# -*- coding: utf-8 -*-
# dao_manager/base_dao.py

import decimal # 用于高精度数字
import json
import logging
import asyncio
import re # 用于异步操作
from django.utils import timezone
from zoneinfo import ZoneInfo
import time
import random
from django.db.utils import OperationalError
from typing import Dict, List, Any, Optional, Type, Union, TypeVar, Generic # 类型提示
from datetime import datetime, date
from django.db import connection, models, IntegrityError, ProgrammingError # Django 模型
from decimal import Decimal

import numpy as np
import pandas as pd
import tushare as ts
from django.conf import settings # Django 设置
from asgiref.sync import sync_to_async # 异步转换工具
from utils.cache_manager import CacheManager


logger = logging.getLogger("dao") # 获取日志记录器

# 定义泛型类型变量 T，限定为 Django 模型
T = TypeVar('T', bound=models.Model)

class BaseDAO(Generic[T]):
    """
    基础数据访问对象 (DAO) 类。
    【V2.0 - 依赖注入版】
    """
    def __init__(self,
                 cache_manager_instance: CacheManager, # <--- 1. 新增参数
                 model_class: Optional[Type[T]] = None,
                 api_service: Any = None,
                 cache_timeout: int = 3600):
        """
        初始化 BaseDAO。
        【V2.0 - 依赖注入版】
        - 不再自己创建 CacheManager，而是接收一个外部传入的实例。
        Args:
            cache_manager_instance: 一个已经初始化的 CacheManager 实例。
            model_class: 此 DAO 主要操作的 Django 模型类。
            api_service: (可选) 用于从外部 API 获取数据的服务实例。
            cache_timeout: (可选) 默认缓存超时时间（秒）。
        """
        self.model_class = model_class
        self.api_service = api_service
        self.cache_timeout = cache_timeout
        # 直接使用传入的 CacheManager 实例
        self.cache_manager = cache_manager_instance # <--- 2. 赋值
        self.ts_pro = ts.pro_api(settings.API_LICENCES_TUSHARE)
        # ts.set_token(...) 返回 None，所以不需要赋值
        ts.set_token(settings.API_LICENCES_TUSHARE)
        self.model_name = model_class._meta.model_name if model_class else "multi_model"
    def _get_cache_key(self, key_suffix: str) -> str:
        """
        (内部方法) 根据模型名称和后缀生成标准化的缓存键。
        Args:
            key_suffix: 缓存键的后缀，用于区分不同的缓存项 (例如 "id:123", "all", "filter:status=active")。
        Returns:
            完整的 Redis 缓存键字符串。
        """
        # 确保 self.model_name 有意义
        prefix = self.model_name if self.model_name != "multi_model" else "base_dao"
        return f"{prefix}:{key_suffix}"
    # ==================== 缓存辅助函数 ====================
    async def _prepare_data_for_cache(self, data: Union[models.Model, Dict],
                                      related_field_map: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """
        (内部方法) 将模型实例或字典转换为适合缓存的字典格式。
        - 处理外键：将关联对象替换为其主键或指定的字段值（例如 stock -> stock_code）。
        - 处理 Decimal：转换为字符串。
        - 处理 datetime/date：转换为 ISO 格式字符串。
        - 移除值为 None 的键。
        - 特殊处理 Level5 的 bids/asks 字段 (如果需要，可以在子类 DAO 中覆盖此方法添加逻辑)。
        Args:
            data: 要处理的模型实例或字典。
            related_field_map: (可选) 字典，映射外键字段名到需要提取的关联对象属性名。
                               例如 {'stock': 'stock_code', 'user': 'username'}。
                               如果为 None，外键将默认被替换为其主键。
        Returns:
            适合缓存的可序列化字典，如果转换失败则返回 None。
        """
        if isinstance(data, models.Model):
            try:
                # 从模型实例提取字段和值
                data_dict = {}
                for field in data._meta.fields:
                    field_name = field.name
                    value = getattr(data, field_name)
                    # 处理外键
                    if field.is_relation and value is not None:
                        if related_field_map and field_name in related_field_map:
                            related_attr = related_field_map[field_name]
                            # 检查关联对象是否有该属性
                            if hasattr(value, related_attr):
                                data_dict[field_name] = getattr(value, related_attr)
                            else:
                                logger.warning(f"关联对象 {field_name} 没有属性 {related_attr}，将使用主键。")
                                data_dict[field_name] = value.pk # 回退到主键
                        else:
                            data_dict[field_name] = value.pk # 默认使用主键
                    else:
                        data_dict[field_name] = value
            except Exception as e:
                logger.error(f"从模型实例 {type(data)} 提取数据失败: {e}", exc_info=True)
                return None
        elif isinstance(data, dict):
            data_dict = data.copy() # 复制字典
        else:
            logger.error(f"不支持的数据类型进行缓存准备: {type(data)}")
            return None
        cache_dict = {}
        for key, value in data_dict.items():
            if value is None:
                continue # 跳过 None 值
            if isinstance(value, Decimal):
                cache_dict[key] = str(value) # Decimal 转字符串
            elif isinstance(value, datetime):
                # 确保 datetime 对象是 timezone-aware (如果设置了 USE_TZ) 或 naive
                if settings.USE_TZ and timezone.is_naive(value):
                    value = timezone.make_aware(value, timezone.get_current_timezone()) # 使用当前时区
                elif not settings.USE_TZ and timezone.is_aware(value):
                    value = timezone.make_naive(value)
                cache_dict[key] = value.isoformat() # 日期时间转 ISO 字符串
            elif isinstance(value, date):
                cache_dict[key] = value.isoformat() # 日期转 ISO 字符串
            # 可以添加对 bids/asks 的特殊处理逻辑 (如果 BaseDAO 需要处理 Level5)
            # elif key in ('bids', 'asks') and isinstance(value, (list, tuple)):
            #     try: cache_dict[key] = [[str(p), v] for p, v in value]
            #     except: cache_dict[key] = []
            elif isinstance(value, (list, tuple, dict)):
                 # 尝试序列化复杂类型，确保可缓存
                 try:
                     self._ensure_cache_objects() # 确保 cache_manager 存在以访问序列化方法
                     # 使用 CacheManager 的内部序列化方法（如果它处理复杂类型）
                     # 或者简单地使用 JSON
                     json.dumps(value) # 检查是否可 JSON 序列化
                     cache_dict[key] = value
                 except TypeError:
                     logger.warning(f"字段 '{key}' 包含不可直接序列化的类型: {type(value)}，尝试转换为字符串。")
                     cache_dict[key] = str(value)
            else:
                # 其他基本类型 (int, str, bool, etc.) 或已处理的外键 ID
                cache_dict[key] = value
        return cache_dict
    async def _build_model_from_cache(self, model_class: Type[T], cached_data: Dict,
                                      related_dao_map: Optional[Dict[str, 'BaseDAO']] = None) -> Optional[T]:
        """
        (内部方法) 从缓存字典构建模型实例。
        - 此方法现在更宽容：如果缓存字典中缺少某个字段（即使是非空字段如'id'），
          它会跳过该字段的设置，将最终验证交给模型实例化过程。
        - 处理外键：根据缓存中的 ID (或指定字段值) 从提供的 DAO 获取关联对象。
        - 处理 Decimal：将字符串转回 Decimal。
        - 处理 datetime/date：将 ISO 字符串转回 datetime/date 对象。
        Args:
            model_class: 要构建的目标模型类。
            cached_data: 从缓存中获取的字典数据。
            related_dao_map: (可选) 字典，映射外键字段名到用于获取关联对象的 DAO 实例。
                               例如 {'stock': self.stock_basic_dao, 'user': self.user_dao}。
                               如果为 None 或缺少某个外键的 DAO，将无法构建该关联。
        Returns:
            构建成功的模型实例，如果字典数据不足以实例化模型或发生其他错误则返回 None。
        """
        if not cached_data or not isinstance(cached_data, dict):
            logger.debug(f"无效的缓存数据用于构建 {model_class.__name__}: {cached_data}")
            return None
        model_data = {} # 用于存储准备好的模型字段数据
        try:
            # 遍历目标模型的所有字段定义
            for field in model_class._meta.fields:
                field_name = field.name
                # --- 修改点：处理缓存中字段缺失的情况 ---
                if field_name not in cached_data:
                    # 如果缓存数据中不存在模型定义的这个字段名
                    # 不再检查 field.null，直接跳过对此字段的处理
                    # 让模型实例化时的 __init__ 方法负责最终检查是否缺少必要字段
                    logger.debug(f"缓存数据缺少字段 '{field_name}' for model {model_class.__name__}，跳过设置。")
                    continue # 继续处理下一个字段
                # --- 结束修改点 ---
                # 获取缓存中对应字段的值
                cached_value = cached_data[field_name]
                # 如果缓存中的值是 None，也跳过处理（除非字段允许 null）
                # 注意：如果字段不允许 null，但在缓存中是 None，实例化时可能会失败
                if cached_value is None and not field.null:
                     logger.debug(f"缓存中字段 '{field_name}' 的值为 None，但模型中该字段不允许 null，跳过设置。")
                     continue
                elif cached_value is None and field.null:
                     model_data[field_name] = None # 显式设置 None
                     continue
                # --- 处理各种字段类型 ---
                # 处理外键 (Relation Field)
                if field.is_relation:
                    # 检查是否有对应的 DAO 来获取关联对象
                    if related_dao_map and field_name in related_dao_map:
                        related_dao = related_dao_map[field_name]
                        # 假设 cached_value 是关联对象的主键值
                        related_pk_value = cached_value
                        try:
                            # 异步调用关联 DAO 的方法获取关联对象实例
                            # 注意：假设 related_dao 有 get_by_id 或类似方法
                            related_obj = await related_dao.get_by_id(related_pk_value)
                            if not related_obj:
                                # 未找到关联对象
                                logger.warning(f"无法找到外键 '{field_name}' 对应的关联对象 (ID: {related_pk_value}) using DAO {type(related_dao).__name__}")
                                # 如果模型字段允许 null，则设置为 None，否则构建失败
                                if field.null:
                                    model_data[field_name] = None
                                    continue
                                else:
                                    # 关联对象是必需的但找不到，无法构建模型
                                    logger.error(f"构建模型 {model_class.__name__} 失败：必需的外键 '{field_name}' 关联对象 (ID: {related_pk_value}) 未找到。")
                                    return None
                            # 成功找到关联对象，存入 model_data
                            model_data[field_name] = related_obj
                        except Exception as dao_err:
                            # 调用关联 DAO 时发生异常
                            logger.error(f"使用 DAO {type(related_dao).__name__} 获取关联对象 {field_name} (ID: {related_pk_value}) 时出错: {dao_err}", exc_info=True)
                            if field.null:
                                model_data[field_name] = None
                                continue
                            else:
                                # 获取必需的关联对象时出错，无法构建模型
                                logger.error(f"构建模型 {model_class.__name__} 失败：获取必需的外键 '{field_name}' 关联对象时出错。")
                                return None
                    else:
                        # 没有提供用于获取关联对象的 DAO
                        logger.warning(f"缺少用于获取外键 '{field_name}' 的 DAO 实例，无法设置该字段。")
                        # 如果字段允许 null，则跳过；否则构建失败
                        if field.null:
                            model_data[field_name] = None # 或者 continue，取决于是否想显式设置 None
                            continue
                        else:
                            logger.error(f"构建模型 {model_class.__name__} 失败：必需的外键 '{field_name}' 缺少对应的 DAO。")
                            return None
                # 处理 Decimal 字段
                elif isinstance(field, models.DecimalField):
                    try:
                        # 尝试将缓存值（通常是字符串）转换为 Decimal
                        model_data[field_name] = Decimal(str(cached_value))
                    except decimal.InvalidOperation:
                        logger.warning(f"无效的 Decimal 值 '{cached_value}' for field '{field_name}' in model {model_class.__name__}")
                        # 如果 Decimal 字段允许 null，可以设为 None 或跳过，否则构建失败
                        if field.null: continue
                        else: return None
                # 处理 DateTime 字段
                elif isinstance(field, models.DateTimeField):
                    try:
                        # 尝试将缓存值（通常是 ISO 格式字符串）转换为 datetime 对象
                        dt = datetime.fromisoformat(str(cached_value))
                        # 根据 Django 设置处理时区
                        if settings.USE_TZ and timezone.is_naive(dt):
                             # 如果项目使用时区且时间是 naive，则设置为项目当前时区
                             dt = timezone.make_aware(dt, timezone.get_current_timezone())
                        elif not settings.USE_TZ and timezone.is_aware(dt):
                             # 如果项目不使用时区且时间是 aware，则转换为 naive
                             dt = timezone.make_naive(dt)
                        model_data[field_name] = dt
                    except ValueError:
                        logger.warning(f"无效的 ISO datetime 格式 '{cached_value}' for field '{field_name}' in model {model_class.__name__}")
                        if field.null: continue
                        else: return None
                # 处理 Date 字段
                elif isinstance(field, models.DateField):
                    try:
                        # 尝试将缓存值（通常是 ISO 格式字符串）转换为 date 对象
                        model_data[field_name] = date.fromisoformat(str(cached_value))
                    except ValueError:
                        logger.warning(f"无效的 ISO date 格式 '{cached_value}' for field '{field_name}' in model {model_class.__name__}")
                        if field.null: continue
                        else: return None
                # 处理其他基本类型字段 (int, str, bool, etc.)
                else:
                    # 直接将缓存中的值赋给 model_data
                    # 注意：这里假设缓存中的类型与模型字段类型兼容
                    # 如果需要更严格的类型检查或转换，可以在这里添加逻辑
                    model_data[field_name] = cached_value
            # --- 尝试使用准备好的 model_data 字典实例化模型 ---
            # 如果 model_data 缺少模型 __init__ 所需的非空字段（且无默认值），这里会抛出 TypeError
            logger.debug(f"准备好用于实例化 {model_class.__name__} 的数据: {model_data}")
            instance = model_class(**model_data)
            logger.debug(f"成功从缓存构建 {model_class.__name__} 实例。")
            return instance
        except Exception as e:
            # 捕获在处理字段或实例化模型过程中发生的任何其他异常
            logger.error(f"从缓存数据构建 {model_class.__name__} 实例时发生未知错误: {e}, data: {cached_data}", exc_info=True)
            return None # 构建失败返回 None
    def _sanitize_for_json(self, data: Any) -> Any:
        """
        递归地清洗数据，使其完全兼容JSON序列化。
        - 将 Decimal 转换为 float。
        - 将 numpy 数字类型 (如 float64, int64) 转换为原生 Python 类型。
        - 将 NaN, NaT, pd.NA 转换为 None。
        """
        if isinstance(data, dict):
            # 如果是字典，递归处理其所有值
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            # 如果是列表，递归处理其所有元素
            return [self._sanitize_for_json(item) for item in data]
        if isinstance(data, Decimal):
            # 将 Decimal 转换为 float
            return float(data)
        if isinstance(data, np.generic):
            # 将所有 numpy 的通用数值类型转换为其对应的 Python 原生类型
            return data.item()
        if pd.isna(data):
            # 将所有 pandas/numpy 的空值表示 (NaN, NaT, pd.NA) 转换为 None
            return None
        # 其他类型原样返回
        return data
    # ==================== CRUD 操作 ====================
    async def get_by_id(self, id_value: Any, related_dao_map: Optional[Dict[str, 'BaseDAO']] = None) -> Optional[T]:
        """
        (异步) 根据主键 ID 获取单个实体。
        查询顺序：缓存 -> 数据库 -> API (如果配置了 api_service 且子类实现了 _fetch_from_api_by_id)。
        Args:
            id_value: 主键值。
            related_dao_map: (可选) 用于从缓存构建模型时获取外键关联对象的 DAO 映射。
        Returns:
            模型实例 T，如果找不到则返回 None。
        """
        if self.model_class is None:
            raise TypeError("model_class 未在 BaseDAO 初始化时设置，无法执行 get_by_id")
        # 确保缓存管理器实例存在
        self._ensure_cache_objects()
        # 生成缓存键
        cache_key = self._get_cache_key(f"id:{id_value}")
        # 1. 先从缓存获取
        try:
            cached_data_dict = await self.cache_manager.get(key=cache_key)
            if cached_data_dict and isinstance(cached_data_dict, dict):
                logger.debug(f"缓存命中: {cache_key}")
                # 尝试从缓存构建模型
                instance = await self._build_model_from_cache(self.model_class, cached_data_dict, related_dao_map)
                if instance:
                    return instance
                else:
                    logger.warning(f"缓存数据无效或构建模型失败，删除缓存 key: {cache_key}")
                    await self.cache_manager.delete(cache_key)
            else:
                logger.debug(f"缓存未命中: {cache_key}")
        except Exception as e:
            logger.error(f"从缓存获取 {self.model_name} (ID: {id_value}) 时发生异常: {e}", exc_info=True)
        # 2. 从数据库获取
        try:
            instance = await self._get_from_db_by_id(id_value)
            if instance:
                logger.debug(f"数据库命中: {self.model_name} (ID: {id_value})")
                # --- 写入缓存 ---
                try:
                    data_to_cache = await self._prepare_data_for_cache(instance)
                    if data_to_cache:
                        await self.cache_manager.set(key=cache_key, data=data_to_cache, timeout=self.cache_timeout)
                        logger.debug(f"已将 {self.model_name} (ID: {id_value}) 写入缓存, key: {cache_key}")
                except Exception as cache_err:
                    logger.error(f"将 {self.model_name} (ID: {id_value}) 写入缓存失败: {cache_err}", exc_info=True)
                # --- 结束写入缓存 ---
                return instance
        except Exception as e:
            logger.error(f"数据库查询 {self.model_name} (ID: {id_value}) 错误: {str(e)}", exc_info=True)
            # 根据策略决定是否继续尝试 API
        # 3. 如果配置了 API 服务，尝试从 API 获取 (需要子类实现 _fetch_from_api_by_id)
        if self.api_service:
            try:
                logger.debug(f"从API获取: {self.model_name} (ID: {id_value})")
                # _fetch_from_api_by_id 应该返回适合保存的字典
                api_data_dict = await self._fetch_from_api_by_id(id_value)
                if api_data_dict and isinstance(api_data_dict, dict):
                    # 保存到数据库 (使用 upsert 方法更健壮)
                    # 注意：需要提供 unique_fields
                    pk_name = self.model_class._meta.pk.name
                    unique_fields = [pk_name] # 假设主键是唯一标识
                    save_result = await self._save_all_to_db_native_upsert(
                        model_class=self.model_class,
                        data_list=[api_data_dict],
                        unique_fields=unique_fields
                    )
                    if save_result.get("创建/更新成功", 0) > 0:
                        # 保存成功后，再次从数据库获取以确保得到完整对象
                        instance = await self._get_from_db_by_id(id_value)
                        if instance:
                            # --- 写入缓存 ---
                            try:
                                data_to_cache = await self._prepare_data_for_cache(instance)
                                if data_to_cache:
                                    await self.cache_manager.set(key=cache_key, data=data_to_cache, timeout=self.cache_timeout)
                            except Exception as cache_err:
                                logger.error(f"将新获取的 {self.model_name} (ID: {id_value}) 写入缓存失败: {cache_err}")
                            return instance
                else:
                    logger.warning(f"API 未返回有效的 {self.model_name} (ID: {id_value}) 数据")
            except NotImplementedError:
                 logger.debug(f"子类未实现 _fetch_from_api_by_id 方法 for {self.model_name}")
            except Exception as e:
                logger.error(f"API 获取 {self.model_name} (ID: {id_value}) 数据错误: {str(e)}", exc_info=True)
        return None # 所有尝试失败
    async def _get_from_db_by_id(self, id_value: Any) -> Optional[T]:
        """
        (内部方法/异步) 从数据库根据主键获取单个实体。
        Args:
            id_value: 主键值。
        Returns:
            模型实例 T，如果找不到则返回 None。
        """
        if self.model_class is None: return None
        try:
            pk_name = self.model_class._meta.pk.name
            filter_kwargs = {pk_name: id_value}
            # 使用 sync_to_async 执行 ORM 查询
            instance = await sync_to_async(
                self.model_class.objects.filter(**filter_kwargs).first,
                thread_sensitive=True
            )()
            return instance
        except Exception as e:
            logger.error(f"数据库查询 {self.model_name} (ID: {id_value}) 错误: {str(e)}", exc_info=True)
            return None
    async def _fetch_from_api_by_id(self, id_value: Any) -> Optional[Dict[str, Any]]:
        """
        (内部方法/异步/待子类实现) 从 API 获取单个实体数据。
        子类 DAO 必须重写此方法以实现具体的 API 调用逻辑。
        Args:
            id_value: ID 值。
        Returns:
            从 API 获取并处理后的实体数据字典，如果获取失败则返回 None。
        """
        raise NotImplementedError(f"{self.__class__.__name__} 必须实现 _fetch_from_api_by_id 方法")
    async def get_all(self, related_dao_map: Optional[Dict[str, 'BaseDAO']] = None) -> List[T]:
        """
        (异步) 获取模型的所有实体列表。
        查询顺序：缓存 -> 数据库 -> API (如果配置了 api_service 且子类实现了 _fetch_all_from_api)。
        Args:
            related_dao_map: (可选) 用于从缓存构建模型时获取外键关联对象的 DAO 映射。
        Returns:
            包含所有实体的模型实例列表 T。
        """
        if self.model_class is None:
            raise TypeError("model_class 未在 BaseDAO 初始化时设置，无法执行 get_all")
        instances = []
        cache_hit = False
        # 确保缓存管理器实例存在
        self._ensure_cache_objects()
        # 生成缓存键
        cache_key = self._get_cache_key("all")
        # 1. 先从缓存获取
        try:
            cached_list = await self.cache_manager.get(key=cache_key)
            if cached_list and isinstance(cached_list, list):
                logger.debug(f"缓存命中: {cache_key}, 命中 {len(cached_list)} 条")
                # 尝试从缓存构建模型列表
                build_tasks = [self._build_model_from_cache(self.model_class, item_dict, related_dao_map) for item_dict in cached_list]
                results = await asyncio.gather(*build_tasks)
                instances = [item for item in results if item is not None] # 过滤 None
                if len(instances) == len(cached_list): # 检查是否所有项都成功构建
                    cache_hit = True
                else:
                    logger.warning(f"部分缓存数据无效或构建模型失败，删除缓存 key: {cache_key}")
                    await self.cache_manager.delete(cache_key)
                    cache_hit = False # 标记为未命中
            else:
                logger.debug(f"缓存未命中: {cache_key}")
        except Exception as e:
            logger.error(f"从缓存获取所有 {self.model_name} 时发生异常: {e}", exc_info=True)
        # 2. 从数据库获取
        if not cache_hit:
            try:
                instances = await self._get_all_from_db()
                if instances: # 检查是否为空列表
                    logger.debug(f"数据库命中: 共{len(instances)}条 {self.model_name} 记录")
                    # --- 写入缓存 ---
                    try:
                        prepare_tasks = [self._prepare_data_for_cache(instance) for instance in instances]
                        data_to_cache = await asyncio.gather(*prepare_tasks)
                        data_to_cache = [d for d in data_to_cache if d] # 过滤 None
                        if data_to_cache:
                            await self.cache_manager.set(key=cache_key, data=data_to_cache, timeout=self.cache_timeout)
                            logger.debug(f"已将所有 {self.model_name} 列表写入缓存, key: {cache_key}")
                    except Exception as cache_err:
                        logger.error(f"将所有 {self.model_name} 列表写入缓存失败: {cache_err}", exc_info=True)
                    # --- 结束写入缓存 ---
                else:
                     logger.info(f"数据库中没有 {self.model_name} 数据")
                     # 数据库为空，尝试 API
                     if self.api_service:
                         try:
                             logger.debug(f"从API获取所有 {self.model_name} 数据")
                             # _fetch_all_from_api 应该返回适合保存的字典列表
                             api_data_list = await self._fetch_all_from_api()
                             if api_data_list and isinstance(api_data_list, list):
                                 # 保存到数据库 (需要确定 unique_fields)
                                 # 假设主键是唯一标识，或者需要子类提供
                                 # unique_fields = [self.model_class._meta.pk.name] # 可能不适用所有情况
                                 # logger.warning("无法确定 unique_fields for _fetch_all_from_api, 保存可能失败或重复")
                                 # 暂时跳过保存和缓存更新，子类应重写 get_all 或提供保存逻辑
                                 logger.warning(f"从 API 获取到 {len(api_data_list)} 条数据，但 BaseDAO 未实现保存逻辑，请在子类处理")
                                 # 如果需要 BaseDAO 处理，需要更复杂的逻辑来确定 unique_fields
                                 # 或者直接返回 API 数据转换后的模型列表（不持久化）
                                 # instances = [self.model_class(**item) for item in api_data_list] # 示例：直接转换
                                 instances = [] # 安全起见，返回空列表
                             else:
                                 logger.warning(f"API 未返回有效的 {self.model_name} 数据列表")
                                 instances = []
                         except NotImplementedError:
                              logger.debug(f"子类未实现 _fetch_all_from_api 方法 for {self.model_name}")
                              instances = []
                         except Exception as e:
                             logger.error(f"API 获取所有 {self.model_name} 数据错误: {str(e)}", exc_info=True)
                             instances = []
                     else:
                          instances = [] # 没有 API 服务，返回空列表
            except Exception as e:
                logger.error(f"数据库查询所有 {self.model_name} 错误: {str(e)}", exc_info=True)
                instances = [] # 查询失败返回空列表
        return instances
    async def _get_all_from_db(self) -> List[T]:
        """
        (内部方法/异步) 从数据库获取所有实体。
        Returns:
            模型实例 T 的列表。
        """
        if self.model_class is None: return []
        try:
            # 使用 sync_to_async 执行 ORM 查询
            instances = await sync_to_async(
                list, # 将 QuerySet 转换为列表
                thread_sensitive=True
            )(self.model_class.objects.all())
            return instances
        except Exception as e:
            logger.error(f"数据库查询所有 {self.model_name} 错误: {str(e)}", exc_info=True)
            return []
    async def _fetch_all_from_api(self) -> List[Dict[str, Any]]:
        """
        (内部方法/异步/待子类实现) 从 API 获取所有实体数据。
        子类 DAO 必须重写此方法以实现具体的 API 调用逻辑。
        Returns:
            从 API 获取并处理后的实体数据字典列表。
        """
        raise NotImplementedError(f"{self.__class__.__name__} 必须实现 _fetch_all_from_api 方法")
    async def filter(self, related_dao_map: Optional[Dict[str, 'BaseDAO']] = None, **kwargs) -> List[T]:
        """
        (异步) 根据提供的关键字参数从数据库筛选实体。
        注意：此方法目前不实现缓存，因为筛选条件的组合可能非常多，缓存效率不高。
              如果特定筛选条件非常常用，可以考虑在子类中实现针对性的缓存。
        Args:
            related_dao_map: (可选) 用于构建模型时获取外键关联对象的 DAO 映射 (如果需要从缓存构建)。
            **kwargs: Django ORM filter 支持的查询参数。
        Returns:
            符合条件的模型实例 T 的列表。
        """
        if self.model_class is None:
            raise TypeError("model_class 未在 BaseDAO 初始化时设置，无法执行 filter")
        # 移除缓存逻辑，直接查询数据库
        # filter_str = ":".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        # cache_key = self._get_cache_key(f"filter:{filter_str}")
        # self._ensure_cache_objects()
        # cached_data = await self.cache_manager.get(key=cache_key) ...
        logger.debug(f"执行数据库筛选 for {self.model_name} with filters: {kwargs}")
        try:
            # 使用 sync_to_async 执行 ORM filter 操作
            instances = await sync_to_async(
                list, # 将 QuerySet 转换为列表
                thread_sensitive=True
            )(self.model_class.objects.filter(**kwargs))
            # 不再写入缓存
            # await self.cache_manager.set(cache_key, instances, self.cache_timeout)
            return instances
        except Exception as e:
            logger.error(f"数据库筛选 {self.model_name} 错误: {str(e)}", exc_info=True)
            return []
    # ==================== 批量保存方法 ====================
    async def _save_all_to_db_native_upsert(self, model_class, data_list: list, unique_fields: list, batch_size=10000):
        """
        【V4.0 - 流程重构版】的入口方法。
        此版本彻底移除了中间的DataFrame转换，解决了因数据转换导致的列丢失问题。
        """
        # 1. 初始检查
        if not data_list:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        total_records = len(data_list)
        # 2. 准备字段映射
        # 这个映射包含了模型所有字段及其对应的数据库列名
        field_to_column_map = {f.name: f.column for f in model_class._meta.fields}
        # 3. 将包含对象的 data_list 转换为 SQL-ready 的字典列表
        sql_ready_data_list = []
        for record in data_list:
            sql_record = {}
            for field_name, value in record.items():
                # 检查这个key是否是模型的一个字段
                field_obj = model_class._meta.get_field(field_name)
                # 如果值是一个模型实例 (代表外键)
                if isinstance(value, models.Model):
                    column_name = field_obj.column
                    target_field_name = field_obj.target_field.name
                    # 获取外键引用的值 (可能是pk，也可能是to_field指定的值)
                    fk_value = getattr(value, target_field_name)
                    sql_record[column_name] = fk_value
                # 如果key是模型的字段，但值不是对象 (常规字段)
                elif field_name in field_to_column_map:
                    sql_record[field_to_column_map[field_name]] = value
                # 如果key不是模型的字段，则忽略 (例如上游传入的临时辅助字段)
                
            sql_ready_data_list.append(sql_record)
        if not sql_ready_data_list:
            logger.warning("所有记录在准备阶段均失败或为空，不执行数据库操作。")
            return {"尝试处理": total_records, "失败": total_records, "创建/更新成功": 0}
        # 4. 确定需要更新的数据库列
        # 从上游接收的 unique_fields 是模型字段名，需要转换为数据库列名
        unique_db_columns = {field_to_column_map.get(f, f) for f in unique_fields}
        # 从准备好的SQL数据中获取所有将要操作的列
        all_db_columns_in_data = list(sql_ready_data_list[0].keys())
        # 计算出需要UPDATE的列
        update_db_columns = [col for col in all_db_columns_in_data if col not in unique_db_columns]
        # 5. 调用下游异步批处理方法
        try:
            # 注意：现在传递的是 sql_ready_data_list 和 update_db_columns
            success_count = await self.process_batch_async(
                model_class=model_class,
                data_list=sql_ready_data_list,
                unique_fields=unique_fields, # 底层仍然需要这个来识别唯一键
                update_fields=update_db_columns, # 传递计算好的、以数据库列为准的更新列表
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"调用 process_batch_async 时发生未知错误: {e}", exc_info=True)
            success_count = 0
        # 6. 返回结果
        failed_count = total_records - success_count
        return {"尝试处理": total_records, "失败": failed_count, "创建/更新成功": success_count}
    async def process_batch_async(self, model_class, data_list: list, unique_fields: list, update_fields: list, batch_size=5000):
        """
        【V4.0 - 流程重构版】的异步批处理调度器。
        - 不再接收和处理DataFrame，直接处理SQL-ready的字典列表。
        """
        if not data_list:
            return 0
        lock_key = f"db_lock:upsert:{model_class._meta.db_table}"
        total_processed = 0
        try:
            redis_client = await self.cache_manager._ensure_client()
            if not redis_client:
                raise ConnectionError("无法获取 Redis 客户端，数据库操作被中止以保证数据安全。")
            async with redis_client.lock(lock_key, timeout=120, blocking_timeout=130):
                # 直接对字典列表进行分批
                for i in range(0, len(data_list), batch_size):
                    batch_list = data_list[i:i + batch_size]
                    try:
                        processed_count = await sync_to_async(self._process_batch_mysql_upsert_sync)(
                            model_class=model_class,
                            data_list=batch_list,
                            unique_fields=unique_fields,
                            update_fields=update_fields # 直接透传已计算好的update_fields
                        )
                        if processed_count is not None:
                            total_processed += processed_count
                    except Exception as e:
                        logger.error(f"原生SQL批处理时遇到意外错误 (在锁内): {e}", exc_info=True)
                logger.info(f"异步批处理完成，共处理 {model_class._meta.db_table} 模型 - {total_processed} 条记录。")
        except Exception as lock_error:
            logger.error(f"获取Redis分布式锁 {lock_key} 失败或在持有锁期间发生未捕获的异常: {lock_error}", exc_info=True)
            return 0
        return total_processed
    def _process_batch_mysql_upsert_sync(self, model_class, data_list, unique_fields, update_fields=None, **kwargs):
        """
        【V3.2 死锁重试修复版】
        使用原生SQL `INSERT ... ON DUPLICATE KEY UPDATE` 执行批量更新或插入操作。
        此版本新增了对数据库死锁 (error 1213) 的自动重试机制。
        - 核心逻辑:
          1. 信任 data_list 中字典的 keys 作为数据库列名。
          2. 在构建参数时，将所有 pd.isna() 为 True 的值（如 np.nan）转换成 None。
          3. 智能地将模型层面的 unique_fields（如 'stock'）转换为数据库层面的列名（如 'stock_id'）。
          4. 当捕获到 OperationalError 且错误码为 1213 时，进行最多3次重试，并采用指数退避+抖动策略。
        Args:
            model_class: Django模型类。
            data_list (List[Dict]): 包含待处理数据的字典列表。键应为数据库列名。
            unique_fields (List[str]): 用于确定记录唯一性的【模型字段名】列表。
            update_fields (List[str], optional): 需要更新的【模型字段名】列表。
            **kwargs: 接收并忽略任何其他关键字参数。
        Returns:
            int: 受影响的总行数。
        """
        total_affected_rows = 0
        if not data_list:
            logger.info(f"模型 {model_class.__name__} 的数据列表为空，跳过批量保存。")
            return total_affected_rows
        table_name = model_class._meta.db_table
        all_db_columns = list(data_list[0].keys())
        params_list_of_tuples = []
        for item in data_list:
            param_tuple = []
            for col in all_db_columns:
                value = item.get(col)
                if pd.isna(value):
                    value = None
                param_tuple.append(value)
            params_list_of_tuples.append(tuple(param_tuple))
        field_to_column_map = {f.name: f.column for f in model_class._meta.fields}
        unique_db_columns = {field_to_column_map.get(f, f) for f in unique_fields}
        if update_fields is None:
            update_db_columns = [col for col in all_db_columns if col not in unique_db_columns]
        else:
            update_db_columns = [field_to_column_map.get(f, f) for f in update_fields]
        field_list_str = ", ".join([f"`{f}`" for f in all_db_columns])
        value_placeholders = ", ".join(["%s"] * len(all_db_columns))
        if not update_db_columns:
            final_sql_template = f"INSERT IGNORE INTO `{table_name}` ({field_list_str}) VALUES ({value_placeholders})"
        else:
            update_clause = ", ".join([f"`{f}` = VALUES(`{f}`)" for f in update_db_columns])
            final_sql_template = (
                f"INSERT INTO `{table_name}` ({field_list_str}) VALUES ({value_placeholders}) "
                f"ON DUPLICATE KEY UPDATE {update_clause}"
            )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with connection.cursor() as cursor:
                    if not params_list_of_tuples:
                        return 0
                    total_affected_rows = cursor.executemany(final_sql_template, params_list_of_tuples)
                return total_affected_rows
            except (IntegrityError, OperationalError, ProgrammingError) as e:
                if isinstance(e, OperationalError) and e.args[0] == 1213:
                    if attempt < max_retries - 1:
                        sleep_time = (0.1 * (2 ** attempt)) + random.uniform(0.05, 0.1)
                        print(f"DEBUG: 捕获到数据库死锁 (1213)，正在进行第 {attempt + 2}/{max_retries} 次重试... 等待 {sleep_time:.2f} 秒。模型: {model_class.__name__}")
                        logger.warning(
                            f"捕获到数据库死锁 (1213)，正在进行第 {attempt + 2}/{max_retries} 次重试... "
                            f"等待 {sleep_time:.2f} 秒。模型: {model_class.__name__}"
                        )
                        time.sleep(sleep_time)
                        continue
                print("="*50)
                print(f"FATAL DEBUG: [base_dao] 批量 Upsert 时捕获到数据库错误！ (尝试 {attempt + 1}/{max_retries})")
                print(f"FATAL DEBUG: 模型 (Model): {model_class.__name__}")
                print(f"FATAL DEBUG: SQL 模板: {final_sql_template}")
                if params_list_of_tuples:
                    print(f"FATAL DEBUG: 待插入/更新的第一条数据 (元组): {params_list_of_tuples[0]}")
                print(f"FATAL DEBUG: 异常类型: {type(e).__name__}")
                print(f"FATAL DEBUG: 异常信息: {e}")
                print("="*50)
                logger.error(f"原生SQL批处理时遇到数据库错误 (在锁内): {e}")
                return 0
            except Exception as e:
                print("="*50)
                print(f"FATAL DEBUG: [base_dao] 批量 Upsert 时捕获到未知异常！ (尝试 {attempt + 1}/{max_retries})")
                print(f"FATAL DEBUG: 模型 (Model): {model_class.__name__}")
                print(f"FATAL DEBUG: SQL 模板: {final_sql_template}")
                if params_list_of_tuples:
                    print(f"FATAL DEBUG: 待插入/更新的第一条数据 (元组): {params_list_of_tuples[0]}")
                print(f"FATAL DEBUG: 异常类型: {type(e).__name__}")
                print(f"FATAL DEBUG: 异常信息: {e}")
                print("="*50)
                logger.error(f"原生SQL批处理时遇到意外错误 (在锁内): {e}")
                return 0
        logger.error(f"模型 {model_class.__name__} 的批量 Upsert 在 {max_retries} 次尝试后均失败。")
        return 0
    async def _prepare_model_instance(self, model_class, prepared_data, fk_fields_map):
        """
        【V8 - 修复字段移除问题版】
        准备单个模型实例的数据字典。
        核心功能：处理外键，将 'stock_code' 这样的字段或 {'stock': {'stock_code': ...}} 这样的结构，统一转换为实际的 'stock' 对象。
        """
        for fk_field_name, code_field_name in fk_fields_map.items():
            code_value = None
            if fk_field_name in prepared_data and isinstance(prepared_data[fk_field_name], dict):
                fk_dict = prepared_data[fk_field_name]
                code_value = fk_dict.get(code_field_name)
            elif code_field_name in prepared_data:
                # 改为直接获取值，以保留 code_field_name (如 'index_code') 在数据中，
                # 因为目标模型可能同时需要外键对象和代码字段本身。
                code_value = prepared_data[code_field_name]
                # print(f"DEBUG: Preserving '{code_field_name}' in data dictionary. Value: '{code_value}'")
            if code_value is not None:
                if code_value is None:
                    prepared_data[fk_field_name] = None
                    continue
                fk_model = model_class._meta.get_field(fk_field_name).related_model
                fk_instance = await self.get_or_create_fk_instance(fk_model, code_field_name, code_value)
                if fk_instance is None:
                    error_msg = (
                        f"为外键字段 '{fk_field_name}' 准备实例失败！"
                        f"无法为代码 '{code_value}' (使用字段 '{code_field_name}') 找到或创建对应的 '{fk_model.__name__}' 实例。"
                    )
                    raise ValueError(error_msg)
                prepared_data[fk_field_name] = fk_instance
        # 使用 model_class._meta.fields 确保只清理模型中实际定义的字段。
        model_field_names = {f.name for f in model_class._meta.fields}
        cleaned_data = {k: v for k, v in prepared_data.items() if k in model_field_names}
        # print(f"DEBUG: Final cleaned data for model instance: {cleaned_data.keys()}")
        return cleaned_data
    @staticmethod
    @sync_to_async
    def _get_or_create_fk_sync(fk_model: Type[models.Model], lookup_field: str, code_value: str) -> models.Model | None:
        """
        同步的辅助方法，用于执行数据库的 get_or_create 操作。
        这是被 sync_to_async 包装的核心，确保数据库调用在同步线程中执行。
        """
        # 关键修复：不再硬编码 'stock_code'，而是使用从上层动态传入的 lookup_field。
        try:
            # 使用 Django ORM 的 get_or_create，它会原子性地尝试获取，如果不存在则创建。
            # 它返回一个元组 (instance, created_boolean)。
            # print(f"DEBUG: _get_or_create_fk_sync: fk_model={fk_model.__name__}, lookup_field='{lookup_field}', code_value='{code_value}'")
            instance, created = fk_model.objects.get_or_create(
                **{lookup_field: code_value},
            )
            # if created:
            #     print(f"DEBUG: Created new {fk_model.__name__} instance for {lookup_field}={code_value}")
            return instance
        except Exception as e:
            # 捕获可能的数据库错误或其他问题，并记录详细日志。
            logger.error(f"在 _get_or_create_fk_sync 中为代码 '{code_value}' (字段: {lookup_field}) 操作 '{fk_model.__name__}' 时出错: {e}", exc_info=True)
            return None # 出错时返回 None，上层会捕获并抛出 ValueError
    async def get_or_create_fk_instance(self, fk_model: Type[models.Model], code_field_name: str, code_value: str) -> models.Model | None:
        """
        【V6 动态字段版】
        异步获取或创建外键实例。
        它调用一个被 @sync_to_async 包装的同步方法来安全地与数据库交互。
        这个版本接收要查询的字段名(code_field_name)，使其更具通用性。
        """
        # 调试信息：打印即将用于查询的参数
        # print(f"DEBUG: get_or_create_fk_instance: fk_model={fk_model.__name__}, code_field_name='{code_field_name}', code_value='{code_value}'")
        # 关键方法签名已更新为 (self, fk_model, code_field_name, code_value)，现在可以正确地将参数传递给下一层。
        return await self._get_or_create_fk_sync(fk_model, code_field_name, code_value)
    # ==================== 更新和删除操作 ====================
    async def update(self, id_value: Any, data: Dict[str, Any]) -> Optional[T]:
        """
        (异步) 更新指定 ID 的实体。
        会先从数据库获取实体，然后更新字段并保存。
        成功更新后会删除对应的缓存项。
        Args:
            id_value: 要更新的实体的主键值。
            data: 包含要更新字段和新值的字典。
        Returns:
            更新后的模型实例 T，如果实体不存在或更新失败则返回 None。
        """
        if self.model_class is None:
            raise TypeError("model_class 未在 BaseDAO 初始化时设置，无法执行 update")
        try:
            # 异步获取要更新的实例
            instance = await self._get_from_db_by_id(id_value)
            if not instance:
                logger.warning(f"更新失败: {self.model_name} 实体不存在 (ID: {id_value})")
                return None
            # 更新实例的字段
            has_changes = False
            for key, value in data.items():
                if hasattr(instance, key) and getattr(instance, key) != value:
                    setattr(instance, key, value)
                    has_changes = True
            if not has_changes:
                logger.debug(f"实体 {self.model_name} (ID: {id_value}) 无需更新")
                return instance # 没有变化，直接返回原实例
            # 异步保存更新
            await sync_to_async(instance.save, thread_sensitive=True)()
            # --- 更新成功后，删除相关缓存 ---
            self._ensure_cache_objects()
            cache_key_id = self._get_cache_key(f"id:{id_value}")
            cache_key_all = self._get_cache_key("all")
            try:
                await self.cache_manager.delete(cache_key_id)
                await self.cache_manager.delete(cache_key_all) # 删除 'all' 缓存，因为它可能已失效
                # 注意：更复杂的缓存策略可能需要删除更多相关的缓存键 (例如 filter 缓存)
                logger.debug(f"已删除缓存: {cache_key_id}, {cache_key_all}")
            except Exception as cache_err:
                 logger.error(f"删除缓存失败 after update for {self.model_name} (ID: {id_value}): {cache_err}", exc_info=True)
            # --- 结束删除缓存 ---
            logger.info(f"成功更新实体: {self.model_name} (ID: {id_value})")
            return instance
        except Exception as e:
            logger.error(f"更新实体 {self.model_name} (ID: {id_value}) 错误: {str(e)}", exc_info=True)
            return None
    async def delete(self, id_value: Any) -> bool:
        """
        (异步) 删除指定 ID 的实体。
        成功删除后会删除对应的缓存项。
        Args:
            id_value: 要删除的实体的主键值。
        Returns:
            如果成功删除返回 True，否则返回 False。
        """
        if self.model_class is None:
            raise TypeError("model_class 未在 BaseDAO 初始化时设置，无法执行 delete")
        try:
            # 获取主键字段名
            pk_name = self.model_class._meta.pk.name
            filter_kwargs = {pk_name: id_value}
            # 异步执行删除操作
            # delete() 返回一个元组 (count, detailed_counts)
            count, _ = await sync_to_async(
                self.model_class.objects.filter(**filter_kwargs).delete,
                thread_sensitive=True
            )()
            if count > 0:
                # --- 删除成功后，删除相关缓存 ---
                self._ensure_cache_objects()
                cache_key_id = self._get_cache_key(f"id:{id_value}")
                cache_key_all = self._get_cache_key("all")
                try:
                    await self.cache_manager.delete(cache_key_id)
                    await self.cache_manager.delete(cache_key_all)
                    logger.debug(f"已删除缓存: {cache_key_id}, {cache_key_all}")
                except Exception as cache_err:
                    logger.error(f"删除缓存失败 after delete for {self.model_name} (ID: {id_value}): {cache_err}", exc_info=True)
                # --- 结束删除缓存 ---
                logger.info(f"成功删除实体: {self.model_name} (ID: {id_value})")
                return True
            else:
                logger.warning(f"删除失败: {self.model_name} 实体不存在 (ID: {id_value})")
                return False
        except Exception as e:
            logger.error(f"删除实体 {self.model_name} (ID: {id_value}) 错误: {str(e)}", exc_info=True)
            return False
    # ==================== 数据解析工具方法 ====================
    def _parse_datetime(self, value, default_format=None):
        """
        (内部方法/同步) 解析各种格式的日期时间值，返回 timezone-aware 或 naive datetime 对象。
        """
        # 获取项目配置的时区，如果未配置则使用 UTC
        tz_name = getattr(settings, 'TIME_ZONE', 'UTC')
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            logger.warning(f"未知的时区名称 '{tz_name}'，将使用 UTC。")
            tz = ZoneInfo("UTC")
        if isinstance(value, datetime):
            # 如果已经是 datetime 对象，确保其时区正确
            if settings.USE_TZ:
                dt = timezone.make_aware(value, tz) if timezone.is_naive(value) else value.astimezone(tz)
            else:
                dt = timezone.make_naive(value, tz) if timezone.is_aware(value) else value
            return dt
        if isinstance(value, date) and not isinstance(value, datetime):
            # 如果是 date 对象，转换为 datetime (午夜)
            dt = datetime.combine(value, datetime.min.time())
            dt = timezone.make_aware(dt, tz) if settings.USE_TZ else dt
            return dt
        if value is None or str(value).strip() in ['', '-', 'N/A', '暂无']:
            return None # 处理空值
        # 尝试解析字符串
        if isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    return None
            value = value.strip()
            # 修正所有 +08:xx 或 -09:xx 变成 +08:00 或 -09:00
            value = re.sub(r'([+-]\d{2}):\d{2}$', r'\1:00', value)
            # 尝试 ISO 格式
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                if settings.USE_TZ:
                    dt = timezone.make_aware(dt, tz) if timezone.is_naive(dt) else dt.astimezone(tz)
                else:
                    dt = timezone.make_naive(dt, tz) if timezone.is_aware(dt) else dt
                return dt
            except ValueError:
                pass # 继续尝试
            # 尝试指定格式
            if default_format:
                try:
                    dt = datetime.strptime(value, default_format)
                    dt = timezone.make_aware(dt, tz) if settings.USE_TZ else dt
                    return dt
                except ValueError:
                    pass # 继续尝试
            # 尝试常见格式列表
            common_formats = [
                '%Y%m%d%H:%M:%S','%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M',
                '%Y%m%d%H%M%S', '%Y%m%d%H%M', '%Y%m%d', '%Y-%m-%d', '%Y/%m/%d',
                '%Y年%m月%d日 %H时%M分%S秒', '%Y年%m月%d日 %H时%M分', '%Y年%m月%d日',
            ]
            for fmt in common_formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    dt = timezone.make_aware(dt, tz) if settings.USE_TZ else dt
                    return dt
                except ValueError:
                    continue
        # 尝试解析时间戳 (秒或毫秒)
        try:
            timestamp = float(str(value))
            # 简单判断秒或毫秒
            if timestamp > 2000000000: # 大约 2033 年后的秒数，可能是毫秒
                timestamp /= 1000
            dt = datetime.fromtimestamp(timestamp, tz)
            return dt if settings.USE_TZ else timezone.make_naive(dt, tz)
        except (ValueError, TypeError):
            pass # 不是时间戳，继续尝试其他格式
        logger.warning(f"无法解析日期时间值: {value}")
        return None # 所有尝试失败
    def _parse_number(self, value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
        """
        (内部方法/同步) 解析数字，处理各种格式的数字字符串，返回 Decimal 类型以保证精度。
        Args:
            value: 要解析的值 (数字、字符串等)。
            default: (可选) 解析失败时返回的默认值，默认为 None。
        Returns:
            解析后的 Decimal 对象，如果解析失败则返回 default。
        """
        if value is None or str(value).strip() in ['', '-', 'N/A', '暂无']:
            return default
        try:
            # 如果已经是 Decimal，直接返回
            if isinstance(value, Decimal):
                return value
            # 如果是 int 或 float，转换为 Decimal
            if isinstance(value, (int, float)):
                return Decimal(value)
            # 处理字符串
            value_str = str(value).strip()
            # 移除千位分隔符
            value_str = value_str.replace(',', '')
            # 处理百分号
            is_percent = '%' in value_str
            if is_percent:
                value_str = value_str.replace('%', '')
            # 尝试直接转换为 Decimal
            number = Decimal(value_str)
            # 如果是百分比，除以 100
            if is_percent:
                number /= Decimal(100)
            return number
        except decimal.InvalidOperation:
            # 如果直接转换失败，尝试提取数字并处理单位 (万/亿)
            try:
                import re
                # 匹配数字部分 (包括负号和小数点)
                number_match = re.search(r'([-+]?\d*\.?\d+)', value_str)
                if not number_match:
                    logger.warning(f"无法从字符串提取数字: {value}")
                    return default
                number = Decimal(number_match.group(1))
                # 处理单位
                if '万亿' in value_str or '兆' in value_str: number *= Decimal('1000000000000')
                elif '亿' in value_str: number *= Decimal('100000000')
                elif '万' in value_str: number *= Decimal('10000')
                elif '千' in value_str: number *= Decimal('1000')
                # 再次检查百分号 (可能单位和百分号并存)
                if '%' in value_str and not is_percent: # 避免重复除以 100
                    number /= Decimal(100)
                return number
            except Exception as e_inner:
                logger.warning(f"解析带单位的数字失败: {value}, 错误: {e_inner}")
                return default
        except Exception as e_outer:
            logger.warning(f"解析数字失败: {value}, 错误: {e_outer}")
            return default
    @staticmethod
    def _get_model_fields(model_class: Type[models.Model]) -> List[str]:
        """
        (静态内部方法) 获取指定 Django 模型的字段名称列表。
        Args:
            model_class: Django 模型类。
        Returns:
            包含所有字段名称的字符串列表。
        """
        if not model_class:
            return []
        try:
            return [field.name for field in model_class._meta.fields]
        except Exception as e:
            logger.error(f"获取模型 {model_class.__name__} 字段时出错: {e}")
            return []
    # ==================== 日期方法 ====================
    # 获取本周一和本周五的日期
    def get_this_monday_and_friday():
        """获取本周一和本周五的日期"""
        today = datetime.date.today()
        this_monday = today - datetime.timedelta(days=today.weekday())
        this_friday = this_monday + datetime.timedelta(days=4)
        return this_monday, this_friday
    # 获取上周一和上周五的日期
    def get_last_monday_and_friday():
        """获取上周一和上周五的日期"""
        today = datetime.date.today()
        this_monday = today - datetime.timedelta(days=today.weekday())
        last_monday = this_monday - datetime.timedelta(days=7)
        last_friday = last_monday + datetime.timedelta(days=4)
        return last_monday, last_friday










