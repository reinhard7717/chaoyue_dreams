# -*- coding: utf-8 -*-
# dao_manager/base_dao.py

import decimal # 用于高精度数字
import json
import logging
import asyncio
import re # 用于异步操作
from django.utils import timezone
from zoneinfo import ZoneInfo
from django.db.models import Q, Model # Django 查询和模型基类
import math
from functools import reduce
from typing import Dict, List, Any, Optional, Type, Union, TypeVar, Generic, Callable # 类型提示
from datetime import datetime, date, timedelta
import dateutil.parser
from django.db import connection, models # Django 模型
from decimal import Decimal

import numpy as np
import pandas as pd # 导入 Decimal
from stock_models.index import IndexInfo
from stock_models.stock_basic import StockInfo
import tushare as ts
from django.conf import settings # Django 设置
import pytz # 时区处理

from django.db import DatabaseError, IntegrityError, models, transaction # Django 数据库异常和事务
# 移除 django.core.cache 的导入，改用 CacheManager
# from django.core.cache import cache
from asgiref.sync import sync_to_async # 异步转换工具

# 导入自定义的 CacheManager
from utils.cache_manager import CacheManager

logger = logging.getLogger("dao") # 获取日志记录器

# 定义泛型类型变量 T，限定为 Django 模型
T = TypeVar('T', bound=models.Model)

class BaseDAO(Generic[T]):
    """
    基础数据访问对象 (DAO) 类。
    提供通用的异步 CRUD (创建、读取、更新、删除) 操作和基于 Redis 的缓存机制。
    旨在作为所有具体 DAO 类的基类，统一数据访问逻辑和缓存策略。
    使用 CacheManager 和全局 Redis 连接池。
    """

    def __init__(self, model_class: Optional[Type[T]] = None, api_service: Any = None, cache_timeout: int = 3600):
        """
        初始化 BaseDAO。
        Args:
            model_class: 此 DAO 主要操作的 Django 模型类。可以为 None，表示 DAO 管理多个模型。
            api_service: (可选) 用于从外部 API 获取数据的服务实例。
            cache_timeout: (可选) 默认缓存超时时间（秒），默认为 3600 秒 (1 小时)。
        """
        self.model_class = model_class
        self.api_service = api_service # API 服务实例，子类可以覆盖或使用
        self.cache_timeout = cache_timeout # 默认缓存超时时间
        self.ts_pro = ts.pro_api(settings.API_LICENCES_TUSHARE)
        self.ts = ts.set_token(settings.API_LICENCES_TUSHARE)

        # 只有当 model_class 不为 None 时才设置 model_name，用于生成缓存键前缀
        self.model_name = model_class._meta.model_name if model_class else "multi_model"

        # 初始化 CacheManager 为 None，按需创建
        self.cache_manager: Optional[CacheManager] = None

    def _ensure_cache_objects(self):
        """
        (内部方法) 确保 CacheManager 实例存在。
        如果实例不存在，则创建它。CacheManager 内部使用全局 Redis 连接池。
        """
        if self.cache_manager is None:
            self.cache_manager = CacheManager()
            logger.debug(f"CacheManager instance created for DAO (model: {self.model_name}).")

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
    async def _save_all_to_db_native_upsert(self, model_class, data_list: list, unique_fields: list, batch_size=5000):
        """
        【V10 - 修正版】使用原生SQL实现高效的批量更新或插入（Upsert）。
        此版本修正了获取外键目标字段名的错误，使用 Django 提供的 f.target_field 属性，
        使其真正健壮和正确。
        """
        # 1. 初始检查和准备
        if not data_list:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        total_records = len(data_list)
        failed_count = 0

        # 2. 字段元数据分析
        all_model_fields = {f.name for f in model_class._meta.get_fields() if getattr(f, 'editable', True) and not f.auto_created}
        unique_field_set = set(unique_fields)
        for field_name in unique_fields:
            if field_name.endswith('_id'):
                unique_field_set.add(field_name[:-3])
        pk_name = model_class._meta.pk.name
        update_fields = [f for f in all_model_fields if f not in unique_field_set and f != pk_name]

        # ▼▼▼【核心代码修改】: 修正获取外键目标字段名的方式 ▼▼▼
        # 解释: 旧代码使用了不存在的 `f.to_field` 属性，导致了 AttributeError。
        #      新的正确代码使用 `f.target_field.name`。
        #      `f.target_field` 是 Django 提供的标准API，它直接返回外键所指向的目标字段对象。
        #      无论 ForeignKey 是否定义了 `to_field` 参数，`f.target_field` 都会正确地指向目标字段（自定义字段或默认主键）。
        #      `.name` 则获取该字段的名称字符串。
        fk_fields_map = {}
        # print("--- [DAO调试] 开始动态构建外键映射 ---")
        for f in model_class._meta.fields:
            if isinstance(f, models.ForeignKey):
                # 旧的错误代码: target_field_name = f.to_field or f.related_model._meta.pk.name
                # 新的正确代码:
                target_field_name = f.target_field.name
                fk_fields_map[f.name] = target_field_name
                # print(f"调试信息: [BaseDAO] 发现外键 '{f.name}'，映射到原始数据键 '{target_field_name}'。")
        # print("--- [DAO调试] 外键映射构建完成 ---")
        # ▲▲▲【核心代码修改】: 修改结束 ▲▲▲

        # 3. 准备数据记录，将字典转换为包含ORM对象的字典
        prepared_data_list = []
        failed_records = []
        for record in data_list:
            try:
                sanitized_record = self._sanitize_for_json(record)
                instance_data = await self._prepare_model_instance(model_class, sanitized_record, fk_fields_map)
                prepared_data_list.append(instance_data)
            except Exception as e:
                logger.error(f"在准备模型实例数据时出错: {e}, 记录: {record}", exc_info=True)
                failed_records.append(record)
        failed_count += len(failed_records)
        if not prepared_data_list:
            logger.warning("所有记录在准备阶段均失败，不执行数据库操作。")
            return {"尝试处理": total_records, "失败": failed_count, "创建/更新成功": 0}

        # 4. 将准备好的数据转换为DataFrame
        df = pd.DataFrame(prepared_data_list)

        # 5. 转换关系字段列 (这部分逻辑已是动态的，无需修改)
        print("--- [DAO调试] 开始转换DataFrame中的关系字段 ---")
        for field in model_class._meta.get_fields():
            if field.is_relation and not field.auto_created:
                field_name = field.name
                column_name = field.column
                if field_name in df.columns:
                    print(f"调试信息: [BaseDAO] 发现关系列 '{field_name}'，将转换为数据库列 '{column_name}'。")
                    df[column_name] = df[field_name].apply(lambda x: x.pk if pd.notna(x) else None)
                    df = df.drop(columns=[field_name])
                    # print(f"调试信息: [BaseDAO] 转换完成。新列 '{column_name}' 已创建，旧列 '{field_name}' 已删除。")
        print("--- [DAO调试] DataFrame关系字段转换完成 ---")

        # 6. 调用下游异步批处理方法
        try:
            success_count = await self.process_batch_async(
                df=df,
                model_class=model_class,
                update_fields=update_fields,
                unique_key_fields=unique_fields,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"调用 process_batch_async 时发生未知错误: {e}", exc_info=True)
            failed_count += len(df)
            success_count = 0

        # 7. 返回结果
        return {"尝试处理": total_records, "失败": failed_count, "创建/更新成功": success_count}

    async def _prepare_model_instance(self, model_class, prepared_data, fk_fields_map):
        """
        【V6 - 终极健壮版】- 此方法保持不变
        准备单个模型实例的数据字典。
        核心功能：处理外键，将 'stock_code' 这样的字段或 {'stock': {'stock_code': ...}} 这样的结构，统一转换为实际的 'stock' 对象。
        """
        for fk_field_name, code_field_name in fk_fields_map.items():
            code_value = None
            if fk_field_name in prepared_data and isinstance(prepared_data[fk_field_name], dict):
                fk_dict = prepared_data[fk_field_name]
                code_value = fk_dict.get(code_field_name)
            elif code_field_name in prepared_data:
                code_value = prepared_data.pop(code_field_name)

            if code_value is not None:
                if code_value is None:
                    prepared_data[fk_field_name] = None
                    continue

                fk_model = model_class._meta.get_field(fk_field_name).related_model
                fk_instance = await self.get_or_create_fk_instance(fk_model, code_value, prepared_data)

                if fk_instance is None:
                    error_msg = (
                        f"为外键字段 '{fk_field_name}' 准备实例失败！"
                        f"无法为代码 '{code_value}' 找到或创建对应的 '{fk_model.__name__}' 实例。"
                    )
                    raise ValueError(error_msg)
                
                prepared_data[fk_field_name] = fk_instance

        model_field_names = {f.name for f in model_class._meta.get_fields()}
        cleaned_data = {k: v for k, v in prepared_data.items() if k in model_field_names}
        return cleaned_data

    # 【代码修改处】重构了异步处理方法，以调用新的同步原生SQL方法
    async def process_batch_async(self, df: pd.DataFrame, model_class, update_fields: list, unique_key_fields: list, batch_size=1000):
        """
        【修改版】异步处理批量数据，确保所有参数都被正确传递。
        """
        if df.empty:
            return 0
        
        total_processed = 0
        # 代码修改处: 在日志中也加入 unique_key_fields 的信息，便于调试
        logger.info(f"开始异步批处理 {len(df)} 条数据到表 {model_class._meta.db_table}。更新字段: {update_fields}, 唯一键: {unique_key_fields}")
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            try:
                # 使用 sync_to_async 来在异步事件循环中运行同步的数据库操作
                # 代码修改处: 将 unique_key_fields 参数传递给同步方法
                processed_count = await sync_to_async(self._process_batch_mysql_upsert_sync)(
                    df=batch_df,
                    model_class=model_class,
                    update_fields=update_fields,
                    unique_key_fields=unique_key_fields  # 将参数传递下去
                )
                if processed_count is not None:
                    total_processed += processed_count
            except Exception as e:
                logger.error(f"原生SQL批处理时遇到意外错误: {e}", exc_info=True)
                # 根据业务需求，可以选择在这里继续处理下一个批次或直接中断
                # continue or break
        
        logger.info(f"异步批处理完成，共处理 {total_processed} 条记录。")
        return total_processed

    # 【代码新增处】这是全新的、替代 process_batch_sync_for_mysql 的方法
    def _process_batch_mysql_upsert_sync(self, df: pd.DataFrame, model_class, update_fields: list, unique_key_fields: list) -> int:
        """
        【V17 - 终极完美版】在同步环境中，使用原生SQL处理一个批次的数据。
        此版本在V16的可靠执行策略基础上，增加了对JSON字段的显式序列化处理，解决了(3140)错误。
        - 策略: 保持V16的事务中逐行执行策略，确保执行的健壮性。
        - 新增: 在数据准备阶段，自动识别模型中的JSONField，并使用 `json.dumps` 将对应的DataFrame列中的Python对象（list/dict）转换为JSON字符串。
        - 结果: 保证了传递给数据库驱动的每一种数据类型都完全符合MySQL的期望，是功能最完整、最可靠的最终版本。
        """
        if df.empty:
            return 0

        table_name = model_class._meta.db_table
        all_columns = list(df.columns)

        # 数据预处理步骤 (保持对NULL值的处理)
        print("--- [DAO调试] 开始数据预处理：使用模型默认值填充缺失数据 ---")
        for field in model_class._meta.fields:
            if field.default is not models.NOT_PROVIDED and field.column in df.columns and df[field.column].isnull().any():
                default_value = field.get_default()
                if default_value is not None:
                    df[field.column].fillna(default_value, inplace=True)
                    print(f"调试信息: [BaseDAO] 在列 '{field.column}' 中发现缺失值，已使用模型默认值 '{default_value}' 进行填充。")
        
        # ▼▼▼【核心代码修改】: 显式进行JSON序列化 ▼▼▼
        # 1. 识别模型中所有的JSON字段
        json_field_names = [
            field.column for field in model_class._meta.fields
            if isinstance(field, models.JSONField)
        ]
        print(f"--- [DAO调试] 发现模型中的JSON字段: {json_field_names} ---")

        # 2. 对DataFrame中对应的JSON列进行序列化
        for col_name in json_field_names:
            if col_name in df.columns:
                print(f"--- [DAO调试] 正在对JSON字段 '{col_name}' 进行序列化处理... ---")
                # 使用.apply()将非空的、非字符串的列表或字典对象转换为JSON字符串
                # `ensure_ascii=False` 保证中文字符能被正确处理
                df[col_name] = df[col_name].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
                )
        print("--- [DAO调试] JSON字段序列化完成 ---")
        # ▲▲▲【核心代码修改】: 修改结束 ▲▲▲

        # 构建用于单行操作的SQL模板 (逻辑同V16)
        cols_sql = ", ".join([f"`{col}`" for col in all_columns])
        placeholders_sql = f"({', '.join(['%s'] * len(all_columns))})"
        
        if not update_fields:
            update_sql = f"`{model_class._meta.pk.name}` = `{model_class._meta.pk.name}`"
        else:
            update_sql = ", ".join([f"`{field}` = VALUES(`{field}`)" for field in update_fields])

        final_sql_template = (
            f"INSERT INTO `{table_name}` ({cols_sql}) "
            f"VALUES {placeholders_sql} "
            f"ON DUPLICATE KEY UPDATE {update_sql}"
        )
        print(f"--- [DAO调试] 生成的单行SQL模板: {final_sql_template} ---")

        # 准备参数列表，现在DataFrame中的JSON列已经是字符串了
        params_list_of_tuples = [tuple(row) for row in df.replace({np.nan: None}).to_numpy()]
        
        total_affected_rows = 0
        try:
            # 在事务中逐行执行 (逻辑同V16)
            with transaction.atomic():
                with connection.cursor() as cursor:
                    for i, params_tuple in enumerate(params_list_of_tuples):
                        try:
                            cursor.execute(final_sql_template, params_tuple)
                            total_affected_rows += cursor.rowcount
                        except Exception as inner_e:
                            # 增加行内错误调试信息
                            print(f"--- [DAO调试] 在处理第 {i+1} 行数据时出错。数据: {params_tuple} ---")
                            raise inner_e # 重新抛出内部异常
            return total_affected_rows
        except Exception as e:
            logger.error(f"执行批量Upsert时数据库出错。SQL模板: {final_sql_template[:500]}... Error: {e}", exc_info=True)
            raise

    @staticmethod
    @sync_to_async
    def _get_or_create_fk_sync(fk_model: Type[models.Model], code_value: str) -> models.Model | None:
        """
        【新增】同步的辅助方法，用于执行数据库的 get_or_create 操作。
        这是被 sync_to_async 包装的核心，确保数据库调用在同步线程中执行。
        """
        # 关键假设：我们假设您的 StockInfo 模型中，用来唯一标识股票的字段名叫 'stock_code'。
        # 如果字段名是 'code' 或其他名称，请务必修改下面这行。
        lookup_field = 'stock_code'
        
        try:
            # 使用 Django ORM 的 get_or_create，它会原子性地尝试获取，如果不存在则创建。
            # 它返回一个元组 (instance, created_boolean)。
            instance, created = fk_model.objects.get_or_create(
                **{lookup_field: code_value},
                # 如果需要创建，可以提供默认值
                # defaults={'stock_name': '未知', ...} 
            )
            if created:
                # 如果是新创建的，打印一条日志，方便调试。
                # print(f"调试信息: [FK-Sync] 在 '{fk_model.__name__}' 表中新创建了记录: {code_value}")
                pass
            return instance
        except Exception as e:
            # 捕获可能的数据库错误或其他问题，并记录详细日志。
            logger.error(f"在 _get_or_create_fk_sync 中为代码 '{code_value}' 操作 '{fk_model.__name__}' 时出错: {e}", exc_info=True)
            return None # 出错时返回 None，上层会捕获并抛出 ValueError

    async def get_or_create_fk_instance(self, fk_model: Type[models.Model], code_value: str, prepared_data: dict) -> models.Model | None:
        """
        【V5 最终修正版 - 健壮版】
        异步获取或创建外键实例。
        它调用一个被 @sync_to_async 包装的同步方法来安全地与数据库交互。
        这个版本简化了接口，直接接收 code_value。
        """
        # 调用我们上面定义的、被包装的同步方法
        return await self._get_or_create_fk_sync(fk_model, code_value)

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

        def fix_bj_time(dt):
            """
            检查并修正为标准北京时间（+08:00），如不是则强制转换。
            """
            if isinstance(dt, datetime):
                offset = dt.tzinfo.utcoffset(dt) if dt.tzinfo else None
                tzkey = getattr(dt.tzinfo, 'key', None) or str(dt.tzinfo)
                # 只要不是Asia/Shanghai或offset不是+8小时，都强制修正
                if offset != timedelta(hours=8) or tzkey != 'Asia/Shanghai':
                    # print(f"警告：检测到非标准北京时间，已强制修正为+08:00，原tzinfo: {dt.tzinfo}, offset: {offset}, tzkey: {tzkey}")
                    dt = dt.replace(tzinfo=None)
                    dt = dt.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            return dt

        if isinstance(value, datetime):
            # 如果已经是 datetime 对象，确保其时区正确
            if settings.USE_TZ:
                dt = timezone.make_aware(value, tz) if timezone.is_naive(value) else value.astimezone(tz)
            else:
                dt = timezone.make_naive(value, tz) if timezone.is_aware(value) else value
            dt = fix_bj_time(dt)  # 校正时区
            return dt

        if isinstance(value, date) and not isinstance(value, datetime):
            # 如果是 date 对象，转换为 datetime (午夜)
            dt = datetime.combine(value, datetime.min.time())
            dt = timezone.make_aware(dt, tz) if settings.USE_TZ else dt
            dt = fix_bj_time(dt)  # 校正时区
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
                dt = fix_bj_time(dt)  # 校正时区
                return dt
            except ValueError:
                pass # 继续尝试

            # 尝试指定格式
            if default_format:
                try:
                    dt = datetime.strptime(value, default_format)
                    dt = timezone.make_aware(dt, tz) if settings.USE_TZ else dt
                    dt = fix_bj_time(dt)  # 校正时区
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
                    dt = fix_bj_time(dt)  # 校正时区
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
            dt = fix_bj_time(dt)  # 校正时区
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










