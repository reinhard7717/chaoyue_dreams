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
import functools
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
        【V2.1 - 避免N+1查询版】
        将模型实例或字典转换为适合缓存的字典格式。
        优化：默认情况下使用 field.attname 获取外键ID，避免访问关联字段时触发额外的数据库查询。
        """
        if isinstance(data, models.Model):
            try:
                data_dict = {}
                for field in data._meta.fields:
                    field_name = field.name
                    # 处理外键
                    if field.is_relation:
                        if related_field_map and field_name in related_field_map:
                            # 特殊情况：需要关联对象的具体属性 (会触发DB查询)
                            value = getattr(data, field_name)
                            if value is not None:
                                related_attr = related_field_map[field_name]
                                if hasattr(value, related_attr):
                                    data_dict[field_name] = getattr(value, related_attr)
                                else:
                                    data_dict[field_name] = value.pk
                            else:
                                data_dict[field_name] = None
                        else:
                            # 默认情况：只取主键，使用 attname (如 stock_id) 避免触发 DB 查询
                            # field.attname 通常是 "field_id"
                            data_dict[field_name] = getattr(data, field.attname)
                    else:
                        # 常规字段
                        data_dict[field_name] = getattr(data, field_name)
            except Exception as e:
                logger.error(f"从模型实例 {type(data)} 提取数据失败: {e}", exc_info=True)
                return None
        elif isinstance(data, dict):
            data_dict = data.copy()
        else:
            logger.error(f"不支持的数据类型进行缓存准备: {type(data)}")
            return None
        cache_dict = {}
        for key, value in data_dict.items():
            if value is None:
                continue
            if isinstance(value, Decimal):
                cache_dict[key] = str(value)
            elif isinstance(value, datetime):
                if settings.USE_TZ and timezone.is_naive(value):
                    value = timezone.make_aware(value, timezone.get_current_timezone())
                elif not settings.USE_TZ and timezone.is_aware(value):
                    value = timezone.make_naive(value)
                cache_dict[key] = value.isoformat()
            elif isinstance(value, date):
                cache_dict[key] = value.isoformat()
            elif isinstance(value, (list, tuple, dict)):
                 try:
                     self._ensure_cache_objects()
                     json.dumps(value)
                     cache_dict[key] = value
                 except TypeError:
                     cache_dict[key] = str(value)
            else:
                cache_dict[key] = value
        return cache_dict

    async def _batch_prepare_data_for_cache(self, instances: List[models.Model]) -> List[Dict]:
        """
        【V1.0 - 向量化缓存准备】
        批量将模型实例列表转换为缓存字典列表。
        利用 Pandas 向量化处理 Decimal 转 str 和 Date 转 ISO 格式，比循环处理快数倍。
        """
        if not instances:
            return []
        
        try:
            model_class = instances[0].__class__
            fields = model_class._meta.fields
            # 1. 快速提取数据 (使用 attname 避免外键查询)
            # 构建一个包含所有字段数据的列表，外键直接取 ID
            raw_data = []
            for obj in instances:
                record = {}
                for field in fields:
                    # 优先使用 attname (如 stock_id) 获取原始值，避免触发外键查询
                    record[field.name] = getattr(obj, field.attname)
                raw_data.append(record)
            # 2. 转为 DataFrame 进行向量化处理
            df = pd.DataFrame(raw_data)
            # 3. 向量化类型转换
            for field in fields:
                field_name = field.name
                if field_name not in df.columns:
                    continue
                # 处理 Decimal -> String
                if isinstance(field, models.DecimalField):
                    # 转换为字符串，处理 None
                    df[field_name] = df[field_name].apply(lambda x: str(x) if x is not None else None)
                # 处理 DateTime/Date -> ISO String
                elif isinstance(field, (models.DateTimeField, models.DateField)):
                    # 使用 Pandas 的 dt 访问器加速格式化 (注意：需要先转为 datetime 类型以防混合)
                    # 这里假设数据已经是 datetime 对象或 None
                    # 批量转换为 ISO 格式
                    # 注意：Pandas 的 isoformat 可能带时区，需保持一致性，这里简化处理
                    # 更快的方式是 map(lambda x: x.isoformat() if x else None)
                    df[field_name] = df[field_name].map(lambda x: x.isoformat() if x else None)
            # 4. 清洗 NaN 并转回字典列表
            # where(pd.notnull(df), None) 将 NaN 替换为 None
            return df.where(pd.notnull(df), None).to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"批量准备缓存数据失败: {e}", exc_info=True)
            # 降级处理：如果批量处理失败，回退到逐个处理
            tasks = [self._prepare_data_for_cache(inst) for inst in instances]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]

    async def _batch_build_models_from_cache(self, model_class: Type[T], cached_list: List[Dict], related_dao_map: Optional[Dict[str, 'BaseDAO']] = None) -> List[T]:
        """
        【V1.0 - 向量化模型构建】
        从缓存字典列表批量构建模型实例。
        利用 Pandas 批量解析日期和数字，减少 Python 循环中的开销。
        """
        if not cached_list:
            return []
            
        try:
            # 1. 转为 DataFrame
            df = pd.DataFrame(cached_list)
            # 2. 向量化类型还原
            for field in model_class._meta.fields:
                field_name = field.name
                if field_name not in df.columns:
                    continue
                # 还原 Decimal
                if isinstance(field, models.DecimalField):
                    # 批量转为 Decimal，处理 None
                    # 注意：apply(Decimal) 在大量数据下可能仍有开销，但比纯循环略好，且代码更整洁
                    df[field_name] = df[field_name].apply(lambda x: Decimal(str(x)) if x is not None else None)
                # 还原 DateTime
                elif isinstance(field, models.DateTimeField):
                    # 使用 pd.to_datetime 批量解析 ISO 字符串
                    df[field_name] = pd.to_datetime(df[field_name], errors='coerce')
                    # 转换为 Python datetime 对象 (带时区处理)
                    if settings.USE_TZ:
                        tz = timezone.get_current_timezone()
                        df[field_name] = df[field_name].apply(
                            lambda x: timezone.make_aware(x, tz) if pd.notnull(x) and timezone.is_naive(x) else (x if pd.notnull(x) else None)
                        )
                    else:
                        df[field_name] = df[field_name].apply(
                            lambda x: timezone.make_naive(x) if pd.notnull(x) and timezone.is_aware(x) else (x if pd.notnull(x) else None)
                        )
                # 还原 Date
                elif isinstance(field, models.DateField):
                    df[field_name] = pd.to_datetime(df[field_name], errors='coerce').dt.date
                    # 处理 NaT (转为 None)
                    df[field_name] = df[field_name].where(pd.notnull(df[field_name]), None)
            # 3. 转换回字典列表用于实例化
            # replace({np.nan: None}) 确保所有 NaN 变回 None
            processed_data = df.where(pd.notnull(df), None).to_dict(orient='records')
            # 4. 实例化模型 (这一步仍需循环，因为 __init__ 不能向量化)
            instances = []
            # 预处理外键 DAO
            fk_tasks = []
            # 如果有外键需要处理，目前仍需逐个处理或后续优化为批量获取
            # 这里保持简单，若无 related_dao_map 则直接实例化
            if not related_dao_map:
                for data in processed_data:
                    try:
                        instances.append(model_class(**data))
                    except Exception:
                        continue
            else:
                # 如果有外键 DAO，回退到逐个构建以复用 _build_model_from_cache 的复杂外键逻辑
                # 但利用了已经解析好的数据类型
                tasks = [self._build_model_from_cache(model_class, data, related_dao_map) for data in processed_data]
                results = await asyncio.gather(*tasks)
                instances = [r for r in results if r is not None]
            return instances
        except Exception as e:
            logger.error(f"批量构建模型失败: {e}", exc_info=True)
            # 降级回退
            tasks = [self._build_model_from_cache(model_class, item, related_dao_map) for item in cached_list]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _get_model_field_map(model_class: Type[models.Model]) -> Dict[str, models.Field]:
        """
        【V1.0 - 元数据缓存】
        (静态内部方法) 获取模型字段名到字段对象的映射，带 LRU 缓存。
        避免在循环中重复访问 model._meta.fields。
        """
        return {field.name: field for field in model_class._meta.fields}

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _get_model_field_names_set(model_class: Type[models.Model]) -> frozenset:
        """
        【V1.0 - 元数据缓存】
        (静态内部方法) 获取模型字段名集合，带 LRU 缓存。
        使用 frozenset 优化 'in' 操作的查找速度。
        """
        return frozenset(field.name for field in model_class._meta.fields)

    async def _build_model_from_cache(self, model_class: Type[T], cached_data: Dict, related_dao_map: Optional[Dict[str, 'BaseDAO']] = None) -> Optional[T]:
        """
        【V2.2 - 查表优化版】
        (内部方法) 从缓存字典构建模型实例。
        优化：
        1. 使用 cached_field_map 替代 _meta.fields 遍历，复杂度由 O(Fields) 降为 O(CacheKeys)。
        2. 优化外键处理：如果没有提供 DAO，尝试直接设置 field_id (attname)，提高容错率。
        """
        if not cached_data or not isinstance(cached_data, dict):
            logger.debug(f"无效的缓存数据用于构建 {model_class.__name__}: {cached_data}")
            return None
        
        model_data = {}
        try:
            # 1. 获取缓存的字段映射表 (O(1) 访问)
            field_map = self._get_model_field_map(model_class)
            # 2. 遍历缓存数据
            for field_name, cached_value in cached_data.items():
                # 忽略不在模型中的字段 (过滤脏数据)
                if field_name not in field_map:
                    continue
                field = field_map[field_name]
                # 跳过 None 值 (如果字段不允许 Null，将在实例化时由 Django 抛出错误，这里保持原逻辑)
                if cached_value is None:
                    if field.null:
                        model_data[field_name] = None
                    continue
                # --- 处理各种字段类型 ---
                if field.is_relation:
                    # 检查是否有对应的 DAO 来获取关联对象
                    if related_dao_map and field_name in related_dao_map:
                        related_dao = related_dao_map[field_name]
                        try:
                            # 异步获取关联对象
                            related_obj = await related_dao.get_by_id(cached_value)
                            if related_obj:
                                model_data[field_name] = related_obj
                            elif not field.null:
                                # 必需关联对象缺失
                                logger.warning(f"构建 {model_class.__name__} 失败: 必需外键 {field_name} (ID: {cached_value}) 未找到")
                                return None
                        except Exception as e:
                            logger.error(f"获取关联对象 {field_name} 失败: {e}")
                            if not field.null: return None
                    else:
                        # 优化：没有 DAO 时，尝试直接设置 ID 字段 (如 stock_id)
                        # field.attname 通常是 "stock_id"，而 field.name 是 "stock"
                        if hasattr(field, 'attname') and field.attname != field.name:
                             model_data[field.attname] = cached_value
                        elif field.null:
                            model_data[field_name] = None
                        else:
                            # 必需外键且无法设置 ID，无法构建
                            logger.debug(f"缺少 DAO 且无法直接设置 ID for {field_name}")
                            return None
                elif isinstance(field, models.DecimalField):
                    try:
                        model_data[field_name] = Decimal(str(cached_value))
                    except:
                        if not field.null: return None
                elif isinstance(field, models.DateTimeField):
                    try:
                        dt = datetime.fromisoformat(str(cached_value))
                        if settings.USE_TZ and timezone.is_naive(dt):
                             dt = timezone.make_aware(dt, timezone.get_current_timezone())
                        elif not settings.USE_TZ and timezone.is_aware(dt):
                             dt = timezone.make_naive(dt)
                        model_data[field_name] = dt
                    except:
                        if not field.null: return None
                elif isinstance(field, models.DateField):
                    try:
                        model_data[field_name] = date.fromisoformat(str(cached_value))
                    except:
                        if not field.null: return None
                else:
                    # 基本类型直接赋值
                    model_data[field_name] = cached_value
            # 3. 实例化模型
            instance = model_class(**model_data)
            return instance
        except Exception as e:
            logger.error(f"从缓存构建 {model_class.__name__} 实例时发生未知错误: {e}", exc_info=True)
            return None

    def _sanitize_for_json(self, data: Any) -> Any:
        """
        【V2.1 - 向量化增强版】
        递归地清洗数据，使其完全兼容JSON序列化。
        新增：直接支持 pandas DataFrame 和 Series 的向量化清洗，大幅提升大数据量下的处理效率。
        - 将 Decimal 转换为 float。
        - 将 numpy 数字类型 (如 float64, int64) 转换为原生 Python 类型。
        - 将 NaN, NaT, pd.NA 转换为 None。
        """
        # 1. 向量化处理 Pandas 对象 (新增优化)
        if isinstance(data, pd.DataFrame):
            # 使用向量化方法一次性替换所有空值为 None
            # to_dict('records') 会返回 [{col: val}, ...] 格式
            return data.where(pd.notnull(data), None).to_dict(orient='records')
        if isinstance(data, pd.Series):
            # Series 转为字典
            return data.where(pd.notnull(data), None).to_dict()
        # 2. 递归处理常规 Python 结构
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_for_json(item) for item in data]
        # 3. 处理标量类型
        if isinstance(data, Decimal):
            return float(data)
        if isinstance(data, np.generic):
            return data.item()
        if pd.isna(data):
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
        【V2.1 - 批量优化版】
        (异步) 获取模型的所有实体列表。
        优化：使用批量方法处理缓存的读写，显著减少序列化/反序列化的时间开销。
        """
        if self.model_class is None:
            raise TypeError("model_class 未在 BaseDAO 初始化时设置，无法执行 get_all")
        
        instances = []
        cache_hit = False
        self._ensure_cache_objects()
        cache_key = self._get_cache_key("all")
        
        # 1. 先从缓存获取
        try:
            cached_list = await self.cache_manager.get(key=cache_key)
            if cached_list and isinstance(cached_list, list):
                logger.debug(f"缓存命中: {cache_key}, 命中 {len(cached_list)} 条")
                # 使用批量构建方法
                instances = await self._batch_build_models_from_cache(self.model_class, cached_list, related_dao_map)
                if len(instances) == len(cached_list):
                    cache_hit = True
                else:
                    logger.warning(f"部分缓存数据无效或构建模型失败，删除缓存 key: {cache_key}")
                    await self.cache_manager.delete(cache_key)
                    cache_hit = False
            else:
                logger.debug(f"缓存未命中: {cache_key}")
        except Exception as e:
            logger.error(f"从缓存获取所有 {self.model_name} 时发生异常: {e}", exc_info=True)
        
        # 2. 从数据库获取
        if not cache_hit:
            try:
                instances = await self._get_all_from_db()
                if instances:
                    logger.debug(f"数据库命中: 共{len(instances)}条 {self.model_name} 记录")
                    # --- 写入缓存 (使用批量准备方法) ---
                    try:
                        data_to_cache = await self._batch_prepare_data_for_cache(instances)
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
                             api_data_list = await self._fetch_all_from_api()
                             if api_data_list and isinstance(api_data_list, list):
                                 logger.warning(f"从 API 获取到 {len(api_data_list)} 条数据，但 BaseDAO 未实现保存逻辑，请在子类处理")
                                 instances = []
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
                          instances = []
            except Exception as e:
                logger.error(f"数据库查询所有 {self.model_name} 错误: {str(e)}", exc_info=True)
                instances = []
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
        【V4.1 - 预计算优化版】
        优化了字段元数据的获取逻辑，避免在循环中重复调用 _meta.get_field。
        """
        if not data_list:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        total_records = len(data_list)
        # 预计算字段映射信息，避免在循环中重复查找元数据
        # 结构: field_name -> (column_name, is_relation, target_field_name)
        field_info_map = {}
        for field in model_class._meta.fields:
            field_info_map[field.name] = (
                field.column,
                field.is_relation,
                field.target_field.name if field.is_relation else None
            )
        sql_ready_data_list = []
        for record in data_list:
            sql_record = {}
            for field_name, value in record.items():
                # 直接从预计算的 map 中获取信息，速度远快于 model_class._meta.get_field
                if field_name in field_info_map:
                    column_name, is_relation, target_field_name = field_info_map[field_name]
                    if is_relation and isinstance(value, models.Model):
                        # 获取外键引用的值
                        fk_value = getattr(value, target_field_name)
                        sql_record[column_name] = fk_value
                    else:
                        # 常规字段或外键已经是ID值的情况
                        sql_record[column_name] = value
                # 如果key不是模型的字段，则忽略
            sql_ready_data_list.append(sql_record)
        if not sql_ready_data_list:
            logger.warning("所有记录在准备阶段均失败或为空，不执行数据库操作。")
            return {"尝试处理": total_records, "失败": total_records, "创建/更新成功": 0}
        # 确定需要更新的数据库列
        # unique_fields 是模型字段名，需要转换为数据库列名
        unique_db_columns = set()
        for f in unique_fields:
            if f in field_info_map:
                unique_db_columns.add(field_info_map[f][0])
            else:
                unique_db_columns.add(f) # 容错
        all_db_columns_in_data = list(sql_ready_data_list[0].keys())
        update_db_columns = [col for col in all_db_columns_in_data if col not in unique_db_columns]
        try:
            success_count = await self.process_batch_async(
                model_class=model_class,
                data_list=sql_ready_data_list,
                unique_fields=unique_fields,
                update_fields=update_db_columns,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"调用 process_batch_async 时发生未知错误: {e}", exc_info=True)
            success_count = 0
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
        【V3.3 - 向量化加速版】
        使用 pandas 进行向量化数据处理，替代原有的双重 for 循环，大幅提升参数构建效率。
        保留了死锁重试机制。
        """
        total_affected_rows = 0
        if not data_list:
            logger.info(f"模型 {model_class.__name__} 的数据列表为空，跳过批量保存。")
            return total_affected_rows
        table_name = model_class._meta.db_table
        # 1. 确定所有涉及的数据库列名 (以第一条数据为准)
        all_db_columns = list(data_list[0].keys())
        # 2. 使用 Pandas 进行向量化处理，构建 params_list_of_tuples
        try:
            # 将字典列表转换为 DataFrame
            df = pd.DataFrame(data_list)
            # 确保 DataFrame 列顺序与 all_db_columns 一致，并处理可能缺失的列（虽然理论上 data_list 结构一致）
            # reindex 会自动引入 NaN 填充缺失值
            df = df.reindex(columns=all_db_columns)
            # 核心优化：向量化处理空值。将所有 NaN/NaT 替换为 Python None
            # where(cond, other): 当 cond 为 False 时，替换为 other
            df = df.where(pd.notnull(df), None)
            # 转换为列表的列表 (MySQL cursor 接受 list of lists 或 list of tuples)
            params_list_of_tuples = df.values.tolist()
        except Exception as e:
            logger.error(f"Pandas 向量化处理数据失败，回退到普通模式: {e}")
            # 回退逻辑：防止 pandas 处理异常导致整个保存失败
            params_list_of_tuples = []
            for item in data_list:
                param_tuple = []
                for col in all_db_columns:
                    value = item.get(col)
                    if pd.isna(value):
                        value = None
                    param_tuple.append(value)
                params_list_of_tuples.append(tuple(param_tuple))
        # 3. 准备 SQL 语句
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
        # 4. 执行数据库操作 (带重试机制)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with connection.cursor() as cursor:
                    if not params_list_of_tuples:
                        return 0
                    total_affected_rows = cursor.executemany(final_sql_template, params_list_of_tuples)
                return total_affected_rows
            except (IntegrityError, OperationalError, ProgrammingError) as e:
                if isinstance(e, OperationalError) and e.args[0] == 1213: # Deadlock
                    if attempt < max_retries - 1:
                        sleep_time = (0.1 * (2 ** attempt)) + random.uniform(0.05, 0.1)
                        logger.warning(
                            f"捕获到数据库死锁 (1213)，正在进行第 {attempt + 2}/{max_retries} 次重试... "
                            f"等待 {sleep_time:.2f} 秒。模型: {model_class.__name__}"
                        )
                        time.sleep(sleep_time)
                        continue
                logger.error(f"原生SQL批处理时遇到数据库错误 (在锁内): {e}")
                # 打印详细调试信息
                print("="*50)
                print(f"FATAL DEBUG: [base_dao] 批量 Upsert 失败 (尝试 {attempt + 1}/{max_retries})")
                print(f"FATAL DEBUG: 模型: {model_class.__name__}")
                print(f"FATAL DEBUG: 异常: {e}")
                print("="*50)
                return 0
            except Exception as e:
                logger.error(f"原生SQL批处理时遇到意外错误 (在锁内): {e}")
                return 0
        logger.error(f"模型 {model_class.__name__} 的批量 Upsert 在 {max_retries} 次尝试后均失败。")
        return 0

    async def _prepare_model_instance(self, model_class, prepared_data, fk_fields_map):
        """
        【V8.1 - 缓存优化版】
        准备单个模型实例的数据字典。
        优化：使用 lru_cache 缓存模型字段集合，避免重复计算。
        """
        # 1. 处理外键 (使用已优化的 get_or_create_fk_instance)
        for fk_field_name, code_field_name in fk_fields_map.items():
            code_value = None
            if fk_field_name in prepared_data and isinstance(prepared_data[fk_field_name], dict):
                fk_dict = prepared_data[fk_field_name]
                code_value = fk_dict.get(code_field_name)
            elif code_field_name in prepared_data:
                code_value = prepared_data[code_field_name]
            if code_value is not None:
                fk_model = model_class._meta.get_field(fk_field_name).related_model
                fk_instance = await self.get_or_create_fk_instance(fk_model, code_field_name, code_value)
                if fk_instance is None:
                    error_msg = (
                        f"为外键字段 '{fk_field_name}' 准备实例失败！"
                        f"无法为代码 '{code_value}' (使用字段 '{code_field_name}') 找到或创建对应的 '{fk_model.__name__}' 实例。"
                    )
                    raise ValueError(error_msg)
                prepared_data[fk_field_name] = fk_instance
        
        # 2. 过滤非模型字段
        # 优化：直接获取缓存的 frozenset，O(1) 获取，O(1) 查找
        model_field_names = self._get_model_field_names_set(model_class)
        
        cleaned_data = {k: v for k, v in prepared_data.items() if k in model_field_names}
        return cleaned_data

    @staticmethod
    @sync_to_async
    @functools.lru_cache(maxsize=4096)
    def _get_or_create_fk_sync(fk_model: Type[models.Model], lookup_field: str, code_value: str) -> models.Model | None:
        """
        【V2.1 - 缓存加速版】
        同步的辅助方法，用于执行数据库的 get_or_create 操作。
        优化：增加了 LRU 缓存，避免在批量处理相同外键（如同一只股票的多条行情）时重复查询数据库。
        注意：缓存基于参数哈希，适用于基础数据（Stock, User等）的查找。
        """
        try:
            # 使用 Django ORM 的 get_or_create
            # lru_cache 会缓存返回的模型实例
            instance, created = fk_model.objects.get_or_create(
                **{lookup_field: code_value},
            )
            return instance
        except Exception as e:
            logger.error(f"在 _get_or_create_fk_sync 中为代码 '{code_value}' (字段: {lookup_field}) 操作 '{fk_model.__name__}' 时出错: {e}", exc_info=True)
            return None

    async def _batch_prepare_model_instances(self, model_class: Type[T], data_list: List[Dict], 
                                             fk_config: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """
        【V1.0 - 向量化外键解析】
        批量准备模型数据，核心解决外键的批量解析问题 (Bulk Resolve)。
        替代循环调用 _prepare_model_instance，将 N 次 DB 查询降低为 1 次。
        
        Args:
            model_class: 目标模型类
            data_list: 原始数据字典列表
            fk_config: 外键配置字典，格式为:
                       {
                           'model_field_name': {
                               'model': RelatedModelClass,
                               'source_field': 'api_field_name',  # 数据源中的字段名
                               'lookup_field': 'db_column_name'   # 关联模型中的查找字段名
                           }
                       }
        Returns:
            清洗好并填充了外键对象的字典列表，可直接用于 _save_all_to_db_native_upsert
        """
        if not data_list:
            return []
        try:
            # 1. 转换为 DataFrame
            df = pd.DataFrame(data_list)
            # 2. 批量处理每个外键配置
            for model_field, config in fk_config.items():
                related_model = config['model']
                source_field = config.get('source_field', model_field) # 默认源字段同名
                lookup_field = config.get('lookup_field', 'pk')        # 默认查找主键
                if source_field not in df.columns:
                    continue
                # 提取所有非空的唯一代码
                unique_codes = df[source_field].dropna().unique().tolist()
                if not unique_codes:
                    continue
                # 3. 批量查询存在的对象 (Bulk Query)
                # 构造查询条件: {lookup_field__in: unique_codes}
                filter_kwargs = {f"{lookup_field}__in": unique_codes}
                # 异步查询数据库
                existing_objects = await sync_to_async(list)(
                    related_model.objects.filter(**filter_kwargs)
                )
                # 4. 构建映射字典: code -> model_instance
                # 注意：这里假设 lookup_field 的值在关联表中是唯一的
                code_to_obj_map = {getattr(obj, lookup_field): obj for obj in existing_objects}
                # 5. 处理缺失对象 (可选：如果需要自动创建)
                # 找出哪些代码在数据库中没找到
                found_codes = set(code_to_obj_map.keys())
                missing_codes = set(unique_codes) - found_codes
                if missing_codes:
                    logger.info(f"字段 {model_field} 有 {len(missing_codes)} 个关联对象未找到，尝试逐个创建...")
                    # 对于缺失的，回退到逐个创建 (无法批量创建并返回带ID的对象，除非用复杂SQL)
                    # 这里利用已有的带缓存的方法
                    for code in missing_codes:
                        obj = await self._get_or_create_fk_sync(related_model, lookup_field, code)
                        if obj:
                            code_to_obj_map[code] = obj
                # 6. 向量化映射回 DataFrame
                # map 可能会产生 NaN (如果没找到)，后续会被处理为 None
                df[model_field] = df[source_field].map(code_to_obj_map)
            # 7. 清理非模型字段并导出
            # 获取模型所有字段名
            model_fields = {f.name for f in model_class._meta.fields}
            # 只保留模型中存在的字段 (且已经在 DF 中的)
            final_columns = [col for col in df.columns if col in model_fields]
            df_final = df[final_columns]
            # 将 NaN 替换为 None 并转为字典列表
            return df_final.where(pd.notnull(df_final), None).to_dict(orient='records')
        except Exception as e:
            logger.error(f"批量准备模型实例失败: {e}", exc_info=True)
            return []

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
        【V2.1 - Pandas加速版】
        解析各种格式的日期时间值，返回 timezone-aware 或 naive datetime 对象。
        利用 pandas.to_datetime 进行底层 C 加速解析，替代低效的循环尝试。
        """
        # 获取项目配置的时区
        tz_name = getattr(settings, 'TIME_ZONE', 'UTC')
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")
        # 1. 快速处理 None 和空值
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            value = value.strip()
            if value in ['', '-', 'N/A', '暂无', 'null', 'None']:
                return None
        
        # 2. 快速处理已有对象
        if isinstance(value, datetime):
            if settings.USE_TZ:
                return timezone.make_aware(value, tz) if timezone.is_naive(value) else value.astimezone(tz)
            else:
                return timezone.make_naive(value, tz) if timezone.is_aware(value) else value
        
        if isinstance(value, date):
            dt = datetime.combine(value, datetime.min.time())
            if settings.USE_TZ:
                dt = timezone.make_aware(dt, tz)
            return dt
        # 3. 使用 Pandas 进行智能解析 (核心优化)
        try:
            # pd.to_datetime 能自动处理 ISO、时间戳(秒/毫秒/纳秒)、常见字符串格式
            # errors='coerce' 会将无法解析的转为 NaT
            pd_dt = pd.to_datetime(value, errors='coerce')
            if pd.isna(pd_dt):
                # Pandas 解析失败，尝试最后的正则修正 (处理特殊的 +08:xx 格式)
                if isinstance(value, str):
                    # 修正时区偏移量格式，例如 +08:05 -> +08:00
                    value_fixed = re.sub(r'([+-]\d{2}):\d{2}$', r'\1:00', value)
                    if value_fixed != value:
                        pd_dt = pd.to_datetime(value_fixed, errors='coerce')
            if pd.isna(pd_dt):
                return None
            # 转换为 Python datetime
            dt = pd_dt.to_pydatetime()
            # 4. 统一时区处理
            if settings.USE_TZ:
                # 如果解析出的时间是 naive，加上项目时区
                # 如果是 aware，转为项目时区
                if timezone.is_naive(dt):
                    return timezone.make_aware(dt, tz)
                else:
                    return dt.astimezone(tz)
            else:
                # 如果项目不使用时区，统一转为 naive
                return timezone.make_naive(dt, tz) if timezone.is_aware(dt) else dt
        except Exception as e:
            logger.warning(f"日期解析发生未预期错误: {value}, error: {e}")
            return None

    def _parse_number(self, value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
        """
        【V2.1 - 逻辑优化版】
        解析数字，处理各种格式的数字字符串，返回 Decimal 类型以保证精度。
        优化了执行路径，优先处理标准数字，减少正则开销。
        """
        if value is None:
            return default
        
        # 1. 快速路径：直接是数字类型
        if isinstance(value, (int, float)):
            return Decimal(str(value)) # 转 str 再转 Decimal 避免 float 精度问题
        if isinstance(value, Decimal):
            return value
            
        # 2. 字符串预处理
        value_str = str(value).strip()
        if not value_str or value_str in ['-', 'N/A', '暂无', 'null', 'None']:
            return default
        # 移除千位分隔符
        value_str = value_str.replace(',', '')
        
        # 3. 尝试直接转换 (最常见情况)
        try:
            # 处理百分号
            if value_str.endswith('%'):
                return Decimal(value_str[:-1]) / Decimal(100)
            return Decimal(value_str)
        except decimal.InvalidOperation:
            pass # 继续尝试复杂解析
        # 4. 复杂单位处理 (万/亿)
        try:
            # 使用文件头导入的 re，不再在方法内导入
            # 匹配数字部分
            match = re.search(r'([-+]?\d*\.?\d+)', value_str)
            if not match:
                return default
            number = Decimal(match.group(1))
            # 处理中文单位
            if '万亿' in value_str or '兆' in value_str:
                number *= Decimal('1000000000000')
            elif '亿' in value_str:
                number *= Decimal('100000000')
            elif '万' in value_str:
                number *= Decimal('10000')
            elif '千' in value_str:
                number *= Decimal('1000')
            # 再次检查百分号 (防止 "10%万" 这种怪异数据，虽然少见)
            if '%' in value_str and not value_str.endswith('%'): 
                number /= Decimal(100)
            return number
        except Exception as e:
            logger.warning(f"解析数字失败: {value}, 错误: {e}")
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
    @staticmethod
    def get_this_monday_and_friday():
        """
        【V1.1 - 静态方法修复】
        获取本周一和本周五的日期
        """
        today = datetime.date.today()
        this_monday = today - datetime.timedelta(days=today.weekday())
        this_friday = this_monday + datetime.timedelta(days=4)
        return this_monday, this_friday

    @staticmethod
    def get_last_monday_and_friday():
        """
        【V1.1 - 静态方法修复】
        获取上周一和上周五的日期
        """
        today = datetime.date.today()
        this_monday = today - datetime.timedelta(days=today.weekday())
        last_monday = this_monday - datetime.timedelta(days=7)
        last_friday = last_monday + datetime.timedelta(days=4)
        return last_monday, last_friday











