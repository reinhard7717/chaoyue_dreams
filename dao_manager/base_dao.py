# -*- coding: utf-8 -*-
# dao_manager/base_dao.py

import decimal # 用于高精度数字
import json
import logging
import asyncio # 用于异步操作
from django.utils import timezone
from django.db.models import Q, Model # Django 查询和模型基类
import operator
from functools import reduce
from typing import Dict, List, Any, Optional, Type, Union, TypeVar, Generic, Callable # 类型提示
from datetime import datetime, date # 日期时间类型
from decimal import Decimal # 导入 Decimal
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
    # 保留 Django 5 原生 Upsert 方法作为推荐实现
    async def _save_all_to_db_native_upsert( self, model_class: Type[models.Model], data_list: List[Dict[str, Any]],
        unique_fields: List[str], # 用于冲突检测的字段 (应有唯一约束)
        **extra_fields: Any,
    ) -> Dict[str, int]:
        """
        (内部方法/异步) 使用 Django 5+ 原生的 bulk_create 实现 Upsert (Update or Create) 的批量处理方法。
        这是推荐的批量保存方式，性能较好。
        Args:
            model_class: 要操作的 Django 模型类。
            data_list: 包含待处理数据的字典列表。
            unique_fields: 用于确定唯一记录并检测冲突的字段列表 (必须在数据库中有唯一约束)。
            **extra_fields: (可选) 额外的字段和值，将添加到所有创建/更新的记录中。
        Returns:
            一个字典，包含处理结果的统计信息:
                - "尝试处理": 本次调用尝试处理的总记录数。
                - "失败": 处理过程中失败的记录数 (通常是整个批次失败)。
                - "创建/更新成功": 成功创建或更新的记录数 (原生方法不区分两者)。
        """
        if not data_list:
            logger.warning(f"未提供任何数据用于处理 - {model_class.__name__}")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        if not isinstance(data_list, list):
            data_list = [data_list]

        total_attempted = len(data_list)
        failed_count = 0

        # --- 确定需要更新的字段 (update_fields) ---
        all_keys = set()
        if data_list:
            # 从第一个数据项获取基础键，并合并 extra_fields 的键
            all_keys = set(data_list[0].keys()) | set(extra_fields.keys())
        # 排除掉用于匹配唯一性的字段
        update_fields = list(all_keys - set(unique_fields))
        # 可选：移除主键，因为通常不更新主键
        pk_name = model_class._meta.pk.name
        if pk_name in update_fields:
            update_fields.remove(pk_name)

        if not update_fields:
            logger.warning(
                f"模型 {model_class.__name__} 没有可用于更新的字段（除了唯一字段 {unique_fields}）。"
                f" bulk_create 在 update_conflicts=True 模式下仍会尝试创建新记录，但现有记录不会被修改。"
            )

        batch_size = 2000 # 定义每个数据库事务处理的批次大小
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]
            current_batch_size = len(batch)

            # --- 准备模型实例列表 ---
            objs_to_process = []
            for item in batch:
                # 合并 extra_fields 到每个 item
                prepared_data = {**item, **extra_fields}
                # 创建模型实例
                # 注意：这里假设 prepared_data 中的字段名和类型与模型匹配
                # 如果 API 返回的数据需要转换 (如日期字符串、数字格式)，应在调用此方法前完成
                try:
                    objs_to_process.append(model_class(**prepared_data))
                except Exception as model_init_err:
                    logger.error(f"创建模型 {model_class.__name__} 实例失败: {model_init_err}, data: {prepared_data}", exc_info=True)
                    failed_count += 1 # 单条记录创建失败

            # 如果当前批次所有记录都创建失败，则跳过数据库操作
            if len(objs_to_process) == 0 and current_batch_size > 0:
                 logger.error(f"批次 {i // batch_size + 1} 所有记录模型实例化失败，跳过数据库操作")
                 # failed_count 已经在上面累加
                 continue # 处理下一个批次

            # 使用 sync_to_async 执行同步的数据库操作
            @sync_to_async
            @transaction.atomic # 确保每个批次在事务中执行
            def process_batch_sync():
                nonlocal failed_count # 允许内部函数修改外部的 failed_count
                try:
                    # --- 调用 Django 原生 bulk_create 实现 Upsert ---
                    model_class.objects.bulk_create(
                        objs_to_process,
                        update_conflicts=True,       # 启用 Upsert 模式
                        # unique_fields=unique_fields, # 指定用于冲突检测的唯一字段
                        update_fields=update_fields, # 指定冲突时需要更新的字段
                        batch_size=current_batch_size # 处理当前批次的所有对象
                    )
                    # logger.info(f"批次 {i // batch_size + 1} 处理成功")
                except (IntegrityError, DatabaseError) as e:
                    # 捕获数据库层面的错误 (例如唯一约束未设置、外键问题等)
                    logger.error(
                        f"批次 {i // batch_size + 1} (大小: {len(objs_to_process)}) "
                        f"使用原生 bulk_create (upsert) 时遇到数据库错误: {str(e)}",
                        exc_info=True
                    )
                    failed_count += len(objs_to_process) # 整个批次标记为失败
                except Exception as e:
                    # 捕获其他潜在错误
                    logger.error(
                        f"批次 {i // batch_size + 1} (大小: {len(objs_to_process)}) "
                        f"使用原生 bulk_create (upsert) 时遇到意外错误: {str(e)}",
                        exc_info=True
                    )
                    failed_count += len(objs_to_process) # 整个批次标记为失败

            # 执行异步包裹的数据库操作
            await process_batch_sync()

        # --- 计算最终结果 ---
        successful_count = total_attempted - failed_count
        result = {
            "尝试处理": total_attempted,
            "失败": failed_count,
            "创建/更新成功": successful_count, # 原生方法不区分创建和更新
        }
        # logger.info(f"完成 {model_class.__name__} 数据批量处理 (使用原生 bulk_create upsert): {result}")
        return result

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

    # ==================== 缓存管理方法 ====================

    async def refresh_cache(self, id_value: Any = None) -> None:
        """
        (异步) 刷新缓存。
        如果提供了 id_value，则只刷新该实体的缓存；否则刷新 'all' 缓存。
        刷新操作包括删除旧缓存，并尝试从数据库重新加载数据写入缓存。
        Args:
            id_value: (可选) 要刷新缓存的实体的主键值。
        """
        self._ensure_cache_objects()
        if id_value:
            # 刷新单个实体缓存
            cache_key = self._get_cache_key(f"id:{id_value}")
            try:
                await self.cache_manager.delete(cache_key)
                logger.debug(f"已删除实体缓存: {cache_key}")
                # 尝试重新从数据库加载并缓存
                await self.get_by_id(id_value) # get_by_id 内部会写缓存
                logger.info(f"已刷新实体缓存: {self.model_name} (ID: {id_value})")
            except Exception as e:
                 logger.error(f"刷新实体缓存 {cache_key} 时出错: {e}", exc_info=True)
        else:
            # 刷新 'all' 缓存
            cache_key_all = self._get_cache_key("all")
            try:
                await self.cache_manager.delete(cache_key_all)
                logger.debug(f"已删除 'all' 缓存: {cache_key_all}")
                # 尝试重新从数据库加载并缓存
                await self.get_all() # get_all 内部会写缓存
                logger.info(f"已刷新所有实体缓存 for {self.model_name}")
            except Exception as e:
                 logger.error(f"刷新 'all' 缓存 {cache_key_all} 时出错: {e}", exc_info=True)

    async def clear_cache(self) -> None:
        """
        (异步) 清除与此 DAO 模型相关的所有已知缓存键 ('all' 和可能的 ID 键)。
        注意：此方法可能无法清除所有 filter 缓存键。如果需要更彻底的清除，
              建议使用 Redis 的 SCAN 命令配合模式匹配（需要在 CacheManager 中实现）。
        """
        self._ensure_cache_objects()
        cache_key_all = self._get_cache_key("all")
        logger.warning(f"准备清除模型 {self.model_name} 的 'all' 缓存 (key: {cache_key_all})...")
        try:
            await self.cache_manager.delete(cache_key_all)
            logger.info(f"已清除 'all' 缓存 for {self.model_name}")
            # 简化处理：不再尝试使用 keys 命令删除模式匹配的键，因为这在生产环境有风险
            # 如果确实需要，应在 CacheManager 中实现基于 SCAN 的 delete_pattern 方法
        except Exception as e:
            logger.error(f"清除 'all' 缓存 {cache_key_all} 时出错: {str(e)}", exc_info=True)

    # ==================== 数据解析工具方法 ====================

    def _parse_datetime(self, value: Any, default_format: Optional[str] = None) -> Optional[datetime]:
        """
        (内部方法/同步) 解析各种格式的日期时间值，返回 timezone-aware 或 naive datetime 对象。
        Args:
            value: 要解析的日期时间值 (字符串、时间戳、datetime 对象等)。
            default_format: (可选) 指定输入字符串的格式 (例如 '%Y-%m-%d %H:%M:%S')。
        Returns:
            解析后的 datetime 对象，如果解析失败则返回 None。
        """
        # 获取项目配置的时区，如果未配置则使用 UTC
        tz_name = getattr(settings, 'TIME_ZONE', 'UTC')
        try:
            tz = pytz.timezone(tz_name)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"未知的时区名称 '{tz_name}'，将使用 UTC。")
            tz = pytz.UTC

        if isinstance(value, datetime):
            # 如果已经是 datetime 对象，确保其时区正确
            if settings.USE_TZ:
                return timezone.make_aware(value, tz) if timezone.is_naive(value) else value.astimezone(tz)
            else:
                return timezone.make_naive(value, tz) if timezone.is_aware(value) else value
        if isinstance(value, date) and not isinstance(value, datetime):
             # 如果是 date 对象，转换为 datetime (午夜)
             dt = datetime.combine(value, datetime.min.time())
             return timezone.make_aware(dt, tz) if settings.USE_TZ else dt

        if value is None or str(value).strip() in ['', '-', 'N/A', '暂无']:
            return None # 处理空值

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

        # 尝试解析字符串
        if isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
                try: value = value.decode('utf-8')
                except UnicodeDecodeError: return None
            value = value.strip()

            # 尝试 ISO 格式
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                if settings.USE_TZ:
                    return timezone.make_aware(dt, tz) if timezone.is_naive(dt) else dt.astimezone(tz)
                else:
                    return timezone.make_naive(dt, tz) if timezone.is_aware(dt) else dt
            except ValueError:
                pass # 继续尝试

            # 尝试指定格式
            if default_format:
                try:
                    dt = datetime.strptime(value, default_format)
                    return timezone.make_aware(dt, tz) if settings.USE_TZ else dt
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
                    return timezone.make_aware(dt, tz) if settings.USE_TZ else dt
                except ValueError:
                    continue

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

