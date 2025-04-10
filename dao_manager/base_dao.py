# dao_manager/basedao.py

import decimal
import json
import logging
from django.db.models import Q
import operator
from functools import reduce
from typing import Dict, List, Any, Optional, Type, Union, TypeVar, Generic
from datetime import datetime

from django.conf import settings
import pytz

from django.db import DatabaseError, IntegrityError, models
from django.core.cache import cache
from django.conf import settings
from django.db import transaction
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)

# 定义泛型类型变量
T = TypeVar('T', bound=models.Model)

class BaseDAO(Generic[T]):
    """
    基础数据访问对象(DAO)类，提供通用的CRUD操作和缓存机制
    实现三层缓存架构：
    1. Redis缓存
    2. MySQL数据库
    3. 外部API调用
    """
    
    def __init__(self, model_class: Optional[Type[T]] = None, api_service=None, cache_timeout: int = 3600):
        """
        初始化DAO
        
        Args:
            model_class: 模型类，可以为None，表示这是一个管理多个模型的DAO
            api_service: API服务实例
            cache_timeout: 缓存超时时间(秒)
        """
        self.model_class = model_class
        self.api_service = api_service
        self.cache_timeout = cache_timeout
        # 只有当model_class不为None时才设置model_name
        self.model_name = model_class._meta.model_name if model_class else "multi_model"
    
    def _get_cache_key(self, key_suffix: str) -> str:
        """
        生成缓存键
        
        Args:
            key_suffix: 缓存键后缀
            
        Returns:
            str: 完整的缓存键
        """
        return f"{self.model_name}:{key_suffix}"
    
    async def get_by_id(self, id_value: Any) -> Optional[T]:
        """
        根据ID获取单个实体
        
        实现顺序：
        1. 先从Redis缓存获取
        2. 若缓存未命中，从数据库查询
        3. 若数据库未命中，且配置了API服务，则从API获取并保存
        
        Args:
            id_value: ID值
            
        Returns:
            Optional[T]: 实体对象，不存在则为None
        """
        # 生成缓存键
        cache_key = self._get_cache_key(f"id:{id_value}")
        
        # 1. 先从缓存获取
        cached_data = await sync_to_async(cache.get)(cache_key)
        if cached_data:
            logger.debug(f"缓存命中: {cache_key}")
            # 如果缓存中有数据，则反序列化为模型实例
            if isinstance(cached_data, dict):
                return self.model_class(**cached_data)
            return cached_data
        
        logger.debug(f"缓存未命中: {cache_key}")
        
        # 2. 从数据库获取
        try:
            instance = await self._get_from_db_by_id(id_value)
            if instance:
                logger.debug(f"数据库命中: {id_value}")
                # 如果数据库中存在，则更新缓存
                await sync_to_async(cache.set)(cache_key, instance, self.cache_timeout)
                return instance
        except Exception as e:
            logger.error(f"数据库查询错误: {str(e)}")
        
        # 3. 如果配置了API服务，尝试从API获取
        if self.api_service:
            try:
                logger.debug(f"从API获取: {id_value}")
                api_data = await self._fetch_from_api_by_id(id_value)
                if api_data:
                    # 保存到数据库
                    instance = await self._save_to_db(api_data)
                    # 更新缓存
                    await sync_to_async(cache.set)(cache_key, instance, self.cache_timeout)
                    return instance
            except Exception as e:
                logger.error(f"API获取数据错误: {str(e)}")
        
        return None
    
    async def _get_from_db_by_id(self, id_value: Any) -> Optional[T]:
        """
        从数据库获取单个实体
        
        Args:
            id_value: ID值
            
        Returns:
            Optional[T]: 实体对象，不存在则为None
        """
        try:
            # 获取主键字段名
            pk_name = self.model_class._meta.pk.name
            # 构建查询条件
            filter_kwargs = {pk_name: id_value}
            # 执行查询
            return await sync_to_async(self.model_class.objects.filter(**filter_kwargs).first)()
        except Exception as e:
            logger.error(f"数据库查询错误: {str(e)}")
            return None
    
    async def _fetch_from_api_by_id(self, id_value: Any) -> Optional[Dict[str, Any]]:
        """
        从API获取单个实体数据
        
        这个方法需要在子类中重写，因为不同的API有不同的调用方式
        
        Args:
            id_value: ID值
            
        Returns:
            Optional[Dict[str, Any]]: API返回的实体数据，不存在则为None
        """
        # 在子类中实现具体逻辑
        raise NotImplementedError("子类必须实现_fetch_from_api_by_id方法")
    
    async def get_all(self) -> List[T]:
        """
        获取所有实体
        
        实现顺序：
        1. 先从Redis缓存获取
        2. 若缓存未命中，从数据库查询
        3. 若数据库为空，且配置了API服务，则从API获取并保存
        
        Returns:
            List[T]: 实体对象列表
        """
        # 生成缓存键
        cache_key = self._get_cache_key("all")
        
        # 1. 先从缓存获取
        cached_data = await sync_to_async(cache.get)(cache_key)
        if cached_data:
            logger.debug(f"缓存命中: {cache_key}")
            return cached_data
        
        logger.debug(f"缓存未命中: {cache_key}")
        
        # 2. 从数据库获取
        try:
            instances = await self._get_all_from_db()
            if instances and len(instances) > 0:
                logger.debug(f"数据库命中: 共{len(instances)}条记录")
                # 如果数据库中存在，则更新缓存
                await sync_to_async(cache.set)(cache_key, instances, self.cache_timeout)
                return instances
        except Exception as e:
            logger.error(f"数据库查询错误: {str(e)}")
        
        # 3. 如果配置了API服务，尝试从API获取
        try:
            logger.debug("从API获取所有数据")
            api_data_list = await self._fetch_all_from_api()
            if api_data_list and len(api_data_list) > 0:
                # 保存到数据库
                instances = await self._save_all_to_db(api_data_list)
                # 更新缓存
                await sync_to_async(cache.set)(cache_key, instances, self.cache_timeout)
                return instances
        except Exception as e:
            logger.error(f"API获取数据错误: {str(e)}")
            
        return []
    
    async def _get_all_from_db(self) -> List[T]:
        """
        从数据库获取所有实体
        
        Returns:
            List[T]: 实体对象列表
        """
        try:
            return await sync_to_async(list)(self.model_class.objects.all())
        except Exception as e:
            logger.error(f"数据库查询错误: {str(e)}")
            return []
    
    async def _fetch_all_from_api(self) -> List[Dict[str, Any]]:
        """
        从API获取所有实体数据
        
        这个方法需要在子类中重写，因为不同的API有不同的调用方式
        
        Returns:
            List[Dict[str, Any]]: API返回的实体数据列表
        """
        # 在子类中实现具体逻辑
        raise NotImplementedError("子类必须实现_fetch_all_from_api方法")
    
    async def filter(self, **kwargs) -> List[T]:
        """
        根据条件筛选实体
        
        实现顺序：
        1. 先从Redis缓存获取
        2. 若缓存未命中，从数据库查询
        
        Args:
            **kwargs: 筛选条件
            
        Returns:
            List[T]: 符合条件的实体对象列表
        """
        # 生成缓存键，根据筛选条件构建
        filter_str = ":".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        cache_key = self._get_cache_key(f"filter:{filter_str}")
        
        # 1. 先从缓存获取
        cached_data = await sync_to_async(cache.get)(cache_key)
        if cached_data:
            logger.debug(f"缓存命中: {cache_key}")
            return cached_data
        
        logger.debug(f"缓存未命中: {cache_key}")
        
        # 2. 从数据库获取
        try:
            # 将同步的filter操作转换为异步
            instances = await sync_to_async(list)(self.model_class.objects.filter(**kwargs))
            # 更新缓存
            await sync_to_async(cache.set)(cache_key, instances, self.cache_timeout)
            return instances
        except Exception as e:
            logger.error(f"数据库筛选错误: {str(e)}")
            return []

    async def _save_all_to_db(self, model_class, data_list, unique_fields,
        ignore_conflicts_on_create=False,  # 新增参数，控制 bulk_create 是否忽略冲突
        **extra_fields, ) -> Dict:
        """
        优化的通用异步数据批量处理方法，使用bulk_create和bulk_update提高性能, 并增强了错误处理和日志记录
        Args:
            model_class: Django模型类
            data_list: 要处理的数据列表，每项都是模型对应的字段字典
            unique_fields: 用于确定唯一记录的字段列表
            ignore_conflicts_on_create: (可选) Boolean, 默认为 False。如果为 True，在 bulk_create 时忽略唯一性冲突错误，跳过冲突记录。
            **extra_fields: 额外的字段，将添加到所有创建/更新的记录中
        Returns:
            dict: 包含创建、更新、跳过和失败的记录数
        """
        if not data_list:
            logger.warning(f"未提供任何数据用于处理 - {model_class.__name__}")
            return {"创建": 0, "更新": 0, "未更改": 0, "失败": 0}

        # 如果传入的不是列表，转换为列表
        if not isinstance(data_list, list):
            data_list = [data_list]

        # 统计计数器，更清晰的命名
        created_count = 0
        updated_count = 0
        unchanged_count = 0  # 原 skipped_count 更名为 unchanged_count
        failed_count = 0  # 新增 failed_count

        batch_size = 2000
        for batch_index, i in enumerate(range(0, len(data_list), batch_size)):
            batch = data_list[i : i + batch_size]

            @transaction.atomic
            def process_batch():
                nonlocal created_count, updated_count, unchanged_count, failed_count

                existing_items = {}
                query_filters = []

                for item in batch:
                    filter_kwargs = {
                        field: item.get(field) for field in unique_fields if field in item
                    }
                    query_condition = Q(**filter_kwargs)
                    query_filters.append(query_condition)
                    key = tuple(
                        (field, item.get(field)) for field in unique_fields if field in item
                    )
                    existing_items[key] = item

                existing_records = {}
                if query_filters:
                    query = model_class.objects.filter(reduce(operator.or_, query_filters))
                    for record in query:
                        key = tuple(
                            (field, getattr(record, field)) for field in unique_fields
                        )
                        existing_records[key] = record

                to_create = []
                to_update = []
                update_fields = set()

                for key, item in existing_items.items():
                    if key in existing_records:
                        # 记录已存在，检查是否需要更新
                        record = existing_records[key]
                        has_changes = False

                        for field, value in item.items():
                            if hasattr(record, field) and getattr(record, field) != value:
                                setattr(record, field, value)
                                has_changes = True
                                update_fields.add(field)

                        if has_changes:
                            to_update.append(record)
                            updated_count += 1
                        else:
                            unchanged_count += 1  # 更名 skipped_count 为 unchanged_count
                    else:
                        # 记录不存在，创建新记录
                        create_data = {**item, **extra_fields} # 合并 extra_fields
                        to_create.append(model_class(**create_data))
                        created_count += 1

                # 批量创建新记录
                if to_create:
                    try:
                        model_class.objects.bulk_create(
                            to_create, ignore_conflicts=ignore_conflicts_on_create
                        )  # 使用 ignore_conflicts 参数
                    except IntegrityError as e:  # 更具体的异常处理
                        error_msg = f"批次 {batch_index + 1} 批量创建记录时遇到唯一性冲突 (忽略冲突: {ignore_conflicts_on_create}): {str(e)}"
                        logger.warning(error_msg) # 警告级别日志，因为可能是预期内的冲突
                        if not ignore_conflicts_on_create: # 如果不忽略冲突，则计入失败
                            failed_count += len(to_create)
                            created_count -= len(to_create) # 修正 created_count
                        else:
                            unchanged_count += len(to_create) # 如果忽略冲突，则计入 unchanged_count (可以根据业务调整)
                            created_count -= len(to_create) # 修正 created_count
                    except DatabaseError as e: # 捕获数据库错误，例如连接问题
                        error_msg = f"批次 {batch_index + 1} 批量创建记录时遇到数据库错误: {str(e)}"
                        logger.error(error_msg) # 错误级别日志，数据库错误通常需要关注
                        failed_count += len(to_create)
                        created_count -= len(to_create) # 修正 created_count
                    except Exception as e: # 捕获其他异常，例如字段类型错误等
                        error_msg = f"批次 {batch_index + 1} 批量创建记录时遇到未知错误: {str(e)}, 数据示例: {to_create[:2]}" # 记录数据示例方便排查
                        logger.error(error_msg)
                        failed_count += len(to_create)
                        created_count -= len(to_create) # 修正 created_count


                # 批量更新已有记录
                if to_update and update_fields:
                    # 排除主键和不可更新字段
                    update_fields_final = [
                        f
                        for f in update_fields
                        if f not in unique_fields
                        and f not in getattr(model_class._meta, "read_only_fields", [])
                    ]
                    if update_fields_final:
                        try:
                            model_class.objects.bulk_update(to_update, update_fields_final)
                        except IntegrityError as e: # 更具体的异常处理
                            error_msg = f"批次 {batch_index + 1} 批量更新记录时遇到唯一性冲突: {str(e)}"
                            logger.error(error_msg)
                            failed_count += len(to_update)
                            updated_count -= len(to_update) # 修正 updated_count
                        except DatabaseError as e: # 捕获数据库错误
                            error_msg = f"批次 {batch_index + 1} 批量更新记录时遇到数据库错误: {str(e)}"
                            logger.error(error_msg)
                            failed_count += len(to_update)
                            updated_count -= len(to_update) # 修正 updated_count
                        except Exception as e: # 捕获其他异常
                            error_msg = f"批次 {batch_index + 1} 批量更新记录时遇到未知错误: {str(e)}, 更新字段: {update_fields_final}, 数据示例: {to_update[:2]}" # 记录更新字段和数据示例
                            logger.error(error_msg)
                            failed_count += len(to_update)
                            updated_count -= len(to_update) # 修正 updated_count

            await sync_to_async(process_batch)()

        result = {
            "创建": created_count,
            "更新": updated_count,
            "未更改": unchanged_count, # 更名 skipped 为 未更改
            "失败": failed_count, # 新增 失败 计数
        }

        # logger.info(f"完成{model_class.__name__}数据批量处理: {result}")
        return result

    async def _save_all_to_db_refactored(
        self, # 假设这是类的一部分
        model_class: Type[models.Model],
        data_list: List[Dict[str, Any]],
        unique_fields: List[str],
        # ignore_conflicts_on_create 参数不再需要，upsert 逻辑会处理
        **extra_fields: Any,
    ) -> Dict[str, int]:
        """
        使用 django-bulk-update-or-create 改造后的通用异步数据批量处理方法。
        利用数据库原生的 UPSERT 功能 (如 MySQL 的 INSERT ... ON DUPLICATE KEY UPDATE) 提高性能。

        Args:
            model_class: Django模型类
            data_list: 要处理的数据列表，每项都是模型对应的字段字典
            unique_fields: 用于匹配现有记录的字段列表 (应对应数据库唯一约束)
            **extra_fields: 额外的字段，将添加到所有创建/更新的记录中

        Returns:
            dict: 包含处理尝试的总数和失败的记录数。
                注意：此实现不区分创建和更新计数，且不跟踪未更改的记录。
                失败计数是基于批次的，如果一个批次失败，整个批次计为失败。
                如果需要更精确的创建/更新计数，需要检查所用库版本是否提供相应方法。
        """
        if not data_list:
            logger.warning(f"未提供任何数据用于处理 - {model_class.__name__}")
            # 返回与原始结构类似的字典，但计数逻辑已改变
            return {"尝试处理": 0, "失败": 0, "创建": 0, "更新": 0} # 添加创建/更新占位符

        if not isinstance(data_list, list):
            data_list = [data_list] # 确保是列表

        total_attempted = len(data_list)
        failed_count = 0
        # 创建和更新计数在此简化实现中无法精确获取，除非库明确返回
        created_count = 0 # 占位符
        updated_count = 0 # 占位符

        # --- 确定 update_fields ---
        # 假设 data_list 不为空且所有字典具有相似（但不一定完全相同）的结构
        # 我们合并所有遇到的键，然后排除 unique_fields
        all_keys = set()
        for item in data_list:
            all_keys.update(item.keys())
        all_keys.update(extra_fields.keys()) # 包含 extra_fields 的键

        update_fields = list(all_keys - set(unique_fields))

        # 可选：从 update_fields 中移除主键（通常不需要更新主键）
        pk_name = model_class._meta.pk.name
        if pk_name in update_fields:
            update_fields.remove(pk_name)
        # 可选：移除其他只读或不应更新的字段
        # read_only_fields = getattr(model_class._meta, "read_only_fields", [])
        # update_fields = [f for f in update_fields if f not in read_only_fields]

        if not update_fields:
            logger.warning(
                f"模型 {model_class.__name__} 没有可用于更新的字段（除了唯一字段 {unique_fields}）。"
                f" 这意味着只会尝试创建新记录，现有记录不会被修改。"
            )
            # 即使没有更新字段，库通常也能处理仅创建的情况

        batch_size = 2000 # 保持与原逻辑一致的批处理大小，用于事务和错误范围控制
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]
            current_batch_size = len(batch)

            # --- 准备包含 extra_fields 的批处理数据 ---
            # 确保 extra_fields 不会意外覆盖 item 中已有的同名字段的值
            # （Python 字典合并行为：后面的键值对会覆盖前面的）
            batch_data_prepared = [{**item, **extra_fields} for item in batch]

            @transaction.atomic # 每个批次仍在事务中执行
            def process_batch_sync():
                nonlocal failed_count # , created_count, updated_count # 如果能获取计数，也声明 nonlocal
                try:
                    # --- 核心改动: 调用 bulk_update_or_create ---
                    # 检查你使用的库版本文档，确认方法名和参数
                    # `match_field` 通常用于指定唯一键字段
                    # `update_fields` 指定当记录存在时要更新的字段
                    model_class.objects.bulk_update_or_create(
                        objs=batch_data_prepared,
                        match_field=unique_fields,
                        update_fields=update_fields,
                        batch_size=None, # 传递给库的 batch_size，None 表示处理整个 objs 列表
                        # yield_objects=False, # 通常不需要返回对象本身
                        # case_insensitive_match=False # 根据需要设置
                    )
                    # --- 处理计数 (如果库支持) ---
                    # 如果你的库版本/fork 提供了计数方法，例如:
                    # counts = model_class.objects.bulk_update_or_create_counts(...)
                    # created_count += counts.created
                    # updated_count += counts.updated
                    # logger.info(f"批次 {i // batch_size + 1} 处理成功: {counts}")
                    # 如果不支持，我们无法在此处精确更新 created/updated 计数

                except (IntegrityError, DatabaseError) as e:
                    # 数据库层面的错误 (如约束冲突、连接问题)
                    logger.error(
                        f"批次 {i // batch_size + 1} (大小: {current_batch_size}) "
                        f"使用 bulk_update_or_create 时遇到数据库错误: {str(e)}"
                    )
                    failed_count += current_batch_size # 整个批次标记为失败
                except Exception as e:
                    # 其他潜在错误 (如数据准备、库内部错误)
                    logger.error(
                        f"批次 {i // batch_size + 1} (大小: {current_batch_size}) "
                        f"使用 bulk_update_or_create 时遇到意外错误: {str(e)}",
                        exc_info=True # 记录完整的 traceback
                    )
                    failed_count += current_batch_size # 整个批次标记为失败

            # 使用 sync_to_async 运行同步的数据库操作
            await sync_to_async(process_batch_sync)()

        # --- 最终结果 ---
        # 注意：创建和更新计数在此简化版本中为 0，除非 process_batch_sync 中有逻辑填充它们
        successful_count = total_attempted - failed_count
        result = {
            "尝试处理": total_attempted,
            "成功": successful_count, # 添加成功计数以提供更多信息
            "失败": failed_count,
            "创建": created_count, # 保持字段存在，但值可能不精确
            "更新": updated_count, # 保持字段存在，但值可能不精确
            # "未更改" 计数通常无法通过此方法获得
        }

        logger.info(f"完成 {model_class.__name__} 数据批量处理 (使用 bulk_update_or_create): {result}")
        return result

    async def _save_all_to_db_native_upsert( self, model_class: Type[models.Model], data_list: List[Dict[str, Any]],
        unique_fields: List[str], # 用于冲突检测的字段 (应有唯一约束)
        # extra_fields 仍然有用
        **extra_fields: Any,
    ) -> Dict[str, int]:
        """
        使用 Django 5 原生的 bulk_create 实现 Upsert (Update or Create) 的批量处理方法。
        不再依赖外部库 django-bulk-update-or-create。
        Args:
            model_class: Django模型类
            data_list: 要处理的数据列表，每项都是模型对应的字段字典
            unique_fields: 用于确定唯一记录并检测冲突的字段列表 (必须在数据库中有唯一约束)
            **extra_fields: 额外的字段，将添加到所有创建/更新的记录中
        Returns:
            dict: 包含处理尝试的总数和失败的记录数。
                注意：Django 原生 bulk_create 在 upsert 模式下不直接返回创建/更新计数。
        """
        if not data_list:
            logger.warning(f"未提供任何数据用于处理 - {model_class.__name__}")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0} # 简化返回结构
        if not isinstance(data_list, list):
            data_list = [data_list]
        total_attempted = len(data_list)
        failed_count = 0
        # --- 确定 update_fields ---
        # 这些是当记录存在时，需要被更新的字段
        all_keys = set()
        if data_list: # 确保 data_list 不为空
            # 从第一个数据项获取基础键，并合并 extra_fields 的键
            all_keys = set(data_list[0].keys()) | set(extra_fields.keys())
            # 可以选择遍历所有项来获取所有可能的键，但这可能效率稍低
            # for item in data_list:
            #     all_keys.update(item.keys())
            # all_keys.update(extra_fields.keys())
        # update_fields 是所有字段中排除了 unique_fields 的部分
        update_fields = list(all_keys - set(unique_fields))
        # 可选：从 update_fields 中移除主键（通常不需要更新主键）
        pk_name = model_class._meta.pk.name
        if pk_name in update_fields:
            update_fields.remove(pk_name)
        # 可选：移除其他不应更新的字段
        # read_only_fields = getattr(model_class._meta, "read_only_fields", [])
        # update_fields = [f for f in update_fields if f not in read_only_fields]
        if not update_fields:
            logger.warning(
                f"模型 {model_class.__name__} 没有可用于更新的字段（除了唯一字段 {unique_fields}）。"
                f" bulk_create 在 update_conflicts=True 模式下仍会尝试，但现有记录不会被修改。"
            )
            # 即使 update_fields 为空，只要 unique_fields 设置正确，创建逻辑仍然有效
        batch_size = 2000 # 保持批处理大小
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]
            current_batch_size = len(batch)
            # --- 准备模型实例列表 ---
            # bulk_create 需要的是模型实例列表，而不是字典列表
            objs_to_process = []
            for item in batch:
                # 合并 extra_fields 到每个 item
                # 确保 extra_fields 不会意外覆盖 item 中已有的同名字段的值
                prepared_data = {**item, **extra_fields}
                objs_to_process.append(model_class(**prepared_data))
            @transaction.atomic # 每个批次仍在事务中执行
            def process_batch_sync():
                nonlocal failed_count
                try:
                    # --- 核心改动: 移除 unique_fields 参数 ---
                    model_class.objects.bulk_create(
                        objs_to_process,
                        update_conflicts=True,       # 启用 Upsert
                        # unique_fields=unique_fields, # <-- 移除这一行
                        update_fields=update_fields, # 指定冲突时更新的字段
                        batch_size=current_batch_size
                    )
                    # logger.info(f"批次 {i // batch_size + 1} 处理成功")
                except (IntegrityError, DatabaseError) as e:
                    # 捕获数据库层面的错误
                    # 注意：如果 unique_fields 没有在数据库层面设置唯一约束，这里可能不会按预期触发冲突
                    # 或者如果 update_fields 包含无法更新的字段（如外键指向不存在的对象），也可能报错
                    logger.error(
                        f"批次 {i // batch_size + 1} (大小: {current_batch_size}) "
                        f"使用原生 bulk_create (upsert) 时遇到数据库错误: {str(e)}"
                    )
                    failed_count += current_batch_size # 整个批次标记为失败
                except Exception as e:
                    # 捕获其他潜在错误 (如数据准备阶段的类型错误)
                    logger.error(
                        f"批次 {i // batch_size + 1} (大小: {current_batch_size}) "
                        f"使用原生 bulk_create (upsert) 时遇到意外错误: {str(e)}",
                        exc_info=True # 记录完整的 traceback
                    )
                    failed_count += current_batch_size # 整个批次标记为失败
            # 使用 sync_to_async 运行同步的数据库操作
            await sync_to_async(process_batch_sync)()
        # --- 最终结果 ---
        successful_count = total_attempted - failed_count
        result = {
            "尝试处理": total_attempted,
            "失败": failed_count,
            "创建/更新成功": successful_count, # 合并计数
        }
        # logger.info(f"完成 {model_class.__name__} 数据批量处理 (使用原生 bulk_create upsert): {result}")
        return result

    async def update(self, id_value: Any, data: Dict[str, Any]) -> Optional[T]:
        """
        更新实体
        
        Args:
            id_value: ID值
            data: 更新数据
            
        Returns:
            Optional[T]: 更新后的实体对象，不存在则为None
        """
        # 生成缓存键
        cache_key = self._get_cache_key(f"id:{id_value}")
        
        try:
            # 获取主键字段名
            pk_name = self.model_class._meta.pk.name
            
            # 检查实体是否存在
            filter_kwargs = {pk_name: id_value}
            instance = await sync_to_async(self.model_class.objects.filter(**filter_kwargs).first)()
            
            if not instance:
                logger.error(f"更新失败: 实体不存在 {id_value}")
                return None
            
            # 更新实体
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            await sync_to_async(instance.save)()
            
            # 删除缓存
            await sync_to_async(cache.delete)(cache_key)
            await sync_to_async(cache.delete)(self._get_cache_key("all"))
            
            logger.debug(f"更新实体: {id_value}")
            return instance
        except Exception as e:
            logger.error(f"更新实体错误: {str(e)}")
            return None
    
    async def delete(self, id_value: Any) -> bool:
        """
        删除实体
        
        Args:
            id_value: ID值
            
        Returns:
            bool: 是否成功删除
        """
        # 生成缓存键
        cache_key = self._get_cache_key(f"id:{id_value}")
        
        try:
            # 获取主键字段名
            pk_name = self.model_class._meta.pk.name
            
            # 执行删除
            filter_kwargs = {pk_name: id_value}
            delete_func = sync_to_async(lambda: self.model_class.objects.filter(**filter_kwargs).delete())
            count, _ = await delete_func()
            
            if count > 0:
                # 删除缓存
                await sync_to_async(cache.delete)(cache_key)
                await sync_to_async(cache.delete)(self._get_cache_key("all"))
                
                logger.debug(f"删除实体: {id_value}")
                return True
            else:
                logger.error(f"删除失败: 实体不存在 {id_value}")
                return False
        except Exception as e:
            logger.error(f"删除实体错误: {str(e)}")
            return False
    
    async def refresh_cache(self, id_value: Any = None) -> None:
        """
        刷新缓存
        
        Args:
            id_value: 指定刷新某个实体的缓存，为None则刷新所有缓存
        """
        if id_value:
            # 刷新单个实体缓存
            cache_key = self._get_cache_key(f"id:{id_value}")
            await sync_to_async(cache.delete)(cache_key)
            logger.debug(f"刷新实体缓存: {id_value}")
            
            # 从数据库或API重新获取
            await self.get_by_id(id_value)
        else:
            # 刷新所有实体缓存
            await sync_to_async(cache.delete)(self._get_cache_key("all"))
            logger.debug("刷新所有实体缓存")
            
            # 从数据库或API重新获取
            await self.get_all()
    
    async def clear_cache(self) -> None:
        """
        清除所有相关缓存
        """
        # 清除所有实体缓存
        await sync_to_async(cache.delete)(self._get_cache_key("all"))
        
        # 删除所有以model_name为前缀的缓存
        try:
            from django_redis import get_redis_connection
            # 异步获取Redis连接
            redis_client = await sync_to_async(get_redis_connection)("default")
            pattern = f"{self.model_name}:*"
            
            # 异步获取匹配的键
            keys = await sync_to_async(redis_client.keys)(pattern)
            
            if keys:
                # 异步删除所有匹配的键
                await sync_to_async(redis_client.delete)(*keys)
                
            logger.debug(f"清除所有缓存: {pattern}")
        except Exception as e:
            logger.error(f"清除缓存错误: {str(e)}")

    def _parse_datetime(self, value: Any, default_format: str = None, default: datetime = None) -> Optional[datetime]:
        """
        解析各种格式的日期时间，返回统一格式的datetime对象
        
        参数:
            value: 要解析的日期时间值，可以是字符串、时间戳、datetime对象等
            default_format: 指定日期时间字符串的格式，如 '%Y-%m-%d %H:%M:%S'
            default: 解析失败时返回的默认值，默认为None
            
        返回:
            Optional[datetime]: 解析后的datetime对象，解析失败则返回default
        """
        # 获取系统时区
        self.tz = pytz.timezone(settings.TIME_ZONE) if hasattr(settings, 'TIME_ZONE') else pytz.UTC
        
        if default is None and hasattr(self, 'model') and hasattr(self.model, 'created_at'):
            # 如果模型有created_at字段，使用当前时间作为默认值
            default = datetime.now(self.tz)
            
        try:
            # 如果已经是datetime对象
            if isinstance(value, datetime):
                # 确保有时区信息
                if value.tzinfo is None:
                    return value.replace(tzinfo=self.tz)
                return value
                
            # 处理None或空值
            if value is None or value == '' or value == 'N/A' or value == '暂无' or value == '-':
                return default
                
            # 确保value是字符串
            if not isinstance(value, (str, int, float)):
                value = str(value)
                
            # 处理时间戳（秒或毫秒）
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                timestamp = float(value)
                # 判断是秒还是毫秒
                if timestamp > 10000000000:  # 大于10^10的可能是毫秒
                    timestamp /= 1000
                return datetime.fromtimestamp(timestamp, self.tz)
                
            # 处理ISO格式
            if isinstance(value, str) and ('T' in value or '+' in value or 'Z' in value):
                try:
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=self.tz)
                    return dt
                except ValueError:
                    pass  # 继续尝试其他格式
                    
            # 处理指定格式
            if default_format:
                try:
                    dt = datetime.strptime(value, default_format)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=self.tz)
                    return dt
                except ValueError:
                    pass  # 继续尝试其他格式
                    
            # 尝试常见格式
            common_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M:%S',
                '%Y/%m/%d %H:%M',
                '%Y-%m-%d',
                '%H:%M:%S',
                # '%d/%m/%Y %H:%M:%S',
                # '%d/%m/%Y %H:%M',
                # '%d-%m-%Y %H:%M:%S',
                # '%d-%m-%Y %H:%M',
                # '%Y/%m/%d',
                # '%d/%m/%Y',
                # '%d-%m-%Y',
                # '%d-%b-%Y',
                # '%d %b %Y',
                # '%b %d, %Y',
                # '%B %d, %Y',
                '%Y年%m月%d日',
                '%Y年%m月%d日 %H时%M分%S秒',
            ]
            
            for fmt in common_formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=self.tz)
                    return dt
                except ValueError:
                    continue
                    
            # 如果所有尝试都失败
            logger.warning(f"无法解析日期时间值: {value}，使用默认值")
            return default
            
        except Exception as e:
            logger.warning(f"解析日期时间值时发生错误: {value}, 错误: {str(e)}")
            return default

    def _parse_number(self, value: Any, default: float = 0.0) -> float:
        """
        解析数字，处理各种格式的数字字符串
        
        参数:
            value: 要解析的值，可以是数字、字符串等
            default: 解析失败时返回的默认值
            
        返回:
            float: 解析后的数字
        """
        try:
            # 如果是数字类型，直接返回float
            if isinstance(value, (int, float)):
                return float(value)
            
            # 如果是None或空字符串，返回默认值
            if value is None or value == '' or value == '-' or value == '暂无':
                return default
            
            # 确保value是字符串
            value = str(value).strip()
            
            # 处理特殊字符
            value = value.replace(',', '')  # 移除千位分隔符
            
            # 提取数字部分
            import re
            number_match = re.search(r'([-+]?\d*\.?\d+)', value)
            if not number_match:
                return default
            
            number = float(number_match.group(1))
            
            # 处理单位
            if '万亿' in value or '兆' in value:
                number *= 1000000000000
            elif '亿' in value:
                number *= 100000000
            elif '万' in value:
                number *= 10000
            elif '千' in value:
                number *= 1000
            
            # 处理百分比
            if '%' in value:
                number /= 100
            
            # 处理括号中的负数
            if '(' in value and ')' in value:
                number = -abs(number)
            
            return number
            
        except Exception as e:
            logger.warning(f"解析数字失败: {value}, 使用默认值{default}, 错误: {str(e)}")
            return default


    @staticmethod
    def _get_model_fields(model_class):
        """
        获取模型的字段列表
        
        Args:
            model_class: Django模型类
            
        Returns:
            list: 字段名称列表
        """
        return [field.name for field in model_class._meta.fields]
