# dao_manager/basedao.py

import json
import logging
from typing import Dict, List, Any, Optional, Type, Union, TypeVar, Generic
from datetime import datetime, timedelta

from django.db import models
from django.core.cache import cache
from django.conf import settings
from django.db import transaction

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
    
    def __init__(self, model_class: Type[T], api_service=None, cache_timeout: int = 3600):
        """
        初始化DAO
        
        Args:
            model_class: 模型类
            api_service: API服务实例
            cache_timeout: 缓存超时时间(秒)
        """
        self.model_class = model_class
        self.api_service = api_service
        self.cache_timeout = cache_timeout
        self.model_name = model_class._meta.model_name
        logger.info(f"初始化{self.model_name}DAO")
    
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
        cached_data = cache.get(cache_key)
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
                cache.set(cache_key, instance, self.cache_timeout)
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
                    cache.set(cache_key, instance, self.cache_timeout)
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
            return await self.model_class.objects.filter(**filter_kwargs).afirst()
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
        cached_data = cache.get(cache_key)
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
                cache.set(cache_key, instances, self.cache_timeout)
                return instances
        except Exception as e:
            logger.error(f"数据库查询错误: {str(e)}")
        
        # 3. 如果配置了API服务，尝试从API获取
        if self.api_service:
            try:
                logger.debug("从API获取所有数据")
                api_data_list = await self._fetch_all_from_api()
                if api_data_list and len(api_data_list) > 0:
                    # 保存到数据库
                    instances = await self._save_all_to_db(api_data_list)
                    # 更新缓存
                    cache.set(cache_key, instances, self.cache_timeout)
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
            return await self.model_class.objects.all()
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
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.debug(f"缓存命中: {cache_key}")
            return cached_data
        
        logger.debug(f"缓存未命中: {cache_key}")
        
        # 2. 从数据库获取
        try:
            instances = await self.model_class.objects.filter(**kwargs)
            # 更新缓存
            cache.set(cache_key, instances, self.cache_timeout)
            return instances
        except Exception as e:
            logger.error(f"数据库筛选错误: {str(e)}")
            return []
    
    @transaction.atomic
    async def _save_to_db(self, data: Dict[str, Any]) -> T:
        """
        保存单个实体到数据库
        
        Args:
            data: 实体数据
            
        Returns:
            T: 保存后的实体对象
        """
        try:
            # 获取主键字段名
            pk_name = self.model_class._meta.pk.name
            
            # 检查实体是否已存在
            if pk_name in data:
                pk_value = data[pk_name]
                try:
                    instance = await self.model_class.objects.filter(**{pk_name: pk_value}).afirst()
                    if instance:
                        # 更新已存在的实体
                        for key, value in data.items():
                            setattr(instance, key, value)
                        await instance.asave()
                        logger.debug(f"更新实体: {pk_value}")
                        return instance
                except Exception as e:
                    logger.error(f"查询实体错误: {str(e)}")
            
            # 创建新实体
            instance = self.model_class(**data)
            await instance.asave()
            logger.debug(f"创建新实体: {instance}")
            return instance
        except Exception as e:
            logger.error(f"保存实体错误: {str(e)}")
            raise
    
    @transaction.atomic
    async def _save_all_to_db(self, data_list: List[Dict[str, Any]]) -> List[T]:
        """
        保存多个实体到数据库
        
        Args:
            data_list: 实体数据列表
            
        Returns:
            List[T]: 保存后的实体对象列表
        """
        instances = []
        for data in data_list:
            try:
                instance = await self._save_to_db(data)
                instances.append(instance)
            except Exception as e:
                logger.error(f"保存实体错误: {str(e)}")
        return instances
    
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
            instance = await self.model_class.objects.filter(**filter_kwargs).afirst()
            
            if not instance:
                logger.error(f"更新失败: 实体不存在 {id_value}")
                return None
            
            # 更新实体
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            await instance.asave()
            
            # 删除缓存
            cache.delete(cache_key)
            cache.delete(self._get_cache_key("all"))
            
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
            count, _ = await self.model_class.objects.filter(**filter_kwargs).adelete()
            
            if count > 0:
                # 删除缓存
                cache.delete(cache_key)
                cache.delete(self._get_cache_key("all"))
                
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
            cache.delete(cache_key)
            logger.debug(f"刷新实体缓存: {id_value}")
            
            # 从数据库或API重新获取
            await self.get_by_id(id_value)
        else:
            # 刷新所有实体缓存
            cache.delete(self._get_cache_key("all"))
            logger.debug("刷新所有实体缓存")
            
            # 从数据库或API重新获取
            await self.get_all()
    
    async def clear_cache(self) -> None:
        """
        清除所有相关缓存
        """
        # 清除所有实体缓存
        cache.delete(self._get_cache_key("all"))
        
        # 删除所有以model_name为前缀的缓存
        # 注意：Django的cache接口不支持pattern删除，如果使用的是Redis，可以通过redis_client直接删除
        try:
            from django_redis import get_redis_connection
            redis_client = get_redis_connection("default")
            pattern = f"{self.model_name}:*"
            
            for key in redis_client.keys(pattern):
                redis_client.delete(key)
                
            logger.debug(f"清除所有缓存: {pattern}")
        except Exception as e:
            logger.error(f"清除缓存错误: {str(e)}")
