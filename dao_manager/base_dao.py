# dao_manager/basedao.py

import decimal
import json
import logging
from typing import Dict, List, Any, Optional, Type, Union, TypeVar, Generic
from datetime import datetime, date, time

from django.db import models
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
                    instance = await sync_to_async(self.model_class.objects.filter(**{pk_name: pk_value}).first)()
                    if instance:
                        # 更新已存在的实体
                        for key, value in data.items():
                            setattr(instance, key, value)
                        await sync_to_async(instance.save)()
                        logger.debug(f"更新实体: {pk_value}")
                        return instance
                except Exception as e:
                    logger.error(f"查询实体错误: {str(e)}")
            
            # 创建新实体
            instance = self.model_class(**data)
            await sync_to_async(instance.save)()
            logger.debug(f"创建新实体: {instance}")
            return instance
        except Exception as e:
            logger.error(f"保存实体错误: {str(e)}")
            raise
    
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
        from django.conf import settings
        import pytz
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
                '%d/%m/%Y %H:%M:%S',
                '%d/%m/%Y %H:%M',
                '%d-%m-%Y %H:%M:%S',
                '%d-%m-%Y %H:%M',
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%d/%m/%Y',
                '%d-%m-%Y',
                '%d-%b-%Y',
                '%d %b %Y',
                '%b %d, %Y',
                '%B %d, %Y',
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

    def _serialize_model(self, model_instance) -> dict:
        """
        将Django模型实例序列化为可JSON化的字典
        
        Args:
            model_instance: Django模型实例
            
        Returns:
            dict: 序列化后的字典
        """
        import datetime  # 确保正确导入
        from decimal import Decimal  # 导入Decimal类
        
        if model_instance is None:
            return None
            
        # 如果已经是字典，直接返回
        if isinstance(model_instance, dict):
            return model_instance
        
        # 确保是模型实例    
        if not hasattr(model_instance, '_meta'):
            raise TypeError(f"Expected Django model instance, got {type(model_instance)}")
            
        result = {}
        
        # 遍历所有字段
        for field in model_instance._meta.fields:
            field_name = field.name
            value = getattr(model_instance, field_name)
            
            # 处理None值
            if value is None:
                result[field_name] = None
                continue
                
            # 处理各种类型
            if isinstance(value, (str, int, float, bool)):
                # 基本类型可直接使用
                result[field_name] = value
                
            elif isinstance(value, datetime.datetime):
                # 日期时间转ISO格式字符串
                result[field_name] = value.isoformat()
                
            elif isinstance(value, datetime.date):
                # 日期转ISO格式字符串
                result[field_name] = value.isoformat()
                
            elif isinstance(value, datetime.time):
                # 时间转ISO格式字符串
                result[field_name] = value.isoformat()
                
            elif isinstance(value, Decimal):
                # Decimal转浮点数
                result[field_name] = float(value)
                
            elif hasattr(value, 'pk') and hasattr(value, '_meta'):
                # 处理外键关系-只保存主键
                result[field_name] = value.pk
                
            elif isinstance(value, (list, tuple)):
                # 列表或元组-尝试序列化内部元素
                try:
                    result[field_name] = [
                        self._serialize_model(item) if hasattr(item, '_meta') else item 
                        for item in value
                    ]
                except:
                    # 无法序列化的列表元素
                    result[field_name] = str(value)
                    
            elif hasattr(value, '__dict__'):
                # 尝试使用__dict__
                try:
                    result[field_name] = value.__dict__
                except:
                    result[field_name] = str(value)
                    
            else:
                # 其他类型转为字符串
                result[field_name] = str(value)
        
        return result


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
