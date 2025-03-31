import json
import zlib
import logging
import inspect
from typing import Any, Dict, List, Optional, Type, Union, TypeVar
from datetime import datetime, timedelta
from django.core.cache import cache
from django_redis import get_redis_connection
from redis.client import Pipeline

from stock_models.stock_realtime import StockRealtimeData

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheManager:
    """
    统一管理股票量化系统的Redis缓存
    """
    
    # 缓存类型枚举
    CACHE_TYPES = {
        'rt': 'realtime',     # 实时数据
        'st': 'static',       # 静态数据
        'ts': 'timeseries',   # 时间序列
        'calc': 'calculation', # 计算结果
        'user': 'user',       # 用户数据
    }
    
    # 默认过期时间（秒）
    DEFAULT_TIMEOUTS = {
        'rt': 60,             # 实时数据缓存1分钟
        'st': 86400,          # 静态数据缓存1天
        'ts': 3600,           # 时间序列缓存1小时
        'calc': 300,          # 计算结果缓存5分钟
        'user': 1800,         # 用户数据缓存30分钟
    }
    
    # 是否压缩大数据集（超过阈值的数据会被压缩）
    COMPRESSION_THRESHOLD = 10 * 1024  # 10KB
    
    def __init__(self):
        self.redis_client = get_redis_connection("default")
    
    def generate_key(self, cache_type: str, entity_type: str, entity_id: str, 
                subtype: Optional[str] = None, params: Optional[Dict] = None,
                date: Optional[Union[str, datetime]] = None) -> str:
        """
        生成标准化的缓存键
        
        Args:
            cache_type: 缓存类型前缀 (rt, st, ts, calc, user)
            entity_type: 实体类型 (stock, index, strategy等)
            entity_id: 实体ID (股票代码、指数代码、策略ID等)
            subtype: 子类型 (可选，如daily, price等)
            date: 日期 (可选，对于时间相关数据)
            
        Returns:
            格式化的缓存键
        """
        if cache_type not in self.CACHE_TYPES:
            raise ValueError(f"无效的缓存类型: {cache_type}")
        
        # 构建基本键
        key_parts = [cache_type, entity_type, entity_id]
        
        # 添加可选部分
        if subtype:
            key_parts.append(subtype)

        if params:
            param_parts = []
            for k, v in sorted(params.items()):  # 排序确保顺序一致
                if v is not None:
                    param_parts.append(f"{k}:{v}")
            if param_parts:
                key_parts.append(":".join(param_parts))
        
        if date:
            if isinstance(date, datetime):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date)
            key_parts.append(date_str)
        
        # 生成冒号分隔的键
        return ':'.join(key_parts)
    
    def get_timeout(self, cache_type: str) -> int:
        """获取指定缓存类型的过期时间"""
        return self.DEFAULT_TIMEOUTS.get(cache_type, 300)  # 默认5分钟
    
    def _serialize(self, data: Any) -> bytes:
        """序列化数据，大数据自动压缩"""
        json_data = json.dumps(data, ensure_ascii=False)
        
        # 检查是否需要压缩
        if len(json_data) > self.COMPRESSION_THRESHOLD:
            # 压缩并添加标记前缀
            return b'c:' + zlib.compress(json_data.encode())
        
        # 不压缩但添加标记前缀
        return b'r:' + json_data.encode()
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据，自动处理压缩"""
        if not data:
            return None
            
        # 检查前缀判断是否需要解压
        prefix = data[:2]
        content = data[2:]
        
        if prefix == b'c:':
            # 解压缩
            json_data = zlib.decompress(content).decode()
        elif prefix == b'r:':
            # 不需要解压
            json_data = content.decode()
        else:
            # 兼容旧数据
            try:
                return json.loads(data.decode())
            except:
                logger.error("无法解析缓存数据")
                return None
                
        return json.loads(json_data)
    
    def set(self, key: str, data: Any, timeout: Optional[int] = None, nx: bool = False) -> bool:
        """
        保存数据到缓存
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            timeout: 过期时间(秒)，默认根据键类型自动选择
            nx: 仅在键不存在时设置
            
        Returns:
            操作是否成功
        """
        try:
            if timeout is None:
                # 根据键前缀自动选择过期时间
                prefix = key.split(':')[0]
                timeout = self.get_timeout(prefix)
            
            serialized_data = self._serialize(data)
            
            if nx:
                return self.redis_client.set(key, serialized_data, ex=timeout, nx=True)
            else:
                return self.redis_client.set(key, serialized_data, ex=timeout)
        except Exception as e:
            logger.error(f"缓存保存失败: {key}, 错误: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        从缓存获取数据
        
        Args:
            key: 缓存键
            default: 默认值(如果键不存在)
            
        Returns:
            缓存的数据，如不存在则返回default
        """
        try:
            data = self.redis_client.get(key)
            if data:
                return self._deserialize(data)
            return default
        except Exception as e:
            logger.error(f"缓存读取失败: {key}, 错误: {str(e)}")
            return default
    
    def get_model(self, key: str, model_class: Type[T]) -> Optional[T]:
        """
        从缓存获取数据并转换为指定模型实例
        
        Args:
            key: 缓存键
            model_class: 要转换成的模型类
            
        Returns:
            模型实例，如不存在则返回None
        """
        data = self.get(key)
        if data:
            try:
                if isinstance(data, dict):
                    return model_class(**data)
                else:
                    logger.warning(f"缓存数据不是字典类型，无法转换为模型: {key}")
                    return None
            except Exception as e:
                logger.error(f"模型转换失败: {key}, 错误: {str(e)}")
                return None
        return None
    
    def delete(self, key: str) -> bool:
        """删除缓存键"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"缓存删除失败: {key}, 错误: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"缓存检查失败: {key}, 错误: {str(e)}")
            return False
    
    def ttl(self, key: str) -> int:
        """获取键的剩余生存时间(秒)"""
        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"获取TTL失败: {key}, 错误: {str(e)}")
            return -2  # 不存在的键
    
    def pipeline(self) -> Pipeline:
        """获取Redis管道对象用于批量操作"""
        return self.redis_client.pipeline()
    
    # Hash类型操作
    def hset(self, key: str, field: str, value: Any, timeout: Optional[int] = None) -> bool:
        """设置Hash字段值"""
        try:
            serialized = self._serialize(value)
            result = self.redis_client.hset(key, field, serialized)
            
            # 如果设置了过期时间，更新键过期时间
            if timeout is not None:
                self.redis_client.expire(key, timeout)
            elif not self.redis_client.ttl(key) > 0:
                # 如果键没有过期时间，根据键前缀设置默认过期时间
                prefix = key.split(':')[0]
                self.redis_client.expire(key, self.get_timeout(prefix))
                
            return bool(result)
        except Exception as e:
            logger.error(f"Hash设置失败: {key}.{field}, 错误: {str(e)}")
            return False
    
    def hget(self, key: str, field: str, default: Any = None) -> Any:
        """获取Hash字段值"""
        try:
            data = self.redis_client.hget(key, field)
            if data:
                return self._deserialize(data)
            return default
        except Exception as e:
            logger.error(f"Hash获取失败: {key}.{field}, 错误: {str(e)}")
            return default
    
    def hgetall(self, key: str) -> Dict[str, Any]:
        """获取Hash所有字段"""
        try:
            data = self.redis_client.hgetall(key)
            result = {}
            for field, value in data.items():
                field_name = field.decode() if isinstance(field, bytes) else field
                result[field_name] = self._deserialize(value)
            return result
        except Exception as e:
            logger.error(f"Hash获取全部失败: {key}, 错误: {str(e)}")
            return {}
    
    # 批量操作
    def mget(self, keys: List[str]) -> List[Any]:
        """批量获取多个键的值"""
        try:
            values = self.redis_client.mget(keys)
            return [self._deserialize(v) if v else None for v in values]
        except Exception as e:
            logger.error(f"批量获取失败: {keys}, 错误: {str(e)}")
            return [None] * len(keys)
    
    def mget_models(self, keys: List[str], model_class: Type[T]) -> List[Optional[T]]:
        """批量获取多个键并转换为模型实例"""
        values = self.mget(keys)
        result = []
        
        for value in values:
            if value and isinstance(value, dict):
                try:
                    result.append(model_class(**value))
                except Exception as e:
                    logger.error(f"模型转换失败, 错误: {str(e)}")
                    result.append(None)
            else:
                result.append(None)
                
        return result


# 使用示例

cache_manager = CacheManager()

# 1. 保存股票实时数据
def save_stock_realtime(stock_code: str, data: dict):
    key = cache_manager.generate_key('rt', 'stock', stock_code, 'quote')
    cache_manager.set(key, data)

# 2. 获取股票实时数据并转换为模型实例
def get_stock_realtime(stock_code: str) -> Optional['StockRealtimeData']:
    from .models import StockRealTimeData  # 导入模型
    
    key = cache_manager.generate_key('rt', 'stock', stock_code, 'quote')
    return cache_manager.get_model(key, StockRealTimeData)

# 3. 保存指数K线数据
def save_index_kline(index_code: str, period: str, date: str, data: dict):
    key = cache_manager.generate_key('ts', 'index', index_code, period, date)
    # 时间序列数据较大，可能会自动压缩
    cache_manager.set(key, data)

# 4. 批量获取多只股票实时数据
def get_batch_stock_realtime(stock_codes: List[str]) -> Dict[str, dict]:
    keys = [cache_manager.generate_key('rt', 'stock', code, 'quote') for code in stock_codes]
    values = cache_manager.mget(keys)
    
    result = {}
    for code, value in zip(stock_codes, values):
        if value:
            result[code] = value
    
    return result

# 5. 使用Hash存储市场概览数据
def save_market_overview(overview_data: dict):
    key = cache_manager.generate_key('rt', 'market', 'overview')
    # 逐字段保存
    for field, value in overview_data.items():
        cache_manager.hset(key, field, value)

# 6. 获取市场概览数据
def get_market_overview() -> dict:
    key = cache_manager.generate_key('rt', 'market', 'overview')
    return cache_manager.hgetall(key)
