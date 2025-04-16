import json
import urllib.parse  # 已存在，不变
from redis.asyncio import Redis  # 替换 import aioredis
from django.conf import settings
import logging
import umsgpack
from typing import Any, Dict, List, Optional, Type, Union, TypeVar, Mapping
from datetime import datetime
from utils import cache_constants as cc  # 导入常量
from redis.asyncio.client import Pipeline  # 替换为 redis-py 的 Pipeline 类型提示

logger = logging.getLogger("dao")

T = TypeVar('T')

class CacheManager:
    """
    统一管理股票量化系统的Redis缓存
    """
    
    # 默认过期时间（秒）
    DEFAULT_TIMEOUTS = {
        'rt': 60 * 5,             # 实时数据缓存5分钟
        'st': 86400 * 7,          # 静态数据缓存1天
        'ts': 86400 * 7,      # 时间序列缓存7天 (根据需要调整)
        'calc': 300,          # 计算结果缓存5分钟
        'user': 1800,         # 用户数据缓存30分钟
        'strategy': 86400,     # 策略数据缓存1天
    }
    
    def __init__(self):
        self.redis_client = None  # 同步初始化

    async def initialize(self):
        cache_config = settings.CACHES['default']
        location = cache_config.get('LOCATION', 'redis://localhost:6379/0')
        password = cache_config.get('OPTIONS', {}).get('PASSWORD', None)
        
        parsed_url = urllib.parse.urlparse(location)
        
        # 手动构建 netloc，包含用户名和密码
        netloc_parts = []
        if parsed_url.username:
            netloc_parts.append(parsed_url.username)
        if password:
            netloc_parts.append(f":{password}")
        netloc_parts.append(f"@{parsed_url.hostname}")
        if parsed_url.port:
            netloc_parts.append(f":{parsed_url.port}")
        
        new_netloc = ''.join(netloc_parts).lstrip('@')  # 移除前导 '@' 如果存在
        if not new_netloc:  # 如果 netloc 为空，使用原始 hostname
            new_netloc = parsed_url.hostname
        
        new_url = parsed_url._replace(netloc=new_netloc).geturl()  # 现在只使用有效的字段
        
        self.redis_client = await Redis.from_url(
            new_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=cache_config.get('OPTIONS', {}).get('SOCKET_CONNECT_TIMEOUT', 10),
            socket_timeout=cache_config.get('OPTIONS', {}).get('SOCKET_TIMEOUT', 15),
            retry_on_timeout=cache_config.get('OPTIONS', {}).get('RETRY_ON_TIMEOUT', True),
            max_connections=cache_config.get('OPTIONS', {}).get('MAX_CONNECTIONS', 100)
        )
    
    def get_timeout(self, cache_type: str) -> int:
        """获取指定缓存类型的过期时间"""
        return self.DEFAULT_TIMEOUTS.get(cache_type, 300)  # 默认5分钟
    
    def _serialize(self, data: Any) -> bytes:
        """序列化数据，大数据自动压缩"""
        try:
            return umsgpack.packb(data, use_bin_type=True)
        except Exception as e:
            logger.error(f"序列化失败: {e}", exc_info=True)
            return json.dumps(str(data)).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        if not data:
            return None
        try:
            return umsgpack.unpackb(data, raw=False)
        except Exception as e:
            logger.error(f"反序列化失败: {e}", exc_info=True)
            return None

    def _restore_objects(self, data: Any) -> Any:
        """还原特殊对象"""
        if not isinstance(data, dict):
            return data
        result = {}
        if "index" in data and isinstance(data["index"], str):
            code = data["index"]
            try:
                from stock_models.index import IndexInfo
                try:
                    index_info = IndexInfo.objects.get(code=code)
                except Exception:
                    index_info = IndexInfo(code=code)
                    logger.warning(f"无法从数据库获取IndexInfo(code={code})，创建了基本对象")
                result["index"] = index_info
            except ImportError as e:
                logger.error(f"导入模型类失败: {e}")
                result["index"] = data["index"]
            except Exception as e:
                logger.error(f"还原指数/股票对象失败: {e}")
                result["index"] = data["index"]
        
        if "stock" in data and isinstance(data["stock"], str):
            code = data["stock"]
            from stock_models.stock_basic import StockInfo
            try:
                stock_info = StockInfo.objects.get(stock_code=code)
            except Exception:
                stock_info = StockInfo(stock_code=code)
                logger.warning(f"无法从数据库获取StockInfo(code={code})，创建了基本对象")
            result["stock"] = stock_info
        
        if "trade_time" in data and isinstance(data["trade_time"], str):
            try:
                result["trade_time"] = datetime.fromisoformat(data["trade_time"])
            except Exception as e:
                logger.error(f"解析时间字段失败: {e}")
                result["trade_time"] = data["trade_time"]
        
        for key, value in data.items():
            if key not in result:
                result[key] = value
        
        return result

    async def set(self, key: str, data: Any, timeout: Optional[int] = None, nx: bool = False) -> bool:
        serialized_data = self._serialize(data)
        try:
            if timeout is None:
                prefix = key.split(':')[0]
                timeout = self.get_timeout(prefix)
            if nx:
                return await self.redis_client.set(key, serialized_data, ex=timeout, nx=True)
            else:
                return await self.redis_client.set(key, serialized_data, ex=timeout)
        except Exception as e:
            logger.error(f"缓存保存失败: {key}, 错误: {str(e)}")
            return False
    
    async def get(self, key: str, default: Any = None) -> Any:
        try:
            data = await self.redis_client.get(key)
            if data:
                return self._deserialize(data)
            return default
        except Exception as e:
            logger.error(f"缓存读取失败: {key}, 错误: {str(e)}")
            return default
    
    async def get_model(self, key: str, model_class: Type[T]) -> Optional[T]:
        data = await self.get(key)
        if data and isinstance(data, dict):
            try:
                return model_class(**data)
            except Exception as e:
                logger.error(f"模型转换失败: {key}, 错误: {str(e)}")
                return None
        return None
    
    async def delete(self, key: str) -> bool:
        try:
            return bool(await self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"缓存删除失败: {key}, 错误: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"缓存检查失败: {key}, 错误: {str(e)}")
            return False
    
    async def ttl(self, key: str) -> int:
        try:
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"获取TTL失败: {key}, 错误: {str(e)}")
            return -2  # 不存在的键
    
    async def pipeline(self) -> Pipeline:
        return self.redis_client.pipeline()  # redis-py 的 pipeline 直接可用
    
    async def hset(self, key: str, field: str, value: Any, timeout: Optional[int] = None) -> bool:
        serialized = self._serialize(value)
        try:
            result = await self.redis_client.hset(key, field, serialized)
            if timeout is not None:
                await self.redis_client.expire(key, timeout)
            elif not (await self.redis_client.ttl(key) > 0):
                prefix = key.split(':')[0]
                await self.redis_client.expire(key, self.get_timeout(prefix))
            return bool(result)
        except Exception as e:
            logger.error(f"Hash设置失败: {key}.{field}, 错误: {str(e)}")
            return False
    
    async def hget(self, key: str, field: str, default: Any = None) -> Any:
        try:
            data = await self.redis_client.hget(key, field)
            if data:
                return self._deserialize(data)
            return default
        except Exception as e:
            logger.error(f"Hash获取失败: {key}.{field}, 错误: {str(e)}")
            return default
    
    async def hgetall(self, key: str) -> Dict[str, Any]:
        try:
            data = await self.redis_client.hgetall(key)
            result = {}
            for field, value in data.items():
                field_name = field.decode() if isinstance(field, bytes) else field  # 注意：decode_responses=True 可能已处理
                result[field_name] = self._deserialize(value)
            return result
        except Exception as e:
            logger.error(f"Hash获取全部失败: {key}, 错误: {str(e)}",exc_info=True)
            return {}
    
    async def mget(self, keys: List[str]) -> List[Any]:
        try:
            values = await self.redis_client.mget(keys)
            return [self._deserialize(v) if v else None for v in values]
        except Exception as e:
            logger.error(f"批量获取失败: {keys}, 错误: {str(e)}")
            return [None] * len(keys)
    
    async def zadd(self, key: str, mapping: Mapping[bytes, float], timeout: Optional[int] = None) -> Optional[int]:
        if not mapping:
            return 0
        try:
            if timeout is None:
                prefix = key.split(':')[0]
                timeout = self.get_timeout(prefix)
            pipe = self.redis_client.pipeline()
            pipe.zadd(key, mapping)
            pipe.expire(key, timeout)
            results = await pipe.execute()
            return results[0] if results and isinstance(results[0], int) else None
        except Exception as e:
            logger.error(f"ZADD 操作失败: key={key}, 错误: {str(e)}")
            return None
    
    async def zrangebyscore(self, key: str, min_score: Union[float, str], max_score: Union[float, str]) -> Optional[List[Any]]:
        try:
            serialized_members = await self.redis_client.zrangebyscore(key, min_score, max_score)
            if serialized_members is None:
                return []
            deserialized_list = [self._deserialize(sm) for sm in serialized_members if self._deserialize(sm) is not None]
            return deserialized_list
        except Exception as e:
            logger.error(f"ZRANGEBYSCORE 操作失败: key={key}, 错误: {str(e)}")
            return None
    
    async def zrange_by_limit(self, key: str, limit: int) -> Optional[List[Any]]:
        try:
            serialized_members = await self.redis_client.zrevrange(key, 0, limit - 1)
            if serialized_members is None:
                return []
            deserialized_list = [self._deserialize(sm) for sm in serialized_members if self._deserialize(sm) is not None]
            return deserialized_list
        except Exception as e:
            logger.error(f"ZREVRANGE 操作失败: key={key}, 错误: {str(e)}")
            return None
    
    async def ztrim_by_rank(self, key: str, keep_latest: int) -> Optional[int]:
        if keep_latest <= 0:
            return 0
        try:
            current_size = await self.redis_client.zcard(key)
            if current_size is None or current_size <= keep_latest:
                return 0
            remove_count = current_size - keep_latest
            end_rank = remove_count - 1
            removed_count = await self.redis_client.zremrangebyrank(key, 0, end_rank)
            return removed_count
        except Exception as e:
            logger.error(f"ZTRIMBYRANK 操作失败: key={key}, 错误: {str(e)}")
            return None
    
    async def zadd_and_trim(self, key: str, mapping: Mapping[Any, float], limit: int, timeout: Optional[int] = None) -> Optional[int]:
        if not mapping or not isinstance(mapping, dict):
            return 0
        try:
            serialized_mapping = {self._serialize(member): score for member, score in mapping.items()}
            if timeout is None:
                prefix = key.split(':')[0]
                timeout = self.get_timeout(prefix)
            pipe = self.redis_client.pipeline()
            pipe.zadd(key, serialized_mapping)
            pipe.zremrangebyrank(key, 0, -(limit + 1))
            pipe.expire(key, timeout)
            results = await pipe.execute()
            return results[0] if results and isinstance(results[0], int) else None
        except Exception as e:
            logger.error(f"ZADD_AND_TRIM 操作失败: key={key}, 错误: {str(e)}")
            return None
    
    async def trim_cache_zset(self, cache_key: str, limit: int) -> Optional[int]:
        if limit <= 0:
            return 0
        try:
            return await self.ztrim_by_rank(key=cache_key, keep_latest=limit)
        except Exception as e:
            logger.error(f"缓存修剪时发生异常: key={cache_key}, error={e}")
            return None

# 使用示例（异步版本）保持不变
async def save_stock_realtime(cache_manager: CacheManager, stock_code: str, data: dict):
    key = cache_manager.generate_key('rt', 'stock', stock_code, 'quote')
    await cache_manager.set(key, data)

async def get_stock_realtime(cache_manager: CacheManager, stock_code: str) -> Optional[Any]:
    from stock_models.stock_realtime import StockRealtimeData
    key = cache_manager.generate_key('rt', 'stock', stock_code, 'quote')
    return await cache_manager.get_model(key, StockRealtimeData)

async def save_index_kline(cache_manager: CacheManager, index_code: str, period: str, date: str, data: dict):
    key = cache_manager.generate_key('ts', 'index', index_code, period, date)
    await cache_manager.set(key, data)

async def get_batch_stock_realtime(cache_manager: CacheManager, stock_codes: List[str]) -> Dict[str, dict]:
    keys = [cache_manager.generate_key('rt', 'stock', code, 'quote') for code in stock_codes]
    values = await cache_manager.mget(keys)
    result = {}
    for code, value in zip(stock_codes, values):
        if value:
            result[code] = value
    return result

async def save_market_overview(cache_manager: CacheManager, overview_data: dict):
    key = cache_manager.generate_key('rt', 'market', 'overview')
    for field, value in overview_data.items():
        await cache_manager.hset(key, field, value)

async def get_market_overview(cache_manager: CacheManager) -> dict:
    key = cache_manager.generate_key('rt', 'market', 'overview')
    return await cache_manager.hgetall(key)
