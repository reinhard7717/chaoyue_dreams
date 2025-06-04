from decimal import Decimal
import json
import urllib.parse  # 已存在，不变
from redis.asyncio import Redis  # 替换 import aioredis
from django.conf import settings
import logging
import umsgpack
from typing import Any, Dict, List, Optional, Type, Union, TypeVar, Mapping
from datetime import date, datetime
from utils import cache_constants as cc  # 导入常量
from redis.asyncio.client import Pipeline  # 替换为 redis-py 的 Pipeline 类型提示

logger = logging.getLogger("dao")

T = TypeVar('T')

# --- 自定义编码函数 ---
def custom_encode_default(obj):
    """
    为 umsgpack.packb 提供 default 回调函数。
    处理 umsgpack 默认不支持的类型，如 Decimal, datetime, date。
    Args:
        obj: 需要序列化的对象。
    Returns:
        对象的序列化友好表示（通常是字符串）。
    Raises:
        TypeError: 如果对象类型无法处理。
    """
    if isinstance(obj, Decimal):
        # 将 Decimal 对象转换为字符串
        return str(obj)
    elif isinstance(obj, datetime):
        # 将 datetime 对象转换为 ISO 格式字符串
        # (注意: BaseDAO 的解析逻辑应能处理 ISO 格式)
        return obj.isoformat()
    elif isinstance(obj, date):
        # 将 date 对象转换为 ISO 格式字符串
        return obj.isoformat()
    # 对于 umsgpack 本身不支持且此处未处理的类型，抛出 TypeError
    raise TypeError(f"对象类型 {type(obj)} 不支持 MessagePack 序列化")
# --- 结束：自定义编码函数 ---

class CacheManager:
    """
    统一管理股票量化系统的Redis缓存
    """
    
    # 默认过期时间（秒）
    DEFAULT_TIMEOUTS = {
        'rt': 60 * 60 * 24,   # 实时数据缓存1天
        'st': 86400 * 7,      # 静态数据缓存7天
        'ts': 86400 * 7,      # 时间序列缓存7天 (根据需要调整)
        'calc': 300,          # 计算结果缓存5分钟
        'user': 1800,         # 用户数据缓存30分钟
        'strategy': 86400,    # 策略数据缓存1天
    }
    
    def __init__(self):
        """初始化时不创建连接，采用懒加载方式"""
        self.redis_client: Optional[Redis] = None # 明确类型提示

    async def initialize(self):
        """异步初始化Redis客户端连接"""
        # 避免重复初始化
        if self.redis_client is not None:
            return

        # logger.info("正在初始化 Redis 客户端...")
        try:
            cache_config = settings.CACHES['default']
            location = cache_config.get('LOCATION', 'redis://localhost:6379/0')
            password = cache_config.get('OPTIONS', {}).get('PASSWORD', None)

            # --- URL 解析和密码处理逻辑保持不变 ---
            parsed_url = urllib.parse.urlparse(location)
            # 检查密码是否已在 URL 中
            if not parsed_url.password and password:
                 # 如果 URL 中没有密码但 OPTIONS 中有，则重建 URL
                 new_netloc = f"{parsed_url.username or ''}:{password}@{parsed_url.hostname}"
                 if parsed_url.port:
                     new_netloc += f":{parsed_url.port}"
                 # 使用 _replace 创建新 URL，确保路径和查询参数保留
                 new_url_parts = parsed_url._replace(netloc=new_netloc)
                 new_url = new_url_parts.geturl()
                 logger.debug("已将密码添加到 Redis 连接 URL 中。")
            else:
                 new_url = location # 使用原始或已包含密码的 URL

            # --- Redis 连接创建 ---
            # 注意：decode_responses=False 是关键，因为 umsgpack 处理 bytes
            self.redis_client = await Redis.from_url(
                new_url,
                # encoding="utf-8", # decode_responses=False 时，encoding 通常不生效或不需要
                decode_responses=False, # <--- 确保为 False
                socket_connect_timeout=cache_config.get('OPTIONS', {}).get('SOCKET_CONNECT_TIMEOUT', 10),
                socket_timeout=cache_config.get('OPTIONS', {}).get('SOCKET_TIMEOUT', 15),
                retry_on_timeout=cache_config.get('OPTIONS', {}).get('RETRY_ON_TIMEOUT', True),
                max_connections=cache_config.get('OPTIONS', {}).get('MAX_CONNECTIONS', 100)
            )
            # 测试连接 (可选但推荐)
            await self.redis_client.ping()
            # logger.info("Redis 客户端初始化并连接成功。")
        except Exception as e:
            logger.error(f"初始化 Redis 客户端失败: {e}", exc_info=True)
            self.redis_client = None # 初始化失败，重置为 None
            raise # 将异常重新抛出，以便调用者知道初始化失败

    async def _ensure_client(self):
        """(内部方法) 确保 Redis 客户端已初始化"""
        if self.redis_client is None:
            await self.initialize()
        # 再次检查，如果 initialize 失败则 redis_client 仍为 None
        if self.redis_client is None:
             raise ConnectionError("无法连接到 Redis 服务器，请检查配置和网络。")

    def get_timeout(self, cache_type: str) -> int:
        """获取指定缓存类型的过期时间"""
        return self.DEFAULT_TIMEOUTS.get(cache_type, 300)  # 默认5分钟
    
    def _serialize(self, data: Any) -> bytes:
        """
        (内部方法) 使用 umsgpack 将 Python 对象序列化为字节。
        通过 default 回调处理 Decimal, datetime, date 等特殊类型。
        """
        try:
            # 使用 default 参数传入自定义编码函数
            return umsgpack.packb(data, default=custom_encode_default, use_bin_type=True)
        except (umsgpack.UnsupportedTypeException, TypeError) as e:
            # 捕获序列化过程中无法处理的类型错误
            logger.error(f"序列化失败: 不支持的类型或自定义编码器未处理 - {e}", exc_info=True)
            # 抛出 ValueError，让调用者知道序列化失败
            raise ValueError(f"Serialization failed due to unsupported type: {e}") from e
        except Exception as e:
            # 捕获其他 umsgpack 可能的错误
            logger.error(f"序列化时发生意外错误: {e}", exc_info=True)
            raise ValueError(f"Unexpected serialization error: {e}") from e

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
        """(异步) 将数据序列化后存入 Redis"""
        try:
            await self._ensure_client() # 确保客户端已初始化
            serialized_data = self._serialize(data) # 调用已修改的序列化方法

            # 确定超时时间
            effective_timeout = timeout
            if effective_timeout is None:
                try:
                    prefix = key.split(':')[0]
                    effective_timeout = self.get_timeout(prefix)
                except IndexError:
                    effective_timeout = self.get_timeout('') # 使用默认超时

            # 执行 Redis set 命令
            if nx:
                # set if not exists
                success = await self.redis_client.set(key, serialized_data, ex=effective_timeout, nx=True)
            else:
                # 普通 set (覆盖)
                success = await self.redis_client.set(key, serialized_data, ex=effective_timeout)

            if success:
                 logger.debug(f"缓存设置成功: key='{key}', timeout={effective_timeout}s, nx={nx}")
                 return True
            else:
                 # nx=True 且键已存在时，set 返回 False
                 logger.debug(f"缓存设置失败 (nx=True 且键已存在?): key='{key}'")
                 return False
        except ValueError as e: # 捕获序列化失败
             logger.error(f"缓存设置失败 (序列化错误): key='{key}', error='{e}'")
             return False
        except ConnectionError as e: # 捕获连接错误
             logger.error(f"缓存设置失败 (Redis 连接错误): key='{key}', error='{e}'")
             return False
        except Exception as e:
            logger.error(f"缓存设置时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return False

    async def get(self, key: str, default: Any = None) -> Any:
        """(异步) 从 Redis 获取数据并反序列化"""
        try:
            await self._ensure_client()
            serialized_data = await self.redis_client.get(key)
            if serialized_data:
                logger.debug(f"缓存命中: key='{key}'")
                # 调用反序列化
                deserialized_data = self._deserialize(serialized_data)
                # 可选：如果需要，在这里调用 _restore_objects
                # return self._restore_objects(deserialized_data)
                return deserialized_data # 返回原始反序列化结果
            else:
                logger.debug(f"缓存未命中: key='{key}'")
                return default
        except ConnectionError as e:
             logger.error(f"缓存获取失败 (Redis 连接错误): key='{key}', error='{e}'")
             return default
        except Exception as e:
            logger.error(f"缓存获取时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return default

    async def get_model(self, key: str, model_class: Type[T]) -> Optional[T]:
        """(异步) 获取缓存数据并尝试转换为指定模型实例"""
        # 注意：此方法依赖于 model_class 的 __init__ 能够处理
        #       反序列化后的数据类型（例如，字符串形式的 Decimal/datetime）。
        #       如果不行，应使用 DAO 层的方法（如 BaseDAO.get_by_id）
        #       它包含更健壮的 _build_model_from_cache 逻辑。
        data = await self.get(key)
        if data and isinstance(data, dict):
            try:
                # 尝试直接用反序列化的字典初始化模型
                # 这里可能需要先调用 _restore_objects 来处理特殊字段
                # restored_data = self._restore_objects(data)
                # return model_class(**restored_data)
                # 简化：直接使用 data，假设模型能处理
                return model_class(**data)
            except TypeError as e:
                 # 捕获模型初始化时因类型不匹配等原因引发的 TypeError
                 logger.error(f"模型转换失败 (TypeError): key='{key}', model={model_class.__name__}, data={data}, error='{e}'", exc_info=True)
                 return None
            except Exception as e:
                logger.error(f"模型转换时发生未知错误: key='{key}', model={model_class.__name__}, error='{e}'", exc_info=True)
                return None
        elif data is not None:
             logger.warning(f"缓存数据不是字典，无法转换为模型: key='{key}', type={type(data)}")
             return None
        return None # 缓存未命中或数据为空

    async def delete(self, key: str) -> bool:
        """(异步) 删除指定的缓存键"""
        try:
            await self._ensure_client()
            result = await self.redis_client.delete(key)
            deleted = bool(result > 0)
            if deleted:
                 logger.debug(f"缓存删除成功: key='{key}'")
            else:
                 logger.debug(f"尝试删除缓存但键不存在: key='{key}'")
            return deleted
        except ConnectionError as e:
             logger.error(f"缓存删除失败 (Redis 连接错误): key='{key}', error='{e}'")
             return False
        except Exception as e:
            logger.error(f"缓存删除时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return False

    async def exists(self, key: str) -> bool:
        """(异步) 检查缓存键是否存在"""
        try:
            await self._ensure_client()
            # exists 返回整数 (存在的键数量)，需要转为 bool
            return bool(await self.redis_client.exists(key))
        except ConnectionError as e:
             logger.error(f"缓存检查失败 (Redis 连接错误): key='{key}', error='{e}'")
             return False # 连接失败视为不存在
        except Exception as e:
            logger.error(f"缓存检查时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return False

    async def ttl(self, key: str) -> int:
        """(异步) 获取缓存键的剩余生存时间 (秒)"""
        try:
            await self._ensure_client()
            # ttl 返回 -2 (键不存在), -1 (无过期时间), 或剩余秒数
            return await self.redis_client.ttl(key)
        except ConnectionError as e:
             logger.error(f"获取 TTL 失败 (Redis 连接错误): key='{key}', error='{e}'")
             return -2 # 连接失败视为不存在
        except Exception as e:
            logger.error(f"获取 TTL 时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return -2

    async def pipeline(self) -> Pipeline:
        """(异步) 获取一个 Redis pipeline 对象"""
        await self._ensure_client() # 确保客户端已初始化
        # redis-py 的 pipeline() 方法直接返回 pipeline 对象
        return self.redis_client.pipeline()

    async def hset(self, key: str, field: str, value: Any, timeout: Optional[int] = None) -> bool:
        """(异步) 设置哈希表中的字段值"""
        try:
            await self._ensure_client()
            serialized_value = self._serialize(value) # 序列化值

            # 使用 pipeline 保证 hset 和 expire 原子性（如果需要设置超时）
            async with self.redis_client.pipeline() as pipe:
                pipe.hset(key, field, serialized_value) # 设置哈希字段

                # 处理超时逻辑
                effective_timeout = timeout
                if effective_timeout is None:
                    # 如果未指定超时，检查哈希键是否已存在且有 TTL
                    # 如果不存在或无 TTL，则设置默认 TTL
                    current_ttl = await self.redis_client.ttl(key) # 在 pipeline 外检查 TTL
                    if current_ttl == -2 or current_ttl == -1: # 键不存在或无过期
                        try:
                            prefix = key.split(':')[0]
                            effective_timeout = self.get_timeout(prefix)
                        except IndexError:
                            effective_timeout = self.get_timeout('')
                        pipe.expire(key, effective_timeout) # 在 pipeline 中设置过期
                elif effective_timeout > 0:
                     pipe.expire(key, effective_timeout) # 设置指定的过期时间

                results = await pipe.execute()
                # hset 返回 1 (新字段) 或 0 (更新字段)
                success = isinstance(results[0], int) # 检查第一个结果是否是整数
                if success:
                     logger.debug(f"Hash 设置成功: key='{key}', field='{field}', timeout={effective_timeout}s")
                return success
        except ValueError as e: # 捕获序列化失败
             logger.error(f"Hash 设置失败 (序列化错误): key='{key}', field='{field}', error='{e}'")
             return False
        except ConnectionError as e:
             logger.error(f"Hash 设置失败 (Redis 连接错误): key='{key}', field='{field}', error='{e}'")
             return False
        except Exception as e:
            logger.error(f"Hash 设置时发生未知 Redis 错误: key='{key}', field='{field}', error='{e}'", exc_info=True)
            return False

    async def hget(self, key: str, field: str, default: Any = None) -> Any:
        """(异步) 获取哈希表中的字段值"""
        try:
            await self._ensure_client()
            serialized_value = await self.redis_client.hget(key, field)
            if serialized_value:
                logger.debug(f"Hash 获取命中: key='{key}', field='{field}'")
                return self._deserialize(serialized_value)
            else:
                logger.debug(f"Hash 获取未命中: key='{key}', field='{field}'")
                return default
        except ConnectionError as e:
             logger.error(f"Hash 获取失败 (Redis 连接错误): key='{key}', field='{field}', error='{e}'")
             return default
        except Exception as e:
            logger.error(f"Hash 获取时发生未知 Redis 错误: key='{key}', field='{field}', error='{e}'", exc_info=True)
            return default

    async def hgetall(self, key: str) -> Dict[str, Any]:
        """(异步) 获取哈希表中的所有字段和值"""
        result_dict = {}
        try:
            await self._ensure_client()
            # hgetall 返回 bytes:bytes 字典 (因为 decode_responses=False)
            raw_dict = await self.redis_client.hgetall(key)
            if not raw_dict:
                 logger.debug(f"Hash 获取全部未命中或为空: key='{key}'")
                 return {}

            logger.debug(f"Hash 获取全部命中: key='{key}', 包含 {len(raw_dict)} 个字段")
            for field_bytes, value_bytes in raw_dict.items():
                try:
                    # 需要手动解码字段名 (通常是 utf-8)
                    field_name = field_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    logger.warning(f"无法解码 Hash 字段名: key='{key}', field_bytes={field_bytes}")
                    field_name = field_bytes # 保留原始 bytes 作为 key

                # 反序列化值
                result_dict[field_name] = self._deserialize(value_bytes)
            return result_dict
        except ConnectionError as e:
             logger.error(f"Hash 获取全部失败 (Redis 连接错误): key='{key}', error='{e}'")
             return {}
        except Exception as e:
            logger.error(f"Hash 获取全部时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return {} # 出错时返回空字典

    async def mget(self, keys: List[str]) -> List[Any]:
        """(异步) 批量获取多个键的值"""
        if not keys:
            return []
        try:
            await self._ensure_client()
            # mget 返回 bytes 列表或 None 列表
            serialized_values = await self.redis_client.mget(keys)
            logger.debug(f"批量获取完成: keys={keys}")
            # 逐个反序列化
            return [self._deserialize(v) if v is not None else None for v in serialized_values]
        except ConnectionError as e:
             logger.error(f"批量获取失败 (Redis 连接错误): keys='{keys}', error='{e}'")
             return [None] * len(keys) # 保持长度一致
        except Exception as e:
            logger.error(f"批量获取时发生未知 Redis 错误: keys='{keys}', error='{e}'", exc_info=True)
            return [None] * len(keys)
    
    async def zadd(self, key: str, mapping: Mapping[Any, float], timeout: Optional[int] = None) -> Optional[int]:
        """(异步)向有序集合添加一个或多个成员，或者更新已存在成员的分数"""
        if not mapping:
            return 0
        try:
            await self._ensure_client()
            # 序列化 mapping 中的成员 (member)
            serialized_mapping = {}
            for member, score in mapping.items():
                 try:
                     serialized_mapping[self._serialize(member)] = float(score) # 分数必须是 float
                 except ValueError as e: # 捕获序列化或 float 转换错误
                      logger.error(f"ZADD 序列化/转换失败: member={member}, score={score}, error='{e}'")
                      return None # 单个成员失败则整个操作失败
            # 确定超时
            effective_timeout = timeout
            if effective_timeout is None:
                try:
                    prefix = key.split(':')[0]
                    effective_timeout = self.get_timeout(prefix)
                except IndexError:
                    effective_timeout = self.get_timeout('')
            # 使用 pipeline 保证原子性
            async with self.redis_client.pipeline() as pipe:
                pipe.zadd(key, serialized_mapping)
                if effective_timeout is not None and effective_timeout > 0:
                     pipe.expire(key, effective_timeout) # 仅在需要时设置过期
                results = await pipe.execute()
                # zadd 返回成功添加的新成员数量 (不包括更新的成员)
                added_count = results[0] if results and isinstance(results[0], int) else None
                if added_count is not None:
                     logger.debug(f"ZADD 操作成功: key='{key}', 添加/更新 {len(serialized_mapping)} 个成员, 新增 {added_count} 个, timeout={effective_timeout}s")
                return added_count
        except ConnectionError as e:
             logger.error(f"ZADD 操作失败 (Redis 连接错误): key='{key}', error='{e}'")
             return None
        except Exception as e:
            logger.error(f"ZADD 操作时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return None

    async def zrangebyscore(self, key: str, min_score: Union[float, str], max_score: Union[float, str],
                            withscores: bool = False) -> Optional[List[Any]]:
        """(异步) 通过分数区间返回有序集合的成员"""
        try:
            await self._ensure_client()
            # zrangebyscore 返回 bytes 列表 (成员) 或 (bytes, float) 元组列表 (如果 withscores=True)
            serialized_result = await self.redis_client.zrangebyscore(key, min_score, max_score, withscores=withscores)

            if serialized_result is None:
                logger.debug(f"ZRANGEBYSCORE 未找到匹配项: key='{key}', range=[{min_score}, {max_score}]")
                return [] # 返回空列表

            logger.debug(f"ZRANGEBYSCORE 命中: key='{key}', range=[{min_score}, {max_score}], withscores={withscores}")
            deserialized_list = []
            if withscores:
                # 处理 (成员, 分数) 元组列表
                for member_bytes, score in serialized_result:
                    deserialized_member = self._deserialize(member_bytes)
                    if deserialized_member is not None:
                        deserialized_list.append((deserialized_member, score))
                    else:
                        logger.warning(f"ZRANGEBYSCORE 反序列化成员失败: key='{key}', member_bytes={member_bytes}")
            else:
                # 处理成员列表
                for member_bytes in serialized_result:
                    deserialized_member = self._deserialize(member_bytes)
                    if deserialized_member is not None:
                        deserialized_list.append(deserialized_member)
                    else:
                         logger.warning(f"ZRANGEBYSCORE 反序列化成员失败: key='{key}', member_bytes={member_bytes}")

            return deserialized_list
        except ConnectionError as e:
             logger.error(f"ZRANGEBYSCORE 操作失败 (Redis 连接错误): key='{key}', error='{e}'")
             return None
        except Exception as e:
            logger.error(f"ZRANGEBYSCORE 操作时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return None

    async def zrange_by_limit(self, key: str, limit: int, desc: bool = True, withscores: bool = False) -> Optional[List[Any]]:
        """(异步) 按排名获取有序集合成员 (默认降序)"""
        if limit <= 0:
            return []
        try:
            await self._ensure_client()
            start = 0
            end = limit - 1
            if desc:
                # zrevrange 返回 bytes 列表或 (bytes, float) 元组列表
                serialized_result = await self.redis_client.zrevrange(key, start, end, withscores=withscores)
            else:
                # zrange 返回 bytes 列表或 (bytes, float) 元组列表
                serialized_result = await self.redis_client.zrange(key, start, end, withscores=withscores)

            if serialized_result is None:
                logger.debug(f"ZRANGE/ZREVRANGE 未找到成员: key='{key}', limit={limit}, desc={desc}")
                return []

            logger.debug(f"ZRANGE/ZREVRANGE 命中: key='{key}', limit={limit}, desc={desc}, withscores={withscores}")
            deserialized_list = []
            if withscores:
                for member_bytes, score in serialized_result:
                    deserialized_member = self._deserialize(member_bytes)
                    if deserialized_member is not None:
                        deserialized_list.append((deserialized_member, score))
                    else:
                        logger.warning(f"ZRANGE/ZREVRANGE 反序列化成员失败: key='{key}', member_bytes={member_bytes}")
            else:
                for member_bytes in serialized_result:
                    deserialized_member = self._deserialize(member_bytes)
                    if deserialized_member is not None:
                        deserialized_list.append(deserialized_member)
                    else:
                        logger.warning(f"ZRANGE/ZREVRANGE 反序列化成员失败: key='{key}', member_bytes={member_bytes}")

            return deserialized_list
        except ConnectionError as e:
             logger.error(f"ZRANGE/ZREVRANGE 操作失败 (Redis 连接错误): key='{key}', error='{e}'")
             return None
        except Exception as e:
            logger.error(f"ZRANGE/ZREVRANGE 操作时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return None

    async def ztrim_by_rank(self, key: str, keep_latest: int) -> Optional[int]:
        """(异步) 修剪有序集合，只保留最新的 N 个成员 (按分数降序)"""
        if keep_latest <= 0:
            logger.warning(f"ZTRIMBYRANK: keep_latest 必须大于 0, key='{key}'")
            # 可以选择删除所有元素或返回 0
            # return await self.redis_client.delete(key) # 删除所有
            return 0 # 不做任何操作

        try:
            await self._ensure_client()
            # zremrangebyrank(key, start, stop) 移除指定排名范围内的成员
            # 排名从 0 开始，负数表示从尾部开始 (-1 是最高分，-2 是第二高分)
            # 要保留最新的 keep_latest 个，需要移除排名在 0 到 -(keep_latest + 1) 之间的成员
            # 例如：保留 100 个，移除排名 0 到 -101 的成员
            stop_rank = -(keep_latest + 1)
            removed_count = await self.redis_client.zremrangebyrank(key, 0, stop_rank)
            if removed_count is not None:
                 logger.debug(f"ZTRIMBYRANK 操作成功: key='{key}', 保留 {keep_latest} 个, 移除了 {removed_count} 个")
            return removed_count
        except ConnectionError as e:
             logger.error(f"ZTRIMBYRANK 操作失败 (Redis 连接错误): key='{key}', error='{e}'")
             return None
        except Exception as e:
            logger.error(f"ZTRIMBYRANK 操作时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return None

    async def zadd_and_trim(self, key: str, mapping: Mapping[Any, float], limit: int, timeout: Optional[int] = None) -> Optional[int]:
        """(异步) 原子地添加成员并修剪有序集合，保持指定数量的最新成员"""
        if not mapping or not isinstance(mapping, dict):
            return 0
        if limit <= 0:
             logger.warning(f"ZADD_AND_TRIM: limit 必须大于 0, key='{key}'")
             return 0
        try:
            await self._ensure_client()
            # 序列化成员
            serialized_mapping = {}
            for member, score in mapping.items():
                 try:
                     serialized_mapping[self._serialize(member)] = float(score)
                 except ValueError as e:
                      logger.error(f"ZADD_AND_TRIM 序列化/转换失败: member={member}, score={score}, error='{e}'")
                      return None
            # 确定超时
            effective_timeout = timeout
            if effective_timeout is None:
                try:
                    prefix = key.split(':')[0]
                    effective_timeout = self.get_timeout(prefix)
                except IndexError:
                    effective_timeout = self.get_timeout('')
            # 使用 pipeline 保证原子性
            async with self.redis_client.pipeline() as pipe:
                # 1. 添加新成员
                pipe.zadd(key, serialized_mapping)
                # 2. 修剪旧成员 (保留最新的 limit 个)
                stop_rank = -(limit + 1)
                pipe.zremrangebyrank(key, 0, stop_rank)
                # 3. 设置/更新过期时间
                if effective_timeout is not None and effective_timeout > 0:
                    pipe.expire(key, effective_timeout)
                results = await pipe.execute()
                # results[0] 是 zadd 的结果 (新增数量)
                # results[1] 是 zremrangebyrank 的结果 (移除数量)
                added_count = results[0] if results and len(results) > 0 and isinstance(results[0], int) else None
                removed_count = results[1] if results and len(results) > 1 and isinstance(results[1], int) else None
                if added_count is not None and removed_count is not None:
                     logger.debug(f"ZADD_AND_TRIM 操作成功: key='{key}', 添加/更新 {len(serialized_mapping)} 个, 新增 {added_count} 个, 移除 {removed_count} 个, 限制 {limit}, timeout={effective_timeout}s")
                     return added_count # 返回新增的数量
                else:
                     logger.error(f"ZADD_AND_TRIM pipeline 执行结果异常: results={results}")
                     return None
        except ConnectionError as e:
             logger.error(f"ZADD_AND_TRIM 操作失败 (Redis 连接错误): key='{key}', error='{e}'")
             return None
        except Exception as e:
            logger.error(f"ZADD_AND_TRIM 操作时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return None
        
    # --- 辅助方法 ---
    def generate_key(self, cache_type: str, *args: str) -> str:
        """生成标准化的缓存键"""
        # 使用常量中定义的模板或默认格式
        key_template = getattr(cc, f"{cache_type.upper()}_CACHE_KEY_TEMPLATE", None)
        if key_template:
            try:
                # 尝试格式化模板，需要确保 args 数量匹配
                return key_template.format(*args)
            except (IndexError, KeyError) as e:
                 logger.warning(f"格式化缓存键模板失败: template='{key_template}', args={args}, error='{e}'. 使用默认格式。")
                 # 回退到默认格式
                 return f"{cache_type}:" + ":".join(map(str, args))
        else:
            # 默认格式
            return f"{cache_type}:" + ":".join(map(str, args))

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
