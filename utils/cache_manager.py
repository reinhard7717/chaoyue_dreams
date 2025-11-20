from decimal import Decimal
import json
import urllib.parse  # 已存在，不变
from redis.asyncio import Redis  # 替换 import aioredis
from django.conf import settings
import threading
import asyncio
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

# --- 递归数据清洗函数 ---
def _clean_data_for_serialization(data: Any) -> Any:
    """
    递归地遍历数据结构，将不支持 umsgpack 的类型转换为支持的类型。
    - Decimal -> str
    - datetime -> str (ISO format)
    - date -> str (ISO format)
    """
    if isinstance(data, dict):
        # 如果是字典，递归处理它的每一个值
        return {k: _clean_data_for_serialization(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 如果是列表，递归处理它的每一个元素
        return [_clean_data_for_serialization(i) for i in data]
    elif isinstance(data, Decimal):
        # 将 Decimal 转换为字符串
        return str(data)
    elif isinstance(data, datetime):
        # 将 datetime 转换为 ISO 格式字符串
        return data.isoformat()
    elif isinstance(data, date):
        # 将 date 转换为 ISO 格式字符串
        return data.isoformat()
    else:
        # 对于其他类型，保持原样
        return data

class CacheManager:
    """
    【V3.3 - 工业级并发安全单例】
    - 使用一个同步的 threading.Lock 来保护一个 per-loop 上下文的字典。
    - 每个事件循环都有自己独立的 Redis 客户端和 asyncio.Lock，从根本上杜绝跨循环调用问题。
    - 实现了在复杂并发环境（如 Celery + asyncio.gather）下的终极健壮性。
    """
    _instance = None
    _lock = threading.Lock() # 用于保护 _instance 创建的同步锁
    # 默认过期时间（秒）
    DEFAULT_TIMEOUTS = {
        'rt': 60 * 60 * 24,   # 实时数据缓存1天
        'st': 86400 * 7,      # 静态数据缓存7天
        'ts': 86400 * 7,      # 时间序列缓存7天 (根据需要调整)
        'calc': 300,          # 计算结果缓存5分钟
        'user': 1800,         # 用户数据缓存30分钟
        'strategy': 86400,    # 策略数据缓存1天
    }
    def __new__(cls, *args, **kwargs):
        # 使用线程安全的双重检查锁定来创建单例
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    print("DEBUG: Initializing new CacheManager singleton instance...")
                    instance = super().__new__(cls)
                    # MODIFIED: 初始化上下文管理器和保护它的同步锁
                    instance._contexts = {}
                    instance._context_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance
    def __init__(self):
        # 初始化方法保持不变
        if hasattr(self, 'is_initialized') and self.is_initialized:
            return
        self.is_initialized = True
        print("DEBUG: CacheManager __init__ is executing (once per instance).")
    # MODIFIED: initialize 方法现在是私有的，因为它由 _ensure_client 内部调用
    async def _initialize_for_loop(self, loop: asyncio.AbstractEventLoop):
        """
        为指定的 event loop 初始化 Redis 客户端连接。
        """
        # print(f"DEBUG: 正在为 Event Loop {id(loop)} 初始化 Redis 客户端连接池...")
        try:
            cache_config = settings.CACHES['default']
            location = cache_config.get('LOCATION', 'redis://localhost:6379/0')
            options = cache_config.get('OPTIONS', {})
            password = options.get('PASSWORD', None)
            # ... URL 和连接池参数处理逻辑保持不变 ...
            parsed_url = urllib.parse.urlparse(location)
            if not parsed_url.password and password:
                 new_netloc = f"{parsed_url.username or ''}:{password}@{parsed_url.hostname}"
                 if parsed_url.port:
                     new_netloc += f":{parsed_url.port}"
                 new_url_parts = parsed_url._replace(netloc=new_netloc)
                 new_url = new_url_parts.geturl()
            else:
                 new_url = location
            pool_kwargs = options.get('CONNECTION_POOL_KWARGS', {})
            max_conns = pool_kwargs.get('max_connections', options.get('MAX_CONNECTIONS', 100))
            # print(f"DEBUG: Redis 连接池最大连接数设置为: {max_conns}")
            redis_client = await Redis.from_url(
                new_url,
                decode_responses=False,
                socket_connect_timeout=options.get('SOCKET_CONNECT_TIMEOUT', 10),
                socket_timeout=options.get('SOCKET_TIMEOUT', 15),
                retry_on_timeout=options.get('RETRY_ON_TIMEOUT', True),
                max_connections=max_conns
            )
            await redis_client.ping()
            # print(f"DEBUG: Redis 客户端连接池为 Event Loop {id(loop)} 初始化并连接成功。")
            return redis_client
        except Exception as e:
            logger.error(f"为 Event Loop {id(loop)} 初始化 Redis 客户端失败: {e}", exc_info=True)
            raise
    # MODIFIED: 这是最核心的修改，重写 _ensure_client 方法
    async def _ensure_client(self) -> Redis:
        """
        【核心 V3.4】确保返回一个与当前事件循环匹配的、已初始化的 Redis 客户端。
        此方法修复了可能返回 None 的竞争条件。
        """
        current_loop = asyncio.get_running_loop()
        loop_id = id(current_loop)
        # 快速路径: 如果客户端已完全初始化，直接返回
        if loop_id in self._contexts and self._contexts[loop_id].get('client'):
            return self._contexts[loop_id]['client']
        # 慢速路径: 上下文不存在或客户端未初始化
        # 使用同步锁保护上下文词典的创建，防止多线程冲突
        with self._context_lock:
            # 双重检查：可能在等待锁时，其他线程已创建了上下文
            if loop_id not in self._contexts:
                # print(f"DEBUG: 正在为 Event Loop {loop_id} 创建新的上下文...")
                # 只创建上下文结构，客户端初始化由异步锁保护
                self._contexts[loop_id] = {
                    'client': None,
                    'lock': asyncio.Lock()
                }
        # 获取当前事件循环专属的异步锁
        loop_async_lock = self._contexts[loop_id]['lock']
        # 使用异步锁来保护 I/O 密集型的初始化过程，防止同一事件循环内的多协程冲突
        async with loop_async_lock:
            # 三重检查：可能在等待异步锁时，已有协程完成了初始化
            if self._contexts[loop_id]['client']:
                return self._contexts[loop_id]['client']
            # print(f"DEBUG: Event loop {loop_id} 的 Redis 客户端不存在，开始异步初始化...")
            try:
                # 调用私有初始化方法
                new_client = await self._initialize_for_loop(current_loop)
                # 将新创建的客户端存入上下文
                self._contexts[loop_id]['client'] = new_client
                return new_client
            except Exception as e:
                logger.error(f"Redis 客户端初始化失败，loop_id: {loop_id}。正在清理损坏的上下文以便重试。错误: {e}", exc_info=True)
                # 如果初始化失败，必须移除损坏的上下文，否则后续调用将永远等待一个无法被创建的客户端
                with self._context_lock:
                    self._contexts.pop(loop_id, None)
                # 将异常向上抛出，让调用者知道操作失败了
                raise
    # NEW: 新增并修正 get_redis_lock 方法
    async def get_redis_lock(self, lock_name: str, timeout: int = 30, blocking_timeout: int = 5):
        """
        (异步) 获取一个 Redis 分布式锁。
        如果获取客户端失败，会向上抛出 ConnectionError。
        """
        try:
            # 1. 正确地 await 获取客户端
            redis_client = await self._ensure_client()
            # 2. 使用获取到的客户端实例来创建锁
            return redis_client.lock(lock_name, timeout=timeout, blocking_timeout=blocking_timeout)
        except Exception as e:
            # 捕获 _ensure_client() 抛出的连接异常，或 redis.lock() 本身的异常
            logger.error(f"获取 Redis 锁 '{lock_name}' 时发生异常: {e}", exc_info=True)
            # 向上抛出一个明确的异常，让调用者（如 BaseDAO）知道操作失败，而不是静默返回 None。
            # 这使得 BaseDAO 可以正确地中止数据库操作。
            raise ConnectionError(f"获取 Redis 锁 '{lock_name}' 失败。") from e
    def get_timeout(self, cache_type: str) -> int:
        """获取指定缓存类型的过期时间"""
        return self.DEFAULT_TIMEOUTS.get(cache_type, 300)  # 默认5分钟
    def _serialize(self, data: Any) -> bytes:
        """
        (内部方法) 使用 umsgpack 将 Python 对象序列化为字节。
        - 先调用 _clean_data_for_serialization 清洗数据，再进行打包。
        """
        try:
            # 1. 在序列化之前，主动清洗数据
            cleaned_data = _clean_data_for_serialization(data)
            # 2. 对清洗后的数据进行打包，不再需要 default 回调
            return umsgpack.packb(cleaned_data, use_bin_type=True)
        except (umsgpack.UnsupportedTypeException, TypeError) as e:
            logger.error(f"序列化失败: 存在未处理的不支持类型 - {e}", exc_info=True)
            raise ValueError(f"Serialization failed due to unsupported type: {e}") from e
        except Exception as e:
            logger.error(f"序列化时发生意外错误: {e}", exc_info=True)
            raise ValueError(f"Unexpected serialization error: {e}") from e
    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        if not data:
            # MODIFIED: 添加调试打印，显示当传入空数据时的情况
            print("DEBUG: _deserialize received empty data.")
            return None
        try:
            return umsgpack.unpackb(data, raw=False)
        except Exception as e:
            # MODIFIED: 添加调试打印，显示反序列化失败时的原始数据（截断）和错误信息
            print(f"DEBUG: _deserialize failed for data: {data[:100]}... (truncated). Error: {e}")
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
            redis_client = await self._ensure_client() # 获取当前循环的客户端
            serialized_data = self._serialize(data)
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
                success = await redis_client.set(key, serialized_data, ex=effective_timeout, nx=True)
            else:
                # 普通 set (覆盖)
                success = await redis_client.set(key, serialized_data, ex=effective_timeout)
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
            redis_client = await self._ensure_client()
            serialized_data = await redis_client.get(key)
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
        await self._ensure_client()
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
            redis_client = await self._ensure_client()
            result = await redis_client.delete(key)
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
            redis_client = await self._ensure_client()
            # exists 返回整数 (存在的键数量)，需要转为 bool
            return bool(await redis_client.exists(key))
        except ConnectionError as e:
             logger.error(f"缓存检查失败 (Redis 连接错误): key='{key}', error='{e}'")
             return False # 连接失败视为不存在
        except Exception as e:
            logger.error(f"缓存检查时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return False
    async def ttl(self, key: str) -> int:
        """(异步) 获取缓存键的剩余生存时间 (秒)"""
        try:
            redis_client = await self._ensure_client()
            # ttl 返回 -2 (键不存在), -1 (无过期时间), 或剩余秒数
            return await redis_client.ttl(key)
        except ConnectionError as e:
             logger.error(f"获取 TTL 失败 (Redis 连接错误): key='{key}', error='{e}'")
             return -2 # 连接失败视为不存在
        except Exception as e:
            logger.error(f"获取 TTL 时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return -2
    async def pipeline(self) -> Pipeline:
        """(异步) 获取一个 Redis pipeline 对象"""
        redis_client = await self._ensure_client() # 确保客户端已初始化
        # redis-py 的 pipeline() 方法直接返回 pipeline 对象
        return redis_client.pipeline()
    async def hset(self, key: str, field: str, value: Any, timeout: Optional[int] = None) -> bool:
        """(异步) 设置哈希表中的字段值"""
        try:
            redis_client = await self._ensure_client()
            serialized_value = self._serialize(value) # 序列化值
            # 使用 pipeline 保证 hset 和 expire 原子性（如果需要设置超时）
            async with redis_client.pipeline() as pipe:
                pipe.hset(key, field, serialized_value) # 设置哈希字段
                # 处理超时逻辑
                effective_timeout = timeout
                if effective_timeout is None:
                    # 如果未指定超时，检查哈希键是否已存在且有 TTL
                    # 如果不存在或无 TTL，则设置默认 TTL
                    current_ttl = await redis_client.ttl(key) # 在 pipeline 外检查 TTL
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
            redis_client = await self._ensure_client()
            serialized_value = await redis_client.hget(key, field)
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
            redis_client = await self._ensure_client()
            # hgetall 返回 bytes:bytes 字典 (因为 decode_responses=False)
            raw_dict = await redis_client.hgetall(key)
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
            redis_client = await self._ensure_client()
            # mget 返回 bytes 列表或 None 列表
            serialized_values = await redis_client.mget(keys)
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
            redis_client = await self._ensure_client()
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
            async with redis_client.pipeline() as pipe:
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
            redis_client = await self._ensure_client()
            serialized_result = await redis_client.zrangebyscore(key, min_score, max_score, withscores=withscores)
            if serialized_result is None:
                logger.debug(f"ZRANGEBYSCORE 未找到匹配项: key='{key}', range=[{min_score}, {max_score}]")
                # MODIFIED: 添加调试打印，显示zrangebyscore返回None的情况
                print(f"DEBUG: zrangebyscore returned None for key: {key}")
                return []
            logger.debug(f"ZRANGEBYSCORE 命中: key='{key}', range=[{min_score}, {max_score}], withscores={withscores}")
            deserialized_list = []
            if withscores:
                for member_bytes, score in serialized_result:
                    deserialized_member = self._deserialize(member_bytes)
                    if deserialized_member is not None:
                        deserialized_list.append((deserialized_member, score))
                    else:
                        # MODIFIED: 添加调试打印，显示zrangebyscore中成员反序列化失败的情况
                        print(f"DEBUG: zrangebyscore: Deserialized member is None for key: {key}, member_bytes: {member_bytes[:50]}...")
                        logger.warning(f"ZRANGEBYSCORE 反序列化成员失败: key='{key}', member_bytes={member_bytes}")
            else:
                for member_bytes in serialized_result:
                    deserialized_member = self._deserialize(member_bytes)
                    if deserialized_member is not None:
                        deserialized_list.append(deserialized_member)
                    else:
                         # MODIFIED: 添加调试打印，显示zrangebyscore中成员反序列化失败的情况
                         print(f"DEBUG: zrangebyscore: Deserialized member is None for key: {key}, member_bytes: {member_bytes[:50]}...")
                         logger.warning(f"ZRANGEBYSCORE 反序列化成员失败: key='{key}', member_bytes={member_bytes}")
            # MODIFIED: 添加调试打印，显示最终反序列化成功的项目数量
            print(f"DEBUG: zrangebyscore returning {len(deserialized_list)} deserialized items for key: {key}")
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
            redis_client = await self._ensure_client()
            start = 0
            end = limit - 1
            if desc:
                # zrevrange 返回 bytes 列表或 (bytes, float) 元组列表
                serialized_result = await redis_client.zrevrange(key, start, end, withscores=withscores)
            else:
                # zrange 返回 bytes 列表或 (bytes, float) 元组列表
                serialized_result = await redis_client.zrange(key, start, end, withscores=withscores)
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
            return 0 # 不做任何操作
        try:
            redis_client =await self._ensure_client()
            # zremrangebyrank(key, start, stop) 移除指定排名范围内的成员
            # 排名从 0 开始，负数表示从尾部开始 (-1 是最高分，-2 是第二高分)
            # 要保留最新的 keep_latest 个，需要移除排名在 0 到 -(keep_latest + 1) 之间的成员
            # 例如：保留 100 个，移除排名 0 到 -101 的成员
            stop_rank = -(keep_latest + 1)
            removed_count = await redis_client.zremrangebyrank(key, 0, stop_rank)
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
            redis_client = await self._ensure_client()
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
            async with redis_client.pipeline() as pipe:
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
    async def scan_keys(self, pattern: str):
        """
        (异步) 扫描并返回所有匹配 pattern 的 key 列表
        :param pattern: Redis key 匹配模式，如 'strategy:stock:*:trend_following'
        :return: 匹配到的 key 列表（str 类型）
        """
        try:
            redis_client = await self._ensure_client()  # 确保 Redis 客户端已初始化
            keys = []
            cursor = 0  # 初始游标
            while True:
                # aioredis 的 scan 返回 (cursor, [keys])
                cursor, batch = await redis_client.scan(cursor=cursor, match=pattern, count=100)
                # 兼容 bytes 和 str
                for k in batch:
                    if isinstance(k, bytes):
                        keys.append(k.decode())
                    else:
                        keys.append(k)
                if cursor == 0:
                    break
            print(f"scan_keys: pattern={pattern}, 共找到{len(keys)}个key")  # 调试信息
            return keys
        except ConnectionError as e:
            logger.error(f"scan_keys 失败 (Redis 连接错误): pattern='{pattern}', error='{e}'")
            return []
        except Exception as e:
            logger.error(f"scan_keys 时发生未知 Redis 错误: pattern='{pattern}', error='{e}'", exc_info=True)
            return []
    async def sadd(self, key: str, *values: Any) -> Optional[int]:
        """
        (异步) 向集合添加一个或多个成员。
        Args:
            key (str): 集合的键。
            *values: 一个或多个要添加到集合的成员。
        Returns:
            Optional[int]: 成功添加到集合中的新成员数量，如果发生错误则返回None。
        """
        if not values:
            return 0
        try:
            redis_client = await self._ensure_client()
            # 序列化所有要添加的值
            serialized_values = [self._serialize(v) for v in values]
            # SADD 命令返回成功添加的新成员数量
            added_count = await redis_client.sadd(key, *serialized_values)
            logger.debug(f"SADD 操作成功: key='{key}', 添加了 {len(values)} 个成员, 新增 {added_count} 个。")
            return added_count
        except ValueError as e: # 捕获序列化失败
            logger.error(f"SADD 操作失败 (序列化错误): key='{key}', error='{e}'")
            return None
        except ConnectionError as e:
            logger.error(f"SADD 操作失败 (Redis 连接错误): key='{key}', error='{e}'")
            return None
        except Exception as e:
            logger.error(f"SADD 操作时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return None
    async def smembers(self, key: str) -> Optional[List[Any]]:
        """
        (异步) 获取集合中的所有成员。
        Args:
            key (str): 集合的键。
        Returns:
            Optional[List[Any]]: 包含所有成员的列表，如果键不存在或发生错误则返回None。
        """
        try:
            redis_client = await self._ensure_client()
            # SMEMBERS 返回一个包含所有成员（bytes）的集合
            serialized_members = await redis_client.smembers(key)
            if not serialized_members:
                logger.debug(f"SMEMBERS 未找到成员或集合为空: key='{key}'")
                return [] # 返回空列表表示集合为空
            logger.debug(f"SMEMBERS 命中: key='{key}', 找到 {len(serialized_members)} 个成员。")
            # 反序列化每个成员
            deserialized_list = [self._deserialize(member) for member in serialized_members]
            return deserialized_list
        except ConnectionError as e:
            logger.error(f"SMEMBERS 操作失败 (Redis 连接错误): key='{key}', error='{e}'")
            return None
        except Exception as e:
            logger.error(f"SMEMBERS 操作时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return None
    async def expire(self, key: str, timeout: int) -> bool:
        """
        (异步) 为指定的键设置过期时间。
        Args:
            key (str): 目标键。
            timeout (int): 过期时间，单位为秒。
        Returns:
            bool: 如果成功设置了过期时间则返回True，否则返回False。
        """
        if timeout <= 0:
            logger.warning(f"EXPIRE: 无效的超时时间 {timeout}，操作取消。")
            return False
        try:
            redis_client = await self._ensure_client()
            # EXPIRE 命令返回 1 (成功) 或 0 (键不存在或未设置成功)
            success = await redis_client.expire(key, timeout)
            if success:
                logger.debug(f"EXPIRE 操作成功: key='{key}', timeout={timeout}s")
            else:
                logger.warning(f"EXPIRE 操作失败: key='{key}' 可能不存在。")
            return bool(success)
        except ConnectionError as e:
            logger.error(f"EXPIRE 操作失败 (Redis 连接错误): key='{key}', error='{e}'")
            return False
        except Exception as e:
            logger.error(f"EXPIRE 操作时发生未知 Redis 错误: key='{key}', error='{e}'", exc_info=True)
            return False
    async def close(self):
        """
        (异步) 优雅地关闭所有已创建的 Redis 客户端连接。
        这是一个重要的清理方法，应在应用或一个完整任务周期结束时调用。
        """
        # print("DEBUG: CacheManager 正在关闭所有 Redis 连接...")
        # 使用同步锁来安全地访问和修改 _contexts 字典
        with self._context_lock:
            if not self._contexts:
                # print("DEBUG: 没有活动的 Redis 连接需要关闭。")
                return
            # 收集所有需要关闭的客户端的 close() 协程
            close_tasks = []
            for loop_id, context in self._contexts.items():
                client = context.get('client')
                if client:
                    # print(f"  -> 正在安排关闭 Event Loop {loop_id} 的连接。")
                    close_tasks.append(client.close())
            # 在持有锁的期间，立即清空上下文，防止新的请求进来
            self._contexts.clear()
        # 在锁之外，并发地执行所有关闭任务
        if close_tasks:
            try:
                # 使用 asyncio.gather 并发关闭所有连接，提高效率
                await asyncio.gather(*close_tasks, return_exceptions=True)
                # print(f"DEBUG: CacheManager 已成功关闭 {len(close_tasks)} 个 Redis 连接。")
            except Exception as e:
                logger.error(f"关闭 Redis 连接过程中发生意外错误: {e}", exc_info=True)
        else:
            print("DEBUG: 没有找到活动的 Redis 客户端实例来关闭。")
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

