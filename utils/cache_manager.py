import json
import zlib
import logging
import inspect
import msgpack
from typing import Any, Dict, List, Optional, Type, Union, TypeVar, Mapping # 引入 Mapping
from datetime import datetime
from utils import cache_constants as cc # 导入常量
# from django.core.cache import cache
from django_redis import get_redis_connection
from redis.client import Pipeline, Redis # 引入 Redis 以便类型提示



logger = logging.getLogger("dao")

T = TypeVar('T')

# 添加自定义JSON编码器
class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理不可直接序列化的对象"""
    def default(self, obj):
        # 处理datetime对象
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        if hasattr(obj, '__code__'):
            return obj.__code__()

        # 处理自定义对象（如IndexInfo）- 尝试使用__str__方法
        elif hasattr(obj, '__str__'):
            return str(obj)
        
        # 尝试将对象转换为字典（优先使用to_dict方法，然后是__dict__）
        elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
            
        # 默认行为
        return super().default(obj)

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
        'st': 86400 * 7,          # 静态数据缓存1天
        'ts': 86400 * 7,      # 时间序列缓存7天 (根据需要调整)
        'calc': 300,          # 计算结果缓存5分钟
        'user': 1800,         # 用户数据缓存30分钟
    }
    
    # 是否压缩大数据集（超过阈值的数据会被压缩）
    COMPRESSION_THRESHOLD = 100 * 1024  # 10KB
    
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
        try:
            # 步骤3: 使用msgpack序列化
            msgpack_data = msgpack.packb(data, use_bin_type=True)
            return msgpack_data
        except Exception as e:
            logger.error(f"序列化失败: {e}", exc_info=True)
            # 应急方案：返回简单字符串
            return json.dumps(str(data)).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据，自动处理压缩"""
        if not data:
            return None
        try:
            # 步骤1: 使用msgpack反序列化
            msgpack_data = msgpack.unpackb(data, raw=False)
            return msgpack_data
        except Exception as e:
            logger.error(f"反序列化失败: {e}", exc_info=True)
            return None

    def _restore_objects(self, data: Any) -> Any:
        """还原特殊对象（根据上下文识别对象类型）"""
        if not isinstance(data, dict):
            return data
        # 创建一个新字典用于存储处理后的数据
        result = {}
        # 处理索引/股票代码
        if "index" in data and isinstance(data["index"], str):
            code = data["index"]
            try:
                from stock_models.index import IndexInfo  # 根据实际路径调整
                # 尝试从数据库或缓存获取IndexInfo对象
                try:
                    # 优先尝试从数据库获取完整对象
                    index_info = IndexInfo.objects.get(code=code)
                except Exception:
                    # 如果获取失败，创建一个基本对象（可能缺少名称等信息）
                    index_info = IndexInfo(code=code)
                    logger.warning(f"无法从数据库获取IndexInfo(code={code})，创建了基本对象")
                result["index"] = index_info
            except ImportError as e:
                logger.error(f"导入模型类失败: {e}")
                # 保留原始字符串
                result["index"] = data["index"]
            except Exception as e:
                logger.error(f"还原指数/股票对象失败: {e}")
                result["index"] = data["index"]
        
        if "stock" in data and isinstance(data["stock"], str):
            code = data["stock"]
            # 股票代码处理
            from stock_models.stock_basic import StockInfo  # 根据实际路径调整
            try:
                stock_info = StockInfo.objects.get(stock_code=code)
            except Exception:
                # 如果获取失败，创建一个基本对象（可能缺少名称等信息）
                stock_info = StockInfo(stock_code=code)
                logger.warning(f"无法从数据库获取StockInfo(code={code})，创建了基本对象")
            
            result["stock"] = stock_info
        # 处理时间字段
        if "trade_time" in data and isinstance(data["trade_time"], str):
            try:
                result["trade_time"] = datetime.fromisoformat(data["trade_time"])
            except Exception as e:
                logger.error(f"解析时间字段失败: {e}")
                result["trade_time"] = data["trade_time"]
        
        # 复制所有其他字段
        for key, value in data.items():
            if key not in result:
                result[key] = value
        
        return result

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
        serialized_data = None
        try:
            if timeout is None:
                # 根据键前缀自动选择过期时间
                prefix = key.split(':')[0]
                timeout = self.get_timeout(prefix)
            # logger.warning(f"缓存保存数据111: {key}, 数据长度: {len(json.dumps(data))}")
            serialized_data = self._serialize(data)
            # logger.warning(f"缓存保存数据222: 数据长度: {len(serialized_data)}, key: {key}")
            if nx:
                return self.redis_client.set(key, serialized_data, ex=timeout, nx=True)
            else:
                return self.redis_client.set(key, serialized_data, ex=timeout)
        except Exception as e:
            logger.error(f"缓存保存失败: {key}, 错误: {str(e)}， 数据: {serialized_data}, type: {type(serialized_data)}")
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
                return_data = self._deserialize(data)
                # logger.warning(f"return_data获取数据: {key}, {return_data}")
                return return_data
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
    
    # --- Sorted Set (有序集合) 操作 ---

    # 修改 zadd 方法签名和内部逻辑
    def zadd(self, key: str, mapping: Mapping[bytes, float], timeout: Optional[int] = None) -> Optional[int]:
        """
        向有序集合添加一个或多个成员(bytes)，或者更新已存在成员的分数。
        Args:
            key: 有序集合的键。
            mapping: 一个字典，键是成员(bytes)，值是分数(float)。
            timeout: 过期时间(秒)。
        Returns:
            Optional[int]: 成功添加的新成员数量。错误则返回 None。
        """
        if not mapping:
            logger.warning(f"ZADD 操作跳过: mapping 为空, key: {key}")
            return 0
        # 可选：验证 mapping 的键是否为 bytes
        # if not all(isinstance(m, bytes) for m in mapping.keys()):
        #     logger.error(f"ZADD 内部错误: mapping 的键必须是 bytes 类型, key: {key}")
        #     return None # 或者抛出异常

        # logger.info(f"ZADD 操作 (接收 bytes member): key={key}, mapping size={len(mapping)}")
        try:
            # 确定超时时间
            if timeout is None:
                prefix = key.split(':')[0]
                timeout = self.get_timeout(prefix)

            # *** 直接将接收到的 {bytes: float} mapping 传递给 redis 客户端 ***
            # 不再需要内部序列化循环
            # serialized_mapping = {self._serialize(member): score for member, score in mapping.items()} # <--- 删除或注释掉这行

            pipe = self.redis_client.pipeline()
            # 直接使用传入的 mapping
            pipe.zadd(key, mapping)
            pipe.expire(key, timeout)
            results = pipe.execute()

            if results and len(results) > 0 and isinstance(results[0], int):
                return results[0]
            else:
                logger.error(f"ZADD pipeline 执行结果异常: key={key}, results={results}")
                return None

        except Exception as e:
            logger.error(f"ZADD 操作失败: key={key}, 错误: {str(e)}", exc_info=True)
            return None
        
    def zrangebyscore(self, key: str, min_score: Union[float, str], max_score: Union[float, str]) -> Optional[List[Any]]:
        """
        通过分数范围获取有序集合的成员列表。

        Args:
            key: 有序集合的键。
            min_score: 最小分数 (包含)。可以是 float 或 '-inf'。
            max_score: 最大分数 (包含)。可以是 float 或 '+inf'。

        Returns:
            Optional[List[Any]]: 按分数升序排列的成员列表 (已反序列化)。
                                 如果键不存在或范围内无成员，返回空列表 []。
                                 如果发生错误，返回 None。
        """
        try:
            # 直接调用 redis-py 的 zrangebyscore
            # 它返回的是成员列表 (bytes)
            serialized_members: List[bytes] = self.redis_client.zrangebyscore(key, min_score, max_score)

            if serialized_members is None: # 理论上 redis-py 不会返回 None，但以防万一
                 logger.warning(f"ZRANGEBYSCORE 返回 None: key={key}, range=[{min_score}, {max_score}]")
                 return [] # 返回空列表表示未找到或键不存在

            # 反序列化每个成员
            deserialized_list = []
            for sm in serialized_members:
                deserialized_member = self._deserialize(sm)
                if deserialized_member is not None: # 跳过无法反序列化的成员
                    deserialized_list.append(deserialized_member)
                else:
                    logger.warning(f"ZRANGEBYSCORE: 跳过无法反序列化的成员, key={key}")

            logger.debug(f"ZRANGEBYSCORE 成功: key={key}, range=[{min_score}, {max_score}], 返回 {len(deserialized_list)} 个成员")
            return deserialized_list

        except Exception as e:
            logger.error(f"ZRANGEBYSCORE 操作失败: key={key}, range=[{min_score}, {max_score}], 错误: {str(e)}", exc_info=True)
            return None # 发生错误时返回 None

    def zrange_by_limit(self, key: str, limit: int) -> Optional[List[Any]]:
        """
        获取有序集合中最新的limit条数据。
        在Redis有序集合中，通常较新的数据会有较高的分数，因此使用zrevrange获取分数最高的数据。
        Args:
            key: 有序集合的键。
            limit: 要返回的成员数量限制。
        Returns:
            Optional[List[Any]]: 按分数降序排列的成员列表 (已反序列化)。
                                如果键不存在，返回空列表 []。
                                如果发生错误，返回 None。
        """
        try:
            # 使用 zrevrange 获取分数最高的 limit 条数据
            # 这会返回分数从高到低排序的成员列表
            serialized_members: List[bytes] = self.redis_client.zrevrange(key, 0, limit - 1)

            if serialized_members is None:  # 理论上 redis-py 不会返回 None，但以防万一
                logger.warning(f"ZREVRANGE 返回 None: key={key}, limit={limit}")
                return []  # 返回空列表表示未找到或键不存在

            # 反序列化每个成员
            deserialized_list = []
            for sm in serialized_members:
                deserialized_member = self._deserialize(sm)
                if deserialized_member is not None:  # 跳过无法反序列化的成员
                    deserialized_list.append(deserialized_member)
                else:
                    logger.warning(f"ZREVRANGE: 跳过无法反序列化的成员, key={key}")

            logger.debug(f"ZREVRANGE 成功: key={key}, limit={limit}, 返回 {len(deserialized_list)} 个成员")
            return deserialized_list

        except Exception as e:
            logger.error(f"ZREVRANGE 操作失败: key={key}, limit={limit}, 错误: {str(e)}", exc_info=True)
            return None  # 发生错误时返回 None

    def ztrim_by_rank(self, key: str, keep_latest: int) -> Optional[int]:
        """
        修剪有序集合，只保留最新的 N 个成员（按分数排序）。
        移除分数最低的成员，直到集合大小等于 keep_latest。

        Args:
            key: 有序集合的键。
            keep_latest: 希望保留的最新成员数量。

        Returns:
            Optional[int]: 被移除的成员数量。如果键不存在、无需移除或发生错误，返回 0 或 None。
        """
        if keep_latest <= 0:
            logger.warning(f"ZTRIMBYRANK 跳过: keep_latest ({keep_latest}) 必须大于 0, key: {key}")
            # 可以选择删除整个 key 或返回 0
            # self.delete(key)
            return 0

        try:
            # 获取当前集合大小
            current_size = self.redis_client.zcard(key)

            if current_size is None or current_size <= keep_latest:
                # 键不存在或大小未超限，无需修剪
                # logger.debug(f"ZTRIMBYRANK 无需修剪: key={key}, current_size={current_size}, keep_latest={keep_latest}")
                return 0

            # 计算需要移除的数量
            remove_count = current_size - keep_latest
            # 计算要移除的最高排名 (从 0 开始)
            # 例如，保留 200，当前 210，移除 10 个，移除排名 0 到 9
            end_rank = remove_count - 1

            # logger.info(f"ZTRIMBYRANK: 准备移除 key={key} 中排名 0 到 {end_rank} 的 {remove_count} 个成员，保留最新 {keep_latest} 个。")

            # 执行移除操作
            removed_count = self.redis_client.zremrangebyrank(key, 0, end_rank)

            if removed_count is not None:
                # logger.info(f"ZTRIMBYRANK 成功: key={key}, 移除了 {removed_count} 个成员。")
                return removed_count
            else:
                # 理论上 zremrangebyrank 失败的可能性较低，除非连接问题
                logger.error(f"ZTRIMBYRANK 命令执行失败 (返回 None): key={key}")
                return None

        except Exception as e:
            logger.error(f"ZTRIMBYRANK 操作失败: key={key}, 错误: {str(e)}", exc_info=True)
            return None

    def zadd_and_trim(self, key: str, mapping: Mapping[Any, float], limit: int, timeout: Optional[int] = None) -> Optional[int]:
        """
        向有序集合添加成员，并确保集合大小不超过指定限制（保留最新的成员）。
        这是一个组合操作，尽量保证原子性。

        Args:
            key: 有序集合的键。
            mapping: {member: score} 的映射，成员会被序列化。
            limit: 集合的最大成员数限制。
            timeout: 整个集合的过期时间(秒)。

        Returns:
            Optional[int]: 成功添加的新成员数量。如果发生错误则返回 None。
                           注意：此返回值不反映修剪操作的结果。
        """
        if not mapping or not isinstance(mapping, dict):
             logger.warning(f"ZADD_AND_TRIM 操作跳过: mapping 为空或不是字典, key: {key}")
             return 0

        try:
            serialized_mapping = {self._serialize(member): score for member, score in mapping.items()}

            if timeout is None:
                prefix = key.split(':')[0]
                timeout = self.get_timeout(prefix)

            # 使用 Lua 脚本实现原子性 (推荐，如果 Redis 版本支持且熟悉 Lua)
            # 或者使用 Pipeline 尽量连续执行
            pipe = self.redis_client.pipeline()
            pipe.zadd(key, serialized_mapping)
            # 计算需要移除的数量 (负数索引表示从尾部开始)
            # 保留 limit 个，则移除范围是 0 到 -(limit + 1)
            # 例如 limit=200, 移除 0 到 -201
            pipe.zremrangebyrank(key, 0, -(limit + 1))
            pipe.expire(key, timeout)
            results = pipe.execute()

            # results[0] = zadd result (added count)
            # results[1] = zremrangebyrank result (removed count)
            # results[2] = expire result (bool)
            if results and len(results) >= 2 and isinstance(results[0], int):
                 added_count = results[0]
                 removed_count = results[1] if isinstance(results[1], int) else 'N/A'
                #  logger.info(f"ZADD_AND_TRIM 成功: key={key}, 新增 {added_count}, 尝试移除旧数据 (移除数量: {removed_count}), limit={limit}, timeout={timeout}s")
                 return added_count
            else:
                 logger.error(f"ZADD_AND_TRIM pipeline 执行结果异常: key={key}, results={results}")
                 return None

        except Exception as e:
            logger.error(f"ZADD_AND_TRIM 操作失败: key={key}, 错误: {str(e)}", exc_info=True)
            return None

    # =============== Dao实际使用方法 ===============

    # 修剪指定键的 Sorted Set，只保留最新的 N 条记录。
    async def trim_cache_zset(self, cache_key: str, limit: int) -> Optional[int]:
        """
        通用的公共方法：修剪指定键的 Sorted Set，只保留最新的 N 条记录。

        Args:
            cache_key: 要修剪的 Redis Sorted Set 的键。
            limit: 希望保留的最新记录数量。

        Returns:
            Optional[int]: 被成功移除的旧记录数量 (0 表示无需移除或键不存在)。
                           如果发生错误，返回 None。
        """
        if limit <= 0:
            logger.warning(f"缓存修剪跳过: limit ({limit}) 必须大于 0 for key={cache_key}")
            return 0
        # logger.info(f"开始修剪缓存: key={cache_key}, 保留最新 {limit} 条。")
        try:
            # 直接调用 CacheManager 的修剪方法
            removed_count = self.ztrim_by_rank(key=cache_key, keep_latest=limit)
            if removed_count is None:
                logger.error(f"缓存修剪失败 (ztrim_by_rank 返回 None): key={cache_key}")
                return None
            elif removed_count > 0:
                # logger.info(f"缓存修剪成功: key={cache_key}, 移除了 {removed_count} 条旧记录。")
                return removed_count
            else:
                # logger.info(f"缓存无需修剪或键不存在: key={cache_key}")
                return 0
        except Exception as e:
            logger.error(f"缓存修剪时发生异常: key={cache_key}, error={e}", exc_info=True)
            return None
# 使用示例

cache_manager = CacheManager()

# 1. 保存股票实时数据
def save_stock_realtime(stock_code: str, data: dict):
    key = cache_manager.generate_key('rt', 'stock', stock_code, 'quote')
    cache_manager.set(key, data)

# 2. 获取股票实时数据并转换为模型实例
def get_stock_realtime(stock_code: str) -> Optional['StockRealtimeData']:
    from stock_models.stock_realtime import StockRealtimeData
    key = cache_manager.generate_key('rt', 'stock', stock_code, 'quote')
    return cache_manager.get_model(key, StockRealtimeData)

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


