import logging
import pickle
from typing import Any, Dict, List
from asgiref.sync import sync_to_async
import pandas as pd
import umsgpack
from datetime import datetime, date # 同时导入 date 类，如果需要处理的话
from decimal import Decimal # 导入 Decimal
from utils import cache_constants as cc
import json

from utils.cache_manager import CacheManager
from utils.cash_key import IndexCashKey, StockCashKey, UserCashKey
from utils.data_format_process import IndexDataFormatProcess

logger = logging.getLogger("dao")

# 定义独立的辅助函数来处理 msgpack 不支持的类型
def _msgpack_default_packer(obj):
    """
    为 msgpack 提供 default hook，处理 datetime 和 Decimal。
    """
    if isinstance(obj, datetime):
        # 返回 ISO 格式字符串 (推荐) 或 timestamp
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        # 将 Decimal 转换为字符串以保持精度
        return str(obj)
    # 对于其他无法序列化的类型，可以抛出错误或返回特定值
    logger.warning(f"Msgpack无法序列化类型 {type(obj)}: {obj}")
    # 或者 return str(obj) 作为最后的尝试
    raise TypeError(f"Object of type {obj.__class__.__name__} is not MSGPACK serializable")

# 新增辅助函数：递归转换数据结构中的 Decimal 对象
def convert_decimals(obj):
    """
    递归遍历对象，转换所有 Decimal 类型为字符串。
    支持字典、列表和其他嵌套结构。
    """
    if isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    elif isinstance(obj, Decimal):
        return str(obj)  # 转换 Decimal 为字符串
    else:
        return obj  # 其他类型保持不变


class CacheSet():
    def __init__(self):
        from utils.cache_manager import CacheManager
        from utils.cash_key import IndexCashKey, StockCashKey, StrategyCashKey, UserCashKey
        from utils.data_format_process import IndexDataFormatProcess
        # 注意：CacheManager 是异步的，这里无法直接初始化。建议在异步上下文中使用。
        self.cache_key_index = IndexCashKey()
        self.cache_key_stock = StockCashKey()
        self.cache_key_strategy = StrategyCashKey()
        self.data_format_process = IndexDataFormatProcess()
        self.cache_key_user = UserCashKey()

    async def get_cache_manager(self):
        from utils.cache_manager import CacheManager
        cm = CacheManager()
        await cm.initialize()
        return cm

    async def _index_latest_data(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any], cache_key: str) -> bool:
        if not data_to_cache:
            logger.warning(f"试图缓存指数[{index_code}] 时间级别[{time_level}] 的空时间序列数据，操作跳过。")
            return False
        cache_manager = await self.get_cache_manager()
        try:
            cache_timeout = cache_manager.get_timeout(cc.TYPE_REALTIME)
            success = await cache_manager.set(  # 添加 await
                key=cache_key,
                data=data_to_cache,
                timeout=cache_timeout
            )
            if success:
                return True
            else:
                logger.warning(f"缓存指数[{index_code}] 时间级别[{time_level}] 最新时间序列数据失败, key: {cache_key}")
                return False
        except Exception as e:
            logger.error(f"缓存指数[{index_code}] 时间级别[{time_level}] 最新时间序列数据时发生异常: {str(e)}", exc_info=True)
            return False
        
    async def _stock_latest_data(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any], cache_key: str) -> bool:
        if not data_to_cache:
            logger.warning(f"试图缓存股票[{stock_code}] 时间级别[{time_level}] 的空时间序列数据，操作跳过。")
            return False
        try:
            cache_manager = await self.get_cache_manager()
            cache_timeout = cache_manager.get_timeout(cc.TYPE_REALTIME) # 或者 cc.TYPE_TIMESERIES
            # 3. 调用 CacheManager 设置缓存
            success = await cache_manager.set(
                key=cache_key,
                data=data_to_cache,
                timeout=cache_timeout
            )
            # print(f"_stock_latest_data - {stock_code}: {success}")
            if success:
                # print(f"股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据缓存成功, key: {cache_key}")
                return True
            else:
                logger.warning(f"缓存股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据失败 (CacheManager.set 返回 False), key: {cache_key}")
                return False
        except Exception as e:
            logger.error(f"StockIndicatorsDAO._stock_latest_data缓存股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据时发生异常: {str(e)}, key: (生成失败或未知)", exc_info=True)
            return False

    async def _realtime_data(self, stock_code: str, data_to_cache: Dict[str, Any], cache_key: str) -> bool:
        if not data_to_cache:
            logger.warning(f"试图缓存股票[{stock_code}] 的空实时数据，操作跳过。")
            return False
        try:    
            cache_manager = await self.get_cache_manager()
            cache_timeout = cache_manager.get_timeout(cc.TYPE_REALTIME) # 或者 cc.TYPE_TIMESERIES
            success = await cache_manager.set(
                key=cache_key,
                data=data_to_cache,
                timeout=cache_timeout
            )
            if success:
                return True
            else:
                logger.warning(f"缓存股票[{stock_code}] 实时数据失败 (CacheManager.set 返回 False), key: {cache_key}")
                return False
        except Exception as e:
            logger.error(f"StockIndicatorsDAO._stock_latest_data缓存股票[{stock_code}] 实时数据时发生异常: {str(e)}, key: (生成失败或未知)", exc_info=True)
            return False

    async def _stock_strategy_data(self, stock_code: str, data_to_cache: Dict[str, Any], cache_key: str) -> bool: # 修改: 添加 timestamp 参数
        if not data_to_cache:
            logger.warning(f"试图缓存股票[{stock_code}] 的空时间序列数据，操作跳过。")
            return False
        try:
            cache_manager = await self.get_cache_manager()
            cache_timeout = cache_manager.get_timeout(cc.TYPE_STRATEGY)
            success = await cache_manager.set(
                key=cache_key,
                data=data_to_cache,
                timeout=cache_timeout
            )
            if success:
                return True
            else:
                logger.warning(f"缓存股票[{stock_code}] 策略结果数据失败 (CacheManager.set 返回 False), key: {cache_key}")
                return False
        except Exception as e:
            logger.error(f"StockIndicatorsDAO._stock_strategy_data缓存股票[{stock_code}] 策略数据时发生异常: {str(e)}, key: (生成失败或未知)", exc_info=True)
            return False

    # --- 修正后的写入缓存方法 (使用 ZADD) ---
    async def _history_data(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any], cache_key: str) -> bool:
        from dao_manager.base_dao import BaseDAO
        base_dao = BaseDAO()
        if not data_to_cache:
            logger.warning(f"试图缓存指数[{stock_code}] 时间级别[{time_level}] 的空时间序列数据，操作跳过。")
            return False
        # 1. 提取时间并转换为时间戳 (分数)
        trade_time_str = data_to_cache.get('trade_time')
        if not trade_time_str:
            logger.error(f"缓存失败: 数据点缺少 'trade_time' 字段。数据: {data_to_cache}")
            return False
        try:
            cache_manager = await self.get_cache_manager()
            trade_datetime = base_dao._parse_datetime(trade_time_str)
            score = trade_datetime.timestamp()
            data_to_serialize = data_to_cache.copy()
            
            # 新增：递归转换数据结构中的 Decimal 对象
            data_to_serialize = convert_decimals(data_to_serialize)
            
            if 'stock' in data_to_serialize:
                stock_obj = data_to_serialize.pop('stock', None)
                if stock_obj and hasattr(stock_obj, 'stock_code'):
                    data_to_serialize['stock_code'] = stock_obj.stock_code  # 注意：这里可能需要额外检查 stock_obj 中的 Decimal
            
            member_bytes = umsgpack.packb(data_to_serialize, use_bin_type=True, default=_msgpack_default_packer)
            mapping_to_send = {member_bytes: score}
            cache_timeout = cache_manager.get_timeout(cc.TYPE_TIMESERIES)
            success = await cache_manager.zadd(cache_key, mapping_to_send, cache_timeout)
            if success is not None:
                return True
            else:
                logger.warning(f"添加到缓存 (ZSET) 失败, key: {cache_key}")
                return False
        except Exception as e:
            logger.error(f"添加到缓存 (ZSET) 时发生异常: {str(e)}", exc_info=True)
            return False

    async def _format_conversion(self, data_to_cache: Dict[str, Any]) -> Dict[str, Any]:
        from stock_models.index import IndexInfo
        from stock_models.stock_basic import StockInfo
        for key, value in data_to_cache.items():
            if isinstance(value, datetime):
                data_to_cache[key] = value.isoformat()
            elif isinstance(value, StockInfo):
                try:
                    data_to_cache[key] = value.__code__()
                except AttributeError:
                    logger.error(f"StockInfo模型 for key '{key}' 没有找到 '__code__' 的方法.")
                    return None
            elif isinstance(value, IndexInfo):
                data_to_cache[key] = value.__code__()
        return data_to_cache

class UserCacheSet(CacheSet):
    def __init__(self):
        self.cache_key_user = UserCashKey()

    async def user_favorites(self, user_id: int, data_to_cache: List[Dict]) -> bool:
        """
        将用户自选股列表缓存到 Redis，使用 Hash 类型。
        """
        cache_key = self.cache_key_user.user_favorites(int(user_id))  # 假设返回如 "user:favorites:123"
        try:
            cache_manager = await self.get_cache_manager()
            for index, item in enumerate(data_to_cache):
                field = str(item.get('id', index))  # 使用 'id' 作为 field，如果没有则用索引
                success = await cache_manager.hset(cache_key, field, item)  # 异步调用 hset
                if not success:
                    logger.warning(f"缓存用户 {user_id} 自选股失败: field {field}")
                    return False  # 如果任何一个 field 失败，整个操作失败
            logger.debug(f"成功缓存用户 {user_id} 的自选股列表到 Hash: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"缓存用户 {user_id} 自选股列表失败: {str(e)}", exc_info=True)
            return False
        
    async def all_favorites(self, fav_data: Dict) -> bool:
        """
        将所有自选股列表缓存到 Redis，使用 Hash 类型。
        """
        
        user_id = fav_data.user_id
        cache_key = self.cache_key_user.user_favorites(user_id)
        return await self.user_favorites(user_id, fav_data)

class IndexCacheSet(CacheSet):
    def __init__(self):
        self.cache_key_index = IndexCashKey()
        self.data_format_process = IndexDataFormatProcess()
    
    async def index_info(self, index_code: str, data_to_cache: Dict) -> bool:
        """
        将指数基本信息缓存到 Redis，使用 Hash 类型。
        """
        cache_key = self.cache_key_index.index_data(index_code)  # 假设返回如 "index:info:000001"
        try:
            cache_manager = await self.get_cache_manager()
            cache_timeout = cache_manager.get_timeout(cc.TYPE_STATIC)
            return await cache_manager.set(key=cache_key, data=data_to_cache, timeout=cache_timeout)
        except Exception as e:
            logger.error(f"缓存指数 {index_code} 基本信息失败: {str(e)}", exc_info=True)

    async def all_indexes(self, indexes: List[Dict]) -> bool:
        """
        将提供的指数数据列表（简单字典格式）设置到缓存中。
        此方法用于手动将准备好的、符合缓存结构的数据放入缓存。
        数据格式应与 fetch_and_save_indexes 写入缓存的格式一致。
        Args:
            indexes_data: 包含指数信息的字典列表，格式应为
                          [{'code': 'xxx', 'name': 'yyy', 'exchange': 'zzz'}, ...]
                          注意：不应包含 id, created_at 等数据库特有字段。
        Returns:
            bool: 操作是否成功。
        """
        cache_manager = await self.get_cache_manager()
        # 1. 输入验证 (可选但推荐)
        if not isinstance(indexes, list):
            logger.error("set_indexes_to_cache 失败: 输入数据不是列表")
            return False
        # 可以添加更详细的验证，例如检查列表中的元素是否为字典，是否包含必要的键等
        data_dicts = []
        for item in indexes:
            data_dict = self.data_format_process.set_index_info_data(item)
            data_dicts.append(data_dict)
        logger.debug(f"从数据库获取股票指数列表，共{len(indexes)}条")
        # 2. 生成缓存键 (与 get_all_indexes/fetch_and_save_indexes 保持一致)
        cache_key = self.cache_key_index.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_INDEX,
            entity_id=cc.ID_ALL
        )
        # 3. 获取缓存超时时间
        cache_timeout = cache_manager.get_timeout(cc.TYPE_STATIC)
        logger.info(f"准备将 {len(data_dicts)} 条指数数据设置到缓存, key: {cache_key}, timeout: {cache_timeout}s")
        # 4. 调用 CacheManager 设置缓存
        try:
            # 直接使用传入的 indexes_data，因为它已经是期望的格式
            success = await cache_manager.set(
                key=cache_key,
                data=data_dicts,
                timeout=cache_timeout
            )
            if success:
                logger.info(f"指数数据成功设置到缓存, key: {cache_key}")
                return True
            else:
                # cache_manager.set 内部通常会记录错误，这里记录一个警告
                logger.warning(f"设置指数数据到缓存失败 (CacheManager.set 返回 False), key: {cache_key}")
                return False
        except Exception as e:
            logger.error(f"设置指数数据到缓存时发生异常: {str(e)}, key: {cache_key}", exc_info=True)
            return False

    async def realtime_data(self, index_code: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将处理后的指数实时数据字典缓存到 Redis。
        Args:
            index_code: 指数代码，用于生成缓存键。
            data_to_cache: 经过处理的、可JSON序列化的实时数据字典。
                           这个字典的结构应该适合直接用于前端展示或后续处理，
                           并且与 get_realtime_from_cache 方法期望的格式一致。
        Returns:
            bool: 缓存操作是否成功。
        """
        if not data_to_cache:
            logger.warning(f"试图缓存指数[{index_code}]的空实时数据，操作跳过。")
            return False
        cache_manager = await self.get_cache_manager()
        try:
            # 1. 生成缓存键
            cache_key = self.cache_key_index.realtime_data(index_code)
            # 2. 获取缓存超时时间 (实时数据通常较短)
            cache_timeout = cache_manager.get_timeout(cc.TYPE_REALTIME)
            # 3. 调用 CacheManager 设置缓存
            # data_to_cache 应该是可以直接序列化的字典
            success = await cache_manager.set(
                key=cache_key,
                data=data_to_cache,
                timeout=cache_timeout
            )
            if success:
                # logger.info(f"指数[{index_code}]实时数据缓存成功, key: {cache_key}")
                return True
            else:
                logger.warning(f"缓存指数[{index_code}]实时数据失败 (CacheManager.set 返回 False), key: {cache_key}")
                return False
        except Exception as e:
            logger.error(f"缓存指数[{index_code}]实时数据时发生异常: {str(e)}, key: {cache_key}", exc_info=True)
            return False

    async def latest_time_series(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将处理后的单个最新指数时间序列数据点缓存到 Redis。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 经过处理的、可JSON序列化的最新时间序列数据字典。
                           格式应与 get_latest_time_series_from_cache 期望的格式一致。
        Returns:
            bool: 缓存操作是否成功。
        """
        # 使用 'latest' 作为 subtype 或 id 来标识这是最新的数据点
        # ***核心修改：转换 StockInfo 和 datetime 对象***
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_time_series.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.latest_time_series(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, data_to_cache, cache_key)

    async def history_time_series(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将处理后的单个时间序列数据点缓存到 Redis。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 经过处理的、可JSON序列化的时间序列数据字典。
        """
        # ***核心修改：转换 StockInfo 和 datetime 对象***
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"history_time_series.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.history_time_series(index_code, time_level)
        return await self._history_data(index_code, time_level, data_to_cache, cache_key)

class StockInfoCacheSet(CacheSet):
    def __init__(self):
        self.cache_key_stock = StockCashKey()
   
    async def all_stocks(self, data_to_cache: Dict[str, Any]) -> bool:
        cache_manager = await self.get_cache_manager()
        cache_key = self.cache_key_stock.stocks_data()
        cache_timeout = cache_manager.get_timeout(cc.TYPE_STATIC)
        return await cache_manager.set(key=cache_key, data=data_to_cache, timeout=cache_timeout)
    
    async def stock_basic_info(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_manager = await self.get_cache_manager()
        cache_key = self.cache_key_stock.stock_data(stock_code)
        # print(f"StockInfoCacheSet.stock_basic_info.cache_key: {cache_key}")
        cache_timeout = cache_manager.get_timeout(cc.TYPE_STATIC)
        return await cache_manager.set(key=cache_key, data=data_to_cache, timeout=cache_timeout)

class StockTimeTradeCacheSet(CacheSet):
    def __init__(self):
        """
        初始化方法。
        修改点:
        1. 调用 super().__init__() 继承父类初始化。
        2. 创建并持有 CacheManager 的实例，供所有方法使用。
        """
        super().__init__() # 调用父类的 __init__
        self.cache_manager = CacheManager()

    async def latest_time_trade(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将处理后的单个最新指数时间序列数据点缓存到 Redis。
        Args:
            stock_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 经过处理的、可JSON序列化的最新时间序列数据字典。
                           格式应与 get_latest_time_series_from_cache 期望的格式一致。
        Returns:
            bool: 缓存操作是否成功。
        """
        # 使用 'latest' 作为 subtype 或 id 来标识这是最新的数据点
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_time_trade.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_time_trade(stock_code, time_level)
        # print(f"latest_time_trade.cache_key: {cache_key}")
        return await self._stock_latest_data(stock_code, time_level, data_to_cache, cache_key)

    async def history_time_trade(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.history_time_trade(stock_code, time_level)
        return await self._history_data(stock_code, time_level, data_to_cache, cache_key)

    async def stock_day_basic_info(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.stock_day_basic_info(stock_code)
        return await self._history_data(stock_code, "Day_Basic_Info", data_to_cache, cache_key)

    async def batch_set_latest_time_trade(self, cache_payload: Dict[str, dict], time_level: str) -> bool:
        """
        【V1 - 高效版】使用 Redis Pipeline 批量缓存最新的分钟线数据。
        
        Args:
            cache_payload (Dict[str, dict]): 一个字典，键是股票代码，值是待缓存的数据字典。
                                             例如: {'000001.SZ': data_dict_1, '600519.SH': data_dict_2}
            time_level (str): 时间级别 (e.g., '5', '30', 'Day')。

        Returns:
            bool: 批量缓存操作是否成功提交。
        """
        # 1. 处理空输入
        if not cache_payload:
            print("调试信息: [Cache] 批量写入任务收到空数据，跳过执行。")
            return True # 空操作视为成功

        # 2. 准备 MSET 所需的数据
        mset_data = {}
        keys_to_expire = []
        
        print(f"调试信息: [Cache] 准备批量处理 {len(cache_payload)} 条分钟线数据...")
        for stock_code, data_to_cache in cache_payload.items():
            # 2.1 对每条数据进行格式转换，与单个写入的逻辑保持一致
            # 注意：_format_conversion 是您类中一个未提供但存在的方法，我们假设它在这里
            formatted_data = await self._format_conversion(data_to_cache)
            if formatted_data is None:
                logger.warning(f"批量缓存中，股票 {stock_code} 的数据格式化失败，已跳过。")
                continue

            # 2.2 生成缓存键
            cache_key = self.cache_key_stock.latest_time_trade(stock_code, time_level)
            keys_to_expire.append(cache_key)

            # 2.3 序列化值，为 pipeline 做准备
            # CacheManager 的 pipeline 直接操作 redis-py 客户端，需要我们手动序列化
            serialized_value = self.cache_manager._serialize(formatted_data)
            mset_data[cache_key] = serialized_value

        if not mset_data:
            logger.warning("批量缓存任务中，所有数据均处理失败，无数据写入。")
            return False

        # 3. 使用 Pipeline 执行批量写入和设置过期时间
        try:
            # 确保 Redis 客户端已连接
            await self.cache_manager._ensure_client()
            
            # 从 CacheManager 获取底层的 redis-py pipeline 对象
            async with self.cache_manager.redis_client.pipeline() as pipe:
                # 步骤 A: 一次性设置所有键值对
                pipe.mset(mset_data)

                # 步骤 B: 为每一个键设置过期时间
                # 我们从 cache_key 推断缓存类型为 'st' (static/timeseries)
                timeout = self.cache_manager.get_timeout('st') 
                for key in keys_to_expire:
                    pipe.expire(key, timeout)

                # 步骤 C: 原子化地执行所有命令
                await pipe.execute()
            
            print(f"调试信息: [Cache] 成功批量写入 {len(mset_data)} 条分钟线数据到Redis。")
            return True

        except Exception as e:
            logger.error(f"批量写入分钟线缓存时发生异常: {e}", exc_info=True)
            return False

class StockIndicatorsCacheSet(CacheSet):
    def __init__(self):
        self.cache_key_stock = StockCashKey()

    async def latest_time_trade(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将处理后的单个最新指数时间序列数据点缓存到 Redis。
        Args:
            stock_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 经过处理的、可JSON序列化的最新时间序列数据字典。
                           格式应与 get_latest_time_series_from_cache 期望的格式一致。
        Returns:
            bool: 缓存操作是否成功。
        """
        # 使用 'latest' 作为 subtype 或 id 来标识这是最新的数据点
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_time_trade.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_time_trade(stock_code, time_level)
        return await self._stock_latest_data(stock_code, time_level, data_to_cache, cache_key)

    async def history_time_trade(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.history_time_trade(stock_code, time_level)
        return await self._history_data(stock_code, time_level, data_to_cache, cache_key)

class StockRealtimeCacheSet(CacheSet):
    def __init__(self):
        self.cache_key_stock = StockCashKey()

    async def latest_realtime_data(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_realtime_data.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_realtime_data(stock_code)
        # logger.info(f"latest_realtime_data.cache_key: {cache_key}")
        return await self._realtime_data(stock_code, data_to_cache, cache_key)
    
    async def history_realtime_data(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.history_realtime_data(stock_code)
        return await self._history_data(stock_code, data_to_cache, cache_key)
    
    async def latest_level5_data(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_level5_data.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_level5_data(stock_code)
        return await self._stock_latest_data(stock_code, "Day", data_to_cache, cache_key)
    
    async def history_level5_data(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.history_level5_data(stock_code)
        return await self._history_data(stock_code, data_to_cache, cache_key)
    
class StrategyCacheSet(CacheSet):
    async def lastest_analyze_signals_trend_following_data(self, stock_code: str, data_to_cache: Dict[str, Any]):
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"lastest_analyze_signals_trend_following_data.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_strategy.analyze_signals_trend_following(stock_code=stock_code)
        # print(f"lastest_analyze_signals_trend_following_data.cache_key: {cache_key}")
        return await self._stock_strategy_data(stock_code=stock_code, data_to_cache=data_to_cache, cache_key=cache_key)

    async def analyze_signals_trend_following(self, stock_code: str, data_to_cache: Dict[str, Any], timestamp: pd.Timestamp) -> bool: # 修改: 添加 timestamp 参数
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"analyze_signals_trend_following.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_strategy.analyze_signals_trend_following(stock_code=stock_code)
        print(f"analyze_signals_trend_following.cache_key: {cache_key}")
        # 修改: 传递 timestamp 参数
        return await self._stock_strategy_data(stock_code=stock_code, data_to_cache=data_to_cache, cache_key=cache_key, timestamp=timestamp)












