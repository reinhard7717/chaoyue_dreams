import logging
import pickle
from typing import Any, Dict, List
from asgiref.sync import sync_to_async
import umsgpack
from datetime import datetime, date # 同时导入 date 类，如果需要处理的话
from decimal import Decimal # 导入 Decimal
from utils import cache_constants as cc
import json

from utils.cash_key import IndexCashKey, StockCashKey, UserCashKey

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
        self.cache_manager = None  # 临时设置为 None，实际使用时需异步初始化
        self.cache_key_index = IndexCashKey()
        self.cache_key_stock = StockCashKey()
        self.cache_key_strategy = StrategyCashKey()
        self.data_format_process = IndexDataFormatProcess()
        self.cache_key_user = UserCashKey()
    
    async def initialize_cache_manager(self):
        from utils.cache_manager import CacheManager
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 初始化方法  # 异步初始化

    async def _index_latest_data(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any], cache_key: str) -> bool:
        if not data_to_cache:
            logger.warning(f"试图缓存指数[{index_code}] 时间级别[{time_level}] 的空时间序列数据，操作跳过。")
            return False
        if self.cache_manager is None:
            await self.initialize()  # 在需要时调用初始化
        try:
            cache_timeout = self.cache_manager.get_timeout(cc.TYPE_REALTIME)
            success = await self.cache_manager.set(  # 添加 await
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
            if self.cache_manager is None:
                await self.initialize_cache_manager()  # 确保初始化
            cache_timeout = self.cache_manager.get_timeout(cc.TYPE_REALTIME) # 或者 cc.TYPE_TIMESERIES
            # 3. 调用 CacheManager 设置缓存
            success = await self.cache_manager.set(
                key=cache_key,
                data=data_to_cache,
                timeout=cache_timeout
            )
            if success:
                # logger.info(f"股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据缓存成功, key: {cache_key}")
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
            if self.cache_manager is None:
                await self.initialize_cache_manager()  # 确保初始化
            cache_timeout = self.cache_manager.get_timeout(cc.TYPE_REALTIME) # 或者 cc.TYPE_TIMESERIES
            success = await self.cache_manager.set(
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

    async def _stock_strategy_data(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any], cache_key: str) -> bool:
        if not data_to_cache:
            logger.warning(f"试图缓存股票[{stock_code}] 时间级别[{time_level}] 的空时间序列数据，操作跳过。")
            return False
        try:
            if self.cache_manager is None:
                await self.initialize_cache_manager()  # 确保初始化
            cache_timeout = self.cache_manager.get_timeout(cc.TYPE_STRATEGY) # 或者 cc.TYPE_TIMESERIES
            # 3. 调用 CacheManager 设置缓存
            success = await self.cache_manager.set(
                key=cache_key,
                data=data_to_cache,
                timeout=cache_timeout
            )
            if success:
                # logger.info(f"股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据缓存成功, key: {cache_key}")
                return True
            else:
                logger.warning(f"缓存股票[{stock_code}] 策略数据失败 (CacheManager.set 返回 False), key: {cache_key}")
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
            if self.cache_manager is None:
                await self.initialize_cache_manager()
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
            cache_timeout = self.cache_manager.get_timeout(cc.TYPE_TIMESERIES)
            success = await self.cache_manager.zadd(cache_key, mapping_to_send, cache_timeout)
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
        self.cache_manager = None  # 初始为 None
        self.cache_key_user = UserCashKey()

    async def initialize(self):
        from utils.cache_manager import CacheManager  # 导入
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 初始化方法  # 异步初始化

    async def user_favorites(self, user_id: int, data_to_cache: List[Dict]) -> bool:
        """
        将用户自选股列表缓存到 Redis，使用 Hash 类型。
        """
        if self.cache_manager is None:
            await self.initialize()  # 在需要时调用初始化
        cache_key = self.cache_key_user.user_favorites(int(user_id))  # 假设返回如 "user:favorites:123"
        try:
            if self.cache_manager is None:
                await self.initialize_cache_manager()
            for index, item in enumerate(data_to_cache):
                field = str(item.get('id', index))  # 使用 'id' 作为 field，如果没有则用索引
                success = await self.cache_manager.hset(cache_key, field, item)  # 异步调用 hset
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
        self.cache_manager = None  # 初始为 None
        self.cache_key_index = IndexCashKey()

    async def initialize(self):
        from utils.cache_manager import CacheManager  # 导入
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 初始化方法  # 异步初始化

    async def indexes(self, indexes: List[Dict]) -> bool:
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
        if self.cache_manager is None:
            await self.initialize_cache_manager()
        # 1. 输入验证 (可选但推荐)
        if not isinstance(indexes, list):
            logger.error("set_indexes_to_cache 失败: 输入数据不是列表")
            return False
        # 可以添加更详细的验证，例如检查列表中的元素是否为字典，是否包含必要的键等
        data_dicts = []
        for item in indexes:
            data_dict = await self.data_format_process.set_index_data(item)
            data_dicts.append(data_dict)
        logger.debug(f"从数据库获取股票指数列表，共{len(indexes)}条")
        # 2. 生成缓存键 (与 get_all_indexes/fetch_and_save_indexes 保持一致)
        cache_key = self.cache_key_index.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_INDEX,
            entity_id=cc.ID_ALL
        )
        # 3. 获取缓存超时时间
        cache_timeout = self.cache_manager.get_timeout(cc.TYPE_STATIC)
        logger.info(f"准备将 {len(data_dicts)} 条指数数据设置到缓存, key: {cache_key}, timeout: {cache_timeout}s")
        # 4. 调用 CacheManager 设置缓存
        try:
            # 直接使用传入的 indexes_data，因为它已经是期望的格式
            success = await self.cache_manager.set(
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
        if self.cache_manager is None:
            await self.initialize_cache_manager()
        try:
            # 1. 生成缓存键
            cache_key = self.cache_key_index.realtime_data(index_code)
            # 2. 获取缓存超时时间 (实时数据通常较短)
            cache_timeout = self.cache_manager.get_timeout(cc.TYPE_REALTIME)
            # 3. 调用 CacheManager 设置缓存
            # data_to_cache 应该是可以直接序列化的字典
            success = await self.cache_manager.set(
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

    async def latest_macd(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
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
            logger.error(f"latest_macd.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.latest_macd(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, data_to_cache, cache_key)

    async def history_macd(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将单个时间序列数据点添加到 Redis 有序集合中进行缓存。
        使用数据点的时间戳作为分数，数据本身序列化后作为成员。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 包含单个时间序列数据点的字典，必须包含可转换为时间戳的时间字段 (如 'trade_time')。
        Returns:
            bool: 操作是否成功。
        """
        # ***核心修改：转换 StockInfo 和 datetime 对象***
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"history_macd.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.history_macd(index_code, time_level)
        return await self._history_data(index_code, time_level, data_to_cache, cache_key)

    async def latest_kdj(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
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
            logger.error(f"latest_kdj.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.latest_kdj(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, data_to_cache, cache_key)

    async def history_kdj(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将单个时间序列数据点添加到 Redis 有序集合中进行缓存。
        使用数据点的时间戳作为分数，数据本身序列化后作为成员。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 包含单个时间序列数据点的字典，必须包含可转换为时间戳的时间字段 (如 'trade_time')。
        Returns:
            bool: 操作是否成功。
        """
        # ***核心修改：转换 StockInfo 和 datetime 对象***
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"history_kdj.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.history_kdj(index_code, time_level)
        return await self._history_data(index_code, time_level, data_to_cache, cache_key)

    async def latest_ma(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将处理后的单个最新指数时间序列数据点缓存到 Redis。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 经过处理的、可JSON序列化的最新时间序列数据字典。
                           格式应与 get_latest_time_series_from_cache 期望的格式一致。
        """
        # ***核心修改：转换 StockInfo 和 datetime 对象***
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_ma.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.latest_ma(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, data_to_cache, cache_key)

    async def history_ma(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将单个时间序列数据点添加到 Redis 有序集合中进行缓存。
        使用数据点的时间戳作为分数，数据本身序列化后作为成员。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 包含单个时间序列数据点的字典，必须包含可转换为时间戳的时间字段 (如 'trade_time')。
        Returns:
            bool: 操作是否成功。
        """
        # ***核心修改：转换 StockInfo 和 datetime 对象***
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"history_ma.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.history_ma(index_code, time_level)
        return await self._history_data(index_code, time_level, data_to_cache, cache_key)

    async def latest_boll(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将处理后的单个最新指数时间序列数据点缓存到 Redis。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 经过处理的、可JSON序列化的最新时间序列数据字典。
                           格式应与 get_latest_time_series_from_cache 期望的格式一致。
        """
        # ***核心修改：转换 StockInfo 和 datetime 对象***
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_boll.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.latest_boll(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, data_to_cache, cache_key)

    async def history_boll(self, index_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将单个时间序列数据点添加到 Redis 有序集合中进行缓存。
        使用数据点的时间戳作为分数，数据本身序列化后作为成员。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            data_to_cache: 包含单个时间序列数据点的字典，必须包含可转换为时间戳的时间字段 (如 'trade_time')。
        Returns:
            bool: 操作是否成功。
        """
        # ***核心修改：转换 StockInfo 和 datetime 对象***
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"history_boll.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_index.history_boll(index_code, time_level)
        return await self._history_data(index_code, time_level, data_to_cache, cache_key)

class StockIndicatorsCacheSet(CacheSet):
    def __init__(self):
        self.cache_manager = None  # 初始为 None
        self.cache_key_stock = StockCashKey()

    async def initialize(self):
        from utils.cache_manager import CacheManager  # 导入
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 初始化方法  # 异步初始化

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

    async def latest_kdj(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        """
        将处理后的单个最新KDJ指标数据缓存到 Redis。
        Args:
            stock_code: 股票代码
            time_level: 时间级别 (e.g., '5', '30', 'Day')
            data_to_cache: 经过处理的、可JSON序列化的最新KDJ指标数据字典
        Returns:
            bool: 缓存操作是否成功
        """
        try:
            # ***核心修改：转换 StockInfo 和 datetime 对象***
            data_to_cache = await self._format_conversion(data_to_cache)
            if data_to_cache is None:
                logger.error(f"latest_kdj.data_to_cache转换失败。")
                return False
            cache_key = self.cache_key_stock.latest_kdj(stock_code, time_level)
            return await self._stock_latest_data(stock_code, time_level, data_to_cache, cache_key)
        except Exception as e:
            logger.error(f"缓存最新KDJ数据失败: {str(e)}")
            return False
    
    async def history_kdj(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.history_kdj(stock_code, time_level)
        return await self._history_data(stock_code, time_level, data_to_cache, cache_key)
    
    async def latest_macd(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_macd.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_macd(stock_code, time_level)
        return await self._stock_latest_data(stock_code, time_level, data_to_cache, cache_key)
    
    async def history_macd(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.history_macd(stock_code, time_level)
        return await self._history_data(stock_code, time_level, data_to_cache, cache_key)
    
    async def latest_ma(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_ma.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_ma(stock_code, time_level)
        # logger.warning(f"latest_ma.data_to_cache: {data_to_cache}, cache_key: {cache_key}")
        return await self._stock_latest_data(stock_code, time_level, data_to_cache, cache_key)
    
    async def history_ma(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.history_ma(stock_code, time_level)
        return await self._history_data(stock_code, time_level, data_to_cache, cache_key)

    async def latest_boll(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"latest_boll.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_boll(stock_code, time_level)
        # logger.warning(f"latest_boll.data_to_cache: {data_to_cache}")
        return_data = await self._stock_latest_data(stock_code, time_level, data_to_cache, cache_key)
        return return_data

    async def history_boll(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        cache_key = self.cache_key_stock.history_boll(stock_code, time_level)
        return await self._history_data(stock_code, time_level, data_to_cache, cache_key)

class StockRealtimeCacheSet(CacheSet):
    def __init__(self):
        self.cache_manager = None  # 初始为 None
        self.cache_key_stock = StockCashKey()

    async def initialize(self):
        from utils.cache_manager import CacheManager  # 导入
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 初始化方法  # 异步初始化

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
    
    async def onebyone_trade(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"onebyone_trade.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_onebyone_trade(stock_code)
        return await self._stock_latest_data(stock_code, "Day", data_to_cache, cache_key)
    
    async def time_deal(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"time_deal.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_time_deal(stock_code)
        return await self._stock_latest_data(stock_code, "Day", data_to_cache, cache_key)

    async def real_percent(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"real_percent.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_real_percent(stock_code)
        return await self._stock_latest_data(stock_code, "Day", data_to_cache, cache_key)
    
    async def big_deal(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"big_deal.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_big_deal(stock_code)
        return await self._stock_latest_data(stock_code, "Day", data_to_cache, cache_key)
    
    async def abnormal_movement(self, stock_code: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"abnormal_movement.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_stock.latest_abnormal_movement(stock_code)
        return await self._stock_latest_data(stock_code, "Day", data_to_cache, cache_key)

class StrategyCacheSet(CacheSet):
    def __init__(self):
        self.cache_manager = None  # 初始为 None

    async def initialize(self):
        from utils.cache_manager import CacheManager  # 导入
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 初始化方法  # 异步初始化

    async def macd_rsi_kdj_boll_data(self, stock_code: str, time_level: str, data_to_cache: Dict[str, Any]) -> bool:
        data_to_cache = await self._format_conversion(data_to_cache)
        if data_to_cache is None:
            logger.error(f"macd_rsi_kdj_boll_data.data_to_cache转换失败。")
            return False
        cache_key = self.cache_key_strategy.macd_rsi_kdj_boll_data(stock_code, time_level)
        # logger.info(f"macd_rsi_kdj_boll_data.cache_key: {cache_key}")
        return await self._stock_strategy_data(stock_code, time_level, data_to_cache, cache_key)











