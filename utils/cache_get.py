import datetime
import logging
import json
from typing import Any, Dict, List, Optional
from utils import cache_constants as cc
from utils.cache_manager import CacheManager
from utils.cash_key import IndexCashKey, StockCashKey, UserCashKey
from utils.data_format_process import IndexDataFormatProcess

logger = logging.getLogger("dao")

class CacheGet():

    def __init__(self):
        self.cache_manager = None
        self.cache_key_user = UserCashKey()
        self.cache_key_index = IndexCashKey()
        self.cache_key_stock = StockCashKey()
        self.data_format_process = IndexDataFormatProcess()

    async def initialize_cache_manager(self):
        from utils.cache_manager import CacheManager
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()

    async def _index_latest_data(self, index_code: str, time_level: str, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            if self.cache_manager is None:
                await self.initialize_cache_manager()
            cached_data = await self.cache_manager.get(key=cache_key)
            if cached_data is not None and isinstance(cached_data, dict):
                return cached_data
            return None
        except Exception as e:
            logger.error(f"从缓存获取指数[{index_code}] 时间级别[{time_level}] 最新时间序列数据时发生异常: {str(e)}", exc_info=True)
            return None

    # --- 修正后的读取缓存方法 (使用 ZRANGEBYSCORE) ---
    async def _history_data_by_date_range(self, stock_code: str, time_level: str, start_time: datetime, end_time: datetime, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
            return None
        min_score = start_time.timestamp()
        max_score = end_time.timestamp()
        try:
            if self.cache_manager is None:
                await self.initialize_cache_manager()
            cached_data_list = await self.cache_manager.zrangebyscore(key=cache_key, min_score=min_score, max_score=max_score)
            return cached_data_list
        except Exception as e:
            logger.error(f"从缓存获取时间序列数据时发生异常: {str(e)}", exc_info=True)
            return None

    async def _history_data_by_limit(self, cache_key: str, limit: int,) -> Optional[List[Dict[str, Any]]]:
        try:
            if self.cache_manager is None:
                await self.initialize_cache_manager()
            cached_data_list = await self.cache_manager.zrange_by_limit(key=cache_key, limit=limit)
            return cached_data_list
        except Exception as e:
            logger.error(f"从缓存 (ZSET) 获取时间序列数据时发生异常: {str(e)}, key: {cache_key}", exc_info=True)
            return None

    async def _realtime_data(self, stock_code: str, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            if self.cache_manager is None:
                await self.initialize_cache_manager()
            cached_data = await self.cache_manager.get(key=cache_key)
            if cached_data is not None:
                if isinstance(cached_data, dict):
                    logger.info(f"缓存命中: 成功获取到股票[{stock_code}]实时数据, key: {cache_key}")
                    return cached_data
                else:
                    logger.warning(f"缓存数据格式错误: 股票[{stock_code}]实时数据的缓存值不是字典类型 (实际类型: {type(cached_data)}), key: {cache_key}. 将视为未命中。")
            return cached_data
        except Exception as e:
            logger.error(f"从缓存获取股票[{stock_code}]实时数据时发生异常: {str(e)}, key: {cache_key}", exc_info=True)
            return None

    async def _stock_latest_data(self, stock_code: str, time_level: str, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            # 1. 生成缓存键 (必须与写入时使用的键完全一致)
            logger.info(f"尝试从缓存获取股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据, key: {cache_key}")
            # 2. 调用 CacheManager 获取数据
            if self.cache_manager is None:
                await self.initialize_cache_manager()
            cached_data = await self.cache_manager.get(key=cache_key)
            if cached_data is not None:
                if isinstance(cached_data, dict):
                    logger.info(f"缓存命中: 成功获取到股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据, key: {cache_key}")
                    return cached_data
                else:
                    logger.warning(f"缓存数据格式错误: 股票[{stock_code}] 时间级别[{time_level}] 的缓存值不是字典类型 (实际类型: {type(cached_data)}), key: {cache_key}. 将视为未命中。")
                    self.cache_manager.delete(cache_key) # 可选：删除错误数据
                    return None
            else:
                logger.info(f"缓存未命中: 未找到股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据, key: {cache_key}")
                return None

        except Exception as e:
            logger.error(f"StockIndicatorsDAO._stock_latest_data从缓存获取股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据时发生异常: {str(e)}, key: (生成失败或未知)", exc_info=True)
            return None

    async def _stock_strategy_data(self, stock_code: str, time_level: str, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            # 1. 生成缓存键 (必须与写入时使用的键完全一致)
            logger.info(f"尝试从缓存获取股票[{stock_code}] 时间级别[{time_level}] 策略数据, key: {cache_key}")
            # 2. 调用 CacheManager 获取数据
            if self.cache_manager is None:
                await self.initialize_cache_manager()
            cached_data = await self.cache_manager.get(key=cache_key)
            if cached_data is not None:
                if isinstance(cached_data, dict):
                    logger.info(f"缓存命中: 成功获取到股票[{stock_code}] 时间级别[{time_level}] 策略数据, key: {cache_key}")
                    return cached_data
                else:
                    logger.warning(f"缓存数据格式错误: 股票[{stock_code}] 时间级别[{time_level}] 的缓存值不是字典类型 (实际类型: {type(cached_data)}), key: {cache_key}. 将视为未命中。")
                    self.cache_manager.delete(cache_key) # 可选：删除错误数据
                    return None
            else:
                logger.info(f"缓存未命中: 未找到股票[{stock_code}] 时间级别[{time_level}] 策略数据, key: {cache_key}")
                return None

        except Exception as e:
            logger.error(f"StockIndicatorsDAO._stock_strategy_data从缓存获取股票[{stock_code}] 时间级别[{time_level}] 策略数据时发生异常: {str(e)}, key: (生成失败或未知)", exc_info=True)
            return None

class UserCacheGet(CacheGet):
    async def initialize(self):
        """
        初始化缓存管理器。代理调用父类的 initialize_cache_manager 方法。
        这确保了与 user_dao.py 中代码的兼容性。
        """
        await super().initialize_cache_manager()  # 调用父类方法初始化 cache_manager


    async def user_favorites(self, user_id: int) -> Optional[List['FavoriteStock']]:
        """
        从缓存中异步读取用户自选股列表，并将字典转换为模型实例。
        """
        from users.models import FavoriteStock
        if self.cache_manager is None:
            await self.initialize()  # 在需要时调用初始化
        cache_key = self.cache_key_user.user_favorites(user_id)  # 例如 "user:favorites:123"
        try:
            cached_data_dict = await self.cache_manager.hgetall(cache_key)  # 获取 Hash 数据，返回 Dict[str, Dict]
            logger.info(f"缓存命中用户 {user_id} 的自选股列表: {cached_data_dict}, key: {cache_key}")
            if cached_data_dict:
                favorite_list = []  # 用于存储转换后的模型实例
                for field, item_dict in cached_data_dict.items():
                    try:
                        # 将字典转换为 FavoriteStock 实例
                        # 注意：这会创建一个未保存的模型实例
                        fav_instance = FavoriteStock(**item_dict)  # 假设 item_dict 包含所有必需字段
                        favorite_list.append(fav_instance)  # 添加到列表
                    except Exception as e:  # 捕获可能的 TypeError 或 ValidationError
                        logger.error(f"转换字典到模型失败: field {field}, 数据: {item_dict}, 错误: {str(e)}")
                        continue  # 跳过失败的项，继续处理其他项
                if favorite_list:  # 如果列表不为空
                    logger.debug(f"从缓存命中用户 {user_id} 的自选股列表: {len(favorite_list)} 项")
                    return favorite_list  # 返回 List[FavoriteStock]
                else:
                    logger.debug(f"缓存数据为空或转换失败: 未找到用户[{user_id}] 的自选股列表, key: {cache_key}")
                    return None
            else:
                logger.debug(f"缓存未命中: 未找到用户[{user_id}] 的自选股列表, key: {cache_key}")
                return None
        except Exception as e:
            logger.error(f"获取用户 {user_id} 自选股列表失败: {str(e)}", exc_info=True)
            return None  # 出错时返回 None

    async def all_favorites(self) -> Optional[List['FavoriteStock']]:
        """
        从缓存中异步读取所有自选股列表，并将字典转换为模型实例。
        """
        from users.models import FavoriteStock
        cache_key = self.cache_key_user.all_favorites()  # 例如 "user:favorites:all"
        

class IndexCacheGet(CacheGet):
    async def initialize(self):
        """
        初始化缓存管理器。代理调用父类的 initialize_cache_manager 方法。
        这确保了与 user_dao.py 中代码的兼容性。
        """
        await super().initialize_cache_manager()  # 调用父类方法初始化 cache_manager
    async def all_indexes(self) -> Optional[List[Dict]]:
        """
        从缓存中获取所有指数列表，按照指数代码排序
        
        Returns:
            Optional[List[Dict]]: 按指数代码排序的指数列表，如果缓存未命中则返回None
        """
        try:
            if self.cache_manager is None:
                await self.initialize()  # 在需要时调用初始化
            cache_key = self.cache_key_index.indexs_data()
            # 获取缓存数据
            cached_data = self.cache_manager.get(cache_key)
            logger.info(f"cache_key：{cache_key}, cached_data: {cached_data}")
            if cached_data:
                # 对缓存数据按code字段排序
                return sorted(cached_data, key=lambda x: x['code'])
            return None
        except Exception as e:
            logger.error(f"从缓存获取所有指数列表时发生错误: {str(e)}", exc_info=True)
            return None

    async def realtime_data(self, index_code: str) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定指数的实时数据。
        Args:
            index_code: 指数代码。
        Returns:
            Optional[Dict[str, Any]]: 缓存中的实时数据字典，如果未命中或发生错误则返回 None。
                                      返回的字典格式应与 _cache_realtime_data 存入的格式一致。
        """
        try:
            if self.cache_manager is None:
                await self.initialize()  # 在需要时调用初始化
            # 1. 生成缓存键 (必须与写入时使用的键完全一致)
            cache_key = self.cache_key_index.realtime_data(index_code)
            logger.info(f"尝试从缓存获取指数[{index_code}]实时数据, key: {cache_key}")
            # 2. 调用 CacheManager 获取数据
            # cache_manager.get 会自动处理反序列化和解压缩
            cached_data = self.cache_manager.get(key=cache_key)
            if cached_data is not None:
                # 验证数据类型是否符合预期（可选但推荐）
                if isinstance(cached_data, dict):
                    logger.info(f"缓存命中: 成功获取到指数[{index_code}]的实时数据, key: {cache_key}")
                    return cached_data
                else:
                    logger.warning(f"缓存数据格式错误: 指数[{index_code}]的缓存值不是字典类型 (实际类型: {type(cached_data)}), key: {cache_key}. 将视为未命中。")
                    # 可以考虑删除错误的缓存项
                    self.cache_manager.delete(cache_key)
                    return None
            else:
                logger.info(f"缓存未命中: 未找到指数[{index_code}]的实时数据, key: {cache_key}")
                return None
        except Exception as e:
            logger.error(f"从缓存获取指数[{index_code}]实时数据时发生异常: {str(e)}, key: (生成失败或未知)", exc_info=True)
            # 确保在异常情况下返回 None
            return None

    async def latest_time_series(self, index_code: str, time_level: str,) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定指数和时间级别的最新时间序列数据点。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
        Returns:
            Optional[Dict[str, Any]]: 缓存中的最新时间序列数据字典，如果未命中或发生错误则返回 None。
        """
        cache_key = self.cache_key_index.latest_time_series(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, cache_key)

    async def history_time_series(self, index_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_index.history_time_series(index_code, time_level)
        return await self._history_data_by_date_range(index_code, time_level, start_time, end_time, cache_key)

    async def latest_macd(self, index_code: str, time_level: str,) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定指数和时间级别的最新时间序列数据点。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
        Returns:
            Optional[Dict[str, Any]]: 缓存中的最新时间序列数据字典，如果未命中或发生错误则返回 None。
        """
        cache_key = self.cache_key_index.latest_macd(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, cache_key)

    async def history_macd(self, index_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        从缓存 (Redis Sorted Set) 中获取指定时间范围内的指数时间序列数据。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            start_time: 开始时间 (datetime 对象, 包含)。
            end_time: 结束时间 (datetime 对象, 包含)。
        Returns:
            Optional[List[Dict[str, Any]]]: 时间范围内的数据点字典列表，按时间升序排列。
                                           如果未命中、范围内无数据或发生错误则返回 None 或空列表。
        """
        if self.cache_manager is None:
            await self.initialize()  # 在需要时调用初始化
        cache_key = self.cache_key_index.history_macd(index_code, time_level)
        return await self._history_data_by_date_range(index_code, time_level, start_time, end_time, cache_key)

    async def latest_kdj(self, index_code: str, time_level: str,) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定指数和时间级别的最新KDJ数据点。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
        Returns:
            Optional[Dict[str, Any]]: 缓存中的最新KDJ数据字典，如果未命中或发生错误则返回 None。
        """
        if self.cache_manager is None:
            await self.initialize()  # 在需要时调用初始化
        cache_key = self.cache_key_index.latest_kdj(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, cache_key)

    async def history_kdj(self, index_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        从缓存 (Redis Sorted Set) 中获取指定时间范围内的指数时间序列数据。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            start_time: 开始时间 (datetime 对象, 包含)。
            end_time: 结束时间 (datetime 对象, 包含)。
        Returns:
            Optional[List[Dict[str, Any]]]: 时间范围内的数据点字典列表，按时间升序排列。
            如果未命中、范围内无数据或发生错误则返回 None 或空列表。
        """
        cache_key = self.cache_key_index.history_kdj(index_code, time_level)
        return await self._history_data_by_date_range(index_code, time_level, start_time, end_time, cache_key)

    async def latest_ma(self, index_code: str, time_level: str,) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定指数和时间级别的最新MA数据点。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
        Returns:
            Optional[Dict[str, Any]]: 缓存中的最新MA数据字典，如果未命中或发生错误则返回 None。
        """
        cache_key = self.cache_key_index.latest_ma(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, cache_key)

    async def history_ma(self, index_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        从缓存 (Redis Sorted Set) 中获取指定时间范围内的指数时间序列数据。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            start_time: 开始时间 (datetime 对象, 包含)。
            end_time: 结束时间 (datetime 对象, 包含)。
        Returns:
            Optional[List[Dict[str, Any]]]: 时间范围内的数据点字典列表，按时间升序排列。
            如果未命中、范围内无数据或发生错误则返回 None 或空列表。
        """
        cache_key = self.cache_key_index.history_ma(index_code, time_level)
        return await self._history_data_by_date_range(index_code, time_level, start_time, end_time, cache_key)

    async def latest_boll(self, index_code: str, time_level: str,) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定指数和时间级别的最新BOLL数据点。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
        Returns:
            Optional[Dict[str, Any]]: 缓存中的最新BOLL数据字典，如果未命中或发生错误则返回 None。
        """
        cache_key = self.cache_key_index.latest_boll(index_code, time_level)
        return await self._index_latest_data(index_code, time_level, cache_key)

    async def history_boll(self, index_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        从缓存 (Redis Sorted Set) 中获取指定时间范围内的指数时间序列数据。
        Args:
            index_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
            start_time: 开始时间 (datetime 对象, 包含)。
            end_time: 结束时间 (datetime 对象, 包含)。
        Returns:
            Optional[List[Dict[str, Any]]]: 时间范围内的数据点字典列表，按时间升序排列。
            如果未命中、范围内无数据或发生错误则返回 None 或空列表。
        """
        cache_key = self.cache_key_index.history_boll(index_code, time_level)
        return await self._history_data_by_date_range(index_code, time_level, start_time, end_time, cache_key)

class StockInfoCacheGet(CacheGet):
    async def initialize(self):
        """
        初始化缓存管理器。代理调用父类的 initialize_cache_manager 方法。
        这确保了与 user_dao.py 中代码的兼容性。
        """
        await super().initialize_cache_manager()  # 调用父类方法初始化 cache_manager

    async def all_stocks(self) -> Optional[List[Dict]]:
        """
        从缓存中获取所有股票列表，按照股票代码排序
        """
        if self.cache_manager is None:
            await self.initialize()  # 在需要时调用初始化
        cache_key = self.cache_key.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_STOCK,
            entity_id=cc.ID_ALL
        )
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            return sorted(cached_data, key=lambda x: x['stock_code'])
        return None

class StockIndicatorsCacheGet(CacheGet):
    async def initialize(self):
        """
        初始化缓存管理器。代理调用父类的 initialize_cache_manager 方法。
        这确保了与 user_dao.py 中代码的兼容性。
        """
        await super().initialize_cache_manager()  # 调用父类方法初始化 cache_manager

    async def latest_time_trade(self, stock_code: str, time_level: str,) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定指数和时间级别的最新时间序列数据点。
        Args:
            stock_code: 指数代码。
            time_level: 时间级别 (e.g., '5', '30', 'Day').
        Returns:
            Optional[Dict[str, Any]]: 缓存中的最新时间序列数据字典，如果未命中或发生错误则返回 None。
        """
        cache_key = self.cache_key_stock.latest_time_trade(stock_code, time_level)
        return await self._stock_latest_data(stock_code, time_level, cache_key)

    async def history_time_trade(self, stock_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_time_trade(stock_code, time_level)
        return await self._history_data_by_date_range(stock_code, time_level, start_time, end_time, cache_key)
    
    async def history_time_trade_by_limit(self, stock_code: str, time_level: str, limit: int) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_time_trade(stock_code, time_level)
        return await self._history_data_by_limit(cache_key, limit)

    async def latest_kdj(self, stock_code: str, time_level: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_kdj(stock_code, time_level)
        data = await self._stock_latest_data(stock_code, time_level, cache_key)
        return data

    async def history_kdj_by_date_range(self, stock_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_kdj(stock_code, time_level)
        data = await self._history_data_by_date_range(stock_code, time_level, start_time, end_time, cache_key)
        return data

    async def history_kdj_by_limit(self, stock_code: str, time_level: str, limit: int) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_kdj(stock_code, time_level)
        data = await self._history_data_by_limit(cache_key, limit)
        return data

    async def latest_macd(self, stock_code: str, time_level: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_macd(stock_code, time_level)
        return await self._stock_latest_data(stock_code, time_level, cache_key)

    async def history_macd(self, stock_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_macd(stock_code, time_level)
        return await self._history_data_by_date_range(stock_code, time_level, start_time, end_time, cache_key)

    async def latest_ma(self, stock_code: str, time_level: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_ma(stock_code, time_level)
        return await self._stock_latest_data(stock_code, time_level, cache_key)

    async def history_ma(self, stock_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_ma(stock_code, time_level)
        return await self._history_data_by_date_range(stock_code, time_level, start_time, end_time, cache_key)

    async def latest_boll(self, stock_code: str, time_level: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_boll(stock_code, time_level)
        return await self._stock_latest_data(stock_code, time_level, cache_key)

    async def history_boll(self, stock_code: str, time_level: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_boll(stock_code, time_level)
        return await self._history_data_by_date_range(stock_code, time_level, start_time, end_time, cache_key)

class StockRealtimeCacheGet(CacheGet):
    async def initialize(self):
        """
        初始化缓存管理器。代理调用父类的 initialize_cache_manager 方法。
        这确保了与 user_dao.py 中代码的兼容性。
        """
        await super().initialize_cache_manager()  # 调用父类方法初始化 cache_manager

    async def latest_realtime_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_realtime_data(stock_code)
        # logger.info(f"尝试从缓存获取股票[{stock_code}]最新实时数据, key: {cache_key}")
        return await self._realtime_data(stock_code=stock_code, cache_key=cache_key)
    
    async def history_realtime_data(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_realtime_data(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)

    async def latest_onebyone_trade(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_onebyone_trade(stock_code)
        return await self._stock_latest_data(stock_code, cache_key)
    
    async def history_onebyone_trade(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_onebyone_trade(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)

    async def latest_time_deal(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_time_deal(stock_code)
        return await self._stock_latest_data(stock_code, cache_key)

    async def history_time_deal(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_time_deal(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)

    async def latest_real_percent(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_real_percent(stock_code)
        return await self._stock_latest_data(stock_code, cache_key)

    async def history_real_percent(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_real_percent(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)
    
    async def latest_big_deal(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_big_deal(stock_code)
        return await self._stock_latest_data(stock_code, cache_key)

    async def history_big_deal(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_big_deal(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)
    
    async def latest_abnormal_movement(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_abnormal_movement(stock_code)
        return await self._stock_latest_data(stock_code, cache_key)

    async def history_abnormal_movement(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_abnormal_movement(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)
    
    async def latest_level5_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_level5_data(stock_code)
        return await self._stock_latest_data(stock_code, cache_key)

    async def history_level5_data(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_level5_data(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)
    
class StrategyCacheGet(CacheGet):
    async def initialize(self):
        """
        初始化缓存管理器。代理调用父类的 initialize_cache_manager 方法。
        这确保了与 user_dao.py 中代码的兼容性。
        """
        await super().initialize_cache_manager()  # 调用父类方法初始化 cache_manager

    async def macd_rsi_kdj_boll_data(self, stock_code: str, time_level: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_strategy.macd_rsi_kdj_boll_data(stock_code, time_level)
        return await self._stock_strategy_data(stock_code, time_level, cache_key)







