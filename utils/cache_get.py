import datetime
import logging
import asyncio
import json
import pandas as pd
from typing import Any, Dict, List, Optional
from users.models import FavoriteStock
from utils import cache_constants as cc
from utils.cache_manager import CacheManager
from utils.cash_key import IndexCashKey, StockCashKey, StrategyCashKey, UserCashKey
from utils.data_format_process import IndexDataFormatProcess

logger = logging.getLogger("dao")

class CacheGet():
    def __init__(self, cache_manager_instance):
        # 调用父类构造函数时，传递 cache_manager_instance
        self.cache_manager = cache_manager_instance
        self.cache_key_user = UserCashKey()
        self.cache_key_index = IndexCashKey()
        self.cache_key_stock = StockCashKey()
        self.cache_key_strategy = StrategyCashKey()
        self.data_format_process = IndexDataFormatProcess(cache_manager_instance)
    async def _index_latest_data(self, index_code: str, time_level: str, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
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
            cached_data_list = await self.cache_manager.zrangebyscore(key=cache_key, min_score=min_score, max_score=max_score)
            return cached_data_list
        except Exception as e:
            logger.error(f"从缓存获取时间序列数据时发生异常: {str(e)}", exc_info=True)
            return None
    async def _history_data_by_limit(self, cache_key: str, limit: int,) -> Optional[List[Dict[str, Any]]]:
        try:
            cached_data_list = await self.cache_manager.zrange_by_limit(key=cache_key, limit=limit)
            return cached_data_list
        except Exception as e:
            logger.error(f"从缓存 (ZSET) 获取时间序列数据时发生异常: {str(e)}, key: {cache_key}", exc_info=True)
            return None
    async def _realtime_data(self, stock_code: str, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            cached_data = await self.cache_manager.get(key=cache_key)
            if cached_data is not None:
                if isinstance(cached_data, dict):
                    # logger.info(f"缓存命中: 成功获取到股票[{stock_code}]实时数据, key: {cache_key}")
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
            # logger.info(f"尝试从缓存获取股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据, key: {cache_key}")
            # 2. 调用 CacheManager 获取数据
            cached_data = await self.cache_manager.get(key=cache_key)
            if cached_data is not None:
                if isinstance(cached_data, dict):
                    # logger.info(f"缓存命中: 成功获取到股票[{stock_code}] 时间级别[{time_level}] 最新时间序列数据, key: {cache_key}")
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
    async def _stock_strategy_datas(self, stock_code: str, cache_key: str, days_count: int) -> List[Dict[str, Any]]:
        try:
            logger.info(f"尝试从缓存获取股票[{stock_code}] 近三日策略数据, key: {cache_key}")
            end_timestamp = datetime.now().timestamp()
            start_datetime = datetime.now() - datetime.timedelta(days=days_count)
            start_timestamp = start_datetime.timestamp()
            cached_members = await self.cache_manager.zrangebyscore(
                key=cache_key,
                min=start_timestamp,
                max=end_timestamp,
                withscores=False
            )
            result_data: List[Dict[str, Any]] = []
            if cached_members:
                logger.info(f"缓存命中: 找到股票[{stock_code}] 近三日策略数据, 共 {len(cached_members)} 条原始数据, key: {cache_key}")
                for json_data_bytes in cached_members:
                    try:
                        # 使用 CacheManager 的 _deserialize 方法来反序列化数据
                        cached_data = self.cache_manager._deserialize(json_data_bytes)
                        if isinstance(cached_data, dict):
                            result_data.append(cached_data)
                        else:
                            logger.warning(f"缓存数据格式错误: 股票[{stock_code}] 的缓存值解析后不是字典类型 (实际类型: {type(cached_data)}), key: {cache_key}, 值: {json_data_bytes.decode('utf-8', errors='ignore')}. 将跳过此条数据。")
                    # 捕获更通用的 Exception，因为 _deserialize 内部已处理具体错误
                    except Exception as e:
                        logger.error(f"缓存数据解析失败: 股票[{stock_code}] 的缓存值不是有效的 MessagePack 格式, key: {cache_key}, 值: {json_data_bytes.decode('utf-8', errors='ignore')}, 错误: {e}. 将跳过此条数据。")
                if result_data:
                    logger.info(f"成功解析股票[{stock_code}] 近三日策略数据, 共 {len(result_data)} 条有效数据, key: {cache_key}")
                else:
                    logger.warning(f"股票[{stock_code}] 近三日策略数据虽然有缓存成员，但无有效解析数据, key: {cache_key}")
            else:
                logger.info(f"缓存未命中: 未找到股票[{stock_code}] 近三日策略数据或 ZSET 在指定时间范围内为空, key: {cache_key}")
            return result_data
        except Exception as e:
            logger.error(f"StrategyCacheGet._stock_strategy_data从缓存获取股票[{stock_code}] 策略数据时发生异常: {str(e)}, key: {cache_key}", exc_info=True)
            return []
    
class UserCacheGet(CacheGet):
    def __init__(self, cache_manager_instance):
        # 调用父类并传递实例
        super().__init__(cache_manager_instance)
    async def initialize(self):
        """
        初始化缓存管理器。代理调用父类的 initialize_cache_manager 方法。
        这确保了与 user_dao.py 中代码的兼容性。
        """
    async def user_favorites(self, user_id: int) -> Optional[List['FavoriteStock']]:
        """
        从缓存中异步读取用户自选股列表，并将字典转换为模型实例。
        """
        from users.models import FavoriteStock
        cache_key = self.cache_key_user.user_favorites(user_id)  # 例如 "user:favorites:123"
        try:
            cached_data_dict = await self.cache_manager.hgetall(cache_key)  # 获取 Hash 数据，返回 Dict[str, Dict]
            # logger.info(f"缓存命中用户 {user_id} 的自选股列表: {cached_data_dict}, key: {cache_key}")
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
    def __init__(self, cache_manager_instance):
        # 调用父类并传递实例
        super().__init__(cache_manager_instance)
    async def all_indexes(self) -> Optional[List[Dict]]:
        """
        从缓存中获取所有指数列表，按照指数代码排序
        Returns:
            Optional[List[Dict]]: 按指数代码排序的指数列表，如果缓存未命中则返回None
        """
        try:
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
    async def index_data_by_code(self, index_code: str) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定指数的基础信息。
        Args:
            index_code: 指数代码。
        Returns:
            Optional[Dict[str, Any]]: 缓存中的基础信息字典，如果未命中或发生错误则返回 None。
        """
        cache_key = self.cache_key_index.index_data(index_code)
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        return None
    async def index_weight(self, index_code: str) -> Optional[List[Dict[str, Any]]]:
        """
        从缓存中获取指定指数的成分权重。
        Args:
            index_code: 指数代码。
        Returns:
            Optional[List[Dict[str, Any]]]: 缓存中的成分权重列表，如果未命中或发生错误则返回 None。
        """
        cache_key = self.cache_key_index.index_weight(index_code)
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
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

class StockInfoCacheGet(CacheGet):
    def __init__(self, cache_manager_instance):
        # 调用父类并传递实例
        super().__init__(cache_manager_instance)
    async def all_stocks(self) -> Optional[List[Dict]]:
        """
        从缓存中获取所有股票列表，按照股票代码排序
        """
        cache_key = self.cache_key_stock.stocks_data()
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return sorted(cached_data, key=lambda x: x['stock_code'])
        return None
    async def stock_data_by_code(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        从缓存中获取指定股票代码的股票数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[Dict[str, Any]]: 缓存中的股票数据字典，如果未命中或发生错误则返回 None。
        """
        cache_key = self.cache_key_stock.stock_data(stock_code)
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        return None

class StockTimeTradeCacheGet(CacheGet):
    def __init__(self, cache_manager_instance):
        # 调用父类并传递实例
        super().__init__(cache_manager_instance)
    async def stock_day_basic_info_by_limit(self, stock_code: str, limit: int) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.stock_day_basic_info(stock_code)
        return await self._history_data_by_limit(cache_key, limit)
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

class StockRealtimeCacheGet(CacheGet):
    def __init__(self, cache_manager_instance):
        # 调用父类构造函数时，传递 cache_manager_instance
        super().__init__(cache_manager_instance)
    async def get_intraday_ticks(self, stock_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        【V1.1 - 修复 datetime 导入问题】从Redis ZSET缓存中获取指定股票、指定日期的【全部】Tick快照数据。
        """
        try:
            # 1. 统一日期格式
            try:
                date_obj = pd.to_datetime(trade_date)
                date_str_yyyymmdd = date_obj.strftime('%Y%m%d')
            except ValueError:
                logger.error(f"无效的日期格式: {trade_date}。无法继续获取Ticks。")
                return None
            # 2. 定义当天的缓存键
            realtime_key = self.cache_key_stock.intraday_ticks_realtime(stock_code, date_str_yyyymmdd)
            level5_key = self.cache_key_stock.intraday_ticks_level5(stock_code, date_str_yyyymmdd)
            # 3. 并发地从Redis获取两种Tick数据
            tasks = [
                self.cache_manager.zrangebyscore(realtime_key, '-inf', '+inf', withscores=True),
                self.cache_manager.zrangebyscore(level5_key, '-inf', '+inf', withscores=True)
            ]
            realtime_ticks, level5_ticks = await asyncio.gather(*tasks, return_exceptions=True)
            # 4. 检查并处理结果
            if isinstance(realtime_ticks, Exception) or not realtime_ticks:
                logger.warning(f"未能从缓存获取 {stock_code} on {date_str_yyyymmdd} 的实时行情Ticks。Key: '{realtime_key}'")
                return None
            if isinstance(level5_ticks, Exception):
                logger.error(f"从Redis获取level5_ticks时出错: {level5_ticks}", exc_info=level5_ticks)
                level5_ticks = []
            # 5. 将原始数据转换为DataFrame并合并
            df_realtime = pd.DataFrame(
                [data for data, score in realtime_ticks],
                index=pd.to_datetime([datetime.datetime.fromtimestamp(score) for data, score in realtime_ticks])
            )
            if level5_ticks:
                df_level5 = pd.DataFrame(
                    [data for data, score in level5_ticks],
                    index=pd.to_datetime([datetime.datetime.fromtimestamp(score) for data, score in level5_ticks])
                )
                df_ticks = pd.merge_asof(df_realtime.sort_index(), df_level5.sort_index(), left_index=True, right_index=True, direction='backward')
            else:
                df_ticks = df_realtime
            logger.debug(f"成功从Redis获取并合并了 {len(df_ticks)} 条Tick数据 for {stock_code}")
            return df_ticks
        except Exception as e:
            # 明确地记录下当前方法名，便于追踪
            logger.error(f"在 get_intraday_ticks 中发生异常 for {stock_code}: {e}", exc_info=True)
            return None
    async def latest_tick_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_realtime_data(stock_code)
        # logger.info(f"尝试从缓存获取股票[{stock_code}]最新实时数据, key: {cache_key}")
        return await self._realtime_data(stock_code=stock_code, cache_key=cache_key)
    async def history_realtime_data(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_realtime_data(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)
    async def latest_level5_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_stock.latest_level5_data(stock_code)
        return await self._stock_latest_data(stock_code, cache_key)
    async def history_level5_data(self, stock_code: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        cache_key = self.cache_key_stock.history_level5_data(stock_code)
        return await self._history_data_by_date_range(stock_code, start_time, end_time, cache_key)
    async def get_intraday_ticks(self, stock_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        【V1.0 - 新增】从Redis ZSET缓存中获取指定股票、指定日期的【全部】Tick快照数据。
        这是从 DAO 层重构过来的标准缓存获取方法。
        """
        try:
            # 1. 统一日期格式
            try:
                date_obj = pd.to_datetime(trade_date)
                date_str_yyyymmdd = date_obj.strftime('%Y%m%d')
            except ValueError:
                logger.error(f"无效的日期格式: {trade_date}。无法继续获取Ticks。")
                return None
            # 2. 定义当天的缓存键
            realtime_key = self.cache_key_stock.intraday_ticks_realtime(stock_code, date_str_yyyymmdd)
            level5_key = self.cache_key_stock.intraday_ticks_level5(stock_code, date_str_yyyymmdd)
            print(f"DEBUG_READ: [Ticks] ZRANGE from realtime_key='{realtime_key}'")
            print(f"DEBUG_READ: [Ticks] ZRANGE from level5_key='{level5_key}'")
            # 3. 并发地从Redis获取两种Tick数据
            tasks = [
                self.cache_manager.zrangebyscore(realtime_key, '-inf', '+inf', withscores=True),
                self.cache_manager.zrangebyscore(level5_key, '-inf', '+inf', withscores=True)
            ]
            realtime_ticks, level5_ticks = await asyncio.gather(*tasks, return_exceptions=True)
            # 4. 检查并处理结果
            if isinstance(realtime_ticks, Exception) or not realtime_ticks:
                logger.warning(f"未能从缓存获取 {stock_code} on {date_str_yyyymmdd} 的实时行情Ticks。Key: '{realtime_key}'")
                return None
            if isinstance(level5_ticks, Exception):
                logger.error(f"从Redis获取level5_ticks时出错: {level5_ticks}", exc_info=level5_ticks)
                level5_ticks = [] # 出错时视为空，不影响主流程
            # 5. 将原始数据转换为DataFrame并合并
            df_realtime = pd.DataFrame(
                [data for data, score in realtime_ticks],
                index=pd.to_datetime([datetime.datetime.fromtimestamp(score) for data, score in realtime_ticks])
            )
            if level5_ticks:
                df_level5 = pd.DataFrame(
                    [data for data, score in level5_ticks],
                    index=pd.to_datetime([datetime.datetime.fromtimestamp(score) for data, score in level5_ticks])
                )
                # 使用 merge_asof 进行高效合并
                df_ticks = pd.merge_asof(df_realtime.sort_index(), df_level5.sort_index(), left_index=True, right_index=True, direction='backward')
            else:
                df_ticks = df_realtime
            logger.debug(f"成功从Redis获取并合并了 {len(df_ticks)} 条Tick数据 for {stock_code}")
            return df_ticks
        except Exception as e:
            logger.error(f"在 get_intraday_ticks 中发生异常 for {stock_code}: {e}", exc_info=True)
            return None
    # ▼▼▼ 新增方法: 从缓存读取真实的逐笔成交数据 ▼▼▼
    async def get_daily_real_ticks(self, stock_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        从Redis ZSET缓存中获取指定股票、指定日期的【全部真实逐笔成交数据】。
        """
        try:
            # 1. 统一日期格式
            date_obj = pd.to_datetime(trade_date)
            date_str_yyyymmdd = date_obj.strftime('%Y%m%d')
            # 2. 获取新的缓存键
            cache_key = self.cache_key_stock.intraday_real_ticks(stock_code, date_str_yyyymmdd)
            # 3. 从Redis获取数据
            # 使用 zrangebyscore 获取指定分数（时间戳）范围内的所有成员
            # '-inf', '+inf' 表示获取该键下的所有成员
            ticks_with_scores = await self.cache_manager.zrangebyscore(cache_key, '-inf', '+inf', withscores=True)
            if not ticks_with_scores:
                logger.info(f"缓存未命中或为空: 未找到 {stock_code} on {date_str_yyyymmdd} 的真实逐笔数据。Key: '{cache_key}'")
                return None
            # 4. 将原始数据转换为DataFrame
            deserialized_data = [self.cache_manager._deserialize(data) for data, score in ticks_with_scores]
            df_ticks = pd.DataFrame(deserialized_data)
            if df_ticks.empty:
                return None
            # 使用分数（时间戳）创建索引，这比依赖数据内部的时间字段更可靠
            df_ticks.index = pd.to_datetime([score for data, score in ticks_with_scores], unit='s')
            df_ticks.index.name = 'trade_time'
            # 确保时区正确
            if df_ticks.index.tz is None:
                df_ticks.index = df_ticks.index.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            logger.debug(f"缓存命中: 成功从Redis获取了 {len(df_ticks)} 条真实逐笔数据 for {stock_code}")
            return df_ticks
        except Exception as e:
            logger.error(f"在 get_daily_real_ticks (cache) 中发生异常 for {stock_code}: {e}", exc_info=True)
            return None

class StockIndicatorsCacheGet(CacheGet):
    def __init__(self, cache_manager_instance):
        # 调用父类并传递实例
        super().__init__(cache_manager_instance)
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

class StrategyCacheGet(CacheGet):
    def __init__(self, cache_manager_instance):
        # 调用父类并传递实例
        super().__init__(cache_manager_instance)
    async def lastest_analyze_signals_trend_following_data(self, stock_code: str):
        cache_key = self.cache_key_strategy.analyze_signals_trend_following(stock_code=stock_code)
        # 2. 调用 CacheManager 获取数据
        cached_data = await self.cache_manager.get(key=cache_key)
        if cached_data is not None:
            if isinstance(cached_data, dict):
                # logger.info(f"缓存命中: 成功获取到股票[{stock_code}] 最新策略判断, key: {cache_key}")
                return cached_data
            else:
                logger.warning(f"缓存数据格式错误: 股票[{stock_code}] 最新策略判断的缓存值不是字典类型 (实际类型: {type(cached_data)}), key: {cache_key}. 将视为未命中。")
                self.cache_manager.delete(cache_key) # 可选：删除错误数据
                return None
        else:
            logger.info(f"缓存未命中: 未找到股票[{stock_code}] 最新策略判断, key: {cache_key}")
            return None
    async def all_analyze_signals_trend_following_data(self):
        """
        获取全部股票的最新趋势跟踪策略数据（从Redis缓存）。
        返回: dict，key为stock_code，value为策略数据
        """
        # 构造key的前缀
        key_prefix = "strategy:stock:"
        pattern = f"{key_prefix}*:trend_following"
        # 获取所有匹配的key
        keys = await self.cache_manager.scan_keys(pattern)
        print(f"all_analyze_signals_trend_following_data: 匹配到{len(keys)}个key")
        result = {}
        for key in keys:
            # 提取stock_code
            # key格式: strategy:stock:{stock_code}:trend_following
            try:
                stock_code = key.split(":")[2]
            except Exception as e:
                print(f"解析key出错: {key}, 错误: {e}")
                continue
            data = await self.cache_manager.get(key)
            if data is not None:
                result[stock_code] = data
        return result
    async def analyze_signals_trend_following_datas(self, stock_code: str, days_count: int = 1) -> Optional[Dict[str, Any]]:
        cache_key = self.cache_key_strategy.analyze_signals_trend_following(stock_code=stock_code)
        return await self._stock_strategy_datas(stock_code=stock_code, cache_key=cache_key, days_count=days_count)
    async def analyze_signals_trend_following_datas_by_timestamp(self, stock_code: str, timestamp: int) -> Optional[List[Any]]:
        cache_key = self.cache_key_strategy.analyze_signals_trend_following(stock_code=stock_code)
        try:
            members = await self.cache_manager.zrangebyscore(
                key=cache_key,
                min_score=timestamp,
                max_score=timestamp,
                withscores=False
            )
            return members if members else None
        except Exception as e:
            logger.error(f"调用 zrangebyscore 失败: {e}")
            return None
    







