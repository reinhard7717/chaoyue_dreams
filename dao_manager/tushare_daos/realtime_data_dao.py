import asyncio
import logging
import pandas as pd
from typing import List, Optional
import tushare as ts
from datetime import datetime
from chaoyue_dreams import settings
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_realtime import StockLevel5Data, StockRealtimeData
from utils.cache_get import StockInfoCacheGet, StockRealtimeCacheGet
from utils.cache_set import StockRealtimeCacheSet
from utils.data_format_process import StockRealtimeDataFormatProcess

logger = logging.getLogger("dao")

class StockRealtimeDAO(BaseDAO):
    """
    股票实时数据DAO，整合所有相关的实时数据访问功能
    """
    def __init__(self):
        """初始化StockRealtimeDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.stock_basic_dao = StockBasicInfoDao()
        self.data_format_process = StockRealtimeDataFormatProcess()
        self.cache_set = StockRealtimeCacheSet()  # 先实例化
        self.cache_get = StockRealtimeCacheGet()  # 先实例化
        self.stock_cache_get = StockInfoCacheGet()
        self.ts = ts.set_token(settings.API_LICENCES_TUSHARE)

    # ================= 实时盘口TICK快照(爬虫版) =================
    # 获取所有股票的实时盘口TICK快照数据并保存到数据库
    async def save_all_tick_data(self) -> Optional[StockRealtimeData]:
        """
        通过tushare获取实时盘口TICK快照数据并保存到数据库
        """
        ts.set_token('0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c')
        stocks = await self.stock_cache_get.all_stocks()
        # 先处理带后缀的stock_code
        stock_codes_list = [stock.stock_code for stock in stocks]
        # 每50个拼接成一个字符串
        grouped_stock_codes = [
            ','.join(stock_codes_list[i:i+50])
            for i in range(0, len(stock_codes_list), 50)
        ]
        for stock_codes in grouped_stock_codes:
            real_data_dicts = []
            level5_data_dicts = []
            # sina数据
            df = ts.realtime_quote(ts_code=stock_codes)
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    real_dict = self.data_format_process.set_realtime_tick_data(stock, row)
                    level5_dict = self.data_format_process.set_level5_data(stock, row)
                    await self.cache_set.latest_realtime_data(stock.stock_code, real_dict)
                    real_data_dicts.append(real_dict)
                    level5_data_dicts.append(level5_dict)
            # 保存数据
            result = await self._save_all_to_db_native_upsert(
                model_class=StockRealtimeData,
                data_list=real_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            result = await self._save_all_to_db_native_upsert(
                model_class=StockLevel5Data,
                data_list=level5_data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    # 根据传入的股票代码列表，获取实时盘口TICK快照数据并保存到数据库
    async def save_tick_data_by_stock_codes(self, stock_codes: List[str]) -> List:
        """
        【V3.0 - 双轨持久化版】
        只调用一次API，然后将Tick数据并发地写入：
        1. 数据库 (PostgreSQL): 用于长期历史归档。
        2. Redis SET: 存储最新的Tick快照，用于实时展示。
        3. Redis ZSET: 追加存储当日的Tick时间序列，用于盘中引擎聚合计算。
        """
        if not stock_codes:
            return []
        
        try:
            # 1. 数据采集：只调用一次API
            stock_codes_str = ','.join(stock_codes)
            df = self.ts.realtime_quote(ts_code=stock_codes_str)
            if df.empty:
                logger.warning(f"Tushare未返回股票 {stock_codes_str} 的实时行情数据。")
                return []

            # 2. 数据预处理与载荷准备
            stocks_dict = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
            
            db_realtime_list = []       # 轨道1: 数据库
            db_level5_list = []
            
            cache_latest_realtime = {}  # 轨道2: Redis SET (最新快照)
            cache_latest_level5 = {}
            
            cache_append_realtime = {}  # 轨道3: Redis ZSET (当日时间序列)
            cache_append_level5 = {}

            for row in df.itertuples():
                stock = stocks_dict.get(row.TS_CODE)
                if stock:
                    # 准备数据库数据
                    real_dict_db = self.data_format_process.set_realtime_tick_data(stock, row)
                    level5_dict_db = self.data_format_process.set_level5_data(stock, row)
                    db_realtime_list.append(real_dict_db)
                    db_level5_list.append(level5_dict_db)
                    
                    # 准备缓存数据 (注意：缓存数据不需要stock对象)
                    real_dict_cache = self.data_format_process.set_realtime_tick_data(None, row)
                    level5_dict_cache = self.data_format_process.set_level5_data(None, row)
                    
                    cache_latest_realtime[row.TS_CODE] = real_dict_cache
                    cache_latest_level5[row.TS_CODE] = level5_dict_cache
                    
                    cache_append_realtime[row.TS_CODE] = real_dict_cache
                    cache_append_level5[row.TS_CODE] = level5_dict_cache

            if not db_realtime_list:
                return []

            # 3. 并发执行所有持久化任务
            tasks = [
                # 轨道1: 写入数据库
                self._save_all_to_db_native_upsert(StockRealtimeData, db_realtime_list, ['stock', 'trade_time']),
                self._save_all_to_db_native_upsert(StockLevel5Data, db_level5_list, ['stock', 'trade_time']),
                
                # 轨道2: 写入Redis SET (最新快照)
                self.cache_set.batch_set_latest_realtime_data(cache_latest_realtime),
                self.cache_set.batch_set_latest_level5_data(cache_latest_level5),
                
                # 轨道3: 写入Redis ZSET (当日时间序列)
                self.cache_set.batch_append_intraday_ticks(cache_append_realtime, cache_append_level5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 检查并记录异常
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"并发持久化任务 {i} 执行失败: {result}", exc_info=result)

            return results[0] if not isinstance(results[0], Exception) else []

        except Exception as e:
            logger.error(f"save_tick_data_by_stock_codes 发生严重异常: {e}", exc_info=True)
            return []

    async def get_daily_ticks_from_cache(self, stock_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        【V3.0】从Redis ZSET缓存中获取指定股票、指定日期的【全部】Tick快照数据。
        """
        try:
            await self.cache_manager.initialize() # 确保客户端已初始化
            today_str = pd.to_datetime(trade_date).strftime('%Y%m%d')

            # 1. 定义当天的缓存键
            realtime_key = self.cache_key_stock.intraday_ticks_realtime(stock_code, today_str)
            level5_key = self.cache_key_stock.intraday_ticks_level5(stock_code, today_str)

            # 2. 并发地从Redis获取两种Tick数据
            tasks = [
                self.cache_manager.zrangebyscore(realtime_key, '-inf', '+inf', withscores=True),
                self.cache_manager.zrangebyscore(level5_key, '-inf', '+inf', withscores=True)
            ]
            realtime_ticks, level5_ticks = await asyncio.gather(*tasks)

            if not realtime_ticks:
                logger.warning(f"未能从缓存获取 {stock_code} on {today_str} 的实时行情Ticks。")
                return None

            # 3. 将原始数据转换为DataFrame
            df_realtime = pd.DataFrame(
                [data for data, score in realtime_ticks],
                index=pd.to_datetime([datetime.fromtimestamp(score) for data, score in realtime_ticks])
            )
            
            df_level5 = None
            if level5_ticks:
                df_level5 = pd.DataFrame(
                    [data for data, score in level5_ticks],
                    index=pd.to_datetime([datetime.fromtimestamp(score) for data, score in level5_ticks])
                )

            # 4. 合并两种Tick数据
            if df_level5 is not None and not df_level5.empty:
                df_ticks = pd.merge_asof(df_realtime.sort_index(), df_level5.sort_index(), left_index=True, right_index=True, direction='backward')
            else:
                df_ticks = df_realtime

            logger.debug(f"成功从Redis获取并合并了 {len(df_ticks)} 条Tick数据 for {stock_code}")
            return df_ticks

        except Exception as e:
            logger.error(f"获取每日Ticks数据时发生异常 for {stock_code}: {e}", exc_info=True)
            return None

    async def get_latest_tick_data(self, stock_code: str) -> dict:
        """
        获取最新价格
        """
        # 从Redis缓存中获取数据
        data_dict = await self.cache_get.latest_tick_data(stock_code)
        if data_dict:
            change_percent = (data_dict.get('current_price') - data_dict.get('prev_close_price')) / data_dict.get('prev_close_price') * 100
            change_percent = round(change_percent, 2)  # 保留两位小数
            data_dict['change_percent'] = change_percent  # 计算涨跌幅（change_percent），并加入data_dict
            volume = data_dict.get('volume')
            volume = round(volume / 100, 2)
            data_dict['volume'] = volume
            return data_dict
        else:
            return None

























