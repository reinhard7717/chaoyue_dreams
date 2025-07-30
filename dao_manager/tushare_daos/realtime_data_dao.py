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
from utils.cache_manager import CacheManager
from utils.cache_set import StockRealtimeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockRealtimeDataFormatProcess

logger = logging.getLogger("dao")

class StockRealtimeDAO(BaseDAO):
    """
    股票实时数据DAO，整合所有相关的实时数据访问功能
    """
    def __init__(self, cache_manager_instance: CacheManager):
        # MODIFIED: 调用父类构造函数时，传递 cache_manager_instance 和 model_class=None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
        self.data_format_process = StockRealtimeDataFormatProcess(cache_manager_instance)
        self.cache_set = StockRealtimeCacheSet(self.cache_manager)
        self.cache_get = StockRealtimeCacheGet(self.cache_manager)
        self.stock_cache_get = StockInfoCacheGet(self.cache_manager)
        self.cache_key_stock = StockCashKey()
        self.ts = ts

    # ================= 实时盘口TICK快照(爬虫版) =================
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
        【V3.1 - 增强调试版】
        增加调试打印，以跟踪从API到缓存写入前的数据流。
        """
        if not stock_codes:
            return []
        
        try:
            # 1. 数据采集
            stock_codes_str = ','.join(stock_codes)
            df = self.ts.realtime_quote(ts_code=stock_codes_str)
            
            # MODIFIED: 添加调试打印，显示从Tushare获取的原始DataFrame
            print(f"DEBUG: Tushare realtime_quote DF for codes '{stock_codes_str[:100]}...':\n{df.head().to_string()}")

            if df.empty:
                logger.warning(f"Tushare未返回股票 {stock_codes_str} 的实时行情数据。")
                return []

            # 2. 数据预处理与载荷准备
            stocks_dict = await self.stock_basic_dao.get_stocks_by_codes(stock_codes)
            
            db_realtime_list = []
            db_level5_list = []
            cache_latest_realtime = {}
            cache_latest_level5 = {}
            cache_append_realtime = {}
            cache_append_level5 = {}

            for row in df.itertuples():
                stock = stocks_dict.get(row.TS_CODE)
                if stock:
                    real_dict_db = self.data_format_process.set_realtime_tick_data(stock, row)
                    level5_dict_db = self.data_format_process.set_level5_data(stock, row)
                    db_realtime_list.append(real_dict_db)
                    db_level5_list.append(level5_dict_db)
                    
                    real_dict_cache = self.data_format_process.set_realtime_tick_data(None, row)
                    level5_dict_cache = self.data_format_process.set_level5_data(None, row)
                    
                    cache_latest_realtime[row.TS_CODE] = real_dict_cache
                    cache_latest_level5[row.TS_CODE] = level5_dict_cache
                    
                    cache_append_realtime[row.TS_CODE] = real_dict_cache
                    cache_append_level5[row.TS_CODE] = level5_dict_cache

            if not db_realtime_list:
                return []

            # MODIFIED: 添加调试打印，显示准备写入Redis ZSET的载荷内容
            print(f"DEBUG: Prepared realtime ZSET payload (first 2 items): {dict(list(cache_append_realtime.items())[:2])}")
            print(f"DEBUG: Prepared level5 ZSET payload (first 2 items): {dict(list(cache_append_level5.items())[:2])}")

            # 3. 并发执行所有持久化任务
            tasks = [
                self._save_all_to_db_native_upsert(StockRealtimeData, db_realtime_list, ['stock', 'trade_time']),
                self._save_all_to_db_native_upsert(StockLevel5Data, db_level5_list, ['stock', 'trade_time']),
                self.cache_set.batch_set_latest_realtime_data(cache_latest_realtime),
                self.cache_set.batch_set_latest_level5_data(cache_latest_level5),
                self.cache_set.batch_append_intraday_ticks(cache_append_realtime, cache_append_level5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"并发持久化任务 {i} 执行失败: {result}", exc_info=result)

            return results[0] if not isinstance(results[0], Exception) else []

        except Exception as e:
            logger.error(f"save_tick_data_by_stock_codes 发生严重异常: {e}", exc_info=True)
            return []

    async def get_daily_ticks_from_cache(self, stock_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        【V3.2 - 重构版】从缓存中获取指定股票、指定日期的【全部】Tick快照数据。
        此方法现在是 StockRealtimeCacheGet.get_intraday_ticks 的一个简单代理，
        保持了DAO层接口的稳定性，同时将实现细节移交给了缓存层。
        """
        # 直接调用重构后的缓存获取方法
        return await self.cache_get.get_intraday_ticks(stock_code, trade_date)

    async def get_latest_tick_data(self, stock_code: str) -> dict:
        """
        获取最新价格
        """
        data_dict = await self.cache_get.latest_tick_data(stock_code)
        if data_dict:
            # 增加健壮性检查，防止因数据缺失导致计算错误
            current_price = data_dict.get('current_price')
            prev_close_price = data_dict.get('prev_close_price')
            if current_price is not None and prev_close_price is not None and prev_close_price != 0:
                change_percent = (current_price - prev_close_price) / prev_close_price * 100
                data_dict['change_percent'] = round(change_percent, 2)
            else:
                data_dict['change_percent'] = 0.0

            volume = data_dict.get('volume')
            if volume is not None:
                data_dict['volume'] = round(volume / 100, 2)
            
            return data_dict
        else:
            return None























