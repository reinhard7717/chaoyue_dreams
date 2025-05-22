import logging
from typing import List, Optional
import tushare as ts
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
    async def save_tick_data_by_stock_codes(self, stock_codes: List[str]) -> Optional[StockRealtimeData]:
        """
        通过tushare获取实时盘口TICK快照数据并保存到数据库
        """
        ts.set_token('0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c')
        stock_codes_str = ','.join(stock_codes)
        # sina数据
        print(stock_codes_str)
        df = ts.realtime_quote(ts_code=stock_codes_str)
        real_data_dicts = []
        level5_data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_basic_dao.get_stock_by_code(row.TS_CODE)
            if stock:
                real_dict = self.data_format_process.set_realtime_tick_data(stock, row)
                level5_dict = self.data_format_process.set_level5_data(stock, row)
                print(f"real_dict: {real_dict}")
                real_data_dicts.append(real_dict)
                await self.cache_set.latest_realtime_data(row.TS_CODE, real_dict)
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

    # ================= 实时成交快照(爬虫版) =================
    async def save_all_time_trade_data(self) -> None:
        """
        通过tushare获取所有股票的实时成交快照数据并保存到数据库
        """
        # 获取所有股票代码
        stocks = await self.stock_basic_dao.get_stock_list()
        pass

























