import logging
from typing import List, Optional
import tushare as ts
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.basic_info_dao import BasicInfoDao
from stock_models.stock_realtime import StockLevel5Data, StockRealtimeData
from utils.cache_get import StockRealtimeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockRealtimeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockRealtimeDataFormatTuShare

logger = logging.getLogger("dao")

class StockRealtimeDAO(BaseDAO):
    """
    股票实时数据DAO，整合所有相关的实时数据访问功能
    """
    
    def __init__(self):
        """初始化StockRealtimeDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.stock_basic_dao = BasicInfoDao()
        self.data_format_tushare = StockRealtimeDataFormatTuShare()
        self.cache_manager = None  # 初始化缓存管理器
        self.cache_get = None
        self.cache_set = None
        self.cache_key = StockCashKey()

    async def initialize_cache_objects(self):
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 其异步初始化方法，如果存在

        self.cache_set = StockRealtimeCacheSet()  # 先实例化
        await self.cache_set.initialize()  # 添加异步初始化方法，如果需要

        self.cache_get = StockRealtimeCacheGet()  # 先实例化
        await self.cache_get.initialize()  # 添加异步初始化方法，如果需要

    # ================= 实时盘口TICK快照(爬虫版) =================
    # 获取所有股票的实时盘口TICK快照数据并保存到数据库
    async def tushare_save_all_tick_data(self) -> Optional[StockRealtimeData]:
        """
        通过tushare获取实时盘口TICK快照数据并保存到数据库
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        ts.set_token('0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c')
        stocks = await self.stock_basic_dao.get_stock_list()
        # 先处理带后缀的stock_code
        stock_codes_list = [
            f"{stock.stock_code}.SH" if stock.stock_code.startswith('6') else f"{stock.stock_code}.SZ"
            for stock in stocks
        ]
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
                db_stock_code = row.ts_code.split('.')[0]
                stock = await self.stock_basic_dao.get_stock_by_code(db_stock_code)
                real_dict = self.data_format_tushare.set_realtime_data(stock, row)
                level5_dict = self.data_format_tushare.set_level5_data(stock, row)
                cache_key = f"stock_realtime_data_{stock.stock_code}"
                await self.cache_set.latest_realtime_data(cache_key, real_dict)
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
            # dfs.append(df)
        # 合并所有DataFrame
        # all_df = pd.concat(dfs, ignore_index=True)
        # print(all_df)

    # 根据传入的股票代码列表，获取实时盘口TICK快照数据并保存到数据库
    async def tushare_save_tick_data_by_stock_codes(self, stock_codes: List[str]) -> Optional[StockRealtimeData]:
        """
        通过tushare获取实时盘口TICK快照数据并保存到数据库
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        ts.set_token('0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c')
        stock_codes_str = ','.join(stock_codes)
        # sina数据
        df = ts.realtime_quote(ts_code=stock_codes_str)
        real_data_dicts = []
        level5_data_dicts = []
        for row in df.itertuples():
            real_dict = self.data_format_tushare.set_realtime_data(row)
            level5_dict = self.data_format_tushare.set_level5_data(row)
            real_data_dicts.append(real_dict)
            await self.cache_set.latest_realtime_data(row.ts_code, real_dict)
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

    # ================= 分钟级实时数据 =================
    # 保存股票的分钟级实时数据
    async def save_realtime_min_data(self, stock_code: str, time_level: str) -> None:
        """
        保存股票的分钟级实时数据
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        # 拉取数据
        df = self.ts_pro.rt_min(**{
            "topic": "", "freq": time_level + "MIN", "ts_code": stock_code, "limit": "", "offset": ""
        }, fields=[
            "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
        ])
        real_data_dicts = []
        level5_data_dicts = []
        if df is not None:
            for row in df.itertuples():
                real_dict = self.data_format_tushare.set_realtime_data(stock_code, time_level, row)
                level5_dict = self.data_format_tushare.set_level5_data(row)
                real_data_dicts.append(real_dict)
                await self.cache_set.latest_realtime_data(row.ts_code, real_dict)
                level5_data_dicts.append(level5_dict)
            await self._save_all_to_db_native_upsert(
                model_class=StockRealtimeData,
                data_list=real_data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
            await self._save_all_to_db_native_upsert(
                model_class=StockLevel5Data,
                data_list=level5_data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def save_realtime_min_data_by_stock_codes(self, stock_codes: List[str], time_level: str) -> None:
        """
        根据传入的股票代码列表，获取分钟级实时数据并保存到数据库
        接口：rt_min
        描述：获取全A股票实时分钟数据，包括1~60min
        限量：单次最大1000行数据，可以通过股票代码提取数据，支持逗号分隔的多个代码同时提取
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        stock_codes_str = ','.join(stock_codes)
        df = self.ts_pro.rt_min(**{
            "topic": "", "freq": time_level + "MIN", "ts_code": stock_codes_str, "limit": "", "offset": ""
        }, fields=[
            "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
        ])
        data_dicts = []
        if df is not None:
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                data_dict = self.data_format_tushare.set_realtime_data(row)
                data_dicts.append(data_dict)
                cache_data = data_dict.copy()
        return data_dicts





























