from datetime import date
import logging
import pandas as pd

from api_manager.apis.stock_indicators_api import StockIndicatorsAPI
from dao_manager.base_dao import BaseDAO
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.user_dao import UserDAO
from stock_models.stock_basic import StockInfo
from stock_models.stock_realtime import StockRealtimeData
from utils.cache_get import StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockTimeTradeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockIndicatorsDataFormatProcess, StockTimeTradeFormatTuShare

logger = logging.getLogger("dao")

class StockTimeTradeDAO(BaseDAO):
    def __init__(self):
        """初始化StockIndicatorsDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockIndicatorsAPI()
        self.stock_basic_dao = StockBasicDAO()
        self.cache_timeout = 300  # 默认缓存5分钟
        self.cache_limit = 450 # 定义缓存数量上限
        self.user_dao = UserDAO()
        self.cache_key = StockCashKey()
        self.data_format_tushare = StockTimeTradeFormatTuShare()
        self.cache_manager = None
        self.cache_get = None
        self.cache_set = None

    async def initialize_cache_objects(self):
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 其异步初始化方法，如果存在

        self.cache_set = StockTimeTradeCacheSet()  # 先实例化
        await self.cache_set.initialize()  # 添加异步初始化方法，如果需要

        self.cache_get = StockTimeTradeCacheGet()  # 先实例化
        await self.cache_get.initialize()  # 添加异步初始化方法，如果需要


    async def save_history_daily_time_deals(self, stock_code: str, trade_date: date) -> None:
        """
        保存股票的历史日线交易数据
        接口：daily，可以通过数据工具调试和查看数据
        数据说明：交易日每天15点～16点之间入库。本接口是未复权行情，停牌期间不提供数据
        调取说明：120积分每分钟内最多调取500次，每次6000条数据，相当于单次提取23年历史
        描述：获取股票行情数据，或通过通用行情接口获取数据，包含了前后复权数据
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        df = self.ts_pro.daily(**{ "ts_code": "", "trade_date": trade_date.strftime("%Y%m%d"), "start_date": "",
                                  "end_date": "", "offset": "", "limit": "" }, 
            fields=[ "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"])
        if df is not None:
            data_dicts = []
            for row in df.itertuples():
                stock = self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_tushare.set_time_trade_data(stock, "daily", row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = cache_data_dict['stock'].stock_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.latest_realtime_data(stock_code, prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            # 使用包含 StockInfo 实例的列表
            result = await self._save_all_to_db_native_upsert(
                model_class=StockRealtimeData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def save_realtime_daily_time_deals(self, stock_code: str, trade_date: date) -> None:
        """
        保存股票的实时日线交易数据
        接口：rt_k
        描述：获取实时日k线行情，支持按股票代码及股票代码通配符一次性提取全部股票实时日k线行情
        限量：单次最大可提取6000条数据
        积分：本接口是单独开权限的数据，单独申请权限请参考权限列表
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        # 拉取数据
        df = self.ts_pro.rt_k(**{
            "topic": "", "ts_code": stock_code, "limit": "", "offset": ""
        }, fields=[
            "ts_code", "name", "pre_close", "high", "open", "low", "close", "vol", "amount", "num"
        ])
        if df is not None:
            data_dicts = []
            for row in df.itertuples():
                stock = self.stock_basic_dao.get_stock_by_code(row.ts_code)
                data_dict = self.data_format_tushare.set_time_trade_data(stock, "daily", row)
                # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                data_dicts.append(data_dict)
                # 2. 准备缓存数据
                cache_data_dict = data_dict.copy()
                if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                    # 替换为 stock_code
                    cache_data_dict['stock_code'] = cache_data_dict['stock'].stock_code
                    del cache_data_dict['stock'] # 删除实例键
                prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                if prepared_data:
                    await self.cache_set.latest_realtime_data(stock_code, prepared_data)
                else:
                    logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            # 使用包含 StockInfo 实例的列表
            result = await self._save_all_to_db_native_upsert(
                model_class=StockRealtimeData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def save_realtime_min_time_deals(self, stock_code: str, time_level: str) -> None:
        """
        保存股票的实时分钟级交易数据
        接口：rt_min
        描述：获取全A股票实时分钟数据，包括1~60min
        限量：单次最大1000行数据，可以通过股票代码提取数据，支持逗号分隔的多个代码同时提取
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        # 拉取数据
        df = self.ts_pro.rt_min(**{
            "topic": "", "freq": time_level + "min", "ts_code": stock_code, "limit": "", "offset": ""
        }, fields=[
            "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
        ])
        if df is not None:
            data_dicts = []
            for row in df.itertuples():
                stock = self.stock_basic_dao.get_stock_by_code(row.ts_code)
                data_dict = self.data_format_tushare.set_time_trade_data(stock, time_level, row)
                # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                data_dicts.append(data_dict)
                # 2. 准备缓存数据
                cache_data_dict = data_dict.copy()
                if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                    # 替换为 stock_code
                    cache_data_dict['stock_code'] = row.ts_code
                    del cache_data_dict['stock'] # 删除实例键
                prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                if prepared_data:
                    await self.cache_set.latest_realtime_data(stock_code, prepared_data)
                else:
                    logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock.stock_code, time_level)
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            result = await self._save_all_to_db_native_upsert(
                model_class=StockRealtimeData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def save_history_min_time_deals(self, stock_code: str, time_level: str, trade_date: date) -> None:
        """
        保存股票的分钟级交易数据
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        # 拉取数据
        df = self.ts_pro.stk_mins(**{
            "ts_code": stock_code, "freq": time_level + "min", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"
        ])
        if df is not None:
            data_dicts = []
            for row in df.itertuples():
                stock = self.stock_basic_dao.get_stock_by_code(row.ts_code)
                data_dict = self.data_format_tushare.set_time_trade_data(stock, time_level, row)
                # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                data_dicts.append(data_dict)
                # 2. 准备缓存数据
                cache_data_dict = data_dict.copy()
                if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                    # 替换为 stock_code
                    cache_data_dict['stock_code'] = row.ts_code
                    del cache_data_dict['stock'] # 删除实例键
                prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                if prepared_data:
                    await self.cache_set.history_time_trade(stock_code, prepared_data)
                else:
                    logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock.stock_code, time_level)
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            result = await self._save_all_to_db_native_upsert(
                model_class=StockRealtimeData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

























