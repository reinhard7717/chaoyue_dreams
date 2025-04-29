from datetime import date
import logging
from typing import List
import numpy as np
import pandas as pd
from math import ceil
from datetime import datetime
from api_manager.apis.stock_indicators_api import StockIndicatorsAPI
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockCyqChips, StockCyqPerf, StockDailyBasic, StockDailyData, StockMinuteData, StockWeeklyData, StockMonthlyData
from utils.cache_get import StockInfoCacheGet, StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockInfoCacheSet, StockTimeTradeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockInfoFormatProcess, StockTimeTradeFormatProcess

logger = logging.getLogger("dao")
time_levels = ["1", "5", "15", "30", "60"]

class StockTimeTradeDAO(BaseDAO):
    def __init__(self):
        """初始化StockIndicatorsDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockIndicatorsAPI()
        self.stock_basic_dao = StockBasicInfoDao()
        self.cache_limit = 333 # 定义缓存数量上限
        self.cache_manager = CacheManager()
        self.cache_key = StockCashKey()
        self.data_format_process_trade = StockTimeTradeFormatProcess()
        self.data_format_process_stock = StockInfoFormatProcess()
        self.cache_set = StockTimeTradeCacheSet()
        self.cache_get = StockTimeTradeCacheGet()
        self.stock_cache_set = StockInfoCacheSet()
        self.stock_cache_get = StockInfoCacheGet()

    # =============== A股日线行情 ===============
    async def save_daily_time_trade_history_by_trade_date(self, trade_date: date) -> None:
        """
        保存股票的历史日线交易数据
        接口：stk_factor
        描述：获取股票每日技术面因子数据，用于跟踪股票当前走势情况，数据由Tushare社区自产，覆盖全历史
        限量：单次最大10000条，可以循环或者分页提取
        积分：5000积分每分钟可以请求100次，8000积分以上每分钟500次，具体请参阅积分获取办法
        """
        df = self.ts_pro.stk_factor(**{ "ts_code": "", "trade_date": trade_date.strftime("%Y%m%d"), "start_date": "",
                                  "end_date": "", "offset": "", "limit": "" }, 
            fields=[ "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol", "amount", "adj_factor",
                    "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq", "low_qfq", "pre_close_hfq", "pre_close_qfq",])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = stock.stock_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.history_time_trade(stock.stock_code, "Day", prepared_data)
                        # --- 函数末尾执行最终修剪 ---
                        cache_key =  self.cache_key.history_time_trade(stock.stock_code, "Day")
                        await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                        # --- 修剪调用结束 ---
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            # 使用包含 StockInfo 实例的列表
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_daily_time_trade_history_by_stock_code(self, stock_code: str) -> None:
        """
        保存股票的历史日线交易数据
        接口：stk_factor
        描述：获取股票每日技术面因子数据，用于跟踪股票当前走势情况，数据由Tushare社区自产，覆盖全历史
        限量：单次最大10000条，可以循环或者分页提取
        积分：5000积分每分钟可以请求100次，8000积分以上每分钟500次，具体请参阅积分获取办法
        """
        df = self.ts_pro.stk_factor(**{ "ts_code": stock_code, "trade_date": "", "start_date": "","end_date": "", "offset": "", "limit": "" }, 
            fields=[ "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol", "amount", "adj_factor",
                    "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq", "low_qfq", "pre_close_hfq", "pre_close_qfq",])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = stock.stock_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.history_time_trade(stock.stock_code, "Day", prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            # 使用包含 StockInfo 实例的列表
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
            # --- 函数末尾执行最终修剪 ---
            cache_key =  self.cache_key.history_time_trade(stock_code, "Day")
            await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
            # --- 修剪调用结束 ---
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_daily_time_trade_history_by_stock_codes(self, stock_codes: List[str]) -> None:
        """
        保存股票的历史日线交易数据
        接口：stk_factor
        描述：获取股票每日技术面因子数据，用于跟踪股票当前走势情况，数据由Tushare社区自产，覆盖全历史
        限量：单次最大10000条，可以循环或者分页提取
        积分：5000积分每分钟可以请求100次，8000积分以上每分钟500次，具体请参阅积分获取办法
        """
        stock_codes_str = ",".join(stock_codes)
        df = self.ts_pro.stk_factor(**{ "ts_code": stock_codes_str, "trade_date": "", "start_date": "","end_date": "", "offset": "", "limit": "" }, 
            fields=[ "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol", "amount", "adj_factor",
                    "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq", "low_qfq", "pre_close_hfq", "pre_close_qfq",])
        logger.info(f"开始执行{len(stock_codes)}个股票的日线数据保存. 获取df数量: {len(df)}，stock_codes: {stock_codes_str}")
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    # logger.info(f"trade_date: {row.trade_date} - {self._parse_datetime(row.trade_date)}")
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = stock.stock_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.history_time_trade(stock.stock_code, "Day", prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            # 使用包含 StockInfo 实例的列表
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
            for stock_code in stock_codes:
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock_code, "Day")
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_daily_time_trade_today(self) -> None:
        """
        保存股票的今日日线交易数据
        接口：daily
        数据说明：交易日每天15点～16点之间入库。本接口是未复权行情，停牌期间不提供数据
        调取说明：120积分每分钟内最多调取500次，每次6000条数据，相当于单次提取23年历史
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        result = await self.save_daily_time_trade_history_by_trade_date(today_str)
        return result

    # 未复权信息，慎用
    async def save_daily_time_trade_realtime(self, stock_code: str) -> None:
        """
        未复权信息，慎用
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
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
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
                        await self.cache_set.latest_time_trade(stock_code, "Day", prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            # 使用包含 StockInfo 实例的列表
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def get_daily_time_trade_history_by_stock_code(self, stock_code: str) -> None:
        """
        获取股票的历史日线交易数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.history_time_trade(stock_code, "Day")
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        stock_daily_data_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_daily_data_list.append(StockDailyData(**data_dict))
            return stock_daily_data_list
        # 从数据库中获取数据
        stock_daily_data_list = StockDailyData.objects.filter(stock_code=stock_code).order_by('-trade_date')[:self.cache_limit]
        return stock_daily_data_list

    # =============== A股分钟行情 ===============
    async def save_minute_time_trade_history_by_time_level(self, stock_code: str, time_level: str) -> None:
        """
        保存股票的历史分钟级交易数据
        接口：stk_mins
        描述：获取A股分钟数据，支持1min/5min/15min/30min/60min行情，提供Python SDK和 http Restful API两种方式
        限量：单次最大8000行数据，可以通过股票代码和时间循环获取，本接口可以提供超过10年历史分钟数据
        """
        df = self.ts_pro.stk_mins(**{
                "ts_code": stock_code, "freq": time_level + "min", "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"
            ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(stock, StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = row.ts_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.history_time_trade(stock_code, time_level, prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMinuteData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_minute_time_trade_history_by_stock_code(self, stock_code: str) -> None:
        """
        保存股票的历史分钟级交易数据
        接口：stk_mins
        描述：获取A股分钟数据，支持1min/5min/15min/30min/60min行情，提供Python SDK和 http Restful API两种方式
        限量：单次最大8000行数据，可以通过股票代码和时间循环获取，本接口可以提供超过10年历史分钟数据
        """
        result_df = pd.DataFrame()
        for time_level in time_level:
            df = self.ts_pro.stk_mins(**{
                "ts_code": stock_code, "freq": time_level + "min", "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"
            ])
            if df is not None:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                result_df = pd.concat([result_df, df])
        if result_df is not None:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(stock, StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = row.ts_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.history_time_trade(stock_code, time_level, prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMinuteData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_minute_time_trade_history_by_stock_codes(self, stock_codes: List[str]) -> None:
        """
        保存股票的历史分钟级交易数据
        接口：stk_mins
        描述：获取A股分钟数据，支持1min/5min/15min/30min/60min行情，提供Python SDK和 http Restful API两种方式
        限量：单次最大8000行数据，可以通过股票代码和时间循环获取，本接口可以提供超过10年历史分钟数据
        """
        stock_codes_str = ",".join(stock_codes)
        data_dicts = []
        for time_level in time_levels:
            df = self.ts_pro.stk_mins(**{
                "ts_code": stock_codes_str, "freq": time_level + "min", "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"
            ])
            if df is not None:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(stock, StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = row.ts_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.history_time_trade(row.ts_code, time_level, prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            for stock_code in stock_codes:
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
        result = await self._save_all_to_db_native_upsert(
            model_class=StockMinuteData,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
        )
        logger.info(f"保存股票 {stock_codes_str} 的分钟级交易数据完成. 结果: {result}")
        return result

    async def save_minute_time_trade_history_by_stock_codes_and_time_level(self, stock_codes: List[str], time_level: str) -> None:
        """
        保存股票的历史分钟级交易数据
        接口：stk_mins
        描述：获取A股分钟数据，支持1min/5min/15min/30min/60min行情，提供Python SDK和 http Restful API两种方式
        限量：单次最大8000行数据，可以通过股票代码和时间循环获取，本接口可以提供超过10年历史分钟数据
        """
        result_df = pd.DataFrame()
        stock_codes_str = ",".join(stock_codes)
        for time_level in time_level:
            df = self.ts_pro.stk_mins(**{
                "ts_code": stock_codes_str, "freq": time_level + "min", "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"
            ])
            if df is not None:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                result_df = pd.concat([result_df, df])
        if result_df is not None:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
                    # 2. 准备缓存数据
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(stock, StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = row.ts_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.history_time_trade(row.ts_code, time_level, prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMinuteData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
            for stock_code in stock_codes:
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            logger.info(f"保存股票 {stock_codes_str} 的分钟级交易数据完成. 结果: {result}")
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_minute_time_trade_history_all(self) -> None:
        """
        保存股票的历史分钟级交易数据
        """
        stocks = self.stock_basic_dao.get_stock_list()
        data_dicts = []
        for stock in stocks:
            for time_level in time_levels:
                df = self.ts_pro.stk_mins(**{
                    "ts_code": stock.stock_code, "freq": time_level + "min", "start_date": "", "end_date": "", "limit": "", "offset": ""
                    }, fields=["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"])
                if df is not None and not df.empty:
                    df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                    df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                    for row in df.itertuples():
                        data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
                        # 2. 准备缓存数据
                        cache_data_dict = data_dict.copy()
                        if 'stock' in cache_data_dict and isinstance(cache_data_dict['stock'], StockInfo):
                            cache_data_dict['stock_code'] = row.ts_code
                            del cache_data_dict['stock']
                        prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                        if prepared_data:
                            await self.cache_set.history_time_trade(row.ts_code, time_level, prepared_data)
                        else:
                            logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
        if data_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMinuteData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            return result
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

    async def save_minute_time_trade_history_today(self, trade_time_str=None) -> None:
        """
        保存股票的历史分钟级交易数据，分批每1000个stock_code循环一次
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        stock_codes = [stock.stock_code for stock in stocks]
        # 获取当前日期
        if trade_time_str:
            today_str = trade_time_str
        else:
            today = datetime.today()
            today_str = today.strftime('%Y-%m-%d')
        result = {}
        batch_size = 200
        num_batches = ceil(len(stock_codes) / batch_size)
        for batch_index in range(num_batches):
            data_dicts = []
            batch_codes = stock_codes[batch_index * batch_size:(batch_index + 1) * batch_size]
            stock_codes_str = ",".join(batch_codes)
            for time_level in time_levels:
                # 拉取数据
                all_dfs = []
                offset = 0
                limit = 8000  # tushare pro接口最大limit一般为8000
                while True:
                    df = self.ts_pro.stk_mins(**{
                        "ts_code": stock_codes_str,
                        "freq": time_level + "min",
                        "start_date": today_str + " 09:30:00",
                        "end_date": today_str + " 15:00:00",
                        "limit": limit,
                        "offset": offset
                    }, fields=[ "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq" ])
                    # print(f"{today_str} 拉取数据: {df}")
                    all_dfs.append(df)
                    if len(df) < limit:
                        break
                    offset += limit
                if all_dfs:
                    result_df = pd.concat(all_dfs, ignore_index=True)
                else:
                    result_df = pd.DataFrame()
                if not result_df.empty:
                    result_df = result_df.replace(['nan', 'NaN', ''], np.nan)  # 把字符串nan变成np.nan
                    result_df = result_df.where(pd.notnull(result_df), None)  # np.nan变成None
                    for row in result_df.itertuples():
                        stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                        if stock:
                            data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                            # print(f"处理股票: {stock.stock_code}, 时间: {row.trade_time}, 数据: {data_dict}")
                            data_dicts.append(data_dict)
                            # 准备缓存数据
                            cache_data_dict = data_dict.copy()
                            if 'stock' in cache_data_dict and isinstance(stock, StockInfo):
                                cache_data_dict['stock_code'] = row.ts_code
                                del cache_data_dict['stock']
                            prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                            if prepared_data:
                                await self.cache_set.history_time_trade(row.ts_code, time_level, prepared_data)
                            else:
                                logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
                for stock_code in batch_codes:
                    cache_key = self.cache_key.history_time_trade(stock_code, time_level)
                    await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
            print(f"data_dicts长度: {len(data_dicts)}")
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMinuteData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            logger.info(f"保存股票 {len(batch_codes)} 个代码的分钟级交易数据完成。结果: {result}")
        return result
    # =============== A股分钟行情(实时) ===============
    async def save_minute_time_trade_realtime(self, stock_code: str, time_level: str) -> None:
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
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
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
                        await self.cache_set.history_time_trade(stock_code, time_level, prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
            if data_dicts:
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockMinuteData,
                    data_list=data_dicts,
                    unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
                )
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock.stock_code, time_level)
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
        else:
            result = {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_minute_time_trade_realtime_by_stock_codes(self, stock_codes: List[str]) -> None:
        """
        保存股票的实时分钟级交易数据
        接口：rt_min
        描述：获取全A股票实时分钟数据，包括1~60min
        限量：单次最大1000行数据，可以通过股票代码提取数据，支持逗号分隔的多个代码同时提取
        """
        stock_codes_str = ",".join(stock_codes)
        # 拉取数据
        data_dicts = []
        for time_level in time_levels:
            df = self.ts_pro.rt_min(**{
                "topic": "", "freq": time_level + "min", "ts_code": stock_codes_str, "limit": "", "offset": ""
            }, fields=[
                "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
            ])
            if df is not None:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
        if data_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMinuteData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            for stock_code in stock_codes:
                for time_level in time_levels:
                    # --- 函数末尾执行最终修剪 ---
                    cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
                    await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                    # --- 修剪调用结束 ---
            return result
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

    async def save_minute_time_trade_realtime_by_stock_codes_and_time_level(self, stock_codes: List[str], time_level: str) -> None:
        """
        保存股票的实时分钟级交易数据
        接口：rt_min
        描述：获取全A股票实时分钟数据，包括1~60min
        限量：单次最大1000行数据，可以通过股票代码提取数据，支持逗号分隔的多个代码同时提取
        """
        stock_codes_str = ",".join(stock_codes)
        # 拉取数据
        data_dicts = []
        df = self.ts_pro.rt_min(**{
            "topic": "", "freq": time_level + "min", "ts_code": stock_codes_str, "limit": "", "offset": ""
        }, fields=[
            "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
        ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
                    cache_data_dict = data_dict.copy()
                    if 'stock' in cache_data_dict and isinstance(stock, StockInfo):
                        # 替换为 stock_code
                        cache_data_dict['stock_code'] = row.ts_code
                        del cache_data_dict['stock'] # 删除实例键
                    prepared_data = await self._prepare_data_for_cache(cache_data_dict, related_field_map=None)
                    if prepared_data:
                        await self.cache_set.history_time_trade(row.ts_code, time_level, prepared_data)
                    else:
                        logger.warning(f"为股票 {stock} 准备缓存数据失败，跳过缓存写入。原始数据: {data_dict}")
        if data_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMinuteData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            for stock_code in stock_codes:
                # --- 函数末尾执行最终修剪 ---
                cache_key =  self.cache_key.history_time_trade(stock_code, time_level)
                await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
                # --- 修剪调用结束 ---
            return result
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

    async def get_minute_time_trade_history(self, stock_code: str, time_level: str) -> None:
        """
        获取股票的历史分钟级交易数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.history_time_trade(stock_code, time_level)
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        stock_minute_data_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_minute_data_list.append(StockMinuteData(**data_dict))
        # 从数据库中获取数据
        stock_minute_data_list = StockMinuteData.objects.filter(stock_code=stock_code, time_level=time_level).order_by('-trade_time')[:self.cache_limit]
        return stock_minute_data_list

    #  =============== A股周线行情 ===============
    async def save_weekly_time_trade(self, stock_code: str) -> None:
        """
        保存股票的周线交易数据
        接口：weekly
        描述：获取A股周线行情
        限量：单次最大4500行，总量不限制
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        # 拉取数据
        df = self.ts_pro.stk_weekly_monthly(**{
            "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "freq": "week", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "change", "pct_chg"
        ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_week_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockWeeklyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def save_weekly_time_trade_by_stock_codes(self, stock_codes: List[str]) -> None:
        """
        保存股票的周线交易数据
        接口：weekly
        描述：获取A股周线行情
        限量：单次最大4500行，总量不限制
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        stock_codes_str = ",".join(stock_codes)
        # 拉取数据
        df = self.ts_pro.stk_weekly_monthly(**{
            "ts_code": stock_codes_str, "trade_date": "", "start_date": "", "end_date": "", "freq": "week", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "change", "pct_chg"
        ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_week_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockWeeklyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def get_weekly_time_trade_history(self, stock_code: str) -> None:
        """
        获取股票的历史周线交易数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.history_time_trade(stock_code, "Week")
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        stock_weekly_data_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_weekly_data_list.append(StockWeeklyData(**data_dict))
        # 从数据库中获取数据
        stock_weekly_data_list = StockWeeklyData.objects.filter(stock_code=stock_code, time_level="Week").order_by('-trade_date')[:self.cache_limit]
        return stock_weekly_data_list
    
    #  =============== A股月线行情 ===============
    async def save_monthly_time_trade(self, stock_code: str) -> None:
        """
        保存股票的月线交易数据
        接口：monthly
        描述：获取A股月线行情
        限量：单次最大4500行，总量不限制
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        # 拉取数据
        df = self.ts_pro.stk_weekly_monthly(**{
            "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "freq": "month", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "change", "pct_chg"
        ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_month_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMonthlyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def save_monthly_time_trade_by_stock_codes(self, stock_codes: List[str]) -> None:
        """
        保存股票的月线交易数据
        接口：monthly
        描述：获取A股月线行情
        限量：单次最大4500行，总量不限制
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        stock_codes_str = ",".join(stock_codes)
        # 拉取数据
        df = self.ts_pro.stk_weekly_monthly(**{
            "ts_code": stock_codes_str, "trade_date": "", "start_date": "", "end_date": "", "freq": "month", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "change", "pct_chg"
        ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_month_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMonthlyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def get_monthly_time_trade_history(self, stock_code: str) -> None:
        """
        获取股票的历史月线交易数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.history_time_trade(stock_code, "Month")
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        stock_monthly_data_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_monthly_data_list.append(StockMonthlyData(**data_dict))
        # 从数据库中获取数据
        stock_monthly_data_list = StockMonthlyData.objects.filter(stock_code=stock_code, time_level="Month").order_by('-trade_date')[:self.cache_limit]
        return stock_monthly_data_list
    
    #  =============== A股日线基本信息 ===============
    async def save_today_stock_basic_info(self) -> None:
        """
        保存股票的日线基本信息
        接口：daily_basic，可以通过数据工具调试和查看数据。
        更新时间：交易日每日15点～17点之间
        描述：获取全部股票每日重要的基本面指标，可用于选股分析、报表展示等。
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 拉取数据
        df = self.ts_pro.daily_basic(**{
            "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
            "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv"
        ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_stock_daily_basic_data(stock=stock, df_data=row)
                    await self.cache_set.stock_day_basic_info(row.ts_code, data_dict)
                    data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyBasic,
                data_list=data_dicts,
                unique_fields=['stock_code', 'trade_date'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        return result

    async def save_stock_daily_basic_history_by_stock_code(self, stock_code: str) -> None:
        """
        保存股票的日线基本信息
        """
        # 拉取数据
        df = self.ts_pro.daily_basic(**{
            "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
            "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
        ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_stock_daily_basic_data(stock=stock, df_data=row)
                    await self.cache_set.stock_day_basic_info(row.ts_code, data_dict)
                    data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyBasic,
                data_list=data_dicts,
                unique_fields=['stock_code', 'trade_date'] # ORM 能处理 stock 实例
            )
        else:
            result = []
        # --- 函数末尾执行最终修剪 ---
        cache_key = self.cache_key.stock_day_basic_info(stock_code)
        await self.cache_manager.ztrim_by_rank(cache_key, self.cache_limit)
        # --- 修剪调用结束 ---
        return result

    async def get_stock_daily_basic(self, stock_code: str) -> None:
        """
        获取股票的日线基本信息
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.today_basic_info(stock_code)
        data_dicts = await self.cache_get.stock_day_basic_info_by_limit(cache_key, self.cache_limit)
        stock_daily_basic_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_daily_basic_list.append(StockDailyBasic(**data_dict))
        # 从数据库中获取数据
        stock_daily_basic_list = StockDailyBasic.objects.filter(stock_code=stock_code).order_by('-trade_date')[:self.cache_limit]
        return stock_daily_basic_list
        
    #  =============== A股筹码及胜率 ===============
    # 每日筹码及胜率
    async def save_today_cyq_chips(self) -> None:
        """
        保存股票的每日筹码分布数据
        接口：cyq_perf
        描述：获取A股每日筹码平均成本和胜率情况，每天17~18点左右更新，数据从2018年开始
        来源：Tushare社区
        限量：单次最大5000条，可以分页或者循环提取
        积分：120积分可以试用(查看数据)，5000积分每天20000次，10000积分每天200000次，15000积分每天不限总量
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        stocks = self.stock_cache_get.all_stocks()
        # 提取所有股票代码
        stock_codes = [stock.stock_code for stock in stocks]
        # 每5000个为一组，拼接成字符串
        group_size = 5000
        stock_code_groups = [
            ','.join(stock_codes[i:i + group_size])
            for i in range(0, len(stock_codes), group_size)
        ]
        data_dicts = []
        for stock_codes_str in stock_code_groups:
            # 拉取数据
            df = self.ts_pro.cyq_perf(**{
                "ts_code": stock_codes_str, "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", 
                "cost_95pct", "weight_avg", "winner_rate"
            ])
            if df is not None:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        row['stock'] = stock  # 先转换好
                        data_dict = self.data_format_process_trade.set_cyq_perf_data(stock=stock, df_data=row)
                        await self.stock_cache_set.stock_basic_info(row.ts_code, data_dict)
                        data_dicts.append(data_dict)
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqPerf,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date']
            )
        return result

    async def save_cyq_chips_history_by_stock_code(self, stock_code: str) -> None:
        """
        保存股票的每日筹码分布数据
        """
        # 拉取数据
        df = self.ts_pro.cyq_perf(**{
                "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", 
                "cost_95pct", "weight_avg", "winner_rate"
            ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_cyq_perf_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqPerf,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date']
            )
        return result

    async def get_cyq_chips_history(self, stock_code: str) -> None:
        """
        获取股票的每日筹码分布数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.cyq_chips(stock_code)
        data_dicts = await self.cache_get.cyq_chips(cache_key)
        stock_cyq_chips_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_cyq_chips_list.append(StockCyqChips(**data_dict))
        # 从数据库中获取数据
        stock_cyq_chips_list = StockCyqChips.objects.filter(stock_code=stock_code).order_by('-trade_date')[:self.cache_limit]
        return stock_cyq_chips_list

    # 每日筹码分布
    async def save_today_cyq_chips(self) -> None:
        """
        保存股票的每日筹码分布数据
        接口：cyq_chips
        描述：获取A股每日的筹码分布情况，提供各价位占比，数据从2018年开始，每天17~18点之间更新当日数据
        来源：Tushare社区
        限量：单次最大2000条，可以按股票代码和日期循环提取
        积分：120积分可以试用查看数据，5000积分每天20000次，10000积分每天200000次，15000积分每天不限总量
        """
        if self.cache_set is None:
            await self.initialize_cache_objects()
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        result_all = { "尝试处理": 0, "失败": 0, "创建/更新成功": 0 }
        stocks = self.stock_cache_get.all_stocks()
        # 提取所有股票代码
        stock_codes = [stock.stock_code for stock in stocks]
        # 每5000个为一组，拼接成字符串
        group_size = 2000
        stock_code_groups = [
            ','.join(stock_codes[i:i + group_size])
            for i in range(0, len(stock_codes), group_size)
        ]
        for stock_codes_str in stock_code_groups:
            data_dicts = []
            # 拉取数据
            df = self.ts_pro.cyq_chips(**{
                "ts_code": stock_codes_str, "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "price", "percent"
            ])
            if df is not None:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_cyq_chips_data(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqChips,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date', 'price']
            )
            result_all['尝试处理'] += result['尝试处理']
            result_all['失败'] += result['失败']
            result_all['创建/更新成功'] += result['创建/更新成功']
        return result_all

    async def save_cyq_chips_history_by_stock_code(self, stock_code: str) -> None:
        """
        保存股票的每日筹码分布数据
        """
        # 拉取数据
        df = self.ts_pro.cyq_chips(**{
            "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "price", "percent"
        ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_cyq_chips_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqChips,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_date', 'price']
            )
        return result

    async def get_cyq_chips_history(self, stock_code: str) -> None:
        """
        获取股票的每日筹码分布数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.cyq_chips(stock_code)
        data_dicts = await self.cache_get.cyq_chips(cache_key)
        stock_cyq_chips_list = []
        if data_dicts:
            for data_dict in data_dicts:
                stock_cyq_chips_list.append(StockCyqChips(**data_dict))
        # 从数据库中获取数据
        stock_cyq_chips_list = StockCyqChips.objects.filter(stock_code=stock_code).order_by('-trade_date')[:self.cache_limit]
        return stock_cyq_chips_list
    






















