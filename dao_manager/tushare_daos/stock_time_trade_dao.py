# dao_manager\tushare_daos\stock_time_trade_dao.py
import asyncio
import logging
import time
from asgiref.sync import sync_to_async
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockCyqChips, StockCyqPerf, StockDailyBasic, StockMinuteData, StockWeeklyData, StockMonthlyData
from utils.cache_get import StockInfoCacheGet, StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockInfoCacheSet, StockTimeTradeCacheSet
from utils.cash_key import StockCashKey
from utils.data_format_process import StockInfoFormatProcess, StockTimeTradeFormatProcess
from stock_models.time_trade import StockDailyData_SZ, StockDailyData_SH, StockDailyData_CY, StockDailyData_KC, StockDailyData_BJ
from stock_models.time_trade import (
            StockMinuteData_5_SZ, StockMinuteData_5_SH, StockMinuteData_5_BJ, StockMinuteData_5_CY, StockMinuteData_5_KC,
            StockMinuteData_15_SZ, StockMinuteData_15_SH, StockMinuteData_15_BJ, StockMinuteData_15_CY, StockMinuteData_15_KC,
            StockMinuteData_30_SZ, StockMinuteData_30_SH, StockMinuteData_30_BJ, StockMinuteData_30_CY, StockMinuteData_30_KC,
            StockMinuteData_60_SZ, StockMinuteData_60_SH, StockMinuteData_60_BJ, StockMinuteData_60_CY, StockMinuteData_60_KC,
        )


logger = logging.getLogger("dao")
time_levels = ["5", "15", "30", "60"] # "1", 

class StockTimeTradeDAO(BaseDAO):
    def __init__(self):
        """初始化StockIndicatorsDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.stock_basic_dao = StockBasicInfoDao()
        self.cache_limit = 500 # 定义缓存数量上限
        self.cache_manager = CacheManager()
        self.cache_key = StockCashKey()
        self.data_format_process_trade = StockTimeTradeFormatProcess()
        self.data_format_process_stock = StockInfoFormatProcess()
        self.cache_set = StockTimeTradeCacheSet()
        self.cache_get = StockTimeTradeCacheGet()
        self.stock_cache_set = StockInfoCacheSet()
        self.stock_cache_get = StockInfoCacheGet()

    # =============== A股日线行情 ===============
    def get_daily_data_model_by_code(self, stock_code: str):
        """
        根据股票代码返回对应的日线数据表Model
        """
        if stock_code.startswith('3') and stock_code.endswith('.SZ'):
            return StockDailyData_CY
        elif stock_code.endswith('.SZ'):
            return StockDailyData_SZ
        elif stock_code.startswith('68') and stock_code.endswith('.SH'):
            return StockDailyData_KC
        elif stock_code.endswith('.SH'):
            return StockDailyData_SH
        elif stock_code.endswith('.BJ'):
            return StockDailyData_BJ
        else:
            print(f"未识别的股票代码: {stock_code}，默认使用SZ主板表")
            return StockDailyData_SZ  # 默认返回深市主板

    async def save_daily_time_trade_history_by_trade_date(self, trade_date: date) -> None:
        """
        保存指定交易日的所有股票日线交易数据，自动分表
        """
        trade_date_str = ""
        if isinstance(trade_date, str):
            trade_date_str = trade_date
        elif isinstance(trade_date, date):
            trade_date_str = trade_date.strftime("%Y%m%d")
        else:
            raise ValueError("trade_date 必须是 str 或 date 类型")
        df = self.ts_pro.stk_factor(
            **{
                "ts_code": "", "trade_date": trade_date_str, "start_date": "",
                "end_date": "", "offset": "", "limit": ""
            },
            fields=[
                "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol", "amount", "adj_factor",
                "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq", "low_qfq", "pre_close_hfq", "pre_close_qfq"
            ]
        )
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            # 按分表分组
            data_dicts_by_model = {}
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    model_class = self.get_daily_data_model_by_code(row.ts_code)
                    if model_class not in data_dicts_by_model:
                        data_dicts_by_model[model_class] = []
                    data_dicts_by_model[model_class].append(data_dict)
            result = {}
            for model_class, data_list in data_dicts_by_model.items():
                res = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_list,
                    unique_fields=['stock', 'trade_time']
                )
                result[model_class.__name__] = res
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_daily_time_trade_history_by_trade_dates(self, start_date: date = None, end_date: date = None) -> None:
        """
        保存指定日期区间的所有股票日线交易数据，自动分表
        """
        start_date_str = ""
        end_date_str = ""
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        offset = 0
        limit = 6000
        data_dicts_by_model = {}
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。{start_date_str} - {end_date_str}, freq=Day")
                break
            df = self.ts_pro.stk_factor(
                **{
                    "ts_code": "", "trade_date": "", "start_date": start_date_str,
                    "end_date": end_date_str, "offset": offset, "limit": limit
                },
                fields=[
                    "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol",
                    "amount", "adj_factor", "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq",
                    "low_qfq", "pre_close_hfq", "pre_close_qfq"
                ]
            )
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                        model_class = self.get_daily_data_model_by_code(row.ts_code)
                        if model_class not in data_dicts_by_model:
                            data_dicts_by_model[model_class] = []
                        data_dicts_by_model[model_class].append(data_dict)
            if len(df) < limit:
                break
            offset += limit
        result = {}
        for model_class, data_list in data_dicts_by_model.items():
            res = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            result[model_class.__name__] = res
        return result

    async def save_daily_time_trade_history_by_stock_code(self, stock_code: str) -> None:
        """
        保存指定股票的历史日线交易数据，自动分表
        """
        df = self.ts_pro.stk_factor(
            **{
                "ts_code": stock_code, "trade_date": "", "start_date": "",
                "end_date": "", "offset": "", "limit": ""
            },
            fields=[
                "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol", "amount", "adj_factor",
                "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq", "low_qfq", "pre_close_hfq", "pre_close_qfq"
            ]
        )
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            if data_dicts:
                model_class = self.get_daily_data_model_by_code(stock_code)
                result = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_daily_time_trade_history_by_stock_codes(self, stock_codes: List[str]) -> None:
        """
        保存多只股票的历史日线交易数据，自动分表
        """
        stock_codes_str = ",".join(stock_codes)
        data_dicts_by_model = {}
        offset = 0
        limit = 6000
        print(f"开始日线历史任务：{stock_codes_str}")
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。ts_code={stock_codes_str}, freq=Day")
                break
            df = self.ts_pro.stk_factor(
                **{
                    "ts_code": stock_codes_str, "trade_date": "", "start_date": "2000-01-01 00:00:00",
                    "end_date": "", "offset": offset, "limit": limit
                },
                fields=[
                    "ts_code", "trade_date", "close", "open", "high", "low", "pre_close", "change", "pct_change", "vol",
                    "amount", "adj_factor", "open_hfq", "open_qfq", "close_hfq", "close_qfq", "high_hfq", "high_qfq", "low_hfq",
                    "low_qfq", "pre_close_hfq", "pre_close_qfq"
                ]
            )
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                        model_class = self.get_daily_data_model_by_code(row.ts_code)
                        if model_class not in data_dicts_by_model:
                            data_dicts_by_model[model_class] = []
                        data_dicts_by_model[model_class].append(data_dict)
            if len(df) < limit:
                break
            offset += limit
        result = {}
        for model_class, data_list in data_dicts_by_model.items():
            res = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            result[model_class.__name__] = res
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

    async def save_daily_time_trade_yesterday(self) -> None:
        """
        保存股票的今日日线交易数据
        接口：daily
        数据说明：交易日每天15点～16点之间入库。本接口是未复权行情，停牌期间不提供数据
        调取说明：120积分每分钟内最多调取500次，每次6000条数据，相当于单次提取23年历史
        """
        # 获取当前日期
        today = datetime.today()
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        day_str = yesterday.strftime('%Y%m%d')
        result = await self.save_daily_time_trade_history_by_trade_date(day_str)
        return result

    # 未复权信息，慎用
    async def save_daily_time_trade_realtime(self, stock_code: str) -> None:
        """
        保存指定股票的实时日线交易数据，自动分表
        """
        df = self.ts_pro.rt_k(
            **{
                "topic": "", "ts_code": stock_code, "limit": "", "offset": ""
            },
            fields=[
                "ts_code", "name", "pre_close", "high", "open", "low", "close", "vol", "amount", "num"
            ]
        )
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_day_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            model_class = self.get_daily_data_model_by_code(stock_code)
            result = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def get_daily_time_trade_history_by_stock_code(self, stock_code: str) -> None:
        """
        获取指定股票的历史日线交易数据，自动分表
        """
        # 先尝试从Redis缓存获取
        cache_key = self.cache_key.history_time_trade(stock_code, "Day")
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        # 路由到正确的分表Model
        model_class = self.get_daily_data_model_by_code(stock_code)
        stock_daily_data_list = []
        if data_dicts:
            # 如果缓存有数据，直接反序列化为Model实例
            for data_dict in data_dicts:
                stock_daily_data_list.append(model_class(**data_dict))
            return stock_daily_data_list
        # 缓存没有数据，则从数据库查找对应分表
        stock_daily_data_list = model_class.objects.filter(stock_code=stock_code).order_by('-trade_time')[:self.cache_limit]
        return stock_daily_data_list

    # =============== A股分钟行情 ===============
    def get_minute_model(self, stock_code: str, time_level: str):
        """
        根据stock_code和time_level返回对应的分钟数据模型
        """
        # 只分5/15/30/60，1min默认用原表
        if time_level not in ['5', '15', '30', '60']:
            return StockMinuteData
        if stock_code.endswith('.SZ'):
            if stock_code.startswith('3'):
                return {
                    '5': StockMinuteData_5_CY, '15': StockMinuteData_15_CY, '30': StockMinuteData_30_CY, '60': StockMinuteData_60_CY
                }[time_level]
            else:
                return {
                    '5': StockMinuteData_5_SZ, '15': StockMinuteData_15_SZ, '30': StockMinuteData_30_SZ, '60': StockMinuteData_60_SZ
                }[time_level]
        elif stock_code.endswith('.SH'):
            if stock_code.startswith('68'):
                return {
                    '5': StockMinuteData_5_KC, '15': StockMinuteData_15_KC, '30': StockMinuteData_30_KC, '60': StockMinuteData_60_KC
                }[time_level]
            else:
                return {
                    '5': StockMinuteData_5_SH, '15': StockMinuteData_15_SH, '30': StockMinuteData_30_SH, '60': StockMinuteData_60_SH
                }[time_level]
        elif stock_code.endswith('.BJ'):
            return {
                '5': StockMinuteData_5_BJ, '15': StockMinuteData_15_BJ, '30': StockMinuteData_30_BJ, '60': StockMinuteData_60_BJ
            }[time_level]
        else:
            return StockMinuteData

    async def get_5_min_kline_time_by_day(self, stock_code: str, date: datetime.date = None) -> List[str]:
        """
        获取指定日期当天的所有5分钟K线的交易时间
        """
        if not date:
            date = datetime.today().date()
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            print(f"未找到股票代码：{stock_code}")
            return []
        start_datetime = datetime.combine(date, datetime.min.time())
        end_datetime = start_datetime + timedelta(days=1)
        # 用sync_to_async包装ORM查询
        @sync_to_async
        def get_trade_times():
            model = self.get_minute_model(stock_code, '5')  # 修改：自动分表
            qs = model.objects.filter(
                stock=stock,
                trade_time__gte=start_datetime,
                trade_time__lt=end_datetime
            ).values_list('trade_time', flat=True)
            return list(qs)
        records = await get_trade_times()
        trade_times = [record.strftime('%Y-%m-%d %H:%M:%S') for record in records]
        return trade_times

    async def get_latest_5_min_kline(self, stock_code: str) -> Optional[Dict]:
        """
        获取指定股票最新一条5分钟K线数据
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            print(f"未找到股票代码：{stock_code}")
            return None
        cache_data = await self.cache_get.latest_time_trade(stock_code=stock_code, time_level=5)
        if cache_data is not None:
            stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
            cache_data['stock'] = stock
            return self.get_minute_model(stock_code, '5')(**cache_data)  # 修改：自动分表
        @sync_to_async
        def get_latest_kline():
            model = self.get_minute_model(stock_code, '5')  # 修改：自动分表
            record = (model.objects
                    .filter(stock=stock)
                    .order_by('-trade_time')
                    .first())
            return record
        latest_kline = await get_latest_kline()
        if not latest_kline:
            print(f"{stock_code} 未查询到5分钟K线数据")
        return latest_kline

    async def save_minute_time_trade_history_by_time_level(self, stock_code: str, time_level: str) -> None:
        """
        保存股票的历史分钟级交易数据
        """
        df = self.ts_pro.stk_mins(**{
                "ts_code": stock_code, "freq": time_level + "min", "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"
            ])
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            # 新增：模型分组字典
            model_grouped_data_dicts = {}
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                    model_class = self.get_minute_model(row.ts_code, time_level)
                    if model_class not in model_grouped_data_dicts:
                        model_grouped_data_dicts[model_class] = []
                    model_grouped_data_dicts[model_class].append(data_dict)
            result = {}
            for model_class, data_list in model_grouped_data_dicts.items():
                result = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_list,
                    unique_fields=['stock', 'trade_time']
                )
        else:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_minute_time_trade_history_by_stock_code(self, stock_code: str) -> None:
        """
        保存股票的历史分钟级交易数据
        """
        result = {}
        for time_level in ['5', '15', '30', '60']:
            offset = 0
            limit = 8000
            # 新增：模型分组字典
            model_grouped_data_dicts = {}
            df = self.ts_pro.stk_mins(**{
                "ts_code": stock_code, "freq": time_level + "min", "start_date": "2020-01-01 00:00:00", "end_date": "", "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq"
            ])
            if df.empty:
                continue
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                        model_class = self.get_minute_model(row.ts_code, time_level)
                        if model_class not in model_grouped_data_dicts:
                            model_grouped_data_dicts[model_class] = []
                        model_grouped_data_dicts[model_class].append(data_dict)
            for model_class, data_list in model_grouped_data_dicts.items():
                result = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_list,
                    unique_fields=['stock', 'trade_time']
                )
        if not result:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_minute_time_trade_history_by_stock_codes(self, stock_codes: List[str], start_date_str: str="2020-01-01 00:00:00", end_date_str: str="") -> None:
        """
        保存股票的历史分钟级交易数据 (已优化)
        """
        stock_codes_str = ",".join(stock_codes)
        # 遍历不同的分钟级别
        for time_level in ['5', '15', '30', '60']:
            offset = 0
            limit = 8000 # Tushare Pro单次最大返回8000条
            # 循环拉取分页数据
            while True:
                # 新增：模型分组字典在循环内部初始化，处理每个批次的数据
                model_grouped_data_dicts = {}
                if offset >= 100000: # Tushare Pro对stk_mins接口的offset有10万的限制
                    logger.warning(f"offset已达10万，停止拉取。ts_code={stock_codes_str}, freq={time_level}min")
                    break
                
                # 调试信息：打印当前拉取的参数
                print(f"调试: 正在拉取 {time_level}min 数据, stock_codes: {len(stock_codes)}个, offset: {offset}, limit: {limit}")

                df = self.ts_pro.stk_mins(**{
                    "ts_code": stock_codes_str, "freq": time_level + "min", "start_date": start_date_str, "end_date": end_date_str, 
                    "limit": limit, "offset": offset
                }, fields=[ "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq" ])
                
                # 如果返回的DataFrame为空，说明没有更多数据，结束循环
                if df.empty:
                    print(f"调试: 拉取到空数据帧，结束 {time_level}min 的拉取。")
                    break
                
                # --- 核心效率优化 ---
                # 新增：从当前批次数据中提取所有唯一的股票代码
                unique_ts_codes = df['ts_code'].unique().tolist()
                # 新增：一次性从数据库查询所有相关的股票基础信息，避免在循环中逐条查询
                # 假设 stock_basic_dao 中有一个 get_stocks_by_codes 的方法进行批量查询
                related_stocks = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
                # 新增：创建一个从股票代码到股票对象的映射，便于快速查找
                stock_map = {stock.stock_code: stock for stock in related_stocks}
                print(f"调试: 批量获取了 {len(stock_map)} 个相关的股票信息对象。")
                # --- 优化结束 ---

                # 数据清洗
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)

                # 遍历处理后的DataFrame数据
                for row in df.itertuples():
                    # 修改：从预先查好的映射中获取股票对象，避免了N+1查询
                    stock = stock_map.get(row.ts_code)
                    if stock:
                        # 格式化数据
                        data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                        # 根据股票代码和时间级别获取对应的模型类
                        model_class = self.get_minute_model(row.ts_code, time_level)
                        # 将数据按模型分组
                        if model_class not in model_grouped_data_dicts:
                            model_grouped_data_dicts[model_class] = []
                        model_grouped_data_dicts[model_class].append(data_dict)
                
                # 批量保存分组后的数据
                for model_class, data_list in model_grouped_data_dicts.items():
                    if not data_list:
                        continue
                    result = await self._save_all_to_db_native_upsert(
                        model_class=model_class,
                        data_list=data_list,
                        unique_fields=['stock', 'trade_time']
                    )
                    logger.info(f"保存 {model_class.__name__} 的 {time_level}分钟级交易数据 offset={offset} 完成. 插入/更新了 {len(data_list)} 条记录。")
                
                # 如果本次返回的数据量小于请求的limit，说明是最后一页，结束循环
                if len(df) < limit:
                    break
                # 否则，增加offset以拉取下一页数据
                offset += limit

        logger.info(f"保存 {len(stock_codes)}个股票 的分钟级交易数据全部完成.")
        return

    async def save_minute_time_trade_history_by_stock_code_and_time_level(self, stock_code: str, time_level: str, start_date: str="2020-01-01 00:00:00", end_date: str="") -> None:
        """
        保存股票的历史分钟级交易数据
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        offset = 0
        limit = 8000
        # 新增：模型分组字典
        model_grouped_data_dicts = {}
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。{stock}, time_level={time_level}min")
                break
            df = self.ts_pro.stk_mins(**{
                "ts_code": stock_code, "freq": time_level + "min", "start_date": start_date, "end_date": end_date, "limit": limit, "offset": offset
            }, fields=[ "ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount", "freq" ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)
                for row in df.itertuples():
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                        model_class = self.get_minute_model(row.ts_code, time_level)
                        if model_class not in model_grouped_data_dicts:
                            model_grouped_data_dicts[model_class] = []
                        model_grouped_data_dicts[model_class].append(data_dict)
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        for model_class, data_list in model_grouped_data_dicts.items():
            result = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
        return result

    # =============== A股分钟行情(实时) ===============
    async def save_minute_time_trade_realtime(self, stock_code: str, time_level: str) -> None:
        """
        保存股票的实时分钟级交易数据
        """
        df = self.ts_pro.rt_min(**{
            "topic": "", "freq": time_level + "MIN", "ts_code": stock_code, "limit": "", "offset": ""
        }, fields=[
            "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
        ])
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            if data_dicts:
                # 自动分表
                model_class = self.get_minute_model(stock_code, time_level)
                result = await self._save_all_to_db_native_upsert(
                    model_class=model_class,
                    data_list=data_dicts,
                    unique_fields=['stock', 'trade_time']
                )
            else:
                result = {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        else:
            result = {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        return result

    async def save_minute_time_trade_realtime_by_stock_codes(self, stock_codes: List[str]) -> None:
        """
        保存股票的实时分钟级交易数据（所有分钟级别）
        """
        stock_codes_str = ",".join(stock_codes)
        for time_level in ['5', '15', '30', '60']:
            data_dicts = []
            df = self.ts_pro.rt_min(**{
                "topic": "", "freq": time_level + "MIN", "ts_code": stock_codes_str, "limit": "", "offset": ""
            }, fields=[
                "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
            ])
            if not df.empty:
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
            if data_dicts:
                # 按stock_code分组写入
                code_group = {}
                for d in data_dicts:
                    code = d['stock'].stock_code
                    code_group.setdefault(code, []).append(d)
                for code, group in code_group.items():
                    model_class = self.get_minute_model(code, time_level)
                    result = await self._save_all_to_db_native_upsert(
                        model_class=model_class,
                        data_list=group,
                        unique_fields=['stock', 'trade_time']
                    )
        return result if data_dicts else {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

    async def save_minute_time_trade_realtime_by_stock_codes_and_time_level(self, stock_codes: List[str], time_level: str):
        """
        【V2 - 优化版】保存股票的实时分钟级交易数据（指定分钟级别）
        
        核心优化:
        1.  【解决N+1查询】在循环外一次性获取所有stock对象，避免循环内数据库查询。
        2.  【批量写入缓存】在循环外一次性将所有数据写入Redis，减少网络I/O。
        3.  【并发执行】使用 asyncio.gather 并发执行数据库和缓存的写入操作。
        """
        if not stock_codes:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        # 1. 从API获取数据 (保持不变)
        stock_codes_str = ",".join(stock_codes)
        df = self.ts_pro.rt_min(**{
            "topic": "", "freq": time_level + "MIN", "ts_code": stock_codes_str, "limit": "", "offset": ""
        }, fields=[
            "ts_code", "freq", "time", "open", "close", "high", "low", "vol", "amount"
        ])

        if df.empty:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        # 2. 数据准备阶段 (核心优化)
        
        # 【代码优化处】一次性获取所有相关的stock对象，存入字典以便快速查找
        stocks_map = await self.stock_basic_dao.get_stocks_by_codes(
            [row.ts_code for row in df.itertuples()]
        )
        
        model_grouped_data_dicts = {}
        cache_payload = {} # 【代码新增处】用于存储待写入缓存的数据

        # 清理DataFrame (保持不变)
        df = df.replace(['nan', 'NaN', ''], np.nan)
        df = df.where(pd.notnull(df), None)

        # 循环内只做CPU密集型的数据准备工作，不做任何 await I/O 操作
        for row in df.itertuples():
            # 【代码优化处】从内存中的字典直接获取stock对象，速度极快
            stock = stocks_map.get(row.ts_code)
            
            if stock:
                data_dict = self.data_format_process_trade.set_time_trade_minute_data(stock=stock, df_data=row)
                if data_dict.get('trade_time'):
                    model_class = self.get_minute_model(row.ts_code, time_level)
                    
                    # 按模型分组，准备数据库数据
                    if model_class not in model_grouped_data_dicts:
                        model_grouped_data_dicts[model_class] = []
                    model_grouped_data_dicts[model_class].append(data_dict)
                    
                    # 【代码优化处】准备缓存数据，暂不写入
                    cache_payload[row.ts_code] = data_dict.copy()

        # 3. 数据持久化阶段 (核心优化)
        if not model_grouped_data_dicts:
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        # 创建数据库保存任务
        db_save_tasks = []
        for model_class, data_list in model_grouped_data_dicts.items():
            task = self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            db_save_tasks.append(task)

        # 【代码优化处】创建缓存批量写入任务
        cache_save_task = self.cache_set.batch_set_latest_time_trade(cache_payload, time_level)

        # 【代码优化处】使用 asyncio.gather 并发执行所有数据库保存任务和缓存写入任务
        all_tasks = db_save_tasks + [cache_save_task]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # 4. 结果统计
        final_result = {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        total_records = sum(len(data) for data in model_grouped_data_dicts.values())
        final_result["尝试处理"] = total_records

        # 遍历数据库保存任务的结果
        for i in range(len(db_save_tasks)):
            res = results[i]
            if isinstance(res, Exception):
                # 如果某个批次保存失败，需要知道是哪个批次
                model_class = list(model_grouped_data_dicts.keys())[i]
                batch_size = len(list(model_grouped_data_dicts.values())[i])
                final_result["失败"] += batch_size
                logger.error(f"保存模型 {model_class.__name__} 的批次时发生异常: {res}", exc_info=res)
            elif isinstance(res, dict):
                final_result["失败"] += res.get("失败", 0)
                final_result["创建/更新成功"] += res.get("创建/更新成功", 0)
        
        # 检查缓存任务的结果
        cache_result = results[-1]
        if isinstance(cache_result, Exception):
            logger.error(f"批量写入分钟线缓存时发生异常: {cache_result}", exc_info=cache_result)

        return final_result

    async def get_minute_time_trade_history(self, stock_code: str, time_level: str) -> None:
        """
        获取股票的历史分钟级交易数据
        """
        # 从Redis缓存中获取数据
        cache_key = self.cache_key.history_time_trade(stock_code, time_level)
        data_dicts = await self.cache_get.history_time_trade(cache_key)
        stock_minute_data_list = []
        if data_dicts:
            model_class = self.get_minute_model(stock_code, time_level)
            for data_dict in data_dicts:
                stock_minute_data_list.append(model_class(**data_dict))
            return stock_minute_data_list
        # 从数据库中获取数据
        model_class = self.get_minute_model(stock_code, time_level)
        stock_minute_data_list = model_class.objects.filter(
            stock__stock_code=stock_code
        ).order_by('-trade_time')[:self.cache_limit]
        return stock_minute_data_list

    #  =============== A股周线行情 ===============
    async def save_weekly_time_trade(self, stock_code: str) -> None:
        """
        保存股票的周线交易数据
        接口：weekly
        描述：获取A股周线行情
        限量：单次最大4500行，总量不限制
        """
        # 拉取数据
        df = self.ts_pro.stk_week_month_adj(**{
            "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "freq": "week", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "change", "pct_chg"
        ])
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_week_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
            if data_dicts is not None:
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockWeeklyData,
                    data_list=data_dicts,
                    unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
                )
        else:
            result = []
        return result

    async def save_weekly_time_trade_by_stock_codes(self, stock_codes: List[str], start_date: str = "2020-01-01") -> None:
        """
        保存股票的周线交易数据
        接口：weekly
        描述：获取A股周线行情
        限量：单次最大4500行，总量不限制
        """
        stock_codes_str = ",".join(stock_codes)
        # 拉取数据
        df = self.ts_pro.stk_week_month_adj(**{
            "ts_code": stock_codes_str, "trade_date": "", "start_date": start_date, "end_date": "", "freq": "week", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "change", "pct_chg"
        ])
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_week_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
            if data_dicts is not None:
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockWeeklyData,
                    data_list=data_dicts,
                    unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
                )
        else:
            result = []
        print(f"result：{result}")
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
    async def save_monthly_time_trade(self, stock_code: str, start_date: str = "2010-01-01") -> None:
        """
        保存股票的月线交易数据
        接口：monthly
        描述：获取A股月线行情
        限量：单次最大4500行，总量不限制
        """
        # 拉取数据
        df = self.ts_pro.stk_week_month_adj(**{
            "ts_code": stock_code, "trade_date": "", "start_date": start_date, "end_date": "", "freq": "month", "limit": "", "offset": ""
        }, fields=[ "ts_code", "trade_date", "freq", "pre_close", "open_qfq", "high_qfq", "low_qfq", 
                       "close_qfq", "vol", "amount", "change", "pct_chg"])
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_time_trade_month_data(stock=stock, df_data=row)
                    # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                    data_dicts.append(data_dict)
            if data_dicts is not None:
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockMonthlyData,
                    data_list=data_dicts,
                    unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
                )
        else:
            result = []
        return result

    async def save_monthly_time_trade_by_trade_date(self, start_date: str = "2010-01-01") -> None:
        """
        保存股票的月线交易数据
        接口：monthly
        描述：获取A股月线行情
        限量：单次最大4500行，总量不限制
        """
        offset = 0
        limit = 6000
        while True:
            if offset >= 100000:
                logger.warning(f"月线行情offset已达10万，停止拉取。")
                break
            # 拉取数据
            df = self.ts_pro.stk_week_month_adj(**{
                "ts_code": "", "trade_date": "", "start_date": start_date, "end_date": "", "freq": "month", "limit": "", "offset": ""
            }, fields=[ "ts_code", "trade_date", "freq", "pre_close", "open_qfq", "high_qfq", "low_qfq", 
                       "close_qfq", "vol", "amount", "change", "pct_chg"])
            if not df.empty:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                data_dicts = []
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_month_data(stock=stock, df_data=row)
                        # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                        data_dicts.append(data_dict)
                if data_dicts is not None:
                    result = await self._save_all_to_db_native_upsert(
                        model_class=StockMonthlyData,
                        data_list=data_dicts,
                        unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
                    )
            else:
                result = []
        return result

    async def save_monthly_time_trade_by_stock_codes(self, stock_codes: List[str], start_date: str = "2010-01-01") -> None:
        """
        保存股票的月线交易数据
        接口：monthly
        描述：获取A股月线行情
        限量：单次最大4500行，总量不限制
        """
        stock_codes_str = ",".join(stock_codes)
        offset = 0
        limit = 6000
        data_dicts = []
        result = []
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。")
                break
            # 拉取数据
            df = self.ts_pro.stk_week_month_adj(**{
                "ts_code": stock_codes_str, "trade_date": "", "start_date": start_date, "end_date": "", "freq": "month", "limit": limit, "offset": offset
            }, fields=[ "ts_code", "trade_date", "freq", "pre_close", "open_qfq", "high_qfq", "low_qfq", 
                        "close_qfq", "vol", "amount", "change", "pct_chg"])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_time_trade_month_data(stock=stock, df_data=row)
                        # 1. 添加到数据库保存列表 (包含 StockInfo 实例)
                        data_dicts.append(data_dict)
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockMonthlyData,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time'] # ORM 能处理 stock 实例
            )
        print(f"result: {result}, data_dicts: {len(data_dicts)}")
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
        trade_date_str = today.strftime('%Y%m%d')
        # 拉取数据
        df = self.ts_pro.daily_basic(**{
            "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
            "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv"
        ])
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_stock_daily_basic_data(stock=stock, df_data=row)
                    await self.cache_set.stock_day_basic_info(row.ts_code, data_dict)
                    data_dicts.append(data_dict)
            if data_dicts is not None:
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockDailyBasic,
                    data_list=data_dicts,
                    unique_fields=['stock_code', 'trade_time'] # ORM 能处理 stock 实例
                )
                print(f"保存 {trade_date_str} 股票重要的基本面指标 完成。result: {result}")
        else:
            result = []
        return result

    async def save_yesterday_stock_basic_info(self) -> None:
        """
        保存股票的日线基本信息
        接口：daily_basic，可以通过数据工具调试和查看数据。
        更新时间：交易日每日15点～17点之间
        描述：获取全部股票每日重要的基本面指标，可用于选股分析、报表展示等。
        """
        # 获取当前日期
        today = datetime.today()
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        trade_date_str = yesterday.strftime('%Y%m%d')
        # 拉取数据
        df = self.ts_pro.daily_basic(**{
            "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
            "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv"
        ])
        if not df.empty:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process_trade.set_stock_daily_basic_data(stock=stock, df_data=row)
                    await self.cache_set.stock_day_basic_info(row.ts_code, data_dict)
                    data_dicts.append(data_dict)
            if data_dicts is not None:
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockDailyBasic,
                    data_list=data_dicts,
                    unique_fields=['stock_code', 'trade_date'] # ORM 能处理 stock 实例
                )
                print(f"保存 {trade_date_str} 股票重要的基本面指标 完成。result: {result}")
        else:
            result = []
        return result

    async def save_stock_daily_basic_history_by_stock_code(self, stock_code: str, start_date: date=None, end_date: date=None) -> None:
        """
        保存股票的日线基本信息
        """
        stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
        if stock:
            start_date_str = ""
            end_date_str = ""
            if start_date is not None:
                start_date_str = start_date.strftime('%Y%m%d')
            if end_date is not None:
                end_date_str = end_date.strftime('%Y%m%d')
            data_dicts = []
            # 拉取数据
            df = self.ts_pro.daily_basic(**{
                "ts_code": stock_code, "trade_date": "", "start_date": start_date_str, "end_date": end_date_str, "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
                "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
            ])
            if not df.empty:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    data_dict = self.data_format_process_trade.set_stock_daily_basic_data(stock=stock, df_data=row)
                    await self.cache_set.stock_day_basic_info(row.ts_code, data_dict)
                    data_dicts.append(data_dict)
            if data_dicts is not None:
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockDailyBasic,
                    data_list=data_dicts,
                    unique_fields=['stock_code', 'trade_date'] # ORM 能处理 stock 实例
                )
            return result
        else:
            return None

    async def save_stock_daily_basic_history_by_stock_codes(self, stock_codes: List[str], start_date: str = "2020-01-01", end_date: date=None) -> None:
        """
        保存股票的日线基本信息
        """
        # 拉取数据
        # 拉取数据
        stock_codes_str = ",".join(stock_codes)
        data_dicts = []
        offset = 0
        limit = 6000  # tushare pro接口最大limit一般为8000
        while True:
            if offset >= 100000:
                logger.warning(f"offset已达10万，停止拉取。ts_code={stock_codes_str}, freq=Day")
                break
            df = self.ts_pro.daily_basic(**{
                "ts_code": stock_codes_str, "trade_date": "", "start_date": start_date, "end_date": "", "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", 
                "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv", "limit_status"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_stock_daily_basic_data(stock=stock, df_data=row)
                        await self.cache_set.stock_day_basic_info(row.ts_code, data_dict)
                        data_dicts.append(data_dict)
            if len(df) < limit:
                break
            offset += limit
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockDailyBasic,
                data_list=data_dicts,
                unique_fields=['stock_code', 'trade_date'] # ORM 能处理 stock 实例
            )
        else:
            result = []
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
    async def save_today_cyq_perf(self) -> None:
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
        trade_date_str = today.strftime('%Y%m%d')
        data_dicts = []
        # 拉取数据
        offset = 0
        limit = 5000  # tushare pro接口最大limit一般为8000
        while True:
            if offset >= 100000:
                logger.warning(f"每日筹码及胜率 offset已达10万，停止拉取。")
                break
            # 拉取数据
            df = self.ts_pro.cyq_perf(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", 
                "cost_95pct", "weight_avg", "winner_rate"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_cyq_perf_data(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
            if len(df) < limit:
                break
            offset += limit
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqPerf,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_yesterday_cyq_perf(self) -> None:
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
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        trade_date_str = yesterday.strftime('%Y%m%d')
        data_dicts = []
        # 拉取数据
        offset = 0
        limit = 5000  # tushare pro接口最大limit一般为8000
        while True:
            if offset >= 100000:
                logger.warning(f"每日筹码及胜率 offset已达10万，停止拉取。")
                break
            # 拉取数据
            df = self.ts_pro.cyq_perf(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", 
                "cost_95pct", "weight_avg", "winner_rate"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_cyq_perf_data(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
            if len(df) < limit:
                break
            offset += limit
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqPerf,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_cyq_perf_history(self, start_date: date=None, end_date: date=None) -> None:
        """
        保存股票的每日筹码及胜率数据
        """
        start_date_str = ""
        end_date_str = ""
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        data_dicts = []
        # 拉取数据
        offset = 0
        limit = 6000  # tushare pro接口最大limit一般为8000
        while True:
            if offset >= 100000:
                logger.warning(f"每日筹码及胜率 offset已达10万，停止拉取。")
                break
            # 拉取数据
            df = self.ts_pro.cyq_perf(**{
                "ts_code": "", "trade_date": "", "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", 
                "cost_95pct", "weight_avg", "winner_rate"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_basic_dao.get_stock_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process_trade.set_cyq_perf_data(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
            if len(df) < limit:
                break
            offset += limit
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqPerf,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            # logger.info(f"完成每日筹码及胜率：{start_date_str} - {end_date_str}, 结果：{result}")
        return result

    async def get_cyq_perf_history(self, stock_code: str) -> None:
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
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        trade_date_str = today.strftime('%Y%m%d')
        # 拉取数据
        offset = 0
        limit = 2000
        all_stocks = await self.stock_basic_dao.get_stock_list()
        data_dicts = []
        for stock in all_stocks:
            while True:
                if offset >= 100000:
                    logger.warning(f"每日筹码分布 offset已达10万，停止拉取。")
                    break
                # 拉取数据
                df = self.ts_pro.cyq_chips(**{
                    "ts_code": stock.stock_code, "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "price", "percent"
                ])
                if df.empty:
                    break
                else:
                    df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                    df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                    for row in df.itertuples():
                        if stock:
                            data_dict = self.data_format_process_trade.set_cyq_chips_data(stock=stock, df_data=row)
                            data_dicts.append(data_dict)
                time.sleep(0.5)
                if len(df) < limit:
                    break
                offset += limit
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqChips,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time', 'price']
            )
            logger.info(f"完成每日筹码分布，结果：{result}")
        return result

    async def save_yesterday_cyq_chips(self) -> None:
        """
        保存股票的每日筹码分布数据
        接口：cyq_chips
        描述：获取A股每日的筹码分布情况，提供各价位占比，数据从2018年开始，每天17~18点之间更新当日数据
        来源：Tushare社区
        限量：单次最大2000条，可以按股票代码和日期循环提取
        积分：120积分可以试用查看数据，5000积分每天20000次，10000积分每天200000次，15000积分每天不限总量
        """
        # 获取当前日期
        today = datetime.today()
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        trade_date_str = yesterday.strftime('%Y%m%d')
        # 拉取数据
        offset = 0
        limit = 2000
        all_stocks = await self.stock_basic_dao.get_stock_list()
        data_dicts = []
        for stock in all_stocks:
            while True:
                if offset >= 100000:
                    logger.warning(f"每日筹码分布 offset已达10万，停止拉取。")
                    break
                # 拉取数据
                df = self.ts_pro.cyq_chips(**{
                    "ts_code": stock.stock_code, "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": limit, "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "price", "percent"
                ])
                if df.empty:
                    break
                else:
                    df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                    df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                    for row in df.itertuples():
                        if stock:
                            data_dict = self.data_format_process_trade.set_cyq_chips_data(stock=stock, df_data=row)
                            data_dicts.append(data_dict)
                time.sleep(0.5)
                if len(df) < limit:
                    break
                offset += limit
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqChips,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time', 'price']
            )
            logger.info(f"完成每日筹码分布，结果：{result}")
        return result

    async def save_cyq_chips_history(self, stock: StockInfo, start_date: date=None, end_date: date=None) -> None:
        """
        保存股票的每日筹码分布数据
        """
        start_date_str = "20250101"
        end_date_str = ""
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        # 拉取数据
        offset = 0
        limit = 2000
        data_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"每日筹码分布 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.cyq_chips(**{
                "ts_code": stock.stock_code, "trade_date": "", "start_date": start_date_str, "end_date": end_date_str, "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "price", "percent"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    if stock:
                        data_dict = self.data_format_process_trade.set_cyq_chips_data(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
            time.sleep(0.5)
            if len(df) < limit:
                break
            offset += limit
        if data_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCyqChips,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time', 'price']
            )
            logger.info(f"完成每日筹码分布：{stock}, 结果：{result}")
        return result
    






















