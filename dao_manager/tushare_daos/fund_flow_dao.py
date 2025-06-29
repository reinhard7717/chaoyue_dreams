
import logging
from datetime import date, datetime, timedelta
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from stock_models.fund_flow import FundFlowCntDC, FundFlowCntTHS, FundFlowDaily, FundFlowDailyDC, FundFlowDailyTHS, FundFlowIndustryTHS, FundFlowMarketDc, TopInst, TopList
from stock_models.market import LimitListThs
from utils.data_format_process import FundFlowFormatProcess

logger = logging.getLogger("dao")

class FundFlowDao(BaseDAO):
    def __init__(self):
        super().__init__()
        from utils.cache_get import StockInfoCacheGet, UserCacheGet
        from utils.cache_set import StockInfoCacheSet, UserCacheSet
        from utils.cash_key import StockCashKey

        self.data_format_process = FundFlowFormatProcess()
        self.index_dao = IndexBasicDAO()
        self.industry_dao = IndustryDao()
        self.stock_cache_key = StockCashKey()
        self.stock_cache_set = StockInfoCacheSet()
        self.stock_cache_get = StockInfoCacheGet()
        self.user_cache_set = UserCacheSet()
        self.user_cache_get = UserCacheGet()

    # ============== 日级资金流向数据 ==============
    async def save_today_fund_flow_daily_data(self) -> Dict:
        """
        保存今天的日级资金流向数据
        接口：moneyflow，可以通过数据工具调试和查看数据。
        描述：获取沪深A股票资金流向数据，分析大单小单成交情况，用于判别资金动向，数据开始于2010年。
        限量：单次最大提取6000行记录，总量不限制
        积分：用户需要至少2000积分才可以调取，基础积分有流量控制，积分越多权限越大，请自行提高积分，具体请参考积分获取办法
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取日级资金流向数据
        df = self.ts_pro.moneyflow(**{
            "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount", "buy_md_vol", "buy_md_amount",
            "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount",
            "sell_elg_vol", "sell_elg_amount", "net_mf_vol", "net_mf_amount"
        ])
        print(f"今天的日级资金流向数据 数量: {len(df)}")
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDaily,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_yesterday_fund_flow_daily_data(self) -> Dict:
        """
        保存今天的日级资金流向数据
        接口：moneyflow，可以通过数据工具调试和查看数据。
        描述：获取沪深A股票资金流向数据，分析大单小单成交情况，用于判别资金动向，数据开始于2010年。
        限量：单次最大提取6000行记录，总量不限制
        积分：用户需要至少2000积分才可以调取，基础积分有流量控制，积分越多权限越大，请自行提高积分，具体请参考积分获取办法
        """
        # 获取当前日期
        today = datetime.today()
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        day_str = yesterday.strftime('%Y%m%d')
        # 获取日级资金流向数据
        df = self.ts_pro.moneyflow(**{
            "ts_code": "", "trade_date": day_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount", "buy_md_vol", "buy_md_amount",
            "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount",
            "sell_elg_vol", "sell_elg_amount", "net_mf_vol", "net_mf_amount"
        ])
        print(f"今天的日级资金流向数据 数量: {len(df)}")
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDaily,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_history_fund_flow_daily_data_by_trade_date(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        保存历史日级资金流向数据
        """
        trade_date_str = "20240101"
        start_date_str = ""
        end_date_str = ""
        if trade_date is not None:
            trade_date_str = trade_date.strftime('%Y%m%d')
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        # 获取历史日级资金流向数据
        offset = 0
        limit = 6000
        data_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"板块资金流向数据 - 同花顺 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.moneyflow(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount", "buy_md_vol", "buy_md_amount",
                "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount",
                "sell_elg_vol", "sell_elg_amount", "net_mf_vol", "net_mf_amount"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process.set_fund_flow_data(stock=stock, df_data=row)
                        # print(f"日级资金流向数据。trade_date_str: {trade_date_str}, stock: {stock}, dict: {data_dict}")
                        data_dicts.append(data_dict)
                print(f"{trade_date} 历史日级资金流向数据，len(df): {len(df)}, len(data_dicts): {len(data_dicts)}")
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDaily,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        print(f"完成 {trade_date} 历史日级资金流向数据，result: {result}")
        return result

    async def save_history_fund_flow_daily_data_by_stock_code(self, stock_code: str) -> Dict:
        """
        保存历史日级资金流向数据
        """
        # 获取历史日级资金流向数据
        df = self.ts_pro.moneyflow(**{
            "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount", "buy_md_vol", "buy_md_amount",
            "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount",
            "sell_elg_vol", "sell_elg_amount", "net_mf_vol", "net_mf_amount"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDaily,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_history_fund_flow_daily_data_by_stock_codes(self, stock_codes: List[str]) -> Dict:
        """
        保存历史日级资金流向数据
        """
        # 获取历史日级资金流向数据
        stock_codes_str = ",".join(stock_codes)
        df = self.ts_pro.moneyflow(**{
            "ts_code": stock_codes_str, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount", "buy_md_vol", "buy_md_amount",
            "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount",
            "sell_elg_vol", "sell_elg_amount", "net_mf_vol", "net_mf_amount"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDaily,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result


    # ============== 个股日级资金流向数据 - 同花顺 ==============
    async def save_today_fund_flow_daily_ths_data(self) -> Dict:
        """
        保存今天的日级资金流向数据 - 同花顺
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取日级资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_ths(**{
            "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "latest", "net_amount", "net_d5_amount", "buy_lg_amount", 
            "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        print(f"今天的日级资金流向数据 - 同花顺 数量: {len(df)}")
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data_ths(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDailyTHS,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_yesterday_fund_flow_daily_ths_data(self) -> Dict:
        """
        保存今天的日级资金流向数据 - 同花顺
        """
        # 获取当前日期
        today = datetime.today()
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        day_str = yesterday.strftime('%Y%m%d')
        # 获取日级资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_ths(**{
            "ts_code": "", "trade_date": day_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "latest", "net_amount", "net_d5_amount", "buy_lg_amount", 
            "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        print(f"今天的日级资金流向数据 - 同花顺 数量: {len(df)}")
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data_ths(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDailyTHS,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_history_fund_flow_daily_ths_data_by_trade_date(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        保存历史日级资金流向数据 - 同花顺
        """
        # 获取历史日级资金流向数据 - 同花顺
        trade_date_str = ""
        start_date_str = ""
        end_date_str = ""
        if trade_date is not None:
            trade_date_str = trade_date.strftime('%Y%m%d')
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        offset = 0
        limit = 4000
        data_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"板块资金流向数据 - 同花顺 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.moneyflow_ths(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "ts_code", "name", "pct_change", "latest", "net_amount", "net_d5_amount", "buy_lg_amount", 
                "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process.set_fund_flow_data_ths(stock=stock, df_data=row)
                        data_dicts.append(data_dict)
                print(f"{trade_date} 历史日级资金流向数据 - 同花顺，len(df): {len(df)}, len(data_dicts): {len(data_dicts)}")
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        print(f"完成 {trade_date} 历史日级资金流向数据 - 同花顺，result: {result}")
        return result

    async def save_history_fund_flow_daily_ths_data_by_stock_code(self, stock_code: str) -> Dict:
        """
        保存历史日级资金流向数据 - 同花顺
        接口：moneyflow_ths
        描述：获取同花顺个股资金流向数据，每日盘后更新
        限量：单次最大6000，可根据日期或股票代码循环提取数据
        积分：用户需要至少5000积分才可以调取，具体请参阅积分获取办法
        """
        # 获取历史日级资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_ths(**{
            "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "latest", "net_amount", "net_d5_amount", "buy_lg_amount", 
            "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data_ths(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDailyTHS,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_history_fund_flow_daily_ths_data_by_stock_codes(self, stock_codes: List[str]) -> Dict:
        """
        保存历史日级资金流向数据 - 同花顺
        接口：moneyflow_ths
        描述：获取同花顺个股资金流向数据，每日盘后更新
        限量：单次最大6000，可根据日期或股票代码循环提取数据
        积分：用户需要至少5000积分才可以调取，具体请参阅积分获取办法
        """
        # 获取历史日级资金流向数据 - 同花顺
        stock_codes_str = ",".join(stock_codes)
        df = self.ts_pro.moneyflow_ths(**{
            "ts_code": stock_codes_str, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "latest", "net_amount", "net_d5_amount", "buy_lg_amount", 
            "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data_ths(stock=stock, df_data=row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDailyTHS,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    # ============== 日级资金流向数据 - 东方财富 ==============
    async def save_today_fund_flow_daily_dc_data(self) -> Dict:
        """
        保存今天的日级资金流向数据 - 东方财富
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取日级资金流向数据 - 东方财富
        df = self.ts_pro.moneyflow_dc(**{
            "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
            "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        print(f"今天的日级资金流向数据 - 东方财富 数量: {len(df)}")
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDailyDC,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_yesterday_fund_flow_daily_dc_data(self) -> Dict:
        """
        保存今天的日级资金流向数据 - 东方财富
        """
        # 获取当前日期
        today = datetime.today()
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        day_str = yesterday.strftime('%Y%m%d')
        # 获取日级资金流向数据 - 东方财富
        df = self.ts_pro.moneyflow_dc(**{
            "ts_code": "", "trade_date": day_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
            "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        print(f"今天的日级资金流向数据 - 东方财富 数量: {len(df)}")
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDailyDC,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_history_fund_flow_daily_dc_data_trade_date(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        保存历史日级资金流向数据 - 东方财富
        """
        # 获取历史日级资金流向数据 - 东方财富
        trade_date_str = ""
        start_date_str = ""
        end_date_str = ""
        if trade_date is not None:
            trade_date_str = trade_date.strftime('%Y%m%d')
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        offset = 0
        limit = 4000
        data_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"板块资金流向数据 - 同花顺 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.moneyflow_dc(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
                "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                        data_dicts.append(data_dict)
                print(f"{trade_date} 历史日级资金流向数据 - 东方财富，len(df): {len(df)}, len(data_dicts): {len(data_dicts)}")
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        return result

    async def save_history_fund_flow_daily_dc_data_stock_code(self, stock_code: str) -> Dict:
        """
        保存历史日级资金流向数据 - 东方财富
        接口：moneyflow_dc
        描述：获取东方财富个股资金流向数据，每日盘后更新，数据开始于20230911
        限量：单次最大获取6000条数据，可根据日期或股票代码循环提取数据
        积分：用户需要至少5000积分才可以调取，具体请参阅积分获取办法
        """
        # 获取历史日级资金流向数据 - 东方财富
        df = self.ts_pro.moneyflow_dc(**{
            "ts_code": stock_code, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
            "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDailyDC,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_history_fund_flow_daily_dc_data_stock_codes(self, stock_codes: List[str]) -> Dict:
        """
        保存历史日级资金流向数据 - 东方财富
        接口：moneyflow_dc
        描述：获取东方财富个股资金流向数据，每日盘后更新，数据开始于20230911
        限量：单次最大获取6000条数据，可根据日期或股票代码循环提取数据
        积分：用户需要至少5000积分才可以调取，具体请参阅积分获取办法
        """
        # 获取历史日级资金流向数据 - 东方财富
        stock_codes_str = ",".join(stock_codes)
        df = self.ts_pro.moneyflow_dc(**{
            "ts_code": stock_codes_str, "trade_date": "", "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
            "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowDailyDC,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    # ============== 板块资金流向数据 - 同花顺 ==============
    async def save_today_fund_flow_cnt_ths_data(self) -> Dict:
        """
        保存今天的板块资金流向数据 - 同花顺
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取板块资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_cnt_ths(**{
            "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "lead_stock", "close_price", "pct_change", "industry_index", "company_num", "pct_change_stock", 
            "net_buy_amount", "net_sell_amount", "net_amount"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                ths_index = await self.industry_dao.get_ths_index_by_code(row.ts_code)
                if ths_index:
                    data_dict = self.data_format_process.set_fund_flow_cnt_ths_data(ths_index=ths_index, df_data=row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowCntTHS,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            print(f"完成 {today} 板块资金流向数据（同花顺），result: {result}")
        return result

    async def save_yesterday_fund_flow_cnt_ths_data(self) -> Dict:
        """
        保存今天的板块资金流向数据 - 同花顺
        """
        # 获取当前日期
        today = datetime.today()
        yesterday = today - timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
        # 转换为YYYYMMDD格式
        day_str = yesterday.strftime('%Y%m%d')
        # 获取板块资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_cnt_ths(**{
            "ts_code": "", "trade_date": day_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "lead_stock", "close_price", "pct_change", "industry_index", "company_num", "pct_change_stock", 
            "net_buy_amount", "net_sell_amount", "net_amount"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                ths_index = await self.industry_dao.get_ths_index_by_code(row.ts_code)
                if ths_index:
                    data_dict = self.data_format_process.set_fund_flow_cnt_ths_data(ths_index=ths_index, df_data=row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowCntTHS,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            print(f"完成 {today} 板块资金流向数据（同花顺），result: {result}")
        return result

    async def save_history_fund_flow_cnt_ths_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        保存历史板块资金流向数据 - 同花顺
        """
        trade_date_str = ""
        start_date_str = ""
        end_date_str = ""
        if trade_date is not None:
            trade_date_str = trade_date.strftime('%Y%m%d')
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        # 获取历史板块资金流向数据 - 同花顺
        offset = 0
        limit = 4000
        data_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"板块资金流向数据 - 同花顺 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.moneyflow_cnt_ths(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "ts_code", "name", "lead_stock", "close_price", "pct_change", "industry_index", "company_num", "pct_change_stock", 
                "net_buy_amount", "net_sell_amount", "net_amount"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    ths_index = await self.industry_dao.get_ths_index_by_code(row.ts_code)
                    if ths_index:
                        data_dict = self.data_format_process.set_fund_flow_cnt_ths_data(ths_index=ths_index, df_data=row)
                        data_dicts.append(data_dict)
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        print(f"完成 {trade_date} 板块资金流向数据（同花顺），result: {result}")
        return result

    # ============== 板块资金流向数据 - 东方财富 ==============
    async def save_today_fund_flow_cnt_dc_data(self) -> Dict:
        """
        保存今天的板块资金流向数据 - 东方财富
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取板块资金流向数据 - 东方财富
        df = self.ts_pro.moneyflow_ind_dc(**{
            "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "content_type": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "content_type", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", 
            "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", 
            "buy_sm_amount_rate", "buy_sm_amount_stock", "rank"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_cnt_dc_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowCntDC,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            print(f"完成 {today} 板块资金流向数据（东方财富），result: {result}")
        return result

    async def save_history_fund_flow_cnt_dc_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        保存历史板块资金流向数据 - 东方财富
        """
        trade_date_str = ""
        start_date_str = ""
        end_date_str = ""
        if trade_date is not None:
            trade_date_str = trade_date.strftime('%Y%m%d')
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        # 获取历史板块资金流向数据 - 东方财富
        offset = 0
        limit = 5000
        data_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"历史板块资金流向数据 - 东方财富 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.moneyflow_ind_dc(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "content_type", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", 
                "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", 
                "buy_sm_amount_rate", "buy_sm_amount_stock", "rank"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process.set_fund_flow_cnt_dc_data(stock, row)
                        data_dicts.append(data_dict)
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        print(f"完成 {trade_date} 历史板块资金流向数据（东方财富），result: {result}")
        return result
    
    # ============== 行业资金流向数据 - 同花顺 ==============
    async def save_today_fund_flow_industry_ths_data(self) -> Dict:
        """
        保存今天的行业资金流向数据 - 同花顺
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取行业资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_ind_ths(**{
            "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "industry", "lead_stock", "close", "pct_change", "company_num", "pct_change_stock", "close_price", 
            "net_buy_amount", "net_sell_amount", "net_amount"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                ths_index = await self.industry_dao.get_ths_index_by_code(row.ts_code)
                if ths_index:
                    data_dict = self.data_format_process.set_fund_flow_industry_ths_data(ths_index=ths_index, df_data=row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowIndustryTHS,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            print(f"完成 {today} 板块资金流向数据（同花顺），result: {result}")
        return result

    async def save_history_fund_flow_industry_ths_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        保存历史行业资金流向数据 - 同花顺
        """
        trade_date_str = ""
        start_date_str = ""
        end_date_str = ""
        if trade_date is not None:
            trade_date_str = trade_date.strftime('%Y%m%d')
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        # 获取历史行业资金流向数据 - 同花顺
        offset = 0
        limit = 5000
        data_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"行业资金流向数据 - 同花顺 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.moneyflow_ind_ths(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "ts_code", "industry", "lead_stock", "close", "pct_change", "company_num", "pct_change_stock", "close_price", 
                "net_buy_amount", "net_sell_amount", "net_amount"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    ths_index = await self.industry_dao.get_ths_index_by_code(row.ts_code)
                    if ths_index:
                        data_dict = self.data_format_process.set_fund_flow_industry_ths_data(ths_index=ths_index, df_data=row)
                        data_dicts.append(data_dict)
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowIndustryTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        print(f"完成 {trade_date} 板块资金流向数据（同花顺），result: {result}")
        return result

    # ============== 大盘资金流向数据 - 东方财富 ==============
    async def save_today_fund_flow_market_dc_data(self) -> Dict:
        """
        保存今天的行业资金流向数据 - 东方财富
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取大盘资金流向数据 - 东方财富
        df = self.ts_pro.moneyflow_mkt_dc(**{
            "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "close_sh", "pct_change_sh", "close_sz", "pct_change_sz", "net_buy_amount", "net_buy_amount_rate", 
            "buy_elg_amount", "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", 
            "buy_sm_amount", "buy_sm_amount_rate"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_fund_flow_market_dc_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=FundFlowMarketDc,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_history_fund_flow_market_dc_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        保存历史大盘资金流向数据 - 东方财富
        """
        trade_date_str = ""
        start_date_str = ""
        end_date_str = ""
        if trade_date is not None:
            trade_date_str = trade_date.strftime('%Y%m%d')
        if start_date is not None:
            start_date_str = start_date.strftime('%Y%m%d')
        if end_date is not None:
            end_date_str = end_date.strftime('%Y%m%d')
        # 获取历史大盘资金流向数据 - 东方财富
        offset = 0
        limit = 500
        data_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"板块资金流向数据 - 同花顺 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.moneyflow_mkt_dc(**{
                "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "close_sh", "pct_change_sh", "close_sz", "pct_change_sz", "net_buy_amount", "net_buy_amount_rate", 
                "buy_elg_amount", "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", 
                "buy_sm_amount", "buy_sm_amount_rate"
            ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process.set_fund_flow_market_dc_data(stock, row)
                        data_dicts.append(data_dict)
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowMarketDc,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        print(f"完成 {trade_date} 历史大盘资金流向数据 - 东方财富，result: {result}")
        return result

    # ============== 龙虎榜每日明细 ==============
    async def save_today_lhb_daily_data(self) -> Dict:
        """
        保存今天的龙虎榜每日明细
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取龙虎榜每日明细
        df = self.ts_pro.top_list(**{
            "trade_date": today_str, "ts_code": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "close", "pct_change", "turnover_rate", "amount", "l_sell",
            "l_buy", "l_amount", "net_amount", "net_rate", "amount_rate", "float_values", "reason"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_lhb_daily_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=TopList,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_hisroty_lhb_daily_data(self, trade_date: str) -> Dict:
        """
        保存历史龙虎榜每日数据
        """
        # 获取龙虎榜每日明细
        df = self.ts_pro.top_list(**{
            "trade_date": trade_date, "ts_code": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "close", "pct_change", "turnover_rate", "amount", "l_sell",
            "l_buy", "l_amount", "net_amount", "net_rate", "amount_rate", "float_values", "reason"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_lhb_daily_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=TopList,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
            logger.info(f"{trade_date} 的龙虎榜每日明细保存完成。")
        return result

    # ============== 龙虎榜机构明细 ==============
    async def save_today_lhb_inst_data(self) -> Dict:
        """
        保存今天的龙虎榜机构明细
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        # 获取龙虎榜机构明细
        df = self.ts_pro.top_inst(**{
            "trade_date": today_str, "ts_code": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "exalter", "buy", "buy_rate", "sell", "sell_rate", "net_buy", "side", "reason"
        ])
        result = {}
        if not df.empty:
            data_dicts = []
            for row in df.itertuples():
                stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                if stock:
                    data_dict = self.data_format_process.set_lhb_inst_data(stock, row)
                    data_dicts.append(data_dict)
            result =  await self._save_all_to_db_native_upsert(
                model_class=TopInst,
                data_list=data_dicts,
                unique_fields=['stock', 'trade_time']
            )
        return result

    async def save_hisroty_lhb_inst_data(self, trade_date: str) -> Dict:
        """
        【V2 - 优化版】保存历史龙虎榜机构明细，并支持分页拉取
        
        优化点:
        1. [核心功能] 增加了分页逻辑，通过循环和 offset 参数获取指定日期的全部数据，最大支持10万行。
        2. [核心性能] 使用 `get_stocks_by_codes` 解决了N+1数据库查询问题。
        3. 整合了分页数据，进行统一的批量关联和批量保存。
        4. 增强了代码的健壮性和日志清晰度。
        
        Args:
            trade_date (str): 日期，格式为 YYYYMMDD
        """
        print(f"调试: 开始执行 save_hisroty_lhb_inst_data 任务, trade_date={trade_date}")
        
        try:
            # 1. [代码修改处] 分页循环拉取API数据
            all_dfs = []
            limit = 10000  # Tushare单次最大返回量
            offset = 0
            max_records = 100000 # 最大拉取10万行作为保护

            print(f"调试: 开始分页拉取龙虎榜数据，每页最多 {limit} 条。")
            while True:
                print(f"调试: 正在拉取数据，offset={offset}...")
                df = self.ts_pro.top_inst(**{
                    "trade_date": trade_date,
                    "ts_code": "",
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "trade_date", "ts_code", "exalter", "buy", "buy_rate", "sell", "sell_rate", "net_buy", "side", "reason"
                ])

                if df.empty:
                    print("调试: API返回空数据帧，分页拉取结束。")
                    break
                
                all_dfs.append(df)

                # 如果返回的数据量小于请求的limit，说明已经是最后一页
                if len(df) < limit:
                    print(f"调试: 已获取最后一页数据({len(df)}条)，分页拉取结束。")
                    break
                
                offset += limit
                # 增加保护，防止意外的无限循环
                if offset >= max_records:
                    logger.warning(f"拉取数据已达到 {max_records} 条上限，自动停止。")
                    print(f"调试: 拉取数据已达到 {max_records} 条上限，自动停止。")
                    break
            
            if not all_dfs:
                logger.info(f"交易日 {trade_date} 没有龙虎榜机构明细数据。")
                return {"status": "success", "message": "No data for this trade date.", "saved_count": 0}

            # 合并所有分页数据
            final_df = pd.concat(all_dfs, ignore_index=True)
            print(f"调试: 分页拉取完成，共获取 {len(final_df)} 条数据。")

            # 2. [代码修改处] 批量获取所有相关的股票基础信息对象
            unique_ts_codes = final_df['ts_code'].unique().tolist()
            print(f"调试: 数据涉及 {len(unique_ts_codes)} 个独立股票代码。")
            
            # [代码修改处] 使用 get_stocks_by_codes 方法，一次性查询数据库，解决N+1问题
            # 注意：这里我们用 stock_basic_dao 替换了原先的 stock_cache_get 以实现批量操作
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            print(f"调试: 批量从数据库获取了 {len(stock_map)} 个股票对象。")

            # 3. 准备批量写入的数据
            data_dicts_to_save = []
            for row in final_df.itertuples():
                # [代码修改处] 从预先查好的映射中获取股票对象，高效且无N+1问题
                stock_instance = stock_map.get(row.ts_code)
                
                if stock_instance:
                    data_dict = self.data_format_process.set_lhb_inst_data(stock_instance, row)
                    data_dicts_to_save.append(data_dict)
                else:
                    logger.warning(f"在数据库中未找到股票代码 {row.ts_code} 的基础信息，已跳过该条龙虎榜数据。")

            # 4. 批量写入数据库
            result = {}
            if data_dicts_to_save:
                print(f"调试: 准备批量保存 {len(data_dicts_to_save)} 条龙虎榜数据到数据库...")
                # 注意：请确保 unique_fields 中的字段名与 TopInst 模型中的字段名完全一致。
                # Tushare返回的是 trade_date，如果您的模型中是 trade_time，请确保 set_lhb_inst_data 方法做了转换。
                result = await self._save_all_to_db_native_upsert(
                    model_class=TopInst,
                    data_list=data_dicts_to_save,
                    unique_fields=['stock', 'trade_date'] # 建议使用 trade_date，与API字段保持一致
                )
                print(f"调试: 成功保存 {len(data_dicts_to_save)} 条数据。")
            else:
                logger.info("经过筛选后，没有需要保存到数据库的数据。")
                print("调试: 经过筛选后，没有需要保存的数据。")
                result = {"status": "success", "message": "No new data to save.", "saved_count": 0}
                
            return result

        except Exception as e:
            logger.error(f"保存龙虎榜机构明细时发生严重错误: {e}", exc_info=True)
            print(f"调试: 发生异常: {e}")
            raise

    # ============== 游资每日明细 ==============
    


    # ============== 涨跌停榜单 - 同花顺 无权限，不用 ==============
    async def save_limit_list_ths(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【修改】保存同花顺涨跌停榜单数据 (limit_list_ths接口)
        """
        # --- 参数处理，保持不变 ---
        # 优雅地处理日期参数，如果为None则生成空字符串
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        
        # --- 分页拉取逻辑，保持并优化 ---
        offset = 0
        limit = 500  # Tushare单次最大返回5000，这里用500作为分页大小是稳妥的
        data_dicts = []
        
        print(f"开始拉取同花顺涨跌停榜单数据... trade_date: {trade_date_str}, start_date: {start_date_str}, end_date: {end_date_str}")

        while True:
            # 【新增】增加安全退出机制，防止意外的无限循环
            if offset >= 100000:
                logger.warning(f"同花顺涨跌停榜单 offset已达10万，为安全起见停止拉取。")
                break
            
            try:
                # 【修改】调用正确的Tushare接口：limit_list_ths
                df = self.ts_pro.limit_list_ths(**{
                    "trade_date": trade_date_str,
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "limit": limit,
                    "offset": offset
                    # ts_code, limit_type, market 等参数留空，表示获取全部
                }, fields=[
                    "trade_date", "ts_code", "name", "price", "pct_chg", "open_num",
                    "lu_desc", "limit_type", "tag", "status", "limit_order", "limit_amount",
                    "turnover_rate", "free_float", "lu_limit_order", "limit_up_suc_rate",
                    "turnover", "market_type", "first_lu_time", "last_lu_time",
                    "first_ld_time", "last_ld_time", "rise_rate", "sum_float"
                ])
            except Exception as e:
                # 【新增】增加异常捕获，防止因API问题导致程序崩溃
                logger.error(f"拉取同花顺涨跌停榜单时发生异常: {e}")
                break # 发生异常时，终止循环

            # --- 数据处理逻辑，保持并优化 ---
            if df.empty:
                # 如果当前分页返回数据为空，说明已经拉取完毕
                print(f"拉取完成，在 offset={offset} 处未获取到更多数据。")
                break
            else:
                print(f"成功拉取到 {len(df)} 条数据，offset={offset}, limit={limit}")
                # 【保持】优秀的数据清洗逻辑：将各种空值统一处理为None，便于数据库存储
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)
                
                # 遍历DataFrame，准备存入数据库的数据
                for row in df.itertuples(index=False):
                    # 【保持】通过缓存高效获取关联的StockInfo对象
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    if stock:
                        # 【修改】调用与LimitListThs模型匹配的数据格式化方法
                        # 你需要确保有一个 `set_limit_list_ths_data` 方法来处理数据转换
                        data_dict = self.data_format_process.set_limit_list_ths_data(stock, row)
                        data_dicts.append(data_dict)
                    else:
                        # 【新增】增加日志，记录未找到对应股票信息的情况
                        logger.warning(f"在涨跌停榜单中找到股票 {row.ts_code}，但在StockInfo表中未找到，已跳过。")

            # 【保持】礼貌地请求API，避免请求过于频繁
            time.sleep(0.2)
            
            # 如果返回的数据量小于请求的limit，说明是最后一页，可以提前退出循环
            if len(df) < limit:
                print("已到达数据末尾，退出循环。")
                break
            
            # 准备下一次分页请求
            offset += limit

        # --- 批量入库逻辑，保持并优化 ---
        if not data_dicts:
            # 【新增】如果没有任何数据需要保存，提前告知并返回
            msg = f"没有需要保存的同花顺涨跌停榜单数据。查询参数: trade_date={trade_date_str}, start_date={start_date_str}, end_date={end_date_str}"
            print(msg)
            return {"status": "success", "message": msg, "saved_count": 0}

        # 【修改】调用批量保存方法，并使用正确的模型和唯一键
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitListThs,  # 【修改】使用正确的模型 LimitListThs
            data_list=data_dicts,
            # 【修改】使用与模型定义匹配的联合唯一键
            unique_fields=['trade_date', 'stock', 'limit_type'] 
        )
        
        # 打印最终结果
        print(f"完成同花顺涨跌停榜单数据保存，结果: {result}")
        return result













