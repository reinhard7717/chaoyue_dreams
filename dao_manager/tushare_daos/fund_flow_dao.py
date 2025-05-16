
import logging
from datetime import date, datetime
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from stock_models.fund_flow import FundFlowCntDC, FundFlowCntTHS, FundFlowDaily, FundFlowDailyDC, FundFlowDailyTHS, FundFlowIndustryTHS, FundFlowMarketDc, TopInst, TopList
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
        保存历史龙虎榜机构明细
        Args:
            trade_date (str): 日期，格式为 YYYYMMDD
        """
        # 获取龙虎榜机构明细
        df = self.ts_pro.top_inst(**{
            "trade_date": trade_date, "ts_code": "", "limit": "", "offset": ""
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















