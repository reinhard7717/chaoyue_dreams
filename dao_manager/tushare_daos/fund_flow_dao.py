
import logging
from datetime import datetime
from typing import Dict, List
from dao_manager.base_dao import BaseDAO
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDaily,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_daily_data_by_trade_date(self, trade_date: str) -> Dict:
        """
        保存历史日级资金流向数据
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        # 获取历史日级资金流向数据
        df = self.ts_pro.moneyflow(**{
            "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount", "buy_md_vol", "buy_md_amount",
            "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount",
            "sell_elg_vol", "sell_elg_amount", "net_mf_vol", "net_mf_amount"
        ])
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDaily,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDaily,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDaily,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result


    # ============== 日级资金流向数据 - 同花顺 ==============
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data_ths(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_daily_ths_data_by_trade_date(self, trade_date: str) -> Dict:
        """
        保存历史日级资金流向数据 - 同花顺
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        # 获取历史日级资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_ths(**{
            "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "latest", "net_amount", "net_d5_amount", "buy_lg_amount", 
            "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data_ths(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data_ths(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data_ths(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_daily_dc_data_trade_date(self, trade_date: str) -> Dict:
        """
        保存历史日级资金流向数据 - 东方财富
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        # 获取历史日级资金流向数据 - 东方财富
        df = self.ts_pro.moneyflow_dc(**{
            "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
            "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ])
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"日级资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"板块资金流向数据获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_cnt_ths_data(row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_cnt_ths_data(self, trade_date: str) -> Dict:
        """
        保存历史板块资金流向数据 - 同花顺
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        # 获取历史板块资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_cnt_ths(**{
            "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "name", "lead_stock", "close_price", "pct_change", "industry_index", "company_num", "pct_change_stock", 
            "net_buy_amount", "net_sell_amount", "net_amount"
        ])
        if df is None:
            logger.error(f"板块资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_cnt_ths_data(row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
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
        if df is None:
            logger.error(f"板块资金流向数据获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_cnt_dc_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_cnt_dc_data(self, trade_date: str) -> Dict:
        """
        保存历史板块资金流向数据 - 东方财富
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        # 获取历史板块资金流向数据 - 东方财富
        df = self.ts_pro.moneyflow_ind_dc(**{
            "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "content_type": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "content_type", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", 
            "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", 
            "buy_sm_amount_rate", "buy_sm_amount_stock", "rank"
        ])
        if df is None:
            logger.error(f"板块资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_cnt_dc_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
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
        if df is None:
            logger.error(f"行业资金流向数据获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_industry_ths_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowIndustryTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_industry_ths_data(self, trade_date: str) -> Dict:
        """
        保存历史行业资金流向数据 - 同花顺
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        # 获取历史行业资金流向数据 - 同花顺
        df = self.ts_pro.moneyflow_ind_ths(**{
            "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "ts_code", "industry", "lead_stock", "close", "pct_change", "company_num", "pct_change_stock", "close_price", 
            "net_buy_amount", "net_sell_amount", "net_amount"
        ])
        if df is None:
            logger.error(f"行业资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_industry_ths_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowIndustryTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
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
        if df is None:
            logger.error(f"大盘资金流向数据获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_market_dc_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowMarketDc,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_market_dc_data(self, trade_date: str) -> Dict:
        """
        保存历史大盘资金流向数据 - 东方财富
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        # 获取历史大盘资金流向数据 - 东方财富
        df = self.ts_pro.moneyflow_mkt_dc(**{
            "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
        }, fields=[
            "trade_date", "close_sh", "pct_change_sh", "close_sz", "pct_change_sz", "net_buy_amount", "net_buy_amount_rate", 
            "buy_elg_amount", "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", 
            "buy_sm_amount", "buy_sm_amount_rate"
        ])
        if df is None:
            logger.error(f"大盘资金流向数据获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_fund_flow_market_dc_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowMarketDc,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
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
        if df is None:
            logger.error(f"龙虎榜每日明细获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_lhb_daily_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=TopList,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"龙虎榜每日明细获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_lhb_daily_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=TopList,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"龙虎榜机构明细获取失败，日期：{today_str}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_lhb_inst_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=TopInst,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
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
        if df is None:
            logger.error(f"龙虎榜机构明细获取失败，日期：{trade_date}")
            return
        data_dicts = []
        for row in df.itertuples():
            stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
            if stock:
                data_dict = self.data_format_process.set_lhb_inst_data(stock, row)
                data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=TopInst,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result















