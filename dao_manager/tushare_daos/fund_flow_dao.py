
import logging
from datetime import datetime
from dao_manager.base_dao import BaseDAO
from stock_models.fund_flow import FundFlowCntDC, FundFlowCntTHS, FundFlowDaily, FundFlowDailyDC, FundFlowDailyTHS, FundFlowIndustryTHS, FundFlowMarketDc
from utils.data_format_process import FundFlowFormatTuShare

logger = logging.getLogger("dao")

class FundFlowDao(BaseDAO):
    def __init__(self):
        super().__init__()
        from utils.cache_get import StockInfoCacheGet, UserCacheGet
        from utils.cache_set import StockCacheSet, UserCacheSet
        from utils.cash_key import StockCashKey

        self.data_format_process = FundFlowFormatTuShare()
        self.stock_cache_key = StockCashKey()
        self.stock_cache_set = StockCacheSet()
        self.stock_cache_get = StockInfoCacheGet()
        self.user_cache_set = UserCacheSet()
        self.user_cache_get = UserCacheGet()

    # ============== 日级资金流向数据 ==============
    async def save_today_fund_flow_daily_data(self) -> None:
        """
        保存今天的日级资金流向数据
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_data(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDaily,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_daily_data(self, trade_date: str) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_data(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDaily,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    # ============== 日级资金流向数据 - 同花顺 ==============
    async def save_today_fund_flow_daily_ths_data(self) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_data_ths(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_daily_ths_data(self, trade_date: str) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_data_ths(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result
    
    # ============== 日级资金流向数据 - 东方财富 ==============
    async def save_today_fund_flow_daily_dc_data(self) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_daily_dc_data(self, trade_date: str) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    # ============== 板块资金流向数据 - 同花顺 ==============
    async def save_today_fund_flow_cnt_ths_data(self) -> None:
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
            data_dict = self.data_format_process.set_fund_flow_cnt_ths_data(row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_cnt_ths_data(self, trade_date: str) -> None:
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
            data_dict = self.data_format_process.set_fund_flow_cnt_ths_data(row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    # ============== 板块资金流向数据 - 东方财富 ==============
    async def save_today_fund_flow_cnt_dc_data(self) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_cnt_dc_data(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_cnt_dc_data(self, trade_date: str) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_cnt_dc_data(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result
    
    # ============== 行业资金流向数据 - 同花顺 ==============
    async def save_today_fund_flow_industry_ths_data(self) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_industry_ths_data(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowIndustryTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_industry_ths_data(self, trade_date: str) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_industry_ths_data(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowIndustryTHS,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    # ============== 大盘资金流向数据 - 东方财富 ==============
    async def save_today_fund_flow_market_dc_data(self) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_market_dc_data(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowMarketDc,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result

    async def save_history_fund_flow_market_dc_data(self, trade_date: str) -> None:
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
            if stock is None:
                logger.error(f"股票代码：{row.ts_code} 不存在")
                continue
            data_dict = self.data_format_process.set_fund_flow_market_dc_data(stock, row)
            data_dicts.append(data_dict)
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowMarketDc,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_date']
        )
        return result





