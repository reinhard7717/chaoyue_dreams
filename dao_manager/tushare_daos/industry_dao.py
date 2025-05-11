# dao_manager\tushare_daos\industry_dao.py
import logging
from time import sleep
import time
from asgiref.sync import sync_to_async
from typing import Any, List, Dict, Optional
from datetime import date, datetime

import numpy as np
import pandas as pd

from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from stock_models.industry import DcIndexDaily, DcIndexMember, SwIndustry, SwIndustryDaily, SwIndustryMember, ThsIndex, ThsIndexMember, ThsIndexDaily, DcIndex
from utils.cache_get import StockInfoCacheGet
from utils.data_format_process import IndustryFormatProcess

logger = logging.getLogger("dao")

class IndustryDao(BaseDAO):
    def __init__(self):
        super().__init__(None, None, 3600)
        self.data_format_process = IndustryFormatProcess()
        self.stock_cache_get = StockInfoCacheGet()

    # ============== 申万行业分类 ==============
    async def get_swan_industry_list(self) -> List['SwIndustry']:
        """
        获取所有申万行业的基本信息
        Returns:
            List[SwIndustry]: 申万行业基本信息列表
        """
        return_data = []
        # 从数据库获取
        industry_list = await sync_to_async(lambda: list(SwIndustry.objects.filter(is_new='Y').all()))()
        if industry_list:
            for industry in industry_list:
                return_data.append(industry)
        return return_data

    async def get_swan_industry_l1_list(self) -> List['SwIndustry']:
        """
        获取所有申万行业的基本信息
        Returns:
            List[SwIndustry]: 申万行业基本信息列表
        """
        return_data = []
        # 从数据库获取
        industry_list = await sync_to_async(lambda: list(SwIndustry.objects.filter(is_new='Y', level="L1").all()))()
        if industry_list:
            for industry in industry_list:
                return_data.append(industry)
        return return_data

    async def get_swan_industry_by_code(self, index_code: str) -> Optional['SwIndustry']:
        """
        获取指定申万行业的基本信息
        Args:
            index_code: 申万行业代码
        Returns:
            SwIndustry: 申万行业基本信息
        """
        # 从数据库获取
        industry = await sync_to_async(lambda: SwIndustry.objects.filter(index_code=index_code).first())()
        if industry:
            return industry
        return None

    async def save_swan_industry_list(self) -> Dict:
        """
        接口：index_classify
        描述：获取申万行业分类，可以获取申万2014年版本（28个一级分类，104个二级分类，227个三级分类）和2021年本版（31个一级分类，134个二级分类，346个三级分类）列表信息
        权限：用户需2000积分可以调取，具体请参阅积分获取办法
        Returns:
            Dict: 保存结果
        """
        result = {}
        # 拉取数据
        df = self.ts_pro.index_classify(**{
            "index_code": "", "level": "", "src": "", "parent_code": "", "limit": "", "offset": ""
        }, fields=[ "index_code", "industry_name", "level", "industry_code", "is_pub", "parent_code", "src" ])
        industry_dicts = []
        index_info_dao = IndexBasicDAO()
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                index_basic = await index_info_dao.get_index_by_code(row.index_code)
                industry_dict = self.data_format_process.set_sw_industry_data(index=index_basic,df_data=row)
                industry_dicts.append(industry_dict)
        if industry_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=SwIndustry,
                data_list=industry_dicts,
                unique_fields=['index_code', 'src']
            )
        return result

    # ============== 申万行业成分 ==============
    async def get_sw_industry_member(self, industry_code: str) -> List['SwIndustryMember']:
        """
        获取指定申万行业的成分
        Args:
            industry_code: 申万行业代码
        Returns:
            List[IndustryWeight]: 申万行业成分列表
        """
        # 从数据库获取
        industry_weight = await sync_to_async(lambda: SwIndustryMember.objects.filter(industry_code=industry_code).all())()
        return industry_weight

    async def save_sw_industry_member(self) -> Dict:
        """
        接口：index_member_all
        描述：按三级分类提取申万行业成分，可提供某个分类的所有成分，也可按股票代码提取所属分类，参数灵活
        限量：单次最大2000行，总量不限制
        权限：用户需2000积分可调取，积分获取方法请参阅积分获取办法
        Returns:
            Dict: 保存结果
        """
        result = {}
        index_info_dao = IndexBasicDAO()
        industry_member_dicts = []
        # 获取所有申万一级行业
        sw_l1_indexs = await self.get_swan_industry_l1_list()
        # 拉取数据
        for sw_l1_index in sw_l1_indexs:
            df = self.ts_pro.index_member_all(**{
                "l1_code": sw_l1_index.index_code, "l2_code": "", "l3_code": "", "is_new": "", "ts_code": "", "src": "", "limit": "", "offset": ""
                }, fields=[
                    "l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name", "ts_code", "name", "in_date", "out_date", "is_new"
                ])
            if df is not None:
                df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
                for row in df.itertuples():
                    swan_industry = await self.get_swan_industry_by_code(row.l3_code)
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    industry_member_dict = self.data_format_process.set_sw_industry_member_data(sw_industry=swan_industry, stock=stock, df_data=row)
                    industry_member_dicts.append(industry_member_dict)
        if industry_member_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=SwIndustryMember,
                data_list=industry_member_dicts,
                unique_fields=['industry_code', 'ts_code'] 
            )
        return result

    # ============== 申万行业日线行情 ==============
    async def get_sw_industry_daily(self, ts_code: str) -> List['SwIndustryDaily']:
        """
        获取指定申万行业的日线行情
        Args:
            ts_code: 申万行业代码
        Returns:
            List[SwIndustryDaily]: 申万行业日线行情列表
        """
        # 从数据库获取
        industry_daily_basic = await sync_to_async(lambda: SwIndustryDaily.objects.filter(ts_code=ts_code).all())()
        return industry_daily_basic

    async def save_sw_industry_daily(self, trade_time: Any) -> Dict:
        """
        接口：sw_daily
        描述：获取申万行业日线行情（默认是申万2021版行情）
        限量：单次最大4000行数据，可通过指数代码和日期参数循环提取，5000积分可调取
        Returns:
            Dict: 保存结果
        """
        result = {}
        index_basic_dao = IndexBasicDAO()
        industry_daily_basic_dicts = []
        if trade_time is None:
            # 获取当前日期
            trade_time = datetime.today()
        # 转换为YYYYMMDD格式
        trade_time_str = trade_time.strftime('%Y%m%d')
        # 拉取数据
        df = self.ts_pro.sw_daily(**{
                "ts_code": "", "trade_date": trade_time_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "name", "open", "low", "high", "close", "change", "pct_change", "vol",
                "amount", "pe", "pb", "float_mv", "total_mv", "weight"
            ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                index_basic = index_basic_dao.get_index_by_code(row.ts_code)
                industry_daily_basic_dict = self.data_format_process.set_sw_industry_daily_data(index=index_basic,df_data=row)
                industry_daily_basic_dicts.append(industry_daily_basic_dict)
        if industry_daily_basic_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=SwIndustryDaily,
                data_list=industry_daily_basic_dicts,
                unique_fields=['index', 'trade_time']
            )
        return result

    # ============== 同花顺概念和行业指数 ==============
    async def get_ths_index_list(self) -> List['ThsIndex']:
        """
        获取所有同花顺概念和行业指数的基本信息
        Returns:
            List[ThsIndex]: 同花顺概念和行业指数基本信息列表
        """
        return_data = []
        # 从数据库获取
        industry_list = await sync_to_async(lambda: ThsIndex.objects.all())()
        if industry_list:
            for industry in industry_list:
                return_data.append(industry)
        return return_data

    async def get_ths_index_by_code(self, index_code: str) -> Optional['ThsIndex']:
        """
        获取指定同花顺概念和行业指数的基本信息
        Args:
            index_code: 同花顺概念和行业指数代码
        Returns:
            ThsIndex: 同花顺概念和行业指数基本信息
        """
        # 从数据库获取
        industry = await sync_to_async(lambda: ThsIndex.objects.filter(index_code=index_code).first())()
        if industry:
            return industry
        return None

    async def save_ths_index_list(self) -> Dict:
        """
        接口：ths_index
        描述：获取同花顺板块指数。注：数据版权归属同花顺，如做商业用途，请主动联系同花顺，如需帮助请联系微信：waditu_a
        限量：本接口需获得5000积分，单次最大5000，一次可提取全部数据，请勿循环提取。
        Returns:
            Dict: 保存结果
        """
        result = {}
        # 拉取数据
        df = self.ts_pro.ths_index(**{
                "ts_code": "", "exchange": "", "type": "", "name": "", "limit": "", "offset": ""
            }, fields=[ "ts_code", "name", "count", "exchange", "list_date", "type" ])
        industry_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
            df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
            for row in df.itertuples():
                industry_dict = self.data_format_process.set_ths_index_data(df_data=row)
                industry_dicts.append(industry_dict)
        if industry_dicts:
            try:
                # 保存到数据库
                result = await self._save_all_to_db_native_upsert(
                    model_class=ThsIndex,
                    data_list=industry_dicts,
                    unique_fields=['ts_code']
                )
            except Exception as e:
                logger.error("同花顺概念和行业指数保存失败。", exc_info=True)
        return result

    # ============== 同花顺概念板块成分 ==============
    async def get_ths_index_member(self, ts_code: str) -> List['ThsIndexMember']:
        """
        获取指定同花顺概念和行业指数的成分
        Args:
            ts_code: 同花顺概念和行业指数代码
        Returns:
            List[ThsIndexMember]: 同花顺概念和行业指数成分列表
        """
        # 从数据库获取
        ths_index_members = await sync_to_async(lambda: ThsIndexMember.objects.filter(ts_code=ts_code).all())()
        return ths_index_members

    async def save_ths_index_member(self) -> Dict:
        """
        接口：ths_member
        描述：获取同花顺概念板块成分列表注：数据版权归属同花顺，如做商业用途，请主动联系同花顺。
        限量：用户积累5000积分可调取，每分钟可调取200次，可按概念板块代码循环提取所有成分
        Returns:
            Dict: 保存结果
        """
        result = {}
        ths_index_member_dicts = []
        ths_index_list = self.get_ths_index_list()  # 获取所有的ThsIndex model实例列表
        # 拉取数据
        for ths_index in ths_index_list:
            df = self.ts_pro.ths_member(**{
                    "ts_code": ths_index.ts_code, "con_code": "", "offset": "", "limit": ""
                }, fields=[ "ts_code", "con_code", "con_name", "weight", "in_date", "out_date", "is_new" ])
            if df is not None:
                df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
                for row in df.itertuples():
                    ths_index = await self.get_ths_index_by_code(row.ts_code)
                    stock = await self.stock_cache_get.stock_data_by_code(row.con_code)
                    ths_index_member_dict = self.data_format_process.set_ths_index_member_data(ths_index=ths_index, stock=stock, df_data=row)
                    ths_index_member_dicts.append(ths_index_member_dict)
            time.sleep(0.5)
        if ths_index_member_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=ThsIndexMember,
                data_list=ths_index_member_dicts,
                unique_fields=['ths_index', 'stock']
            )
        return result

    # ============== 同花顺板块指数行情 ==============
    async def get_ths_index_daily(self, ts_code: str) -> List['ThsIndexDaily']:
        """
        获取指定同花顺概念和行业指数的日线行情
        Args:
            ts_code: 同花顺概念和行业指数代码
        Returns:
            List[ThsIndexDaily]: 同花顺概念和行业指数日线行情列表
        """
        # 从数据库获取
        ths_index_daily_basic = await sync_to_async(lambda: ThsIndexDaily.objects.filter(ts_code=ts_code).all())()
        return ths_index_daily_basic

    async def save_ths_index_daily_today(self) -> Dict:
        """
        接口：ths_daily
        描述：获取同花顺板块指数行情。注：数据版权归属同花顺，如做商业用途，请主动联系同花顺，如需帮助请联系微信：waditu_a
        限量：单次最大3000行数据（5000积分），可根据指数代码、日期参数循环提取。
        Args:
        Returns:
            Dict: 保存结果
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        result = {}
        index_basic_dao = IndexBasicDAO()
        # 拉取数据
        df = self.ts_pro.ths_daily(**{
                "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq"
            ])
        ths_index_daily_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                index_basic = index_basic_dao.get_index_by_code(row.ts_code)
                ths_index_daily_dict = self.data_format_process.set_ths_index_daily_data(index=index_basic,df_data=row)
                ths_index_daily_dicts.append(ths_index_daily_dict)
        if ths_index_daily_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=ThsIndexDaily,
                data_list=ths_index_daily_dicts,
                unique_fields=['ths_index', 'trade_time']
            )
        return result

    async def save_ths_index_daily_by_trade_date(self, trade_date: date) -> Dict:
        """
        接口：ths_daily
        描述：获取同花顺板块指数行情。注：数据版权归属同花顺，如做商业用途，请主动联系同花顺，如需帮助请联系微信：waditu_a
        限量：单次最大3000行数据（5000积分），可根据指数代码、日期参数循环提取。
        Args:
        Returns:
            Dict: 保存结果
        """
        # 转换为YYYYMMDD格式
        trade_date_str = trade_date.strftime('%Y%m%d')
        result = {}
        index_basic_dao = IndexBasicDAO()
        # 拉取数据
        df = self.ts_pro.ths_daily(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq"
            ])
        ths_index_daily_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                index_basic = index_basic_dao.get_index_by_code(row.ts_code)
                ths_index_daily_dict = self.data_format_process.set_ths_index_daily_data(index=index_basic,df_data=row)
                ths_index_daily_dicts.append(ths_index_daily_dict)
        if ths_index_daily_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=ThsIndexDaily,
                data_list=ths_index_daily_dicts,
                unique_fields=['ths_index', 'trade_time']
            )
        return result

    async def save_ths_index_daily_history(self, start_date: date, end_date: date) -> Dict:
        """
        接口：ths_daily
        描述：获取同花顺板块指数行情。注：数据版权归属同花顺，如做商业用途，请主动联系同花顺，如需帮助请联系微信：waditu_a
        限量：单次最大3000行数据（5000积分），可根据指数代码、日期参数循环提取。
        Args:
        Returns:
            Dict: 保存结果
        """
        start_date_str = start_date.strftime.strftime('%Y%m%d')
        end_date_str = end_date.strftime.strftime('%Y%m%d')
        result = {}
        index_basic_dao = IndexBasicDAO()
        # 拉取数据
        offset = 0
        limit = 3000
        # 拉取数据
        ths_index_daily_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"每日筹码分布 offset已达10万，停止拉取。")
                break
            df = self.ts_pro.ths_daily(**{
                    "ts_code": "", "trade_date": "", "start_date": start_date_str, "end_date": end_date_str, "limit": "", "offset": ""
                }, fields=[
                    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                    "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq"
                ])
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    index_basic = index_basic_dao.get_index_by_code(row.ts_code)
                    ths_index_daily_dict = self.data_format_process.set_ths_index_daily_data(index=index_basic,df_data=row)
                    ths_index_daily_dicts.append(ths_index_daily_dict)
            time.sleep(0.5)
            if len(df) < limit:
                break
            offset += limit
        if ths_index_daily_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=ThsIndexDaily,
                data_list=ths_index_daily_dicts,
                unique_fields=['ths_index', 'trade_time']
            )
        return result

    # ============== 东方财富概念板块 ==============
    async def get_dc_index_list(self) -> List['DcIndex']:
        """
        获取所有东方财富概念板块的基本信息
        Returns:
            List[DcIndex]: 东方财富概念板块基本信息列表
        """
        return_data = []
        # 从数据库获取
        industry_list = await sync_to_async(lambda: DcIndex.objects.all())()
        if industry_list:
            for industry in industry_list:
                return_data.append(industry)
        return return_data

    async def get_dc_index_by_code(self, ts_code: str) -> Optional['DcIndex']:
        """
        获取指定东方财富概念板块的基本信息
        Args:
            ts_code: 东方财富概念板块代码
        Returns:
            DcIndex: 东方财富概念板块基本信息
        """
        # 从数据库获取
        industry = await sync_to_async(lambda: DcIndex.objects.filter(ts_code=ts_code).first())()

    # ============== 东方财富板块指数行情 ==============
    async def get_dc_index_daily(self, ts_code: str) -> List['DcIndexDaily']:
        """
        获取指定东方财富概念板块的日线行情
        Args:
            ts_code: 东方财富概念板块代码
        Returns:
            List[DcIndexDaily]: 东方财富概念板块日线行情列表
        """
        # 从数据库获取
        dc_index_daily_basic = await sync_to_async(lambda: DcIndexDaily.objects.filter(ts_code=ts_code).all())()
        return dc_index_daily_basic

    async def save_dc_index_daily_today(self) -> Dict:
        """
        接口：ths_daily
        描述：获取东方财富概念板块指数行情。
        限量：单次最大3000行数据（5000积分），可根据指数代码、日期参数循环提取。
        Args:
        Returns:
            Dict: 保存结果
        """
        # 获取当前日期
        today = datetime.today()
        # 转换为YYYYMMDD格式
        today_str = today.strftime('%Y%m%d')
        result = {}
        df = self.ts_pro.dc_index(**{
                "ts_code": "", "name": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "name", "leading", "leading_code", "pct_change", "leading_pct", "total_mv", "turnover_rate",
                "up_num", "down_num"
            ])
        dc_index_daily_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                dc_index = await self.get_dc_index_by_code(row.ts_code)
                if dc_index is None:
                    dc_index = self.data_format_process.set_dc_index_data(df_data=row)
                    dc_index_model = DcIndex(**dc_index)
                    await dc_index_model.save()
                dc_index_daily_dict = self.data_format_process.set_dc_index_daily_data(dc_index=dc_index_model, df_data=row)
                dc_index_daily_dicts.append(dc_index_daily_dict)
                dc_index = self.get_dc_index_by_code(row.ts_code)
        if dc_index_daily_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=DcIndexDaily,
                data_list=dc_index_daily_dicts,
                unique_fields=['index', 'trade_time']
            )
        return result

    async def save_dc_index_daily_by_trade_time(self, trade_time: date=None) -> Dict:
        """
        接口：ths_daily
        描述：获取东方财富概念板块指数行情。
        限量：单次最大3000行数据（5000积分），可根据指数代码、日期参数循环提取。
        Args:
        Returns:
            Dict: 保存结果
        """
        if trade_time is None:
            # 获取当前日期
            today = datetime.today()
            # 转换为YYYYMMDD格式
            trade_time_str = today.strftime('%Y%m%d')
        else:
            trade_time_str = trade_time.strftime('%Y%m%d')
        result = {}
        df = self.ts_pro.dc_index(**{
                "ts_code": "", "name": "", "trade_date": trade_time_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "name", "leading", "leading_code", "pct_change", "leading_pct", "total_mv", "turnover_rate",
                "up_num", "down_num"
            ])
        dc_index_daily_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                dc_index = await self.get_dc_index_by_code(row.ts_code)
                if dc_index is None:
                    dc_index = self.data_format_process.set_dc_index_data(df_data=row)
                    dc_index_model = DcIndex(**dc_index)
                    # dc_index_model.exchange = "DC"
                    await dc_index_model.save()
                dc_index_daily_dict = self.data_format_process.set_dc_index_daily_data(dc_index=dc_index_model, df_data=row)
                dc_index_daily_dicts.append(dc_index_daily_dict)
                dc_index = self.get_dc_index_by_code(row.ts_code)
        if dc_index_daily_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=DcIndexDaily,
                data_list=dc_index_daily_dicts,
                unique_fields=['index', 'trade_time']
            )
        return result

    # ============== 东方财富板块成分 ==============
    async def get_dc_index_member(self, ts_code: str) -> List['DcIndexMember']:
        """
        获取指定东方财富概念板块的成分
        Args:
            ts_code: 东方财富概念板块代码
        Returns:
            List[DcIndexMember]: 东方财富概念板块成分列表
        """
        # 从数据库获取
        dc_index_members = await sync_to_async(lambda: DcIndexMember.objects.filter(ts_code=ts_code).all())()
        return dc_index_members

    async def save_dc_index_member_by_ts_code(self, ts_code: str) -> Dict:
        """
        接口：dc_member
        描述：获取东方财富板块每日成分数据，可以根据概念板块代码和交易日期，获取历史成分
        限量：单次最大获取5000条数据，可以通过日期和代码循环获取
        权限：用户积累5000积分可调取，具体请参阅积分获取办法
        Returns:
            Dict: 保存结果
        """
        result = {}
        dc_index_member_dicts = []
        df = self.ts_pro.dc_member(**{
                "trade_date": "", "ts_code": ts_code, "con_code": "", "limit": "", "offset": ""
            }, fields=[ "trade_date", "ts_code", "con_code", "name" ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                dc_index = await self.get_dc_index_by_code(row.ts_code)
                stock = await self.stock_cache_get.stock_data_by_code(row.con_code)
                dc_index_member_dict = self.data_format_process.set_dc_index_member_data(dc_index=dc_index, stock=stock, df_data=row)
                dc_index_member_dicts.append(dc_index_member_dict)
        if dc_index_member_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=DcIndexMember,
                data_list=dc_index_member_dicts,
                unique_fields=['ts_code', 'con_code']
            )
        return result


































