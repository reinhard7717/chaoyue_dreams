# dao_manager\tushare_daos\industry_dao.py
import asyncio
import logging
from time import sleep
import time
from django.db.models import F, Window
from django.db.models.functions import RowNumber
from asgiref.sync import sync_to_async
from typing import Any, List, Dict, Optional
from datetime import date, datetime

import numpy as np
import pandas as pd

from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from stock_models.industry import DcIndexDaily, DcIndexMember, SwIndustry, SwIndustryDaily, SwIndustryMember, ThsIndex, ThsIndexMember, ThsIndexDaily, DcIndex
from stock_models.stock_basic import StockInfo
from utils.cache_get import StockInfoCacheGet
from utils.cache_manager import CacheManager
from utils.data_format_process import IndustryFormatProcess

logger = logging.getLogger("dao")
BATCH_SAVE_SIZE = 100000

class IndustryDao(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.index_info_dao = IndexBasicDAO(self.cache_manager)
        self.data_format_process = IndustryFormatProcess(cache_manager_instance)
        self.stock_cache_get = StockInfoCacheGet(self.cache_manager)

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
        
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                index_basic = await self.index_info_dao.get_index_by_code(row.index_code)
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
                index_basic = await self.index_info_dao.get_index_by_code(row.ts_code)
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
        # 从数据库获取
        # 用 sync_to_async 包装整个 list(ThsIndex.objects.all())
        industry_list = await sync_to_async(lambda: list(ThsIndex.objects.all()))()
        return industry_list

    async def get_ths_indices_by_codes(self, codes: list) -> dict:
        """
        根据ts_code列表，一次性从数据库获取所有ThsIndex对象，并返回一个 code -> object 的映射字典。
        """
        if not codes:
            return {}
        # 使用 Django ORM 的异步接口 afilter
        indices = ThsIndex.objects.filter(ts_code__in=codes)
        # 使用 avalues 或 aiterator 进行异步迭代，构建字典
        return {index.ts_code: index async for index in indices}

    async def get_ths_index_by_code(self, index_code: str) -> Optional['ThsIndex']:
        """
        获取指定同花顺概念和行业指数的基本信息
        Args:
            index_code: 同花顺概念和行业指数代码
        Returns:
            ThsIndex: 同花顺概念和行业指数基本信息
        """
        # 从数据库获取
        industry = await sync_to_async(lambda: ThsIndex.objects.filter(ts_code=index_code).first())()
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
                if row.count is None:
                    logger.info(f"count为0: {row}")
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

    # 获取某只股票所属的所有同花顺板块/行业/概念
    async def get_stock_ths_indices(self, stock_code: str) -> list:
        """
        获取某只股票所属的所有同花顺板块/行业/概念
        """
        return await sync_to_async(list)(
            ThsIndexMember.objects.filter(stock__con_code=stock_code, is_new='Y').select_related('ths_index')
        )

    @sync_to_async
    def get_stock_codes_by_industry(self, industry_code: str) -> List[str]:
        """根据同花顺行业代码获取所有成分股代码列表"""
        # print(f"    [DAO] 正在查询行业 {industry_code} 的所有成分股代码...")
        try:
            # 假设 ThsIndexMember 模型通过外键 ths_index 和 stock 关联
            members = ThsIndexMember.objects.filter(
                ths_index__ts_code=industry_code, is_new='Y'
            ).select_related('stock').values_list('stock__stock_code', flat=True)
            return list(members)
        except Exception as e:
            logger.error(f"查询行业 {industry_code} 成分股代码时出错: {e}")
            return []

    async def save_ths_index_member(self) -> Dict:
        """
        【重构修复版】获取同花顺概念板块成分列表并保存
        修复点:
        1.  **根本解决外键约束错误**: 通过修改底层的_save_all_to_db_native_upsert方法，正确处理to_field非主键的场景。
        2.  (保留已有优化) 消除N+1查询, 修复逻辑Bug, 移除冗余查询, 批量保存, 异步优化。
        """
        # --- 1. 初始化和准备 ---
        API_CALL_DELAY_SECONDS = 0.3
        final_result = {}
        data_to_save = []
        # --- 2. 获取所有概念板块信息 ---
        ths_index_list = await self.get_ths_index_list()
        if not ths_index_list:
            logger.warning("数据库中未找到任何同花顺概念板块信息，任务结束。")
            return {"status": "warning", "message": "No ThsIndex found."}
        logger.info(f"开始处理 {len(ths_index_list)} 个同花顺概念板块...")
        # --- 3. 循环调用API，收集原始数据和所有需要的股票代码 ---
        all_raw_members = []
        all_stock_codes = set()
        for i, ths_index in enumerate(ths_index_list):
            print(f"进度: {i+1}/{len(ths_index_list)} | 正在获取板块 [{ths_index.name} ({ths_index.ts_code})] 的成分股...")
            try:
                # 明确指定需要的字段，减少不必要的数据传输
                df = self.ts_pro.ths_member(ts_code=ths_index.ts_code, fields="ts_code,con_code,name,weight,in_date,out_date,is_new")
                if df is None or df.empty:
                    logger.warning(f"板块 [{ths_index.name}] 未返回任何成分股数据，跳过。")
                    await asyncio.sleep(API_CALL_DELAY_SECONDS)
                    continue
                for row in df.itertuples(index=False):
                    all_raw_members.append((ths_index, row))
                    all_stock_codes.add(row.con_code)
            except Exception as e:
                logger.error(f"获取板块 [{ths_index.name}] 成分股时发生API错误: {e}", exc_info=True)
            await asyncio.sleep(API_CALL_DELAY_SECONDS)
        logger.info(f"所有板块API数据获取完成，共 {len(all_raw_members)} 条成分股记录，涉及 {len(all_stock_codes)} 个独立股票。")
        # --- 4. (核心优化) 一次性从数据库获取所有需要的股票信息 ---
        print("正在一次性获取所有涉及的股票信息...")
        stock_queryset = StockInfo.objects.filter(stock_code__in=list(all_stock_codes))
        stock_map = {stock.stock_code: stock async for stock in stock_queryset}
        print(f"成功获取并映射了 {len(stock_map)} 个股票的信息。")
        # --- 5. 组装最终数据并批量保存 ---
        print("开始组装最终数据并准备写入数据库...")
        for ths_index, row_data in all_raw_members:
            stock = stock_map.get(row_data.con_code)
            if not stock:
                # logger.warning(f"在数据库中未找到股票代码为 {row_data.con_code} 的信息，该成分股将被忽略。")
                continue
            cleaned_row_data = {field: getattr(row_data, field) for field in row_data._fields}
            df_temp = pd.Series(cleaned_row_data).replace(['nan', 'NaN', ''], np.nan).where(pd.notnull, None)
            api_data_dict = df_temp.to_dict()
            # 【代码保留】这个步骤依然是好的实践，可以防止API返回的ts_code意外污染数据。
            # 虽然根本问题在下游修复了，但保留此防御性编程措施没有坏处。
            api_data_dict.pop('ts_code', None)
            # 现在传递给处理函数的数据是干净的，不包含冲突的外键信息
            ths_index_member_dict = self.data_format_process.set_ths_index_member_data(
                ths_index=ths_index, # 这个是我们数据库里的对象，是“可信”的
                stock=stock,
                df_data=api_data_dict # 这个是API数据，但已经移除了冲突键
            )
            data_to_save.append(ths_index_member_dict)
            if len(data_to_save) >= BATCH_SAVE_SIZE:
                print(f"数据缓存池达到 {len(data_to_save)} 条，开始批量写入数据库...")
                final_result = await self._save_all_to_db_native_upsert(
                    model_class=ThsIndexMember,
                    data_list=data_to_save,
                    unique_fields=['ths_index', 'stock']
                )
                print(f"批量写入完成。")
                data_to_save.clear()
        if data_to_save:
            print(f"正在保存最后剩余的 {len(data_to_save)} 条数据...")
            final_result = await self._save_all_to_db_native_upsert(
                model_class=ThsIndexMember,
                data_list=data_to_save,
                unique_fields=['ths_index', 'stock']
            )
            print(f"最后的批量写入完成。")
        logger.info("同花顺概念板块成分保存任务全部完成。")
        return final_result

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

    # 获取某个板块/行业/概念在指定日期的行情特征
    async def get_ths_index_daily_feature(self, ts_code: str, trade_date: str) -> dict:
        """
        获取某个板块/行业/概念在指定日期的行情特征
        """
        obj = await sync_to_async(ThsIndexDaily.objects.filter(
            ths_index__ts_code=ts_code, trade_time=trade_date
        ).first)()
        if obj:
            return {
                "close": obj.close,
                "pct_change": obj.pct_change,
                "turnover_rate": obj.turnover_rate,
                "total_mv": obj.total_mv,
                "float_mv": obj.float_mv,
                "pe_ttm": obj.pe_ttm,
                "pb_mrq": obj.pb_mrq,
                # ...可扩展
            }
        return {}

    # 一次性获取所有ths_codes的最后limit条数据
    async def get_latest_n_per_ths_code_async(self, ths_codes, limit: int = 333):
        qs = ThsIndexDaily.objects.filter(
            ths_index__ts_code__in=ths_codes
        ).annotate(
            row_number=Window(
                expression=RowNumber(),
                partition_by=[F('ths_index__ts_code')],
                order_by=F('trade_time').desc()
            )
        ).filter(row_number__lte=limit)
        objs = await sync_to_async(list)(qs)
        return objs

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
                index_basic = await self.index_info_dao.get_index_by_code(row.ts_code)
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
        # 拉取数据
        df = self.ts_pro.ths_daily(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq"
            ])
        ths_index_daily_dicts = []
        if df.empty:
            return {}
        df = df.replace(['nan', 'NaN', ''], np.nan).where(pd.notnull, None)
        # 1. 批量获取所有需要的 ths_index 对象
        all_index_codes = df['ts_code'].unique().tolist()
        ths_index_map = await self.get_ths_indices_by_codes(all_index_codes)
        # 2. 循环组装数据
        for row in df.itertuples(index=False):
            ths_index = ths_index_map.get(row.ts_code)
            if ths_index:
                ths_index_daily_dict = self.data_format_process.set_ths_index_daily_data(ths_index=ths_index, df_data=row)
                ths_index_daily_dicts.append(ths_index_daily_dict)
            else:
                logger.warning(f"在处理日期 {trade_date_str} 的行情时，未在数据库中找到板块 {row.ts_code}。")
        if ths_index_daily_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=ThsIndexDaily,
                data_list=ths_index_daily_dicts,
                unique_fields=['ths_index', 'trade_time']
            )
        return result

    async def _save_ths_index_daily_history_by_index(self, start_date: date, end_date: date = None) -> Dict:
        """
        接口：ths_daily
        描述：获取同花顺板块指数行情。注：数据版权归属同花顺，如做商业用途，请主动联系同花顺，如需帮助请联系微信：waditu_a
        限量：单次最大3000行数据（5000积分），可根据指数代码、日期参数循环提取。
        Args:
        Returns:
            Dict: 保存结果
        """
        start_date_str = start_date.strftime.strftime('%Y%m%d')
        if end_date is None:
            end_date_str = ""
        else:
            end_date_str = end_date.strftime.strftime('%Y%m%d')
        result = {}
        # 拉取数据
        offset = 0
        limit = 3000
        # 拉取数据
        ths_index_daily_dicts = []
        while True:
            if offset >= 100000:
                logger.warning(f"同花顺板块指数行情 offset已达10万，停止拉取。")
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
                    index_basic = await self.index_info_dao.get_index_by_code(row.ts_code)
                    ths_index_daily_dict = self.data_format_process.set_ths_index_daily_data(ths_index=index_basic,df_data=row)
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

    async def save_ths_index_daily_history(self, trade_dates: List[date]) -> Dict:
        """
        【V2.0 按天并行版】
        描述：接收一个交易日列表，并为每一天调用按天获取行情的方法。适配并行任务框架。
        """
        if not trade_dates:
            logger.warning("save_ths_index_daily_history 接收到的交易日列表为空，任务跳过。")
            return {}
        total_results = []
        for trade_date in trade_dates:
            try:
                result = await self.save_ths_index_daily_by_trade_date(trade_date)
                total_results.append(result)
            except Exception as e:
                logger.error(f"在 save_ths_index_daily_history 中处理日期 {trade_date} 时失败: {e}", exc_info=True)
        # 此处可以对 total_results 进行汇总，但对于并行任务，单个日志已足够
        return {"status": "completed", "processed_days": len(trade_dates)}

    # ============== 开盘啦题材与榜单 ============== 
    async def save_kpl_concept_list_by_date(self, trade_date: date) -> Dict:
        """
        接口：kpl_concept
        描述：获取并保存指定交易日的开盘啦题材库列表。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的开盘啦题材库...")
        try:
            df = self.ts_pro.kpl_concept(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 kpl_concept 失败: {e}", exc_info=True)
            return {"status": "error", "message": f"API call failed: {e}"}
        if df is None or df.empty:
            logger.warning(f"Tushare接口 kpl_concept 未返回 {trade_date_str} 的数据。")
            return {"status": "warning", "message": "API returned no data."}
        df = df.replace(['nan', 'NaN', ''], np.nan).where(pd.notnull, None)
        concept_dicts = [
            self.data_format_process.set_kpl_concept_data(df_data=row)
            for row in df.itertuples(index=False)
        ]
        if not concept_dicts:
            return {}
        result = await self._save_all_to_db_native_upsert(
            model_class=KplConcept,
            data_list=concept_dicts,
            unique_fields=['trade_time', 'ts_code']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(concept_dicts)} 条开盘啦题材。")
        return result

    async def save_kpl_concept_members_by_date(self, trade_date: date) -> Dict:
        """
        接口：kpl_concept_cons
        描述：高效获取并保存指定交易日的所有开盘啦题材的成分股。
        采用批量查询、内存映射、批量插入的优化策略。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的开盘啦题材成分股...")
        try:
            df = self.ts_pro.kpl_concept_cons(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 kpl_concept_cons 失败: {e}", exc_info=True)
            return {"status": "error", "message": f"API call failed: {e}"}
        if df is None or df.empty:
            logger.warning(f"Tushare接口 kpl_concept_cons 未返回 {trade_date_str} 的数据。")
            return {"status": "warning", "message": "API returned no data."}
        df = df.replace(['nan', 'NaN', ''], np.nan).where(pd.notnull, None)
        # 1. 收集所有需要的 concept_code 和 stock_code
        all_concept_codes = df['ts_code'].unique().tolist()
        all_stock_codes = df['con_code'].unique().tolist()
        # 2. 一次性从数据库获取所有需要的 KplConcept 和 StockInfo 对象
        concept_qs = KplConcept.objects.filter(ts_code__in=all_concept_codes, trade_time=trade_date_str)
        concept_map = {c.ts_code: c async for c in concept_qs}
        stock_qs = StockInfo.objects.filter(stock_code__in=all_stock_codes)
        stock_map = {s.stock_code: s async for s in stock_qs}
        # 3. 组装最终数据
        members_to_save = []
        for row in df.itertuples(index=False):
            concept = concept_map.get(row.ts_code)
            stock = stock_map.get(row.con_code)
            if concept and stock:
                member_dict = self.data_format_process.set_kpl_concept_member_data(
                    kpl_concept=concept,
                    stock=stock,
                    df_data=row
                )
                members_to_save.append(member_dict)
            else:
                if not concept:
                    logger.warning(f"未在数据库中找到日期为 {trade_date_str} 的题材 {row.ts_code}，成分股 {row.con_code} 将被忽略。")
                if not stock:
                    logger.warning(f"未在数据库中找到股票 {row.con_code}，成分股记录将被忽略。")
        if not members_to_save:
            return {}
        # 4. 批量保存
        result = await self._save_all_to_db_native_upsert(
            model_class=KplConceptConstituent,
            data_list=members_to_save,
            unique_fields=['concept', 'stock', 'trade_time']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(members_to_save)} 条开盘啦题材成分股。")
        return result

    async def save_kpl_list_by_date(self, trade_date: date) -> Dict:
        """
        接口：kpl_list
        描述：获取并保存指定交易日的开盘啦所有榜单数据（涨停、跌停、炸板等）。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的开盘啦榜单数据...")
        # 定义要获取的榜单类型
        tags_to_fetch = ['涨停', '跌停', '炸板', '自然涨停', '竞价']
        all_list_items = []
        for tag in tags_to_fetch:
            try:
                df = self.ts_pro.kpl_list(trade_date=trade_date_str, tag=tag)
                if df is not None and not df.empty:
                    all_list_items.append(df)
                await asyncio.sleep(0.3) # API调用之间稍作停顿
            except Exception as e:
                logger.error(f"调用Tushare接口 kpl_list (tag={tag}) 失败: {e}", exc_info=True)
                continue
        if not all_list_items:
            logger.warning(f"Tushare接口 kpl_list 未返回 {trade_date_str} 的任何榜单数据。")
            return {"status": "warning", "message": "API returned no data for any tag."}
        combined_df = pd.concat(all_list_items, ignore_index=True)
        combined_df = combined_df.replace(['nan', 'NaN', ''], np.nan).where(pd.notnull, None)
        # 批量获取股票信息
        all_stock_codes = combined_df['ts_code'].unique().tolist()
        stock_qs = StockInfo.objects.filter(stock_code__in=all_stock_codes)
        stock_map = {s.stock_code: s async for s in stock_qs}
        # 组装数据
        items_to_save = []
        for row in combined_df.itertuples(index=False):
            stock = stock_map.get(row.ts_code)
            if stock:
                item_dict = self.data_format_process.set_kpl_list_data(stock=stock, df_data=row)
                items_to_save.append(item_dict)
            else:
                logger.warning(f"未在数据库中找到股票 {row.ts_code}，榜单记录将被忽略。")
        if not items_to_save:
            return {}
        # 批量保存
        result = await self._save_all_to_db_native_upsert(
            model_class=KplLimitList,
            data_list=items_to_save,
            unique_fields=['stock', 'trade_date', 'tag']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条开盘啦榜单数据。")
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

    async def get_dc_indices_by_codes(self, codes: list) -> dict:
        """
        根据ts_code列表，一次性从数据库获取所有DcIndex对象，并返回一个 code -> object 的映射字典。
        """
        if not codes:
            return {}
        # 使用 Django ORM 的异步接口 afilter 和异步推导式
        indices = DcIndex.objects.filter(ts_code__in=codes)
        return {index.ts_code: index async for index in indices}

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

    # ============== 市场情绪与涨跌停数据 ==============

    async def save_limit_list_ths_by_date(self, trade_date: date) -> Dict:
        """
        接口：limit_list_ths
        描述：获取并保存指定交易日的同花顺涨跌停榜单数据。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的同花顺涨跌停榜单...")
        # 定义要获取的榜单类型
        limit_types = ['涨停池', '连扳池', '冲刺涨停', '炸板池', '跌停池']
        all_items_df = []
        for l_type in limit_types:
            try:
                df = self.ts_pro.limit_list_ths(trade_date=trade_date_str, limit_type=l_type)
                if df is not None and not df.empty:
                    all_items_df.append(df)
                await asyncio.sleep(0.2) # API调用之间稍作停顿
            except Exception as e:
                logger.error(f"调用Tushare接口 limit_list_ths (limit_type={l_type}) 失败: {e}", exc_info=True)
                continue
        if not all_items_df:
            logger.warning(f"Tushare接口 limit_list_ths 未返回 {trade_date_str} 的任何榜单数据。")
            return {}
        combined_df = pd.concat(all_items_df, ignore_index=True)
        combined_df = combined_df.replace([np.nan, 'nan', 'NaN', ''], None)
        # 批量获取股票信息
        all_stock_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_cache_get.get_stocks_by_codes_map(all_stock_codes)
        # 组装数据
        items_to_save = [
            self.data_format_process.set_limit_list_ths_data(stock=stock_map.get(row.ts_code), df_data=row)
            for row in combined_df.itertuples(index=False) if stock_map.get(row.ts_code)
        ]
        if not items_to_save:
            return {}
        # 批量保存
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitListThs,
            data_list=items_to_save,
            unique_fields=['stock', 'trade_date', 'limit_type']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条同花顺涨跌停数据。")
        return result

    async def save_limit_list_d_by_date(self, trade_date: date) -> Dict:
        """
        接口：limit_list_d
        描述：获取并保存指定交易日的Tushare版A股涨跌停列表数据。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的Tushare涨跌停列表...")
        limit_types = ['U', 'D', 'Z'] # 涨停, 跌停, 炸板
        all_items_df = []
        for l_type in limit_types:
            try:
                df = self.ts_pro.limit_list_d(trade_date=trade_date_str, limit_type=l_type)
                if df is not None and not df.empty:
                    all_items_df.append(df)
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"调用Tushare接口 limit_list_d (limit_type={l_type}) 失败: {e}", exc_info=True)
                continue
        if not all_items_df:
            logger.warning(f"Tushare接口 limit_list_d 未返回 {trade_date_str} 的任何数据。")
            return {}
        combined_df = pd.concat(all_items_df, ignore_index=True)
        combined_df = combined_df.replace([np.nan, 'nan', 'NaN', ''], None)
        all_stock_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_cache_get.get_stocks_by_codes_map(all_stock_codes)
        items_to_save = [
            self.data_format_process.set_limit_list_d_data(stock=stock_map.get(row.ts_code), df_data=row)
            for row in combined_df.itertuples(index=False) if stock_map.get(row.ts_code)
        ]
        if not items_to_save:
            return {}
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitListD,
            data_list=items_to_save,
            unique_fields=['stock', 'trade_date', 'limit']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条Tushare涨跌停数据。")
        return result

    async def save_limit_step_by_date(self, trade_date: date) -> Dict:
        """
        接口：limit_step
        描述：获取并保存指定交易日的连板天梯数据。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的连板天梯数据...")
        try:
            df = self.ts_pro.limit_step(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 limit_step 失败: {e}", exc_info=True)
            return {}
        if df is None or df.empty:
            logger.warning(f"Tushare接口 limit_step 未返回 {trade_date_str} 的数据。")
            return {}
        df = df.replace([np.nan, 'nan', 'NaN', ''], None)
        all_stock_codes = df['ts_code'].unique().tolist()
        stock_map = await self.stock_cache_get.get_stocks_by_codes_map(all_stock_codes)
        items_to_save = [
            self.data_format_process.set_limit_step_data(stock=stock_map.get(row.ts_code), df_data=row)
            for row in df.itertuples(index=False) if stock_map.get(row.ts_code)
        ]
        if not items_to_save:
            return {}
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitStep,
            data_list=items_to_save,
            unique_fields=['stock', 'trade_date']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条连板天梯数据。")
        return result

    async def save_limit_cpt_list_by_date(self, trade_date: date) -> Dict:
        """
        接口：limit_cpt_list
        描述：获取并保存指定交易日的最强板块统计数据。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的最强板块统计...")
        try:
            df = self.ts_pro.limit_cpt_list(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 limit_cpt_list 失败: {e}", exc_info=True)
            return {}
        if df is None or df.empty:
            logger.warning(f"Tushare接口 limit_cpt_list 未返回 {trade_date_str} 的数据。")
            return {}
        df = df.replace([np.nan, 'nan', 'NaN', ''], None)
        all_index_codes = df['ts_code'].unique().tolist()
        index_map = await self.get_ths_indices_by_codes(all_index_codes)
        items_to_save = [
            self.data_format_process.set_limit_cpt_list_data(ths_index=index_map.get(row.ts_code), df_data=row)
            for row in df.itertuples(index=False) if index_map.get(row.ts_code)
        ]
        if not items_to_save:
            return {}
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitCptList,
            data_list=items_to_save,
            unique_fields=['ths_index', 'trade_date']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条最强板块数据。")
        return result
































