# dao_manager\tushare_daos\industry_dao.py
import os
import asyncio
import logging
from time import sleep
import time
from django.db.models import Q, F, Window, Max
from django.db.models.functions import RowNumber
from asgiref.sync import sync_to_async
from typing import Any, List, Dict, Optional
from datetime import date, datetime
from utils.rate_limiter import rate_limiter_factory
import numpy as np
import pandas as pd
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from utils.cash_key import StockCashKey # 导入 StockCashKey
from utils import cache_constants as cc # 导入 cache_constants
from stock_models.market import LimitCptList, LimitListD, LimitListThs, LimitStep
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from stock_models.industry import (
    ConceptMaster, ConceptDaily, ConceptMember,
    IndustryLifecycle, KplConceptInfo, KplConceptDaily, KplConceptConstituent, KplLimitList, 
    DcIndexDaily, DcIndexMember, SwIndustry, SwIndustryDaily, SwIndustryMember, 
    ThsIndex, ThsIndexMember, ThsIndexDaily, DcIndex
)
from stock_models.stock_basic import StockInfo
from utils.cache_get import StockInfoCacheGet
from utils.cache_manager import CacheManager
from utils.data_format_process import IndustryFormatProcess, MarketFormatProcess

logger = logging.getLogger("dao")
BATCH_SAVE_SIZE = 100000

class IndustryDao(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.index_info_dao = IndexBasicDAO(self.cache_manager)
        self.data_format_process = IndustryFormatProcess(cache_manager_instance)
        self.stock_cache_get = StockInfoCacheGet(self.cache_manager)
        self.stock_basic_info_dao = StockBasicInfoDao(self.cache_manager)
        self.market_format_process = MarketFormatProcess(self.cache_manager)
        self.cache_key_stock = StockCashKey()

    # ============== 申万行业分类 ==============
    async def get_swan_industry_list(self) -> List['SwIndustry']:
        """
        获取所有申万行业的基本信息
        Returns:
            List[SwIndustry]: 申万行业基本信息列表
        """
        return_data = []
        # 从数据库获取
        industry_list = await sync_to_async(lambda: list(SwIndustry.objects.all()))()
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
        industry_list = await sync_to_async(lambda: list(SwIndustry.objects.filter(level="L1").all()))()
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
        【V2.0 向量化与N+1优化版】保存申万行业分类列表
        - 核心优化:
          1. 【消除N+1查询】通过一次性批量获取所有 `IndexInfo` 对象，彻底解决了原先在循环中频繁查询数据库的性能瓶颈。
          2. 【向量化处理】使用Pandas的向量化操作替代了原有的 `itertuples()` 循环，大幅提升了数据处理效率。
        """
        # 拉取数据
        df = self.ts_pro.index_classify(**{
            "index_code": "", "level": "", "src": "", "parent_code": "", "limit": "", "offset": ""
        }, fields=["index_code", "industry_name", "level", "industry_code", "is_pub", "parent_code", "src"])
        if df is None or df.empty:
            return {}
        # --- 开始向量化处理 ---
        # 1. 数据清洗
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        df.dropna(subset=['index_code'], inplace=True)
        if df.empty:
            return {}
        # 2. 批量获取关联对象 (消除N+1查询)
        unique_index_codes = df['index_code'].unique().tolist()
        index_map = await self.index_info_dao.get_indices_by_codes(unique_index_codes)
        # 3. 向量化映射与过滤
        df['index'] = df['index_code'].map(index_map)
        df.dropna(subset=['index'], inplace=True)
        if df.empty:
            logger.warning("所有申万行业数据都无法关联到已知的指数信息，任务终止。")
            return {}
        # 4. 向量化重命名与列选择
        df.rename(columns={'industry_name': 'name'}, inplace=True)
        final_df = df[['index', 'name', 'level', 'parent_code', 'src']]
        # 5. 转换为字典列表，并将NaN转为None
        industry_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
        # --- 向量化处理结束，原有的itertuples()循环已被移除 ---
        if not industry_dicts:
            return {}
        # 保存到数据库
        result = await self._save_all_to_db_native_upsert(
            model_class=SwIndustry,
            data_list=industry_dicts,
            unique_fields=['index', 'src'] # 优化：直接使用index对象作为唯一键
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
        【V3.0 - 向量化重构版】获取申万行业成分并保存
        优化：
        1. 移除所有 Python for 循环，改用 Pandas 向量化操作。
        2. 批量提取并创建缺失的 L1/L2/L3 行业，消除循环内的 get_or_create。
        3. 使用 melt 转换数据结构，批量生成 ConceptMember 同步数据。
        """
        print("  - [DAO] 开始获取申万行业成分...")
        API_CALL_DELAY_SECONDS = 0.3
        all_dfs = []
        
        # 1. 获取所有L1行业并循环调用API (IO密集型，保持循环)
        sw_l1_indexs = await self.get_swan_industry_l1_list()
        if not sw_l1_indexs:
            logger.warning("数据库中未找到任何申万L1行业信息，任务结束。")
            return {"status": "warning", "message": "No SW L1 Index found."}
            
        for i, sw_l1_index in enumerate(sw_l1_indexs):
            print(f"    - 进度: {i+1}/{len(sw_l1_indexs)} | 获取L1行业 [{sw_l1_index.industry_name}] 的成分...")
            try:
                df = self.ts_pro.index_member_all(
                    l1_code=sw_l1_index.index_code,
                    is_new='Y',
                    fields="l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new"
                )
                if df is not None and not df.empty:
                    all_dfs.append(df)
                await asyncio.sleep(API_CALL_DELAY_SECONDS)
            except Exception as e:
                logger.error(f"获取申万行业 [{sw_l1_index.industry_name}] 成分时API错误: {e}")
                continue

        if not all_dfs:
            return {}
            
        # --- 开始向量化处理 ---
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        # 2. 动态创建缺失的行业 (L1, L2, L3)
        # 提取所有涉及的行业代码
        all_industry_codes = set(full_df['l1_code'].dropna()) | set(full_df['l2_code'].dropna()) | set(full_df['l3_code'].dropna())
        
        # 批量获取已存在的行业
        existing_industries = await sync_to_async(list)(SwIndustry.objects.filter(index_code__in=all_industry_codes))
        existing_map = {ind.index_code: ind for ind in existing_industries}
        
        new_industries = []
        # L1
        l1_df = full_df[['l1_code', 'l1_name']].dropna().drop_duplicates('l1_code')
        for row in l1_df.itertuples(index=False):
            if row.l1_code not in existing_map:
                new_industries.append(SwIndustry(index_code=row.l1_code, industry_name=row.l1_name, level='L1', parent_code='0', src='SW2021', industry_code=row.l1_code))
                existing_map[row.l1_code] = True # 标记为已处理
        
        # L2
        l2_df = full_df[['l2_code', 'l2_name', 'l1_code']].dropna().drop_duplicates('l2_code')
        for row in l2_df.itertuples(index=False):
            if row.l2_code not in existing_map:
                new_industries.append(SwIndustry(index_code=row.l2_code, industry_name=row.l2_name, level='L2', parent_code=row.l1_code, src='SW2021', industry_code=row.l2_code))
                existing_map[row.l2_code] = True

        # L3
        l3_df = full_df[['l3_code', 'l3_name', 'l2_code']].dropna().drop_duplicates('l3_code')
        for row in l3_df.itertuples(index=False):
            if row.l3_code not in existing_map:
                new_industries.append(SwIndustry(index_code=row.l3_code, industry_name=row.l3_name, level='L3', parent_code=row.l2_code, src='SW2021', industry_code=row.l3_code))
                existing_map[row.l3_code] = True
                
        if new_industries:
            print(f"    - 批量创建 {len(new_industries)} 个新申万行业...")
            await SwIndustry.objects.abulk_create(new_industries, ignore_conflicts=True)
            # 重新获取完整映射
            existing_industries = await sync_to_async(list)(SwIndustry.objects.filter(index_code__in=all_industry_codes))
            existing_map = {ind.index_code: ind for ind in existing_industries}

        # 3. 准备 SwIndustryMember 数据
        # 批量获取股票
        all_stock_codes = full_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        
        # 映射对象
        full_df['l3_industry'] = full_df['l3_code'].map(existing_map)
        full_df['stock'] = full_df['ts_code'].map(stock_map)
        
        # 过滤无效数据
        valid_df = full_df.dropna(subset=['l3_industry', 'stock'])
        
        # 转换日期
        valid_df['in_date'] = pd.to_datetime(valid_df['in_date']).dt.date
        valid_df['out_date'] = pd.to_datetime(valid_df['out_date'], errors='coerce').dt.date
        
        # 构建 SwIndustryMember 字典列表
        member_df = valid_df[['l3_industry', 'stock', 'in_date', 'out_date']].copy()
        member_df['weight'] = None
        member_df['is_new'] = 'Y'
        industry_member_dicts = member_df.where(pd.notnull(member_df), None).to_dict('records')

        # 4. 准备 ConceptMember 同步数据 (使用 melt 向量化处理 L1/L2/L3)
        # 保留需要的列
        sync_df = valid_df[['ts_code', 'stock', 'in_date', 'out_date', 'l1_code', 'l2_code', 'l3_code']].copy()
        # 熔断 (Melt)
        melted_df = sync_df.melt(
            id_vars=['ts_code', 'stock', 'in_date', 'out_date'], 
            value_vars=['l1_code', 'l2_code', 'l3_code'],
            value_name='concept_code'
        ).dropna(subset=['concept_code'])
        
        melted_df['source'] = 'sw'
        # 转换为字典列表
        concept_member_sync_list = melted_df[['concept_code', 'stock', 'source', 'in_date', 'out_date']].where(pd.notnull(melted_df), None).to_dict('records')

        # 5. 保存
        final_result = {}
        if industry_member_dicts:
            print(f"    - 准备保存 {len(industry_member_dicts)} 条申万行业成分数据...")
            final_result = await self._save_all_to_db_native_upsert(
                model_class=SwIndustryMember,
                data_list=industry_member_dicts,
                unique_fields=['l3_industry', 'stock', 'in_date']
            )
            await self._sync_to_concept_member(concept_member_sync_list, 'sw')
            
        return final_result

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

    async def save_sw_industry_daily(self, trade_date: Any = None) -> Dict:
        """
        【V3.1 向量化与N+1优化版】获取申万行业日线行情，并同步到 ConceptDaily。
        - 核心优化:
          1. 【消除N+1查询】对于API返回的、数据库中不存在的新指数，采用批量创建模式，避免了在循环中逐个创建。
          2. 【向量化处理】使用Pandas的向量化操作替代了 `itertuples()` 循环，大幅提升了数据处理效率。
        """
        if trade_date is None:
            trade_date = datetime.today().date()
        elif isinstance(trade_date, datetime):
            trade_date = trade_date.date()
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  -> [申万日线] 开始获取 {trade_date_str} 的数据...")
        try:
            df = self.ts_pro.sw_daily(trade_date=trade_date_str, fields=[
                "ts_code", "trade_date", "name", "open", "low", "high", "close", "change", "pct_change", "vol",
                "amount", "pe", "pb", "float_mv", "total_mv", "weight"
            ])
        except Exception as e:
            logger.error(f"调用Tushare接口 sw_daily 失败: {e}", exc_info=True)
            return {"status": "error", "message": f"API call failed: {e}"}
        if df is None or df.empty:
            return {}
        # --- 开始向量化处理 ---
        df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
        if df.empty:
            return {}
        # 1. 批量获取已存在的指数信息
        all_index_codes = df['ts_code'].unique().tolist()
        index_map = await self.index_info_dao.get_indices_by_codes(all_index_codes)
        # 2. 识别并批量创建缺失的指数信息 (消除N+1)
        missing_codes_df = df[~df['ts_code'].isin(index_map.keys())][['ts_code', 'name']].drop_duplicates()
        if not missing_codes_df.empty:
            print(f"     ...发现 {len(missing_codes_df)} 个新的申万指数，将进行批量创建...")
            new_indices_to_create = [
                {'index_code': row['ts_code'], 'name': row['name'], 'market': 'SW', 'publisher': '申万指数'}
                for _, row in missing_codes_df.iterrows()
            ]
            await self._save_all_to_db_native_upsert(
                model_class=IndexInfo, data_list=new_indices_to_create, unique_fields=['index_code']
            )
            # 重新获取完整的指数映射
            index_map = await self.index_info_dao.get_indices_by_codes(all_index_codes)
        # 3. 向量化映射、转换和选择
        df['index'] = df['ts_code'].map(index_map)
        df.dropna(subset=['index'], inplace=True)
        if df.empty: return {}
        df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
        numeric_cols = ['open', 'low', 'high', 'close', 'change', 'pct_change', 'vol', 'amount', 'pe', 'pb', 'float_mv', 'total_mv', 'weight']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        model_cols = ['index', 'trade_time', 'name'] + numeric_cols
        final_df = df[[col for col in model_cols if col in df.columns]]
        industry_daily_basic_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
        # --- 向量化处理结束 ---
        result = {}
        if industry_daily_basic_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=SwIndustryDaily,
                data_list=industry_daily_basic_dicts,
                unique_fields=['index', 'trade_time']
            )
            print(f"     ...成功保存 {len(industry_daily_basic_dicts)} 条记录到 [SwIndustryDaily] 表。")
            # --- 后续同步逻辑保持不变 ---
            print(f"     -> [同步任务] 开始同步 {trade_date_str} 的申万日线行情到 ConceptDaily...")
            concept_map = await self.get_concepts_by_codes(all_index_codes)
            concept_daily_to_save = [
                self.data_format_process.adapt_to_concept_daily('sw', item, concept_map.get(item['index'].index_code))
                for item in industry_daily_basic_dicts if concept_map.get(item['index'].index_code)
            ]
            if concept_daily_to_save:
                await ConceptDaily.objects.abulk_create(concept_daily_to_save, ignore_conflicts=True)
                print(f"        ...同步完成，处理 {len(concept_daily_to_save)} 条记录到 [ConceptDaily] 表。")
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
        【V2.0 向量化优化版】保存同花顺板块指数列表
        - 核心优化: 使用Pandas的向量化操作替代了原有的 `itertuples()` 循环，大幅提升了数据处理效率。
        """
        # 拉取数据
        df = self.ts_pro.ths_index(**{
            "ts_code": "", "exchange": "", "type": "", "name": "", "limit": "", "offset": ""
        }, fields=["ts_code", "name", "count", "exchange", "list_date", "type"])
        if df is None or df.empty:
            return {}
        # --- 开始向量化处理 ---
        # 1. 数据清洗
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 2. 向量化类型转换
        df['count'] = pd.to_numeric(df['count'], errors='coerce').astype('Int64') # 使用可空整数类型
        df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce').dt.date
        # 3. 转换为字典列表，并将Pandas的NA/NaT值转为Python的None
        industry_dicts = df.where(pd.notnull(df), None).to_dict('records')
        # --- 向量化处理结束，原有的itertuples()循环已被移除 ---
        if not industry_dicts:
            return {}
        try:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=ThsIndex,
                data_list=industry_dicts,
                unique_fields=['ts_code']
            )
            return result
        except Exception as e:
            logger.error("同花顺概念和行业指数保存失败。", exc_info=True)
            return {}

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
            ThsIndexMember.objects.filter(stock__stock_code=stock_code, is_new='Y').select_related('ths_index')
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
        【V3.1 - 向量化版】获取同花顺概念板块成分列表并保存
        优化：
        1. 使用 Pandas 向量化处理替代 Python 循环，大幅提升数据组装速度。
        2. 保持“先删后插”的事务逻辑，确保数据一致性。
        """
        final_result = {}
        limiter = rate_limiter_factory.get_limiter(name='api_ths_member')
        today = datetime.now().date()
        PROXY_IN_DATE = date(1990, 1, 1)
        
        # 1. 获取所有概念板块
        ths_index_list = await self.get_ths_index_list()
        if not ths_index_list:
            logger.warning("数据库中未找到任何同花顺概念板块信息，任务结束。")
            return {"status": "warning", "message": "No ThsIndex found."}
        logger.info(f"开始处理 {len(ths_index_list)} 个同花顺概念板块...")
        
        all_dfs = []
        all_ths_index_codes = set()
        
        # 2. 循环调用 API (IO密集型，保持循环)
        for i, ths_index in enumerate(ths_index_list):
            print(f"进度: {i+1}/{len(ths_index_list)} | [同花顺概念] 获取 [{ths_index.name}] 成分...")
            all_ths_index_codes.add(ths_index.ts_code)
            offset = 0
            limit = 6000
            while True:
                if offset >= 100000: break
                try:
                    while not await limiter.acquire():
                        await asyncio.sleep(20)
                    df = self.ts_pro.ths_member(ts_code=ths_index.ts_code, fields="ts_code,con_code,con_name", limit=limit, offset=offset)
                    if df is None or df.empty: break
                    
                    # 直接在 DF 中标记所属板块对象，避免后续查找
                    df['ths_index'] = ths_index
                    all_dfs.append(df)
                    
                    if len(df) < limit: break
                    offset += limit
                except Exception as e:
                    logger.error(f"获取板块 [{ths_index.name}] 成分失败: {e}")
                    break
        
        if not all_dfs:
            return {"status": "completed", "saved_count": 0}
            
        # --- 向量化处理 ---
        # 3. 合并数据
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        # 4. 批量映射股票 (消除 N+1)
        all_stock_codes = full_df['con_code'].unique().tolist()
        print(f"正在映射 {len(all_stock_codes)} 个股票信息...")
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        
        full_df['stock'] = full_df['con_code'].map(stock_map)
        valid_df = full_df.dropna(subset=['stock']).copy()
        
        if valid_df.empty:
            return {"status": "completed", "saved_count": 0}
            
        # 5. 准备 ThsIndexMember 数据
        valid_df['weight'] = None
        valid_df['in_date'] = today
        valid_df['out_date'] = None
        valid_df['is_new'] = 'Y'
        
        # 使用 to_dict 转为字典列表，再构建模型实例
        ths_records = valid_df[['ths_index', 'stock', 'weight', 'in_date', 'out_date', 'is_new']].to_dict('records')
        # 列表推导式构建实例比循环 append 快
        data_to_save_ths = [ThsIndexMember(**record) for record in ths_records]
        
        # 6. 准备 ConceptMember 同步数据
        sync_df = valid_df[['ts_code', 'stock']].copy()
        sync_df.rename(columns={'ts_code': 'concept_code'}, inplace=True)
        sync_df['source'] = 'ths'
        sync_df['in_date'] = PROXY_IN_DATE
        sync_df['out_date'] = None
        
        concept_member_sync_list = sync_df.to_dict('records')
        
        # 7. 执行数据库操作
        if data_to_save_ths:
            print(f"准备清空并写入 {len(data_to_save_ths)} 条最新的同花顺成分数据...")
            
            # 清空旧数据
            await sync_to_async(ThsIndexMember.objects.filter(ths_index__ts_code__in=all_ths_index_codes).delete)()
            print("  - 旧的 ThsIndexMember 数据已清空。")
            
            # 批量插入
            await ThsIndexMember.objects.abulk_create(data_to_save_ths, batch_size=5000)
            print(f"  - 批量写入完成。")
            
            # 同步 ConceptMember
            await sync_to_async(ConceptMember.objects.filter(concept__code__in=all_ths_index_codes, source='ths').delete)()
            print("  - 旧的 ConceptMember (ths来源) 数据已清空。")
            
            await self._sync_to_concept_member(concept_member_sync_list, 'ths')
            
        return {"status": "completed", "saved_count": len(data_to_save_ths)}

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

    async def get_ths_index_daily_for_range(self, ts_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        根据代码和日期范围获取同花顺板块指数行情。
        """
        qs = ThsIndexDaily.objects.filter(
            ths_index__ts_code=ts_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date
        ).order_by('trade_time')
        # 使用 avalues 异步获取数据
        data = await sync_to_async(list)(qs.values())
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(data)
        df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
        df.set_index('trade_time', inplace=True)
        return df

    async def get_stock_ths_industry_info(self, stock_code: str) -> Optional[Dict[str, str]]:
        """
        根据股票代码获取其当前所属的同花顺行业/概念板块信息。
        """
        print(f"      - [DAO查询] 正在查询股票 {stock_code} 的所属行业...")
        try:
            # 筛选 is_new='Y' 表示当前最新的成分关系
            membership = await ThsIndexMember.objects.select_related('ths_index').filter(
                stock__stock_code=stock_code,
                is_new='Y'
            ).afirst()
            if membership and membership.ths_index:
                industry_info = {
                    'code': membership.ths_index.ts_code,
                    'name': membership.ths_index.name
                }
                print(f"      - [DAO查询] 成功找到 {stock_code} 所属行业: {industry_info['name']} ({industry_info['code']})")
                return industry_info
            else:
                print(f"      - [DAO查询] 未能找到 {stock_code} 的当前所属行业。")
                return None
        except Exception as e:
            logger.error(f"查询股票 {stock_code} 的行业信息时发生数据库错误: {e}", exc_info=True)
            return None

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
        【V2.0 - 向量化版】接口：ths_daily
        描述：获取同花顺板块指数今日行情。
        """
        today = datetime.today()
        today_str = today.strftime('%Y%m%d')
        
        df = self.ts_pro.ths_daily(**{
                "ts_code": "", "trade_date": today_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq"
            ])
            
        if df is None or df.empty:
            return {}
            
        df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        all_index_codes = df['ts_code'].unique().tolist()
        ths_index_map = await self.get_ths_indices_by_codes(all_index_codes)
        
        df['ths_index'] = df['ts_code'].map(ths_index_map)
        df.dropna(subset=['ths_index'], inplace=True)
        
        if df.empty:
            return {}
            
        df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
        
        # 数值转换
        numeric_cols = ["open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        model_cols = ['ths_index', 'trade_time'] + numeric_cols
        final_df = df[[c for c in model_cols if c in df.columns]]
        
        ths_index_daily_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
        
        result = await self._save_all_to_db_native_upsert(
            model_class=ThsIndexDaily,
            data_list=ths_index_daily_dicts,
            unique_fields=['ths_index', 'trade_time']
        )
        return result

    async def save_ths_index_daily_by_trade_date(self, trade_date: date) -> Dict:
        """
        【V3.1 向量化优化版】获取同花顺板块日线行情，并同步到 ConceptDaily。
        - 核心优化: 使用Pandas的向量化操作替代了 `itertuples()` 循环，大幅提升了数据处理效率。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  -> [同花顺日线] 开始获取 {trade_date_str} 的数据...")
        try:
            df = self.ts_pro.ths_daily(trade_date=trade_date_str, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq", "amount"
            ])
        except Exception as e:
            logger.error(f"调用Tushare接口 ths_daily 失败: {e}", exc_info=True)
            return {"status": "error", "message": f"API call failed: {e}"}
        if df is None or df.empty:
            logger.warning(f"Tushare接口 ths_daily 未返回 {trade_date_str} 的数据。")
            return {}
        # --- 开始向量化处理 ---
        df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        df.dropna(subset=['ts_code'], inplace=True)
        if df.empty: return {}
        # 1. 批量获取关联对象
        all_index_codes = df['ts_code'].unique().tolist()
        ths_index_map = await self.get_ths_indices_by_codes(all_index_codes)
        # 2. 向量化映射、转换和选择
        df['ths_index'] = df['ts_code'].map(ths_index_map)
        df.dropna(subset=['ths_index'], inplace=True)
        if df.empty: return {}
        df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
        numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'avg_price', 'change', 'pct_change', 'vol', 'turnover_rate', 'total_mv', 'float_mv', 'pe_ttm', 'pb_mrq', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        model_cols = ['ths_index', 'trade_time'] + numeric_cols
        final_df = df[[col for col in model_cols if col in df.columns]]
        ths_index_daily_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
        # --- 向量化处理结束 ---
        result = {}
        if ths_index_daily_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=ThsIndexDaily,
                data_list=ths_index_daily_dicts,
                unique_fields=['ths_index', 'trade_time']
            )
            print(f"     ...成功保存 {len(ths_index_daily_dicts)} 条记录到 [ThsIndexDaily] 表。")
            # --- 后续同步逻辑保持不变 ---
            print(f"     -> [同步任务] 开始同步 {trade_date_str} 的同花顺日线行情到 ConceptDaily...")
            concept_map = await self.get_concepts_by_codes(all_index_codes)
            concept_daily_to_save = [
                self.data_format_process.adapt_to_concept_daily('ths', item, concept_map.get(item['ths_index'].ts_code))
                for item in ths_index_daily_dicts if concept_map.get(item['ths_index'].ts_code)
            ]
            if concept_daily_to_save:
                await ConceptDaily.objects.abulk_create(concept_daily_to_save, ignore_conflicts=True)
                print(f"        ...同步完成，处理 {len(concept_daily_to_save)} 条记录到 [ConceptDaily] 表。")
        return result

    async def _save_ths_index_daily_history_by_index(self, start_date: date, end_date: date = None) -> Dict:
        """
        【V2.0 - 向量化版】接口：ths_daily
        描述：获取同花顺板块指数历史行情。
        优化：向量化处理，消除 N+1 查询。
        """
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        
        offset = 0
        limit = 3000
        all_dfs = []
        
        # 1. 循环拉取数据
        while True:
            if offset >= 100000:
                logger.warning(f"同花顺板块指数行情 offset已达10万，停止拉取。")
                break
            try:
                df = self.ts_pro.ths_daily(**{
                        "ts_code": "", "trade_date": "", "start_date": start_date_str, "end_date": end_date_str, "limit": "", "offset": ""
                    }, fields=[
                        "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                        "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq"
                    ])
            except Exception as e:
                logger.error(f"获取同花顺历史行情失败: {e}")
                break
                
            if df is None or df.empty:
                break
                
            all_dfs.append(df)
            
            if len(df) < limit:
                break
            offset += limit
            await asyncio.sleep(0.5)
            
        if not all_dfs:
            return {}
            
        # --- 向量化处理 ---
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        # 2. 批量获取 ThsIndex 对象 (消除 N+1)
        all_codes = full_df['ts_code'].unique().tolist()
        ths_index_map = await self.get_ths_indices_by_codes(all_codes)
        
        full_df['ths_index'] = full_df['ts_code'].map(ths_index_map)
        full_df.dropna(subset=['ths_index'], inplace=True)
        
        if full_df.empty:
            return {}
            
        full_df['trade_time'] = pd.to_datetime(full_df['trade_date']).dt.date
        
        # 3. 数值转换
        numeric_cols = ["open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq"]
        for col in numeric_cols:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
                
        model_cols = ['ths_index', 'trade_time'] + numeric_cols
        final_df = full_df[[c for c in model_cols if c in full_df.columns]]
        
        ths_index_daily_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
        
        # 4. 批量保存
        result = await self._save_all_to_db_native_upsert(
            model_class=ThsIndexDaily,
            data_list=ths_index_daily_dicts,
            unique_fields=['ths_index', 'trade_time']
        )
        return result

    async def save_ths_index_daily_history(self, trade_dates: List[date]) -> Dict:
        """
        【V3.1 向量化内循环优化版】按天并行获取同花顺历史行情，并同步到 ConceptDaily。
        - 核心优化: 将内层的 `itertuples()` 循环替换为Pandas向量化操作，在处理每一天的数据时获得更高的效率。
        """
        if not trade_dates:
            logger.warning("save_ths_index_daily_history 接收到的交易日列表为空，任务跳过。")
            return {}
        print(f"开始为 {len(trade_dates)} 个交易日补全同花顺历史行情...")
        all_days_ths_daily_dicts = []
        for trade_date in trade_dates:
            trade_date_str = trade_date.strftime('%Y%m%d')
            print(f"  -> [同花顺历史日线] 正在处理日期: {trade_date_str}")
            try:
                df = self.ts_pro.ths_daily(trade_date=trade_date_str, fields=[
                    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "avg_price", "change", "pct_change",
                    "vol", "turnover_rate", "total_mv", "float_mv", "pe_ttm", "pb_mrq", "amount"
                ])
            except Exception as e:
                logger.error(f"获取 {trade_date_str} 同花顺历史行情时API失败: {e}", exc_info=True)
                continue
            if df is None or df.empty:
                continue
            # --- 开始向量化处理内循环 ---
            df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
            df.dropna(subset=['ts_code'], inplace=True)
            if df.empty: continue
            all_index_codes = df['ts_code'].unique().tolist()
            ths_index_map = await self.get_ths_indices_by_codes(all_index_codes)
            df['ths_index'] = df['ts_code'].map(ths_index_map)
            df.dropna(subset=['ths_index'], inplace=True)
            if df.empty: continue
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'avg_price', 'change', 'pct_change', 'vol', 'turnover_rate', 'total_mv', 'float_mv', 'pe_ttm', 'pb_mrq', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            model_cols = ['ths_index', 'trade_time'] + numeric_cols
            final_df = df[[col for col in model_cols if col in df.columns]]
            daily_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
            # --- 向量化处理结束 ---
            if daily_dicts:
                await self._save_all_to_db_native_upsert(
                    model_class=ThsIndexDaily,
                    data_list=daily_dicts,
                    unique_fields=['ths_index', 'trade_time']
                )
                all_days_ths_daily_dicts.extend(daily_dicts)
        # --- 后续的统一同步逻辑保持不变 ---
        if all_days_ths_daily_dicts:
            print(f"\n  -> [同步任务] 所有日期处理完毕，开始将 {len(all_days_ths_daily_dicts)} 条历史行情统一同步到 ConceptDaily...")
            all_concept_codes = {item['ths_index'].ts_code for item in all_days_ths_daily_dicts}
            concept_map = await self.get_concepts_by_codes(list(all_concept_codes))
            concept_daily_to_save = [
                self.data_format_process.adapt_to_concept_daily('ths', item, concept_map.get(item['ths_index'].ts_code))
                for item in all_days_ths_daily_dicts if concept_map.get(item['ths_index'].ts_code)
            ]
            if concept_daily_to_save:
                BATCH_SIZE = 5000
                for i in range(0, len(concept_daily_to_save), BATCH_SIZE):
                    batch = concept_daily_to_save[i:i + BATCH_SIZE]
                    await ConceptDaily.objects.abulk_create(batch, ignore_conflicts=True)
                    print(f"     ...已同步 {i + len(batch)} / {len(concept_daily_to_save)} 条记录。")
            print("     ...历史数据同步完成。")
        return {"status": "completed", "processed_days": len(trade_dates)}

    # ============== 开盘啦题材与榜单 ============== 
    async def save_kpl_concept_list_by_date(self, trade_date: date) -> Dict:
        """
        【V2.1 - 向量化版】接口：kpl_concept
        描述：获取开盘啦概念题材列表，并同时更新主表(KplConceptInfo)和每日快照表(KplConceptDaily)。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的开盘啦题材列表...")
        try:
            df = self.ts_pro.kpl_concept(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 kpl_concept 失败: {e}", exc_info=True)
            return {}
        if df is None or df.empty:
            return {}
            
        # --- 向量化处理 ---
        df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        # 1. 更新/创建题材主表 (KplConceptInfo)
        # 只需要 ts_code 和 name
        info_df = df[['ts_code', 'name']].drop_duplicates('ts_code')
        concept_info_list = info_df.to_dict('records')
        
        await self._save_all_to_db_native_upsert(
            model_class=KplConceptInfo,
            data_list=concept_info_list,
            unique_fields=['ts_code'],
        )
        
        # 2. 准备每日快照数据 (KplConceptDaily)
        # 批量获取外键对象
        all_codes = df['ts_code'].unique().tolist()
        concept_map = await self.get_kpl_concepts_by_codes(all_codes)
        
        df['concept_info'] = df['ts_code'].map(concept_map)
        df.dropna(subset=['concept_info'], inplace=True)
        
        # 转换日期和数值
        df['trade_time'] = pd.to_datetime(df['trade_date']).dt.date
        numeric_cols = ['z_t_num', 'up_num', 'down_num', 'w_z_t_num', 'z_t_num_l', 'o_num']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # 映射列名到模型字段
        # 假设 API 列名与模型字段名一致或通过 rename 调整
        # 这里假设 API 返回的列名已经是 snake_case 或者需要简单映射
        # 如果 API 返回的是 'z_t_num' 等，直接使用
        
        model_cols = ['concept_info', 'trade_time'] + numeric_cols
        final_df = df[[c for c in model_cols if c in df.columns]]
        
        daily_items_to_save = final_df.where(pd.notnull(final_df), None).to_dict('records')
        
        if not daily_items_to_save:
            return {}
            
        result = await self._save_all_to_db_native_upsert(
            model_class=KplConceptDaily,
            data_list=daily_items_to_save,
            unique_fields=['concept_info', 'trade_time']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(daily_items_to_save)} 条题材每日快照。")
        return result

    async def save_kpl_concept_members_by_date(self, trade_date: date) -> Dict:
        """
        【V2.1 - 向量化版】接口：kpl_concept_cons
        描述：获取并保存指定日期的开盘啦题材成分股。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的开盘啦题材成分...")
        try:
            df = self.ts_pro.kpl_concept_cons(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 kpl_concept_cons 失败: {e}", exc_info=True)
            return {}
        if df is None or df.empty:
            return {}
            
        # --- 向量化处理 ---
        df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        # 批量获取外键
        all_concept_codes = df['ts_code'].unique().tolist()
        all_stock_codes = df['con_code'].unique().tolist()
        
        concept_map = await self.get_kpl_concepts_by_codes(all_concept_codes)
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        
        df['concept_info'] = df['ts_code'].map(concept_map)
        df['stock'] = df['con_code'].map(stock_map)
        
        # 过滤无效行
        valid_df = df.dropna(subset=['concept_info', 'stock']).copy()
        if valid_df.empty:
            return {}
            
        valid_df['trade_time'] = pd.to_datetime(valid_df['trade_date']).dt.date
        
        # 1. 准备 KplConceptConstituent 数据
        # 假设模型字段: concept_info, stock, trade_time, name (con_name)
        valid_df.rename(columns={'con_name': 'name'}, inplace=True)
        model_cols = ['concept_info', 'stock', 'trade_time', 'name']
        items_to_save = valid_df[model_cols].where(pd.notnull(valid_df), None).to_dict('records')
        
        # 2. 准备 ConceptMember 同步数据
        # 构造同步所需的字典列表
        sync_df = valid_df[['ts_code', 'stock', 'trade_time']].copy()
        sync_df.rename(columns={'ts_code': 'concept_code', 'trade_time': 'in_date'}, inplace=True)
        sync_df['source'] = 'kpl'
        sync_df['out_date'] = None
        concept_member_sync_list = sync_df.where(pd.notnull(sync_df), None).to_dict('records')
        
        # 保存
        result = await self._save_all_to_db_native_upsert(
            model_class=KplConceptConstituent,
            data_list=items_to_save,
            unique_fields=['concept_info', 'stock', 'trade_time']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条题材成分数据。")
        await self._sync_to_concept_member(concept_member_sync_list, 'kpl')
        return result

    async def get_kpl_concepts_by_codes(self, codes: List[str]) -> Dict[str, KplConceptInfo]:
        """【V2.0 新增】根据题材代码列表批量获取 KplConceptInfo 对象"""
        if not codes:
            return {}
        instances = await sync_to_async(list)(KplConceptInfo.objects.filter(ts_code__in=codes))
        return {instance.ts_code: instance for instance in instances}

    async def get_kpl_themes_for_stock(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """【V1.2 修复版】获取股票在指定日期范围内所属的KPL题材列表。返回 tz-naive 的日期。"""
        # print(f"    - [DAO-KPL] 正在为 {stock_code} 获取 {start_date} 到 {end_date} 的KPL题材归属...")
        query = KplConceptConstituent.objects.filter(
            stock__stock_code=stock_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date
        )
        data = await sync_to_async(list)(query.values('trade_time', concept_code=F('concept_info__ts_code')))
        if not data:
            print(f"    - [DAO-KPL] 未找到数据。")
            return pd.DataFrame()
        df = pd.DataFrame.from_records(data)
        df.rename(columns={'trade_time': 'trade_date'}, inplace=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'], utc=True)
        # print(f"    - [DAO-KPL] 成功获取 {len(df)} 条题材归属记录。")
        return df

    async def get_kpl_themes_hotness(self, concept_codes: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """获取一批KPL题材在指定日期范围内的热度指标"""
        query = KplConceptDaily.objects.filter(
            concept_info__ts_code__in=concept_codes,
            trade_time__gte=start_date,
            trade_time__lte=end_date
        )
        data = await sync_to_async(list)(query.values(
            'trade_time', 'z_t_num', 'up_num',
            concept_code=F('concept_info__ts_code')
        ))
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(data)
        df.rename(columns={'trade_time': 'trade_date'}, inplace=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'], utc=True)
        return df

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
                item_dict = self.market_format_process.set_kpl_list_data(stock=stock, df_data=row)
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
        # 从数据库获取
        industry_list = await sync_to_async(list)(DcIndex.objects.all())
        return industry_list

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
        industry = await sync_to_async(DcIndex.objects.filter(ts_code=ts_code).first)()
        return industry

    async def save_dc_index_list_by_date(self, trade_date: date) -> Dict:
        """
        【V1.1 修复版】接口：dc_index
        描述：获取东方财富每日的概念板块列表，并用此数据更新 DcIndex 主表。
        修复：修复了因错误处理数据导致 'ts_code' 为 NULL 的问题。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"    -> 开始获取 [东方财富板块列表] 数据, 日期: {trade_date_str}...")
        all_df_rows = []
        offset = 0
        limit = 5000
        max_offset = 100000 # Tushare 限制
        while offset < max_offset:
            try:
                df = self.ts_pro.dc_index(trade_date=trade_date_str, limit=limit, offset=offset)
                if df is None or df.empty:
                    print(f"    - 在 offset={offset} 处未获取到更多数据，分页结束。")
                    break
                all_df_rows.append(df)
                if len(df) < limit:
                    print(f"    - 获取到 {len(df)} 条数据，少于 limit={limit}，认定为最后一页。")
                    break
                offset += limit
                print(f"    - 已获取 {offset} 条数据，继续下一页...")
                await asyncio.sleep(0.2) # API调用间隔
            except Exception as e:
                logger.error(f"调用Tushare接口 dc_index (offset={offset}) 失败: {e}", exc_info=True)
                break # 出错则终止循环
        if not all_df_rows:
            logger.warning(f"Tushare接口 dc_index 未返回 {trade_date_str} 的任何数据。")
            return {"status": "warning", "message": "API returned no data."}
        combined_df = pd.concat(all_df_rows, ignore_index=True)
        combined_df = combined_df.replace([np.nan, 'nan', 'NaN', ''], None)
        # 提取板块元数据并去重，结果是一个字典列表，可以直接用于保存
        unique_indices = combined_df[['ts_code', 'name']].drop_duplicates().to_dict('records')
        if not unique_indices:
            return {}
        print(f"    - 发现 {len(unique_indices)} 个不重复的东方财富板块，准备进行更新/创建...")
        result = await self._save_all_to_db_native_upsert(
            model_class=DcIndex,
            data_list=unique_indices,  # 直接使用 unique_indices
            unique_fields=['ts_code']
        )
        print(f"    -- 完成 [东方财富板块列表] 更新，共处理 {len(unique_indices)} 条板块元数据。") # 更新日志输出变量
        return result

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

    async def save_dc_index_daily_by_trade_time(self, trade_time: date = None) -> Dict:
        """
        【V3.3 向量化优化版】获取东方财富板块日线行情，并同步到 ConceptDaily。
        - 核心优化: 使用Pandas的向量化操作替代了 `itertuples()` 循环，大幅提升了数据处理效率。
        """
        if trade_time is None:
            trade_time = datetime.today().date()
        trade_time_str = trade_time.strftime('%Y%m%d')
        print(f"  -> [东方财富日线] 开始获取 {trade_time_str} 的数据...")
        all_dfs = []
        offset = 0
        limit = 2000
        max_offset = 100000
        while offset < max_offset:
            try:
                df = self.ts_pro.dc_daily(trade_date=trade_time_str, limit=limit, offset=offset)
                if df is None or df.empty: break
                all_dfs.append(df)
                if len(df) < limit: break
                offset += limit
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"调用Tushare接口 dc_daily (offset={offset}) 失败: {e}", exc_info=True)
                break
        if not all_dfs:
            logger.warning(f"Tushare接口 dc_daily 未返回 {trade_time_str} 的数据。")
            return {}
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # --- 开始向量化处理 ---
        combined_df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        combined_df.dropna(subset=['ts_code'], inplace=True)
        if combined_df.empty: return {}
        # 1. 批量获取/创建关联对象
        all_index_codes = combined_df['ts_code'].unique().tolist()
        dc_index_map = await self.get_dc_indices_by_codes(all_index_codes)
        new_indices_to_create = [{'ts_code': code, 'name': code} for code in all_index_codes if code not in dc_index_map]
        if new_indices_to_create:
            await self._save_all_to_db_native_upsert(model_class=DcIndex, data_list=new_indices_to_create, unique_fields=['ts_code'])
            dc_index_map.update(await self.get_dc_indices_by_codes([d['ts_code'] for d in new_indices_to_create]))
        # 2. 向量化映射、转换和选择
        combined_df['dc_index'] = combined_df['ts_code'].map(dc_index_map)
        combined_df.dropna(subset=['dc_index'], inplace=True)
        if combined_df.empty: return {}
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date'], format='%Y%m%d').dt.date
        numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        model_cols = ['dc_index', 'trade_time'] + numeric_cols
        final_df = combined_df[[col for col in model_cols if col in combined_df.columns]]
        dc_index_daily_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
        # --- 向量化处理结束 ---
        result = {}
        if dc_index_daily_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=DcIndexDaily,
                data_list=dc_index_daily_dicts,
                unique_fields=['dc_index', 'trade_time']
            )
            print(f"     ...成功保存 {len(dc_index_daily_dicts)} 条记录到 [DcIndexDaily] 表。")
            # --- 后续同步逻辑保持不变 ---
            print(f"     -> [同步任务] 开始同步 {trade_time_str} 的东方财富日线行情到 ConceptDaily...")
            concept_map = await self.get_concepts_by_codes(all_index_codes)
            concept_daily_to_save = [
                self.data_format_process.adapt_to_concept_daily('dc', item, concept_map.get(item['dc_index'].ts_code))
                for item in dc_index_daily_dicts if concept_map.get(item['dc_index'].ts_code)
            ]
            if concept_daily_to_save:
                await ConceptDaily.objects.abulk_create(concept_daily_to_save, ignore_conflicts=True)
                print(f"        ...同步完成，处理 {len(concept_daily_to_save)} 条记录到 [ConceptDaily] 表。")
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
        dc_index_members = await sync_to_async(list)(DcIndexMember.objects.filter(ts_code=ts_code).all())
        return dc_index_members

    async def save_dc_index_member_history_by_code(self, ts_code: str) -> Dict:
        """
        【V3.1 - 向量化版】接口：dc_member
        描述：获取指定东方财富板块(ts_code)的【全部历史】成分数据。
        """
        print(f"  - [DAO-历史回补] [东方财富] 开始获取板块 {ts_code} 的全部历史成分...")
        limiter = rate_limiter_factory.get_limiter(name='api_dc_member')
        dc_index = await self.get_dc_index_by_code(ts_code)
        if not dc_index:
            return {}
            
        all_dfs = []
        offset = 0
        limit = 5000
        
        while True:
            if offset >= 100000: break
            try:
                while not await limiter.acquire():
                    await asyncio.sleep(20)
                df = self.ts_pro.dc_member(
                    ts_code=ts_code,
                    limit=limit,
                    offset=offset,
                    fields=["trade_date", "ts_code", "con_code", "name"]
                )
                if df is None or df.empty: break
                all_dfs.append(df)
                if len(df) < limit: break
                offset += limit
            except Exception as e:
                logger.error(f"获取板块 {ts_code} 历史成分失败: {e}")
                break
                
        if not all_dfs:
            return {}
            
        # --- 向量化处理 ---
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        full_df.dropna(subset=['con_code', 'trade_date'], inplace=True)
        
        all_stock_codes = full_df['con_code'].unique().tolist()
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        
        full_df['stock'] = full_df['con_code'].map(stock_map)
        full_df['dc_index'] = dc_index
        
        valid_df = full_df.dropna(subset=['stock']).copy()
        if valid_df.empty:
            return {}
            
        valid_df['trade_time'] = pd.to_datetime(valid_df['trade_date']).dt.date
        
        # 1. DcIndexMember
        members_to_save = valid_df[['dc_index', 'stock', 'trade_time']].where(pd.notnull(valid_df), None).to_dict('records')
        
        # 2. ConceptMember
        sync_df = valid_df[['ts_code', 'stock', 'trade_time']].copy()
        sync_df.rename(columns={'ts_code': 'concept_code', 'trade_time': 'in_date'}, inplace=True)
        sync_df['source'] = 'dc'
        sync_df['out_date'] = None
        concept_member_sync_list = sync_df.where(pd.notnull(sync_df), None).to_dict('records')
        
        print(f"    - [DAO-历史回补] 准备为板块 {ts_code} 保存 {len(members_to_save)} 条历史成分数据...")
        result = await self._save_all_to_db_native_upsert(
            model_class=DcIndexMember,
            data_list=members_to_save,
            unique_fields=['dc_index', 'stock', 'trade_time']
        )
        await self._sync_to_concept_member(concept_member_sync_list, 'dc')
        return result

    async def save_dc_index_members_by_date(self, trade_date: date) -> Dict:
        """
        【V1.3 - 向量化版】接口：dc_member
        描述：按天获取所有东方财富板块的成分股。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"    -> 开始获取 [东方财富板块成分] 数据, 日期: {trade_date_str}...")
        limiter = rate_limiter_factory.get_limiter(name='api_dc_member')
        all_dc_indices = await self.get_dc_index_list()
        
        if not all_dc_indices:
            return {"status": "warning", "message": "No DcIndex found in DB."}
            
        all_dfs = []
        
        for i, dc_index in enumerate(all_dc_indices):
            print(f"      - 进度 {i+1}/{len(all_dc_indices)}: [东方财富板块成分] 获取板块 [{dc_index.name or dc_index.ts_code}] 成分...")
            offset = 0
            limit = 8000
            while True:
                if offset >= 100000: break
                try:
                    while not await limiter.acquire():
                        await asyncio.sleep(20)
                    df = self.ts_pro.dc_member(trade_date=trade_date_str, ts_code=dc_index.ts_code, limit=limit, offset=offset)
                    if df is None or df.empty: break
                    
                    # 关键：在 DF 中标记所属板块对象
                    df['dc_index'] = dc_index
                    all_dfs.append(df)
                    
                    if len(df) < limit: break
                    offset += limit
                except Exception as e:
                    logger.error(f"获取板块 {dc_index.ts_code} 成分失败: {e}")
                    if '次' in str(e): await asyncio.sleep(10)
                    break
                    
        if not all_dfs:
            return {}
            
        # --- 向量化处理 ---
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        all_stock_codes = full_df['con_code'].unique().tolist()
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        
        full_df['stock'] = full_df['con_code'].map(stock_map)
        valid_df = full_df.dropna(subset=['stock']).copy()
        
        if valid_df.empty:
            return {}
            
        valid_df['trade_time'] = pd.to_datetime(valid_df['trade_date']).dt.date
        
        # 1. 准备 DcIndexMember 数据
        members_to_save = valid_df[['dc_index', 'stock', 'trade_time']].where(pd.notnull(valid_df), None).to_dict('records')
        
        # 2. 准备 ConceptMember 同步数据
        sync_df = valid_df[['ts_code', 'stock', 'trade_time']].copy()
        sync_df.rename(columns={'ts_code': 'concept_code', 'trade_time': 'in_date'}, inplace=True)
        sync_df['source'] = 'dc'
        sync_df['out_date'] = None
        concept_member_sync_list = sync_df.where(pd.notnull(sync_df), None).to_dict('records')
        
        print(f"    - 准备保存 {len(members_to_save)} 条东方财富板块成分数据...")
        result = await self._save_all_to_db_native_upsert(
            model_class=DcIndexMember,
            data_list=members_to_save,
            unique_fields=['trade_time', 'dc_index', 'stock']
        )
        
        await self._sync_to_concept_member(concept_member_sync_list, 'dc')
        return result

    async def _sync_to_concept_member(self, sync_list: List[Dict], source: str):
        """
        【V2.0 - 向量化版】将格式化后的成分股数据同步到统一的 ConceptMember 表。
        优化：使用 Pandas 向量化处理对象映射，提升大批量同步时的性能。
        """
        if not sync_list:
            return
            
        print(f"    -> [同步任务] 开始同步 {len(sync_list)} 条 [{source.upper()}] 成分股数据到 [ConceptMember]...")
        
        # 1. 转为 DataFrame
        df = pd.DataFrame(sync_list)
        if df.empty:
            return
            
        # 2. 批量获取 ConceptMaster 对象
        all_concept_codes = df['concept_code'].unique().tolist()
        concept_map = await self.get_concepts_by_codes(all_concept_codes)
        
        # 3. 向量化映射
        df['concept'] = df['concept_code'].map(concept_map)
        
        # 4. 过滤无效数据 (未找到对应 ConceptMaster 的记录)
        valid_df = df.dropna(subset=['concept']).copy()
        
        if len(valid_df) < len(df):
            logger.warning(f"同步过程中有 {len(df) - len(valid_df)} 条记录因未找到 ConceptMaster 而被忽略。")
            
        if valid_df.empty:
            return
            
        # 5. 准备保存的数据
        # 移除临时列 concept_code，保留模型所需字段
        # 假设 sync_list 中已包含 stock, in_date, out_date, source 等字段
        cols_to_keep = ['concept', 'stock', 'in_date', 'out_date', 'source']
        final_data_to_save = valid_df[cols_to_keep].to_dict('records')
        
        # 6. 批量保存
        await self._save_all_to_db_native_upsert(
            model_class=ConceptMember,
            data_list=final_data_to_save,
            unique_fields=['concept', 'stock', 'in_date', 'source']
        )
        print(f"    -- [同步任务] 完成，成功处理 {len(final_data_to_save)} 条 [{source.upper()}] 成分股数据。")

    # ============== 市场情绪与涨跌停数据 ==============
    async def save_limit_list_ths_by_date(self, trade_date: date) -> Dict:
        """
        【V1.2 - 向量化版】接口：limit_list_ths
        描述：获取并保存指定交易日的同花顺涨跌停榜单数据。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的同花顺涨跌停榜单...")
        limiter = rate_limiter_factory.get_limiter(name='api_limit_list_ths')
        limit_types = ['涨停池', '连扳池', '冲刺涨停', '炸板池', '跌停池']
        all_dfs = []
        
        for l_type in limit_types:
            try:
                while not await limiter.acquire():
                    await asyncio.sleep(20)
                df = self.ts_pro.limit_list_ths(trade_date=trade_date_str, limit_type=l_type)
                if df is not None and not df.empty:
                    df['limit_type'] = l_type # 标记类型
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"调用Tushare接口 limit_list_ths (limit_type={l_type}) 失败: {e}")
                continue
                
        if not all_dfs:
            return {}
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        # 向量化映射
        all_stock_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        
        if combined_df.empty:
            return {}
            
        combined_df['trade_date'] = pd.to_datetime(combined_df['trade_date']).dt.date
        
        # 列名映射 (API -> Model)
        # 假设模型字段: stock, trade_date, limit_type, name, close, pct_chg, amp, fc_ratio, fl_ratio, turnover_rate, first_time, last_time, open_times, strth, limit_reason
        # 需要根据实际 API 返回列名调整
        # 这里做简单的数值转换
        numeric_cols = ['close', 'pct_chg', 'amp', 'fc_ratio', 'fl_ratio', 'turnover_rate', 'open_times', 'strth']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                
        items_to_save = combined_df.where(pd.notnull(combined_df), None).to_dict('records')
        
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitListThs,
            data_list=items_to_save,
            unique_fields=['stock', 'trade_date', 'limit_type']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条同花顺涨跌停数据。")
        return result

    async def save_limit_list_d_by_date(self, trade_date: date) -> Dict:
        """
        【V1.2 - 向量化版】接口：limit_list_d
        描述：获取并保存指定交易日的Tushare版A股涨跌停列表数据。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的Tushare涨跌停列表...")
        limiter = rate_limiter_factory.get_limiter(name='api_limit_list_d')
        limit_types = ['U', 'D', 'Z']
        all_dfs = []
        
        for l_type in limit_types:
            try:
                while not await limiter.acquire():
                    await asyncio.sleep(20)
                df = self.ts_pro.limit_list_d(trade_date=trade_date_str, limit_type=l_type)
                if df is not None and not df.empty:
                    df['limit'] = l_type # 注意模型字段名为 limit
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"调用Tushare接口 limit_list_d (limit_type={l_type}) 失败: {e}")
                continue
                
        if not all_dfs:
            return {}
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        all_stock_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        
        if combined_df.empty:
            return {}
            
        combined_df['trade_date'] = pd.to_datetime(combined_df['trade_date']).dt.date
        
        # 数值转换
        numeric_cols = ['close', 'pct_chg', 'amp', 'fc_ratio', 'fl_ratio', 'turnover_rate', 'limit_times', 'open_times', 'strth']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                
        items_to_save = combined_df.where(pd.notnull(combined_df), None).to_dict('records')
        
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitListD,
            data_list=items_to_save,
            unique_fields=['stock', 'trade_date', 'limit']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条Tushare涨跌停数据。")
        return result

    async def save_limit_step_by_date(self, trade_date: date) -> Dict:
        """
        【V1.2 - 向量化版】接口：limit_step
        描述：获取并保存指定交易日的连板天梯数据。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的连板天梯数据...")
        limiter = rate_limiter_factory.get_limiter(name='api_limit_step')
        try:
            while not await limiter.acquire():
                await asyncio.sleep(20)
            df = self.ts_pro.limit_step(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 limit_step 失败: {e}", exc_info=True)
            return {}
            
        if df is None or df.empty:
            return {}
            
        df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        all_stock_codes = df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        
        df['stock'] = df['ts_code'].map(stock_map)
        df.dropna(subset=['stock'], inplace=True)
        
        if df.empty:
            return {}
            
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        # 假设模型字段: stock, trade_date, step, name
        # API 返回字段可能包含 step (连板数)
        
        items_to_save = df.where(pd.notnull(df), None).to_dict('records')
        
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitStep,
            data_list=items_to_save,
            unique_fields=['stock', 'trade_date']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条连板天梯数据。")
        return result

    async def save_limit_cpt_list_by_date(self, trade_date: date) -> Dict:
        """
        【V1.2 - 向量化版】接口：limit_cpt_list
        描述：获取并保存指定交易日的最强板块统计数据。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的最强板块统计...")
        limiter = rate_limiter_factory.get_limiter(name='api_limit_cpt_list')
        try:
            while not await limiter.acquire():
                await asyncio.sleep(20)
            df = self.ts_pro.limit_cpt_list(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 limit_cpt_list 失败: {e}", exc_info=True)
            return {}
            
        if df is None or df.empty:
            return {}
            
        df.replace([np.nan, 'nan', 'NaN', ''], None, inplace=True)
        
        all_index_codes = df['ts_code'].unique().tolist()
        index_map = await self.get_ths_indices_by_codes(all_index_codes)
        
        df['ths_index'] = df['ts_code'].map(index_map)
        df.dropna(subset=['ths_index'], inplace=True)
        
        if df.empty:
            return {}
            
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        
        # 数值转换
        numeric_cols = ['rank', 'cons_nums', 'up_nums', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        items_to_save = df.where(pd.notnull(df), None).to_dict('records')
        
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitCptList,
            data_list=items_to_save,
            unique_fields=['ths_index', 'trade_date']
        )
        print(f"  - [DAO] 完成保存 {trade_date_str} 的 {len(items_to_save)} 条最强板块数据。")
        return result

    # ============== 市场情绪与涨跌停数据 (Getter方法) ==============
    async def get_limit_list_d_for_range(self, start_date: date, end_date: date, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """根据日期范围和股票代码列表，获取Tushare版涨跌停数据。"""
        query = LimitListD.objects.filter(trade_date__gte=start_date, trade_date__lte=end_date)
        if stock_codes:
            query = query.filter(stock__stock_code__in=stock_codes)
        # 使用 avalues 异步获取数据并转换为 DataFrame
        data = await sync_to_async(lambda: list(query.values(
            'trade_date', 'stock__stock_code', 'limit', 'limit_times', 'open_times'
        )))()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data).rename(columns={'stock__stock_code': 'stock_code'})
        df['trade_date'] = pd.to_datetime(df['trade_date'], utc=True)
        return df

    async def get_limit_step_for_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """根据日期范围，获取连板天梯数据。"""
        query = LimitStep.objects.filter(trade_date__gte=start_date, trade_date__lte=end_date)
        data = await sync_to_async(lambda: list(query.values(
            'trade_date', 'stock__stock_code', 'nums'
        )))()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data).rename(columns={'stock__stock_code': 'stock_code'})
        df['trade_date'] = pd.to_datetime(df['trade_date'], utc=True)
        return df

    async def get_limit_cpt_list_for_range(self, start_date: date, end_date: date, industry_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """根据日期范围和板块代码列表，获取最强板块统计数据。"""
        query = LimitCptList.objects.filter(trade_date__gte=start_date, trade_date__lte=end_date)
        if industry_codes:
            query = query.filter(ths_index__ts_code__in=industry_codes)
        data = await sync_to_async(lambda: list(query.values(
            'trade_date', 'ths_index__ts_code', 'rank', 'up_stat', 'cons_nums', 'up_nums', 'pct_chg'
        )))()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data).rename(columns={'ths_index__ts_code': 'industry_code'})
        df['trade_date'] = pd.to_datetime(df['trade_date'], utc=True)
        return df

    async def get_limit_cpt_list_for_industry_and_date(self, industry_code: str, trade_date: date) -> Optional[Dict]:
        """获取单个板块在指定日期的最强板块统计数据。"""
        record = await sync_to_async(
            LimitCptList.objects.filter(ths_index__ts_code=industry_code, trade_date=trade_date).first
        )()
        if record:
            return {
                'rank': int(record.rank),
                'up_stat': record.up_stat,
                'cons_nums': record.cons_nums,
                'up_nums': int(record.up_nums),
                'pct_chg': record.pct_chg
            }
        return None

    # ============== 板块/概念主数据 (ConceptMaster) ==============
    async def get_all_concepts_by_source(self, source: str) -> List[ConceptMaster]:
        """
        【V3.0 新增】根据来源获取所有板块/概念主数据。
        """
        return await sync_to_async(list)(ConceptMaster.objects.filter(source=source))

    async def get_concepts_by_codes(self, codes: List[str]) -> Dict[str, ConceptMaster]:
        """
        【V3.0 新增】根据代码列表批量获取 ConceptMaster 对象。
        """
        if not codes:
            return {}
        concepts = await sync_to_async(list)(ConceptMaster.objects.filter(code__in=codes))
        return {concept.code: concept for concept in concepts}

    async def get_concept_members_on_date(self, concept_code: str, trade_date: date) -> List[ConceptMember]:
        """
        【V2.1 终极版】获取指定板块在指定日期的所有有效成分股。
        - 核心升级: 能够智能区分“历史快照”('dc', 'kpl')和“当前快照”('ths')数据源。
          - 对于 'ths' 来源，直接返回所有当前成员，作为历史日期的最佳代理。
        """
        try:
            concept = await ConceptMaster.objects.aget(code=concept_code)
            source = concept.source
        except ConceptMaster.DoesNotExist:
            logger.warning(f"查询成分股时未在 ConceptMaster 中找到代码 {concept_code}。")
            return []
        interval_sources = ['sw', 'ci']
        historical_snapshot_sources = ['dc', 'kpl']
        # 'ths' 不再属于 snapshot_sources，因为它不提供历史快照
        if source in interval_sources:
            # 区间模式: 精确查找
            query = Q(concept__code=concept_code) & \
                    Q(in_date__lte=trade_date) & \
                    (Q(out_date__isnull=True) | Q(out_date__gt=trade_date))
            members = await sync_to_async(list)(
                ConceptMember.objects.filter(query).select_related('stock')
            )
        elif source in historical_snapshot_sources:
            # 历史快照模式: 查找小于等于trade_date的最新快照
            latest_date_subquery = ConceptMember.objects.filter(
                concept__code=concept_code,
                in_date__lte=trade_date
            ).values('concept__code').annotate(
                latest_date=Max('in_date')
            ).values('latest_date')
            members = await sync_to_async(list)(
                ConceptMember.objects.filter(
                    concept__code=concept_code,
                    in_date=Subquery(latest_date_subquery[:1])
                ).select_related('stock')
            )
        elif source == 'ths':
            # 当前快照模式: 直接返回所有成员作为代理
            # print(f"    - [DAO-成员查询] 板块 {concept_code} (ths) 为当前快照模式，返回所有当前成员作为代理。")
            members = await sync_to_async(list)(
                ConceptMember.objects.filter(
                    concept__code=concept_code, source='ths'
                ).select_related('stock')
            )
        else:
            logger.warning(f"未知的概念来源 '{source}' (板块代码: {concept_code})，无法确定成分股查询策略。")
            return []
        return members

    async def get_limit_list_for_stocks(self, stock_codes: List[str], trade_date: date, tag: str) -> pd.DataFrame:
        """
        根据股票代码列表和日期，获取KPL榜单数据。
        - 返回一个DataFrame，便于后续分析。
        """
        if not stock_codes:
            return pd.DataFrame()
        # 异步查询 KplLimitList
        query = KplLimitList.objects.filter(
            stock__stock_code__in=stock_codes,
            trade_date=trade_date,
            tag=tag
        )
        # 使用 avalues 异步获取数据，只选择需要的列
        data = await sync_to_async(list)(query.values(
            'stock_id', 'name', 'lu_time', 'status', 'turnover_rate', 'limit_order', 'free_float'
        ))
        if not data:
            return pd.DataFrame()
        return pd.DataFrame.from_records(data)

    # ============== 板块/概念日线行情 (ConceptDaily) ==============
    async def get_concept_daily_for_range(self, concept_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        【V3.0 新增】根据代码和日期范围获取通用板块日线行情。
        """
        qs = ConceptDaily.objects.filter(
            concept__code=concept_code,
            trade_date__gte=start_date,
            trade_date__lte=end_date
        ).order_by('trade_date')
        data = await sync_to_async(list)(qs.values('trade_date', 'open', 'high', 'low', 'close', 'turnover_rate'))
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(data)
        df['trade_date'] = pd.to_datetime(df['trade_date'], utc=True)
        df.set_index('trade_date', inplace=True)
        return df

    # ============== 行业生命周期预计算 ==============
    async def save_industry_lifecycle(self, lifecycle_data: List[Dict]) -> Dict:
        """
        【V3.0 改造】批量保存行业生命周期预计算结果。
        - 适配通用的 ConceptMaster 模型。
        """
        if not lifecycle_data:
            return {}
        all_concept_codes = [item['concept_code'] for item in lifecycle_data]
        concept_map = await self.get_concepts_by_codes(all_concept_codes)
        records_to_save = []
        for item in lifecycle_data:
            concept = concept_map.get(item['concept_code'])
            if concept:
                records_to_save.append({
                    "concept": concept,
                    "trade_date": item['trade_date'],
                    "source": concept.source,
                    "strength_rank": item.get('strength_rank'),
                    "rank_slope": item.get('rank_slope'),
                    "rank_accel": item.get('rank_accel'),
                    "lifecycle_stage": item.get('lifecycle_stage'),
                    "breadth_score": item.get('breadth_score'),
                    "leader_score": item.get('leader_score')
                })
        return await self._save_all_to_db_native_upsert(
            model_class=IndustryLifecycle,
            data_list=records_to_save,
            unique_fields=['concept', 'trade_date']
        )

    async def get_raw_lifecycle_data_for_stock(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        【V6.0 职责净化版】根据股票代码，仅获取其所属的所有行业/概念在指定日期范围内的【原始】生命周期数据。
        - 核心重构: 移除了所有计算和融合逻辑，此方法现在只负责从数据库查询并返回原始数据。
                      所有计算逻辑已移至 ContextualAnalysisService。
        - 返回: 一个包含原始生命周期数据的长格式 DataFrame。
        """
        # print(f"    - [DAO-查询] 正在为 {stock_code} 获取原始行业生命周期数据...")
        # 1. 获取股票所属的所有概念
        all_concepts = await self.get_stock_all_concepts(stock_code)
        if not all_concepts:
            print(f"    - [DAO-查询] 未找到 {stock_code} 的任何行业/概念归属。")
            return pd.DataFrame()
        all_concept_codes = [c['code'] for c in all_concepts]
        # 2. 批量查询这些概念的生命周期数据
        query = IndustryLifecycle.objects.filter(
            concept__code__in=all_concept_codes,
            trade_date__gte=start_date,
            trade_date__lte=end_date
        ).select_related('concept').order_by('trade_date')
        # 直接返回查询结果，不做任何处理
        data = await sync_to_async(list)(query.values(
            'trade_date',
            'concept__code',
            'concept__source',
            'strength_rank',
            'rank_slope',
            'rank_accel',
            'lifecycle_stage',
            'breadth_score',
            'leader_score'
        ))
        if not data:
            print(f"    - [DAO-查询] 未查询到 {stock_code} 相关概念的生命周期数据。")
            return pd.DataFrame()
        df = pd.DataFrame.from_records(data)
        # 重命名以保持一致性
        df.rename(columns={'concept__code': 'concept_code', 'concept__source': 'source'}, inplace=True)
        # print(f"    - [DAO-查询] 成功获取 {len(df)} 条原始生命周期记录。")
        return df

    # ============ V3.0 多维概念融合分析 - 核心数据获取方法 ============
    async def get_stock_all_concepts(self, stock_code: str) -> List[Dict[str, str]]:
        """
        【V3.0 生产级】获取一个股票所属的所有行业/概念及其来源。
        - 核心特性: 异步并行、缓存优先、健壮容错、输出标准化。
        - 数据源: 申万(sw), 中信(ci), 开盘啦(kpl), 同花顺(ths), 东方财富(dc)。
        Args:
            stock_code (str): 股票代码。
        Returns:
            List[Dict[str, str]]: 一个包含该股票所有概念标签的列表，
                                  每个元素是一个字典，格式为 {'code': '...', 'name': '...', 'source': '...'}.
        """
        # print(f"    - [多维概念融合DAO] 开始为 {stock_code} 并行获取所有概念标签...")
        # 1. 定义所有需要并行执行的查询任务
        tasks = [
            self._get_sw_concepts(stock_code),
            self._get_ci_concepts(stock_code),
            self._get_kpl_concepts(stock_code),
            self._get_ths_concepts(stock_code),
            self._get_dc_concepts(stock_code),
        ]
        # 2. 使用 asyncio.gather 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 3. 汇总所有查询结果
        all_concepts = []
        for res in results:
            if isinstance(res, Exception):
                # 记录异常，但不中断整个流程
                logger.error(f"在 get_stock_all_concepts 中获取某个数据源时失败: {res}", exc_info=False)
                continue
            if res:
                all_concepts.extend(res)
        # 4. 去重并返回
        # 使用元组作为集合元素进行去重，因为字典是不可哈希的
        unique_concepts_tuples = { (c['code'], c['source']) for c in all_concepts }
        # 为了保持原始的name，我们需要一个映射
        code_source_to_name_map = { (c['code'], c['source']): c['name'] for c in all_concepts }
        final_concepts = [
            {'code': code, 'name': code_source_to_name_map[(code, source)], 'source': source}
            for code, source in unique_concepts_tuples
        ]
        # print(f"    - [多维概念融合DAO] 完成。为 {stock_code} 共获取到 {len(final_concepts)} 个唯一概念标签。")
        return final_concepts

    async def _get_sw_concepts(self, stock_code: str) -> List[Dict[str, str]]:
        """
        【V1.3 - 缓存优化版】获取申万行业概念。
        优化：在实例层面缓存 SwIndustry 映射表，避免每次调用都全量查询数据库。
        """
        cache_key = self.cache_key_stock.stock_concepts(stock_code, 'sw')
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
            
        concepts = []
        try:
            # 1. 获取该股票所属的L3行业成员关系
            memberships = SwIndustryMember.objects.filter(
                stock__stock_code=stock_code, is_new='Y'
            ).select_related('l3_industry')
            
            # 2. 获取申万行业映射 (带实例级缓存)
            if not hasattr(self, '_sw_industry_map_cache'):
                all_sw = await sync_to_async(list)(SwIndustry.objects.all())
                self._sw_industry_map_cache = {ind.index_code: ind for ind in all_sw}
            sw_industry_map = self._sw_industry_map_cache
            
            async for member in memberships:
                l3 = member.l3_industry
                if l3 and l3.index_code in sw_industry_map:
                    concepts.append({'code': l3.index_code, 'name': l3.industry_name, 'source': 'sw'})
                    
                    # 查找父级
                    l2 = sw_industry_map.get(l3.parent_code)
                    if l2:
                        concepts.append({'code': l2.index_code, 'name': l2.industry_name, 'source': 'sw'})
                        l1 = sw_industry_map.get(l2.parent_code)
                        if l1:
                            concepts.append({'code': l1.index_code, 'name': l1.industry_name, 'source': 'sw'})
                            
            await self.cache_manager.set(cache_key, concepts, timeout=3600 * 24)
            return concepts
        except Exception as e:
            logger.error(f"查询股票 {stock_code} 的申万行业时出错: {e}", exc_info=True)
            return []

    async def _get_ci_concepts(self, stock_code: str) -> List[Dict[str, str]]:
        """获取中信行业概念"""
        cache_key = self.cache_key_stock.stock_concepts(stock_code, 'ci')
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        concepts = []
        try:
            # 假设中信行业模型为 CiIndexMember，请根据实际情况调整
            from stock_models.industry import CiIndexMember # 确保导入
            memberships = CiIndexMember.objects.filter(stock__stock_code=stock_code, is_new='Y')
            async for member in memberships:
                if member.l1_code:
                    concepts.append({'code': member.l1_code, 'name': member.l1_name, 'source': 'ci'})
                if member.l2_code:
                    concepts.append({'code': member.l2_code, 'name': member.l2_name, 'source': 'ci'})
                if member.l3_code:
                    concepts.append({'code': member.l3_code, 'name': member.l3_name, 'source': 'ci'})
            await self.cache_manager.set(cache_key, concepts, timeout=3600 * 24)
            return concepts
        except Exception as e:
            logger.error(f"查询股票 {stock_code} 的中信行业时出错: {e}", exc_info=True)
            return []
 
    async def _get_kpl_concepts(self, stock_code: str) -> List[Dict[str, str]]:
        """【V1.1 修复版】获取开盘啦题材概念，修复了 SynchronousOnlyOperation 错误。"""
        cache_key = self.cache_key_stock.stock_concepts(stock_code, 'kpl')
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        concepts = []
        try:
            latest_date_obj = await sync_to_async(
                lambda: KplConceptConstituent.objects.latest('trade_time')
            )()
            if not latest_date_obj:
                return []
            latest_date = latest_date_obj.trade_time
            memberships = KplConceptConstituent.objects.filter(
                stock__stock_code=stock_code, trade_time=latest_date
            ).select_related('concept_info')
            async for member in memberships:
                if member.concept_info:
                    concepts.append({'code': member.concept_info.ts_code, 'name': member.concept_info.name, 'source': 'kpl'})
            await self.cache_manager.set(cache_key, concepts, timeout=3600 * 12)
            return concepts
        except KplConceptConstituent.DoesNotExist:
            return []
        except Exception as e:
            logger.error(f"查询股票 {stock_code} 的开盘啦题材时出错: {e}", exc_info=True)
            return []

    async def _get_ths_concepts(self, stock_code: str) -> List[Dict[str, str]]:
        """获取同花顺行业与概念"""
        cache_key = self.cache_key_stock.stock_concepts(stock_code, 'ths')
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        concepts = []
        try:
            memberships = ThsIndexMember.objects.filter(
                stock__stock_code=stock_code, is_new='Y'
            ).select_related('ths_index')
            async for member in memberships:
                if member.ths_index:
                    concepts.append({'code': member.ths_index.ts_code, 'name': member.ths_index.name, 'source': 'ths'})
            await self.cache_manager.set(cache_key, concepts, timeout=3600 * 24)
            return concepts
        except Exception as e:
            logger.error(f"查询股票 {stock_code} 的同花顺概念时出错: {e}", exc_info=True)
            return []
 
    async def _get_dc_concepts(self, stock_code: str) -> List[Dict[str, str]]:
        """【V1.1 修复版】获取东方财富概念，修复了 SynchronousOnlyOperation 错误。"""
        cache_key = self.cache_key_stock.stock_concepts(stock_code, 'dc')
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        concepts = []
        try:
            latest_date_obj = await sync_to_async(
                lambda: DcIndexMember.objects.latest('trade_time')
            )()
            if not latest_date_obj:
                return []
            latest_date = latest_date_obj.trade_time
            memberships = DcIndexMember.objects.filter(
                stock__stock_code=stock_code, trade_time=latest_date
            ).select_related('dc_index')
            async for member in memberships:
                if member.dc_index:
                    concepts.append({'code': member.dc_index.ts_code, 'name': member.dc_index.name, 'source': 'dc'})
            await self.cache_manager.set(cache_key, concepts, timeout=3600 * 12)
            return concepts
        except DcIndexMember.DoesNotExist:
            return []
        except Exception as e:
            logger.error(f"查询股票 {stock_code} 的东方财富概念时出错: {e}", exc_info=True)
            return []























