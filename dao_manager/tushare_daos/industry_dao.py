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
        【V2.3 统一模型同步修复版】获取申万行业成分并保存
        - 核心修复: 补全了将申万 L1, L2, L3 三个层级的行业成分关系同步到统一的 ConceptMember 表的核心逻辑。
        - 收益: 确保了申万行业数据能够被下游的通用分析模块（如行业生命周期、多维概念融合）正确消费。
        """
        print("  - [DAO] 开始获取申万行业成分...")
        # --- 1. 初始化和准备 ---
        API_CALL_DELAY_SECONDS = 0.3
        final_result = {}
        all_raw_members = []
        all_l1_codes, all_l2_codes, all_l3_codes = set(), set(), set()
        all_stock_codes = set()
        # --- 2. 获取所有L1行业并循环调用API ---
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
                    df = df.replace([np.nan, 'nan', 'NaN', ''], None)
                    for row in df.itertuples(index=False):
                        all_raw_members.append(row)
                        if row.l1_code: all_l1_codes.add(row.l1_code)
                        if row.l2_code: all_l2_codes.add(row.l2_code)
                        if row.l3_code: all_l3_codes.add(row.l3_code)
                        if row.ts_code: all_stock_codes.add(row.ts_code)
                await asyncio.sleep(API_CALL_DELAY_SECONDS)
            except Exception as e:
                logger.error(f"获取申万行业 [{sw_l1_index.industry_name}] 成分时发生API错误: {e}", exc_info=True)
                continue
        logger.info(f"API数据获取完成，共 {len(all_raw_members)} 条成分记录。")
        # --- 3. 批量获取所有需要的外键对象 ---
        print("    - 正在一次性获取所有涉及的申万行业和股票信息...")
        all_industry_codes = list(all_l1_codes | all_l2_codes | all_l3_codes)
        sw_industry_qs = SwIndustry.objects.filter(index_code__in=all_industry_codes)
        sw_industry_map = {ind.index_code: ind async for ind in sw_industry_qs}
        stock_qs = StockInfo.objects.filter(stock_code__in=list(all_stock_codes))
        stock_map = {stock.stock_code: stock async for stock in stock_qs}
        print(f"    - 成功获取并映射了 {len(sw_industry_map)} 个行业和 {len(stock_map)} 个股票信息。")
        # --- 4. 组装最终数据 (包含动态创建缺失行业的逻辑) ---
        industry_member_dicts = []
        concept_member_sync_list = [] # 确保列表已初始化
        for row in all_raw_members:
            # 按层级检查并创建缺失的行业 (此部分逻辑保持不变)
            try:
                if row.l1_code and row.l1_code not in sw_industry_map:
                    l1_obj, _ = await sync_to_async(SwIndustry.objects.get_or_create)(
                        index_code=row.l1_code,
                        defaults={'industry_name': row.l1_name, 'level': 'L1', 'parent_code': '0', 'src': 'SW2021', 'industry_code': row.l1_code}
                    )
                    sw_industry_map[row.l1_code] = l1_obj
                if row.l2_code and row.l2_code not in sw_industry_map:
                    l2_obj, _ = await sync_to_async(SwIndustry.objects.get_or_create)(
                        index_code=row.l2_code,
                        defaults={'industry_name': row.l2_name, 'level': 'L2', 'parent_code': row.l1_code, 'src': 'SW2021', 'industry_code': row.l2_code}
                    )
                    sw_industry_map[row.l2_code] = l2_obj
                if row.l3_code and row.l3_code not in sw_industry_map:
                    l3_obj, _ = await sync_to_async(SwIndustry.objects.get_or_create)(
                        index_code=row.l3_code,
                        defaults={'industry_name': row.l3_name, 'level': 'L3', 'parent_code': row.l2_code, 'src': 'SW2021', 'industry_code': row.l3_code}
                    )
                    sw_industry_map[row.l3_code] = l3_obj
            except Exception as e:
                logger.error(f"动态创建申万行业时失败: {e}, 数据: {row}", exc_info=True)
                continue
            swan_industry = sw_industry_map.get(row.l3_code)
            stock = stock_map.get(row.ts_code)
            if swan_industry and stock:
                # 保存到 SwIndustryMember 表的逻辑
                industry_member_dict = self.data_format_process.set_sw_industry_member_data(
                    sw_industry=swan_industry,
                    stock=stock,
                    df_data=row
                )
                industry_member_dicts.append(industry_member_dict)
                # 每一行申万成分数据，都需要生成 L1, L2, L3 三条记录到 concept_member_sync_list
                for level in ['l1', 'l2', 'l3']:
                    concept_code = getattr(row, f'{level}_code', None)
                    if concept_code:
                        concept_member_sync_list.append({
                            'concept_code': concept_code,
                            'stock': stock,
                            'source': 'sw',
                            'in_date': self.data_format_process._parse_datetime(row.in_date),
                            'out_date': self.data_format_process._parse_datetime(row.out_date)
                        })
            else:
                if not swan_industry: logger.warning(f"动态创建后仍未找到申万三级行业 {row.l3_code}，成分股 {row.ts_code} 将被忽略。")
        # --- 5. 批量保存 ---
        if industry_member_dicts:
            print(f"    - 准备保存 {len(industry_member_dicts)} 条申万行业成分数据...")
            final_result = await self._save_all_to_db_native_upsert(
                model_class=SwIndustryMember,
                data_list=industry_member_dicts,
                unique_fields=['l3_industry', 'stock', 'in_date']
            )
            print(f"  - [DAO] 完成保存申万行业成分。结果: {final_result}")
            # 确保在保存完主数据后，调用同步方法
            await self._sync_to_concept_member(concept_member_sync_list, 'sw')
        else:
            logger.warning("没有可供保存的申万行业成分数据。")
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
        【V3.0 终极建模版】获取同花顺概念板块成分列表并保存
        - 核心修复:
          1. [正确建模] 承认同花顺API只提供“当前快照”的现实。
          2. [优雅代理] 在同步到 ConceptMember 时，将 in_date 设置为一个极早的日期(1990-01-01)，
             将当前成员关系建模为“长期有效”，作为历史分析的最佳代理。
          3. [数据完整性] 在 ths_index_member 表中，in_date 记录为数据获取当天，作为更新时间戳。
        - 收益: 彻底解决在历史回溯时因 in_date 问题导致“未找到成分股”的致命错误。
        """
        # --- 1. 初始化和准备 ---
        final_result = {}
        limiter = rate_limiter_factory.get_limiter(name='api_ths_member')
        today = datetime.now().date() # 数据获取日期，作为 ths_index_member 的时间戳
        PROXY_IN_DATE = date(1990, 1, 1) # 用于 ConceptMember 的代理“纳入日期”
        # --- 2. 获取所有概念板块信息 ---
        ths_index_list = await self.get_ths_index_list()
        if not ths_index_list:
            logger.warning("数据库中未找到任何同花顺概念板块信息，任务结束。")
            return {"status": "warning", "message": "No ThsIndex found."}
        logger.info(f"开始处理 {len(ths_index_list)} 个同花顺概念板块...")
        # --- 3. 循环调用API，收集数据 ---
        all_raw_members = []
        all_stock_codes = set()
        all_ths_index_codes = {idx.ts_code for idx in ths_index_list} # 收集所有板块代码用于清空操作
        for i, ths_index in enumerate(ths_index_list):
            print(f"进度: {i+1}/{len(ths_index_list)} | [同花顺概念]正在获取板块 [{ths_index.name} ({ths_index.ts_code})] 的成分股...")
            offset = 0
            limit = 6000
            while True:
                if offset >= 100000: break
                try:
                    while not await limiter.acquire():
                        await asyncio.sleep(20)
                    df = self.ts_pro.ths_member(ts_code=ths_index.ts_code, fields="ts_code,con_code,con_name", limit=limit, offset=offset)
                    if df is None or df.empty: break
                    for row in df.itertuples(index=False):
                        all_raw_members.append((ths_index, row))
                        all_stock_codes.add(row.con_code)
                    if len(df) < limit: break
                    offset += limit
                except Exception as e:
                    logger.error(f"[同花顺概念]获取板块 [{ths_index.name}] 成分股时(offset={offset})发生API错误: {e}", exc_info=True)
                    break
        logger.info(f"API数据获取完成，共 {len(all_raw_members)} 条成分股记录。")
        # --- 4. 批量获取股票信息 ---
        print("正在一次性获取所有涉及的股票信息...")
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(list(all_stock_codes))
        print(f"成功获取并映射了 {len(stock_map)} 个股票的信息。")
        # --- 5. 组装数据 ---
        print("开始组装最终数据...")
        data_to_save_ths = []
        concept_member_sync_list = []
        for ths_index, row_data in all_raw_members:
            stock = stock_map.get(row_data.con_code)
            if not stock: continue
            # 为 ths_index_member 表准备数据 (in_date 是数据获取日期)
            data_to_save_ths.append(ThsIndexMember(
                ths_index=ths_index, stock=stock, weight=None,
                in_date=today, out_date=None, is_new='Y'
            ))
            # 为 concept_member 表准备同步数据 (in_date 是代理日期)
            concept_member_sync_list.append({
                'concept_code': ths_index.ts_code, 'stock': stock, 'source': 'ths',
                'in_date': PROXY_IN_DATE, 'out_date': None
            })
        # --- 6. 批量保存与同步 (采用“先删后插”策略) ---
        if data_to_save_ths:
            print(f"准备清空并写入 {len(data_to_save_ths)} 条最新的同花顺成分数据...")
            # 事务性地执行删除和插入
            # 1. 清空 ThsIndexMember 表中涉及的所有板块的旧数据
            await sync_to_async(ThsIndexMember.objects.filter(ths_index__ts_code__in=all_ths_index_codes).delete)()
            print("  - 旧的 ThsIndexMember 数据已清空。")
            # 2. 批量插入新数据到 ThsIndexMember
            await ThsIndexMember.objects.abulk_create(data_to_save_ths, batch_size=5000)
            print(f"  - 批量写入 {len(data_to_save_ths)} 条数据到 ths_index_member 完成。")
            # 3. 同步到 ConceptMember (同样采用先删后插)
            # 注意：这里的删除逻辑也需要调整，只删除 ths 来源的
            await sync_to_async(ConceptMember.objects.filter(concept__code__in=all_ths_index_codes, source='ths').delete)()
            print("  - 旧的 ConceptMember (ths来源) 数据已清空。")
            await self._sync_to_concept_member(concept_member_sync_list, 'ths')
        logger.info("同花顺概念板块成分保存与同步任务全部完成。")
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
        if df is not None and not df.empty:
            df = df.replace(['nan', 'NaN', ''], None)
            all_index_codes = df['ts_code'].unique().tolist()
            ths_index_map = await self.get_ths_indices_by_codes(all_index_codes)
            print(f"DEBUG: 批量获取到 {len(ths_index_map)} 个 ThsIndex 对象用于关联。")
            for row in df.itertuples():
                ths_index = ths_index_map.get(row.ts_code)
                if ths_index:
                    ths_index_daily_dict = self.data_format_process.set_ths_index_daily_data(ths_index=ths_index, df_data=row)
                    ths_index_daily_dicts.append(ths_index_daily_dict)
                else:
                    logger.warning(f"处理今日（{today_str}）行情时，未在数据库中找到板块 {row.ts_code}，该条记录将被跳过。")
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
        【V2.0 重构】接口：kpl_concept
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
            logger.warning(f"Tushare接口 kpl_concept 未返回 {trade_date_str} 的数据。")
            return {}
        df = df.replace([np.nan, 'nan', 'NaN', ''], None)
        # 1. 更新/创建题材主表 (KplConceptInfo)
        concept_info_list = [
            self.data_format_process.set_kpl_concept_info_data(df_data=row)
            for row in df.itertuples(index=False)
        ]
        await self._save_all_to_db_native_upsert(
            model_class=KplConceptInfo,
            data_list=concept_info_list,
            unique_fields=['ts_code'],
        )
        print(f"    - [DAO] 完成 {len(concept_info_list)} 条题材主数据更新。")
        # 2. 准备并保存每日快照数据 (KplConceptDaily)
        all_concept_codes = df['ts_code'].unique().tolist()
        concept_map = await self.get_kpl_concepts_by_codes(all_concept_codes)
        daily_items_to_save = [
            self.data_format_process.set_kpl_concept_daily_data(
                concept_info=concept_map.get(row.ts_code),
                df_data=row
            )
            for row in df.itertuples(index=False) if concept_map.get(row.ts_code)
        ]
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
        【V2.0 重构】接口：kpl_concept_cons
        描述：获取并保存指定日期的开盘啦题材成分股。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的开盘啦题材成分...")
        try:
            # 按日期一次性获取所有成分
            df = self.ts_pro.kpl_concept_cons(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 kpl_concept_cons 失败: {e}", exc_info=True)
            return {}
        if df is None or df.empty:
            logger.warning(f"Tushare接口 kpl_concept_cons 未返回 {trade_date_str} 的数据。")
            return {}
        df = df.replace([np.nan, 'nan', 'NaN', ''], None)
        # 批量获取所需的外键对象
        all_concept_codes = df['ts_code'].unique().tolist()
        all_stock_codes = df['con_code'].unique().tolist()
        concept_map = await self.get_kpl_concepts_by_codes(all_concept_codes)
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        items_to_save = []
        concept_member_sync_list = []
        for row in df.itertuples(index=False):
            concept_info = concept_map.get(row.ts_code)
            stock = stock_map.get(row.con_code)
            if concept_info and stock:
                items_to_save.append(
                    self.data_format_process.set_kpl_concept_member_data(
                        concept_info=concept_info,
                        stock=stock,
                        df_data=row
                    )
                )
                # --- 准备同步到 ConceptMember 的数据 ---
                # 开盘啦是每日快照，in_date是当天，out_date是None
                concept_member_sync_list.append({
                    'concept_code': concept_info.ts_code,
                    'stock': stock,
                    'source': 'kpl',
                    'in_date': self.data_format_process._parse_datetime(row.trade_date),
                    'out_date': None
                })
            else:
                if not concept_info:
                    logger.warning(f"未在主表中找到题材 {row.ts_code}，成分股 {row.con_code} 将被忽略。")
                if not stock:
                    logger.warning(f"未在主表中找到股票 {row.con_code}，其成分记录将被忽略。")
        if not items_to_save:
            return {}
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
        【V3.0 历史回补专用版】接口：dc_member
        描述：获取指定东方财富板块(ts_code)的【全部历史】成分数据。
        - 核心逻辑: 不传入 trade_date，并使用 offset 分页，拉取该板块自成立以来的所有成分变动记录。
        - 性能优化: 移除了 N+1 查询，通过批量获取股票信息大幅提升性能。
        - 速率控制: 集成了速率限制器，确保调用安全。
        """
        print(f"  - [DAO-历史回补] [东方财富] 开始获取板块 {ts_code} 的全部历史成分...")
        # 1. 获取速率限制器和板块对象
        limiter = rate_limiter_factory.get_limiter(name='api_dc_member')
        dc_index = await self.get_dc_index_by_code(ts_code)
        if not dc_index:
            logger.error(f"无法在数据库中找到 ts_code={ts_code} 的东方财富板块，任务终止。")
            return {}
        # 2. 分页循环获取所有历史数据
        all_dfs = []
        offset = 0
        limit = 5000 # 根据API文档，单次最大5000
        while True:
            if offset >= 100000: # Tushare对某些接口有10万行的总数据量限制
                logger.warning(f"板块 {ts_code} 历史成分获取已达10万行上限，停止获取。")
                break
            try:
                # 在每次API调用前检查速率限制
                while not await limiter.acquire():
                    print(f"PID[{os.getpid()}] API[api_dc_member] 速率超限，等待20秒后重试... (板块: {ts_code}, offset: {offset})")
                    await asyncio.sleep(20)
                # API调用，不指定trade_date以获取全部历史
                df = self.ts_pro.dc_member(
                    ts_code=ts_code, 
                    limit=limit, 
                    offset=offset,
                    fields=["trade_date", "ts_code", "con_code", "name"]
                )
            except Exception as e:
                logger.error(f"[东方财富板块成分] 获取板块 {ts_code} 历史成分时(offset={offset})失败: {e}", exc_info=True)
                break
            if df is None or df.empty:
                break
            all_dfs.append(df)
            if len(df) < limit:
                break
            offset += limit
        if not all_dfs:
            logger.warning(f"未获取到板块 {ts_code} 的任何历史成分数据。")
            return {}
        # 3. 合并数据并批量处理
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.replace(['nan', 'NaN', ''], None).dropna(subset=['con_code', 'trade_date'])
        # 4. 批量获取所有涉及的股票信息 (消除N+1查询)
        all_stock_codes = combined_df['con_code'].unique().tolist()
        print(f"    - [DAO-历史回补] 批量获取 {len(all_stock_codes)} 个相关股票信息...")
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        # 5. 组装数据
        members_to_save = []
        concept_member_sync_list = []
        for row in combined_df.itertuples(index=False):
            stock = stock_map.get(row.con_code)
            if stock:
                # 组装 DcIndexMember 数据
                member_dict = self.data_format_process.set_dc_index_member_data(dc_index=dc_index, stock=stock, df_data=row)
                members_to_save.append(member_dict)
                # 准备同步到 ConceptMember 的数据
                concept_member_sync_list.append({
                    'concept_code': dc_index.ts_code,
                    'stock': stock,
                    'source': 'dc',
                    'in_date': self.data_format_process._parse_datetime(row.trade_date),
                    'out_date': None # 快照模式没有out_date
                })
        # 6. 批量保存到 DcIndexMember 和 ConceptMember
        result = {}
        if members_to_save:
            print(f"    - [DAO-历史回补] 准备为板块 {ts_code} 保存 {len(members_to_save)} 条历史成分数据...")
            result = await self._save_all_to_db_native_upsert(
                model_class=DcIndexMember,
                data_list=members_to_save,
                unique_fields=['dc_index', 'stock', 'trade_time']
            )
            await self._sync_to_concept_member(concept_member_sync_list, 'dc')
            print(f"    - [DAO-历史回补] 板块 {ts_code} 历史成分保存完成。")
        return result
    async def save_dc_index_members_by_date(self, trade_date: date) -> Dict:
        """
        【V1.2 limit更新版】接口：dc_member
        描述：按天获取所有东方财富板块的成分股。
        修复：
        1. (保留) 增加了对 Tushare API 的速率限制。
        2. 将 limit 更新为接口支持的最大值 8000。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"    -> 开始获取 [东方财富板块成分] 数据, 日期: {trade_date_str}...")
        # 1. 获取速率限制器实例
        limiter = rate_limiter_factory.get_limiter(name='api_dc_member')
        # 2. 获取所有已知的东方财富板块
        all_dc_indices = await self.get_dc_index_list()
        if not all_dc_indices:
            logger.warning("数据库中未找到任何东方财富板块，无法获取成分股。")
            return {"status": "warning", "message": "No DcIndex found in DB."}
        print(f"    - 将为 {len(all_dc_indices)} 个板块获取成分股...")
        all_raw_members = []
        all_stock_codes = set()
        # 3. 循环获取每个板块的成分
        for i, dc_index in enumerate(all_dc_indices):
            print(f"      - 进度 {i+1}/{len(all_dc_indices)}: [东方财富板块成分] 获取板块 [{dc_index.name or dc_index.ts_code}] 成分...")
            offset = 0
            limit = 8000 # 根据Tushare最新文档，dc_member单次最大可获取8000条
            while True:
                if offset >= 100000:
                    logger.warning(f"板块 {dc_index.ts_code} 成分股获取已达10万行上限，停止获取。")
                    break
                try:
                    # 在每次API调用前检查速率限制
                    while not await limiter.acquire():
                        print(f"PID[{os.getpid()}] API[api_dc_member] 速率超限，等待20秒后重试... (板块: {dc_index.ts_code})")
                        await asyncio.sleep(20)
                    df = self.ts_pro.dc_member(trade_date=trade_date_str, ts_code=dc_index.ts_code, limit=limit, offset=offset)
                    if df is None or df.empty:
                        break
                    for row in df.itertuples(index=False):
                        all_raw_members.append((dc_index, row))
                        all_stock_codes.add(row.con_code)
                    if len(df) < limit:
                        break
                    offset += limit
                except Exception as e:
                    logger.error(f"[东方财富板块成分] 获取板块 {dc_index.ts_code} 成分时(offset={offset})失败: {e}", exc_info=True)
                    if '次' in str(e):
                        await asyncio.sleep(10)
                    break 
        if not all_raw_members:
            logger.warning(f"未获取到 {trade_date_str} 的任何东方财富板块成分数据。")
            return {}
        # 4. 批量获取股票信息
        print(f"    - [东方财富板块成分] 正在批量获取 {len(all_stock_codes)} 个股票的信息...")
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(list(all_stock_codes))
        # 5. 组装数据
        members_to_save = []
        concept_member_sync_list = []
        for dc_index, row_data in all_raw_members:
            stock = stock_map.get(row_data.con_code)
            if stock:
                member_dict = self.data_format_process.set_dc_index_member_data(
                    dc_index=dc_index,
                    stock=stock,
                    df_data=row_data
                )
                members_to_save.append(member_dict)
                # --- 准备同步到 ConceptMember 的数据 ---
                # 东方财富是每日快照，in_date是当天，out_date是None
                concept_member_sync_list.append({
                    'concept_code': dc_index.ts_code,
                    'stock': stock,
                    'source': 'dc',
                    'in_date': self.data_format_process._parse_datetime(row_data.trade_date),
                    'out_date': None
                })
            # else:
                # logger.warning(f"未在DB中找到股票 {row_data.con_code}，板块 {dc_index.ts_code} 的此条成分记录将被忽略。")
        if not members_to_save:
            return {}
        # 6. 批量保存
        print(f"    - 准备保存 {len(members_to_save)} 条东方财富板块成分数据...")
        result = await self._save_all_to_db_native_upsert(
            model_class=DcIndexMember,
            data_list=members_to_save,
            unique_fields=['trade_time', 'dc_index', 'stock']
        )
        print(f"    -- 完成 [东方财富板块成分] 数据获取，共保存 {len(members_to_save)} 条。")
        # --- 步骤7: 同步到 ConceptMember ---
        await self._sync_to_concept_member(concept_member_sync_list, 'dc')
        return result
    async def _sync_to_concept_member(self, sync_list: List[Dict], source: str):
        """
        将格式化后的成分股数据同步到统一的 ConceptMember 表。
        """
        if not sync_list:
            return
        print(f"    -> [同步任务] 开始同步 {len(sync_list)} 条 [{source.upper()}] 成分股数据到 [ConceptMember]...")
        # 1. 批量获取所有需要的 ConceptMaster 对象
        all_concept_codes = {item['concept_code'] for item in sync_list}
        concept_map = await self.get_concepts_by_codes(list(all_concept_codes))
        # 2. 组装最终数据，将 concept_code 替换为 concept_id
        final_data_to_save = []
        for item in sync_list:
            concept_master = concept_map.get(item['concept_code'])
            if concept_master:
                item['concept'] = concept_master
                item.pop('concept_code') # 移除临时的 code 字段
                final_data_to_save.append(item)
            else:
                logger.warning(f"同步到 ConceptMember 时未找到代码为 {item['concept_code']} 的主概念记录，已跳过。")
        # 3. 批量保存
        if final_data_to_save:
            await self._save_all_to_db_native_upsert(
                model_class=ConceptMember,
                data_list=final_data_to_save,
                unique_fields=['concept', 'stock', 'in_date', 'source']
            )
            print(f"    -- [同步任务] 完成，成功处理 {len(final_data_to_save)} 条 [{source.upper()}] 成分股数据。")
    # ============== 市场情绪与涨跌停数据 ==============
    async def save_limit_list_ths_by_date(self, trade_date: date) -> Dict:
        """
        【V1.1 速率限制修复版】接口：limit_list_ths
        描述：获取并保存指定交易日的同花顺涨跌停榜单数据。
        修复：增加了对 Tushare API 的速率限制，确保代码健壮性。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的同花顺涨跌停榜单...")
        # 1. 获取速率限制器实例
        limiter = rate_limiter_factory.get_limiter(name='api_limit_list_ths')
        # 2. 定义要获取的榜单类型
        limit_types = ['涨停池', '连扳池', '冲刺涨停', '炸板池', '跌停池']
        all_items_df = []
        for l_type in limit_types:
            try:
                # 在每次API调用前检查速率限制
                while not await limiter.acquire():
                    print(f"PID[{os.getpid()}] API[api_limit_list_ths] 速率超限，等待20秒后重试... (类型: {l_type})")
                    await asyncio.sleep(20)
                df = self.ts_pro.limit_list_ths(trade_date=trade_date_str, limit_type=l_type)
                if df is not None and not df.empty:
                    all_items_df.append(df)
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
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        # 组装数据
        items_to_save = [
            self.market_format_process.set_limit_list_ths_data(stock=stock_map.get(row.ts_code), df_data=row)
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
        【V1.1 速率限制修复版】接口：limit_list_d
        描述：获取并保存指定交易日的Tushare版A股涨跌停列表数据。
        修复：增加了对 Tushare API 的速率限制 (200次/分钟)。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的Tushare涨跌停列表...")
        # 1. 获取速率限制器实例
        limiter = rate_limiter_factory.get_limiter(name='api_limit_list_d')
        limit_types = ['U', 'D', 'Z'] # 涨停, 跌停, 炸板
        all_items_df = []
        for l_type in limit_types:
            try:
                # 2. 在每次API调用前检查速率限制
                while not await limiter.acquire():
                    print(f"PID[{os.getpid()}] API[api_limit_list_d] 速率超限，等待20秒后重试... (类型: {l_type})")
                    await asyncio.sleep(20)
                df = self.ts_pro.limit_list_d(trade_date=trade_date_str, limit_type=l_type)
                if df is not None and not df.empty:
                    all_items_df.append(df)
            except Exception as e:
                logger.error(f"调用Tushare接口 limit_list_d (limit_type={l_type}) 失败: {e}", exc_info=True)
                continue
        if not all_items_df:
            logger.warning(f"Tushare接口 limit_list_d 未返回 {trade_date_str} 的任何数据。")
            return {}
        combined_df = pd.concat(all_items_df, ignore_index=True)
        combined_df = combined_df.replace([np.nan, 'nan', 'NaN', ''], None)
        all_stock_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        items_to_save = [
            self.market_format_process.set_limit_list_d_data(stock=stock_map.get(row.ts_code), df_data=row)
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
        【V1.1 速率限制修复版】接口：limit_step
        描述：获取并保存指定交易日的连板天梯数据。
        修复：增加了对 Tushare API 的速率限制 (500次/分钟)。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的连板天梯数据...")
        # 1. 获取速率限制器实例
        limiter = rate_limiter_factory.get_limiter(name='api_limit_step')
        try:
            # 2. 在API调用前检查速率限制
            while not await limiter.acquire():
                print(f"PID[{os.getpid()}] API[api_limit_step] 速率超限，等待20秒后重试...")
                await asyncio.sleep(20)
            df = self.ts_pro.limit_step(trade_date=trade_date_str)
        except Exception as e:
            logger.error(f"调用Tushare接口 limit_step 失败: {e}", exc_info=True)
            return {}
        if df is None or df.empty:
            logger.warning(f"Tushare接口 limit_step 未返回 {trade_date_str} 的数据。")
            return {}
        df = df.replace([np.nan, 'nan', 'NaN', ''], None)
        all_stock_codes = df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        items_to_save = [
            self.market_format_process.set_limit_step_data(stock=stock_map.get(row.ts_code), df_data=row)
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
        【V1.1 速率限制修复版】接口：limit_cpt_list
        描述：获取并保存指定交易日的最强板块统计数据。
        修复：增加了对 Tushare API 的速率限制 (500次/分钟)。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"  - [DAO] 开始获取 {trade_date_str} 的最强板块统计...")
        # 1. 获取速率限制器实例
        limiter = rate_limiter_factory.get_limiter(name='api_limit_cpt_list')
        try:
            # 2. 在API调用前检查速率限制
            while not await limiter.acquire():
                print(f"PID[{os.getpid()}] API[api_limit_cpt_list] 速率超限，等待20秒后重试...")
                await asyncio.sleep(20)
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
            self.market_format_process.set_limit_cpt_list_data(ths_index=index_map.get(row.ts_code), df_data=row)
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
        """【V1.2 终极修复版】获取申万行业概念，修复 FieldError，并使用内存映射优化性能。"""
        cache_key = self.cache_key_stock.stock_concepts(stock_code, 'sw')
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        concepts = []
        try:
            # 步骤1: 获取该股票所属的L3行业成员关系
            memberships = SwIndustryMember.objects.filter(
                stock__stock_code=stock_code, is_new='Y'
            ).select_related('l3_industry') # 仅预加载直接关联的L3行业
            # 步骤2: 一次性获取所有申万行业数据到内存字典中，以供快速查找
            all_sw_industries_qs = await sync_to_async(list)(SwIndustry.objects.all())
            sw_industry_map = {ind.index_code: ind for ind in all_sw_industries_qs}
            async for member in memberships:
                l3 = member.l3_industry
                if l3 and l3.index_code in sw_industry_map:
                    concepts.append({'code': l3.index_code, 'name': l3.industry_name, 'source': 'sw'})
                    # 步骤3: 在内存中查找父级行业
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























