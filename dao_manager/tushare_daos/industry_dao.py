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
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.market import LimitCptList, LimitListD, LimitListThs, LimitStep
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from stock_models.industry import KplConceptInfo, KplConceptDaily, KplConceptConstituent, KplLimitList, DcIndexDaily, DcIndexMember, SwIndustry, SwIndustryDaily, SwIndustryMember, ThsIndex, ThsIndexMember, ThsIndexDaily, DcIndex
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
        【V2.2 健壮性修复版】获取申万行业成分并保存
        修复与优化:
        1.  【数据不一致处理】新增逻辑：当发现成分股所属行业在数据库中不存在时，不再忽略，而是利用成分股数据中的层级信息，自动创建缺失的L1, L2, L3行业。
        2.  (保留) 修复API调用错误、方法缺失错误、字段错误。
        3.  (保留) 消除N+1查询，采用批量获取、内存映射的方式。
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
        for row in all_raw_members:
            # 按层级检查并创建缺失的行业
            try:
                # 检查并创建L1
                if row.l1_code and row.l1_code not in sw_industry_map:
                    print(f"    - 发现新的L1行业，正在创建: {row.l1_name} ({row.l1_code})")
                    l1_obj, _ = await sync_to_async(SwIndustry.objects.get_or_create)(
                        index_code=row.l1_code,
                        defaults={
                            'industry_name': row.l1_name, 'level': 'L1', 'parent_code': '0', 'src': 'SW2021', 'industry_code': row.l1_code
                        }
                    )
                    sw_industry_map[row.l1_code] = l1_obj
                # 检查并创建L2
                if row.l2_code and row.l2_code not in sw_industry_map:
                    print(f"    - 发现新的L2行业，正在创建: {row.l2_name} ({row.l2_code})")
                    l2_obj, _ = await sync_to_async(SwIndustry.objects.get_or_create)(
                        index_code=row.l2_code,
                        defaults={
                            'industry_name': row.l2_name, 'level': 'L2', 'parent_code': row.l1_code, 'src': 'SW2021', 'industry_code': row.l2_code
                        }
                    )
                    sw_industry_map[row.l2_code] = l2_obj
                # 检查并创建L3
                if row.l3_code and row.l3_code not in sw_industry_map:
                    print(f"    - 发现新的L3行业，正在创建: {row.l3_name} ({row.l3_code})")
                    l3_obj, _ = await sync_to_async(SwIndustry.objects.get_or_create)(
                        index_code=row.l3_code,
                        defaults={
                            'industry_name': row.l3_name, 'level': 'L3', 'parent_code': row.l2_code, 'src': 'SW2021', 'industry_code': row.l3_code
                        }
                    )
                    sw_industry_map[row.l3_code] = l3_obj
            except Exception as e:
                logger.error(f"动态创建申万行业时失败: {e}, 数据: {row}", exc_info=True)
                continue
            swan_industry = sw_industry_map.get(row.l3_code)
            stock = stock_map.get(row.ts_code)
            # 修改判断条件，确保在动态创建后，行业和股票都存在
            if swan_industry and stock:
                industry_member_dict = self.data_format_process.set_sw_industry_member_data(
                    sw_industry=swan_industry,
                    stock=stock,
                    df_data=row
                )
                industry_member_dicts.append(industry_member_dict)
            else:
                # 这里的 warning 现在只会在股票不存在或行业创建失败时触发
                if not swan_industry: logger.warning(f"动态创建后仍未找到申万三级行业 {row.l3_code}，成分股 {row.ts_code} 将被忽略。")
                if not stock: logger.warning(f"未在DB中找到股票 {row.ts_code}，成分股记录将被忽略。")
        # --- 5. 批量保存 ---
        if industry_member_dicts:
            print(f"    - 准备保存 {len(industry_member_dicts)} 条申万行业成分数据...")
            final_result = await self._save_all_to_db_native_upsert(
                model_class=SwIndustryMember,
                data_list=industry_member_dicts,
                unique_fields=['l3_industry', 'stock', 'in_date']
            )
            print(f"  - [DAO] 完成保存申万行业成分。结果: {final_result}")
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
        接口：sw_daily
        描述：获取申万行业日线行情（默认是申万2021版行情）
        限量：单次最大4000行数据，可通过指数代码和日期参数循环提取，5000积分可调取
        修复:
        1.  【方法签名错误】修复：将参数从 `trade_time` 修改为 `trade_date`，并设为关键字参数，以匹配调用方的传参方式。
        Returns:
            Dict: 保存结果
        """
        result = {}
        industry_daily_basic_dicts = []
        if trade_date is None:
            trade_date = datetime.today()
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"DEBUG: save_sw_industry_daily - trade_date_str: {trade_date_str}")
        # 拉取数据
        df = self.ts_pro.sw_daily(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": "", "end_date": "", "limit": "", "offset": ""
            }, fields=[
                "ts_code", "trade_date", "name", "open", "low", "high", "close", "change", "pct_change", "vol",
                "amount", "pe", "pb", "float_mv", "total_mv", "weight"
            ])
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                index_basic = await self.index_info_dao.get_index_by_code(row.ts_code)
                industry_daily_basic_dict = self.data_format_process.set_sw_industry_daily_data(index=index_basic, df_data=row)
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
                    # logger.warning(f"板块 [{ths_index.name}] 未返回任何成分股数据，跳过。")
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
            update_fields=['name']
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
        return result

    async def get_kpl_concepts_by_codes(self, codes: List[str]) -> Dict[str, KplConceptInfo]:
        """【V2.0 新增】根据题材代码列表批量获取 KplConceptInfo 对象"""
        if not codes:
            return {}
        instances = await self.filter_async(KplConceptInfo, ts_code__in=codes)
        return {instance.ts_code: instance for instance in instances}

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
            data_list=unique_indices,  # 修改行: 直接使用 unique_indices
            unique_fields=['ts_code']
        )
        
        print(f"    -- 完成 [东方财富板块列表] 更新，共处理 {len(unique_indices)} 条板块元数据。") # 修改行: 更新日志输出变量
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

    async def save_dc_index_daily_by_trade_time(self, trade_time: date = None) -> Dict:
        """
        【V3.0 重构修复版】接口：dc_daily
        描述：获取东方财富概念板块的日线行情数据。
        修复：
        1.  【KeyError修复】完全重写数据处理逻辑，以匹配 `dc_daily` API的实际返回字段（OHLC等）。
        2.  【分页逻辑】新增了 while 循环和 offset，以完整获取当日所有板块的行情数据。
        3.  【模型匹配】确保数据格式化与重构后的 `DcIndexDaily` 模型完全对应。
        4.  (保留) 保留了“按需创建新板块”的健壮性设计。
        """
        if trade_time is None:
            trade_time = datetime.today().date()
        trade_time_str = trade_time.strftime('%Y%m%d')
        print(f"    -> 开始获取 [东方财富板块行情] 数据, 日期: {trade_time_str}...")
        all_dfs = []
        offset = 0
        limit = 2000 # dc_daily 的 limit 是 2000
        max_offset = 100000
        while offset < max_offset:
            try:
                df = self.ts_pro.dc_daily(trade_date=trade_time_str, limit=limit, offset=offset)
                if df is None or df.empty:
                    print(f"    - 在 offset={offset} 处未获取到更多行情数据，分页结束。")
                    break
                all_dfs.append(df)
                if len(df) < limit:
                    print(f"    - 获取到 {len(df)} 条行情，少于 limit={limit}，认定为最后一页。")
                    break
                offset += limit
                print(f"    - 已获取 {offset} 条行情，继续下一页...")
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"调用Tushare接口 dc_daily (offset={offset}) 失败: {e}", exc_info=True)
                break
        if not all_dfs:
            logger.warning(f"Tushare接口 dc_daily 未返回 {trade_time_str} 的数据。")
            return {"status": "warning", "message": "API returned no data."}
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.replace([np.nan, 'nan', 'NaN', ''], None)
        # --- 批量获取和创建板块主数据 ---
        all_index_codes = combined_df['ts_code'].unique().tolist()
        dc_index_map = await self.get_dc_indices_by_codes(all_index_codes)
        new_indices_to_create = []
        for row in combined_df[['ts_code']].drop_duplicates().itertuples(index=False):
            if row.ts_code not in dc_index_map:
                # 注意：dc_daily不返回name，所以这里我们只用ts_code创建，name可以后续通过dc_index任务补全
                new_indices_to_create.append({'ts_code': row.ts_code, 'name': row.ts_code})
        if new_indices_to_create:
            print(f"    - 发现 {len(new_indices_to_create)} 个新的东方财富板块，正在创建...")
            await self._save_all_to_db_native_upsert(
                model_class=DcIndex,
                data_list=new_indices_to_create,
                unique_fields=['ts_code'],
                update_fields=['name'] # 如果已存在，可以尝试更新name
            )
            dc_index_map.update(await self.get_dc_indices_by_codes([d['ts_code'] for d in new_indices_to_create]))
            print(f"    - 新板块创建完成。")
        # --- 组装日线数据 ---
        dc_index_daily_dicts = []
        for row in combined_df.itertuples(index=False):
            dc_index = dc_index_map.get(row.ts_code)
            if not dc_index:
                logger.warning(f"创建后仍未找到东方财富板块 {row.ts_code}，跳过此条日线数据。")
                continue
            # 关键修复：调用新的数据格式化方法，匹配 dc_daily API 和新模型
            daily_dict = self.data_format_process.set_dc_index_daily_data(
                dc_index=dc_index,
                df_data=row
            )
            dc_index_daily_dicts.append(daily_dict)
        if not dc_index_daily_dicts:
            return {}
        # --- 批量保存日线数据 ---
        result = await self._save_all_to_db_native_upsert(
            model_class=DcIndexDaily,
            data_list=dc_index_daily_dicts,
            unique_fields=['dc_index', 'trade_time']
        )
        print(f"    -- 完成 [东方财富板块行情] 数据获取，共 {len(dc_index_daily_dicts)} 条。")
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
                dc_index_member_dict = self.data_format_process.set_dc_index_member_data(dc_index=dc_index, stock=stock, df_data=row) # 修改行: 调用修正后的方法名
                dc_index_member_dicts.append(dc_index_member_dict)
        if dc_index_member_dicts:
            # 保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=DcIndexMember,
                data_list=dc_index_member_dicts,
                unique_fields=['dc_index', 'stock', 'trade_time']
            )
        return result

    async def save_dc_index_members_by_date(self, trade_date: date) -> Dict:
        """
        【新增】接口：dc_member
        描述：按天获取所有东方财富板块的成分股。
        优化：
        1. 先获取所有板块，再循环调用API，逻辑清晰。
        2. 对 dc_member 接口调用增加了分页逻辑，确保单个板块成分过多时也能完整获取。
        3. 采用批量获取、批量保存的策略，提升数据库效率。
        """
        trade_date_str = trade_date.strftime('%Y%m%d')
        print(f"    -> 开始获取 [东方财富板块成分] 数据, 日期: {trade_date_str}...")
        # 1. 获取所有已知的东方财富板块
        all_dc_indices = await self.get_dc_index_list()
        if not all_dc_indices:
            logger.warning("数据库中未找到任何东方财富板块，无法获取成分股。")
            return {"status": "warning", "message": "No DcIndex found in DB."}
        print(f"    - 将为 {len(all_dc_indices)} 个板块获取成分股...")
        all_raw_members = []
        all_stock_codes = set()
        # 2. 循环获取每个板块的成分
        for i, dc_index in enumerate(all_dc_indices):
            print(f"      - 进度 {i+1}/{len(all_dc_indices)}: 获取板块 [{dc_index.name or dc_index.ts_code}] 成分...")
            offset = 0
            limit = 5000
            while True:
                try:
                    df = self.ts_pro.dc_member(trade_date=trade_date_str, ts_code=dc_index.ts_code, limit=limit, offset=offset)
                    if df is None or df.empty:
                        break
                    
                    for row in df.itertuples(index=False):
                        all_raw_members.append((dc_index, row))
                        all_stock_codes.add(row.con_code)

                    if len(df) < limit:
                        break
                    offset += limit
                    await asyncio.sleep(0.2)
                except Exception as e:
                    logger.error(f"获取板块 {dc_index.ts_code} 成分时(offset={offset})失败: {e}", exc_info=True)
                    break # 单个板块出错，继续下一个

        if not all_raw_members:
            logger.warning(f"未获取到 {trade_date_str} 的任何板块成分数据。")
            return {}
        # 3. 批量获取股票信息
        print(f"    - 正在批量获取 {len(all_stock_codes)} 个股票的信息...")
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(list(all_stock_codes))
        # 4. 组装数据
        members_to_save = []
        for dc_index, row_data in all_raw_members:
            stock = stock_map.get(row_data.con_code)
            if stock:
                member_dict = self.data_format_process.set_dc_index_member_data(
                    dc_index=dc_index,
                    stock=stock,
                    df_data=row_data
                )
                members_to_save.append(member_dict)
            else:
                logger.warning(f"未在DB中找到股票 {row_data.con_code}，板块 {dc_index.ts_code} 的此条成分记录将被忽略。")
        if not members_to_save:
            return {}
        # 5. 批量保存
        print(f"    - 准备保存 {len(members_to_save)} 条东方财富板块成分数据...")
        result = await self._save_all_to_db_native_upsert(
            model_class=DcIndexMember,
            data_list=members_to_save,
            unique_fields=['trade_time', 'dc_index', 'stock']
        )
        print(f"    -- 完成 [东方财富板块成分] 数据获取，共保存 {len(members_to_save)} 条。")
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
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
        # 组装数据
        items_to_save = [
            self.market_format_process.set_limit_step_data(stock=stock_map.get(row.ts_code), df_data=row)
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
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
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
        stock_map = await self.stock_basic_info_dao.get_stocks_by_codes(all_stock_codes)
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
































