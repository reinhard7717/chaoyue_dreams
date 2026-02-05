
import asyncio
import logging
from datetime import date, datetime, timedelta
import time
from typing import Dict, List
from asgiref.sync import sync_to_async
import numpy as np
import pandas as pd
from utils.cache_manager import CacheManager
from utils.cache_get import StockInfoCacheGet, UserCacheGet
from utils.cache_set import StockInfoCacheSet, UserCacheSet
from utils.cash_key import StockCashKey
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from utils.model_helpers import get_fund_flow_model_by_code, get_fund_flow_ths_model_by_code, get_fund_flow_dc_model_by_code
from stock_models.fund_flow import FundFlowCntDC, FundFlowCntTHS,FundFlowIndustryTHS, FundFlowMarketDc, TopInst, TopList
from stock_models.market import HmDetail, HmList, LimitListThs
from utils.data_format_process import FundFlowFormatProcess

logger = logging.getLogger("dao")

class FundFlowDao(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.data_format_process = FundFlowFormatProcess(cache_manager_instance)
        self.index_dao = IndexBasicDAO(cache_manager_instance)
        self.stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
        self.industry_dao = IndustryDao(cache_manager_instance)
        self.stock_cache_key = StockCashKey()
        self.stock_cache_set = StockInfoCacheSet(self.cache_manager)
        self.stock_cache_get = StockInfoCacheGet(self.cache_manager)
        self.user_cache_set = UserCacheSet(self.cache_manager)
        self.user_cache_get = UserCacheGet(self.cache_manager)

    # ============== 日级资金流向数据 ==============
    async def get_fund_flow_daily_data(self, stock_code: str, trade_date: date, limit: int) -> pd.DataFrame:
        """
        获取单个股票的历史日级资金流向数据 (V2.0 - 读取性能优化)
        优化：使用 sync_to_async(list) 替代异步推导式，减少大量数据行遍历时的上下文切换开销。
        """
        model_class = get_fund_flow_model_by_code(stock_code)
        if not model_class:
            logger.warning(f"无法为股票 {stock_code} 确定常规资金流向数据模型。")
            return pd.DataFrame()
        try:
            qs = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__lte=trade_date
            ).order_by('-trade_time')[:limit]
            # 优化点：将 QuerySet 的求值和列表构建放入同步线程，避免逐行 await
            data_list = await sync_to_async(list)(qs.values())
            if not data_list:
                return pd.DataFrame()
            df = pd.DataFrame(data_list)
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.drop(columns=['id', 'stock_id'], inplace=True, errors='ignore')
            return df
        except Exception as e:
            logger.error(f"查询常规日级资金流数据时出错 (stock: {stock_code}): {e}", exc_info=True)
            return pd.DataFrame()

    async def save_history_fund_flow_daily_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        保存历史日级资金流向数据 (V3.2 - 向量化模型映射与类型优化)
        1. [优化] 使用 map 替代 apply 进行模型匹配，复杂度由 O(N) 降为 O(M)。
        2. [优化] 显式批量转换数值列类型，确保数据纯净。
        3. 保留了分块分页获取和动态分表存储逻辑。
        """
        if start_date and end_date and start_date > end_date:
            logger.error(f"日期范围无效：起始日期 {start_date} 不能晚于结束日期 {end_date}。任务终止。")
            return
        if start_date and end_date:
            logger.info(f"接收到范围任务，将对 {start_date} 到 {end_date} 的数据采用客户端分块策略处理。")
        elif trade_date:
            start_date = end_date = trade_date
            logger.info(f"接收到单日任务: {trade_date}")
        else:
            start_date = end_date = date.today()
            logger.info(f"未提供日期，默认获取今日数据: {start_date}")
        date_chunks = []
        chunk_size_days = 10
        current_chunk_end = end_date
        while current_chunk_end >= start_date:
            current_chunk_start = max(start_date, current_chunk_end - timedelta(days=chunk_size_days - 1))
            date_chunks.append((current_chunk_start, current_chunk_end))
            current_chunk_end = current_chunk_start - timedelta(days=1)
        all_dfs_for_market = []
        for chunk_start, chunk_end in date_chunks:
            chunk_start_str = chunk_start.strftime('%Y%m%d')
            chunk_end_str = chunk_end.strftime('%Y%m%d')
            print(f"DAO: 开始处理日期块: {chunk_start_str} 到 {chunk_end_str}")
            offset = 0
            limit = 6000
            while True:
                try:
                    df = self.ts_pro.moneyflow(**{
                        "ts_code": "", "trade_date": "",
                        "start_date": chunk_start_str, "end_date": chunk_end_str,
                        "limit": limit, "offset": offset
                    }, fields=[
                        "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount",
                        "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount",
                        "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount",
                        "net_mf_vol", "net_mf_amount", "trade_count"
                    ])
                    await asyncio.sleep(0.55)
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (moneyflow, chunk: {chunk_start_str}-{chunk_end_str}): {e}")
                    await asyncio.sleep(5)
                    df = pd.DataFrame()
                if df.empty:
                    break
                all_dfs_for_market.append(df)
                if len(df) < limit:
                    break
                offset += limit
        if not all_dfs_for_market:
            logger.info("在所有日期块中均未获取到任何资金流向数据。")
            return
        combined_df = pd.concat(all_dfs_for_market, ignore_index=True)
        combined_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 批量数值转换
        numeric_cols = [
            "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount",
            "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount",
            "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount",
            "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount",
            "net_mf_vol", "net_mf_amount", "trade_count"
        ]
        # 仅处理存在的列
        cols_to_convert = [c for c in numeric_cols if c in combined_df.columns]
        if cols_to_convert:
            combined_df[cols_to_convert] = combined_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        all_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_ts_codes)
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info("数据关联股票基础信息后为空，任务结束。")
            return
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 优化：使用 map 替代 apply 进行模型匹配
        unique_codes = combined_df['ts_code'].unique()
        model_map = {code: get_fund_flow_model_by_code(code) for code in unique_codes}
        combined_df['target_model'] = combined_df['ts_code'].map(model_map)
        total_rows = 0
        for model, group_df in combined_df.groupby('target_model', sort=False):
            if group_df.empty:
                continue
            final_df = group_df.drop(columns=['ts_code', 'trade_date', 'target_model'])
            # 将 NaN 替换为 None (针对数值列转换后可能产生的 NaN)
            final_df = final_df.where(pd.notnull(final_df), None)
            data_list = final_df.to_dict('records')
            await self._save_all_to_db_native_upsert(
                model_class=model,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            total_rows += len(data_list)
        return

    # ============== 个股日级资金流向数据 - 同花顺 ==============
    async def get_fund_flow_ths_data(self, stock_code: str, trade_date: date, limit: int) -> pd.DataFrame:
        """
        获取单个股票的历史日级资金流向数据 (同花顺) (V2.0 - 读取性能优化)
        优化：使用 sync_to_async(list) 替代异步推导式。
        """
        model_class = get_fund_flow_ths_model_by_code(stock_code)
        if not model_class:
            logger.warning(f"无法为股票 {stock_code} 确定同花顺资金流向数据模型。")
            return pd.DataFrame()
        try:
            qs = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__lte=trade_date
            ).order_by('-trade_time')[:limit]
            # 优化点：批量获取
            data_list = await sync_to_async(list)(qs.values())
            if not data_list:
                return pd.DataFrame()
            df = pd.DataFrame(data_list)
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.drop(columns=['id', 'stock_id'], inplace=True, errors='ignore')
            return df
        except Exception as e:
            logger.error(f"查询同花顺资金流数据时出错 (stock: {stock_code}): {e}", exc_info=True)
            return pd.DataFrame()

    async def save_history_fund_flow_daily_ths_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        保存历史日级资金流向数据 - 同花顺 (V3.2 - 向量化模型映射与类型优化)
        1. [优化] 使用 map 替代 apply 进行模型匹配。
        2. [优化] 显式批量转换数值列类型。
        """
        if start_date and end_date and start_date > end_date:
            logger.error(f"日期范围无效：起始日期 {start_date} 不能晚于结束日期 {end_date}。任务终止。")
            return
        elif trade_date:
            start_date = end_date = trade_date
            logger.info(f"接收到单日任务: {trade_date}")
        else:
            start_date = end_date = date.today()
            logger.info(f"未提供日期，默认获取今日数据: {start_date}")
        date_chunks = []
        chunk_size_days = 10
        current_chunk_end = end_date
        while current_chunk_end >= start_date:
            current_chunk_start = max(start_date, current_chunk_end - timedelta(days=chunk_size_days - 1))
            date_chunks.append((current_chunk_start, current_chunk_end))
            current_chunk_end = current_chunk_start - timedelta(days=1)
        all_dfs_for_market = []
        for chunk_start, chunk_end in date_chunks:
            chunk_start_str = chunk_start.strftime('%Y%m%d')
            chunk_end_str = chunk_end.strftime('%Y%m%d')
            print(f"DAO: 开始处理同花顺资金流日期块: {chunk_start_str} 到 {chunk_end_str}")
            offset = 0
            limit = 5000
            while True:
                try:
                    df = self.ts_pro.moneyflow_ths(**{
                        "ts_code": "", "trade_date": "",
                        "start_date": chunk_start_str, "end_date": chunk_end_str,
                        "limit": limit, "offset": offset
                    }, fields=[
                        "trade_date", "ts_code", "pct_change", "net_amount", "net_d5_amount", "buy_lg_amount",
                        "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
                    ])
                    await asyncio.sleep(0.85)
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (moneyflow_ths, chunk: {chunk_start_str}-{chunk_end_str}): {e}")
                    await asyncio.sleep(5)
                    df = pd.DataFrame()
                if df.empty:
                    break
                all_dfs_for_market.append(df)
                if len(df) < limit:
                    break
                offset += limit
        if not all_dfs_for_market:
            logger.info("在所有日期块中均未获取到任何同花顺资金流向数据。")
            return
        print("DAO: 所有同花顺分块数据获取完毕，开始进行数据整合与处理...")
        combined_df = pd.concat(all_dfs_for_market, ignore_index=True)
        combined_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 批量数值转换
        numeric_cols = [
            "pct_change", "net_amount", "net_d5_amount", "buy_lg_amount",
            "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ]
        cols_to_convert = [c for c in numeric_cols if c in combined_df.columns]
        if cols_to_convert:
            combined_df[cols_to_convert] = combined_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        all_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_ts_codes)
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info("数据关联股票基础信息后为空，任务结束。")
            return
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 优化：使用 map 替代 apply 进行模型匹配
        unique_codes = combined_df['ts_code'].unique()
        model_map = {code: get_fund_flow_ths_model_by_code(code) for code in unique_codes}
        combined_df['target_model'] = combined_df['ts_code'].map(model_map)
        total_rows = 0
        for model, group_df in combined_df.groupby('target_model', sort=False):
            if group_df.empty:
                continue
            final_df = group_df.drop(columns=['ts_code', 'trade_date', 'target_model'])
            final_df = final_df.where(pd.notnull(final_df), None)
            data_list = final_df.to_dict('records')
            await self._save_all_to_db_native_upsert(
                model_class=model,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            total_rows += len(data_list)
        print(f"所有历史日级资金流向数据(同花顺)处理完成，共保存 {total_rows} 条记录。")
        return

    # ============== 日级资金流向数据 - 东方财富 ==============
    async def get_fund_flow_dc_data(self, stock_code: str, trade_date: date, limit: int) -> pd.DataFrame:
        """
        获取单个股票的历史日级资金流向数据 (东方财富) (V2.0 - 读取性能优化)
        优化：使用 sync_to_async(list) 替代异步推导式。
        """
        model_class = get_fund_flow_dc_model_by_code(stock_code)
        if not model_class:
            logger.warning(f"无法为股票 {stock_code} 确定东方财富资金流向数据模型。")
            return pd.DataFrame()
        try:
            qs = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__lte=trade_date
            ).order_by('-trade_time')[:limit]
            # 优化点：批量获取
            data_list = await sync_to_async(list)(qs.values())
            if not data_list:
                return pd.DataFrame()
            df = pd.DataFrame(data_list)
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.drop(columns=['id', 'stock_id'], inplace=True, errors='ignore')
            return df
        except Exception as e:
            logger.error(f"查询东方财富资金流数据时出错 (stock: {stock_code}): {e}", exc_info=True)
            return pd.DataFrame()

    async def save_history_fund_flow_daily_dc_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        保存历史日级资金流向数据 - 东方财富 (V3.2 - 向量化模型映射与类型优化)
        1. [优化] 使用 map 替代 apply 进行模型匹配。
        2. [优化] 显式批量转换数值列类型。
        """
        if start_date and end_date and start_date > end_date:
            logger.error(f"日期范围无效：起始日期 {start_date} 不能晚于结束日期 {end_date}。任务终止。")
            return
        if start_date and end_date:
            logger.info(f"接收到范围任务，将对 {start_date} 到 {end_date} 的东方财富资金流数据采用客户端分块策略处理。")
        elif trade_date:
            start_date = end_date = trade_date
            logger.info(f"接收到单日任务: {trade_date}")
        else:
            start_date = end_date = date.today()
            logger.info(f"未提供日期，默认获取今日数据: {start_date}")
        date_chunks = []
        chunk_size_days = 10
        current_chunk_end = end_date
        while current_chunk_end >= start_date:
            current_chunk_start = max(start_date, current_chunk_end - timedelta(days=chunk_size_days - 1))
            date_chunks.append((current_chunk_start, current_chunk_end))
            current_chunk_end = current_chunk_start - timedelta(days=1)
        all_dfs_for_market = []
        for chunk_start, chunk_end in date_chunks:
            chunk_start_str = chunk_start.strftime('%Y%m%d')
            chunk_end_str = chunk_end.strftime('%Y%m%d')
            print(f"DAO: 开始处理东方财富资金流日期块: {chunk_start_str} 到 {chunk_end_str}")
            offset = 0
            limit = 5000
            while True:
                try:
                    df = self.ts_pro.moneyflow_dc(**{
                        "ts_code": "", "trade_date": "",
                        "start_date": chunk_start_str, "end_date": chunk_end_str,
                        "limit": limit, "offset": offset
                    }, fields=[
                        "trade_date", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
                        "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
                    ])
                    await asyncio.sleep(0.85)
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (moneyflow_dc, chunk: {chunk_start_str}-{chunk_end_str}): {e}")
                    await asyncio.sleep(5)
                    df = pd.DataFrame()
                if df.empty:
                    break
                all_dfs_for_market.append(df)
                if len(df) < limit:
                    break
                offset += limit
        if not all_dfs_for_market:
            logger.info("在所有日期块中均未获取到任何东方财富资金流向数据。")
            return
        print("DAO: 所有东方财富分块数据获取完毕，开始进行数据整合与处理...")
        combined_df = pd.concat(all_dfs_for_market, ignore_index=True)
        combined_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 批量数值转换
        numeric_cols = [
            "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
            "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
        ]
        cols_to_convert = [c for c in numeric_cols if c in combined_df.columns]
        if cols_to_convert:
            combined_df[cols_to_convert] = combined_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        all_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_ts_codes)
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info("数据关联股票基础信息后为空，任务结束。")
            return
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 优化：使用 map 替代 apply 进行模型匹配
        unique_codes = combined_df['ts_code'].unique()
        model_map = {code: get_fund_flow_dc_model_by_code(code) for code in unique_codes}
        combined_df['target_model'] = combined_df['ts_code'].map(model_map)
        total_rows = 0
        for model, group_df in combined_df.groupby('target_model', sort=False):
            if group_df.empty:
                continue
            final_df = group_df.drop(columns=['ts_code', 'trade_date', 'target_model'])
            final_df = final_df.where(pd.notnull(final_df), None)
            data_list = final_df.to_dict('records')
            await self._save_all_to_db_native_upsert(
                model_class=model,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            total_rows += len(data_list)
        print(f"所有历史日级资金流向数据(东方财富)处理完成，共保存 {total_rows} 条记录。")
        return

    # ============== 板块资金流向数据 - 同花顺 ==============
    async def save_history_fund_flow_cnt_ths_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【V2.1 - 类型优化版】保存历史板块资金流向数据 - 同花顺
        1. [优化] 增加数值列的显式转换，防止字符串入库。
        2. 保留了向量化外键关联和分页逻辑。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        offset = 0
        limit = 6000
        all_data_to_save = []
        while True:
            if offset >= 100000:
                logger.warning(f"板块资金流向数据 - 同花顺 offset已达10万，停止拉取。")
                break
            print(f"调试信息: 正在从Tushare拉取数据, offset={offset}, limit={limit}")
            df = self.ts_pro.moneyflow_cnt_ths(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "ts_code", "name", "lead_stock", "close_price", "pct_change", "industry_index", "company_num", "pct_change_stock",
                "net_buy_amount", "net_sell_amount", "net_amount"
            ])
            if df.empty:
                break
            original_count = len(df)
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                time.sleep(0.2)
                if original_count < limit: break
                offset += limit
                continue
            unique_codes = df['ts_code'].unique().tolist()
            ths_index_map = await self.industry_dao.get_ths_indices_by_codes(unique_codes)
            df['ths_index'] = df['ts_code'].map(ths_index_map)
            df.dropna(subset=['ths_index'], inplace=True)
            if df.empty:
                logger.warning(f"当前批次所有ts_code均未在ThsIndex表中找到对应记录。")
                time.sleep(0.2)
                if original_count < limit: break
                offset += limit
                continue
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 批量数值转换
            numeric_cols = [
                'close_price', 'pct_change', 'industry_index', 'company_num', 'pct_change_stock',
                'net_buy_amount', 'net_sell_amount', 'net_amount'
            ]
            cols_to_convert = [c for c in numeric_cols if c in df.columns]
            if cols_to_convert:
                df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
            final_columns = [
                'ths_index', 'trade_time', 'lead_stock', 'close_price', 'pct_change',
                'industry_index', 'company_num', 'pct_change_stock', 'net_buy_amount',
                'net_sell_amount', 'net_amount'
            ]
            df_final = df[final_columns]
            df_processed = df_final.where(pd.notnull(df_final), None)
            all_data_to_save.extend(df_processed.to_dict('records'))
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何有效数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        result = await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntTHS,
            data_list=all_data_to_save,
            unique_fields=['ths_index', 'trade_time']
        )
        date_range_info = f"trade_date={trade_date_str}" if trade_date_str else f"start={start_date_str}, end={end_date_str}"
        print(f"完成 {date_range_info} 板块资金流向数据（同花顺），result: {result}")
        return result

    # ============== 板块资金流向数据 - 东方财富 ==============
    async def save_history_fund_flow_cnt_dc_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【V2.1 - 类型优化版】保存历史板块资金流向数据 - 东方财富
        1. [优化] 增加数值列的显式转换。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        offset = 0
        limit = 6000
        all_data_to_save = []
        while True:
            if offset >= 100000:
                logger.warning(f"历史板块资金流向数据 - 东方财富 offset已达10万，停止拉取。")
                break
            print(f"调试信息: 正在从Tushare拉取东方财富数据, offset={offset}, limit={limit}")
            df = self.ts_pro.moneyflow_ind_dc(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "content_type", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount",
                "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount",
                "buy_sm_amount_rate", "buy_sm_amount_stock", "rank"
            ])
            if df.empty:
                break
            original_count = len(df)
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                time.sleep(0.2)
                if original_count < limit: break
                offset += limit
                continue
            unique_codes = df['ts_code'].unique().tolist()
            dc_index_map = await self.industry_dao.get_dc_indices_by_codes(unique_codes)
            df['dc_index'] = df['ts_code'].map(dc_index_map)
            df.dropna(subset=['dc_index'], inplace=True)
            if df.empty:
                logger.warning(f"当前批次所有ts_code均未在DcIndex表中找到对应记录。")
                time.sleep(0.2)
                if original_count < limit: break
                offset += limit
                continue
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 批量数值转换
            numeric_cols = [
                'pct_change', 'close', 'net_amount', 'net_amount_rate', 'buy_elg_amount', 'buy_elg_amount_rate',
                'buy_lg_amount', 'buy_lg_amount_rate', 'buy_md_amount', 'buy_md_amount_rate', 'buy_sm_amount',
                'buy_sm_amount_rate', 'buy_sm_amount_stock'
            ]
            cols_to_convert = [c for c in numeric_cols if c in df.columns]
            if cols_to_convert:
                df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
            final_columns = [
                'dc_index', 'trade_time', 'content_type', 'name', 'pct_change', 'close', 'net_amount',
                'net_amount_rate', 'buy_elg_amount', 'buy_elg_amount_rate', 'buy_lg_amount',
                'buy_lg_amount_rate', 'buy_md_amount', 'buy_md_amount_rate', 'buy_sm_amount',
                'buy_sm_amount_rate', 'buy_sm_amount_stock'
            ]
            df_final = df[final_columns]
            df_processed = df_final.where(pd.notnull(df_final), None)
            all_data_to_save.extend(df_processed.to_dict('records'))
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何东方财富板块资金流数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        result = await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntDC,
            data_list=all_data_to_save,
            unique_fields=['dc_index', 'trade_time']
        )
        date_range_info = f"trade_date={trade_date_str}" if trade_date_str else f"start={start_date_str}, end={end_date_str}"
        print(f"完成 {date_range_info} 历史板块资金流向数据（东方财富），result: {result}")
        return result

    # ============== 行业资金流向数据 - 同花顺 ==============
    @sync_to_async
    def get_industry_fund_flow(self, industry_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """获取行业的历史资金流数据"""
        # print(f"    [DAO] 正在获取行业 {industry_code} 从 {start_date} 到 {end_date} 的资金流...")
        qs = FundFlowIndustryTHS.objects.filter(
            ths_index__ts_code=industry_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date
        ).order_by('trade_time')
        df = pd.DataFrame(list(qs.values()))
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
        return df

    async def save_history_fund_flow_industry_ths_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【V2.1 - 类型优化版】保存历史行业资金流向数据 - 同花顺
        1. [优化] 增加数值列的显式转换。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        offset = 0
        limit = 5000
        all_data_to_save = []
        while True:
            if offset >= 100000:
                logger.warning(f"行业资金流向数据 - 同花顺 offset已达10万，停止拉取。")
                break
            print(f"调试信息: 正在从Tushare拉取同花顺行业资金流数据, offset={offset}, limit={limit}")
            df = self.ts_pro.moneyflow_ind_ths(**{
                "ts_code": "", "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
            }, fields=[
                "trade_date", "ts_code", "industry", "lead_stock", "close", "pct_change", "company_num", "pct_change_stock", "close_price",
                "net_buy_amount", "net_sell_amount", "net_amount"
            ])
            if df.empty:
                break
            original_count = len(df)
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                time.sleep(0.2)
                if original_count < limit: break
                offset += limit
                continue
            unique_codes = df['ts_code'].unique().tolist()
            ths_index_map = await self.industry_dao.get_ths_indices_by_codes(unique_codes)
            df['ths_index'] = df['ts_code'].map(ths_index_map)
            df.dropna(subset=['ths_index'], inplace=True)
            if df.empty:
                logger.warning(f"当前批次所有ts_code均未在ThsIndex表中找到对应记录。")
                time.sleep(0.2)
                if original_count < limit: break
                offset += limit
                continue
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 批量数值转换
            numeric_cols = [
                'close', 'pct_change', 'company_num', 'pct_change_stock', 'close_price',
                'net_buy_amount', 'net_sell_amount', 'net_amount'
            ]
            cols_to_convert = [c for c in numeric_cols if c in df.columns]
            if cols_to_convert:
                df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
            final_columns = [
                'ths_index', 'trade_time', 'industry', 'lead_stock', 'close', 'pct_change',
                'company_num', 'pct_change_stock', 'close_price', 'net_buy_amount',
                'net_sell_amount', 'net_amount'
            ]
            df_final = df[final_columns]
            df_processed = df_final.where(pd.notnull(df_final), None)
            all_data_to_save.extend(df_processed.to_dict('records'))
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何同花顺行业资金流数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        result = await self._save_all_to_db_native_upsert(
            model_class=FundFlowIndustryTHS,
            data_list=all_data_to_save,
            unique_fields=['ths_index', 'trade_time']
        )
        date_range_info = f"trade_date={trade_date_str}" if trade_date_str else f"start={start_date_str}, end={end_date_str}"
        print(f"完成 {date_range_info} 行业资金流向数据（同花顺），result: {result}")
        return result

    # ============== 大盘资金流向数据 - 东方财富 ==============
    async def save_history_fund_flow_market_dc_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【V2.1 - 类型优化版】保存历史大盘资金流向数据 - 东方财富
        1. [优化] 增加数值列的显式转换。
        """
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        offset = 0
        limit = 5000
        all_dfs = []
        while True:
            if offset >= 100000:
                logger.warning(f"大盘资金流向数据 - 东方财富 offset已达10万，停止拉取。")
                break
            try:
                df = self.ts_pro.moneyflow_mkt_dc(**{
                    "trade_date": trade_date_str, "start_date": start_date_str, "end_date": end_date_str, "limit": limit, "offset": offset
                }, fields=[
                    "trade_date", "close_sh", "pct_change_sh", "close_sz", "pct_change_sz", "net_buy_amount", "net_buy_amount_rate",
                    "buy_elg_amount", "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate",
                    "buy_sm_amount", "buy_sm_amount_rate"
                ])
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"Tushare API调用失败 (moneyflow_mkt_dc): {e}")
                break
            if df.empty:
                break
            all_dfs.append(df)
            if len(df) < limit:
                break
            offset += limit
        if not all_dfs:
            logger.info("未获取到任何大盘资金流向数据。")
            return {}
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=['trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 批量数值转换
        numeric_cols = [
            "close_sh", "pct_change_sh", "close_sz", "pct_change_sz", "net_buy_amount", "net_buy_amount_rate",
            "buy_elg_amount", "buy_elg_amount_rate", "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate",
            "buy_sm_amount", "buy_sm_amount_rate"
        ]
        cols_to_convert = [c for c in numeric_cols if c in combined_df.columns]
        if cols_to_convert:
            combined_df[cols_to_convert] = combined_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        df_processed = combined_df.drop(columns=['trade_date']).where(pd.notnull(combined_df), None)
        data_dicts = df_processed.to_dict('records')
        if not data_dicts:
            logger.info("处理后无有效大盘资金流向数据可供保存。")
            return {}
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowMarketDc,
            data_list=data_dicts,
            unique_fields=['trade_time']
        )
        date_range_info = f"trade_date={trade_date_str}" if trade_date_str else f"start={start_date_str}, end={end_date_str}"
        print(f"完成 {date_range_info} 历史大盘资金流向数据（东方财富），result: {result}")
        return result

    # ============== 龙虎榜每日明细 ==============
    async def save_today_lhb_daily_data(self) -> Dict:
        """
        【V2.0 - 向量化优化版】保存今天的龙虎榜每日明细
        核心优化: 消除N+1查询，改为批量获取股票信息并进行向量化处理。
        """
        today_str = datetime.today().strftime('%Y%m%d')
        # 直接调用重构后的历史数据方法，传入当天日期
        print(f"调用 save_hisroty_lhb_daily_data 保存今日 {today_str} 的龙虎榜数据。")
        return await self.save_hisroty_lhb_daily_data(trade_date=today_str)

    async def save_hisroty_lhb_daily_data(self, trade_date: str) -> Dict:
        """
        【V2.2 - 类型优化版】保存历史龙虎榜每日数据
        1. [优化] 增加数值列的显式转换。
        """
        print(f"开始执行 save_hisroty_lhb_daily_data, trade_date={trade_date}")
        all_dfs = []
        limit = 5000
        offset = 0
        while True:
            try:
                df = self.ts_pro.top_list(**{
                    "trade_date": trade_date, "ts_code": "", "limit": limit, "offset": offset
                }, fields=[
                    "trade_date", "ts_code", "name", "close", "pct_change", "turnover_rate", "amount", "l_sell",
                    "l_buy", "l_amount", "net_amount", "net_rate", "amount_rate", "float_values", "reason"
                ])
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"Tushare API调用失败 (top_list, date: {trade_date}): {e}")
                break
            if df.empty:
                break
            all_dfs.append(df)
            if len(df) < limit:
                break
            offset += limit
        if not all_dfs:
            logger.info(f"交易日 {trade_date} 没有龙虎榜每日明细数据。")
            return {}
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
        unique_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info(f"交易日 {trade_date} 的龙虎榜数据关联股票信息后为空。")
            return {}
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 批量数值转换
        numeric_cols = [
            'close', 'pct_change', 'turnover_rate', 'amount', 'l_sell',
            'l_buy', 'l_amount', 'net_amount', 'net_rate', 'amount_rate'
        ]
        cols_to_convert = [c for c in numeric_cols if c in combined_df.columns]
        if cols_to_convert:
            combined_df[cols_to_convert] = combined_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        model_cols = [
            'stock', 'trade_time', 'name', 'close', 'pct_change', 'turnover_rate',
            'amount', 'l_sell', 'l_buy', 'l_amount', 'net_amount', 'net_rate',
            'amount_rate', 'float_values', 'reason'
        ]
        final_cols = [col for col in model_cols if col in combined_df.columns]
        final_df = combined_df[final_cols]
        final_df = final_df.where(pd.notnull(final_df), None)
        data_dicts = final_df.to_dict('records')
        result =  await self._save_all_to_db_native_upsert(
            model_class=TopList,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        logger.info(f"{trade_date} 的龙虎榜每日明细保存完成。")
        return result

    async def get_top_list_data(self, start_date: date, end_date: date, stock_codes: list[str] = None) -> pd.DataFrame:
        """
        根据日期范围和股票代码列表，获取龙虎榜每日明细数据。(V2.0 - 读取性能优化)
        优化：使用 sync_to_async(list) 替代异步推导式。
        """
        qs = TopList.objects.filter(trade_date__range=(start_date, end_date))
        if stock_codes:
            qs = qs.filter(stock__stock_code__in=stock_codes)
        qs = qs.select_related('stock')
        fields_to_get = [
            'trade_date', 'stock__stock_code',
            'net_amount', 'l_buy', 'l_sell'
        ]
        # 优化点：批量获取
        data_list = await sync_to_async(list)(qs.values(*fields_to_get))
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df.rename(columns={'stock__stock_code': 'ts_code'}, inplace=True)
        return df

    # ============== 龙虎榜机构明细 ==============
    async def save_today_lhb_inst_data(self) -> Dict:
        """
        【V2.0 - 向量化优化版】保存今天的龙虎榜机构明细
        核心优化: 直接复用已优化的 `save_hisroty_lhb_inst_data` 方法，消除N+1查询。
        """
        today_str = datetime.today().strftime('%Y%m%d')
        print(f"调用 save_hisroty_lhb_inst_data 保存今日 {today_str} 的龙虎榜机构数据。")
        # 复用已包含分页和向量化逻辑的健壮方法
        return await self.save_hisroty_lhb_inst_data(trade_date=today_str)

    async def save_hisroty_lhb_inst_data(self, trade_date: str) -> Dict:
        """
        【V2.2 - 类型优化版】保存历史龙虎榜机构明细
        1. [优化] 增加数值列的显式转换。
        """
        print(f"调试: 开始执行 save_hisroty_lhb_inst_data 任务, trade_date={trade_date}")
        try:
            all_dfs = []
            limit = 10000
            offset = 0
            max_records = 100000
            while True:
                df = self.ts_pro.top_inst(**{
                    "trade_date": trade_date,
                    "ts_code": "",
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "trade_date", "ts_code", "exalter", "buy", "buy_rate", "sell", "sell_rate", "net_buy", "side", "reason"
                ])
                if df.empty:
                    break
                all_dfs.append(df)
                if len(df) < limit:
                    break
                offset += limit
                if offset >= max_records:
                    logger.warning(f"拉取数据已达到 {max_records} 条上限，自动停止。")
                    break
            if not all_dfs:
                logger.info(f"交易日 {trade_date} 没有龙虎榜机构明细数据。")
                return {"status": "success", "message": "No data for this trade date.", "saved_count": 0}
            final_df = pd.concat(all_dfs, ignore_index=True)
            unique_ts_codes = final_df['ts_code'].unique().tolist()
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            final_df['stock'] = final_df['ts_code'].map(stock_map)
            final_df.dropna(subset=['stock'], inplace=True)
            final_df['trade_date'] = pd.to_datetime(final_df['trade_date']).dt.date
            # 批量数值转换
            numeric_cols = ['buy', 'buy_rate', 'sell', 'sell_rate', 'net_buy']
            cols_to_convert = [c for c in numeric_cols if c in final_df.columns]
            if cols_to_convert:
                final_df[cols_to_convert] = final_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
            model_cols = [
                'stock', 'trade_date', 'exalter', 'buy', 'buy_rate', 'sell',
                'sell_rate', 'net_buy', 'side', 'reason'
            ]
            final_cols = [col for col in model_cols if col in final_df.columns]
            df_to_save = final_df[final_cols]
            df_to_save = df_to_save.where(pd.notnull(df_to_save), None)
            data_dicts_to_save = df_to_save.to_dict('records')
            result = {}
            if data_dicts_to_save:
                result = await self._save_all_to_db_native_upsert(
                    model_class=TopInst,
                    data_list=data_dicts_to_save,
                    unique_fields=['stock', 'trade_date']
                )
            else:
                logger.info("经过筛选后，没有需要保存到数据库的数据。")
                result = {"status": "success", "message": "No new data to save.", "saved_count": 0}
            return result
        except Exception as e:
            logger.error(f"保存龙虎榜机构明细时发生严重错误: {e}", exc_info=True)
            raise

    async def get_top_inst_data(self, start_date: date, end_date: date, stock_codes: list[str] = None) -> pd.DataFrame:
        """
        根据日期范围和股票代码列表，获取龙虎榜机构明细数据。(V2.0 - 读取性能优化)
        优化：使用 sync_to_async(list) 替代异步推导式。
        """
        qs = TopInst.objects.filter(trade_date__range=(start_date, end_date))
        if stock_codes:
            qs = qs.filter(stock__stock_code__in=stock_codes)
        qs = qs.select_related('stock')
        fields_to_get = [
            'trade_date', 'stock__stock_code', 'net_buy'
        ]
        # 优化点：批量获取
        data_list = await sync_to_async(list)(qs.values(*fields_to_get))
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df.rename(columns={'stock__stock_code': 'ts_code'}, inplace=True)
        return df

    # ============== 游资每日明细 ==============
    async def save_hm_detail_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        保存游资每日明细数据 (V2.1 - 向量化计算优化)
        1. [优化] 对金额列进行全向量化的类型转换、填充和单位换算，替代循环处理。
        2. 保留了 Redis API 频率限制逻辑。
        """
        if not hasattr(self, 'cache_manager'):
            logger.error("DAO实例中未找到 cache_manager，无法执行API调用限制。")
            return
        try:
            redis_client = await self.cache_manager._ensure_client()
            if not redis_client: raise ConnectionError("无法获取 Redis 客户端。")
        except Exception as e:
            logger.error(f"初始化Redis客户端失败: {e}", exc_info=True)
            return
        api_limit_key = f"api_limit:hm_detail:{date.today().isoformat()}"
        API_DAILY_LIMIT = 2
        all_dfs = []
        offset = 0
        limit = 2000
        while True:
            try:
                current_count = await redis_client.incr(api_limit_key)
                if current_count == 1:
                    await redis_client.expire(api_limit_key, 3600 * 25)
                if current_count > API_DAILY_LIMIT:
                    logger.warning(f"接口 hm_detail 今日调用次数已达上限({API_DAILY_LIMIT}次)")
                    await redis_client.decr(api_limit_key)
                    break
                print(f"调试信息: 正在进行今日第 {current_count}/{API_DAILY_LIMIT} 次 hm_detail 接口调用...")
            except Exception as e:
                logger.error(f"Redis API限制检查出错: {e}")
                break
            try:
                df = self.ts_pro.hm_detail(**{
                    "trade_date": "", "start_date": "", "end_date": "", "limit": limit, "offset": offset
                }, fields=["trade_date", "ts_code", "ts_name", "buy_amount", "sell_amount", "net_amount", "hm_name", "hm_orgs"])
                await asyncio.sleep(0.55)
            except Exception as e:
                logger.error(f"Tushare API调用失败 (hm_detail): {e}")
                await asyncio.sleep(5)
                df = pd.DataFrame()
            if df.empty: break
            all_dfs.append(df)
            if len(df) < limit: break
            offset += limit
        if not all_dfs:
            logger.info("未获取到任何游资明细数据。")
            return
        print("DAO: 数据获取完毕，开始进行数据整合与处理...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=['trade_date', 'ts_code', 'hm_name'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)
        combined_df.dropna(subset=['ts_code', 'trade_date', 'hm_name'], inplace=True)
        # 更新游资名录
        hm_list_df = combined_df[['hm_name', 'hm_orgs']].copy()
        hm_list_df.drop_duplicates(subset=['hm_name'], keep='first', inplace=True)
        hm_list_df.rename(columns={'hm_name': 'name', 'hm_orgs': 'orgs'}, inplace=True)
        hm_list_df['orgs'] = hm_list_df['orgs'].fillna('')
        hm_list_data = hm_list_df.to_dict('records')
        if hm_list_data:
            await self._save_all_to_db_native_upsert(
                model_class=HmList,
                data_list=hm_list_data,
                unique_fields=['name']
            )
        # 处理明细数据
        all_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_ts_codes)
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info("游资数据关联股票基础信息后为空，任务结束。")
            return
        combined_df['trade_date'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 优化：全向量化数值处理 (转换 -> 填充 -> 运算)
        amount_cols = ['buy_amount', 'sell_amount', 'net_amount']
        # 确保列存在
        valid_cols = [c for c in amount_cols if c in combined_df.columns]
        if valid_cols:
            # 一次性处理所有金额列：转数字 -> 填0 -> 乘10000 (单位：万 -> 元)
            combined_df[valid_cols] = combined_df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0) * 10000
        final_df = combined_df.drop(columns=['ts_code'])
        # 将 NaN 替换为 None (针对非金额列)
        final_df = final_df.where(pd.notnull(final_df), None)
        data_list = final_df.to_dict('records')
        if not data_list: return
        await self._save_all_to_db_native_upsert(
            model_class=HmDetail,
            data_list=data_list,
            unique_fields=['trade_date', 'stock', 'hm_name']
        )
        print(f"所有游资每日明细数据处理完成，共处理/保存 {len(data_list)} 条记录。")
        return

    async def get_hm_detail_data(self, start_date: date, end_date: date, stock_codes: list[str] = None, hm_names: list[str] = None) -> pd.DataFrame:
        """
        查询游资每日明细数据 (V2.0 - 读取性能优化)
        优化：使用 sync_to_async(list) 替代异步推导式。
        """
        qs = HmDetail.objects.filter(trade_date__range=(start_date, end_date))
        if stock_codes:
            qs = qs.filter(stock__stock_code__in=stock_codes)
        if hm_names:
            qs = qs.filter(hm_name__in=hm_names)
        qs = qs.select_related('stock')
        fields_to_get = [
            'trade_date', 'stock__stock_code', 'ts_name',
            'buy_amount', 'sell_amount', 'net_amount',
            'hm_name', 'hm_orgs'
        ]
        # 优化点：批量获取
        data_list = await sync_to_async(list)(qs.values(*fields_to_get))
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df.rename(columns={'stock__stock_code': 'ts_code'}, inplace=True)
        return df











