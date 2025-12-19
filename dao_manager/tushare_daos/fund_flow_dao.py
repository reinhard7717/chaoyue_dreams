
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
from utils.model_helpers import get_advanced_fund_flow_metrics_model_by_code, get_fund_flow_model_by_code, get_fund_flow_ths_model_by_code, get_fund_flow_dc_model_by_code
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
        获取单个股票的历史日级资金流向数据 (Tushare moneyflow 接口)
        :param stock_code: 股票代码
        :param trade_date: 查询的截止日期
        :param limit: 返回的数据条数
        :return: 包含资金流向数据的DataFrame，以trade_time为索引
        """
        # print(f"DAO: 正在获取 {stock_code} 的常规日级资金流数据，截止日期 {trade_date}，数量 {limit}...")
        # 直接调用导入的辅助函数，而不是通过 self.
        model_class = get_fund_flow_model_by_code(stock_code)
        if not model_class:
            logger.warning(f"无法为股票 {stock_code} 确定常规资金流向数据模型。")
            return pd.DataFrame()
        try:
            # 异步查询数据库
            qs = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__lte=trade_date
            ).order_by('-trade_time')[:limit]
            # 使用异步推导式高效获取数据
            data_list = [item async for item in qs.values()]
            if not data_list:
                print(f"DAO: 未找到 {stock_code} 的常规日级资金流数据。")
                return pd.DataFrame()
            # 转换为DataFrame
            df = pd.DataFrame(data_list)
            # 将 trade_time 设置为UTC时区的DatetimeIndex，以满足上层服务要求
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            # 移除ORM生成的id和外键id列，保持数据纯净
            df.drop(columns=['id', 'stock_id'], inplace=True, errors='ignore')
            # print(f"DAO: 成功获取 {len(df)} 条常规日级资金流数据。")
            return df
        except Exception as e:
            logger.error(f"查询常规日级资金流数据时出错 (stock: {stock_code}): {e}", exc_info=True)
            return pd.DataFrame()
    async def save_history_fund_flow_daily_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        保存历史日级资金流向数据 (终极优化版 V3.1 - 字段补全)
        1. [重构] 废弃低效的“10万行追溯”逻辑。
        2. 采用客户端分块策略，将大日期范围切分为多个小块（如90天/块）进行处理。
        3. [优化] 对每个小块使用 limit/offset 进行高效分页，避免Tushare的查询限制和性能瓶颈。
        4. [保留] 保留了向量化处理和分表存储的核心优势。
        """
        # --- 1. 日期参数处理与验证 ---
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
        # --- 2. 客户端日期分块逻辑 ---
        date_chunks = []
        chunk_size_days = 10
        current_chunk_end = end_date
        while current_chunk_end >= start_date:
            current_chunk_start = max(start_date, current_chunk_end - timedelta(days=chunk_size_days - 1))
            date_chunks.append((current_chunk_start, current_chunk_end))
            current_chunk_end = current_chunk_start - timedelta(days=1)
        all_dfs_for_market = []
        # --- 3. [重构] 遍历分块并使用分页获取数据 ---
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
                        "net_mf_vol", "net_mf_amount", "trade_count"  # 新增 trade_count 字段，以获取交易笔数
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
        # --- 4. [保留] 向量化数据处理与入库 (此部分逻辑完全不变) ---
        # print("DAO: 所有分块数据获取完毕，开始进行数据整合与处理...")
        combined_df = pd.concat(all_dfs_for_market, ignore_index=True)
        combined_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        all_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_ts_codes)
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info("数据关联股票基础信息后为空，任务结束。")
            return
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        combined_df['target_model'] = combined_df['ts_code'].apply(get_fund_flow_model_by_code)
        total_rows = 0
        for model, group_df in combined_df.groupby('target_model', sort=False):
            if group_df.empty:
                continue
            final_df = group_df.drop(columns=['ts_code', 'trade_date', 'target_model'])
            data_list = final_df.to_dict('records')
            await self._save_all_to_db_native_upsert(
                model_class=model,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            total_rows += len(data_list)
        # print(f"所有历史日级资金流向数据处理完成，共保存 {total_rows} 条记录。")
        return
    # ============== 个股日级资金流向数据 - 同花顺 ==============
    async def get_fund_flow_ths_data(self, stock_code: str, trade_date: date, limit: int) -> pd.DataFrame:
        """
        获取单个股票的历史日级资金流向数据 (同花顺)
        :param stock_code: 股票代码
        :param trade_date: 查询的截止日期
        :param limit: 返回的数据条数
        :return: 包含资金流向数据的DataFrame，以trade_time为索引
        """
        # print(f"DAO: 正在获取 {stock_code} 的同花顺资金流数据，截止日期 {trade_date}，数量 {limit}...")
        # 直接调用导入的辅助函数，而不是通过 self.
        model_class = get_fund_flow_ths_model_by_code(stock_code)
        if not model_class:
            logger.warning(f"无法为股票 {stock_code} 确定同花顺资金流向数据模型。")
            return pd.DataFrame()
        try:
            # 异步查询数据库
            qs = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__lte=trade_date
            ).order_by('-trade_time')[:limit]
            # 使用异步推导式高效获取数据
            data_list = [item async for item in qs.values()]
            if not data_list:
                print(f"DAO: 未找到 {stock_code} 的同花顺资金流数据。")
                return pd.DataFrame()
            # 转换为DataFrame
            df = pd.DataFrame(data_list)
            # 将 trade_time 设置为UTC时区的DatetimeIndex
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            # 移除ORM生成的id和外键id列
            df.drop(columns=['id', 'stock_id'], inplace=True, errors='ignore')
            # print(f"DAO: 成功获取 {len(df)} 条同花顺资金流数据。")
            return df
        except Exception as e:
            logger.error(f"查询同花顺资金流数据时出错 (stock: {stock_code}): {e}", exc_info=True)
            return pd.DataFrame()
    async def save_history_fund_flow_daily_ths_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        保存历史日级资金流向数据 - 同花顺 (参照save_history_fund_flow_daily_data重构)
        1. 采用客户端分块策略，将大日期范围切分为多个小块进行处理。
        2. [优化] 对每个小块使用 limit/offset 进行高效分页，避免Tushare的查询限制和性能瓶颈。
        3. [重构] 使用向量化处理替代逐行循环，一次性获取所有股票信息，避免N+1查询。
        4. [重构] 实现动态分表存储，根据股票代码自动存入对应的板块数据表。
        """
        # --- 1. 日期参数处理与验证 ---
        if start_date and end_date and start_date > end_date:
            logger.error(f"日期范围无效：起始日期 {start_date} 不能晚于结束日期 {end_date}。任务终止。")
            return
        # 如果是范围查询
        if start_date and end_date:
            logger.info(f"接收到范围任务，将对 {start_date} 到 {end_date} 的同花顺资金流数据采用客户端分块策略处理。")
        # 如果是单日查询
        elif trade_date:
            start_date = end_date = trade_date
            logger.info(f"接收到单日任务: {trade_date}")
        # 默认情况，获取当天
        else:
            start_date = end_date = date.today()
            logger.info(f"未提供日期，默认获取今日数据: {start_date}")
        # --- 2. 客户端日期分块逻辑 ---
        date_chunks = []
        chunk_size_days = 10  # 每个分块的大小（天数）
        current_chunk_end = end_date
        while current_chunk_end >= start_date:
            current_chunk_start = max(start_date, current_chunk_end - timedelta(days=chunk_size_days - 1))
            date_chunks.append((current_chunk_start, current_chunk_end))
            current_chunk_end = current_chunk_start - timedelta(days=1)
        all_dfs_for_market = [] # 用于收集所有分块的数据
        # --- 3. [重构] 遍历分块并使用分页获取数据 ---
        for chunk_start, chunk_end in date_chunks:
            chunk_start_str = chunk_start.strftime('%Y%m%d')
            chunk_end_str = chunk_end.strftime('%Y%m%d')
            print(f"DAO: 开始处理同花顺资金流日期块: {chunk_start_str} 到 {chunk_end_str}")
            offset = 0
            limit = 5000 # Tushare建议的单次最大limit
            while True:
                try:
                    # API调用现在使用分块的起止日期，并调用 moneyflow_ths 接口
                    df = self.ts_pro.moneyflow_ths(**{
                        "ts_code": "", "trade_date": "", # 范围查询时，trade_date应为空
                        "start_date": chunk_start_str, "end_date": chunk_end_str,
                        "limit": limit, "offset": offset
                    }, fields=[
                        "trade_date", "ts_code", "pct_change", "net_amount", "net_d5_amount", "buy_lg_amount",
                        "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
                    ])
                    await asyncio.sleep(0.85) # 保持友好的API调用频率
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (moneyflow_ths, chunk: {chunk_start_str}-{chunk_end_str}): {e}")
                    await asyncio.sleep(5) # 出错时等待更长时间
                    df = pd.DataFrame()
                if df.empty:
                    break # 当前分块的当前分页无数据，结束此分块的分页
                all_dfs_for_market.append(df)
                if len(df) < limit:
                    break # 当前分块的数据已全部获取完毕
                offset += limit
        if not all_dfs_for_market:
            logger.info("在所有日期块中均未获取到任何同花顺资金流向数据。")
            return
        # --- 4. [重构] 向量化数据处理与入库 ---
        print("DAO: 所有同花顺分块数据获取完毕，开始进行数据整合与处理...")
        combined_df = pd.concat(all_dfs_for_market, ignore_index=True)
        combined_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 一次性批量获取所有涉及的股票信息
        all_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_ts_codes)
        # 使用向量化操作进行数据关联和清洗
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info("数据关联股票基础信息后为空，任务结束。")
            return
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 调用 get_fund_flow_ths_model_by_code 进行动态分表模型匹配
        combined_df['target_model'] = combined_df['ts_code'].apply(get_fund_flow_ths_model_by_code)
        total_rows = 0
        # 按目标模型（即目标数据表）进行分组并批量保存
        for model, group_df in combined_df.groupby('target_model', sort=False):
            if group_df.empty:
                continue
            # 准备最终要存入数据库的数据，丢弃辅助列
            final_df = group_df.drop(columns=['ts_code', 'trade_date', 'target_model'])
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
        获取单个股票的历史日级资金流向数据 (东方财富)
        :param stock_code: 股票代码
        :param trade_date: 查询的截止日期
        :param limit: 返回的数据条数
        :return: 包含资金流向数据的DataFrame，以trade_time为索引
        """
        # print(f"DAO: 正在获取 {stock_code} 的东方财富资金流数据，截止日期 {trade_date}，数量 {limit}...")
        # 直接调用导入的辅助函数，而不是通过 self.
        model_class = get_fund_flow_dc_model_by_code(stock_code)
        if not model_class:
            logger.warning(f"无法为股票 {stock_code} 确定东方财富资金流向数据模型。")
            return pd.DataFrame()
        try:
            # 异步查询数据库
            qs = model_class.objects.filter(
                stock__stock_code=stock_code,
                trade_time__lte=trade_date
            ).order_by('-trade_time')[:limit]
            # 使用异步推导式高效获取数据
            data_list = [item async for item in qs.values()]
            if not data_list:
                print(f"DAO: 未找到 {stock_code} 的东方财富资金流数据。")
                return pd.DataFrame()
            # 转换为DataFrame
            df = pd.DataFrame(data_list)
            # 将 trade_time 设置为UTC时区的DatetimeIndex
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            # 移除ORM生成的id和外键id列
            df.drop(columns=['id', 'stock_id'], inplace=True, errors='ignore')
            # print(f"DAO: 成功获取 {len(df)} 条东方财富资金流数据。")
            return df
        except Exception as e:
            logger.error(f"查询东方财富资金流数据时出错 (stock: {stock_code}): {e}", exc_info=True)
            return pd.DataFrame()
    async def save_history_fund_flow_daily_dc_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        保存历史日级资金流向数据 - 东方财富 (参照save_history_fund_flow_daily_data重构)
        1. 采用客户端分块策略，将大日期范围切分为多个小块进行处理。
        2. [优化] 对每个小块使用 limit/offset 进行高效分页，避免Tushare的查询限制和性能瓶颈。
        3. [重构] 使用向量化处理替代逐行循环，一次性获取所有股票信息，避免N+1查询。
        4. [重构] 实现动态分表存储，根据股票代码自动存入对应的板块数据表。
        """
        # --- 1. 日期参数处理与验证 ---
        if start_date and end_date and start_date > end_date:
            logger.error(f"日期范围无效：起始日期 {start_date} 不能晚于结束日期 {end_date}。任务终止。")
            return
        # 如果是范围查询
        if start_date and end_date:
            logger.info(f"接收到范围任务，将对 {start_date} 到 {end_date} 的东方财富资金流数据采用客户端分块策略处理。")
        # 如果是单日查询
        elif trade_date:
            start_date = end_date = trade_date
            logger.info(f"接收到单日任务: {trade_date}")
        # 默认情况，获取当天
        else:
            start_date = end_date = date.today()
            logger.info(f"未提供日期，默认获取今日数据: {start_date}")
        # --- 2. 客户端日期分块逻辑 ---
        date_chunks = []
        chunk_size_days = 10  # 每个分块的大小（天数）
        current_chunk_end = end_date
        while current_chunk_end >= start_date:
            current_chunk_start = max(start_date, current_chunk_end - timedelta(days=chunk_size_days - 1))
            date_chunks.append((current_chunk_start, current_chunk_end))
            current_chunk_end = current_chunk_start - timedelta(days=1)
        all_dfs_for_market = [] # 用于收集所有分块的数据
        # --- 3. [重构] 遍历分块并使用分页获取数据 ---
        for chunk_start, chunk_end in date_chunks:
            chunk_start_str = chunk_start.strftime('%Y%m%d')
            chunk_end_str = chunk_end.strftime('%Y%m%d')
            print(f"DAO: 开始处理东方财富资金流日期块: {chunk_start_str} 到 {chunk_end_str}")
            offset = 0
            limit = 5000 # Tushare建议的单次最大limit
            while True:
                try:
                    # API调用现在使用分块的起止日期，并调用 moneyflow_dc 接口
                    df = self.ts_pro.moneyflow_dc(**{
                        "ts_code": "", "trade_date": "", # 范围查询时，trade_date应为空
                        "start_date": chunk_start_str, "end_date": chunk_end_str,
                        "limit": limit, "offset": offset
                    }, fields=[
                        "trade_date", "ts_code", "name", "pct_change", "close", "net_amount", "net_amount_rate", "buy_elg_amount", "buy_elg_amount_rate",
                        "buy_lg_amount", "buy_lg_amount_rate", "buy_md_amount", "buy_md_amount_rate", "buy_sm_amount", "buy_sm_amount_rate"
                    ])
                    await asyncio.sleep(0.85) # 保持友好的API调用频率
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (moneyflow_dc, chunk: {chunk_start_str}-{chunk_end_str}): {e}")
                    await asyncio.sleep(5) # 出错时等待更长时间
                    df = pd.DataFrame()
                if df.empty:
                    break # 当前分块的当前分页无数据，结束此分块的分页
                all_dfs_for_market.append(df)
                if len(df) < limit:
                    break # 当前分块的数据已全部获取完毕
                offset += limit
        if not all_dfs_for_market:
            logger.info("在所有日期块中均未获取到任何东方财富资金流向数据。")
            return
        # --- 4. [重构] 向量化数据处理与入库 ---
        print("DAO: 所有东方财富分块数据获取完毕，开始进行数据整合与处理...")
        combined_df = pd.concat(all_dfs_for_market, ignore_index=True)
        combined_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # 一次性批量获取所有涉及的股票信息
        all_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_ts_codes)
        # 使用向量化操作进行数据关联和清洗
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info("数据关联股票基础信息后为空，任务结束。")
            return
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 调用 get_fund_flow_dc_model_by_code 进行动态分表模型匹配
        combined_df['target_model'] = combined_df['ts_code'].apply(get_fund_flow_dc_model_by_code)
        total_rows = 0
        # 按目标模型（即目标数据表）进行分组并批量保存
        for model, group_df in combined_df.groupby('target_model', sort=False):
            if group_df.empty:
                continue
            # 准备最终要存入数据库的数据，丢弃辅助列。注意：'name'字段在DC模型中是需要的，所以不丢弃。
            final_df = group_df.drop(columns=['ts_code', 'trade_date', 'target_model'])
            data_list = final_df.to_dict('records')
            await self._save_all_to_db_native_upsert(
                model_class=model,
                data_list=data_list,
                unique_fields=['stock', 'trade_time']
            )
            total_rows += len(data_list)
        print(f"所有历史日级资金流向数据(东方财富)处理完成，共保存 {total_rows} 条记录。")
        return
    # ============== 资金流向高级指标 ==============
    async def get_advanced_fund_flow_metrics_data(self, stock_code: str, trade_date: datetime.date, limit: int) -> pd.DataFrame:
        """
        从 AdvancedFundFlowMetrics 模型获取预计算的高级资金指标。
        """
        # 动态获取对应市场的模型
        # 直接调用导入的辅助函数，而不是通过 self.
        model = get_advanced_fund_flow_metrics_model_by_code(stock_code)
        if not model:
            return pd.DataFrame()
        # 构建查询
        end_date = trade_date or datetime.date.today()
        qs = model.objects.filter(
            stock__stock_code=stock_code,
            trade_time__lte=end_date
        ).order_by('-trade_time')[:limit]
        # 异步执行查询并返回DataFrame
        df = await sync_to_async(lambda: pd.DataFrame.from_records(qs.values()))()
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.set_index('trade_time')
        return df
    # ============== 板块资金流向数据 - 同花顺 ==============
    async def save_history_fund_flow_cnt_ths_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【V2.0 - 向量化优化版】保存历史板块资金流向数据 - 同花顺
        核心优化:
        1. 【消除N+1查询】将循环内单次数据库查询，改为批处理模式。一次性获取所有ts_code，一次性查询数据库。
        2. 【向量化数据处理】使用Pandas的向量化操作(map, to_datetime)替代低效的逐行循环，大幅提升数据预处理性能。
        3. 【修正逻辑错误】修正了unique_fields参数，使其与模型定义一致。
        4. 【代码健壮性】增加了对空数据和无效关联的过滤，使数据管道更稳定。
        """
        # 1. 准备API请求参数 
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        # 2. 分页拉取并处理数据
        offset = 0
        limit = 6000
        all_data_to_save = [] # 用于累积所有批次处理好的数据
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
            # 3. 向量化数据清洗与转换
            # 3.1 清洗空值和无效行
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                time.sleep(0.2)
                if original_count < limit:
                    break
                offset += limit
                continue
            # 3.2 向量化处理外键关联 (核心性能优化)
            unique_codes = df['ts_code'].unique().tolist()
            # 假设 industry_dao 中有一个批量获取的方法，这是最佳实践
            # 如果没有，您需要添加它。实现示例: return {obj.ts_code: obj for obj in ThsIndex.objects.filter(ts_code__in=codes)}
            ths_index_map = await self.industry_dao.get_ths_indices_by_codes(unique_codes)
            df['ths_index'] = df['ts_code'].map(ths_index_map)
            # 过滤掉数据库中不存在对应ThsIndex的记录
            df.dropna(subset=['ths_index'], inplace=True)
            if df.empty:
                logger.warning(f"当前批次所有ts_code均未在ThsIndex表中找到对应记录。")
                time.sleep(0.2)
                if original_count < limit:
                    break
                offset += limit
                continue
            # 3.3 向量化数据类型转换
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 3.4 将NaN替换为None，以适应数据库存储
            # 选择模型需要的列，并确保顺序和命名正确
            final_columns = [
                'ths_index', 'trade_time', 'lead_stock', 'close_price', 'pct_change',
                'industry_index', 'company_num', 'pct_change_stock', 'net_buy_amount',
                'net_sell_amount', 'net_amount'
            ]
            df_final = df[final_columns]
            # 将Pandas的NaN转换为Python的None
            df_processed = df_final.where(pd.notnull(df_final), None)
            # 4. 累积处理好的数据
            all_data_to_save.extend(df_processed.to_dict('records'))
            # 5. 分页逻辑 
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit
        # 6. 一次性批量保存到数据库
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何有效数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        result = await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntTHS,
            data_list=all_data_to_save,
            unique_fields=['ths_index', 'trade_time']
        )
        # 使用 f-string 格式化输出
        date_range_info = f"trade_date={trade_date_str}" if trade_date_str else f"start={start_date_str}, end={end_date_str}"
        print(f"完成 {date_range_info} 板块资金流向数据（同花顺），result: {result}")
        return result
    # ============== 板块资金流向数据 - 东方财富 ==============
    async def save_history_fund_flow_cnt_dc_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【V2.0 - 向量化优化版】保存历史板块资金流向数据 - 东方财富
        核心优化:
        1. 【消除N+1查询】将循环内单次数据库/缓存查询，改为批处理模式。一次性获取所有ts_code，一次性查询数据库。
        2. 【向量化数据处理】使用Pandas的向量化操作(map, to_datetime)替代低效的逐行循环，大幅提升数据预处理性能。
        3. 【修正逻辑错误】修正了unique_fields参数，使其与模型定义一致。
        4. 【代码健壮性】增加了对空数据和无效关联的过滤，使数据管道更稳定。
        """
        # 1. 准备API请求参数 
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        # 2. 分页拉取并处理数据
        offset = 0
        limit = 6000
        all_data_to_save = [] # 用于累积所有批次处理好的数据
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
            # 3. 向量化数据清洗与转换
            # 3.1 清洗空值和无效行
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                time.sleep(0.2)
                if original_count < limit:
                    break
                offset += limit
                continue
            # 3.2 向量化处理外键关联 (核心性能优化)
            unique_codes = df['ts_code'].unique().tolist()
            dc_index_map = await self.industry_dao.get_dc_indices_by_codes(unique_codes)
            df['dc_index'] = df['ts_code'].map(dc_index_map)
            # 过滤掉数据库中不存在对应DcIndex的记录
            df.dropna(subset=['dc_index'], inplace=True)
            if df.empty:
                logger.warning(f"当前批次所有ts_code均未在DcIndex表中找到对应记录。")
                time.sleep(0.2)
                if original_count < limit:
                    break
                offset += limit
                continue
            # 3.3 向量化数据类型转换
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 3.4 准备用于保存的数据
            # 选择模型需要的列，并确保命名与模型字段一致
            # 注意：模型中没有rank字段，所以我们不选择它
            final_columns = [
                'dc_index', 'trade_time', 'content_type', 'name', 'pct_change', 'close', 'net_amount',
                'net_amount_rate', 'buy_elg_amount', 'buy_elg_amount_rate', 'buy_lg_amount',
                'buy_lg_amount_rate', 'buy_md_amount', 'buy_md_amount_rate', 'buy_sm_amount',
                'buy_sm_amount_rate', 'buy_sm_amount_stock'
            ]
            df_final = df[final_columns]
            # 将Pandas的NaN转换为Python的None
            df_processed = df_final.where(pd.notnull(df_final), None)
            # 4. 累积处理好的数据
            all_data_to_save.extend(df_processed.to_dict('records'))
            # 5. 分页逻辑 
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit
        # 6. 一次性批量保存到数据库
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何东方财富板块资金流数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        # unique_fields 从 ['stock', 'trade_time'] 修正为 ['dc_index', 'trade_time']
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
        【V2.0 - 向量化优化版】保存历史行业资金流向数据 - 同花顺
        核心优化:
        1. 【消除N+1查询】将循环内单次数据库查询，改为批处理模式。一次性获取所有ts_code，一次性查询数据库。
        2. 【向量化数据处理】使用Pandas的向量化操作替代低效的逐行循环和数据格式化函数调用。
        3. 【修正逻辑错误】修正了unique_fields参数，使其与模型定义一致。
        4. 【代码健壮性】增加了对空数据和无效关联的过滤，使数据管道更稳定。
        """
        # 1. 准备API请求参数 
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        # 2. 分页拉取并处理数据
        offset = 0
        limit = 5000
        all_data_to_save = [] # 用于累积所有批次处理好的数据
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
            # 3. 向量化数据清洗与转换
            # 3.1 清洗空值和无效行
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                time.sleep(0.2)
                if original_count < limit:
                    break
                offset += limit
                continue
            # 3.2 向量化处理外键关联 (核心性能优化)
            unique_codes = df['ts_code'].unique().tolist()
            # 假设 industry_dao 中有批量获取的方法，这是最佳实践
            ths_index_map = await self.industry_dao.get_ths_indices_by_codes(unique_codes)
            df['ths_index'] = df['ts_code'].map(ths_index_map)
            # 过滤掉数据库中不存在对应ThsIndex的记录
            df.dropna(subset=['ths_index'], inplace=True)
            if df.empty:
                logger.warning(f"当前批次所有ts_code均未在ThsIndex表中找到对应记录。")
                time.sleep(0.2)
                if original_count < limit:
                    break
                offset += limit
                continue
            # 3.3 向量化数据类型转换
            df['trade_time'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
            # 3.4 准备用于保存的数据
            # 选择模型需要的列，确保命名与模型字段一致
            final_columns = [
                'ths_index', 'trade_time', 'industry', 'lead_stock', 'close', 'pct_change',
                'company_num', 'pct_change_stock', 'close_price', 'net_buy_amount',
                'net_sell_amount', 'net_amount'
            ]
            df_final = df[final_columns]
            # 将Pandas的NaN转换为Python的None
            df_processed = df_final.where(pd.notnull(df_final), None)
            # 4. 累积处理好的数据
            all_data_to_save.extend(df_processed.to_dict('records'))
            # 5. 分页逻辑 
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit
        # 6. 一次性批量保存到数据库
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何同花顺行业资金流数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}
        # unique_fields 从 ['stock', 'trade_time'] 修正为 ['ths_index', 'trade_time']
        result = await self._save_all_to_db_native_upsert(
            model_class=FundFlowIndustryTHS,
            data_list=all_data_to_save,
            unique_fields=['ths_index', 'trade_time']
        )
        date_range_info = f"trade_date={trade_date_str}" if trade_date_str else f"start={start_date_str}, end={end_date_str}"
        # 【修正打印信息】将“板块”修正为“行业”
        print(f"完成 {date_range_info} 行业资金流向数据（同花顺），result: {result}")
        return result
    # ============== 大盘资金流向数据 - 东方财富 ==============
    async def save_history_fund_flow_market_dc_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【V2.0 - 逻辑修正与向量化优化版】保存历史大盘资金流向数据 - 东方财富
        核心优化:
        1. 移除错误的 stock 关联。大盘资金流数据与个股无关，其唯一性由交易日期保证。
        2. 【性能优化】采用向量化处理替代逐行循环，一次性完成数据清洗和类型转换。
        3. 【代码健壮性】保留分页逻辑以支持大数据量拉取，并增强日志信息。
        """
        # 1. 准备API请求参数 
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        # 2. 分页拉取数据
        offset = 0
        limit = 5000 # Tushare建议的单次最大limit
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
                await asyncio.sleep(0.2) # 保持友好的API调用频率
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
        # 3. 向量化数据处理
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=['trade_date'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        # trade_time 是 trade_date 转换而来，不再需要 stock 关联
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 将Pandas的NaN转换为Python的None以适应数据库
        df_processed = combined_df.drop(columns=['trade_date']).where(pd.notnull(combined_df), None)
        data_dicts = df_processed.to_dict('records')
        if not data_dicts:
            logger.info("处理后无有效大盘资金流向数据可供保存。")
            return {}
        # 4. 批量保存
        # unique_fields 应该是 'trade_time'，而不是 'stock', 'trade_time'
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
        【V2.1 - 全面向量化版】保存历史龙虎榜每日数据
        核心优化:
        1. 【消除循环】用Pandas的向量化操作 (map, 列赋值) 彻底取代了原有的 `itertuples()` 循环，显著提升数据处理效率。
        2. 【批量关联】将股票代码(ts_code)与股票对象(stock)的关联，通过一次 `map` 操作完成，避免了逐行查找。
        3. 【代码简化】数据处理流程更清晰、更符合Pandas的最佳实践。
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
        # 使用向量化操作替代原有的 itertuples 循环
        # 1. 一次性批量获取所有相关的股票对象
        unique_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
        # 2. 向量化映射：将股票对象映射到DataFrame的新列'stock'中
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        # 3. 向量化过滤：移除没有成功关联到股票对象的行
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info(f"交易日 {trade_date} 的龙虎榜数据关联股票信息后为空。")
            return {}
        # 4. 向量化转换：转换日期格式并重命名为模型字段'trade_time'
        combined_df['trade_time'] = pd.to_datetime(combined_df['trade_date']).dt.date
        # 5. 准备数据：选择模型需要的列
        model_cols = [
            'stock', 'trade_time', 'name', 'close', 'pct_change', 'turnover_rate',
            'amount', 'l_sell', 'l_buy', 'l_amount', 'net_amount', 'net_rate',
            'amount_rate', 'float_values', 'reason'
        ]
        # 筛选出DataFrame中实际存在的列，以增强代码健壮性
        final_cols = [col for col in model_cols if col in combined_df.columns]
        final_df = combined_df[final_cols]
        # 6. 格式化：将Pandas的NaN/NaT转换成数据库能接受的None
        final_df = final_df.where(pd.notnull(final_df), None)
        # 7. 转换为字典列表，准备入库
        data_dicts = final_df.to_dict('records')
        # 原有的 itertuples 循环已被完全移除
        result =  await self._save_all_to_db_native_upsert(
            model_class=TopList,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        logger.info(f"{trade_date} 的龙虎榜每日明细保存完成。")
        return result
    async def get_top_list_data(self, start_date: date, end_date: date, stock_codes: list[str] = None) -> pd.DataFrame:
        """
        根据日期范围和股票代码列表，获取龙虎榜每日明细数据。
        """
        qs = TopList.objects.filter(trade_date__range=(start_date, end_date))
        if stock_codes:
            qs = qs.filter(stock__stock_code__in=stock_codes)
        # 使用select_related优化查询
        qs = qs.select_related('stock')
        # 定义需要返回的字段
        fields_to_get = [
            'trade_date',
            'stock__stock_code',
            'net_amount', # 龙虎榜净买入额
            'l_buy',      # 龙虎榜总买入
            'l_sell'      # 龙虎榜总卖出
        ]
        data_list = [item async for item in qs.values(*fields_to_get)]
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
        【V2.1 - 全面向量化版】保存历史龙虎榜机构明细
        核心优化:
        1. 【消除循环】用Pandas的向量化操作 (map, 列赋值) 彻底取代了原有的 `itertuples()` 循环，显著提升数据处理效率。
        2. 【批量关联】将股票代码(ts_code)与股票对象(stock)的关联，通过一次 `map` 操作完成，避免了逐行查找。
        3. 【代码简化】数据处理流程更清晰、更符合Pandas的最佳实践，同时保持了原有的分页和异常处理逻辑。
        """
        print(f"调试: 开始执行 save_hisroty_lhb_inst_data 任务, trade_date={trade_date}")
        try:
            all_dfs = []
            limit = 10000
            offset = 0
            max_records = 100000
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
                if len(df) < limit:
                    print(f"调试: 已获取最后一页数据({len(df)}条)，分页拉取结束。")
                    break
                offset += limit
                if offset >= max_records:
                    logger.warning(f"拉取数据已达到 {max_records} 条上限，自动停止。")
                    print(f"调试: 拉取数据已达到 {max_records} 条上限，自动停止。")
                    break
            if not all_dfs:
                logger.info(f"交易日 {trade_date} 没有龙虎榜机构明细数据。")
                return {"status": "success", "message": "No data for this trade date.", "saved_count": 0}
            final_df = pd.concat(all_dfs, ignore_index=True)
            print(f"调试: 分页拉取完成，共获取 {len(final_df)} 条数据。")
            # 使用向量化操作替代原有的 itertuples 循环
            # 1. 一次性批量获取所有相关的股票对象
            unique_ts_codes = final_df['ts_code'].unique().tolist()
            print(f"调试: 数据涉及 {len(unique_ts_codes)} 个独立股票代码。")
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            print(f"调试: 批量从数据库获取了 {len(stock_map)} 个股票对象。")
            # 2. 向量化映射：将股票对象映射到DataFrame的新列'stock'中
            final_df['stock'] = final_df['ts_code'].map(stock_map)
            # 3. 向量化过滤：移除没有成功关联到股票对象的行
            final_df.dropna(subset=['stock'], inplace=True,
                            after_stat=lambda dropped: logger.warning(f"在数据库中未找到 {dropped['ts_code'].nunique()} 个股票代码的基础信息，已跳过 {len(dropped)} 条龙虎榜数据。") if not dropped.empty else None)
            # 4. 向量化转换：转换日期格式以匹配模型字段
            final_df['trade_date'] = pd.to_datetime(final_df['trade_date']).dt.date
            # 5. 准备数据：选择模型需要的列
            model_cols = [
                'stock', 'trade_date', 'exalter', 'buy', 'buy_rate', 'sell',
                'sell_rate', 'net_buy', 'side', 'reason'
            ]
            final_cols = [col for col in model_cols if col in final_df.columns]
            df_to_save = final_df[final_cols]
            # 6. 格式化：将Pandas的NaN/NaT转换成数据库能接受的None
            df_to_save = df_to_save.where(pd.notnull(df_to_save), None)
            # 7. 转换为字典列表，准备入库
            data_dicts_to_save = df_to_save.to_dict('records')
            # 原有的 itertuples 循环已被完全移除
            result = {}
            if data_dicts_to_save:
                print(f"调试: 准备批量保存 {len(data_dicts_to_save)} 条龙虎榜数据到数据库...")
                result = await self._save_all_to_db_native_upsert(
                    model_class=TopInst,
                    data_list=data_dicts_to_save,
                    unique_fields=['stock', 'trade_date']
                )
                print(f"调试: 数据库操作完成，结果: {result}")
            else:
                logger.info("经过筛选后，没有需要保存到数据库的数据。")
                print("调试: 经过筛选后，没有需要保存的数据。")
                result = {"status": "success", "message": "No new data to save.", "saved_count": 0}
            return result
        except Exception as e:
            logger.error(f"保存龙虎榜机构明细时发生严重错误: {e}", exc_info=True)
            print(f"调试: 发生异常: {e}")
            raise
    async def get_top_inst_data(self, start_date: date, end_date: date, stock_codes: list[str] = None) -> pd.DataFrame:
        """
        根据日期范围和股票代码列表，获取龙虎榜机构明细数据。
        """
        qs = TopInst.objects.filter(trade_date__range=(start_date, end_date))
        if stock_codes:
            qs = qs.filter(stock__stock_code__in=stock_codes)
        qs = qs.select_related('stock')
        fields_to_get = [
            'trade_date',
            'stock__stock_code',
            'net_buy' # 机构净买入额
        ]
        data_list = [item async for item in qs.values(*fields_to_get)]
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df.rename(columns={'stock__stock_code': 'ts_code'}, inplace=True)
        return df
    # ============== 游资每日明细 ==============
    async def save_hm_detail_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        【V2 - API调用限制版】保存游资每日明细数据。
        - 策略:
        1. 引入基于Redis的每日API调用计数器，确保 hm_detail 接口每天最多被调用2次。
        2. 在每次分页循环（即每次API调用）前，检查并增加计数器。
        3. 如果超出限制，则立即停止获取数据，并处理已获取的数据。
        """
        # --- 1. 初始化API调用限制相关的变量 ---
        # 确保可以访问到在 BaseDAO.__init__ 中创建的 cache_manager
        if not hasattr(self, 'cache_manager'):
            logger.error("DAO实例中未找到 cache_manager，无法执行API调用限制。")
            return
        try:
            redis_client = await self.cache_manager._ensure_client()
            if not redis_client:
                raise ConnectionError("无法从 CacheManager 获取 Redis 客户端。")
        except Exception as e:
            logger.error(f"初始化Redis客户端以进行API限制检查时失败: {e}", exc_info=True)
            return
        # 定义一个每日更新的、用于API计数器的Redis键
        api_limit_key = f"api_limit:hm_detail:{date.today().isoformat()}"
        API_DAILY_LIMIT = 2 # 每日API调用上限
        # --- 2. 调整分页获取逻辑以集成限制检查 ---
        all_dfs = []
        offset = 0
        limit = 2000
        while True:
            # 在每次API调用前，检查并更新调用次数
            try:
                # 对Redis键执行原子+1操作，返回操作后的值
                current_count = await redis_client.incr(api_limit_key)
                # 如果是当天的第一次调用 (值为1)，则为该键设置过期时间，确保第二天计数器自动重置
                if current_count == 1:
                    # 设置25小时过期，比一天稍长，可避免午夜时区的微小误差
                    await redis_client.expire(api_limit_key, 3600 * 25)
                # 检查是否已超出每日限制
                if current_count > API_DAILY_LIMIT:
                    logger.warning(f"接口 hm_detail 今日调用次数已达上限({API_DAILY_LIMIT}次)，Key: {api_limit_key}")
                    # 将刚刚多加的1次减回去，保持计数准确
                    await redis_client.decr(api_limit_key)
                    break # 退出循环，不再调用API
                print(f"调试信息: 正在进行今日第 {current_count}/{API_DAILY_LIMIT} 次 hm_detail 接口调用...")
            except Exception as e:
                logger.error(f"执行Redis API调用限制检查时出错: {e}", exc_info=True)
                # 如果Redis检查失败，为安全起见，直接中断任务
                break
            # --- 原有的API调用逻辑 ---
            try:
                df = self.ts_pro.hm_detail(**{
                    "trade_date": "",
                    "start_date": "",
                    "end_date": "",
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "trade_date", "ts_code", "ts_name", "buy_amount", "sell_amount",
                    "net_amount", "hm_name", "hm_orgs"
                ])
                await asyncio.sleep(0.55)
            except Exception as e:
                logger.error(f"Tushare API调用失败 (hm_detail): {e}")
                await asyncio.sleep(5)
                df = pd.DataFrame()
            if df.empty:
                break
            all_dfs.append(df)
            if len(df) < limit:
                break
            offset += limit
        # --- 后续的数据处理逻辑保持不变 ---
        if not all_dfs:
            logger.info("未获取到任何游资明细数据。")
            return
        print("DAO: 数据获取完毕，开始进行数据整合与处理...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=['trade_date', 'ts_code', 'hm_name'], keep='first', inplace=True)
        combined_df.replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)
        combined_df.dropna(subset=['ts_code', 'trade_date', 'hm_name'], inplace=True)
        # 更新游资名录 (HmList)
        hm_list_df = combined_df[['hm_name', 'hm_orgs']].copy()
        hm_list_df.drop_duplicates(subset=['hm_name'], keep='first', inplace=True)
        hm_list_df.rename(columns={'hm_name': 'name', 'hm_orgs': 'orgs'}, inplace=True)
        hm_list_df['orgs'] = hm_list_df['orgs'].fillna('')
        hm_list_data = hm_list_df.to_dict('records')
        if hm_list_data:
            print(f"DAO: 准备更新游资名录，共 {len(hm_list_data)} 条...")
            await self._save_all_to_db_native_upsert(
                model_class=HmList,
                data_list=hm_list_data,
                unique_fields=['name']
            )
            print(f"DAO: 游资名录更新完成。")
        # 处理并保存游资每日明细 (HmDetail)
        all_ts_codes = combined_df['ts_code'].unique().tolist()
        stock_map = await self.stock_basic_dao.get_stocks_by_codes(all_ts_codes)
        combined_df['stock'] = combined_df['ts_code'].map(stock_map)
        combined_df.dropna(subset=['stock'], inplace=True)
        if combined_df.empty:
            logger.info("游资数据关联股票基础信息后为空，任务结束。")
            return
        combined_df['trade_date'] = pd.to_datetime(combined_df['trade_date']).dt.date
        amount_cols = ['buy_amount', 'sell_amount', 'net_amount']
        for col in amount_cols:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0) * 10000
        final_df = combined_df.drop(columns=['ts_code'])
        data_list = final_df.to_dict('records')
        if not data_list:
            logger.info("最终处理后没有可供保存的游资数据。")
            return
        await self._save_all_to_db_native_upsert(
            model_class=HmDetail,
            data_list=data_list,
            unique_fields=['trade_date', 'stock', 'hm_name']
        )
        print(f"所有游资每日明细数据处理完成，共处理/保存 {len(data_list)} 条记录。")
        return
    async def get_hm_detail_data(self, start_date: date, end_date: date, stock_codes: list[str] = None, hm_names: list[str] = None) -> pd.DataFrame:
        """
        查询游资每日明细数据
        :param start_date: 起始日期
        :param end_date: 结束日期
        :param stock_codes: 股票代码列表 (可选)
        :param hm_names: 游资名称列表 (可选)
        :return: 包含查询结果的DataFrame
        """
        # print(f"DAO: 开始查询游资数据，日期范围: {start_date} to {end_date}, 股票: {stock_codes}, 游资: {hm_names}")
        # 构建基础查询
        qs = HmDetail.objects.filter(trade_date__range=(start_date, end_date))
        # 应用可选的股票代码过滤
        if stock_codes:
            qs = qs.filter(stock__stock_code__in=stock_codes)
        # 应用可选的游资名称过滤
        if hm_names:
            qs = qs.filter(hm_name__in=hm_names)
        # 使用select_related优化查询，避免N+1问题
        qs = qs.select_related('stock')
        # 定义需要返回的字段
        fields_to_get = [
            'trade_date',
            'stock__stock_code', # 通过外键获取股票代码
            'ts_name',
            'buy_amount',
            'sell_amount',
            'net_amount',
            'hm_name',
            'hm_orgs'
        ]
        # 异步执行查询并转换为列表
        data_list = [item async for item in qs.values(*fields_to_get)]
        if not data_list:
            print("DAO: 未查询到符合条件的游资数据。")
            return pd.DataFrame()
        # 将结果转换为DataFrame并重命名列以保持一致性
        df = pd.DataFrame(data_list)
        df.rename(columns={'stock__stock_code': 'ts_code'}, inplace=True)
        # print(f"DAO: 查询完成，共返回 {len(df)} 条记录。")
        return df











