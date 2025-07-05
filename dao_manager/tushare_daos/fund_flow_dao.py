
import asyncio
import logging
from datetime import date, datetime, timedelta
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from dao_manager.base_dao import BaseDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.fund_flow import FundFlowCntDC, FundFlowCntTHS, FundFlowDailyBJ, FundFlowDailyCY, FundFlowDailyDC, FundFlowDailyKC, FundFlowDailySH, FundFlowDailySZ, FundFlowDailyTHS, FundFlowIndustryTHS, FundFlowMarketDc, TopInst, TopList
from stock_models.market import LimitListThs
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
        self.stock_basic_dao = StockBasicInfoDao()
        self.industry_dao = IndustryDao()
        self.stock_cache_key = StockCashKey()
        self.stock_cache_set = StockInfoCacheSet()
        self.stock_cache_get = StockInfoCacheGet()
        self.user_cache_set = UserCacheSet()
        self.user_cache_get = UserCacheGet()

    # ============== 日级资金流向数据 ==============
    def get_fund_flow_model_by_code(self, stock_code: str):
        """
        根据股票代码返回对应的【日级资金流向】数据表Model
        """
        if stock_code.startswith('3') and stock_code.endswith('.SZ'):
            return FundFlowDailyCY
        elif stock_code.endswith('.SZ'):
            return FundFlowDailySZ
        elif stock_code.startswith('68') and stock_code.endswith('.SH'):
            return FundFlowDailyKC
        elif stock_code.endswith('.SH'):
            return FundFlowDailySH
        elif stock_code.endswith('.BJ'):
            return FundFlowDailyBJ
        else:
            logger.warning(f"未识别的股票代码: {stock_code}，资金流向默认使用SZ主板表")
            return FundFlowDailySZ  # 默认返回深市主板

    async def save_history_fund_flow_daily_data_by_trade_date(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> None:
        """
        保存历史日级资金流向数据 (终极优化版 V3 - 客户端分块策略)
        1. [重构] 废弃低效的“10万行追溯”逻辑。
        2. [新增] 采用客户端分块策略，将大日期范围切分为多个小块（如90天/块）进行处理。
        3. [优化] 对每个小块使用 limit/offset 进行高效分页，避免Tushare的查询限制和性能瓶颈。
        4. [保留] 保留了向量化处理和分表存储的核心优势。
        """
        # --- 1. 日期参数处理与验证 ---
        if start_date and end_date and start_date > end_date:
            logger.error(f"日期范围无效：起始日期 {start_date} 不能晚于结束日期 {end_date}。任务终止。")
            return

        # 如果是范围查询
        if start_date and end_date:
            logger.info(f"接收到范围任务，将对 {start_date} 到 {end_date} 的数据采用客户端分块策略处理。")
        # 如果是单日查询
        elif trade_date:
            start_date = end_date = trade_date
            logger.info(f"接收到单日任务: {trade_date}")
        # 默认情况，获取当天
        else:
            start_date = end_date = date.today()
            logger.info(f"未提供日期，默认获取今日数据: {start_date}")

        # --- 2. [新增] 客户端日期分块逻辑 ---
        date_chunks = []
        chunk_size_days = 30  # 每个分块的大小（天数），90天是一个比较安全且高效的选择
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
            print(f"DAO: 开始处理日期块: {chunk_start_str} 到 {chunk_end_str}")

            offset = 0
            limit = 6000 # Tushare建议的单次最大limit
            
            # 内层分页循环 (此逻辑保持不变，但现在作用于小块)
            while True:
                try:
                    # [修改] API调用现在使用分块的起止日期
                    df = self.ts_pro.moneyflow(**{
                        "ts_code": "", "trade_date": "", # 范围查询时，trade_date应为空
                        "start_date": chunk_start_str, "end_date": chunk_end_str, 
                        "limit": limit, "offset": offset
                    }, fields=[
                        "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount", 
                        "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount", 
                        "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount", 
                        "net_mf_vol", "net_mf_amount"
                    ])
                    await asyncio.sleep(0.55) # 保持友好的API调用频率
                except Exception as e:
                    logger.error(f"Tushare API调用失败 (moneyflow, chunk: {chunk_start_str}-{chunk_end_str}): {e}")
                    await asyncio.sleep(5) # 出错时等待更长时间
                    df = pd.DataFrame()

                if df.empty:
                    break # 当前分块的当前分页无数据，结束此分块的分页
                
                all_dfs_for_market.append(df)
                
                if len(df) < limit:
                    break # 当前分块的数据已全部获取完毕
                
                offset += limit
                # [修改] 移除旧的10万行限制逻辑，因为分块策略使其不再必要
        
        if not all_dfs_for_market:
            logger.info("在所有日期块中均未获取到任何资金流向数据。")
            return

        # --- 4. [保留] 向量化数据处理与入库 (此部分逻辑完全不变) ---
        print("DAO: 所有分块数据获取完毕，开始进行数据整合与处理...")
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
        combined_df['target_model'] = combined_df['ts_code'].apply(self.get_fund_flow_model_by_code)

        total_rows = 0
        for model, group_df in combined_df.groupby('target_model'):
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

        print(f"所有历史日级资金流向数据处理完成，共保存 {total_rows} 条记录。")
        return
    
    # ============== 个股日级资金流向数据 - 同花顺 ==============
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
                print(f"{trade_date} 历史日级资金流向数据 - 同花顺，len(df): {len(df)}, len(data_dicts): {len(data_dicts)}")
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

    # ============== 日级资金流向数据 - 东方财富 ==============
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
            if df.empty:
                break
            else:
                df = df.replace(['nan', 'NaN', ''], np.nan)  # 先把字符串nan等变成np.nan
                df = df.where(pd.notnull(df), None)          # 再把所有np.nan变成None
                for row in df.itertuples():
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    if stock:
                        data_dict = self.data_format_process.set_fund_flow_data_dc(stock, row)
                        data_dicts.append(data_dict)
                print(f"{trade_date} 历史日级资金流向数据 - 东方财富，len(df): {len(df)}, len(data_dicts): {len(data_dicts)}")
            time.sleep(0.2)
            if len(df) < limit:
                break
            offset += limit
        result =  await self._save_all_to_db_native_upsert(
            model_class=FundFlowDailyDC,
            data_list=data_dicts,
            unique_fields=['stock', 'trade_time']
        )
        return result

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
        
        # 1. 准备API请求参数 (逻辑不变)
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""

        # 2. 分页拉取并处理数据
        offset = 0
        limit = 4000
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
                print("调试信息: Tushare返回数据为空，拉取结束。")
                break
            
            original_count = len(df)
            print(f"调试信息: 成功拉取 {original_count} 条数据，开始进行向量化处理。")

            # 3. 向量化数据清洗与转换
            # 3.1 清洗空值和无效行
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                print("调试信息: 数据清洗后，当前批次无有效数据。")
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
                print("调试信息: 关联ThsIndex后，当前批次无有效数据。")
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
            print(f"调试信息: 当前批次处理完成，获得 {len(df_processed)} 条有效数据。累计待保存数据: {len(all_data_to_save)} 条。")

            # 5. 分页逻辑 (逻辑不变)
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit

        # 6. 一次性批量保存到数据库
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何有效数据。")
            print("调试信息: 没有可保存的数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        print(f"调试信息: 所有数据拉取和处理完成，准备将 {len(all_data_to_save)} 条数据一次性保存到数据库。")
        # 【逻辑修正】unique_fields 从 ['stock', 'trade_time'] 修正为 ['ths_index', 'trade_time']
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

        # 1. 准备API请求参数 (逻辑不变)
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""

        # 2. 分页拉取并处理数据
        offset = 0
        limit = 5000
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
                print("调试信息: Tushare返回数据为空，拉取结束。")
                break
            
            original_count = len(df)
            print(f"调试信息: 成功拉取 {original_count} 条数据，开始进行向量化处理。")

            # 3. 向量化数据清洗与转换
            # 3.1 清洗空值和无效行
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                print("调试信息: 数据清洗后，当前批次无有效数据。")
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
                print("调试信息: 关联DcIndex后，当前批次无有效数据。")
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
            print(f"调试信息: 当前批次处理完成，获得 {len(df_processed)} 条有效数据。累计待保存数据: {len(all_data_to_save)} 条。")

            # 5. 分页逻辑 (逻辑不变)
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit

        # 6. 一次性批量保存到数据库
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何东方财富板块资金流数据。")
            print("调试信息: 没有可保存的数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        print(f"调试信息: 所有数据拉取和处理完成，准备将 {len(all_data_to_save)} 条数据一次性保存到数据库。")
        # 【逻辑修正】unique_fields 从 ['stock', 'trade_time'] 修正为 ['dc_index', 'trade_time']
        result = await self._save_all_to_db_native_upsert(
            model_class=FundFlowCntDC,
            data_list=all_data_to_save,
            unique_fields=['dc_index', 'trade_time']
        )

        date_range_info = f"trade_date={trade_date_str}" if trade_date_str else f"start={start_date_str}, end={end_date_str}"
        print(f"完成 {date_range_info} 历史板块资金流向数据（东方财富），result: {result}")
        return result
    
    # ============== 行业资金流向数据 - 同花顺 ==============
    async def save_history_fund_flow_industry_ths_data(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【V2.0 - 向量化优化版】保存历史行业资金流向数据 - 同花顺
        核心优化:
        1. 【消除N+1查询】将循环内单次数据库查询，改为批处理模式。一次性获取所有ts_code，一次性查询数据库。
        2. 【向量化数据处理】使用Pandas的向量化操作替代低效的逐行循环和数据格式化函数调用。
        3. 【修正逻辑错误】修正了unique_fields参数，使其与模型定义一致。
        4. 【代码健壮性】增加了对空数据和无效关联的过滤，使数据管道更稳定。
        """
        # --- 代码修改开始 ---

        # 1. 准备API请求参数 (逻辑不变)
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
                print("调试信息: Tushare返回数据为空，拉取结束。")
                break
            
            original_count = len(df)
            print(f"调试信息: 成功拉取 {original_count} 条数据，开始进行向量化处理。")

            # 3. 向量化数据清洗与转换
            # 3.1 清洗空值和无效行
            df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
            df.dropna(subset=['ts_code', 'trade_date'], inplace=True)
            if df.empty:
                print("调试信息: 数据清洗后，当前批次无有效数据。")
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
                print("调试信息: 关联ThsIndex后，当前批次无有效数据。")
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
            print(f"调试信息: 当前批次处理完成，获得 {len(df_processed)} 条有效数据。累计待保存数据: {len(all_data_to_save)} 条。")

            # 5. 分页逻辑 (逻辑不变)
            time.sleep(0.2)
            if original_count < limit:
                break
            offset += limit

        # 6. 一次性批量保存到数据库
        if not all_data_to_save:
            logger.warning(f"在日期范围 {start_date_str}-{end_date_str} 内没有找到或处理任何同花顺行业资金流数据。")
            print("调试信息: 没有可保存的数据。")
            return {"尝试处理": 0, "失败": 0, "创建/更新成功": 0}

        print(f"调试信息: 所有数据拉取和处理完成，准备将 {len(all_data_to_save)} 条数据一次性保存到数据库。")
        # 【逻辑修正】unique_fields 从 ['stock', 'trade_time'] 修正为 ['ths_index', 'trade_time']
        result = await self._save_all_to_db_native_upsert(
            model_class=FundFlowIndustryTHS,
            data_list=all_data_to_save,
            unique_fields=['ths_index', 'trade_time']
        )
        
        # --- 代码修改结束 ---
        
        date_range_info = f"trade_date={trade_date_str}" if trade_date_str else f"start={start_date_str}, end={end_date_str}"
        # 【修正打印信息】将“板块”修正为“行业”
        print(f"完成 {date_range_info} 行业资金流向数据（同花顺），result: {result}")
        return result

    # ============== 大盘资金流向数据 - 东方财富 ==============
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
        【V2 - 优化版】保存历史龙虎榜机构明细，并支持分页拉取
        
        优化点:
        1. [核心功能] 增加了分页逻辑，通过循环和 offset 参数获取指定日期的全部数据，最大支持10万行。
        2. [核心性能] 使用 `get_stocks_by_codes` 解决了N+1数据库查询问题。
        3. 整合了分页数据，进行统一的批量关联和批量保存。
        4. 增强了代码的健壮性和日志清晰度。
        
        Args:
            trade_date (str): 日期，格式为 YYYYMMDD
        """
        print(f"调试: 开始执行 save_hisroty_lhb_inst_data 任务, trade_date={trade_date}")
        
        try:
            # 1. [代码修改处] 分页循环拉取API数据
            all_dfs = []
            limit = 10000  # Tushare单次最大返回量
            offset = 0
            max_records = 100000 # 最大拉取10万行作为保护

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

                # 如果返回的数据量小于请求的limit，说明已经是最后一页
                if len(df) < limit:
                    print(f"调试: 已获取最后一页数据({len(df)}条)，分页拉取结束。")
                    break
                
                offset += limit
                # 增加保护，防止意外的无限循环
                if offset >= max_records:
                    logger.warning(f"拉取数据已达到 {max_records} 条上限，自动停止。")
                    print(f"调试: 拉取数据已达到 {max_records} 条上限，自动停止。")
                    break
            
            if not all_dfs:
                logger.info(f"交易日 {trade_date} 没有龙虎榜机构明细数据。")
                return {"status": "success", "message": "No data for this trade date.", "saved_count": 0}

            # 合并所有分页数据
            final_df = pd.concat(all_dfs, ignore_index=True)
            print(f"调试: 分页拉取完成，共获取 {len(final_df)} 条数据。")

            # 2. [代码修改处] 批量获取所有相关的股票基础信息对象
            unique_ts_codes = final_df['ts_code'].unique().tolist()
            print(f"调试: 数据涉及 {len(unique_ts_codes)} 个独立股票代码。")
            
            # [代码修改处] 使用 get_stocks_by_codes 方法，一次性查询数据库，解决N+1问题
            # 注意：这里我们用 stock_basic_dao 替换了原先的 stock_cache_get 以实现批量操作
            stock_map = await self.stock_basic_dao.get_stocks_by_codes(unique_ts_codes)
            print(f"调试: 批量从数据库获取了 {len(stock_map)} 个股票对象。")

            # 3. 准备批量写入的数据
            data_dicts_to_save = []
            for row in final_df.itertuples():
                # [代码修改处] 从预先查好的映射中获取股票对象，高效且无N+1问题
                stock_instance = stock_map.get(row.ts_code)
                
                if stock_instance:
                    data_dict = self.data_format_process.set_lhb_inst_data(stock_instance, row)
                    data_dicts_to_save.append(data_dict)
                else:
                    logger.warning(f"在数据库中未找到股票代码 {row.ts_code} 的基础信息，已跳过该条龙虎榜数据。")

            # 4. 批量写入数据库
            result = {}
            if data_dicts_to_save:
                print(f"调试: 准备批量保存 {len(data_dicts_to_save)} 条龙虎榜数据到数据库...")
                # 注意：请确保 unique_fields 中的字段名与 TopInst 模型中的字段名完全一致。
                # Tushare返回的是 trade_date，如果您的模型中是 trade_time，请确保 set_lhb_inst_data 方法做了转换。
                result = await self._save_all_to_db_native_upsert(
                    model_class=TopInst,
                    data_list=data_dicts_to_save,
                    unique_fields=['stock', 'trade_date'] # 建议使用 trade_date，与API字段保持一致
                )
                print(f"调试: 成功保存 {len(data_dicts_to_save)} 条数据。")
            else:
                logger.info("经过筛选后，没有需要保存到数据库的数据。")
                print("调试: 经过筛选后，没有需要保存的数据。")
                result = {"status": "success", "message": "No new data to save.", "saved_count": 0}
                
            return result

        except Exception as e:
            logger.error(f"保存龙虎榜机构明细时发生严重错误: {e}", exc_info=True)
            print(f"调试: 发生异常: {e}")
            raise

    # ============== 游资每日明细 ==============
    


    # ============== 涨跌停榜单 - 同花顺 无权限，不用 ==============
    async def save_limit_list_ths(self, trade_date: date = None, start_date: date = None, end_date: date = None) -> Dict:
        """
        【修改】保存同花顺涨跌停榜单数据 (limit_list_ths接口)
        """
        # --- 参数处理，保持不变 ---
        # 优雅地处理日期参数，如果为None则生成空字符串
        trade_date_str = trade_date.strftime('%Y%m%d') if trade_date else ""
        start_date_str = start_date.strftime('%Y%m%d') if start_date else ""
        end_date_str = end_date.strftime('%Y%m%d') if end_date else ""
        
        # --- 分页拉取逻辑，保持并优化 ---
        offset = 0
        limit = 500  # Tushare单次最大返回5000，这里用500作为分页大小是稳妥的
        data_dicts = []
        
        print(f"开始拉取同花顺涨跌停榜单数据... trade_date: {trade_date_str}, start_date: {start_date_str}, end_date: {end_date_str}")

        while True:
            # 【新增】增加安全退出机制，防止意外的无限循环
            if offset >= 100000:
                logger.warning(f"同花顺涨跌停榜单 offset已达10万，为安全起见停止拉取。")
                break
            
            try:
                # 【修改】调用正确的Tushare接口：limit_list_ths
                df = self.ts_pro.limit_list_ths(**{
                    "trade_date": trade_date_str,
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "limit": limit,
                    "offset": offset
                    # ts_code, limit_type, market 等参数留空，表示获取全部
                }, fields=[
                    "trade_date", "ts_code", "name", "price", "pct_chg", "open_num",
                    "lu_desc", "limit_type", "tag", "status", "limit_order", "limit_amount",
                    "turnover_rate", "free_float", "lu_limit_order", "limit_up_suc_rate",
                    "turnover", "market_type", "first_lu_time", "last_lu_time",
                    "first_ld_time", "last_ld_time", "rise_rate", "sum_float"
                ])
            except Exception as e:
                # 【新增】增加异常捕获，防止因API问题导致程序崩溃
                logger.error(f"拉取同花顺涨跌停榜单时发生异常: {e}")
                break # 发生异常时，终止循环

            # --- 数据处理逻辑，保持并优化 ---
            if df.empty:
                # 如果当前分页返回数据为空，说明已经拉取完毕
                print(f"拉取完成，在 offset={offset} 处未获取到更多数据。")
                break
            else:
                print(f"成功拉取到 {len(df)} 条数据，offset={offset}, limit={limit}")
                # 【保持】优秀的数据清洗逻辑：将各种空值统一处理为None，便于数据库存储
                df = df.replace(['nan', 'NaN', ''], np.nan)
                df = df.where(pd.notnull(df), None)
                
                # 遍历DataFrame，准备存入数据库的数据
                for row in df.itertuples(index=False):
                    # 【保持】通过缓存高效获取关联的StockInfo对象
                    stock = await self.stock_cache_get.stock_data_by_code(row.ts_code)
                    if stock:
                        # 【修改】调用与LimitListThs模型匹配的数据格式化方法
                        # 你需要确保有一个 `set_limit_list_ths_data` 方法来处理数据转换
                        data_dict = self.data_format_process.set_limit_list_ths_data(stock, row)
                        data_dicts.append(data_dict)
                    else:
                        # 【新增】增加日志，记录未找到对应股票信息的情况
                        logger.warning(f"在涨跌停榜单中找到股票 {row.ts_code}，但在StockInfo表中未找到，已跳过。")

            # 【保持】礼貌地请求API，避免请求过于频繁
            time.sleep(0.2)
            
            # 如果返回的数据量小于请求的limit，说明是最后一页，可以提前退出循环
            if len(df) < limit:
                print("已到达数据末尾，退出循环。")
                break
            
            # 准备下一次分页请求
            offset += limit

        # --- 批量入库逻辑，保持并优化 ---
        if not data_dicts:
            # 【新增】如果没有任何数据需要保存，提前告知并返回
            msg = f"没有需要保存的同花顺涨跌停榜单数据。查询参数: trade_date={trade_date_str}, start_date={start_date_str}, end_date={end_date_str}"
            print(msg)
            return {"status": "success", "message": msg, "saved_count": 0}

        # 【修改】调用批量保存方法，并使用正确的模型和唯一键
        result = await self._save_all_to_db_native_upsert(
            model_class=LimitListThs,  # 【修改】使用正确的模型 LimitListThs
            data_list=data_dicts,
            # 【修改】使用与模型定义匹配的联合唯一键
            unique_fields=['trade_date', 'stock', 'limit_type'] 
        )
        
        # 打印最终结果
        print(f"完成同花顺涨跌停榜单数据保存，结果: {result}")
        return result













