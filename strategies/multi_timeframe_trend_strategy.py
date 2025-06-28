# 文件: strategies/multi_timeframe_trend_strategy.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pandas_ta
from services.indicator_services import IndicatorService
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.weekly_trend_follow_strategy import WeeklyTrendFollowStrategy

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        self.weekly_strategy = WeeklyTrendFollowStrategy()
        self.tactical_strategy = TrendFollowStrategy()
        self.indicator_service = IndicatorService()

    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V18.0 适配版】为单个股票执行完整的多时间框架分析。
        - 适配 WeeklyTrendFollowStrategy V18.0 的新接口。
        """
        logger.info(f"--- 开始为【{stock_code}】执行多时间框架分析 (V18.0 适配版) ---")

        # --- 步骤 1: 准备战略层数据 (日线+周线合并) ---
        logger.info(f"--- 步骤1: 准备战略分析所需数据 (日线为中心)... ---")
        
        df_daily_centric = await self.indicator_service.prepare_daily_centric_dataframe(
            stock_code=stock_code,
            daily_config_path='config/trend_follow_strategy.json',
            weekly_config_path='config/weekly_trend_follow_strategy.json',
            trade_time=trade_time
        )

        if df_daily_centric is None or df_daily_centric.empty:
            logger.warning(f"[{stock_code}] indicator_service.prepare_daily_centric_dataframe 返回了 None，分析终止。")
            return None
        
        print(f"    [调试-协同层] 生成的DataFrame包含列: {df_daily_centric.columns.tolist()[:5]}... 等 {len(df_daily_centric.columns)} 列")

        # --- 步骤 2: 运行战略层策略 (周线)，生成“战略背景”DataFrame ---
        logger.info(f"--- 步骤2: 运行周线战略策略，生成多剧本'战略背景'... ---")
        
        # ▼▼▼【修改】适配 V18.0 的新接口 ▼▼▼
        # 解释: WeeklyTrendFollowStrategy.apply_strategy 现在内部管理其参数，
        # 并且直接返回一个包含所有原始及合成信号的 DataFrame。
        strategic_context_df = await self.weekly_strategy.apply_strategy(df_daily_centric)
        # ▲▲▲ 修改结束 ▲▲▲

        if strategic_context_df is None or strategic_context_df.empty:
            logger.warning(f"[{stock_code}] 周线战略策略未能生成战略背景，后续流程终止。")
            return None

        # --- 步骤 3: 将战略背景整合到日线数据中 ---
        logger.info(f"--- 步骤3: 整合战略背景到日线数据，并进行'前向填充'... ---")
        # 注意：由于 strategic_context_df 是基于 df_daily_centric 计算的，它们的索引是相同的。
        # 但为了安全和清晰，我们只选择周线策略生成的列进行合并。
        weekly_cols = [col for col in strategic_context_df.columns if '_W' in col]
        df_daily_with_context = pd.merge(df_daily_centric, strategic_context_df[weekly_cols], left_index=True, right_index=True, how='left')
        
        # 对所有周线信号进行前向填充，使其在日线级别上连续有效
        if weekly_cols:
            # 对布尔型和数值型信号都进行填充
            df_daily_with_context[weekly_cols] = df_daily_with_context[weekly_cols].fillna(method='ffill')
            # 对填充后可能仍然存在的NaN（通常在数据开头）进行处理
            for col in weekly_cols:
                if df_daily_with_context[col].dtype == 'bool':
                    df_daily_with_context[col] = df_daily_with_context[col].fillna(False)
                else:
                    df_daily_with_context[col] = df_daily_with_context[col].fillna(0)
            print(f"    [调试-协同层] 已将 {len(weekly_cols)} 个周线信号(原始+合成)前向填充到日线。")
        
        # --- 步骤 4: 准备战术层数据 (传入预计算结果) ---
        logger.info(f"--- 步骤4: 准备战术分析所需数据 (复用已整合战略背景的数据)... ---")
        all_dfs = {
            'D': df_daily_with_context
        }

        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
            logger.warning(f"[{stock_code}] 准备战术层数据(all_dfs)失败或缺少核心日线数据，分析终止。")
            return None

        # --- 步骤 5: 运行战术层策略 (日线/分钟线) ---
        logger.info(f"--- 步骤5: 运行多时间框架战术策略 (V18.0 协同版)... ---")
        final_df, atomic_signals = await self.tactical_strategy.apply_strategy(
            all_dfs, self.tactical_strategy.daily_params
        )

        # --- 步骤 6: 打包最终结果并返回 ---
        if final_df is None or final_df.empty:
            logger.info(f"\n--- 【{stock_code}】战术策略运行未产生有效结果DataFrame ---")
            return None
        
        has_entry = final_df.get('signal_entry', pd.Series(False)).any()
        has_exit = final_df.get('take_profit_signal', pd.Series(0)).any()
        if not (has_entry or has_exit):
             logger.info(f"\n--- 【{stock_code}】战术策略运行完成，但未触发任何买入或卖出信号。 ---")
             return None

        logger.info(f"[{stock_code}] 战术策略分析完成，准备数据库记录...")
        db_records = self.tactical_strategy.prepare_db_records(
            stock_code, 
            final_df, 
            atomic_signals, 
            params=self.tactical_strategy.daily_params
        )
        
        logger.info(f"--- 【{stock_code}】多时间框架分析完成，共生成 {len(db_records) if db_records else 0} 条信号记录。 ---")
        return db_records
    