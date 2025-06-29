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
        self.tactical_strategy = TrendFollowStrategy() # 包含了分钟线共振逻辑
        self.indicator_service = IndicatorService()
        self.tactical_config_path = 'config/trend_follow_strategy.json'
        self.strategic_config_path = 'config/weekly_trend_follow_strategy.json'
        # 假设未来会有月线配置
        # self.monthly_config_path = 'config/monthly_trend_follow_strategy.json'

    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V19.2 终极简化版】为单个股票执行完整的多时间框架分析。
        - 核心逻辑: 回归简单、明确的编排。总指挥按顺序调用专职数据准备方法。
        """
        logger.info(f"--- 开始为【{stock_code}】执行多时间框架分析 (V19.2 终极简化版) ---")

        # --- 步骤 1: 准备战略层数据 (日线+周线+月线合并) ---
        logger.info(f"--- 步骤1: 准备战略分析所需数据 (日线为中心)... ---")
        
        # ▼▼▼【修改】调用专职方法准备日线为中心的数据 ▼▼▼
        # 这个方法内部处理日、周、月配置的加载和数据合并
        df_daily_centric = await self.indicator_service.prepare_daily_centric_dataframe(
            stock_code=stock_code,
            trade_time=trade_time,
            daily_config_path=self.tactical_config_path,
            weekly_config_path=self.strategic_config_path
            # monthly_config_path=self.monthly_config_path # 未来可扩展
        )
        # ▲▲▲ 修改结束 ▲▲▲

        if df_daily_centric is None or df_daily_centric.empty:
            logger.warning(f"[{stock_code}] indicator_service.prepare_daily_centric_dataframe 返回了 None，分析终止。")
            return None
        
        print(f"    [调试-协同层] 生成的日线中心DataFrame包含列: {df_daily_centric.columns.tolist()[:5]}... 等 {len(df_daily_centric.columns)} 列")

        # --- 步骤 2: 运行战略层策略 (周线)，生成“战略信号” ---
        logger.info(f"--- 步骤2: 运行周线战略策略，生成'战略信号'... ---")
        # 周线策略直接在已经包含周线指标的 df_daily_centric 上运行
        strategic_context_df = await self.weekly_strategy.apply_strategy(df_daily_centric)

        if strategic_context_df is None or strategic_context_df.empty:
            logger.warning(f"[{stock_code}] 周线战略策略未能生成战略背景，后续流程终止。")
            return None

        # --- 步骤 3: 将战略信号整合回日线数据中 ---
        logger.info(f"--- 步骤3: 整合战略信号到日线数据... ---")
        # 仅合并周线策略新生成的信号列，避免重复合并指标列
        weekly_signal_cols = [col for col in strategic_context_df.columns if '_W' in col and col not in df_daily_centric.columns]
        df_daily_with_signals = pd.merge(df_daily_centric, strategic_context_df[weekly_signal_cols], left_index=True, right_index=True, how='left')
        
        if weekly_signal_cols:
            df_daily_with_signals[weekly_signal_cols] = df_daily_with_signals[weekly_signal_cols].fillna(method='ffill')
            # 对填充后可能仍然存在的NaN进行处理
            for col in weekly_signal_cols:
                if df_daily_with_signals[col].dtype == 'bool':
                    df_daily_with_signals[col] = df_daily_with_signals[col].fillna(False)
                else:
                    df_daily_with_signals[col] = df_daily_with_signals[col].fillna(0)
            print(f"    [调试-协同层] 已将 {len(weekly_signal_cols)} 个周线策略信号前向填充到日线。")
        
        # --- 步骤 4: 并行准备所有战术层(分钟线)数据 ---
        logger.info(f"--- 步骤4: 并行准备战术分析所需分钟线数据... ---")
        
        # 明确指定需要哪些分钟线周期 
        minute_timeframes = ['60', '15', '5'] # 从日线配置中可以动态读取，但这里写死更简单
        
        minute_tasks = [
            self.indicator_service.prepare_minute_centric_dataframe(
                stock_code=stock_code,
                params_file=self.tactical_config_path, # 所有分钟线指标配置都在日线配置文件里
                timeframe=tf,
                trade_time=trade_time
            ) for tf in minute_timeframes
        ]
        
        minute_results = await asyncio.gather(*minute_tasks)
        
        # --- 步骤 5: 组装最终的数据字典 all_dfs ---
        all_dfs = {'D': df_daily_with_signals}
        for i, tf in enumerate(minute_timeframes):
            df_minute, _ = minute_results[i] if minute_results[i] else (None, None)
            if df_minute is not None and not df_minute.empty:
                all_dfs[tf] = df_minute
            else:
                logger.warning(f"[{stock_code}] 未能获取到 {tf} 分钟数据，后续策略可能受影响。")
        
        print(f"    [调试-协同层] 最终数据集包含的周期: {list(all_dfs.keys())}")

        # --- 步骤 6: 运行战术层策略 (日线/分钟线) ---
        logger.info(f"--- 步骤6: 运行多时间框架战术策略 (V19.2 协同版)... ---")
        final_df, atomic_signals = await self.tactical_strategy.apply_strategy(
            all_dfs, self.tactical_strategy.daily_params
        )

        # --- 步骤 7: 打包最终结果并返回 ---
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





