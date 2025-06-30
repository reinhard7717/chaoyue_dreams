# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V2.1 - 初始化依赖注入修复版
import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from services.indicator_services import IndicatorService
# ▼▼▼【代码修改】: 导入 TrendFollowStrategy 以便实例化 ▼▼▼
from strategies.trend_following_strategy import TrendFollowStrategy
# ▲▲▲【代码修改】: 修改结束 ▲▲▲
from strategies.weekly_trend_follow_strategy import WeeklyTrendFollowStrategy
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        """
        【V2.1 初始化依赖注入修复版】
        - 修复: 在创建 TrendFollowStrategy 实例时，将已加载的战术配置字典传递给它。
        """
        # 1. 加载所有需要的配置文件
        self.tactical_config_path = 'config/trend_follow_strategy.json'
        self.strategic_config_path = 'config/weekly_trend_follow_strategy.json'
        self.tactical_config = load_strategy_config(self.tactical_config_path)
        self.strategic_config = load_strategy_config(self.strategic_config_path)

        # 2. 实例化所有需要的服务和子策略
        self.indicator_service = IndicatorService()
        self.weekly_strategy = WeeklyTrendFollowStrategy() # 周线策略的__init__不需要参数

        # ▼▼▼【代码修改】: 这是本次修复的核心 ▼▼▼
        # 解释: TrendFollowStrategy的构造函数需要一个配置字典。
        # 我们在这里将刚刚加载的 self.tactical_config 传递给它。
        self.tactical_strategy = TrendFollowStrategy(config=self.tactical_config)
        # ▲▲▲【代码修改】: 修改结束 ▲▲▲


    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V2.0 逻辑修正版】为单个股票执行完整的多时间框架分析。
        - 核心修正: 严格分离战略层和战术层的数据处理，确保周线策略在纯周线数据上运行。
        """
        logger.info(f"--- 开始为【{stock_code}】执行多时间框架分析 (V2.1 修复版) ---")

        # --- 步骤 1: 并行准备战略(周)和战术(日/分钟)数据 ---
        logger.info(f"--- 步骤1: 调用 IndicatorService 并行准备所有数据... ---")
        task_strategic = self.indicator_service.prepare_data_for_strategy(
            stock_code=stock_code, config=self.strategic_config, trade_time=trade_time
        )
        task_tactical = self.indicator_service.prepare_data_for_strategy(
            stock_code=stock_code, config=self.tactical_config, trade_time=trade_time
        )
        strategic_dfs, tactical_dfs = await asyncio.gather(task_strategic, task_tactical)

        if not strategic_dfs or 'W' not in strategic_dfs:
            logger.warning(f"[{stock_code}] 战略层(周线)数据准备失败，分析终止。")
            return None
        if not tactical_dfs or 'D' not in tactical_dfs:
            logger.warning(f"[{stock_code}] 战术层(日线)数据准备失败，分析终止。")
            return None

        df_weekly = strategic_dfs['W']
        df_daily = tactical_dfs['D']

        # --- 步骤 2: 运行战略层策略 (在纯周线数据上) ---
        logger.info(f"--- 步骤2: 运行周线战略策略，生成'战略信号'... ---")
        weekly_signals_df = self.weekly_strategy.apply_strategy(df_weekly)

        if weekly_signals_df is None or weekly_signals_df.empty:
            logger.warning(f"[{stock_code}] 周线战略策略未能生成战略背景，但将继续进行战术分析。")
            weekly_signals_df = pd.DataFrame(index=df_weekly.index)

        # --- 步骤 3: 将战略信号整合到日线数据中 ---
        logger.info(f"--- 步骤3: 整合战略信号到日线数据... ---")
        df_daily.index = pd.to_datetime(df_daily.index).tz_localize(None)
        weekly_signals_df.index = pd.to_datetime(weekly_signals_df.index).tz_localize(None)
        
        df_daily_with_signals = pd.merge_asof(
            left=df_daily.sort_index(),
            right=weekly_signals_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        if 'signal_breakout_trigger_W' in df_daily_with_signals.columns:
            df_daily_with_signals.rename(columns={'signal_breakout_trigger_W': 'BASE_SIGNAL_BREAKOUT_TRIGGER'}, inplace=True)
            print("    - [协同层] 已将周线王牌信号 'signal_breakout_trigger_W' 重命名为 'BASE_SIGNAL_BREAKOUT_TRIGGER'")
        
        signal_cols = list(weekly_signals_df.columns)
        if 'BASE_SIGNAL_BREAKOUT_TRIGGER' in df_daily_with_signals.columns:
            signal_cols.append('BASE_SIGNAL_BREAKOUT_TRIGGER')
            
        for col in signal_cols:
            if col in df_daily_with_signals.columns:
                if df_daily_with_signals[col].dtype == 'bool':
                    df_daily_with_signals[col] = df_daily_with_signals[col].fillna(False)
                else:
                    df_daily_with_signals[col] = df_daily_with_signals[col].fillna(0)
        
        print(f"    - [协同层] 已将 {len(weekly_signals_df.columns)} 个周线策略信号合并到日线。")

        # --- 步骤 4: 组装最终的数据字典 all_dfs ---
        all_dfs = {'D': df_daily_with_signals}
        for tf, df_minute in tactical_dfs.items():
            if tf != 'D':
                all_dfs[tf] = df_minute
        
        print(f"    - [协同层] 最终数据集包含的周期: {list(all_dfs.keys())}")

        # --- 步骤 5: 运行战术层策略 (日线/分钟线) ---
        logger.info(f"--- 步骤5: 运行多时间框架战术策略... ---")
        final_df, atomic_signals = self.tactical_strategy.apply_strategy(
            all_dfs, self.tactical_config
        )

        # --- 步骤 6: 打包最终结果并返回 ---
        if final_df is None or final_df.empty:
            logger.info(f"\n--- 【{stock_code}】战术策略运行未产生有效结果DataFrame ---")
            return None
        
        logger.info(f"[{stock_code}] 战术策略分析完成，准备数据库记录...")
        db_records = self.tactical_strategy.prepare_db_records(
            stock_code, 
            final_df, 
            atomic_signals, 
            params=self.tactical_config
        )
        
        logger.info(f"--- 【{stock_code}】多时间框架分析完成，共生成 {len(db_records) if db_records else 0} 条信号记录。 ---")
        return db_records
