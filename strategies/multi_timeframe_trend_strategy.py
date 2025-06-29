# 文件: strategies/multi_timeframe_trend_strategy.py
import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from services.indicator_services import IndicatorService
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.weekly_trend_follow_strategy import WeeklyTrendFollowStrategy
# ▼▼▼【代码修改】: 导入新的配置加载工具 ▼▼▼
from utils.config_loader import load_strategy_config
# ▲▲▲【代码修改】: 修改结束 ▲▲▲

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        self.weekly_strategy = WeeklyTrendFollowStrategy()
        self.tactical_strategy = TrendFollowStrategy() # 包含了分钟线共振逻辑
        self.indicator_service = IndicatorService()
        self.tactical_config_path = 'config/trend_follow_strategy.json'
        self.strategic_config_path = 'config/weekly_trend_follow_strategy.json'
        # ▼▼▼【代码修改】: 在初始化时预加载配置，提高效率 ▼▼▼
        # 解释: 将配置加载从运行循环中提前到初始化阶段，避免重复IO操作。
        print("--- [总指挥初始化] 正在预加载策略配置文件... ---")
        self.tactical_config = load_strategy_config(self.tactical_config_path)
        self.strategic_config = load_strategy_config(self.strategic_config_path)
        print(f"    - 战术配置 '{self.tactical_config_path}' 加载完成。")
        print(f"    - 战略配置 '{self.strategic_config_path}' 加载完成。")
        # ▲▲▲【代码修改】: 修改结束 ▲▲▲

    # ▼▼▼【代码修改】: 对核心运行逻辑进行重构，实现真正的职责分离 ▼▼▼
    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V21.0 职责分离重构版】为单个股票执行完整的多时间框架分析。
        - 核心逻辑: 总指挥全权负责数据准备的编排，IndicatorService回归单一职责。
        - 优化: 不再使用有风险的resample方法，所有数据均从数据库原生获取。
        """
        logger.info(f"--- 开始为【{stock_code}】执行多时间框架分析 (V21.0 职责分离版) ---")

        # --- 步骤 1: 并行准备所有需要的基础数据 ---
        logger.info(f"--- 步骤1: 并行准备战略层(周)与战术层(日+分钟)数据... ---")
        
        # 创建并行任务列表
        # 任务1: 根据周线配置，准备周线数据
        # 解释: 调用新的、通用的prepare_data方法，只传入周线配置字典。
        task_strategic = self.indicator_service.prepare_data(
            stock_code=stock_code,
            config=self.strategic_config, # 直接传入加载好的配置字典
            trade_time=trade_time
        )
        # 任务2: 根据日线配置，准备日线及所有分钟线数据
        # 解释: 同样调用通用的prepare_data方法，只传入日线/分钟线配置字典。
        task_tactical = self.indicator_service.prepare_data(
            stock_code=stock_code,
            config=self.tactical_config, # 直接传入加载好的配置字典
            trade_time=trade_time
        )
        
        # 并行执行
        strategic_dfs, tactical_dfs = await asyncio.gather(task_strategic, task_tactical)

        # 检查数据准备结果
        if not strategic_dfs or 'W' not in strategic_dfs:
            logger.warning(f"[{stock_code}] 战略层(周线)数据准备失败，分析终止。")
            return None
        if not tactical_dfs or 'D' not in tactical_dfs:
            logger.warning(f"[{stock_code}] 战术层(日线)数据准备失败，分析终止。")
            return None

        # --- 步骤 2: 由总指挥负责，将战略数据合并到战术数据中 ---
        logger.info(f"--- 步骤2: 整合战略(周)指标到战术(日)数据... ---")
        df_daily = tactical_dfs['D']
        df_weekly = strategic_dfs['W']
        
        # 解释: 使用 merge_asof 实现周线数据向日线数据的精确“投影”。
        # 这是将不同时间框架数据对齐的关键步骤，由总指挥完成是正确的。
        df_daily_centric = pd.merge_asof(
            left=df_daily.sort_index(),
            right=df_weekly.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward' # 使用上一周的数据填充本周的每一天
        )
        # 清理因合并产生的可能的全NaN行（例如，在数据历史的开端）
        reliable_col = next((col for col in df_daily_centric.columns if col.endswith('_W')), None)
        if reliable_col:
            df_daily_centric.dropna(subset=[reliable_col], inplace=True)
        
        print(f"    [调试-协同层] 生成的日线中心DataFrame包含列: {df_daily_centric.columns.tolist()[:5]}... 等 {len(df_daily_centric.columns)} 列")

        # --- 步骤 3: 运行战略层策略 (周线)，生成“战略信号” ---
        logger.info(f"--- 步骤3: 运行周线战略策略，生成'战略信号'... ---")
        # 周线策略在已经包含周线指标的 df_daily_centric 上运行
        strategic_context_df = await self.weekly_strategy.apply_strategy(df_daily_centric)

        if strategic_context_df is None or strategic_context_df.empty:
            logger.warning(f"[{stock_code}] 周线战略策略未能生成战略背景，后续流程终止。")
            return None

        # --- 步骤 4: 将战略信号整合回日线数据中 ---
        logger.info(f"--- 步骤4: 整合战略信号到日线数据... ---")
        # 仅合并周线策略新生成的信号列，避免重复合并指标列
        weekly_signal_cols = [col for col in strategic_context_df.columns if '_W' in col and col not in df_daily_centric.columns]
        df_daily_with_signals = pd.merge(df_daily_centric, strategic_context_df[weekly_signal_cols], left_index=True, right_index=True, how='left')
        
        # 对新合并的信号列进行前向填充
        if weekly_signal_cols:
            df_daily_with_signals[weekly_signal_cols] = df_daily_with_signals[weekly_signal_cols].ffill()
            # 对填充后可能仍然存在的NaN进行处理
            for col in weekly_signal_cols:
                if df_daily_with_signals[col].dtype == 'bool':
                    df_daily_with_signals[col] = df_daily_with_signals[col].fillna(False)
                else:
                    df_daily_with_signals[col] = df_daily_with_signals[col].fillna(0)
            print(f"    [调试-协同层] 已将 {len(weekly_signal_cols)} 个周线策略信号前向填充到日线。")
        
        # --- 步骤 5: 组装最终的数据字典 all_dfs ---
        # 日线数据已经是最完整的了，直接使用
        all_dfs = {'D': df_daily_with_signals}
        # 将战术数据准备结果中的分钟线数据加入
        for tf, df_minute in tactical_dfs.items():
            if tf != 'D': # 避免重复添加日线
                all_dfs[tf] = df_minute
        
        print(f"    [调试-协同层] 最终数据集包含的周期: {list(all_dfs.keys())}")

        # --- 步骤 6: 运行战术层策略 (日线/分钟线) ---
        logger.info(f"--- 步骤6: 运行多时间框架战术策略... ---")
        final_df, atomic_signals = await self.tactical_strategy.apply_strategy(
            all_dfs, self.tactical_strategy.daily_params
        )

        # --- 步骤 7: 打包最终结果并返回 (此部分逻辑保持不变) ---
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
    # ▲▲▲【代码修改】: 重构结束 ▲▲▲
