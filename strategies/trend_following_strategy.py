# 文件: strategies/trend_following_strategy.py
# 版本: V101.0 · 终极信号适配版
import logging
import pandas as pd
from typing import Dict, Optional, Tuple, List

from .trend_following.intelligence_layer import IntelligenceLayer
from .trend_following.offensive_layer import OffensiveLayer
from .trend_following.warning_layer import WarningLayer
from .trend_following.structural_defense_layer import StructuralDefenseLayer
from .trend_following.judgment_layer import JudgmentLayer
from .trend_following.simulation_layer import SimulationLayer
from .trend_following.reporting_layer import ReportingLayer
from .trend_following.utils import ensure_numeric_types, optimize_df_memory

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    【V101.0 · 终极信号适配版】
    - 核心重构: 净化了 apply_strategy 的主流程，现在完全依赖 IntelligenceLayer 来完成所有情报的生成与融合。
    """
    def __init__(self, config: dict):
        self.unified_config = config
        self.params = {}
        self.atomic_states = {}
        self.playbook_states = {}
        self.trigger_events = {}
        self.df_indicators = pd.DataFrame()
        
        self.intelligence_layer = IntelligenceLayer(self)
        self.offensive_layer = OffensiveLayer(self)
        self.warning_layer = WarningLayer(self)
        self.structural_defense_layer = StructuralDefenseLayer(self)
        self.judgment_layer = JudgmentLayer(self)
        self.simulation_layer = SimulationLayer(self)
        self.reporting_layer = ReportingLayer(self)

    def apply_strategy(self, all_dfs: Dict[str, pd.DataFrame], params: dict, start_date_str: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V405.0 · 普罗米修斯盗火协议版】
        - 核心修复: 修复了因调用链断裂导致的 TypeError。
        - 核心逻辑: 在情报层运行后，立即计算权威的上下文分数，并将其作为参数传递给进攻层，
                      确保计分引擎在执行时能够感知上下文，完成“赫淮斯托斯协议”的闭环。
        """
        self.params = params
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_indicators = self._merge_all_timeframes(all_dfs)
        # --- 指挥链 1/7: 情报层 (现在是所有情报的唯一入口，包括硬性离场) ---
        self.intelligence_layer.run_all_diagnostics()
        # 步骤1.5: 普罗米修斯盗火 - 计算并获取权威的上下文分数
        from .trend_following.utils import calculate_context_scores
        bottom_context_score, top_context_score = calculate_context_scores(self.df_indicators, self.atomic_states)
        # --- 指挥链 2/7: 进攻层 ---
        # 将上下文分数作为“火种”注入计分引擎
        entry_score, score_details_df = self.offensive_layer.calculate_entry_score(
            self.trigger_events,
            bottom_context_score,
            top_context_score
        )
        self.df_indicators['entry_score'] = entry_score
        # --- 指挥链 3/7: 预警层 ---
        risk_details_df = self.warning_layer.run_all_warnings()
        # --- 指挥链 4/7: 统合判断层 (最终决策者) ---
        self.judgment_layer.make_final_decisions(score_details_df, risk_details_df)
        # --- 指挥链 6/7 & 7/7: 模拟层与报告层 ---
        self.simulation_layer.run_position_management_simulation()
        self.df_indicators = optimize_df_memory(self.df_indicators, verbose=False)
        if risk_details_df is None:
            risk_details_df = pd.DataFrame(index=self.df_indicators.index)
        return self.df_indicators, score_details_df, risk_details_df

    def _merge_all_timeframes(self, all_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        将所有时间框架的数据合并到一个DataFrame中。
        """
        if 'D' not in all_dfs:
            return pd.DataFrame()
            
        merged_df = all_dfs['D'].copy()
        
        for tf, df in all_dfs.items():
            if tf == 'D' or df.empty:
                continue
            # 对齐索引并向前填充
            aligned_df = df.reindex(merged_df.index, method='ffill')
            # 合并，避免重复列
            merged_df = merged_df.merge(
                aligned_df.drop(columns=[col for col in aligned_df.columns if col in merged_df.columns and col != 'trade_date'], errors='ignore'),
                left_index=True,
                right_index=True,
                how='left'
            )
        return ensure_numeric_types(merged_df)

    async def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> Tuple[List, List, List, List, List]:
        """
        对外暴露的报告生成接口。
        """
        return await self.reporting_layer.prepare_db_records(stock_code, result_df, score_details_df, risk_details_df, params, result_timeframe)
