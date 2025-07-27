# 文件: strategies/trend_following_strategy.py
# 版本: V100.0 - 模块化重构版
import logging
import pandas as pd
from typing import Tuple

from .trend_following.intelligence_layer import IntelligenceLayer
from .trend_following.offensive_layer import OffensiveLayer
from .trend_following.warning_layer import WarningLayer
from .trend_following.exit_layer import ExitLayer
from .trend_following.judgment_layer import JudgmentLayer
from .trend_following.simulation_layer import SimulationLayer
from .trend_following.reporting_layer import ReportingLayer
from .trend_following.utils import ensure_numeric_types, format_debug_dates, get_param_value, get_params_block

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    趋势跟踪策略 (V100.0 - 模块化重构版)
    - 核心重构: 将单一的策略文件，按照“情报、进攻、预警、离场、判断、模拟、报告”的
                职责，拆分为七个高度内聚、低耦合的独立模块。
    - 收益: 极大地提升了代码的可读性、可维护性和可扩展性，使复杂的策略逻辑变得清晰可控。
    """
    def __init__(self, config: dict):
        self.unified_config = config
        self.strategy_info = get_params_block(self, 'strategy_info')
        
        # 初始化所有分层模块，并将主策略实例(self)传递给它们，以便共享配置和状态
        self.intelligence_layer = IntelligenceLayer(self)
        self.offensive_layer = OffensiveLayer(self)
        self.warning_layer = WarningLayer(self)
        self.exit_layer = ExitLayer(self)
        self.judgment_layer = JudgmentLayer(self)
        self.simulation_layer = SimulationLayer(self)
        self.reporting_layer = ReportingLayer(self)

        # 共享状态初始化
        self.atomic_states = {}
        self.playbook_states = {}
        self.setup_scores = {}
        self.df_indicators = pd.DataFrame()

    def apply_strategy(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V100.0 模块化版】
        - 新职责: 作为总指挥，按顺序调用各大作战分层，完成从情报收集到最终决策的全过程。
        """
        print("======================================================================")
        print(f"====== 日期: {df.index[-1].date()} | 正在执行【模块化战术引擎 V100.0】 ======")
        print("======================================================================")

        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        self.df_indicators = ensure_numeric_types(df)

        # --- 指挥链 1/7: 情报层 ---
        # 收集所有战场情报，生成原子状态、认知状态和主力行为序列
        print("    --- [指挥链 1/7] 情报层: 正在收集所有战场情报... ---")
        trigger_events = self.intelligence_layer.run_all_diagnostics()

        # --- 指挥链 2/7: 进攻层 ---
        # 评估所有进攻机会，计算 entry_score
        print("    --- [指挥链 2/7] 进攻层: 正在评估所有进攻方案... ---")
        entry_score, score_details_df = self.offensive_layer.calculate_entry_score(trigger_events)
        self.df_indicators['entry_score'] = entry_score

        # --- 指挥链 3/7: 预警层 ---
        # 评估所有风险信号，计算 risk_score
        print("    --- [指挥链 3/7] 预警层: 正在评估所有战场风险... ---")
        risk_score, risk_details_df = self.warning_layer.calculate_risk_score()
        self.df_indicators['risk_score'] = risk_score

        # --- 指挥链 4/7: 离场层 ---
        # 在非潜在买入日，计算具体的离场信号
        print("    --- [指挥链 4/7] 离场层: 正在计算所有离场指令... ---")
        # 注意：离场层现在在判断层内部被调用，以确保决策顺序正确
        
        # --- 指挥链 5/7: 统合判断层 ---
        # 综合攻防分数，并应用绝对否决权和战略审查，做出最终决策
        print("    --- [指挥链 5/7] 统合判断层: 正在进行最终决策... ---")
        self.judgment_layer.make_final_decisions()

        # --- 指挥链 6/7: 模拟层 ---
        # 根据最终决策，进行沙盘推演
        print("    --- [指挥链 6/7] 模拟层: 正在进行全程战术推演... ---")
        self.simulation_layer.run_position_management_simulation()

        print(f"    ====== 【模块化战术引擎 V100.0】执行完毕 ======")

        # --- 指挥链 7/7: 报告层 (隐式调用) ---
        # prepare_db_records 方法现在由外部调用，使用 self.reporting_layer.prepare_db_records
        
        return self.df_indicators, score_details_df, risk_details_df

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str):
        """对外暴露的报告生成接口"""
        return self.reporting_layer.prepare_db_records(stock_code, result_df, score_details_df, risk_details_df, params, result_timeframe)

