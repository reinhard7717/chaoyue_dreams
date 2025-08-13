# 文件: strategies/trend_following_strategy.py
# 版本: V100.1 - 初始化顺序与逻辑净化修复版
import logging
import pandas as pd
from typing import Tuple
import gc # [代码新增] 导入gc模块以支持内存回收

from .trend_following.intelligence_layer import IntelligenceLayer
from .trend_following.offensive_layer import OffensiveLayer
from .trend_following.warning_layer import WarningLayer
from .trend_following.exit_layer import ExitLayer
from .trend_following.judgment_layer import JudgmentLayer
from .trend_following.simulation_layer import SimulationLayer
from .trend_following.reporting_layer import ReportingLayer
from .trend_following.intelligence.dynamic_mechanics_engine import DynamicMechanicsEngine
from .trend_following.utils import ensure_numeric_types, optimize_df_memory, get_param_value, get_params_block

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    趋势跟踪策略 (V100.1 - 初始化顺序与逻辑净化修复版)
    - 核心修复: 调整了构造函数中共享状态属性的初始化顺序，确保它们在被子模块
                使用前已经被创建，从根本上解决了 AttributeError。
    - 逻辑净化: 移除了对已被重构掉的方法的调用，并清理了未使用的中间属性。
    """
    def __init__(self, config: dict):
        self.unified_config = config
        self.strategy_info = get_params_block(self, 'strategy_info')

        self.atomic_states = {}
        self.playbook_states = {}
        self.setup_scores = {}
        self.df_indicators = pd.DataFrame()

        # 初始化所有分层模块，并将主策略实例(self)传递给它们，以便共享配置和状态
        self.intelligence_layer = IntelligenceLayer(self)
        self.offensive_layer = OffensiveLayer(self)
        self.warning_layer = WarningLayer(self)
        self.exit_layer = ExitLayer(self)
        self.judgment_layer = JudgmentLayer(self)
        self.simulation_layer = SimulationLayer(self)
        self.reporting_layer = ReportingLayer(self)
        self.mechanics_engine = DynamicMechanicsEngine(self)

    def apply_strategy(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V400.1 逻辑净化版】
        - 核心重组: 调整了指挥链，以适配风险计算的统一流程。
        - 新流程: 基础情报 -> 进攻评分 -> 致命风险评估 -> 常规风险评估(并合并) -> 力学分析 -> 最终决策。
        """

        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        self.df_indicators = ensure_numeric_types(df)

        # --- 指挥链 1/8: 基础情报层 ---
        trigger_events = self.intelligence_layer.run_all_diagnostics()

        # --- 指挥链 2/8: 进攻层 ---
        entry_score, score_details_df = self.offensive_layer.calculate_entry_score(trigger_events)
        self.df_indicators['entry_score'] = entry_score

        # --- 指挥链 3/8: 离场层 (仅计算致命风险) ---
        # ExitLayer 现在只返回一个包含致命风险详情的DataFrame
        critical_risk_details_df = self.exit_layer.calculate_critical_risks()

        # --- 指挥链 4/8: 预警层 (合并所有风险) ---
        # WarningLayer 接收致命风险详情，并与常规风险合并，计算最终的总 risk_score
        risk_score, combined_risk_details_df, risk_change_summary = self.warning_layer.calculate_risk_score(
            critical_risk_details=critical_risk_details_df
        )
        self.df_indicators['risk_score'] = risk_score
        self.df_indicators['risk_change_summary'] = risk_change_summary

        # --- 指挥链 5/8: 力学分析层 ---
        self.mechanics_engine.run_force_vector_analysis()

        # --- 指挥链 6/8: 统合判断层 ---
        # JudgmentLayer 现在接收合并后的完整风险详情
        self.judgment_layer.make_final_decisions(score_details_df, combined_risk_details_df)

        # --- 指挥链 7/8 & 8/8: 模拟层与报告层 ---
        self.simulation_layer.run_position_management_simulation()
        # print(f"    ====== 【ORM适配引擎 V400.1】执行完毕 ======")
        
        # 1. 对主DataFrame进行内存压缩
        self.df_indicators = optimize_df_memory(self.df_indicators, verbose=False)
        
        # 2. 只删除那些确定不再需要的、过程中的大型变量
        try:
            del trigger_events, entry_score
            del critical_risk_details_df, risk_score, risk_change_summary
            gc.collect()
        except NameError:
            pass # 忽略可能已删除的变量

        # 返回进攻详情和合并后的完整风险详情
        return self.df_indicators, score_details_df, combined_risk_details_df

    async def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str):
        """对外暴露的报告生成接口"""
        return await self.reporting_layer.prepare_db_records(stock_code, result_df, score_details_df, risk_details_df, params, result_timeframe)
