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
from .trend_following.utils import ensure_numeric_types, optimize_df_memory, get_params_block, get_param_value

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    【V101.0 · 终极信号适配版】
    - 核心重构: 净化了 apply_strategy 的主流程，现在完全依赖 IntelligenceLayer 来完成所有情报的生成与融合。
    """
    def __init__(self, orchestrator_instance, strategy_config: dict):
        """
        【V103.0 · 主权配置协议版】
        - 初始化方法现在接收一份由总指挥分发的、纯净的专属配置 (strategy_config)。
        - 收益: 策略单元不再需要关心配置的来源和净化过程，实现了更高的内聚和独立性。
        """
        self.orchestrator = orchestrator_instance
        # 不再从总指挥处继承，而是使用注入的专属配置
        self.unified_config = strategy_config
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

    def apply_strategy(self, all_dfs: Dict[str, pd.DataFrame], start_date_str: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V416.2 · 数据完整性卫兵与进攻风险分离适配版】
        - 核心修复: 移除了上一版中错误的、基于 start_date_str 的数据切片逻辑。
        - 核心新增: 增加了“数据完整性卫兵”。在策略执行开始时，会检查传入的数据帧长度是否
                      满足配置文件中 `base_needed_bars` 的要求。如果数据长度不足，将打印
                      明确警告并提前终止，从根本上防止因上游数据供给不足导致的所有后续错误。
        - 核心适配: 调整 OffensiveLayer 的调用，以捕获并存储总进攻得分和总风险惩罚。
        - 签名修复: 修正 calculate_context_scores 的调用，传入 strategy 实例。
        """
        self.params = self.unified_config
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_indicators = self._merge_all_timeframes(all_dfs)
        fe_params = get_params_block(self, 'feature_engineering_params', {})
        required_bars = get_param_value(fe_params.get('base_needed_bars'), 250) # 默认至少需要250条
        if len(self.df_indicators) < required_bars:
            print(f"    -> [策略执行终止] 错误：数据完整性检查失败！")
            print(f"       需要至少 {required_bars} 条数据来进行指标计算，但只收到了 {len(self.df_indicators)} 条。")
            print(f"       请检查调用本策略的上层代码，确保为回测或分析提供了足够的历史回溯数据。")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # 步骤1: 情报层完成所有诊断与合成，包括专业层、融合层和认知层。这是唯一的情报生成入口。
        # IntelligenceLayer 负责填充 self.atomic_states, self.playbook_states, self.trigger_events
        self.intelligence_layer.run_all_diagnostics(self.df_indicators)
        # 步骤2: 基于完整的诊断结果，进行顶层上下文分析
        from .trend_following.utils import calculate_context_scores
        # 修正 calculate_context_scores 的调用，传入 self (strategy_instance)
        bottom_context_score, top_context_score = calculate_context_scores(self.df_indicators, self.atomic_states, self)
        self.atomic_states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] = bottom_context_score
        self.atomic_states['SCORE_CONTEXT_TOP_ZONE'] = top_context_score
        # 步骤3: 执行攻防决策与模拟
        # OffensiveLayer.calculate_entry_score 现在返回三个值：total_offensive_score, total_risk_sum, score_details_df
        total_offensive_score, total_risk_sum, score_details_df = self.offensive_layer.calculate_entry_score(
            self.trigger_events,
            bottom_context_score,
            top_context_score
        )
        self.df_indicators['entry_score'] = total_offensive_score # 存储总进攻分
        self.df_indicators['total_risk_sum'] = total_risk_sum # 存储总风险惩罚分
        risk_details_df = self.warning_layer.run_all_warnings()
        # JudgmentLayer 现在会使用 df_indicators 中的 'entry_score' 和 'total_risk_sum'
        self.judgment_layer.make_final_decisions(score_details_df, risk_details_df)
        self.simulation_layer.run_position_management_simulation()
        self.df_indicators = optimize_df_memory(self.df_indicators, verbose=False)
        if risk_details_df is None:
            risk_details_df = pd.DataFrame(index=self.df_indicators.index)
        # 将 score_details_df 存储为实例属性，以便 ReportingLayer 访问
        self.score_details_df = score_details_df 
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
