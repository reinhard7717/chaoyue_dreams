# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict
from .structural_defense_layer import StructuralDefenseLayer
# --- 将所有情报模块和工具函数的导入整合到文件顶部 ---
from .intelligence.foundation_intelligence import FoundationIntelligence
from .intelligence.structural_intelligence import StructuralIntelligence
from .intelligence.chip_intelligence import ChipIntelligence
from .intelligence.behavioral_intelligence import BehavioralIntelligence
from .intelligence.cognitive_intelligence import CognitiveIntelligence
from .intelligence.micro_behavior_engine import MicroBehaviorEngine
from .intelligence.fund_flow_intelligence import FundFlowIntelligence
from .intelligence.dynamic_mechanics_engine import DynamicMechanicsEngine
from .intelligence.cyclical_intelligence import CyclicalIntelligence
from .intelligence.pattern_intelligence import PatternIntelligence
from .intelligence.process_intelligence import ProcessIntelligence
from .intelligence.predictive_intelligence import PredictiveIntelligence
from .intelligence.fusion_intelligence import FusionIntelligence
from .intelligence.intraday_behavior_engine import IntradayBehaviorEngine
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from strategies.trend_following.utils import (
    get_params_block, get_param_value, calculate_context_scores, 
    normalize_score, normalize_to_bipolar, _calculate_gaia_bedrock_support, 
    _calculate_historical_low_support, get_unified_score
)
from .forensic_probes import ForensicProbes

class IntelligenceLayer:
    """
    【V407.0 · 终极信号适配版】情报层总指挥官
    - 核心职责: 1. 实例化所有专业化的情报子模块。
                2. 按照“原子信号生成 -> 跨域认知融合 -> 战术剧本生成”的顺序，编排和调用这些子模块。
                3. 整合所有模块产出的原子状态和触发器，供下游层使用。
    - 全面适配所有情报引擎的“大一统”重构，确保调用流程和数据流正确无误。
    """
    def __init__(self, strategy_instance):
        """
        【V407.3 · 导入净化版】
        - 核心修复: 将所有模块的导入语句移至文件顶部，遵循Python最佳实践，
                      彻底解决因作用域问题导致的 UnboundLocalError。
        """
        self.strategy = strategy_instance
        # 移除所有在此方法内部的import语句
        # 现在所有依赖都在文件顶部导入，可以直接使用
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')
        # self.strategy.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)
        # self.foundation_intel = FoundationIntelligence(self.strategy)
        # self.structural_intel = StructuralIntelligence(self.strategy, {})
        # self.chip_intel = ChipIntelligence(self.strategy)
        # self.behavioral_intel = BehavioralIntelligence(self.strategy)
        # self.micro_behavior_engine = MicroBehaviorEngine(self.strategy)
        # self.intraday_behavior_engine = IntradayBehaviorEngine(self.strategy)
        # self.fund_flow_intel = FundFlowIntelligence(self.strategy)
        # self.mechanics_engine = DynamicMechanicsEngine(self.strategy)
        # self.pattern_intel = PatternIntelligence(strategy_instance)
        # self.cyclical_intel = CyclicalIntelligence(self.strategy)
        self.process_intel = ProcessIntelligence(self.strategy)
        self.fusion_intel = FusionIntelligence(self.strategy)
        self.cognitive_intel = CognitiveIntelligence(self.strategy)
        self.structural_defense_layer = StructuralDefenseLayer(self.strategy)
        self.predictive_intel = PredictiveIntelligence(self.strategy)
        # self.probes = ForensicProbes(self)

    def run_all_diagnostics(self, df: pd.DataFrame) -> Dict:
        """
        【V426.5 · 命名规范化版】
        - 核心重构: 移除了具有宗教和神话色彩的内部命名，例如“神力引擎”、“普罗米修斯神坛”。
        """
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)
        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)
        update_states(self.process_intel.run_process_diagnostics(df, task_type_filter=None))
        self._calculate_relational_dynamics_power()
        return self.strategy.atomic_states

    def _calculate_relational_dynamics_power(self):
        """
        【V1.1 · 关系动力引擎规范版】
        - 核心职责: 计算关系动力分。已废弃 stormborn (风暴降生) 等神话命名，采用标准化金融动能描述。
        """
        df = self.strategy.df_indicators
        power_transfer = (self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df.index)).clip(-1, 1) * 0.5 + 0.5)
        stealth_accumulation = (self.strategy.atomic_states.get('PROCESS_META_STEALTH_ACCUMULATION', pd.Series(0.0, index=df.index)).clip(-1, 1) * 0.5 + 0.5)
        winner_conviction = (self.strategy.atomic_states.get('PROCESS_META_WINNER_CONVICTION', pd.Series(0.0, index=df.index)).clip(-1, 1) * 0.5 + 0.5)
        loser_capitulation = (self.strategy.atomic_states.get('PROCESS_META_LOSER_CAPITULATION', pd.Series(0.0, index=df.index)).clip(-1, 1) * 0.5 + 0.5)
        volatility_breakout_power = (power_transfer * loser_capitulation)**0.5
        self.strategy.atomic_states['SCORE_ATOMIC_VOLATILITY_BREAKOUT_POWER'] = volatility_breakout_power.astype(np.float32)
        stealth_accumulation_power = (stealth_accumulation * winner_conviction)**0.5
        self.strategy.atomic_states['SCORE_ATOMIC_STEALTH_ACCUMULATION_POWER'] = stealth_accumulation_power.astype(np.float32)
        relational_dynamics_power = np.maximum(volatility_breakout_power, stealth_accumulation_power)
        self.strategy.atomic_states['SCORE_ATOMIC_RELATIONAL_DYNAMICS'] = relational_dynamics_power.astype(np.float32)

    def deploy_forensic_probes(self):
        """
        【V2.25 · 赢家信念探针激活版 & 调试标志精简版】法医探针调度中心
        - 核心扩展: 新增对赢家信念探针的调用。
        - **新增业务逻辑：移除调试标志的设置逻辑，专注于探针报告的调度和打印。**
        """
        # 调试标志已在 IntelligenceLayer.run_all_diagnostics 中前置设置
        # if not self.probes.should_probe:
        #     return
        # probe_dates_list = list(self.probes.probe_dates_set) # 从已设置的集合中获取日期列表
        # if not probe_dates_list:
        #     return
        # print("\n" + "="*30 + f" [法医探针部署中心 V2.25] 开始对 {len(probe_dates_list)} 个目标日期进行解剖... " + "="*30)
        # debug_params = get_params_block(self.strategy, 'debug_params', {}) # 仍然需要获取debug_params来判断具体探针是否启用
        # for probe_date_date in probe_dates_list:
        #     probe_date = pd.Timestamp(probe_date_date) # 转换为Timestamp以便与df_indicators索引匹配
        #     if self.strategy.df_indicators.index.tz is not None:
        #         try:
        #             probe_date = probe_date.tz_localize(self.strategy.df_indicators.index.tz)
        #         except Exception:
        #             try:
        #                 probe_date = probe_date.tz_convert(self.strategy.df_indicators.index.tz)
        #             except Exception as e_conv:
        #                  print(f"    -> [法医探针] 错误: 转换探针日期 {probe_date_date} 时区失败: {e_conv}。")
        #                  continue
        #     if probe_date not in self.strategy.df_indicators.index:
        #         print(f"    -> [法医探针] 警告: 探针日期 {probe_date_date} (校准后: {probe_date}) 不在数据索引中，跳过该日期。")
        #         continue
        #     print("\n" + "="*25 + f" 正在解剖 {probe_date_date.strftime('%Y-%m-%d')} " + "="*25)
        #     if debug_params.get('enable_trend_quality_probe', False):
        #         self.probes._deploy_trend_quality_probe(probe_date)
        #     if debug_params.get('enable_liquidity_trap_probe', False):
        #         self.probes._deploy_liquidity_trap_probe(probe_date)
        #     if debug_params.get('enable_ff_distribution_resonance_probe', False):
        #         self.probes._deploy_ff_distribution_resonance_probe(probe_date)
        #     if debug_params.get('enable_structural_health_probe', False):
        #         self.probes._deploy_structural_health_probe(probe_date)
        #     if debug_params.get('enable_structural_pillar_fusion_probe', False):
        #         self.probes._deploy_structural_pillar_fusion_probe(probe_date)
        #     if debug_params.get('enable_structural_pillar_dissection_probe', False):
        #         self.probes._deploy_structural_pillar_dissection_probe(probe_date, pillar_name='structural_stability')
        #     if debug_params.get('enable_comprehensive_top_risk_probe', False):
        #         self.probes._deploy_comprehensive_top_risk_probe(probe_date)
        #     if debug_params.get('enable_euphoric_acceleration_probe', False):
        #         self.probes._deploy_euphoric_acceleration_transmutation_probe(probe_date)
        #     if debug_params.get('enable_chip_lockdown_probe', False):
        #         self.probes._deploy_bottom_accumulation_lockdown_probe(probe_date)
        #     if debug_params.get('enable_winner_conviction_probe', False):
        #         self.probes._deploy_winner_conviction_probe(probe_date)
        #     if debug_params.get('enable_profit_taking_pressure_probe', False):
        #         self.probes._deploy_profit_taking_pressure_probe(probe_date)
        #     if debug_params.get('enable_lockdown_scramble_probe', False):
        #         self.probes._deploy_lockdown_scramble_probe(probe_date)
        # print("\n" + "="*35 + " [法医探针部署中心] 所有目标解剖完毕 " + "="*35 + "\n")
        return






