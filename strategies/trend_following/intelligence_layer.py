# 文件: strategies/trend_following/intelligence_layer.py
# 情报层总指挥官 (重构版)
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict
from .structural_defense_layer import StructuralDefenseLayer
# --- 从新目录导入所有情报模块 ---
from .intelligence.foundation_intelligence import FoundationIntelligence
from .intelligence.structural_intelligence import StructuralIntelligence
from .intelligence.chip_intelligence import ChipIntelligence
from .intelligence.behavioral_intelligence import BehavioralIntelligence
from .intelligence.cognitive_intelligence import CognitiveIntelligence
from .intelligence.micro_behavior_engine import MicroBehaviorEngine
from .intelligence.fund_flow_intelligence import FundFlowIntelligence
from .intelligence.dynamic_mechanics_engine import DynamicMechanicsEngine
from .intelligence.cyclical_intelligence import CyclicalIntelligence
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from .intelligence.pattern_intelligence import PatternIntelligence
from .intelligence.process_intelligence import ProcessIntelligence
from .intelligence.predictive_intelligence import PredictiveIntelligence
from .intelligence.fusion_intelligence import FusionIntelligence
from strategies.trend_following.utils import get_params_block, calculate_holographic_dynamics, get_param_value, calculate_context_scores, normalize_score, normalize_to_bipolar, _calculate_gaia_bedrock_support, _calculate_historical_low_support, get_unified_score

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
        【V407.2 · 探针调度器修复版】
        - 核心修复: 补上了对主探针调度器 ForensicProbes 的导入，解决了 NameError 启动错误。
        """
        self.strategy = strategy_instance
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')
        self.strategy.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)
        self.foundation_intel = FoundationIntelligence(self.strategy)
        self.structural_intel = StructuralIntelligence(self.strategy, {})
        self.chip_intel = ChipIntelligence(self.strategy, {})
        self.behavioral_intel = BehavioralIntelligence(self.strategy)
        self.micro_behavior_engine = MicroBehaviorEngine(self.strategy)
        self.fund_flow_intel = FundFlowIntelligence(self.strategy)
        self.mechanics_engine = DynamicMechanicsEngine(self.strategy)
        self.pattern_intel = PatternIntelligence(strategy_instance)
        self.cyclical_intel = CyclicalIntelligence(self.strategy)
        self.process_intel = ProcessIntelligence(self.strategy)
        self.fusion_intel = FusionIntelligence(self.strategy)
        self.cognitive_intel = CognitiveIntelligence(self.strategy)
        self.structural_defense_layer = StructuralDefenseLayer(self.strategy)
        self.predictive_intel = PredictiveIntelligence(self.strategy)
        # 导入主探针调度器
        from .forensic_probes import ForensicProbes
        # ForensicProbes 现在会内部加载和管理所有专业探针模块
        self.probes = ForensicProbes(self)

    def run_all_diagnostics(self, df: pd.DataFrame) -> Dict:
        """
        【V424.0 · 指挥链修复版】情报层总指挥官
        - 核心重构: 彻底重组了引擎的调用顺序，以修复因执行时序错乱导致的情报真空问题。
        - 【V424.0 修复】接收 df 参数，并将其作为所有情报计算的统一数据上下文，确保索引一致性。
        """
        # df = self.strategy.df_indicators # [代码删除] 不再使用全局 df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)
        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)
        # --- 阶段一：基础原子情报层 (Foundation & Atomic Layer) ---
        update_states(self.cyclical_intel.run_cyclical_analysis_command(df))
        update_states(self.behavioral_intel.run_behavioral_analysis_command())
        update_states(self.micro_behavior_engine.run_micro_behavior_synthesis(df))
        update_states(self.foundation_intel.run_foundation_analysis_command())
        chip_states_from_intel = self.chip_intel.run_chip_intelligence_command(df)
        update_states(chip_states_from_intel)
        # --- Debugging output ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [IntelligenceLayer Debug] @ {probe_date_for_loop.date()}: atomic_states after ChipIntelligence:")
                # ... (调试代码省略)
        # --- End Debugging output ---
        update_states(self.fund_flow_intel.diagnose_fund_flow_states(df))
        update_states(self.structural_intel.diagnose_structural_states(df))
        update_states(self.mechanics_engine.run_dynamic_analysis_command())
        update_states(self.pattern_intel.run_pattern_analysis_command(df))
        # --- 阶段二：过程关系情报层 (Process & Relational Layer) ---
        update_states(self.process_intel.run_process_diagnostics(task_type_filter=None))
        # --- 阶段三：融合态势情报层 (Fusion & Situational Layer) ---
        update_states(self.fusion_intel.run_fusion_diagnostics())
        # --- 阶段四：认知推演层 (Cognitive & Playbook Layer) ---
        self._ignite_relational_dynamics_engine()
        self.cognitive_intel.synthesize_cognitive_scores(df) # [代码修改] 使用传入的 df
        # [代码修改开始] 植入指挥层探针，检查 playbook_states 的最终状态
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            print("\n" + "="*20 + f" [指挥层-真理探针] @ {probe_date_naive.date()} " + "="*20)
            print("--- [探针] 检查 run_all_diagnostics 返回前 'playbook_states' 的最终内容 ---")
            if not self.strategy.playbook_states:
                print("  -> [探针警告] 致命错误: 在情报层执行完毕后, 'playbook_states' 为空！")
            else:
                for key, signal_series in self.strategy.playbook_states.items():
                    if isinstance(signal_series, pd.Series) and probe_date_for_loop in signal_series.index:
                        raw_value = signal_series.loc[probe_date_for_loop]
                        print(f"  -> 信号: {key:<50} | 最终值: {raw_value:.4f}")
                    else:
                        print(f"  -> 信号: {key:<50} | [探针警告] 无法在探针日期找到该信号值或信号非Series类型。")
            print("="*70)
        # [代码修改结束]
        return self.strategy.atomic_states

    def deploy_forensic_probes(self):
        """
        【V2.24 · 赢家信念探针激活版】法医探针调度中心
        - 核心扩展: 新增对赢家信念探针的调用。
        """
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        if not debug_params.get('enabled', {}).get('value', False):
            return
        probe_dates_list = debug_params.get('probe_dates')
        if not probe_dates_list:
            single_date = debug_params.get('probe_date')
            if single_date:
                probe_dates_list = [single_date]
        if not probe_dates_list or not isinstance(probe_dates_list, list):
            return
        print("\n" + "="*30 + f" [法医探针部署中心 V2.24] 开始对 {len(probe_dates_list)} 个目标日期进行解剖... " + "="*30)
        for probe_date_str in probe_dates_list:
            if not probe_date_str:
                continue
            probe_date = pd.to_datetime(probe_date_str)
            if self.strategy.df_indicators.index.tz is not None:
                try:
                    probe_date = probe_date.tz_localize(self.strategy.df_indicators.index.tz)
                except Exception:
                    try:
                        probe_date = probe_date.tz_convert(self.strategy.df_indicators.index.tz)
                    except Exception as e_conv:
                         print(f"    -> [法医探针] 错误: 转换探针日期 {probe_date_str} 时区失败: {e_conv}。")
                         continue
            if probe_date not in self.strategy.df_indicators.index:
                print(f"    -> [法医探针] 警告: 探针日期 {probe_date_str} (校准后: {probe_date}) 不在数据索引中，跳过该日期。")
                continue
            print("\n" + "="*25 + f" 正在解剖 {probe_date_str} " + "="*25)
            if debug_params.get('enable_trend_quality_probe', False):
                self.probes._deploy_trend_quality_probe(probe_date)
            if debug_params.get('enable_liquidity_trap_probe', False):
                self.probes._deploy_liquidity_trap_probe(probe_date)
            if debug_params.get('enable_ff_distribution_resonance_probe', False):
                self.probes._deploy_ff_distribution_resonance_probe(probe_date)
            if debug_params.get('enable_structural_health_probe', False):
                self.probes._deploy_structural_health_probe(probe_date)
            if debug_params.get('enable_structural_pillar_fusion_probe', False):
                self.probes._deploy_structural_pillar_fusion_probe(probe_date)
            if debug_params.get('enable_structural_pillar_dissection_probe', False):
                self.probes._deploy_structural_pillar_dissection_probe(probe_date, pillar_name='structural_stability')
            if debug_params.get('enable_comprehensive_top_risk_probe', False):
                self.probes._deploy_comprehensive_top_risk_probe(probe_date)
            if debug_params.get('enable_euphoric_acceleration_probe', False):
                self.probes._deploy_euphoric_acceleration_transmutation_probe(probe_date)
            if debug_params.get('enable_chip_lockdown_probe', False):
                self.probes._deploy_bottom_accumulation_lockdown_probe(probe_date)
            if debug_params.get('enable_winner_conviction_probe', False):
                self.probes._deploy_winner_conviction_probe(probe_date)
            if debug_params.get('enable_profit_taking_pressure_probe', False):
                self.probes._deploy_profit_taking_pressure_probe(probe_date)
            if debug_params.get('enable_lockdown_scramble_probe', False):
                self.probes._deploy_lockdown_scramble_probe(probe_date)
        print("\n" + "="*35 + " [法医探针部署中心] 所有目标解剖完毕 " + "="*35 + "\n")

    def _ignite_relational_dynamics_engine(self):
        """
        【V1.0】关系动力引擎（普罗米修斯神坛）
        - 核心职责: 作为跨领域的通用“神力”引擎，计算“关系动力分”，为所有终极信号提供力量倍增。
        - 架构意义: 将此通用逻辑从行为情报模块中解放出来，提升至最高指挥部，实现架构净化。
        """
        # print("    - [神力引擎] 正在点燃“关系动力”引擎...")
        df = self.strategy.df_indicators
        power_transfer = (self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df.index)).clip(-1, 1) * 0.5 + 0.5)
        stealth_accumulation = (self.strategy.atomic_states.get('PROCESS_META_STEALTH_ACCUMULATION', pd.Series(0.0, index=df.index)).clip(-1, 1) * 0.5 + 0.5)
        winner_conviction = (self.strategy.atomic_states.get('PROCESS_META_WINNER_CONVICTION', pd.Series(0.0, index=df.index)).clip(-1, 1) * 0.5 + 0.5)
        loser_capitulation = (self.strategy.atomic_states.get('PROCESS_META_LOSER_CAPITULATION', pd.Series(0.0, index=df.index)).clip(-1, 1) * 0.5 + 0.5)
        stormborn_power = (power_transfer * loser_capitulation)**0.5
        self.strategy.atomic_states['SCORE_ATOMIC_STORM_BORN_POWER'] = stormborn_power.astype(np.float32)
        still_waters_power = (stealth_accumulation * winner_conviction)**0.5
        self.strategy.atomic_states['SCORE_ATOMIC_STILL_WATERS_POWER'] = still_waters_power.astype(np.float32)
        relational_dynamics_power = np.maximum(stormborn_power, still_waters_power)
        self.strategy.atomic_states['SCORE_ATOMIC_RELATIONAL_DYNAMICS'] = relational_dynamics_power.astype(np.float32)






