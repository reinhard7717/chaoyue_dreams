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

    def run_all_diagnostics(self, df: pd.DataFrame) -> None:
        """
        【V25.9 · 剧本调用顺序优化版】
        - 核心优化: 调整了情报模块的调用顺序，确保各层级情报的依赖关系得到满足。
        - 核心职责: 协调所有情报模块的运行，并将生成的原子状态统一存储到 `self.strategy.atomic_states`。
        - 【新增】增加探针，检查 `PROCESS_META_MAIN_FORCE_RALLY_INTENT` 在 `atomic_states` 中的值。
        """
        print("启动【V25.9 · 剧本调用顺序优化版】认知情报分析...")
        # 1. 运行基础情报模块
        foundation_states = self.foundation_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(foundation_states)
        # 2. 运行力学情报模块
        dynamic_states = self.dynamic_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(dynamic_states)
        # 3. 运行结构情报模块
        structural_states = self.structural_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(structural_states)
        # 4. 运行行为情报模块
        behavioral_states = self.behavioral_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(behavioral_states)
        # 5. 运行微观情报模块
        micro_states = self.micro_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(micro_states)
        # 6. 运行形态情报模块
        pattern_states = self.pattern_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(pattern_states)
        # 7. 运行资金流情报模块
        fund_flow_states = self.fund_flow_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(fund_flow_states)
        # 8. 运行筹码情报模块
        chip_states = self.chip_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(chip_states)
        # 9. 运行过程层情报模块 (依赖所有原子层信号)
        process_states = self.process_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(process_states)

        # [代码修改开始]
        # 增加探针，检查 PROCESS_META_MAIN_FORCE_RALLY_INTENT 在 atomic_states 中的值
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                if 'PROCESS_META_MAIN_FORCE_RALLY_INTENT' in self.strategy.atomic_states:
                    signal_series = self.strategy.atomic_states['PROCESS_META_MAIN_FORCE_RALLY_INTENT']
                    if isinstance(signal_series, pd.Series) and probe_date_for_loop in signal_series.index:
                        value_in_atomic_states = signal_series.loc[probe_date_for_loop]
                        print(f"    -> [IntelligenceLayer Debug] @ {probe_date_for_loop.date()}: PROCESS_META_MAIN_FORCE_RALLY_INTENT in atomic_states: {value_in_atomic_states:.4f}")
                    else:
                        print(f"    -> [IntelligenceLayer Debug] @ {probe_date_for_loop.date()}: PROCESS_META_MAIN_FORCE_RALLY_INTENT series not found or date not in index.")
                else:
                    print(f"    -> [IntelligenceLayer Debug] @ {probe_date_for_loop.date()}: PROCESS_META_MAIN_FORCE_RALLY_INTENT not in atomic_states.")
        # [代码修改结束]

        # 10. 运行融合层情报模块 (依赖所有原子层和过程层信号)
        fusion_states = self.fusion_intel.run_fusion_diagnostics()
        self.strategy.atomic_states.update(fusion_states)
        # 11. 运行周期情报模块
        cyclical_states = self.cyclical_intel.run_all_diagnostics(df)
        self.strategy.atomic_states.update(cyclical_states)
        # 12. 运行认知层情报模块 (依赖所有原子层、过程层和融合层信号)
        self.cognitive_intel.run_all_diagnostics(df)
        print(f"【V25.9 · 剧本调用顺序优化版】分析完成，生成 {len(self.strategy.playbook_states)} 个剧本信号并存入专属状态库。")

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






