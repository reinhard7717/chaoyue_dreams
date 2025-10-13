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
from .intelligence.playbook_engine import PlaybookEngine
from .intelligence.fund_flow_intelligence import FundFlowIntelligence
from .intelligence.dynamic_mechanics_engine import DynamicMechanicsEngine
from .intelligence.cyclical_intelligence import CyclicalIntelligence
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from .intelligence.pattern_intelligence import PatternIntelligence
from .intelligence.process_intelligence import ProcessIntelligence
from .intelligence.predictive_intelligence import PredictiveIntelligence
from strategies.trend_following.forensic_probes import ForensicProbes
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
        初始化情报层总指挥官。
        """
        self.strategy = strategy_instance
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')
        self.strategy.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)
        # 实例化所有子模块，注入依赖
        self.foundation_intel = FoundationIntelligence(self.strategy)
        self.structural_intel = StructuralIntelligence(self.strategy, {}) # dynamic_thresholds 已废弃
        self.chip_intel = ChipIntelligence(self.strategy, {}) # dynamic_thresholds 已废弃
        self.behavioral_intel = BehavioralIntelligence(self.strategy)
        self.fund_flow_intel = FundFlowIntelligence(self.strategy)
        self.mechanics_engine = DynamicMechanicsEngine(self.strategy)
        self.pattern_intel = PatternIntelligence(strategy_instance)
        self.cyclical_intel = CyclicalIntelligence(self.strategy)
        self.process_intel = ProcessIntelligence(self.strategy)
        self.cognitive_intel = CognitiveIntelligence(self.strategy)
        self.playbook_engine = PlaybookEngine(self.strategy)
        self.structural_defense_layer = StructuralDefenseLayer(self.strategy)
        # 实例化先知引擎
        self.predictive_intel = PredictiveIntelligence(self.strategy)
        # 实例化法医探针集合
        self.probes = ForensicProbes(self)

    def run_all_diagnostics(self) -> Dict:
        """
        【V415.1 · 指挥链审查版】情报层总指挥官
        - 核心升级: 部署“指挥链审查”探针，监控对认知层的调用。
        """
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)
        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)
        update_states(self.cyclical_intel.run_cyclical_analysis_command(df))
        base_process_states = self.process_intel.run_process_diagnostics(task_type_filter='base')
        update_states(base_process_states)
        self._ignite_relational_dynamics_engine()
        update_states(self.behavioral_intel.run_behavioral_analysis_command())
        update_states(self.foundation_intel.run_foundation_analysis_command())
        update_states(self.chip_intel.run_chip_intelligence_command(df))
        update_states(self.structural_intel.diagnose_structural_states(df))
        update_states(self.fund_flow_intel.diagnose_fund_flow_states(df))
        update_states(self.mechanics_engine.run_dynamic_analysis_command())
        update_states(self.pattern_intel.run_pattern_analysis_command(df))
        strategy_process_states = self.process_intel.run_process_diagnostics(task_type_filter='strategy')
        update_states(strategy_process_states)
        self.cognitive_intel.synthesize_cognitive_scores(df, pullback_enhancements={})
        update_states(self.predictive_intel.run_predictive_diagnostics())
        trigger_events = self.playbook_engine.define_trigger_events(df)
        self.strategy.trigger_events.update(trigger_events)
        _, playbook_states = self.playbook_engine.generate_playbook_states(self.strategy.trigger_events)
        self.strategy.playbook_states.update(playbook_states)
        exit_triggers_df = self.structural_defense_layer.generate_hard_exit_triggers()
        self.strategy.exit_triggers = exit_triggers_df
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        return self.strategy.trigger_events

    def deploy_forensic_probes(self):
        """
        【V2.4 · 调用链路修复版】法医探针调度中心
        - 核心修复: 修正了对 _deploy_zeus_thunderbolt_probe 的调用，移除了多余的参数，解决 TypeError 崩溃问题。
        """
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        if not debug_params.get('enabled', False):
            return
        probe_dates_list = debug_params.get('probe_dates')
        if not probe_dates_list:
            single_date = debug_params.get('probe_date')
            if single_date:
                probe_dates_list = [single_date]
        if not probe_dates_list or not isinstance(probe_dates_list, list):
            return
        print("\n" + "="*30 + f" [法医探针部署中心 V2.3] 开始对 {len(probe_dates_list)} 个目标日期进行解剖... " + "="*30)
        # [代码删除] 移除对上下文分数的预计算，因为新的探针链会自行处理
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
            # 修正调用签名，不再传递多余参数
            self._deploy_zeus_thunderbolt_probe(probe_date)
            # 自动调度“哈迪斯凝视”探针
            if probe_date_str == '2025-09-17':
                print("\n" + "="*25 + f" 检测到特定风险日期，启动哈迪斯凝视探针 " + "="*25)
                self.probes._deploy_hades_gaze_probe(probe_date, 'CHIP', 'BEARISH_RESONANCE')
                self.probes._deploy_hades_gaze_probe(probe_date, 'CHIP', 'TOP_REVERSAL')
                self.probes._deploy_hades_gaze_probe(probe_date, 'FUND_FLOW', 'BEARISH_RESONANCE')
                self.probes._deploy_hades_gaze_probe(probe_date, 'FUND_FLOW', 'TOP_REVERSAL')
        print("\n" + "="*35 + " [法医探针部署中心] 所有目标解剖完毕 " + "="*35 + "\n")

    def _ignite_relational_dynamics_engine(self):
        """
        【V1.0 · 新增】关系动力引擎（普罗米修斯神坛）
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

    # 注入全新的“宙斯之雷”终极探针
    def _deploy_zeus_thunderbolt_probe(self, probe_date: pd.Timestamp):
        """
        【V3.7.0 · 宙斯敕令版】终极得分构成解剖探针
        - 核心修正: 部署“宙斯敕令协议”，正确处理“奇美拉冲突”的双重神格。不再从风险总分中剔除奇美拉，
                      而是将其作为标准风险项计入前置总分，再应用其阻尼效果。
        - 收益: 彻底解决了探针重算与主引擎最终得分之间的偏差，完美复现最终裁决逻辑。
        """
        print(f"\n--- [探针] 正在召唤⚡️【宙斯之雷 · 终极得分解剖探针 V3.7.0】⚡️---")
        # self.probes._deploy_themis_scales_probe(probe_date)
        # self.probes._deploy_archangel_diagnosis_probe(probe_date)
        # self.probes._deploy_athena_wisdom_probe(probe_date)
        # self.probes._deploy_hephaestus_forge_probe(probe_date)
        # self.probes._deploy_hermes_caduceus_probe(probe_date)
        # self.probes._deploy_hermes_verdict_probe(probe_date)
        # self.probes._deploy_ares_spear_probe(probe_date)
        self.probes._deploy_hephaestus_chip_forge_probe(probe_date)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        print("\n  [链路层 1] 最终裁决")
        final_score = df.get('final_score', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        final_signal = df.get('signal_type', pd.Series('N/A', index=df.index)).get(probe_date, 'N/A')
        print(f"    - 【最终信号】: {final_signal}")
        if isinstance(final_score, (float, np.floating)):
            print(f"    - 【最终得分】: {final_score:.0f}")
        else:
            print(f"    - 【最终得分】: {final_score}")
        print("\n  [链路层 2] 激活的进攻项 (按贡献度排序)")
        score_details_json_str = df.get('signal_details_cn', pd.Series('{}', index=df.index)).get(probe_date, '{}')
        try:
            score_details = json.loads(score_details_json_str) if isinstance(score_details_json_str, str) else score_details_json_str
            if not isinstance(score_details, dict): score_details = {}
        except (json.JSONDecodeError, TypeError):
            score_details = {}
        offense_items = score_details.get('offense', [])
        offense_total = 0
        if offense_items:
            if not isinstance(offense_items, list): offense_items = []
            for item in sorted(offense_items, key=lambda x: x.get('score', 0), reverse=True):
                if not isinstance(item, dict): continue
                contribution = item.get('score', 0)
                raw_score = item.get('raw_score', 0)
                base_score = item.get('base_score', 0)
                item_name = item.get('name', 'N/A')
                # 增加对幻影信号的过滤，使探针更健壮
                if '筹码行为同步' in item_name or '风险' in item_name:
                    print(f"    - 【幻影/错误信号已忽略】: {item_name} (贡献: {contribution})")
                    continue
                print(f"    - 【{item_name}】: {contribution: <5.0f} (原始值: {raw_score:.4f} * 基础分: {base_score})")
                offense_total += contribution
        print("    ----------------------------------")
        print(f"    - 【进攻项总分】: {offense_total:.0f}")
        print("\n  [链路层 3] 激活的风险项 (按贡献度排序)")
        risk_items = score_details.get('risk', [])
        risk_total = 0
        if risk_items:
            if not isinstance(risk_items, list): risk_items = []
            for item in sorted(risk_items, key=lambda x: abs(x.get('score', 0)), reverse=True):
                if not isinstance(item, dict): continue
                contribution = item.get('score', 0)
                raw_score = item.get('raw_score', 0)
                base_score = item.get('base_score', 0)
                item_name = item.get('name', 'N/A')
                print(f"    - 【{item_name}】: {contribution: <5.0f} (原始值: {raw_score:.4f} * 基础分: {base_score})")
                risk_total += contribution
        print("    ----------------------------------")
        print(f"    - 【风险项总分】: {risk_total:.0f}")
        print("\n  [链路层 4] 终极对质 (宙斯最终敕令)")
        # 修改开始: 部署“宙斯敕令”协议
        # 1. 计算前置裁决分，直接将进攻总分与风险总分相加
        pre_damper_score = offense_total + risk_total
        print(f"    - [探针重算] 前置裁决分 = {offense_total:.0f} (进攻) + {risk_total:.0f} (风险) = {pre_damper_score:.0f}")
        # 2. 获取奇美拉阻尼器
        chimera_conflict_score = atomic.get('COGNITIVE_SCORE_CHIMERA_CONFLICT', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        # 3. 应用阻尼器
        final_score_recalc = pre_damper_score * (1 - chimera_conflict_score)
        print(f"    - [探针重算] 最终得分 = {pre_damper_score:.0f} * (1 - 奇美拉冲突调节器:{chimera_conflict_score:.2f}) = {final_score_recalc:.0f}")
        # 修改结束
        if isinstance(final_score, (float, np.floating)):
            print(f"    - [对比]: 实际值 {final_score:.0f} vs 重算值 {final_score_recalc:.0f}")
        else:
            print(f"    - [对比]: 实际值 {final_score} vs 重算值 {final_score_recalc:.0f}")
        print("\n--- “宙斯之雷”审查完毕 ---")





