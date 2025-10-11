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
from strategies.trend_following.utils import get_params_block, get_param_value, calculate_context_scores, normalize_score, normalize_to_bipolar, _calculate_gaia_bedrock_support, _calculate_historical_low_support, get_unified_score

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
                self._deploy_hades_gaze_probe(probe_date, 'CHIP', 'BEARISH_RESONANCE')
                self._deploy_hades_gaze_probe(probe_date, 'CHIP', 'TOP_REVERSAL')
                self._deploy_hades_gaze_probe(probe_date, 'FUND_FLOW', 'BEARISH_RESONANCE')
                self._deploy_hades_gaze_probe(probe_date, 'FUND_FLOW', 'TOP_REVERSAL')
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

    def _deploy_prophet_entry_probe(self, probe_date: pd.Timestamp):
        """
        【V2.5 · 赫淮斯托斯重铸协议同步版】“先知入场神谕”专属法医探针
        - 核心革命: 探针的重算逻辑已与主引擎的“赫淮斯托斯重铸协议”版本完全同步。
        - 新核心公式: intraday_low_pct_change 被 clip(upper=0)，确保只有真下跌才会计分。
        - 收益: 确保探针能够正确解剖和验证“只买真恐慌”的最终哲学。
        """
        print("\n--- [探针] 正在解剖: 【神谕 · 先知入场】 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators

        def get_val(name, date, default=np.nan):
            series = atomic.get(name)
            if series is None: return default
            return series.get(date, default)

        print("\n  [链路层 1] 解剖 -> 最终预测机会 (PREDICTIVE_OPP_CAPITULATION_REVERSAL)")
        final_opp_score = get_val('PREDICTIVE_OPP_CAPITULATION_REVERSAL', probe_date, 0.0)
        print(f"    - 【最终预测值】: {final_opp_score:.4f}")
        print(f"    - [核心公式]: 预测机会 = 恐慌战备分 (SCORE_SETUP_PANIC_SELLING)")

        print("\n  [链路层 2] 解剖 -> 核心输入: 恐慌战备分 (SCORE_SETUP_PANIC_SELLING)")
        panic_setup_score = get_val('SCORE_SETUP_PANIC_SELLING', probe_date, 0.0)
        print(f"    - 【恐慌战备分】: {panic_setup_score:.4f}")
        print(f"    - [核心公式]: (六大支柱加权和 * 静谧度 * 反弹强度) * 赫尔墨斯调节器 (当满足价格暴跌门槛时)")

        print("\n  [链路层 3] 钻透 -> 六大支柱 & 调节器")
        
        p_panic = get_params_block(self.strategy, 'panic_selling_setup_params', {})
        pillar_weights = get_param_value(p_panic.get('pillar_weights'), {})
        min_price_drop_pct = get_param_value(p_panic.get('min_price_drop_pct'), -0.025)

        # 同步“赫淮斯托斯重铸协议”
        intraday_low_pct_change_raw = (df.at[probe_date, 'low_D'] - df.at[probe_date, 'pre_close_D']) / df.at[probe_date, 'pre_close_D'] if df.at[probe_date, 'pre_close_D'] > 0 else 0.0
        intraday_low_pct_change_series = ((df['low_D'] - df['pre_close_D']) / df['pre_close_D'].replace(0, np.nan)).clip(upper=0)
        
        price_drop_score_recalc = normalize_score(intraday_low_pct_change_series, df.index, window=60, ascending=False).get(probe_date, 0.0)
        print(f"    --- 支柱一: 价格暴跌 (权重: {pillar_weights.get('price_drop', 0):.2f}) ---")
        print(f"      - 当日盘中最大跌幅 (原始值): {intraday_low_pct_change_raw:.2%}")
        print(f"      - [探针重算] 价格暴跌分 (经clip修正): {price_drop_score_recalc:.4f}")

        # ... 其他支柱和调节器的解剖逻辑保持不变 ...
        volume_spike_score_recalc = normalize_score(df['volume_D'] / df['VOL_MA_21_D'], df.index, window=60, ascending=True).get(probe_date, 0.0)
        print(f"    --- 支柱二: 成交天量 (权重: {pillar_weights.get('volume_spike', 0):.2f}) ---")
        print(f"      - [探针重算] 成交天量分: {volume_spike_score_recalc:.4f}")
        from .utils import get_unified_score
        chip_breakdown_score_recalc = get_unified_score(atomic, df.index, 'CHIP_BEARISH_RESONANCE').get(probe_date, 0.0)
        chip_integrity_score_recalc = 1.0 - chip_breakdown_score_recalc
        print(f"    --- 支柱三: 结构完整度 (权重: {pillar_weights.get('chip_integrity', 0):.2f}) ---")
        print(f"      - [探针重算] 结构完整度分: {chip_integrity_score_recalc:.4f}")
        tactic_engine_probe = self.cognitive_intel.tactic_engine
        despair_context_score_recalc = tactic_engine_probe._calculate_despair_context_score(df, p_panic).get(probe_date, 0.0)
        print(f"    --- 支柱四: 绝望背景 (权重: {pillar_weights.get('despair_context', 0):.2f}) ---")
        print(f"      - [探针重算] 绝望背景分: {despair_context_score_recalc:.4f}")
        structural_test_score_recalc = tactic_engine_probe.calculate_structural_test_score(df, p_panic).get(probe_date, 0.0)
        print(f"    --- 支柱五: 结构支撑测试 (权重: {pillar_weights.get('structural_test', 0):.2f}) ---")
        print(f"      - [探针重算] 结构支撑测试分: {structural_test_score_recalc:.4f}")
        ma_structure_score_recalc = tactic_engine_probe._calculate_ma_trend_context(df, [5, 13, 21, 55]).get(probe_date, 0.5)
        print(f"    --- 支柱六: 均线结构 (权重: {pillar_weights.get('ma_structure', 0):.2f}) ---")
        print(f"      - [探针重算] 均线结构分: {ma_structure_score_recalc:.4f}")
        print(f"    --- 调节器 I: 成交量静谧度 ---")
        logic_params = get_param_value(p_panic.get('volume_calmness_logic'), {})
        lifeline_ma_period = get_param_value(logic_params.get('lifeline_ma_period'), 5)
        lifeline_base_score = get_param_value(logic_params.get('lifeline_base_score'), 1.0)
        p_depth_bonus = get_param_value(logic_params.get('absolute_depth_bonus'), {})
        p_gradient_bonus = get_param_value(logic_params.get('structural_gradient_bonus'), {})
        raw_volume_calmness_score_recalc = 0.0
        lifeline_ma_col = f'VOL_MA_{lifeline_ma_period}_D'
        if lifeline_ma_col in df.columns and df.at[probe_date, 'volume_D'] < df.at[probe_date, lifeline_ma_col]:
            raw_volume_calmness_score_recalc = lifeline_base_score
            for p_str, weight in p_depth_bonus.items():
                ma_col = f'VOL_MA_{p_str}_D'
                if ma_col in df.columns and df.at[probe_date, 'volume_D'] < df.at[probe_date, ma_col]:
                    raw_volume_calmness_score_recalc += weight
            if get_param_value(p_gradient_bonus.get('enabled'), False):
                level_weights = get_param_value(p_gradient_bonus.get('level_weights'), {})
                ma5, ma13, ma21, ma55 = df.get(f'VOL_MA_5_D').at[probe_date], df.get(f'VOL_MA_13_D').at[probe_date], df.get(f'VOL_MA_21_D').at[probe_date], df.get(f'VOL_MA_55_D').at[probe_date]
                is_level_1, is_level_2, is_level_3 = (ma5 < ma13), (ma5 < ma13) and (ma13 < ma21), (ma5 < ma13) and (ma13 < ma21) and (ma21 < ma55)
                if is_level_1: raw_volume_calmness_score_recalc += level_weights.get('level_1', 0.0)
                if is_level_2: raw_volume_calmness_score_recalc += level_weights.get('level_2', 0.0)
                if is_level_3: raw_volume_calmness_score_recalc += level_weights.get('level_3', 0.0)
        final_calmness_score_recalc = raw_volume_calmness_score_recalc
        print(f"      - [探针重算] 最终静谧度分: {final_calmness_score_recalc:.4f}")
        print(f"    --- 调节器 II: 反弹强度 ---")
        day_range_raw = df.at[probe_date, 'high_D'] - df.at[probe_date, 'low_D']
        rebound_strength_score_recalc = ((df.at[probe_date, 'close_D'] - df.at[probe_date, 'low_D']) / day_range_raw) if day_range_raw > 0 else 0.5
        print(f"      - [探针重算] 反弹强度分: {rebound_strength_score_recalc:.4f}")
        print(f"    --- 调节器 III: 赫尔墨斯信使 (日内博弈) ---")
        upper_shadow_raw = df.at[probe_date, 'high_D'] - max(df.at[probe_date, 'open_D'], df.at[probe_date, 'close_D'])
        lower_shadow_raw = min(df.at[probe_date, 'open_D'], df.at[probe_date, 'close_D']) - df.at[probe_date, 'low_D']
        hermes_score_raw = ((lower_shadow_raw - upper_shadow_raw) / day_range_raw) if day_range_raw > 0 else 0.0
        hermes_regulator_recalc = (hermes_score_raw + 1) / 2.0
        print(f"      - [探针重算] 赫尔墨斯调节器: {hermes_regulator_recalc:.4f}")

        print("\n  [链路层 4] 最终验证")
        snapshot_panic_recalc = (
            price_drop_score_recalc * pillar_weights.get('price_drop', 0) +
            volume_spike_score_recalc * pillar_weights.get('volume_spike', 0) +
            chip_integrity_score_recalc * pillar_weights.get('chip_integrity', 0) +
            despair_context_score_recalc * pillar_weights.get('despair_context', 0) +
            structural_test_score_recalc * pillar_weights.get('structural_test', 0) +
            ma_structure_score_recalc * pillar_weights.get('ma_structure', 0)
        )
        print(f"    - [探针重算] 六大支柱加权和 (瞬时恐慌快照分): {snapshot_panic_recalc:.4f}")
        
        is_significant_drop = intraday_low_pct_change_raw < min_price_drop_pct
        print(f"    - [探针检查] 价格暴跌门槛 ({min_price_drop_pct:.2%}) 是否满足? {'✅ 是' if is_significant_drop else '❌ 否'}")

        base_recalculated_score = snapshot_panic_recalc * final_calmness_score_recalc * rebound_strength_score_recalc
        final_recalculated_score = base_recalculated_score * hermes_regulator_recalc if is_significant_drop else 0
        
        print(f"    - [探针重算恐慌战备分]: ({snapshot_panic_recalc:.4f} * {final_calmness_score_recalc:.4f} * {rebound_strength_score_recalc:.4f}) * {hermes_regulator_recalc:.4f} = {final_recalculated_score:.4f}")
        print(f"    - [对比]: 实际值 {panic_setup_score:.4f} vs 重算值 {final_recalculated_score:.4f}")
        print("--- 先知入场神谕探针解剖完毕 ---")

    # 部署“德尔菲神谕-离场探针协议”
    def _deploy_prophet_exit_probe(self, probe_date: pd.Timestamp):
        """
        【V1.0 · 新增】“德尔菲神谕-离场探针”
        - 核心职责: 钻透式解剖“高潮衰竭”风险信号 (PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION) 及其警报触发逻辑。
        """
        print("\n--- [探针] 正在解剖: 【神谕 · 先知离场】 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators

        def get_val(name, date, default=np.nan):
            series = atomic.get(name)
            if series is None: return default
            return series.get(date, default)

        print("\n  [链路层 1] 解剖 -> 最终预测风险 (PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION)")
        final_risk_score = get_val('PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION', probe_date, 0.0)
        print(f"    - 【最终风险值】: {final_risk_score:.4f}")
        print(f"    - [核心公式]: (亢奋分 * 天量分 * K线疲弱分) ^ (1/3)")

        print("\n  [链路层 2] 钻透 -> 风险三位一体")
        p_pred = get_params_block(self.strategy, 'predictive_intelligence_params', {})
        fusion_weights = get_param_value(p_pred.get('trinity_fusion_weights'), {})

        # 2.1 亢奋分
        euphoria_score_recalc = get_val('COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION', probe_date, 0.0)
        print(f"    --- 支柱一: 亢奋分 (权重: {fusion_weights.get('euphoria', 0):.2f}) ---")
        print(f"      - [探针获取] 亢奋分: {euphoria_score_recalc:.4f}")

        # 2.2 天量分
        vol_spike_series = df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)
        volume_spike_score_recalc = normalize_score(vol_spike_series, df.index, window=60, ascending=True).get(probe_date, 0.0)
        print(f"    --- 支柱二: 天量分 (权重: {fusion_weights.get('volume', 0):.2f}) ---")
        print(f"      - [探针重算] 天量分: {volume_spike_score_recalc:.4f}")

        # 2.3 K线疲弱分
        day_range = df.at[probe_date, 'high_D'] - df.at[probe_date, 'low_D']
        upper_shadow = df.at[probe_date, 'high_D'] - max(df.at[probe_date, 'open_D'], df.at[probe_date, 'close_D'])
        upper_shadow_ratio = (upper_shadow / day_range) if day_range > 0 else 0.0
        is_negative_close = df.at[probe_date, 'close_D'] < df.at[probe_date, 'open_D']
        kline_weakness_score_recalc = upper_shadow_ratio * float(is_negative_close)
        print(f"    --- 支柱三: K线疲弱分 (权重: {fusion_weights.get('kline', 0):.2f}) ---")
        print(f"      - [探针重算] K线疲弱分: {kline_weakness_score_recalc:.4f} (上影线率: {upper_shadow_ratio:.2f}, 是否阴线: {is_negative_close})")

        print("\n  [链路层 3] 最终验证 -> 风险融合")
        recalculated_risk_score = (
            (euphoria_score_recalc ** fusion_weights.get('euphoria', 0.33)) *
            (volume_spike_score_recalc ** fusion_weights.get('volume', 0.33)) *
            (kline_weakness_score_recalc ** fusion_weights.get('kline', 0.33))
        )
        print(f"    - [探针重算风险值]: ({euphoria_score_recalc:.4f}^{fusion_weights.get('euphoria', 0.33):.2f} * {volume_spike_score_recalc:.4f}^{fusion_weights.get('volume', 0.33):.2f} * {kline_weakness_score_recalc:.4f}^{fusion_weights.get('kline', 0.33):.2f}) = {recalculated_risk_score:.4f}")
        print(f"    - [对比]: 实际值 {final_risk_score:.4f} vs 重算值 {recalculated_risk_score:.4f}")

        print("\n  [链路层 4] 最终验证 -> 警报触发逻辑")
        p_judge = get_params_block(self.strategy, 'judgment_params', {})
        prophet_threshold = get_param_value(p_judge.get('prophet_alert_threshold'), 0.7)
        is_uptrend_context = df.at[probe_date, 'close_D'] > df.at[probe_date, 'EMA_5_D']
        
        alert_level = get_val('ALERT_LEVEL', probe_date, 0)
        alert_reason = get_val('ALERT_REASON', probe_date, '')

        print(f"    - [条件一] 风险分 > 阈值?  ({final_risk_score:.4f} > {prophet_threshold}) -> {'✅ 是' if final_risk_score > prophet_threshold else '❌ 否'}")
        print(f"    - [条件二] 处于上升趋势 (close > EMA5)? -> {'✅ 是' if is_uptrend_context else '❌ 否'}")
        
        recalculated_alert = (final_risk_score > prophet_threshold) and is_uptrend_context
        print(f"    - [探针重算警报]: {'触发' if recalculated_alert else '不触发'}")
        print(f"    - [实际警报]: Level {alert_level} ({alert_reason})")
        print("--- 先知离场神谕探针解剖完毕 ---")

    def _deploy_hephaestus_forge_probe(self, probe_date: pd.Timestamp):
        """
        【V2.1 · 赫尔墨斯信使协议版】“赫淮斯托斯熔炉”探针
        - 核心修复: 探针的数据源已修正为 COGNITIVE_INTERNAL_RISK_SNAPSHOT，确保能获取到
                      主引擎发布的、经“神盾协议”调节后的风险快照分，从而进行正确的动态锻造重算。
        - 收益: 彻底解决了探针与主引擎在风险计算上逻辑脱节的问题。
        """
        print("\n--- [探针] 正在启用: 🔥【赫淮斯托斯熔炉 · 风险融合解剖 V2.1】🔥 ---") # 修改: 更新探针版本
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        p_fused_risk = get_params_block(self.strategy, 'fused_risk_scoring')
        if not get_param_value(p_fused_risk.get('enabled'), True):
            print("    - 风险融合模块未启用，跳过解剖。")
            return
        print("  --- [阶段1] 信号输入审查 ---")
        risk_categories = p_fused_risk.get('risk_categories', {})
        all_required_signals = {s for signals in risk_categories.values() if isinstance(signals, dict) for s in signals if s != "说明"}
        signal_value_cache = {}
        for sig_name in sorted(list(all_required_signals)):
            val = atomic.get(sig_name, pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
            signal_value_cache[sig_name] = val
            if "ARCHANGEL" in sig_name:
                print(f"    - 关键输入信号 [{sig_name}]: {val:.4f}  <-- 风险源头")
            else:
                print(f"    - 输入信号 [{sig_name}]: {val:.4f}")
        print("\n  --- [阶段2] 维度内融合解剖 (修正为 max() 算法) ---")
        fused_dimension_scores = {}
        for category_name, signals in risk_categories.items():
            if category_name == "说明": continue
            print(f"\n    -> 正在处理维度: [{category_name.upper()}]")
            category_signal_scores = []
            for signal_name, signal_params in signals.items():
                if signal_name == "说明": continue
                atomic_score = signal_value_cache.get(signal_name, 0.0)
                processed_score = 1.0 - atomic_score if signal_params.get('inverse', False) else atomic_score
                weighted_score = processed_score * signal_params.get('weight', 1.0)
                category_signal_scores.append(weighted_score)
                print(f"      - 信号 '{signal_name}':")
                print(f"        - 原始值: {atomic_score:.4f} -> 处理后: {processed_score:.4f} -> 加权后: {weighted_score:.4f} (权重: {signal_params.get('weight', 1.0)})")
            if category_signal_scores:
                dimension_risk = max(category_signal_scores)
                fused_dimension_scores[category_name] = dimension_risk
                print(f"      - 维度内融合计算 (max() 算法):")
                print(f"        - 维度总风险 = max({[f'{s:.4f}' for s in category_signal_scores]}) = {dimension_risk:.4f}")
            else:
                fused_dimension_scores[category_name] = 0.0
                print(f"      - 维度内无信号，总风险为 0.0")
        print("\n  --- [阶段3] 跨维度融合解剖 ---")
        valid_scores = list(fused_dimension_scores.values())
        if valid_scores:
            total_fused_risk = max(valid_scores)
            print(f"    - 所有维度风险分: { {k: f'{v:.4f}' for k, v in fused_dimension_scores.items()} }")
            print(f"    - 裁决: 取最大值 -> {total_fused_risk:.4f}")
        else:
            total_fused_risk = 0.0
            print(f"    - 无有效维度风险分，总分为 0.0")
        print("\n  --- [阶段4] 共振惩罚解剖 ---")
        p_resonance = p_fused_risk.get('resonance_penalty_params', {})
        if get_param_value(p_resonance.get('enabled'), True):
            core_dims = p_resonance.get('core_risk_dimensions', [])
            min_dims = p_resonance.get('min_dimensions_for_resonance', 2)
            threshold = p_resonance.get('risk_score_threshold', 0.6)
            penalty_multiplier = p_resonance.get('penalty_multiplier', 1.2)
            high_risk_dimension_count = sum(1 for dim in core_dims if fused_dimension_scores.get(dim, 0.0) > threshold)
            is_resonance_triggered = (high_risk_dimension_count >= min_dims)
            print(f"    - 共振诊断: {high_risk_dimension_count}个核心维度 > {threshold} (要求: {min_dims}个) -> 触发: {is_resonance_triggered}")
            if is_resonance_triggered:
                final_risk_score = total_fused_risk * penalty_multiplier
                print(f"    - 共振惩罚: {total_fused_risk:.4f} * {penalty_multiplier} = {final_risk_score:.4f}")
            else:
                final_risk_score = total_fused_risk
                print(f"    - 未触发共振惩罚。")
        else:
            final_risk_score = total_fused_risk
            print(f"    - 共振惩罚模块未启用。")
        print("\n  --- [阶段5] 神盾协议 & 动态锻造解剖 ---")
        trend_quality_score = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        healthy_pullback_score = atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        aegis_shield_strength = max(trend_quality_score, healthy_pullback_score)
        suppression_factor = 1.0 - aegis_shield_strength
        risk_snapshot_score_recalc = final_risk_score * suppression_factor
        print(f"    - 神盾强度 (Aegis Strength): max(趋势质量:{trend_quality_score:.2f}, 健康回踩:{healthy_pullback_score:.2f}) = {aegis_shield_strength:.4f}")
        print(f"    - 风险抑制因子 (Suppression): 1.0 - {aegis_shield_strength:.2f} = {suppression_factor:.4f}")
        print(f"    - 风险快照分 (经神盾调节): {final_risk_score:.4f} * {suppression_factor:.2f} = {risk_snapshot_score_recalc:.4f}")
        # 修改开始: 从正确的信号源读取风险快照序列
        risk_snapshot_series = atomic.get('COGNITIVE_INTERNAL_RISK_SNAPSHOT', pd.Series(0.0, index=df.index))
        # 修改结束
        p_meta_cog = get_params_block(self.strategy, 'cognitive_intelligence_params', {}).get('relational_meta_analysis_params', {})
        w_state = get_param_value(p_meta_cog.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta_cog.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta_cog.get('acceleration_weight'), 0.4)
        state_val = risk_snapshot_series.clip(0, 2.0).get(probe_date, 0.0) # 注意clip上限为2.0
        vel_series = normalize_to_bipolar(risk_snapshot_series.diff(5).fillna(0), df.index, 55)
        vel_val = vel_series.get(probe_date, 0.0)
        accel_series = normalize_to_bipolar(vel_series.diff(5).fillna(0), df.index, 55) # 应该是 vel_series.diff()
        accel_val = accel_series.get(probe_date, 0.0)
        final_dynamic_risk_recalc = (state_val * w_state + vel_val * w_velocity + accel_val * w_acceleration).clip(0, 1)
        final_dynamic_risk_actual = atomic.get('COGNITIVE_FUSED_RISK_SCORE', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        print(f"    - 动态锻造: State({state_val:.2f})*w + Velocity({vel_val:.2f})*w + Accel({accel_val:.2f})*w = {final_dynamic_risk_recalc:.4f}")
        print(f"\n  --- [最终裁决] ---")
        print(f"    - 🔥 熔炉产物 (COGNITIVE_FUSED_RISK_SCORE): {final_dynamic_risk_actual:.4f}")
        print(f"    - [对比]: 实际值 {final_dynamic_risk_actual:.4f} vs 重算值 {final_dynamic_risk_recalc:.4f}")
        print("--- “赫淮斯托斯熔炉”探针运行完毕 ---\n")

    # 注入全新的“宙斯之雷”终极探针
    def _deploy_zeus_thunderbolt_probe(self, probe_date: pd.Timestamp):
        """
        【V2.5 · 信号细节修复版】终极得分构成解剖探针
        - 核心修复: 修正了从 df_indicators['signal_details_cn'] 中解析进攻项和风险项的逻辑，确保探针能正确显示所有得分细节。
        """
        print(f"\n--- [探针] 正在召唤⚡️【宙斯之雷 · 终极得分解剖探针⚡️⚡️】---")
        self._deploy_themis_scales_probe(probe_date)
        self._deploy_archangel_diagnosis_probe(probe_date)
        # 新增开始
        self._deploy_athena_wisdom_probe(probe_date)
        # 新增结束
        self._deploy_hephaestus_forge_probe(probe_date)
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
        # 修正了从 df_indicators 中解析信号细节的逻辑
        score_details_json_str = df.get('signal_details_cn', pd.Series('{}', index=df.index)).get(probe_date, '{}')
        try:
            # 检查是否已经是字典，如果不是（是字符串），则加载
            score_details = json.loads(score_details_json_str) if isinstance(score_details_json_str, str) else score_details_json_str
            if not isinstance(score_details, dict): score_details = {} # 防御性编程
        except (json.JSONDecodeError, TypeError):
            score_details = {}
        offense_items = score_details.get('offense', [])
        offense_total = 0
        if offense_items:
            # 确保 offense_items 是一个列表
            if not isinstance(offense_items, list): offense_items = []
            for item in sorted(offense_items, key=lambda x: x.get('score', 0), reverse=True):
                if not isinstance(item, dict): continue # 防御性编程
                contribution = item.get('score', 0) # 直接使用 'score' 字段，它已经是贡献度
                raw_score = item.get('raw_score', 0)
                base_score = item.get('base_score', 0)
                print(f"    - 【{item.get('name', 'N/A')}】: {contribution:.0f}  (原始值: {raw_score:.4f} * 基础分: {base_score})")
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
                print(f"    - 【{item.get('name', 'N/A')}】: {contribution:.0f}  (原始值: {raw_score:.4f} * 基础分: {base_score})")
                risk_total += contribution
        print("    ----------------------------------")
        print(f"    - 【风险项总分】: {risk_total:.0f}")
        print("\n  [链路层 4] 终极对质")
        entry_score_recalc = offense_total + risk_total
        chimera_conflict_score = atomic.get('COGNITIVE_SCORE_CHIMERA_CONFLICT', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        final_score_recalc = entry_score_recalc * (1 - chimera_conflict_score)
        print(f"    - [探针重算入场分(entry_score)]: {offense_total:.0f} (进攻) + {risk_total:.0f} (风险) = {entry_score_recalc:.0f}")
        print(f"    - [探针重算最终分(final_score)]: {entry_score_recalc:.0f} * (1 - 奇美拉冲突:{chimera_conflict_score:.2f}) = {final_score_recalc:.0f}")
        if isinstance(final_score, (float, np.floating)):
            print(f"    - [对比]: 实际值 {final_score:.0f} vs 重算值 {final_score_recalc:.0f}")
        else:
            print(f"    - [对比]: 实际值 {final_score} vs 重算值 {final_score_recalc:.0f}")
        print("\n--- “宙斯之雷”审查完毕 ---")

    def _deploy_themis_scales_probe(self, probe_date: pd.Timestamp):
        """
        【V1.26 · 神盾协议探针版】“忒弥斯天平”上下文解剖探针
        - 核心升级: 历史高点显微镜现在调用通用的阿波罗之箭探针，并展示“质量x重要性”的融合逻辑。
        """
        print("\n--- [探针] 正在启用: ⚖️【忒弥斯天平 · 上下文解剖】⚖️ ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        strategy_instance_ref = self.strategy
        p_synthesis = get_params_block(strategy_instance_ref, 'ultimate_signal_synthesis_params', {})
        gaia_params = get_param_value(p_synthesis.get('gaia_bedrock_params'), {})
        cooldown_reset_volume_ma_period = get_param_value(gaia_params.get('cooldown_reset_volume_ma_period'), 55)
        ares_vol_ma_col = 'VOL_MA_5_D'
        cooldown_vol_ma_col = f'VOL_MA_{cooldown_reset_volume_ma_period}_D'
        print("\n  --- [结构性支撑审查] 关键均线系统快照 ---")
        ma_periods_to_probe = [5, 55, 144, 233, 377]
        close_price = df.get('close_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        open_price = df.get('open_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        low_price = df.get('low_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        high_price = df.get('high_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        volume = df.get('volume_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        ares_volume_ma = df.get(ares_vol_ma_col, pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        cooldown_volume_ma = df.get(cooldown_vol_ma_col, pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        if isinstance(close_price, (float, np.floating)):
            print(f"    - {'high_D':<12}: {high_price:.2f}  (当日最高价)")
            print(f"    - {'open_D':<12}: {open_price:.2f}  (当日开盘价)")
            print(f"    - {'close_D':<12}: {close_price:.2f}  (当日收盘价)")
            print(f"    - {'low_D':<12}: {low_price:.2f}  (当日最低价)")
            print(f"    - {'volume_D':<12}: {volume:,.0f}  (当日成交量)")
            print(f"    - {ares_vol_ma_col:<12}: {ares_volume_ma:,.0f}  (阿瑞斯之矛-成交量均线)")
            print(f"    - {cooldown_vol_ma_col:<12}: {cooldown_volume_ma:,.0f}  (冷却重置-成交量均线)")
        else:
            print(f"    - {'high_D':<12}: {high_price}")
            print(f"    - {'open_D':<12}: {open_price}")
            print(f"    - {'close_D':<12}: {close_price}")
            print(f"    - {'low_D':<12}: {low_price}")
            print(f"    - {'volume_D':<12}: {volume}")
            print(f"    - {ares_vol_ma_col:<12}: {ares_volume_ma}")
            print(f"    - {cooldown_vol_ma_col:<12}: {cooldown_volume_ma}")
        for period in ma_periods_to_probe:
            col_name = f'MA_{period}_D'
            ma_value = df.get(col_name, pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
            if isinstance(ma_value, (float, np.floating)):
                print(f"    - {col_name:<12}: {ma_value:.2f}")
            else:
                print(f"    - {col_name:<12}: {ma_value}")
        print("\n  --- [天平左侧] 底部上下文解剖 ---")
        depth_threshold = get_param_value(p_synthesis.get('deep_bearish_threshold'), 0.05)
        ma55_lifeline = df.get('MA_55_D', df['close_D'])
        is_deep_bearish_zone = (df['close_D'] < ma55_lifeline * (1 - depth_threshold)).astype(float)
        ma55_slope = ma55_lifeline.diff(3).fillna(0)
        slope_moderator = (0.5 + 0.5 * np.tanh(ma55_slope * 100)).fillna(0.5)
        distance_from_ma55 = (df['close_D'] - ma55_lifeline) / ma55_lifeline.replace(0, np.nan)
        lifeline_support_score_raw = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2).fillna(0.0)
        lifeline_support_score = lifeline_support_score_raw * slope_moderator
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        absolute_value_zone_score = 1.0 - price_pos_yearly
        deep_bottom_context_score_values = np.maximum.reduce([
            lifeline_support_score.values,
            absolute_value_zone_score.values
        ])
        deep_bottom_context_score = pd.Series(deep_bottom_context_score_values, index=df.index, dtype=np.float32)
        rsi_w_col = 'RSI_13_W'
        rsi_w_oversold_score = normalize_score(df.get(rsi_w_col, pd.Series(50, index=df.index)), df.index, window=52, ascending=False, default_value=0.5)
        cycle_phase = atomic.get('DOMINANT_CYCLE_PHASE', pd.Series(0.0, index=df.index)).fillna(0.0)
        cycle_trough_score = (1 - cycle_phase) / 2.0
        context_weights = get_param_value(p_synthesis.get('bottom_context_weights'), {'price_pos': 0.5, 'rsi_w': 0.3, 'cycle': 0.2})
        score_components = {'price_pos': deep_bottom_context_score, 'rsi_w': rsi_w_oversold_score, 'cycle': cycle_trough_score}
        valid_scores, valid_weights = [], []
        for name, weight in context_weights.items():
            if name in score_components and weight > 0:
                valid_scores.append(score_components[name].values)
                valid_weights.append(weight)
        if not valid_scores:
            bottom_context_score_raw = pd.Series(0.5, index=df.index, dtype=np.float32)
        else:
            weights_array = np.array(valid_weights)
            total_weight = weights_array.sum()
            normalized_weights = weights_array / total_weight if total_weight > 0 else np.full_like(weights_array, 1.0 / len(weights_array))
            stacked_scores = np.stack(valid_scores, axis=0)
            safe_scores = np.maximum(stacked_scores, 1e-9)
            weighted_log_sum = np.sum(np.log(safe_scores) * normalized_weights[:, np.newaxis], axis=0)
            bottom_context_score_raw = pd.Series(np.exp(weighted_log_sum), index=df.index, dtype=np.float32)
        conventional_bottom_score = bottom_context_score_raw * is_deep_bearish_zone
        print(f"    - [组件1] 常规底部得分 (经深度熊市过滤): {conventional_bottom_score.get(probe_date, 0.0):.4f}")
        # 在探针调用时，传入 atomic_states
        gaia_bedrock_support_score = _calculate_gaia_bedrock_support(df, gaia_params, atomic)
        
        print(f"    - [组件2] 盖亚基石支撑分: {gaia_bedrock_support_score.get(probe_date, 0.0):.4f}")
        print("      --- [盖亚显微镜] 深入解剖 ---")
        support_levels = get_param_value(gaia_params.get('support_levels'), [55, 89, 144, 233, 377])
        confirmation_window = get_param_value(gaia_params.get('confirmation_window'), 3)
        aegis_lookback_window = get_param_value(gaia_params.get('aegis_lookback_window'), 5)
        confirmation_cooldown_period = get_param_value(gaia_params.get('confirmation_cooldown_period'), 10)
        influence_zone_pct = get_param_value(gaia_params.get('influence_zone_pct'), 0.03)
        defense_base_score = get_param_value(gaia_params.get('defense_base_score'), 0.4)
        defense_yang_line_weight = get_param_value(gaia_params.get('defense_yang_line_weight'), 0.1)
        defense_dominance_weight = get_param_value(gaia_params.get('defense_dominance_weight'), 0.2)
        defense_volume_weight = get_param_value(gaia_params.get('defense_volume_weight'), 0.3)
        confirmation_score = get_param_value(gaia_params.get('confirmation_score'), 0.8)
        aegis_quality_bonus_factor = get_param_value(gaia_params.get('aegis_quality_bonus_factor'), 0.25)
        g_ma_cols = [f'MA_{p}_D' for p in support_levels if f'MA_{p}_D' in df.columns]
        g_ma_df = df[g_ma_cols]
        g_ma_df_below_price = g_ma_df.where(g_ma_df.le(df['close_D'], axis=0))
        g_acting_lifeline = g_ma_df_below_price.max(axis=1).ffill()
        g_is_in_influence_zone = pd.Series(False, index=df.index)
        g_valid_indices = g_acting_lifeline.dropna().index
        g_upper_bound = g_acting_lifeline[g_valid_indices] * (1 + influence_zone_pct)
        g_is_in_influence_zone.loc[g_valid_indices] = df.loc[g_valid_indices, 'close_D'].between(g_acting_lifeline[g_valid_indices], g_upper_bound)
        g_base_defense_condition = (df['low_D'] < g_acting_lifeline) & g_is_in_influence_zone & (df['close_D'] > df['low_D'])
        g_is_yang_line = df['close_D'] > df['open_D']
        g_upper_shadow = df['high_D'] - np.maximum(df['open_D'], df['close_D'])
        g_lower_shadow = np.minimum(df['open_D'], df['close_D']) - df['low_D']
        g_has_dominance = g_lower_shadow > g_upper_shadow
        g_has_volume_spike = df['volume_D'] > df[ares_vol_ma_col]
        g_is_cassandra_warning = (g_upper_shadow > g_lower_shadow) & g_has_volume_spike
        g_defense_quality_score = pd.Series(0.0, index=df.index)
        g_defense_quality_score.loc[g_base_defense_condition] = defense_base_score
        g_defense_quality_score.loc[g_base_defense_condition & g_is_yang_line] += defense_yang_line_weight
        g_defense_quality_score.loc[g_base_defense_condition & g_has_dominance] += defense_dominance_weight
        g_defense_quality_score.loc[g_base_defense_condition & g_has_dominance & g_has_volume_spike] += defense_volume_weight
        g_defense_quality_score.loc[g_is_in_influence_zone & g_is_cassandra_warning] = 0.0
        g_defense_quality_score = g_defense_quality_score.clip(0, 1.0)
        g_max_recent_defense_quality = g_defense_quality_score.rolling(window=aegis_lookback_window, min_periods=1).max()
        g_is_standing_firm_in_zone = (df['close_D'] > g_acting_lifeline) & g_is_in_influence_zone
        g_is_confirmed_base = g_is_standing_firm_in_zone.rolling(window=confirmation_window, min_periods=confirmation_window).sum() >= confirmation_window
        g_is_cooldown_reset_signal = (g_upper_shadow > g_lower_shadow) & (df['volume_D'] > df[cooldown_vol_ma_col])
        g_confirmation_score_series = pd.Series(0.0, index=df.index)
        g_last_confirmation_date = pd.NaT
        g_is_in_cooldown_on_probe_date = False
        for idx in df.index:
            if idx > probe_date: break
            if pd.notna(g_last_confirmation_date) and (idx - g_last_confirmation_date).days < confirmation_cooldown_period:
                if idx == probe_date: g_is_in_cooldown_on_probe_date = True
                if g_is_cooldown_reset_signal.get(idx, False): g_last_confirmation_date = pd.NaT
                continue
            if g_is_confirmed_base.get(idx, False):
                recent_quality = g_max_recent_defense_quality.get(idx, 0.0)
                if recent_quality > 0:
                    aegis_score = confirmation_score + recent_quality * aegis_quality_bonus_factor
                    g_confirmation_score_series.loc[idx] = min(aegis_score, 1.0)
                else:
                    g_confirmation_score_series.loc[idx] = confirmation_score
                g_last_confirmation_date = idx
        print(f"        - acting_lifeline (代理总指挥): {g_acting_lifeline.get(probe_date, np.nan):.4f}")
        print("        --- [防守质量解剖 (赫尔墨斯信使)] ---")
        base_cond_val, yang_line_val, dominance_val, volume_val, cassandra_val, in_zone_val = (
            g_base_defense_condition.get(probe_date, False), g_is_yang_line.get(probe_date, False),
            g_has_dominance.get(probe_date, False), g_has_volume_spike.get(probe_date, False),
            g_is_cassandra_warning.get(probe_date, False), g_is_in_influence_zone.get(probe_date, False)
        )
        defense_score_today = g_defense_quality_score.get(probe_date, 0.0)
        print(f"          - 预警判定 (卡珊德拉): (在影响区内 {in_zone_val}) AND (上影>下影 AND 放量) -> {cassandra_val and in_zone_val}")
        if cassandra_val and in_zone_val:
            print(f"          - 裁决: 触发卡珊德拉预警，防守质量分强制归零。")
        else:
            print(f"          - 基础条件 (触线+下影): {base_cond_val} -> 基础分 {defense_base_score if base_cond_val else 0.0:.2f}")
            print(f"          - 权重1 (主权宣告-收阳): {yang_line_val} -> 加分 {defense_yang_line_weight if yang_line_val and base_cond_val else 0.0:.2f}")
            print(f"          - 权重2 (韧性胜利-下影优势): {dominance_val} -> 加分 {defense_dominance_weight if dominance_val and base_cond_val else 0.0:.2f}")
            print(f"          - 权重3 (主力参战-放量): {volume_val and dominance_val} (需下影优势) -> 加分 {defense_volume_weight if volume_val and dominance_val and base_cond_val else 0.0:.2f}")
        print(f"          - 当日独立防守分: {defense_score_today:.4f}")
        print("        --- [确认质量评估 (所罗门审判)] ---")
        is_confirmed_val = g_is_confirmed_base.get(probe_date, False)
        recent_quality_val = g_max_recent_defense_quality.get(probe_date, 0.0)
        confirmation_score_today = g_confirmation_score_series.get(probe_date, 0.0)
        print(f"          - is_confirmed_base (是否满足站稳天数): {is_confirmed_val}")
        print(f"          - is_in_cooldown (是否处于确认冷却期): {g_is_in_cooldown_on_probe_date}")
        print(f"          - 近期最高防守质量分 (lookback={aegis_lookback_window}d): {recent_quality_val:.4f}")
        if not g_is_in_cooldown_on_probe_date and is_confirmed_val:
            if recent_quality_val > 0:
                print(f"          - 审判类型: 神盾构筑 (基础分 {confirmation_score:.2f} + 质量奖励 {recent_quality_val:.2f} * {aegis_quality_bonus_factor:.2f})")
            else:
                print(f"          - 审判类型: 常规确认 (固定分 {confirmation_score:.2f})")
        else:
            print(f"          - 审判类型: 无 (is_confirmed={is_confirmed_val}, in_cooldown={g_is_in_cooldown_on_probe_date})")
        print(f"          - 当日独立确认分: {confirmation_score_today:.4f}")
        print("        --- [最终审判 (忒弥斯天平)] ---")
        final_gaia_score = max(defense_score_today, confirmation_score_today)
        print(f"          - 裁决: max(独立防守分, 独立确认分) = max({defense_score_today:.4f}, {confirmation_score_today:.4f}) = {final_gaia_score:.4f}")
        print("        --- [冷却重置解剖 (哈迪斯面纱)] ---")
        reset_cond1 = g_upper_shadow.get(probe_date, 0) > g_lower_shadow.get(probe_date, 0)
        reset_cond2 = df.get('volume_D').get(probe_date, 0) > df.get(cooldown_vol_ma_col).get(probe_date, np.inf)
        print(f"          - 条件1 (上影优势): upper_shadow > lower_shadow -> {reset_cond1}")
        print(f"          - 条件2 (成交放量): volume > {cooldown_vol_ma_col} -> {reset_cond2}")
        print(f"          - 综合判定 (is_cooldown_reset): {g_is_cooldown_reset_signal.get(probe_date, False)}")
        print("      --------------------------")
        p_fib_support = get_param_value(p_synthesis.get('fibonacci_support_params'), {})
        historical_low_support_score = _calculate_historical_low_support(df, p_fib_support)
        print(f"    - [组件3] 历史低点支撑分: {historical_low_support_score.get(probe_date, 0.0):.4f}")
        structural_support_score = np.maximum(gaia_bedrock_support_score, historical_low_support_score)
        final_bottom_context_score = np.maximum(conventional_bottom_score, structural_support_score)
        print(f"    - [融合步骤1] 结构支撑分 (盖亚 vs 历史低点): {structural_support_score.get(probe_date, 0.0):.4f}")
        print(f"    - [最终裁决] 底部上下文总分 (常规 vs 结构): {final_bottom_context_score.get(probe_date, 0.0):.4f}")
        print("\n  --- [天平右侧] 顶部上下文解剖 ---")
        atomic['strategy_instance_ref'] = self.strategy
        _, top_context = calculate_context_scores(df, atomic)
        del atomic['strategy_instance_ref']
        ma55 = df.get('MA_55_D', df['close_D'])
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        wave_channel_height = (rolling_high_55d - ma55).replace(0, 1e-9)
        stretch_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
        ma_periods = [5, 13, 21, 55]
        short_ma_cols = [f'MA_{p}_D' for p in ma_periods[:-1]]
        long_ma_cols = [f'MA_{p}_D' for p in ma_periods[1:]]
        if all(col in df for col in short_ma_cols + long_ma_cols):
            short_mas = df[short_ma_cols].values
            long_mas = df[long_ma_cols].values
            misalignment_matrix = (short_mas < long_mas).astype(np.float32)
            misalignment_score_values = np.mean(misalignment_matrix, axis=1)
            misalignment_score = pd.Series(misalignment_score_values, index=df.index)
        else:
            misalignment_score = pd.Series(0.5, index=df.index)
        bias_params = get_param_value(p_synthesis.get('bias_overheat_params'), {})
        warning_threshold = get_param_value(bias_params.get('warning_threshold'), 0.15)
        danger_threshold = get_param_value(bias_params.get('danger_threshold'), 0.25)
        bias_abs = df.get('BIAS_21_D', pd.Series(0, index=df.index)).abs()
        denominator = danger_threshold - warning_threshold
        if denominator <= 1e-6:
            overheat_score = (bias_abs > danger_threshold).astype(float)
        else:
            overheat_score = ((bias_abs - warning_threshold) / denominator).clip(0, 1)
        overheat_score = overheat_score.fillna(0.0)
        conventional_top_score = (stretch_score * misalignment_score * overheat_score)**(1/3)
        print(f"    - [组件1] 常规顶部得分 (传统三因子融合): {conventional_top_score.get(probe_date, 0.0):.4f}")
        print(f"      - 价格拉伸分: {stretch_score.get(probe_date, 0.0):.4f}")
        print(f"      - 均线混乱分: {misalignment_score.get(probe_date, 0.0):.4f}")
        print(f"      - 乖离过热分: {overheat_score.get(probe_date, 0.0):.4f} (原始BIAS: {bias_abs.get(probe_date, 0.0):.2%})")
        uranus_params = get_param_value(p_synthesis.get('uranus_ceiling_params'), {})
        from .utils import _calculate_uranus_ceiling_resistance, _calculate_historical_high_resistance
        uranus_ceiling_resistance_score_series = _calculate_uranus_ceiling_resistance(df, uranus_params)
        uranus_ceiling_resistance_score = uranus_ceiling_resistance_score_series.get(probe_date, 0.0)
        self._deploy_uranus_ceiling_probe(probe_date)
        p_fib_resistance = get_param_value(p_synthesis.get('fibonacci_resistance_params'), {})
        print("      --- [历史高点显微镜] 深入解剖 ---")
        historical_high_resistance_score_series = _calculate_historical_high_resistance(df, p_fib_resistance, uranus_params)
        final_historical_high_score = historical_high_resistance_score_series.get(probe_date, 0.0)
        if get_param_value(p_fib_resistance.get('enabled'), False):
            fib_periods = get_param_value(p_fib_resistance.get('periods'), [34, 55, 89, 144, 233])
            level_scores = get_param_value(p_fib_resistance.get('level_scores'), {})
            for period in fib_periods:
                period_str = str(period)
                if period_str not in level_scores: continue
                rolling_high_series = df['high_D'].rolling(window=period, min_periods=max(1, int(period*0.8))).max().shift(1)
                historical_high_val = rolling_high_series.get(probe_date)
                if pd.notna(historical_high_val):
                    historical_high_date = rolling_high_series.loc[:probe_date].idxmax()
                    print(f"\n        - {period}日周期: 找到前高 {historical_high_val:.2f} (日期: {historical_high_date.strftime('%Y-%m-%d')})")
                else:
                    print(f"\n        - {period}日周期: 未找到有效前高。")
                    continue
                temp_resistance_series = pd.Series(np.nan, index=df.index)
                temp_resistance_series.at[probe_date] = historical_high_val
                rejection_quality = self._deploy_apollo_arrow_probe(probe_date, uranus_params, temp_resistance_series)
                strategic_importance = level_scores[period_str]
                final_period_score = rejection_quality * strategic_importance
                print(f"        --- [融合裁决] ---")
                print(f"          - 战术质量分 (来自阿波罗之箭): {rejection_quality:.4f}")
                print(f"          - 战略重要性分 (来自配置): {strategic_importance:.2f}")
                print(f"          - ⚖️ 周期风险分 = 质量 * 重要性 = {rejection_quality:.4f} * {strategic_importance:.2f} = {final_period_score:.4f}")
        structural_resistance_score = np.maximum(uranus_ceiling_resistance_score, final_historical_high_score)
        print(f"\n    - [组件2] 结构性阻力得分: {structural_resistance_score:.4f}")
        print(f"      - 乌拉诺斯穹顶(均线)阻力分: {uranus_ceiling_resistance_score:.4f}")
        print(f"      - 历史高点(价格)阻力分: {final_historical_high_score:.4f}")
        final_top_context_score = np.maximum(conventional_top_score.get(probe_date, 0.0), structural_resistance_score)
        print(f"    - [最终裁决] 顶部上下文总分 (常规 vs 结构): {final_top_context_score:.4f}")
        print("\n--- “忒弥斯天平”称量完毕 ---")

    def _get_dominant_offense_type_for_probe(self, total_offense_score: float, active_offense: list) -> str:
        """
        【V1.1 · 记忆烙印协议版】为“宙斯之雷”探针专门提供的、用于模拟“最强进攻信号类型”判断的辅助方法。
        - 核心升级: 不再依赖脆弱的中文名反查，而是直接使用 `internal_name` 进行精准查找。
        """
        if total_offense_score <= 0 or not active_offense:
            return 'unknown'
        # 直接获取主导信号的内部名称
        dominant_signal_internal_name = active_offense[0].get('internal_name')
        if not dominant_signal_internal_name:
            return 'unknown'
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        # 使用内部名称进行精准、可靠的查找
        meta = score_map.get(dominant_signal_internal_name, {})
        return meta.get('type', 'unknown')

    def _deploy_uranus_ceiling_probe(self, probe_date: pd.Timestamp):
        """
        【V1.3 · 神盾协议探针版】“乌拉诺斯穹顶”法医探针
        - 核心升级: 调用通用的 _deploy_apollo_arrow_probe 探针来解剖拒绝质量。
        """
        print("\n      --- [乌拉诺斯显微镜] 深入解剖 ---")
        df = self.strategy.df_indicators
        strategy_instance_ref = self.strategy
        p_synthesis = get_params_block(strategy_instance_ref, 'ultimate_signal_synthesis_params', {})
        uranus_params = get_param_value(p_synthesis.get('uranus_ceiling_params'), {})
        if not get_param_value(uranus_params.get('enabled'), False):
            print("        - 乌拉诺斯穹顶系统在配置中被禁用。")
            return 0.0
        
        # [代码重构] 大部分参数获取移至阿波罗之箭探针，这里只保留确认压制所需的参数
        resistance_levels = get_param_value(uranus_params.get('resistance_levels'), [55, 89, 144, 233, 377])
        confirmation_window = get_param_value(uranus_params.get('confirmation_window'), 3)
        rejection_lookback_window = get_param_value(uranus_params.get('rejection_lookback_window'), 5)
        confirmation_cooldown_period = get_param_value(uranus_params.get('confirmation_cooldown_period'), 10)
        confirmation_score = get_param_value(uranus_params.get('confirmation_score'), 0.8)
        rejection_quality_bonus_factor = get_param_value(uranus_params.get('rejection_quality_bonus_factor'), 0.25)
        cooldown_reset_volume_ma_period = get_param_value(uranus_params.get('cooldown_reset_volume_ma_period'), 55)
        
        close_col, open_col, low_col, high_col, vol_col = 'close_D', 'open_D', 'low_D', 'high_D', 'volume_D'
        cooldown_vol_ma_col = f'VOL_MA_{cooldown_reset_volume_ma_period}_D'
        
        # 1. 寻找代理天花板
        ma_cols = [f'MA_{p}_D' for p in resistance_levels if f'MA_{p}_D' in df.columns]
        ma_df = df[ma_cols]
        ma_df_above_price = ma_df.where(ma_df.ge(df[close_col], axis=0))
        acting_ceiling = ma_df_above_price.min(axis=1).ffill()
        
        print(f"        - acting_ceiling (代理天花板): {acting_ceiling.get(probe_date, np.nan):.4f}")
        
        # 2. 调用通用的阿波罗之箭探针来获取拒绝质量分
        rejection_score_today = self._deploy_apollo_arrow_probe(probe_date, uranus_params, acting_ceiling)
        
        # 3. [代码重构] 确认压制评估逻辑保持不变，但需要重新计算 rejection_quality_score 序列
        from .utils import _calculate_rejection_quality_score
        rejection_quality_score = _calculate_rejection_quality_score(df, uranus_params, acting_ceiling)

        print("        --- [确认压制评估 (哈迪斯之锁)] ---")
        max_recent_rejection_quality = rejection_quality_score.rolling(window=rejection_lookback_window, min_periods=1).max()
        
        influence_zone_pct = get_param_value(uranus_params.get('influence_zone_pct'), 0.03)
        is_in_influence_zone = pd.Series(False, index=df.index)
        valid_indices = acting_ceiling.dropna().index
        if not valid_indices.empty:
            lower_bound = acting_ceiling[valid_indices] * (1 - influence_zone_pct)
            is_in_influence_zone.loc[valid_indices] = df.loc[valid_indices, close_col].between(lower_bound, acting_ceiling[valid_indices])

        is_failing_to_break = (df[close_col] < acting_ceiling) & is_in_influence_zone
        is_confirmed_rejection = is_failing_to_break.rolling(window=confirmation_window, min_periods=confirmation_window).sum() >= confirmation_window
        
        upper_shadow = df[high_col] - np.maximum(df[open_col], df[close_col])
        lower_shadow = np.minimum(df[open_col], df[close_col]) - df[low_col]
        is_cooldown_reset_signal = (lower_shadow > upper_shadow) & (df[vol_col] > df[cooldown_vol_ma_col])
        
        confirmation_score_series = pd.Series(0.0, index=df.index)
        last_confirmation_date = pd.NaT
        is_in_cooldown_on_probe_date = False
        for idx in df.index:
            if idx > probe_date: break
            if pd.notna(last_confirmation_date) and (idx - last_confirmation_date).days < confirmation_cooldown_period:
                if idx == probe_date: is_in_cooldown_on_probe_date = True
                if is_cooldown_reset_signal.get(idx, False): last_confirmation_date = pd.NaT
                continue
            if is_confirmed_rejection.get(idx, False):
                recent_quality = max_recent_rejection_quality.get(idx, 0.0)
                if recent_quality > 0:
                    rejection_score = confirmation_score + recent_quality * rejection_quality_bonus_factor
                    confirmation_score_series.loc[idx] = min(rejection_score, 1.0)
                else:
                    confirmation_score_series.loc[idx] = confirmation_score
                last_confirmation_date = idx
        
        is_confirmed_val = is_confirmed_rejection.get(probe_date, False)
        recent_quality_val = max_recent_rejection_quality.get(probe_date, 0.0)
        confirmation_score_today = confirmation_score_series.get(probe_date, 0.0)
        
        print(f"          - is_confirmed_rejection (是否满足压制天数): {is_confirmed_val}")
        print(f"          - is_in_cooldown (是否处于确认冷却期): {is_in_cooldown_on_probe_date}")
        print(f"          - 近期最高拒绝质量分 (lookback={rejection_lookback_window}d): {recent_quality_val:.4f}")
        if not is_in_cooldown_on_probe_date and is_confirmed_val:
            if recent_quality_val > 0:
                print(f"          - 审判类型: 强化压制 (基础分 {confirmation_score:.2f} + 质量奖励 {recent_quality_val:.2f} * {rejection_quality_bonus_factor:.2f})")
            else:
                print(f"          - 审判类型: 常规压制 (固定分 {confirmation_score:.2f})")
        else:
            print(f"          - 审判类型: 无 (is_confirmed={is_confirmed_val}, in_cooldown={is_in_cooldown_on_probe_date})")
        print(f"          - 当日独立确认分: {confirmation_score_today:.4f}")
        
        print("        --- [最终审判 (塔纳托斯之镰)] ---")
        final_score = max(rejection_score_today, confirmation_score_today)
        print(f"          - 裁决: max(独立拒绝分, 独立确认分) = max({rejection_score_today:.4f}, {confirmation_score_today:.4f}) = {final_score:.4f}")
        return final_score

    def _deploy_hades_gaze_probe(self, probe_date: pd.Timestamp, domain: str, signal_type: str):
        """
        【V1.1 · 赫淮斯托斯重铸版】“哈迪斯凝视”终极风险探针
        - 核心修复: 彻底修正了计算顺序的颠倒错误。探针现在严格遵循“先在周期内相乘，再跨周期融合”的
                      正确逻辑，确保重算结果与主引擎完全一致。
        - 核心职责: 钻透式解剖终极风险信号，揭示其从支柱健康度到最终分数的完整计算链路。
        - 调用示例: self._deploy_hades_gaze_probe(probe_date, 'CHIP', 'BEARISH_RESONANCE')
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_{signal_type}'
        print(f"\n--- [探针] 正在启用: 💀【哈迪斯凝视】💀 -> 解剖信号【{signal_name}】 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        def get_val(name, date, default=np.nan):
            series = atomic.get(name)
            if series is None: return default
            return series.get(date, default)
        # 链路层 1: 获取最终信号值
        final_score_raw = get_val(signal_name, probe_date, 0.0)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        signal_meta = score_map.get(signal_name, {})
        base_score = signal_meta.get('penalty_weight', signal_meta.get('score', 0))
        final_score_contribution = final_score_raw * base_score
        print(f"\n  [链路层 1] 最终风险贡献: {final_score_contribution:.0f}")
        print(f"    - [公式]: 原始信号值 * 基础分")
        print(f"    - [计算]: {final_score_raw:.4f} * {base_score} = {final_score_contribution:.2f}")
        # 链路层 2: 反推到中央合成引擎的输出
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        resonance_tf_weights = get_param_value(p_synthesis.get('resonance_tf_weights'), {})
        reversal_tf_weights = get_param_value(p_synthesis.get('reversal_tf_weights'), {})
        overall_health_cache = atomic.get(f'__{domain_upper}_overall_health', {})
        if not overall_health_cache:
            print("    - [探针错误] 无法找到领域健康度缓存。解剖终止。")
            return
        # 提取探针日的整体健康度
        s_bull_health = {p: v.get(probe_date, 0.5) for p, v in overall_health_cache.get('s_bull', {}).items()}
        s_bear_health = {p: v.get(probe_date, 0.5) for p, v in overall_health_cache.get('s_bear', {}).items()}
        d_intensity_health = {p: v.get(probe_date, 0.5) for p, v in overall_health_cache.get('d_intensity', {}).items()}
        print(f"\n  [链路层 2] 反推 -> 中央合成引擎 (transmute_health_to_ultimate_signals)")
        # 修正计算顺序：先在每个周期内相乘，再跨周期融合
        recalc_raw_score = 0.0
        if signal_type == 'BEARISH_RESONANCE':
            print(f"    - [公式]: 看跌共振 = Fuse(各周期s_bear * 各周期d_intensity)")
            period_scores = {p: s_bear_health.get(p, 0.5) * d_intensity_health.get(p, 0.5) for p in s_bear_health.keys()}
            print(f"    - [周期内计算]:")
            for p, score in period_scores.items():
                print(f"      - {p:<2}日周期: s_bear({s_bear_health.get(p, 0.5):.4f}) * d_intensity({d_intensity_health.get(p, 0.5):.4f}) = {score:.4f}")
            short_force = (period_scores.get(1, 0.5) * period_scores.get(5, 0.5))**0.5
            medium_trend = (period_scores.get(13, 0.5) * period_scores.get(21, 0.5))**0.5
            long_inertia = period_scores.get(55, 0.5)
            recalc_raw_score = (short_force**resonance_tf_weights.get('short', 0.2) * 
                                medium_trend**resonance_tf_weights.get('medium', 0.5) * 
                                long_inertia**resonance_tf_weights.get('long', 0.3))
            print(f"    - [跨周期融合计算]: Fuse(...) = {recalc_raw_score:.4f}")
        elif signal_type == 'TOP_REVERSAL':
            print(f"    - [公式]: 顶部反转 = Fuse(各周期s_bear * (1 - 各周期s_bull))")
            period_scores = {p: s_bear_health.get(p, 0.5) * (1.0 - s_bull_health.get(p, 0.5)) for p in s_bear_health.keys()}
            print(f"    - [周期内计算]:")
            for p, score in period_scores.items():
                print(f"      - {p:<2}日周期: s_bear({s_bear_health.get(p, 0.5):.4f}) * (1 - s_bull({s_bull_health.get(p, 0.5):.4f})) = {score:.4f}")
            short_force = (period_scores.get(1, 0.5) * period_scores.get(5, 0.5))**0.5
            medium_trend = (period_scores.get(13, 0.5) * period_scores.get(21, 0.5))**0.5
            long_inertia = period_scores.get(55, 0.5)
            recalc_raw_score = (short_force**reversal_tf_weights.get('short', 0.6) * 
                                medium_trend**reversal_tf_weights.get('medium', 0.3) * 
                                long_inertia**reversal_tf_weights.get('long', 0.1))
            print(f"    - [跨周期融合计算]: Fuse(...) = {recalc_raw_score:.4f}")
        else:
            print(f"    - [探针警告] 不支持的风险信号类型: {signal_type}")
            return
        print(f"    - [对比]: 实际原始值 {final_score_raw:.4f} vs 重算原始值 {recalc_raw_score:.4f}")
        # 链路层3和4保持不变，但现在其分析更有意义
        print(f"\n  [链路层 3] 钻透 -> 整体三维健康度来源 (融合后的参考值)")
        overall_bullish_health = ( ( (s_bull_health.get(1,0.5)*s_bull_health.get(5,0.5))**0.5 )**resonance_tf_weights.get('short',0.2) * ( (s_bull_health.get(13,0.5)*s_bull_health.get(21,0.5))**0.5 )**resonance_tf_weights.get('medium',0.5) * (s_bull_health.get(55,0.5))**resonance_tf_weights.get('long',0.3) )
        overall_bearish_health = ( ( (s_bear_health.get(1,0.5)*s_bear_health.get(5,0.5))**0.5 )**resonance_tf_weights.get('short',0.2) * ( (s_bear_health.get(13,0.5)*s_bear_health.get(21,0.5))**0.5 )**resonance_tf_weights.get('medium',0.5) * (s_bear_health.get(55,0.5))**resonance_tf_weights.get('long',0.3) )
        overall_dynamic_intensity = ( ( (d_intensity_health.get(1,0.5)*d_intensity_health.get(5,0.5))**0.5 )**resonance_tf_weights.get('short',0.2) * ( (d_intensity_health.get(13,0.5)*d_intensity_health.get(21,0.5))**0.5 )**resonance_tf_weights.get('medium',0.5) * (d_intensity_health.get(55,0.5))**resonance_tf_weights.get('long',0.3) )
        print(f"    - 整体看涨健康度 (s_bull): {overall_bullish_health:.4f}")
        print(f"    - 整体看跌健康度 (s_bear): {overall_bearish_health:.4f}  <-- 风险的主要来源之一")
        print(f"    - 整体动态强度 (d_intensity): {overall_dynamic_intensity:.4f}  <-- 风险的主要来源之一")
        print(f"\n  [链路层 4] 终极解剖 -> 各支柱健康度贡献 (以5日周期为例)")
        pillar_configs = {
            'CHIP': ['quantitative', 'advanced', 'internal', 'holder', 'fault'],
            'FUND_FLOW': ['consensus', 'conviction', 'conflict', 'sentiment'],
            'DYN': ['volatility', 'efficiency', 'momentum', 'inertia'],
            'STRUCTURE': ['ma', 'mechanics', 'mtf', 'pattern'],
            'BEHAVIOR': ['price', 'volume', 'kline'],
            'FOUNDATION': ['ema', 'rsi', 'macd', 'cmf']
        }
        pillars = pillar_configs.get(domain_upper, [])
        if not pillars:
            print(f"    - [探针警告] 未找到领域 {domain_upper} 的支柱配置。")
            return
        print(f"    {'Pillar':<15} | {'s_bull':<10} | {'s_bear':<10} | {'d_intensity':<10}")
        print(f"    {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10}")
        for pillar_name in pillars:
            pillar_health = atomic.get(f'_PILLAR_HEALTH_{domain_upper}_{pillar_name}')
            if not pillar_health:
                print(f"    {pillar_name:<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
                continue
            s_b = pillar_health.get('s_bull', {}).get(5, pd.Series(0.5)).get(probe_date, 0.5)
            s_br = pillar_health.get('s_bear', {}).get(5, pd.Series(0.5)).get(probe_date, 0.5)
            d_i = pillar_health.get('d_intensity', {}).get(5, pd.Series(0.5)).get(probe_date, 0.5)
            print(f"    {pillar_name:<15} | {s_b:<10.4f} | {s_br:<10.4f} | {d_i:<10.4f}")
        print("\n--- “哈迪斯凝视”解剖完毕 ---")

    def _deploy_apollo_arrow_probe(self, probe_date: pd.Timestamp, params: Dict, resistance_line: pd.Series) -> float:
        """
        【V1.3 · 塔纳托斯之镰探针版】通用拒绝质量评估探针 (阿波罗之箭)
        - 核心升级: 完全同步“伊卡洛斯之陨”的风险叠加逻辑，并清晰打印奖励分。
        """
        print("        --- [阿波罗之箭评估] ---")
        df = self.strategy.df_indicators
        # 从参数中获取所有必要的配置
        influence_zone_pct = get_param_value(params.get('influence_zone_pct'), 0.03)
        rejection_base_score = get_param_value(params.get('rejection_base_score'), 0.4)
        rejection_yin_line_weight = get_param_value(params.get('rejection_yin_line_weight'), 0.1)
        rejection_dominance_weight = get_param_value(params.get('rejection_dominance_weight'), 0.2)
        rejection_volume_weight = get_param_value(params.get('rejection_volume_weight'), 0.3)
        min_shadow_ratio = get_param_value(params.get('min_shadow_ratio'), 0.15) # 已按您的要求修改
        # 废除icarus_fall_base_score，引入icarus_fall_bonus
        icarus_fall_bonus = get_param_value(params.get('icarus_fall_bonus'), 0.5)
        cooldown_reset_volume_ma_period = get_param_value(params.get('cooldown_reset_volume_ma_period'), 55)
        close_col, open_col, low_col, high_col, vol_col = 'close_D', 'open_D', 'low_D', 'high_D', 'volume_D'
        ares_vol_ma_col = 'VOL_MA_5_D'
        # 检查必需列
        required_cols = [close_col, open_col, low_col, high_col, vol_col, ares_vol_ma_col, 'up_limit_D']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 1. 获取当日关键数据
        res_val = resistance_line.get(probe_date, np.nan)
        if pd.isna(res_val):
            print("          - 目标阻力位当日无有效值，评估跳过。")
            return 0.0
        # 2. 计算影响区和基础条件
        lower_bound_val = res_val * (1 - influence_zone_pct)
        is_in_influence_zone_val = lower_bound_val <= df.at[probe_date, close_col] <= res_val
        base_rejection_condition_val = (df.at[probe_date, high_col] > res_val) & is_in_influence_zone_val & (df.at[probe_date, close_col] < df.at[probe_date, high_col])
        # 3. 计算各项质量加权分
        rejection_quality_score_val = 0.0
        if base_rejection_condition_val:
            rejection_quality_score_val = rejection_base_score
        is_yin_line_val = df.at[probe_date, close_col] < df.at[probe_date, open_col]
        upper_shadow_val = df.at[probe_date, high_col] - max(df.at[probe_date, open_col], df.at[probe_date, close_col])
        lower_shadow_val = min(df.at[probe_date, open_col], df.at[probe_date, close_col]) - df.at[probe_date, low_col]
        kline_range_val = (df.at[probe_date, high_col] - df.at[probe_date, low_col])
        upper_shadow_ratio_val = upper_shadow_val / kline_range_val if kline_range_val > 0 else 0
        is_upper_shadow_significant_val = upper_shadow_ratio_val > min_shadow_ratio
        has_dominance_val = (upper_shadow_val > lower_shadow_val) & is_upper_shadow_significant_val
        yin_line_bonus = rejection_yin_line_weight if base_rejection_condition_val and is_yin_line_val else 0.0
        dominance_bonus = rejection_dominance_weight if base_rejection_condition_val and has_dominance_val else 0.0
        has_volume_spike_val = df.at[probe_date, vol_col] > df.at[probe_date, ares_vol_ma_col]
        proportional_volume_score_val = normalize_score(df[vol_col] / df[ares_vol_ma_col].replace(0, np.nan), df.index, window=cooldown_reset_volume_ma_period, ascending=True).get(probe_date, 0.0)
        volume_bonus = rejection_volume_weight * proportional_volume_score_val if base_rejection_condition_val and has_dominance_val and has_volume_spike_val else 0.0
        rejection_quality_score_val += yin_line_bonus + dominance_bonus + volume_bonus
        # 4. 应用绝对否决/奖励规则
        limit_up_price_val = df.at[probe_date, 'up_limit_D']
        is_icarus_fall_val = (df.at[probe_date, high_col] >= limit_up_price_val * 0.995) & (df.at[probe_date, close_col] < df.at[probe_date, high_col] * 0.98)
        # 将伊卡洛斯之陨的逻辑从“取最大值”改为“叠加奖励分”
        icarus_bonus_val = icarus_fall_bonus if is_icarus_fall_val else 0.0
        rejection_quality_score_val += icarus_bonus_val
        is_apollo_absorption_val = (lower_shadow_val > upper_shadow_val) & has_volume_spike_val
        if is_in_influence_zone_val and is_apollo_absorption_val:
            rejection_quality_score_val = 0.0
        final_score = np.clip(rejection_quality_score_val, 0, 1.0)
        # 5. 打印详细解剖过程
        print(f"          - 目标阻力线: {res_val:.2f}")
        print(f"          - 影响区: [{lower_bound_val:.2f}, {res_val:.2f}] -> 收盘价({df.at[probe_date, close_col]:.2f})是否在内: {'✅' if is_in_influence_zone_val else '❌'}")
        print(f"          - 基础条件 (触顶+回落): {base_rejection_condition_val} -> 基础分 {rejection_base_score if base_rejection_condition_val else 0.0:.2f}")
        print(f"          - 权重1 (空头宣告-收阴): {is_yin_line_val} -> 加分 {yin_line_bonus:.2f}")
        print(f"          - 影线长度: 上影 {upper_shadow_val:.2f} vs 下影 {lower_shadow_val:.2f}")
        print(f"          - 上影线显著性检查: (上影率 {upper_shadow_ratio_val:.2f} > 阈值 {min_shadow_ratio:.2f}) -> {is_upper_shadow_significant_val}")
        print(f"          - 权重2 (空头胜利-上影优势): (上影>下影 AND 上影显著) -> {has_dominance_val} -> 加分 {dominance_bonus:.2f}")
        print(f"          - 权重3 (主力派发-放量): {has_volume_spike_val and has_dominance_val} -> 加分 {volume_bonus:.2f}")
        print(f"          ---")
        # 修改打印说明，反映叠加逻辑
        print(f"          - 💀 塔纳托斯之镰 (涨停回落): {is_icarus_fall_val} -> 额外奖励分 {icarus_bonus_val:.2f}")
        print(f"          - ☀️ 阿波罗吸收 (多头反噬): {is_apollo_absorption_val and is_in_influence_zone_val} -> 若触发，分数强制归零")
        print(f"          - 最终裁决 (战术质量分): {final_score:.4f}")
        return final_score

    def _deploy_archangel_diagnosis_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 权限修复版】“天使长诊断探针” - 四骑士审查
        - 核心修复: 在调用 calculate_context_scores 之前，为其注入必需的 'strategy_instance_ref' 上下文引用，
                      解决因缺少权限导致计算结果为0的致命BUG。
        """
        print("\n--- [探针] 正在启用: 👼【天使长诊断探针 · 四骑士审查】👼 ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        # 步骤一：获取“第四骑士” - 结构性压力分
        # 注入计算所需的上下文引用
        atomic['strategy_instance_ref'] = self.strategy
        _, top_context_score_series = calculate_context_scores(df, atomic)
        # 计算完毕后，立即移除临时引用，保持状态纯净
        del atomic['strategy_instance_ref']
        top_context_score = top_context_score_series.get(probe_date, 0.0)
        # 步骤二：获取其他三位骑士的信号值
        upthrust_risk = atomic.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        heaven_earth_risk = atomic.get('SCORE_BOARD_HEAVEN_EARTH', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        post_peak_risk = atomic.get('COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        print("  --- [输入审查] 启示录四骑士信号值 ---")
        print(f"    - 骑士1 (上冲派发): {upthrust_risk:.4f}")
        print(f"    - 骑士2 (天地板): {heaven_earth_risk:.4f}")
        print(f"    - 骑士3 (高位回落): {post_peak_risk:.4f}")
        print(f"    - 骑士4 (结构性压力): {top_context_score:.4f}  <-- 来自“忒弥斯天平”")
        # 步骤三：复刻融合逻辑并提供证据
        risk_values = [upthrust_risk, heaven_earth_risk, post_peak_risk, top_context_score]
        archangel_score = max(risk_values)
        print("\n  --- [融合裁决] ---")
        print(f"    - 融合算法: max(骑士1, 骑士2, 骑士3, 骑士4)")
        print(f"    - 计算过程: max({upthrust_risk:.4f}, {heaven_earth_risk:.4f}, {post_peak_risk:.4f}, {top_context_score:.4f}) = {archangel_score:.4f}")
        print(f"    - 最终结论 (SCORE_ARCHANGEL_TOP_REVERSAL 的真实值): {archangel_score:.4f}")
        print("--- “天使长诊断探针”运行完毕 ---")

    def _deploy_athena_wisdom_probe(self, probe_date: pd.Timestamp):
        """
        【V3.0 · 普罗米修斯之火同步版】“雅典娜智慧”探针
        - 核心革命: 探针的重算逻辑已与主引擎的“普罗米修斯之火”协议完全同步。
        - 新核心逻辑: 探针内部完美复刻“加权几何平均 + 关系元分析”的两阶段认知过程。
        - 收益: 彻底解决了探针与主引擎逻辑脱节导致的巨大验证偏差，恢复了探针的诊断能力。
        """
        print("\n--- [探针] 正在启用: 🦉【雅典娜智慧 · 终极底部确认解剖 V3.0】🦉 ---") # 修改: 更新探针版本
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        def get_val(name, date, default=0.0):
            series = atomic.get(name)
            if series is None:
                print(f"      - [警告] 探针无法在 atomic_states 中找到信号: {name}")
                return default
            return series.get(date, default)
        final_score = get_val('COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION', probe_date)
        print(f"\n  [链路层 1] 最终确认成品: COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION = {final_score:.4f}")
        print(f"    - [核心公式]: 终极确认分 = 原始终极底部确认分 * 底部上下文分数")
        print("\n  [链路层 2] 解剖 -> 原始终极底部确认分 (ultimate_bottom_raw)")
        print(f"    - [核心公式]: 原始分 = 认知融合底部反转分 * 形态底部反转分")
        fusion_bottom_val = get_val('COGNITIVE_FUSION_BOTTOM_REVERSAL', probe_date)
        print(f"\n    --- [组件 A] 认知融合底部反转分 (COGNITIVE_FUSION_BOTTOM_REVERSAL): {fusion_bottom_val:.4f} ---")
        # 新增开始: 部署与主引擎完全同步的“普罗米修斯之火”重算逻辑
        print(f"      - [核心公式]: MetaAnalysis(GeometricMean(基础, 结构, 行为))")
        print("\n        --- [组件A显微镜 · 普罗米修斯之火重算] ---")
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        fusion_weights_conf = get_param_value(p_cognitive.get('cognitive_fusion_weights'), {})
        foundation_bottom = get_unified_score(atomic, df.index, 'FOUNDATION_BOTTOM_REVERSAL')
        structure_bottom = get_unified_score(atomic, df.index, 'STRUCTURE_BOTTOM_REVERSAL')
        behavior_bottom = get_unified_score(atomic, df.index, 'BEHAVIOR_BOTTOM_REVERSAL')
        print(f"        - 输入1: 基础层反转分 = {foundation_bottom.get(probe_date, 0.0):.4f}")
        print(f"        - 输入2: 结构层反转分 = {structure_bottom.get(probe_date, 0.0):.4f}")
        print(f"        - 输入3: 行为层反转分 = {behavior_bottom.get(probe_date, 0.0):.4f}")
        print("\n        --- [阶段一: 奥林匹斯众神殿 · 共识快照] ---")
        scores_to_fuse = [foundation_bottom.values, structure_bottom.values, behavior_bottom.values]
        weights_to_fuse = [
            fusion_weights_conf.get('foundation', 0.33),
            fusion_weights_conf.get('structure', 0.33),
            fusion_weights_conf.get('behavior', 0.34)
        ]
        weights_array = np.array(weights_to_fuse)
        weights_array /= weights_array.sum()
        stacked_scores = np.stack(scores_to_fuse, axis=0)
        safe_scores = np.maximum(stacked_scores, 1e-9)
        log_signals = np.log(safe_scores)
        weighted_log_sum = np.sum(log_signals * weights_array[:, np.newaxis], axis=0)
        consensus_snapshot_score = pd.Series(np.exp(weighted_log_sum), index=df.index, dtype=np.float32)
        print(f"          - [探针重算] 共识快照分 (GeometricMean) @ {probe_date.date()}: {consensus_snapshot_score.get(probe_date, 0.0):.4f}")
        print("\n        --- [阶段二: 普罗米修斯之火 · 动态锻造] ---")
        p_meta = get_param_value(p_cognitive.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        meta_window = 5
        norm_window = 55
        state_score_val = consensus_snapshot_score.clip(0, 1).get(probe_date, 0.0)
        relationship_trend = consensus_snapshot_score.diff(meta_window).fillna(0)
        velocity_score_series = normalize_to_bipolar(series=relationship_trend, target_index=df.index, window=norm_window, sensitivity=1.0)
        velocity_score_val = velocity_score_series.get(probe_date, 0.0)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score_series = normalize_to_bipolar(series=relationship_accel, target_index=df.index, window=norm_window, sensitivity=1.0)
        acceleration_score_val = acceleration_score_series.get(probe_date, 0.0)
        print(f"          - 状态分 (State): {state_score_val:.4f}")
        print(f"          - 速度分 (Velocity): {velocity_score_val:.4f}")
        print(f"          - 加速度分 (Acceleration): {acceleration_score_val:.4f}")
        fusion_bottom_recalc = (state_score_val * w_state + velocity_score_val * w_velocity + acceleration_score_val * w_acceleration).clip(0, 1)
        print(f"          - [探针重算] 最终融合分 = ({state_score_val:.2f}*{w_state} + {velocity_score_val:.2f}*{w_velocity} + {acceleration_score_val:.2f}*{w_acceleration}) = {fusion_bottom_recalc:.4f}")
        print(f"          - [对比]: 实际值 {fusion_bottom_val:.4f} vs 重算值 {fusion_bottom_recalc:.4f}")
        # 新增结束
        pattern_bottom_val = get_val('SCORE_PATTERN_BOTTOM_REVERSAL', probe_date)
        print(f"\n    --- [组件 B] 形态底部反转分 (SCORE_PATTERN_BOTTOM_REVERSAL): {pattern_bottom_val:.4f} ---")
        print(f"      - [核心公式]: max(RSI反转, 平台突破, MACD金叉, 动能衰竭)")
        print("\n        --- [组件B显微镜] ---")
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        macd_hist = df.get('MACDh_13_34_8_D', pd.Series(0, index=df.index))
        was_oversold = (rsi.rolling(window=5, min_periods=1).min() < 35)
        is_recovering = (df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)) > 0)
        score_rsi_reversal = (was_oversold & is_recovering).astype(float).get(probe_date, 0.0)
        is_breaking_consolidation = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        score_consolidation_breakout = (is_breaking_consolidation * 0.8).get(probe_date, 0.0)
        is_macd_bull_cross = ((macd_hist > 0) & (macd_hist.shift(1) <= 0)).astype(float)
        score_macd_bullish_cross = is_macd_bull_cross.get(probe_date, 0.0)
        rsi_slope_abs = df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)).abs()
        macd_hist_slope_abs = df.get('SLOPE_1_MACDh_13_34_8_D', pd.Series(0, index=df.index)).abs()
        rsi_exhaustion_score = normalize_score(rsi_slope_abs, df.index, window=60, ascending=False)
        macd_exhaustion_score = normalize_score(macd_hist_slope_abs, df.index, window=60, ascending=False)
        score_momentum_exhaustion = ((rsi_exhaustion_score * macd_exhaustion_score)**0.5).get(probe_date, 0.0)
        print(f"        - 模式1: RSI反转分: {score_rsi_reversal:.4f}")
        print(f"        - 模式2: 平台突破分: {score_consolidation_breakout:.4f}")
        print(f"        - 模式3: MACD金叉分: {score_macd_bullish_cross:.4f}")
        print(f"        - 模式4: 动能衰竭分: {score_momentum_exhaustion:.4f}")
        pattern_bottom_recalc = max(score_rsi_reversal, score_consolidation_breakout, score_macd_bullish_cross, score_momentum_exhaustion)
        print(f"        - [探针重算] 形态分 = max(...) = {pattern_bottom_recalc:.4f}")
        print("\n    --- [调节器] 底部上下文分数 (bottom_context_score) ---")
        atomic['strategy_instance_ref'] = self.strategy
        bottom_context_score_series, _ = calculate_context_scores(df, atomic)
        del atomic['strategy_instance_ref']
        bottom_context_score = bottom_context_score_series.get(probe_date, 0.0)
        print(f"      - [探针获取] 底部上下文分数: {bottom_context_score:.4f} (详情请见“忒弥斯天平”探针)")
        print("\n  [最终验证]")
        ultimate_bottom_raw_recalc = fusion_bottom_recalc * pattern_bottom_recalc
        print(f"    - [探针重算] 原始终极底部确认分 = {fusion_bottom_recalc:.4f} * {pattern_bottom_recalc:.4f} = {ultimate_bottom_raw_recalc:.4f}")
        final_score_recalc = ultimate_bottom_raw_recalc * bottom_context_score
        print(f"    - [探针重算] 终极确认分 = {ultimate_bottom_raw_recalc:.4f} * {bottom_context_score:.4f} = {final_score_recalc:.4f}")
        print(f"    - [对比]: 实际值 {final_score:.4f} vs 重算值 {final_score_recalc:.4f}")
        print("--- “雅典娜智慧”探针解剖完毕 ---")






