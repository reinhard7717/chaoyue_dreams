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
from strategies.trend_following.utils import get_params_block, get_param_value, calculate_context_scores, normalize_score, calculate_trend_confirmation_context, _calculate_gaia_bedrock_support, _calculate_historical_low_support

class IntelligenceLayer:
    """
    【V407.0 · 终极信号适配版】情报层总指挥官
    - 核心职责: 1. 实例化所有专业化的情报子模块。
                2. 按照“原子信号生成 -> 跨域认知融合 -> 战术剧本生成”的顺序，编排和调用这些子模块。
                3. 整合所有模块产出的原子状态和触发器，供下游层使用。
    - 本次修改: 全面适配所有情报引擎的“大一统”重构，确保调用流程和数据流正确无误。
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
        【V415.0 · 先知计划版】情报层总指挥官
        - 核心升级: 在认知层融合之后，审判日引擎裁决之前，插入“先知引擎”的预测诊断。
        """
        # print("--- [情报层总指挥官 V415.0 · 先知计划版] 开始执行所有诊断模块... ---")
        df = self.strategy.df_indicators
        self.strategy.atomic_states = {}
        self.strategy.trigger_events = {}
        self.strategy.playbook_states = {}
        self.strategy.exit_triggers = pd.DataFrame(index=df.index)
        def update_states(new_states: Dict):
            if isinstance(new_states, dict):
                self.strategy.atomic_states.update(new_states)
        # --- 阶段一: 基础信号生成 (按依赖关系重构顺序) ---
        # print("    - [阶段 1/6] 正在执行周期与基础过程诊断...")
        update_states(self.cyclical_intel.run_cyclical_analysis_command(df))
        base_process_states = self.process_intel.run_process_diagnostics(task_type_filter='base')
        update_states(base_process_states)
        # 阶段 1.5: 点燃关系动力引擎（解放普罗米修斯）
        # 这个引擎依赖过程信号，且必须在所有终极信号引擎之前运行
        self._ignite_relational_dynamics_engine()
        # --- 阶段二: 状态情报与战略过程诊断 ---
        # print("    - [阶段 2/6] 正在执行状态情报与战略过程诊断...")
        update_states(self.behavioral_intel.run_behavioral_analysis_command())
        update_states(self.foundation_intel.run_foundation_analysis_command())
        update_states(self.chip_intel.run_chip_intelligence_command(df))
        update_states(self.structural_intel.diagnose_structural_states(df))
        update_states(self.fund_flow_intel.diagnose_fund_flow_states(df))
        update_states(self.mechanics_engine.run_dynamic_analysis_command())
        update_states(self.pattern_intel.run_pattern_analysis_command(df))
        strategy_process_states = self.process_intel.run_process_diagnostics(task_type_filter='strategy')
        update_states(strategy_process_states)
        # --- 阶段三: 跨域认知融合 ---
        # print("    - [阶段 3/6] 正在执行认知层跨域元融合...")
        self.cognitive_intel.synthesize_cognitive_scores(df, pullback_enhancements={})
        # --- 阶段四: 先知引擎预测 ---
        # print("    - [阶段 4/6] 正在启动“先知引擎”进行风险预测...")
        update_states(self.predictive_intel.run_predictive_diagnostics())
        # --- 阶段五: 最终战法与剧本生成 ---
        # print("    - [阶段 5/6] 正在生成最终战法与剧本...")
        trigger_events = self.playbook_engine.define_trigger_events(df)
        self.strategy.trigger_events.update(trigger_events)
        _, playbook_states = self.playbook_engine.generate_playbook_states(self.strategy.trigger_events)
        self.strategy.playbook_states.update(playbook_states)
        # --- 阶段六: 硬性离场信号生成 ---
        # print("    - [阶段 6/6] 正在生成硬性离场信号...")
        exit_triggers_df = self.structural_defense_layer.generate_hard_exit_triggers()
        self.strategy.exit_triggers = exit_triggers_df
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        if get_param_value(debug_params.get('enabled'), False):
            self.deploy_forensic_probes()
        # print("--- [情报层总指挥官 V415.0] 所有诊断模块执行完毕。 ---")
        return self.strategy.trigger_events

    def deploy_forensic_probes(self):
        """
        【V2.1 · 双神谕探针版】法医探针调度中心
        - 核心升级: 新增对“神谕离场”信号的专属解剖探针。
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
            
        print("\n" + "="*30 + f" [法医探针部署中心 V2.1] 开始对 {len(probe_dates_list)} 个目标日期进行解剖... " + "="*30)

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
            
            # 部署双神谕探针
            # self._deploy_prophet_entry_probe(probe_date) # 将原函数重命名，权责更清晰
            # self._deploy_prophet_exit_probe(probe_date)  # 新增神谕离场探针
            
            # “宙斯之雷”终极探针
            self._deploy_zeus_thunderbolt_probe(probe_date)
        
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

    def _deploy_hephaestus_forge_probe(self, probe_date: pd.Timestamp, domain: str, signal_type: str):
        """
        【V1.2 · 雅典娜智慧版】“赫淮斯托斯熔炉”探针
        - 核心升级: 1. 移除了对隐藏衰减因子 `bottom_context_score` 的错误计算。
                    2. 新增对“雅典娜智慧”抑制因子的解剖，清晰展示底部反转信号的“荣誉退役”过程。
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_{signal_type}'
        print(f"\n--- [探针] 正在启用: 🔥【赫淮斯托斯熔炉】🔥 -> 解剖信号【{signal_name}】 ---")
        
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        
        def get_val(name, date, default=np.nan):
            series = atomic.get(name)
            if series is None: return default
            return series.get(date, default)

        # 链路层 1: 获取最终信号值
        final_score = get_val(signal_name, probe_date, 0.0)
        print(f"\n  [链路层 1] 最终锻造成品: {signal_name} = {final_score:.4f}")

        # 链路层 2: 反推到中央合成引擎的输出
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        reversal_tf_weights = get_param_value(p_synthesis.get('reversal_tf_weights'), {})
        bottom_context_bonus_factor = get_param_value(p_synthesis.get('bottom_context_bonus_factor'), 0.5)
        
        overall_health_cache = atomic.get(f'__{domain_upper}_overall_health', {})
        if not overall_health_cache:
            print("    - [探针错误] 无法找到领域健康度缓存。解剖终止。")
            return

        # 获取所有新的上下文因子
        recent_reversal_context = get_val('SCORE_CONTEXT_RECENT_REVERSAL', probe_date, 0.0)
        memory_retention_factor = 1.0 - get_val('CONTEXT_NEW_HIGH_STRENGTH', probe_date, 0.0)
        recent_reversal_context_modulated = recent_reversal_context * memory_retention_factor
        trend_confirmation_context = get_val('CONTEXT_TREND_CONFIRMED', probe_date, 0.0)
        
        # 模拟 transmute_health_to_ultimate_signals 的逻辑
        bullish_reversal_health = {p: recent_reversal_context_modulated * get_val('SCORE_ATOMIC_RELATIONAL_DYNAMICS', probe_date, 0.5) * overall_health_cache.get('d_intensity', {}).get(p, pd.Series(0.5)).get(probe_date, 0.5) for p in [1, 5, 13, 21, 55]}
        
        bullish_short_force_rev = (bullish_reversal_health.get(1, 0.5) * bullish_reversal_health.get(5, 0.5))**0.5
        bullish_medium_trend_rev = (bullish_reversal_health.get(13, 0.5) * bullish_reversal_health.get(21, 0.5))**0.5
        bullish_long_inertia_rev = bullish_reversal_health.get(55, 0.5)
        
        overall_bullish_reversal_trigger = ((bullish_short_force_rev ** reversal_tf_weights.get('short', 0.6)) * 
                                            (bullish_medium_trend_rev ** reversal_tf_weights.get('medium', 0.3)) * 
                                            (bullish_long_inertia_rev ** reversal_tf_weights.get('long', 0.1)))
        
        # 修正重算公式，移除隐藏的 bottom_context_score，并加入雅典娜抑制因子
        raw_recalc_score = (overall_bullish_reversal_trigger * (1 + recent_reversal_context_modulated * bottom_context_bonus_factor)).clip(0, 1)
        recalc_final_score = raw_recalc_score * (1 - trend_confirmation_context)

        print(f"\n  [链路层 2] 反推 -> 中央合成引擎 (utils.transmute_health_to_ultimate_signals)")
        print(f"    - [公式]: (原始分 * (1 - 趋势确认分))")
        print(f"    - [探针重算]: ({raw_recalc_score:.4f} * (1 - {trend_confirmation_context:.4f})) = {recalc_final_score:.4f}")
        print(f"    - [对比]: 实际值 {final_score:.4f} vs 重算值 {recalc_final_score:.4f}")
        print(f"    - [雅典娜的智慧] 🦉: “趋势确认分”为 {trend_confirmation_context:.2f}，导致底部反转信号被抑制了 {(trend_confirmation_context*100):.1f}%。")

        # 链路层5的公式也需要更新，以反映 modulated context
        print(f"\n  [链路层 5] 终极解剖 -> 1日健康度 ({bullish_reversal_health.get(1, 0.5):.4f})")
        print(f"    - [公式]: (反转回声 * 记忆保留因子) * 关系动力 * 动态强度")
        relational_power = get_val('SCORE_ATOMIC_RELATIONAL_DYNAMICS', probe_date, 0.5)
        d_intensity_1d = overall_health_cache.get('d_intensity', {}).get(1, pd.Series(0.5)).get(probe_date, 0.5)
        print(f"    - [探针重算]: ({recent_reversal_context:.4f} * {memory_retention_factor:.4f}) * {relational_power:.4f} * {d_intensity_1d:.4f} = {bullish_reversal_health.get(1, 0.5):.4f}")
        print(f"    - [阿波罗的日冕] ☀️: “新高强度分”为 {1-memory_retention_factor:.2f}，导致反转回声被削弱了 {((1-memory_retention_factor)*100):.1f}%。")
        
        print("\n--- “赫淮斯托斯熔炉”解剖完毕 ---")

    # 注入全新的“宙斯之雷”终极探针
    def _deploy_zeus_thunderbolt_probe(self, probe_date: pd.Timestamp):
        """
        【V2.2 · 忒弥斯天平协议版】“宙斯之雷”终极法医探针
        - 核心升级: 签署“忒弥斯天平协议”，在主探针执行前，调用全新的子探针 `_deploy_themis_scales_probe`，
                      彻底解剖并展示 bottom/top 上下文分数的每一个组成部分。
        """
        print("\n--- [探针] 正在召唤:⚡️【宙斯之雷 · 终极得分解剖探针.⚡️⚡    ---")
        # 在主审判前，先由忒弥斯天平公开称量所有证据
        self._deploy_themis_scales_probe(probe_date)
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        def get_val(name, date, default=0.0):
            series = atomic.get(name)
            if series is None: return default
            return series.get(date, default)
        # 1. 调取最终判决
        final_score = df.loc[probe_date].get('final_score', 0)
        final_signal = df.loc[probe_date].get('signal_type', '未知')
        print(f"\n  [链路层 1] 最终裁决")
        print(f"    - 【最终信号】: {final_signal}")
        print(f"    - 【最终得分】: {final_score:.0f}")
        # 2. 加载信号字典和参数
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        # 直接调用导入的公共函数，实现真正的独立验证
        atomic['strategy_instance_ref'] = self.strategy
        bottom_context, top_context = calculate_context_scores(df, atomic)
        trend_confirmation_context = calculate_trend_confirmation_context(df, p_synthesis.get('trend_confirmation_context_params', {}), p_synthesis.get('norm_window', 55))
        del atomic['strategy_instance_ref']
        bottom_context_val = bottom_context.get(probe_date, 0.0)
        top_context_val = top_context.get(probe_date, 0.0)
        trend_confirmation_val = trend_confirmation_context.get(probe_date, 0.0)
        active_offense = []
        active_risks = []
        # 3. 遍历所有可能的信号，重演审判过程
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            signal_value_raw = get_val(signal_name, probe_date, 0.0)
            if abs(signal_value_raw) < 1e-6:
                continue
            base_score = 0
            is_risk = meta.get('type') == 'risk'
            if is_risk:
                base_score = meta.get('penalty_weight', 0) * -1
            elif 'score' in meta:
                base_score = meta.get('score', 0)
            if abs(base_score) < 1e-6:
                continue
            signal_value_final = signal_value_raw
            explanation = f"原始值: {signal_value_raw:.4f} * 基础分: {base_score:.0f}"
            if 'BOTTOM_REVERSAL' in signal_name:
                signal_value_final = signal_value_raw * bottom_context_val * (1 - trend_confirmation_val)
                explanation = (f"原始值: {signal_value_raw:.4f} "
                               f"x 底部上下文: {bottom_context_val:.2f} "
                               f"x (1 - 趋势确认: {trend_confirmation_val:.2f}) "
                               f"* 基础分: {base_score:.0f}")
            elif 'TACTICAL_REVERSAL' in signal_name:
                pass
            elif 'TOP_REVERSAL' in signal_name and not is_risk:
                 signal_value_final = signal_value_raw * top_context_val
                 explanation = (f"原始值: {signal_value_raw:.4f} "
                                f"x 顶部上下文: {top_context_val:.2f} "
                                f"* 基础分: {base_score:.0f}")
            contribution = signal_value_final * base_score
            if abs(contribution) < 0.5:
                continue
            signal_info = {
                'name': meta.get('cn_name', signal_name),
                'raw_value': signal_value_raw,
                'final_value': signal_value_final,
                'base_score': base_score,
                'contribution': contribution,
                'explanation': explanation,
                'category': meta.get('category', '未知类别')
            }
            if is_risk:
                active_risks.append(signal_info)
            else:
                active_offense.append(signal_info)
        # 4. 按贡献度排序并展示
        active_offense.sort(key=lambda x: x['contribution'], reverse=True)
        active_risks.sort(key=lambda x: x['contribution'], reverse=False)
        print("\n  [链路层 2] 激活的进攻项 (按贡献度排序)")
        total_offense = 0
        if not active_offense:
            print("    - 当日无任何激活的进攻信号。")
        else:
            for item in active_offense:
                print(f"    - 【{item['name']}】: {item['contribution']:.0f}  ({item['explanation']})")
                total_offense += item['contribution']
        print(f"    ----------------------------------")
        print(f"    - 【进攻项总分】: {total_offense:.0f}")
        print("\n  [链路层 3] 激活的风险项 (按贡献度排序)")
        total_risk = 0
        if not active_risks:
            print("    - 当日无任何激活的风险信号。")
        else:
            for item in active_risks:
                print(f"    - 【{item['name']}】: {item['contribution']:.0f}  ({item['explanation']})")
                total_risk += item['contribution']
        print(f"    ----------------------------------")
        print(f"    - 【风险项总分】: {total_risk:.0f}")
        # 5. 终极对质
        print("\n  [链路层 4] 终极对质")
        recalculated_final_score = total_offense + total_risk
        print(f"    - [探针重算总分]: {total_offense:.0f} (进攻) + {total_risk:.0f} (风险) = {recalculated_final_score:.0f}")
        print(f"    - [对比]: 实际值 {final_score:.0f} vs 重算值 {recalculated_final_score:.0f}")
        print("\n--- “宙斯之雷”审查完毕 ---")

    def _deploy_themis_scales_probe(self, probe_date: pd.Timestamp):
        """
        【V1.12 · 哈迪斯面纱协议版】“忒弥斯天平”上下文解剖探针
        - 核心升级: 探针现在能解剖“冷却重置”信号，展示其所有判断细节，包括成交量对比。
        """
        print("\n--- [探针] 正在启用: ⚖️【忒弥斯天平 · 上下文解剖】⚖️ ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        strategy_instance_ref = self.strategy
        p_synthesis = get_params_block(strategy_instance_ref, 'ultimate_signal_synthesis_params', {})
        gaia_params = get_param_value(p_synthesis.get('gaia_bedrock_params'), {}) # 提前获取参数
        cooldown_reset_volume_ma_period = get_param_value(gaia_params.get('cooldown_reset_volume_ma_period'), 55)
        vol_ma_col = f'VOL_MA_{cooldown_reset_volume_ma_period}_D'
        print("\n  --- [结构性支撑审查] 关键均线系统快照 ---")
        ma_periods_to_probe = [5, 55, 144, 233, 377]
        close_price = df.get('close_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        low_price = df.get('low_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        high_price = df.get('high_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        volume = df.get('volume_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A') # 新增
        volume_ma = df.get(vol_ma_col, pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A') # 新增
        if isinstance(close_price, (float, np.floating)):
            print(f"    - {'high_D':<12}: {high_price:.2f}  (当日最高价)")
            print(f"    - {'close_D':<12}: {close_price:.2f}  (当日收盘价)")
            print(f"    - {'low_D':<12}: {low_price:.2f}  (当日最低价)")
            print(f"    - {'volume_D':<12}: {volume:,.0f}  (当日成交量)") # 新增
            print(f"    - {vol_ma_col:<12}: {volume_ma:,.0f}  (成交量均线)") # 新增
        else:
            # ... 省略N/A情况的打印 ...
            pass
        for period in ma_periods_to_probe:
            col_name = f'EMA_{period}_D'
            ma_value = df.get(col_name, pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
            if isinstance(ma_value, (float, np.floating)):
                print(f"    - {col_name:<12}: {ma_value:.2f}")
            else:
                print(f"    - {col_name:<12}: {ma_value}")
        print("\n  --- [天平左侧] 底部上下文解剖 ---")
        # ... 此处到盖亚显微镜之间的代码无变化，省略 ...
        depth_threshold = get_param_value(p_synthesis.get('deep_bearish_threshold'), 0.05)
        ma55_lifeline = df.get('EMA_55_D', df['close_D'])
        is_deep_bearish_zone = (df['close_D'] < ma55_lifeline * (1 - depth_threshold)).astype(float)
        ma55_slope = ma55_lifeline.diff(3).fillna(0)
        slope_moderator = (0.5 + 0.5 * np.tanh(ma55_slope * 100)).fillna(0.5)
        distance_from_ma55 = (df['close_D'] - ma55_lifeline) / ma55_lifeline.replace(0, np.nan)
        lifeline_support_score_raw = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2).fillna(0.0)
        lifeline_support_score = lifeline_support_score_raw * slope_moderator
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        absolute_value_zone_score = 1.0 - price_pos_yearly
        deep_bottom_context_score = np.maximum(lifeline_support_score, absolute_value_zone_score)
        rsi_w_oversold_score = normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), df.index, window=52, ascending=False, default_value=0.5)
        cycle_phase = atomic.get('DOMINANT_CYCLE_PHASE', pd.Series(0.0, index=df.index)).fillna(0.0)
        cycle_trough_score = (1 - cycle_phase) / 2.0
        context_weights = get_param_value(p_synthesis.get('bottom_context_weights'), {'price_pos': 0.5, 'rsi_w': 0.3, 'cycle': 0.2})
        bottom_context_score_raw = (deep_bottom_context_score**context_weights['price_pos'] * rsi_w_oversold_score**context_weights['rsi_w'] * cycle_trough_score**context_weights['cycle'])
        conventional_bottom_score = bottom_context_score_raw * is_deep_bearish_zone
        print(f"    - [组件1] 常规底部得分 (经深度熊市过滤): {conventional_bottom_score.get(probe_date, 0.0):.4f}")
        gaia_bedrock_support_score = _calculate_gaia_bedrock_support(df, gaia_params)
        print(f"    - [组件2] 盖亚基石支撑分: {gaia_bedrock_support_score.get(probe_date, 0.0):.4f}")
        print("      --- [盖亚显微镜] 深入解剖 ---")
        # [代码修改] 探针同步新的冷却重置逻辑
        support_levels = get_param_value(gaia_params.get('support_levels'), [55, 89, 144, 233, 377])
        confirmation_window = get_param_value(gaia_params.get('confirmation_window'), 3)
        aegis_lookback_window = get_param_value(gaia_params.get('aegis_lookback_window'), 5)
        confirmation_cooldown_period = get_param_value(gaia_params.get('confirmation_cooldown_period'), 10)
        influence_zone_pct = get_param_value(gaia_params.get('influence_zone_pct'), 0.03)
        lower_shadow_strength_pct = get_param_value(gaia_params.get('lower_shadow_strength_pct'), 0.01)
        g_ma_cols = [f'EMA_{p}_D' for p in support_levels if f'EMA_{p}_D' in df.columns]
        g_ma_df = df[g_ma_cols]
        g_ma_df_below_price = g_ma_df.where(g_ma_df.le(df['close_D'], axis=0))
        g_acting_lifeline = g_ma_df_below_price.max(axis=1).ffill()
        g_valid_indices = g_acting_lifeline.dropna().index
        g_is_in_influence_zone = pd.Series(False, index=df.index)
        g_upper_bound = g_acting_lifeline[g_valid_indices] * (1 + influence_zone_pct)
        g_is_in_influence_zone.loc[g_valid_indices] = df.loc[g_valid_indices, 'close_D'].between(g_acting_lifeline[g_valid_indices], g_upper_bound)
        g_defense_quality_score = pd.Series(0.0, index=df.index)
        g_touched_lifeline = (df['low_D'] < g_acting_lifeline) & g_is_in_influence_zone
        g_has_lower_shadow = df['close_D'] > df['low_D']
        g_base_defense_condition = g_touched_lifeline & g_has_lower_shadow
        g_lower_shadow = df['close_D'] - df['low_D']
        g_upper_shadow = df['high_D'] - df['close_D']
        g_is_cooldown_reset_signal = (g_upper_shadow > g_lower_shadow) & (df['volume_D'] > df[vol_ma_col])
        g_was_recently_defended = (g_base_defense_condition).rolling(window=aegis_lookback_window, min_periods=1).sum() > 0
        g_is_standing_firm = (df['close_D'] > g_acting_lifeline).astype(float)
        g_is_standing_firm_in_zone = g_is_standing_firm.copy()
        g_is_standing_firm_in_zone.loc[~g_is_in_influence_zone] = np.nan
        g_is_confirmed_base = g_is_standing_firm_in_zone.rolling(window=confirmation_window, min_periods=confirmation_window).sum() >= confirmation_window
        g_is_in_cooldown = False
        g_last_confirmation_date = pd.NaT
        for idx in g_valid_indices:
            if idx > probe_date: break
            if pd.notna(g_last_confirmation_date) and (idx - g_last_confirmation_date).days < confirmation_cooldown_period:
                if idx == probe_date: g_is_in_cooldown = True
                if g_is_cooldown_reset_signal.get(idx, False):
                    g_last_confirmation_date = pd.NaT
                continue
            if g_is_confirmed_base.get(idx, False):
                g_last_confirmation_date = idx
        print(f"        - acting_lifeline (代理总指挥): {g_acting_lifeline.get(probe_date, np.nan):.4f}")
        print(f"        - is_confirmed_base (是否满足站稳天数): {g_is_confirmed_base.get(probe_date, False)}")
        print(f"        - was_recently_defended (近期是否防守过): {g_was_recently_defended.get(probe_date, False)}")
        print(f"        - is_in_cooldown (是否处于冷却期): {g_is_in_cooldown}")
        print("        --- [防守质量解剖 (三层迷宫)] ---")
        # ... 此处到下一个修改点之间的代码无变化，省略 ...
        lifeline_val = g_acting_lifeline.get(probe_date, np.nan)
        low_val = df.get('low_D').get(probe_date, np.nan)
        close_val = df.get('close_D').get(probe_date, np.nan)
        high_val = df.get('high_D').get(probe_date, np.nan)
        in_zone = g_is_in_influence_zone.get(probe_date, False)
        touched = g_touched_lifeline.get(probe_date, False)
        has_ls = g_has_lower_shadow.get(probe_date, False)
        strength_condition = (g_lower_shadow / df['close_D'].replace(0, np.nan)) > lower_shadow_strength_pct
        strength = strength_condition.get(probe_date, False)
        dominance_condition = g_lower_shadow > g_upper_shadow
        dominance = dominance_condition.get(probe_date, False)
        lower_shadow_val = g_lower_shadow.get(probe_date, 0)
        upper_shadow_val = g_upper_shadow.get(probe_date, 0)
        strength_pct = (lower_shadow_val / close_val) * 100 if close_val > 0 else 0
        print(f"          - 前提: is_in_influence_zone -> {in_zone}")
        print(f"          - Tier 1 (基础): (low < lifeline -> {touched}) AND (close > low -> {has_ls})")
        print(f"          - Tier 2 (力量): (c-l)/c > {lower_shadow_strength_pct:.0%} -> {strength_pct:.2f}% > {lower_shadow_strength_pct:.0%} -> {strength}")
        print(f"          - Tier 3 (优势): (c-l) > (h-c) -> {lower_shadow_val:.2f} > {upper_shadow_val:.2f} -> {dominance}")
        # [代码新增] 增加冷却重置信号的解剖
        print("        --- [冷却重置解剖 (哈迪斯面纱)] ---")
        reset_cond1 = g_upper_shadow.get(probe_date, 0) > g_lower_shadow.get(probe_date, 0)
        reset_cond2 = df.get('volume_D').get(probe_date, 0) > df.get(vol_ma_col).get(probe_date, np.inf)
        print(f"          - 条件1 (上影优势): upper_shadow > lower_shadow -> {reset_cond1}")
        print(f"          - 条件2 (成交放量): volume > {vol_ma_col} -> {reset_cond2}")
        print(f"          - 综合判定 (is_cooldown_reset): {g_is_cooldown_reset_signal.get(probe_date, False)}")
        print("      --------------------------")
        # ... 此处到函数结尾的代码无变化，省略 ...
        p_fib_support = get_params_block(strategy_instance_ref, 'fibonacci_support_params', {})
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
        ma55 = df.get('EMA_55_D', df['close_D'])
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        wave_channel_height = (rolling_high_55d - ma55).replace(0, 1e-9)
        stretch_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
        ma_periods = [5, 13, 21, 55]
        short_ma_cols = [f'EMA_{p}_D' for p in ma_periods[:-1]]
        long_ma_cols = [f'EMA_{p}_D' for p in ma_periods[1:]]
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
        print(f"    - [组件1] 价格拉伸分: {stretch_score.get(probe_date, 0.0):.4f}")
        print(f"    - [组件2] 均线混乱分: {misalignment_score.get(probe_date, 0.0):.4f}")
        print(f"    - [组件3] 乖离过热分 (绝对值): {overheat_score.get(probe_date, 0.0):.4f} (原始BIAS: {bias_abs.get(probe_date, 0.0):.2%})")
        print(f"    - [最终裁决] 顶部上下文总分: {top_context.get(probe_date, 0.0):.4f}")
        print("\n--- “忒弥斯天平”称量完毕 ---")










