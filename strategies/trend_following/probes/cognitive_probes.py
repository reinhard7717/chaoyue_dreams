import pandas as pd
import numpy as np
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, get_unified_score

class CognitiveProbes:
    """
    【探针模块】认知情报专属探针
    """
    def __init__(self, intel_layer):
        self.intelligence_layer = intel_layer
        self.strategy = intel_layer.strategy
        self.cognitive_intel = intel_layer.cognitive_intel

    def _deploy_thanatos_scythe_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 时空同步版】“塔纳托斯之镰”探针 - 真实撤退风险全要素解剖
        - 核心修复: 完全同步主引擎V5.3版的逻辑，确保探针与主引擎使用相同的“盈利盘信念”数据源进行重算。
        """
        print("\n--- [探针] 正在启用: 💀【真实撤退风险解剖 V1.1】💀 ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        norm_window = 55
        p = 5
        def get_val(series, date, default=np.nan):
            if series is None or not isinstance(series, (pd.Series, dict)): return default
            if isinstance(series, dict): return series.get(date, default)
            return series.get(date, default)
        print("\n  [链路层 1] 最终裁决 (Final Verdict)")
        final_score = get_val(atomic.get('COGNITIVE_SCORE_TRUE_RETREAT_RISK'), probe_date, 0.0)
        print(f"    - 【最终风险值】: {final_score:.4f}")
        print("\n  [链路层 2] 前提条件 (Premise): 近期派发强度")
        to_main = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True) * normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True))**0.5
        to_retail = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False) * normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False))**0.5
        short_term_transfer_snapshot = (to_main - to_retail).astype(np.float32)
        recent_distribution_strength = (short_term_transfer_snapshot.rolling(3).mean().clip(-1, 0) * -1).astype(np.float32)
        print(f"    - 【近期派发强度】: {get_val(recent_distribution_strength, probe_date):.4f}")
        print("\n  [链路层 3] 证据链 (Evidence Chain): 真实撤退的四大迹象")
        trend_quality_context = get_val(atomic.get('COGNITIVE_SCORE_TREND_QUALITY'), probe_date, 0.0)
        trend_decay_context = 1.0 - trend_quality_context
        panic_absorption_score = get_val(atomic.get('SCORE_MICRO_PANIC_ABSORPTION'), probe_date, 0.0)
        no_absorption_score = 1.0 - panic_absorption_score
        winner_conviction_0_1 = (get_val(atomic.get('PROCESS_META_WINNER_CONVICTION'), probe_date, 0.0) * 0.5 + 0.5)
        winner_capitulation_score = (1.0 - winner_conviction_0_1) ** 0.7
        dyn_bullish_resonance = get_val(atomic.get('SCORE_DYN_BULLISH_RESONANCE'), probe_date, 0.0)
        behavior_bullish_resonance = get_val(atomic.get('SCORE_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        reversal_dynamic_quality = (dyn_bullish_resonance * behavior_bullish_resonance)**0.5
        bull_trap_evidence = 1.0 - reversal_dynamic_quality
        retreat_evidence_chain = (trend_decay_context * no_absorption_score * winner_capitulation_score * bull_trap_evidence)
        print(f"    - 【证据链总强度】: {retreat_evidence_chain:.4f}")
        print("\n--- “真实撤退风险探针”解剖完毕 ---")

    def _deploy_profit_taking_pressure_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0】利润兑现压力风险探针
        - 核心使命: 深度解剖重构后的 COGNITIVE_RISK_PROFIT_TAKING_PRESSURE 信号。
        - 解剖链路: 1. 最终风险分 -> 2. 跨周期融合 -> 3. 单周期元分析 -> 4. 核心证据快照分 -> 5. 四维质量评估。
        """
        print("\n" + "="*35 + f" [认知探针] 正在启用 💸【利润兑现压力风险探针 V1.0】💸 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel.micro_behavior_engine # 引擎实例现在位于微观行为引擎下
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_RISK_PROFIT_TAKING_PRESSURE'
        # 链路层 1: 最终输出
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        # 链路层 2: 核心证据四维质量评估 (以 p=5 周期为例)
        print("\n  [链路层 2] 核心证据四维质量评估 (p=5 周期为例)")
        p_tactical, p_context = 5, 13
        # 证据一: 了结紧迫度
        urgency_quality_series = engine._calculate_4d_metric_quality(df, 'profit_taking_urgency_D', p_tactical, p_context, ascending=True)
        urgency_quality_val = get_val(urgency_quality_series, probe_date)
        print(f"    - [证据一: 了结紧迫度] 质量分: {urgency_quality_val:.4f}")
        # 证据二: 兑现溢价
        premium_quality_series = engine._calculate_4d_metric_quality(df, 'profit_realization_premium_D', p_tactical, p_context, ascending=True)
        premium_quality_val = get_val(premium_quality_series, probe_date)
        print(f"    - [证据二: 兑现溢价] 质量分: {premium_quality_val:.4f}")
        # 链路层 3: 快照分融合
        print("\n  [链路层 3] 快照分融合 (p=5 周期)")
        snapshot_score_val = (urgency_quality_val * premium_quality_val)**0.5
        print(f"    - [融合公式]: (紧迫度质量 * 溢价质量) ** 0.5")
        print(f"    - 【探针重算快照分】: ({urgency_quality_val:.4f} * {premium_quality_val:.4f})**0.5 = {snapshot_score_val:.4f}")
        # 链路层 4: 关系元分析
        print("\n  [链路层 4] 关系元分析 (p=5 周期)")
        snapshot_series = (urgency_quality_series * premium_quality_series)**0.5
        recalc_period_score_series = engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_series)
        recalc_period_score_val = get_val(recalc_period_score_series, probe_date)
        print(f"    - 【探针重算周期风险分】: {recalc_period_score_val:.4f}")
        # 链路层 5: 跨周期融合
        print("\n  [链路层 5] 跨周期融合")
        # 重新执行完整的计算以获得所有周期的分数
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        pressure_scores_by_period = {}
        for i, p_tac in enumerate(sorted_periods):
            p_con = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tac
            urgency_q = engine._calculate_4d_metric_quality(df, 'profit_taking_urgency_D', p_tac, p_con, ascending=True)
            premium_q = engine._calculate_4d_metric_quality(df, 'profit_realization_premium_D', p_tac, p_con, ascending=True)
            snapshot_s = (urgency_q * premium_q)**0.5
            pressure_scores_by_period[p_tac] = engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_s)
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        recalc_final_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        calc_str = []
        for p in periods:
            weight = tf_weights.get(p, 0) / total_weight
            score_val = get_val(pressure_scores_by_period.get(p), probe_date, 0.0)
            recalc_final_score += score_val * weight
            calc_str.append(f"({score_val:.2f}*{weight:.2f})")
        print(f"    - [融合公式]: Σ (周期分 * 权重)")
        print(f"    - [计算过程]: {' + '.join(calc_str)} = {recalc_final_score.get(probe_date):.4f}")
        print("\n  [链路层 6] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针正确值 {recalc_final_score.get(probe_date):.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_final_score.get(probe_date)) else '❌ 不一致'}")
        print("\n--- “利润兑现压力风险探针”解剖完毕 ---")

    def _deploy_comprehensive_top_risk_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V2.1 · 神盾解剖版】综合顶部风险探针
        - 核心升级: 增强对“趋势韧性神盾”的解剖能力，清晰展示其“静态韧性”和“动态韧性”的计算过程。
        """
        print("\n" + "="*25 + f" [认知探针] 正在启用 🛡️【综合顶部风险探针 V2.1】🛡️ " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_RISK_COMPREHENSIVE_TOP'
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        print("\n  [链路层 2] 三柱风险分析 (Tri-Pillar Risk Analysis)")
        euphoric_signals = {
            'EUPHORIC_ACCELERATION': engine._get_atomic_score(df, 'COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION', 0.0),
            'ICARUS_FALL': engine._get_atomic_score(df, 'SCORE_RISK_ICARUS_FALL', 0.0),
            'BOARD_HEAVEN_EARTH': engine._get_atomic_score(df, 'SCORE_BOARD_HEAVEN_EARTH', 0.0),
        }
        euphoric_scores = {name: get_val(s, probe_date) for name, s in euphoric_signals.items()}
        euphoric_risk_score = max(euphoric_scores.values()) if euphoric_scores else 0.0
        print(f"    - [支柱 I: 亢奋/高潮风险] -> 得分: {euphoric_risk_score:.4f}")
        distribution_signals = {
            'MAIN_FORCE_INTENT_DUEL': engine._get_atomic_score(df, 'COGNITIVE_RISK_MAIN_FORCE_HIGH_COST_VS_DISTRIBUTION', 0.0),
            'UPTHRUST_DISTRIBUTION': engine._get_atomic_score(df, 'SCORE_RISK_UPTHRUST_DISTRIBUTION', 0.0),
            'RETAIL_FOMO_RETREAT': engine._get_atomic_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_MAIN_FORCE_RETREAT', 0.0),
            'TRUE_RETREAT': engine._get_atomic_score(df, 'COGNITIVE_SCORE_TRUE_RETREAT_RISK', 0.0),
        }
        distribution_scores = {name: get_val(s, probe_date) for name, s in distribution_signals.items()}
        distribution_risk_score = max(distribution_scores.values()) if distribution_scores else 0.0
        print(f"    - [支柱 II: 派发/背叛风险] -> 得分: {distribution_risk_score:.4f}")
        structural_signals = {
            'CONTEXT_TOP': engine._get_atomic_score(df, 'CONTEXT_TOP_SCORE', 0.0),
            'CYCLICAL_TOP': engine._get_atomic_score(df, 'COGNITIVE_RISK_CYCLICAL_TOP', 0.0),
        }
        structural_scores = {name: get_val(s, probe_date) for name, s in structural_signals.items()}
        structural_risk_score = max(structural_scores.values()) if structural_scores else 0.0
        print(f"    - [支柱 III: 结构/周期风险] -> 得分: {structural_risk_score:.4f}")
        print("\n  [链路层 3] 原始风险融合 (Raw Risk Fusion)")
        recalc_raw_risk = max(euphoric_risk_score, distribution_risk_score, structural_risk_score)
        print(f"    - 【探针重算原始风险】: max({euphoric_risk_score:.2f}, {distribution_risk_score:.2f}, {structural_risk_score:.2f}) = {recalc_raw_risk:.4f}")
        print("\n  [链路层 4] 趋势韧性神盾解剖 (Aegis Shield Dissection)")
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p_shield = get_param_value(p_cognitive.get('trend_resilience_shield_params'), {})
        weights = get_param_value(p_shield.get('fusion_weights'), {})
        pillars = {
            'trend_quality': engine._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0),
            'structural_health': engine._get_atomic_score(df, 'SCORE_STRUCTURE_BULLISH_RESONANCE', 0.0),
            'fund_flow_health': engine._get_atomic_score(df, 'SCORE_FF_BULLISH_RESONANCE', 0.0),
            'chip_health': engine._get_atomic_score(df, 'SCORE_CHIP_BULLISH_RESONANCE', 0.0)
        }
        static_resilience_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        total_weight = sum(weights.get(name, 0) for name in pillars.keys())
        if total_weight > 0:
            for name, score_series in pillars.items():
                weight = weights.get(name, 0.25)
                static_resilience_series += score_series * (weight / total_weight)
                print(f"    - [静态支柱: {name}] 得分: {get_val(score_series, probe_date):.4f}, 权重贡献: {(get_val(score_series, probe_date) * (weight / total_weight)):.4f}")
        recalc_static_resilience = get_val(static_resilience_series, probe_date)
        print(f"    - 【静态韧性分】: {recalc_static_resilience:.4f}")
        p_meta = get_param_value(p_cognitive.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window, meta_window = 55, 5
        velocity_score = normalize_to_bipolar(static_resilience_series.diff(meta_window).fillna(0), df.index, norm_window)
        acceleration_score = normalize_to_bipolar(static_resilience_series.diff(meta_window).fillna(0).diff(meta_window).fillna(0), df.index, norm_window)
        recalc_dynamic_bonus = (get_val(velocity_score, probe_date, 0.0) * w_velocity + get_val(acceleration_score, probe_date, 0.0) * w_acceleration)
        print(f"    - 【动态韧性加成】: {recalc_dynamic_bonus:.4f}")
        recalc_shield_score = (recalc_static_resilience * (1 + recalc_dynamic_bonus)).clip(0, 1)
        print(f"    - 【探针重算神盾总分】: {recalc_static_resilience:.4f} * (1 + {recalc_dynamic_bonus:.4f}) = {recalc_shield_score:.4f}")
        print("\n  [链路层 5] 最终风险裁决 (Final Risk Adjudication)")
        recalc_final_score = recalc_raw_risk * (1.0 - recalc_shield_score)
        print(f"    - 【探针重算最终风险】: {recalc_raw_risk:.4f} * (1.0 - {recalc_shield_score:.4f}) = {recalc_final_score:.4f}")
        print("\n  [链路层 6] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针正确值 {recalc_final_score:.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_final_score) else '❌ 不一致'}")
        print("\n--- “综合顶部风险探针”解剖完毕 ---")

    def _deploy_main_force_intent_duel_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V4.0 · 全息意图对决版】主力意图对决风险探针
        - 核心重构: 同步主引擎的MTF分析逻辑，解剖从“分周期对决”到“全息融合”的完整链路。
        """
        print("\n" + "="*25 + f" [认知探针] 正在启用 ⚔️【主力意图对决风险探针 V4.0】⚔️ " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_RISK_MAIN_FORCE_HIGH_COST_VS_DISTRIBUTION'
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        periods = get_param_value(p_cognitive.get('expansion_engine_periods'), [1, 5, 13, 21, 55])
        tf_weights = get_param_value(p_cognitive.get('expansion_engine_tf_weights'), {})
        numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_tf_weights.values())
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        print("\n  [链路层 2] 分周期意图对决 (Per-Period Intent Duel)")
        bipolar_intent_by_period = {}
        urgency_score_raw_series = engine._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_URGENCY', 0.0)
        distribution_score_raw_series = df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index))
        for p in periods:
            urgency_norm = get_val(normalize_score(urgency_score_raw_series, df.index, p), probe_date, 0.0)
            distribution_norm = get_val(normalize_score(distribution_score_raw_series, df.index, p), probe_date, 0.0)
            bipolar_intent = urgency_norm - distribution_norm
            bipolar_intent_by_period[p] = bipolar_intent
            print(f"    - [周期 {p}d]: 决心分 {urgency_norm:.2f} - 背叛分 {distribution_norm:.2f} = 净意图 {bipolar_intent:.4f}")
        print("\n  [链路层 3] 全息意图融合 (Holographic Intent Fusion)")
        recalc_final_bipolar_intent = 0.0
        calc_str_list = []
        if total_weight > 0:
            for p in periods:
                weight = numeric_tf_weights.get(p, 0) / total_weight
                intent_score = bipolar_intent_by_period.get(p, 0.0)
                recalc_final_bipolar_intent += intent_score * weight
                calc_str_list.append(f"({intent_score:.2f}*{weight:.2f})")
        print(f"    - [融合公式]: Σ (净意图_p * 权重_p)")
        print(f"    - [计算过程]: {' + '.join(calc_str_list)} = {recalc_final_bipolar_intent:.4f}")
        print("\n  [链路层 4] 风险转化 (Risk Transformation)")
        recalc_final_score = -min(0, recalc_final_bipolar_intent)
        print(f"    - [风险公式]: -min(0, 综合净意图)")
        print(f"    - 【探针重算最终风险分】: {recalc_final_score:.4f}")
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针正确值 {recalc_final_score:.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_final_score) else '❌ 不一致'}")
        print("\n--- “主力意图对决风险探针”解剖完毕 ---")

    def _deploy_trend_quality_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.1 · 算术平均同步版】趋势质量探针
        - 核心修复: 同步主引擎的融合算法变更，将探针重算“领域共识分”的逻辑从
                      过时的几何平均改为与主引擎一致的加权算术平均。
        """
        print("\n" + "="*25 + f" [认知探针] 正在启用 📈【趋势质量探针 V1.1】📈 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_SCORE_TREND_QUALITY'
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终信号分】: {actual_final_score:.4f}")
        print("\n  [链路层 2 & 3] 快照分重算 (Snapshot Recalculation)")
        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = p.get('domain_weights', {})
        direct_trend_health_series = engine._calculate_cognitive_trend_health(df)
        direct_trend_health_val = get_val(direct_trend_health_series, probe_date)
        print(f"    - [组件 I: 直接趋势健康度] -> 得分: {direct_trend_health_val:.4f}")
        domain_scores_map = {
            'behavior': 1.0 - get_unified_score(atomic, df.index, 'STRUCT_BEHAVIOR_TOP_REVERSAL'),
            'chip': get_unified_score(atomic, df.index, 'CHIP_BULLISH_RESONANCE'),
            'fund_flow': get_unified_score(atomic, df.index, 'FF_BULLISH_RESONANCE'),
            'structural': get_unified_score(atomic, df.index, 'STRUCTURE_BULLISH_RESONANCE'),
            'mechanics': get_unified_score(atomic, df.index, 'DYN_BULLISH_RESONANCE'),
            'regime': (engine._get_atomic_score(df, 'SCORE_TRENDING_REGIME') * engine._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT'))**0.5,
            'cyclical': 1.0 - engine._get_atomic_score(df, 'SCORE_CYCLICAL_REGIME')
        }
        # --- 使用加权算术平均数进行重算 ---
        recalc_consensus_val = 0.0
        total_weight = 0.0
        valid_weights = {name: w for name, w in weights.items() if w > 0 and name in domain_scores_map}
        total_weight = sum(valid_weights.values())
        if total_weight > 0:
            for name, score_series in domain_scores_map.items():
                weight = valid_weights.get(name, 0)
                if weight > 0:
                    score_val = get_val(score_series, probe_date, 0.5)
                    recalc_consensus_val += score_val * (weight / total_weight)
        else:
            recalc_consensus_val = 0.5
        print(f"    - [组件 II: 领域共识分] -> 得分: {recalc_consensus_val:.4f}")
        recalc_snapshot_score = (direct_trend_health_val * recalc_consensus_val)**0.5
        print(f"    - 【探针重算快照总分】: ({direct_trend_health_val:.4f} * {recalc_consensus_val:.4f})**0.5 = {recalc_snapshot_score:.4f}")
        print("\n  [链路层 4] 领域共识分解剖 (Component Dissection)")
        for name, series in domain_scores_map.items():
            weight = weights.get(name, 0)
            if weight > 0:
                score = get_val(series, probe_date)
                print(f"      - [领域: {name:<12}] 得分: {score:.4f}, 权重: {weight}")
        print("\n--- “趋势质量探针”解剖完毕 ---")

    def _deploy_liquidity_trap_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.1 · 信号同步版】穿透式解剖 COGNITIVE_RISK_LIQUIDITY_TRAP 信号
        - 核心修复: 将引用的信号名从旧的 'SCORE_RISK_LIQUIDITY_DRAIN' 同步为新的 'SCORE_RISK_LIQUIDITY_VACUUM'。
        """
        print("\n" + "="*25 + f" [认知探针] 正在启用 💧【流动性陷阱探针 V1.1】💧 " + "="*25)
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        signal_name = 'COGNITIVE_RISK_LIQUIDITY_TRAP'
        
        def get_val(series, date, default=0.0):
            val = series.get(date)
            return default if pd.isna(val) else val

        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        system_score = get_val(atomic_states.get(signal_name, pd.Series(0.0, index=df.index)), probe_date)
        print(f"    - 【最终信号分】: {system_score:.4f}")

        print("\n  [链路层 2] 快照分重算 (Snapshot Recalculation)")
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        periods = get_param_value(p_cognitive.get('expansion_engine_periods'), [1, 5, 13, 21, 55])
        tf_weights = get_param_value(p_cognitive.get('expansion_engine_tf_weights'), {})
        numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_tf_weights.values())

        capital_flight_raw = df.get('main_force_net_flow_consensus_sum_5d_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs()
        capital_flight_fused = pd.Series(0.0, index=df.index)
        if total_weight > 0:
            for p in periods:
                weight = numeric_tf_weights.get(p, 0) / total_weight
                capital_flight_fused += normalize_score(capital_flight_raw, df.index, p) * weight
        
        # [代码修改开始]
        # 证据二：流动性真空 - 使用正确的信号名称
        liquidity_vacuum_fused = atomic_states.get('SCORE_RISK_LIQUIDITY_VACUUM', pd.Series(0.0, index=df.index))
        # [代码修改结束]

        buyer_apathy_raw = 1.0 - normalize_score(df.get('realized_support_intensity_D', pd.Series(0.0, index=df.index)), df.index, 55)
        buyer_apathy_fused = pd.Series(0.0, index=df.index)
        if total_weight > 0:
            for p in periods:
                weight = numeric_tf_weights.get(p, 0) / total_weight
                buyer_apathy_fused += normalize_score(buyer_apathy_raw, df.index, p) * weight

        probe_snapshot_score = (capital_flight_fused * liquidity_vacuum_fused * buyer_apathy_fused)**(1/3)
        probe_snapshot_val = get_val(probe_snapshot_score, probe_date)
        print(f"    - 【探针重算快照分】: {probe_snapshot_val:.4f}")

        print("\n  [链路层 3] 关系元分析 (Relational Meta-Analysis)")
        probe_dynamic_score = self.intelligence_layer.cognitive_intel._perform_cognitive_relational_meta_analysis(df, probe_snapshot_score)
        probe_dynamic_val = get_val(probe_dynamic_score, probe_date)
        print(f"    - 【探针重算动态分】: {probe_dynamic_val:.4f}")

        print("\n  [链路层 4] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {system_score:.4f} vs. 探针重算值 {probe_dynamic_val:.4f} -> {'✅ 一致' if np.isclose(system_score, probe_dynamic_val) else '❌ 不一致'}")

        print("\n  [链路层 5] 证据链分解 (Component Dissection)")
        print(f"    - [证据一: 主力持续出逃] 原始值: {get_val(capital_flight_raw, probe_date):.2f}, 融合归一化后: {get_val(capital_flight_fused, probe_date):.4f}")
        print(f"    - [证据二: 流动性真空] 融合归一化后: {get_val(liquidity_vacuum_fused, probe_date):.4f}")
        print(f"    - [证据三: 买盘真空] 原始值: {get_val(buyer_apathy_raw, probe_date):.4f}, 融合归一化后: {get_val(buyer_apathy_fused, probe_date):.4f}")
        
        print("\n--- “流动性陷阱探针”解剖完毕 ---")










