# 文件: strategies/trend_following/forensic_probes.py
# 法医探针集合
import pandas as pd
import numpy as np
import pandas_ta as ta
import json
from typing import Dict
from strategies.trend_following.utils import get_params_block, calculate_holographic_dynamics, get_param_value, calculate_context_scores, normalize_score, normalize_to_bipolar, _calculate_gaia_bedrock_support, _calculate_historical_low_support, get_unified_score

class ForensicProbes:
    """
    【V1.1 · 依赖重铸版】法医探针集合
    - 核心修正: 构造函数现在接收 IntelligenceLayer 实例，解决了因错误的对象引用导致的 AttributeError。
    - 架构意义: 建立了清晰的依赖关系，探针集合现在是情报层的直接附属，可以访问其所有子模块。
    """
    def __init__(self, intelligence_layer_instance):
        # 接收 intelligence_layer_instance 而非 strategy_instance
        self.strategy = intelligence_layer_instance.strategy
        # 探针可能需要访问认知引擎等子模块，通过 intelligence_layer_instance 传递
        self.cognitive_intel = intelligence_layer_instance.cognitive_intel
        # 为新的筹码探针获取 chip_intel 引用
        self.chip_intel = intelligence_layer_instance.chip_intel

    def _deploy_thanatos_scythe_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 时空同步版】“塔纳托斯之镰”探针 - 真实撤退风险全要素解剖
        - 核心修复: 完全同步主引擎V5.3版的逻辑，确保探针与主引擎使用相同的“盈利盘信念”数据源进行重算。
        """
        print("\n--- [探针] 正在启用: 💀【塔纳托斯之镰 · 真实撤退风险解剖 V1.1】💀 ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        norm_window = 55
        p = 5
        def get_val(series, date, default=np.nan):
            if series is None or not isinstance(series, (pd.Series, dict)): return default
            if isinstance(series, dict): return series.get(date, default)
            return series.get(date, default)
        # --- 链路层 1: 最终裁决 ---
        print("\n  [链路层 1] 最终裁决 (Final Verdict)")
        final_score = get_val(atomic.get('COGNITIVE_SCORE_TRUE_RETREAT_RISK'), probe_date, 0.0)
        print(f"    - 【最终风险值】: {final_score:.4f}")
        print(f"    - [核心公式]: 真实撤退风险 = 近期派发强度 * 真实撤退证据链")
        # --- 链路层 2: 前提条件 - 近期派发强度 ---
        print("\n  [链路层 2] 前提条件 (Premise): 近期派发强度")
        to_main = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True) *
                   normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True))**0.5
        to_retail = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False) *
                     normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False))**0.5
        short_term_transfer_snapshot = (to_main - to_retail).astype(np.float32)
        recent_distribution_strength = (short_term_transfer_snapshot.rolling(3).mean().clip(-1, 0) * -1).astype(np.float32)
        print(f"    - 【近期派发强度】: {get_val(recent_distribution_strength, probe_date):.4f}")
        print(f"    - [钻透]:")
        print(f"      - 当日短期筹码转移快照: {get_val(short_term_transfer_snapshot, probe_date):.4f}")
        # --- 链路层 3: 证据链 - 真实撤退证据链 ---
        print("\n  [链路层 3] 证据链 (Evidence Chain): 真实撤退的四大迹象")
        trend_quality_context = get_val(atomic.get('COGNITIVE_SCORE_TREND_QUALITY'), probe_date, 0.0)
        trend_decay_context = 1.0 - trend_quality_context
        panic_absorption_score = get_val(atomic.get('SCORE_MICRO_PANIC_ABSORPTION'), probe_date, 0.0)
        no_absorption_score = 1.0 - panic_absorption_score
        # 修改开始(V1.1): 与主引擎V5.3同步，使用相同的源计算“赢家投降分”
        winner_conviction_0_1 = (get_val(atomic.get('PROCESS_META_WINNER_CONVICTION'), probe_date, 0.0) * 0.5 + 0.5)
        winner_capitulation_score = (1.0 - winner_conviction_0_1) ** 0.7
        # 修改结束(V1.1)
        dyn_bullish_resonance = get_val(atomic.get('SCORE_DYN_BULLISH_RESONANCE'), probe_date, 0.0)
        behavior_bullish_resonance = get_val(atomic.get('SCORE_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        reversal_dynamic_quality = (dyn_bullish_resonance * behavior_bullish_resonance)**0.5
        bull_trap_evidence = 1.0 - reversal_dynamic_quality
        retreat_evidence_chain = (trend_decay_context * no_absorption_score * winner_capitulation_score * bull_trap_evidence)
        print(f"    - 【证据链总强度】: {retreat_evidence_chain:.4f}")
        print(f"    - [核心公式]: 趋势衰退 * 缺乏承接 * 赢家投降 * 牛市陷阱")
        print(f"    - [计算]: {trend_decay_context:.2f} * {no_absorption_score:.2f} * {winner_capitulation_score:.2f} * {bull_trap_evidence:.2f} = {retreat_evidence_chain:.4f}")
        print(f"    - [钻透]:")
        print(f"      - 证据1 (趋势衰退): {trend_decay_context:.4f} (来自 1 - 趋势质量分 {trend_quality_context:.4f})")
        print(f"      - 证据2 (缺乏承接): {no_absorption_score:.4f} (来自 1 - 恐慌吸收分 {panic_absorption_score:.4f})")
        print(f"      - 证据3 (赢家投降): {winner_capitulation_score:.4f} (来自 盈利盘信念[0,1] {winner_conviction_0_1:.4f})")
        print(f"      - 证据4 (牛市陷阱): {bull_trap_evidence:.4f} (来自 1 - 反转动态质量 {reversal_dynamic_quality:.4f})")
        # --- 链路层 4: 最终验证 ---
        print("\n  [链路层 4] 最终验证 (Final Verification)")
        recalc_score = (get_val(recent_distribution_strength, probe_date) * retreat_evidence_chain).clip(0, 1)
        print(f"    - [探针重算]: {get_val(recent_distribution_strength, probe_date):.4f} (派发强度) * {retreat_evidence_chain:.4f} (证据链) = {recalc_score:.4f}")
        print(f"    - [对比]: 实际值 {final_score:.4f} vs 重算值 {recalc_score:.4f}")
        print("--- “塔纳托斯之镰”探针解剖完毕 ---")

    def _deploy_prometheus_torch_probe(self, probe_date: pd.Timestamp):
        """
        【V1.0 · 新增】“普罗米修斯火炬”探针 - 行为智能引擎全要素解剖
        - 核心职责: 深度解剖 BehavioralIntelligence 的核心引擎 _calculate_structural_behavior_health
                     和 transmute_health_to_ultimate_signals 的完整计算链路。
        - 解剖路径:
          1. 最终输出: 展示 SCORE_STRUCT_BEHAVIOR_* 系列终极信号的最终值。
          2. 复合状态构建: 解剖 bullish_composite_state 和 bearish_composite_state 的合成过程。
          3. 健康度计算: 以周期 p=5 为例，展示 s_bull 和 s_bear 的详细计算过程。
          4. 终极信号嬗变: 重算并验证最终的看涨/看跌共振分和反转分。
        """
        print("\n" + "="*35 + f" [行为探针] 正在点燃 🔥【普罗米修斯火炬 · 行为引擎解剖 V1.0】🔥 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        # --- 链路层 1: 最终输出 ---
        print("\n  [链路层 1] 最终输出 (Final Output)")
        bull_res = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_rev = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_rev = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_TOP_REVERSAL'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res:.4f}")
        print(f"    - 【看跌共振】: {bear_res:.4f}")
        print(f"    - 【底部反转】: {bottom_rev:.4f}")
        print(f"    - 【顶部反转】: {top_rev:.4f}")
        # --- 链路层 2: 复合状态构建 (Composite State Construction) ---
        print("\n  [链路层 2] 复合状态构建 (来自 _calculate_structural_behavior_health)")
        # 原材料
        csi = get_val(df.get('closing_strength_index_D'), probe_date, 0.5)
        cvv = get_val(df.get('close_vs_vwap_ratio_D'), probe_date, 1.0)
        fdiv = get_val(df.get('flow_divergence_mf_vs_retail_D'), probe_date, 0.0)
        fhm = get_val(df.get('final_hour_momentum_D'), probe_date, 0.0)
        ite = get_val(df.get('intraday_trend_efficiency_D'), probe_date, 0.5)
        print("    - [原材料 Raw Ingredients]:")
        print(f"      - closing_strength_index_D: {csi:.4f}")
        print(f"      - close_vs_vwap_ratio_D: {cvv:.4f}")
        print(f"      - flow_divergence_mf_vs_retail_D: {fdiv:.4f}")
        print(f"      - final_hour_momentum_D: {fhm:.4f}")
        print(f"      - intraday_trend_efficiency_D: {ite:.4f}")
        # 中间件
        csi_score = get_val(normalize_score(df.get('closing_strength_index_D'), df.index, norm_window), probe_date)
        vwap_score = get_val(normalize_score(df.get('close_vs_vwap_ratio_D'), df.index, norm_window), probe_date)
        rev_strength = (csi_score * vwap_score)**0.5
        rev_weakness = ((1.0 - csi_score) * (1.0 - vwap_score))**0.5
        bull_div = get_val(normalize_score(df.get('flow_divergence_mf_vs_retail_D').clip(0), df.index, norm_window), probe_date)
        auction_power = get_val(normalize_score(df.get('final_hour_momentum_D').clip(0), df.index, norm_window), probe_date)
        trend_eff = get_val(normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window), probe_date)
        bear_div = get_val(normalize_score(df.get('flow_divergence_mf_vs_retail_D').clip(upper=0).abs(), df.index, norm_window), probe_date)
        auction_weak = get_val(normalize_score(df.get('final_hour_momentum_D').clip(upper=0).abs(), df.index, norm_window), probe_date)
        trend_ineff = 1 - trend_eff
        bullish_composite_state = (rev_strength * csi_score * (1 + bull_div) * auction_power * trend_eff)**(1/5)
        bearish_composite_state = (rev_weakness * (1.0 - csi_score) * (1 + bear_div) * auction_weak * trend_ineff)**(1/5)
        print("\n    - [复合状态计算 Composite State Calculation]:")
        print(f"      - 看涨复合状态 (Bullish Composite): {bullish_composite_state:.4f}")
        print(f"        - [公式]: (反转强度 * 下影线力量 * (1+主力背离) * 尾盘动能 * 趋势效率)^(1/5)")
        print(f"        - [计算]: ({rev_strength:.2f} * {csi_score:.2f} * (1+{bull_div:.2f}) * {auction_power:.2f} * {trend_eff:.2f})^(1/5) = {bullish_composite_state:.4f}")
        print(f"      - 看跌复合状态 (Bearish Composite): {bearish_composite_state:.4f}")
        print(f"        - [公式]: (反转疲弱 * 上影线压力 * (1+散户背离) * 尾盘疲弱 * 趋势无效)^(1/5)")
        print(f"        - [计算]: ({rev_weakness:.2f} * {(1.0-csi_score):.2f} * (1+{bear_div:.2f}) * {auction_weak:.2f} * {trend_ineff:.2f})^(1/5) = {bearish_composite_state:.4f}")
        # --- 链路层 3: 健康度计算 (Health Calculation) ---
        print("\n  [链路层 3] 健康度计算 (以周期 p=5 为例)")
        p = 5
        context_p = 13
        # 重新计算 bullish_composite_state series
        _csi_s = normalize_score(df.get('closing_strength_index_D'), df.index, norm_window)
        _vwap_s = normalize_score(df.get('close_vs_vwap_ratio_D'), df.index, norm_window)
        _rev_s = (_csi_s * _vwap_s)**0.5
        _bull_div_s = normalize_score(df.get('flow_divergence_mf_vs_retail_D').clip(0), df.index, norm_window)
        _ap_s = normalize_score(df.get('final_hour_momentum_D').clip(0), df.index, norm_window)
        _te_s = normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window)
        bullish_composite_state_series = (_rev_s * _csi_s * (1 + _bull_div_s) * _ap_s * _te_s)**(1/5)
        bull_static_norm = get_val(normalize_score(bullish_composite_state_series, df.index, p, True), probe_date)
        bull_slope_raw = get_val(bullish_composite_state_series.diff(p), probe_date)
        bull_slope_norm = get_val(normalize_score(bullish_composite_state_series.diff(p).fillna(0), df.index, p, True), probe_date)
        bull_accel_raw = get_val(bullish_composite_state_series.diff(p).fillna(0).diff(1), probe_date)
        bull_accel_norm = get_val(normalize_score(bullish_composite_state_series.diff(p).fillna(0).diff(1).fillna(0), df.index, p, True), probe_date)
        tactical_bull_health = (bull_static_norm * bull_slope_norm * bull_accel_norm)**(1/3)
        context_bull_static_norm = get_val(normalize_score(bullish_composite_state_series, df.index, context_p, True), probe_date)
        context_bull_slope_norm = get_val(normalize_score(bullish_composite_state_series.diff(p).fillna(0), df.index, context_p, True), probe_date)
        context_bull_accel_norm = get_val(normalize_score(bullish_composite_state_series.diff(p).fillna(0).diff(1).fillna(0), df.index, context_p, True), probe_date)
        context_bull_health = (context_bull_static_norm * context_bull_slope_norm * context_bull_accel_norm)**(1/3)
        s_bull_p5 = (tactical_bull_health * context_bull_health)**0.5
        print("    - [看涨健康度 s_bull[5]]:")
        print(f"      - 战术层健康度 (p=5): {tactical_bull_health:.4f} (状态: {bull_static_norm:.2f}, 斜率: {bull_slope_norm:.2f}, 加速度: {bull_accel_norm:.2f})")
        print(f"      - 上下文健康度 (p=13): {context_bull_health:.4f} (状态: {context_bull_static_norm:.2f}, 斜率: {context_bull_slope_norm:.2f}, 加速度: {context_bull_accel_norm:.2f})")
        print(f"      - 【融合后 s_bull[5]】: {s_bull_p5:.4f}")
        # --- 链路层 4: 终极信号嬗变 (Ultimate Signal Transmutation) ---
        print("\n  [链路层 4] 终极信号嬗变 (来自 transmute_health_to_ultimate_signals)")
        overall_health = atomic.get('__BEHAVIOR_overall_health', {})
        s_bull_all = overall_health.get('s_bull', {})
        bull_scores_by_period = {p: get_val(s_bull_all.get(p), probe_date, 0.5) for p in periods}
        print("    - [看涨健康度 s_bull 快照]:")
        for p, val in bull_scores_by_period.items():
            print(f"      - s_bull[{p}]: {val:.4f}")
        resonance_tf_weights = get_param_value(p_synthesis.get('resonance_tf_weights'), {})
        short_periods = [p for p in periods if p <= 5]
        medium_periods = [p for p in periods if 5 < p <= 21]
        long_periods = [p for p in periods if p > 21]
        bull_short_score = np.mean([bull_scores_by_period.get(p, 0.5) for p in short_periods]) if short_periods else 0.5
        bull_medium_score = np.mean([bull_scores_by_period.get(p, 0.5) for p in medium_periods]) if medium_periods else 0.5
        bull_long_score = np.mean([bull_scores_by_period.get(p, 0.5) for p in long_periods]) if long_periods else 0.5
        bullish_static_score = (
            bull_short_score * resonance_tf_weights.get('short', 0.2) +
            bull_medium_score * resonance_tf_weights.get('medium', 0.5) +
            bull_long_score * resonance_tf_weights.get('long', 0.3)
        )
        print("\n    - [看涨共振重算]:")
        print(f"      - 短期(<=5)均分: {bull_short_score:.4f}, 中期(>5,<=21)均分: {bull_medium_score:.4f}, 长期(>21)均分: {bull_long_score:.4f}")
        print(f"      - 融合后静态分: {bullish_static_score:.4f}")
        # 重新计算动态分
        bullish_static_series = pd.Series(0.0, index=df.index)
        total_weight = resonance_tf_weights.get('short', 0.2) + resonance_tf_weights.get('medium', 0.5) + resonance_tf_weights.get('long', 0.3)
        if total_weight > 0:
            weight_short = resonance_tf_weights.get('short', 0.2) / total_weight
            weight_medium = resonance_tf_weights.get('medium', 0.5) / total_weight
            weight_long = resonance_tf_weights.get('long', 0.3) / total_weight
            for p in short_periods: bullish_static_series += s_bull_all.get(p, pd.Series(0.5, index=df.index)) * (weight_short / len(short_periods))
            for p in medium_periods: bullish_static_series += s_bull_all.get(p, pd.Series(0.5, index=df.index)) * (weight_medium / len(medium_periods))
            for p in long_periods: bullish_static_series += s_bull_all.get(p, pd.Series(0.5, index=df.index)) * (weight_long / len(long_periods))
        d_intensity_all = overall_health.get('d_intensity', {})
        d_intensity_series = pd.Series(0.0, index=df.index)
        if d_intensity_all:
            for p in periods:
                d_intensity_series += d_intensity_all.get(p, pd.Series(0.5, index=df.index)) * (1/len(periods))
        bullish_dynamic_score = get_val(d_intensity_series, probe_date, 0.5)
        recalc_bull_res = (bullish_static_score * bullish_dynamic_score)**0.5
        print(f"      - 动态强度分: {bullish_dynamic_score:.4f}")
        print(f"      - 【重算看涨共振】: (静态分 {bullish_static_score:.2f} * 动态分 {bullish_dynamic_score:.2f})^0.5 = {recalc_bull_res:.4f}")
        print(f"      - [对比]: 实际值 {bull_res:.4f} vs 重算值 {recalc_bull_res:.4f}")
        print("\n--- “普罗米修斯火炬”探针解剖完毕 ---")






