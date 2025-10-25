# 文件: strategies/trend_following/forensic_probes.py
# 法医探针集合
import pandas as pd
import numpy as np
import pandas_ta as ta
import json
from typing import Dict
from strategies.trend_following.utils import get_params_block, calculate_holographic_dynamics, transmute_health_to_ultimate_signals, get_param_value, calculate_context_scores, normalize_score, normalize_to_bipolar, _calculate_gaia_bedrock_support, _calculate_historical_low_support, get_unified_score

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
        【V2.6 · 阿波罗战车同步版】“普罗米修斯火炬”探针
        - 核心升级: 与主引擎 `_calculate_structural_behavior_health` V4.2 版完全同步。
                      在“链路层2”中，精确复刻了“阿波罗战车”协议，引入“日内质量分”来计算“净有效强度”。
        """
        print("\n" + "="*35 + f" [行为探针] 正在点燃 🔥【普罗米修斯火炬 · 行为引擎解剖 V2.6】🔥 " + "="*35)
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
        print("\n  [链路层 1] 最终输出 (Final Output)")
        bull_res_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_rev_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_rev_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_TOP_REVERSAL'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print(f"    - 【底部反转】: {bottom_rev_actual:.4f}")
        print(f"    - 【顶部反转】: {top_rev_actual:.4f}")
        print("\n  [链路层 2] 复合状态构建 (来自 _calculate_structural_behavior_health)")
        # [代码修改开始] 同步 V4.2 阿波罗战车协议
        # 2.1 原材料
        raw_ingredients = {
            'pre_close_D': get_val(df.get('pre_close_D'), probe_date),
            'open_D': get_val(df.get('open_D'), probe_date),
            'high_D': get_val(df.get('high_D'), probe_date),
            'low_D': get_val(df.get('low_D'), probe_date),
            'close_D': get_val(df.get('close_D'), probe_date),
            'pct_change_D': get_val(df.get('pct_change_D'), probe_date),
            'closing_strength_index_D': get_val(df.get('closing_strength_index_D'), probe_date, 0.5),
            'flow_divergence_mf_vs_retail_D': get_val(df.get('flow_divergence_mf_vs_retail_D'), probe_date, 0.0),
            'final_hour_momentum_D': get_val(df.get('final_hour_momentum_D'), probe_date, 0.0),
            'intraday_trend_efficiency_D': get_val(df.get('intraday_trend_efficiency_D'), probe_date, 0.5),
            'intraday_volume_gini_D': get_val(df.get('intraday_volume_gini_D'), probe_date, 0.5)
        }
        print("    - [原材料 Raw Ingredients]:")
        for name, val in raw_ingredients.items():
            print(f"      - {name}: {val:.4f}")
        # 2.2 阿波罗战车协议计算
        gap_up = df['open_D'] > df['pre_close_D']
        body_up = df['close_D'] > df['open_D']
        trajectory_score = pd.Series(0.0, index=df.index)
        trajectory_score.loc[gap_up & body_up] = 1.0
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        lower_shadow = (np.minimum(df['open_D'], df['close_D']) - df['low_D']).clip(lower=0)
        shadow_modifier = ((lower_shadow - upper_shadow) / kline_range).fillna(0)
        day_quality_score = (trajectory_score * 0.7 + shadow_modifier * 0.3).clip(-1, 1)
        quality_adjustment_factor = (1 + day_quality_score) / 2
        positive_day_strength_raw = df['pct_change_D'].clip(0)
        net_effective_bullish_strength = (positive_day_strength_raw * 0.5) + (positive_day_strength_raw * quality_adjustment_factor * 0.5)
        positive_day_strength = normalize_score(net_effective_bullish_strength, df.index, norm_window) * (net_effective_bullish_strength > 0)
        print("\n    - [阿波罗战车协议计算]:")
        print(f"      - 轨迹得分: {get_val(trajectory_score, probe_date):.4f} (高开高走=1.0)")
        print(f"      - 影线修正分: {get_val(shadow_modifier, probe_date):.4f} ((下影-上影)/振幅)")
        print(f"      - 日内质量分: {get_val(day_quality_score, probe_date):.4f} (轨迹*0.7+影线*0.3)")
        print(f"      - 质量调整因子: {get_val(quality_adjustment_factor, probe_date):.4f} (->[0,1])")
        print(f"      - 净有效看涨强度(原始): {get_val(net_effective_bullish_strength, probe_date):.4f}")
        # 2.3 中间件
        csi_score = get_val(normalize_score(df.get('closing_strength_index_D'), df.index, norm_window), probe_date)
        bull_div_score = get_val(normalize_score(df.get('flow_divergence_mf_vs_retail_D').clip(0), df.index, norm_window), probe_date)
        auction_power_score = get_val(normalize_score(df.get('final_hour_momentum_D').clip(0), df.index, norm_window), probe_date)
        trend_eff_score = get_val(normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window), probe_date)
        efficiency_holo_bull, _ = calculate_holographic_dynamics(df, 'intraday_trend_efficiency_D', norm_window)
        gini_holo_bull, _ = calculate_holographic_dynamics(df, 'intraday_volume_gini_D', norm_window)
        eff_holo_bull_val = get_val(efficiency_holo_bull, probe_date)
        gini_holo_bull_val = get_val(gini_holo_bull, probe_date)
        bullish_d_intensity = (eff_holo_bull_val + gini_holo_bull_val) / 2.0
        print("\n    - [中间件 Normalized Ingredients]:")
        print(f"      - 净有效看涨强度(归一化): {get_val(positive_day_strength, probe_date):.4f}")
        print(f"      - csi_score: {csi_score:.4f}, bull_div_score: {bull_div_score:.4f}")
        print(f"      - auction_power_score: {auction_power_score:.4f}, trend_eff_score: {trend_eff_score:.4f}")
        print(f"      - bullish_d_intensity: {bullish_d_intensity:.4f}")
        # 2.4 复合状态计算
        bullish_composite_state = (
            get_val(positive_day_strength, probe_date) * csi_score * (1 + bull_div_score) *
            auction_power_score * trend_eff_score * bullish_d_intensity
        )**(1/6)
        bearish_composite_state = 0.0
        print("\n    - [复合状态计算 Composite State Calculation]:")
        print(f"      - 看涨复合状态 (Bullish Composite): {bullish_composite_state:.4f}")
        print(f"      - 看跌复合状态 (Bearish Composite): {bearish_composite_state:.4f} (因上涨日，净有效看跌强度为0)")
        # [代码修改结束]
        print("\n  [链路层 3.5] 健康度全景 (Full Health Panorama)")
        overall_health = atomic.get('__BEHAVIOR_overall_health', {})
        bipolar_health = overall_health.get('s_bull', {})
        for p in periods:
            health_val = get_val(bipolar_health.get(p), probe_date, 0.0)
            print(f"    - [双极性健康分 s_bull[{p}]]: {health_val:.4f}")
        print("\n  [链路层 4] 融合过程尸检 (Fusion Autopsy)")
        if not overall_health:
            print("    - [错误] 无法在 atomic_states 中找到 '__BEHAVIOR_overall_health'，无法进行重算。")
            return
        resonance_tf_weights = get_param_value(p_synthesis.get('resonance_tf_weights'), {})
        reversal_tf_weights = get_param_value(p_synthesis.get('reversal_tf_weights'), {})
        neutral_zone_threshold = get_param_value(p_synthesis.get('neutral_zone_threshold'), 0.1)
        period_map = {'short': 5, 'medium': 21, 'long': 55}
        print("    - [融合权重] 共振权重: " + json.dumps(resonance_tf_weights))
        print("    - [融合权重] 反转权重: " + json.dumps(reversal_tf_weights))
        print("    - [罗塞塔石碑] 周期映射: " + json.dumps(period_map))
        def probe_fuse_bipolar(health_dict, weights, name):
            final_score = 0.0
            total_weight = sum(weights.values())
            calc_str = []
            if total_weight > 0:
                for tf_name, weight in weights.items():
                    period_key = period_map.get(tf_name)
                    if period_key is None: continue
                    score = get_val(health_dict.get(period_key), probe_date, 0.0)
                    final_score += score * (weight / total_weight)
                    calc_str.append(f"({score:.2f} * {weight})")
            print(f"\n    - [融合计算 - {name}]:")
            print(f"      - [公式]: Σ (s_bull[p] * weight) / Σ(weight)")
            print(f"      - [计算]: ({' + '.join(calc_str)}) / {total_weight:.2f} = {final_score:.4f}")
            return final_score
        final_bipolar_resonance = probe_fuse_bipolar(bipolar_health, resonance_tf_weights, "共振信号")
        final_bipolar_reversal = probe_fuse_bipolar(bipolar_health, reversal_tf_weights, "反转信号")
        print(f"\n    - [壁炉审判] 中性区阈值: {neutral_zone_threshold:.2f}")
        from strategies.trend_following.utils import bipolar_to_exclusive_unipolar
        bull_res_recalc, bear_res_recalc = bipolar_to_exclusive_unipolar(pd.Series([final_bipolar_resonance]), neutral_zone_threshold)
        bottom_rev_recalc, top_rev_recalc = bipolar_to_exclusive_unipolar(pd.Series([final_bipolar_reversal]), neutral_zone_threshold)
        print(f"      - 共振分 {final_bipolar_resonance:.4f} -> 看涨: {bull_res_recalc.iloc[0]:.4f}, 看跌: {bear_res_recalc.iloc[0]:.4f}")
        print(f"      - 反转分 {final_bipolar_reversal:.4f} -> 底部: {bottom_rev_recalc.iloc[0]:.4f}, 顶部: {top_rev_recalc.iloc[0]:.4f}")
        print("\n    - [终极对质 Final Verdict]:")
        print(f"      - 【看涨共振】: 实际值 {bull_res_actual:.4f} vs. 探针重算 {bull_res_recalc.iloc[0]:.4f} -> {'✅ 一致' if np.isclose(bull_res_actual, bull_res_recalc.iloc[0]) else '❌ 不一致'}")
        print(f"      - 【看跌共振】: 实际值 {bear_res_actual:.4f} vs. 探针重算 {bear_res_recalc.iloc[0]:.4f} -> {'✅ 一致' if np.isclose(bear_res_actual, bear_res_recalc.iloc[0]) else '❌ 不一致'}")
        print(f"      - 【底部反转】: 实际值 {bottom_rev_actual:.4f} vs. 探针重算(trigger) {bottom_rev_recalc.iloc[0]:.4f} -> (仅供参考，未计入上下文)")
        print(f"      - 【顶部反转】: 实际值 {top_rev_actual:.4f} vs. 探针重算(trigger) {top_rev_recalc.iloc[0]:.4f} -> (仅供参考，未计入上下文)")
        print("\n--- “普罗米修斯火炬”探针解剖完毕 ---")

    def _deploy_hephaestus_forge_probe(self, probe_date: pd.Timestamp):
        """
        【V1.2 · 商神杖同步版】“赫菲斯托斯熔炉”探针
        - 核心升级: 与主引擎 V4.5 完全同步，精确复刻“赫尔墨斯的商神杖”协议，对背离信号进行“阴阳分离”，确保反转信号计算的纯净性。
        """
        print("\n" + "="*35 + f" [筹码探针] 正在点燃 🔥【赫菲斯托斯熔炉 · 筹码引擎解剖 V1.2】🔥 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        chip_intel = self.chip_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        print("\n  [链路层 1] 最终输出 (Final Output)")
        bull_res_actual = get_val(atomic.get('SCORE_CHIP_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_actual = get_val(atomic.get('SCORE_CHIP_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_rev_actual = get_val(atomic.get('SCORE_CHIP_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_rev_actual = get_val(atomic.get('SCORE_CHIP_TOP_REVERSAL'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print(f"    - 【底部反转】: {bottom_rev_actual:.4f}")
        print(f"    - 【顶部反转】: {top_rev_actual:.4f}")
        print("\n  [链路层 2] 四大公理诊断 (Axiom Diagnostics)")
        periods = [1, 5, 13, 21, 55]
        concentration_scores = chip_intel._diagnose_concentration_dynamics(df, periods)
        accumulation_scores = chip_intel._diagnose_main_force_action(df, periods)
        power_transfer_scores = chip_intel._diagnose_power_transfer(df, periods)
        peak_integrity_scores = chip_intel._diagnose_peak_integrity_dynamics(df, periods)
        axiom_scores_by_period = {}
        for p in periods:
            axiom_scores_by_period[p] = {
                'concentration': get_val(concentration_scores.get(p), probe_date, 0.0),
                'accumulation': get_val(accumulation_scores.get(p), probe_date, 0.0),
                'power_transfer': get_val(power_transfer_scores.get(p), probe_date, 0.0),
                'peak_integrity': get_val(peak_integrity_scores.get(p), probe_date, 0.0),
            }
            print(f"    - [周期 {p:2d}] 公理得分: 聚散({axiom_scores_by_period[p]['concentration']:.2f}), 吸派({axiom_scores_by_period[p]['accumulation']:.2f}), 转移({axiom_scores_by_period[p]['power_transfer']:.2f}), 峰健康({axiom_scores_by_period[p]['peak_integrity']:.2f})")
        print("\n  [链路层 3] 双极性健康分合成 (Bipolar Health Synthesis)")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {})
        print(f"    - [公理权重 Axiom Weights]: " + json.dumps(axiom_weights))
        bipolar_health_recalc = {}
        for p in periods:
            scores = axiom_scores_by_period[p]
            health = (
                scores['concentration'] * axiom_weights.get('concentration', 0.3) +
                scores['accumulation'] * axiom_weights.get('accumulation', 0.3) +
                scores['power_transfer'] * axiom_weights.get('power_transfer', 0.25) +
                scores['peak_integrity'] * axiom_weights.get('peak_integrity', 0.15)
            )
            bipolar_health_recalc[p] = np.clip(health, -1, 1)
            print(f"    - [周期 {p:2d}] 双极性健康分: {bipolar_health_recalc[p]:.4f}")
        print("\n  [链路层 4] 终极信号融合 (Ultimate Signal Fusion)")
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {})
        print(f"    - [周期权重 TF Weights]: " + json.dumps(tf_weights))
        bullish_scores_recalc = {p: max(0, score) for p, score in bipolar_health_recalc.items()}
        bearish_scores_recalc = {p: max(0, -score) for p, score in bipolar_health_recalc.items()}
        bull_res_recalc, bear_res_recalc = 0.0, 0.0
        numeric_weights = {k: v for k, v in tf_weights.items() if isinstance(v, (int, float))}
        total_weight = sum(numeric_weights.values())
        bull_calc_str, bear_calc_str = [], []
        if total_weight > 0:
            for p in periods:
                weight = numeric_weights.get(str(p), 0) / total_weight
                bull_res_recalc += bullish_scores_recalc.get(p, 0.0) * weight
                bear_res_recalc += bearish_scores_recalc.get(p, 0.0) * weight
                bull_calc_str.append(f"({bullish_scores_recalc.get(p, 0.0):.2f}*{weight:.2f})")
                bear_calc_str.append(f"({bearish_scores_recalc.get(p, 0.0):.2f}*{weight:.2f})")
        print(f"\n    - [融合计算 - 看涨共振]: {' + '.join(bull_calc_str)} = {bull_res_recalc:.4f}")
        print(f"    - [融合计算 - 看跌共振]: {' + '.join(bear_calc_str)} = {bear_res_recalc:.4f}")
        bipolar_health_series = {}
        for p in periods:
            conc_s = concentration_scores.get(p, pd.Series(0.0, index=df.index))
            acc_s = accumulation_scores.get(p, pd.Series(0.0, index=df.index))
            pow_s = power_transfer_scores.get(p, pd.Series(0.0, index=df.index))
            peak_s = peak_integrity_scores.get(p, pd.Series(0.0, index=df.index))
            bipolar_health_series[p] = (conc_s * axiom_weights.get('concentration', 0.3) + acc_s * axiom_weights.get('accumulation', 0.3) + pow_s * axiom_weights.get('power_transfer', 0.25) + peak_s * axiom_weights.get('peak_integrity', 0.15)).clip(-1, 1)
        bullish_series = {p: s.clip(0, 1) for p, s in bipolar_health_series.items()}
        bearish_series = {p: (s.clip(-1, 0) * -1) for p, s in bipolar_health_series.items()}
        bottom_rev_recalc, top_rev_recalc = 0.0, 0.0
        bottom_calc_str, top_calc_str = [], []
        if total_weight > 0:
            for p in periods:
                weight = numeric_weights.get(str(p), 0) / total_weight
                context_p = periods[periods.index(p) + 1] if periods.index(p) + 1 < len(periods) else p
                # [代码修改开始] 复刻“赫尔墨斯的商神杖”协议
                holographic_bull_divergence = chip_intel._calculate_holographic_divergence(bullish_series.get(p), 1, p, context_p)
                holographic_bear_divergence = chip_intel._calculate_holographic_divergence(bearish_series.get(p), 1, p, context_p)
                bull_divergence_val = get_val(holographic_bull_divergence, probe_date, 0.0)
                bear_divergence_val = get_val(holographic_bear_divergence, probe_date, 0.0)
                # 阴阳分离
                bottom_rev_recalc += max(0, bull_divergence_val) * weight
                top_rev_recalc += max(0, bear_divergence_val) * weight
                bottom_calc_str.append(f"({max(0, bull_divergence_val):.2f}*{weight:.2f})")
                top_calc_str.append(f"({max(0, bear_divergence_val):.2f}*{weight:.2f})")
                # [代码修改结束]
        print(f"    - [融合计算 - 底部反转]: {' + '.join(bottom_calc_str)} = {bottom_rev_recalc:.4f}")
        print(f"    - [融合计算 - 顶部反转]: {' + '.join(top_calc_str)} = {top_rev_recalc:.4f}")
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"      - 【看涨共振】: 实际值 {bull_res_actual:.4f} vs. 探针重算 {bull_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bull_res_actual, bull_res_recalc) else '❌ 不一致'}")
        print(f"      - 【看跌共振】: 实际值 {bear_res_actual:.4f} vs. 探针重算 {bear_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bear_res_actual, bear_res_recalc) else '❌ 不一致'}")
        print(f"      - 【底部反转】: 实际值 {bottom_rev_actual:.4f} vs. 探针重算 {bottom_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(bottom_rev_actual, bottom_rev_recalc) else '❌ 不一致'}")
        print(f"      - 【顶部反转】: 实际值 {top_rev_actual:.4f} vs. 探针重算 {top_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(top_rev_actual, top_rev_recalc) else '❌ 不一致'}")
        print("\n--- “赫菲斯托斯熔炉”探针解剖完毕 ---")




