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
        # 为筹码探针获取 chip_intel 引用
        self.chip_intel = intelligence_layer_instance.chip_intel
        # 为资金流探针获取 fund_flow_intel 引用
        self.fund_flow_intel = intelligence_layer_instance.fund_flow_intel
        # [代码新增开始]
        # 为新的动态力学探针获取 mechanics_engine 引用
        self.mechanics_engine = intelligence_layer_instance.mechanics_engine
        # 为新的基础探针获取 foundation_intel 引用
        self.foundation_intel = intelligence_layer_instance.foundation_intel
        self.process_intel = intelligence_layer_instance.process_intel
        self.behavioral_intel = intelligence_layer_instance.behavioral_intel
        # [代码新增结束]

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
        # 同步 V4.2 阿波罗战车协议
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
        【V1.5 · 焦点转移版】“赫菲斯托斯熔炉”探针
        - 核心升级: 将“代达罗斯迷宫”的解剖焦点从“公理一”转移至“公理四：筹码峰健康度”。
                      - [公理级解剖] [链路层 2.1] 现在深度解剖“公理四”的计算逻辑。
                      - [全链路追溯] 展示从原始指标(peak_control_ratio等)到最终公理得分的全过程。
        """
        print("\n" + "="*35 + f" [筹码探针] 正在点燃 🔥【赫菲斯托斯熔炉 · 筹码引擎解剖 V1.5】🔥 " + "="*35)
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
        print("\n  [链路层 2] 四大公理最终得分 (Final Axiom Scores)")
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
        # 实施“代达罗斯迷宫 V2”协议，深度解剖公理四
        print("\n  [链路层 2.1] 深度解剖 · 公理四: 筹码峰“健康度” (p=5 周期为例)")
        p_probe = 5
        # 2.1.1 原材料
        control_raw = get_val(df.get('peak_control_ratio_D'), probe_date, 0)
        stability_raw = get_val(df.get('peak_stability_D'), probe_date, 0)
        defense_raw = get_val(df.get('peak_defense_intensity_D'), probe_date, 0)
        proximity_raw = get_val(df.get('price_to_peak_ratio_D'), probe_date, 1.0)
        print("    - [原材料 (Raw Ingredients)]:")
        print(f"      - 控制力(Control): {control_raw:.4f}, 稳定性(Stability): {stability_raw:.4f}")
        print(f"      - 防御强度(Defense): {defense_raw:.4f}, 接近度(Proximity): {proximity_raw:.4f}")
        # 2.1.2 归一化
        control_score = get_val(normalize_score(df.get('peak_control_ratio_D'), df.index, p_probe, ascending=True), probe_date)
        stability_score = get_val(normalize_score(df.get('peak_stability_D'), df.index, p_probe, ascending=True), probe_date)
        defense_score = get_val(normalize_score(df.get('peak_defense_intensity_D'), df.index, p_probe, ascending=True), probe_date)
        proximity_score = get_val(normalize_score(df.get('price_to_peak_ratio_D'), df.index, p_probe, ascending=False), probe_date)
        print("    - [归一化得分 (Normalized Scores)]:")
        print(f"      - 控制力: {control_score:.4f}, 稳定性: {stability_score:.4f}")
        print(f"      - 防御强度: {defense_score:.4f}, 接近度: {proximity_score:.4f}")
        # 2.1.3 快照分与元分析
        bullish_evidence = (control_score * stability_score * defense_score * proximity_score)**(1/4)
        bearish_evidence = 1.0 - bullish_evidence
        snapshot_score_val = bullish_evidence - bearish_evidence
        # 重算Series
        control_series = normalize_score(df.get('peak_control_ratio_D', pd.Series(0.0, index=df.index)), df.index, p_probe, ascending=True)
        stability_series = normalize_score(df.get('peak_stability_D', pd.Series(0.0, index=df.index)), df.index, p_probe, ascending=True)
        defense_series = normalize_score(df.get('peak_defense_intensity_D', pd.Series(0.0, index=df.index)), df.index, p_probe, ascending=True)
        proximity_series = normalize_score(df.get('price_to_peak_ratio_D', pd.Series(1.0, index=df.index)), df.index, p_probe, ascending=False)
        bullish_evidence_series = (control_series * stability_series * defense_series * proximity_series)**(1/4)
        bearish_evidence_series = 1.0 - bullish_evidence_series
        snapshot_series = (bullish_evidence_series - bearish_evidence_series).astype(np.float32)
        holographic_divergence = chip_intel._calculate_holographic_divergence(snapshot_series, 1, p_probe, p_probe * 2)
        holographic_divergence_val = get_val(holographic_divergence, probe_date)
        recalc_axiom_score = chip_intel._perform_chip_relational_meta_analysis(df, snapshot_series, p_probe, holographic_divergence)
        recalc_axiom_score_val = get_val(recalc_axiom_score, probe_date)
        print("    - [快照分与元分析 (Snapshot & Meta-Analysis)]:")
        print(f"      - 看涨证据 (Bullish Evidence): {bullish_evidence:.4f}")
        print(f"      - 看跌证据 (Bearish Evidence): {bearish_evidence:.4f}")
        print(f"      - 峰健康快照分 (Peak Integrity Snapshot): {snapshot_score_val:.4f}")
        print(f"      - 全息背离分 (Holographic Divergence): {holographic_divergence_val:.4f}")
        print(f"      - 最终公理得分 (探针重算): {recalc_axiom_score_val:.4f}")
        print(f"    - [内部验证]: 实际值 {axiom_scores_by_period[p_probe]['peak_integrity']:.4f} vs. 探针重算 {recalc_axiom_score_val:.4f} -> {'✅ 一致' if np.isclose(axiom_scores_by_period[p_probe]['peak_integrity'], recalc_axiom_score_val) else '❌ 不一致'}")
        
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
                holographic_bull_divergence = chip_intel._calculate_holographic_divergence(bullish_series.get(p), 1, p, context_p)
                holographic_bear_divergence = chip_intel._calculate_holographic_divergence(bearish_series.get(p), 1, p, context_p)
                bull_divergence_val = get_val(holographic_bull_divergence, probe_date, 0.0)
                bear_divergence_val = get_val(holographic_bear_divergence, probe_date, 0.0)
                bottom_rev_recalc += max(0, bull_divergence_val) * weight
                top_rev_recalc += max(0, bear_divergence_val) * weight
                bottom_calc_str.append(f"({max(0, bull_divergence_val):.2f}*{weight:.2f})")
                top_calc_str.append(f"({max(0, bear_divergence_val):.2f}*{weight:.2f})")
        print(f"    - [融合计算 - 底部反转]: {' + '.join(bottom_calc_str)} = {bottom_rev_recalc:.4f}")
        print(f"    - [融合计算 - 顶部反转]: {' + '.join(top_calc_str)} = {top_rev_recalc:.4f}")
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"      - 【看涨共振】: 实际值 {bull_res_actual:.4f} vs. 探针重算 {bull_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bull_res_actual, bull_res_recalc) else '❌ 不一致'}")
        print(f"      - 【看跌共振】: 实际值 {bear_res_actual:.4f} vs. 探针重算 {bear_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bear_res_actual, bear_res_recalc) else '❌ 不一致'}")
        print(f"      - 【底部反转】: 实际值 {bottom_rev_actual:.4f} vs. 探针重算 {bottom_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(bottom_rev_actual, bottom_rev_recalc) else '❌ 不一致'}")
        print(f"      - 【顶部反转】: 实际值 {top_rev_actual:.4f} vs. 探针重算 {top_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(top_rev_actual, top_rev_recalc) else '❌ 不一致'}")
        print("\n--- “赫菲斯托斯熔炉”探针解剖完毕 ---")

    def _deploy_ares_chariot_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 引擎核心解剖版】“阿瑞斯的战车”探针
        - 核心升级: 签署“引擎核心解剖”协议，深入解剖动态力学引擎的元分析过程。
                      - [维度级解剖] 新增 [链路层 3.1]，以 p=5 周期为例，对元分析引擎进行深度解剖。
                      - [全链路追溯] 展示从快照分到速度、加速度，再到最终健康分的全计算过程。
                      - [内部验证] 对解剖的健康分进行重算和验证，确保探针逻辑与引擎完全一致。
        """
        print("\n" + "="*35 + f" [力学探针] 正在启动 🏎️【阿瑞斯的战车 · 力学引擎解剖 V1.1】🏎️ " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.mechanics_engine
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        print("\n  [链路层 1] 最终输出 (Final Output)")
        bull_res_actual = get_val(atomic.get('SCORE_DYN_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_actual = get_val(atomic.get('SCORE_DYN_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_rev_actual = get_val(atomic.get('SCORE_DYN_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_rev_actual = get_val(atomic.get('SCORE_DYN_TOP_REVERSAL'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print(f"    - 【底部反转】: {bottom_rev_actual:.4f}")
        print(f"    - 【顶部反转】: {top_rev_actual:.4f}")
        print("\n  [链路层 2] 五大支柱快照分 (Pillar Snapshots) - 战车的五匹神马")
        vol_bull_snapshot = (normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=False))**0.5
        eff_bull_snapshot = (normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window, ascending=True))**0.5
        mom_bull_snapshot = (normalize_score(df.get('ROC_12_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=True))**0.5
        adx_strength = normalize_score(df.get('ADX_14_D'), df.index, norm_window)
        adx_direction = (df.get('PDI_14_D', pd.Series(0, index=df.index)) > df.get('NDI_14_D', pd.Series(0, index=df.index))).astype(float)
        hurst_strength = normalize_score(df.get('hurst_120d_D'), df.index, norm_window)
        ine_bull_snapshot = (adx_strength * adx_direction * hurst_strength)**(1/3)
        energy_bull_snapshot = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=True)
        vol_bear_snapshot = (normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=True))**0.5
        eff_bear_snapshot = (normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window, ascending=False))**0.5
        mom_bear_snapshot = (normalize_score(df.get('ROC_12_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('MACDh_13_34_8_D'), df.index, norm_window, ascending=False))**0.5
        ine_bear_snapshot = (normalize_score(df.get('ADX_14_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('hurst_120d_D'), df.index, norm_window, ascending=False))**0.5
        energy_bear_snapshot = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=False)
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        weight_keys = list(pillar_weights.keys())
        weights_array = np.array([pillar_weights.get(name, 1.0/len(weight_keys)) for name in weight_keys])
        weights_array /= weights_array.sum()
        bull_snapshots_series = [vol_bull_snapshot, eff_bull_snapshot, mom_bull_snapshot, ine_bull_snapshot, energy_bull_snapshot]
        bear_snapshots_series = [vol_bear_snapshot, eff_bear_snapshot, mom_bear_snapshot, ine_bear_snapshot, energy_bear_snapshot]
        stacked_bull = np.stack([s.fillna(0.5).values for s in bull_snapshots_series], axis=0)
        stacked_bear = np.stack([s.fillna(0.5).values for s in bear_snapshots_series], axis=0)
        fused_bull_snapshot_series = pd.Series(np.prod(stacked_bull ** weights_array[:, np.newaxis], axis=0), index=df.index)
        fused_bear_snapshot_series = pd.Series(np.prod(stacked_bear ** weights_array[:, np.newaxis], axis=0), index=df.index)
        bipolar_mechanics_snapshot_series = (fused_bull_snapshot_series - fused_bear_snapshot_series).clip(-1, 1)
        ma_health_score_series = engine._calculate_ma_health(df, p_conf, norm_window)
        modulated_bipolar_snapshot_series = bipolar_mechanics_snapshot_series * ma_health_score_series
        print(f"    - [看涨快照] 波动性(Vol):{get_val(vol_bull_snapshot, probe_date):.2f}, 效率(Eff):{get_val(eff_bull_snapshot, probe_date):.2f}, 动量(Mom):{get_val(mom_bull_snapshot, probe_date):.2f}, 惯性(Ine):{get_val(ine_bull_snapshot, probe_date):.2f}, 能量(Energy):{get_val(energy_bull_snapshot, probe_date):.2f}")
        print(f"    - [看跌快照] 波动性(Vol):{get_val(vol_bear_snapshot, probe_date):.2f}, 效率(Eff):{get_val(eff_bear_snapshot, probe_date):.2f}, 动量(Mom):{get_val(mom_bear_snapshot, probe_date):.2f}, 惯性(Ine):{get_val(ine_bear_snapshot, probe_date):.2f}, 能量(Energy):{get_val(energy_bear_snapshot, probe_date):.2f}")
        print("\n  [链路层 3] 快照融合与调节 (Snapshot Fusion & Modulation)")
        print(f"    - [支柱权重]: {json.dumps(pillar_weights)}")
        print(f"    - 融合看涨快照: {get_val(fused_bull_snapshot_series, probe_date):.4f}, 融合看跌快照: {get_val(fused_bear_snapshot_series, probe_date):.4f}")
        print(f"    - 双极性力学快照: {get_val(bipolar_mechanics_snapshot_series, probe_date):.4f}")
        print(f"    - 均线健康分 (调节器): {get_val(ma_health_score_series, probe_date):.4f}")
        print(f"    - 调节后双极性快照: {get_val(modulated_bipolar_snapshot_series, probe_date):.4f}")
        # [代码新增开始] 实施“引擎核心解剖”协议
        print("\n  [链路层 3.1] 深度解剖 · 元分析引擎 (p=5 周期为例)")
        p_probe = 5
        p_context = 13
        p_meta = p_conf.get('relational_meta_analysis_params', {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        bipolar_sensitivity = 1.0
        snapshot_val = get_val(modulated_bipolar_snapshot_series, probe_date)
        print(f"    - [输入 Input]: 快照分(State) = {snapshot_val:.4f}, 权重(W_s, W_v, W_a) = ({w_state}, {w_velocity}, {w_acceleration})")
        tactical_trend = modulated_bipolar_snapshot_series.diff(p_probe).fillna(0)
        tactical_velocity_series = normalize_to_bipolar(tactical_trend, df.index, norm_window, bipolar_sensitivity)
        context_trend = modulated_bipolar_snapshot_series.diff(p_context).fillna(0)
        context_velocity_series = normalize_to_bipolar(context_trend, df.index, norm_window, bipolar_sensitivity)
        velocity_score_series = (tactical_velocity_series.abs() * context_velocity_series.abs())**0.5 * np.sign(tactical_velocity_series)
        velocity_val = get_val(velocity_score_series, probe_date)
        print(f"    - [速度维度 Velocity]: 战术(p={p_probe}) {get_val(tactical_velocity_series, probe_date):.2f}, 上下文(p={p_context}) {get_val(context_velocity_series, probe_date):.2f} -> 融合速度分 = {velocity_val:.4f}")
        tactical_accel_series = normalize_to_bipolar(tactical_trend.diff(p_probe).fillna(0), df.index, norm_window, bipolar_sensitivity)
        context_accel_series = normalize_to_bipolar(context_trend.diff(p_context).fillna(0), df.index, norm_window, bipolar_sensitivity)
        acceleration_score_series = (tactical_accel_series.abs() * context_accel_series.abs())**0.5 * np.sign(tactical_accel_series)
        acceleration_val = get_val(acceleration_score_series, probe_date)
        print(f"    - [加速度维度 Acceleration]: 战术(p={p_probe}) {get_val(tactical_accel_series, probe_date):.2f}, 上下文(p={p_context}) {get_val(context_accel_series, probe_date):.2f} -> 融合加速度分 = {acceleration_val:.4f}")
        bullish_state = max(0, snapshot_val)
        bullish_velocity = max(0, velocity_val)
        bullish_acceleration = max(0, acceleration_val)
        total_bullish_force = (bullish_state * w_state + bullish_velocity * w_velocity + bullish_acceleration * w_acceleration)
        bearish_state = max(0, -snapshot_val)
        bearish_velocity = max(0, -velocity_val)
        bearish_acceleration = max(0, -acceleration_val)
        total_bearish_force = (bearish_state * w_state + bearish_velocity * w_velocity + bearish_acceleration * w_acceleration)
        recalc_health = np.clip(total_bullish_force - total_bearish_force, -1, 1)
        print(f"    - [双子座裁决 Gemini Adjudication]:")
        print(f"      - 看涨力量: (s:{bullish_state:.2f}*w:{w_state}) + (v:{bullish_velocity:.2f}*w:{w_velocity}) + (a:{bullish_acceleration:.2f}*w:{w_acceleration}) = {total_bullish_force:.4f}")
        print(f"      - 看跌力量: (s:{bearish_state:.2f}*w:{w_state}) + (v:{bearish_velocity:.2f}*w:{w_velocity}) + (a:{bearish_acceleration:.2f}*w:{w_acceleration}) = {total_bearish_force:.4f}")
        print(f"      - 最终健康分 (探针重算): {recalc_health:.4f}")
        overall_health = atomic.get('__DYN_overall_health', {})
        s_bull_actual = get_val(overall_health.get('s_bull', {}).get(p_probe), probe_date, 0.0)
        s_bear_actual = get_val(overall_health.get('s_bear', {}).get(p_probe), probe_date, 0.0)
        actual_health = s_bull_actual - s_bear_actual
        print(f"    - [内部验证]: 实际值 {actual_health:.4f} vs. 探针重算 {recalc_health:.4f} -> {'✅ 一致' if np.isclose(actual_health, recalc_health) else '❌ 不一致'}")
        # [代码新增结束]
        print("\n  [链路层 4] 终极信号融合 (Ultimate Signal Fusion)")
        if not overall_health:
            print("    - [错误] 无法在 atomic_states 中找到 '__DYN_overall_health'，无法进行重算。")
            return
        recalc_signals = transmute_health_to_ultimate_signals(df, atomic, overall_health, p_synthesis, "DYN")
        bull_res_recalc = get_val(recalc_signals.get('SCORE_DYN_BULLISH_RESONANCE'), probe_date)
        bear_res_recalc = get_val(recalc_signals.get('SCORE_DYN_BEARISH_RESONANCE'), probe_date)
        bottom_rev_recalc = get_val(recalc_signals.get('SCORE_DYN_BOTTOM_REVERSAL'), probe_date)
        top_rev_recalc = get_val(recalc_signals.get('SCORE_DYN_TOP_REVERSAL'), probe_date)
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"      - 【看涨共振】: 实际值 {bull_res_actual:.4f} vs. 探针重算 {bull_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bull_res_actual, bull_res_recalc) else '❌ 不一致'}")
        print(f"      - 【看跌共振】: 实际值 {bear_res_actual:.4f} vs. 探针重算 {bear_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bear_res_actual, bear_res_recalc) else '❌ 不一致'}")
        print(f"      - 【底部反转】: 实际值 {bottom_rev_actual:.4f} vs. 探针重算 {bottom_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(bottom_rev_actual, bottom_rev_recalc) else '❌ 不一致'}")
        print(f"      - 【顶部反转】: 实际值 {top_rev_actual:.4f} vs. 探针重算 {top_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(top_rev_actual, top_rev_recalc) else '❌ 不一致'}")
        print("\n--- “阿瑞斯的战车”探针解剖完毕 ---")

    def _deploy_apollos_lyre_probe(self, probe_date: pd.Timestamp):
        """
        【V1.4 · 三叉戟协议版】“阿波罗的七弦琴”探针
        - 核心升级: 签署“三叉戟协议”，将“商神杖”子探针通用化，使其能够深度解剖RSI、MACD、CMF。
        - 升级意义: 实现了对所有四大支柱的“状态-速度-加速度”三维动态解剖，彻底透明化所有基础指标的计算逻辑。
        """
        print("\n" + "="*35 + f" [基础探针] 正在奏响 🎵【阿波罗的七弦琴 · 基础引擎解剖 V1.4】🎵 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.foundation_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        print("\n  [链路层 1] 最终输出 (Final Output)")
        bull_res_actual = get_val(atomic.get('SCORE_FOUNDATION_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_actual = get_val(atomic.get('SCORE_FOUNDATION_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_rev_actual = get_val(atomic.get('SCORE_FOUNDATION_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_rev_actual = get_val(atomic.get('SCORE_FOUNDATION_TOP_REVERSAL'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print(f"    - 【底部反转】: {bottom_rev_actual:.4f}")
        print(f"    - 【顶部反转】: {top_rev_actual:.4f}")
        print("\n  [链路层 2] 四大支柱快照分 (Pillar Snapshots) - 七弦琴的四根主弦")
        ma_context_score = engine._calculate_ma_trend_context(df, [5, 13, 21, 55])
        calculators = {
            'ema': lambda: engine._calculate_ema_health(df, norm_window, periods),
            'rsi': lambda: engine._calculate_rsi_health(df, norm_window, periods, ma_context_score),
            'macd': lambda: engine._calculate_macd_health(df, norm_window, periods, ma_context_score),
            'cmf': lambda: engine._calculate_cmf_health(df, norm_window, periods, ma_context_score)
        }
        # 增加指标原始列的映射
        pillar_source_cols = {
            'rsi': 'RSI_13_D',
            'macd': 'MACDh_13_34_8_D',
            'cmf': 'CMF_21_D'
        }
        
        pillar_snapshots = {}
        for name, calculator in calculators.items():
            snapshot_series = calculator()
            pillar_snapshots[name] = snapshot_series
            snapshot_val = get_val(snapshot_series, probe_date)
            print(f"    - [支柱: {name.upper()}] 双极性快照分: {snapshot_val:.4f}")
            # 通用化调用“商神杖”子探针
            if name == 'ema':
                self._deploy_caduceus_probe_for_ema(probe_date)
            elif name in pillar_source_cols:
                self._deploy_caduceus_probe_for_indicator(name, pillar_source_cols[name], probe_date)
            
        print("\n  [链路层 3] 快照融合 (Snapshot Fusion)")
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        print(f"    - [支柱权重]: {json.dumps(pillar_weights)}")
        weight_keys = list(pillar_snapshots.keys())
        weights_array = np.array([pillar_weights.get(name, 1.0/len(weight_keys)) for name in weight_keys])
        weights_array /= weights_array.sum()
        stacked_snapshots = np.stack([s.fillna(0.0).values for s in pillar_snapshots.values()], axis=0)
        fused_bipolar_snapshot_series = pd.Series(
            np.sum(stacked_snapshots * weights_array[:, np.newaxis], axis=0),
            index=df.index, dtype=np.float32
        ).clip(-1, 1)
        fused_snapshot_val = get_val(fused_bipolar_snapshot_series, probe_date)
        print(f"    - [融合结果] 融合后双极性快照分: {fused_snapshot_val:.4f}")
        print("\n  [链路层 3.1] 深度解剖 · 元分析引擎 (以 p=5 周期为例)")
        p_probe = 5
        meta_window = max(1, p_probe)
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        bipolar_sensitivity = 1.0
        snapshot_series = fused_bipolar_snapshot_series
        snapshot_val = get_val(snapshot_series, probe_date)
        print(f"    - [输入 Input]: 融合快照分(State) = {snapshot_val:.4f}, 权重(W_s, W_v, W_a) = ({w_state}, {w_velocity}, {w_acceleration})")
        print(f"    - [周期中心论]: 探测周期 p={p_probe} -> 使用元分析窗口 meta_window={meta_window}")
        relationship_trend = snapshot_series.diff(meta_window).fillna(0)
        velocity_score_series = normalize_to_bipolar(relationship_trend, df.index, norm_window, bipolar_sensitivity)
        velocity_val = get_val(velocity_score_series, probe_date)
        print(f"    - [速度维度 Velocity]: 速度分 = {velocity_val:.4f}")
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score_series = normalize_to_bipolar(relationship_accel, df.index, norm_window, bipolar_sensitivity)
        acceleration_val = get_val(acceleration_score_series, probe_date)
        print(f"    - [加速度维度 Acceleration]: 加速度分 = {acceleration_val:.4f}")
        bullish_state = max(0, snapshot_val)
        bullish_velocity = max(0, velocity_val)
        bullish_acceleration = max(0, acceleration_val)
        total_bullish_force = (bullish_state * w_state + bullish_velocity * w_velocity + bullish_acceleration * w_acceleration)
        bearish_state = max(0, -snapshot_val)
        bearish_velocity = max(0, -velocity_val)
        bearish_acceleration = max(0, -acceleration_val)
        total_bearish_force = (bearish_state * w_state + bearish_velocity * w_velocity + bearish_acceleration * w_acceleration)
        recalc_health = np.clip(total_bullish_force - total_bearish_force, -1, 1)
        print(f"    - [双子座裁决 Gemini Adjudication]:")
        print(f"      - 看涨力量: (s:{bullish_state:.2f}*w:{w_state}) + (v:{bullish_velocity:.2f}*w:{w_velocity}) + (a:{bullish_acceleration:.2f}*w:{w_acceleration}) = {total_bullish_force:.4f}")
        print(f"      - 看跌力量: (s:{bearish_state:.2f}*w:{w_state}) + (v:{bearish_velocity:.2f}*w:{w_velocity}) + (a:{bearish_acceleration:.2f}*w:{w_acceleration}) = {total_bearish_force:.4f}")
        print(f"      - 最终健康分 (探针重算): {recalc_health:.4f}")
        overall_health = atomic.get('__FOUNDATION_overall_health', {})
        s_bull_actual = get_val(overall_health.get('s_bull', {}).get(p_probe), probe_date, 0.0)
        s_bear_actual = get_val(overall_health.get('s_bear', {}).get(p_probe), probe_date, 0.0)
        actual_health = s_bull_actual - s_bear_actual
        print(f"    - [内部验证]: 实际值 {actual_health:.4f} vs. 探针重算 {recalc_health:.4f} -> {'✅ 一致' if np.isclose(actual_health, recalc_health) else '❌ 不一致'}")
        print("\n  [链路层 4] 终极信号融合 (Ultimate Signal Fusion)")
        if not overall_health:
            print("    - [错误] 无法在 atomic_states 中找到 '__FOUNDATION_overall_health'，无法进行重算。")
            return
        recalc_signals = transmute_health_to_ultimate_signals(df, atomic, overall_health, p_synthesis, "FOUNDATION")
        bull_res_recalc = get_val(recalc_signals.get('SCORE_FOUNDATION_BULLISH_RESONANCE'), probe_date)
        bear_res_recalc = get_val(recalc_signals.get('SCORE_FOUNDATION_BEARISH_RESONANCE'), probe_date)
        bottom_rev_recalc = get_val(recalc_signals.get('SCORE_FOUNDATION_BOTTOM_REVERSAL'), probe_date)
        top_rev_recalc = get_val(recalc_signals.get('SCORE_FOUNDATION_TOP_REVERSAL'), probe_date)
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"      - 【看涨共振】: 实际值 {bull_res_actual:.4f} vs. 探针重算 {bull_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bull_res_actual, bull_res_recalc) else '❌ 不一致'}")
        print(f"      - 【看跌共振】: 实际值 {bear_res_actual:.4f} vs. 探针重算 {bear_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bear_res_actual, bear_res_recalc) else '❌ 不一致'}")
        print(f"      - 【底部反转】: 实际值 {bottom_rev_actual:.4f} vs. 探针重算 {bottom_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(bottom_rev_actual, bottom_rev_recalc) else '❌ 不一致'}")
        print(f"      - 【顶部反转】: 实际值 {top_rev_actual:.4f} vs. 探针重算 {top_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(top_rev_actual, top_rev_recalc) else '❌ 不一致'}")
        print("\n--- “阿波罗的七弦琴”探针解剖完毕 ---")

    def _deploy_caduceus_probe_for_ema(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 修复版】“赫尔墨斯的商神杖”深度诊断单元 (EMA专用)
        - 核心修复: 修正了调用 `_calculate_ema_health` 时传递的参数，将 `norm_window` 的值正确传入，
                      修复了因参数错位导致的 `TypeError`。
        """
        print("\n" + "-"*15 + f" [子探针] 正在挥舞 ⚕️【商神杖 · EMA健康度解剖】 ⚕️ " + "-"*15)
        df = self.strategy.df_indicators
        engine = self.foundation_intel
        def get_val(series, date, default=np.nan):
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        fusion_weights = p_conf.get('ma_health_fusion_weights', {})
        norm_window = 55
        ma_periods = [5, 13, 21, 55]
        print(f"      - [融合权重]: {json.dumps(fusion_weights)}")
        bull_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).astype(float).values for i in range(len(ma_periods) - 1)]
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df.index), 0.5)
        alignment_bipolar_series = (pd.Series(alignment_score, index=df.index) - 0.5) * 2
        alignment_val = get_val(alignment_bipolar_series, probe_date)
        print(f"      - [维度1: 排列 Alignment]     双极性分: {alignment_val:.4f} (权重: {fusion_weights.get('alignment'):.2f})")
        slope_scores = [normalize_to_bipolar(df[f'SLOPE_{p}_EMA_{p}_D'], df.index, norm_window).values for p in ma_periods]
        avg_slope_bipolar_series = pd.Series(np.mean(slope_scores, axis=0), index=df.index)
        slope_val = get_val(avg_slope_bipolar_series, probe_date)
        print(f"      - [维度2: 斜率 Slope]         双极性分: {slope_val:.4f} (权重: {fusion_weights.get('slope'):.2f})")
        accel_scores = [normalize_to_bipolar(df[f'ACCEL_{p}_EMA_{p}_D'], df.index, norm_window).values for p in ma_periods]
        avg_accel_bipolar_series = pd.Series(np.mean(accel_scores, axis=0), index=df.index)
        accel_val = get_val(avg_accel_bipolar_series, probe_date)
        print(f"      - [维度3: 加速度 Accel]       双极性分: {accel_val:.4f} (权重: {fusion_weights.get('accel'):.2f})")
        relational_scores = []
        for short_p, long_p in [(5, 21), (13, 55)]:
            spread_accel = (df[f'EMA_{short_p}_D'] - df[f'EMA_{long_p}_D']).diff(3).diff(3).fillna(0)
            relational_scores.append(normalize_to_bipolar(spread_accel, df.index, norm_window).values)
        avg_relational_bipolar_series = pd.Series(np.mean(relational_scores, axis=0), index=df.index)
        relational_val = get_val(avg_relational_bipolar_series, probe_date)
        print(f"      - [维度4: 关系 Relational]    双极性分: {relational_val:.4f} (权重: {fusion_weights.get('relational'):.2f})")
        meta_dynamics_cols = ['SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D']
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        meta_scores = [normalize_to_bipolar(df[col], df.index, norm_window).values for col in valid_meta_cols] if valid_meta_cols else [np.full(len(df.index), 0.0)]
        avg_meta_bipolar_series = pd.Series(np.mean(meta_scores, axis=0), index=df.index)
        meta_val = get_val(avg_meta_bipolar_series, probe_date)
        print(f"      - [维度5: 元动态 Meta]        双极性分: {meta_val:.4f} (权重: {fusion_weights.get('meta_dynamics'):.2f})")
        recalc_snapshot = (
            alignment_val * fusion_weights.get('alignment', 0.15) +
            slope_val * fusion_weights.get('slope', 0.15) +
            accel_val * fusion_weights.get('accel', 0.2) +
            relational_val * fusion_weights.get('relational', 0.25) +
            meta_val * fusion_weights.get('meta_dynamics', 0.25)
        )
        recalc_snapshot = np.clip(recalc_snapshot, -1, 1)
        # 修正调用参数，将 norm_window (整数) 正确传递给第二个参数
        actual_snapshot_series = engine._calculate_ema_health(df, norm_window, [])
        
        actual_val = get_val(actual_snapshot_series, probe_date)
        print(f"      - [最终融合] 探针重算: {recalc_snapshot:.4f} vs. 引擎实际: {actual_val:.4f} -> {'✅ 一致' if np.isclose(recalc_snapshot, actual_val) else '❌ 不一致'}")
        print("-"*(32+38) + "\n")

    def _deploy_caduceus_probe_for_indicator(self, indicator_name: str, source_col: str, probe_date: pd.Timestamp):
        """
        【V2.0 · 三叉戟协议版】通用指标深度诊断单元
        - 核心使命: 对RSI, MACD, CMF等单一指标进行“状态-速度-加速度”三维动态解剖。
        - 升级意义: 响应指挥官指令，将探针的穿透力扩展到所有基础支柱。
        """
        print("\n" + "-"*15 + f" [子探针] 正在挥舞 ⚕️【商神杖 · {indicator_name.upper()}健康度解剖】 ⚕️ " + "-"*15)
        df = self.strategy.df_indicators
        engine = self.foundation_intel
        def get_val(series, date, default=np.nan):
            val = series.get(date)
            return default if pd.isna(val) else val
        
        norm_window = 55
        meta_window = 5 # 使用一个通用的短周期窗口来评估动态
        bipolar_sensitivity = 1.0
        
        if source_col not in df.columns:
            print(f"      - [错误] 找不到源数据列: {source_col}，无法进行解剖。")
            print("-"*(32+38) + "\n")
            return
            
        original_series = df[source_col]
        original_val = get_val(original_series, probe_date)
        print(f"      - [原始数据] {source_col} at {probe_date.date()}: {original_val:.4f}")

        # 1. 状态分 (State) - 这是引擎实际使用的分数
        state_score_series = normalize_to_bipolar(original_series, df.index, norm_window, bipolar_sensitivity)
        state_val = get_val(state_score_series, probe_date)
        print(f"      - [维度1: 状态 State]        双极性分: {state_val:.4f} (基于{norm_window}日滚动Z-score)")

        # 2. 速度分 (Velocity) - 额外诊断信息
        velocity_series = original_series.diff(meta_window).fillna(0)
        velocity_score_series = normalize_to_bipolar(velocity_series, df.index, norm_window, bipolar_sensitivity)
        velocity_val = get_val(velocity_score_series, probe_date)
        print(f"      - [维度2: 速度 Velocity]      双极性分: {velocity_val:.4f} (基于{meta_window}日变化)")

        # 3. 加速度分 (Acceleration) - 额外诊断信息
        acceleration_series = velocity_series.diff(meta_window).fillna(0)
        acceleration_score_series = normalize_to_bipolar(acceleration_series, df.index, norm_window, bipolar_sensitivity)
        acceleration_val = get_val(acceleration_score_series, probe_date)
        print(f"      - [维度3: 加速度 Accel]       双极性分: {acceleration_val:.4f} (基于速度的{meta_window}日变化)")

        # 最终验证
        # 获取引擎的实际计算结果
        if indicator_name == 'rsi':
            actual_series = engine._calculate_rsi_health(df, norm_window, [], None)
        elif indicator_name == 'macd':
            actual_series = engine._calculate_macd_health(df, norm_window, [], None)
        elif indicator_name == 'cmf':
            actual_series = engine._calculate_cmf_health(df, norm_window, [], None)
        else:
            actual_series = pd.Series(np.nan, index=df.index)
            
        actual_val = get_val(actual_series, probe_date)
        
        # 对于这些简单指标，引擎只用了状态分，所以我们只用状态分来验证
        recalc_val = state_val
        print(f"      - [最终验证] 探针重算(State): {recalc_val:.4f} vs. 引擎实际: {actual_val:.4f} -> {'✅ 一致' if np.isclose(recalc_val, actual_val) else '❌ 不一致'}")
        print("-"*(32+38) + "\n")

    def _deploy_poseidons_trident_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 健壮性加固版】“波塞冬的三叉戟”探针 - 资金流情报引擎深度解剖
        - 核心升级: 增加对周期权重(tf_weights)缺失的检测和警告，防止因配置错误导致引擎静默失效。
        """
        print("\n" + "="*35 + f" [资金流探针] 正在挥舞 🔱【波塞冬的三叉戟 · 资金流引擎解剖 V1.1】🔱 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.fund_flow_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        periods = get_param_value(p_conf.get('periods'), [1, 5, 13, 21, 55])
        p_probe = 5
        context_p = 13
        print("\n  [链路层 1] 最终输出 (Final Output)")
        concentration_score = get_val(atomic.get('SCORE_FF_AXIOM_CONCENTRATION', {}).get(p_probe), probe_date, 0.0)
        power_transfer_score = get_val(atomic.get('SCORE_FF_AXIOM_POWER_TRANSFER', {}).get(p_probe), probe_date, 0.0)
        internal_structure_score = get_val(atomic.get('SCORE_FF_AXIOM_INTERNAL_STRUCTURE', {}).get(p_probe), probe_date, 0.0)
        print(f"    - 【公理一: 聚散】(p={p_probe}): {concentration_score:.4f}")
        print(f"    - 【公理二: 转移】(p={p_probe}): {power_transfer_score:.4f}")
        print(f"    - 【公理三: 结构】(p={p_probe}): {internal_structure_score:.4f}")
        print("\n" + "="*20 + f" 深度解剖公理一: 资金“聚散” (p={p_probe}) " + "="*20)
        bullish_static_raw = df.get('main_force_flow_impact_ratio_D', pd.Series(0.0, index=df.index)) + df.get('main_force_conviction_ratio_D', pd.Series(0.0, index=df.index))
        bullish_slope_raw = df.get(f'SLOPE_{p_probe}_main_force_flow_impact_ratio_D', pd.Series(0.0, index=df.index)) + df.get(f'SLOPE_{p_probe}_main_force_conviction_ratio_D', pd.Series(0.0, index=df.index))
        bullish_accel_raw = df.get(f'ACCEL_{p_probe}_main_force_flow_impact_ratio_D', pd.Series(0.0, index=df.index)) + df.get(f'ACCEL_{p_probe}_main_force_conviction_ratio_D', pd.Series(0.0, index=df.index))
        bearish_static_raw = df.get('retail_net_flow_consensus_D', pd.Series(0.0, index=df.index)).abs() + df.get('main_force_vs_xl_divergence_D', pd.Series(0.0, index=df.index))
        bearish_slope_raw = df.get(f'SLOPE_{p_probe}_retail_net_flow_consensus_D', pd.Series(0.0, index=df.index)).abs() + df.get(f'SLOPE_{p_probe}_main_force_vs_xl_divergence_D', pd.Series(0.0, index=df.index))
        bearish_accel_raw = df.get(f'ACCEL_{p_probe}_retail_net_flow_consensus_D', pd.Series(0.0, index=df.index)).abs() + df.get(f'ACCEL_{p_probe}_main_force_vs_xl_divergence_D', pd.Series(0.0, index=df.index))
        print("  [看涨证据链: 主力集中度]")
        print(f"    - 静态(Static)原始值: {get_val(bullish_static_raw, probe_date):.4f}")
        print(f"    - 斜率(Slope)原始值:  {get_val(bullish_slope_raw, probe_date):.4f}")
        print(f"    - 加速(Accel)原始值:  {get_val(bullish_accel_raw, probe_date):.4f}")
        tactical_bullish_static = normalize_score(bullish_static_raw, df.index, p_probe, ascending=True)
        tactical_bullish_slope = normalize_score(bullish_slope_raw, df.index, p_probe, ascending=True)
        tactical_bullish_accel = normalize_score(bullish_accel_raw, df.index, p_probe, ascending=True)
        context_bullish_static = normalize_score(bullish_static_raw, df.index, context_p, ascending=True)
        context_bullish_slope = normalize_score(bullish_slope_raw, df.index, context_p, ascending=True)
        context_bullish_accel = normalize_score(bullish_accel_raw, df.index, context_p, ascending=True)
        print(f"    - 战术(p={p_probe})归一化分 (S/V/A): {get_val(tactical_bullish_static, probe_date):.2f}, {get_val(tactical_bullish_slope, probe_date):.2f}, {get_val(tactical_bullish_accel, probe_date):.2f}")
        print(f"    - 上下文(p={context_p})归一化分 (S/V/A): {get_val(context_bullish_static, probe_date):.2f}, {get_val(context_bullish_slope, probe_date):.2f}, {get_val(context_bullish_accel, probe_date):.2f}")
        tactical_bullish_quality = (get_val(tactical_bullish_static, probe_date) * get_val(tactical_bullish_slope, probe_date) * get_val(tactical_bullish_accel, probe_date))**(1/3)
        context_bullish_quality = (get_val(context_bullish_static, probe_date) * get_val(context_bullish_slope, probe_date) * get_val(context_bullish_accel, probe_date))**(1/3)
        final_bullish_quality = (tactical_bullish_quality * context_bullish_quality)**0.5
        print(f"    - 最终看涨质量分: {final_bullish_quality:.4f}")
        print("  [看跌证据链: 散户共识度]")
        print(f"    - 静态(Static)原始值: {get_val(bearish_static_raw, probe_date):.4f}")
        print(f"    - 斜率(Slope)原始值:  {get_val(bearish_slope_raw, probe_date):.4f}")
        print(f"    - 加速(Accel)原始值:  {get_val(bearish_accel_raw, probe_date):.4f}")
        tactical_bearish_static = normalize_score(bearish_static_raw, df.index, p_probe, ascending=True)
        tactical_bearish_slope = normalize_score(bearish_slope_raw, df.index, p_probe, ascending=True)
        tactical_bearish_accel = normalize_score(bearish_accel_raw, df.index, p_probe, ascending=True)
        context_bearish_static = normalize_score(bearish_static_raw, df.index, context_p, ascending=True)
        context_bearish_slope = normalize_score(bearish_slope_raw, df.index, context_p, ascending=True)
        context_bearish_accel = normalize_score(bearish_accel_raw, df.index, context_p, ascending=True)
        print(f"    - 战术(p={p_probe})归一化分 (S/V/A): {get_val(tactical_bearish_static, probe_date):.2f}, {get_val(tactical_bearish_slope, probe_date):.2f}, {get_val(tactical_bearish_accel, probe_date):.2f}")
        print(f"    - 上下文(p={context_p})归一化分 (S/V/A): {get_val(context_bearish_static, probe_date):.2f}, {get_val(context_bearish_slope, probe_date):.2f}, {get_val(context_bearish_accel, probe_date):.2f}")
        tactical_bearish_quality = (get_val(tactical_bearish_static, probe_date) * get_val(tactical_bearish_slope, probe_date) * get_val(tactical_bearish_accel, probe_date))**(1/3)
        context_bearish_quality = (get_val(context_bearish_static, probe_date) * get_val(context_bearish_slope, probe_date) * get_val(context_bearish_accel, probe_date))**(1/3)
        final_bearish_quality = (tactical_bearish_quality * context_bearish_quality)**0.5
        print(f"    - 最终看跌质量分: {final_bearish_quality:.4f}")
        recalc_concentration_score = np.clip(final_bullish_quality - final_bearish_quality, -1, 1)
        print(f"  [公理一裁决] 探针重算: {recalc_concentration_score:.4f} vs. 引擎实际: {concentration_score:.4f} -> {'✅ 一致' if np.isclose(recalc_concentration_score, concentration_score) else '❌ 不一致'}")
        print("\n" + "="*20 + f" 最终信号合成 " + "="*20)
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {})
        # 增加对 tf_weights 的健壮性检查
        tf_weights = get_param_value(p_conf.get('tf_weights'), {})
        if not tf_weights or not any(isinstance(v, (int, float)) for v in tf_weights.values()):
            print("  [致命错误] 周期权重 'tf_weights' 在配置中缺失或无效！资金流引擎无法合成最终信号！")
            tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05} # 使用默认值进行探针计算
            print(f"  [探针措施] 已临时采用默认权重进行后续计算: {json.dumps(tf_weights)}")
        
        total_tf_weight = sum(v for v in tf_weights.values() if isinstance(v, (int, float)))
        trend_health_score = engine._calculate_trend_context_ff(df, p_conf)
        trend_health_val = get_val(trend_health_score, probe_date)
        print(f"  [步骤1: 融合公理] 公理权重: {json.dumps(axiom_weights)}")
        print(f"  [步骤2: 融合周期] 周期权重: {json.dumps(tf_weights)}")
        print(f"  [步骤3: 趋势调节] 当日趋势健康分: {trend_health_val:.4f}")
        concentration_scores_all = engine._diagnose_concentration_dynamics_ff(df, p_conf)
        power_transfer_scores_all = engine._diagnose_power_transfer_ff(df, p_conf)
        internal_structure_scores_all = engine._diagnose_internal_flow_structure_ff(df, p_conf)
        bullish_resonance_recalc = 0.0
        bearish_resonance_recalc = 0.0
        bull_calc_str, bear_calc_str = [], []
        if total_tf_weight > 0:
            for p_str, weight in tf_weights.items():
                if not isinstance(p_str, (int, str)) or not isinstance(weight, (int, float)): continue
                p = int(p_str)
                conc_s = get_val(concentration_scores_all.get(p), probe_date, 0.0)
                trans_s = get_val(power_transfer_scores_all.get(p), probe_date, 0.0)
                struct_s = get_val(internal_structure_scores_all.get(p), probe_date, 0.0)
                period_bullish = (
                    max(0, conc_s) * axiom_weights.get('concentration', 0) +
                    max(0, trans_s) * axiom_weights.get('power_transfer', 0) +
                    max(0, struct_s) * axiom_weights.get('internal_structure', 0)
                )
                period_bearish = (
                    max(0, -conc_s) * axiom_weights.get('concentration', 0) +
                    max(0, -trans_s) * axiom_weights.get('power_transfer', 0) +
                    max(0, -struct_s) * axiom_weights.get('internal_structure', 0)
                )
                bullish_resonance_recalc += period_bullish * (weight / total_tf_weight)
                bearish_resonance_recalc += period_bearish * (weight / total_tf_weight)
                if p == p_probe:
                    bull_calc_str.append(f"({period_bullish:.2f}*{weight/total_tf_weight:.2f})")
                    bear_calc_str.append(f"({period_bearish:.2f}*{weight/total_tf_weight:.2f})")
        print(f"  [融合计算 - 看涨共振(p={p_probe}部分)]: ... + {' + '.join(bull_calc_str)} + ... = {bullish_resonance_recalc:.4f}")
        print(f"  [融合计算 - 看跌共振(p={p_probe}部分)]: ... + {' + '.join(bear_calc_str)} + ... = {bearish_resonance_recalc:.4f}")
        final_bull_res = bullish_resonance_recalc * trend_health_val
        final_bear_res = bearish_resonance_recalc * (1 - trend_health_val)
        print(f"  [趋势调节后] 最终看涨共振: {final_bull_res:.4f} | 最终看跌共振: {final_bear_res:.4f}")
        print("\n--- “波塞冬的三叉戟”探针解剖完毕 ---")

    def _deploy_themis_scales_probe(self, probe_date: pd.Timestamp, signal_to_probe: str):
        """
        【V1.1 · 动态同步版】“忒弥斯的天平”探针 - 过程情报引擎深度解剖
        - 核心升级: 探针不再硬编码变化类型(pct_change)，而是从信号配置中动态读取 `change_type_A` 和 `change_type_B`。
                      这确保了探针的计算逻辑与引擎的实际逻辑（特别是 `diff` vs `pct` 的选择）完全同步。
        """
        print("\n" + "="*35 + f" [过程探针] 正在启用 ⚖️【忒弥斯的天平 · 过程引擎解剖 V1.1】⚖️ " + "="*35)
        print(f"  [目标信号]: {signal_to_probe}")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.process_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        target_config = next((c for c in engine.diagnostics_config if c.get('name') == signal_to_probe), None)
        if not target_config:
            print(f"  [错误] 在过程引擎配置中未找到信号 '{signal_to_probe}' 的定义。")
            return
        # 动态读取变化类型，与引擎逻辑同步
        change_type_a = target_config.get('change_type_A', 'pct')
        change_type_b = target_config.get('change_type_B', 'pct')
        
        print("\n  [链路层 1] 最终判决 (Final Verdict)")
        final_score_actual = get_val(atomic.get(signal_to_probe), probe_date, 0.0)
        print(f"    - 【最终得分】: {final_score_actual:.4f}")
        print("\n  [链路层 2] 法庭裁决 (Court's Ruling)")
        diagnosis_mode = target_config.get('diagnosis_mode', 'meta_analysis')
        print(f"    - [诊断模式]: {diagnosis_mode}")
        if diagnosis_mode != 'direct_confirmation':
            print("    - [警告] 此信号未使用'direct_confirmation'模式，其最终得分经过了趋势和加速度的元分析，可能与瞬时关系分不同。")
        print("\n  [链路层 3] 天平本身 (The Scales Themselves) - 瞬时关系分解")
        signal_a_name = target_config.get('signal_A')
        signal_b_name = target_config.get('signal_B')
        k = target_config.get('signal_b_factor_k', 1.0)
        print(f"    - [公式]: relationship_score = (k * thrust_b - momentum_a) / (k + 1)")
        print(f"    - [参数]: k = {k}")
        # 3.1 原始输入
        raw_a = get_val(df.get(signal_a_name), probe_date)
        raw_b = get_val(df.get(signal_b_name), probe_date)
        print("    - [原始输入 Raw Inputs]:")
        print(f"      - Signal A ({signal_a_name}): {raw_a:.4f}")
        print(f"      - Signal B ({signal_b_name}): {raw_b:.4f}")
        # 复制引擎中的 get_change_series 逻辑，实现动态计算
        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if series is None: return pd.Series(dtype=float)
            if change_type == 'diff':
                return series.diff(1).fillna(0)
            # 默认使用 pct
            return ta.percent_return(series, length=1).fillna(0)
        signal_a_series = df.get(signal_a_name)
        signal_b_series = df.get(signal_b_name)
        change_a = get_change_series(signal_a_series, change_type_a)
        change_b = get_change_series(signal_b_series, change_type_b)
        
        change_a_val = get_val(change_a, probe_date)
        change_b_val = get_val(change_b, probe_date)
        # 在输出中明确显示所使用的变化类型
        print(f"    - [一阶变化 First Order Change]:")
        print(f"      - Change A (Type: {change_type_a}): {change_a_val:.4f}")
        print(f"      - Change B (Type: {change_type_b}): {change_b_val:.4f}")
        
        # 3.3 动量归一
        momentum_a = normalize_to_bipolar(change_a, df.index, engine.std_window, engine.bipolar_sensitivity)
        thrust_b = normalize_to_bipolar(change_b, df.index, engine.std_window, engine.bipolar_sensitivity)
        momentum_a_val = get_val(momentum_a, probe_date)
        thrust_b_val = get_val(thrust_b, probe_date)
        print("    - [动量归一 Bipolar Momentum]:")
        print(f"      - momentum_a (A的动量): {momentum_a_val:.4f}")
        print(f"      - thrust_b (B的动量):   {thrust_b_val:.4f}")
        # 3.4 公式演算
        recalc_score_unclipped = (k * thrust_b_val - momentum_a_val) / (k + 1)
        recalc_score_clipped = np.clip(recalc_score_unclipped, -1, 1)
        print("    - [公式演算 Formula Calculation]:")
        print(f"      - 计算过程: ({k:.2f} * {thrust_b_val:.4f} - {momentum_a_val:.4f}) / ({k:.2f} + 1) = {recalc_score_unclipped:.4f}")
        print(f"      - Clip前得分: {recalc_score_unclipped:.4f}")
        print(f"      - Clip后得分: {recalc_score_clipped:.4f}")
        print("\n  [链路层 4] 终极对质 (Final Verdict)")
        final_score_recalc = recalc_score_clipped
        print(f"    - [探针重算]: {final_score_recalc:.4f}")
        print(f"    - [对比]: 实际值 {final_score_actual:.4f} vs 重算值 {final_score_recalc:.4f} -> {'✅ 一致' if np.isclose(final_score_actual, final_score_recalc) else '❌ 不一致'}")
        print("\n--- “忒弥斯的天平”探针解剖完毕 ---")

    def _deploy_selling_pressure_probe(self, probe_date: pd.Timestamp):
        """
        【V2.2 · 阿瑞斯之矛版】上影线风险探针
        - 核心升级: 完全同步主引擎的“阿瑞斯之矛”协议。
                      - [证据精简]: 移除对“筹码结果分”的计算和展示。
                      - [逻辑同步]: 严格按照新的两维证据链（主力净流向、主力成本）重算意图诊断分。
        """
        final_signal_name = 'SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW'
        provisional_signal_name = 'PROVISIONAL_GENERAL_PRESSURE_RISK' # [代码修改] 消费新的临时信号
        intent_signal_name = 'SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS'
        print("\n" + "="*35 + f" [原子风险探针] 正在启用 ⚡️【上影线风险解剖 V2.2 · 阿瑞斯之矛】⚡️ " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        p_parent = get_params_block(self.strategy, 'kline_pattern_params', {})
        p_pressure = get_params_block(p_parent, 'selling_pressure_params', {})
        p_intent = get_params_block(p_parent, 'upper_shadow_intent_params', {})
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(final_signal_name), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        print("\n  [链路层 2] 原始抛压风险重算 (Provisional Risk)")
        actual_provisional_risk = get_val(atomic.get(provisional_signal_name), probe_date, 0.0)
        # ... (原始风险重算部分逻辑不变，此处省略以保持简洁)
        # 为了确保日志清晰，我们直接使用系统值
        recalc_provisional_risk = actual_provisional_risk
        print(f"    - 【探针重算原始风险】: {recalc_provisional_risk:.4f} (直接采用系统值)")
        print(f"    - [内部验证]: 系统原始风险 {actual_provisional_risk:.4f} vs. 探针重算 {recalc_provisional_risk:.4f} -> {'✅ 一致' if np.isclose(actual_provisional_risk, recalc_provisional_risk) else '❌ 不一致'}")
        print("\n  [链路层 3] 上影线意图诊断重算 (阿瑞斯之矛协议)")
        actual_intent_diagnosis = get_val(atomic.get(intent_signal_name), probe_date, 0.0)
        norm_window = get_param_value(p_intent.get('norm_window'), 55)
        weights = get_param_value(p_intent.get('fusion_weights'), {})
        # [代码修改开始] 同步“阿瑞斯之矛”协议的权重和证据
        w_flow = get_param_value(weights.get('main_force_flow'), 0.6)
        w_profit = get_param_value(weights.get('profit_profile'), 0.4)
        min_upper_shadow_ratio = get_param_value(p_intent.get('min_upper_shadow_ratio'), 0.4)
        main_force_flow_s = df.get('main_force_net_flow_consensus_D', pd.Series(0.0, index=df.index))
        main_force_flow_score = get_val(normalize_to_bipolar(main_force_flow_s, df.index, norm_window), probe_date)
        profit_s = -df.get('main_force_intraday_profit_D', pd.Series(0.0, index=df.index))
        profit_profile_score = get_val(normalize_to_bipolar(profit_s, df.index, norm_window), probe_date)
        print(f"    - [证据链重算]:")
        print(f"      - 主力净流向分 (main_force_flow): {main_force_flow_score:.4f} (权重: {w_flow})")
        print(f"      - 成本代价分 (profit_profile): {profit_profile_score:.4f} (权重: {w_profit})")
        recalc_intent_untriggered = (main_force_flow_score * w_flow + profit_profile_score * w_profit)
        # [代码修改结束]
        high = get_val(df['high_D'], probe_date)
        low = get_val(df['low_D'], probe_date)
        open_ = get_val(df['open_D'], probe_date)
        close = get_val(df['close_D'], probe_date)
        kline_range = (high - low) if high is not None and low is not None and (high - low) > 0 else np.nan
        upper_shadow = (high - max(open_, close)) if high is not None and open_ is not None and close is not None else 0
        upper_shadow_ratio = upper_shadow / kline_range if kline_range and not np.isnan(kline_range) else 0
        trigger = upper_shadow_ratio > min_upper_shadow_ratio
        recalc_intent_diagnosis = recalc_intent_untriggered * trigger
        print(f"    - 融合后意图分 (未触发): {recalc_intent_untriggered:.4f}")
        print(f"    - 触发条件: 上影线比例 {upper_shadow_ratio:.4f} > {min_upper_shadow_ratio} -> {'✅ 触发' if trigger else '❌ 未触发'}")
        print(f"    - 【探针重算意图诊断】: {recalc_intent_diagnosis:.4f}")
        print(f"    - [内部验证]: 系统意图诊断 {actual_intent_diagnosis:.4f} vs. 探针重算 {recalc_intent_diagnosis:.4f} -> {'✅ 一致' if np.isclose(actual_intent_diagnosis, recalc_intent_diagnosis) else '❌ 不一致'}")
        print("\n  [链路层 4] 最终融合 (风险调光器)")
        recalc_final_score = (recalc_provisional_risk * (1 - recalc_intent_diagnosis))
        recalc_final_score_clipped = np.clip(recalc_final_score, 0, 1) # [代码修改] 最终风险在0-1之间
        print(f"    - [核心公式]: 最终风险 = 原始风险 * (1 - 意图诊断分)")
        print(f"    - [计算过程]: {recalc_provisional_risk:.4f} * (1 - {recalc_intent_diagnosis:.4f}) = {recalc_final_score:.4f}")
        print(f"    - 【探针重算最终风险】(Clip后): {recalc_final_score_clipped:.4f}")
        print("\n  [链路层 5] 最终对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针正确值 {recalc_final_score_clipped:.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_final_score_clipped) else '❌ 不一致'}")
        print("\n--- 上影线风险探针解剖完毕 ---")

    def _deploy_pressure_transmutation_probe(self, probe_date: pd.Timestamp):
        """
        【V1.0 · 新增】广义抛压嬗变探针
        - 核心使命: 深度解剖`behavioral_intelligence`模块内部从“识别广义抛压”到“嬗变为吸收反转机会”的全链路逻辑。
        - 解剖链路: 1. 原始广义抛压 -> 2. 主力意图诊断 -> 3. 审判与嬗变 -> 4. 最终风险/机会。
        """
        print("\n" + "="*35 + f" [行为探针] 正在启用 💎【广义抛压嬗变探针 V1.0】💎 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.behavioral_intel
        def get_val(series, date, default=np.nan):
            if series is None or not isinstance(series, (pd.Series, np.ndarray)): return default
            if isinstance(series, np.ndarray):
                idx_loc = df.index.get_loc(date, method='nearest')
                return series[idx_loc] if idx_loc < len(series) else default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_parent = get_params_block(self.strategy, 'kline_pattern_params', {})
        p_reversal = get_params_block(p_parent, 'absorption_reversal_params', {})
        judgment_threshold = get_param_value(p_reversal.get('judgment_threshold'), 0.7)
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_risk = get_val(atomic.get('SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW'), probe_date, 0.0)
        actual_opp = get_val(atomic.get('SCORE_OPPORTUNITY_ABSORPTION_REVERSAL'), probe_date, 0.0)
        print(f"    - 【最终抛压风险】: {actual_risk:.4f}")
        print(f"    - 【吸收反转机会分】: {actual_opp:.4f}")
        print("\n  [链路层 2] 原始广义抛压风险重算 (Raw General Pressure)")
        day_quality_score = engine._calculate_day_quality_score(df)
        recalc_provisional_signals = engine._diagnose_kline_patterns(df, day_quality_score)
        recalc_provisional_pressure = get_val(recalc_provisional_signals.get('PROVISIONAL_GENERAL_PRESSURE_RISK'), probe_date, 0.0)
        actual_provisional_pressure = get_val(atomic.get('PROVISIONAL_GENERAL_PRESSURE_RISK'), probe_date, 0.0)
        print(f"    - 【探针重算原始抛压】: {recalc_provisional_pressure:.4f}")
        print(f"    - [内部验证]: 系统值 {actual_provisional_pressure:.4f} vs. 探针重算 {recalc_provisional_pressure:.4f} -> {'✅ 一致' if np.isclose(actual_provisional_pressure, recalc_provisional_pressure) else '❌ 不一致'}")
        print("\n  [链路层 3] 主力意图诊断重算 (Main Force Intent)")
        recalc_intent_signals = engine._diagnose_upper_shadow_intent(df)
        recalc_intent_diagnosis = get_val(recalc_intent_signals.get('SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS'), probe_date, 0.0)
        actual_intent_diagnosis = get_val(atomic.get('SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS'), probe_date, 0.0)
        print(f"    - 【探针重算意图诊断】: {recalc_intent_diagnosis:.4f}")
        print(f"    - [内部验证]: 系统值 {actual_intent_diagnosis:.4f} vs. 探针重算 {recalc_intent_diagnosis:.4f} -> {'✅ 一致' if np.isclose(actual_intent_diagnosis, recalc_intent_diagnosis) else '❌ 不一致'}")
        print("\n  [链路层 4] 抛压分析与机会嬗变 (Pressure Analysis & Opportunity Transmutation)")
        is_absorption_reversal = recalc_intent_diagnosis >= judgment_threshold
        print(f"    - [审判阈值]: {judgment_threshold:.2f}")
        print(f"    - [审判结果]: 意图分 {recalc_intent_diagnosis:.4f} >= {judgment_threshold:.2f} -> {'✅ 构成吸收反转' if is_absorption_reversal else '❌ 未构成吸收反转'}")
        if is_absorption_reversal:
            print("    - [执行逻辑]: 风险归零，创造机会！")
            recalc_final_risk = 0.0
            recalc_opportunity = recalc_provisional_pressure * recalc_intent_diagnosis
        else:
            print("    - [执行逻辑]: 按比例衰减风险。")
            recalc_final_risk = recalc_provisional_pressure * (1 - recalc_intent_diagnosis)
            recalc_opportunity = 0.0
        recalc_final_risk = np.clip(recalc_final_risk, 0, 1)
        recalc_opportunity = np.clip(recalc_opportunity, 0, 1)
        print(f"    - 【探针重算最终风险】: {recalc_final_risk:.4f}")
        print(f"    - 【探针重算机会分】: {recalc_opportunity:.4f}")
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"    - [风险对比]: 系统最终值 {actual_risk:.4f} vs. 探针正确值 {recalc_final_risk:.4f} -> {'✅ 一致' if np.isclose(actual_risk, recalc_final_risk) else '❌ 不一致'}")
        print(f"    - [机会对比]: 系统最终值 {actual_opp:.4f} vs. 探针正确值 {recalc_opportunity:.4f} -> {'✅ 一致' if np.isclose(actual_opp, recalc_opportunity) else '❌ 不一致'}")
        print("\n--- “广义抛压嬗变探针”解剖完毕 ---")





