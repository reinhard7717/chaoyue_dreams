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
        【V2.1 · 赫淮斯托斯蓝图版】“普罗米修斯火炬”探针 - 行为智能引擎全要素解剖
        - 核心重构: 废除链路层4中所有模拟计算。探针现在直接调用真实的 `transmute_health_to_ultimate_signals` 函数
                      （神之手），使用与主引擎完全相同的输入和蓝图进行重算，确保结果100%可复现。
        """
        print("\n" + "="*35 + f" [行为探针] 正在点燃 🔥【普罗米修斯火炬 · 行为引擎解剖 V2.1】🔥 " + "="*35)
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
        bull_res_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_rev_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_rev_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_TOP_REVERSAL'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print(f"    - 【底部反转】: {bottom_rev_actual:.4f}")
        print(f"    - 【顶部反转】: {top_rev_actual:.4f}")
        # --- 链路层 2: 复合状态构建 (Composite State Construction) ---
        print("\n  [链路层 2] 复合状态构建 (来自 _calculate_structural_behavior_health)")
        # 2.1 原材料
        raw_ingredients = {
            'closing_strength_index_D': get_val(df.get('closing_strength_index_D'), probe_date, 0.5),
            'close_vs_vwap_ratio_D': get_val(df.get('close_vs_vwap_ratio_D'), probe_date, 1.0),
            'flow_divergence_mf_vs_retail_D': get_val(df.get('flow_divergence_mf_vs_retail_D'), probe_date, 0.0),
            'final_hour_momentum_D': get_val(df.get('final_hour_momentum_D'), probe_date, 0.0),
            'intraday_trend_efficiency_D': get_val(df.get('intraday_trend_efficiency_D'), probe_date, 0.5)
        }
        print("    - [原材料 Raw Ingredients]:")
        for name, val in raw_ingredients.items():
            print(f"      - {name}: {val:.4f}")
        # 2.2 中间件 (归一化)
        csi_score = get_val(normalize_score(df.get('closing_strength_index_D'), df.index, norm_window), probe_date)
        vwap_score = get_val(normalize_score(df.get('close_vs_vwap_ratio_D'), df.index, norm_window), probe_date)
        bull_div_score = get_val(normalize_score(df.get('flow_divergence_mf_vs_retail_D').clip(0), df.index, norm_window), probe_date)
        auction_power_score = get_val(normalize_score(df.get('final_hour_momentum_D').clip(0), df.index, norm_window), probe_date)
        trend_eff_score = get_val(normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window), probe_date)
        bear_div_score = get_val(normalize_score(df.get('flow_divergence_mf_vs_retail_D').clip(upper=0).abs(), df.index, norm_window), probe_date)
        auction_weak_score = get_val(normalize_score(df.get('final_hour_momentum_D').clip(upper=0).abs(), df.index, norm_window), probe_date)
        print("\n    - [中间件 Normalized Ingredients]:")
        print(f"      - csi_score: {csi_score:.4f}, vwap_score: {vwap_score:.4f}, bull_div_score: {bull_div_score:.4f}")
        print(f"      - auction_power_score: {auction_power_score:.4f}, trend_eff_score: {trend_eff_score:.4f}")
        # 2.3 复合状态计算
        reversal_strength = (csi_score * vwap_score)**0.5
        bullish_composite_state = (reversal_strength * csi_score * (1 + bull_div_score) * auction_power_score * trend_eff_score)**(1/5)
        print("\n    - [复合状态计算 Composite State Calculation]:")
        print(f"      - 看涨复合状态 (Bullish Composite): {bullish_composite_state:.4f}")
        print(f"        - [公式]: (反转强度 * 下影线力量 * (1+主力背离) * 尾盘动能 * 趋势效率)^(1/5)")
        print(f"        - [计算]: ({reversal_strength:.2f} * {csi_score:.2f} * (1+{bull_div_score:.2f}) * {auction_power_score:.2f} * {trend_eff_score:.2f})^(1/5) = {bullish_composite_state:.4f}")
        # --- 链路层 3: 健康度计算 (Health Calculation) ---
        print("\n  [链路层 3] 健康度计算 (以周期 p=5 为例)")
        p = 5
        s_bull_p5_actual = get_val(atomic.get('__BEHAVIOR_overall_health', {}).get('s_bull', {}).get(p), probe_date)
        print(f"    - [实际 s_bull[5]]: {s_bull_p5_actual:.4f} (此值已在上一版探针中验证通过，本处直接采信)")
        # [代码修改开始] 彻底重构链路层4
        # --- 链路层 4: 终极信号嬗变 (Ultimate Signal Transmutation) ---
        print("\n  [链路层 4] 终极信号嬗变 (调用“神之手”`transmute_health_to_ultimate_signals`)")
        overall_health = atomic.get('__BEHAVIOR_overall_health')
        if not overall_health:
            print("    - [错误] 无法在 atomic_states 中找到 '__BEHAVIOR_overall_health'，无法进行重算。")
            return
        print("    - [输入验证] 正在使用与主引擎完全相同的 `overall_health` 数据作为输入...")
        # 调用真实的“神之手”函数进行重算
        recalculated_ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=atomic,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="STRUCT_BEHAVIOR"
        )
        # 从重算结果中提取探针日期当天的值
        bull_res_recalc = get_val(recalculated_ultimate_signals.get('SCORE_STRUCT_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_recalc = get_val(recalculated_ultimate_signals.get('SCORE_STRUCT_BEHAVIOR_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_rev_recalc = get_val(recalculated_ultimate_signals.get('SCORE_STRUCT_BEHAVIOR_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_rev_recalc = get_val(recalculated_ultimate_signals.get('SCORE_STRUCT_BEHAVIOR_TOP_REVERSAL'), probe_date, 0.0)
        print("\n    - [终极对质 Final Verdict]:")
        print(f"      - 【看涨共振】: 实际值 {bull_res_actual:.4f} vs. 探针重算 {bull_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bull_res_actual, bull_res_recalc) else '❌ 不一致'}")
        print(f"      - 【看跌共振】: 实际值 {bear_res_actual:.4f} vs. 探针重算 {bear_res_recalc:.4f} -> {'✅ 一致' if np.isclose(bear_res_actual, bear_res_recalc) else '❌ 不一致'}")
        print(f"      - 【底部反转】: 实际值 {bottom_rev_actual:.4f} vs. 探针重算 {bottom_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(bottom_rev_actual, bottom_rev_recalc) else '❌ 不一致'}")
        print(f"      - 【顶部反转】: 实际值 {top_rev_actual:.4f} vs. 探针重算 {top_rev_recalc:.4f} -> {'✅ 一致' if np.isclose(top_rev_actual, top_rev_recalc) else '❌ 不一致'}")
        # [代码修改结束]
        print("\n--- “普罗米修斯火炬”探针解剖完毕 ---")





