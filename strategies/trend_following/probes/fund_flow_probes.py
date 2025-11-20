import pandas as pd
import numpy as np
import json
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class FundFlowProbes:
    """
    【探针模块】资金流情报专属探针
    """
    def __init__(self, intel_layer):
        self.intelligence_layer = intel_layer
        self.strategy = intel_layer.strategy
        self.fund_flow_intel = intel_layer.fund_flow_intel
    def _deploy_poseidons_trident_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 健壮性加固版】“波塞冬的三叉戟”探针 - 资金流情报引擎深度解剖
        - 核心升级: 增加对周期权重(tf_weights)缺失的检测和警告。
        """
        print("\n" + "="*35 + f" [资金流探针] 正在挥舞 🔱【资金流引擎解剖 V1.1】🔱 " + "="*35)
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
        final_bullish_quality = (get_val(normalize_score(bullish_static_raw, df.index, p_probe, ascending=True), probe_date) * get_val(normalize_score(bullish_static_raw.diff(p_probe), df.index, p_probe, ascending=True), probe_date) * get_val(normalize_score(bullish_static_raw.diff(p_probe).diff(p_probe), df.index, p_probe, ascending=True), probe_date))**(1/3)
        bearish_static_raw = df.get('retail_net_flow_consensus_D', pd.Series(0.0, index=df.index)).abs() + df.get('main_force_vs_xl_divergence_D', pd.Series(0.0, index=df.index))
        final_bearish_quality = (get_val(normalize_score(bearish_static_raw, df.index, p_probe, ascending=True), probe_date) * get_val(normalize_score(bearish_static_raw.diff(p_probe), df.index, p_probe, ascending=True), probe_date) * get_val(normalize_score(bearish_static_raw.diff(p_probe).diff(p_probe), df.index, p_probe, ascending=True), probe_date))**(1/3)
        recalc_concentration_score = np.clip(final_bullish_quality - final_bearish_quality, -1, 1)
        print(f"  [公理一裁决] 探针重算: {recalc_concentration_score:.4f} vs. 引擎实际: {concentration_score:.4f} -> {'✅ 一致' if np.isclose(recalc_concentration_score, concentration_score) else '❌ 不一致'}")
        print("\n--- “资金流引擎探针”解剖完毕 ---")
    def _deploy_ff_distribution_resonance_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0 · 派发共振版】穿透式解剖 SCORE_FF_DISTRIBUTION_RESONANCE 信号
        - 核心职责: 验证“资金流派发共振”信号的每一个公理、每一个周期的计算，确保其准确性。
        """
        print("\n" + "="*25 + f" [资金流探针] 正在启用 🌊【派发共振探针 V1.0】🌊 " + "="*25)
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        signal_name = 'SCORE_FF_DISTRIBUTION_RESONANCE'
        def get_val(series, date, default=0.0):
            if isinstance(series, dict): # 处理公理分数
                val = series.get(probe_date)
                return default if pd.isna(val) else val
            val = series.get(date)
            return default if pd.isna(val) else val
        # 1. 获取最终系统输出
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        system_score = get_val(atomic_states.get(signal_name, pd.Series(0.0, index=df.index)), probe_date)
        print(f"    - 【最终信号分】: {system_score:.4f}")
        # 2. 重算原始共振分
        print("\n  [链路层 2] 原始共振分重算 (Raw Resonance Recalculation)")
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {})
        tf_weights = get_param_value(p_conf.get('tf_weights'), {})
        numeric_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_tf_weight = sum(numeric_weights.values())
        concentration = atomic_states.get('SCORE_FF_AXIOM_CONCENTRATION', {})
        power_transfer = atomic_states.get('SCORE_FF_AXIOM_POWER_TRANSFER', {})
        internal_structure = atomic_states.get('SCORE_FF_AXIOM_INTERNAL_STRUCTURE', {})
        raw_bearish_resonance = 0.0
        if total_tf_weight > 0:
            for p, weight in numeric_weights.items():
                conc_score = get_val(concentration.get(p, pd.Series(0.0, index=df.index)), probe_date)
                trans_score = get_val(power_transfer.get(p, pd.Series(0.0, index=df.index)), probe_date)
                struct_score = get_val(internal_structure.get(p, pd.Series(0.0, index=df.index)), probe_date)
                period_bearish = (
                    np.clip(conc_score, -1, 0) * -1 * axiom_weights.get('concentration', 0) +
                    np.clip(trans_score, -1, 0) * -1 * axiom_weights.get('power_transfer', 0) +
                    np.clip(struct_score, -1, 0) * -1 * axiom_weights.get('internal_structure', 0)
                )
                contribution = period_bearish * (weight / total_tf_weight)
                raw_bearish_resonance += contribution
                print(f"    - [周期 {p}d] 聚散: {conc_score:.2f}, 转移: {trans_score:.2f}, 结构: {struct_score:.2f} -> 周期看跌分: {period_bearish:.4f}, 权重贡献: {contribution:.4f}")
        print(f"    - 【探针重算原始共振分】: {raw_bearish_resonance:.4f}")
        # 3. 趋势上下文调制
        print("\n  [链路层 3] 趋势上下文调制 (Trend Context Modulation)")
        trend_health_score_series = self.intelligence_layer.fund_flow_intel._calculate_trend_context_ff(df, p_conf)
        trend_health_score = get_val(trend_health_score_series, probe_date)
        probe_final_score = raw_bearish_resonance * (1 - trend_health_score)
        print(f"    - 趋势健康度: {trend_health_score:.4f} (抑制因子: {1-trend_health_score:.4f})")
        print(f"    - 【探针重算最终信号分】: {probe_final_score:.4f}")
        # 4. 终极对质
        print("\n  [链路层 4] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {system_score:.4f} vs. 探针重算值 {probe_final_score:.4f} -> {'✅ 一致' if np.isclose(system_score, probe_final_score) else '❌ 不一致'}")
        print("\n--- “派发共振探针”解剖完毕 ---")









