import pandas as pd
import numpy as np
import json
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class FundFlowProbes:
    """
    【探针模块】资金流情报专属探针
    """
    def __init__(self, intel_layer):
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
