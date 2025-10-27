import pandas as pd
import numpy as np
import json
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class ChipProbes:
    """
    【探针模块】筹码情报专属探针
    """
    def __init__(self, intel_layer):
        self.intelligence_layer = intel_layer
        self.strategy = intel_layer.strategy
        self.chip_intel = intel_layer.chip_intel

    def _deploy_hephaestus_forge_probe(self, probe_date: pd.Timestamp):
        """
        【V1.5 · 焦点转移版】“赫菲斯托斯熔炉”探针
        - 核心升级: 将解剖焦点从“公理一”转移至“公理四：筹码峰健康度”。
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
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
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
        print("\n--- “赫菲斯托斯熔炉”探针解剖完毕 ---")

    def _deploy_chip_resonance_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 健壮性修复版】筹码共振探针
        - 核心修复: 修复了对 tf_fusion_weights 键进行排序时，未过滤 'description' 键导致的 ValueError。
        """
        import json
        print("\n" + "="*35 + f" [筹码探针] 正在启用 🏛️【筹码共振探针 V1.1】🏛️ " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.chip_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights_raw = p_conf.get('tf_fusion_weights', {})
        periods_str = [k for k in tf_weights_raw.keys() if str(k).isdigit()]
        periods = sorted([int(p) for p in periods_str])
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {})
        tf_weights = {int(k): v for k, v in tf_weights_raw.items() if str(k).isdigit()}
        numeric_tf_weights = tf_weights
        total_tf_weight = sum(numeric_tf_weights.values())
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_bull_res = get_val(atomic.get('SCORE_CHIP_BULLISH_RESONANCE'), probe_date, 0.0)
        actual_bear_res = get_val(atomic.get('SCORE_CHIP_BEARISH_RESONANCE'), probe_date, 0.0)
        print(f"    - 【看涨共振分】: {actual_bull_res:.4f}")
        print(f"    - 【看跌共振分】: {actual_bear_res:.4f}")
        print("\n  [链路层 2] 多周期健康分 (MTF Health Scores)")
        bipolar_health_by_period = {}
        bullish_scores_by_period = {}
        bearish_scores_by_period = {}
        concentration_scores = engine._diagnose_concentration_dynamics(df, periods)
        accumulation_scores = engine._diagnose_main_force_action(df, periods)
        power_transfer_scores = engine._diagnose_power_transfer(df, periods)
        peak_integrity_scores = engine._diagnose_peak_integrity_dynamics(df, periods)
        for p in periods:
            conc_score = get_val(concentration_scores.get(p), probe_date, 0.0)
            acc_score = get_val(accumulation_scores.get(p), probe_date, 0.0)
            pow_score = get_val(power_transfer_scores.get(p), probe_date, 0.0)
            peak_score = get_val(peak_integrity_scores.get(p), probe_date, 0.0)
            health = (conc_score * axiom_weights.get('concentration', 0) + acc_score * axiom_weights.get('accumulation', 0) + pow_score * axiom_weights.get('power_transfer', 0) + peak_score * axiom_weights.get('peak_integrity', 0))
            bipolar_health_by_period[p] = np.clip(health, -1, 1)
            bullish_scores_by_period[p] = max(0, bipolar_health_by_period[p])
            bearish_scores_by_period[p] = max(0, -bipolar_health_by_period[p])
            print(f"    - [周期 {p:2d}] 双极性健康分: {bipolar_health_by_period[p]:.4f} -> 看涨: {bullish_scores_by_period[p]:.4f}, 看跌: {bearish_scores_by_period[p]:.4f}")
        print("\n--- “筹码共振探针”解剖完毕 ---")
