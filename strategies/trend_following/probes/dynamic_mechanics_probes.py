import pandas as pd
import numpy as np
import json
from strategies.trend_following.utils import get_params_block, get_param_value, transmute_health_to_ultimate_signals, normalize_to_bipolar

class DynamicMechanicsProbes:
    """
    【探针模块】动态力学情报专属探针
    """
    def __init__(self, intel_layer):
        self.intelligence_layer = intel_layer
        self.strategy = intel_layer.strategy
        self.mechanics_engine = intel_layer.mechanics_engine
    def _deploy_ares_chariot_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 引擎核心解剖版】“阿瑞斯的战车”探针
        - 核心升级: 深入解剖动态力学引擎的元分析过程。
        """
        print("\n" + "="*35 + f" [力学探针] 正在启动 🏎️【力学引擎解剖 V1.1】🏎️ " + "="*35)
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
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print("\n  [链路层 2] 快照融合与调节 (Snapshot Fusion & Modulation)")
        bipolar_mechanics_snapshot_series = engine._calculate_bipolar_snapshot(df, p_conf, norm_window)
        ma_health_score_series = engine._calculate_ma_health(df, p_conf, norm_window)
        modulated_bipolar_snapshot_series = bipolar_mechanics_snapshot_series * ma_health_score_series
        print(f"    - 双极性力学快照: {get_val(bipolar_mechanics_snapshot_series, probe_date):.4f}")
        print(f"    - 均线健康分 (调节器): {get_val(ma_health_score_series, probe_date):.4f}")
        print(f"    - 调节后双极性快照: {get_val(modulated_bipolar_snapshot_series, probe_date):.4f}")
        print("\n  [链路层 3] 终极信号融合 (Ultimate Signal Fusion)")
        overall_health = atomic.get('__DYN_overall_health', {})
        if not overall_health:
            print("    - [错误] 无法在 atomic_states 中找到 '__DYN_overall_health'，无法进行重算。")
            return
        recalc_signals = transmute_health_to_ultimate_signals(df, atomic, overall_health, p_synthesis, "DYN")
        bull_res_recalc = get_val(recalc_signals.get('SCORE_DYN_BULLISH_RESONANCE'), probe_date)
        bear_res_recalc = get_val(recalc_signals.get('SCORE_DYN_BEARISH_RESONANCE'), probe_date)
        print(f"    - 【看涨共振】探针重算: {bull_res_recalc:.4f}")
        print(f"    - 【看跌共振】探针重算: {bear_res_recalc:.4f}")
        print("\n--- “力学引擎探针”解剖完毕 ---")
