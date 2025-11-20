import pandas as pd
import numpy as np
import json
from strategies.trend_following.utils import get_params_block, get_param_value, transmute_health_to_ultimate_signals, normalize_to_bipolar

class FoundationProbes:
    """
    【探针模块】基础情报专属探针
    """
    def __init__(self, intel_layer):
        self.intelligence_layer = intel_layer
        self.strategy = intel_layer.strategy
        self.foundation_intel = intel_layer.foundation_intel
    def _deploy_apollos_lyre_probe(self, probe_date: pd.Timestamp):
        """
        【V1.4 · 三叉戟协议版】“阿波罗的七弦琴”探针
        - 核心升级: 签署“三叉戟协议”，将“商神杖”子探针通用化，使其能够深度解剖RSI、MACD、CMF。
        """
        print("\n" + "="*35 + f" [基础探针] 正在奏响 🎵【基础引擎解剖 V1.4】🎵 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.foundation_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        print("\n  [链路层 1] 最终输出 (Final Output)")
        bull_res_actual = get_val(atomic.get('SCORE_FOUNDATION_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_actual = get_val(atomic.get('SCORE_FOUNDATION_BEARISH_RESONANCE'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print("\n  [链路层 2] 四大支柱快照分 (Pillar Snapshots)")
        ma_context_score = engine._calculate_ma_trend_context(df, [5, 13, 21, 55])
        calculators = {
            'ema': lambda: engine._calculate_ema_health(df, norm_window, []),
            'rsi': lambda: engine._calculate_rsi_health(df, norm_window, [], ma_context_score),
            'macd': lambda: engine._calculate_macd_health(df, norm_window, [], ma_context_score),
            'cmf': lambda: engine._calculate_cmf_health(df, norm_window, [], ma_context_score)
        }
        pillar_source_cols = {'rsi': 'RSI_13_D', 'macd': 'MACDh_13_34_8_D', 'cmf': 'CMF_21_D'}
        for name, calculator in calculators.items():
            snapshot_series = calculator()
            snapshot_val = get_val(snapshot_series, probe_date)
            print(f"    - [支柱: {name.upper()}] 双极性快照分: {snapshot_val:.4f}")
            if name == 'ema':
                self._deploy_caduceus_probe_for_ema(probe_date)
            elif name in pillar_source_cols:
                self._deploy_caduceus_probe_for_indicator(name, pillar_source_cols[name], probe_date)
        print("\n--- “基础引擎探针”解剖完毕 ---")
    def _deploy_caduceus_probe_for_ema(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 修复版】“赫尔墨斯的商神杖”深度诊断单元 (EMA专用)
        """
        print("\n" + "-"*15 + f" [子探针] 正在挥舞 ⚕️【EMA健康度解剖】 ⚕️ " + "-"*15)
        df = self.strategy.df_indicators
        engine = self.foundation_intel
        def get_val(series, date, default=np.nan):
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        fusion_weights = p_conf.get('ma_health_fusion_weights', {})
        norm_window = 55
        ma_periods = [5, 13, 21, 55]
        bull_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).astype(float).values for i in range(len(ma_periods) - 1)]
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df.index), 0.5)
        alignment_bipolar_series = (pd.Series(alignment_score, index=df.index) - 0.5) * 2
        alignment_val = get_val(alignment_bipolar_series, probe_date)
        print(f"      - [维度1: 排列] 双极性分: {alignment_val:.4f}")
        slope_scores = [normalize_to_bipolar(df[f'SLOPE_{p}_EMA_{p}_D'], df.index, norm_window).values for p in ma_periods]
        avg_slope_bipolar_series = pd.Series(np.mean(slope_scores, axis=0), index=df.index)
        slope_val = get_val(avg_slope_bipolar_series, probe_date)
        print(f"      - [维度2: 斜率] 双极性分: {slope_val:.4f}")
        accel_scores = [normalize_to_bipolar(df[f'ACCEL_{p}_EMA_{p}_D'], df.index, norm_window).values for p in ma_periods]
        avg_accel_bipolar_series = pd.Series(np.mean(accel_scores, axis=0), index=df.index)
        accel_val = get_val(avg_accel_bipolar_series, probe_date)
        print(f"      - [维度3: 加速度] 双极性分: {accel_val:.4f}")
        relational_scores = []
        for short_p, long_p in [(5, 21), (13, 55)]:
            spread_accel = (df[f'EMA_{short_p}_D'] - df[f'EMA_{long_p}_D']).diff(3).diff(3).fillna(0)
            relational_scores.append(normalize_to_bipolar(spread_accel, df.index, norm_window).values)
        avg_relational_bipolar_series = pd.Series(np.mean(relational_scores, axis=0), index=df.index)
        relational_val = get_val(avg_relational_bipolar_series, probe_date)
        print(f"      - [维度4: 关系] 双极性分: {relational_val:.4f}")
        meta_dynamics_cols = ['SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D']
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        meta_scores = [normalize_to_bipolar(df[col], df.index, norm_window).values for col in valid_meta_cols] if valid_meta_cols else [np.full(len(df.index), 0.0)]
        avg_meta_bipolar_series = pd.Series(np.mean(meta_scores, axis=0), index=df.index)
        meta_val = get_val(avg_meta_bipolar_series, probe_date)
        print(f"      - [维度5: 元动态] 双极性分: {meta_val:.4f}")
        recalc_snapshot = (alignment_val * fusion_weights.get('alignment', 0.15) + slope_val * fusion_weights.get('slope', 0.15) + accel_val * fusion_weights.get('accel', 0.2) + relational_val * fusion_weights.get('relational', 0.25) + meta_val * fusion_weights.get('meta_dynamics', 0.25))
        recalc_snapshot = np.clip(recalc_snapshot, -1, 1)
        actual_snapshot_series = engine._calculate_ema_health(df, norm_window, [])
        actual_val = get_val(actual_snapshot_series, probe_date)
        print(f"      - [最终融合] 探针重算: {recalc_snapshot:.4f} vs. 引擎实际: {actual_val:.4f} -> {'✅ 一致' if np.isclose(recalc_snapshot, actual_val) else '❌ 不一致'}")
        print("-"*(32+22) + "\n")
    def _deploy_caduceus_probe_for_indicator(self, indicator_name: str, source_col: str, probe_date: pd.Timestamp):
        """
        【V2.0 · 三叉戟协议版】通用指标深度诊断单元
        """
        print("\n" + "-"*15 + f" [子探针] 正在挥舞 ⚕️【{indicator_name.upper()}健康度解剖】 ⚕️ " + "-"*15)
        df = self.strategy.df_indicators
        engine = self.foundation_intel
        def get_val(series, date, default=np.nan):
            val = series.get(date)
            return default if pd.isna(val) else val
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        if source_col not in df.columns:
            print(f"      - [错误] 找不到源数据列: {source_col}，无法进行解剖。")
            print("-"*(32+22) + "\n")
            return
        original_series = df[source_col]
        state_score_series = normalize_to_bipolar(original_series, df.index, norm_window, bipolar_sensitivity)
        state_val = get_val(state_score_series, probe_date)
        print(f"      - [维度1: 状态] 双极性分: {state_val:.4f}")
        if indicator_name == 'rsi':
            actual_series = engine._calculate_rsi_health(df, norm_window, [], None)
        elif indicator_name == 'macd':
            actual_series = engine._calculate_macd_health(df, norm_window, [], None)
        elif indicator_name == 'cmf':
            actual_series = engine._calculate_cmf_health(df, norm_window, [], None)
        else:
            actual_series = pd.Series(np.nan, index=df.index)
        actual_val = get_val(actual_series, probe_date)
        recalc_val = state_val
        print(f"      - [最终验证] 探针重算(State): {recalc_val:.4f} vs. 引擎实际: {actual_val:.4f} -> {'✅ 一致' if np.isclose(recalc_val, actual_val) else '❌ 不一致'}")
        print("-"*(32+22) + "\n")
