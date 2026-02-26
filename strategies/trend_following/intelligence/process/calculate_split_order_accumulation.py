# strategies/trend_following/intelligence/process/calculate_split_order_accumulation.py
# 拆单吸筹强度计算器 已完成 deepThink
import json
import os
import pandas as pd
import numpy as np
from numba import jit, float64, int64
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

@jit(nopython=True)
def _numba_robust_dynamics(data, win=5, abs_threshold=1e-4, change_threshold=1e-5):
    """V8.0.1 · 鲁棒动力学算子 (引入 Tanh Threshold Gate 防止零基粉碎)"""
    n = len(data)
    slope = np.zeros(n, dtype=np.float32)
    accel = np.zeros(n, dtype=np.float32)
    jerk = np.zeros(n, dtype=np.float32)
    clean_data = np.copy(data).astype(np.float32)
    for i in range(n):
        if np.abs(clean_data[i]) < abs_threshold: clean_data[i] = 0.0
    for i in range(win, n):
        delta = clean_data[i] - clean_data[i-win]
        norm_delta = 0.0 if np.abs(delta) < change_threshold else np.tanh(np.abs(delta) / (change_threshold * 10.0 + 1e-9)) * delta
        slope[i] = norm_delta / win
    for i in range(win, n):
        delta_s = slope[i] - slope[i-1]
        norm_s = 0.0 if np.abs(delta_s) < (change_threshold / 5.0) else np.tanh(np.abs(delta_s) / (change_threshold + 1e-9)) * delta_s
        accel[i] = norm_s
    for i in range(win, n):
        delta_a = accel[i] - accel[i-1]
        norm_a = 0.0 if np.abs(delta_a) < (change_threshold / 25.0) else np.tanh(np.abs(delta_a) / (change_threshold / 5.0 + 1e-9)) * delta_a
        jerk[i] = norm_a
    return slope, accel, jerk

@jit(nopython=True)
def _numba_rolling_robust_norm(data, window=21):
    """V8.0.1 · 零点锚定极性归一化算子 (Zero-Centered MAD)"""
    n = len(data)
    out = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        slice_data = data[i - window + 1 : i + 1]
        median = np.median(slice_data)
        mad = np.median(np.abs(slice_data - median)) + 1e-4
        out[i] = np.tanh((data[i] - median) / (mad * 3.0)) * 3.0
    return out

@jit(nopython=True)
def _numba_power_law_activation(x, power=1.5, leak=0.1):
    """V8.0.1 · 幂律极化泄露算子"""
    n = len(x)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        out[i] = x[i] ** power if x[i] > 0 else - (np.abs(x[i]) ** power) * leak
    return out

@jit(nopython=True)
def _numba_hab_impact(data, windows):
    """V8.0.1 · HAB 历史累积与冲击强度算子"""
    n = len(data)
    num_wins = len(windows)
    hab_stock = np.zeros((num_wins, n), dtype=np.float32)
    shock_intensity = np.zeros((num_wins, n), dtype=np.float32)
    for w_idx in range(num_wins):
        w = windows[w_idx]
        for i in range(w, n):
            stock = 0.0
            for j in range(i - w, i): stock += np.abs(data[j])
            hab_stock[w_idx, i] = stock
            avg_stock = stock / w
            shock_intensity[w_idx, i] = data[i] / (avg_stock + 1e-5)
    return hab_stock, shock_intensity

@jit(nopython=True)
def _numba_hawkes_process(events, decay_rate=0.2):
    """V8.0.1 · 霍克斯过程自激算子"""
    n = len(events)
    intensity = np.zeros(n, dtype=np.float32)
    curr_intensity = 0.0
    for i in range(n):
        curr_intensity = curr_intensity * np.exp(-decay_rate) + events[i]
        intensity[i] = curr_intensity
    return intensity

@jit(nopython=True)
def _numba_langevin_dynamics(data, window=21):
    """V8.0.1 · 朗之万动力学漂移与扩散算子"""
    n = len(data)
    drift = np.zeros(n, dtype=np.float32)
    diffusion = np.zeros(n, dtype=np.float32)
    for i in range(window, n):
        diffs = np.empty(window, dtype=np.float32)
        for j in range(window): diffs[j] = data[i - window + j + 1] - data[i - window + j]
        drift[i] = np.mean(diffs)
        diffusion[i] = np.std(diffs) + 1e-5
    return drift, diffusion

@jit(nopython=True)
def _numba_quantum_tunneling(kinetic, potential, mass=1.0):
    """V8.0.1 · 量子隧穿概率算子"""
    n = len(kinetic)
    prob = np.zeros(n, dtype=np.float32)
    for i in range(n):
        E, V = kinetic[i], potential[i]
        if E >= V: prob[i] = 1.0
        else: prob[i] = np.exp(-2.0 * np.sqrt(2.0 * mass * (V - E + 1e-5)))
    return prob

class CalculateSplitOrderAccumulation:
    """PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY 拆单吸筹计算引擎"""
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.process_params = self.helper.params

    def _setup_debug_info(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp], Dict]:
        is_debug_enabled_for_method = get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None: is_debug_enabled_for_method = False
        _temp_debug_values = {}
        if is_debug_enabled_for_method and probe_ts:
            _temp_debug_values[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        return is_debug_enabled_for_method, probe_ts, _temp_debug_values

    def _validate_all_required_signals(self, df: pd.DataFrame, method_name: str, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> bool:
        base_req = ['close_D', 'volume_D', 'turnover_rate_f_D', 'pct_change_D', 'tick_clustering_index_D', 'high_freq_flow_skewness_D', 'high_freq_flow_kurtosis_D', 'stealth_flow_ratio_D', 'tick_abnormal_volume_ratio_D', 'chip_convergence_ratio_D', 'intraday_chip_entropy_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'hidden_accumulation_intensity_D', 'VPA_MF_ADJUSTED_EFF_D', 'price_flow_divergence_D', 'pressure_trapped_D', 'STATE_PARABOLIC_WARNING_D', 'IS_MARKET_LEADER_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 'large_order_anomaly_D']
        return self.helper._validate_required_signals(df, base_req, method_name)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Dict[str, pd.Series]:
        raw_cols = ['close_D', 'volume_D', 'turnover_rate_f_D', 'pct_change_D', 'tick_clustering_index_D', 'high_freq_flow_skewness_D', 'high_freq_flow_kurtosis_D', 'stealth_flow_ratio_D', 'tick_abnormal_volume_ratio_D', 'chip_convergence_ratio_D', 'intraday_chip_entropy_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'hidden_accumulation_intensity_D', 'VPA_MF_ADJUSTED_EFF_D', 'price_flow_divergence_D', 'pressure_trapped_D', 'STATE_PARABOLIC_WARNING_D', 'IS_MARKET_LEADER_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', 'large_order_anomaly_D']
        raw_signals = {}
        for col in raw_cols:
            if col in df.columns: raw_signals[col] = df[col].ffill().fillna(0.0).astype(np.float32)
            else: raw_signals[col] = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        if raw_signals['pct_change_D'].abs().max() > 0.5: raw_signals['pct_change_D'] *= np.float32(0.01)
        if raw_signals['turnover_rate_f_D'].max() > 1.5: raw_signals['turnover_rate_f_D'] *= np.float32(0.01)
        
        hab_windows = np.array([13, 21, 34, 55], dtype=np.int32)
        hab_targets = ['stealth_flow_ratio_D', 'hidden_accumulation_intensity_D', 'tick_clustering_index_D']
        prefix_map = ['STEALTH', 'HIDDEN_ACCUM', 'TICK_CLUS']
        for target, prefix in zip(hab_targets, prefix_map):
            stock, shock = _numba_hab_impact(raw_signals[target].values.astype(np.float32), hab_windows)
            for i, w in enumerate([13, 21, 34, 55]):
                raw_signals[f'HAB_STOCK_{w}_{prefix}'] = pd.Series(stock[i], index=df.index, dtype=np.float32)
                raw_signals[f'HAB_SHOCK_{w}_{prefix}'] = pd.Series(shock[i], index=df.index, dtype=np.float32)
        
        drift, diffusion = _numba_langevin_dynamics(raw_signals['hidden_accumulation_intensity_D'].values.astype(np.float32), 21)
        raw_signals['LANGEVIN_DRIFT_ACCUM'] = pd.Series(drift, index=df.index, dtype=np.float32)
        raw_signals['LANGEVIN_DIFF_ACCUM'] = pd.Series(diffusion, index=df.index, dtype=np.float32)
        
        threshold_map = {'tick_clustering_index_D': (0.01, 0.005), 'stealth_flow_ratio_D': (0.01, 0.005), 'hidden_accumulation_intensity_D': (0.01, 0.005), 'intraday_chip_entropy_D': (0.01, 0.005)}
        fib_wins = [3, 5, 8, 13, 21]
        for col, (abs_th, chg_th) in threshold_map.items():
            if col in raw_signals:
                base_vals = raw_signals[col].values
                for win in fib_wins:
                    s, a, j = _numba_robust_dynamics(base_vals, win=win, abs_threshold=abs_th, change_threshold=chg_th)
                    raw_signals[f"SLOPE_{win}_{col}"] = pd.Series(s, index=df.index, dtype=np.float32)
                    raw_signals[f"ACCEL_{win}_{col}"] = pd.Series(a, index=df.index, dtype=np.float32)
                    raw_signals[f"JERK_{win}_{col}"] = pd.Series(j, index=df.index, dtype=np.float32)
        return raw_signals

    def _calculate_kinetic_energy(self, raw: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        hawkes = _numba_hawkes_process(np.clip(raw['tick_clustering_index_D'].values.astype(np.float32), 0.0, None), 0.2)
        hawkes_z = _numba_rolling_robust_norm(hawkes, 21)
        skew_z = _numba_rolling_robust_norm(raw['high_freq_flow_skewness_D'].values, 21)
        kurt_z = _numba_rolling_robust_norm(raw['high_freq_flow_kurtosis_D'].values, 21)
        stealth_z = _numba_rolling_robust_norm(raw['stealth_flow_ratio_D'].values, 21)
        vpa = _numba_rolling_robust_norm(raw['VPA_MF_ADJUSTED_EFF_D'].values, 21)
        
        w_stealth = np.clip(1.0 + vpa * 0.2, 0.5, 1.5).astype(np.float32)
        base_ke = (hawkes_z * 0.35 + skew_z * 0.15 + kurt_z * 0.15 + stealth_z * 0.35 * w_stealth).astype(np.float32)
        accel_bonus = np.where(raw['ACCEL_5_stealth_flow_ratio_D'].values > 0, 0.5, 0.0).astype(np.float32)
        jerk_bonus = np.where(raw['JERK_3_stealth_flow_ratio_D'].values > 0.5, 0.3, 0.0).astype(np.float32)
        ke = _numba_rolling_robust_norm(base_ke + accel_bonus + jerk_bonus, 21)
        return pd.Series(ke, index=df_index, dtype=np.float32)

    def _calculate_potential_barrier(self, raw: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        conv_z = _numba_rolling_robust_norm(raw['chip_convergence_ratio_D'].values, 21)
        ent_z = _numba_rolling_robust_norm(raw['intraday_chip_entropy_D'].values, 21)
        comp_z = _numba_rolling_robust_norm(raw['MA_POTENTIAL_COMPRESSION_RATE_D'].values, 21)
        trap_z = _numba_rolling_robust_norm(raw['pressure_trapped_D'].values, 21)
        
        base_pb = (comp_z * 0.3 + conv_z * 0.3 - ent_z * 0.2 + trap_z * 0.2).astype(np.float32)
        pct = raw['pct_change_D'].values
        break_penalty = np.where(pct < -0.05, 1.0, 0.0).astype(np.float32)
        pb = _numba_rolling_robust_norm(base_pb + break_penalty, 21)
        return pd.Series(pb, index=df_index, dtype=np.float32)

    def _calculate_hab_langevin(self, raw: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        shock_stealth = raw['HAB_SHOCK_21_STEALTH'].values
        shock_hidden = raw['HAB_SHOCK_21_HIDDEN_ACCUM'].values
        drift = raw['LANGEVIN_DRIFT_ACCUM'].values
        diffusion = raw['LANGEVIN_DIFF_ACCUM'].values
        
        snr = drift / (diffusion + 1e-9)
        snr_z = _numba_rolling_robust_norm(snr, 21)
        combined = (shock_stealth * 0.3 + shock_hidden * 0.4 + snr_z * 0.3).astype(np.float32)
        return pd.Series(_numba_rolling_robust_norm(combined, 21), index=df_index, dtype=np.float32)

    def _calculate_anti_spoofing_penalty(self, raw: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        div_z = _numba_rolling_robust_norm(raw['price_flow_divergence_D'].values, 21)
        turn = raw['turnover_rate_f_D'].values
        turn_ma = pd.Series(turn).rolling(21).mean().replace(0, 1e-9).values.astype(np.float32)
        turn_ratio = turn / turn_ma
        pct = raw['pct_change_D'].values
        
        is_spoofing = (div_z > 1.5) & (turn_ratio > 2.0) & (pct < 0)
        is_dist = (raw['stealth_flow_ratio_D'].values < 0) & (pct < -0.03) & (raw['pressure_trapped_D'].values > 0.8)
        sm_div = raw['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'].values > 0.8
        large_anomaly = raw['large_order_anomaly_D'].values > 0.8
        
        penalty = np.where(is_spoofing | is_dist | sm_div | large_anomaly, 0.1, 1.0).astype(np.float32)
        return pd.Series(penalty, index=df_index, dtype=np.float32)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V8.0.1 · 单极性 ReLU 截断修复，零容忍虚假吸筹假阳性"""
        method_name = "calculate_split_order_accumulation"
        is_debug, probe_ts, debug_output = self._setup_debug_info(df, method_name)
        df_index = df.index
        if not self._validate_all_required_signals(df, method_name, is_debug, probe_ts): return pd.Series(0.0, index=df_index, dtype=np.float32)
        
        raw = self._get_raw_signals(df, method_name)
        ke = self._calculate_kinetic_energy(raw, df_index)
        pb = self._calculate_potential_barrier(raw, df_index)
        hl = self._calculate_hab_langevin(raw, df_index)
        penalty = self._calculate_anti_spoofing_penalty(raw, df_index)
        
        ke_val = np.clip(1.0 + ke.values * 0.5, 0.1, 3.0)
        pb_val = np.clip(1.0 - pb.values * 0.5, 0.1, 3.0) 
        tunnel_prob = _numba_quantum_tunneling(ke_val, np.clip(pb.values, 0.1, 3.0), mass=1.0)
        tunnel_val = np.clip(tunnel_prob * 3.0, 0.1, 3.0)
        hl_val = np.clip(1.0 + hl.values * 0.5, 0.1, 3.0)
        
        num_dim = 4.0
        geom_mean = np.power(np.clip(ke_val * pb_val * tunnel_val * hl_val, 1e-20, None), 1.0/num_dim).astype(np.float32)
        arith_mean = ((ke_val + pb_val + tunnel_val + hl_val) / num_dim).astype(np.float32)
        
        matrix_scores = np.vstack([ke_val, pb_val, tunnel_val, hl_val])
        min_dim_score = np.min(matrix_scores, axis=0)
        collapse_penalty = np.where(min_dim_score < 0.5, np.clip((min_dim_score / 0.5) ** 2, 0.01, 1.0), 1.0).astype(np.float32)
        
        blend = (0.7 * geom_mean + 0.3 * arith_mean) * collapse_penalty * penalty.values
        
        # 【核心修复：单极性截断 Unipolar ReLU】
        # 只要融合分 < 1.0 (未突破吸筹基准线，或被防爆网压制)，直接物理归零。
        accumulation_raw = np.maximum(blend - 1.0, 0.0).astype(np.float32)
        activated = _numba_power_law_activation(accumulation_raw, power=1.5, leak=0.0)
        
        parabolic = raw['STATE_PARABOLIC_WARNING_D'].values > 0.5
        is_leader = raw['IS_MARKET_LEADER_D'].values > 0.5
        fatal_trap = parabolic & (~is_leader)
        
        # 截断后输出 0.0 ~ 1.0 的纯净吸筹烈度
        final_vals = np.where(fatal_trap, 0.0, np.clip(activated, 0.0, 1.0)).astype(np.float32)
        final_score = pd.Series(final_vals, index=df_index, dtype=np.float32)
        
        self._persist_hab_state(raw, df_index, method_name)
        if is_debug and probe_ts in df_index:
            self._print_full_chain_probe(probe_ts, raw, ke, pb, hl, tunnel_prob, penalty, blend, min_dim_score, collapse_penalty, accumulation_raw, activated, fatal_trap, final_score)
        
        return final_score

    def _persist_hab_state(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str):
        if len(df_index) == 0: return
        last_idx, last_ts = -1, df_index[-1]
        hab_snapshot = {"timestamp": last_ts.strftime('%Y-%m-%d'), "updated_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), "metrics": {}}
        for key, series in raw.items():
            if any(p in key for p in ['HAB_STOCK_', 'HAB_SHOCK_']): hab_snapshot["metrics"][key] = float(series.values[last_idx])
        self.latest_hab_state = hab_snapshot
        if hasattr(self.helper, 'update_shared_state'):
            try: self.helper.update_shared_state('HAB_SPLIT_ACCUM', hab_snapshot)
            except Exception: pass

    def _print_full_chain_probe(self, probe_ts: pd.Timestamp, raw: Dict[str, pd.Series], ke: pd.Series, pb: pd.Series, hl: pd.Series, tunnel_prob: np.ndarray, penalty: pd.Series, blend: np.ndarray, min_dim: np.ndarray, collapse: np.ndarray, acc_raw: np.ndarray, activated: np.ndarray, fatal_trap: np.ndarray, final_score: pd.Series):
        idx = raw['close_D'].index.get_loc(probe_ts)
        tc = raw['tick_clustering_index_D'].values[idx]
        stealth = raw['stealth_flow_ratio_D'].values[idx]
        ent = raw['intraday_chip_entropy_D'].values[idx]
        div = raw['price_flow_divergence_D'].values[idx]
        drift = raw['LANGEVIN_DRIFT_ACCUM'].values[idx]
        diff = raw['LANGEVIN_DIFF_ACCUM'].values[idx]
        snr = drift / (diff + 1e-9)
        is_fatal = fatal_trap[idx]
        is_leader = raw['IS_MARKET_LEADER_D'].values[idx] > 0.5
        
        # 修正状态标签显示逻辑
        f_score = final_score.values[idx]
        if f_score > 0.6: status_str = "🚀 强力隐蔽吸筹确认"
        elif f_score > 0.1: status_str = "👀 弱势试探吸筹"
        elif is_fatal: status_str = "核按钮级派发/抛物线" 
        else: status_str = "未达门槛 / 虚假诱多防爆拦截 (0.0000)"

        print(f"\n{'='*30} [全息拆单吸筹微观探针 V8.0.1 @ {probe_ts.strftime('%Y-%m-%d')}] {'='*30}")
        print(f"【底层张量场】")
        print(f"  ├─ 聚类算法碎单(Tick Clustering): {tc:.4f}")
        print(f"  ├─ 隐秘潜行占比(Stealth Ratio): {stealth:.4f}")
        print(f"  ├─ 筹码熵序收敛(Intraday Entropy): {ent:.4f}")
        print(f"  └─ 价流背离防爆(Price-Flow Div): {div:.4f}")
        print(f"【高维物理测度与模型】")
        print(f"  ├─ 霍克斯动能张量(Hawkes Kinetic): {ke.values[idx]:.4f}σ")
        print(f"  ├─ 势垒压力张量(Potential Barrier): {pb.values[idx]:.4f}σ")
        print(f"  ├─ 量子隧穿渗透率(Quantum Tunneling): {tunnel_prob[idx]*100:.1f}%")
        print(f"  └─ 朗之万漂移与扩散(Langevin SNR): Drift={drift:.4f} | Diff={diff:.4f} | SNR={snr:.2f}")
        print(f"【HAB全域存量冲击池】")
        print(f"  ├─ 隐蔽吸筹21日冲击波: {raw['HAB_SHOCK_21_HIDDEN_ACCUM'].values[idx]:.4f}x")
        print(f"  └─ 潜行流21日冲击波: {raw['HAB_SHOCK_21_STEALTH'].values[idx]:.4f}x")
        print(f"【AG-Blend 全息压缩与防爆坍缩网】")
        print(f"  ├─ 原始虚假繁荣锁(Anti-Spoofing): {penalty.values[idx]:.2f}x")
        print(f"  ├─ 短板侦测(Min Dim): {min_dim[idx]:.4f} -> 触发木桶坍缩倍率: {collapse[idx]:.4f}x")
        print(f"  ├─ 算术-几何混合均值(AG-Blend): {blend[idx]:.4f}")
        print(f"  ├─ 单极性绝对截断(Unipolar ReLU): {acc_raw[idx]:.4f}")
        print(f"  └─ 幂律极化激活(Power Law Activation): {activated[idx]:.4f}")
        print(f"【信号最终输出 (Unipolar 0.0 - 1.0)】")
        print(f"  ★ 状态标签: {status_str}")
        print(f"{'-'*85}")
        print(f" >>> PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY 最终绝对吸筹烈度: {f_score:.4f}")
        print(f"{'='*85}\n")