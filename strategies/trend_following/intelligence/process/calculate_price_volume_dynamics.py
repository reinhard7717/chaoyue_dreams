# strategies/trend_following/intelligence/process/calculate_price_volume_dynamics.py
# 价格和成交量的动态变化 已完成
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import jit, float64, int64
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import get_param_value
from strategies.trend_following.intelligence.process.price_volume_modules.calculate_power_transfer_raw_score import CalculatePowerTransferRawScore
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
@jit(nopython=True)
def _numba_fractal_dimension(flows, window=21):
    """V7.0.0 · 分形维数算子"""
    n_series = len(flows)
    n_len = len(flows[0])
    out = np.ones((n_series, n_len), dtype=np.float32) * 1.5
    all_scales = np.array([1, 2, 3, 5, 8, 13, 21], dtype=np.int32)
    for s_idx in range(n_series):
        series = flows[s_idx]
        for i in range(window, n_len):
            if np.std(series[i-window:i]) < 1e-6:
                out[s_idx, i] = 1.5
                continue
            flucts = []
            current_scales = []
            for scale in all_scales:
                if scale >= window: break
                reshaped_len = (window // scale) * scale
                data_slice = series[i-reshaped_len:i]
                seg_flucts = 0.0
                n_segs = 0
                for k in range(0, reshaped_len, scale):
                    seg_std = np.std(data_slice[k:k+scale])
                    seg_flucts += seg_std
                    n_segs += 1
                if n_segs > 0 and seg_flucts > 1e-9:
                    flucts.append(seg_flucts / n_segs)
                    current_scales.append(float(scale))
            if len(flucts) >= 3:
                y = np.log(np.array(flucts, dtype=np.float32))
                x = np.log(np.array(current_scales, dtype=np.float32))
                A = np.vstack((x, np.ones(len(x), dtype=np.float32))).T
                slope, _ = np.linalg.lstsq(A, y, rcond=-1)[0]
                h_val = slope
                if h_val < 0.0: h_val = 0.0
                if h_val > 1.0: h_val = 1.0
                out[s_idx, i] = 2.0 - h_val
    return out
@jit(nopython=True)
def _numba_adaptive_denoise_dynamics(data, vol_adj, confidence, process_noise=0.05):
    """V7.0.0 · 自适应去噪算子"""
    n = len(data)
    est = np.zeros(n)
    p = np.zeros(n)
    first_valid = 0
    for i in range(n):
        if not np.isnan(data[i]) and not np.isnan(vol_adj[i]) and not np.isnan(confidence[i]):
            est[i] = data[i]
            first_valid = i
            break
    p[first_valid] = 1.0
    for i in range(first_valid + 1, n):
        if np.isnan(data[i]) or np.isnan(vol_adj[i]) or np.isnan(confidence[i]):
            est[i] = est[i-1]
            p[i] = p[i-1]
            continue
        meas_noise = (vol_adj[i] * 2.0) / (confidence[i] + 1e-9)
        curr_est = est[i-1]
        curr_p = p[i-1] + process_noise
        k_gain = curr_p / (curr_p + meas_noise)
        est[i] = curr_est + k_gain * (data[i] - curr_est)
        p[i] = (1 - k_gain) * curr_p
    return est
@jit(nopython=True)
def _numba_power_activation(x, alpha=0.01, gain=1.5):
    """V7.0.0 · 非对称动力学激活算子"""
    res = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            res[i] = x[i] * gain
        else:
            res[i] = x[i] * alpha
    return res
@jit(nopython=True)
def _numba_power_law_activation(x, power=1.5, leak=0.1):
    """V7.0.0 · 幂律极化泄露算子"""
    n = len(x)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if x[i] > 0:
            out[i] = x[i] ** power
        else:
            out[i] = - (np.abs(x[i]) ** power) * leak
    return out
@jit(nopython=True)
def _numba_rolling_robust_norm(data, window=21):
    """V7.0.0 · 零点锚定极性归一化算子 (Zero-Centered MAD)"""
    n = len(data)
    out = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        slice_data = data[i - window + 1 : i + 1]
        median = np.median(slice_data)
        mad = np.median(np.abs(slice_data - median)) + 1e-4
        out[i] = np.tanh((data[i] - median) / (mad * 3.0)) * 3.0
    return out
@jit(nopython=True)
def _numba_fast_rolling_dynamics(data, windows):
    """V7.0.0 · Numba 原生多尺度动力学算子"""
    n = len(data)
    num_wins = len(windows)
    means = np.zeros((num_wins, n))
    slopes = np.zeros((num_wins, n))
    for w_idx in range(num_wins):
        w = windows[w_idx]
        for i in range(w, n):
            window_data = data[i-w:i]
            means[w_idx, i] = np.mean(window_data)
            if window_data[0] != 0:
                slopes[w_idx, i] = (window_data[-1] - window_data[0]) / w
    return means, slopes
@jit(nopython=True)
def _numba_hmm_regime_probability(flow_n, vol_n, price_n, vwap_dist_n):
    """V7.0.0 · HMM 体制概率算子"""
    n = len(flow_n)
    markup_probs = np.zeros(n, dtype=np.float32)
    centroid_acc = np.array([1.0, -0.5, -0.2, -0.5], dtype=np.float32)
    centroid_markup = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32)
    centroid_dist = np.array([-1.0, 1.0, 0.2, 0.0], dtype=np.float32)
    for i in range(n):
        current_vec = np.array([flow_n[i], vol_n[i], price_n[i], vwap_dist_n[i]], dtype=np.float32)
        d_acc = np.sum((current_vec - centroid_acc)**2)
        d_mar = np.sum((current_vec - centroid_markup)**2)
        d_dis = np.sum((current_vec - centroid_dist)**2)
        exp_acc = np.exp(-d_acc)
        exp_mar = np.exp(-d_mar)
        exp_dis = np.exp(-d_dis)
        markup_probs[i] = exp_mar / (exp_acc + exp_mar + exp_dis + 1e-9)
    return markup_probs
@jit(nopython=True)
def _numba_robust_dynamics(data, win=5, abs_threshold=1e-4, change_threshold=1e-5):
    """V7.0.0 · 鲁棒动力学算子 (引入 Tanh Threshold Gate 防止零基粉碎)"""
    n = len(data)
    slope = np.zeros(n, dtype=np.float32)
    accel = np.zeros(n, dtype=np.float32)
    jerk = np.zeros(n, dtype=np.float32)
    clean_data = np.copy(data).astype(np.float32)
    for i in range(n):
        if np.abs(clean_data[i]) < abs_threshold:
            clean_data[i] = 0.0
    for i in range(win, n):
        delta = clean_data[i] - clean_data[i-win]
        if np.abs(delta) < change_threshold:
            norm_delta = 0.0
        else:
            norm_delta = np.tanh(np.abs(delta) / (change_threshold * 10.0 + 1e-9)) * delta
        slope[i] = norm_delta / win
    for i in range(win, n):
        delta_s = slope[i] - slope[i-1]
        if np.abs(delta_s) < (change_threshold / 5.0):
            norm_s = 0.0
        else:
            norm_s = np.tanh(np.abs(delta_s) / (change_threshold + 1e-9)) * delta_s
        accel[i] = norm_s
    for i in range(win, n):
        delta_a = accel[i] - accel[i-1]
        if np.abs(delta_a) < (change_threshold / 25.0):
            norm_a = 0.0
        else:
            norm_a = np.tanh(np.abs(delta_a) / (change_threshold / 5.0 + 1e-9)) * delta_a
        jerk[i] = norm_a
    return slope, accel, jerk
@jit(nopython=True)
def _numba_rolling_accumulation(data, window=13):
    """V7.0.0 · 滚动累积算子"""
    n = len(data)
    res = np.zeros(n, dtype=np.float32)
    for i in range(window, n):
        accum = 0.0
        for j in range(i - window, i):
            accum += data[j]
        res[i] = accum
    return res
@jit(nopython=True)
def _numba_rolling_rank(arr, window):
    """V7.0.0 · Numba 滚动排名算子"""
    n = len(arr)
    out = np.empty(n, dtype=np.float32)
    out[:] = np.nan
    for i in range(window - 1, n):
        count = 0.0
        target = arr[i]
        for j in range(i - window + 1, i + 1):
            if arr[j] < target:
                count += 1.0
        out[i] = count / (window - 1)
    return out
@jit(nopython=True)
def _numba_hab_impact(data, windows):
    """V7.0.0 · HAB 历史累积与冲击强度全域感知缓冲器"""
    n = len(data)
    num_wins = len(windows)
    hab_stock = np.zeros((num_wins, n), dtype=np.float32)
    shock_intensity = np.zeros((num_wins, n), dtype=np.float32)
    for w_idx in range(num_wins):
        w = windows[w_idx]
        for i in range(w, n):
            stock = 0.0
            for j in range(i - w, i):
                stock += np.abs(data[j])
            hab_stock[w_idx, i] = stock
            avg_stock = stock / w
            shock_intensity[w_idx, i] = data[i] / (avg_stock + 1e-5)
    return hab_stock, shock_intensity
@jit(nopython=True)
def _numba_langevin_dynamics(price, window=21):
    """V7.0.0 · 朗之万动力学漂移与扩散算子"""
    n = len(price)
    drift = np.zeros(n, dtype=np.float32)
    diffusion = np.zeros(n, dtype=np.float32)
    for i in range(window, n):
        diffs = np.empty(window, dtype=np.float32)
        for j in range(window):
            diffs[j] = price[i - window + j + 1] - price[i - window + j]
        drift[i] = np.mean(diffs)
        diffusion[i] = np.std(diffs) + 1e-5
    return drift, diffusion
@jit(nopython=True)
def _numba_lotka_volterra_phase(predator, prey, window=21):
    """V7.0.0 · Lotka-Volterra 捕食者-猎物相空间角动量算子"""
    n = len(predator)
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        x, y = prey[i-1] + 2.0, predator[i-1] + 2.0
        dx, dy = prey[i] - prey[i-1], predator[i] - predator[i-1]
        out[i] = x * dy - y * dx
    return out
@jit(nopython=True)
def _numba_quantum_harmonic_oscillator(price, vwap, price_slope, window=21):
    """V7.0.0 · 量子谐振子引力场"""
    n = len(price)
    energy = np.zeros(n, dtype=np.float32)
    for i in range(n):
        bias = (price[i] - vwap[i]) / (vwap[i] + 1e-9)
        potential = 0.5 * (bias ** 2) * np.sign(bias)
        kinetic = 0.5 * (price_slope[i] ** 2) * np.sign(price_slope[i])
        energy[i] = potential + kinetic
    return energy
@jit(nopython=True)
def _numba_navier_stokes_viscosity(pressure_trapped, cost_volatility, flow_persistence, window=21):
    """V7.0.0 · Navier-Stokes 订单流运动学粘滞度算子"""
    n = len(pressure_trapped)
    viscosity = np.zeros(n, dtype=np.float32)
    for i in range(n):
        density = pressure_trapped[i]
        turbulence = cost_volatility[i]
        channel_width = flow_persistence[i] + 1e-5
        viscosity[i] = (density * turbulence) / channel_width
    return viscosity
class CalculatePriceVolumeDynamics:
    """PROCESS_META_POWER_TRANSFER 计算价量动态与量子引力场的全息中枢系统"""
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.process_params = self.helper.params
        self.std_window = self.helper.std_window
    def _setup_debug_info(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp], Dict]:
        is_debug_enabled_for_method = get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        _temp_debug_values = {}
        if is_debug_enabled_for_method and probe_ts:
            _temp_debug_values[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            _temp_debug_values[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价量动态..."] = ""
        return is_debug_enabled_for_method, probe_ts, _temp_debug_values
    def _print_pvd_debug_output(self, debug_values: Dict, probe_ts: pd.Timestamp, method_name: str, final_message: str):
        pass
    def _get_pvd_params(self, config: Dict) -> Dict:
        return get_param_value(config.get('price_volume_dynamics_params'), {})
    def _validate_all_required_signals(self, df: pd.DataFrame, pvd_params: Dict, mtf_slope_accel_weights: Dict, method_name: str, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> bool:
        """V7.0.0 · 核心防爆验证：融合微观与流体力学高价值新维度拦截器"""
        base_required = ['close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'net_amount_rate_D', 'winner_rate_D', 'up_limit_D', 'down_limit_D', 'closing_flow_ratio_D', 'T1_PREMIUM_EXPECTATION_D', 'pressure_release_index_D', 'BBW_21_2.0_D', 'VPA_EFFICIENCY_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'turnover_rate_f_D', 'STATE_ROUNDING_BOTTOM_D', 'STATE_GOLDEN_PIT_D', 'STATE_TRENDING_STAGE_D', 'price_percentile_position_D', 'TURNOVER_STABILITY_INDEX_D', 'STATE_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 'THEME_HOTNESS_SCORE_D', 'chip_entropy_D', 'cost_50pct_D', 'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'trade_count_D', 'market_sentiment_score_D', 'industry_strength_rank_D', 'industry_rank_accel_D', 'net_mf_amount_D', 'pressure_trapped_D', 'stealth_flow_ratio_D', 'ADX_14_D', 'MACDh_13_34_8_D', 'tick_large_order_net_D', 'intraday_main_force_activity_D', 'CMF_21_D', 'game_intensity_D', 'hidden_accumulation_intensity_D', 'IS_MARKET_LEADER_D', 'STATE_PARABOLIC_WARNING_D', 'VPA_MF_ADJUSTED_EFF_D', 'MA_COHERENCE_RESONANCE_D', 'intraday_distribution_confidence_D', 'price_flow_divergence_D', 'profit_pressure_D', 'tick_clustering_index_D', 'PRICE_ENTROPY_D', 'intraday_cost_center_volatility_D', 'flow_persistence_minutes_D', 'tick_abnormal_volume_ratio_D', 'uptrend_strength_D', 'circ_mv_D']
        return self.helper._validate_required_signals(df, base_required, method_name)
    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Dict[str, pd.Series]:
        """V7.0.0 · 原料加载层：贯通高频结构特征与宏观筹码势垒，扩容全量 HAB 冲击池"""
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        raw_signals = {}
        base_cols = ['close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'pct_change_D', 'net_amount_rate_D', 'trade_count_D', 'turnover_rate_f_D', 'circ_mv_D', 'up_limit_D', 'down_limit_D']
        struct_cols = ['winner_rate_D', 'chip_concentration_ratio_D', 'chip_entropy_D', 'cost_50pct_D', 'absorption_energy_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'price_percentile_position_D', 'pressure_trapped_D', 'PRICE_ENTROPY_D', 'intraday_cost_center_volatility_D', 'chip_stability_D']
        tech_cols = ['SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D', 'pressure_release_index_D', 'closing_flow_ratio_D', 'TURNOVER_STABILITY_INDEX_D', 'STATE_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 'industry_strength_rank_D', 'industry_rank_accel_D', 'STATE_ROUNDING_BOTTOM_D', 'STATE_GOLDEN_PIT_D', 'STATE_TRENDING_STAGE_D', 'THEME_HOTNESS_SCORE_D', 'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'market_sentiment_score_D', 'ADX_14_D', 'MACDh_13_34_8_D', 'CMF_21_D', 'hidden_accumulation_intensity_D', 'game_intensity_D', 'tick_large_order_net_D', 'intraday_main_force_activity_D', 'net_mf_amount_D', 'stealth_flow_ratio_D', 'IS_MARKET_LEADER_D', 'STATE_PARABOLIC_WARNING_D', 'VPA_MF_ADJUSTED_EFF_D', 'MA_COHERENCE_RESONANCE_D', 'intraday_distribution_confidence_D', 'price_flow_divergence_D', 'profit_pressure_D', 'tick_clustering_index_D', 'flow_persistence_minutes_D', 'tick_abnormal_volume_ratio_D', 'uptrend_strength_D', 'downtrend_strength_D', 'net_energy_flow_D']
        for col in base_cols + struct_cols + tech_cols:
            if col in df.columns:
                raw_signals[col] = df[col].ffill().fillna(0.0).astype(np.float32)
            else:
                raw_signals[col] = pd.Series(0.0, index=df.index, dtype=np.float32)
        sm_v = raw_signals['SMART_MONEY_HM_NET_BUY_D'].values
        if np.sum(np.abs(sm_v)) < 1e-5: 
            raw_signals['SMART_MONEY_HM_NET_BUY_D'] = (raw_signals['buy_elg_amount_D'] - raw_signals['sell_elg_amount_D'] + raw_signals['buy_lg_amount_D'] - raw_signals['sell_lg_amount_D'])
        if raw_signals['pct_change_D'].abs().max() > 0.5: raw_signals['pct_change_D'] *= np.float32(0.01)
        if raw_signals['turnover_rate_f_D'].max() > 1.5: raw_signals['turnover_rate_f_D'] *= np.float32(0.01)
        if raw_signals['winner_rate_D'].max() > 1.1: raw_signals['winner_rate_D'] *= np.float32(0.01)
        raw_signals['circ_mv_D'] = raw_signals['circ_mv_D'] * np.float32(10000.0)
        valid_mv = raw_signals.get('circ_mv_D', pd.Series([0], dtype=np.float32))
        valid_mv = valid_mv[valid_mv > 1e7]
        avg_mv = valid_mv.median() if len(valid_mv) > 0 else 0.0
        w_s, w_l = 13, 21
        if avg_mv > 0:
            if avg_mv < 50e8: w_s, w_l = 8, 13
            elif avg_mv > 500e8: w_s, w_l = 21, 34
        raw_signals['META_HAB_WINDOWS'] = pd.Series([w_s, w_l], index=df.index[:2], dtype=np.float32)
        raw_vwap = (raw_signals['amount_D'] / (raw_signals['volume_D'] + 1e-9)).fillna(raw_signals['close_D'])
        r_c, r_v = raw_signals['close_D'].values[-60:], raw_vwap.values[-60:]
        v_m, s_f = r_v > 0, np.float32(1.0)
        if np.any(v_m):
            m_r = np.median(r_v[v_m] / r_c[v_m])
            if m_r < 0.5 or m_r > 2.0: s_f = np.float32(10.0 ** (-np.round(np.log10(m_r))))
        raw_signals['VWAP_D'] = (raw_vwap * s_f).astype(np.float32)
        hab_windows = np.array([13, 21, 34, 55], dtype=np.int32)
        hab_targets = ['SMART_MONEY_HM_NET_BUY_D', 'volume_D', 'tick_large_order_net_D', 'profit_pressure_D', 'flow_persistence_minutes_D', 'net_energy_flow_D']
        prefix_map = ['SMART_MONEY', 'VOLUME', 'TICK_NET', 'PROFIT_PRES', 'FLOW_PERSIST', 'ENERGY']
        for target, prefix in zip(hab_targets, prefix_map):
            stock, shock = _numba_hab_impact(raw_signals[target].values.astype(np.float32), hab_windows)
            for i, w in enumerate([13, 21, 34, 55]):
                raw_signals[f'HAB_STOCK_{w}_{prefix}'] = pd.Series(stock[i], index=df.index, dtype=np.float32)
                raw_signals[f'HAB_SHOCK_{w}_{prefix}'] = pd.Series(shock[i], index=df.index, dtype=np.float32)
        raw_signals['ACCUM_13_SMART_MONEY'] = raw_signals['HAB_STOCK_13_SMART_MONEY']
        raw_signals['ACCUM_21_SMART_MONEY'] = raw_signals['HAB_STOCK_21_SMART_MONEY']
        raw_signals['ACCUM_21_VOLUME'] = raw_signals['HAB_STOCK_21_VOLUME']
        raw_signals['ACCUM_13_CLOSING_FLOW'] = pd.Series(_numba_rolling_accumulation(raw_signals['closing_flow_ratio_D'].values, w_s), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_THEME_HOTNESS'] = pd.Series(_numba_rolling_accumulation(raw_signals['THEME_HOTNESS_SCORE_D'].values, w_l), index=df.index, dtype=np.float32)
        raw_signals['MEAN_13_STABILITY'] = pd.Series(_numba_rolling_accumulation(raw_signals['TURNOVER_STABILITY_INDEX_D'].values, w_s), index=df.index, dtype=np.float32) / float(w_s)
        raw_signals['HIST_VOL_SQUEEZE'] = pd.Series(_numba_rolling_accumulation(np.clip(1.0 - raw_signals['BBW_21_2.0_D'].values, 0.0, 1.0).astype(np.float32), w_l), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_TR'] = pd.Series(_numba_rolling_accumulation((raw_signals['high_D'] - raw_signals['low_D']).values, w_l), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_13_SENTIMENT'] = pd.Series(_numba_rolling_accumulation(raw_signals['market_sentiment_score_D'].values, w_s), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_13_ELASTICITY'] = pd.Series(_numba_rolling_accumulation((np.abs(raw_signals['pct_change_D'].values) / (np.abs(pd.Series(raw_signals['volume_D'].values, index=df.index).pct_change().fillna(0).values) + 0.1)).astype(np.float32), w_s), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_ENTROPY_STABILITY'] = pd.Series(_numba_rolling_accumulation(np.where(pd.Series(raw_signals['chip_entropy_D'].values, index=df.index).diff().fillna(0).values < 0, 1.0, 0.0).astype(np.float32), w_l), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_POS_SLOPE'] = pd.Series(_numba_rolling_accumulation(np.where(raw_signals['close_D'].diff(5).fillna(0) > 0, 1.0, 0.0).astype(np.float32), w_l), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_ABOVE_VWAP'] = pd.Series(_numba_rolling_accumulation(np.where(raw_signals['close_D'].values > raw_signals['VWAP_D'].values, 1.0, 0.0).astype(np.float32), w_l), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_13_HIGH_WINNER'] = pd.Series(_numba_rolling_accumulation(np.where(raw_signals['winner_rate_D'].values > 0.8, 1.0, 0.0).astype(np.float32), w_s), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_5_LIQUIDITY_LOCK'] = pd.Series(_numba_rolling_accumulation(np.where((raw_signals['turnover_rate_f_D'].values < 0.03) & (raw_signals['pct_change_D'].values > 0), 1.0, 0.0).astype(np.float32), 5), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_HIGH_RANK'] = pd.Series(_numba_rolling_accumulation(np.where(raw_signals['industry_strength_rank_D'].values <= 20, 1.0, 0.0).astype(np.float32), w_l), index=df.index, dtype=np.float32)
        turn = raw_signals['turnover_rate_f_D'].values.astype(np.float32)
        mf_amount = raw_signals['net_mf_amount_D'].values.astype(np.float32)
        tick_net = raw_signals['tick_large_order_net_D'].values.astype(np.float32)
        tick_norm = _numba_rolling_robust_norm(tick_net, 21)
        sm_shock = raw_signals['HAB_SHOCK_13_SMART_MONEY'].values.astype(np.float32)
        pct = raw_signals['pct_change_D'].values.astype(np.float32)
        sm_hm = raw_signals['SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float32)
        dist_conf = raw_signals['intraday_distribution_confidence_D'].values.astype(np.float32)
        close = raw_signals['close_D'].values.astype(np.float32)
        up_limit = raw_signals['up_limit_D'].values.astype(np.float32)
        flow_collapse = (sm_shock < -0.3) & ((tick_norm < -1.0) | (mf_amount < 0) | (sm_hm < 0) | (dist_conf > 60.0))
        stagnant_dist = (turn > 0.15) & (pct >= -0.04) & (pct <= 0.02) & flow_collapse
        price_crash = (pct < -0.04) & flow_collapse
        rally_dist = (turn > 0.10) & (pct > 0.02) & ((tick_norm < -1.0) | (dist_conf > 70.0)) & (mf_amount < 0) & (sm_shock < -0.3)
        raw_signals['FLAG_STAGNANT_DISTRIBUTION'] = pd.Series(np.where(stagnant_dist, 1.0, 0.0).astype(np.float32), index=df.index)
        raw_signals['FLAG_PRICE_CRASH'] = pd.Series(np.where(price_crash, 1.0, 0.0).astype(np.float32), index=df.index)
        raw_signals['FLAG_RALLY_DISTRIBUTION'] = pd.Series(np.where(rally_dist, 1.0, 0.0).astype(np.float32), index=df.index)
        raw_signals['FLAG_FATAL_TRAP'] = pd.Series(np.where(stagnant_dist | price_crash | rally_dist, 1.0, 0.0).astype(np.float32), index=df.index)
        is_limit_up = close >= up_limit * 0.995
        ignition = (pct > 0.05) & (mf_amount > 0) & (sm_shock > 0.5)
        raw_signals['FLAG_LIMIT_UP'] = pd.Series(np.where(is_limit_up, 1.0, 0.0).astype(np.float32), index=df.index)
        raw_signals['FLAG_IGNITION'] = pd.Series(np.where(ignition, 1.0, 0.0).astype(np.float32), index=df.index)
        threshold_map = {'net_amount_rate_D': (0.01, 0.005), 'winner_rate_D': (0.01, 0.005), 'SMART_MONEY_HM_NET_BUY_D': (10, 10), 'VPA_EFFICIENCY_D': (0.01, 0.01), 'BBW_21_2.0_D': (0.001, 0.0001), 'THEME_HOTNESS_SCORE_D': (0.1, 0.1), 'chip_entropy_D': (0.01, 0.001), 'volume_D': (100, 10), 'pct_change_D': (0.0001, 0.0001), 'close_D': (0.1, 0.01), 'market_sentiment_score_D': (0.1, 0.1), 'industry_strength_rank_D': (0.001, 0.001), 'turnover_rate_f_D': (0.01, 0.01), 'trade_count_D': (10, 5), 'VWAP_D': (0.1, 0.01), 'closing_flow_ratio_D': (0.01, 0.01), 'net_mf_amount_D': (10000, 5000), 'pressure_trapped_D': (0.01, 0.005), 'stealth_flow_ratio_D': (0.01, 0.005), 'tick_large_order_net_D': (10000, 5000), 'intraday_main_force_activity_D': (1.0, 0.5), 'ADX_14_D': (0.5, 0.5), 'MACDh_13_34_8_D': (0.01, 0.005), 'VPA_MF_ADJUSTED_EFF_D': (0.01, 0.01), 'MA_COHERENCE_RESONANCE_D': (0.01, 0.01), 'price_flow_divergence_D': (0.01, 0.01), 'intraday_distribution_confidence_D': (1.0, 1.0), 'tick_clustering_index_D': (0.01, 0.005), 'intraday_cost_center_volatility_D': (0.01, 0.005), 'flow_persistence_minutes_D': (1.0, 0.5), 'PRICE_ENTROPY_D': (0.01, 0.005), 'chip_stability_D': (0.01, 0.005), 'net_energy_flow_D': (1000, 500), 'profit_pressure_D': (0.01, 0.01), 'uptrend_strength_D': (0.01, 0.01), 'downtrend_strength_D': (0.01, 0.01)}
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
    def _calculate_power_transfer_raw_score(self, df_index: pd.Index, raw: Dict[str, pd.Series], method_name: str) -> pd.Series:
        vol_adj = raw['BBW_21_2.0_D'].values.astype(np.float32)
        tc_values = raw['trade_count_D'].values.astype(np.float32)
        tc_means, _ = _numba_fast_rolling_dynamics(tc_values, np.array([21], dtype=np.int64))
        roll_tc = np.where(np.abs(tc_means[0]) < 1e-9, 1.0, tc_means[0]).astype(np.float32)
        conf = (tc_values / roll_tc).astype(np.float32)
        s_c = _numba_adaptive_denoise_dynamics(raw['SLOPE_5_net_mf_amount_D'].values.astype(np.float32), vol_adj, conf)
        a_c = _numba_adaptive_denoise_dynamics(raw['ACCEL_5_net_mf_amount_D'].values.astype(np.float32), vol_adj, conf)
        j_c = _numba_adaptive_denoise_dynamics(raw['JERK_3_net_mf_amount_D'].values.astype(np.float32), vol_adj, conf)
        mf_val = raw['net_mf_amount_D'].values.astype(np.float32)
        mf_norm = _numba_rolling_robust_norm(mf_val, 21)
        raw_impulse = (mf_norm * 0.40 + s_c * 0.30 + a_c * 0.20 + j_c * 0.10).astype(np.float32)
        norm_imp = _numba_rolling_robust_norm(raw_impulse, 21)
        norm_imp = np.where((mf_val < 0) & (norm_imp > 0), -norm_imp, norm_imp)
        comp_factor = np.where(raw['price_percentile_position_D'].values.astype(np.float32) < 0.2, 1.2, np.where(raw['price_percentile_position_D'].values.astype(np.float32) > 0.8, 0.8, 1.0)).astype(np.float32)
        comp_imp = norm_imp * comp_factor
        _, f_slopes = _numba_fast_rolling_dynamics(raw['net_mf_amount_D'].values.astype(np.float32), np.array([3, 5, 8, 13, 21], dtype=np.int64))
        mcv = np.dot(np.array([0.35, 0.25, 0.20, 0.10, 0.10], dtype=np.float32), f_slopes.astype(np.float32))
        mcv_norm = _numba_rolling_robust_norm(mcv, 21)
        mcv_norm = np.where((mf_val < 0) & (mcv_norm > 0), -mcv_norm, mcv_norm)
        mcv_weight = np.where((comp_imp > 0) & (mcv_norm < 0), 0.0, 0.35).astype(np.float32)
        imp_weight = 0.60 + (0.35 - mcv_weight)
        mass_m = np.clip(np.log1p(np.abs(raw['HAB_STOCK_21_SMART_MONEY'].values.astype(np.float32))) / 10.0, 0.8, 1.3).astype(np.float32)
        mass_v = np.clip(np.log1p(raw['HAB_STOCK_21_VOLUME'].values.astype(np.float32)) / 7.0, 0.8, 1.3).astype(np.float32)
        final_mass = np.where(mass_m <= 0.81, mass_v, mass_m).astype(np.float32)
        shock_sm = raw['HAB_SHOCK_21_SMART_MONEY'].values.astype(np.float32)
        vol_shock = raw['HAB_SHOCK_21_VOLUME'].values.astype(np.float32)
        shock_mult = np.where(shock_sm < -0.2, np.clip(1.0 + np.abs(shock_sm) + np.abs(vol_shock) * 0.2, 1.0, 5.0), np.where(shock_sm > 0.5, np.clip(1.0 + shock_sm * 0.8 + vol_shock * 0.2, 1.0, 5.0), np.clip(1.0 + np.tanh(shock_sm) * 0.3 + np.tanh(vol_shock) * 0.1, 0.5, 2.0))).astype(np.float32)
        base_score = comp_imp * 2.0 * imp_weight + mcv_norm * mcv_weight
        return pd.Series((base_score * final_mass * shock_mult).astype(np.float32), index=df_index, dtype=np.float32).clip(-5.0, 5.0)
    def _calculate_vwap_propulsion_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        vwap = raw['VWAP_D'].values.astype(np.float32)
        close = raw['close_D'].values.astype(np.float32)
        pct_change = raw['pct_change_D'].values.astype(np.float32)
        norm_propulsion = np.clip(_numba_rolling_robust_norm(pct_change, 21) * 3.0, -3.0, 3.0)
        bias = (close - vwap) / (vwap + 1e-9)
        instant_polarity = np.where((bias < -0.015) & (pct_change < -0.04), -1.0, np.where((bias > 0.015) & (pct_change > 0.04), 1.0, np.sign(norm_propulsion)))
        propulsion_score = (np.tanh(np.abs(norm_propulsion) * 0.8) * 1.5 * instant_polarity).astype(np.float32)
        bias_penalty = np.where(np.abs(bias) > 0.08, 0.8, 1.0).astype(np.float32)
        kinematic_boost = np.where(raw['ACCEL_5_close_D'].values.astype(np.float32) * instant_polarity > 0, 1.3, 1.0).astype(np.float32)
        days_above = raw['ACCUM_21_ABOVE_VWAP'].values.astype(np.float32)
        thickness_bonus = np.where(propulsion_score > 0, np.clip(days_above / 10.0, 0.8, 1.4), np.clip((21.0 - days_above) / 10.0, 0.8, 1.4)).astype(np.float32)
        qho_energy = _numba_quantum_harmonic_oscillator(close, vwap, pct_change, 21)
        qho_norm = _numba_rolling_robust_norm(qho_energy, 21)
        qho_resonance = np.clip(1.0 + np.abs(qho_norm) * 0.5 * instant_polarity, 0.1, 2.0).astype(np.float32)
        return pd.Series((propulsion_score * bias_penalty * kinematic_boost * thickness_bonus * qho_resonance).astype(np.float32), index=df_index, dtype=np.float32).clip(-5.0, 5.0)
    def _calculate_trend_inertia_momentum(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        ma5 = raw['close_D'].rolling(5).mean().fillna(0).values.astype(np.float32)
        ma21 = raw['close_D'].rolling(21).mean().fillna(0).values.astype(np.float32)
        ma55 = raw['close_D'].rolling(55).mean().fillna(0).values.astype(np.float32)
        alignment = np.where((ma5 > ma21) & (ma21 > ma55), 1.0, 0.8).astype(np.float32)
        norm_s = _numba_rolling_robust_norm(raw['SLOPE_5_close_D'].values.astype(np.float32), 21) * 3.0
        norm_a = _numba_rolling_robust_norm(raw['ACCEL_5_close_D'].values.astype(np.float32), 21) * 3.0
        drift, diffusion = _numba_langevin_dynamics(raw['close_D'].values.astype(np.float32), 21)
        snr = _numba_rolling_robust_norm(drift / (diffusion + 1e-9), 21)
        kinematic_score = np.ones(len(df_index), dtype=np.float32) + np.where((norm_s > 1.0) & (norm_a > 0.5), 0.3, 0.0).astype(np.float32) + np.where((norm_a > 1.5) & (raw['JERK_3_close_D'].values.astype(np.float32) > 0), 0.4, 0.0).astype(np.float32) - np.where((norm_s > 0) & (norm_a < -1.0), 0.3, 0.0).astype(np.float32) + np.clip(snr * 0.2, -0.3, 0.5)
        adx_bonus = np.where(raw['ADX_14_D'].values.astype(np.float32) > 25, 1.2, np.where(raw['ADX_14_D'].values.astype(np.float32) < 15, 0.8, 1.0)).astype(np.float32)
        macd_z = _numba_rolling_robust_norm(raw['MACDh_13_34_8_D'].values.astype(np.float32), 21)
        macd_bonus = np.clip(1.0 + macd_z * 0.3, 0.8, 1.3).astype(np.float32)
        pos_days = raw['ACCUM_21_POS_SLOPE'].values.astype(np.float32)
        consistency_bonus = np.clip(pos_days / 15.0, 0.8, 1.25).astype(np.float32)
        r2_vals = raw['GEOM_REG_R2_D'].values.astype(np.float32)
        divergence_penalty = np.clip(1.0 - raw['price_flow_divergence_D'].values.astype(np.float32) * 0.5, 0.5, 1.0).astype(np.float32)
        final_inertia = alignment * np.maximum(kinematic_score, 0.1) * np.maximum(consistency_bonus, 0.1) * np.maximum((0.6 + np.clip(r2_vals, 0.0, 1.0) * 0.4), 0.1) * np.maximum(adx_bonus, 0.1) * np.maximum(macd_bonus, 0.1) * divergence_penalty
        return pd.Series(final_inertia, index=df_index, dtype=np.float32).clip(0.1, 2.5)
    def _calculate_market_permeability_index(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        sent_rank = _numba_rolling_rank(raw['market_sentiment_score_D'].values, 21)
        accel = raw['ACCEL_5_market_sentiment_score_D'].values
        is_ignition = raw['FLAG_IGNITION'].values > 0.5
        is_leader = raw['IS_MARKET_LEADER_D'].values.astype(np.float32) > 0.5
        perm = np.where((sent_rank < 0.2) & (accel > 0), 1.3, np.where((sent_rank > 0.8) & (accel > 0) & (is_ignition | is_leader), 1.4, np.where((sent_rank > 0.8) & (accel > 0), 0.7, 1.0))).astype(np.float32)
        acc_sent = raw['ACCUM_13_SENTIMENT'].values
        saturation_decay = np.where((acc_sent > pd.Series(acc_sent, index=df_index).rolling(21).median().fillna(0).values * 1.5) & (~is_ignition) & (~is_leader), 0.8, 1.0).astype(np.float32)
        bonus = np.where(raw['JERK_3_industry_strength_rank_D'].values < -2.0, 1.3, np.where(raw['industry_strength_rank_D'].values < 0.1, 1.1, 0.9)).astype(np.float32)
        game_z = _numba_rolling_robust_norm(raw['game_intensity_D'].values.astype(np.float32), 21) * 3.0
        game_boost = np.clip(1.0 + game_z * 0.2, 0.8, 1.2).astype(np.float32)
        leader_boost = np.where(is_leader, 1.5, 1.0).astype(np.float32)
        final_perm = perm * np.maximum(bonus, 0.1) * np.maximum(saturation_decay, 0.1) * np.maximum(game_boost, 0.1) * leader_boost
        return pd.Series(final_perm, index=df_index, dtype=np.float32).clip(0.1, 2.5)
    def _calculate_entropic_ordering_bonus(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        s5 = raw['SLOPE_5_chip_entropy_D'].values.astype(np.float32)
        a5 = raw['ACCEL_5_chip_entropy_D'].values.astype(np.float32)
        j3 = raw['JERK_3_chip_entropy_D'].values.astype(np.float32)
        pct = raw['pct_change_D'].values.astype(np.float32)
        locking_force = -(s5 * 0.7 + a5 * 0.3)
        force_s = pd.Series(locking_force, index=df_index).values.astype(np.float32)
        force_norm = _numba_rolling_robust_norm(force_s, 21) * 3.0
        base_bonus = np.clip(np.tanh(force_norm - 1.0) + 0.5, 0.0, 1.5).astype(np.float32)
        jerk_bonus = np.where((j3 < -0.01) & (force_norm > 1.5), 1.3, 1.0).astype(np.float32)
        ent_stab = raw['ACCUM_21_ENTROPY_STABILITY'].values.astype(np.float32)
        stab_ratio = ent_stab / 21.0
        stability_mult = np.clip(stab_ratio * 2.0, 0.8, 1.4).astype(np.float32)
        penalty = np.where((pct > 0) & (s5 > 0), 0.7, 1.0).astype(np.float32)
        p_ent_norm = _numba_rolling_robust_norm(raw['PRICE_ENTROPY_D'].values.astype(np.float32), 21)
        p_ent_bonus = np.clip(1.0 - p_ent_norm * 0.15, 0.8, 1.2).astype(np.float32)
        final_factor = pd.Series((1.0 + base_bonus) * jerk_bonus * stability_mult * penalty * p_ent_bonus, index=df_index, dtype=np.float32).clip(0.1, 2.5)
        return final_factor
    def _calculate_fractal_efficiency_resonance(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        f_all = _numba_fractal_dimension(np.vstack((raw['close_D'].values.astype(np.float32), raw['volume_D'].values.astype(np.float32))), window=21)
        h_p, h_v = (2.0 - f_all[0]).astype(np.float32), (2.0 - f_all[1]).astype(np.float32)
        gap_z = _numba_rolling_robust_norm(np.abs(h_p - h_v), 21) * 3.0
        res_score = np.clip(1.2 - np.abs(gap_z) * 0.2, 0.5, 1.3).astype(np.float32)
        is_super_trend = (h_p > 0.8) & (raw['ADX_14_D'].values.astype(np.float32) > 25.0)
        res_score = np.where(is_super_trend, 1.2, res_score).astype(np.float32)
        struct_stab = _numba_rolling_accumulation(np.where(h_p > 0.55, 1.0, 0.0).astype(np.float32), int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[1]))
        adx_gate = np.clip(raw['ADX_14_D'].values.astype(np.float32) / 20.0, 0.5, 1.5).astype(np.float32)
        ma_coherence = raw['MA_COHERENCE_RESONANCE_D'].values.astype(np.float32)
        ma_boost = np.clip(1.0 + _numba_rolling_robust_norm(ma_coherence, 21) * 0.3, 0.7, 1.3).astype(np.float32)
        final = res_score * np.clip(struct_stab / 15.0, 0.9, 1.3).astype(np.float32) * adx_gate * ma_boost
        final_series = pd.Series(np.where((h_p > 0.55) & (pd.Series(h_p, index=df_index).diff(5).fillna(0).values.astype(np.float32) > 0), final * 1.2, final), index=df_index, dtype=np.float32)
        return pd.Series(final_series.values.astype(np.float32), index=df_index, dtype=np.float32).clip(0.1, 2.5)
    def _calculate_hmm_regime_confirmation(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        f_n = _numba_rolling_robust_norm(raw['SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float32), 21) * 3.0
        v_n = _numba_rolling_robust_norm(raw['volume_D'].values.astype(np.float32), 21) * 3.0
        p_n = _numba_rolling_robust_norm(raw['pct_change_D'].values.astype(np.float32), 21) * 3.0
        v_d = _numba_rolling_robust_norm(((raw['close_D'].values.astype(np.float32) - raw['VWAP_D'].values.astype(np.float32)) / (raw['VWAP_D'].values.astype(np.float32) + 1e-9)), 21) * 3.0
        m_p_s = pd.Series(_numba_hmm_regime_probability(f_n, v_n, p_n, v_d), index=df_index, dtype=np.float32)
        regime_bias = np.where(raw['ACCUM_21_SMART_MONEY'].values.astype(np.float32) > 0, 1.15, 0.9).astype(np.float32)
        realization_ratio = np.clip(raw['ACCUM_21_POS_SLOPE'].values.astype(np.float32) / (float(int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[1])) * 0.6), 0.7, 1.2).astype(np.float32)
        p_a = m_p_s.diff(3).diff(3).fillna(0).values.astype(np.float32)
        p_j = m_p_s.diff(1).diff(1).diff(1).fillna(0).values.astype(np.float32)
        b_f = np.where(m_p_s.values > 0.5, 1.0 + (m_p_s.values - 0.5), 0.8 + m_p_s.values * 0.4).astype(np.float32)
        d_b = 1.0 + np.where((m_p_s.values > 0.6) & (p_a > 0), 0.2, 0.0) + np.where((m_p_s.values > 0.4) & (p_j > 0.1), 0.3, 0.0)
        return pd.Series(b_f * d_b * regime_bias * realization_ratio, index=df_index, dtype=np.float32)
    def _calculate_chip_lock_efficiency(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        high_win_days = raw['ACCUM_13_HIGH_WINNER'].values.astype(np.float32)
        c_50 = np.where(raw['cost_50pct_D'].values.astype(np.float32) == 0, raw['close_D'].values.astype(np.float32), raw['cost_50pct_D'].values.astype(np.float32))
        eff_win = np.maximum(raw['winner_rate_D'].values.astype(np.float32), np.maximum(np.where((raw['close_D'].values.astype(np.float32) > c_50), 0.6, 0.0).astype(np.float32), np.clip(high_win_days / 13.0, 0.0, 1.0) * 0.8))
        turn = raw['turnover_rate_f_D'].values.astype(np.float32)
        rel_turn = turn / (pd.Series(turn, index=df_index).rolling(21).median().replace(0, 0.01).values.astype(np.float32) + 1e-9)
        final_decay = np.clip(np.where((raw['close_D'].values.astype(np.float32) > raw['close_D'].rolling(21).mean().values.astype(np.float32)), 0.6, 1.0).astype(np.float32) - np.where(raw['absorption_energy_D'].values.astype(np.float32) > 0, 0.3, 0.0).astype(np.float32), 0.3, 1.2).astype(np.float32)
        lock_factor = np.clip(2.0 / (1.0 + np.exp((rel_turn - 1.0) * 2.0 * final_decay)), 0.3, 1.5)
        k_bonus = 1.0 + np.where((raw['ACCEL_5_winner_rate_D'].values.astype(np.float32) > 0) & (raw['ACCEL_5_turnover_rate_f_D'].values.astype(np.float32) < 0), 0.3, 0.0).astype(np.float32) + np.where(raw['JERK_3_winner_rate_D'].values.astype(np.float32) > 0.1, 0.2, 0.0).astype(np.float32)
        deep_lock_mult = np.where(high_win_days > 10, 1.5, np.where(high_win_days > 5, 1.2, 1.0)).astype(np.float32)
        chip_stab_z = _numba_rolling_robust_norm(raw['chip_stability_D'].values.astype(np.float32), 21)
        stab_bonus = np.clip(1.0 + chip_stab_z * 0.2, 0.8, 1.2).astype(np.float32)
        final_chip = eff_win * lock_factor * k_bonus * deep_lock_mult * stab_bonus * (0.8 + (raw['close_D'].values.astype(np.float32) > c_50).astype(np.float32) * 0.2)
        return pd.Series(final_chip, index=df_index, dtype=np.float32).clip(0.1, 2.5)
    def _calculate_microstructure_attack_vector(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        e_n = (raw['buy_elg_amount_D'].values - raw['sell_elg_amount_D'].values).astype(np.float32)
        l_n = (raw['buy_lg_amount_D'].values - raw['sell_lg_amount_D'].values).astype(np.float32)
        t_a = (raw['amount_D'].values + 1e-9).astype(np.float32)
        z_score_flow = _numba_rolling_robust_norm((e_n + l_n) / t_a, 21) * 3.0
        z_score_jerk = _numba_rolling_robust_norm(raw['JERK_3_close_D'].values.astype(np.float32), 21) * 3.0
        base_score = 0.5 + (1.0 / (1.0 + np.exp(-z_score_flow)))
        close = raw['close_D'].values.astype(np.float32)
        is_stealth = (((e_n + l_n) / t_a) < 0) & (close > pd.Series(close, index=df_index).rolling(5).mean().values) & (z_score_jerk > 0)
        is_benign_dip = (z_score_flow > -1.0) & (close > raw['close_D'].rolling(21).mean().values) & (z_score_jerk > -2.0)
        adjusted_base = np.where(is_stealth | is_benign_dip, np.maximum(base_score, 1.0), base_score).astype(np.float32)
        sync_score = np.where(((e_n / t_a) > 0) & ((l_n / t_a) > 0), 1.1, np.where(((e_n / t_a) * (l_n / t_a)) < 0, 0.9, 1.0)).astype(np.float32)
        jerk_bonus = np.where(raw['JERK_3_SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float32) > 0.5, 1.2, 1.0).astype(np.float32)
        mf_norm = _numba_rolling_robust_norm(raw['net_mf_amount_D'].values.astype(np.float32), 21)
        pt_norm = _numba_rolling_robust_norm(raw['pressure_trapped_D'].values.astype(np.float32), 21)
        lv_phase = _numba_lotka_volterra_phase(mf_norm, pt_norm, 21)
        lv_bonus = np.clip(1.0 + _numba_rolling_robust_norm(lv_phase, 21) * 0.5, 0.5, 2.0).astype(np.float32)
        tick_net = raw['tick_large_order_net_D'].values.astype(np.float32)
        tick_norm = _numba_rolling_robust_norm(tick_net, 21) * 3.0
        tick_bonus = np.clip(1.0 + tick_norm * 0.3, 0.5, 1.3).astype(np.float32)
        stealth_bonus = np.clip(1.0 + _numba_rolling_robust_norm(raw['stealth_flow_ratio_D'].values.astype(np.float32), 21) * 0.2, 0.8, 1.2).astype(np.float32)
        clus_z = _numba_rolling_robust_norm(raw['tick_clustering_index_D'].values.astype(np.float32), 21)
        clus_bonus = np.clip(1.0 + clus_z * 0.2, 0.8, 1.4).astype(np.float32)
        final_score = adjusted_base * np.maximum(sync_score, 0.1) * np.maximum(jerk_bonus, 0.1) * np.maximum(lv_bonus, 0.1) * np.maximum(tick_bonus, 0.1) * np.maximum(stealth_bonus, 0.1) * clus_bonus
        return pd.Series(final_score.astype(np.float32), index=df_index, dtype=np.float32).clip(0.1, 3.0)
    def _calculate_vpa_elasticity_reflexivity(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        current_elasticity = np.abs(raw['pct_change_D'].values.astype(np.float32)) / (np.abs(pd.Series(raw['volume_D'].values, index=df_index).pct_change().fillna(0).values.astype(np.float32)) + 0.1)
        ela_median = pd.Series(current_elasticity, index=df_index).rolling(21).median().replace(0, 0.01).values.astype(np.float32)
        score = np.clip(np.tanh(current_elasticity / ela_median - 0.8) + 0.5, 0.5, 1.8).astype(np.float32)
        hab_score = np.clip(raw['ACCUM_13_ELASTICITY'].values.astype(np.float32) / (ela_median * 13.0 + 1e-9), 0.8, 1.4).astype(np.float32)
        slope_bonus = np.where(pd.Series(current_elasticity, index=df_index).diff(3).fillna(0).values.astype(np.float32) > 0, 1.2, 1.0).astype(np.float32)
        vpa_adj = raw['VPA_MF_ADJUSTED_EFF_D'].values.astype(np.float32)
        vpa_boost = np.clip(1.0 + _numba_rolling_robust_norm(vpa_adj, 21) * 0.4, 0.6, 1.5).astype(np.float32)
        final_score = score * np.maximum(slope_bonus, 0.1) * np.maximum(hab_score, 0.1) * vpa_boost
        return pd.Series(final_score.astype(np.float32), index=df_index, dtype=np.float32).clip(0.1, 2.5)
    def _calculate_wyckoff_breakout_quality(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        high = raw['high_D'].values.astype(np.float32)
        low = raw['low_D'].values.astype(np.float32)
        close = raw['close_D'].values.astype(np.float32)
        vol = raw['volume_D'].values.astype(np.float32)
        close_prev = raw['close_D'].shift(1).fillna(raw['close_D']).values.astype(np.float32)
        tr = np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
        tr_norm = tr / (close + 1e-9)
        tr_slope = (pd.Series(tr_norm, index=df_index).diff(5).fillna(0) / 5.0).values.astype(np.float32)
        tr_accel = (pd.Series(tr_slope, index=df_index).diff(5).fillna(0) / 5.0).values.astype(np.float32)
        raw_jerk = raw['JERK_3_close_D'].values.astype(np.float32)
        jerk_z = _numba_rolling_robust_norm(raw_jerk, 21) * 3.0
        vol_ma = pd.Series(vol, index=df_index).rolling(21).mean().replace(0, 1).values.astype(np.float32)
        tr_ma = pd.Series(tr, index=df_index).rolling(21).mean().replace(0, 1e-9).values.astype(np.float32)
        acc_compression = pd.Series(np.clip((vol / vol_ma) / (tr / tr_ma + 0.1), 0.5, 3.0).astype(np.float32), index=df_index).rolling(5).mean().fillna(0).values.astype(np.float32)
        highest_21 = pd.Series(high, index=df_index).rolling(21).max().shift(1).fillna(99999).values.astype(np.float32)
        is_breakout = close > highest_21
        quality_mult = np.where(acc_compression > 1.2, 1.5, 1.0).astype(np.float32)
        is_explosive = (jerk_z > 1.5)
        base_q = np.where(is_breakout & is_explosive, 1.6, np.where(is_breakout, 0.9, 0.0)).astype(np.float32)
        prep_q = np.where((tr_slope < 0) & (tr_accel > -0.01), 1.1, 0.9).astype(np.float32)
        final_score = pd.Series(np.maximum(base_q, prep_q) * np.where((tr_slope < 0), 1.3, 1.0).astype(np.float32) * quality_mult, index=df_index, dtype=np.float32).clip(0.1, 2.5)
        return final_score
    def _calculate_premium_reversal_risk(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        turnover = raw['turnover_rate_f_D'].values.astype(np.float32)
        exhaustion_rate = np.clip(turnover / pd.Series(turnover, index=df_index).rolling(21).max().replace(0, 1.0).values.astype(np.float32), 0.0, 1.5).astype(np.float32)
        hab_risk_norm = np.where(raw['ACCUM_13_CLOSING_FLOW'].values.astype(np.float32) > pd.Series(raw['closing_flow_ratio_D'].values.astype(np.float32), index=df_index).rolling(13).mean().replace(0, 0.1).values.astype(np.float32) * 13 * 1.5, 1.25, 1.0).astype(np.float32)
        pressure_penalty = np.where(raw['pressure_trapped_D'].values.astype(np.float32) > 0.5, 1.2, 1.0).astype(np.float32)
        parabolic_risk = raw['STATE_PARABOLIC_WARNING_D'].values.astype(np.float32)
        t1_premium = raw['T1_PREMIUM_EXPECTATION_D'].values.astype(np.float32)
        parabolic_mult = np.where(parabolic_risk > 0.5, 2.0, 1.0).astype(np.float32)
        t1_exhaustion = np.where((t1_premium > 0.8) & (exhaustion_rate > 1.0), 1.5, 1.0).astype(np.float32)
        prof_shock = raw['HAB_SHOCK_21_PROFIT_PRES'].values.astype(np.float32)
        prof_mult = np.clip(1.0 + prof_shock * 0.5, 1.0, 2.0).astype(np.float32)
        reversal_pressure = raw['closing_flow_ratio_D'].values.astype(np.float32) * raw['STATE_EMOTIONAL_EXTREME_D'].values.astype(np.float32) * exhaustion_rate * hab_risk_norm * pressure_penalty * parabolic_mult * t1_exhaustion * prof_mult
        base_risk = 1.0 - reversal_pressure * 0.4
        is_ignition = raw['FLAG_IGNITION'].values > 0.5
        is_limit_up = raw['FLAG_LIMIT_UP'].values > 0.5
        risk_adjustment = np.where(is_ignition | is_limit_up, np.maximum(base_risk, 1.2), base_risk).astype(np.float32)
        return pd.Series(risk_adjustment, index=df_index).clip(0.1, 1.5)
    def _calculate_intraday_decay_model(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        stab = raw['TURNOVER_STABILITY_INDEX_D'].values.astype(np.float32)
        base_decay = np.clip(1.0 + _numba_rolling_robust_norm(stab, 21) * 0.15, 0.7, 1.2).astype(np.float32)
        bad_board = (raw['close_D'].values.astype(np.float32) >= raw['up_limit_D'].values.astype(np.float32) * 0.999) & (raw['closing_flow_ratio_D'].values.astype(np.float32) > 0.4) & (stab < 0.4)
        fragility = np.where(raw['MEAN_13_STABILITY'].values.astype(np.float32) < 0.5, 0.9, 1.0).astype(np.float32)
        repair = np.where((raw['winner_rate_D'].values.astype(np.float32) < 0.15) & (stab < 0.3), 1.5, 1.0).astype(np.float32)
        return pd.Series(base_decay * np.where(bad_board, 0.6, 1.0).astype(np.float32) * repair * fragility, index=df_index, dtype=np.float32).clip(0.4, 1.5)
    def _calculate_sector_resonance_modifier(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        impulse_factor = (1.0 / (1.0 + np.exp(-np.clip(_numba_rolling_robust_norm(raw['SLOPE_5_THEME_HOTNESS_SCORE_D'].values.astype(np.float32), 21) * 3.0, -3.0, 3.0)))).astype(np.float32)
        leadership_bonus = 1.0 + np.clip(raw['ACCUM_21_HIGH_RANK'].values.astype(np.float32) / 21.0, 0.0, 1.0).astype(np.float32) * 0.4
        persistence = np.where((raw['industry_rank_accel_D'].values.astype(np.float32) > 0) & (raw['flow_consistency_D'].values.astype(np.float32) > 0.65), 1.2, 0.8).astype(np.float32)
        rank_pulse = np.where(raw['JERK_3_industry_strength_rank_D'].values.astype(np.float32) < -2.0, 1.3, 1.0).astype(np.float32)
        return pd.Series((0.8 + impulse_factor * 0.4) * persistence * leadership_bonus * rank_pulse, index=df_index, dtype=np.float32).clip(0.6, 2.2)
    def _calculate_volatility_clustering_adjustment(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        bbw_z = _numba_rolling_robust_norm(raw['BBW_21_2.0_D'].values.astype(np.float32), 21) * 3.0
        is_squeeze = (bbw_z < -1.0)
        vcp_ignite = np.where(is_squeeze & (raw['ACCEL_5_BBW_21_2.0_D'].values.astype(np.float32) > 0), 1.4, 1.0).astype(np.float32)
        trap = np.where((bbw_z > 2.0) & (raw['SLOPE_5_BBW_21_2.0_D'].values.astype(np.float32) < 0), 0.7, 1.0).astype(np.float32)
        squeeze_bonus = np.clip(raw['HIST_VOL_SQUEEZE'].values.astype(np.float32) / 10.0, 1.0, 1.3).astype(np.float32)
        adj = np.ones(len(df_index), dtype=np.float32)
        adj = np.where(is_squeeze & (raw['JERK_3_close_D'].values.astype(np.float32) > 0), 1.5 * vcp_ignite * squeeze_bonus, adj)
        adj = np.where(is_squeeze & (raw['JERK_3_close_D'].values.astype(np.float32) < 0), 0.5, adj)
        final_score = adj * trap
        return pd.Series(final_score.astype(np.float32), index=df_index, dtype=np.float32).clip(0.1, 2.5)
    def _calculate_sector_overflow_decay(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        hot = raw['THEME_HOTNESS_SCORE_D'].values.astype(np.float32)
        risk_level = np.where((hot / pd.Series(hot, index=df_index).rolling(60).max().replace(0, 1.0).values.astype(np.float32) > 0.9) & (raw['ACCUM_21_THEME_HOTNESS'].values.astype(np.float32) > 1500.0), 1.0, 0.0)
        fd_vals = _numba_fractal_dimension(np.expand_dims(hot, axis=0), window=13)[0].astype(np.float32)
        slope = pd.Series(fd_vals, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        accel_fd = pd.Series(slope, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        avalanche = ((risk_level == 1.0) | (hot > 80.0) | (raw['ACCEL_5_THEME_HOTNESS_SCORE_D'].values.astype(np.float32) > 0.5)) & (slope > 0) & (accel_fd > 0)
        base = (1.5 / (fd_vals + 1e-9)).clip(0.5, 1.1).astype(np.float32)
        return pd.Series(np.where(avalanche, base * 0.5, base).astype(np.float32), index=df_index, dtype=np.float32).clip(0.1, 1.2)
    def _calculate_entry_accessibility_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        rel_liq = raw['turnover_rate_f_D'].values.astype(np.float32) / pd.Series(raw['turnover_rate_f_D'].values.astype(np.float32), index=df_index).rolling(21).mean().replace(0, 0.1).values.astype(np.float32)
        base_access = np.where(rel_liq < 0.5, 0.4 + rel_liq * 0.5, np.where(rel_liq <= 2.5, 1.0, np.clip(2.5/rel_liq, 0.5, 1.0))).astype(np.float32)
        is_limit_up = raw['FLAG_LIMIT_UP'].values > 0.5
        limit_penalty = np.where(is_limit_up, 1.5, np.where(raw['close_D'].values.astype(np.float32) >= raw['up_limit_D'].values.astype(np.float32) * 0.999, 0.2 + 0.3 * (1.0 - np.clip(raw['closing_flow_intensity_D'].values.astype(np.float32), 0.0, 1.0)), 1.0)).astype(np.float32)
        congestion = np.where((raw['ACCEL_5_trade_count_D'].values.astype(np.float32) > 0.1) & (raw['ACCUM_5_LIQUIDITY_LOCK'].values.astype(np.float32) >= 3) & (~is_limit_up), 0.6, 1.0).astype(np.float32)
        cmf_bonus = np.clip(1.0 + raw['CMF_21_D'].values.astype(np.float32), 0.8, 1.2).astype(np.float32)
        access = base_access * np.maximum(limit_penalty, 0.1) * np.maximum(congestion, 0.1) * np.maximum(cmf_bonus, 0.1)
        return pd.Series(access.astype(np.float32), index=df_index, dtype=np.float32).clip(0.1, 1.8)
    def _calculate_flow_viscosity_drag(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        density = raw['pressure_trapped_D'].values.astype(np.float32)
        turbulence = raw['intraday_cost_center_volatility_D'].values.astype(np.float32)
        persistence = raw['flow_persistence_minutes_D'].values.astype(np.float32)
        viscosity_raw = _numba_navier_stokes_viscosity(density, turbulence, persistence, 21)
        viscosity_norm = _numba_rolling_robust_norm(viscosity_raw, 21)
        viscosity_drag_factor = np.clip(1.0 - np.tanh(viscosity_norm * 0.5), 0.3, 1.5).astype(np.float32)
        adx = raw['ADX_14_D'].values.astype(np.float32)
        lubricant_bonus = np.where((adx > 25) & (viscosity_norm < 0), 1.2, 1.0).astype(np.float32)
        return pd.Series(viscosity_drag_factor * lubricant_bonus, index=df_index, dtype=np.float32).clip(0.1, 2.5)
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V7.0.0 · 全局动力学总线：引入算术-几何混合全息压缩 (AG-Blend)，建立短板坍缩防爆墙"""
        method_name = "calculate_price_volume_dynamics"
        df_index = df.index
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        if not self._validate_all_required_signals(df, {}, {}, method_name, is_debug, probe_ts): return pd.Series(0.0, index=df_index, dtype=np.float32)
        raw = self._get_raw_signals(df, method_name)
        scores = {}
        scores['physical'] = self._calculate_power_transfer_raw_score(df_index, raw, method_name)
        s_geo = (_numba_power_activation(raw['GEOM_ARC_CURVATURE_D'].values, alpha=0.5, gain=2.0) * (0.5 + np.clip(raw['GEOM_REG_R2_D'].values.astype(np.float32), 0.0, 1.0) * 0.5) * (raw['STATE_ROUNDING_BOTTOM_D'].values.astype(np.float32) * 1.2 + 0.5))
        scores['geo'] = pd.Series(s_geo, index=df_index, dtype=np.float32)
        scores['vwap'] = self._calculate_vwap_propulsion_score(raw, df_index, method_name)
        scores['inertia'] = self._calculate_trend_inertia_momentum(raw, df_index, method_name)
        scores['perm'] = self._calculate_market_permeability_index(raw, df_index, method_name)
        scores['entropy'] = self._calculate_entropic_ordering_bonus(raw, df_index, method_name)
        scores['fractal'] = self._calculate_fractal_efficiency_resonance(raw, df_index, method_name)
        scores['hmm'] = self._calculate_hmm_regime_confirmation(raw, df_index, method_name)
        scores['chip'] = self._calculate_chip_lock_efficiency(raw, df_index, method_name)
        scores['micro'] = self._calculate_microstructure_attack_vector(raw, df_index, method_name)
        scores['reflex'] = self._calculate_vpa_elasticity_reflexivity(raw, df_index, method_name)
        scores['wyckoff'] = self._calculate_wyckoff_breakout_quality(raw, df_index, method_name)
        scores['risk'] = self._calculate_premium_reversal_risk(raw, df_index, method_name)
        scores['decay'] = self._calculate_intraday_decay_model(raw, df_index, method_name)
        scores['sector_mod'] = self._calculate_sector_resonance_modifier(raw, df_index, method_name)
        scores['vol_gamma'] = self._calculate_volatility_clustering_adjustment(raw, df_index, method_name)
        scores['sector_decay'] = self._calculate_sector_overflow_decay(raw, df_index, method_name)
        scores['access'] = self._calculate_entry_accessibility_score(raw, df_index, method_name)
        scores['viscosity'] = self._calculate_flow_viscosity_drag(raw, df_index, method_name)
        fatal_trap = raw['FLAG_FATAL_TRAP'].values > 0.5
        is_ignition = raw['FLAG_IGNITION'].values > 0.5
        is_leader = raw['IS_MARKET_LEADER_D'].values > 0.5
        keys_eval = ['inertia', 'perm', 'entropy', 'fractal', 'hmm', 'chip', 'micro', 'reflex', 'wyckoff', 'sector_mod', 'vol_gamma', 'sector_decay', 'access', 'viscosity']
        for key in keys_eval:
            base_arr = np.clip(scores[key].values, 0.1, 2.5)
            shielded_arr = np.where((is_leader | is_ignition) & (~fatal_trap) & (base_arr < 1.0), 1.0, base_arr)
            scores[key] = pd.Series(np.where(fatal_trap, 0.1, shielded_arr).astype(np.float32), index=df_index, dtype=np.float32)
        for key in ['risk', 'decay']: 
            scores[key] = pd.Series(np.where(fatal_trap, 0.1, np.clip(scores[key].values, 0.1, 2.5)).astype(np.float32), index=df_index, dtype=np.float32)
        unadjusted = scores['physical'].values * 0.45 + s_geo * 0.20 + scores['vwap'].values * 0.35
        unadjusted = np.where(fatal_trap, -np.abs(unadjusted) - 2.0, unadjusted).astype(np.float32)
        num_dim = float(len(keys_eval) + 2)
        geom_product = np.ones(len(df_index), dtype=np.float32)
        arith_sum = np.zeros(len(df_index), dtype=np.float32)
        for key in keys_eval + ['risk', 'decay']:
            v = scores[key].values
            geom_product *= v
            arith_sum += v
        geom_mean = np.power(np.clip(geom_product, 1e-15, None), 1.0 / num_dim)
        arith_mean = arith_sum / num_dim
        blend_factor = 0.7 * geom_mean + 0.3 * arith_mean
        matrix_scores = np.vstack([scores[k].values for k in keys_eval + ['risk', 'decay']])
        min_dim_score = np.min(matrix_scores, axis=0)
        collapse_penalty = np.where(min_dim_score < 0.3, np.clip((min_dim_score / 0.3) ** 2, 0.01, 1.0), 1.0).astype(np.float32)
        global_factor_activated = _numba_power_law_activation(blend_factor * collapse_penalty, power=1.5, leak=0.5)
        inv_factor = np.clip(1.0 / (global_factor_activated + 1e-9), 0.1, 5.0)
        final_vals = np.where(unadjusted > 0, unadjusted * global_factor_activated, unadjusted * inv_factor)
        final_score = pd.Series(final_vals, index=df_index, dtype=np.float32).clip(-5.0, 6.0)
        self._persist_hab_state(raw, df_index, method_name)
        if is_debug and probe_ts in df_index: self._print_full_chain_probe(probe_ts, raw, scores, final_score.loc[probe_ts], unadjusted, global_factor_activated, blend_factor, min_dim_score, collapse_penalty)
        return final_score
    def _persist_hab_state(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str):
        if len(df_index) == 0: return
        last_idx, last_ts = -1, df_index[-1]
        hab_snapshot = {"timestamp": last_ts.strftime('%Y-%m-%d'), "updated_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), "metrics": {}}
        for key, series in raw.items():
            if any(p in key for p in ['ACCUM_', 'HIST_', 'MEAN_', 'HAB_']): hab_snapshot["metrics"][key] = float(series.values[last_idx])
        self.latest_hab_state = hab_snapshot
        if hasattr(self.helper, 'update_shared_state'):
            try: self.helper.update_shared_state('HAB_LATEST', hab_snapshot)
            except Exception: pass
    def _print_full_chain_probe(self, probe_ts: pd.Timestamp, raw: Dict[str, pd.Series], scores: Dict[str, pd.Series], final_score: float, unadjusted: np.ndarray, global_factor: np.ndarray, blend_factor: np.ndarray, min_dim_score: np.ndarray, collapse_penalty: np.ndarray):
        if probe_ts not in raw['close_D'].index: return
        idx = raw['close_D'].index.get_loc(probe_ts)
        w_s, w_l = int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[0]), int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[1])
        raw_jerk = raw['JERK_3_close_D']
        jerk_std = raw_jerk.rolling(21).std().replace(0, 1e-6).values[idx]
        jerk_z = raw_jerk.values[idx] / (jerk_std + 1e-9)
        smart_shock = raw['HAB_SHOCK_21_SMART_MONEY'].values[idx]
        vol_shock = raw['HAB_SHOCK_21_VOLUME'].values[idx]
        health_score = np.clip(50.0 + final_score * 10.0, 0, 100)
        is_rally_dist = raw['FLAG_RALLY_DISTRIBUTION'].values[idx] > 0.5
        is_stagnant_dist = raw['FLAG_STAGNANT_DISTRIBUTION'].values[idx] > 0.5
        is_crash = raw['FLAG_PRICE_CRASH'].values[idx] > 0.5
        is_ignition = raw['FLAG_IGNITION'].values[idx] > 0.5
        is_limit_up = raw['FLAG_LIMIT_UP'].values[idx] > 0.5
        is_leader = raw['IS_MARKET_LEADER_D'].values[idx] > 0.5
        is_parabolic = raw['STATE_PARABOLIC_WARNING_D'].values[idx] > 0.5
        is_fatal = is_rally_dist or is_stagnant_dist or is_crash
        has_shield = (is_leader or is_ignition) and not is_fatal
        status_str = "拉高出货诱多" if is_rally_dist else ("滞涨派发" if is_stagnant_dist else ("核按钮级深渊" if is_crash else ("👑绝对龙头受保护" if has_shield else ("🚀主升点火" if is_ignition else ("⚠️抛物线预警" if is_parabolic else ("健康发散" if health_score > 60 else "分歧震荡"))))))
        inv_f = np.clip(1.0 / (global_factor[idx] + 1e-9), 0.1, 5.0)
        actual_multiplier = inv_f if unadjusted[idx] < 0 else global_factor[idx]
        drift, diffusion = _numba_langevin_dynamics(raw['close_D'].values.astype(np.float32), 21)
        snr = drift[idx] / (diffusion[idx] + 1e-9)
        print(f"\n{'='*30} [全息动力学与量子物理场探针 V7.0.0 @ {probe_ts.strftime('%Y-%m-%d')}] {'='*30}")
        print(f"【自适应边界】市值窗口: 短周期 {w_s}d / 长周期 {w_l}d")
        print(f"【基础物理学】收盘: {raw['close_D'].values[idx]:.2f} | T0瞬时速度: {raw['pct_change_D'].values[idx]*100:.2f}% | 换手: {raw['turnover_rate_f_D'].values[idx]*100:.2f}%")
        print(f"【HAB 冲击缓冲池 (全量扩容)】")
        print(f"  ├─ 聪明钱底仓池 (21d/55d): {raw['HAB_STOCK_21_SMART_MONEY'].values[idx]:.1f} / {raw['HAB_STOCK_55_SMART_MONEY'].values[idx]:.1f}")
        print(f"  ├─ 主力冲击强度 (13/21d Shock): {raw['HAB_SHOCK_13_SMART_MONEY'].values[idx]:.4f}x / {smart_shock:.4f}x")
        print(f"  ├─ 极端能量冲击强度 (21d Shock): {raw['HAB_SHOCK_21_ENERGY'].values[idx]:.4f}x")
        print(f"  └─ 大单突袭冲击张力 (21d Shock): {raw['HAB_SHOCK_21_TICK_NET'].values[idx]:.4f}x")
        print(f"【高维引擎与物理场测度】")
        print(f"  ├─ 趋势神针 (ADX/MACDh/CMF): {raw['ADX_14_D'].values[idx]:.1f} / {raw['MACDh_13_34_8_D'].values[idx]:.4f} / {raw['CMF_21_D'].values[idx]:.3f}")
        print(f"  ├─ 郎之万动力学(Langevin): 漂移项={drift[idx]:.2f} | 布朗扩散={diffusion[idx]:.2f} | SNR={snr:.2f}")
        print(f"  ├─ 微观相空间(Lotka-Volterra): 隐秘={raw['stealth_flow_ratio_D'].values[idx]:.2f} | 异动大单={raw['tick_large_order_net_D'].values[idx]:.2f}")
        print(f"  └─ 特权与极限压制: 龙头特权={'[开启(免惩罚)]' if has_shield else '关闭'} | 抛物死劫={'[报警]' if is_parabolic else '安全'} | 涨停封死={'[锁定(+1.5x)]' if is_limit_up else '未封'}")
        print(f"【魔鬼代言人: 木桶坍缩防线 (绝对做空权执行)】")
        print(f"  ├─ 16维算术-几何混合(AG-Blend) 原始融合: {blend_factor[idx]:.4f}")
        print(f"  ├─ 寻找致命短板: 维度最低分 = {min_dim_score[idx]:.3f} -> 触发坍缩倍率 = {collapse_penalty[idx]:.3f}x")
        print(f"  └─ 幂律极化激活算子过滤后 -> 真实杠杆系数: {global_factor[idx]:.4f}x")
        print(f"【健康度与引擎分输出 (绝对领域：相空间修正与龙头特权执行)】")
        print(f"  ★ 无量纲体检评分: {health_score:.1f} ({status_str}) | 瞬时脉冲Z: {jerk_z:.2f}σ")
        print(f"  ★ 极性状态: {'[深渊坠落/派发]' if unadjusted[idx] < 0 else '[主升向上势能]'} | 绝对基础向量: {unadjusted[idx]:.4f}")
        print(f"  ★ 真实物理杠杆: {'极性共轭裂变(乘数倒置)' if unadjusted[idx] < 0 else '正向引力放大'} -> {actual_multiplier:.4f}x")
        tag = "(天网剥夺)" if is_fatal else ("(特权庇护)" if has_shield else "")
        print(f"  [物理中枢] 物理:{scores['physical'].values[idx]:.3f} | 几何:{scores['geo'].values[idx]:.3f} | VWAP(QHO势能):{scores['vwap'].values[idx]:.3f} | 流体粘滞:{scores['viscosity'].values[idx]:.3f}")
        print(f"  [基底结构] 筹码{tag}:{scores['chip'].values[idx]:.3f} | 熵序{tag}:{scores['entropy'].values[idx]:.3f} | 威科夫{tag}:{scores['wyckoff'].values[idx]:.3f} | 惯性{tag}:{scores['inertia'].values[idx]:.3f}")
        print(f"  [微观战术] 攻击(LV相位保护){tag}:{scores['micro'].values[idx]:.3f} | 分形{tag}:{scores['fractal'].values[idx]:.3f} | 渗透{tag}:{scores['perm'].values[idx]:.3f} | 风险:{scores['risk'].values[idx]:.3f}")
        print(f"{'-'*85}")
        print(f" >>> PROCESS_META_PRICE_VOLUME_DYNAMICS 最终多维绝对领域总分: {final_score.values[idx]:.4f}")
        print(f"{'='*85}\n")








