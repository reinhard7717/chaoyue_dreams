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
    """V70.0 · 分形维数算子：支持多维度序列并行计算，采用 float32 降级并移除空行"""
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
    """V51.0 · 自适应去噪算子：增加 NaN 容错与断路保护"""
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
    """V32.0 · 非对称动力学激活算子：强化极端正向爆发，抑制负向噪音"""
    res = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            # 正向信号线性增益，捕捉“夺权”爆发力
            res[i] = x[i] * gain
        else:
            # 负向信号渗漏抑制，保留风险底色
            res[i] = x[i] * alpha
    return res

@jit(nopython=True)
def _numba_fast_rolling_dynamics(data, windows):
    """V38.0 · Numba 原生多尺度动力学算子：一次遍历实现全尺度均值与斜率提取"""
    n = len(data)
    num_wins = len(windows)
    means = np.zeros((num_wins, n))
    slopes = np.zeros((num_wins, n))
    for w_idx in range(num_wins):
        w = windows[w_idx]
        for i in range(w, n):
            # 基础窗口均值
            window_data = data[i-w:i]
            m_val = np.mean(window_data)
            means[w_idx, i] = m_val
            # 简易斜率计算 (末值 vs 首值 / 跨度)
            if window_data[0] != 0:
                slopes[w_idx, i] = (window_data[-1] - window_data[0]) / w
    return means, slopes

@jit(nopython=True)
def _numba_hmm_regime_probability(flow_n, vol_n, price_n, vwap_dist_n):
    """V70.0 · HMM 体制概率算子：全面降级为 float32 运算，清除空行"""
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
    """V70.0 · 鲁棒动力学算子：降级为 float32 并移除空行，提升内存效率"""
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
            slope[i] = 0.0
        else:
            slope[i] = delta / win
    for i in range(win, n):
        delta_s = slope[i] - slope[i-1]
        if np.abs(delta_s) < change_threshold / 10.0:
            accel[i] = 0.0
        else:
            accel[i] = delta_s
    for i in range(win, n):
        delta_a = accel[i] - accel[i-1]
        if np.abs(delta_a) < change_threshold / 100.0:
            jerk[i] = 0.0
        else:
            jerk[i] = delta_a
    return slope, accel, jerk

class CalculatePriceVolumeDynamics:
    """
    PROCESS_META_POWER_TRANSFER
    计算价量动态的类，用于分析价格和成交量的变化趋势。
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        # 从 helper 获取参数，确保访问的是 process_intelligence_params 块
        self.process_params = self.helper.params
        self.std_window = self.helper.std_window # Needed for dynamic thresholds

    def _setup_debug_info(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp], Dict]:
        """
        设置调试信息，包括是否启用调试、探针日期和临时调试值字典。
        """
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
        """
        统一打印价量动态计算的调试信息。
        """
        debug_output = {}
        for key, value in debug_values.items():
            if isinstance(value, dict):
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {key} ---"] = ""
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.Series):
                        val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                        debug_output[f"        '{sub_key}': {val:.4f}"] = ""
                    elif isinstance(sub_value, dict):
                        debug_output[f"        '{sub_key}':"] = ""
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            if isinstance(sub_sub_value, pd.Series):
                                val = sub_sub_value.loc[probe_ts] if probe_ts in sub_sub_value.index else np.nan
                                debug_output[f"          {sub_sub_key}: {val:.4f}"] = ""
                            else:
                                debug_output[f"          {sub_sub_key}: {sub_sub_value}"] = ""
                    else:
                        debug_output[f"        '{sub_key}': {sub_value}"] = ""
            else:
                debug_output[key] = value # For initial messages
        final_score_val = debug_values.get("最终融合分数", {}).get("final_score", pd.Series(np.nan)).loc[probe_ts] if probe_ts in debug_values.get("最终融合分数", {}).get("final_score", pd.Series(np.nan)).index else np.nan
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: {final_message}，最终分值: {final_score_val:.4f}"] = ""
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def _get_pvd_params(self, config: Dict) -> Dict:
        """
        从配置中获取价量动态相关的参数。
        """
        return get_param_value(config.get('price_volume_dynamics_params'), {})

    def _validate_all_required_signals(self, df: pd.DataFrame, pvd_params: Dict, mtf_slope_accel_weights: Dict, method_name: str, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> bool:
        """V64.0 · 核心信号校验：新增换手率与成交笔数的动力学校验"""
        fib_windows = [3, 5, 8, 13, 21]
        dynamic_base_cols = [
            'net_amount_rate_D', 'winner_rate_D', 'SMART_MONEY_HM_NET_BUY_D', 
            'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'THEME_HOTNESS_SCORE_D', 'chip_entropy_D',
            'volume_D', 'pct_change_D', 'close_D',
            'market_sentiment_score_D', 'industry_strength_rank_D',
            'turnover_rate_f_D', 'trade_count_D' # 新增动力学目标
        ]
        base_required = [
            'close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'net_amount_rate_D', 'winner_rate_D',
            'up_limit_D', 'down_limit_D', 'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D',
            'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'pressure_release_index_D', 'BBW_21_2.0_D', 
            'VPA_EFFICIENCY_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'turnover_rate_f_D',
            'STATE_ROUNDING_BOTTOM_D', 'STATE_GOLDEN_PIT_D', 'STATE_TRENDING_STAGE_D', 'price_percentile_position_D',
            'TURNOVER_STABILITY_INDEX_D', 'STATE_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 
            'THEME_HOTNESS_SCORE_D', 'chip_entropy_D', 'cost_50pct_D',
            'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'trade_count_D',
            'market_sentiment_score_D', 'industry_strength_rank_D', 'industry_rank_accel_D'
        ]
        dynamic_required = [f"{p}_{w}_{c}" for c in dynamic_base_cols for w in fib_windows for p in ['SLOPE', 'ACCEL', 'JERK']]
        all_required = base_required + dynamic_required
        return self.helper._validate_required_signals(df, all_required, method_name)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Dict[str, pd.Series]:
        """V70.0 · 原料加载层：通过 float32 降级与 NumPy 向量化实现极速量纲校准，移除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        raw_signals = {}
        base_cols = ['close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'pct_change_D', 'net_amount_rate_D', 'trade_count_D', 'turnover_rate_f_D']
        struct_cols = ['winner_rate_D', 'chip_concentration_ratio_D', 'chip_entropy_D', 'cost_50pct_D', 'absorption_energy_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'price_percentile_position_D']
        tech_cols = ['SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D', 'pressure_release_index_D', 'up_limit_D', 'down_limit_D', 'closing_flow_ratio_D', 'TURNOVER_STABILITY_INDEX_D', 'STATE_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 'industry_strength_rank_D', 'industry_rank_accel_D', 'STATE_ROUNDING_BOTTOM_D', 'STATE_GOLDEN_PIT_D', 'STATE_TRENDING_STAGE_D', 'THEME_HOTNESS_SCORE_D', 'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'market_sentiment_score_D']
        for col in base_cols + struct_cols + tech_cols:
            if col not in df.columns: raise KeyError(f"CRITICAL: 军械库缺失关键列 {col}")
            raw_signals[col] = df[col].ffill().fillna(0.0).astype(np.float32)
        raw_vwap = (raw_signals['amount_D'] / (raw_signals['volume_D'] + 1e-9)).fillna(raw_signals['close_D'])
        recent_closes = raw_signals['close_D'].values[-60:]
        recent_vwaps = raw_vwap.values[-60:]
        valid_mask = recent_vwaps > 0
        scale_factor = 1.0
        if np.any(valid_mask):
            median_ratio = np.median(recent_vwaps[valid_mask] / recent_closes[valid_mask])
            if median_ratio < 0.5 or median_ratio > 2.0:
                scale_factor = 10.0 ** (-np.round(np.log10(median_ratio)))
        raw_signals['VWAP_D'] = (raw_vwap * scale_factor).astype(np.float32)
        if raw_signals['winner_rate_D'].max() > 1.1: raw_signals['winner_rate_D'] /= 100.0
        if raw_signals['SMART_MONEY_HM_NET_BUY_D'].abs().sum() < 1e-5:
            raw_signals['SMART_MONEY_HM_NET_BUY_D'] = (raw_signals['buy_elg_amount_D'] - raw_signals['sell_elg_amount_D']) + (raw_signals['buy_lg_amount_D'] - raw_signals['sell_lg_amount_D'])
        threshold_map = {'net_amount_rate_D': (0.01, 0.005), 'winner_rate_D': (0.01, 0.005), 'SMART_MONEY_HM_NET_BUY_D': (10, 10), 'VPA_EFFICIENCY_D': (0.01, 0.01), 'BBW_21_2.0_D': (0.001, 0.0001), 'THEME_HOTNESS_SCORE_D': (0.1, 0.1), 'chip_entropy_D': (0.01, 0.001), 'volume_D': (100, 10), 'pct_change_D': (0.0001, 0.0001), 'close_D': (0.1, 0.01), 'market_sentiment_score_D': (0.1, 0.1), 'industry_strength_rank_D': (0.001, 0.001), 'turnover_rate_f_D': (0.01, 0.01), 'trade_count_D': (10, 5)}
        fib_windows = [3, 5, 8, 13, 21]
        for col, (abs_th, chg_th) in threshold_map.items():
            base_vals = raw_signals[col].values
            for win in fib_windows:
                s, a, j = _numba_robust_dynamics(base_vals, win=win, abs_threshold=abs_th, change_threshold=chg_th)
                raw_signals[f"SLOPE_{win}_{col}"] = pd.Series(s, index=df.index, dtype=np.float32)
                raw_signals[f"ACCEL_{win}_{col}"] = pd.Series(a, index=df.index, dtype=np.float32)
                raw_signals[f"JERK_{win}_{col}"] = pd.Series(j, index=df.index, dtype=np.float32)
        if is_debug and probe_ts in df.index:
            print(f"\n[原料自适应极速探针 V70.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    VWAP对齐系数: {scale_factor}, 最终VWAP: {raw_signals['VWAP_D'].loc[probe_ts]:.2f}")
        return raw_signals

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V70.0 · 全局动力学集成总线：全链路 float32 向量化，移除索引依赖与空行"""
        method_name = "calculate_price_volume_dynamics"
        df_index = df.index
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        if not self._validate_all_required_signals(df, {}, {}, method_name, is_debug, probe_ts): return pd.Series(0.0, index=df_index, dtype=np.float32)
        raw = self._get_raw_signals(df, method_name)
        physical_base = self._calculate_power_transfer_raw_score(df_index, raw, method_name).values
        geo_resonance = (pd.Series(_numba_power_activation(raw['GEOM_ARC_CURVATURE_D'].values, gain=2.0), index=df_index) * (1.0 - raw['GEOM_REG_R2_D']) * (raw['STATE_ROUNDING_BOTTOM_D'].astype(np.float32) * 1.2 + 0.5)).values
        vwap_propulsion = self._calculate_vwap_propulsion_score(raw, df_index, method_name).values
        unadjusted_intensity = physical_base * 0.50 + geo_resonance * 0.25 + vwap_propulsion * 0.25
        final_vals = unadjusted_intensity * self._calculate_trend_inertia_momentum(raw, df_index, method_name).values * \
                     self._calculate_market_permeability_index(raw, df_index, method_name).values * \
                     self._calculate_entropic_ordering_bonus(raw, df_index, method_name).values * \
                     self._calculate_fractal_efficiency_resonance(raw, df_index, method_name).values * \
                     self._calculate_hmm_regime_confirmation(raw, df_index, method_name).values * \
                     self._calculate_chip_lock_efficiency(raw, df_index, method_name).values * \
                     self._calculate_microstructure_attack_vector(raw, df_index, method_name).values * \
                     self._calculate_vpa_elasticity_reflexivity(raw, df_index, method_name).values * \
                     self._calculate_wyckoff_breakout_quality(raw, df_index, method_name).values * \
                     self._calculate_premium_reversal_risk(raw, df_index, method_name).values * \
                     self._calculate_intraday_decay_model(raw, df_index, method_name).values * \
                     self._calculate_sector_resonance_modifier(raw, df_index, method_name).values * \
                     self._calculate_volatility_clustering_adjustment(raw, df_index, method_name).values * \
                     self._calculate_sector_overflow_decay(raw, df_index, method_name).values * \
                     self._calculate_entry_accessibility_score(raw, df_index, method_name).values
        final_score = pd.Series(final_vals, index=df_index).clip(-3.5, 6.0).astype(np.float32)
        if is_debug and probe_ts in df_index:
            print(f"\n[PROCESS_META_POWER_TRANSFER 全链极速探针 V70.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    >>> 最终输出分值: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_power_transfer_raw_score(self, df_index: pd.Index, raw: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V70.0 · 物理动力引擎：采用 float32 向量化合成与降级去噪，消除 Series 对齐开销，移除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        score_calculate = CalculatePowerTransferRawScore(is_debug, probe_ts)
        vol_adj = raw['BBW_21_2.0_D'].values.astype(np.float64)
        rolling_tc = pd.Series(raw['trade_count_D'].values, index=df_index).rolling(21).mean().replace(0, 1).values
        conf = (raw['trade_count_D'].values / (rolling_tc + 1e-9)).astype(np.float64)
        j_c = _numba_adaptive_denoise_dynamics(raw['JERK_3_net_amount_rate_D'].values.astype(np.float64), vol_adj, conf)
        a_c = _numba_adaptive_denoise_dynamics(raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float64), vol_adj, conf)
        act_impulse = pd.Series(_numba_power_activation((j_c * 0.45 + a_c * 0.55), gain=1.8), index=df_index, dtype=np.float32)
        norm_imp = score_calculate._calculate_dynamic_impulse_norm(act_impulse, raw, df_index, method_name).values
        comp_imp = score_calculate._calculate_limit_price_compensation(pd.Series(norm_imp, index=df_index), raw, df_index, method_name).values
        auc_pred = score_calculate._calculate_auction_prediction(raw, df_index, method_name).values
        _, f_slopes = _numba_fast_rolling_dynamics(raw['net_amount_rate_D'].values.astype(np.float64), np.array([3, 5, 8, 13, 21], dtype=np.int64))
        mcv = np.dot(np.array([0.35, 0.25, 0.20, 0.10, 0.10], dtype=np.float32), f_slopes.astype(np.float32))
        phy_score = (comp_imp * 2.0 * 0.30 + auc_pred * 0.35 + mcv * 0.35).astype(np.float32)
        if is_debug and probe_ts in df_index:
            print(f"\n[物理引擎极速探针 V70.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    物理合成总分: {phy_score[df_index.get_loc(probe_ts)]:.4f}")
        return pd.Series(phy_score, index=df_index, dtype=np.float32)

    def _calculate_premium_reversal_risk(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 溢价回吐风险：NumPy 向量化加速，清除所有空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        turnover = raw['turnover_rate_f_D'].values
        extreme = raw['STATE_EMOTIONAL_EXTREME_D'].astype(np.float32).values
        ratio = raw['closing_flow_ratio_D'].values
        exhaustion = np.clip(turnover / 15.0, 0.0, 1.5)
        pressure = ratio * extreme * exhaustion
        risk_adj = pd.Series(1.0 - pressure * 0.4, index=df_index, dtype=np.float32).clip(0.6, 1.0)
        return risk_adj

    def _calculate_intraday_decay_model(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 日内衰减模型：全向量化逻辑判断与 float32 类型降级，提升合成速度，移除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        stab = np.clip(raw['TURNOVER_STABILITY_INDEX_D'].values, 0.0, 1.0).astype(np.float32)
        close = raw['close_D'].values
        up = raw['up_limit_D'].values
        ratio = raw['closing_flow_ratio_D'].values
        bad_mask = (close >= up * 0.999) & (ratio > 0.4) & (stab < 0.4)
        winner = raw['winner_rate_D'].values
        repair = np.where((winner < 0.15) & (stab < 0.3), 1.5, 1.0).astype(np.float32)
        decay = (0.6 + stab * 0.4) * np.where(bad_mask, 0.6, 1.0).astype(np.float32) * repair
        return pd.Series(decay, index=df_index, dtype=np.float32).clip(0.3, 1.5)

    def _calculate_sector_resonance_modifier(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 板块共振算子：全量 NumPy 动力学合成，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        s_hot = raw['SLOPE_5_THEME_HOTNESS_SCORE_D'].values
        a_hot = raw['ACCEL_5_THEME_HOTNESS_SCORE_D'].values
        impulse = (s_hot * 0.6 + a_hot * 0.4)
        persistence = np.where((raw['industry_rank_accel_D'].values > 0) & (raw['flow_consistency_D'].values > 0.65), 1.2, 0.8).astype(np.float32)
        mod = (1.0 + _numba_power_activation(impulse.astype(np.float64), gain=0.5).astype(np.float32)) * persistence
        return pd.Series(mod, index=df_index, dtype=np.float32).clip(0.6, 1.8)

    def _calculate_volatility_clustering_adjustment(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 波动率伽马模型：全向量化 float32 爆炸判定，消除空行并提升内存命中率"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        bbw = raw['BBW_21_2.0_D'].values
        s_bbw = raw['SLOPE_5_BBW_21_2.0_D'].values
        a_bbw = raw['ACCEL_5_BBW_21_2.0_D'].values
        j_bbw = raw['JERK_3_BBW_21_2.0_D'].values
        bbw_ma = pd.Series(bbw, index=df_index).rolling(21).mean().fillna(pd.Series(bbw, index=df_index)).values
        exp = (bbw < bbw_ma * 1.2) & (a_bbw > 0) & (j_bbw > 0.01)
        trap = (bbw > bbw_ma * 1.5) & ((s_bbw < 0) | (a_bbw < 0))
        p_jerk = raw['JERK_3_close_D'].values
        adj = np.ones(len(df_index), dtype=np.float32)
        adj = np.where(exp & (p_jerk > 0), 1.5, adj)
        adj = np.where(exp & (p_jerk < 0), 0.5, adj)
        adj = np.where(trap, 0.8, adj)
        return pd.Series(adj, index=df_index, dtype=np.float32).clip(0.4, 1.6)

    def _calculate_sector_overflow_decay(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 熵增雪崩模型：极速分形动力学向量化，采用 float32 降级，消除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        hot = raw['THEME_HOTNESS_SCORE_D'].values
        fd_all = _numba_fractal_dimension(np.expand_dims(hot, axis=0).astype(np.float64), window=13)
        fd_vals = fd_all[0].astype(np.float32)
        slope = pd.Series(fd_vals, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        accel = pd.Series(slope, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        avalanche = (raw['THEME_HOTNESS_SCORE_D'].values > 80) & (slope > 0) & (accel > 0)
        base = (1.5 / (fd_vals + 1e-9)).clip(0.6, 1.1)
        res = np.where(avalanche, base * 0.7, base).astype(np.float32)
        return pd.Series(res, index=df_index, dtype=np.float32)

    def _calculate_hmm_regime_confirmation(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · HMM 动力学共振：基于 float32 向量化的体制确认逻辑，移除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        large_net = raw['SMART_MONEY_HM_NET_BUY_D']
        roll_mean = large_net.rolling(21).mean().fillna(0)
        roll_std = large_net.rolling(21).std().fillna(1e-9)
        flow_n = ((large_net - roll_mean) / roll_std).values.astype(np.float32)
        vol_n = ((raw['volume_D'] / raw['volume_D'].rolling(21).mean().replace(0, 1)) - 1.0).values.astype(np.float32)
        price_n = (raw['pct_change_D'] * 10.0).values.astype(np.float32)
        vwap_dist = ((raw['close_D'] - raw['VWAP_D']) / (raw['close_D'] * raw['BBW_21_2.0_D'].clip(lower=0.01))).values.astype(np.float32)
        markup_prob = _numba_hmm_regime_probability(flow_n, vol_n, price_n, vwap_dist)
        markup_prob_s = pd.Series(markup_prob, index=df_index, dtype=np.float32)
        prob_accel = markup_prob_s.diff(3).diff(3).fillna(0).values
        prob_jerk = markup_prob_s.diff(1).diff(1).diff(1).fillna(0).values
        base_factor = np.where(markup_prob > 0.5, 1.0 + (markup_prob - 0.5), 0.8 + markup_prob * 0.4)
        dynamic_bonus = 1.0 + np.where((markup_prob > 0.6) & (prob_accel > 0), 0.2, 0.0) + np.where((markup_prob > 0.4) & (prob_jerk > 0.1), 0.3, 0.0)
        res = pd.Series(base_factor * dynamic_bonus, index=df_index, dtype=np.float32)
        if is_debug and probe_ts in df_index:
            print(f"\n[HMM 极速探针 V70.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    拉升概率: {markup_prob_s.loc[probe_ts]:.4f}")
        return res

    def _calculate_fractal_efficiency_resonance(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 分形相干效率模型：移除 Series 索引开销，采用 NumPy 掩码相变判定，消除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        c_vals = raw['close_D'].values.astype(np.float64)
        v_vals = raw['volume_D'].values.astype(np.float64)
        input_arrays = np.vstack((c_vals, v_vals))
        f_all = _numba_fractal_dimension(input_arrays, window=21)
        h_p = (2.0 - f_all[0]).astype(np.float32)
        h_v = (2.0 - f_all[1]).astype(np.float32)
        h_slope = pd.Series(h_p, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        gap = np.abs(h_p - h_v)
        gap_a = pd.Series(gap, index=df_index).diff(3).diff(3).fillna(0).values.astype(np.float32)
        pers = np.where(h_p > 0.55, 1.2, np.where(h_p < 0.45, 0.6, 0.9)).astype(np.float32)
        res_f = np.clip(1.0 - gap * 2.0, 0.5, 1.1).astype(np.float32)
        final = pers * res_f
        final = np.where((h_p > 0.55) & (h_slope > 0), final * 1.2, final)
        final = np.where((gap > 0.3) & (gap_a > 0), final * 0.7, final)
        return pd.Series(final, index=df_index, dtype=np.float32)

    def _calculate_chip_lock_efficiency(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 筹码锁定效率：纯向量化掩码操作，移除 mask 依赖与空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        winner = raw['winner_rate_D'].values
        turnover = raw['turnover_rate_f_D'].values
        turnover_norm = np.clip(turnover / 5.0, 0.0, 1.0)
        static_lock = winner * (1.0 - turnover_norm)
        accel_winner = raw['ACCEL_5_winner_rate_D'].values
        accel_turnover = raw['ACCEL_5_turnover_rate_f_D'].values
        jerk_winner = raw['JERK_3_winner_rate_D'].values
        kinetic_bonus = 1.0 + np.where((accel_winner > 0) & (accel_turnover < 0), 0.3, 0.0) + np.where(jerk_winner > 0.1, 0.2, 0.0)
        cost_series = raw['cost_50pct_D'].values
        close_series = raw['close_D'].values
        cost_50 = np.where(cost_series == 0, close_series, cost_series)
        break_cost = (close_series > cost_50).astype(np.float32)
        final_efficiency = pd.Series(static_lock * kinetic_bonus * (0.8 + break_cost * 0.2), index=df_index, dtype=np.float32)
        if is_debug and probe_ts in df_index:
            print(f"\n[筹码极速探针 V70.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    静态锁定: {static_lock[df_index.get_loc(probe_ts)]:.4f}")
        return final_efficiency

    def _calculate_microstructure_attack_vector(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V64.0 · 微观矢量脉冲模型：基于 Jerk/Accel 的主力狙击识别"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 基础攻击矢量
        elg_net = (raw['buy_elg_amount_D'] - raw['sell_elg_amount_D']).fillna(0)
        lg_net = (raw['buy_lg_amount_D'] - raw['sell_lg_amount_D']).fillna(0)
        total_vol = raw['amount_D'].replace(0, 1e-9)
        elg_ratio = elg_net / total_vol
        lg_ratio = lg_net / total_vol
        net_strength = np.tanh((elg_ratio + lg_ratio) * 10.0)
        # 2. 主力脉冲 (Smart Money Jerk)
        # 使用 SMART_MONEY_HM_NET_BUY_D 的 Jerk 作为主力点火信号
        sm_jerk = raw['JERK_3_SMART_MONEY_HM_NET_BUY_D']
        is_sniper_shot = sm_jerk > 0.5 # 剧烈的资金脉冲
        # 3. 扫货加速度 (Sweeping Acceleration)
        # 逻辑：单笔均额在加速变大 <=> 量能加速 > 笔数加速
        accel_vol = raw['ACCEL_5_volume_D']
        accel_cnt = raw['ACCEL_5_trade_count_D']
        # 差值 > 0 说明大单在进场
        size_accel = accel_vol - accel_cnt
        is_sweeping = size_accel > 0.05
        # 4. 矢量合成
        # 基础分
        base_vector = (0.5 + net_strength * 0.5)
        # 动力学乘数
        # 狙击脉冲 + 扫货加速
        kinetic_multiplier = 1.0 + np.where(is_sniper_shot, 0.4, 0.0) + np.where(is_sweeping, 0.3, 0.0)
        final_vector = pd.Series(base_vector * kinetic_multiplier, index=df_index).clip(0, 2.0)
        if is_debug and probe_ts in df_index:
            print(f"\n[微观矢量脉冲探针 V64 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    主力脉冲(Jerk): {sm_jerk.loc[probe_ts]:.4f} ({'狙击点火' if is_sniper_shot.loc[probe_ts] else '平稳'})")
            print(f"    均额加速度差: {size_accel.loc[probe_ts]:.4f} ({'扫货中' if is_sweeping.loc[probe_ts] else '散户主导'})")
            print(f"    >>> 最终微观攻击矢量: {final_vector.loc[probe_ts]:.4f}")
        return final_vector.astype(np.float32)

    def _calculate_vpa_elasticity_reflexivity(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 流体反身性模型：全面向量化 float32 弹性计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        pp = np.abs(raw['pct_change_D'].values)
        pv = np.abs(pd.Series(raw['volume_D'].values, index=df_index).pct_change().fillna(0).values)
        base_e = pp / (pv + 0.1)
        e_slope = pd.Series(base_e, index=df_index).diff(3).values / 3.0
        jp = raw['JERK_3_close_D'].values
        jv = raw['JERK_3_volume_D'].values
        ap = raw['ACCEL_5_close_D'].values
        av = raw['ACCEL_5_volume_D'].values
        score = np.clip(np.tanh(base_e), 0.5, 1.5)
        final = score * np.where(e_slope > 0, 1.2, 1.0)
        final = np.where((jp > 0.05) & (jv > 0.1), final * 1.5, np.where((ap > 0) & (av > 0), final * 1.2, final))
        return pd.Series(final, index=df_index, dtype=np.float32).clip(0.5, 1.8)

    def _calculate_wyckoff_breakout_quality(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 动态威科夫突破：TR 动力学全向量化，移除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        high = raw['high_D'].values
        low = raw['low_D'].values
        close = raw['close_D'].values
        close_prev = raw['close_D'].shift(1).fillna(raw['close_D']).values
        tr = np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
        tr_norm = tr / (close + 1e-9)
        tr_s = pd.Series(tr_norm, index=df_index)
        tr_slope = (tr_s.diff(5) / 5.0).values
        tr_accel = (pd.Series(tr_slope, index=df_index).diff(5) / 5.0).values
        jerk_p = raw['JERK_3_close_D'].values
        highest_21 = raw['high_D'].rolling(21).max().shift(1).fillna(99999).values
        is_breakout = close > highest_21
        quality = np.where(is_breakout & (jerk_p > 0.1), 1.6, np.where(is_breakout & (jerk_p <= 0.05), 0.8, np.where((tr_slope < 0) & (tr_accel > -0.01), 1.1, 0.9)))
        final_score = pd.Series(quality * np.where((tr_slope < 0) & (tr_accel > -0.01), 1.3, 1.0), index=df_index, dtype=np.float32).clip(0.6, 2.0)
        return final_score

    def _calculate_trend_inertia_momentum(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 趋势运动学模型：全向量化 float32 动能合成，清除所有空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        c_vals = raw['close_D'].values
        ma5 = raw['close_D'].rolling(5).mean().values
        ma21 = raw['close_D'].rolling(21).mean().values
        ma55 = raw['close_D'].rolling(55).mean().values
        s_vals = raw['SLOPE_5_close_D'].values
        a_vals = raw['ACCEL_5_close_D'].values
        j_vals = raw['JERK_3_close_D'].values
        r2_vals = raw['GEOM_REG_R2_D'].values
        alignment = np.where((ma5 > ma21) & (ma21 > ma55), 1.0, 0.8).astype(np.float32)
        kinematic = 1.0 + np.where((s_vals > 0) & (a_vals > 0), 0.3, 0.0) + np.where(j_vals > 0.1, 0.2, 0.0) - np.where((s_vals > 0) & (a_vals < 0), 0.2, 0.0)
        final_inertia = pd.Series(alignment * kinematic * (0.6 + np.clip(r2_vals, 0.0, 1.0) * 0.4), index=df_index, dtype=np.float32).clip(0.6, 1.6)
        if is_debug and probe_ts in df_index:
            print(f"\n[趋势惯性极速探针 V70.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    惯性系数: {final_inertia.loc[probe_ts]:.4f}")
        return final_inertia

    def _calculate_market_permeability_index(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 市场渗透率模型：float32 降级与向量化相变判定，清除所有空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        sent = raw['market_sentiment_score_D'].values
        accel = raw['ACCEL_5_market_sentiment_score_D'].values
        rank = raw['industry_strength_rank_D'].values
        jerk = raw['JERK_3_industry_strength_rank_D'].values
        perm = np.where((sent < 20) & (accel > 0), 1.3, np.where((sent > 80) & (accel > 0), 0.7, 1.0)).astype(np.float32)
        bonus = np.where(jerk < -2.0, 1.3, np.where(rank < 0.1, 1.1, 0.9)).astype(np.float32)
        final_ctx = pd.Series(perm * bonus, index=df_index, dtype=np.float32).clip(0.6, 1.8)
        if is_debug and probe_ts in df_index:
            print(f"\n[市场渗透极速探针 V70.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    渗透系数: {final_ctx.loc[probe_ts]:.4f}")
        return final_ctx

    def _calculate_entry_accessibility_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 入场可获得性：极速向量化根号平滑，移除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        tf = raw['turnover_rate_f_D'].values
        liq = np.clip(np.sqrt(tf) / 1.2247, 0.1, 1.1)
        up_limit = raw['up_limit_D'].values
        close = raw['close_D'].values
        intensity = raw['closing_flow_intensity_D'].values
        premium = raw['T1_PREMIUM_EXPECTATION_D'].values
        sealing = np.clip(intensity * premium, 0.0, 1.0)
        limit_access = np.where(close >= up_limit * 0.999, 0.4 * (1.0 - sealing), 1.0).astype(np.float32)
        return pd.Series(limit_access * liq, index=df_index, dtype=np.float32).clip(0.0, 1.0)

    def _calculate_entropic_ordering_bonus(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · 熵减有序性因子：纯数组掩码计算，消除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        s5 = raw['SLOPE_5_chip_entropy_D'].values
        a5 = raw['ACCEL_5_chip_entropy_D'].values
        pct = raw['pct_change_D'].values
        locking = -(s5 * 0.7 + a5 * 0.3)
        bonus = np.clip(np.tanh(locking * 5.0), 0.0, 1.0) * 1.5
        penalty = np.where((pct > 0) & (s5 > 0), 0.7, 1.0).astype(np.float32)
        final_factor = pd.Series((1.0 + bonus) * penalty, index=df_index, dtype=np.float32).clip(0.7, 1.5)
        return final_factor

    def _calculate_vwap_propulsion_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V70.0 · VWAP 推进力：全向量化逻辑，移除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        close = raw['close_D'].values
        vwap = raw['VWAP_D'].values
        vwap_slope = pd.Series(vwap, index=df_index).diff(3).values / 3.0
        bias = (close - vwap) / (vwap + 1e-9)
        propulsion = np.where(vwap_slope > 0, 1.0, 0.0) + np.where((bias > 0) & (bias < 0.05), 0.2, 0.0) + np.clip(np.tanh(vwap_slope * 10.0), 0.0, 0.3)
        final_score = pd.Series(propulsion, index=df_index, dtype=np.float32).clip(0, 1.5)
        if is_debug and probe_ts in df_index:
            print(f"\n[VWAP 推进极速探针 V70.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    最终得分: {final_score.loc[probe_ts]:.4f}")
        return final_score


