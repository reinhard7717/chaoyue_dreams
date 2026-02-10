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
    """V7.0.0 · 分形维数算子：支持多维度序列并行计算，采用 float32 降级并移除空行"""
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
    """V7.0.0 · HMM 体制概率算子：全面降级为 float32 运算，清除空行"""
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
    """V7.0.0 · 鲁棒动力学算子：降级为 float32 并移除空行，提升内存效率"""
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

@jit(nopython=True)
def _numba_rolling_accumulation(data, window=13):
    """V71.0 · 滚动累积算子：高性能计算底仓能量累积，移除所有空行"""
    n = len(data)
    res = np.zeros(n, dtype=np.float32)
    for i in range(window, n):
        accum = 0.0
        for j in range(i - window, i):
            accum += data[j]
        res[i] = accum
    return res

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
        """V73.0 · 原料加载层：深度集成 HAB 缓冲系统，计算尾盘压力、板块厚度与波动紧致度的周期累积，清除所有空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        raw_signals = {}
        base_cols = ['close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'pct_change_D', 'net_amount_rate_D', 'trade_count_D', 'turnover_rate_f_D']
        struct_cols = ['winner_rate_D', 'chip_concentration_ratio_D', 'chip_entropy_D', 'cost_50pct_D', 'absorption_energy_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'price_percentile_position_D']
        tech_cols = ['SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D', 'pressure_release_index_D', 'up_limit_D', 'down_limit_D', 'closing_flow_ratio_D', 'TURNOVER_STABILITY_INDEX_D', 'STATE_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 'industry_strength_rank_D', 'industry_rank_accel_D', 'STATE_ROUNDING_BOTTOM_D', 'STATE_GOLDEN_PIT_D', 'STATE_TRENDING_STAGE_D', 'THEME_HOTNESS_SCORE_D', 'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'market_sentiment_score_D']
        for col in base_cols + struct_cols + tech_cols:
            if col not in df.columns: raise KeyError(f"CRITICAL: 军械库缺失关键列 {col}")
            raw_signals[col] = df[col].ffill().fillna(0.0).astype(np.float32)
        raw_vwap = (raw_signals['amount_D'] / (raw_signals['volume_D'] + 1e-9)).fillna(raw_signals['close_D'])
        r_c, r_v = raw_signals['close_D'].values[-60:], raw_vwap.values[-60:]
        v_m, s_f = r_v > 0, 1.0
        if np.any(v_m):
            m_r = np.median(r_v[v_m] / r_c[v_m])
            if m_r < 0.5 or m_r > 2.0: s_f = 10.0 ** (-np.round(np.log10(m_r)))
        raw_signals['VWAP_D'] = (raw_vwap * s_f).astype(np.float32)
        if raw_signals['winner_rate_D'].max() > 1.1: raw_signals['winner_rate_D'] /= 100.0
        sm_v = raw_signals['SMART_MONEY_HM_NET_BUY_D'].values
        if np.abs(sm_v).sum() < 1e-5: sm_v = (raw_signals['buy_elg_amount_D'] - raw_signals['sell_elg_amount_D'] + raw_signals['buy_lg_amount_D'] - raw_signals['sell_lg_amount_D']).values
        raw_signals['ACCUM_13_SMART_MONEY'] = pd.Series(_numba_rolling_accumulation(sm_v.astype(np.float32), 13), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_SMART_MONEY'] = pd.Series(_numba_rolling_accumulation(sm_v.astype(np.float32), 21), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_13_CLOSING_FLOW'] = pd.Series(_numba_rolling_accumulation(raw_signals['closing_flow_ratio_D'].values, 13), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_THEME_HOTNESS'] = pd.Series(_numba_rolling_accumulation(raw_signals['THEME_HOTNESS_SCORE_D'].values, 21), index=df.index, dtype=np.float32)
        raw_signals['MEAN_13_STABILITY'] = pd.Series(_numba_rolling_accumulation(raw_signals['TURNOVER_STABILITY_INDEX_D'].values, 13), index=df.index, dtype=np.float32) / 13.0
        bbw_tight = np.clip(1.0 - raw_signals['BBW_21_2.0_D'].values, 0.0, 1.0)
        raw_signals['HIST_VOL_SQUEEZE'] = pd.Series(_numba_rolling_accumulation(bbw_tight.astype(np.float32), 21), index=df.index, dtype=np.float32)
        threshold_map = {'net_amount_rate_D': (0.01, 0.005), 'winner_rate_D': (0.01, 0.005), 'SMART_MONEY_HM_NET_BUY_D': (10, 10), 'VPA_EFFICIENCY_D': (0.01, 0.01), 'BBW_21_2.0_D': (0.001, 0.0001), 'THEME_HOTNESS_SCORE_D': (0.1, 0.1), 'chip_entropy_D': (0.01, 0.001), 'volume_D': (100, 10), 'pct_change_D': (0.0001, 0.0001), 'close_D': (0.1, 0.01), 'market_sentiment_score_D': (0.1, 0.1), 'industry_strength_rank_D': (0.001, 0.001), 'turnover_rate_f_D': (0.01, 0.01), 'trade_count_D': (10, 5)}
        fib_wins = [3, 5, 8, 13, 21]
        for col, (abs_th, chg_th) in threshold_map.items():
            base_vals = raw_signals[col].values
            for win in fib_wins:
                s, a, j = _numba_robust_dynamics(base_vals, win=win, abs_threshold=abs_th, change_threshold=chg_th)
                raw_signals[f"SLOPE_{win}_{col}"] = pd.Series(s, index=df.index, dtype=np.float32)
                raw_signals[f"ACCEL_{win}_{col}"] = pd.Series(a, index=df.index, dtype=np.float32)
                raw_signals[f"JERK_{win}_{col}"] = pd.Series(j, index=df.index, dtype=np.float32)
        return raw_signals

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V74.0 · 全局动力学总线：集成全息探针，实现从 HAB 原料到最终分值的全链路可视化，清除空行"""
        method_name = "calculate_price_volume_dynamics"
        df_index = df.index
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        if not self._validate_all_required_signals(df, {}, {}, method_name, is_debug, probe_ts): return pd.Series(0.0, index=df_index, dtype=np.float32)
        raw = self._get_raw_signals(df, method_name)
        # 1. 核心计算链路 (保持 float32 向量化)
        scores = {}
        scores['physical'] = self._calculate_power_transfer_raw_score(df_index, raw, method_name)
        scores['geo'] = pd.Series(_numba_power_activation(raw['GEOM_ARC_CURVATURE_D'].values, gain=2.0), index=df_index) * (1.0 - raw['GEOM_REG_R2_D']) * (raw['STATE_ROUNDING_BOTTOM_D'].astype(np.float32) * 1.2 + 0.5)
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
        # 2. 调节器链路
        scores['risk'] = self._calculate_premium_reversal_risk(raw, df_index, method_name)
        scores['decay'] = self._calculate_intraday_decay_model(raw, df_index, method_name)
        scores['sector_mod'] = self._calculate_sector_resonance_modifier(raw, df_index, method_name)
        scores['vol_gamma'] = self._calculate_volatility_clustering_adjustment(raw, df_index, method_name)
        scores['sector_decay'] = self._calculate_sector_overflow_decay(raw, df_index, method_name)
        scores['access'] = self._calculate_entry_accessibility_score(raw, df_index, method_name)
        # 3. 最终合成
        unadjusted = scores['physical'] * 0.50 + scores['geo'] * 0.25 + scores['vwap'] * 0.25
        final_vals = unadjusted * scores['inertia'] * scores['perm'] * scores['entropy'] * scores['fractal'] * \
                     scores['hmm'] * scores['chip'] * scores['micro'] * scores['reflex'] * scores['wyckoff'] * \
                     scores['risk'] * scores['decay'] * scores['sector_mod'] * scores['vol_gamma'] * \
                     scores['sector_decay'] * scores['access']
        final_score = final_vals.clip(-3.5, 6.0).astype(np.float32)
        # 4. 触发全息探针
        if is_debug and probe_ts in df_index:
            self._print_full_chain_probe(probe_ts, raw, scores, final_score.loc[probe_ts])
        return final_score

    def _calculate_power_transfer_raw_score(self, df_index: pd.Index, raw: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V74.1 · 物理动力引擎：移除分散探针，保留 V73 HAB 逻辑"""
        # _setup_debug_info 仅保留用于兼容，不再用于打印
        self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        score_calc = CalculatePowerTransferRawScore(False, None) # 禁用内部调试
        vol_adj = raw['BBW_21_2.0_D'].values.astype(np.float64)
        roll_tc = pd.Series(raw['trade_count_D'].values, index=df_index).rolling(21).mean().replace(0, 1).values
        conf = (raw['trade_count_D'].values / (roll_tc + 1e-9)).astype(np.float64)
        j_c = _numba_adaptive_denoise_dynamics(raw['JERK_3_net_amount_rate_D'].values.astype(np.float64), vol_adj, conf)
        a_c = _numba_adaptive_denoise_dynamics(raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float64), vol_adj, conf)
        act_imp = pd.Series(_numba_power_activation((j_c * 0.45 + a_c * 0.55), gain=1.8), index=df_index, dtype=np.float32)
        norm_imp = score_calc._calculate_dynamic_impulse_norm(act_imp, raw, df_index, method_name).values
        comp_imp = score_calc._calculate_limit_price_compensation(pd.Series(norm_imp, index=df_index), raw, df_index, method_name).values
        auc_p = score_calc._calculate_auction_prediction(raw, df_index, method_name).values
        _, f_slopes = _numba_fast_rolling_dynamics(raw['net_amount_rate_D'].values.astype(np.float64), np.array([3, 5, 8, 13, 21], dtype=np.int64))
        mcv = np.dot(np.array([0.35, 0.25, 0.20, 0.10, 0.10], dtype=np.float32), f_slopes.astype(np.float32))
        accum_m = raw['ACCUM_21_SMART_MONEY'].values
        m_mass = np.clip(np.log1p(np.abs(accum_m)) / 10.0, 0.8, 1.3).astype(np.float32)
        phy_score = ((comp_imp * 2.0 * 0.30 + auc_p * 0.35 + mcv * 0.35) * m_mass).astype(np.float32)
        return pd.Series(phy_score, index=df_index, dtype=np.float32)

    def _calculate_premium_reversal_risk(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 溢价风险：移除分散探针，纯净计算"""
        turnover, extreme = raw['turnover_rate_f_D'].values, raw['STATE_EMOTIONAL_EXTREME_D'].values.astype(np.float32)
        daily_ratio, accum_ratio = raw['closing_flow_ratio_D'].values, raw['ACCUM_13_CLOSING_FLOW'].values
        exhaustion = np.clip(turnover / 15.0, 0.0, 1.5)
        hab_risk = np.where(accum_ratio > 3.5, 1.25, 1.0)
        reversal_pressure = daily_ratio * extreme * exhaustion * hab_risk
        risk_adjustment = pd.Series(1.0 - reversal_pressure * 0.4, index=df_index, dtype=np.float32).clip(0.5, 1.0)
        return risk_adjustment

    def _calculate_intraday_decay_model(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 日内衰减：移除分散探针，纯净计算"""
        stab, mean_stab = raw['TURNOVER_STABILITY_INDEX_D'].values, raw['MEAN_13_STABILITY'].values
        close, up, ratio = raw['close_D'].values, raw['up_limit_D'].values, raw['closing_flow_ratio_D'].values
        bad_mask = (close >= up * 0.999) & (ratio > 0.4) & (stab < 0.4)
        fragility = np.where(mean_stab < 0.5, 0.85, 1.0)
        winner = raw['winner_rate_D'].values
        repair = np.where((winner < 0.15) & (stab < 0.3), 1.5, 1.0).astype(np.float32)
        decay = (0.6 + stab * 0.4) * np.where(bad_mask, 0.6, 1.0).astype(np.float32) * repair * fragility
        return pd.Series(decay, index=df_index, dtype=np.float32).clip(0.2, 1.5)

    def _calculate_sector_resonance_modifier(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 板块共振：移除分散探针，纯净计算"""
        s_hot, a_hot = raw['SLOPE_5_THEME_HOTNESS_SCORE_D'].values, raw['ACCEL_5_THEME_HOTNESS_SCORE_D'].values
        accum_hot = raw['ACCUM_21_THEME_HOTNESS'].values
        mainline_bonus = np.clip(accum_hot / 1000.0, 1.0, 1.35)
        impulse = (s_hot * 0.6 + a_hot * 0.4)
        persistence = np.where((raw['industry_rank_accel_D'].values > 0) & (raw['flow_consistency_D'].values > 0.65), 1.2, 0.8).astype(np.float32)
        mod = (1.0 + _numba_power_activation(impulse.astype(np.float64), gain=0.5).astype(np.float32)) * persistence * mainline_bonus
        return pd.Series(mod, index=df_index, dtype=np.float32).clip(0.6, 2.0)

    def _calculate_volatility_clustering_adjustment(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 波动率伽马：移除分散探针，纯净计算"""
        bbw, squeeze_score = raw['BBW_21_2.0_D'].values, raw['HIST_VOL_SQUEEZE'].values
        s_bbw, a_bbw, j_bbw = raw['SLOPE_5_BBW_21_2.0_D'].values, raw['ACCEL_5_BBW_21_2.0_D'].values, raw['JERK_3_BBW_21_2.0_D'].values
        bbw_ma = pd.Series(bbw, index=df_index).rolling(21).mean().fillna(pd.Series(bbw, index=df_index)).values
        vcp_ignite = np.where(squeeze_score > 15.0, 1.4, 1.0)
        exp = (bbw < bbw_ma * 1.2) & (a_bbw > 0) & (j_bbw > 0.01)
        trap = (bbw > bbw_ma * 1.5) & ((s_bbw < 0) | (a_bbw < 0))
        p_jerk = raw['JERK_3_close_D'].values
        adj = np.ones(len(df_index), dtype=np.float32)
        adj = np.where(exp & (p_jerk > 0), 1.5 * vcp_ignite, adj)
        adj = np.where(exp & (p_jerk < 0), 0.5, adj)
        adj = np.where(trap, 0.8, adj)
        return pd.Series(adj, index=df_index, dtype=np.float32).clip(0.3, 2.2)

    def _calculate_sector_overflow_decay(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 熵增雪崩：移除分散探针，纯净计算"""
        hot, accum_hot = raw['THEME_HOTNESS_SCORE_D'].values, raw['ACCUM_21_THEME_HOTNESS'].values
        fd_all = _numba_fractal_dimension(np.expand_dims(hot, axis=0).astype(np.float64), window=13)
        fd_vals = fd_all[0].astype(np.float32)
        slope = pd.Series(fd_vals, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        accel = pd.Series(slope, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        hot_threshold = np.where(accum_hot > 1500.0, 70.0, 85.0)
        avalanche = (hot > hot_threshold) & (slope > 0) & (accel > 0)
        base = (1.5 / (fd_vals + 1e-9)).clip(0.6, 1.1)
        res = np.where(avalanche, base * 0.6, base).astype(np.float32)
        return pd.Series(res, index=df_index, dtype=np.float32)

    def _calculate_hmm_regime_confirmation(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · HMM 体制确认：移除分散探针，保留 HAB 偏置"""
        l_n = raw['SMART_MONEY_HM_NET_BUY_D']
        r_m, r_s = l_n.rolling(21).mean().fillna(0), l_n.rolling(21).std().fillna(1e-9)
        f_n = ((l_n - r_m) / r_s).values.astype(np.float32)
        v_n = ((raw['volume_D'] / raw['volume_D'].rolling(21).mean().replace(0, 1)) - 1.0).values.astype(np.float32)
        p_n = (raw['pct_change_D'] * 10.0).values.astype(np.float32)
        v_d = ((raw['close_D'] - raw['VWAP_D']) / (raw['close_D'] * raw['BBW_21_2.0_D'].clip(lower=0.01))).values.astype(np.float32)
        m_p = _numba_hmm_regime_probability(f_n, v_n, p_n, v_d)
        m_p_s = pd.Series(m_p, index=df_index, dtype=np.float32)
        acc_m = raw['ACCUM_21_SMART_MONEY'].values
        regime_bias = np.where(acc_m > 0, 1.15, 0.9).astype(np.float32)
        p_a, p_j = m_p_s.diff(3).diff(3).fillna(0).values, m_p_s.diff(1).diff(1).diff(1).fillna(0).values
        b_f = np.where(m_p > 0.5, 1.0 + (m_p - 0.5), 0.8 + m_p * 0.4)
        d_b = 1.0 + np.where((m_p > 0.6) & (p_a > 0), 0.2, 0.0) + np.where((m_p > 0.4) & (p_j > 0.1), 0.3, 0.0)
        return pd.Series(b_f * d_b * regime_bias, index=df_index, dtype=np.float32)

    def _calculate_fractal_efficiency_resonance(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 分形效率：移除分散探针，纯净计算"""
        c_vals = raw['close_D'].values.astype(np.float64)
        v_vals = raw['volume_D'].values.astype(np.float64)
        input_arrays = np.vstack((c_vals, v_vals))
        f_all = _numba_fractal_dimension(input_arrays, window=21)
        h_p, h_v = (2.0 - f_all[0]).astype(np.float32), (2.0 - f_all[1]).astype(np.float32)
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
        """V74.1 · 筹码锁定：移除分散探针，纯净计算"""
        win, turn = raw['winner_rate_D'].values, raw['turnover_rate_f_D'].values
        t_norm = np.clip(turn / 5.0, 0.0, 1.0)
        s_lock = win * (1.0 - t_norm)
        win_ma13 = pd.Series(win, index=df_index).rolling(13).mean().values
        hist_lock_qual = np.where(win_ma13 > 0.8, 1.25, 1.0).astype(np.float32)
        a_w, a_t, j_w = raw['ACCEL_5_winner_rate_D'].values, raw['ACCEL_5_turnover_rate_f_D'].values, raw['JERK_3_winner_rate_D'].values
        k_bonus = 1.0 + np.where((a_w > 0) & (a_t < 0), 0.3, 0.0) + np.where(j_w > 0.1, 0.2, 0.0)
        c_s, p_s = raw['cost_50pct_D'].values, raw['close_D'].values
        c_50 = np.where(c_s == 0, p_s, c_s)
        b_c = (p_s > c_50).astype(np.float32)
        final_eff = pd.Series(s_lock * k_bonus * hist_lock_qual * (0.8 + b_c * 0.2), index=df_index, dtype=np.float32)
        return final_eff

    def _calculate_microstructure_attack_vector(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 微观矢量：移除分散探针，保留 HAB 护盾逻辑"""
        e_n = (raw['buy_elg_amount_D'].values - raw['sell_elg_amount_D'].values).astype(np.float32)
        l_n = (raw['buy_lg_amount_D'].values - raw['sell_lg_amount_D'].values).astype(np.float32)
        t_a = (raw['amount_D'].values + 1e-9).astype(np.float32)
        d_sm = (e_n + l_n).astype(np.float32)
        a13, a21 = raw['ACCUM_13_SMART_MONEY'].values.astype(np.float32), raw['ACCUM_21_SMART_MONEY'].values.astype(np.float32)
        e_r, l_r = e_n / t_a, l_n / t_a
        s_s = np.where((e_r > 0) & (l_r > 0), 1.0 + (e_r + l_r) * 2.0, np.where((e_r * l_r) < 0, 0.6, 0.8)).astype(np.float32)
        s_j = raw['JERK_3_SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float32)
        s_a = (raw['ACCEL_5_volume_D'].values - raw['ACCEL_5_trade_count_D'].values).astype(np.float32)
        b_v = (0.5 + np.tanh((e_r + l_r) * 10.0) * 0.5) * (1.0 + np.where(s_j > 0.5, 0.4, 0.0) + np.where(s_a > 0.05, 0.3, 0.0))
        h_s = np.where((d_sm < 0) & (a13 > np.abs(d_sm) * 10.0), 1.2, 1.0)
        h_s = np.where((d_sm < 0) & (a21 > np.abs(d_sm) * 20.0), h_s * 1.15, h_s).astype(np.float32)
        f_v = pd.Series(b_v * s_s * h_s, index=df_index, dtype=np.float32).clip(0, 2.0)
        return f_v

    def _calculate_vpa_elasticity_reflexivity(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 反身性：移除分散探针，纯净计算"""
        pp, pv_s = np.abs(raw['pct_change_D'].values), pd.Series(raw['volume_D'].values, index=df_index)
        pv = np.abs(pv_s.pct_change().fillna(0).values)
        base_e = pp / (pv + 0.1)
        e_s = pd.Series(base_e, index=df_index).diff(3).values / 3.0
        acc_m, acc_v = raw['ACCUM_21_SMART_MONEY'].values, raw['ACCUM_21_VOLUME'].values
        e_density = np.clip((np.abs(acc_m) / (acc_v + 1.0)) * 100.0, 0.9, 1.3).astype(np.float32)
        jp, jv, ap, av = raw['JERK_3_close_D'].values, raw['JERK_3_volume_D'].values, raw['ACCEL_5_close_D'].values, raw['ACCEL_5_volume_D'].values
        score = np.clip(np.tanh(base_e), 0.5, 1.5)
        final = score * np.where(e_s > 0, 1.2, 1.0) * e_density
        final = np.where((jp > 0.05) & (jv > 0.1), final * 1.5, np.where((ap > 0) & (av > 0), final * 1.2, final))
        return pd.Series(final, index=df_index, dtype=np.float32).clip(0.5, 2.0)

    def _calculate_wyckoff_breakout_quality(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 威科夫突破：移除分散探针，纯净计算"""
        high, low = raw['high_D'].values, raw['low_D'].values
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
        """V74.1 · 趋势运动学：移除分散探针，纯净计算"""
        c_vals = raw['close_D'].values
        ma5 = raw['close_D'].rolling(5).mean().values
        ma21 = raw['close_D'].rolling(21).mean().values
        ma55 = raw['close_D'].rolling(55).mean().values
        s_vals, a_vals, j_vals = raw['SLOPE_5_close_D'].values, raw['ACCEL_5_close_D'].values, raw['JERK_3_close_D'].values
        r2_vals = raw['GEOM_REG_R2_D'].values
        alignment = np.where((ma5 > ma21) & (ma21 > ma55), 1.0, 0.8).astype(np.float32)
        kinematic = 1.0 + np.where((s_vals > 0) & (a_vals > 0), 0.3, 0.0) + np.where(j_vals > 0.1, 0.2, 0.0) - np.where((s_vals > 0) & (a_vals < 0), 0.2, 0.0)
        final_inertia = pd.Series(alignment * kinematic * (0.6 + np.clip(r2_vals, 0.0, 1.0) * 0.4), index=df_index, dtype=np.float32).clip(0.6, 1.6)
        return final_inertia

    def _calculate_market_permeability_index(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 市场渗透率：移除分散探针，纯净计算"""
        sent, accel = raw['market_sentiment_score_D'].values, raw['ACCEL_5_market_sentiment_score_D'].values
        rank, jerk = raw['industry_strength_rank_D'].values, raw['JERK_3_industry_strength_rank_D'].values
        perm = np.where((sent < 20) & (accel > 0), 1.3, np.where((sent > 80) & (accel > 0), 0.7, 1.0)).astype(np.float32)
        bonus = np.where(jerk < -2.0, 1.3, np.where(rank < 0.1, 1.1, 0.9)).astype(np.float32)
        final_ctx = pd.Series(perm * bonus, index=df_index, dtype=np.float32).clip(0.6, 1.8)
        return final_ctx

    def _calculate_entry_accessibility_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · 入场可获得性：移除分散探针，纯净计算"""
        tf = raw['turnover_rate_f_D'].values
        liq = np.clip(np.sqrt(tf) / 1.2247, 0.1, 1.1)
        up_limit = raw['up_limit_D'].values
        close = raw['close_D'].values
        intensity = raw['closing_flow_intensity_D'].values
        premium = raw['T1_PREMIUM_EXPECTATION_D'].values
        sealing = np.clip(intensity * premium, 0.0, 1.0)
        limit_access = np.where(close >= up_limit * 0.999, 0.4 * (1.0 - sealing), 1.0).astype(np.float32)
        return pd.Series(limit_access * liq, index=df_index, dtype=np.float32).clip(0.0, 1.0)

    def _calculate_entropic_ordering_bonus(self, df: pd.DataFrame, df_index: pd.Index, method_name: str) -> pd.Series:
        """V3.21.0 · 熵减有序度奖励模型：采用非线性饱和映射替代统计归一化，精准度量系统做功 """
        # 1. 获取基础价格熵 (军械库真实字段)
        # PRICE_ENTROPY_D: 通常在 0~1 之间，1代表最大混乱，0代表最大有序
        # 填充 NaN 为 1.0 (最大混乱)
        price_entropy = df['PRICE_ENTROPY_D'].fillna(1.0)
        # 2. 计算动态熵减流 (Entropy Reduction Flow)
        # 逻辑：计算每日熵的变化量。
        # diff < 0 代表混乱度降低（有序度增加），这是我们需要捕捉的“负熵流”。
        entropy_delta = price_entropy.diff()
        # 只保留熵减少的部分 (取绝对值)，熵增加的部分置为 0
        valid_reduction = entropy_delta.clip(upper=0).abs()
        # 3. 计算 13日 累积熵减做功 (Accumulated Entropic Work)
        # 这代表了过去一个短周期内，市场为恢复有序所做的“总功”
        acc_entropy_reduction = valid_reduction.rolling(window=13).sum().fillna(0)
        # 4. 专用物理归一化：非线性饱和映射 (Non-linear Saturation Mapping)
        # 设定灵敏度系数 lambda = 2.5
        # 当累积熵减达到 0.4 时，tanh(0.4 * 2.5) = tanh(1.0) ≈ 0.76 (显著)
        # 当累积熵减达到 0.8 时，tanh(0.8 * 2.5) = tanh(2.0) ≈ 0.96 (极强)
        sensitivity_lambda = 2.5
        saturation_score = np.tanh(acc_entropy_reduction * sensitivity_lambda)
        # 5. 当前绝对有序度 (Current Absolute Order)
        # 仅仅有熵减过程不够，当前状态必须也处于相对有序区间
        current_absolute_order = (1 - price_entropy).clip(0, 1)
        # 6. 最终融合
        # 熵减过程分(70%) + 当前绝对分(30%)
        # 这种设计强调了“变化的过程”比“当前的状态”更具预测性
        final_entropy_score = pd.Series(saturation_score * 0.7 + current_absolute_order * 0.3, index=df_index)
        # 7. 探针输出
        print(f"  [探针-EntropyDynamics] 13日累积熵减均值: {acc_entropy_reduction.mean():.4f} | 物理饱和分值均值: {saturation_score.mean():.4f}")
        print(f"  [探针-EntropyDynamics] 当前绝对有序均值: {current_absolute_order.mean():.4f} | 最终奖励分值: {final_entropy_score.mean():.4f}")
        return final_entropy_score.astype(np.float32)

    def _calculate_vwap_propulsion_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V74.1 · VWAP 推进力：移除分散探针，纯净计算"""
        close = raw['close_D'].values
        vwap = raw['VWAP_D'].values
        vwap_slope = pd.Series(vwap, index=df_index).diff(3).values / 3.0
        bias = (close - vwap) / (vwap + 1e-9)
        propulsion = np.where(vwap_slope > 0, 1.0, 0.0) + np.where((bias > 0) & (bias < 0.05), 0.2, 0.0) + np.clip(np.tanh(vwap_slope * 10.0), 0.0, 0.3)
        final_score = pd.Series(propulsion, index=df_index, dtype=np.float32).clip(0, 1.5)
        return final_score

    def _print_full_chain_probe(self, probe_ts: pd.Timestamp, raw: Dict[str, pd.Series], scores: Dict[str, pd.Series], final_score: float):
        """V74.0 · 全息探针诊断总线：构建从 HAB 原料到策略结果的全链路透视，清除空行"""
        if probe_ts not in raw['close_D'].index: return
        idx = raw['close_D'].index.get_loc(probe_ts)
        p_str = probe_ts.strftime('%Y-%m-%d')
        print(f"\n{'='*30} [全息动力学探针 V74.0 @ {p_str}] {'='*30}")
        # 1. 基础物理场 (Base Field)
        c, v, vwap = raw['close_D'].values[idx], raw['volume_D'].values[idx], raw['VWAP_D'].values[idx]
        print(f"【基础物理】收盘: {c:.2f} | VWAP: {vwap:.2f} | 量能: {v/10000:.1f}万 | 换手: {raw['turnover_rate_f_D'].values[idx]:.2f}%")
        # 2. HAB 历史累积缓冲 (Historical Accumulation Buffer)
        print(f"【HAB 缓冲】")
        print(f"  ├─ 资金底仓 (13d/21d): {raw['ACCUM_13_SMART_MONEY'].values[idx]:.1f} / {raw['ACCUM_21_SMART_MONEY'].values[idx]:.1f}")
        print(f"  ├─ 尾盘压力 (13d): {raw['ACCUM_13_CLOSING_FLOW'].values[idx]:.2f} (阈值>3.5高危)")
        print(f"  ├─ 板块厚度 (21d): {raw['ACCUM_21_THEME_HOTNESS'].values[idx]:.1f} (主线锚点>1000)")
        print(f"  └─ 波动紧致 (21d): {raw['HIST_VOL_SQUEEZE'].values[idx]:.1f} (VCP点火源)")
        # 3. 动力学特征 (Kinematics)
        print(f"【动力学特征】")
        print(f"  ├─ 价格 (S/A/J): {raw['SLOPE_5_close_D'].values[idx]:.4f} / {raw['ACCEL_5_close_D'].values[idx]:.4f} / {raw['JERK_3_close_D'].values[idx]:.4f}")
        print(f"  └─ 主力 (S/A/J): {raw['SLOPE_5_SMART_MONEY_HM_NET_BUY_D'].values[idx]:.1f} / {raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'].values[idx]:.1f} / {raw['JERK_3_SMART_MONEY_HM_NET_BUY_D'].values[idx]:.1f}")
        # 4. 子模块评分矩阵 (Sub-Module Matrix)
        print(f"【核心引擎评分】")
        print(f"  [物理] 强度: {scores['physical'].values[idx]:.4f} | VWAP: {scores['vwap'].values[idx]:.4f} | 几何: {scores['geo'].values[idx]:.4f}")
        print(f"  [结构] 筹码: {scores['chip'].values[idx]:.4f} | 熵序: {scores['entropy'].values[idx]:.4f} | 威科夫: {scores['wyckoff'].values[idx]:.4f}")
        print(f"  [环境] 渗透: {scores['perm'].values[idx]:.4f} | 惯性: {scores['inertia'].values[idx]:.4f} | HMM: {scores['hmm'].values[idx]:.4f}")
        print(f"  [微观] 攻击: {scores['micro'].values[idx]:.4f} | 反身: {scores['reflex'].values[idx]:.4f} | 分形: {scores['fractal'].values[idx]:.4f}")
        # 5. 风险阀与调节器 (Valves & Modifiers)
        print(f"【安全阀调节】")
        print(f"  ├─ 风险回吐: {scores['risk'].values[idx]:.2f} x 日内衰减: {scores['decay'].values[idx]:.2f}")
        print(f"  └─ 板块溢出: {scores['sector_decay'].values[idx]:.2f} x 波动伽马: {scores['vol_gamma'].values[idx]:.2f}")
        # 6. 最终合成 (Final Synthesis)
        print(f"{'-'*75}")
        print(f" >>> PROCESS_META_POWER_TRANSFER 最终得分: {final_score:.4f}")
        print(f"{'='*75}\n")

