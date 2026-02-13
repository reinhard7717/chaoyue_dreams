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

@jit(nopython=True)
def _numba_rolling_rank(arr, window):
    """V1.0 · Numba 滚动排名算子：高性能计算滑动窗口内的百分位排名，替代 pandas apply lambda"""
    n = len(arr)
    out = np.empty(n, dtype=np.float32)
    out[:] = np.nan  # 初始化为 NaN，与 pandas 行为一致
    # 简单的 O(N*W) 实现，对于小窗口 (W=21) 比维护跳表或排序树更高效且对缓存友好
    for i in range(window - 1, n):
        count = 0.0
        target = arr[i]
        # 遍历窗口统计小于当前值的数量
        for j in range(i - window + 1, i + 1):
            if arr[j] < target:
                count += 1.0
        out[i] = count / (window - 1)
    return out

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
        """V86.2 · 原料加载层：严格 float32 运算，优化市值与 VWAP 计算性能，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        raw_signals = {}
        base_cols = ['close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'pct_change_D', 'net_amount_rate_D', 'trade_count_D', 'turnover_rate_f_D']
        has_mv_col = 'circ_mv_D' in df.columns
        if has_mv_col: base_cols.append('circ_mv_D')
        struct_cols = ['winner_rate_D', 'chip_concentration_ratio_D', 'chip_entropy_D', 'cost_50pct_D', 'absorption_energy_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'price_percentile_position_D']
        tech_cols = ['SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D', 'pressure_release_index_D', 'up_limit_D', 'down_limit_D', 'closing_flow_ratio_D', 'TURNOVER_STABILITY_INDEX_D', 'STATE_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 'industry_strength_rank_D', 'industry_rank_accel_D', 'STATE_ROUNDING_BOTTOM_D', 'STATE_GOLDEN_PIT_D', 'STATE_TRENDING_STAGE_D', 'THEME_HOTNESS_SCORE_D', 'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'market_sentiment_score_D']
        for col in base_cols + struct_cols + tech_cols:
            if col not in df.columns: raise KeyError(f"CRITICAL: 军械库缺失关键列 {col}")
            raw_signals[col] = df[col].ffill().fillna(0.0).astype(np.float32)
        # --- 市值自适应逻辑 ---
        mv_source = "RAW"
        if has_mv_col:
            raw_signals['circ_mv_D'] = raw_signals['circ_mv_D'] * np.float32(10000.0)
        valid_mv = raw_signals.get('circ_mv_D', pd.Series([0], dtype=np.float32))
        valid_mv = valid_mv[valid_mv > 1e7]
        if len(valid_mv) > 0:
            avg_mv = valid_mv.median()
        else:
            amt = raw_signals['amount_D']
            turn = raw_signals['turnover_rate_f_D']
            est_mv = (amt / (turn + 1e-9) * np.float32(100.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            avg_mv = est_mv[est_mv > 1e7].median()
            if np.isnan(avg_mv): avg_mv = 0.0
            mv_source = "ESTIMATED"
        w_s, w_l = 13, 21
        mv_tag = "MID"
        if avg_mv > 0:
            if avg_mv < 50e8:
                w_s, w_l = 8, 13
                mv_tag = "SMALL"
            elif avg_mv > 500e8:
                w_s, w_l = 21, 34
                mv_tag = "LARGE"
        raw_signals['META_HAB_WINDOWS'] = pd.Series([w_s, w_l], index=df.index[:2], dtype=np.float32)
        # --- 核心指标预计算 (优化：显式 float32) ---
        raw_vwap = (raw_signals['amount_D'] / (raw_signals['volume_D'] + 1e-9)).fillna(raw_signals['close_D'])
        r_c, r_v = raw_signals['close_D'].values[-60:], raw_vwap.values[-60:]
        v_m, s_f = r_v > 0, np.float32(1.0)
        if np.any(v_m):
            m_r = np.median(r_v[v_m] / r_c[v_m])
            if m_r < 0.5 or m_r > 2.0: s_f = np.float32(10.0 ** (-np.round(np.log10(m_r))))
        raw_signals['VWAP_D'] = (raw_vwap * s_f).astype(np.float32)
        if raw_signals['winner_rate_D'].max() > 1.1: raw_signals['winner_rate_D'] *= np.float32(0.01)
        sm_v = raw_signals['SMART_MONEY_HM_NET_BUY_D'].values
        if np.sum(np.abs(sm_v)) < 1e-5:
            sm_v = (raw_signals['buy_elg_amount_D'] - raw_signals['sell_elg_amount_D'] + 
                    raw_signals['buy_lg_amount_D'] - raw_signals['sell_lg_amount_D']).values
        # 基础指标 HAB
        raw_signals['ACCUM_13_SMART_MONEY'] = pd.Series(_numba_rolling_accumulation(sm_v.astype(np.float32), w_s), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_SMART_MONEY'] = pd.Series(_numba_rolling_accumulation(sm_v.astype(np.float32), w_l), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_VOLUME'] = pd.Series(_numba_rolling_accumulation(raw_signals['volume_D'].values, w_l), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_13_CLOSING_FLOW'] = pd.Series(_numba_rolling_accumulation(raw_signals['closing_flow_ratio_D'].values, w_s), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_21_THEME_HOTNESS'] = pd.Series(_numba_rolling_accumulation(raw_signals['THEME_HOTNESS_SCORE_D'].values, w_l), index=df.index, dtype=np.float32)
        raw_signals['MEAN_13_STABILITY'] = pd.Series(_numba_rolling_accumulation(raw_signals['TURNOVER_STABILITY_INDEX_D'].values, w_s), index=df.index, dtype=np.float32) / float(w_s)
        raw_signals['HIST_VOL_SQUEEZE'] = pd.Series(_numba_rolling_accumulation(np.clip(1.0 - raw_signals['BBW_21_2.0_D'].values, 0.0, 1.0).astype(np.float32), w_l), index=df.index, dtype=np.float32)
        tr_proxy = (raw_signals['high_D'] - raw_signals['low_D']).values
        raw_signals['ACCUM_21_TR'] = pd.Series(_numba_rolling_accumulation(tr_proxy, w_l), index=df.index, dtype=np.float32)
        raw_signals['ACCUM_13_SENTIMENT'] = pd.Series(_numba_rolling_accumulation(raw_signals['market_sentiment_score_D'].values, w_s), index=df.index, dtype=np.float32)
        pp, pv = np.abs(raw_signals['pct_change_D'].values), np.abs(pd.Series(raw_signals['volume_D'].values, index=df.index).pct_change().fillna(0).values)
        raw_signals['ACCUM_13_ELASTICITY'] = pd.Series(_numba_rolling_accumulation((pp / (pv + 0.1)).astype(np.float32), w_s), index=df.index, dtype=np.float32)
        ent_red = np.where(pd.Series(raw_signals['chip_entropy_D'].values, index=df.index).diff().fillna(0).values < 0, 1.0, 0.0).astype(np.float32)
        raw_signals['ACCUM_21_ENTROPY_STABILITY'] = pd.Series(_numba_rolling_accumulation(ent_red, w_l), index=df.index, dtype=np.float32)
        pos_slope = np.where(raw_signals['close_D'].diff(5) > 0, 1.0, 0.0).astype(np.float32)
        raw_signals['ACCUM_21_POS_SLOPE'] = pd.Series(_numba_rolling_accumulation(pos_slope, w_l), index=df.index, dtype=np.float32)
        ab_vwap = np.where(raw_signals['close_D'].values > raw_signals['VWAP_D'].values, 1.0, 0.0).astype(np.float32)
        raw_signals['ACCUM_21_ABOVE_VWAP'] = pd.Series(_numba_rolling_accumulation(ab_vwap, w_l), index=df.index, dtype=np.float32)
        hi_win = np.where(raw_signals['winner_rate_D'].values > 0.8, 1.0, 0.0).astype(np.float32)
        raw_signals['ACCUM_13_HIGH_WINNER'] = pd.Series(_numba_rolling_accumulation(hi_win, w_s), index=df.index, dtype=np.float32)
        liq_lock = np.where((raw_signals['turnover_rate_f_D'].values < 3.0) & (raw_signals['pct_change_D'].values > 0), 1.0, 0.0).astype(np.float32)
        raw_signals['ACCUM_5_LIQUIDITY_LOCK'] = pd.Series(_numba_rolling_accumulation(liq_lock, 5), index=df.index, dtype=np.float32)
        hi_rank = np.where(raw_signals['industry_strength_rank_D'].values <= 20, 1.0, 0.0).astype(np.float32)
        raw_signals['ACCUM_21_HIGH_RANK'] = pd.Series(_numba_rolling_accumulation(hi_rank, w_l), index=df.index, dtype=np.float32)
        # 动力学衍生
        threshold_map = {'net_amount_rate_D': (0.01, 0.005), 'winner_rate_D': (0.01, 0.005), 'SMART_MONEY_HM_NET_BUY_D': (10, 10), 'VPA_EFFICIENCY_D': (0.01, 0.01), 'BBW_21_2.0_D': (0.001, 0.0001), 'THEME_HOTNESS_SCORE_D': (0.1, 0.1), 'chip_entropy_D': (0.01, 0.001), 'volume_D': (100, 10), 'pct_change_D': (0.0001, 0.0001), 'close_D': (0.1, 0.01), 'market_sentiment_score_D': (0.1, 0.1), 'industry_strength_rank_D': (0.001, 0.001), 'turnover_rate_f_D': (0.01, 0.01), 'trade_count_D': (10, 5), 'VWAP_D': (0.1, 0.01), 'closing_flow_ratio_D': (0.01, 0.01)}
        fib_wins = [3, 5, 8, 13, 21]
        for col, (abs_th, chg_th) in threshold_map.items():
            base_vals = raw_signals[col].values
            for win in fib_wins:
                s, a, j = _numba_robust_dynamics(base_vals, win=win, abs_threshold=abs_th, change_threshold=chg_th)
                raw_signals[f"SLOPE_{win}_{col}"] = pd.Series(s, index=df.index, dtype=np.float32)
                raw_signals[f"ACCEL_{win}_{col}"] = pd.Series(a, index=df.index, dtype=np.float32)
                raw_signals[f"JERK_{win}_{col}"] = pd.Series(j, index=df.index, dtype=np.float32)
        # if is_debug and probe_ts in df.index:
        #     print(f"\n[原料 V86.2 修正 HAB 探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    原始市值包含0值: {'是' if (raw_signals['circ_mv_D']==0).any() else '否'}")
        #     print(f"    代表性市值: {avg_mv/1e8:.1f}亿 ({mv_source}), 类型: {mv_tag}")
        #     print(f"    窗口配置: 短 {w_s}d / 长 {w_l}d")
        return raw_signals

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V77.1 · 全局动力学总线：向量化合成计算优化，移除 Series 算术开销，直接操作 float32 数组，清除空行"""
        method_name = "calculate_price_volume_dynamics"
        print(f"\n ====== [ CalculatePriceVolumeDynamics 渗透率自适应探针 V82.1 ] ======")
        df_index = df.index
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        if not self._validate_all_required_signals(df, {}, {}, method_name, is_debug, probe_ts): return pd.Series(0.0, index=df_index, dtype=np.float32)
        raw = self._get_raw_signals(df, method_name)
        # 1. 核心计算链路 (保持 float32 向量化)
        scores = {}
        scores['physical'] = self._calculate_power_transfer_raw_score(df_index, raw, method_name)
        # 几何评分：直接使用 numpy 运算
        s_geo = (_numba_power_activation(raw['GEOM_ARC_CURVATURE_D'].values, gain=2.0) * (1.0 - raw['GEOM_REG_R2_D'].values) * (raw['STATE_ROUNDING_BOTTOM_D'].values.astype(np.float32) * 1.2 + 0.5))
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
        # 2. 调节器链路
        scores['risk'] = self._calculate_premium_reversal_risk(raw, df_index, method_name)
        scores['decay'] = self._calculate_intraday_decay_model(raw, df_index, method_name)
        scores['sector_mod'] = self._calculate_sector_resonance_modifier(raw, df_index, method_name)
        scores['vol_gamma'] = self._calculate_volatility_clustering_adjustment(raw, df_index, method_name)
        scores['sector_decay'] = self._calculate_sector_overflow_decay(raw, df_index, method_name)
        scores['access'] = self._calculate_entry_accessibility_score(raw, df_index, method_name)
        # 3. 最终合成 (Numpy float32 向量化，性能优化点)
        # 提取所有 values，避免 pandas 索引对齐检查
        s_phy = scores['physical'].values
        s_vwap = scores['vwap'].values
        # 基础得分
        unadjusted = s_phy * 0.50 + s_geo * 0.25 + s_vwap * 0.25
        # 连乘合成
        final_vals = unadjusted * scores['inertia'].values * scores['perm'].values * scores['entropy'].values * scores['fractal'].values * \
                     scores['hmm'].values * scores['chip'].values * scores['micro'].values * scores['reflex'].values * scores['wyckoff'].values * \
                     scores['risk'].values * scores['decay'].values * scores['sector_mod'].values * scores['vol_gamma'].values * \
                     scores['sector_decay'].values * scores['access'].values
        # 封装结果
        final_score = pd.Series(final_vals, index=df_index, dtype=np.float32).clip(-3.5, 6.0)
        # 4. 执行 HAB 状态持久化
        # self._persist_hab_state(raw, df_index, method_name)
        # 5. 触发全息探针
        # if is_debug and probe_ts in df_index:
            # self._print_full_chain_probe(probe_ts, raw, scores, final_score.loc[probe_ts])
        print(f"\n ====== ======================== ======")
        return final_score

    def _persist_hab_state(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str):
        """V78.0 · HAB 状态持久化：扩展弹性密度与熵减稳定性指标固化，供全线调用，清除空行"""
        if len(df_index) == 0: return
        last_idx, last_ts = -1, df_index[-1]
        hab_snapshot = {"timestamp": last_ts.strftime('%Y-%m-%d'), "updated_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), "metrics": {}}
        # 自动提取所有 HAB 与历史稳定性特征
        target_prefixes = ['ACCUM_', 'HIST_', 'MEAN_']
        for key, series in raw.items():
            if any(p in key for p in target_prefixes):
                hab_snapshot["metrics"][key] = float(series.values[last_idx])
        self.latest_hab_state = hab_snapshot
        if hasattr(self.helper, 'update_shared_state'):
            try: self.helper.update_shared_state('HAB_LATEST', hab_snapshot)
            except Exception: pass
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        if is_debug and probe_ts == last_ts:
            print(f"\n[HAB 持久化探针 V78.0 @ {last_ts.strftime('%Y-%m-%d')}]")
            print(f"    新增指标固化 -> 13d弹性: {hab_snapshot['metrics'].get('ACCUM_13_ELASTICITY', 0.0):.4f}")
            print(f"    新增指标固化 -> 21d熵减稳态: {hab_snapshot['metrics'].get('ACCUM_21_ENTROPY_STABILITY', 0.0):.1f}")

    def _calculate_power_transfer_raw_score(self, df_index: pd.Index, raw: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V94.2 · 物理动力引擎：采用 Numba 快速滚动算子替代 Pandas Rolling，全链路 float32 计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 基础物理冲量 (优化：使用 Numba 计算 rolling mean)
        vol_adj = raw['BBW_21_2.0_D'].values.astype(np.float32)
        tc_values = raw['trade_count_D'].values.astype(np.float32)
        # _numba_fast_rolling_dynamics 返回 (n_windows, n_samples)
        tc_means, _ = _numba_fast_rolling_dynamics(tc_values, np.array([21], dtype=np.int64))
        roll_tc = tc_means[0]
        # 避免除零，替换 0 为 1.0 (保持 float32)
        roll_tc = np.where(np.abs(roll_tc) < 1e-9, 1.0, roll_tc).astype(np.float32)
        conf = (tc_values / roll_tc).astype(np.float32)
        j_c = _numba_adaptive_denoise_dynamics(raw['JERK_3_net_amount_rate_D'].values.astype(np.float32), vol_adj, conf)
        a_c = _numba_adaptive_denoise_dynamics(raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float32), vol_adj, conf)
        raw_impulse = (j_c * 0.45 + a_c * 0.55).astype(np.float32)
        # 2. 自适应归一化
        imp_series = pd.Series(raw_impulse, index=df_index)
        imp_mean = imp_series.rolling(21).mean().fillna(0).values.astype(np.float32)
        imp_std = imp_series.rolling(21).std().replace(0, 1e-9).values.astype(np.float32)
        z_score_imp = (raw_impulse - imp_mean) / imp_std
        norm_imp = np.tanh(z_score_imp * 0.8).astype(np.float32)
        # 3. 极值补偿
        price_pos = raw['price_percentile_position_D'].values.astype(np.float32)
        comp_factor = np.where(price_pos < 0.2, 1.2, np.where(price_pos > 0.8, 0.8, 1.0)).astype(np.float32)
        comp_imp = norm_imp * comp_factor
        # 4. MCV 动态门控
        _, f_slopes = _numba_fast_rolling_dynamics(raw['net_amount_rate_D'].values.astype(np.float32), np.array([3, 5, 8, 13, 21], dtype=np.int64))
        mcv = np.dot(np.array([0.35, 0.25, 0.20, 0.10, 0.10], dtype=np.float32), f_slopes.astype(np.float32))
        base_imp_w = 0.60
        base_mcv_w = 0.35
        reversal_mask = (comp_imp > 0) & (mcv < 0)
        mcv_weight = np.where(reversal_mask, 0.0, base_mcv_w).astype(np.float32)
        imp_weight = base_imp_w + (base_mcv_w - mcv_weight)
        # 5. 质量兜底
        accum_m = raw['ACCUM_21_SMART_MONEY'].values.astype(np.float32)
        accum_v = raw['ACCUM_21_VOLUME'].values.astype(np.float32)
        mass_m = np.clip(np.log1p(np.abs(accum_m)) / 10.0, 0.8, 1.3).astype(np.float32)
        mass_v = np.clip(np.log1p(accum_v) / 7.0, 0.8, 1.3).astype(np.float32)
        final_mass = np.where(mass_m <= 0.81, mass_v, mass_m).astype(np.float32)
        # 6. 最终计算
        phy_score = ((comp_imp * 2.0 * imp_weight + mcv * mcv_weight) * final_mass).astype(np.float32)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[物理引擎回补探针 V94.2 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    权重配置: 冲量={imp_weight[p_i]:.2f}, MCV={mcv_weight[p_i]:.2f}")
        #     print(f"    质量因子: {final_mass[p_i]:.2f} (原资金Mass: {mass_m[p_i]:.2f})")
        #     print(f"    >>> 物理合成总分: {phy_score[p_i]:.4f}")
        return pd.Series(phy_score, index=df_index, dtype=np.float32)

    def _calculate_premium_reversal_risk(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V82.1 · 溢价回吐风险：全链路 float32 优化，加速峰值查找计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        turnover = raw['turnover_rate_f_D'].values.astype(np.float32)
        extreme = raw['STATE_EMOTIONAL_EXTREME_D'].values.astype(np.float32)
        daily_ratio = raw['closing_flow_ratio_D'].values.astype(np.float32)
        # 1. 换手竭尽归一化
        max_turn = pd.Series(turnover, index=df_index).rolling(21).max().replace(0, 1.0).values.astype(np.float32)
        exhaustion_rate = np.clip(turnover / max_turn, 0.0, 1.5).astype(np.float32)
        # 2. HAB 尾盘压力
        accum_ratio = raw['ACCUM_13_CLOSING_FLOW'].values.astype(np.float32)
        mean_ratio = pd.Series(daily_ratio, index=df_index).rolling(13).mean().replace(0, 0.1).values.astype(np.float32)
        hab_risk_norm = np.where(accum_ratio > mean_ratio * 13 * 1.5, 1.25, 1.0).astype(np.float32)
        reversal_pressure = daily_ratio * extreme * exhaustion_rate * hab_risk_norm
        risk_adjustment = pd.Series(1.0 - reversal_pressure * 0.4, index=df_index, dtype=np.float32).clip(0.5, 1.0)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[溢价风险自适应探针 V82.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    换手竭尽率: {exhaustion_rate[p_i]:.2f} (当前/21日峰值)")
        return risk_adjustment

    def _calculate_intraday_decay_model(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V88.1 · 日内衰减：全链路 float32 优化，加速 Z-Score 与 Tanh 计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        stab = raw['TURNOVER_STABILITY_INDEX_D'].values.astype(np.float32)
        # 1. 稳定性归一化 (优化：显式 float32)
        stab_s = pd.Series(stab, index=df_index)
        stab_mean = stab_s.rolling(21).mean().values.astype(np.float32)
        stab_std = stab_s.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        stab_z = (stab - stab_mean) / stab_std
        # 2. 软衰减因子映射
        base_decay = np.clip(1.0 + np.tanh(stab_z) * 0.15, 0.7, 1.2).astype(np.float32)
        # 3. 结构性崩塌判定
        close, up, ratio = raw['close_D'].values.astype(np.float32), raw['up_limit_D'].values.astype(np.float32), raw['closing_flow_ratio_D'].values.astype(np.float32)
        mean_stab = raw['MEAN_13_STABILITY'].values.astype(np.float32)
        bad_board = (close >= up * 0.999) & (ratio > 0.4) & (stab < 0.4)
        fragility = np.where(mean_stab < 0.5, 0.9, 1.0).astype(np.float32)
        winner = raw['winner_rate_D'].values.astype(np.float32)
        repair = np.where((winner < 0.15) & (stab < 0.3), 1.5, 1.0).astype(np.float32)
        decay = base_decay * np.where(bad_board, 0.6, 1.0).astype(np.float32) * repair * fragility
        final_decay = pd.Series(decay, index=df_index, dtype=np.float32).clip(0.4, 1.5)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[日内衰减柔化探针 V88.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    稳定性Z分: {stab_z[p_i]:.2f}σ -> 基础衰减: {base_decay[p_i]:.2f}")
        #     print(f"    >>> 最终日内衰减: {final_decay.loc[probe_ts]:.4f}")
        return final_decay

    def _calculate_sector_resonance_modifier(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V82.1 · 板块共振：全链路 float32 优化，加速信噪比计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        s_hot = raw['SLOPE_5_THEME_HOTNESS_SCORE_D'].values.astype(np.float32)
        a_hot = raw['ACCEL_5_THEME_HOTNESS_SCORE_D'].values.astype(np.float32)
        # 1. 动量信噪比归一化 (优化：显式 float32)
        s_hot_std = pd.Series(s_hot, index=df_index).rolling(21).std().replace(0, 1e-9).values.astype(np.float32)
        s_hot_snr = np.clip(s_hot / s_hot_std, -3.0, 3.0)
        impulse_factor = (1.0 / (1.0 + np.exp(-s_hot_snr))).astype(np.float32)
        # 2. HAB 主线霸榜
        high_rank_days = raw['ACCUM_21_HIGH_RANK'].values.astype(np.float32)
        leadership_norm = np.clip(high_rank_days / 21.0, 0.0, 1.0).astype(np.float32)
        leadership_bonus = 1.0 + leadership_norm * 0.4
        persistence = np.where((raw['industry_rank_accel_D'].values > 0) & (raw['flow_consistency_D'].values > 0.65), 1.2, 0.8).astype(np.float32)
        rank_jerk = raw['JERK_3_industry_strength_rank_D'].values.astype(np.float32)
        rank_pulse = np.where(rank_jerk < -2.0, 1.3, 1.0).astype(np.float32)
        mod = (0.8 + impulse_factor * 0.4) * persistence * leadership_bonus * rank_pulse
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[板块信噪比探针 V82.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    热度斜率SNR: {s_hot_snr[p_i]:.2f}, 冲击系数: {impulse_factor[p_i]:.2f}")
        return pd.Series(mod, index=df_index, dtype=np.float32).clip(0.6, 2.2)

    def _calculate_volatility_clustering_adjustment(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V84.1 · 波动率伽马：全链路 float32 优化，加速 Z-Score 计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        bbw = raw['BBW_21_2.0_D'].values.astype(np.float32)
        # 1. 自适应归一化 (优化：显式 float32)
        bbw_s = pd.Series(bbw, index=df_index)
        bbw_mean = bbw_s.rolling(21).mean().values.astype(np.float32)
        bbw_std = bbw_s.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        bbw_z = (bbw - bbw_mean) / bbw_std
        # 2. 状态定义
        is_squeeze = (bbw_z < -1.0)
        is_expansion = (bbw_z > 2.0)
        # 3. 动力学结合
        s_bbw = raw['SLOPE_5_BBW_21_2.0_D'].values.astype(np.float32)
        a_bbw = raw['ACCEL_5_BBW_21_2.0_D'].values.astype(np.float32)
        vcp_ignite = np.where(is_squeeze & (a_bbw > 0), 1.4, 1.0).astype(np.float32)
        trap = np.where(is_expansion & (s_bbw < 0), 0.7, 1.0).astype(np.float32)
        # 4. HAB 紧致度修正
        squeeze_accum = raw['HIST_VOL_SQUEEZE'].values.astype(np.float32)
        squeeze_bonus = np.clip(squeeze_accum / 10.0, 1.0, 1.3).astype(np.float32)
        p_jerk = raw['JERK_3_close_D'].values.astype(np.float32)
        adj = np.ones(len(df_index), dtype=np.float32)
        adj = np.where(is_squeeze & (p_jerk > 0), 1.5 * vcp_ignite * squeeze_bonus, adj)
        adj = np.where(is_squeeze & (p_jerk < 0), 0.5, adj)
        adj = adj * trap
        final_adj = pd.Series(adj, index=df_index, dtype=np.float32).clip(0.3, 2.5)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[波动率自适应探针 V84.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    BBW Z-Score: {bbw_z[p_i]:.2f}σ, 状态: {'挤压' if is_squeeze[p_i] else ('扩张' if is_expansion[p_i] else '常态')}")
        #     print(f"    挤压累积(HAB): {squeeze_accum[p_i]:.1f}, 陷阱惩罚: {trap[p_i]:.2f}")
        return final_adj

    def _calculate_sector_overflow_decay(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V84.1 · 熵增雪崩：全链路 float32 优化，加速分形维数后处理，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        hot = raw['THEME_HOTNESS_SCORE_D'].values.astype(np.float32)
        accum_hot = raw['ACCUM_21_THEME_HOTNESS'].values.astype(np.float32)
        accel_hot = raw['ACCEL_5_THEME_HOTNESS_SCORE_D'].values.astype(np.float32)
        # 1. 相对热度归一化
        max_hot = pd.Series(hot, index=df_index).rolling(60).max().replace(0, 1.0).values.astype(np.float32)
        rel_hot = hot / max_hot
        # 2. 动态雪崩门槛
        risk_level = np.where((rel_hot > 0.9) & (accum_hot > 1500.0), 1.0, 0.0)
        # 3. 分形熵增判定
        fd_all = _numba_fractal_dimension(np.expand_dims(hot, axis=0), window=13)
        fd_vals = fd_all[0].astype(np.float32)
        slope = pd.Series(fd_vals, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        accel_fd = pd.Series(slope, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        is_exploding = (accel_hot > 0.5)
        avalanche = ((risk_level == 1.0) | (hot > 80.0) | is_exploding) & (slope > 0) & (accel_fd > 0)
        base = (1.5 / (fd_vals + 1e-9)).clip(0.5, 1.1).astype(np.float32)
        res = np.where(avalanche, base * 0.5, base).astype(np.float32)
        final_decay = pd.Series(res, index=df_index, dtype=np.float32).clip(0.1, 1.2)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[熵增雪崩自适应探针 V84.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    相对热度: {rel_hot[p_i]:.2f}, 风险等级: {'CRITICAL' if risk_level[p_i]==1.0 else 'NORMAL'}")
        #     print(f"    状态: {'雪崩' if avalanche[p_i] else '稳定'}")
        return final_decay

    def _calculate_hmm_regime_confirmation(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V85.1 · HMM 体制确认：优化数据准备阶段为 float32，无缝对接 Numba，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 资金流 Z-Score
        l_n = raw['SMART_MONEY_HM_NET_BUY_D']
        r_m = l_n.rolling(21).mean().fillna(0).values.astype(np.float32)
        r_s = l_n.rolling(21).std().replace(0, 1e-9).values.astype(np.float32)
        f_n = ((l_n.values.astype(np.float32) - r_m) / r_s)
        # 2. 成交量 Z-Score
        vol = raw['volume_D']
        v_m = vol.rolling(21).mean().fillna(0).values.astype(np.float32)
        v_s = vol.rolling(21).std().replace(0, 1e-9).values.astype(np.float32)
        v_n = ((vol.values.astype(np.float32) - v_m) / v_s)
        # 3. 价格 Z-Score
        pct = raw['pct_change_D']
        p_m = pct.rolling(21).mean().fillna(0).values.astype(np.float32)
        p_s = pct.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        p_n = ((pct.values.astype(np.float32) - p_m) / p_s)
        # 4. VWAP 乖离 Z-Score
        dist = (raw['close_D'] - raw['VWAP_D']) / (raw['VWAP_D'] + 1e-9)
        d_m = dist.rolling(21).mean().fillna(0).values.astype(np.float32)
        d_s = dist.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        v_d = ((dist.values.astype(np.float32) - d_m) / d_s)
        # 5. HMM 概率计算 (输入已确保 float32)
        m_p = _numba_hmm_regime_probability(f_n, v_n, p_n, v_d)
        m_p_s = pd.Series(m_p, index=df_index, dtype=np.float32)
        # 6. HAB 修正
        acc_m = raw['ACCUM_21_SMART_MONEY'].values.astype(np.float32)
        regime_bias = np.where(acc_m > 0, 1.15, 0.9).astype(np.float32)
        pos_days = raw['ACCUM_21_POS_SLOPE'].values.astype(np.float32)
        w_l = int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[1])
        realization_ratio = np.clip(pos_days / (float(w_l) * 0.6), 0.7, 1.2).astype(np.float32)
        # 7. 动力学修正
        p_a = m_p_s.diff(3).diff(3).fillna(0).values.astype(np.float32)
        p_j = m_p_s.diff(1).diff(1).diff(1).fillna(0).values.astype(np.float32)
        b_f = np.where(m_p > 0.5, 1.0 + (m_p - 0.5), 0.8 + m_p * 0.4).astype(np.float32)
        d_b = 1.0 + np.where((m_p > 0.6) & (p_a > 0), 0.2, 0.0) + np.where((m_p > 0.4) & (p_j > 0.1), 0.3, 0.0)
        final_conf = pd.Series(b_f * d_b * regime_bias * realization_ratio, index=df_index, dtype=np.float32)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[HMM 自适应探针 V85.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    输入Z分 -> 资金: {f_n[p_i]:.2f}, 量能: {v_n[p_i]:.2f}, 价格: {p_n[p_i]:.2f}, 乖离: {v_d[p_i]:.2f}")
        #     print(f"    拉升概率: {m_p[p_i]:.4f}, 最终确认: {final_conf.loc[probe_ts]:.4f}")
        return final_conf

    def _calculate_fractal_efficiency_resonance(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V90.1 · 分形相干效率：数据类型降级为 float32，加速矩阵运算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 优化：移除 .astype(np.float64)，使用 float32
        c_vals = raw['close_D'].values
        v_vals = raw['volume_D'].values
        input_arrays = np.vstack((c_vals, v_vals))
        f_all = _numba_fractal_dimension(input_arrays, window=21)
        h_p, h_v = (2.0 - f_all[0]).astype(np.float32), (2.0 - f_all[1]).astype(np.float32)
        h_slope = pd.Series(h_p, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        # 1. 缺口标准化
        raw_gap = np.abs(h_p - h_v)
        gap_s = pd.Series(raw_gap, index=df_index)
        gap_std = gap_s.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        gap_z = raw_gap / gap_std
        # 2. 连续性评分
        res_score = np.clip(1.2 - gap_z * 0.2, 0.5, 1.3).astype(np.float32)
        # 3. 超线性免疫
        is_super_trend = (h_p > 0.8)
        if np.any(is_super_trend):
            res_score = np.where(is_super_trend, 1.2, res_score).astype(np.float32)
        # 4. HAB 稳态
        w_l = int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[1])
        is_ordered = np.where(h_p > 0.55, 1.0, 0.0).astype(np.float32)
        struct_stab = _numba_rolling_accumulation(is_ordered, w_l)
        stab_bonus = np.clip(struct_stab / 15.0, 0.9, 1.3).astype(np.float32)
        final = res_score * stab_bonus
        final = np.where((h_p > 0.55) & (h_slope > 0), final * 1.2, final)
        final_series = pd.Series(final, index=df_index, dtype=np.float32).clip(0.4, 2.0)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[分形免疫探针 V90.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    Hurst指数: {h_p[p_i]:.4f}, 免疫状态: {'激活' if is_super_trend[p_i] else '关闭'}")
        #     print(f"    缺口Z分: {gap_z[p_i]:.2f}σ -> 基础分: {res_score[p_i]:.2f}")
        #     print(f"    >>> 最终分形效率: {final_series.loc[probe_ts]:.4f}")
        return final_series

    def _calculate_chip_lock_efficiency(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V93.1 · 筹码锁定效率：全链路 float32 优化，加速中位数滚动计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        win = raw['winner_rate_D'].values.astype(np.float32)
        turn = raw['turnover_rate_f_D'].values.astype(np.float32)
        c_s, p_s = raw['cost_50pct_D'].values.astype(np.float32), raw['close_D'].values.astype(np.float32)
        # 1. 获利盘惯性平滑
        high_win_days = raw['ACCUM_13_HIGH_WINNER'].values.astype(np.float32)
        inertial_win = np.clip(high_win_days / 13.0, 0.0, 1.0)
        c_50 = np.where(c_s == 0, p_s, c_s)
        is_above_cost = (p_s > c_50)
        cost_fix_win = np.where(is_above_cost, 0.6, 0.0).astype(np.float32)
        eff_win = np.maximum(win, np.maximum(cost_fix_win, inertial_win * 0.8))
        # 2. 相对换手率 (优化：显式 float32)
        turn_median = pd.Series(turn, index=df_index).rolling(21).median().replace(0, 0.01).values.astype(np.float32)
        rel_turn = turn / (turn_median + 1e-9)
        # 3. 趋势缓冲
        ma21 = raw['close_D'].rolling(21).mean().values.astype(np.float32)
        is_trend_safe = (raw['close_D'].values > ma21)
        buffer_rate = np.where(is_trend_safe, 0.6, 1.0).astype(np.float32)
        absorb = raw['absorption_energy_D'].values.astype(np.float32)
        final_decay = np.clip(buffer_rate - np.where(absorb > 0, 0.3, 0.0), 0.3, 1.2).astype(np.float32)
        # 4. 锁定系数
        lock_factor = 2.0 / (1.0 + np.exp((rel_turn - 1.0) * 2.0 * final_decay))
        lock_factor = np.clip(lock_factor, 0.3, 1.5)
        s_lock = eff_win * lock_factor
        # 5. 动力学与 HAB
        a_w = raw['ACCEL_5_winner_rate_D'].values.astype(np.float32)
        a_t = raw['ACCEL_5_turnover_rate_f_D'].values.astype(np.float32)
        j_w = raw['JERK_3_winner_rate_D'].values.astype(np.float32)
        k_bonus = 1.0 + np.where((a_w > 0) & (a_t < 0), 0.3, 0.0) + np.where(j_w > 0.1, 0.2, 0.0)
        deep_lock_mult = np.where(high_win_days > 10, 1.5, np.where(high_win_days > 5, 1.2, 1.0)).astype(np.float32)
        b_c = is_above_cost.astype(np.float32)
        final_eff = pd.Series(s_lock * k_bonus * deep_lock_mult * (0.8 + b_c * 0.2), index=df_index, dtype=np.float32)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[筹码惯性探针 V93.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    原始获利: {win[p_i]:.2f}, 惯性获利: {inertial_win[p_i]:.2f} (HAB={high_win_days[p_i]:.0f}d)")
        #     print(f"    有效获利(平滑后): {eff_win[p_i]:.2f}")
        #     print(f"    >>> 最终筹码效率: {final_eff.loc[probe_ts]:.4f}")
        return final_eff

    def _calculate_microstructure_attack_vector(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V94.1 · 微观矢量：全链路 float32 优化，加速滚动 Z-Score 计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        e_n = (raw['buy_elg_amount_D'].values - raw['sell_elg_amount_D'].values).astype(np.float32)
        l_n = (raw['buy_lg_amount_D'].values - raw['sell_lg_amount_D'].values).astype(np.float32)
        t_a = (raw['amount_D'].values + 1e-9).astype(np.float32)
        net_ratio = (e_n + l_n) / t_a
        # 1. 资金流 Z-Score
        ratio_s = pd.Series(net_ratio, index=df_index)
        ratio_mean = ratio_s.rolling(21).mean().fillna(0).values.astype(np.float32)
        ratio_std = ratio_s.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        z_score_flow = (net_ratio - ratio_mean) / ratio_std
        base_score = 0.5 + (1.0 / (1.0 + np.exp(-z_score_flow)))
        # 2. 价格动力学 Z-Score
        raw_jerk = raw['JERK_3_close_D'].values.astype(np.float32)
        jerk_s = pd.Series(raw_jerk, index=df_index)
        jerk_mean = jerk_s.rolling(21).mean().fillna(0).values.astype(np.float32)
        jerk_std = jerk_s.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        z_score_jerk = (raw_jerk - jerk_mean) / jerk_std
        # 3. 隐蔽吸筹与良性回调
        close = raw['close_D'].values.astype(np.float32)
        ma5 = pd.Series(close, index=df_index).rolling(5).mean().values.astype(np.float32)
        ma21 = raw['close_D'].rolling(21).mean().values.astype(np.float32)
        is_stealth = (net_ratio < 0) & (close > ma5) & (z_score_jerk > 0)
        is_benign_dip = (z_score_flow > -1.0) & (close > ma21) & (z_score_jerk > -2.0)
        adjusted_base = base_score
        if np.any(is_stealth) or np.any(is_benign_dip):
            guard_mask = (is_stealth | is_benign_dip)
            adjusted_base = np.where(guard_mask, np.maximum(base_score, 1.0), base_score).astype(np.float32)
        # 4. 协同性与动力学
        e_r, l_r = e_n / t_a, l_n / t_a
        sync_score = np.where((e_r > 0) & (l_r > 0), 1.1, np.where((e_r * l_r) < 0, 0.9, 1.0)).astype(np.float32)
        s_j = raw['JERK_3_SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float32)
        jerk_bonus = np.where(s_j > 0.5, 1.2, 1.0).astype(np.float32)
        d_sm = (e_n + l_n)
        a13 = raw['ACCUM_13_SMART_MONEY'].values.astype(np.float32)
        h_s = np.where((d_sm < 0) & (a13 > np.abs(d_sm) * 10.0), 1.1, 1.0).astype(np.float32)
        final_v = pd.Series(adjusted_base * sync_score * jerk_bonus * h_s, index=df_index, dtype=np.float32).clip(0.4, 2.0)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[微观自适应探针 V94.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    资金Z: {z_score_flow[p_i]:.2f}σ, 价格Jerk Z: {z_score_jerk[p_i]:.2f}σ")
        #     print(f"    状态判定: {'隐蔽吸筹' if is_stealth[p_i] else ('良性回调' if is_benign_dip[p_i] else '正常')}")
        #     print(f"    >>> 最终攻击矢量: {final_v.loc[probe_ts]:.4f}")
        return final_v

    def _calculate_vpa_elasticity_reflexivity(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V83.1 · 流体反身性：全链路 float32 优化，加速中位数滚动计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        pp = np.abs(raw['pct_change_D'].values).astype(np.float32)
        pv = np.abs(pd.Series(raw['volume_D'].values, index=df_index).pct_change().fillna(0).values).astype(np.float32)
        # 1. 瞬时弹性
        current_elasticity = pp / (pv + 0.1)
        # 2. 自适应归一化 (优化：显式 float32)
        ela_median = pd.Series(current_elasticity, index=df_index).rolling(21).median().replace(0, 0.01).values.astype(np.float32)
        rel_elasticity = current_elasticity / ela_median
        score = np.clip(np.tanh(rel_elasticity - 0.8) + 0.5, 0.5, 1.8).astype(np.float32)
        # 3. HAB 弹性密度修正
        acc_e = raw['ACCUM_13_ELASTICITY'].values.astype(np.float32)
        hab_score = np.clip(acc_e / (ela_median * 13.0 + 1e-9), 0.8, 1.4).astype(np.float32)
        # 4. 动力学修正
        e_s = pd.Series(current_elasticity, index=df_index).diff(3).values.astype(np.float32)
        slope_bonus = np.where(e_s > 0, 1.2, 1.0).astype(np.float32)
        final = pd.Series(score * slope_bonus * hab_score, index=df_index, dtype=np.float32).clip(0.5, 2.5)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[反身性自适应探针 V83.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    当前弹性: {current_elasticity[p_i]:.2f}, 历史中位: {ela_median[p_i]:.2f}")
        #     print(f"    相对比率: {rel_elasticity[p_i]:.2f} -> 基础分: {score[p_i]:.2f}")
        return final

    def _calculate_wyckoff_breakout_quality(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V85.1 · 威科夫突破：全链路 float32 优化，提升 TR 动力学计算速度，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        high = raw['high_D'].values.astype(np.float32)
        low = raw['low_D'].values.astype(np.float32)
        close = raw['close_D'].values.astype(np.float32)
        vol = raw['volume_D'].values.astype(np.float32)
        close_prev = raw['close_D'].shift(1).fillna(raw['close_D']).values.astype(np.float32)
        # 1. TR 动力学
        tr = np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
        tr_norm = tr / (close + 1e-9)
        tr_s = pd.Series(tr_norm, index=df_index)
        tr_slope = (tr_s.diff(5) / 5.0).values.astype(np.float32)
        tr_accel = (pd.Series(tr_slope, index=df_index).diff(5) / 5.0).values.astype(np.float32)
        # 2. 价格脉冲自适应归一化
        raw_jerk = raw['JERK_3_close_D'].values.astype(np.float32)
        jerk_s = pd.Series(raw_jerk, index=df_index)
        jerk_mean = jerk_s.rolling(21).mean().fillna(0).values.astype(np.float32)
        jerk_std = jerk_s.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        jerk_z = (raw_jerk - jerk_mean) / jerk_std
        # 3. 自适应压缩比
        vol_ma = pd.Series(vol, index=df_index).rolling(21).mean().replace(0, 1).values.astype(np.float32)
        tr_ma = pd.Series(tr, index=df_index).rolling(21).mean().replace(0, 1e-9).values.astype(np.float32)
        compression_ratio = np.clip((vol / vol_ma) / (tr / tr_ma + 0.1), 0.5, 3.0).astype(np.float32)
        acc_compression = pd.Series(compression_ratio, index=df_index).rolling(5).mean().values.astype(np.float32)
        # 4. 突破判定
        highest_21 = raw['high_D'].rolling(21).max().shift(1).fillna(99999).values.astype(np.float32)
        is_breakout = close > highest_21
        quality_mult = np.where(acc_compression > 1.2, 1.5, 1.0).astype(np.float32)
        is_explosive = (jerk_z > 1.5)
        base_q = np.where(is_breakout & is_explosive, 1.6, np.where(is_breakout, 0.9, 0.0)).astype(np.float32)
        prep_q = np.where((tr_slope < 0) & (tr_accel > -0.01), 1.1, 0.9).astype(np.float32)
        quality = np.maximum(base_q, prep_q)
        final_score = pd.Series(quality * np.where((tr_slope < 0), 1.3, 1.0) * quality_mult, index=df_index, dtype=np.float32).clip(0.6, 2.5)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[威科夫自适应探针 V85.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    Jerk Z-Score: {jerk_z[p_i]:.2f}σ (原值: {raw_jerk[p_i]:.4f})")
        #     print(f"    状态: {'爆发突破' if is_breakout[p_i] and is_explosive[p_i] else ('普通突破' if is_breakout[p_i] else '蓄势')}")
        #     print(f"    >>> 最终突破质量: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_trend_inertia_momentum(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V83.1 · 趋势运动学：全链路 float32 优化，加速滚动标准差计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 均线排列
        close_s = raw['close_D']
        ma5 = close_s.rolling(5).mean().values.astype(np.float32)
        ma21 = close_s.rolling(21).mean().values.astype(np.float32)
        ma55 = close_s.rolling(55).mean().values.astype(np.float32)
        alignment = np.where((ma5 > ma21) & (ma21 > ma55), 1.0, 0.8).astype(np.float32)
        # 2. 标准化动力学
        s_vals = raw['SLOPE_5_close_D'].values.astype(np.float32)
        a_vals = raw['ACCEL_5_close_D'].values.astype(np.float32)
        j_vals = raw['JERK_3_close_D'].values.astype(np.float32)
        # 优化：float32 滚动计算
        s_std = pd.Series(s_vals, index=df_index).rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        a_std = pd.Series(a_vals, index=df_index).rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        norm_s = s_vals / s_std
        norm_a = a_vals / a_std
        # 3. 状态定义
        is_strong_trend = (norm_s > 1.0) & (norm_a > 0.5)
        is_parabolic = (norm_a > 1.5) & (j_vals > 0)
        # 4. 评分映射
        kinematic_score = np.ones(len(df_index), dtype=np.float32)
        kinematic_score += np.where(is_strong_trend, 0.3, 0.0).astype(np.float32)
        kinematic_score += np.where(is_parabolic, 0.4, 0.0).astype(np.float32)
        kinematic_score -= np.where((norm_s > 0) & (norm_a < -1.0), 0.3, 0.0).astype(np.float32)
        # 5. HAB 韧性修正
        pos_days = raw['ACCUM_21_POS_SLOPE'].values.astype(np.float32)
        consistency_bonus = np.clip(pos_days / 15.0, 0.8, 1.25).astype(np.float32)
        r2_vals = raw['GEOM_REG_R2_D'].values.astype(np.float32)
        final_inertia = pd.Series(alignment * kinematic_score * consistency_bonus * (0.6 + np.clip(r2_vals, 0.0, 1.0) * 0.4), index=df_index, dtype=np.float32).clip(0.6, 2.0)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[运动学自适应探针 V83.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    标准化速度: {norm_s[p_i]:.2f}σ, 标准化加速: {norm_a[p_i]:.2f}σ")
        #     print(f"    状态: {'抛物线加速' if is_parabolic[p_i] else ('强趋势' if is_strong_trend[p_i] else '常态')}")
        return final_inertia

    def _calculate_market_permeability_index(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V82.1 · 市场渗透率：使用 Numba 滚动排名替代 Pandas apply，大幅提升计算效率，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        sent = raw['market_sentiment_score_D'].values
        accel = raw['ACCEL_5_market_sentiment_score_D'].values
        # 1. 历史分位数归一化 (优化：使用 _numba_rolling_rank)
        # 替代原先极其缓慢的 rolling().apply(lambda...)
        sent_rank = _numba_rolling_rank(sent, 21)
        # Numba 函数前 20 个值为 NaN，使用 1.0 (中性/无信号) 填充以保持 safe
        # sent_rank < 0.2 对 NaN 为 False，sent_rank > 0.8 对 NaN 为 False
        # 2. 相对极值判断
        perm = np.where((sent_rank < 0.2) & (accel > 0), 1.3, np.where((sent_rank > 0.8) & (accel > 0), 0.7, 1.0)).astype(np.float32)
        # 3. HAB 饱和度调节
        acc_sent = raw['ACCUM_13_SENTIMENT'].values
        acc_sent_median = pd.Series(acc_sent, index=df_index).rolling(21).median().values
        saturation_decay = np.where(acc_sent > acc_sent_median * 1.5, 0.8, 1.0).astype(np.float32)
        rank_val, jerk = raw['industry_strength_rank_D'].values, raw['JERK_3_industry_strength_rank_D'].values
        bonus = np.where(jerk < -2.0, 1.3, np.where(rank_val < 0.1, 1.1, 0.9)).astype(np.float32)
        final_ctx = pd.Series(perm * bonus * saturation_decay, index=df_index, dtype=np.float32).clip(0.5, 1.8)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"    情绪分位数: {sent_rank[p_i]:.2f}, 相对状态: {'冰点复苏' if sent_rank[p_i] < 0.2 else ('过热' if sent_rank[p_i] > 0.8 else '常态')}")
        return final_ctx

    def _calculate_entry_accessibility_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V86.1 · 入场可获得性：全链路 float32 优化，加速均值滚动计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        turnover = raw['turnover_rate_f_D'].values.astype(np.float32)
        # 1. 相对流动性
        turn_ma = pd.Series(turnover, index=df_index).rolling(21).mean().replace(0, 0.1).values.astype(np.float32)
        rel_liq = turnover / turn_ma
        # 2. 流动性评分
        base_access = np.where(rel_liq < 0.5, 0.4 + rel_liq * 0.5,
                      np.where(rel_liq <= 2.5, 1.0,
                               np.clip(2.5/rel_liq, 0.5, 1.0))).astype(np.float32)
        # 3. 封板硬约束
        up_limit = raw['up_limit_D'].values.astype(np.float32)
        close = raw['close_D'].values.astype(np.float32)
        intensity = raw['closing_flow_intensity_D'].values.astype(np.float32)
        sealing = np.clip(intensity, 0.0, 1.0)
        limit_penalty = np.where(close >= up_limit * 0.999, 0.2 + 0.3 * (1.0 - sealing), 1.0).astype(np.float32)
        # 4. 抢筹拥堵
        tc_accel = raw['ACCEL_5_trade_count_D'].values.astype(np.float32)
        lock_days = raw['ACCUM_5_LIQUIDITY_LOCK'].values.astype(np.float32)
        congestion = np.where((tc_accel > 0.1) & (lock_days >= 3), 0.6, 1.0).astype(np.float32)
        final_access = pd.Series(base_access * limit_penalty * congestion, index=df_index, dtype=np.float32).clip(0.1, 1.2)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[入场保底探针 V86.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    相对流动性: {rel_liq[p_i]:.2f}, 封板惩罚: {limit_penalty[p_i]:.2f}")
        #     print(f"    >>> 最终可获得性: {final_access.loc[probe_ts]:.4f}")
        return final_access

    def _calculate_entropic_ordering_bonus(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V84.1 · 熵减有序性：全链路 float32 优化，加速 Z-Score 计算，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        s5 = raw['SLOPE_5_chip_entropy_D'].values.astype(np.float32)
        a5 = raw['ACCEL_5_chip_entropy_D'].values.astype(np.float32)
        j3 = raw['JERK_3_chip_entropy_D'].values.astype(np.float32)
        pct = raw['pct_change_D'].values.astype(np.float32)
        # 1. 有序化力度合成
        locking_force = -(s5 * 0.7 + a5 * 0.3)
        # 2. 自适应归一化 (优化：显式 float32)
        force_s = pd.Series(locking_force, index=df_index)
        force_std = force_s.rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        force_z = locking_force / force_std
        base_bonus = np.clip(np.tanh(force_z - 1.0) + 0.5, 0.0, 1.5).astype(np.float32)
        # 3. 脉冲修正
        jerk_bonus = np.where((j3 < -0.01) & (force_z > 1.5), 1.3, 1.0).astype(np.float32)
        # 4. HAB 稳定性修正
        ent_stab = raw['ACCUM_21_ENTROPY_STABILITY'].values.astype(np.float32)
        stab_ratio = ent_stab / 21.0
        stability_mult = np.clip(stab_ratio * 2.0, 0.8, 1.4).astype(np.float32)
        penalty = np.where((pct > 0) & (s5 > 0), 0.7, 1.0).astype(np.float32)
        final_factor = pd.Series((1.0 + base_bonus) * jerk_bonus * stability_mult * penalty, index=df_index, dtype=np.float32).clip(0.6, 2.5)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[熵减自适应探针 V84.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    锁定力度Z分: {force_z[p_i]:.2f}σ, 稳定性占比: {stab_ratio[p_i]*100:.1f}%")
        #     print(f"    >>> 最终有序性系数: {final_factor.loc[probe_ts]:.4f}")
        return final_factor

    def _calculate_vwap_propulsion_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V83.1 · VWAP 推进力：全链路 float32 优化，提升 Z-Score 计算效率，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        close = raw['close_D'].values.astype(np.float32)
        vwap = raw['VWAP_D'].values.astype(np.float32)
        # 1. 基础速度 (Slope)
        vwap_slope = pd.Series(vwap, index=df_index).diff(3).values.astype(np.float32) / 3.0
        # 2. 自适应归一化 (Adaptive Normalization)
        # 优化：显式指定 float32
        slope_std = pd.Series(vwap_slope, index=df_index).rolling(21).std().replace(0, 1e-6).values.astype(np.float32)
        norm_propulsion = np.clip(vwap_slope / slope_std, -3.0, 3.0)
        propulsion_score = np.where(norm_propulsion > 0, np.tanh(norm_propulsion * 0.8) * 1.5, 0.0).astype(np.float32)
        # 3. 乖离率修正
        bias = (close - vwap) / (vwap + 1e-9)
        bias_penalty = np.where(bias > 0.08, 0.8, 1.0).astype(np.float32)
        # 4. 动力学增强
        accel_vwap = raw['ACCEL_5_VWAP_D'].values.astype(np.float32)
        kinematic_boost = np.where(accel_vwap > 0, 1.3, 1.0).astype(np.float32)
        # 5. HAB 趋势厚度
        days_above = raw['ACCUM_21_ABOVE_VWAP'].values.astype(np.float32)
        thickness_bonus = np.clip(days_above / 10.0, 0.8, 1.4).astype(np.float32)
        final_score = pd.Series(propulsion_score * bias_penalty * kinematic_boost * thickness_bonus, index=df_index, dtype=np.float32).clip(0, 2.5)
        # if is_debug and probe_ts in df_index:
        #     p_i = df_index.get_loc(probe_ts)
        #     print(f"\n[VWAP 自适应探针 V83.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
        #     print(f"    原始斜率: {vwap_slope[p_i]:.5f}, 历史波动(Std): {slope_std[p_i]:.5f}")
        #     print(f"    标准化推进: {norm_propulsion[p_i]:.2f}σ (Sigma)")
        #     print(f"    >>> 最终推进力: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _print_full_chain_probe(self, probe_ts: pd.Timestamp, raw: Dict[str, pd.Series], scores: Dict[str, pd.Series], final_score: float):
        """V86.0 · 全息探针：优化健康度显示，使用 Z-Score 替代 Raw Jerk，清除空行"""
        if probe_ts not in raw['close_D'].index: return
        idx = raw['close_D'].index.get_loc(probe_ts)
        p_str = probe_ts.strftime('%Y-%m-%d')
        w_s, w_l = int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[0]), int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[1])
        # 临时计算 Jerk Z-Score 用于展示
        raw_jerk = raw['JERK_3_close_D']
        jerk_mean = raw_jerk.rolling(21).mean().fillna(0).values[idx]
        jerk_std = raw_jerk.rolling(21).std().replace(0, 1e-6).values[idx]
        jerk_z = (raw_jerk.values[idx] - jerk_mean) / jerk_std
        # 健康度计算
        impulse_score = jerk_z * 0.5 + raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'].values[idx] / 1000.0
        foundation_score = raw['ACCUM_21_SMART_MONEY'].values[idx] / 10000.0 + raw['ACCUM_21_ABOVE_VWAP'].values[idx] / 5.0
        fragility = 0.0
        if impulse_score > 1.0 and foundation_score < 0.5:
            fragility = (impulse_score - foundation_score) * 0.5
        health_score = np.clip((1.0 + foundation_score) * (1.0 - np.clip(fragility, 0, 0.8)) * 10.0, 0, 100)
        health_status = "健康" if health_score > 60 else ("脆性" if fragility > 0.3 else "虚浮")
        print(f"\n{'='*30} [全息动力学探针 V86.0 @ {p_str}] {'='*30}")
        print(f"【自适应配置】市值窗口: 短周期 {w_s}d / 长周期 {w_l}d")
        print(f"【基础物理】收盘: {raw['close_D'].values[idx]:.2f} | VWAP: {raw['VWAP_D'].values[idx]:.2f} | 换手: {raw['turnover_rate_f_D'].values[idx]:.2f}%")
        print(f"【HAB 缓冲】")
        print(f"  ├─ 资金底仓 (S/L): {raw['ACCUM_13_SMART_MONEY'].values[idx]:.1f} / {raw['ACCUM_21_SMART_MONEY'].values[idx]:.1f}")
        print(f"  ├─ 尾盘压力: {raw['ACCUM_13_CLOSING_FLOW'].values[idx]:.2f} | 板块厚度: {raw['ACCUM_21_THEME_HOTNESS'].values[idx]:.1f}")
        print(f"  └─ 波动紧致: {raw['HIST_VOL_SQUEEZE'].values[idx]:.1f} | 趋势厚度: {raw['ACCUM_21_ABOVE_VWAP'].values[idx]:.0f}d")
        print(f"【动力学特征】")
        print(f"  ├─ 价格 (S/A/J): {raw['SLOPE_5_close_D'].values[idx]:.4f} / {raw['ACCEL_5_close_D'].values[idx]:.4f} / {raw['JERK_3_close_D'].values[idx]:.4f}")
        print(f"  └─ 价格脉冲强度: {jerk_z:.2f}σ (Sigma)")
        print(f"【健康度诊断】")
        print(f"  ★ 综合评分: {health_score:.1f} ({health_status}) | 脉冲Z: {jerk_z:.2f} vs 底仓: {foundation_score:.2f}")
        print(f"【核心引擎评分】")
        print(f"  [物理] {scores['physical'].values[idx]:.3f} | [几何] {scores['geo'].values[idx]:.3f} | [VWAP] {scores['vwap'].values[idx]:.3f}")
        print(f"  [结构] 筹码:{scores['chip'].values[idx]:.3f} 熵序:{scores['entropy'].values[idx]:.3f} 威科夫:{scores['wyckoff'].values[idx]:.3f}")
        print(f"  [环境] 渗透:{scores['perm'].values[idx]:.3f} 惯性:{scores['inertia'].values[idx]:.3f} HMM:{scores['hmm'].values[idx]:.3f}")
        print(f"  [微观] 攻击:{scores['micro'].values[idx]:.3f} 反身:{scores['reflex'].values[idx]:.3f} 分形:{scores['fractal'].values[idx]:.3f}")
        print(f"【安全阀调节】")
        print(f"  风险:{scores['risk'].values[idx]:.2f} x 衰减:{scores['decay'].values[idx]:.2f} x 板块:{scores['sector_decay'].values[idx]:.2f} x 伽马:{scores['vol_gamma'].values[idx]:.2f} x 入场:{scores['access'].values[idx]:.2f}")
        print(f"{'-'*75}")
        print(f" >>> PROCESS_META_POWER_TRANSFER 最终得分: {final_score:.4f}")
        print(f"{'='*75}\n")

