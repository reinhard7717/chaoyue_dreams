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
        """V86.1 · 原料加载层：修复市值单位(万元->元)及0值干扰，确保自适应窗口正确锚定，清除空行"""
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

        # --- 市值自适应逻辑修正 V86.1 ---
        mv_source = "RAW"
        # 1. 单位修正与有效性检查
        # 假设 circ_mv_D 单位为万元，转换为元
        if has_mv_col:
            raw_signals['circ_mv_D'] = raw_signals['circ_mv_D'] * 10000.0

        # 2. 计算代表性市值 (Representative Market Value)
        # 过滤掉 0 值和异常小值 (< 1000万)，取中位数代表该股的整体体量
        valid_mv = raw_signals.get('circ_mv_D', pd.Series([0]))
        valid_mv = valid_mv[valid_mv > 1e7] # 过滤掉 0 和无效值
        
        if len(valid_mv) > 0:
            avg_mv = valid_mv.median() # 使用中位数更稳健
        else:
            # 3. 估算回退逻辑
            amt = raw_signals['amount_D']
            turn = raw_signals['turnover_rate_f_D']
            est_mv = (amt / (turn + 0.1) * 100.0).replace([np.inf, -np.inf], 0).fillna(0)
            avg_mv = est_mv[est_mv > 1e7].median()
            if np.isnan(avg_mv): avg_mv = 0.0
            mv_source = "ESTIMATED"
        
        # 4. 窗口判定
        w_s, w_l = 13, 21 # 默认中盘
        mv_tag = "MID"
        
        if avg_mv > 0:
            if avg_mv < 50e8: # < 50亿
                w_s, w_l = 8, 13
                mv_tag = "SMALL"
            elif avg_mv > 500e8: # > 500亿
                w_s, w_l = 21, 34
                mv_tag = "LARGE"
        
        # 将窗口参数存入信号流
        raw_signals['META_HAB_WINDOWS'] = pd.Series([w_s, w_l], index=df.index[:2])
        
        # --- 后续计算 ---
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
        # 使用自适应窗口计算 HAB
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
        threshold_map = {'net_amount_rate_D': (0.01, 0.005), 'winner_rate_D': (0.01, 0.005), 'SMART_MONEY_HM_NET_BUY_D': (10, 10), 'VPA_EFFICIENCY_D': (0.01, 0.01), 'BBW_21_2.0_D': (0.001, 0.0001), 'THEME_HOTNESS_SCORE_D': (0.1, 0.1), 'chip_entropy_D': (0.01, 0.001), 'volume_D': (100, 10), 'pct_change_D': (0.0001, 0.0001), 'close_D': (0.1, 0.01), 'market_sentiment_score_D': (0.1, 0.1), 'industry_strength_rank_D': (0.001, 0.001), 'turnover_rate_f_D': (0.01, 0.01), 'trade_count_D': (10, 5), 'VWAP_D': (0.1, 0.01), 'closing_flow_ratio_D': (0.01, 0.01)}
        fib_wins = [3, 5, 8, 13, 21]
        for col, (abs_th, chg_th) in threshold_map.items():
            base_vals = raw_signals[col].values
            for win in fib_wins:
                s, a, j = _numba_robust_dynamics(base_vals, win=win, abs_threshold=abs_th, change_threshold=chg_th)
                raw_signals[f"SLOPE_{win}_{col}"] = pd.Series(s, index=df.index, dtype=np.float32)
                raw_signals[f"ACCEL_{win}_{col}"] = pd.Series(a, index=df.index, dtype=np.float32)
                raw_signals[f"JERK_{win}_{col}"] = pd.Series(j, index=df.index, dtype=np.float32)
        if is_debug and probe_ts in df.index:
            print(f"\n[原料 V86.1 修正 HAB 探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    原始市值包含0值: {'是' if (raw_signals['circ_mv_D']==0).any() else '否'}")
            print(f"    代表性市值: {avg_mv/1e8:.1f}亿 ({mv_source}), 类型: {mv_tag}")
            print(f"    窗口配置: 短 {w_s}d / 长 {w_l}d")
        return raw_signals

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V77.0 · 全局动力学总线：集成全息探针与 HAB 持久化，实现从原料到最终分值的全链路可视化与状态保存，清除空行"""
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
        # 4. 执行 HAB 状态持久化
        self._persist_hab_state(raw, df_index, method_name)
        # 5. 触发全息探针
        if is_debug and probe_ts in df_index:
            self._print_full_chain_probe(probe_ts, raw, scores, final_score.loc[probe_ts])
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
        """V91.0 · 物理动力引擎：重构为非线性门控逻辑，彻底消除滞后 MCV 对反转启动的拖累，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 基础物理冲量
        vol_adj = raw['BBW_21_2.0_D'].values.astype(np.float64)
        roll_tc = pd.Series(raw['trade_count_D'].values, index=df_index).rolling(21).mean().replace(0, 1).values
        conf = (raw['trade_count_D'].values / (roll_tc + 1e-9)).astype(np.float64)
        j_c = _numba_adaptive_denoise_dynamics(raw['JERK_3_net_amount_rate_D'].values.astype(np.float64), vol_adj, conf)
        a_c = _numba_adaptive_denoise_dynamics(raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float64), vol_adj, conf)
        raw_impulse = (j_c * 0.45 + a_c * 0.55)
        # 2. 自适应归一化 (敏感度 0.8)
        imp_series = pd.Series(raw_impulse, index=df_index)
        imp_mean = imp_series.rolling(21).mean().fillna(0).values
        imp_std = imp_series.rolling(21).std().replace(0, 1e-9).values
        z_score_imp = (raw_impulse - imp_mean) / imp_std
        norm_imp = np.tanh(z_score_imp * 0.8).astype(np.float32)
        # 3. 极值补偿
        price_pos = raw['price_percentile_position_D'].values
        comp_factor = np.where(price_pos < 0.2, 1.2, np.where(price_pos > 0.8, 0.8, 1.0)).astype(np.float32)
        comp_imp = norm_imp * comp_factor
        # 4. MCV 非线性门控 (Non-linear Gating)
        _, f_slopes = _numba_fast_rolling_dynamics(raw['net_amount_rate_D'].values.astype(np.float64), np.array([3, 5, 8, 13, 21], dtype=np.int64))
        mcv = np.dot(np.array([0.35, 0.25, 0.20, 0.10, 0.10], dtype=np.float32), f_slopes.astype(np.float32))
        # 逻辑重构：
        # Case A: 冲量 > 0, MCV > 0 -> 共振 (Resonance), 全额叠加
        # Case B: 冲量 > 0, MCV < 0 -> 背离 (Divergence), 忽略 MCV (权重=0)
        # Case C: 冲量 < 0 -> 下跌或震荡, 正常叠加 MCV (权重=0.35)
        mcv_weight = np.where((comp_imp > 0) & (mcv < 0), 0.0, 0.35).astype(np.float32)
        # 5. 合成
        accum_m = raw['ACCUM_21_SMART_MONEY'].values
        m_mass = np.clip(np.log1p(np.abs(accum_m)) / 10.0, 0.8, 1.3).astype(np.float32)
        phy_score = ((comp_imp * 2.0 * 0.30 + mcv * mcv_weight) * m_mass).astype(np.float32)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[物理引擎门控探针 V91.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    冲量: {comp_imp[p_i]:.4f}, MCV: {mcv[p_i]:.4f}")
            print(f"    模式: {'背离启动(忽略MCV)' if mcv_weight[p_i] == 0 else '正常叠加'}")
            print(f"    >>> 物理合成总分: {phy_score[p_i]:.4f}")
        return pd.Series(phy_score, index=df_index, dtype=np.float32)

    def _calculate_premium_reversal_risk(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V82.0 · 溢价回吐风险：采用历史天量归一化，识别相对衰竭，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        turnover = raw['turnover_rate_f_D'].values
        extreme = raw['STATE_EMOTIONAL_EXTREME_D'].values.astype(np.float32)
        daily_ratio = raw['closing_flow_ratio_D'].values
        # 1. 换手竭尽归一化 (Exhaustion Normalization)
        # 计算 21 日最大换手率作为短期天花板
        max_turn = pd.Series(turnover, index=df_index).rolling(21).max().replace(0, 1.0).values
        # 竭尽率 = 当前换手 / 周期峰值。接近 1.0 代表到达天量区，风险极大
        exhaustion_rate = np.clip(turnover / max_turn, 0.0, 1.5)
        # 2. HAB 尾盘压力
        accum_ratio = raw['ACCUM_13_CLOSING_FLOW'].values
        # 尾盘压力归一化：相对 13 日均值
        mean_ratio = pd.Series(daily_ratio, index=df_index).rolling(13).mean().replace(0, 0.1).values
        hab_risk_norm = np.where(accum_ratio > mean_ratio * 13 * 1.5, 1.25, 1.0)
        reversal_pressure = daily_ratio * extreme * exhaustion_rate * hab_risk_norm
        risk_adjustment = pd.Series(1.0 - reversal_pressure * 0.4, index=df_index, dtype=np.float32).clip(0.5, 1.0)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[溢价风险自适应探针 V82.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    换手竭尽率: {exhaustion_rate[p_i]:.2f} (当前/21日峰值)")
        return risk_adjustment

    def _calculate_intraday_decay_model(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V88.0 · 日内衰减：采用 Tanh 软衰减，大幅降低对常态波动(轻微不稳定)的惩罚力度，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        stab = raw['TURNOVER_STABILITY_INDEX_D'].values
        # 1. 稳定性归一化 (Z-Score)
        stab_s = pd.Series(stab, index=df_index)
        stab_mean = stab_s.rolling(21).mean().values
        stab_std = stab_s.rolling(21).std().replace(0, 1e-6).values
        stab_z = (stab - stab_mean) / stab_std
        # 2. 软衰减因子映射 (Soft Decay)
        # Z=0 -> 1.0 (中性)
        # Z=-0.5 -> 1.0 + (-0.46)*0.15 = 0.93 (轻微惩罚)
        # Z=-2.0 -> 1.0 + (-0.96)*0.15 = 0.85 (最大惩罚不超过 15%)
        # 这种设计即便是烂板也能给到 0.85 的保底分，不至于直接熔断
        base_decay = np.clip(1.0 + np.tanh(stab_z) * 0.15, 0.7, 1.2).astype(np.float32)
        # 3. 结构性崩塌判定
        close, up, ratio = raw['close_D'].values, raw['up_limit_D'].values, raw['closing_flow_ratio_D'].values
        mean_stab = raw['MEAN_13_STABILITY'].values
        bad_board = (close >= up * 0.999) & (ratio > 0.4) & (stab < 0.4)
        fragility = np.where(mean_stab < 0.5, 0.9, 1.0) # 历史差，打9折
        winner = raw['winner_rate_D'].values
        repair = np.where((winner < 0.15) & (stab < 0.3), 1.5, 1.0).astype(np.float32)
        decay = base_decay * np.where(bad_board, 0.6, 1.0).astype(np.float32) * repair * fragility
        final_decay = pd.Series(decay, index=df_index, dtype=np.float32).clip(0.4, 1.5)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[日内衰减柔化探针 V88.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    稳定性Z分: {stab_z[p_i]:.2f}σ -> 基础衰减: {base_decay[p_i]:.2f}")
            print(f"    >>> 最终日内衰减: {final_decay.loc[probe_ts]:.4f}")
        return final_decay

    def _calculate_sector_resonance_modifier(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V82.0 · 板块共振：采用信噪比(SNR)归一化，剔除无效波动，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        s_hot = raw['SLOPE_5_THEME_HOTNESS_SCORE_D'].values
        a_hot = raw['ACCEL_5_THEME_HOTNESS_SCORE_D'].values
        # 1. 动量信噪比归一化 (Momentum SNR Normalization)
        # 计算斜率的 21 日标准差，用作归一化分母
        s_hot_std = pd.Series(s_hot, index=df_index).rolling(21).std().replace(0, 1e-9).values
        # SNR > 1 代表趋势性显著强于波动性
        s_hot_snr = np.clip(s_hot / s_hot_std, -3.0, 3.0)
        # 使用 sigmoid 将 SNR 映射为 0~1 的冲击系数
        impulse_factor = (1.0 / (1.0 + np.exp(-s_hot_snr))).astype(np.float32)
        # 2. HAB 主线霸榜
        high_rank_days = raw['ACCUM_21_HIGH_RANK'].values
        # 归一化霸榜天数 (0~1)
        leadership_norm = np.clip(high_rank_days / 21.0, 0.0, 1.0).astype(np.float32)
        leadership_bonus = 1.0 + leadership_norm * 0.4 # 最大加成 1.4
        persistence = np.where((raw['industry_rank_accel_D'].values > 0) & (raw['flow_consistency_D'].values > 0.65), 1.2, 0.8).astype(np.float32)
        rank_jerk = raw['JERK_3_industry_strength_rank_D'].values
        rank_pulse = np.where(rank_jerk < -2.0, 1.3, 1.0).astype(np.float32)
        mod = (0.8 + impulse_factor * 0.4) * persistence * leadership_bonus * rank_pulse
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[板块信噪比探针 V82.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    热度斜率SNR: {s_hot_snr[p_i]:.2f}, 冲击系数: {impulse_factor[p_i]:.2f}")
        return pd.Series(mod, index=df_index, dtype=np.float32).clip(0.6, 2.2)

    def _calculate_volatility_clustering_adjustment(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V84.0 · 波动率伽马：采用 BBW Z-Score 自适应归一化，统计学定义挤压与扩张，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        bbw = raw['BBW_21_2.0_D'].values
        # 1. 自适应归一化 (Z-Score Normalization)
        # 计算 BBW 的 21 日均值和标准差
        bbw_s = pd.Series(bbw, index=df_index)
        bbw_mean = bbw_s.rolling(21).mean().values
        bbw_std = bbw_s.rolling(21).std().replace(0, 1e-6).values
        # BBW Z-Score: 衡量当前波动带宽相对于历史的偏离程度
        bbw_z = (bbw - bbw_mean) / bbw_std
        # 2. 状态定义
        # Z < -1.0: 显著挤压 (Squeeze)，能量积蓄
        # Z > 2.0: 异常扩张 (Expansion)，能量释放/耗尽
        is_squeeze = (bbw_z < -1.0)
        is_expansion = (bbw_z > 2.0)
        # 3. 动力学结合
        s_bbw = raw['SLOPE_5_BBW_21_2.0_D'].values
        a_bbw = raw['ACCEL_5_BBW_21_2.0_D'].values
        # VCP 点火：处于挤压状态且波动率开始加速放大 (Accel > 0)
        vcp_ignite = np.where(is_squeeze & (a_bbw > 0), 1.4, 1.0)
        # 波动率陷阱：处于极度扩张状态且开始减速 (Slope < 0)
        trap = np.where(is_expansion & (s_bbw < 0), 0.7, 1.0)
        # 4. HAB 紧致度修正
        squeeze_accum = raw['HIST_VOL_SQUEEZE'].values
        # 归一化 HAB: 长期挤压后的爆发更猛烈
        squeeze_bonus = np.clip(squeeze_accum / 10.0, 1.0, 1.3)
        p_jerk = raw['JERK_3_close_D'].values
        adj = np.ones(len(df_index), dtype=np.float32)
        adj = np.where(is_squeeze & (p_jerk > 0), 1.5 * vcp_ignite * squeeze_bonus, adj) # 挤压+价格脉冲=爆发
        adj = np.where(is_squeeze & (p_jerk < 0), 0.5, adj) # 挤压+向下脉冲=破位
        adj = adj * trap # 扩张陷阱惩罚
        final_adj = pd.Series(adj, index=df_index, dtype=np.float32).clip(0.3, 2.5)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[波动率自适应探针 V84.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    BBW Z-Score: {bbw_z[p_i]:.2f}σ, 状态: {'挤压' if is_squeeze[p_i] else ('扩张' if is_expansion[p_i] else '常态')}")
            print(f"    挤压累积(HAB): {squeeze_accum[p_i]:.1f}, 陷阱惩罚: {trap[p_i]:.2f}")
        return final_adj

    def _calculate_sector_overflow_decay(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V84.0 · 熵增雪崩：采用滚动极值归一化，精准识别相对过热，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        hot = raw['THEME_HOTNESS_SCORE_D'].values
        accum_hot = raw['ACCUM_21_THEME_HOTNESS'].values
        accel_hot = raw['ACCEL_5_THEME_HOTNESS_SCORE_D'].values
        # 1. 相对热度归一化 (Relative Hotness)
        # 计算 60 日滚动最大热度，作为当季度的“天花板”
        max_hot = pd.Series(hot, index=df_index).rolling(60).max().replace(0, 1.0).values
        # 相对热度: 当前热度 / 季度峰值
        rel_hot = hot / max_hot
        # 2. 动态雪崩门槛
        # 如果相对热度 > 0.9 (接近历史极值) 且 长期累积热度也很高，则门槛极低
        risk_level = np.where((rel_hot > 0.9) & (accum_hot > 1500.0), "CRITICAL", "NORMAL")
        # 3. 分形熵增判定
        fd_all = _numba_fractal_dimension(np.expand_dims(hot, axis=0).astype(np.float64), window=13)
        fd_vals = fd_all[0].astype(np.float32)
        slope = pd.Series(fd_vals, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        accel_fd = pd.Series(slope, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        # 雪崩条件：(高危状态 OR 绝对热度过高) AND (分形维数加速上升 -> 混乱度激增)
        is_exploding = (accel_hot > 0.5)
        avalanche = ((risk_level == "CRITICAL") | (hot > 80.0) | is_exploding) & (slope > 0) & (accel_fd > 0)
        base = (1.5 / (fd_vals + 1e-9)).clip(0.5, 1.1)
        res = np.where(avalanche, base * 0.5, base).astype(np.float32)
        final_decay = pd.Series(res, index=df_index, dtype=np.float32).clip(0.1, 1.2)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[熵增雪崩自适应探针 V84.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    相对热度: {rel_hot[p_i]:.2f}, 风险等级: {risk_level[p_i]}")
            print(f"    状态: {'雪崩' if avalanche[p_i] else '稳定'}")
        return final_decay

    def _calculate_hmm_regime_confirmation(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V85.0 · HMM 体制确认：采用全要素 Z-Score 自适应归一化，精准对齐体制质心，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 资金流 Z-Score (Flow)
        l_n = raw['SMART_MONEY_HM_NET_BUY_D']
        r_m, r_s = l_n.rolling(21).mean().fillna(0), l_n.rolling(21).std().replace(0, 1e-9)
        f_n = ((l_n - r_m) / r_s).values.astype(np.float32)
        # 2. 成交量 Z-Score (Volume)
        # 原逻辑是相对比率，现统一为 Z-Score 以匹配质心定义
        vol = raw['volume_D']
        v_m, v_s = vol.rolling(21).mean().fillna(0), vol.rolling(21).std().replace(0, 1e-9)
        v_n = ((vol - v_m) / v_s).values.astype(np.float32)
        # 3. 价格 Z-Score (Price)
        # 移除 *10.0 硬编码，使用 pct_change 的 Z-Score
        pct = raw['pct_change_D']
        p_m, p_s = pct.rolling(21).mean().fillna(0), pct.rolling(21).std().replace(0, 1e-6)
        p_n = ((pct - p_m) / p_s).values.astype(np.float32)
        # 4. VWAP 乖离 Z-Score (VWAP Dist)
        # 原逻辑利用 BBW 近似 Std，现直接计算乖离的 Z-Score
        dist = (raw['close_D'] - raw['VWAP_D']) / (raw['VWAP_D'] + 1e-9)
        d_m, d_s = dist.rolling(21).mean().fillna(0), dist.rolling(21).std().replace(0, 1e-6)
        v_d = ((dist - d_m) / d_s).values.astype(np.float32)
        # 5. HMM 概率计算
        # 输入现已全部归一化为 N(0,1)，适配固定质心
        m_p = _numba_hmm_regime_probability(f_n, v_n, p_n, v_d)
        m_p_s = pd.Series(m_p, index=df_index, dtype=np.float32)
        # 6. HAB 修正
        acc_m = raw['ACCUM_21_SMART_MONEY'].values
        regime_bias = np.where(acc_m > 0, 1.15, 0.9).astype(np.float32)
        pos_days = raw['ACCUM_21_POS_SLOPE'].values
        w_l = int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[1])
        realization_ratio = np.clip(pos_days / (float(w_l) * 0.6), 0.7, 1.2).astype(np.float32)
        # 7. 动力学修正
        p_a = m_p_s.diff(3).diff(3).fillna(0).values
        p_j = m_p_s.diff(1).diff(1).diff(1).fillna(0).values
        b_f = np.where(m_p > 0.5, 1.0 + (m_p - 0.5), 0.8 + m_p * 0.4)
        d_b = 1.0 + np.where((m_p > 0.6) & (p_a > 0), 0.2, 0.0) + np.where((m_p > 0.4) & (p_j > 0.1), 0.3, 0.0)
        final_conf = pd.Series(b_f * d_b * regime_bias * realization_ratio, index=df_index, dtype=np.float32)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[HMM 自适应探针 V85.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    输入Z分 -> 资金: {f_n[p_i]:.2f}, 量能: {v_n[p_i]:.2f}, 价格: {p_n[p_i]:.2f}, 乖离: {v_d[p_i]:.2f}")
            print(f"    拉升概率: {m_p[p_i]:.4f}, 最终确认: {final_conf.loc[probe_ts]:.4f}")
        return final_conf

    def _calculate_fractal_efficiency_resonance(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V90.0 · 分形相干效率：引入超线性免疫(Super Trend Immunity)，保护主升浪结构，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        c_vals = raw['close_D'].values.astype(np.float64)
        v_vals = raw['volume_D'].values.astype(np.float64)
        input_arrays = np.vstack((c_vals, v_vals))
        f_all = _numba_fractal_dimension(input_arrays, window=21)
        h_p, h_v = (2.0 - f_all[0]).astype(np.float32), (2.0 - f_all[1]).astype(np.float32)
        h_slope = pd.Series(h_p, index=df_index).diff(5).fillna(0).values.astype(np.float32)
        # 1. 缺口标准化
        raw_gap = np.abs(h_p - h_v)
        gap_s = pd.Series(raw_gap, index=df_index)
        gap_std = gap_s.rolling(21).std().replace(0, 1e-6).values
        gap_z = raw_gap / gap_std
        # 2. 连续性评分
        res_score = np.clip(1.2 - gap_z * 0.2, 0.5, 1.3)
        # 3. 超线性免疫 (Immunity)
        # 如果 Hurst > 0.8，说明价格走势极度线性有序，此时忽略量能扰动带来的缺口
        is_super_trend = (h_p > 0.8)
        if np.any(is_super_trend):
            # 在超强趋势下，Gap Z 即使很大也不扣分，反而视为动能强劲
            res_score = np.where(is_super_trend, 1.2, res_score)
        # 4. HAB 稳态
        w_l = int(raw.get('META_HAB_WINDOWS', pd.Series([13, 21])).iloc[1])
        is_ordered = np.where(h_p > 0.55, 1.0, 0.0).astype(np.float32)
        struct_stab = _numba_rolling_accumulation(is_ordered, w_l)
        stab_bonus = np.clip(struct_stab / 15.0, 0.9, 1.3).astype(np.float32)
        final = res_score * stab_bonus
        final = np.where((h_p > 0.55) & (h_slope > 0), final * 1.2, final)
        final_series = pd.Series(final, index=df_index, dtype=np.float32).clip(0.4, 2.0)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[分形免疫探针 V90.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    Hurst指数: {h_p[p_i]:.4f}, 免疫状态: {'激活' if is_super_trend[p_i] else '关闭'}")
            print(f"    缺口Z分: {gap_z[p_i]:.2f}σ -> 基础分: {res_score[p_i]:.2f}")
            print(f"    >>> 最终分形效率: {final_series.loc[probe_ts]:.4f}")
        return final_series

    def _calculate_chip_lock_efficiency(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V92.0 · 筹码锁定效率：引入趋势缓冲(Trend Buffer)与平缓衰减曲线，防止良性分歧被误杀，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        win = raw['winner_rate_D'].values
        turn = raw['turnover_rate_f_D'].values
        c_s, p_s = raw['cost_50pct_D'].values, raw['close_D'].values
        # 1. 有效获利盘 (保持 V90 逻辑)
        c_50 = np.where(c_s == 0, p_s, c_s)
        is_above_cost = (p_s > c_50)
        eff_win = np.maximum(win, np.where(is_above_cost, 0.6, 0.0)).astype(np.float32)
        # 2. 相对换手率
        turn_median = pd.Series(turn, index=df_index).rolling(21).median().replace(0, 0.01).values
        rel_turn = turn / (turn_median + 1e-9)
        # 3. 趋势缓冲 (Trend Buffer)
        # 如果价格在 21日均线之上，说明趋势完好，高换手容忍度提升
        ma21 = raw['close_D'].rolling(21).mean().values
        is_trend_safe = (raw['close_D'].values > ma21)
        # 缓冲系数: 趋势安全 -> 0.6 (衰减打6折); 趋势破位 -> 1.0 (全额衰减)
        buffer_rate = np.where(is_trend_safe, 0.6, 1.0).astype(np.float32)
        # 承接力进一步豁免
        absorb = raw['absorption_energy_D'].values
        final_decay = np.clip(buffer_rate - np.where(absorb > 0, 0.3, 0.0), 0.3, 1.2)
        # 4. 平缓锁定系数 (Smoother Curve)
        # 使用 Sigmoid 变体替代陡峭的 exp
        # x = rel_turn。当 x=1.8, decay=0.6 时: 
        # (x-1)*decay = 0.8*0.6 = 0.48
        # Score = 2 / (1 + exp(0.48)) = 2 / 2.61 ≈ 0.76 (及格)
        # 相比 V91 的 0.27 大幅改善
        lock_factor = 2.0 / (1.0 + np.exp((rel_turn - 1.0) * 2.0 * final_decay))
        # 限制范围，避免过度奖励缩量
        lock_factor = np.clip(lock_factor, 0.3, 1.5).astype(np.float32)
        s_lock = eff_win * lock_factor
        # 5. 动力学与 HAB
        a_w, a_t, j_w = raw['ACCEL_5_winner_rate_D'].values, raw['ACCEL_5_turnover_rate_f_D'].values, raw['JERK_3_winner_rate_D'].values
        k_bonus = 1.0 + np.where((a_w > 0) & (a_t < 0), 0.3, 0.0) + np.where(j_w > 0.1, 0.2, 0.0)
        high_win_days = raw['ACCUM_13_HIGH_WINNER'].values
        deep_lock_mult = np.where(high_win_days > 10, 1.5, np.where(high_win_days > 5, 1.2, 1.0)).astype(np.float32)
        b_c = is_above_cost.astype(np.float32)
        final_eff = pd.Series(s_lock * k_bonus * deep_lock_mult * (0.8 + b_c * 0.2), index=df_index, dtype=np.float32)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[筹码缓冲探针 V92.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    相对换手: {rel_turn[p_i]:.2f}x, 趋势安全: {is_trend_safe[p_i]}")
            print(f"    衰减力度: {final_decay[p_i]:.2f} -> 锁定系数: {lock_factor[p_i]:.2f}")
            print(f"    >>> 最终筹码效率: {final_eff.loc[probe_ts]:.4f}")
        return final_eff

    def _calculate_microstructure_attack_vector(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V92.0 · 微观矢量：引入良性回调判定(Dip Guard)，保护趋势中的缩量回吐，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        e_n = (raw['buy_elg_amount_D'].values - raw['sell_elg_amount_D'].values)
        l_n = (raw['buy_lg_amount_D'].values - raw['sell_lg_amount_D'].values)
        t_a = (raw['amount_D'].values + 1e-9)
        net_ratio = (e_n + l_n) / t_a
        # 1. 资金流 Z-Score
        ratio_s = pd.Series(net_ratio, index=df_index)
        ratio_mean = ratio_s.rolling(21).mean().fillna(0).values
        ratio_std = ratio_s.rolling(21).std().replace(0, 1e-6).values
        z_score_flow = (net_ratio - ratio_mean) / ratio_std
        base_score = 0.5 + (1.0 / (1.0 + np.exp(-z_score_flow)))
        # 2. 隐蔽吸筹与良性回调 (Stealth & Dip Guard)
        close = raw['close_D'].values
        ma5 = pd.Series(close, index=df_index).rolling(5).mean().values
        ma21 = raw['close_D'].rolling(21).mean().values
        p_jerk = raw['JERK_3_close_D'].values
        # 隐蔽吸筹: 资金负 + 价格强 (Jerk>0) + 站稳短均线
        is_stealth = (net_ratio < 0) & (close > ma5) & (p_jerk > 0)
        #良性回调: 资金微负 (Z > -1.0) + 趋势向上 (Close > MA21) + 价格无崩塌 (Jerk > -2.0)
        is_benign_dip = (z_score_flow > -1.0) & (close > ma21) & (p_jerk > -2.0)
        # 只要满足任一条件，基础分至少给到 1.0 (中性)，避免误杀
        adjusted_base = base_score
        if np.any(is_stealth) or np.any(is_benign_dip):
            guard_mask = (is_stealth | is_benign_dip)
            adjusted_base = np.where(guard_mask, np.maximum(base_score, 1.0), base_score).astype(np.float32)
        # 3. 协同性与动力学
        e_r, l_r = e_n / t_a, l_n / t_a
        sync_score = np.where((e_r > 0) & (l_r > 0), 1.1, np.where((e_r * l_r) < 0, 0.9, 1.0)).astype(np.float32)
        s_j = raw['JERK_3_SMART_MONEY_HM_NET_BUY_D'].values.astype(np.float32)
        jerk_bonus = np.where(s_j > 0.5, 1.2, 1.0).astype(np.float32)
        d_sm = (e_n + l_n).astype(np.float32)
        a13 = raw['ACCUM_13_SMART_MONEY'].values.astype(np.float32)
        h_s = np.where((d_sm < 0) & (a13 > np.abs(d_sm) * 10.0), 1.1, 1.0).astype(np.float32)
        final_v = pd.Series(adjusted_base * sync_score * jerk_bonus * h_s, index=df_index, dtype=np.float32).clip(0.4, 2.0)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[微观护盾探针 V92.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    原始分: {base_score[p_i]:.4f}, 状态: {'隐蔽吸筹' if is_stealth[p_i] else ('良性回调' if is_benign_dip[p_i] else '正常')}")
            print(f"    >>> 最终攻击矢量: {final_v.loc[probe_ts]:.4f}")
        return final_v

    def _calculate_vpa_elasticity_reflexivity(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V83.0 · 流体反身性：采用历史中位数相对归一化，精准识别高敏状态，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        pp = np.abs(raw['pct_change_D'].values)
        pv = np.abs(pd.Series(raw['volume_D'].values, index=df_index).pct_change().fillna(0).values)
        # 1. 瞬时弹性
        current_elasticity = pp / (pv + 0.1)
        # 2. 自适应归一化 (Median Relative)
        # 计算 21 日中位数弹性，作为基准
        ela_median = pd.Series(current_elasticity, index=df_index).rolling(21).median().replace(0, 0.01).values
        # 相对弹性比率 (Relative Elasticity Ratio)
        # Ratio > 1.0 代表当前比平时更"轻"，少量量能即可推动价格
        rel_elasticity = current_elasticity / ela_median
        # 映射分数：关注 Ratio > 1.0 的区间
        score = np.clip(np.tanh(rel_elasticity - 0.8) + 0.5, 0.5, 1.8)
        # 3. HAB 弹性密度修正
        acc_e = raw['ACCUM_13_ELASTICITY'].values
        # 同样对 HAB 进行相对归一化
        hab_score = np.clip(acc_e / (ela_median * 13.0 + 1e-9), 0.8, 1.4).astype(np.float32)
        # 4. 动力学修正
        e_s = pd.Series(current_elasticity, index=df_index).diff(3).values
        slope_bonus = np.where(e_s > 0, 1.2, 1.0).astype(np.float32)
        final = pd.Series(score * slope_bonus * hab_score, index=df_index, dtype=np.float32).clip(0.5, 2.5)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[反身性自适应探针 V83.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    当前弹性: {current_elasticity[p_i]:.2f}, 历史中位: {ela_median[p_i]:.2f}")
            print(f"    相对比率: {rel_elasticity[p_i]:.2f} -> 基础分: {score[p_i]:.2f}")
        return final

    def _calculate_wyckoff_breakout_quality(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V85.0 · 威科夫突破：采用 Jerk Z-Score 自适应归一化，精准识别爆发性突破，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        high, low = raw['high_D'].values, raw['low_D'].values
        close = raw['close_D'].values
        vol = raw['volume_D'].values
        close_prev = raw['close_D'].shift(1).fillna(raw['close_D']).values
        # 1. TR 动力学
        tr = np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
        tr_norm = tr / (close + 1e-9)
        tr_s = pd.Series(tr_norm, index=df_index)
        tr_slope = (tr_s.diff(5) / 5.0).values
        tr_accel = (pd.Series(tr_slope, index=df_index).diff(5) / 5.0).values
        # 2. 价格脉冲自适应归一化 (Price Jerk Z-Score)
        # 原始 Jerk 单位是 $/day^3，受股价绝对值影响极大
        raw_jerk = raw['JERK_3_close_D'].values
        jerk_s = pd.Series(raw_jerk, index=df_index)
        jerk_mean = jerk_s.rolling(21).mean().fillna(0).values
        jerk_std = jerk_s.rolling(21).std().replace(0, 1e-6).values
        # Jerk Z-Score: > 2.0 代表异常猛烈的加速
        jerk_z = (raw_jerk - jerk_mean) / jerk_std
        # 3. 自适应压缩比
        vol_ma = pd.Series(vol, index=df_index).rolling(21).mean().replace(0, 1).values
        tr_ma = pd.Series(tr, index=df_index).rolling(21).mean().replace(0, 1e-9).values
        compression_ratio = np.clip((vol / vol_ma) / (tr / tr_ma + 0.1), 0.5, 3.0).astype(np.float32)
        acc_compression = pd.Series(compression_ratio, index=df_index).rolling(5).mean().values
        # 4. 突破判定
        highest_21 = raw['high_D'].rolling(21).max().shift(1).fillna(99999).values
        is_breakout = close > highest_21
        # 质量判定：突破 + 压缩蓄势 + 脉冲爆发(Z>1.5)
        # 之前使用 jerk > 0.1 是硬编码，现在用 Z > 1.5 自适应
        quality_mult = np.where(acc_compression > 1.2, 1.5, 1.0)
        is_explosive = (jerk_z > 1.5)
        base_q = np.where(is_breakout & is_explosive, 1.6, np.where(is_breakout, 0.9, 0.0))
        # 非突破状态下的蓄势评分
        prep_q = np.where((tr_slope < 0) & (tr_accel > -0.01), 1.1, 0.9)
        quality = np.maximum(base_q, prep_q)
        final_score = pd.Series(quality * np.where((tr_slope < 0), 1.3, 1.0) * quality_mult, index=df_index, dtype=np.float32).clip(0.6, 2.5)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[威科夫自适应探针 V85.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    Jerk Z-Score: {jerk_z[p_i]:.2f}σ (原值: {raw_jerk[p_i]:.4f})")
            print(f"    状态: {'爆发突破' if is_breakout[p_i] and is_explosive[p_i] else ('普通突破' if is_breakout[p_i] else '蓄势')}")
            print(f"    >>> 最终突破质量: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_trend_inertia_momentum(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V83.0 · 趋势运动学：采用标准化动力学归一化，精准定义加速状态，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 均线排列 (保持原逻辑，因其为状态量)
        ma5 = raw['close_D'].rolling(5).mean().values
        ma21 = raw['close_D'].rolling(21).mean().values
        ma55 = raw['close_D'].rolling(55).mean().values
        alignment = np.where((ma5 > ma21) & (ma21 > ma55), 1.0, 0.8).astype(np.float32)
        # 2. 标准化动力学 (Standardized Kinematics)
        s_vals = raw['SLOPE_5_close_D'].values
        a_vals = raw['ACCEL_5_close_D'].values
        j_vals = raw['JERK_3_close_D'].values
        # 计算各阶导数的历史标准差
        s_std = pd.Series(s_vals, index=df_index).rolling(21).std().replace(0, 1e-6).values
        a_std = pd.Series(a_vals, index=df_index).rolling(21).std().replace(0, 1e-6).values
        # 标准化值 (Sigma)
        norm_s = s_vals / s_std
        norm_a = a_vals / a_std
        # 3. 状态定义
        # 强趋势: 速度 > 1.0σ 且 加速度 > 0.5σ
        is_strong_trend = (norm_s > 1.0) & (norm_a > 0.5)
        # 抛物线加速: 加速度 > 1.5σ 且 Jerk > 0
        is_parabolic = (norm_a > 1.5) & (j_vals > 0)
        # 4. 评分映射
        kinematic_score = 1.0
        kinematic_score += np.where(is_strong_trend, 0.3, 0.0)
        kinematic_score += np.where(is_parabolic, 0.4, 0.0)
        kinematic_score -= np.where((norm_s > 0) & (norm_a < -1.0), 0.3, 0.0) # 减速惩罚
        # 5. HAB 韧性修正
        pos_days = raw['ACCUM_21_POS_SLOPE'].values
        consistency_bonus = np.clip(pos_days / 15.0, 0.8, 1.25).astype(np.float32)
        r2_vals = raw['GEOM_REG_R2_D'].values
        final_inertia = pd.Series(alignment * kinematic_score * consistency_bonus * (0.6 + np.clip(r2_vals, 0.0, 1.0) * 0.4), index=df_index, dtype=np.float32).clip(0.6, 2.0)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[运动学自适应探针 V83.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    标准化速度: {norm_s[p_i]:.2f}σ, 标准化加速: {norm_a[p_i]:.2f}σ")
            print(f"    状态: {'抛物线加速' if is_parabolic[p_i] else ('强趋势' if is_strong_trend[p_i] else '常态')}")
        return final_inertia

    def _calculate_market_permeability_index(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V82.0 · 市场渗透率：采用历史分位数归一化，精准定位情绪周期，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        sent = raw['market_sentiment_score_D'].values
        accel = raw['ACCEL_5_market_sentiment_score_D'].values
        # 1. 历史分位数归一化 (Rank Percentile Normalization)
        # 计算当前情绪在过去 21 天中的排名百分比 (0~1)
        # 这种方式能自动适应牛熊市不同的情绪基准
        sent_series = pd.Series(sent, index=df_index)
        sent_rank = sent_series.rolling(21).apply(lambda x: (x.argsort().argsort()[-1] / (len(x) - 1)) if len(x) > 1 else 0.5, raw=True).values
        # 2. 相对极值判断
        # 分位数 < 0.2: 相对冰点；分位数 > 0.8: 相对高潮
        perm = np.where((sent_rank < 0.2) & (accel > 0), 1.3, np.where((sent_rank > 0.8) & (accel > 0), 0.7, 1.0)).astype(np.float32)
        # 3. HAB 饱和度调节
        acc_sent = raw['ACCUM_13_SENTIMENT'].values
        # 同样对累积情绪进行相对归一化
        acc_sent_median = pd.Series(acc_sent, index=df_index).rolling(21).median().values
        saturation_decay = np.where(acc_sent > acc_sent_median * 1.5, 0.8, 1.0).astype(np.float32)
        rank_val, jerk = raw['industry_strength_rank_D'].values, raw['JERK_3_industry_strength_rank_D'].values
        bonus = np.where(jerk < -2.0, 1.3, np.where(rank_val < 0.1, 1.1, 0.9)).astype(np.float32)
        final_ctx = pd.Series(perm * bonus * saturation_decay, index=df_index, dtype=np.float32).clip(0.5, 1.8)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[渗透率自适应探针 V82.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    情绪分位数: {sent_rank[p_i]:.2f}, 相对状态: {'冰点复苏' if sent_rank[p_i] < 0.2 else ('过热' if sent_rank[p_i] > 0.8 else '常态')}")
        return final_ctx

    def _calculate_entry_accessibility_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V86.0 · 入场可获得性：设定 0.2 保底阈值，防止极端情况下得分为零，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        turnover = raw['turnover_rate_f_D'].values
        # 1. 相对流动性
        turn_ma = pd.Series(turnover, index=df_index).rolling(21).mean().replace(0, 0.1).values
        rel_liq = turnover / turn_ma
        # 2. 流动性评分 (保底 0.4)
        base_access = np.where(rel_liq < 0.5, 0.4 + rel_liq * 0.5,
                      np.where(rel_liq <= 2.5, 1.0,
                               np.clip(2.5/rel_liq, 0.5, 1.0)))
        # 3. 封板硬约束 (保底 0.2)
        up_limit = raw['up_limit_D'].values
        close = raw['close_D'].values
        intensity = raw['closing_flow_intensity_D'].values
        sealing = np.clip(intensity, 0.0, 1.0)
        # 如果封死涨停，得分为 0.2 (仍有机会排板成交，不完全杀死)，否则为 1.0
        limit_penalty = np.where(close >= up_limit * 0.999, 0.2 + 0.3 * (1.0 - sealing), 1.0).astype(np.float32)
        # 4. 抢筹拥堵
        tc_accel = raw['ACCEL_5_trade_count_D'].values
        lock_days = raw['ACCUM_5_LIQUIDITY_LOCK'].values
        congestion = np.where((tc_accel > 0.1) & (lock_days >= 3), 0.6, 1.0).astype(np.float32)
        final_access = pd.Series(base_access * limit_penalty * congestion, index=df_index, dtype=np.float32).clip(0.1, 1.2)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[入场保底探针 V86.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    相对流动性: {rel_liq[p_i]:.2f}, 封板惩罚: {limit_penalty[p_i]:.2f}")
            print(f"    >>> 最终可获得性: {final_access.loc[probe_ts]:.4f}")
        return final_access

    def _calculate_entropic_ordering_bonus(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V84.0 · 熵减有序性：采用有序流 Z-Score 自适应归一化，精准度量负熵做功，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        s5 = raw['SLOPE_5_chip_entropy_D'].values
        a5 = raw['ACCEL_5_chip_entropy_D'].values
        j3 = raw['JERK_3_chip_entropy_D'].values
        pct = raw['pct_change_D'].values
        # 1. 有序化力度合成 (Ordering Force)
        # 负斜率和负加速度代表有序度增加
        locking_force = -(s5 * 0.7 + a5 * 0.3)
        # 2. 自适应归一化 (Z-Score)
        # 计算锁定力度的历史标准差，衡量当前的有序化是否“异常强烈”
        force_s = pd.Series(locking_force, index=df_index)
        force_std = force_s.rolling(21).std().replace(0, 1e-6).values
        force_z = locking_force / force_std
        # 仅当有序化力度显著强于历史 (>1.0σ) 时给予奖励
        base_bonus = np.clip(np.tanh(force_z - 1.0) + 0.5, 0.0, 1.5)
        # 3. 脉冲修正
        # Jerk < -0.01 且 Z-Score 很高 => 瞬间强力锁仓
        jerk_bonus = np.where((j3 < -0.01) & (force_z > 1.5), 1.3, 1.0).astype(np.float32)
        # 4. HAB 稳定性修正 (相对比率)
        ent_stab = raw['ACCUM_21_ENTROPY_STABILITY'].values
        # 稳定性比率: 21天里有多少天是熵减的 / 21
        stab_ratio = ent_stab / 21.0
        stability_mult = np.clip(stab_ratio * 2.0, 0.8, 1.4).astype(np.float32)
        penalty = np.where((pct > 0) & (s5 > 0), 0.7, 1.0).astype(np.float32)
        final_factor = pd.Series((1.0 + base_bonus) * jerk_bonus * stability_mult * penalty, index=df_index, dtype=np.float32).clip(0.6, 2.5)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[熵减自适应探针 V84.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    锁定力度Z分: {force_z[p_i]:.2f}σ, 稳定性占比: {stab_ratio[p_i]*100:.1f}%")
            print(f"    >>> 最终有序性系数: {final_factor.loc[probe_ts]:.4f}")
        return final_factor

    def _calculate_vwap_propulsion_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V83.0 · VWAP 推进力：采用 Z-Score 波动率自适应归一化，移除硬编码缩放，清除空行"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        close = raw['close_D'].values
        vwap = raw['VWAP_D'].values
        # 1. 基础速度 (Slope)
        vwap_slope = pd.Series(vwap, index=df_index).diff(3).values / 3.0
        # 2. 自适应归一化 (Adaptive Normalization)
        # 计算斜率的波动率 (21日标准差)
        slope_std = pd.Series(vwap_slope, index=df_index).rolling(21).std().replace(0, 1e-6).values
        # 标准化推进力 (Standardized Propulsion): 表示当前斜率是历史波动率的多少倍
        # > 1.0 代表显著推进，> 2.0 代表极端加速
        norm_propulsion = np.clip(vwap_slope / slope_std, -3.0, 3.0)
        # 映射到 0~1.5 分值，仅关注正向推进
        propulsion_score = np.where(norm_propulsion > 0, np.tanh(norm_propulsion * 0.8) * 1.5, 0.0)
        # 3. 乖离率修正 (保持相对比例)
        bias = (close - vwap) / (vwap + 1e-9)
        bias_penalty = np.where(bias > 0.08, 0.8, 1.0) # 乖离过大给予惩罚
        # 4. 动力学增强 (Consensus Acceleration)
        accel_vwap = raw['ACCEL_5_VWAP_D'].values
        kinematic_boost = np.where(accel_vwap > 0, 1.3, 1.0).astype(np.float32)
        # 5. HAB 趋势厚度
        days_above = raw['ACCUM_21_ABOVE_VWAP'].values
        thickness_bonus = np.clip(days_above / 10.0, 0.8, 1.4).astype(np.float32)
        final_score = pd.Series(propulsion_score * bias_penalty * kinematic_boost * thickness_bonus, index=df_index, dtype=np.float32).clip(0, 2.5)
        if is_debug and probe_ts in df_index:
            p_i = df_index.get_loc(probe_ts)
            print(f"\n[VWAP 自适应探针 V83.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    原始斜率: {vwap_slope[p_i]:.5f}, 历史波动(Std): {slope_std[p_i]:.5f}")
            print(f"    标准化推进: {norm_propulsion[p_i]:.2f}σ (Sigma)")
            print(f"    >>> 最终推进力: {final_score.loc[probe_ts]:.4f}")
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

