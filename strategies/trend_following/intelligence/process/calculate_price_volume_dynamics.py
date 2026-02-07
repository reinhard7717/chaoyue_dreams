# strategies/trend_following/intelligence/process/calculate_price_volume_dynamics.py
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
    n = len(flows[0])
    out = np.ones(n) * 1.5
    scales_log = np.log(np.array([1.0, 3.0, 5.0, 10.0, 21.0]))
    for i in range(window, n):
        flucts = []
        for j in range(len(flows)):
            slice_data = flows[j][i-window:i]
            std_val = np.std(slice_data)
            if std_val > 0: flucts.append(std_val)
        if len(flucts) >= 3:
            flucts_log = np.log(np.array(flucts[:len(scales_log)]))
            # Linear regression for slope
            A = np.vstack((scales_log[:len(flucts)], np.ones(len(flucts)))).T
            slope, _ = np.linalg.lstsq(A, flucts_log, rcond=-1)[0]
            out[i] = 2.0 - slope
    return out
@jit(nopython=True)
def _numba_hmm_states(flow, volume, price):
    n = len(flow)
    states = np.zeros(n)
    for i in range(1, n):
        f, v, p = flow[i], volume[i], price[i]
        if f > 0.5 and p < 0 and v > 0: states[i] = 1 # Accumulation
        elif f > 0.8 and p > 0.5 and v > 0.5: states[i] = 2 # Markup
        elif f < -0.5 and p > 0.3 and v > 0.8: states[i] = 3 # Distribution
        else: states[i] = states[i-1]
    return states
@jit(nopython=True)
def _numba_heat_conduction(u, alpha=0.1, steps=3):
    n = len(u)
    res = np.copy(u)
    for _ in range(steps):
        u_old = np.copy(res)
        for i in range(1, n-1):
            res[i] = u_old[i] + alpha * (u_old[i+1] - 2*u_old[i] + u_old[i-1])
    return res
@jit(nopython=True)
def _numba_flow_distribution_analysis(elg, lg, md, sm, close, n):
    # V25.0 Numba加速: 资金流层级与空间分布分析
    elg_dom = np.zeros(n)
    large_sync = np.zeros(n)
    ret_lg_div = np.zeros(n)
    spatial_bal = np.zeros(n)
    total_flow = elg + lg + md + sm
    for i in range(n):
        abs_total = np.abs(total_flow[i]) + 1e-9
        elg_dom[i] = np.abs(elg[i]) / abs_total
        # 大单协同
        if elg[i] * lg[i] > 0: large_sync[i] = 1.0
        elif np.abs(elg[i]) < 1e-6 or np.abs(lg[i]) < 1e-6: large_sync[i] = 0.5
        else: large_sync[i] = 0.0
        # 散户vs主力背离
        large_sum = elg[i] + lg[i]
        retail_sum = md[i] + sm[i]
        if large_sum * retail_sum < 0: ret_lg_div[i] = 1.0
        elif np.abs(large_sum) < 1e-6 or np.abs(retail_sum) < 1e-6: ret_lg_div[i] = 0.5
        else: ret_lg_div[i] = 0.0
    # 空间平衡度 (简化版: 价格分位分析在Numba中较繁琐，此处用价格与流的相关性近似)
    for i in range(30, n):
        f_win = total_flow[i-20:i]
        p_win = close[i-20:i]
        std_f = np.std(f_win)
        std_p = np.std(p_win)
        if std_f > 0 and std_p > 0:
            spatial_bal[i] = 1.0 - np.std(f_win) / (np.mean(np.abs(f_win)) + 1e-9) # 简化的流平衡指标
        else:
            spatial_bal[i] = 0.5
    return elg_dom, large_sync, ret_lg_div, spatial_bal
@jit(nopython=True)
def _numba_stochastic_process(flow, window=21):
    # V25.0 Numba加速: 随机过程特性分析
    n = len(flow)
    martingale = np.zeros(n)
    predictability = np.zeros(n)
    for i in range(window, n):
        w_data = flow[i-window:i]
        if len(w_data) < window/2: continue
        diffs = np.diff(w_data)
        if len(diffs) > 0:
            mean_inc = np.mean(diffs)
            std_inc = np.std(diffs)
            if std_inc == 0: std_inc = 1.0
            martingale[i] = 1.0 - min(np.abs(mean_inc / std_inc), 2.0) / 2.0
        # 自相关性近似预测性
        if len(w_data) >= 5:
            ac_sum = 0.0
            for lag in range(1, 4):
                c = np.corrcoef(w_data[:-lag], w_data[lag:])[0, 1]
                if not np.isnan(c): ac_sum += np.abs(c)
            predictability[i] = min(ac_sum / 3.0, 1.0)
    return martingale * 0.6 + np.clip(predictability, 0.3, 0.6) * 0.4
@jit(nopython=True)
def _numba_network_queuing_game(buy, sell, price_chg, trade_cnt, vol, amt, sentiment, window=13):
    # V25.0 Numba加速: 网络流、排队论与博弈学习综合计算
    n = len(buy)
    net_eff = np.zeros(n)
    queue_eff = np.zeros(n)
    game_learn = np.zeros(n)
    net_flow = buy - sell
    for i in range(window, n):
        # 网络流效率
        b_w = buy[i-window:i]
        s_w = sell[i-window:i]
        p_w = price_chg[i-window:i]
        if np.std(b_w) > 0 and np.std(s_w) > 0:
            corr = np.corrcoef(b_w, s_w)[0, 1]
            corr_score = 1.0 - np.abs(corr) if not np.isnan(corr) else 0.0
        else: corr_score = 0.0
        net_eff[i] = corr_score # 简化计算
        # 排队效率
        tc_w = trade_cnt[i-window:i]
        amt_w = amt[i-window:i]
        if np.sum(tc_w) > 0:
            arr_rate = np.mean(tc_w)
            avg_size = np.mean(amt_w / (tc_w + 1e-9))
            srv_rate = np.mean(amt_w) / (avg_size * window + 1e-9)
            intensity = arr_rate / srv_rate if srv_rate > 0 else 0.0
            if intensity < 0.7: queue_eff[i] = intensity / 0.7
            elif intensity < 1.0: queue_eff[i] = 1.0
            else: queue_eff[i] = max(0.0, 1.3 - intensity)
        # 博弈学习 (基于滞后相关性)
        if i > window * 2:
            tr_flow = net_flow[i-window*2:i-window]
            fut_p = price_chg[i-window*2+3:i-window+3] # 简化的未来收益
            if len(tr_flow) == len(fut_p): # 确保长度一致
                 corr_learn = np.corrcoef(tr_flow, fut_p)[0, 1]
                 game_learn[i] = (corr_learn + 1) / 2 if not np.isnan(corr_learn) else 0.5
    return net_eff, queue_eff, game_learn
@jit(nopython=True)
def _numba_cnn_pattern_sim(price, vol, eff, window=13):
    # V25.0 Numba加速: 模拟CNN卷积模式识别
    n = len(price)
    out = np.zeros(n)
    # 简单的卷积核
    k_trend = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    k_rev = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
    for i in range(window, n):
        p_w = price[i-5:i]
        v_w = vol[i-5:i]
        # 归一化
        p_std = np.std(p_w)
        v_std = np.std(v_w)
        if p_std == 0 or v_std == 0: continue
        p_n = (p_w - np.mean(p_w)) / p_std
        v_n = (v_w - np.mean(v_w)) / v_std
        # 卷积
        conv_p = np.sum(p_n * k_trend)
        conv_rev = np.sum(p_n * k_rev)
        out[i] = np.tanh(conv_p) * 0.6 + np.tanh(np.abs(conv_rev)) * 0.4
    return out
@jit(nopython=True)
def _numba_vpa_mechanisms(price_chg, vol_chg, eff_chg, window=8):
    # V25.0 Numba加速: VPA注意力与长期依赖
    n = len(price_chg)
    attn_scores = np.zeros(n)
    dep_scores = np.zeros(n)
    for i in range(window, n):
        # Attention
        pc = np.abs(price_chg[i-window:i])
        vc = np.abs(vol_chg[i-window:i])
        attn = (pc + vc) / (np.mean(pc) + np.mean(vc) + 1e-9)
        weights = np.exp(attn) / np.sum(np.exp(attn))
        attn_scores[i] = np.sum(eff_chg[i-window:i] * weights)
        # Dependency (Autocorr decay)
        w_eff = eff_chg[i-window:i]
        if len(w_eff) > 2:
            ac1 = np.corrcoef(w_eff[:-1], w_eff[1:])[0, 1]
            dep_scores[i] = 0.5 + 0.5 * (ac1 if not np.isnan(ac1) else 0)
    return np.tanh(attn_scores), dep_scores
@jit(nopython=True)
def _numba_market_regime_scan(close, volume, rsi, adx, n):
    # V25.0 Numba加速: 市场机制识别
    trend_mk = np.zeros(n)
    range_mk = np.zeros(n)
    break_mk = np.zeros(n)
    rev_mk = np.zeros(n)
    for i in range(21, n):
        # 趋势
        ret_20 = close[i]/close[i-20] - 1
        if adx[i] > 25 and np.abs(ret_20) > 0.1: trend_mk[i] = 1.0
        # 震荡
        mx = np.max(close[i-20:i])
        mn = np.min(close[i-20:i])
        if adx[i] < 20 and (mx-mn)/mn < 0.15: range_mk[i] = 1.0
        # 突破
        vol_r = volume[i] / np.mean(volume[i-20:i])
        if (close[i] > mx*1.02 or close[i] < mn*0.98) and vol_r > 1.5: break_mk[i] = 1.0
        # 反转
        if (rsi[i]>70 or rsi[i]<30) and i>5:
             # 简单判断变盘
             if (close[i]/close[i-5]-1) * (close[i-5]/close[i-10]-1) < 0: rev_mk[i] = 1.0
    return trend_mk, range_mk, break_mk, rev_mk
@jit(nopython=True)
def _numba_trend_dynamics(close, volume, n):
    # V25.0 Numba加速: 动态系统与拓扑分析
    attractor = np.zeros(n)
    topo = np.zeros(n)
    for i in range(34, n):
        # 简易Lyapunov (Attractor)
        p_win = close[i-10:i]
        if np.std(p_win) > 0:
            # 轨道分离率近似
            div = np.abs(np.diff(p_win))
            lya = np.mean(np.log(div + 1e-9))
            attractor[i] = 1.0 / (1.0 + np.abs(lya))
        else: attractor[i] = 0.5
        # 拓扑持续性 (局部极值规律)
        # 简化：统计极值点间隔的标准差
        p_w21 = close[i-21:i]
        diffs = np.diff(np.sign(np.diff(p_w21)))
        extrema_indices = np.where(diffs != 0)[0]
        if len(extrema_indices) > 2:
            intervals = np.diff(extrema_indices)
            reg = 1.0 - np.std(intervals)/(np.mean(intervals)+1e-9)
            topo[i] = max(0.0, reg)
        else: topo[i] = 0.5
    return attractor, topo
@jit(nopython=True)
def _numba_complex_systems(close, volume, sentiment, n):
    # V25.0 Numba加速: 复杂系统临界性与适应性
    criticality = np.zeros(n)
    adaptation = np.zeros(n)
    for i in range(34, n):
        p_w = close[i-34:i]
        # 幂律检测 (简化: 价格变化的分布尾部)
        changes = np.abs(np.diff(p_w))
        if np.max(changes) > 0:
            # 简单估算：大波动占比
            extreme_ratio = np.sum(changes > 2*np.std(changes)) / len(changes)
            if 0.05 < extreme_ratio < 0.15: criticality[i] = 1.0 # 临界状态特征
            else: criticality[i] = 0.5
        # 适应性 (环境复杂度 vs 可预测性)
        vol_w = np.std(p_w) / (np.mean(p_w)+1e-9)
        # 简单自相关作为可预测性
        ac = 0.0
        if len(p_w) > 5:
            ac = np.abs(np.corrcoef(p_w[:-1], p_w[1:])[0, 1])
        adaptation[i] = np.tanh(ac / (vol_w + 1e-9))
    return criticality, adaptation
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
        """V50.0 · 核心信号校验：修复题材热度动力学 (THEME_HOTNESS_DYNAMICS) 校验遗漏"""
        fib_windows = [3, 5, 8, 13, 21]
        # 核心修复：将 THEME_HOTNESS_SCORE_D 正式列为动力学基础目标
        dynamic_base_cols = ['net_amount_rate_D', 'winner_rate_D', 'SMART_MONEY_HM_NET_BUY_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'THEME_HOTNESS_SCORE_D']
        base_required = [
            'close_D', 'volume_D', 'amount_D', 'net_amount_rate_D', 'winner_rate_D',
            'up_limit_D', 'down_limit_D', 'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D',
            'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'pressure_release_index_D', 'BBW_21_2.0_D', 
            'VPA_EFFICIENCY_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'turnover_rate_f_D',
            'IS_ROUNDING_BOTTOM_D', 'IS_GOLDEN_PIT_D', 'IS_TRENDING_STAGE_D', 'price_percentile_position_D',
            'TURNOVER_STABILITY_INDEX_D', 'IS_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 'THEME_HOTNESS_SCORE_D'
        ]
        dynamic_required = [f"{p}_{w}_{c}" for c in dynamic_base_cols for w in fib_windows for p in ['SLOPE', 'ACCEL', 'JERK']]
        all_required = base_required + dynamic_required
        return self.helper._validate_required_signals(df, all_required, method_name)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Dict[str, pd.Series]:
        """V50.0 · 原料加载层：修复题材热度及其衍生项加载逻辑并执行 MAD 鲁棒清洗"""
        raw_signals = {}
        base_cols = ['close_D', 'volume_D', 'amount_D', 'pct_change_D', 'net_amount_rate_D', 'trade_count_D', 'turnover_rate_f_D']
        struct_cols = ['winner_rate_D', 'chip_concentration_ratio_D', 'chip_entropy_D', 'cost_50pct_D', 'absorption_energy_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'price_percentile_position_D']
        tech_cols = [
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 
            'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D', 'pressure_release_index_D', 'up_limit_D', 
            'down_limit_D', 'closing_flow_ratio_D', 'TURNOVER_STABILITY_INDEX_D', 'IS_EMOTIONAL_EXTREME_D', 
            'flow_consistency_D', 'industry_strength_rank_D', 'industry_rank_accel_D', 'IS_ROUNDING_BOTTOM_D', 
            'IS_GOLDEN_PIT_D', 'IS_TRENDING_STAGE_D', 'THEME_HOTNESS_SCORE_D'
        ]
        # 核心修复：添加 THEME_HOTNESS_SCORE_D 至动力学清洗目标
        dynamic_targets = ['net_amount_rate_D', 'winner_rate_D', 'SMART_MONEY_HM_NET_BUY_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'THEME_HOTNESS_SCORE_D']
        fib_windows = [3, 5, 8, 13, 21]
        for col in base_cols + struct_cols + tech_cols:
            if col not in df.columns:
                raise KeyError(f"CRITICAL: 军械库缺失关键列 {col}，请检查数据层产出")
            raw_signals[col] = df[col].ffill().fillna(0.0)
        for col in dynamic_targets:
            for win in fib_windows:
                for prefix in ['SLOPE', 'ACCEL', 'JERK']:
                    dyn_col = f"{prefix}_{win}_{col}"
                    if dyn_col not in df.columns:
                        # 降级处理：若缺失则补0，防止 KeyError 中断引擎
                        raw_signals[dyn_col] = pd.Series(0.0, index=df.index)
                        continue
                    d_series = df[dyn_col].fillna(0.0).copy()
                    median = d_series.median()
                    mad = (d_series - median).abs().median()
                    threshold = 5.0 * (mad * 1.4826 + 1e-9)
                    raw_signals[dyn_col] = d_series.clip(lower=median - threshold, upper=median + threshold)
        return raw_signals

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V53.0 · 主力夺权全局总线：集成五层安全阀调节矩阵与入场可获得性修正"""
        method_name = "calculate_price_volume_dynamics"
        df_index = df.index
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        if not self._validate_all_required_signals(df, {}, {}, method_name, is_debug, probe_ts):
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        raw = self._get_raw_signals(df, method_name)
        # 1. 计算物理引擎分与几何势能分（基础强度层）
        physical_base = self._calculate_power_transfer_raw_score(df_index, raw, method_name)
        geo_conf = (1.0 - raw['GEOM_REG_R2_D']).clip(0, 1)
        act_curv = pd.Series(_numba_power_activation(raw['GEOM_ARC_CURVATURE_D'].values, gain=2.0), index=df_index)
        geo_resonance = act_curv * geo_conf * (raw['IS_ROUNDING_BOTTOM_D'].astype(float) * 1.2 + 0.5)
        # 2. 合成未修正的综合强度
        unadjusted_intensity = (physical_base * 0.70 + geo_resonance * 0.30)
        # 3. 五层安全阀调节矩阵（风险控制层）
        risk_v = self._calculate_premium_reversal_risk(raw, df_index, method_name)
        decay_v = self._calculate_intraday_decay_model(raw, df_index, method_name)
        sector_v = self._calculate_sector_resonance_modifier(raw, df_index, method_name)
        vol_v = self._calculate_volatility_clustering_adjustment(raw, df_index, method_name)
        sector_overflow_v = self._calculate_sector_overflow_decay(raw, df_index, method_name)
        # 4. 实战修正与链式乘法合成
        access_f = self._calculate_entry_accessibility_score(raw, df_index, method_name)
        final_score = unadjusted_intensity * risk_v * decay_v * sector_v * vol_v * sector_overflow_v * access_f
        if is_debug and probe_ts in df_index:
            print(f"\n[PROCESS_META_POWER_TRANSFER 全链集成探针 V53 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    [核心] 物理基准: {physical_base.loc[probe_ts]:.4f}, 几何共振: {geo_resonance.loc[probe_ts]:.4f}")
            print(f"    [阀门] 情绪: {risk_v.loc[probe_ts]:.2f}, 衰减: {decay_v.loc[probe_ts]:.2f}, 板块溢出: {sector_overflow_v.loc[probe_ts]:.2f}")
            print(f"    >>> 最终输出分值: {final_score.loc[probe_ts]:.4f}")
        return final_score.clip(-3.5, 6.0).astype(np.float32)

    def _calculate_power_transfer_raw_score(self, df_index: pd.Index, raw: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V53.0 · 物理动力引擎：集成共识速度与量纲压缩后的竞价预判"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        score_calculate = CalculatePowerTransferRawScore(is_debug, probe_ts)
        # 1. 动力学去噪与冲量激活
        vol_adj = raw['BBW_21_2.0_D'].fillna(0.1).values
        rolling_tc = raw['trade_count_D'].rolling(21).mean().replace(0, 1)
        conf = (raw['trade_count_D'] / rolling_tc).fillna(1.0).values
        j_c = _numba_adaptive_denoise_dynamics(raw['JERK_3_net_amount_rate_D'].fillna(0).values, vol_adj, conf)
        a_c = _numba_adaptive_denoise_dynamics(raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'].fillna(0).values, vol_adj, conf)
        act_impulse = pd.Series(_numba_power_activation((j_c * 0.45 + a_c * 0.55), gain=1.8), index=df_index)
        # 2. 标准化与补偿（物理修正）
        norm_impulse = score_calculate._calculate_dynamic_impulse_norm(act_impulse, raw, df_index, method_name)
        comp_impulse = score_calculate._calculate_limit_price_compensation(norm_impulse, raw, df_index, method_name)
        # 3. T+1 竞价预判（已在内部执行量纲压缩）
        auc_pred = score_calculate._calculate_auction_prediction(raw, df_index, method_name)
        # 4. MCV 多尺度共识速度
        fib_wins = np.array([3, 5, 8, 13, 21], dtype=np.int64)
        _, f_slopes = _numba_fast_rolling_dynamics(raw['net_amount_rate_D'].values, fib_wins)
        mcv_consensus = pd.Series(np.dot(np.array([0.35, 0.25, 0.20, 0.10, 0.10]), f_slopes), index=df_index)
        # 5. 物理引擎合成（权重微调：提升持续性占比）
        physical_engine_score = (comp_impulse * 8.0 * 0.30 + auc_pred * 0.30 + mcv_consensus * 0.40)
        if is_debug and probe_ts in df_index:
            print(f"\n[物理引擎量纲审计 V53 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    补偿后冲量: {comp_impulse.loc[probe_ts]:.6f}, 竞价压缩分: {auc_pred.loc[probe_ts]:.4f}")
            print(f"    MCV共识速度: {mcv_consensus.loc[probe_ts]:.4f}, 物理合成总分: {physical_engine_score.loc[probe_ts]:.4f}")
        return physical_engine_score.astype(np.float32)

    def _calculate_premium_reversal_risk(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V35.0 · 溢价回吐风险预判：基于情绪透支与换手率能效比模型"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 核心逻辑：若T日尾盘占比过高且处于情绪极端，T+1容易出现开盘脉冲后的衰竭
        # 换手率归一化因子 (假设换手率 > 15% 为高能耗区)
        exhaustion_factor = (raw['turnover_rate_f_D'] / 15.0).clip(0, 1.5)
        # 溢价回吐压力 = 尾盘占比 * 情绪极端因子 * 换手能耗
        reversal_pressure = raw['closing_flow_ratio_D'] * raw['IS_EMOTIONAL_EXTREME_D'].astype(float) * exhaustion_factor
        # 转化为风险调节系数 (0.6 代表 40% 的衰减，1.0 代表无衰减)
        risk_adjustment = (1.0 - reversal_pressure * 0.4).clip(0.6, 1.0)
        if is_debug and probe_ts in df_index:
            print(f"\n[溢价回吐风险探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    情绪极端: {raw['IS_EMOTIONAL_EXTREME_D'].loc[probe_ts]}, 自由换手: {raw['turnover_rate_f_D'].loc[probe_ts]:.2f}%")
            print(f"    尾盘占比: {raw['closing_flow_ratio_D'].loc[probe_ts]:.4f}, 能耗因子: {exhaustion_factor.loc[probe_ts]:.4f}")
            print(f"    >>> 最终风险调节系数: {risk_adjustment.loc[probe_ts]:.4f}")
        return risk_adjustment.astype(np.float32)

    def _calculate_intraday_decay_model(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V53.0 · 日内衰减模型：修复语法错误并优化烂板修复逻辑"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        stability = raw['TURNOVER_STABILITY_INDEX_D'].fillna(0.5).clip(0, 1)
        is_limit_up = (raw['close_D'] >= raw['up_limit_D'] * 0.999)
        # 1. 基础衰减逻辑
        bad_board_mask = is_limit_up & (raw['closing_flow_ratio_D'] > 0.4) & (stability < 0.4)
        # 2. 修复逻辑
        winner_rate = raw['winner_rate_D'].fillna(0.5)
        repair_potential = np.where((winner_rate < 0.15) & (stability < 0.3), 1.5, 1.0)
        # 3. 结果合成
        decay_resistance = (0.6 + stability * 0.4) * np.where(bad_board_mask, 0.6, 1.0) * repair_potential
        res = pd.Series(decay_resistance, index=df_index).clip(0.3, 1.5)
        return res.astype(np.float32)

    def _calculate_sector_resonance_modifier(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V53.0 · 板块共振算子：移除干扰语法并校准持续性判定"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 动力学冲量
        sector_impulse = (raw['SLOPE_5_THEME_HOTNESS_SCORE_D'] * 0.6 + raw['ACCEL_5_THEME_HOTNESS_SCORE_D'] * 0.4).fillna(0)
        # 2. 持续性校验
        persistence_factor = np.where((raw['industry_rank_accel_D'] > 0) & (raw['flow_consistency_D'] > 0.65), 1.2, 0.8)
        # 3. 调节器合成
        resonance_mod = (1.0 + _numba_power_activation(sector_impulse.values, gain=0.5)) * persistence_factor
        return pd.Series(resonance_mod, index=df_index).clip(0.6, 1.8).astype(np.float32)

    def _calculate_volatility_clustering_adjustment(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V49.0 · 波动率聚集调节：识别“二次加速”与“波动率陷阱”"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        bbw = raw['BBW_21_2.0_D']
        # 利用 5 日斜率与加速度识别波动率惯性
        slope_5 = raw['SLOPE_5_BBW_21_2.0_D']
        accel_5 = raw['ACCEL_5_BBW_21_2.0_D']
        bbw_ma21 = bbw.rolling(21).mean().fillna(bbw)
        # 1. 二次加速判定 (Secondary Acceleration)
        # 逻辑：BBW 处于低位发散状态（斜率 > 0 且 加速度 > 0）代表进入良性扩张期
        is_acceleration = (bbw < bbw_ma21 * 1.2) & (slope_5 > 0) & (accel_5 > 0)
        accel_gain = np.where(is_acceleration, 1.0 + (slope_5 * 2.5 + accel_5 * 1.5).clip(0, 0.4), 1.0)
        # 2. 波动率陷阱判定 (Volatility Trap)
        # 逻辑：BBW 处于高位（> 均值 1.5 倍）且斜率转负，预示高位动能耗竭
        is_vol_trap = (bbw > bbw_ma21 * 1.5) & (slope_5 < 0)
        trap_penalty = np.where(is_vol_trap, 0.85 - (slope_5.abs() * 2.0).clip(0, 0.25), 1.0)
        # 3. 综合调节因子合成
        vol_adj = pd.Series(accel_gain * trap_penalty, index=df_index).clip(0.6, 1.4)
        if is_debug and probe_ts in df_index:
            print(f"\n[波动率聚集探针V49 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    BBW: {bbw.loc[probe_ts]:.4f}, 21日均值: {bbw_ma21.loc[probe_ts]:.4f}")
            print(f"    BBW斜率(5d): {slope_5.loc[probe_ts]:.4f}, 加速度(5d): {accel_5.loc[probe_ts]:.4f}")
            print(f"    >>> 加速增益: {accel_gain[df_index.get_loc(probe_ts)]:.4f}, 陷阱惩罚: {trap_penalty[df_index.get_loc(probe_ts)]:.4f}")
            print(f"    >>> 最终波动率调节系数: {vol_adj.loc[probe_ts]:.4f}")
        return vol_adj.astype(np.float32)

    def _calculate_sector_overflow_decay(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V53.0 · 板块分形衰减：利用分形维数判定热度博弈稳定性"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 分形维数计算
        theme_hotness = raw['THEME_HOTNESS_SCORE_D'].fillna(0).values
        flow_arrays = np.expand_dims(theme_hotness, axis=0) 
        fractal_dim_vals = _numba_fractal_dimension(flow_arrays, window=13)
        fractal_dim = pd.Series(fractal_dim_vals, index=df_index)
        # 2. 衰减系数映射
        decay_factor = (1.5 / (fractal_dim + 1e-9)).clip(0.6, 1.2)
        # 3. 高位耗竭惩罚
        is_exhaustion = (raw['THEME_HOTNESS_SCORE_D'] > 80) & (fractal_dim > 1.6)
        final_decay = np.where(is_exhaustion, decay_factor * 0.8, decay_factor)
        return pd.Series(final_decay, index=df_index).astype(np.float32)

    def _calculate_fractal_market_analysis(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V24.0 · 分形市场分析（Numba加速）"""
        large_net_flow = (raw_signals['buy_elg_amount_D'] + raw_signals['buy_lg_amount_D'] - 
                         raw_signals['sell_elg_amount_D'] - raw_signals['sell_lg_amount_D']).fillna(0)
        scales = [1, 3, 5, 10, 21]
        fractal_scores = []
        flow_arrays = [] # For numba
        for scale in scales:
            s_flow = large_net_flow if scale == 1 else large_net_flow.rolling(window=scale, min_periods=1).sum()
            std = s_flow.std()
            norm_flow = (s_flow - s_flow.mean()) / (std if std > 0 else 1)
            fractal_scores.append(norm_flow)
            flow_arrays.append(s_flow.fillna(0).values)
        # Numba计算分形维数
        fractal_dim = pd.Series(_numba_fractal_dimension(np.array(flow_arrays)), index=df_index).clip(1.0, 2.0)
        # 向量化计算一致性
        signs_sum = sum([np.sign(f.fillna(0)) for f in fractal_scores])
        consistency_score = (signs_sum.abs() / len(scales))
        fractal_score = (1 - (fractal_dim - 1.5).abs() * 2) * consistency_score
        return fractal_score.fillna(0).clip(-1, 1)

    def _calculate_hidden_markov_states(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V24.0 · HMM状态识别（Numba加速）"""
        flow_obs = (raw_signals['buy_elg_amount_D'] - raw_signals['sell_elg_amount_D']).fillna(0)
        volume_obs = raw_signals['volume_D'].fillna(0)
        price_change = raw_signals['pct_change_D'].fillna(0)
        def norm(s): return (s - s.mean()) / (s.std() + 1e-9)
        f_n, v_n, p_n = norm(flow_obs).values, norm(volume_obs).values, norm(price_change).values
        # Numba状态计算
        hmm_states = pd.Series(_numba_hmm_states(f_n, v_n, p_n), index=df_index)
        state_scores = hmm_states.map({0: 0.0, 1: 0.7, 2: 1.0, 3: -0.5})
        # 向量化持续性检测 (shift比较)
        state_persistence = np.where((hmm_states == hmm_states.shift(1)) & (hmm_states == hmm_states.shift(2)), 0.3, 0.0)
        return (state_scores + state_persistence).clip(-1, 1).fillna(0)

    def _calculate_chip_control_score(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V24.0 · 筹码控制分数（深度向量化）"""
        # 1. GMM/多模态分析 (简化为峰度分析以利用向量化)
        conc = raw_signals['chip_concentration_ratio_D'].fillna(0.5)
        # 利用rolling kurtosis代替复杂的KDE峰值计数，峰度低意味着多峰或平坦
        price_pos = raw_signals.get('price_percentile_position_D', pd.Series(0.5, index=df_index))
        window_kurt = price_pos.rolling(21).kurt()
        gmm_score = (1 / (1 + window_kurt.abs())) * conc # 峰度越接近0(正态)或负(多峰)，分数越高
        gmm_score = gmm_score.fillna(0.3)
        # 2. 突变检测 (向量化CUSUM代理)
        stab = raw_signals['chip_stability_D'].fillna(0.5)
        roll_mean = stab.rolling(13).mean()
        roll_std = stab.rolling(13).std()
        change_score = (stab - roll_mean).abs() / (roll_std + 1e-9)
        change_points = (change_score / 3.0).clip(0, 1).rolling(3).mean().fillna(0)
        # 3. 网络分析 (向量化)
        f_dir = raw_signals.get('chip_flow_direction_D', pd.Series(0.0, index=df_index))
        f_int = raw_signals.get('chip_flow_intensity_D', pd.Series(0.0, index=df_index))
        div_r = raw_signals.get('chip_divergence_ratio_D', pd.Series(0.5, index=df_index))
        dir_cons = f_dir.rolling(21).std()
        int_conc = f_int.rolling(21).std() / (f_int.rolling(21).mean().abs() + 1e-9)
        div_bal = 1 - div_r.rolling(21).std()
        net_score = ((0.3/(dir_cons+1e-9)) + int_conc*0.4 + div_bal*0.3).clip(0,1).fillna(0.5)
        # 4. 成本结构压力 (向量化)
        close = raw_signals['close_D']
        costs = {k: raw_signals.get(k, close) for k in ['cost_5pct_D','cost_50pct_D','cost_95pct_D']}
        p_below = np.maximum(0, 1 - close/costs['cost_50pct_D'])
        p_above = np.maximum(0, close/costs['cost_95pct_D'] - 1)
        pressure_bal = (1 - (p_below - p_above).abs()).fillna(0.65)
        # 5. 加权融合
        chip_score = (gmm_score * 0.15 + (1-change_points)*0.15 + net_score*0.15 + pressure_bal*0.15 + 
                      raw_signals.get('chip_entropy_D', 0.5)*0.1 + raw_signals.get('chip_stability_change_5d_D', 0)*0.1)
        # 博弈调整
        ent_norm = raw_signals['chip_entropy_D'] / raw_signals['chip_entropy_D'].max()
        adj = np.tanh(conc * 2) * 0.2 + (1 - ent_norm) * 0.15
        return (chip_score * (1 + adj)).clip(0, 1).fillna(0.5)

    def _calculate_order_flow_dominance(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V25.0 · 订单流主导分数计算（Numba极致加速版）"""
        print(f"    -> [订单流主导探针] {method_name} 开始深度计算订单流主导分数...")
        n_len = len(df_index)
        # 0. 准备Numpy数组
        large_buy = (raw_signals['buy_elg_amount_D'] + raw_signals['buy_lg_amount_D']).fillna(0).values
        large_sell = (raw_signals['sell_elg_amount_D'] + raw_signals['sell_lg_amount_D']).fillna(0).values
        large_net = large_buy - large_sell
        elg_net = (raw_signals['buy_elg_amount_D'] - raw_signals['sell_elg_amount_D']).fillna(0).values
        lg_net = (raw_signals['buy_lg_amount_D'] - raw_signals['sell_lg_amount_D']).fillna(0).values
        md_net = (raw_signals['buy_md_amount_D'] - raw_signals['sell_md_amount_D']).fillna(0).values
        sm_net = (raw_signals['buy_sm_amount_D'] - raw_signals['sell_sm_amount_D']).fillna(0).values
        close_vals = raw_signals['close_D'].fillna(0).values
        pct_change = raw_signals['pct_change_D'].fillna(0).values
        trade_count = raw_signals['trade_count_D'].fillna(1000).values
        vol_vals = raw_signals['volume_D'].fillna(1e6).values
        amt_vals = raw_signals['amount_D'].fillna(1e8).values
        sentiment = raw_signals['market_sentiment_score_D'].fillna(0).values
        # 1. 热传导 (Numba)
        heat_vals = _numba_heat_conduction(large_net, alpha=0.1, steps=3)
        heat_flow_norm = self.helper._normalize_series(pd.Series(heat_vals, index=df_index), df_index, bipolar=True)
        # 2. 资金流分布 (Numba)
        elg_dom, lg_sync, ret_div, spat_bal = _numba_flow_distribution_analysis(elg_net, lg_net, md_net, sm_net, close_vals, n_len)
        dist_composite = (pd.Series(elg_dom, index=df_index) + pd.Series(lg_sync, index=df_index) + 
                          pd.Series(spat_bal, index=df_index)) / 3.0
        # 3. 随机过程 (Numba)
        stoch_score_vals = _numba_stochastic_process(large_net)
        stoch_score = pd.Series(stoch_score_vals, index=df_index)
        # 4. 网络与排队博弈 (Numba)
        net_eff, queue_eff, game_learn = _numba_network_queuing_game(large_buy, large_sell, pct_change, trade_count, vol_vals, amt_vals, sentiment)
        # 5. 强度与稳定性 (向量化)
        flow_intensity = raw_signals.get('flow_intensity_D', pd.Series(0, index=df_index)).fillna(0)
        flow_stab = raw_signals.get('flow_stability_D', pd.Series(0.5, index=df_index)).fillna(0.5)
        strength_stability = (self.helper._normalize_series(flow_intensity, df_index) * 0.6 + flow_stab * 0.4)
        # 6. 综合加权
        weights = {'heat': 0.15, 'stoch': 0.15, 'net': 0.15, 'queue': 0.10, 'learn': 0.15, 'dist': 0.15, 'stab': 0.15}
        total_score = (heat_flow_norm * weights['heat'] + 
                       stoch_score * weights['stoch'] + 
                       pd.Series(net_eff, index=df_index) * weights['net'] + 
                       pd.Series(queue_eff, index=df_index) * weights['queue'] + 
                       pd.Series(game_learn, index=df_index) * weights['learn'] + 
                       dist_composite * weights['dist'] + 
                       strength_stability * weights['stab'])
        # 7. 市场结构适应性 (向量化)
        volatility = raw_signals['BBW_21_2.0_D'].fillna(0.1)
        adj = 1.0 / (1.0 + volatility * 2)
        final_score = total_score * adj
        print(f"    -> [订单流主导探针] 最终分值均值: {final_score.mean():.4f}")
        return final_score.clip(0, 1).fillna(0.5)

    def _calculate_vpa_confirmation(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V25.0 · 量价分析确认分数计算（Numba深度学习增强版）"""
        print(f"    -> [VPA确认探针] {method_name} 开始深度计算量价分析确认分数...")
        # 0. 数据准备
        close = raw_signals['close_D'].fillna(0).values
        vol = raw_signals['volume_D'].fillna(0).values
        eff = raw_signals.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=df_index)).fillna(0.5).values
        pct_change = raw_signals['pct_change_D'].fillna(0).values
        vol_change = raw_signals['volume_D'].pct_change().fillna(0).values
        eff_change = pd.Series(eff, index=df_index).diff().fillna(0).values
        # 1. Numba加速计算模式识别
        cnn_score_vals = _numba_cnn_pattern_sim(close, vol, eff)
        cnn_score = pd.Series(cnn_score_vals, index=df_index)
        # 2. Numba加速注意力与依赖
        attn_vals, dep_vals = _numba_vpa_mechanisms(pct_change, vol_change, eff_change)
        attention_score = pd.Series(attn_vals, index=df_index)
        tcn_score = pd.Series(dep_vals, index=df_index)
        # 3. 向量化VAE潜在表示 (使用Rolling PCA近似)
        # 简化为价格和成交量的Rolling Correlation作为一致性代理
        p_s = pd.Series(close, index=df_index)
        v_s = pd.Series(vol, index=df_index)
        vae_score = p_s.rolling(21).corr(v_s).abs().fillna(0.5)
        # 4. 向量化GAN鲁棒性
        # 添加噪声并计算相关性，完全向量化
        eff_s = pd.Series(eff, index=df_index)
        noise = np.random.normal(0, 0.1 * eff_s.std(), len(eff_s))
        noisy_eff = eff_s + noise
        gan_score = eff_s.rolling(13).corr(noisy_eff).fillna(0.5)
        # 5. 传统指标融合 (向量化)
        trad_score = (raw_signals.get('VPA_EFFICIENCY_D', pd.Series(0.5, index=df_index)) * 0.4 +
                      raw_signals.get('vpa_bullish_divergence_D', pd.Series(0, index=df_index)) * 0.3 +
                      raw_signals.get('vpa_buy_accum_eff_D', pd.Series(0.5, index=df_index)) * 0.3)
        # 6. 加权融合
        deep_score = (cnn_score * 0.25 + attention_score * 0.20 + vae_score * 0.20 + tcn_score * 0.20 + gan_score * 0.15)
        final_score = deep_score * 0.6 + trad_score * 0.4
        # 7. 博弈论优化 (向量化)
        pos = raw_signals.get('price_percentile_position_D', pd.Series(0.5, index=df_index))
        conc = raw_signals['chip_concentration_ratio_D'].fillna(0.5)
        opt_factor = np.where(pos < 0.3, 1.2, np.where(pos > 0.7, 0.8, 1.0)) * (0.7 + conc * 0.6)
        return (final_score * opt_factor).clip(0, 1).fillna(0.5)

    def _calculate_vpa_regime_adaptation(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V25.0 · VPA机制适应分析（Numba加速版）"""
        print(f"    -> [VPA机制探针] {method_name} 开始VPA机制适应分析...")
        n = len(df_index)
        close = raw_signals['close_D'].fillna(0).values
        volume = raw_signals['volume_D'].fillna(0).values
        rsi = raw_signals.get('RSI_13_D', pd.Series(50, index=df_index)).fillna(50).values
        adx = raw_signals.get('ADX_14_D', pd.Series(20, index=df_index)).fillna(20).values
        # Numba计算机制状态
        trend, range_mk, break_mk, rev_mk = _numba_market_regime_scan(close, volume, rsi, adx, n)
        # 向量化加权
        eff_trend = 0.9
        eff_range = 0.6
        eff_break = 0.8
        eff_rev = 0.7
        # 加权平均
        total_w = trend + range_mk + break_mk + rev_mk + 1e-9
        score = (trend * eff_trend + range_mk * eff_range + break_mk * eff_break + rev_mk * eff_rev) / total_w
        # 平滑
        res = pd.Series(score, index=df_index).rolling(5).mean().clip(0.3, 0.9).fillna(0.6)
        print(f"       机制适应分数均值: {res.mean():.4f}")
        return res

    def _calculate_trend_alignment(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V25.1 · 趋势协同分数计算（Numba动力学模型版）"""
        print(f"    -> [趋势协同探针] {method_name} 开始深度计算趋势协同分数...")
        n = len(df_index)
        close = raw_signals['close_D'].fillna(0).values
        volume = raw_signals['volume_D'].fillna(0).values
        # 1. Numba 动态系统与拓扑分析
        attr_vals, topo_vals = _numba_trend_dynamics(close, volume, n)
        attractor = pd.Series(attr_vals, index=df_index)
        topo = pd.Series(topo_vals, index=df_index)
        # 2. 向量化 多时间框架协同
        # 替代原有的嵌套循环
        ma_5 = raw_signals['close_D'].rolling(5).mean()
        ma_21 = raw_signals['close_D'].rolling(21).mean()
        ma_55 = raw_signals['close_D'].rolling(55).mean()
        dir_5 = np.sign(raw_signals['close_D'] - ma_5)
        dir_21 = np.sign(raw_signals['close_D'] - ma_21)
        dir_55 = np.sign(raw_signals['close_D'] - ma_55)
        # 协同度：方向一致给分
        consistency = ((dir_5 == dir_21).astype(float) + (dir_21 == dir_55).astype(float) + (dir_5 == dir_55).astype(float)) / 3.0
        multi_tf_score = pd.Series(consistency, index=df_index).fillna(0.5)
        # 3. 向量化 资金流协同
        buy_flow = raw_signals['buy_elg_amount_D'] + raw_signals['buy_lg_amount_D']
        sell_flow = raw_signals['sell_elg_amount_D'] + raw_signals['sell_lg_amount_D']
        net_flow = (buy_flow - sell_flow).fillna(0)
        price_trend = raw_signals['pct_change_D'].rolling(5).mean().fillna(0)
        # 协同：资金净额与价格趋势同向
        flow_align = np.where(np.sign(net_flow) == np.sign(price_trend), 1.0, 0.3)
        flow_score = pd.Series(flow_align, index=df_index)
        # 4. 融合
        final = (attractor * 0.25 + topo * 0.2 + multi_tf_score * 0.3 + flow_score * 0.25)
        # 5. 阶段调整
        phase = raw_signals.get('MARKET_PHASE_D', pd.Series(0, index=df_index))
        adj = np.where(phase > 0, 1.1, np.where(phase < 0, 0.8, 0.9))
        return (final * adj).clip(0, 1).fillna(0.5)

    def _calculate_market_context_modulator(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V25.0 · 市场情境调节器计算（Numba感知模型版）"""
        print(f"    -> [市场情境探针] {method_name} 开始深度计算市场情境调节器...")
        n = len(df_index)
        close = raw_signals['close_D'].fillna(0).values
        volume = raw_signals['volume_D'].fillna(0).values
        sent = raw_signals.get('market_sentiment_score_D', pd.Series(0.5, index=df_index)).fillna(0.5).values
        # 1. Numba 复杂系统分析
        crit_vals, adapt_vals = _numba_complex_systems(close, volume, sent, n)
        criticality = pd.Series(crit_vals, index=df_index)
        adaptation = pd.Series(adapt_vals, index=df_index)
        # 2. 向量化 模糊认知与图分析 (简化为相关性矩阵特征)
        # 代理指标：量价相关性、波动率与情绪相关性
        corr_pv = raw_signals['close_D'].rolling(21).corr(raw_signals['volume_D']).fillna(0)
        fcm_score = (corr_pv.abs() + 0.5).clip(0, 1) # 简化的系统一致性
        # 3. 向量化 贝叶斯状态 (规则库)
        p_trend = (raw_signals['close_D'] / raw_signals['close_D'].shift(5) - 1).fillna(0)
        vol_r = raw_signals.get('volume_ratio_D', pd.Series(1, index=df_index))
        state_score = np.where((p_trend > 0) & (vol_r > 1), 0.8, 
                      np.where((p_trend < 0) & (vol_r > 1), 0.2, 0.5))
        bayes_score = pd.Series(state_score, index=df_index)
        # 4. 多层级感知
        micro = raw_signals.get('uptrend_strength_D', pd.Series(0.5, index=df_index))
        macro = raw_signals.get('market_sentiment_score_D', pd.Series(0.5, index=df_index))
        multi_level = (micro + macro) / 2.0
        # 5. 融合
        raw_ctx = (criticality * 0.2 + adaptation * 0.2 + fcm_score * 0.15 + bayes_score * 0.25 + multi_level * 0.2)
        # 6. 非线性增强
        bbw = raw_signals.get('BBW_21_2.0_D', pd.Series(0.5, index=df_index))
        final_ctx = np.where(bbw > 0.7, raw_ctx * 1.2, raw_ctx) # 高波动需强确认
        # 7. 记忆平滑
        return pd.Series(final_ctx, index=df_index).ewm(span=21).mean().clip(0, 1).fillna(0.5)

    def _calculate_entry_accessibility_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V45.0 · T+1 入场可获得性模型：基于换手能效与封板惯性的实战买入机会预判"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 换手率门控 (Liquidity Gate)
        # A股实战经验：自由换手 < 1% 且涨停，次日几乎无入场机会
        turnover_f = raw['turnover_rate_f_D']
        liquidity_factor = (turnover_f / 3.0).clip(0.1, 1.2) # 以3%换手为基准，高于3%则认为流动性充足
        # 2. 封板强度与溢价惩罚 (Sealing Penalty)
        is_limit_up = (raw['close_D'] >= raw['up_limit_D'] * 0.999)
        # 如果涨停，且溢价预期和封板强度都高，则入场难度指数级增加
        sealing_intensity = (raw['closing_flow_intensity_D'] * raw['T1_PREMIUM_EXPECTATION_D']).clip(0, 1)
        limit_accessibility = np.where(is_limit_up, 0.4 * (1.0 - sealing_intensity), 1.0)
        limit_accessibility = pd.Series(limit_accessibility, index=df_index)
        # 3. 综合可获得性评分
        # 换手率是乘数效应，低换手一票否决
        accessibility_score = (limit_accessibility * liquidity_factor).clip(0, 1.0)
        if is_debug and probe_ts in df_index:
            print(f"\n[入场可获得性探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    自由换手: {turnover_f.loc[probe_ts]:.2f}%, 流动性因子: {liquidity_factor.loc[probe_ts]:.4f}")
            print(f"    封板状态: {is_limit_up.loc[probe_ts]}, 封板强度: {raw['closing_flow_intensity_D'].loc[probe_ts]:.4f}")
            print(f"    >>> $T+1$ 入场窗口评分: {accessibility_score.loc[probe_ts]:.4f} (1.0为极易买入)")
        return accessibility_score.astype(np.float32)

