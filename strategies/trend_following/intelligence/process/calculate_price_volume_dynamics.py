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
@jit(nopython=True)
def _numba_hmm_regime_probability(flow_n, vol_n, price_n, vwap_dist_n):
    """V57.0 · HMM 体制概率算子：基于质心距离的软分类"""
    n = len(flow_n)
    markup_probs = np.zeros(n)
    # 定义理想状态质心 [Flow, Vol, Price, VWAP_Dist]
    # 1. 吸筹: 资金流入, 缩量/平量, 价格微跌/震荡, 均价下方压制
    centroid_acc = np.array([1.0, -0.5, -0.2, -0.5])
    # 2. 拉升: 资金流入, 放量, 价格大涨, 站稳均价上方
    centroid_markup = np.array([1.0, 1.0, 1.0, 0.5])
    # 3. 派发: 资金流出, 放量, 价格滞涨/震荡, 均价乖离过大或跌破
    centroid_dist = np.array([-1.0, 1.0, 0.2, 0.0])
    for i in range(n):
        # 构建当前特征向量
        current_vec = np.array([flow_n[i], vol_n[i], price_n[i], vwap_dist_n[i]])
        # 计算欧氏距离
        d_acc = np.sum((current_vec - centroid_acc)**2)
        d_mar = np.sum((current_vec - centroid_markup)**2)
        d_dis = np.sum((current_vec - centroid_dist)**2)
        # 转化为概率 (Softmax 变体，距离越小概率越大)
        # 使用负指数转换
        exp_acc = np.exp(-d_acc)
        exp_mar = np.exp(-d_mar)
        exp_dis = np.exp(-d_dis)
        total_exp = exp_acc + exp_mar + exp_dis + 1e-9
        # 输出拉升体制的概率
        markup_probs[i] = exp_mar / total_exp
    return markup_probs
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
        """V61.0 · 核心信号校验：新增市场情绪与行业排名等宏观环境数据的校验"""
        fib_windows = [3, 5, 8, 13, 21]
        dynamic_base_cols = [
            'net_amount_rate_D', 'winner_rate_D', 'SMART_MONEY_HM_NET_BUY_D', 
            'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'THEME_HOTNESS_SCORE_D', 'chip_entropy_D',
            'volume_D', 'pct_change_D'
        ]
        base_required = [
            'close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'net_amount_rate_D', 'winner_rate_D',
            'up_limit_D', 'down_limit_D', 'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D',
            'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'pressure_release_index_D', 'BBW_21_2.0_D', 
            'VPA_EFFICIENCY_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'turnover_rate_f_D',
            'IS_ROUNDING_BOTTOM_D', 'IS_GOLDEN_PIT_D', 'IS_TRENDING_STAGE_D', 'price_percentile_position_D',
            'TURNOVER_STABILITY_INDEX_D', 'IS_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 
            'THEME_HOTNESS_SCORE_D', 'chip_entropy_D', 'cost_50pct_D',
            'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'trade_count_D',
            'market_sentiment_score_D', 'industry_strength_rank_D', 'industry_rank_accel_D' # 新增环境数据
        ]
        dynamic_required = [f"{p}_{w}_{c}" for c in dynamic_base_cols for w in fib_windows for p in ['SLOPE', 'ACCEL', 'JERK']]
        all_required = base_required + dynamic_required
        return self.helper._validate_required_signals(df, all_required, method_name)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Dict[str, pd.Series]:
        """V61.0 · 原料加载层：确保市场情绪与行业排名数据被正确加载"""
        raw_signals = {}
        base_cols = ['close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'pct_change_D', 'net_amount_rate_D', 'trade_count_D', 'turnover_rate_f_D']
        struct_cols = ['winner_rate_D', 'chip_concentration_ratio_D', 'chip_entropy_D', 'cost_50pct_D', 'absorption_energy_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'price_percentile_position_D']
        tech_cols = [
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 
            'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D', 'pressure_release_index_D', 'up_limit_D', 
            'down_limit_D', 'closing_flow_ratio_D', 'TURNOVER_STABILITY_INDEX_D', 'IS_EMOTIONAL_EXTREME_D', 
            'flow_consistency_D', 'industry_strength_rank_D', 'industry_rank_accel_D', 'IS_ROUNDING_BOTTOM_D', 
            'IS_GOLDEN_PIT_D', 'IS_TRENDING_STAGE_D', 'THEME_HOTNESS_SCORE_D',
            'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D',
            'market_sentiment_score_D' # 新增
        ]
        dynamic_targets = [
            'net_amount_rate_D', 'winner_rate_D', 'SMART_MONEY_HM_NET_BUY_D', 
            'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'THEME_HOTNESS_SCORE_D', 'chip_entropy_D',
            'volume_D', 'pct_change_D'
        ]
        fib_windows = [3, 5, 8, 13, 21]
        for col in base_cols + struct_cols + tech_cols:
            if col not in df.columns:
                raise KeyError(f"CRITICAL: 军械库缺失关键列 {col}，请检查数据层产出")
            if col in struct_cols or 'rank' in col:
                raw_signals[col] = df[col].ffill().fillna(0.0)
            else:
                raw_signals[col] = df[col].fillna(0.0)
        if 'VWAP_D' not in df.columns:
            vwap = raw_signals['amount_D'] / (raw_signals['volume_D'] + 1e-9)
            raw_signals['VWAP_D'] = vwap.fillna(raw_signals['close_D'])
        else:
            raw_signals['VWAP_D'] = df['VWAP_D'].fillna(raw_signals['close_D'])
        for col in dynamic_targets:
            for win in fib_windows:
                for prefix in ['SLOPE', 'ACCEL', 'JERK']:
                    dyn_col = f"{prefix}_{win}_{col}"
                    if dyn_col not in df.columns:
                        raw_signals[dyn_col] = pd.Series(0.0, index=df.index)
                        continue
                    d_series = df[dyn_col].fillna(0.0).copy()
                    median = d_series.median()
                    mad = (d_series - median).abs().median()
                    threshold = 5.0 * (mad * 1.4826 + 1e-9)
                    raw_signals[dyn_col] = d_series.clip(lower=median - threshold, upper=median + threshold)
        return raw_signals

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V61.0 · 主力夺权全局总线：集成趋势惯性与市场渗透率"""
        method_name = "calculate_price_volume_dynamics"
        df_index = df.index
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        if not self._validate_all_required_signals(df, {}, {}, method_name, is_debug, probe_ts):
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        raw = self._get_raw_signals(df, method_name)
        # --- 第一阶段：计算基础强度 ---
        physical_base = self._calculate_power_transfer_raw_score(df_index, raw, method_name)
        geo_conf = (1.0 - raw['GEOM_REG_R2_D']).clip(0, 1)
        act_curv = pd.Series(_numba_power_activation(raw['GEOM_ARC_CURVATURE_D'].values, gain=2.0), index=df_index)
        geo_resonance = act_curv * geo_conf * (raw['IS_ROUNDING_BOTTOM_D'].astype(float) * 1.2 + 0.5)
        vwap_propulsion = self._calculate_vwap_propulsion_score(raw, df_index, method_name)
        unadjusted_intensity = (physical_base * 0.50 + geo_resonance * 0.25 + vwap_propulsion * 0.25)
        # --- 第二阶段：调节矩阵 ---
        # 1. 结构与环境调节 (V61 新增集成)
        trend_inertia = self._calculate_trend_inertia_momentum(raw, df_index, method_name)
        market_perm = self._calculate_market_permeability_index(raw, df_index, method_name)
        entropy_adj = self._calculate_entropic_ordering_bonus(raw, df_index, method_name)
        fractal_efficiency = self._calculate_fractal_efficiency_resonance(raw, df_index, method_name)
        hmm_confirm = self._calculate_hmm_regime_confirmation(raw, df_index, method_name)
        chip_lock = self._calculate_chip_lock_efficiency(raw, df_index, method_name)
        micro_attack = self._calculate_microstructure_attack_vector(raw, df_index, method_name)
        vpa_reflexivity = self._calculate_vpa_elasticity_reflexivity(raw, df_index, method_name)
        wyckoff_quality = self._calculate_wyckoff_breakout_quality(raw, df_index, method_name)
        # 2. 安全阀
        risk_v = self._calculate_premium_reversal_risk(raw, df_index, method_name)
        decay_v = self._calculate_intraday_decay_model(raw, df_index, method_name)
        sector_v = self._calculate_sector_resonance_modifier(raw, df_index, method_name)
        vol_v = self._calculate_volatility_clustering_adjustment(raw, df_index, method_name)
        sector_overflow_v = self._calculate_sector_overflow_decay(raw, df_index, method_name)
        # --- 第三阶段：实战入场修正 ---
        access_f = self._calculate_entry_accessibility_score(raw, df_index, method_name)
        # --- 第四阶段：最终合成 ---
        # 逻辑：(强度 * 趋势 * 环境 * 结构 * 微观) * 安全阀 * 执行度
        # 将 trend_inertia 和 market_perm 加入乘法链
        final_score = (unadjusted_intensity * trend_inertia * market_perm * entropy_adj * fractal_efficiency * \
                       hmm_confirm * chip_lock * micro_attack * vpa_reflexivity * wyckoff_quality) * \
                      risk_v * decay_v * sector_v * vol_v * sector_overflow_v * access_f
        if is_debug and probe_ts in df_index:
            print(f"\n[PROCESS_META_POWER_TRANSFER 全链集成探针 V61 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    [基准] 物理: {physical_base.loc[probe_ts]:.4f}, 几何: {geo_resonance.loc[probe_ts]:.4f}")
            print(f"    [背景] 趋势惯性: {trend_inertia.loc[probe_ts]:.2f}, 市场渗透: {market_perm.loc[probe_ts]:.2f}")
            print(f"    [微观] 攻击: {micro_attack.loc[probe_ts]:.2f}, 锁定: {chip_lock.loc[probe_ts]:.2f}")
            print(f"    >>> 最终输出分值: {final_score.loc[probe_ts]:.4f}")
        return final_score.clip(-3.5, 6.0).astype(np.float32)

    def _calculate_power_transfer_raw_score(self, df_index: pd.Index, raw: Dict[str, pd.Series], method_name: str) -> pd.Series:
        """V54.0 · 物理动力引擎：适配新的量纲体系，平衡冲量与共识速度"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        score_calculate = CalculatePowerTransferRawScore(is_debug, probe_ts)
        
        # 1. 动力学去噪与冲量激活
        vol_adj = raw['BBW_21_2.0_D'].fillna(0.1).values
        rolling_tc = raw['trade_count_D'].rolling(21).mean().replace(0, 1)
        conf = (raw['trade_count_D'] / rolling_tc).fillna(1.0).values
        j_c = _numba_adaptive_denoise_dynamics(raw['JERK_3_net_amount_rate_D'].fillna(0).values, vol_adj, conf)
        a_c = _numba_adaptive_denoise_dynamics(raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'].fillna(0).values, vol_adj, conf)
        act_impulse = pd.Series(_numba_power_activation((j_c * 0.45 + a_c * 0.55), gain=1.8), index=df_index)
        
        # 2. 标准化与补偿（含 5.0x 增益）
        norm_impulse = score_calculate._calculate_dynamic_impulse_norm(act_impulse, raw, df_index, method_name)
        comp_impulse = score_calculate._calculate_limit_price_compensation(norm_impulse, raw, df_index, method_name)
        
        # 3. 竞价预判 (含 log/tanh 压缩)
        auc_pred = score_calculate._calculate_auction_prediction(raw, df_index, method_name)
        
        # 4. MCV 多尺度共识速度
        fib_wins = np.array([3, 5, 8, 13, 21], dtype=np.int64)
        _, f_slopes = _numba_fast_rolling_dynamics(raw['net_amount_rate_D'].values, fib_wins)
        mcv_consensus = pd.Series(np.dot(np.array([0.35, 0.25, 0.20, 0.10, 0.10]), f_slopes), index=df_index)
        
        # 5. 物理引擎合成
        # 权重平衡：CompImpulse(2.0x) + AucPred(2.5x) + MCV(2.0x)
        # 降低 CompImpulse 的系数(8.0->2.0)以匹配标准化后的增益
        physical_engine_score = (comp_impulse * 2.0 * 0.30 + auc_pred * 0.35 + mcv_consensus * 0.35)
        
        if is_debug and probe_ts in df_index:
            print(f"\n[物理引擎量纲审计 V54 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    补偿后冲量(Norm*2.0): {(comp_impulse.loc[probe_ts] * 2.0):.4f}")
            print(f"    竞价压缩分: {auc_pred.loc[probe_ts]:.4f}, MCV共识: {mcv_consensus.loc[probe_ts]:.4f}")
            print(f"    >>> 物理合成总分: {physical_engine_score.loc[probe_ts]:.4f}")
            
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

    def _calculate_hmm_regime_confirmation(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V57.0 · HMM 体制共振模型：基于概率距离的拉升体制 (Markup Regime) 确认"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 特征归一化 (Rolling Z-Score)
        # 资金流：主力大单净额
        large_net = (raw['SMART_MONEY_HM_NET_BUY_D']).fillna(0)
        flow_n = (large_net - large_net.rolling(21).mean()) / (large_net.rolling(21).std() + 1e-9)
        # 成交量：相对均量
        vol_n = (raw['volume_D'] / raw['volume_D'].rolling(21).mean().replace(0, 1)) - 1.0
        # 价格：动量
        price_n = raw['pct_change_D'] * 10.0 # 放大以匹配量纲
        # VWAP位置：(收盘 - VWAP) / 波动率
        vwap_dist = (raw['close_D'] - raw['VWAP_D']) / (raw['close_D'] * raw['BBW_21_2.0_D'] + 1e-9)
        # 2. 调用 Numba 算子计算拉升概率
        markup_prob_vals = _numba_hmm_regime_probability(
            flow_n.fillna(0).values, 
            vol_n.fillna(0).values, 
            price_n.fillna(0).values, 
            vwap_dist.fillna(0).values
        )
        markup_prob = pd.Series(markup_prob_vals, index=df_index)
        # 3. 体制共振信号提取
        # 逻辑：只有当拉升概率 > 50% 时才产生正向增益，否则视为噪音或压制
        # 基础系数 1.0，最大增益 1.5，若概率低则衰减至 0.8
        regime_factor = np.where(markup_prob > 0.5, 
                                 1.0 + (markup_prob - 0.5), # [0.5, 1.0] -> [1.0, 1.5]
                                 0.8 + markup_prob * 0.4)   # [0.0, 0.5] -> [0.8, 1.0]
        res = pd.Series(regime_factor, index=df_index)
        if is_debug and probe_ts in df_index:
            print(f"\n[HMM 体制共振探针 V57 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    特征归一化 -> 流: {flow_n.loc[probe_ts]:.2f}, 量: {vol_n.loc[probe_ts]:.2f}, 价: {price_n.loc[probe_ts]:.2f}")
            print(f"    拉升体制概率: {markup_prob.loc[probe_ts]:.4f}")
            print(f"    >>> HMM 确认系数: {res.loc[probe_ts]:.4f}")
        return res.astype(np.float32)

    def _calculate_fractal_efficiency_resonance(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V56.0 · 分形相干效率模型：基于 Hurst 指数的价量共振判定"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 准备数据：价格与成交量
        # 使用 21 日窗口计算分形特征
        close_vals = raw['close_D'].fillna(0).values
        vol_vals = raw['volume_D'].fillna(0).values
        # 将一维转为二维以适配 Numba 算子
        input_arrays = np.vstack((close_vals, vol_vals))
        fractal_dims = _numba_fractal_dimension(input_arrays, window=21)
        # 2. 转换为 Hurst 指数 (H = 2 - D)
        # D 在 [1, 2] 之间，H 在 [0, 1] 之间
        hurst_price = pd.Series(2.0 - fractal_dims[0], index=df_index).clip(0, 1)
        hurst_vol = pd.Series(2.0 - fractal_dims[1], index=df_index).clip(0, 1)
        # 3. 趋势持续性判定 (Persistence Score)
        # H > 0.55 为强趋势，H < 0.45 为均值回归
        persistence = np.where(hurst_price > 0.55, 1.2, 
                      np.where(hurst_price < 0.45, 0.6, 0.9)) # 随机游走区给予中性偏低分
        # 4. 价量分形共振 (Efficiency Gap)
        # 如果价格趋势强 (H高) 但量能杂乱 (H低)，则 gap 大，扣分
        eff_gap = (hurst_price - hurst_vol).abs()
        resonance_factor = (1.0 - eff_gap * 2.0).clip(0.5, 1.1)
        # 5. 综合效率系数
        # 基础分 1.0，根据持续性和共振度调整
        efficiency_score = pd.Series(persistence * resonance_factor, index=df_index)
        if is_debug and probe_ts in df_index:
            print(f"\n[分形相干效率探针 V56 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    价格Hurst: {hurst_price.loc[probe_ts]:.4f} ({'强趋势' if hurst_price.loc[probe_ts]>0.55 else '随机/回归'})")
            print(f"    成交量Hurst: {hurst_vol.loc[probe_ts]:.4f}, 分形缺口: {eff_gap.loc[probe_ts]:.4f}")
            print(f"    >>> 最终分形效率系数: {efficiency_score.loc[probe_ts]:.4f}")
        return efficiency_score.astype(np.float32)

    def _calculate_chip_lock_efficiency(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V58.0 · 筹码磁滞锁定模型：基于获利盘-换手率弹性的控盘度量"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 准备数据
        winner = raw['winner_rate_D'].fillna(0)
        turnover = raw['turnover_rate_f_D'].replace(0, 0.1) # 避免除零
        cost_50 = raw['cost_50pct_D'].replace(0, raw['close_D'])
        close = raw['close_D']
        # 2. 计算磁滞锁定因子 (Hysteresis Lock)
        # 逻辑：获利盘很高 (>80%) 但换手率很低 (<3%)，说明大家都赚钱但都不卖 -> 强控盘
        # 归一化换手：将 0-5% 映射到 1.0-0.0
        turnover_norm = (turnover / 5.0).clip(0, 1)
        # 锁定强度 = 获利盘 * (1 - 归一化换手)
        # 例：获利90% * (1 - 1%/5%) = 0.9 * 0.8 = 0.72 (强)
        # 例：获利90% * (1 - 8%/5%) = 0.9 * 0 = 0 (松动)
        lock_strength = winner * (1.0 - turnover_norm).clip(0, 1)
        # 3. 计算底仓稳定性 (Base Stability)
        # 逻辑：价格上涨，但成本重心上移缓慢 -> 底部筹码未动
        price_rise = close.pct_change(3).fillna(0)
        cost_rise = cost_50.pct_change(3).fillna(0)
        # 只有当价格涨幅显著大于成本涨幅时，才视为底仓稳定
        base_stable = np.where((price_rise > 0.05) & (cost_rise < 0.02), 1.2, 1.0)
        # 4. 穿透力加成 (Penetration Bonus)
        # 如果锁定强且价格正在突破成本位
        break_cost = (close > cost_50).astype(float)
        # 5. 综合效率合成
        # 基础系数 1.0，强锁定可达 1.5
        efficiency_score = pd.Series((1.0 + lock_strength * 0.5) * base_stable, index=df_index) * (0.8 + break_cost * 0.2)
        if is_debug and probe_ts in df_index:
            print(f"\n[筹码磁滞锁定探针 V58 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    获利比例: {winner.loc[probe_ts]:.2f}, 自由换手: {turnover.loc[probe_ts]:.2f}%")
            print(f"    锁定强度: {lock_strength.loc[probe_ts]:.4f} (高获利低换手=强)")
            print(f"    底仓稳定系数: {base_stable[df_index.get_loc(probe_ts)]:.2f}, 突破成本: {break_cost.loc[probe_ts]}")
            print(f"    >>> 最终筹码锁定效率: {efficiency_score.loc[probe_ts]:.4f}")
        return efficiency_score.astype(np.float32)

    def _calculate_microstructure_attack_vector(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V59.0 · 微观结构攻击矢量：基于主力协同与扫货密度的进攻性度量"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 准备微观数据
        # 特大单(ELG)与大单(LG)
        elg_net = (raw['buy_elg_amount_D'] - raw['sell_elg_amount_D']).fillna(0)
        lg_net = (raw['buy_lg_amount_D'] - raw['sell_lg_amount_D']).fillna(0)
        total_vol = raw['amount_D'].replace(0, 1e-9)
        # 2. 计算主力协同度 (Institutional Synchronization)
        # 逻辑：ELG 和 LG 必须同向做多，且占成交额比例要大
        # 归一化净额
        elg_ratio = elg_net / total_vol
        lg_ratio = lg_net / total_vol
        # 协同加成：两者同正，加分；一正一负，减分
        sync_score = np.where((elg_ratio > 0) & (lg_ratio > 0), 
                              1.0 + (elg_ratio + lg_ratio) * 2.0, # 强协同
                              np.where((elg_ratio * lg_ratio) < 0, 0.6, 0.8)) # 分歧或弱势
        # 3. 计算扫货密度 (Sweep Density / Stealth)
        # 逻辑：成交笔数(trade_count)下降，但成交金额(amount)上升 -> 平均单笔金额暴增 -> 主力扫货
        trade_count = raw['trade_count_D'].replace(0, 1)
        avg_trade_size = raw['amount_D'] / trade_count
        # 相对均值 (21日)
        avg_size_ma = avg_trade_size.rolling(21).mean().replace(0, 1)
        density_ratio = (avg_trade_size / avg_size_ma).clip(0.5, 3.0)
        # 密度因子：只有在上涨时，高密度才是进攻；下跌时高密度是恐慌抛售
        is_up = raw['pct_change_D'] > 0
        density_factor = np.where(is_up, np.sqrt(density_ratio), 1.0)
        # 4. 综合攻击矢量合成
        # 基础分 * 协同修正 * 密度加成
        # 我们使用 tanh 将巨大的净额比率压缩到合理区间
        net_strength = np.tanh((elg_ratio + lg_ratio) * 10.0) # 0.1(10%) -> 0.76
        attack_vector = pd.Series((0.5 + net_strength * 0.5) * sync_score * density_factor, index=df_index)
        if is_debug and probe_ts in df_index:
            print(f"\n[微观结构攻击探针 V59 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    ELG净占比: {elg_ratio.loc[probe_ts]:.2%}, LG净占比: {lg_ratio.loc[probe_ts]:.2%}")
            print(f"    主力协同分: {sync_score[df_index.get_loc(probe_ts)]:.2f} (同向>1.0)")
            print(f"    单笔均额比: {density_ratio.loc[probe_ts]:.2f}, 扫货密度因子: {density_factor[df_index.get_loc(probe_ts)]:.2f}")
            print(f"    >>> 最终微观攻击矢量: {attack_vector.loc[probe_ts]:.4f}")
        return attack_vector.astype(np.float32)

    def _calculate_vpa_elasticity_reflexivity(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V60.0 · 量价反身性模型：替代原 VPA Confirmation，计算弹性与反身性"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 量价弹性 (Elasticity)
        # 逻辑：Elasticity = %Price / %Volume
        # 我们关注的是"单位成交量推动的价格幅度"
        pct_price = raw['pct_change_D'].abs()
        pct_vol = raw['volume_D'].pct_change().fillna(0).abs()
        # 弹性 = 价格变化 / (成交量变化 + 基础量)
        # 基础量设为 0.1 以防除零，且体现"缩量上涨"的高弹性
        elasticity = pct_price / (pct_vol + 0.1)
        # 映射：弹性越高，筹码锁定越好 -> 1.0 ~ 1.5
        # 弹性极低（放量滞涨），筹码松动 -> 0.5 ~ 0.8
        elasticity_score = np.tanh(elasticity).clip(0.5, 1.5)
        # 2. 索罗斯反身性 (Reflexivity)
        # 逻辑：价格加速 + 成交量加速 = 正反馈爆发
        accel_p = raw['ACCEL_5_pct_change_D']
        accel_v = raw['ACCEL_5_volume_D']
        # 正反馈：两者同时加速
        positive_feedback = (accel_p > 0) & (accel_v > 0)
        # 负反馈：量加速但价减速 (顶背离风险)
        negative_feedback = (accel_p < 0) & (accel_v > 0)
        reflexivity_score = np.where(positive_feedback, 1.3, 
                            np.where(negative_feedback, 0.7, 1.0))
        # 3. 综合 VPA 确认系数
        final_score = pd.Series(elasticity_score * reflexivity_score, index=df_index).clip(0.5, 1.8)
        if is_debug and probe_ts in df_index:
            print(f"\n[量价反身性探针 V60 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    量价弹性: {elasticity.loc[probe_ts]:.4f}, 弹性系数: {elasticity_score.loc[probe_ts]:.4f}")
            print(f"    P加速: {accel_p.loc[probe_ts]:.4f}, V加速: {accel_v.loc[probe_ts]:.4f}")
            print(f"    反身性状态: {'正反馈(共振)' if positive_feedback.loc[probe_ts] else ('负反馈(背离)' if negative_feedback.loc[probe_ts] else '中性')}")
            print(f"    >>> 最终 VPA 确认系数: {final_score.loc[probe_ts]:.4f}")
        return final_score.astype(np.float32)

    def _calculate_wyckoff_breakout_quality(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V60.0 · 威科夫突破质量：替代原 Regime Adaptation，识别 VCP 与真突破"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 波动率收缩 (VCP - Volatility Contraction)
        # 计算真实波幅 TR 的 5 日斜率
        high = raw['high_D']
        low = raw['low_D']
        close_prev = raw['close_D'].shift(1).fillna(raw['close_D'])
        tr = np.maximum(high - low, np.abs(high - close_prev))
        tr_norm = tr / raw['close_D']
        # 简单的线性回归斜率模拟 (末端 - 首端)
        tr_slope = (tr_norm - tr_norm.shift(5)) / 5.0
        # VCP 特征：波动率在收缩 (斜率 < 0)
        is_vcp = tr_slope < 0
        vcp_bonus = np.where(is_vcp, 1.2, 1.0)
        # 2. 突破强度 (Breakout Intensity)
        # 当前价格是否创 21 日新高
        highest_21 = raw['high_D'].rolling(21).max().shift(1).fillna(99999)
        is_breakout = raw['close_D'] > highest_21
        # 突破且放量 (Volume > MA21 * 1.5)
        vol_ma = raw['volume_D'].rolling(21).mean().fillna(1)
        is_volume_confirmed = raw['volume_D'] > vol_ma * 1.5
        # 3. 综合突破质量
        # 如果是 VCP 后的放量突破 -> 极品 (1.5)
        # 如果是 VCP 但未突破 -> 蓄势 (1.0)
        # 如果非 VCP 的突兀突破 -> 需谨慎 (1.1)
        # 如果未突破 -> 平庸 (0.8)
        quality = np.where(is_breakout & is_volume_confirmed & is_vcp, 1.5,
                  np.where(is_breakout & is_volume_confirmed, 1.2,
                  np.where(is_vcp, 1.0, 0.8)))
        final_score = pd.Series(quality * vcp_bonus, index=df_index).clip(0.6, 1.8)
        if is_debug and probe_ts in df_index:
            print(f"\n[威科夫突破探针 V60 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    波动率斜率: {tr_slope.loc[probe_ts]:.4f} ({'收缩VCP' if is_vcp.loc[probe_ts] else '扩散'})")
            print(f"    突破状态: {is_breakout.loc[probe_ts]}, 放量确认: {is_volume_confirmed.loc[probe_ts]}")
            print(f"    >>> 最终突破质量系数: {final_score.loc[probe_ts]:.4f}")
        return final_score.astype(np.float32)

    def _calculate_trend_inertia_momentum(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V61.0 · 趋势惯性动量模型：基于多周期均线排列与 R2 质量的惯性度量"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        close = raw['close_D']
        # 1. 均线多头排列 (Bullish Alignment)
        ma5 = close.rolling(5).mean()
        ma21 = close.rolling(21).mean()
        ma55 = close.rolling(55).mean()
        # 判定：5 > 21 > 55 且 5 > 5(t-1)
        alignment_score = np.where((ma5 > ma21) & (ma21 > ma55) & (ma5 > ma5.shift(1)), 1.2, 
                          np.where((ma5 > ma21), 1.0, 0.8))
        # 2. 惯性质量 (Inertia Quality - R2)
        # R2 越高，趋势越平滑，噪音越小 -> 惯性越强
        r2_quality = raw['GEOM_REG_R2_D'].clip(0, 1)
        # 3. 动量堆叠 (Momentum Stacking)
        # 短期动量 > 长期动量 -> 加速状态
        mom_5 = raw['pct_change_D'].rolling(5).sum()
        mom_21 = raw['pct_change_D'].rolling(21).sum()
        stacking_bonus = np.where(mom_5 > mom_21 * 0.2, 1.1, 1.0)
        # 4. 综合惯性系数
        # 基础分 * 排列 * 质量 * 堆叠
        final_inertia = pd.Series(alignment_score * (0.5 + r2_quality * 0.5) * stacking_bonus, index=df_index).clip(0.6, 1.5)
        if is_debug and probe_ts in df_index:
            print(f"\n[趋势惯性动量探针 V61 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    均线状态: {'多头排列' if alignment_score[df_index.get_loc(probe_ts)] > 1.0 else '非多头'}")
            print(f"    惯性质量(R2): {r2_quality.loc[probe_ts]:.4f}, 动量堆叠: {stacking_bonus[df_index.get_loc(probe_ts)]:.2f}")
            print(f"    >>> 最终趋势惯性系数: {final_inertia.loc[probe_ts]:.4f}")
        return final_inertia.astype(np.float32)

    def _calculate_market_permeability_index(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V61.0 · 市场渗透率与 Alpha 模型：基于情绪背离与行业强度的环境判读"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 情绪背离 (Sentiment Divergence - Wall of Worry)
        # 最佳状态：股价处于高位 (Percentile > 0.8)，但情绪尚未过热 (Score < 80)
        price_pos = raw['price_percentile_position_D']
        sentiment = raw['market_sentiment_score_D']
        # 渗透率高：价格强 + 情绪冷
        permeability = np.where((price_pos > 0.8) & (sentiment < 80), 1.2,
                       np.where(sentiment > 90, 0.7, 1.0)) # 情绪过热则渗透率下降（拥挤）
        # 2. Alpha 属性 (Industry Resonance)
        # 行业排名靠前 (数值小) 且 加速度为正 (排名在提升，注意 rank 越小越好，所以提升是 rank 变小， diff < 0)
        rank = raw['industry_strength_rank_D']
        rank_accel = raw['industry_rank_accel_D']
        # 假设 accel > 0 代表排名变好（在数据层定义，通常需要确认方向，这里假设 >0 为改善）
        # 如果 rank < 10 (前10名) -> 强 Alpha
        alpha_bonus = np.where(rank < 10, 1.3, 
                      np.where(rank < 30, 1.1, 0.9))
        # 3. 综合环境系数
        final_ctx = pd.Series(permeability * alpha_bonus, index=df_index).clip(0.6, 1.6)
        if is_debug and probe_ts in df_index:
            print(f"\n[市场渗透率探针 V61 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    价格位置: {price_pos.loc[probe_ts]:.2f}, 市场情绪: {sentiment.loc[probe_ts]:.2f}")
            print(f"    行业排名: {rank.loc[probe_ts]:.1f}, Alpha加成: {alpha_bonus[df_index.get_loc(probe_ts)]:.2f}")
            print(f"    >>> 最终环境渗透系数: {final_ctx.loc[probe_ts]:.4f}")
        return final_ctx.astype(np.float32)

    def _calculate_entry_accessibility_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V54.0 · T+1 入场可获得性模型：采用非线性根号平滑优化流动性门控"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 换手率门控 (Liquidity Gate)
        turnover_f = raw['turnover_rate_f_D']
        # 优化策略：根号平滑 (sqrt(x)/sqrt(1.5))
        liquidity_factor = (np.sqrt(turnover_f) / np.sqrt(1.5)).clip(0.1, 1.1)
        # 2. 封板强度与溢价惩罚 (Sealing Penalty)
        is_limit_up = (raw['close_D'] >= raw['up_limit_D'] * 0.999)
        sealing_intensity = (raw['closing_flow_intensity_D'] * raw['T1_PREMIUM_EXPECTATION_D']).clip(0, 1)
        limit_accessibility = np.where(is_limit_up, 0.4 * (1.0 - sealing_intensity), 1.0)
        limit_accessibility = pd.Series(limit_accessibility, index=df_index)
        # 3. 综合可获得性评分
        accessibility_score = (limit_accessibility * liquidity_factor).clip(0, 1.0)
        if is_debug and probe_ts in df_index:
            print(f"\n[入场可获得性探针 V54 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    自由换手: {turnover_f.loc[probe_ts]:.2f}%, 流动性因子: {liquidity_factor.loc[probe_ts]:.4f}")
            print(f"    >>> $T+1$ 入场窗口评分: {accessibility_score.loc[probe_ts]:.4f} (优化后)")
        return accessibility_score.astype(np.float32)

    def _calculate_entropic_ordering_bonus(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V55.0 · 熵减有序性因子：识别筹码锁定（熵减）带来的趋势稳定性"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        entropy = raw['chip_entropy_D']
        slope_5 = raw['SLOPE_5_chip_entropy_D']
        accel_5 = raw['ACCEL_5_chip_entropy_D']
        locking_intensity = -(slope_5 * 0.7 + accel_5 * 0.3)
        ordering_bonus = np.tanh(locking_intensity * 5.0).clip(0, 1) * 1.5
        price_up = raw['pct_change_D'] > 0
        entropy_surge = (slope_5 > 0)
        penalty = np.where(price_up & entropy_surge, 0.7, 1.0)
        final_factor = pd.Series((1.0 + ordering_bonus) * penalty, index=df_index).clip(0.7, 1.5)
        if is_debug and probe_ts in df_index:
            print(f"\n[熵减有序性探针 V55 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    筹码熵: {entropy.loc[probe_ts]:.4f}, 5日斜率: {slope_5.loc[probe_ts]:.4f}")
            print(f"    锁仓强度: {locking_intensity.loc[probe_ts]:.4f}, 有序奖励: {ordering_bonus.loc[probe_ts]:.4f}")
            print(f"    >>> 最终有序性调节系数: {final_factor.loc[probe_ts]:.4f}")
        return final_factor.astype(np.float32)

    def _calculate_vwap_propulsion_score(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V55.0 · VWAP 推进力因子：验证价格上涨的平均成本支撑强度"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        close = raw['close_D']
        vwap = raw['VWAP_D']
        vwap_slope = vwap.diff(3) / 3.0
        bias = (close - vwap) / (vwap + 1e-9)
        healthy_bias = (bias > 0) & (bias < 0.05)
        propulsion = np.where(vwap_slope > 0, 1.0, 0.0)
        bias_bonus = np.where(healthy_bias, 0.2, 0.0)
        trend_bonus = np.tanh(vwap_slope * 10.0).clip(0, 0.3)
        final_score = pd.Series(propulsion + bias_bonus + trend_bonus, index=df_index).clip(0, 1.5)
        if is_debug and probe_ts in df_index:
            print(f"\n[VWAP 推进力探针 V55 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    收盘: {close.loc[probe_ts]:.2f}, VWAP: {vwap.loc[probe_ts]:.2f}")
            print(f"    VWAP斜率(3d): {vwap_slope.loc[probe_ts]:.4f}, 乖离率: {bias.loc[probe_ts]:.2%}")
            print(f"    >>> 最终推进力得分: {final_score.loc[probe_ts]:.4f}")
        return final_score.astype(np.float32)



