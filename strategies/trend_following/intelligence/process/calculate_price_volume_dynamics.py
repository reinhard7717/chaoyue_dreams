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
    """V69.0 · 分形维数算子修复：移除不兼容的 np.clip 标量操作并增强鲁棒性"""
    n = len(flows[0])
    out = np.ones(n) * 1.5
    # 定义采样尺度
    all_scales = np.array([1, 2, 3, 5, 8, 13, 21])
    for i in range(window, n):
        # 1. 检测窗口内是否有有效波动
        has_variance = False
        for j in range(len(flows)):
            if np.std(flows[j][i-window:i]) > 1e-6:
                has_variance = True
                break
        if not has_variance:
            out[i] = 1.5
            continue
        flucts = []
        current_scales = []
        for scale in all_scales:
            if scale >= window: break
            total_fluct = 0.0
            count = 0
            for j in range(len(flows)):
                data_slice = flows[j][i-window:i]
                reshaped_len = (len(data_slice) // scale) * scale
                if reshaped_len < scale: continue
                seg_flucts = 0.0
                n_segs = 0
                for k in range(0, reshaped_len, scale):
                    seg = data_slice[k:k+scale]
                    seg_std = np.std(seg)
                    seg_flucts += seg_std
                    n_segs += 1
                if n_segs > 0:
                    total_fluct += seg_flucts / n_segs
                    count += 1
            if count > 0 and total_fluct > 1e-9:
                flucts.append(total_fluct)
                current_scales.append(float(scale))
        if len(flucts) >= 3:
            y = np.log(np.array(flucts))
            x = np.log(np.array(current_scales))
            # 线性回归 log(fluct) ~ H * log(scale)
            # 确保 x, y 形状匹配
            A = np.vstack((x, np.ones(len(x)))).T
            slope, _ = np.linalg.lstsq(A, y, rcond=-1)[0]
            # 修复点：手动执行标量裁剪，避免 Numba 的 np.clip(float64) 错误
            # Hurst H 限制在 [0, 1]
            h_val = slope
            if h_val < 0.0: h_val = 0.0
            if h_val > 1.0: h_val = 1.0
            # 分形维数 D = 2 - H
            out[i] = 2.0 - h_val
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
@jit(nopython=True)
def _numba_robust_dynamics(data, win=5, abs_threshold=1e-4, change_threshold=1e-5):
    """
    V67.0 · 鲁棒动力学算子：内置绝对幅值门控与最小变动锁，彻底消除微小基数噪音
    :param data: 输入序列
    :param win: 窗口长度
    :param abs_threshold: 绝对幅值门控 (低于此值视为 0)
    :param change_threshold: 最小变动锁 (变化量低于此值视为 0)
    """
    n = len(data)
    slope = np.zeros(n)
    accel = np.zeros(n)
    jerk = np.zeros(n)
    # 预处理：应用绝对幅值门控
    # 如果数据本身极小，直接视为 0，防止计算出巨大的相对倍数
    clean_data = np.copy(data)
    for i in range(n):
        if np.abs(clean_data[i]) < abs_threshold:
            clean_data[i] = 0.0
    # 计算 Slope (速度)
    for i in range(win, n):
        # 线性回归求斜率比简单的 (end-start)/n 更抗噪
        # 这里为了性能，采用加权差分：(x[i] - x[i-win]) / win
        # 但加入最小变动锁
        delta = clean_data[i] - clean_data[i-win]
        if np.abs(delta) < change_threshold:
            slope[i] = 0.0
        else:
            slope[i] = delta / win
    # 计算 Accel (加速度)
    for i in range(win, n):
        delta_s = slope[i] - slope[i-1] # 瞬时加速度
        # 对加速度也应用变动锁
        if np.abs(delta_s) < change_threshold / 10.0: # 加速度阈值通常更小
            accel[i] = 0.0
        else:
            accel[i] = delta_s
    # 计算 Jerk (脉冲)
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
            'IS_ROUNDING_BOTTOM_D', 'IS_GOLDEN_PIT_D', 'IS_TRENDING_STAGE_D', 'price_percentile_position_D',
            'TURNOVER_STABILITY_INDEX_D', 'IS_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 
            'THEME_HOTNESS_SCORE_D', 'chip_entropy_D', 'cost_50pct_D',
            'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'trade_count_D',
            'market_sentiment_score_D', 'industry_strength_rank_D', 'industry_rank_accel_D'
        ]
        dynamic_required = [f"{p}_{w}_{c}" for c in dynamic_base_cols for w in fib_windows for p in ['SLOPE', 'ACCEL', 'JERK']]
        all_required = base_required + dynamic_required
        return self.helper._validate_required_signals(df, all_required, method_name)

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Dict[str, pd.Series]:
        """V69.0 · 原料加载层：增强量纲自适应对齐与鲁棒性过滤"""
        is_debug, probe_ts, _ = self._setup_debug_info(df, method_name)
        raw_signals = {}
        base_cols = ['close_D', 'high_D', 'low_D', 'volume_D', 'amount_D', 'pct_change_D', 'net_amount_rate_D', 'trade_count_D', 'turnover_rate_f_D']
        struct_cols = ['winner_rate_D', 'chip_concentration_ratio_D', 'chip_entropy_D', 'cost_50pct_D', 'absorption_energy_D', 'GEOM_ARC_CURVATURE_D', 'GEOM_REG_R2_D', 'price_percentile_position_D']
        tech_cols = ['SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', 'closing_flow_intensity_D', 'T1_PREMIUM_EXPECTATION_D', 'pressure_release_index_D', 'up_limit_D', 'down_limit_D', 'closing_flow_ratio_D', 'TURNOVER_STABILITY_INDEX_D', 'IS_EMOTIONAL_EXTREME_D', 'flow_consistency_D', 'industry_strength_rank_D', 'industry_rank_accel_D', 'IS_ROUNDING_BOTTOM_D', 'IS_GOLDEN_PIT_D', 'IS_TRENDING_STAGE_D', 'THEME_HOTNESS_SCORE_D', 'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', 'market_sentiment_score_D']
        # 1. 基础数据加载
        for col in base_cols + struct_cols + tech_cols:
            if col not in df.columns: raise KeyError(f"CRITICAL: 军械库缺失关键列 {col}")
            if col in struct_cols or 'rank' in col: raw_signals[col] = df[col].ffill().fillna(0.0)
            else: raw_signals[col] = df[col].fillna(0.0)
        # 2. VWAP 自适应单位对齐 (针对 Amount=万元, Volume=手 的特殊处理)
        raw_vwap = raw_signals['amount_D'] / (raw_signals['volume_D'] + 1e-9)
        # 统计样本对齐
        recent_closes = raw_signals['close_D'].iloc[-60:]
        recent_vwaps = raw_vwap.iloc[-60:].replace(0, np.nan).dropna()
        scale_factor = 1.0
        if not recent_vwaps.empty:
            median_ratio = (recent_vwaps / recent_closes.loc[recent_vwaps.index]).median()
            if median_ratio < 0.5 or median_ratio > 2.0:
                # 寻找最近的 10 的幂次进行补偿 (例如 0.01 -> 100, 0.1 -> 10)
                scale_factor = 10.0 ** (-np.round(np.log10(median_ratio)))
        raw_signals['VWAP_D'] = raw_vwap * scale_factor
        # 3. 筹码与排名数据归一化
        if raw_signals['winner_rate_D'].max() > 1.1: raw_signals['winner_rate_D'] /= 100.0
        # 4. 主力资金降级补偿
        if raw_signals['SMART_MONEY_HM_NET_BUY_D'].abs().sum() < 1e-5:
            raw_signals['SMART_MONEY_HM_NET_BUY_D'] = (raw_signals['buy_elg_amount_D'] - raw_signals['sell_elg_amount_D']) + (raw_signals['buy_lg_amount_D'] - raw_signals['sell_lg_amount_D'])
        # 5. 执行鲁棒动力学提取
        threshold_map = {'net_amount_rate_D': (0.01, 0.005), 'winner_rate_D': (0.01, 0.005), 'SMART_MONEY_HM_NET_BUY_D': (10, 10), 'VPA_EFFICIENCY_D': (0.01, 0.01), 'BBW_21_2.0_D': (0.001, 0.0001), 'THEME_HOTNESS_SCORE_D': (0.1, 0.1), 'chip_entropy_D': (0.01, 0.001), 'volume_D': (100, 10), 'pct_change_D': (0.0001, 0.0001), 'close_D': (0.1, 0.01), 'market_sentiment_score_D': (0.1, 0.1), 'industry_strength_rank_D': (0.001, 0.001), 'turnover_rate_f_D': (0.01, 0.01), 'trade_count_D': (10, 5)}
        fib_windows = [3, 5, 8, 13, 21]
        for col, (abs_th, chg_th) in threshold_map.items():
            base_vals = raw_signals.get(col, pd.Series(0.0, index=df.index)).values
            for win in fib_windows:
                s, a, j = _numba_robust_dynamics(base_vals, win=win, abs_threshold=abs_th, change_threshold=chg_th)
                raw_signals[f"SLOPE_{win}_{col}"] = pd.Series(s, index=df.index)
                raw_signals[f"ACCEL_{win}_{col}"] = pd.Series(a, index=df.index)
                raw_signals[f"JERK_{win}_{col}"] = pd.Series(j, index=df.index)
        if is_debug and probe_ts in df.index:
            print(f"\n[原料自适应探针 V69.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    VWAP对齐系数: {scale_factor}, 最终VWAP: {raw_signals['VWAP_D'].loc[probe_ts]:.2f}")
        return raw_signals

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """V66.0 · 主力夺权全局总线：全模块动力学闭环 (Full Kinematic Loop)"""
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
        # 合成：物理(50%) + 几何(25%) + VWAP(25%)
        unadjusted_intensity = (physical_base * 0.50 + geo_resonance * 0.25 + vwap_propulsion * 0.25)
        # --- 第二阶段：调节矩阵 (全动力学 V66) ---
        # 1. 结构与环境
        trend_inertia = self._calculate_trend_inertia_momentum(raw, df_index, method_name)
        market_perm = self._calculate_market_permeability_index(raw, df_index, method_name)
        entropy_adj = self._calculate_entropic_ordering_bonus(raw, df_index, method_name)
        fractal_efficiency = self._calculate_fractal_efficiency_resonance(raw, df_index, method_name)
        hmm_confirm = self._calculate_hmm_regime_confirmation(raw, df_index, method_name)
        # 2. 筹码与微观
        chip_lock = self._calculate_chip_lock_efficiency(raw, df_index, method_name)
        micro_attack = self._calculate_microstructure_attack_vector(raw, df_index, method_name)
        vpa_reflexivity = self._calculate_vpa_elasticity_reflexivity(raw, df_index, method_name)
        wyckoff_quality = self._calculate_wyckoff_breakout_quality(raw, df_index, method_name)
        # 3. 风险阀 (V66 升级)
        risk_v = self._calculate_premium_reversal_risk(raw, df_index, method_name)
        decay_v = self._calculate_intraday_decay_model(raw, df_index, method_name)
        sector_v = self._calculate_sector_resonance_modifier(raw, df_index, method_name)
        # V66 升级点：波动率与板块溢出引入动力学
        vol_v = self._calculate_volatility_clustering_adjustment(raw, df_index, method_name)
        sector_overflow_v = self._calculate_sector_overflow_decay(raw, df_index, method_name)
        # --- 第三阶段：实战入场修正 ---
        access_f = self._calculate_entry_accessibility_score(raw, df_index, method_name)
        # --- 第四阶段：最终合成 ---
        # 逻辑：(强度 * 趋势 * 环境 * 结构 * 微观) * 安全阀 * 执行度
        final_score = (unadjusted_intensity * trend_inertia * market_perm * entropy_adj * fractal_efficiency * \
                       hmm_confirm * chip_lock * micro_attack * vpa_reflexivity * wyckoff_quality) * \
                      risk_v * decay_v * sector_v * vol_v * sector_overflow_v * access_f
        if is_debug and probe_ts in df_index:
            print(f"\n[PROCESS_META_POWER_TRANSFER 全链集成探针 V66 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    [基准] 物理: {physical_base.loc[probe_ts]:.4f}, 几何: {geo_resonance.loc[probe_ts]:.4f}")
            print(f"    [阀门] 波动率: {vol_v.loc[probe_ts]:.2f}, 板块溢出: {sector_overflow_v.loc[probe_ts]:.2f}")
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
        """V66.0 · 波动率伽马模型：引入 Jerk 识别“波动率爆炸”与“陷阱”"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 波动率动力学 (Volatility Kinematics)
        bbw = raw['BBW_21_2.0_D']
        slope_bbw = raw['SLOPE_5_BBW_21_2.0_D']
        accel_bbw = raw['ACCEL_5_BBW_21_2.0_D']
        jerk_bbw = raw['JERK_3_BBW_21_2.0_D']
        bbw_ma21 = bbw.rolling(21).mean().fillna(bbw)
        # 2. 波动率爆炸 (Volatility Explosion)
        # 逻辑：低位 + 加速扩张 + 脉冲强烈
        is_low_base = bbw < bbw_ma21 * 1.2
        is_exploding = is_low_base & (accel_bbw > 0) & (jerk_bbw > 0.01)
        # 3. 波动率陷阱 (Volatility Trap)
        # 逻辑：高位 + 减速扩张 (Slope>0, Accel<0) 或 掉头向下 (Slope<0)
        is_high_base = bbw > bbw_ma21 * 1.5
        is_trap = is_high_base & ((slope_bbw < 0) | (accel_bbw < 0))
        # 4. 方向性确认 (Directional Confirmation)
        # 波动率变大是好事还是坏事，取决于价格的 Jerk
        price_jerk = raw['JERK_3_close_D']
        # 正向爆炸：波动率炸裂 + 价格向上脉冲
        positive_gamma = is_exploding & (price_jerk > 0)
        # 负向崩塌：波动率炸裂 + 价格向下脉冲
        negative_gamma = is_exploding & (price_jerk < 0)
        # 5. 综合调节系数
        # 基础分
        adj = np.ones(len(df_index))
        # 正向爆炸：极度增强 (1.5x)
        adj = np.where(positive_gamma, 1.5, adj)
        # 负向崩塌：强力熔断 (0.5x)
        adj = np.where(negative_gamma, 0.5, adj)
        # 陷阱/耗竭：温和衰减 (0.8x)
        adj = np.where(is_trap, 0.8, adj)
        final_adj = pd.Series(adj, index=df_index).clip(0.4, 1.6)
        if is_debug and probe_ts in df_index:
            print(f"\n[波动率伽马探针 V66 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    BBW动力学: Accel={accel_bbw.loc[probe_ts]:.4f}, Jerk={jerk_bbw.loc[probe_ts]:.4f}")
            print(f"    价格脉冲: {price_jerk.loc[probe_ts]:.4f}")
            print(f"    状态: {'正向爆炸' if positive_gamma.loc[probe_ts] else ('负向崩塌' if negative_gamma.loc[probe_ts] else ('高位陷阱' if is_trap.loc[probe_ts] else '常态'))}")
            print(f"    >>> 最终波动率调节系数: {final_adj.loc[probe_ts]:.4f}")
            
        return final_adj.astype(np.float32)

    def _calculate_sector_overflow_decay(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V66.0 · 熵增雪崩模型：引入分形维数加速度(Accel)预警板块崩塌"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 计算分形维数
        theme_hotness = raw['THEME_HOTNESS_SCORE_D'].fillna(0).values
        flow_arrays = np.expand_dims(theme_hotness, axis=0) 
        fractal_dim_vals = _numba_fractal_dimension(flow_arrays, window=13)
        fractal_dim = pd.Series(fractal_dim_vals, index=df_index)
        # 2. 熵动力学 (Entropic Kinematics)
        # 分形维数的斜率与加速度
        fd_slope = fractal_dim.diff(5).fillna(0)
        fd_accel = fd_slope.diff(5).fillna(0)
        # 3. 雪崩预警 (Avalanche Warning)
        # 逻辑：板块热度很高 (>80) + 内部结构正在加速混沌化 (Accel > 0)
        is_hot = raw['THEME_HOTNESS_SCORE_D'] > 80
        # 熵增加速：分形维数在变大，且变大的速度在加快
        is_entropic_accel = (fd_slope > 0) & (fd_accel > 0)
        is_avalanche = is_hot & is_entropic_accel
        # 4. 衰减系数计算
        # 基础衰减：分形维数越高，衰减越大 (1.5 -> 1.0)
        base_decay = (1.5 / (fractal_dim + 1e-9)).clip(0.6, 1.1)
        # 雪崩惩罚：如果触发雪崩预警，在基础衰减上再打 7 折
        final_decay = np.where(is_avalanche, base_decay * 0.7, base_decay)
        res = pd.Series(final_decay, index=df_index).astype(np.float32)
        if is_debug and probe_ts in df_index:
            print(f"\n[熵增雪崩探针 V66 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    热度: {raw['THEME_HOTNESS_SCORE_D'].loc[probe_ts]:.1f}, 分形维数: {fractal_dim.loc[probe_ts]:.4f}")
            print(f"    熵动力学: Slope={fd_slope.loc[probe_ts]:.4f}, Accel={fd_accel.loc[probe_ts]:.4f}")
            print(f"    状态: {'雪崩预警(熵增加速)' if is_avalanche.loc[probe_ts] else '稳定'}")
            print(f"    >>> 最终板块溢出衰减: {res.loc[probe_ts]:.4f}")
            
        return res

    def _calculate_hmm_regime_confirmation(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V68.0 · HMM 动力学共振：适配降级后的特征归一化"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 特征归一化 (Rolling Z-Score) with Safety
        large_net = (raw['SMART_MONEY_HM_NET_BUY_D']).fillna(0)
        roll_mean = large_net.rolling(21).mean().fillna(0)
        roll_std = large_net.rolling(21).std().fillna(0)
        # 如果标准差太小，说明数据基本没动，归一化结果置 0
        flow_n = np.where(roll_std > 1e-5, (large_net - roll_mean) / roll_std, 0.0)
        flow_n = pd.Series(flow_n, index=df_index)
        vol_mean = raw['volume_D'].rolling(21).mean().replace(0, 1)
        vol_n = (raw['volume_D'] / vol_mean) - 1.0
        price_n = raw['pct_change_D'] * 10.0
        # VWAP 距离：(Close - VWAP) / BBW
        # 使用 max(BBW, 0.01) 防止除零放大
        safe_bbw = raw['BBW_21_2.0_D'].clip(lower=0.01)
        vwap_dist = (raw['close_D'] - raw['VWAP_D']) / (raw['close_D'] * safe_bbw)
        # 2. 计算拉升概率
        markup_prob_vals = _numba_hmm_regime_probability(
            flow_n.fillna(0).values, 
            vol_n.fillna(0).values, 
            price_n.fillna(0).values, 
            vwap_dist.fillna(0).values
        )
        markup_prob = pd.Series(markup_prob_vals, index=df_index)
        # 3. HMM 动力学
        prob_accel = markup_prob.diff(3).diff(3).fillna(0)
        prob_jerk = markup_prob.diff(1).diff(1).diff(1).fillna(0)
        is_solid_markup = (markup_prob > 0.6) & (prob_accel > 0)
        is_phase_transition = (markup_prob > 0.4) & (prob_jerk > 0.1)
        base_factor = np.where(markup_prob > 0.5, 
                               1.0 + (markup_prob - 0.5), 
                               0.8 + markup_prob * 0.4)
        dynamic_bonus = 1.0 + np.where(is_solid_markup, 0.2, 0.0) + np.where(is_phase_transition, 0.3, 0.0)
        res = pd.Series(base_factor * dynamic_bonus, index=df_index)
        if is_debug and probe_ts in df_index:
            print(f"\n[HMM 动力学探针 V68.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    主力归一化流: {flow_n.loc[probe_ts]:.4f} (原始: {large_net.loc[probe_ts]:.2f})")
            print(f"    VWAP距离因子: {vwap_dist.loc[probe_ts]:.4f}")
            print(f"    拉升概率: {markup_prob.loc[probe_ts]:.4f}")
            print(f"    >>> 最终 HMM 确认系数: {res.loc[probe_ts]:.4f}")
            
        return res.astype(np.float32)

    def _calculate_fractal_efficiency_resonance(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V65.1 · 分形动力学模型：诊断 Hurst=0.5 的原因"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        close_vals = raw['close_D'].fillna(0).values
        vol_vals = raw['volume_D'].fillna(0).values
        input_arrays = np.vstack((close_vals, vol_vals))
        # 1. 计算 Hurst 指数
        fractal_dims = _numba_fractal_dimension(input_arrays, window=21)
        hurst_price = pd.Series(2.0 - fractal_dims[0], index=df_index).clip(0, 1)
        hurst_vol = pd.Series(2.0 - fractal_dims[1], index=df_index).clip(0, 1)
        # 2. 分形动力学
        hurst_slope = hurst_price.diff(5).fillna(0)
        is_hardening = (hurst_price > 0.55) & (hurst_slope > 0)
        # 3. 效率缺口
        eff_gap = (hurst_price - hurst_vol).abs()
        gap_accel = eff_gap.diff(3).diff(3).fillna(0)
        is_collapsing = (eff_gap > 0.3) & (gap_accel > 0)
        # 4. 综合效率系数 (使用 numpy 操作避免 Series 错误)
        persistence = np.where(hurst_price > 0.55, 1.2, 
                      np.where(hurst_price < 0.45, 0.6, 0.9))
        resonance_factor = (1.0 - eff_gap * 2.0).clip(0.5, 1.1)
        base_vals = persistence * resonance_factor
        mask_h = is_hardening.values
        mask_c = is_collapsing.values
        final_vals = np.where(mask_h, base_vals * 1.2, base_vals)
        final_vals = np.where(mask_c, final_vals * 0.7, final_vals)
        final_score = pd.Series(final_vals, index=df_index)
        if is_debug and probe_ts in df_index:
            print(f"\n[分形动力学探针 V65.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
            idx_loc = df_index.get_loc(probe_ts)
            print(f"    当前索引位置: {idx_loc}, 需求窗口: 21 (若索引<21则Hurst为默认值0.5)")
            print(f"    价格Hurst: {hurst_price.loc[probe_ts]:.4f}, Hurst斜率: {hurst_slope.loc[probe_ts]:.4f}")
            print(f"    >>> 最终分形效率系数: {final_score.loc[probe_ts]:.4f}")
            
        return final_score.astype(np.float32)

    def _calculate_chip_lock_efficiency(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V68.0 · 筹码动力学锁定模型：适配归一化后的获利盘数据"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 基础磁滞状态
        # V68: winner_rate_D 已经在 _get_raw_signals 中保证是 0-1
        winner = raw['winner_rate_D'].fillna(0)
        turnover = raw['turnover_rate_f_D'].replace(0, 0.1)
        turnover_norm = (turnover / 5.0).clip(0, 1)
        static_lock = winner * (1.0 - turnover_norm).clip(0, 1)
        # 2. 动力学剪刀差
        accel_winner = raw['ACCEL_5_winner_rate_D']
        accel_turnover = raw['ACCEL_5_turnover_rate_f_D']
        is_vacuum_accel = (accel_winner > 0) & (accel_turnover < 0)
        # 3. 筹码脉冲
        jerk_winner = raw['JERK_3_winner_rate_D']
        is_breakthrough = jerk_winner > 0.1
        # 4. 动力学增强
        kinetic_bonus = 1.0 + np.where(is_vacuum_accel, 0.3, 0.0) + np.where(is_breakthrough, 0.2, 0.0)
        # 5. 成本支撑
        cost_series = raw['cost_50pct_D']
        close_series = raw['close_D']
        cost_50 = cost_series.mask(cost_series == 0, close_series)
        break_cost = (close_series > cost_50).astype(float)
        final_efficiency = pd.Series(static_lock * kinetic_bonus * (0.8 + break_cost * 0.2), index=df_index)
        if is_debug and probe_ts in df_index:
            print(f"\n[筹码动力学探针 V68.0 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    获利比例: {winner.loc[probe_ts]:.2f}, 静态锁定: {static_lock.loc[probe_ts]:.4f}")
            print(f"    >>> 最终筹码效率系数: {final_efficiency.loc[probe_ts]:.4f}")
            
        return final_efficiency.astype(np.float32)

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
        """V63.0 · 流体反身性模型：引入 Jerk/Slope 动力学识别“点火时刻”与“弹性趋势”"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 动态弹性 (Dynamic Elasticity)
        # 基础弹性 = %P / %V
        pct_price = raw['pct_change_D'].abs()
        pct_vol = raw['volume_D'].pct_change().fillna(0).abs()
        base_elasticity = pct_price / (pct_vol + 0.1)
        # 弹性趋势：计算弹性的 3 日斜率
        # 如果斜率 > 0，说明同样的量能带来的涨幅在变大（筹码变轻）
        elasticity_slope = base_elasticity.diff(3) / 3.0
        elasticity_bonus = np.where(elasticity_slope > 0, 1.2, 1.0)
        # 2. 索罗斯点火 (Soros Ignition - Jerk Resonance)
        # 寻找价格和成交量的"双脉冲"
        jerk_p = raw['JERK_3_close_D'] # 价格脉冲
        jerk_v = raw['JERK_3_volume_D'] # 成交量脉冲
        # 点火时刻：价格和量能同时出现正向剧烈脉冲
        is_ignition = (jerk_p > 0.05) & (jerk_v > 0.1)
        # 3. 反身性状态判定
        accel_p = raw['ACCEL_5_close_D'] # 改用 close 的加速度更平滑
        accel_v = raw['ACCEL_5_volume_D']
        # 正反馈：双加速
        positive_feedback = (accel_p > 0) & (accel_v > 0)
        # 4. 综合反身性系数
        # 基础分
        score = np.tanh(base_elasticity).clip(0.5, 1.5)
        # 叠加趋势红利与点火爆发
        final_score = pd.Series(score * elasticity_bonus, index=df_index)
        final_score = np.where(is_ignition, final_score * 1.5, # 点火时刻给予重奖
                      np.where(positive_feedback, final_score * 1.2, final_score))
        if is_debug and probe_ts in df_index:
            print(f"\n[流体反身性探针 V63 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    基础弹性: {base_elasticity.loc[probe_ts]:.4f}, 弹性斜率: {elasticity_slope.loc[probe_ts]:.4f}")
            print(f"    脉冲(Jerk): P={jerk_p.loc[probe_ts]:.4f}, V={jerk_v.loc[probe_ts]:.4f}")
            print(f"    状态: {'索罗斯点火(Ignition)' if is_ignition.loc[probe_ts] else ('正反馈' if positive_feedback.loc[probe_ts] else '常态')}")
            print(f"    >>> 最终 VPA 反身性系数: {final_score[df_index.get_loc(probe_ts)]:.4f}")
        return final_score.astype(np.float32)

    def _calculate_wyckoff_breakout_quality(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V63.0 · 动态威科夫模型：基于波动率加速度(Accel)与突破脉冲(Jerk)的质量判定"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 动态 VCP (Dynamic Volatility Contraction)
        # 计算真实波幅 TR
        high = raw['high_D']
        low = raw['low_D']
        close = raw['close_D']
        close_prev = close.shift(1).fillna(close)
        tr = np.maximum(high - low, np.abs(high - close_prev))
        tr_norm = tr / close
        # 计算 TR 的动力学特征 (手动计算，因 TR 未在 raw 中预先生成动力学)
        tr_slope = tr_norm.diff(5) / 5.0
        tr_accel = tr_slope.diff(5) / 5.0
        # 极致挤压 (Squeeze): 波动率在下降(Slope<0) 且 减速归零(Accel>0但微小, 或趋于平缓)
        # 当 Accel > 0 时意味着下降速度变慢，即 TR 曲线开始走平（死寂）
        is_squeeze = (tr_slope < 0) & (tr_accel > -0.01)
        squeeze_bonus = np.where(is_squeeze, 1.3, 1.0)
        # 2. 牛顿突破 (Newtonian Breakout)
        # 突破需要巨大的 Force (F=ma)，即价格的 Jerk
        jerk_p = raw['JERK_3_close_D']
        # 突破确认
        highest_21 = high.rolling(21).max().shift(1).fillna(99999)
        is_breakout = close > highest_21
        # 3. 质量判定
        # 真突破：价格创新高 + 巨大的正向脉冲 (Jerk > 0.1)
        is_impulsive_breakout = is_breakout & (jerk_p > 0.1)
        # 假突破/弱突破：价格创新高 但 Jerk 很小（由惯性滑过新高，无爆发力）
        is_weak_breakout = is_breakout & (jerk_p <= 0.05)
        # 4. 综合质量合成
        quality = np.where(is_impulsive_breakout, 1.6, # 强力突破
                  np.where(is_weak_breakout, 0.8,      # 惯性突破(弱)
                  np.where(is_squeeze, 1.1, 0.9)))     # 蓄势中
        # 叠加 VCP 挤压红利
        final_score = pd.Series(quality * squeeze_bonus, index=df_index).clip(0.6, 2.0)
        if is_debug and probe_ts in df_index:
            print(f"\n[动态威科夫探针 V63 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    TR斜率: {tr_slope.loc[probe_ts]:.4f}, TR加速度: {tr_accel.loc[probe_ts]:.4f}")
            print(f"    价格脉冲(Jerk): {jerk_p.loc[probe_ts]:.4f}")
            print(f"    状态: {'极致挤压(Squeeze)' if is_squeeze.loc[probe_ts] else '波动中'}")
            print(f"    突破判定: {'爆发性真突破' if is_impulsive_breakout.loc[probe_ts] else ('弱突破' if is_weak_breakout.loc[probe_ts] else '未突破')}")
            print(f"    >>> 最终威科夫质量系数: {final_score.loc[probe_ts]:.4f}")
        return final_score.astype(np.float32)

    def _calculate_trend_inertia_momentum(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V62.0 · 趋势运动学模型：基于高阶导数 (Slope/Accel/Jerk) 的推背感度量"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        close = raw['close_D']
        # 1. 均线多头 (基础状态)
        ma5 = close.rolling(5).mean()
        ma21 = close.rolling(21).mean()
        ma55 = close.rolling(55).mean()
        alignment_score = np.where((ma5 > ma21) & (ma21 > ma55), 1.0, 0.8)
        # 2. 运动学分析 (Kinematics)
        # 使用 5 日窗口的运动特征
        slope = raw['SLOPE_5_close_D'] # 速度
        accel = raw['ACCEL_5_close_D'] # 加速度
        jerk = raw['JERK_3_close_D']   # 脉冲 (使用3日捕捉瞬时突变)
        # 状态判定：
        # A. 逃逸速度 (Escape Velocity): 速度 > 0 且 加速度 > 0 (正在加速脱离引力)
        is_escape = (slope > 0) & (accel > 0)
        # B. 动能衰竭 (Exhaustion): 速度 > 0 但 加速度 < 0 (惯性滑行)
        is_exhaustion = (slope > 0) & (accel < 0)
        # C. 脉冲启动 (Impulse Start): 极大的正向 Jerk
        is_impulse = jerk > 0.1
        # 3. 动能系数合成
        # 基础分 1.0
        # 逃逸速度 +0.3
        # 脉冲启动 +0.2
        # 动能衰竭 -0.2
        kinematic_score = 1.0 + np.where(is_escape, 0.3, 0.0) + \
                          np.where(is_impulse, 0.2, 0.0) - \
                          np.where(is_exhaustion, 0.2, 0.0)
        # 4. R2 质量修正
        r2_quality = raw['GEOM_REG_R2_D'].clip(0, 1)
        final_inertia = pd.Series(alignment_score * kinematic_score * (0.6 + r2_quality * 0.4), index=df_index).clip(0.6, 1.6)
        if is_debug and probe_ts in df_index:
            print(f"\n[趋势运动学探针 V62 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    均线排列: {alignment_score[df_index.get_loc(probe_ts)]:.2f}")
            print(f"    速度(Slope): {slope.loc[probe_ts]:.4f}, 加速度(Accel): {accel.loc[probe_ts]:.4f}, 脉冲(Jerk): {jerk.loc[probe_ts]:.4f}")
            print(f"    状态: {'逃逸加速' if is_escape.loc[probe_ts] else ('动能衰竭' if is_exhaustion.loc[probe_ts] else '中性')}")
            print(f"    >>> 最终趋势惯性系数: {final_inertia.loc[probe_ts]:.4f}")
        return final_inertia.astype(np.float32)

    def _calculate_market_permeability_index(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V62.1 · 市场渗透率与环境相变：增加输入值诊断"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 情绪相变
        sentiment = raw['market_sentiment_score_D']
        sent_accel = raw['ACCEL_5_market_sentiment_score_D']
        is_overheat = (sentiment > 80) & (sent_accel > 0)
        is_recovery = (sentiment < 20) & (sent_accel > 0)
        # 2. 行业排名跃迁
        rank = raw['industry_strength_rank_D']
        rank_jerk = raw['JERK_3_industry_strength_rank_D']
        is_rank_surge = rank_jerk < -2.0 
        # 3. 渗透率合成
        permeability = np.where(is_recovery, 1.3, 
                       np.where(is_overheat, 0.7, 1.0))
        rank_bonus = np.where(is_rank_surge, 1.3, 
                     np.where(rank < 10, 1.1, 0.9))
        final_ctx = pd.Series(permeability * rank_bonus, index=df_index).clip(0.6, 1.8)
        if is_debug and probe_ts in df_index:
            print(f"\n[环境相变探针 V62.1 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    原始情绪: {sentiment.loc[probe_ts]:.4f}, 情绪加速(Accel): {sent_accel.loc[probe_ts]:.4f}")
            print(f"    原始排名: {rank.loc[probe_ts]:.4f}, 排名脉冲(Jerk): {rank_jerk.loc[probe_ts]:.4f}")
            print(f"    判定逻辑 -> 复苏条件(Sent<20 & Acc>0): {is_recovery.loc[probe_ts]}")
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



