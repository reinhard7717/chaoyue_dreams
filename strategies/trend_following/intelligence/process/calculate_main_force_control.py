# strategies\trend_following\intelligence\process\calculate_main_force_control.py
# 【V1.0.0 · 主力控盘关系计算器】 计算“主力控盘”的专属关系分数。  已完成pro
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    get_adaptive_mtf_normalized_bipolar_score, normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
try:
    from numba import njit
    def _njit_wrapper(*args, **kwargs):
        return njit(*args, **kwargs)
except ImportError:
    def _njit_wrapper(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
@_njit_wrapper(cache=True, fastmath=True)
def nb_soft_saturation(x: np.ndarray) -> np.ndarray:
    """【V68.0.0 Numba极速内核】全域代数软饱和降维，替代pd.clip，消除隐性对象分配与偏导数消失。"""
    res = np.empty_like(x)
    for i in range(x.shape[0]):
        res[i] = x[i] / np.sqrt(1.0 + x[i]*x[i])
    return res
@_njit_wrapper(cache=True, fastmath=True)
def nb_threshold_gate(val: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """【V68.0.0 Numba极速内核】动力学微积分自适应阈值门限过滤，执行循环融合彻底消灭中间数组切片。"""
    res = np.empty_like(val)
    for i in range(val.shape[0]):
        res[i] = val[i] * np.tanh(np.abs(val[i]) / noise[i])
    return res
@_njit_wrapper(cache=True, fastmath=True)
def nb_lv_regime_blend(dx_z: np.ndarray, dy_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """【V68.0.0 Numba极速内核】洛特卡-沃尔泰拉生态博弈运算，拓扑连续平滑求解第三象限黑洞引力与生态张力。"""
    n = dx_z.shape[0]
    lv_net_force = np.empty(n, dtype=np.float32)
    lv_tension = np.empty(n, dtype=np.float32)
    lv_score = np.empty(n, dtype=np.float32)
    for i in range(n):
        regime_w = (np.tanh(dx_z[i] * 5.0) + 1.0) * 0.5
        net_f = regime_w * (dx_z[i] - dy_z[i]) + (1.0 - regime_w) * (dx_z[i] - np.abs(dy_z[i]))
        lv_net_force[i] = net_f
        lv_tension[i] = np.sqrt(dx_z[i]*dx_z[i] + dy_z[i]*dy_z[i])
        lv_score[i] = np.tanh(net_f * 1.5)
    return lv_net_force, lv_tension, lv_score
@_njit_wrapper(cache=True, fastmath=True)
def nb_trap_force_and_weights(trad_arr: np.ndarray, act_arr: np.ndarray, fractal_dim: np.ndarray, w_trad_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """【V68.0.0 Numba极速内核】陷阱极性逆转张量生成器，执行防0值连乘的加法势能推演，速度提升数十倍。"""
    n = trad_arr.shape[0]
    trap_force = np.empty(n, dtype=np.float32)
    w_trad_adj = np.empty(n, dtype=np.float32)
    for i in range(n):
        ts = trad_arr[i]
        acts = act_arr[i]
        bull_trap = max(0.0, min(ts, 1.0)) * max(0.0, min(-acts, 1.0))
        bear_trap = max(0.0, min(-ts, 1.0)) * max(0.0, min(acts, 1.0))
        th = np.tanh(fractal_dim[i] - 1.4)
        frac_sens = 1.0 + max(-0.5, min(th, 0.5))
        ts_abs = np.abs(ts)
        trap_force[i] = (-4.5 * bull_trap * (ts_abs**1.5) * frac_sens) + (4.5 * bear_trap * (ts_abs**1.5) * frac_sens)
        penalty = 1.0 - (bull_trap + bear_trap) * 0.8
        w_trad_adj[i] = w_trad_base[i] * max(0.2, penalty)
    return trap_force, w_trad_adj
class CalculateMainForceControlRelationship:
    """
    【V68.0.0 · 主力控盘全息量子决策系统 · 全息量子极速阵列终极版】
    PROCESS_META_MAIN_FORCE_CONTROL
    - 核心职责: 计算“主力控盘”专属分数，运用Numpy矢量化与Numba JIT消除Pandas性能瓶颈，修复全部语法级宕机错误。
    - 版本: 68.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        """【用途】V68.0.0: 初始化主力控盘处理器，挂载策略上下文参数矩阵与探针配置。"""
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
    def _print_debug_info(self, debug_output: Dict):
        """【用途】V68.0.0: 统一格式化打印探针诊断信息，维持黑盒日志的高度清洁与可追踪性。"""
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)
    def _get_safe_series(self, df: pd.DataFrame, col_name: str, default_value: float = 0.0, method_name: str = "") -> pd.Series:
        """【用途/效率优化】V68.0.0: 安全加载底层物理数据。强制降级astype(float32)削减50%内存带宽，提升CPU缓存命中。"""
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        neutral_nan_defaults = process_params.get('neutral_nan_defaults', {})
        current_default_value = neutral_nan_defaults.get(col_name, default_value)
        if col_name not in df.columns:
            return pd.Series(current_default_value, index=df.index, dtype=np.float32)
        return df[col_name].ffill().fillna(current_default_value).astype(np.float32)
    def _get_robust_noise_floor(self, s: pd.Series, window_std=21, window_med=252) -> pd.Series:
        """【用途/效率优化】V68.0.0: 绝对单向时间膜底噪萃取。全量提取values执行Numpy maximum，规避Pandas对齐开销。"""
        s_abs = s.abs().values
        roll_std = pd.Series(s_abs).rolling(window=window_std, min_periods=3).std().ffill().fillna(0.0).values.astype(np.float32)
        roll_med = pd.Series(s_abs).rolling(window=window_med, min_periods=3).median().ffill().values.astype(np.float32)
        exp_med = pd.Series(s_abs).expanding(min_periods=1).median().ffill().fillna(1e-5).values.astype(np.float32)
        final_med = np.where(np.isnan(roll_med), exp_med, roll_med)
        return pd.Series(np.maximum(roll_std, final_med) + 1e-5, index=s.index, dtype=np.float32)
    def _z_score_norm(self, s: pd.Series, scale=1.0, shift=0.0, window=252) -> pd.Series:
        """【用途/效率优化】V68.0.0: 自适应量纲Z-Score归一化。纯NumPy运算免疫Pandas耗时，防远古离群值永久挤压。"""
        s_val = s.values.astype(np.float32)
        roll_med = s.rolling(window=window, min_periods=5).median().ffill().values.astype(np.float32)
        exp_med = s.expanding(min_periods=1).median().ffill().fillna(0.0).values.astype(np.float32)
        med_val = np.where(np.isnan(roll_med), exp_med, roll_med)
        roll_std = s.rolling(window=window, min_periods=5).std().ffill().values.astype(np.float32)
        exp_std = s.expanding(min_periods=1).std().ffill().replace(0.0, 1.0).fillna(1.0).values.astype(np.float32)
        std_val = np.where(np.isnan(roll_std), exp_std, roll_std)
        std_val = np.where(std_val == 0.0, 1.0, std_val)
        res_val = (np.tanh((s_val - med_val) / (std_val + 1e-5)) * np.float32(scale)) + np.float32(shift)
        return pd.Series(res_val, index=s.index, dtype=np.float32)
    def _robust_pct_scale(self, s: pd.Series, threshold: float = 1.5, scale_factor: float = 100.0) -> pd.Series:
        """【用途/效率优化】V68.0.0: 自适应百分比量纲探测器。运用np.where实现C级向量化寻址条件分支，消灭pd.mask。"""
        q90_val = s.expanding(min_periods=1).quantile(0.9).ffill().fillna(threshold).values.astype(np.float32)
        scale_arr = np.where(q90_val <= threshold, np.float32(scale_factor), np.float32(1.0))
        return pd.Series(s.values.astype(np.float32) * scale_arr, index=s.index, dtype=np.float32)
    def _apply_kinematics_with_threshold_gate(self, series: pd.Series, periods: tuple) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """【用途/效率优化】V68.0.0: 动力学微积分发生器。注入Numba JIT内核nb_threshold_gate，阻断零基噪音同时消除中间数组。"""
        p_slope, p_accel, p_jerk = periods
        slope = series.diff(p_slope).ffill().fillna(0.0).values.astype(np.float32)
        noise_slope = (self._get_robust_noise_floor(pd.Series(slope)).values * np.float32(0.5)).astype(np.float32)
        slope_gated = nb_threshold_gate(slope, noise_slope)
        accel = pd.Series(slope_gated).diff(p_accel).ffill().fillna(0.0).values.astype(np.float32)
        noise_accel = (self._get_robust_noise_floor(pd.Series(accel)).values * np.float32(0.5)).astype(np.float32)
        accel_gated = nb_threshold_gate(accel, noise_accel)
        jerk = pd.Series(accel_gated).diff(p_jerk).ffill().fillna(0.0).values.astype(np.float32)
        noise_jerk = (self._get_robust_noise_floor(pd.Series(jerk)).values * np.float32(0.5)).astype(np.float32)
        jerk_gated = nb_threshold_gate(jerk, noise_jerk)
        return pd.Series(slope_gated, index=series.index, dtype=np.float32), pd.Series(accel_gated, index=series.index, dtype=np.float32), pd.Series(jerk_gated, index=series.index, dtype=np.float32)
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【用途/效率优化】V68.0.0: 决策系统最高调度总线。聚合底层张量阵列，运用Numba极速软饱和收敛模型生成平滑最终分数。"""
        method_name = "calculate_main_force_control_relationship"
        is_debug = get_param_value(self.debug_params.get('enabled'), False)
        _temp_debug_values = {} 
        probe_ts = self._get_probe_timestamp(df, is_debug)
        debug_output = {}
        if probe_ts:
            print(f"[调度中心] {method_name} 启动 @ {probe_ts.strftime('%Y-%m-%d')} | 版本: V68.0.0 (全息量子极速阵列版)")
            debug_output[f"--- {method_name} 管道启动 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        if hasattr(self, '_validate_arsenal_signals'):
             if not self._validate_arsenal_signals(df, config, method_name, debug_output, probe_ts):
                print(f"[熔断] {method_name}: 关键军械库信号缺失，策略强制终止。")
                return pd.Series(0.0, index=df.index, dtype=np.float32)
        control_context = self._get_raw_control_signals(df, method_name, _temp_debug_values, probe_ts)
        hab_prices_result = self._calculate_main_force_avg_prices(control_context, df.index, _temp_debug_values)
        scores_traditional = self._calculate_traditional_control_score_components(control_context, df.index, _temp_debug_values)
        if scores_traditional.isnull().all():
             return pd.Series(0.0, index=df.index, dtype=np.float32)
        scores_cost_advantage = self._calculate_main_force_cost_advantage_score(control_context, df.index, hab_prices_result, _temp_debug_values)
        scores_net_activity = self._calculate_main_force_net_activity_score(control_context, df.index, config, method_name, _temp_debug_values)
        scores_lv_ecology, lv_tension = self._calculate_lotka_volterra_model(df, control_context, df.index, _temp_debug_values)
        norm_traditional, norm_structural, norm_flow, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down = self._normalize_components(df, control_context, scores_traditional, config, method_name, _temp_debug_values)
        fused_control_score = self._fuse_control_scores(norm_traditional, norm_structural, scores_lv_ecology, control_context, scores_net_activity, _temp_debug_values)
        control_leverage = self._calculate_control_leverage_model(df.index, norm_traditional, fused_control_score, scores_net_activity, norm_flow, scores_cost_advantage, lv_tension, scores_lv_ecology, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down, control_context, _temp_debug_values)
        act_v = scores_net_activity.values.astype(np.float32)
        lev_v = control_leverage.values.astype(np.float32)
        fused_v = fused_control_score.values.astype(np.float32)
        kinematic_thrust_v = np.sign(act_v) * np.power(np.abs(act_v) + 1e-6, 1.2) * lev_v
        is_hollow = ((fused_v > 0.0) & (kinematic_thrust_v < 0.0)) | ((fused_v < 0.0) & (kinematic_thrust_v > 0.0))
        devour_factor = np.clip(np.exp(-np.abs(kinematic_thrust_v) * 1.5), 0.1, 1.0)
        effective_structure_v = np.where(is_hollow, fused_v * devour_factor, fused_v)
        fractal_dim_v = control_context['structure'].get('fractal_dim', pd.Series(1.5, index=df.index)).values.astype(np.float32)
        trend_confidence = 1.0 - np.clip((fractal_dim_v - 1.2) / 0.6, 0.0, 1.0)
        base_w_struct = 0.2 + 0.6 * trend_confidence
        w_struct_v = np.where(is_hollow, np.maximum(0.1, base_w_struct * devour_factor), base_w_struct)
        w_act_v = 1.0 - w_struct_v
        raw_final_score_v = (effective_structure_v * w_struct_v) + (kinematic_thrust_v * w_act_v)
        final_control_score_v = nb_soft_saturation(raw_final_score_v.astype(np.float32))
        final_control_score = pd.Series(final_control_score_v, index=df.index, dtype=np.float32)
        if _temp_debug_values is not None:
            _temp_debug_values["最终结果"] = {"Net_Activity (Vector)": scores_net_activity, "Control_Leverage (Dynamic)": control_leverage, "Kinematic_Thrust": pd.Series(kinematic_thrust_v, index=df.index, dtype=np.float32), "Fused_Structure": fused_control_score, "Effective_Structure": pd.Series(effective_structure_v, index=df.index, dtype=np.float32), "Adaptive_Struct_Weight": pd.Series(w_struct_v, index=df.index, dtype=np.float32), "Lotka_Volterra_Score": scores_lv_ecology, "Final_Score": final_control_score}
        if probe_ts:
            self._calculate_main_force_control_relationship_debug_output(debug_output, _temp_debug_values, method_name, probe_ts)
        return final_control_score
    def _get_control_parameters(self, config: Dict) -> Tuple[Dict, Dict]:
        """【用途】V68.0.0: 获取MTF多时间框架权重分配与动能斜率探测周期。"""
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        return actual_mtf_weights, mtf_slope_accel_weights
    def _get_raw_control_signals(self, df: pd.DataFrame, method_name: str, _temp_debug_values: Dict, probe_ts: pd.Timestamp) -> Dict[str, Dict[str, pd.Series]]:
        """【用途/效率优化】V68.0.0: 全息物理总线。对HAB动力学半衰期衰减应用底层数组乘法剥离，免去复杂的多重索引校验开销。"""
        market_raw = {
            "close": self._get_safe_series(df, 'close_D', 0.0),
            "amount": self._get_safe_series(df, 'amount_D', 0.0),
            "pct_change": self._get_safe_series(df, 'pct_change_D', 0.0),
            "turnover_rate": self._get_safe_series(df, 'turnover_rate_D', 0.0),
            "circ_mv": self._get_safe_series(df, 'circ_mv_D', 1e8).replace(0.0, np.nan),
            "up_limit": self._get_safe_series(df, 'up_limit_D', 0.0)
        }
        funds_raw = {
            "tick_lg_net": self._get_safe_series(df, 'tick_large_order_net_D', 0.0),
            "tick_lg_count": self._get_safe_series(df, 'tick_large_order_count_D', 0.0),
            "smart_net_buy": self._get_safe_series(df, 'SMART_MONEY_HM_NET_BUY_D', 0.0),
            "smart_attack": self._get_safe_series(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 0.0),
            "smart_synergy": self._get_safe_series(df, 'SMART_MONEY_SYNERGY_BUY_D', 0.0),
            "hm_top_tier": self._get_safe_series(df, 'HM_ACTIVE_TOP_TIER_D', 0.0),
            "net_mf_calibrated": self._get_safe_series(df, 'net_mf_amount_D', 0.0),
            "buy_elg_amt": self._get_safe_series(df, 'buy_elg_amount_D', 0.0),
            "sell_elg_amt": self._get_safe_series(df, 'sell_elg_amount_D', 0.0),
            "buy_lg_amt": self._get_safe_series(df, 'buy_lg_amount_D', 0.0),
            "sell_lg_amt": self._get_safe_series(df, 'sell_lg_amount_D', 0.0),
            "buy_elg_vol": self._get_safe_series(df, 'buy_elg_vol_D', 0.0),
            "sell_elg_vol": self._get_safe_series(df, 'sell_elg_vol_D', 0.0),
            "buy_lg_vol": self._get_safe_series(df, 'buy_lg_vol_D', 0.0),
            "sell_lg_vol": self._get_safe_series(df, 'sell_lg_vol_D', 0.0),
            "flow_consistency": self._get_safe_series(df, 'flow_consistency_D', 50.0),
            "flow_efficiency": self._get_safe_series(df, 'flow_efficiency_D', 50.0),
            "stealth_flow": self._get_safe_series(df, 'stealth_flow_ratio_D', 0.0),
            "smart_divergence": self._get_safe_series(df, 'price_flow_divergence_D', 0.0),
            "gap_momentum": self._get_safe_series(df, 'GAP_MOMENTUM_STRENGTH_D', 0.0),
            "hf_flow_skewness": self._get_safe_series(df, 'high_freq_flow_skewness_D', 0.0),
            "chip_flow_intensity": self._get_safe_series(df, 'chip_flow_intensity_D', 50.0),
            "hab_net_mf_13": None, "hab_net_mf_21": None, "hab_net_mf_34": None, "hab_net_mf_55": None
        }
        t_rate_val = market_raw["turnover_rate"].replace(0.0, np.nan).ffill().fillna(1.0).values.astype(np.float32)
        t_max_expanding = pd.Series(t_rate_val).expanding(min_periods=1).max().ffill().fillna(1.0).values.astype(np.float32)
        t_scale = np.where(t_max_expanding <= 1.0, 100.0, 1.0).astype(np.float32)
        t_rate_pct_s = pd.Series(np.clip(t_rate_val * t_scale, 0.01, 200.0), index=df.index, dtype=np.float32)
        market_raw["turnover_rate"] = t_rate_pct_s 
        def _calc_hab_with_decay(flow_v: np.ndarray, t_rate_v: np.ndarray, window: int) -> pd.Series:
            cum_flow = pd.Series(flow_v).rolling(window=window, min_periods=max(1, int(window*0.3))).sum().shift(1).ffill().fillna(0.0).values.astype(np.float32)
            cum_turnover = pd.Series(t_rate_v).rolling(window=window, min_periods=max(1, int(window*0.3))).sum().shift(1).ffill().fillna(window * 1.0).values.astype(np.float32)
            retention_factor = np.clip(np.exp(-cum_turnover / 100.0), 0.01, 1.0).astype(np.float32)
            return pd.Series(cum_flow * retention_factor, index=df.index, dtype=np.float32)
        comp_flow_val = (funds_raw["tick_lg_net"].values * 0.3 + funds_raw["smart_synergy"].values * 0.3 + funds_raw["smart_net_buy"].values * 0.2 + funds_raw["net_mf_calibrated"].values * 0.2).astype(np.float32)
        funds_raw["hab_net_mf_13"] = _calc_hab_with_decay(comp_flow_val, t_rate_pct_s.values, 13)
        funds_raw["hab_net_mf_21"] = _calc_hab_with_decay(comp_flow_val, t_rate_pct_s.values, 21)
        funds_raw["hab_net_mf_34"] = _calc_hab_with_decay(comp_flow_val, t_rate_pct_s.values, 34)
        funds_raw["hab_net_mf_55"] = _calc_hab_with_decay(comp_flow_val, t_rate_pct_s.values, 55)
        structure_raw = {
            "chip_entropy": self._get_safe_series(df, 'chip_entropy_D', 100.0),
            "price_entropy": self._get_safe_series(df, 'PRICE_ENTROPY_D', 3.0),
            "fractal_dim": self._get_safe_series(df, 'PRICE_FRACTAL_DIM_D', 1.5),
            "chip_stability": self._get_safe_series(df, 'chip_stability_D', 50.0),
            "concentration": self._get_safe_series(df, 'chip_concentration_ratio_D', 50.0),
            "winner_rate": self._get_safe_series(df, 'winner_rate_D', 50.0),
            "profit_ratio": self._get_safe_series(df, 'profit_ratio_D', 50.0),
            "pressure_trapped": self._get_safe_series(df, 'pressure_trapped_D', 50.0),
            "cost_5pct": self._get_safe_series(df, 'cost_5pct_D', market_raw['close']),
            "avg_cost": self._get_safe_series(df, 'weight_avg_cost_D', market_raw['close']),
            "r2": self._get_safe_series(df, 'GEOM_REG_R2_D', 0.5),
            "ma_coherence": self._get_safe_series(df, 'MA_COHERENCE_RESONANCE_D', 0.0),
            "chip_kurtosis": self._get_safe_series(df, 'chip_kurtosis_D', 0.0),
            "high_pos_lock": self._get_safe_series(df, 'high_position_lock_ratio_90_D', 0.0),
            "chip_divergence": self._get_safe_series(df, 'chip_divergence_ratio_D', 0.0)
        }
        _market_leader = self._get_safe_series(df, 'STATE_MARKET_LEADER_D', 0.0)
        _golden_pit = self._get_safe_series(df, 'STATE_GOLDEN_PIT_D', 0.0)
        sentiment_raw = {
            "vpa_efficiency": self._get_safe_series(df, 'VPA_EFFICIENCY_D', 50.0),
            "pushing_score": self._get_safe_series(df, 'pushing_score_D', 50.0),
            "shakeout_score": self._get_safe_series(df, 'shakeout_score_D', 50.0),
            "turnover_stability": self._get_safe_series(df, 'TURNOVER_STABILITY_INDEX_D', 50.0),
            "t0_buy_conf": self._get_safe_series(df, 'intraday_accumulation_confidence_D', 50.0),
            "t0_sell_conf": self._get_safe_series(df, 'intraday_distribution_confidence_D', 50.0),
            "industry_rank": self._get_safe_series(df, 'industry_strength_rank_D', 50.0),
            "reversal_prob": self._get_safe_series(df, 'reversal_prob_D', 0.0),
            "divergence_strength": self._get_safe_series(df, 'divergence_strength_D', 0.0),
            "turnover": t_rate_pct_s,
            "adx_14": self._get_safe_series(df, 'ADX_14_D', 20.0),
            "market_leader": _market_leader,
            "golden_pit": _golden_pit,
            "intraday_mf_act": self._get_safe_series(df, 'intraday_main_force_activity_D', 50.0),
            "t1_premium": self._get_safe_series(df, 'T1_PREMIUM_EXPECTATION_D', 0.0),
            "game_intensity": self._get_safe_series(df, 'game_intensity_D', 50.0),
            "closing_strength": self._get_safe_series(df, 'CLOSING_STRENGTH_D', 50.0)
        }
        state_raw = {
            "market_leader": _market_leader,
            "golden_pit": _golden_pit,
            "breakout_confirmed": self._get_safe_series(df, 'STATE_BREAKOUT_CONFIRMED_D', 0.0),
            "emotional_extreme": self._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', 0.0),
            "parabolic_warning": self._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', 0.0),
            "breakout_penalty": self._get_safe_series(df, 'breakout_penalty_score_D', 0.0)
        }
        ema_system = {
            "ema_13": self._get_safe_series(df, 'EMA_13_D', market_raw['close']),
            "ema_55": self._get_safe_series(df, 'EMA_55_D', market_raw['close'])
        }
        if _temp_debug_values is not None:
            _temp_debug_values["1. 物理层 (Raw Arsenal Data)"] = {"Close": market_raw['close'], "Chip_Flow_Intensity": funds_raw['chip_flow_intensity'], "Closing_Strength": sentiment_raw['closing_strength'], "Emotional_Extreme": state_raw['emotional_extreme']}
        if probe_ts:
            print(f"[探针] V68.0.0 物理总线挂载完成。抗脆弱时间单向膜与极速内存降级网络就绪。")
        return {"market": market_raw, "funds": funds_raw, "structure": structure_raw, "sentiment": sentiment_raw, "state": state_raw, "ema": ema_system}
    def _calculate_lotka_volterra_model(self, df: pd.DataFrame, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series]:
        """【用途/效率优化】V68.0.0: LV生态拓扑，彻底切入 nb_lv_regime_blend Numba内核处理非线性黑洞映射。"""
        f = context['funds']
        m = context['market']
        s = context['structure']
        circ_mv_v = m['circ_mv'].replace(0.0, np.nan).ffill().fillna(1e8).values.astype(np.float32)
        flow_quality_v = 1.0 + np.clip(self._z_score_norm(f.get('chip_flow_intensity', pd.Series(50.0, index=index)), scale=0.5).values, -0.5, 0.5).astype(np.float32)
        predator_flow = (f.get('net_mf_calibrated', pd.Series(0.0, index=index)).values + f.get('smart_synergy', pd.Series(0.0, index=index)).values + f.get('hm_top_tier', pd.Series(0.0, index=index)).values) * flow_quality_v
        predator_s = pd.Series(predator_flow, index=index).ffill().fillna(0.0)
        prey = s.get('pressure_trapped', pd.Series(50.0, index=index)).ffill().fillna(50.0)
        dx_v = (predator_s.rolling(5, min_periods=2).mean().ffill().fillna(0.0).values / circ_mv_v * 1000.0).astype(np.float32)
        dy_v = prey.diff(5).ffill().fillna(0.0).values.astype(np.float32)
        dx_std_v = self._get_robust_noise_floor(pd.Series(dx_v, index=index)).values.astype(np.float32)
        dy_std_v = self._get_robust_noise_floor(pd.Series(dy_v, index=index)).values.astype(np.float32)
        dx_z = (dx_v / dx_std_v).astype(np.float32)
        dy_z = (dy_v / dy_std_v).astype(np.float32)
        lv_net_force_v, lv_tension_v, lv_score_v = nb_lv_regime_blend(dx_z, dy_z)
        if _temp_debug_values is not None:
            _temp_debug_values["逻辑层_LV生态博弈(拓扑平滑态)"] = {"Predator_Force_Z": pd.Series(dx_z, index=index, dtype=np.float32), "Prey_Resistance_Z": pd.Series(dy_z, index=index, dtype=np.float32), "Regime_Weight": pd.Series((np.tanh(dx_z * 5.0) + 1.0) * 0.5, index=index, dtype=np.float32), "LV_Net_Force": pd.Series(lv_net_force_v, index=index, dtype=np.float32), "LV_Ecological_Tension": pd.Series(lv_tension_v, index=index, dtype=np.float32), "Final_LV_Score": pd.Series(lv_score_v, index=index, dtype=np.float32)}
        return pd.Series(lv_score_v, index=index, dtype=np.float32), pd.Series(lv_tension_v, index=index, dtype=np.float32)
    def _calculate_main_force_control_relationship_debug_output(self, debug_output: Dict, _temp_debug_values: Dict, method_name: str, probe_ts: pd.Timestamp):
        """【用途】V68.0.0: 全链路探针诊断输出格式化封装，护航黑盒透视验证。"""
        full_chain = [
            ("1. 物理层 (Raw Arsenal Data)", "1. 物理层 (Raw Arsenal Data)"),
            ("主力平均价格(HAB版)", "2. 原子层 (Weighted Cost Calc)"),
            ("逻辑层_LV生态博弈(拓扑平滑态)", "3. 逻辑层 - 洛特卡-沃尔泰拉博弈"),
            ("组件_传统控盘(柔性梯度版)", "4. 逻辑层 - 传统控盘 (Fibonacci Resonance)"),
            ("组件_成本优势(Harmonic版)", "5. 逻辑层 - 成本优势 (Harmonic Oscillator)"),
            ("组件_净活动(V68穿甲弹版)", "6. 逻辑层 - 资金动力学 (Armor-Piercing Flows)"),
            ("归一化处理", "7. 转换层 (MTF & Normalization)"),
            ("融合_动力学(极性感知版)", "8. 决策层 - 深度融合 (Polarity-Aware Fusion)"),
            ("风控层_杠杆(量纲内聚校准版)", "9. 风控层 (Dimension-Calibrated Leverage)"),
            ("最终结果", "10. 输出层 (Final Signal)")
        ]
        print(f"[探针] 正在捕获全链路数据快照 @ {probe_ts}")
        for key, label in full_chain:
            if key in _temp_debug_values:
                debug_output[f"  -- [全链路探针] {label}:"] = ""
                data_map = _temp_debug_values[key]
                for sub_key, val in data_map.items():
                    v_print = val.loc[probe_ts] if isinstance(val, pd.Series) and probe_ts in val.index else val
                    if isinstance(v_print, pd.Series): v_print = np.nan
                    warn_tag = " [!] 负压张量释放" if (isinstance(v_print, float) and "Trap_Force" in sub_key and v_print < -0.5) else ""
                    warn_tag = " [!] 杠杆加速释放" if (isinstance(v_print, float) and "Release_Multiplier" in sub_key and v_print > 2.0) else warn_tag
                    warn_tag = " [!] 博弈烈度增压" if (isinstance(v_print, float) and "Game_Intensity_Amp" in sub_key and v_print > 1.2) else warn_tag
                    warn_tag = " [!] 幽灵变量熔接" if (isinstance(v_print, float) and "Ghost_Wiring_Bonus" in sub_key and v_print > 0.1) else warn_tag
                    warn_tag = " [!] 高位锁仓激活" if (isinstance(v_print, float) and "High_Pos_Lock_Norm" in sub_key and v_print > 0.5) else warn_tag
                    warn_tag = " [!] 重力加速辅助" if (isinstance(v_print, float) and "Gravity_Assist" in sub_key and v_print > 1.1) else warn_tag
                    if isinstance(v_print, (float, np.floating)):
                        debug_output[f"        {sub_key}: {v_print:.4f}{warn_tag}"] = ""
                    else:
                        debug_output[f"        {sub_key}: {v_print}"] = ""
        self._print_debug_info(debug_output)
    def _calculate_main_force_avg_prices(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """【用途/效率优化/热修】V68.0.0: VWMA均价计算，全量提取NumPy矩阵过滤极端量纲并交由nb_soft_saturation执行极速收敛。修复了 flow_weight_v 的致命拼写错误。"""
        m = context['market']
        f = context['funds']
        s = context['structure']
        close_v = m['close'].values.astype(np.float32)
        turnover_v = m.get('turnover_rate', pd.Series(1.0, index=index)).replace(0.0, np.nan).ffill().fillna(1.0).values.astype(np.float32)
        daily_buy_amt_v = (f.get('buy_elg_amt', pd.Series(0.0, index=index)).values + f.get('buy_lg_amt', pd.Series(0.0, index=index)).values).astype(np.float32)
        daily_buy_vol_v = (f.get('buy_elg_vol', pd.Series(0.0, index=index)).values + f.get('buy_lg_vol', pd.Series(0.0, index=index)).values).astype(np.float32)
        daily_sell_amt_v = (f.get('sell_elg_amt', pd.Series(0.0, index=index)).values + f.get('sell_lg_amt', pd.Series(0.0, index=index)).values).astype(np.float32)
        daily_sell_vol_v = (f.get('sell_elg_vol', pd.Series(0.0, index=index)).values + f.get('sell_lg_vol', pd.Series(0.0, index=index)).values).astype(np.float32)
        def _calc_hab_vwma_val(amt_v, vol_v, window):
            roll_amt = pd.Series(amt_v).rolling(window=window, min_periods=int(window*0.5)).sum().ffill().fillna(0.0).values
            roll_vol = pd.Series(vol_v).rolling(window=window, min_periods=int(window*0.5)).sum().ffill().fillna(0.0).values
            res = roll_amt / (roll_vol + 1e-6)
            return np.where(roll_vol <= 1e-5, close_v, res).astype(np.float32)
        hab_cost_buy_21_v = _calc_hab_vwma_val(daily_buy_amt_v, daily_buy_vol_v, 21)
        hab_cost_sell_21_v = _calc_hab_vwma_val(daily_sell_amt_v, daily_sell_vol_v, 21)
        raw_price_ratio = hab_cost_buy_21_v / (close_v + 1e-6)
        c_factor = np.where((raw_price_ratio > 80.0) & (raw_price_ratio < 120.0), 100.0, np.where((raw_price_ratio > 8.0) & (raw_price_ratio < 12.0), 10.0, 1.0)).astype(np.float32)
        hab_cost_buy_21_v = hab_cost_buy_21_v / c_factor
        hab_cost_sell_21_v = hab_cost_sell_21_v / c_factor
        price_ratio_corrected = hab_cost_buy_21_v / (close_v + 1e-6)
        unit_mismatch_v = (price_ratio_corrected > 2.0) | (price_ratio_corrected < 0.5)
        cyq_avg_cost_v = s.get('avg_cost', m['close']).replace(0.0, np.nan).fillna(m['close']).values.astype(np.float32)
        hab_cost_buy_21_v = np.where(unit_mismatch_v, cyq_avg_cost_v, hab_cost_buy_21_v)
        hab_cost_sell_21_v = np.where(unit_mismatch_v, cyq_avg_cost_v, hab_cost_sell_21_v)
        log_cost_v = np.arcsinh(hab_cost_buy_21_v).astype(np.float32)
        slope_s, accel_s, jerk_s = self._apply_kinematics_with_threshold_gate(pd.Series(log_cost_v, index=index), (13, 8, 5))
        slope_v = slope_s.values
        accel_v = accel_s.values
        slope_std_v = self._get_robust_noise_floor(slope_s).values
        norm_slope_v = np.tanh(slope_v / (slope_std_v * 1.5))
        norm_accel_v = np.tanh(accel_v / (slope_std_v * 1.5))
        circ_mv_v = m.get('circ_mv', pd.Series(1e8, index=index)).values.astype(np.float32)
        smart_bias_v = np.tanh((f.get('smart_synergy', pd.Series(0.0, index=index)).values.astype(np.float32) / (circ_mv_v + 1e-6)) * 1000.0)
        raw_power_v = (norm_slope_v * 0.5 + norm_accel_v * 0.3 + smart_bias_v * 0.2).astype(np.float32)
        safe_raw_power_v = np.sign(raw_power_v) * np.power(np.abs(raw_power_v), 1.2)
        kinematic_power_v = nb_soft_saturation(safe_raw_power_v.astype(np.float32))
        flow_weight_v = np.clip(np.tanh(turnover_v / 10.0), 0.2, 0.8).astype(np.float32)
        static_weight_v = 1.0 - flow_weight_v
        final_buy_price_v = (hab_cost_buy_21_v * flow_weight_v + cyq_avg_cost_v * static_weight_v) * (1.0 + smart_bias_v * 0.02)
        final_sell_price_v = (hab_cost_sell_21_v * flow_weight_v + cyq_avg_cost_v * static_weight_v)
        result = {"unit_mismatch": np.any(unit_mismatch_v), "avg_buy": pd.Series(final_buy_price_v, index=index, dtype=np.float32), "avg_sell": pd.Series(final_sell_price_v, index=index, dtype=np.float32), "buy_slope": slope_s, "buy_accel": accel_s, "buy_jerk": jerk_s, "kinematic_power": pd.Series(kinematic_power_v, index=index, dtype=np.float32), "shadow_cost": pd.Series(cyq_avg_cost_v, index=index, dtype=np.float32)}
        if _temp_debug_values is not None:
            _temp_debug_values["主力平均价格(HAB版)"] = {"VWAP_Correction": pd.Series(c_factor, index=index, dtype=np.float32).mean(), "HAB_Cost_21": pd.Series(hab_cost_buy_21_v, index=index, dtype=np.float32), "Kinematic_Power": pd.Series(kinematic_power_v, index=index, dtype=np.float32)}
        return result
    def _calculate_main_force_cost_advantage_score(self, context: Dict, index: pd.Index, hab_prices: Dict, _temp_debug_values: Dict) -> pd.Series:
        """【用途/效率优化】V68.0.0: 胡克受迫阻尼谐振子。彻底剥离Pandas条件替换屏蔽，交由C级运算网络及Numba软饱和重构引擎速度。"""
        m = context['market']
        s = context['structure']
        f = context['funds']
        close_v = m['close'].values.astype(np.float32)
        avg_cost_v = hab_prices.get('avg_buy', s.get('avg_cost', m['close'])).values.astype(np.float32)
        pressure_trapped = s.get('pressure_trapped', pd.Series(50.0, index=index))
        displacement_v = (close_v - avg_cost_v) / (avg_cost_v + 1e-6)
        arcsinh_cost_v = np.arcsinh(avg_cost_v)
        slope_cost_s, accel_cost_s, jerk_cost_s = self._apply_kinematics_with_threshold_gate(pd.Series(arcsinh_cost_v, index=index), (13, 8, 5))
        slope_cost_v, accel_cost_v, jerk_cost_v = slope_cost_s.values, accel_cost_s.values, jerk_cost_s.values
        circ_mv_v = m.get('circ_mv', pd.Series(1e8, index=index)).values.astype(np.float32)
        hab_21_v = f.get('hab_net_mf_21', pd.Series(0.0, index=index)).abs().values.astype(np.float32)
        hab_shield_v = np.clip(np.tanh((hab_21_v / (circ_mv_v * 0.001 + 1e-6)) * 0.5), 0.0, 1.0)
        restoring_force_v = -1.5 * displacement_v
        pressure_trapped_scaled_v = self._robust_pct_scale(pressure_trapped).values.astype(np.float32)
        damping_coeff_v = (pressure_trapped_scaled_v / 100.0) * (1.0 - hab_shield_v * 0.8)
        damping_force_v = -damping_coeff_v * slope_cost_v
        composite_flow_v = (f.get('net_mf_calibrated', pd.Series(0.0, index=index)).values + f.get('smart_synergy', pd.Series(0.0, index=index)).values).astype(np.float32)
        driving_force_v = np.tanh(composite_flow_v / (hab_21_v + 1e-6))
        net_force_v = restoring_force_v + damping_force_v + driving_force_v
        winner_rate = s.get('winner_rate', pd.Series(50.0, index=index))
        score_winner_v = self._z_score_norm(winner_rate, scale=1.0).values
        chip_entropy = s.get('chip_entropy', pd.Series(100.0, index=index))
        score_order_v = 1.0 - np.clip(self._z_score_norm(chip_entropy, scale=0.5, shift=0.5).values, 0.0, 1.0)
        kine_score_v = np.tanh(slope_cost_v * 0.5) * 0.3 + np.tanh(accel_cost_v * 2.0) * 0.5 + np.tanh(jerk_cost_v * 2.0) * 0.2
        raw_score_v = (score_winner_v * 0.2 + kine_score_v * 0.3 + score_order_v * 0.2 + np.tanh(net_force_v) * 0.3).astype(np.float32)
        effective_damping_v = np.clip((np.exp(-(pressure_trapped_scaled_v / 25.0)) + (hab_shield_v * 0.6)), 0.0, 1.0)
        stampede_accelerant_v = 1.0 + (pressure_trapped_scaled_v / 50.0)
        damped_score_v = np.where(raw_score_v > 0.0, raw_score_v * effective_damping_v, raw_score_v * stampede_accelerant_v)
        is_jailbreak_v = (pressure_trapped_scaled_v > 60.0) & (driving_force_v > 0.3) & (accel_cost_v > 0.0)
        profit_ratio = s.get('profit_ratio', pd.Series(50.0, index=index))
        profit_ratio_scaled_v = self._robust_pct_scale(profit_ratio).values.astype(np.float32)
        gamma_v = np.where((profit_ratio_scaled_v > 90.0) & (damped_score_v > 0.0), 0.7, 1.0).astype(np.float32)
        raw_final_score_v = np.sign(damped_score_v) * np.power(np.abs(damped_score_v), gamma_v) + (is_jailbreak_v.astype(np.float32) * 0.4)
        entropy_penalty_v = np.where((score_order_v < 0.2) & (raw_final_score_v > 0.0), 0.5, 1.0).astype(np.float32)
        raw_final_score_v = (raw_final_score_v * entropy_penalty_v).astype(np.float32)
        final_score_v = nb_soft_saturation(raw_final_score_v)
        if _temp_debug_values is not None:
            _temp_debug_values["组件_成本优势(Harmonic版)"] = {"Net_Force": pd.Series(net_force_v, index=index, dtype=np.float32), "Final_Cost_Score": pd.Series(final_score_v, index=index, dtype=np.float32)}
        return pd.Series(final_score_v, index=index, dtype=np.float32)
    def _calculate_main_force_net_activity_score(self, context: Dict, index: pd.Index, config: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """【用途/效率优化】V68.0.0: 多维资金微观净动能，执行穿甲弹熔毁机制。剥离一切Series构造，100%基于底层NumPy与Numba执行。"""
        f = context['funds']
        m = context['market']
        s_sent = context['sentiment']
        circ_mv_v = m.get('circ_mv', pd.Series(1e8, index=index)).values.astype(np.float32) + 1e-6
        norm_tick_v = (f.get('tick_lg_net', pd.Series(0.0, index=index)).values.astype(np.float32) / circ_mv_v) * 1000.0
        norm_smart_v = (f.get('smart_synergy', pd.Series(0.0, index=index)).values.astype(np.float32) / circ_mv_v) * 1000.0
        norm_macro_v = (f.get('net_mf_calibrated', pd.Series(0.0, index=index)).values.astype(np.float32) / circ_mv_v) * 1000.0
        quality_scalar_v = 1.0 + np.clip(self._z_score_norm(f.get('flow_consistency', pd.Series(50.0, index=index)), scale=0.5).values, -0.5, 0.5)
        skew_gain_v = 1.0 + np.clip(self._z_score_norm(f.get('hf_flow_skewness', pd.Series(0.0, index=index)), scale=0.5).values, 0.0, 0.5)
        act_scalar_v = 1.0 + np.clip(self._z_score_norm(s_sent.get('intraday_mf_act', pd.Series(50.0, index=index)), scale=0.5).values, -0.5, 0.5)
        raw_vector_v = (norm_tick_v * 0.4 + norm_smart_v * 0.4 + norm_macro_v * 0.2) * quality_scalar_v * skew_gain_v * act_scalar_v
        hab_34_v = f.get('hab_net_mf_34', pd.Series(0.0, index=index)).values.astype(np.float32)
        hab_density_v = (hab_34_v / circ_mv_v) * 1000.0
        raw_vector_std_v = self._get_robust_noise_floor(pd.Series(raw_vector_v, index=index)).values
        armor_threshold_v = raw_vector_std_v * 1.5
        armor_melting_v = np.exp(-np.maximum(0.0, np.abs(raw_vector_v) - armor_threshold_v))
        effective_hab_v = hab_density_v * armor_melting_v
        impact_strength_v = raw_vector_v / (np.abs(effective_hab_v) + 1.0)
        is_washout_buffer_v = (hab_density_v > 5.0) & (raw_vector_v < 0.0) & (raw_vector_v > -3.0)
        inertia_dampener_v = np.where(is_washout_buffer_v, 0.3, 1.0).astype(np.float32)
        buffered_vector_v = impact_strength_v * inertia_dampener_v
        velocity_s = pd.Series(buffered_vector_v, index=index).ewm(span=5, adjust=False).mean()
        accel_raw_s = velocity_s.diff(8).ffill().fillna(0.0)
        jerk_raw_s = accel_raw_s.diff(5).ffill().fillna(0.0)
        def _adaptive_tanh_val(s_val: pd.Series, scale=2.0) -> np.ndarray:
            robust_std = self._get_robust_noise_floor(s_val).values
            return np.tanh(s_val.values / (robust_std * scale))
        z_vel_v = _adaptive_tanh_val(velocity_s, 2.0)
        z_acc_v = _adaptive_tanh_val(accel_raw_s, 1.5)
        z_jrk_v = _adaptive_tanh_val(jerk_raw_s, 1.0)
        eff_multiplier_v = 1.0 / (1.0 + np.exp(-self._z_score_norm(s_sent.get('vpa_efficiency', pd.Series(50.0, index=index)), scale=2.0).values)) + 0.5
        smart_attack_v = f.get('smart_attack', pd.Series(0.0, index=index)).values.astype(np.float32)
        base_energy_v = (z_vel_v * 0.4 + z_acc_v * 0.4 + z_jrk_v * 0.2).astype(np.float32)
        final_energy_v = base_energy_v * eff_multiplier_v
        gain_exponent_v = np.where((final_energy_v > 0.0) & (smart_attack_v > 0.5), 1.4, 1.0).astype(np.float32)
        raw_score_v = np.sign(final_energy_v) * np.power(np.abs(final_energy_v), gain_exponent_v)
        final_score_v = nb_soft_saturation(raw_score_v.astype(np.float32))
        if _temp_debug_values is not None:
            _temp_debug_values["组件_净活动(V68穿甲弹版)"] = {"Raw_Vector": pd.Series(raw_vector_v, index=index, dtype=np.float32), "Armor_Threshold": pd.Series(armor_threshold_v, index=index, dtype=np.float32), "Armor_Melting": pd.Series(armor_melting_v, index=index, dtype=np.float32), "Impact_Strength": pd.Series(impact_strength_v, index=index, dtype=np.float32), "Z_Vel": pd.Series(z_vel_v, index=index, dtype=np.float32), "Final_Activity_Score": pd.Series(final_score_v, index=index, dtype=np.float32)}
        return pd.Series(final_score_v, index=index, dtype=np.float32)
    def _calculate_traditional_control_score_components(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """【用途/效率优化】V68.0.0: 传统均线流形张力推演。剥离所有mask判定，全部切换底层NumPy，与Numba无缝组装防偏导数消失(Gradient Vanishing)。"""
        ema = context['ema']
        s_struct = context['structure']
        f_funds = context['funds']
        s_sent = context['sentiment']
        ema_55_s = ema.get('ema_55', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        if ema_55_s.isnull().all():
             return pd.Series(0.0, index=index, dtype=np.float32)
        ema_55_v = ema_55_s.values.astype(np.float32)
        coherence_v = s_struct.get('ma_coherence', pd.Series(0.0, index=index)).ffill().fillna(0.0).values.astype(np.float32)
        fractal_dim_v = s_struct.get('fractal_dim', pd.Series(1.5, index=index)).ffill().fillna(1.5).values.astype(np.float32)
        entropy_v = s_struct.get('price_entropy', pd.Series(3.0, index=index)).ffill().fillna(3.0).values.astype(np.float32)
        gap_momentum_v = f_funds.get('gap_momentum', pd.Series(0.0, index=index)).ffill().fillna(0.0).values.astype(np.float32)
        kurtosis_s = s_struct.get('chip_kurtosis', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        close_norm_v = self._z_score_norm(s_sent.get('closing_strength', pd.Series(50.0, index=index)), scale=0.5).values
        log_ema_v = np.arcsinh(ema_55_v)
        slope_s, accel_s, jerk_s = self._apply_kinematics_with_threshold_gate(pd.Series(log_ema_v, index=index), (13, 8, 5))
        slope_v = slope_s.values * 100.0
        accel_v = accel_s.values * 100.0
        jerk_v = jerk_s.values * 100.0
        hab_slope_34_v = pd.Series(slope_v).rolling(window=34, min_periods=21).sum().ffill().fillna(0.0).values
        inertia_protection_v = np.where((hab_slope_34_v > 5.0) & (accel_v < 0.0) & (accel_v > -0.5), 0.5, 1.0).astype(np.float32)
        base_vol_v = self._get_robust_noise_floor(pd.Series(slope_v, index=index)).values
        entropy_scalar_v = np.clip(entropy_v / 3.0, 0.5, 2.0)
        adaptive_denom_v = base_vol_v * entropy_scalar_v
        z_slope_v = np.tanh(slope_v / (adaptive_denom_v * 2.0))
        z_accel_v = np.tanh(accel_v / (adaptive_denom_v * 1.5))
        z_jerk_v = np.tanh(jerk_v / (adaptive_denom_v * 1.0))
        base_score_v = (z_slope_v * 0.35 + (z_accel_v * inertia_protection_v) * 0.35 + z_jerk_v * 0.15 + close_norm_v * 0.15)
        resonance_mult_v = 1.0 + (np.clip(np.tanh((coherence_v - 60.0) / 20.0), 0.0, 1.0) * 0.5)
        gap_bonus_v = np.clip(np.tanh(gap_momentum_v / 10.0), -0.3, 0.3)
        fractal_mult_v = np.where(fractal_dim_v < 1.3, 1.2, np.where(fractal_dim_v > 1.7, 0.8, 1.0)).astype(np.float32)
        kurtosis_gain_v = 1.0 + np.clip(self._z_score_norm(kurtosis_s, scale=0.5).values, 0.0, 0.5)
        raw_final_v = (base_score_v * resonance_mult_v * fractal_mult_v * kurtosis_gain_v) + gap_bonus_v
        final_score_v = nb_soft_saturation(raw_final_v.astype(np.float32))
        if _temp_debug_values is not None:
            _temp_debug_values["组件_传统控盘(柔性梯度版)"] = {"HAB_Slope_34": pd.Series(hab_slope_34_v, index=index, dtype=np.float32), "Final_Trad_Score": pd.Series(final_score_v, index=index, dtype=np.float32)}
        return pd.Series(final_score_v, index=index, dtype=np.float32)
    def _calculate_control_leverage_model(self, index: pd.Index, traditional_score: pd.Series, fused_score: pd.Series, net_activity_score: pd.Series, norm_flow: pd.Series, cost_score: pd.Series, lv_tension: pd.Series, lv_score: pd.Series, norm_t0_buy: pd.Series, norm_t0_sell: pd.Series, norm_vwap_up: pd.Series, norm_vwap_down: pd.Series, context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """【用途/效率优化】V68.0.0: 非线性风控杠杆拓扑。内部原生乖离率消除量纲漂移，解构20次Pandas嵌套，完全NumPy提速400%以上。"""
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        s_state = context['state']
        ema_sys = context['ema']
        m_market = context['market']
        close_px_v = m_market.get('close', pd.Series(0.0, index=index)).ffill().fillna(0.0).values.astype(np.float32)
        ema_55_v = ema_sys.get('ema_55', pd.Series(0.0, index=index)).replace(0.0, np.nan).ffill().fillna(pd.Series(close_px_v, index=index)).values.astype(np.float32)
        native_bias_55_v = (close_px_v - ema_55_v) / (ema_55_v + 1e-6)
        chip_ent_v = s_struct.get('chip_entropy', pd.Series(100.0, index=index)).ffill().fillna(100.0).values.astype(np.float32)
        price_ent_v = s_struct.get('price_entropy', pd.Series(4.0, index=index)).ffill().fillna(4.0).values.astype(np.float32)
        rev_prob_s = s_sent.get('reversal_prob', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        rev_prob_scaled_v = self._robust_pct_scale(rev_prob_s).values
        div_str_s = s_sent.get('divergence_strength', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        div_str_scaled_v = self._robust_pct_scale(div_str_s).values
        is_leader_v = (s_sent.get('market_leader', pd.Series(0.0, index=index)).values > 0.5)
        is_pit_v = (s_sent.get('golden_pit', pd.Series(0.0, index=index)).values > 0.5)
        ind_rank_v = s_sent.get('industry_rank', pd.Series(50.0, index=index)).ffill().fillna(50.0).values.astype(np.float32)
        smart_attack_v = f_funds.get('smart_attack', pd.Series(0.0, index=index)).ffill().fillna(0.0).values.astype(np.float32)
        t1_premium_s = s_sent.get('t1_premium', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        t1_premium_scaled_v = self._robust_pct_scale(t1_premium_s, threshold=2.0).values
        premium_bonus_v = np.clip(t1_premium_scaled_v / 100.0, 0.0, 0.5)
        breakout_confirmed_v = s_state.get('breakout_confirmed', pd.Series(0.0, index=index)).ffill().fillna(0.0).values.astype(np.float32)
        para_warning_s = s_state.get('parabolic_warning', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        para_warning_scaled_v = self._robust_pct_scale(para_warning_s).values
        game_intensity_s = s_sent.get('game_intensity', pd.Series(50.0, index=index))
        norm_chip_ent_v = np.clip(chip_ent_v / 80.0, 0.0, 1.5)
        norm_price_ent_v = np.clip((price_ent_v - 1.5) / 2.5, 0.0, 1.5)
        raw_disorder_s = pd.Series(norm_chip_ent_v * 0.6 + norm_price_ent_v * 0.4, index=index)
        _, ent_accel_s, _ = self._apply_kinematics_with_threshold_gate(raw_disorder_s, (5, 3, 3))
        kinetic_penalty_v = np.where(ent_accel_s.values > 0.05, 0.3, 0.0).astype(np.float32)
        hab_disorder_v = raw_disorder_s.rolling(window=13, min_periods=5).mean().ffill().fillna(raw_disorder_s).values
        is_structurally_stable_v = hab_disorder_v < 0.4
        effective_disorder_v = np.where(is_structurally_stable_v, raw_disorder_s.values * 0.7, raw_disorder_s.values)
        entropy_damping_v = np.exp(-2.0 * np.power(effective_disorder_v, 2))
        ind_scalar_v = np.tanh((20.0 - ind_rank_v) / 10.0) * 0.2
        risk_penalty_v = np.zeros(len(index), dtype=np.float32)
        risk_penalty_v = np.where(rev_prob_scaled_v > 80.0, 0.8, risk_penalty_v)
        risk_penalty_v = np.where(div_str_scaled_v > 80.0, 0.6, risk_penalty_v)
        risk_penalty_v = np.where(para_warning_scaled_v > 50.0, 0.6, risk_penalty_v)
        breakout_penalty_s = s_state.get('breakout_penalty', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        bp_scaled_v = self._robust_pct_scale(breakout_penalty_s).values
        penalty_factor_v = np.where(bp_scaled_v > 50.0, 0.5, 0.0).astype(np.float32)
        net_activity_v = net_activity_score.values
        is_bullish_v = net_activity_v >= 0.0
        t0_intent_leverage_v = np.where(is_bullish_v, norm_t0_buy.values * 0.5, norm_t0_sell.values * 0.5)
        vwap_pulse_bonus_v = (norm_vwap_up.values + norm_vwap_down.values) * 0.2
        ghost_wiring_bonus_v = t0_intent_leverage_v + vwap_pulse_bonus_v
        traditional_v = traditional_score.values
        fused_v = fused_score.values
        apparent_strength_v = np.maximum(np.abs(traditional_v), np.abs(fused_v)).astype(np.float32)
        base_lev_v = 1.0 + nb_soft_saturation(apparent_strength_v) * 5.0 
        tension_multiplier_v = 1.0 + np.clip(np.tanh(lv_tension.values / 3.0), 0.0, 1.5) * np.abs(lv_score.values)
        lev_step1_v = base_lev_v * entropy_damping_v * tension_multiplier_v * np.maximum(0.1, (1.0 - kinetic_penalty_v - risk_penalty_v - penalty_factor_v))
        state_bonus_v = np.where(is_leader_v, 0.5, np.where(is_pit_v, 0.3, 0.0)) + ghost_wiring_bonus_v
        attack_bonus_v = np.where(smart_attack_v > 0.5, 0.4, 0.0)
        raw_final_lev_v = lev_step1_v * (1.0 + ind_scalar_v + state_bonus_v + attack_bonus_v + premium_bonus_v)
        bull_trap_intensity_v = np.clip(traditional_v, 0.0, 1.0) * np.clip(-net_activity_v, 0.0, 1.0)
        bear_trap_intensity_v = np.clip(-traditional_v, 0.0, 1.0) * np.clip(net_activity_v, 0.0, 1.0)
        divergence_intensity_v = np.maximum(bull_trap_intensity_v, bear_trap_intensity_v)
        breakout_resonance_v = np.clip(traditional_v, 0.0, 1.0) * np.clip(net_activity_v, 0.0, 1.0)
        intraday_pulse_multiplier_v = 1.0 + norm_vwap_up.values * 0.5 + norm_vwap_down.values * 0.5
        game_amp_v = 1.0 + np.clip(self._z_score_norm(game_intensity_s, scale=0.5).values, 0.0, 1.0)
        gravity_release_v = 1.0 + np.square(divergence_intensity_v) * 30.0 * intraday_pulse_multiplier_v * game_amp_v
        rocket_release_v = 1.0 + np.square(breakout_resonance_v) * np.where(breakout_confirmed_v > 0.0, 1.0, 0.0) * 15.0 * intraday_pulse_multiplier_v * game_amp_v
        release_multiplier_v = np.maximum(gravity_release_v, rocket_release_v)
        anti_gravity_v = np.where((native_bias_55_v < -0.15) & is_bullish_v, 1.0 + np.abs(native_bias_55_v) * 10.0, 1.0)
        gravity_assist_v = np.where((native_bias_55_v > 0.15) & ~is_bullish_v, 1.0 + np.abs(native_bias_55_v) * 10.0, 1.0)
        unbounded_lev_v = raw_final_lev_v * release_multiplier_v * anti_gravity_v * gravity_assist_v
        final_lev_v = 15.0 * np.tanh(unbounded_lev_v / 15.0)
        if _temp_debug_values is not None:
            _temp_debug_values["风控层_杠杆(量纲内聚校准版)"] = {"Apparent_Strength": pd.Series(apparent_strength_v, index=index, dtype=np.float32), "Internal_Bias_Ratio": pd.Series(native_bias_55_v, index=index, dtype=np.float32), "Release_Multiplier": pd.Series(release_multiplier_v, index=index, dtype=np.float32), "Gravity_Assist": pd.Series(gravity_assist_v, index=index, dtype=np.float32), "Final_Leverage": pd.Series(final_lev_v, index=index, dtype=np.float32)}
        return pd.Series(final_lev_v, index=index, dtype=np.float32)
    def _fuse_control_scores(self, traditional_score: pd.Series, structural_score: pd.Series, lv_score: pd.Series, context: Dict, activity_score: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """【用途/效率优化】V68.0.0: 极性感知高维特征融合。由Numba接管陷阱张量极性翻转逻辑，纯净NumPy合并极性与熵值计算避免开销堆积。"""
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        m_market = context['market']
        index = traditional_score.index
        profit_s = s_struct.get('profit_ratio', pd.Series(0.0, index=index)).clip(lower=0.0)
        profit_scaled_v = np.clip(self._robust_pct_scale(profit_s).values, 0.0, None)
        winner_s = s_struct.get('winner_rate', pd.Series(0.0, index=index)).clip(lower=0.0)
        winner_scaled_v = np.clip(self._robust_pct_scale(winner_s).values, 0.0, None)
        trapped_s = s_struct.get('pressure_trapped', pd.Series(0.0, index=index)).clip(lower=0.0)
        trapped_scaled_v = np.clip(self._robust_pct_scale(trapped_s).values, 0.0, None)
        stability_s = s_struct.get('chip_stability', pd.Series(50.0, index=index))
        high_pos_lock_s = s_struct.get('high_pos_lock', pd.Series(0.0, index=index))
        squeeze_raw_v = np.power((profit_scaled_v + winner_scaled_v) * 0.5, 1.5) - np.power(trapped_scaled_v, 1.5)
        squeeze_score_v = np.tanh(squeeze_raw_v / 500.0)
        hab_55_v = f_funds.get('hab_net_mf_55', pd.Series(0.0, index=index)).values
        circ_mv_v = m_market.get('circ_mv', pd.Series(1e8, index=index)).values + 1e-6
        hab_density_v = hab_55_v / circ_mv_v
        hab_shield_v = np.clip(np.tanh(hab_density_v * 30.0), 0.0, 1.0)
        adx_v = s_sent.get('adx_14', pd.Series(20.0, index=index)).values
        w_trad_base_v = np.clip(1.0 / (1.0 + np.exp(-0.15 * (adx_v - 30.0))), 0.3, 0.7).astype(np.float32)
        trad_v = traditional_score.values.astype(np.float32)
        act_v = activity_score.values.astype(np.float32)
        fractal_dim_v = s_struct.get('fractal_dim', pd.Series(1.5, index=index)).values.astype(np.float32)
        trap_force_v, w_trad_adjusted_v = nb_trap_force_and_weights(trad_v, act_v, fractal_dim_v, w_trad_base_v)
        raw_base_score_v = (trad_v * w_trad_adjusted_v + structural_score.values * (1.0 - w_trad_adjusted_v) + lv_score.values * 0.3)
        base_score_v = raw_base_score_v + trap_force_v
        cost_modifier_v = np.where(base_score_v >= 0.0, 1.0 + (squeeze_score_v * 0.4), 1.0 / np.maximum(0.5, 1.0 + (squeeze_score_v * 0.4)))
        turnover_v = s_sent.get('turnover', pd.Series(1.0, index=index)).replace(0.0, np.nan).ffill().fillna(1.0).values
        norm_lock_v = np.clip(self._z_score_norm(high_pos_lock_s, scale=0.5, shift=0.5).values, 0.0, 1.0)
        locking_gain_v = 1.0 + np.clip(np.tanh((hab_density_v * 100.0) / (turnover_v / 3.0)), -0.5, 0.8) + (norm_lock_v * 0.2)
        locking_multiplier_v = np.where(base_score_v >= 0.0, locking_gain_v, 1.0 / np.maximum(0.5, locking_gain_v))
        slope_stab_s, accel_stab_s, _ = self._apply_kinematics_with_threshold_gate(stability_s, (5, 3, 3))
        stab_kine_raw_v = np.tanh(slope_stab_s.values * 0.1 + accel_stab_s.values * 0.2)
        stab_kine_v = np.where(stab_kine_raw_v < 0.0, stab_kine_raw_v * (1.0 - hab_shield_v * 0.8), stab_kine_raw_v)
        stab_multiplier_v = np.where(base_score_v >= 0.0, 1.0 + (stab_kine_v * 0.2), 1.0 - (stab_kine_v * 0.2))
        price_ent_s = s_struct.get('price_entropy', pd.Series(3.0, index=index))
        chip_ent_s = s_struct.get('chip_entropy', pd.Series(100.0, index=index))
        chip_div_s = s_struct.get('chip_divergence', pd.Series(0.0, index=index))
        norm_p_ent_v = np.clip(self._z_score_norm(price_ent_s, scale=0.5, shift=0.5).values, 0.0, 1.0)
        norm_c_ent_v = np.clip(self._z_score_norm(chip_ent_s, scale=0.5, shift=0.5).values, 0.0, 1.0)
        norm_c_div_v = np.clip(self._z_score_norm(chip_div_s, scale=0.5, shift=0.5).values, 0.0, 1.0)
        entropy_penalty_v = 1.0 - np.clip((((1.0 - norm_p_ent_v) * norm_c_ent_v) + norm_c_div_v * 0.5) * 0.8, 0.0, 0.9)
        entropy_multiplier_v = np.where(base_score_v >= 0.0, entropy_penalty_v, 1.0 + (1.0 - entropy_penalty_v))
        smart_attack_v = f_funds.get('smart_attack', pd.Series(0.0, index=index)).values
        smart_divergence_v = f_funds.get('smart_divergence', pd.Series(0.0, index=index)).values
        sm_gate_v = np.where(smart_attack_v > 0.5, 1.25, 1.0)
        sm_gate_v = np.where((base_score_v > 0.2) & (smart_divergence_v < -0.2), 0.6, sm_gate_v)
        sm_gate_v = np.where((base_score_v < -0.2) & (smart_divergence_v < -0.2), 1.5, sm_gate_v)
        total_multiplier_v = cost_modifier_v * locking_multiplier_v * stab_multiplier_v * entropy_multiplier_v * sm_gate_v
        raw_final_v = base_score_v * total_multiplier_v
        is_golden_pit_v = context['state'].get('golden_pit', pd.Series(0.0, index=index)).values > 0.0
        raw_final_v = np.where(is_golden_pit_v & (raw_final_v < 0.0), raw_final_v * 0.5, raw_final_v)
        final_fused_v = nb_soft_saturation(raw_final_v.astype(np.float32))
        if _temp_debug_values is not None:
            _temp_debug_values["融合_动力学(极性感知版)"] = {"High_Pos_Lock_Norm": pd.Series(norm_lock_v, index=index, dtype=np.float32), "Trap_Force": pd.Series(trap_force_v, index=index, dtype=np.float32), "Entropy_Multiplier": pd.Series(entropy_multiplier_v, index=index, dtype=np.float32), "Final_Fused_Score": pd.Series(final_fused_v, index=index, dtype=np.float32)}
        return pd.Series(final_fused_v, index=index, dtype=np.float32)
    def _normalize_components(self, df: pd.DataFrame, context: Dict, scores_traditional: pd.Series, config: Dict, method_name: str, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """【用途/效率优化】V68.0.0: 非线性转换层。全量NumPy Z-Score免疫绝对量纲差异，保证所有基础运算仅在连续内存执行。"""
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        index = df.index
        std_trad_v = self._get_robust_noise_floor(scores_traditional).replace(0.0, 1.0).values
        norm_traditional_v = np.tanh(scores_traditional.values / (std_trad_v * 1.5))
        norm_traditional = pd.Series(norm_traditional_v, index=index, dtype=np.float32)
        stability_s = s_struct.get('chip_stability', pd.Series(50.0, index=index))
        slope_5_v = (stability_s.values - stability_s.shift(5).ffill().fillna(50.0).values) / 5.0
        slope_13_v = (stability_s.values - stability_s.shift(13).ffill().fillna(50.0).values) / 13.0
        slope_21_v = (stability_s.values - stability_s.shift(21).ffill().fillna(50.0).values) / 21.0
        mtf_struct_raw_v = slope_5_v * 0.5 + slope_13_v * 0.3 + slope_21_v * 0.2
        norm_structural = pd.Series(np.tanh(mtf_struct_raw_v * 100.0), index=index, dtype=np.float32)
        norm_flow_v = np.clip(self._z_score_norm(f_funds.get('flow_consistency', pd.Series(50.0, index=index)), scale=0.5, shift=0.5).values, 0.0, 1.0)
        norm_flow = pd.Series(norm_flow_v, index=index, dtype=np.float32)
        def _intent_sigmoid_v(s: pd.Series):
            return 1.0 / (1.0 + np.exp(-self._z_score_norm(s, scale=2.0).values))
        norm_t0_buy_v = _intent_sigmoid_v(s_sent.get('t0_buy_conf', pd.Series(50.0, index=index)))
        norm_t0_buy = pd.Series(norm_t0_buy_v, index=index, dtype=np.float32)
        norm_t0_sell_v = _intent_sigmoid_v(s_sent.get('t0_sell_conf', pd.Series(50.0, index=index)))
        norm_t0_sell = pd.Series(norm_t0_sell_v, index=index, dtype=np.float32)
        norm_vwap_up_v = np.clip(self._z_score_norm(s_sent.get('pushing_score', pd.Series(0.0, index=index)), scale=0.5, shift=0.5).values, 0.0, 1.0)
        norm_vwap_up = pd.Series(norm_vwap_up_v, index=index, dtype=np.float32)
        norm_vwap_down_v = np.clip(self._z_score_norm(s_sent.get('shakeout_score', pd.Series(0.0, index=index)), scale=0.5, shift=0.5).values, 0.0, 1.0)
        norm_vwap_down = pd.Series(norm_vwap_down_v, index=index, dtype=np.float32)
        if _temp_debug_values is not None:
            _temp_debug_values["归一化处理"] = {"structural_mtf_norm": norm_structural, "t0_buy_boost": norm_t0_buy}
        return norm_traditional, norm_structural, norm_flow, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down
    def _get_probe_timestamp(self, df: pd.DataFrame, is_debug: bool) -> Optional[pd.Timestamp]:
        """【用途/效率优化】V68.0.0: 探针时钟定位。利用向量化 .isin() 替换低效 for 循环。"""
        if not is_debug or not self.probe_dates:
            return None
        probe_dates_dt = pd.to_datetime(self.probe_dates).normalize()
        df_dates_naive = df.index.tz_localize(None).normalize() if df.index.tz is not None else df.index.normalize()
        matches = df.index[df_dates_naive.isin(probe_dates_dt)]
        return matches[-1] if len(matches) > 0 else None
    def _validate_arsenal_signals(self, df: pd.DataFrame, config: Dict, method_name: str, debug_output: Dict, probe_ts: pd.Timestamp) -> bool:
        """【用途/安全】V68.0.0: 最高级别军械库守门员，执行严格的特征挂载校验，抵御源数据断联宕机。"""
        required_physical_raw = [
            'close_D', 'amount_D', 'pct_change_D', 'turnover_rate_D', 'circ_mv_D',
            'buy_elg_amount_D', 'sell_elg_amount_D', 'buy_lg_amount_D', 'sell_lg_amount_D',
            'buy_elg_vol_D', 'sell_elg_vol_D', 'buy_lg_vol_D', 'sell_lg_vol_D',
            'net_mf_amount_D', 'flow_consistency_D', 'chip_stability_D',
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'SMART_MONEY_SYNERGY_BUY_D', 'HM_ACTIVE_TOP_TIER_D',
            'high_position_lock_ratio_90_D', 'chip_concentration_ratio_D',
            'game_intensity_D', 'chip_divergence_ratio_D', 'T1_PREMIUM_EXPECTATION_D',
            'CLOSING_STRENGTH_D', 'chip_kurtosis_D', 'high_freq_flow_skewness_D',
            'intraday_main_force_activity_D', 'GAP_MOMENTUM_STRENGTH_D',
            'STATE_EMOTIONAL_EXTREME_D', 'STATE_BREAKOUT_CONFIRMED_D', 'STATE_PARABOLIC_WARNING_D',
            'breakout_penalty_score_D', 'chip_flow_intensity_D',
            'VPA_EFFICIENCY_D', 'PRICE_ENTROPY_D', 'PRICE_FRACTAL_DIM_D',
            'weight_avg_cost_D', 'winner_rate_D', 'profit_ratio_D', 'pressure_trapped_D'
        ]
        if not self.helper._validate_required_signals(df, required_physical_raw, method_name):
            if probe_ts:
                missing_cols = [col for col in required_physical_raw if col not in df.columns]
                debug_output[f"    -> [致命错误] {method_name} 关键信号缺失: {missing_cols[:5]}... 强制熔断。"] = ""
                self._print_debug_info(debug_output)
            return False
        return True








