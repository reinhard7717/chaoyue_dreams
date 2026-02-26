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
class CalculateMainForceControlRelationship:
    """
    【V46.0.0 · 主力控盘全息量子决策系统 · 无未来函数与张量反转版】
    PROCESS_META_MAIN_FORCE_CONTROL
    - 核心职责: 计算“主力控盘”的专属关系分数，防御诱多陷阱与未来函数。
    - 版本: 46.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
    def _print_debug_info(self, debug_output: Dict):
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)
    def _get_safe_series(self, df: pd.DataFrame, col_name: str, default_value: float = 0.0, method_name: str = "") -> pd.Series:
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        neutral_nan_defaults = process_params.get('neutral_nan_defaults', {})
        current_default_value = neutral_nan_defaults.get(col_name, default_value)
        if col_name not in df.columns:
            return pd.Series(current_default_value, index=df.index, dtype=np.float32)
        series = df[col_name].astype(np.float32)
        return series.ffill().fillna(current_default_value)
    def _get_robust_noise_floor(self, s: pd.Series, window_std=21) -> pd.Series:
        roll_std = s.abs().rolling(window=window_std, min_periods=5).std().ffill().fillna(0.0)
        exp_med = s.abs().expanding(min_periods=5).median().ffill().fillna(1e-5)
        return np.maximum(roll_std, exp_med) + 1e-5
    def _apply_kinematics_with_threshold_gate(self, series: pd.Series, periods: tuple) -> Tuple[pd.Series, pd.Series, pd.Series]:
        p_slope, p_accel, p_jerk = periods
        slope = series.diff(p_slope).ffill().fillna(0.0)
        noise_slope = self._get_robust_noise_floor(slope) * 0.5
        slope_gated = slope * np.tanh(np.abs(slope) / noise_slope)
        accel = slope_gated.diff(p_accel).ffill().fillna(0.0)
        noise_accel = self._get_robust_noise_floor(accel) * 0.5
        accel_gated = accel * np.tanh(np.abs(accel) / noise_accel)
        jerk = accel_gated.diff(p_jerk).ffill().fillna(0.0)
        noise_jerk = self._get_robust_noise_floor(jerk) * 0.5
        jerk_gated = jerk * np.tanh(np.abs(jerk) / noise_jerk)
        return slope_gated, accel_gated, jerk_gated
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = "calculate_main_force_control_relationship"
        is_debug = get_param_value(self.debug_params.get('enabled'), False)
        _temp_debug_values = {} 
        probe_ts = self._get_probe_timestamp(df, is_debug)
        debug_output = {}
        if probe_ts:
            print(f"[调度中心] {method_name} 启动 @ {probe_ts.strftime('%Y-%m-%d')} | 版本: V46.0.0 (张量反转防御版)")
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
        control_leverage = self._calculate_control_leverage_model(df.index, fused_control_score, scores_net_activity, norm_flow, scores_cost_advantage, lv_tension, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down, control_context, _temp_debug_values)
        kinematic_thrust = np.sign(scores_net_activity) * np.power(np.abs(scores_net_activity) + 1e-6, 1.2) * control_leverage
        is_hollow = (fused_control_score > 0.0) & (kinematic_thrust < 0.0)
        effective_structure = fused_control_score.copy()
        effective_structure = effective_structure.mask(is_hollow, fused_control_score * np.maximum(0.0, 1.0 - np.abs(kinematic_thrust) * 2.0))
        fractal_dim = control_context['structure'].get('fractal_dim', pd.Series(1.5, index=df.index))
        trend_confidence = 1.0 - ((fractal_dim - 1.2) / 0.6).clip(0.0, 1.0)
        w_struct = 0.2 + 0.6 * trend_confidence
        w_act = 1.0 - w_struct
        raw_final_score = (effective_structure * w_struct) + (kinematic_thrust * w_act)
        final_control_score = (raw_final_score / np.sqrt(1.0 + np.square(raw_final_score))).astype(np.float32)
        _temp_debug_values["最终结果"] = {"Net_Activity (Vector)": scores_net_activity, "Control_Leverage (Dynamic)": control_leverage, "Kinematic_Thrust": kinematic_thrust, "Fused_Structure": fused_control_score, "Effective_Structure": effective_structure, "Dynamic_Struct_Weight": w_struct, "Lotka_Volterra_Score": scores_lv_ecology, "Final_Score": final_control_score}
        if probe_ts:
            self._calculate_main_force_control_relationship_debug_output(debug_output, _temp_debug_values, method_name, probe_ts)
        return final_control_score
    def _get_control_parameters(self, config: Dict) -> Tuple[Dict, Dict]:
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        return actual_mtf_weights, mtf_slope_accel_weights
    def _get_raw_control_signals(self, df: pd.DataFrame, method_name: str, _temp_debug_values: Dict, probe_ts: pd.Timestamp) -> Dict[str, Dict[str, pd.Series]]:
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
            "hab_net_mf_13": None, "hab_net_mf_21": None, "hab_net_mf_34": None, "hab_net_mf_55": None
        }
        composite_flow = (funds_raw["tick_lg_net"] * 0.3 + funds_raw["smart_synergy"] * 0.3 + funds_raw["smart_net_buy"] * 0.2 + funds_raw["net_mf_calibrated"] * 0.2)
        funds_raw["hab_net_mf_13"] = composite_flow.rolling(window=13, min_periods=5).sum().shift(1).ffill().fillna(0.0)
        funds_raw["hab_net_mf_21"] = composite_flow.rolling(window=21, min_periods=10).sum().shift(1).ffill().fillna(0.0)
        funds_raw["hab_net_mf_34"] = composite_flow.rolling(window=34, min_periods=15).sum().shift(1).ffill().fillna(0.0)
        funds_raw["hab_net_mf_55"] = composite_flow.rolling(window=55, min_periods=21).sum().shift(1).ffill().fillna(0.0)
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
            "bias_55": self._get_safe_series(df, 'BIAS_55_D', 0.0),
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
            "turnover": self._get_safe_series(df, 'turnover_rate_D', 1.0),
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
            "emotional_extreme": self._get_safe_series(df, 'STATE_EMOTIONAL_EXTREME_D', 0.0)
        }
        ema_system = {
            "ema_13": self._get_safe_series(df, 'EMA_13_D', market_raw['close']),
            "ema_55": self._get_safe_series(df, 'EMA_55_D', market_raw['close'])
        }
        if _temp_debug_values is not None:
            _temp_debug_values["1. 物理层 (Raw Arsenal Data)"] = {"Close": market_raw['close'], "Smart_Synergy": funds_raw['smart_synergy'], "High_Pos_Lock": structure_raw['high_pos_lock'], "Emotional_Extreme": state_raw['emotional_extreme']}
        if probe_ts:
            print(f"[探针] V46.0.0 物理总线挂载完成。消除一切未来函数，建立安全时间单向膜。")
        return {"market": market_raw, "funds": funds_raw, "structure": structure_raw, "sentiment": sentiment_raw, "state": state_raw, "ema": ema_system}
    def _calculate_lotka_volterra_model(self, df: pd.DataFrame, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series]:
        f = context['funds']
        m = context['market']
        s = context['structure']
        circ_mv = m['circ_mv'].replace(0.0, np.nan).ffill().fillna(1e8)
        predator_flow = f['net_mf_calibrated'] + f['smart_synergy'] + f['hm_top_tier']
        predator_flow = predator_flow.ffill().fillna(0.0)
        prey = s['pressure_trapped'].ffill().fillna(50.0)
        dx = predator_flow.rolling(5, min_periods=2).mean().diff(3).ffill().fillna(0.0) / circ_mv * 1000.0
        dy = prey.diff(3).ffill().fillna(0.0)
        dx_std = self._get_robust_noise_floor(dx)
        dy_std = self._get_robust_noise_floor(dy)
        dx_z = dx / dx_std
        dy_z = dy / dy_std
        lv_net_force = dx_z - dy_z
        lv_tension = np.sqrt(dx_z**2 + dy_z**2).astype(np.float32)
        lv_score = np.tanh(lv_net_force).astype(np.float32)
        if _temp_debug_values is not None:
            _temp_debug_values["逻辑层_LV生态博弈(净力模型)"] = {"Predator_Force_Z": dx_z, "Prey_Resistance_Z": dy_z, "LV_Net_Force": lv_net_force, "LV_Ecological_Tension": lv_tension, "Final_LV_Score": lv_score}
        return lv_score, lv_tension
    def _calculate_main_force_control_relationship_debug_output(self, debug_output: Dict, _temp_debug_values: Dict, method_name: str, probe_ts: pd.Timestamp):
        full_chain = [
            ("1. 物理层 (Raw Arsenal Data)", "1. 物理层 (Raw Arsenal Data)"),
            ("主力平均价格(HAB版)", "2. 原子层 (Weighted Cost Calc)"),
            ("逻辑层_LV生态博弈(净力模型)", "3. 逻辑层 - 洛特卡-沃尔泰拉博弈"),
            ("组件_传统控盘(柔性梯度版)", "4. 逻辑层 - 传统控盘 (Fibonacci Resonance)"),
            ("组件_成本优势(Harmonic版)", "5. 逻辑层 - 成本优势 (Harmonic Oscillator)"),
            ("组件_净活动(V46矢量逆零基版)", "6. 逻辑层 - 资金动力学 (Kinematic Flows)"),
            ("归一化处理", "7. 转换层 (MTF & Normalization)"),
            ("融合_动力学(陷阱反转版)", "8. 决策层 - 深度融合 (Shannon-VPA Fusion)"),
            ("风控层_杠杆(张量扩张版)", "9. 风控层 (Asymmetric Leverage & Tension)"),
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
                    warn_tag = " [!] 极性反转触发" if (isinstance(v_print, float) and "Trap_Inversion_Factor" in sub_key and v_print < 0.0) else ""
                    warn_tag = " [!] 梯度激变/饱和" if (isinstance(v_print, float) and abs(v_print - 1.0) < 0.0001 and "Score" in sub_key) else warn_tag
                    if isinstance(v_print, (float, np.floating)):
                        debug_output[f"        {sub_key}: {v_print:.4f}{warn_tag}"] = ""
                    else:
                        debug_output[f"        {sub_key}: {v_print}"] = ""
        self._print_debug_info(debug_output)
    def _calculate_main_force_avg_prices(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        m = context['market']
        f = context['funds']
        s = context['structure']
        close = m['close']
        turnover = m.get('turnover_rate', pd.Series(1.0, index=index)).replace(0.0, np.nan).ffill().fillna(1.0)
        daily_main_buy_amt = f.get('buy_elg_amt', pd.Series(0.0, index=index)) + f.get('buy_lg_amt', pd.Series(0.0, index=index))
        daily_main_buy_vol = f.get('buy_elg_vol', pd.Series(0.0, index=index)) + f.get('buy_lg_vol', pd.Series(0.0, index=index))
        daily_main_sell_amt = f.get('sell_elg_amt', pd.Series(0.0, index=index)) + f.get('sell_lg_amt', pd.Series(0.0, index=index))
        daily_main_sell_vol = f.get('sell_elg_vol', pd.Series(0.0, index=index)) + f.get('sell_lg_vol', pd.Series(0.0, index=index))
        def _calc_hab_vwma(amt_s, vol_s, window):
            roll_amt = amt_s.rolling(window=window, min_periods=int(window*0.5)).sum().ffill().fillna(0.0)
            roll_vol = vol_s.rolling(window=window, min_periods=int(window*0.5)).sum().ffill().fillna(0.0)
            return (roll_amt / (roll_vol + 1e-6)).replace(0.0, np.nan).fillna(close)
        hab_cost_buy_21 = _calc_hab_vwma(daily_main_buy_amt, daily_main_buy_vol, 21)
        hab_cost_sell_21 = _calc_hab_vwma(daily_main_sell_amt, daily_main_sell_vol, 21)
        price_ratio = hab_cost_buy_21 / (close + 1e-6)
        unit_mismatch = (price_ratio > 5.0) | (price_ratio < 0.2)
        cyq_avg_cost = s.get('avg_cost', close)
        hab_cost_buy_21 = hab_cost_buy_21.mask(unit_mismatch, cyq_avg_cost)
        hab_cost_sell_21 = hab_cost_sell_21.mask(unit_mismatch, cyq_avg_cost)
        log_cost = np.arcsinh(hab_cost_buy_21)
        slope_clean, accel_clean, jerk_clean = self._apply_kinematics_with_threshold_gate(log_cost, (13, 8, 5))
        slope_std = self._get_robust_noise_floor(slope_clean)
        norm_slope = np.tanh(slope_clean / (slope_std * 1.5))
        norm_accel = np.tanh(accel_clean / (slope_std * 1.5))
        circ_mv = m.get('circ_mv', pd.Series(1e8, index=index))
        smart_bias = np.tanh((f.get('smart_synergy', pd.Series(0.0, index=index)) / (circ_mv + 1e-6)) * 1000.0)
        raw_power = norm_slope * 0.5 + norm_accel * 0.3 + smart_bias * 0.2
        kinematic_power = np.sign(raw_power) * np.power(np.abs(raw_power), 1.2).clip(-1.0, 1.0)
        flow_weight = np.tanh(turnover / 2.0).clip(0.2, 0.8)
        static_weight = 1.0 - flow_weight
        final_buy_price = (hab_cost_buy_21 * flow_weight + cyq_avg_cost * static_weight) * (1.0 + smart_bias * 0.02)
        final_sell_price = (hab_cost_sell_21 * flow_weight + cyq_avg_cost * static_weight)
        result = {"unit_mismatch": unit_mismatch.any(), "avg_buy": final_buy_price, "avg_sell": final_sell_price, "buy_slope": slope_clean, "buy_accel": accel_clean, "buy_jerk": jerk_clean, "kinematic_power": kinematic_power, "shadow_cost": cyq_avg_cost}
        if _temp_debug_values is not None:
            _temp_debug_values["主力平均价格(HAB版)"] = {"HAB_Cost_21": hab_cost_buy_21, "Kinematic_Power": kinematic_power}
        return result
    def _calculate_main_force_cost_advantage_score(self, context: Dict, index: pd.Index, hab_prices: Dict, _temp_debug_values: Dict) -> pd.Series:
        m = context['market']
        s = context['structure']
        f = context['funds']
        close = m['close']
        avg_cost = hab_prices.get('avg_buy', s.get('avg_cost', close))
        pressure_trapped = s.get('pressure_trapped', pd.Series(50.0, index=index))
        displacement = (close - avg_cost) / (avg_cost + 1e-6)
        arcsinh_cost = np.arcsinh(avg_cost)
        slope_cost, accel_cost, jerk_cost = self._apply_kinematics_with_threshold_gate(arcsinh_cost, (13, 8, 5))
        circ_mv = m.get('circ_mv', pd.Series(1e8, index=index))
        hab_shield = np.tanh((f.get('hab_net_mf_21', pd.Series(0.0, index=index)).abs() / (circ_mv * 0.001 + 1e-6)) * 0.5).clip(0.0, 1.0)
        k_spring = 1.5
        restoring_force = -k_spring * displacement
        damping_coeff = (pressure_trapped / 100.0) * (1.0 - hab_shield * 0.8)
        damping_force = -damping_coeff * slope_cost
        composite_flow = f.get('net_mf_calibrated', pd.Series(0.0, index=index)) + f.get('smart_synergy', pd.Series(0.0, index=index))
        driving_force = np.tanh(composite_flow / (f.get('hab_net_mf_21', pd.Series(1e-6, index=index)).abs() + 1e-6))
        net_force = restoring_force + damping_force + driving_force
        winner_rate = s.get('winner_rate', pd.Series(50.0, index=index))
        chip_entropy = s.get('chip_entropy', pd.Series(100.0, index=index))
        score_winner = (2.0 / (1.0 + np.exp(-(winner_rate - 65.0) * 0.15))) - 1.0
        score_order = 1.0 / (1.0 + np.exp((chip_entropy - 75.0) * 0.1))
        kine_score = np.tanh(slope_cost * 0.5) * 0.3 + np.tanh(accel_cost * 2.0) * 0.5 + np.tanh(jerk_cost * 2.0) * 0.2
        raw_score = (score_winner * 0.2 + kine_score * 0.3 + score_order * 0.2 + np.tanh(net_force) * 0.3)
        effective_damping = (np.exp(-(pressure_trapped / 25.0)) + (hab_shield * 0.6)).clip(0.0, 1.0)
        damped_score = raw_score * effective_damping
        is_jailbreak = (pressure_trapped > 60.0) & (driving_force > 0.3) & (accel_cost > 0.0)
        profit_ratio = s.get('profit_ratio', pd.Series(50.0, index=index))
        gamma = pd.Series(1.0, index=index).mask((profit_ratio > 90.0) & (damped_score > 0.0), 0.7)
        raw_final_score = np.sign(damped_score) * np.power(np.abs(damped_score), gamma) + (is_jailbreak.astype(np.float32) * 0.4)
        entropy_penalty = pd.Series(1.0, index=index).mask((score_order < 0.2) & (raw_final_score > 0.0), 0.5)
        raw_final_score = raw_final_score * entropy_penalty
        final_score = (raw_final_score / np.sqrt(1.0 + np.square(raw_final_score))).astype(np.float32)
        if _temp_debug_values is not None:
            _temp_debug_values["组件_成本优势(Harmonic版)"] = {"Net_Force": net_force, "Final_Cost_Score": final_score}
        return final_score
    def _calculate_main_force_net_activity_score(self, context: Dict, index: pd.Index, config: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        f = context['funds']
        m = context['market']
        s_sent = context['sentiment']
        circ_mv = m.get('circ_mv', pd.Series(1e8, index=index))
        norm_tick = (f.get('tick_lg_net', pd.Series(0.0, index=index)) / (circ_mv + 1e-6)) * 1000.0
        norm_smart = (f.get('smart_synergy', pd.Series(0.0, index=index)) / (circ_mv + 1e-6)) * 1000.0
        norm_macro = (f.get('net_mf_calibrated', pd.Series(0.0, index=index)) / (circ_mv + 1e-6)) * 1000.0
        quality_scalar = np.tanh((f.get('flow_consistency', pd.Series(50.0, index=index)) - 40.0) / 20.0).clip(0.5, 1.5)
        hf_skewness = f.get('hf_flow_skewness', pd.Series(0.0, index=index))
        skew_gain = 1.0 + np.tanh(hf_skewness).clip(0.0, 0.5)
        intraday_act = s_sent.get('intraday_mf_act', pd.Series(50.0, index=index))
        act_scalar = np.tanh((intraday_act - 50.0) / 20.0).clip(0.5, 1.5)
        raw_vector = (norm_tick * 0.4 + norm_smart * 0.4 + norm_macro * 0.2) * quality_scalar * skew_gain * act_scalar
        hab_34 = f.get('hab_net_mf_34', pd.Series(0.0, index=index))
        hab_density = (hab_34 / (circ_mv + 1e-6)) * 1000.0
        impact_strength = raw_vector / (np.abs(hab_density) + 1.0)
        is_washout_buffer = (hab_density > 5.0) & (raw_vector < 0.0) & (raw_vector > -3.0)
        inertia_dampener = pd.Series(1.0, index=index).mask(is_washout_buffer, 0.3)
        buffered_vector = impact_strength * inertia_dampener
        velocity, accel, jerk = self._apply_kinematics_with_threshold_gate(buffered_vector, (13, 8, 5))
        def _adaptive_tanh(s_val: pd.Series, scale=2.0):
            robust_std = self._get_robust_noise_floor(s_val)
            return np.tanh(s_val / (robust_std * scale))
        z_vel = _adaptive_tanh(velocity, 2.0)
        z_acc = _adaptive_tanh(accel, 1.5)
        z_jrk = _adaptive_tanh(jerk, 1.0)
        vpa_eff = s_sent.get('vpa_efficiency', pd.Series(50.0, index=index))
        eff_multiplier = (1.0 / (1.0 + np.exp(-(vpa_eff - 50.0) * 0.1))) + 0.5
        smart_attack = f.get('smart_attack', pd.Series(0.0, index=index))
        base_energy = (z_vel * 0.4 + z_acc * 0.4 + z_jrk * 0.2)
        final_energy = base_energy * eff_multiplier
        gain_exponent = pd.Series(1.0, index=index).mask((final_energy > 0.0) & (smart_attack > 0.5), 1.4)
        raw_score = np.sign(final_energy) * np.power(np.abs(final_energy), gain_exponent)
        final_score = (raw_score / np.sqrt(1.0 + np.square(raw_score))).astype(np.float32)
        if _temp_debug_values is not None:
            _temp_debug_values["组件_净活动(V46矢量逆零基版)"] = {"Impact_Strength": impact_strength, "Final_Activity_Score": final_score}
        return final_score
    def _calculate_traditional_control_score_components(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        ema = context['ema']
        s_struct = context['structure']
        f_funds = context['funds']
        ema_55 = ema.get('ema_55', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        if ema_55.isnull().all():
             return pd.Series(0.0, index=index)
        coherence = s_struct.get('ma_coherence', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        fractal_dim = s_struct.get('fractal_dim', pd.Series(1.5, index=index)).ffill().fillna(1.5)
        entropy = s_struct.get('price_entropy', pd.Series(3.0, index=index)).ffill().fillna(3.0)
        gap_momentum = f_funds.get('gap_momentum', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        kurtosis = s_struct.get('chip_kurtosis', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        log_ema = np.arcsinh(ema_55)
        slope, accel, jerk = self._apply_kinematics_with_threshold_gate(log_ema, (13, 8, 5))
        slope = slope * 100.0
        accel = accel * 100.0
        jerk = jerk * 100.0
        hab_slope_34 = slope.rolling(window=34, min_periods=21).sum().ffill().fillna(0.0)
        inertia_protection = pd.Series(1.0, index=index).mask((hab_slope_34 > 5.0) & (accel < 0.0) & (accel > -0.5), 0.5)
        base_vol = self._get_robust_noise_floor(slope)
        entropy_scalar = (entropy / 3.0).clip(0.5, 2.0)
        adaptive_denom = base_vol * entropy_scalar
        z_slope = np.tanh(slope / (adaptive_denom * 2.0))
        z_accel = np.tanh(accel / (adaptive_denom * 1.5))
        z_jerk = np.tanh(jerk / (adaptive_denom * 1.0))
        base_score = (z_slope * 0.4 + (z_accel * inertia_protection) * 0.4 + z_jerk * 0.2)
        resonance_mult = 1.0 + (np.tanh((coherence - 60.0) / 20.0).clip(0.0, 1.0) * 0.5)
        gap_bonus = np.tanh(gap_momentum / 10.0).clip(-0.3, 0.3)
        fractal_mult = pd.Series(1.0, index=index).mask(fractal_dim < 1.3, 1.2).mask(fractal_dim > 1.7, 0.8)
        kurtosis_gain = 1.0 + np.tanh(kurtosis / 10.0).clip(0.0, 0.5)
        raw_final = (base_score * resonance_mult * fractal_mult * kurtosis_gain) + gap_bonus
        final_score = (raw_final / np.sqrt(1.0 + np.square(raw_final))).astype(np.float32)
        if _temp_debug_values is not None:
            _temp_debug_values["组件_传统控盘(柔性梯度版)"] = {"HAB_Slope_34": hab_slope_34, "Final_Trad_Score": final_score}
        return final_score
    def _calculate_control_leverage_model(self, index: pd.Index, fused_score: pd.Series, net_activity_score: pd.Series, norm_flow: pd.Series, cost_score: pd.Series, lv_tension: pd.Series, norm_t0_buy: pd.Series, norm_t0_sell: pd.Series, norm_vwap_up: pd.Series, norm_vwap_down: pd.Series, context: Dict, _temp_debug_values: Dict) -> pd.Series:
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        s_state = context['state']
        chip_ent = s_struct.get('chip_entropy', pd.Series(100.0, index=index)).ffill().fillna(100.0)
        price_ent = s_struct.get('price_entropy', pd.Series(4.0, index=index)).ffill().fillna(4.0)
        rev_prob = s_sent.get('reversal_prob', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        div_str = s_sent.get('divergence_strength', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        is_leader = s_sent.get('market_leader', pd.Series(0.0, index=index)) > 0.5
        is_pit = s_sent.get('golden_pit', pd.Series(0.0, index=index)) > 0.5
        ind_rank = s_sent.get('industry_rank', pd.Series(50.0, index=index)).ffill().fillna(50.0)
        smart_attack = f_funds.get('smart_attack', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        t1_premium = s_sent.get('t1_premium', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        breakout_penalty = s_state.get('breakout_penalty', pd.Series(0.0, index=index)).ffill().fillna(0.0)
        norm_chip_ent = (chip_ent / 80.0).clip(0.0, 1.5)
        norm_price_ent = ((price_ent - 1.5) / 2.5).clip(0.0, 1.5)
        raw_disorder = (norm_chip_ent * 0.6 + norm_price_ent * 0.4)
        ent_slope, ent_accel, _ = self._apply_kinematics_with_threshold_gate(raw_disorder, (5, 3, 3))
        kinetic_penalty = pd.Series(0.0, index=index).mask(ent_accel > 0.05, 0.3)
        hab_disorder = raw_disorder.rolling(window=13, min_periods=5).mean().ffill().fillna(raw_disorder)
        is_structurally_stable = hab_disorder < 0.4
        effective_disorder = raw_disorder.mask(is_structurally_stable, raw_disorder * 0.7)
        entropy_damping = np.exp(-2.0 * np.power(effective_disorder, 2))
        ind_scalar = np.tanh((20.0 - ind_rank) / 10.0) * 0.2
        risk_penalty = pd.Series(0.0, index=index).mask(rev_prob > 80.0, 0.8).mask(div_str > 80.0, 0.6)
        penalty_factor = pd.Series(0.0, index=index).mask(breakout_penalty > 50.0, 0.5)
        base_lev = 1.0 + (np.abs(fused_score) / np.sqrt(1.0 + np.square(fused_score))) * 5.0 
        tension_multiplier = 1.0 + np.tanh(lv_tension / 5.0).clip(0.0, 1.0) * 0.5
        lev_step1 = base_lev * entropy_damping * tension_multiplier * np.maximum(0.1, (1.0 - kinetic_penalty - risk_penalty - penalty_factor))
        state_bonus = pd.Series(0.0, index=index).mask(is_leader, 0.5).mask(is_pit, 0.3)
        attack_bonus = pd.Series(0.0, index=index).mask(smart_attack > 0.5, 0.4)
        premium_bonus = (t1_premium / 100.0).clip(0.0, 0.5)
        raw_final_lev = lev_step1 * (1.0 + ind_scalar + state_bonus + attack_bonus + premium_bonus)
        is_bullish = net_activity_score >= 0.0
        is_divergence = (fused_score * net_activity_score) < 0.0
        bear_release = np.clip(np.abs(net_activity_score) * 4.0, 1.0, 4.0)
        unbounded_lev = raw_final_lev.copy()
        unbounded_lev = unbounded_lev.mask(is_divergence & ~is_bullish, raw_final_lev * bear_release)
        unbounded_lev = unbounded_lev.mask(is_divergence & is_bullish, raw_final_lev.clip(upper=2.0))
        unbounded_lev = unbounded_lev.mask(~is_divergence & ~is_bullish, raw_final_lev * bear_release)
        final_lev = 12.0 * np.tanh(unbounded_lev / 12.0)
        if _temp_debug_values is not None:
            _temp_debug_values["风控层_杠杆(张量扩张版)"] = {"Effective_Disorder": effective_disorder, "Tension_Multiplier": tension_multiplier, "Final_Leverage": final_lev}
        return final_lev
    def _fuse_control_scores(self, traditional_score: pd.Series, structural_score: pd.Series, lv_score: pd.Series, context: Dict, activity_score: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        m_market = context['market']
        s_state = context['state']
        profit = s_struct.get('profit_ratio', pd.Series(0.0, index=traditional_score.index))
        winner = s_struct.get('winner_rate', pd.Series(0.0, index=traditional_score.index))
        trapped = s_struct.get('pressure_trapped', pd.Series(0.0, index=traditional_score.index))
        stability = s_struct.get('chip_stability', pd.Series(50.0, index=traditional_score.index))
        squeeze_raw = ((profit + winner) * 0.5 - trapped)
        squeeze_score = np.tanh(squeeze_raw / 20.0)
        cost_modifier = 1.0 + (squeeze_score * 0.4)
        hab_55 = f_funds.get('hab_net_mf_55', pd.Series(0.0, index=traditional_score.index))
        hab_density = (hab_55 / (m_market.get('circ_mv', pd.Series(1e8, index=traditional_score.index)) + 1e-6))
        hab_shield = np.tanh(hab_density * 30.0).clip(0.0, 1.0)
        adx = s_sent.get('adx_14', pd.Series(20.0, index=traditional_score.index))
        w_trad_base = (1.0 / (1.0 + np.exp(-0.15 * (adx - 30.0)))).clip(0.3, 0.7)
        bull_trap_intensity = np.clip(traditional_score, 0.0, 1.0) * np.clip(-activity_score, 0.0, 1.0)
        w_trad_adjusted = w_trad_base * (1.0 - bull_trap_intensity * 0.8)
        raw_base_score = (traditional_score * w_trad_adjusted + structural_score * (1.0 - w_trad_adjusted) + lv_score * 0.3) * cost_modifier
        trap_inversion_factor = 1.0 - 2.5 * bull_trap_intensity
        base_score = raw_base_score * trap_inversion_factor
        turnover = s_sent.get('turnover', pd.Series(1.0, index=traditional_score.index)).replace(0.0, np.nan).fillna(1.0)
        locking_gain = 1.0 + np.tanh((hab_density * 100.0) / (turnover / 3.0)).clip(-0.5, 0.8)
        slope_stab, accel_stab, _ = self._apply_kinematics_with_threshold_gate(stability, (5, 3, 3))
        stab_kine = np.tanh(slope_stab * 0.1 + accel_stab * 0.2)
        stab_kine = stab_kine.mask(stab_kine < 0.0, stab_kine * (1.0 - hab_shield * 0.8))
        price_ent = s_struct.get('price_entropy', pd.Series(3.0, index=traditional_score.index))
        chip_ent = s_struct.get('chip_entropy', pd.Series(100.0, index=traditional_score.index))
        chip_div = s_struct.get('chip_divergence', pd.Series(0.0, index=traditional_score.index))
        norm_p_ent = ((price_ent - 1.0) / 4.0).clip(0.0, 1.0)
        norm_c_ent = (chip_ent / 100.0).clip(0.0, 1.0)
        norm_c_div = (chip_div / 100.0).clip(0.0, 1.0)
        entropy_penalty = 1.0 - ((((1.0 - norm_p_ent) * norm_c_ent) + norm_c_div * 0.5) * 0.8).clip(0.0, 0.9)
        smart_attack = f_funds.get('smart_attack', pd.Series(0.0, index=traditional_score.index))
        smart_divergence = f_funds.get('smart_divergence', pd.Series(0.0, index=traditional_score.index))
        sm_gate = pd.Series(1.0, index=traditional_score.index)
        sm_gate = sm_gate.mask(smart_attack > 0.5, 1.25)
        sm_gate = sm_gate.mask((base_score > 0.2) & (smart_divergence < -0.2), 0.6)
        raw_final = base_score * locking_gain * (1.0 + (stab_kine * 0.2)) * entropy_penalty * sm_gate
        is_golden_pit = s_state.get('golden_pit', pd.Series(0.0, index=traditional_score.index)) > 0.0
        raw_final = raw_final.mask(is_golden_pit & (raw_final < 0.0), raw_final * 0.5)
        final_fused = (raw_final / np.sqrt(1.0 + np.square(raw_final))).astype(np.float32)
        if _temp_debug_values is not None:
            _temp_debug_values["融合_动力学(陷阱反转版)"] = {"Squeeze_Score": squeeze_score, "Trap_Inversion_Factor": trap_inversion_factor, "Final_Fused_Score": final_fused}
        return final_fused
    def _normalize_components(self, df: pd.DataFrame, context: Dict, scores_traditional: pd.Series, config: Dict, method_name: str, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        std_trad = self._get_robust_noise_floor(scores_traditional).replace(0.0, 1.0)
        norm_traditional = np.tanh(scores_traditional / (std_trad * 1.5))
        stability = s_struct.get('chip_stability', pd.Series(50.0, index=df.index))
        slope_5 = (stability - stability.shift(5).ffill().fillna(50.0)) / 5.0
        slope_13 = (stability - stability.shift(13).ffill().fillna(50.0)) / 13.0
        slope_21 = (stability - stability.shift(21).ffill().fillna(50.0)) / 21.0
        mtf_struct_raw = slope_5 * 0.5 + slope_13 * 0.3 + slope_21 * 0.2
        norm_structural = np.tanh(mtf_struct_raw * 100.0)
        flow_consistency = f_funds.get('flow_consistency', pd.Series(50.0, index=df.index))
        norm_flow = (flow_consistency / 100.0).clip(0.0, 1.0)
        def _intent_sigmoid(s: pd.Series):
            return 1.0 / (1.0 + np.exp(-10.0 * (s - 0.5)))
        norm_t0_buy = _intent_sigmoid(s_sent.get('t0_buy_conf', pd.Series(50.0, index=df.index)) / 100.0)
        norm_t0_sell = _intent_sigmoid(s_sent.get('t0_sell_conf', pd.Series(50.0, index=df.index)) / 100.0)
        norm_vwap_up = s_sent.get('pushing_score', pd.Series(0.0, index=df.index)).clip(0.0, 1.0)
        norm_vwap_down = s_sent.get('shakeout_score', pd.Series(0.0, index=df.index)).clip(0.0, 1.0)
        if _temp_debug_values is not None:
            _temp_debug_values["归一化处理"] = {"structural_mtf_norm": norm_structural, "t0_buy_boost": norm_t0_buy}
        return norm_traditional, norm_structural, norm_flow, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down
    def _get_probe_timestamp(self, df: pd.DataFrame, is_debug: bool) -> Optional[pd.Timestamp]:
        if not is_debug or not self.probe_dates:
            return None
        probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
        for date in reversed(df.index):
            ts = pd.to_datetime(date)
            ts_naive = ts.tz_localize(None) if ts.tz is not None else ts
            if ts_naive.normalize() in probe_dates_dt:
                return date
        return None
    def _validate_arsenal_signals(self, df: pd.DataFrame, config: Dict, method_name: str, debug_output: Dict, probe_ts: pd.Timestamp) -> bool:
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
            'STATE_EMOTIONAL_EXTREME_D', 'STATE_BREAKOUT_CONFIRMED_D'
        ]
        if not self.helper._validate_required_signals(df, required_physical_raw, method_name):
            if probe_ts:
                missing_cols = [col for col in required_physical_raw if col not in df.columns]
                debug_output[f"    -> [致命错误] {method_name} 关键信号缺失: {missing_cols[:5]}... 强制熔断。"] = ""
                self._print_debug_info(debug_output)
            return False
        return True











