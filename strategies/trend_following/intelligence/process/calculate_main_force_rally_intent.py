# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
# 版本: V19.0 · 物理量纲统整与对数拓扑压缩版
# 说明: 1. 彻底修复变量断层引发的 NameError；2. 大规模修正底层因子的物理量纲映射 ([0,1]与[0,100]分制错位)；3. 引入 np.log1p 对数压缩层，解决抛压/拉升奇点畸变导致的 Z-Score 映射失效问题。
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.trend_following.utils import get_param_value

class CalculateMainForceRallyIntent:
    """
    PROCESS_META_MAIN_FORCE_RALLY_INTENT
    【V28.0.0 · 消除未来函数与绝对尺度正交对称版】
    重写Z-Score消除未来函数泄漏(引入144日Rolling MAD)。
    修复MACD与价格尺度的量纲鸿沟，引入正交对称的多空共振矩阵(Long/Short Resonance)。
    """
    def __init__(self, strategy_instance, process_intelligence_helper_instance):
        self.strategy = strategy_instance
        self.helper = process_intelligence_helper_instance
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self._probe_cache = []
    def _kinematic_gate(self, val: np.ndarray, threshold: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return np.tanh(np.sign(val) * np.maximum(0.0, np.abs(val) - threshold) / scale)
    def _robust_normalize(self, tensor: np.ndarray):
        ts = pd.Series(tensor)
        med = ts.rolling(window=144, min_periods=1).median().fillna(0.0).values
        mad = (ts - med).abs().rolling(window=144, min_periods=1).median().fillna(0.1).values
        robust_mad = np.maximum(mad, 0.1)
        z_scores = (tensor - med) / (robust_mad * 3.0)
        final_scores = 1.0 / (1.0 + np.exp(np.clip(-z_scores, -20.0, 20.0)))
        return final_scores, z_scores, med, mad
    def _get_neutral_defaults(self) -> Dict[str, float]:
        defaults = {k: 0.0 for k in self._get_required_column_map().keys()}
        score_keys = [
            'pushing_score', 'winner_rate', 'peak_conc', 'accumulation_score', 
            'platform_quality', 'dist_score', 'shakeout_score', 'theme_hotness', 
            'lock_ratio', 'consolidation_chip_conc', 'downtrend_str', 't1_premium', 
            'industry_markup', 'breakout_flow', 'flow_consistency', 'closing_intensity', 
            'hf_flow_div', 'dist_energy', 'outflow_qual', 'reversal_prob', 'market_sentiment',
            'breakout_pot', 'turnover_stability', 'trend_confirm', 'intra_consolidation', 
            'foundation_strength', 'cons_accum', 'closing_str', 'rsi', 'intra_support',
            'turnover_intensity', 'vol_adj_conc', 'uptrend_str', 'behav_accum', 'behav_dist',
            'absorption_energy'
        ]
        for k in score_keys: defaults[k] = 50.0
        ratio_keys = [
            'mf_activity', 'ind_downtrend', 'chip_convergence', 'control_solidity', 'chip_stability',
            'intra_acc_conf', 'intraday_dist', 'game_intensity', 'energy_conc', 'ma_compression',
            'vpa_adj', 'robust_trend'
        ]
        for k in ratio_keys: defaults[k] = 0.5
        zero_keys = [
            'ma_coherence', 'ma_tension', 'profit_pressure', 'trapped_pressure', 'intra_skew', 
            'buy_elg_rate', 'chip_divergence', 'gap_momentum', 'instability', 'pressure_release', 
            'pf_div', 'cmf', 'geom_slope', 'net_energy', 'macdh', 'parabolic_warn', 'bias_21', 
            'exp_flow_1d', 'div_strength', 'up_slope_13', 'up_accel_13', 'bias_slope_5'
        ]
        for k in zero_keys: defaults[k] = 0.0
        defaults['hab_structure'] = 0.6
        defaults['tick_abnormal_vol'] = 1.0
        defaults['sr_ratio'] = 1.0
        defaults['chip_entropy'] = 1.0
        defaults['vpa_efficiency'] = 0.0
        defaults['turnover'] = 5.0
        defaults['close'] = 1.0
        defaults['cost_avg'] = 1.0
        defaults['bbp'] = 0.5
        defaults['fractal_dim'] = 1.5
        defaults['vol_burst'] = 1.0
        defaults['circ_mv'] = 500000.0
        return defaults
    def _load_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        data = {}
        col_map = self._get_required_column_map()
        defaults = self._get_neutral_defaults()
        for key, col_name in col_map.items():
            series = self.helper._get_safe_series(df, col_name, np.nan)
            if key not in ['close', 'cost_avg', 'volume', 'vol_13d', 'vol_21d', 'vol_34d', 'vol_55d', 'mf_net_buy', 'hab_inv_13', 'hab_inv_21', 'hab_inv_34', 'hab_inv_55', 'circ_mv', 'exp_flow_1d']:
                series = series.fillna(defaults.get(key, 0.0))
            else:
                series = series.fillna(0.0)
            data[key] = series.astype(np.float32)
        if 'close' in data: data['close'] = data['close'].ffill().bfill().fillna(1.0)
        if 'cost_avg' in data and 'close' in data: data['cost_avg'] = data['cost_avg'].replace(0.0, np.nan).fillna(data['close'])
        return data
    def _get_required_column_map(self) -> Dict[str, str]:
        return {
            'close': 'close_D', 'cost_avg': 'cost_50pct_D', 'mf_net_buy': 'net_mf_amount_D',
            'volume': 'volume_D', 'vol_13d': 'total_volume_13d_D', 'vol_21d': 'total_volume_21d_D',
            'vol_34d': 'total_volume_34d_D', 'vol_55d': 'total_volume_55d_D',
            'hab_inv_13': 'total_net_amount_13d_D', 'hab_inv_21': 'total_net_amount_21d_D',
            'hab_inv_34': 'total_net_amount_34d_D', 'hab_inv_55': 'total_net_amount_55d_D',
            'net_energy': 'net_energy_flow_D', 'vol_burst': 'volume_ratio_D',
            'pf_div': 'price_flow_divergence_D', 'geom_slope': 'GEOM_REG_SLOPE_D',
            'vpa_adj': 'VPA_MF_ADJUSTED_EFF_D', 'cons_accum': 'consolidation_accumulation_score_D',
            'cmf': 'CMF_21_D', 'bbp': 'BBP_21_2.0_D', 'closing_str': 'CLOSING_STRENGTH_D',
            'fractal_dim': 'PRICE_FRACTAL_DIM_D', 'robust_trend': 'STATE_ROBUST_TREND_D',
            'macdh': 'MACDh_13_34_8_D', 'circ_mv': 'circ_mv_D', 'rsi': 'RSI_13_D',
            'parabolic_warn': 'STATE_PARABOLIC_WARNING_D', 'intra_support': 'INTRADAY_SUPPORT_INTENT_D',
            'turnover_intensity': 'intraday_chip_turnover_intensity_D', 'turnover': 'turnover_rate_f_D',
            'vol_adj_conc': 'volatility_adjusted_concentration_D', 'exp_flow_1d': 'expected_flow_next_1d_D',
            'bias_21': 'BIAS_21_D', 'chip_divergence': 'chip_divergence_ratio_D',
            'uptrend_str': 'uptrend_strength_D', 'div_strength': 'divergence_strength_D',
            'behav_accum': 'behavior_accumulation_D', 'behav_dist': 'behavior_distribution_D',
            'absorption_energy': 'absorption_energy_D', 'bias_slope_5': 'SLOPE_5_BIAS_21_D',
            'up_slope_13': 'SLOPE_13_uptrend_strength_D', 'up_accel_13': 'ACCEL_13_uptrend_strength_D',
            'mf_slope_13': 'SLOPE_13_net_mf_amount_D', 'mf_accel_13': 'ACCEL_13_net_mf_amount_D', 'mf_jerk_13': 'JERK_13_net_mf_amount_D',
            'cmf_slope_13': 'SLOPE_13_CMF_21_D', 'cmf_accel_13': 'ACCEL_13_CMF_21_D',
            'hm_synergy': 'HM_COORDINATED_ATTACK_D', 'pushing_score': 'pushing_score_D',
            'market_sentiment': 'market_sentiment_score_D', 'tick_large_net': 'tick_large_order_net_D',
            'intra_accel': 'flow_acceleration_intraday_D', 'breakout_flow': 'breakout_fundflow_score_D',
            'mf_activity': 'intraday_main_force_activity_D', 'energy_conc': 'energy_concentration_D',
            'winner_rate': 'winner_rate_D', 'control_solidity': 'consolidation_chip_stability_D',
            'chip_entropy': 'chip_entropy_D', 'chip_stability': 'chip_stability_D',
            'peak_conc': 'peak_concentration_D', 'accumulation_score': 'accumulation_signal_score_D',
            'ma_coherence': 'MA_COHERENCE_RESONANCE_D', 'hab_structure': 'long_term_chip_ratio_D',
            'conc_slope': 'SLOPE_5_peak_concentration_D', 'winner_accel': 'ACCEL_5_winner_rate_D',
            'platform_quality': 'consolidation_quality_score_D', 'foundation_strength': 'support_strength_D',
            'vpa_efficiency': 'VPA_EFFICIENCY_D', 'profit_pressure': 'profit_pressure_D',
            'trapped_pressure': 'pressure_trapped_D',
            'dist_score': 'distribution_score_D', 'intraday_dist': 'intraday_distribution_confidence_D',
            'instability': 'flow_volatility_21d_D', 'pressure_release': 'pressure_release_index_D',
            'shakeout_score': 'shakeout_score_D', 'chip_divergence': 'chip_divergence_ratio_D',
            'dist_slope_13': 'SLOPE_13_distribution_score_D', 'dist_accel_13': 'ACCEL_13_distribution_score_D', 'dist_jerk_13': 'JERK_13_distribution_score_D',
            'gap_momentum': 'GAP_MOMENTUM_STRENGTH_D', 'emotional_extreme': 'STATE_EMOTIONAL_EXTREME_D',
            'reversal_prob': 'reversal_prob_D', 'is_leader': 'STATE_MARKET_LEADER_D',
            'theme_hotness': 'THEME_HOTNESS_SCORE_D', 'lock_ratio': 'high_position_lock_ratio_90_D',
            'trend_confirm': 'trend_confirmation_score_D', 'buy_elg_rate': 'buy_elg_amount_rate_D',
            'flow_consistency': 'flow_consistency_D', 'flow_persistence': 'flow_persistence_minutes_D',
            'closing_intensity': 'closing_flow_intensity_D', 'industry_markup': 'industry_markup_score_D',
            'tick_abnormal_vol': 'tick_abnormal_volume_ratio_D', 'intra_acc_conf': 'intraday_accumulation_confidence_D',
            'tick_net_slope_13': 'SLOPE_13_tick_large_order_net_D', 'tick_net_accel_13': 'ACCEL_13_tick_large_order_net_D', 'tick_net_jerk_13': 'JERK_13_tick_large_order_net_D',
            'pushing_slope_13': 'SLOPE_13_pushing_score_D', 'pushing_accel_13': 'ACCEL_13_pushing_score_D', 'pushing_jerk_13': 'JERK_13_pushing_score_D',
            'chip_convergence': 'chip_convergence_ratio_D', 'intra_consolidation': 'intraday_chip_consolidation_degree_D',
            'ma_tension': 'MA_POTENTIAL_TENSION_INDEX_D', 'consolidation_chip_conc': 'consolidation_chip_concentration_D',
            'rounding_bottom': 'STATE_ROUNDING_BOTTOM_D', 'sr_ratio': 'support_resistance_ratio_D',
            'ctrl_slope_13': 'SLOPE_13_consolidation_chip_stability_D', 'ctrl_accel_13': 'ACCEL_13_consolidation_chip_stability_D', 'ctrl_jerk_13': 'JERK_13_consolidation_chip_stability_D',
            'outflow_qual': 'outflow_quality_D', 'intra_skew': 'intraday_price_distribution_skewness_D',
            'ind_downtrend': 'industry_downtrend_score_D', 'downtrend_str': 'downtrend_strength_D',
            'dist_energy': 'distribution_energy_D', 'hf_flow_div': 'high_freq_flow_divergence_D',
            'game_intensity': 'game_intensity_D', 'golden_pit': 'STATE_GOLDEN_PIT_D',
            'breakout_conf': 'STATE_BREAKOUT_CONFIRMED_D', 'hm_top_tier': 'HM_ACTIVE_TOP_TIER_D',
            't1_premium': 'T1_PREMIUM_EXPECTATION_D', 'breakout_pot': 'breakout_potential_D',
            'ma_compression': 'MA_POTENTIAL_COMPRESSION_RATE_D', 'turnover_stability': 'TURNOVER_STABILITY_INDEX_D'
        }
    def _get_probe_locs(self, idx: pd.Index, target_tensor: np.ndarray = None) -> List[int]:
        locs = set()
        if self.probe_dates:
            target_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
            current_dates = idx.tz_localize(None).normalize()
            matched = np.where(current_dates.isin(target_dates))[0]
            locs.update(matched.tolist())
        if target_tensor is not None:
            abnormal = np.where(np.isnan(target_tensor) | np.isinf(target_tensor))[0]
            locs.update(abnormal.tolist()[:3])
            valid_mask = ~(np.isnan(target_tensor) | np.isinf(target_tensor))
            if np.any(valid_mask):
                valid_idx = np.where(valid_mask)[0]
                locs.add(valid_idx[np.argmax(target_tensor[valid_mask])])
                locs.add(valid_idx[np.argmin(target_tensor[valid_mask])])
        count = len(idx)
        if count > 0: locs.update([0, count - 1])
        return sorted(list(locs))
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        self._probe_cache = []
        raw = self._load_data(df)
        idx = df.index
        count = len(idx)
        if count < 5: return pd.Series(0.0, index=idx)
        self._probe_cache_raw = raw
        self._probe_cache_idx = idx
        hab_pool_flow = (raw['hab_inv_13'].values * 0.4 + raw['hab_inv_21'].values * 0.3 + raw['hab_inv_34'].values * 0.2 + raw['hab_inv_55'].values * 0.1)
        hab_pool_vol = (raw['vol_13d'].values * 0.4 + raw['vol_21d'].values * 0.3 + raw['vol_34d'].values * 0.2 + raw['vol_55d'].values * 0.1)
        circ_mv_yi = np.maximum(raw['circ_mv'].values / 10000.0, 1.0)
        cap_discount = np.clip(np.log1p(1000.0) / np.log1p(circ_mv_yi), 0.5, 3.0)
        cap_factor = np.clip(circ_mv_yi / 100.0, 0.1, 10.0)
        thrust = self._calc_thrust_component(raw, idx, hab_pool_flow, hab_pool_vol, cap_discount, cap_factor)
        structure = self._calc_structure_component(raw, idx)
        drag = self._calc_drag_component(raw, idx, hab_pool_flow, cap_discount)
        raw_intent, long_res, short_res = self._calc_tensor_synthesis(thrust, structure, drag, raw, idx, hab_pool_flow, cap_discount, cap_factor)
        raw_intent_clean = np.clip(np.nan_to_num(raw_intent, nan=0.0), -1e9, 1e9)
        compressed_intent = np.sign(raw_intent_clean) * np.log1p(np.abs(raw_intent_clean))
        final_scores, z_scores, med_arr, mad_arr = self._robust_normalize(compressed_intent)
        if self._is_probe_enabled(): self._generate_probe_report(idx, raw, thrust, structure, drag, raw_intent_clean, compressed_intent, z_scores, final_scores, cap_discount, cap_factor, long_res, short_res, med_arr, mad_arr)
        return pd.Series(final_scores, index=idx, dtype=np.float32)
    def _calc_thrust_component(self, raw: Dict[str, np.ndarray], idx: pd.Index, hab_pool_flow: np.ndarray, hab_pool_vol: np.ndarray, cap_discount: np.ndarray, cap_factor: np.ndarray) -> np.ndarray:
        mf_net_buy = raw['mf_net_buy'].values
        tick_large_net = raw['tick_large_net'].values
        hm_synergy = raw['hm_synergy'].values
        net_energy = raw['net_energy'].values
        vol_current = raw['volume'].values
        vol_burst = raw['vol_burst'].values
        cmf = raw['cmf'].values
        k_slope = self._kinematic_gate(raw['mf_slope_13'].values, np.array([500.0]) * cap_factor, np.array([5000.0]) * cap_factor)
        k_accel = self._kinematic_gate(raw['mf_accel_13'].values, np.array([200.0]) * cap_factor, np.array([2000.0]) * cap_factor)
        k_jerk = self._kinematic_gate(raw['mf_jerk_13'].values, np.array([50.0]) * cap_factor, np.array([500.0]) * cap_factor)
        tick_net_slope = self._kinematic_gate(raw['tick_net_slope_13'].values, np.array([50.0]) * cap_factor, np.array([2500.0]) * cap_factor)
        tick_net_accel = self._kinematic_gate(raw['tick_net_accel_13'].values, np.array([25.0]) * cap_factor, np.array([1250.0]) * cap_factor)
        tick_net_jerk = self._kinematic_gate(raw['tick_net_jerk_13'].values, np.array([5.0]) * cap_factor, np.array([250.0]) * cap_factor)
        push_slope = self._kinematic_gate(raw['pushing_slope_13'].values, np.array([0.5]), np.array([5.0]))
        push_accel = self._kinematic_gate(raw['pushing_accel_13'].values, np.array([0.25]), np.array([2.5]))
        push_jerk = self._kinematic_gate(raw['pushing_jerk_13'].values, np.array([0.1]), np.array([1.0]))
        cmf_slope = self._kinematic_gate(raw['cmf_slope_13'].values, np.array([0.02]), np.array([0.2]))
        cmf_accel = self._kinematic_gate(raw['cmf_accel_13'].values, np.array([0.01]), np.array([0.1]))
        hab_flow_impact = np.arcsinh(np.where(np.abs(hab_pool_flow) > 1.0, (mf_net_buy * cap_discount) / (np.abs(hab_pool_flow) / 23.8 + 1e-5), 0.0))
        hab_vol_impact = np.arcsinh(np.where(np.abs(hab_pool_vol) > 1.0, (vol_current * cap_discount) / (np.abs(hab_pool_vol) / 23.8 + 1e-5), 0.0))
        synergy_multiplier = 1.0 + np.maximum(0.0, hm_synergy / 100.0)
        net_energy_amp = 1.0 + np.tanh(net_energy / 100.0)
        macro_base = mf_net_buy * net_energy_amp * synergy_multiplier * (1.0 + np.tanh(np.abs(hab_flow_impact)))
        norm_macro_base = np.sign(macro_base) * np.log1p(np.abs(macro_base) / (1000.0 * cap_factor))
        macro_damping = np.tanh(np.abs(mf_net_buy) / (10000.0 * cap_factor))
        tick_damping = np.tanh(np.abs(tick_large_net) / (5000.0 * cap_factor))
        push_damping = np.clip((raw['pushing_score'].values - 50.0) / 50.0, -1.0, 1.0)
        macro_kinematics = (k_slope + k_accel + k_jerk) * macro_damping
        tick_kinematics = (tick_net_slope + tick_net_accel + tick_net_jerk) * tick_damping
        push_kinematics = (push_slope + push_accel + push_jerk) * np.maximum(0.0, push_damping)
        cmf_kinematics = (cmf_slope + cmf_accel) * np.clip(cmf, -1.0, 1.0)
        coupling_field = np.tanh(macro_kinematics + tick_kinematics + push_kinematics + cmf_kinematics)
        kinematic_multiplier = 1.0 + np.maximum(0.0, coupling_field)
        purity_multiplier = 1.0 + np.maximum(0.0, np.tanh(raw['buy_elg_rate'].values / 20.0))
        acc_conf_norm = np.clip((raw['intra_acc_conf'].values - 0.5) * 2.0, -1.0, 1.0)
        acc_confidence_multiplier = 1.0 + np.maximum(0.0, acc_conf_norm)
        norm_uptrend = np.clip(raw['uptrend_str'].values / 100.0, 0.0, 1.0)
        macro_momentum = norm_macro_base * purity_multiplier * acc_confidence_multiplier * (1.0 + coupling_field * 0.5) * (1.0 + norm_uptrend)
        persistence_factor = np.tanh(raw['flow_persistence'].values / 120.0)
        tick_intensity = np.clip(tick_large_net / (np.abs(mf_net_buy) + 1e-9), -50.0, 50.0)
        detonation_boost = 1.0 + np.tanh(np.maximum(0.0, raw['tick_abnormal_vol'].values - 1.0)) + np.clip(vol_burst - 1.0, 0.0, 5.0) * 0.2 * np.tanh(np.abs(hab_vol_impact))
        norm_flow_consistency = np.clip(raw['flow_consistency'].values / 100.0, 0.0, 1.0)
        energy_dissipation = np.maximum(0.01, 1.0 - norm_flow_consistency)
        aligned_jet = np.clip(raw['intra_accel'].values / 10.0, -5.0, 5.0) + tick_intensity
        micro_jet_raw = aligned_jet * (raw['pushing_score'].values / 100.0) * np.maximum(0.1, raw['mf_activity'].values) * persistence_factor * detonation_boost / energy_dissipation
        jet_exponent = np.tanh(micro_jet_raw) * (raw['breakout_flow'].values / 50.0) * np.maximum(0.1, norm_flow_consistency)
        micro_multiplier = np.exp(np.clip(jet_exponent, -2.0, 2.0))
        closing_amplifier = 1.0 + np.maximum(0.0, np.tanh((raw['closing_intensity'].values - 50.0) / 50.0))
        sentiment_amplifier = 1.0 + np.maximum(0.0, np.clip((raw['market_sentiment'].values - 50.0) / 50.0, -1.0, 1.0))
        industry_resonance = 1.0 + np.maximum(0.0, np.clip((raw['industry_markup'].values - 50.0) / 50.0, -1.0, 1.0))
        phase_alignment = np.where((macro_momentum > 0) & (raw['industry_markup'].values > 50), 1.2, 0.8)
        base_final_thrust = macro_momentum * micro_multiplier * kinematic_multiplier * sentiment_amplifier * closing_amplifier * industry_resonance * phase_alignment
        excess_kine = np.maximum(0.0, kinematic_multiplier - 1.0)
        excess_jet = np.maximum(0.0, micro_multiplier - 1.0)
        excess_beta = np.maximum(0.0, industry_resonance - 1.0)
        critical_resonance_index = excess_kine * excess_jet * excess_beta * acc_confidence_multiplier * (1.0 + np.tanh(np.abs(hab_vol_impact)))
        nonlinear_gain = 1.0 + np.expm1(np.clip(np.power(critical_resonance_index, 1.618), 0.0, 5.0))
        return np.clip(np.nan_to_num(base_final_thrust * nonlinear_gain, nan=0.0), -1000.0, 1000.0)
    def _calc_structure_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        cost_gap = (raw['close'].values - raw['cost_avg'].values) / (raw['cost_avg'].values + 1e-9)
        cost_rbf = np.exp(np.clip(-10.0 * (cost_gap - 0.05)**2, -20.0, 20.0))
        entropy_raw = np.maximum(0.01, raw['chip_entropy'].values)
        norm_intra_consolidation = 1.0 / (1.0 + np.exp(np.clip(-0.05 * (raw['intra_consolidation'].values - 50.0), -20.0, 20.0)))
        stability_raw = np.clip(raw['chip_stability'].values, 0.0, 1.0) + norm_intra_consolidation * 0.5
        entropy_penalty = np.maximum(1e-4, entropy_raw / (1.0 + stability_raw))
        norm_coherence = 1.0 / (1.0 + np.exp(np.clip(-2.0 * raw['ma_coherence'].values, -10.0, 10.0)))
        lattice_orderliness = (stability_raw * np.maximum(0.1, norm_coherence)) / entropy_penalty
        norm_convergence = np.clip(raw['chip_convergence'].values, 0.0, 1.0)
        convergence_factor = 1.0 + norm_convergence
        norm_tension = np.tanh(np.maximum(0.0, raw['ma_tension'].values) / 2.0)
        elastic_compression = np.maximum(0.0, norm_tension * norm_convergence)
        norm_peak_conc = np.clip((raw['peak_conc'].values * 0.5 + raw['vol_adj_conc'].values * 0.5) / 100.0, 0.0, 1.0)
        peak_efficiency = norm_peak_conc * np.clip(raw['winner_rate'].values / 100.0, 0.0, 1.0) * convergence_factor
        norm_control = np.clip(raw['control_solidity'].values, 0.0, 1.0)
        control_factor = 1.0 + norm_control * 0.5
        norm_acc = np.clip(raw['accumulation_score'].values / 100.0, 0.0, 1.0)
        norm_cons_accum = np.clip(raw['cons_accum'].values / 100.0, 0.0, 1.0)
        norm_behav_accum = np.clip(raw['behav_accum'].values / 100.0, 0.0, 1.0)
        acc_factor = 1.0 + norm_acc + norm_cons_accum * 0.5 + norm_behav_accum * 0.5
        trend_bonus = 1.0 + np.clip(raw['uptrend_str'].values / 100.0, 0.0, 1.0) * 0.5
        static_lattice_energy = lattice_orderliness * peak_efficiency * cost_rbf * control_factor * acc_factor * trend_bonus * 2.0
        inertia_bonus = 1.0 + np.maximum(0.0, (raw['hab_structure'].values - 0.6) * 1.5)
        ctrl_damping = np.maximum(0.1, norm_control)
        raw_ctrl_kine = self._kinematic_gate(raw['ctrl_slope_13'].values, np.array([0.01]), np.array([0.1])) + self._kinematic_gate(raw['ctrl_accel_13'].values, np.array([0.005]), np.array([0.05])) + self._kinematic_gate(raw['ctrl_jerk_13'].values, np.array([0.001]), np.array([0.01]))
        effective_ctrl_kine = raw_ctrl_kine * ctrl_damping
        k_conc_slope = np.tanh(self._kinematic_gate(raw['conc_slope'].values, np.array([0.5]), np.array([5.0])) * 2.0)
        k_winner_accel = np.tanh(self._kinematic_gate(raw['winner_accel'].values, np.array([1.0]), np.array([10.0])) * 1.5)
        kine_vector = (k_conc_slope * 0.2) + (k_winner_accel * 0.15) + (np.tanh(effective_ctrl_kine) * 0.35)
        geom_multiplier = 1.0 + np.tanh(raw['geom_slope'].values) * 0.5
        turnover_int_boost = 1.0 + np.clip(raw['turnover_intensity'].values / 100.0, 0.0, 1.0) * 0.5
        up_slope = self._kinematic_gate(raw['up_slope_13'].values, np.array([1.0]), np.array([10.0]))
        up_accel = self._kinematic_gate(raw['up_accel_13'].values, np.array([0.5]), np.array([5.0]))
        trend_kinematics = np.tanh(up_slope + up_accel) * 0.5
        evolution_kinematics = 1.0 + kine_vector * (1.0 + elastic_compression * 2.0) * geom_multiplier * turnover_int_boost + trend_kinematics
        norm_consolidation_conc = np.clip(raw['consolidation_chip_conc'].values / 100.0, 0.0, 1.0)
        consolidation_boost = 1.0 + norm_consolidation_conc
        norm_platform = np.clip(raw['platform_quality'].values / 100.0, 0.0, 1.0)
        platform_factor = 1.0 + norm_platform * 0.6 * consolidation_boost
        sr_factor = np.exp(np.clip(np.tanh(raw['sr_ratio'].values - 1.0), -5.0, 5.0))
        norm_foundation = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (raw['foundation_strength'].values - 50.0), -20.0, 20.0)))
        foundation_factor = 1.0 + norm_foundation * 0.4 * sr_factor
        pattern_bonus = 1.0 + (raw['rounding_bottom'].values * 0.3)
        base_structure = static_lattice_energy * inertia_bonus * evolution_kinematics * platform_factor * foundation_factor * pattern_bonus
        sri = (lattice_orderliness * norm_platform * raw['hab_structure'].values * np.maximum(0.1, norm_tension))
        excitation_gain = 1.0 + np.expm1(np.clip(np.maximum(0.0, sri - 0.5), 0.0, 3.0)) * 1.5
        resonance_core = base_structure * excitation_gain
        excess_res = np.clip(np.maximum(0.0, resonance_core - 1.5), 0.0, 5.0)
        avalanche_gain = 1.0 + np.power(excess_res, 1.618) * 1.5
        return np.clip(np.nan_to_num(resonance_core * avalanche_gain, nan=1.0), 0.01, 1000.0)
    def _calc_drag_component(self, raw: Dict[str, np.ndarray], idx: pd.Index, hab_pool_flow: np.ndarray, cap_discount: np.ndarray) -> np.ndarray:
        dist_damping = np.clip(raw['dist_score'].values / 100.0, 0.0, 1.0)
        dist_slope_13 = self._kinematic_gate(raw['dist_slope_13'].values, np.array([1.0]), np.array([10.0]))
        dist_accel_13 = self._kinematic_gate(raw['dist_accel_13'].values, np.array([0.5]), np.array([5.0]))
        dist_jerk_13 = self._kinematic_gate(raw['dist_jerk_13'].values, np.array([0.2]), np.array([2.0]))
        raw_dist_kine = dist_slope_13 + dist_accel_13 + dist_jerk_13
        effective_dist_kine = raw_dist_kine * dist_damping
        kine_multiplier = 1.0 + np.maximum(0.0, np.tanh(effective_dist_kine) * 1.5)
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool_flow * cap_discount / 50000.0, -20.0, 20.0))))
        hab_immunity = np.clip(hab_immunity, 0.0, 0.9)
        hab_drag_penalty = 1.0 + np.tanh(np.maximum(0.0, -hab_pool_flow * cap_discount) / 50000.0)
        norm_profit_pressure = np.expm1(np.clip(np.maximum(0.0, raw['profit_pressure'].values) / 50.0, 0.0, 5.0))
        norm_trapped_pressure = np.expm1(np.clip(np.maximum(0.0, raw['trapped_pressure'].values), 0.0, 1.0)) * 1.5
        norm_dist = np.clip(raw['dist_score'].values / 100.0, 0.0, 1.0)
        norm_intra_dist = np.clip(raw['intraday_dist'].values, 0.0, 1.0)
        norm_instability = np.tanh(np.maximum(0.0, raw['instability'].values) / 100.0)
        norm_downtrend = np.clip(raw['downtrend_str'].values / 100.0, 0.0, 1.0)
        dump_quality_factor = 1.0 + np.clip(raw['outflow_qual'].values / 100.0, 0.0, 1.0) * 1.5
        energy_factor = 1.0 + np.clip(raw['dist_energy'].values / 100.0, 0.0, 1.0)
        norm_behav_dist = np.clip(raw['behav_dist'].values / 100.0, 0.0, 1.0)
        pf_div_penalty = 1.0 + np.maximum(0.0, np.sinh(np.clip(raw['pf_div'].values / 20.0, 0.0, 3.0)))
        chip_div_penalty = 1.0 + np.maximum(0.0, np.sinh(np.clip(raw['chip_divergence'].values * 3.0, 0.0, 3.0)))
        div_str_penalty = 1.0 + np.sinh(np.clip(raw['div_strength'].values / 20.0, 0.0, 3.0))
        coupled_active_dump = (norm_dist + norm_intra_dist * 0.5) * dump_quality_factor * energy_factor * kine_multiplier * pf_div_penalty * chip_div_penalty * div_str_penalty * (1.0 + norm_behav_dist)
        beta_headwind = 1.0 + np.clip(raw['ind_downtrend'].values, 0.0, 1.0)
        friction_vpa = 1.0 + np.maximum(0.0, 1.0 - np.clip(raw['vpa_adj'].values / 100.0, -1.0, 1.0))
        skew_penalty = 1.0 + np.maximum(0.0, -raw['intra_skew'].values) * 0.5
        coupled_viscosity = (1.0 + norm_instability) * beta_headwind * friction_vpa * skew_penalty
        robust_trend_decay = 1.0 / (1.0 + np.clip(raw['robust_trend'].values, 0.0, 1.0) * 0.5)
        bias_val = raw['bias_21'].values
        bias_slope = self._kinematic_gate(raw['bias_slope_5'].values, np.array([0.5]), np.array([5.0]))
        bias_overbought = np.sinh(np.clip(np.maximum(0.0, bias_val) / 10.0, 0.0, 3.0))
        bias_oversold = np.sinh(np.clip(np.maximum(0.0, -bias_val) / 10.0, 0.0, 3.0))
        rsi_val = raw['rsi'].values
        rsi_overbought = np.maximum(0.0, rsi_val - 70.0) / 30.0
        rsi_oversold = np.maximum(0.0, 30.0 - rsi_val) / 30.0
        gravity_amplifier = (1.0 + rsi_overbought * 2.0 + bias_overbought) / (1.0 + rsi_oversold * 2.0 + bias_oversold)
        intra_support = np.clip(raw['intra_support'].values / 100.0, 0.0, 1.0)
        coupled_gravity = (norm_profit_pressure + norm_trapped_pressure) * (1.0 + norm_downtrend) * robust_trend_decay * gravity_amplifier * (1.0 - intra_support * 0.5)
        norm_release = np.clip(raw['pressure_release'].values, 0.0, 1.0)
        norm_shakeout = np.clip(raw['shakeout_score'].values / 100.0, 0.0, 1.0)
        relief_valve = np.maximum(0.1, 1.0 + norm_release * 1.5 + norm_shakeout * 1.0)
        hf_hidden_div = np.clip(raw['hf_flow_div'].values / 50.0, 0.0, 2.0)
        turnover_drag = np.expm1(np.clip((raw['turnover'].values / 100.0) - 0.05, 0.0, 5.0) * 10.0) * 0.5
        parabolic_penalty = 1.0 + np.clip(raw['parabolic_warn'].values, 0.0, 1.0) * 2.0
        core_drag_raw = np.clip(((coupled_gravity + coupled_active_dump) * coupled_viscosity * hab_drag_penalty) / relief_valve, 0.0, 1000.0)
        core_drag_shielded = np.clip(core_drag_raw * (1.0 - hab_immunity) + turnover_drag + hf_hidden_div, 0.0, 1000.0) * parabolic_penalty
        excess_drag = np.clip(np.maximum(0.0, core_drag_shielded - 1.5), 0.0, 10.0)
        avalanche_gain = 1.0 + np.power(excess_drag, 1.618) * 1.5
        return np.clip(np.nan_to_num(core_drag_shielded * avalanche_gain, nan=0.0), 0.0, 10000.0)
    def _calc_tensor_synthesis(self, thrust: np.ndarray, structure: np.ndarray, drag: np.ndarray, raw: Dict[str, np.ndarray], idx: pd.Index, hab_pool_flow: np.ndarray, cap_discount: np.ndarray, cap_factor: np.ndarray):
        norm_theme = np.clip(raw['theme_hotness'].values / 100.0, 0.0, 1.0)
        norm_breakout_pot = np.clip(raw['breakout_pot'].values / 100.0, 0.0, 1.0)
        norm_turnover_stab = np.clip(raw['turnover_stability'].values / 100.0, 0.0, 1.0)
        eco_premium = 1.0 + (raw['is_leader'].values * 0.8) + (raw['hm_top_tier'].values * 0.6) + (raw['breakout_conf'].values * 0.4) + (norm_theme * 0.3) + (np.clip(raw['trend_confirm'].values / 100.0, 0.0, 1.0) * 0.5)
        fractal_efficiency = np.exp(np.clip(1.5 - raw['fractal_dim'].values, -0.5, 0.5) * 2.0)
        eff_structure = np.where(thrust >= 0, np.sqrt(np.clip(structure, 0.01, 100.0)) * fractal_efficiency, 1.0 / np.clip(np.sqrt(np.clip(structure, 0.01, 100.0)) * fractal_efficiency, 0.2, 5.0))
        eff_eco_premium = np.where(thrust >= 0, eco_premium, 1.0 / np.clip(eco_premium, 0.5, 2.0))
        eff_gap_mom = np.where(thrust >= 0, 1.0 + raw['gap_momentum'].values, 1.0 / np.clip(1.0 + raw['gap_momentum'].values, 0.5, 2.0))
        eff_breakout = np.where(thrust >= 0, 1.0 + norm_breakout_pot, 1.0 / np.clip(1.0 + norm_breakout_pot, 0.5, 2.0))
        eff_turnover = np.where(thrust >= 0, 1.0 + norm_turnover_stab * 0.5, 1.0 / np.clip(1.0 + norm_turnover_stab * 0.5, 0.5, 2.0))
        cmf = np.clip(raw['cmf'].values, -1.0, 1.0)
        closing_str_norm = (raw['closing_str'].values - 50.0) / 50.0
        bbp = raw['bbp'].values
        macdh_norm = np.tanh((raw['macdh'].values / (raw['close'].values + 1e-9)) * 50.0)
        cmf_pos = np.maximum(0.0, cmf)
        closing_pos = np.maximum(0.0, closing_str_norm)
        bbp_pos = np.maximum(0.0, bbp - 0.5) * 2.0
        macdh_pos = np.maximum(0.0, macdh_norm)
        long_resonance = 1.0 + (cmf_pos * closing_pos * bbp_pos * macdh_pos * 5.0)
        cmf_neg = np.maximum(0.0, -cmf)
        closing_neg = np.maximum(0.0, -closing_str_norm)
        bbp_neg = np.maximum(0.0, 0.5 - bbp) * 2.0
        macdh_neg = np.maximum(0.0, -macdh_norm)
        short_resonance = 1.0 + (cmf_neg * closing_neg * bbp_neg * macdh_neg * 5.0)
        total_resonance = np.where(thrust >= 0, long_resonance, short_resonance)
        future_flow_premium = np.arcsinh(raw['exp_flow_1d'].values / (10000.0 * cap_factor + 1e-5))
        flow_geom_multiplier = np.exp(np.clip(future_flow_premium * np.sign(thrust) * 0.3, -2.0, 2.0))
        base_tensor = thrust * eff_structure * eff_gap_mom * eff_eco_premium * eff_breakout * eff_turnover * total_resonance * flow_geom_multiplier
        norm_lock_ratio = np.clip(raw['lock_ratio'].values / 100.0, 0.0, 1.0)
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool_flow * cap_discount / 50000.0, -20.0, 20.0))))
        raw_effective_drag = drag * (1.0 - np.maximum(0.0, np.minimum(hab_immunity, 0.90)))
        exp_arg = np.clip(-2.0 * (base_tensor - 1.5 * raw_effective_drag), -10.0, 10.0)
        squeeze_transition = 1.0 / (1.0 + np.exp(exp_arg))
        norm_game_intensity = np.clip(raw['game_intensity'].values, 0.0, 1.0)
        trap_reversal_factor = 1.0 + (raw['golden_pit'].values * 2.0)
        norm_energy = np.clip(raw['energy_conc'].values, 0.0, 1.0)
        absorption_norm = np.clip(raw['absorption_energy'].values / 100.0, 0.0, 1.0)
        squeeze_bonus = np.where(base_tensor >= 0, squeeze_transition * raw_effective_drag * (1.0 + np.abs(raw['emotional_extreme'].values)) * norm_game_intensity * trap_reversal_factor * norm_energy * (1.0 + absorption_norm * 2.0), 0.0)
        norm_reversal_prob = np.clip(raw['reversal_prob'].values / 100.0, 0.0, 1.0)
        final_drag = raw_effective_drag * (1.0 - squeeze_transition) * (1.0 - norm_reversal_prob) * (1.0 - norm_lock_ratio * 0.5)
        raw_intent = np.where(
            base_tensor >= 0,
            (base_tensor / np.maximum(1.0 + final_drag, 1.0)) + squeeze_bonus,
            base_tensor * (1.0 + np.sqrt(np.clip(final_drag, 0.0, 10000.0)))
        )
        t1_multiplier = np.exp(np.clip(np.tanh((raw['t1_premium'].values - 50.0) / 20.0), -1.0, 1.0))
        norm_compression = np.clip(raw['ma_compression'].values, 0.0, 1.0)
        hri = np.where(
            base_tensor >= 0,
            (base_tensor * (1.0 + squeeze_bonus)) / np.maximum(1.0 + final_drag, 1.0),
            base_tensor * (1.0 + np.sqrt(np.clip(final_drag, 0.0, 10000.0)))
        )
        hab_fuel = np.maximum(0.0, np.tanh(hab_pool_flow * cap_discount / 100000.0))
        hri_magnitude = np.abs(hri)
        hri_excess = np.clip(np.maximum(0.0, hri_magnitude - 3.0), 0.0, 10.0)
        exponent_gain = np.clip(hri_excess * t1_multiplier * (1.0 + norm_compression + hab_fuel), 0.0, 8.0)
        singularity_gain = 1.0 + np.expm1(np.clip(np.power(exponent_gain, 1.618), 0.0, 8.0))
        return np.clip(np.nan_to_num(raw_intent * singularity_gain, nan=0.0), -1e9, 1e9), long_resonance, short_resonance
    def _generate_probe_report(self, idx, raw, thrust, structure, drag, raw_intent, compressed_intent, z_scores, final, cap_discount, cap_factor, long_res, short_res, med_arr, mad_arr):
        locs = self._get_probe_locs(idx, compressed_intent)
        for i in locs:
            ts = idx[i]
            net_buy = raw['mf_net_buy'].values[i]
            hab_pool = (raw['hab_inv_13'].values[i] * 0.4 + raw['hab_inv_21'].values[i] * 0.3 + raw['hab_inv_34'].values[i] * 0.2 + raw['hab_inv_55'].values[i] * 0.1)
            circ_mv = raw['circ_mv'].values[i]
            cap_d = cap_discount[i]
            cap_f = cap_factor[i]
            hab_impact = np.arcsinh(np.where(np.abs(hab_pool) > 1.0, (net_buy * cap_d) / (np.abs(hab_pool) / 23.8 + 1e-5), 0.0))
            k_burst = 1.0 + ((self._kinematic_gate(np.array([raw['mf_slope_13'].values[i]]), np.array([500.0 * cap_f]), np.array([5000.0 * cap_f]))[0] * 0.3 + self._kinematic_gate(np.array([raw['mf_accel_13'].values[i]]), np.array([200.0 * cap_f]), np.array([2000.0 * cap_f]))[0] * 0.3 + self._kinematic_gate(np.array([raw['mf_jerk_13'].values[i]]), np.array([50.0 * cap_f]), np.array([500.0 * cap_f]))[0] * 0.4) * np.tanh(np.abs(net_buy) / (10000.0 * cap_f)))
            hab_imm = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool * cap_d / 50000.0, -10.0, 10.0))))
            eff_drag = drag[i] * (1.0 - hab_imm)
            net_energy_amp = 1.0 + np.tanh(raw['net_energy'].values[i] / 100.0)
            macdh_norm = np.tanh((raw['macdh'].values[i] / (raw['close'].values[i] + 1e-9)) * 50.0)
            chip_div_penalty = 1.0 + np.maximum(0.0, np.sinh(np.clip(raw['chip_divergence'].values[i] * 3.0, 0.0, 3.0)))
            report = [
                f"\n=== [PROBE V28.0.0] CalculateMainForceRallyIntent Full-Chain Audit (The Final Edition) @ {ts.strftime('%Y-%m-%d')} ===",
                f"【0. Raw Data Overview (底层核心数据快照)】",
                f"   [Thrust] MF_NetBuy: {net_buy:.2f} | Tick_Large_Net: {raw['tick_large_net'].values[i]:.2f} | ExpFlow1D: {raw['exp_flow_1d'].values[i]:.2f} | UptrendStr: {raw['uptrend_str'].values[i]:.2f}",
                f"   [Struct] Close: {raw['close'].values[i]:.2f} | Cost_Avg: {raw['cost_avg'].values[i]:.2f} | BehavAccum: {raw['behav_accum'].values[i]:.2f} | Control_Solidity: {raw['control_solidity'].values[i]:.4f}",
                f"   [Drag]   Profit_Pres: {raw['profit_pressure'].values[i]:.4f} | Div_Str: {raw['div_strength'].values[i]:.4f} | PF_Div: {raw['pf_div'].values[i]:.4f} | BehavDist: {raw['behav_dist'].values[i]:.4f} | FloatTurnover: {raw['turnover'].values[i]:.4f}",
                f"   [Eco]    Market_Sentiment: {raw['market_sentiment'].values[i]:.4f} | Fractal_Dim: {raw['fractal_dim'].values[i]:.4f} | RSI_13: {raw['rsi'].values[i]:.2f} | BIAS_21: {raw['bias_21'].values[i]:.2f}",
                f"---------------------------------------------------------------",
                f"【A. Kinematics (动态阈值软门限)】 Burst: x{k_burst:.4f} | Soft Gated Jerk: {self._kinematic_gate(np.array([raw['mf_jerk_13'].values[i]]), np.array([50.0 * cap_f]), np.array([500.0 * cap_f]))[0]:.4f}",
                f"【B. HAB (市值加权对数缓冲体系)】 CircMV: {circ_mv:.0f}万 (Cap Factor: {cap_f:.2f}, Discount: {cap_d:.2f}) | Fib Pool: {hab_pool:.0f} | Immunity: {hab_imm*100:.1f}% | Impact: {hab_impact:.4f}",
                f"【C. Ecosystem (脱敏乘数)】 MACDh_Norm: {macdh_norm:.4f} | NetEnergy_Amp: {net_energy_amp:.4f} | Chip_Div_Penalty: {chip_div_penalty:.4f}",
                f"【E. Synthesis (分形流形合成)】 Thrust: {thrust[i]:.4f} | Structure: {structure[i]:.4f} | EffectiveDrag: {eff_drag:.4f}",
                f"                              LongResonance: {long_res[i]:.4f} | ShortResonance: {short_res[i]:.4f}",
                f"【F. Result (最终)】 Raw Intent: {raw_intent[i]:.4f} | Compressed log1p: {compressed_intent[i]:.4f} | Z-Score: {z_scores[i]:.4f}",
                f"                  [Rolling Base] Med(144d): {med_arr[i]:.4f} | MAD(144d): {mad_arr[i]:.4f}",
                f"                  -> Final Normalized Score: {final[i]:.4f}",
                f"===============================================================\n"
            ]
            self._probe_cache.extend(report)
            for line in report: print(line)
    def _is_probe_enabled(self) -> bool:
        return True
















