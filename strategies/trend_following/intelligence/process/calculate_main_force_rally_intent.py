# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
# 版本: V34.1.0 · 真理之镜微观校准版 (Tick级筹码平衡平替 & QHOIM防爆) 已完成DeepThink
# 说明: 修正底层未注册信号 main_force_buy_ofi_D，平替为 tick_chip_balance_ratio_D，确保系统微观结构物理完整性。
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.trend_following.utils import get_param_value

class CalculateMainForceRallyIntent:
    """
    PROCESS_META_MAIN_FORCE_RALLY_INTENT
    【V34.1.0 · 真理之镜微观校准版 (Tick级筹码平衡平替 & QHOIM防爆)】
    重构说明：
    1. 平替底层缺失指标 main_force_buy_ofi_D，启用 tick_chip_balance_ratio_D 保持订单流失衡张量完整。
    2. 彻底斩断净额占比(net_amt_ratio)引发的 0值连乘死锁，引入 np.exp() 对数流形映射。
    3. 部署 fused_net_flow 融合资金阵型，打破主力净流单点失效带来的引擎熄火风险。
    4. 修复 QHOIM 量子谐振调制器极性闭环，利用 qho_match 相位匹配指数级放大极端趋势意图。
    5. 全域部署普朗克常量 1e-9 并消灭所有未来函数 .bfill()，打造绝对时间单向膜。
    """
    def __init__(self, strategy_instance, process_intelligence_helper_instance):
        self.strategy = strategy_instance
        self.helper = process_intelligence_helper_instance
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self._probe_cache = []
        self._probe_tensors = {}

    def _kinematic_gate(self, val: np.ndarray, threshold: np.ndarray, scale: np.ndarray, vol_factor: np.ndarray = 1.0) -> np.ndarray:
        val_clean = np.nan_to_num(val, nan=0.0)
        adj_threshold = threshold * vol_factor
        active_val = np.sign(val_clean) * np.maximum(0.0, np.abs(val_clean) - adj_threshold)
        return np.tanh(active_val / (scale + 1e-9))

    def _absolute_manifold_projection(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_scores = tensor / 4.0
        final_scores = 1.0 / (1.0 + np.exp(np.clip(-z_scores, -20.0, 20.0)))
        return final_scores, z_scores

    def _get_neutral_defaults(self) -> Dict[str, float]:
        """
        【V34.2.0 · 绝对零基防多头偏倚版】
        用途：为所有输入张量提供安全的缺失默认值。
        修改要点：
        1. 修复了“默认值假阳性污染”漏洞。将 chip_flow_int, breakout_chip 等强度类指标彻底剥离 
           50.0 假象池，强行移入 zero_keys，确保数据缺失时绝对死寂。
        2. 将 adx 的物理缺失默认值独立剥离定义为 20.0 (趋势分界底噪)。
        3. 显式列出所有微积分斜率的 zero_keys 键位，摒弃系统的隐式字典默认值。
        """
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
            'absorption_energy', 'flow_conf', 'tick_cluster', 'intra_game', 'resistance_str', 
            'flow_eff', 'breakout_qual', 'abs_change_str', 'flow_cluster'
        ]
        for k in score_keys: defaults[k] = 50.0
        ratio_keys = [
            'mf_activity', 'ind_downtrend', 'chip_convergence', 'control_solidity', 'chip_stability',
            'intra_acc_conf', 'intraday_dist', 'game_intensity', 'energy_conc', 'ma_compression',
            'vpa_adj', 'robust_trend', 'cost_range', 'geom_r2', 'tick_chip_bal'
        ]
        for k in ratio_keys: defaults[k] = 0.5
        zero_keys = [
            'ma_coherence', 'ma_tension', 'profit_pressure', 'trapped_pressure', 'intra_skew', 
            'buy_elg_rate', 'chip_divergence', 'gap_momentum', 'instability', 'pressure_release', 
            'pf_div', 'cmf', 'geom_slope', 'net_energy', 'macdh', 'parabolic_warn', 'bias_21', 
            'exp_flow_1d', 'div_strength', 'up_slope_13', 'up_accel_13', 'bias_slope_5', 
            'consol_duration', 'chip_rsi_div', 'flow_z', 'vwap_dev', 'roc_13', 'hf_skew', 'hf_kurt', 
            'vpa_accel', 'ema_angle_55', 'break_penalty', 'avg_net_13d', 'avg_net_21d', 
            'avg_net_34d', 'avg_net_55d', 'net_amt_ratio', 
            # V34.2.0 新增: 将强度类微观张量彻底封控于极寒防御区
            'chip_flow_int', 'breakout_chip',
            # 显式声明所有微积分衍生键位
            'tick_net_slope_13', 'tick_net_accel_13', 'tick_net_jerk_13',
            'pushing_slope_13', 'pushing_accel_13', 'pushing_jerk_13',
            'ctrl_slope_13', 'ctrl_accel_13', 'ctrl_jerk_13',
            'mf_slope_13', 'mf_accel_13', 'mf_jerk_13',
            'dist_slope_13', 'dist_accel_13', 'dist_jerk_13',
            'cmf_slope_13', 'cmf_accel_13', 'cfi_slope_13', 'tc_bal_slope_13',
            'adx_slope_13', 'adx_accel_13', 'conc_slope', 'winner_accel'
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
        defaults['bbw'] = 20.0
        defaults['fractal_dim'] = 1.5
        defaults['vol_burst'] = 1.0
        defaults['circ_mv'] = 500000.0
        defaults['price_range'] = 5.0
        defaults['price_entropy'] = 1.0
        defaults['adx'] = 20.0
        return defaults

    def _load_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        data = {}
        col_map = self._get_required_column_map()
        defaults = self._get_neutral_defaults()
        for key, col_name in col_map.items():
            series = self.helper._get_safe_series(df, col_name, np.nan)
            if key not in ['close', 'cost_avg', 'volume', 'vol_13d', 'vol_21d', 'vol_34d', 'vol_55d', 'mf_net_buy', 'hab_inv_13', 'hab_inv_21', 'hab_inv_34', 'hab_inv_55', 'circ_mv', 'exp_flow_1d', 'avg_net_13d', 'avg_net_21d', 'avg_net_34d', 'avg_net_55d']:
                series = series.ffill().fillna(defaults.get(key, 0.0))
            else:
                series = series.ffill().fillna(0.0)
            data[key] = series.astype(np.float32)
        if 'close' in data: data['close'] = data['close'].ffill().fillna(1.0)
        if 'cost_avg' in data and 'close' in data: data['cost_avg'] = data['cost_avg'].replace(0.0, np.nan).ffill().fillna(data['close'])
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
            'geom_r2': 'GEOM_REG_R2_D', 'vpa_adj': 'VPA_MF_ADJUSTED_EFF_D', 
            'cons_accum': 'consolidation_accumulation_score_D', 'cmf': 'CMF_21_D', 
            'bbp': 'BBP_21_2.0_D', 'bbw': 'BBW_21_2.0_D', 'closing_str': 'CLOSING_STRENGTH_D',
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
            'cost_range': 'main_cost_range_ratio_D', 'consol_duration': 'dynamic_consolidation_duration_D',
            'tick_cluster': 'tick_clustering_index_D', 'intra_game': 'intraday_chip_game_index_D',
            'chip_rsi_div': 'chip_rsi_divergence_D', 'flow_z': 'flow_zscore_D', 'flow_conf': 'flow_forecast_confidence_D',
            'vwap_dev': 'vwap_deviation_D', 'price_range': 'intraday_price_range_ratio_D', 'price_entropy': 'PRICE_ENTROPY_D', 'roc_13': 'ROC_13_D',
            'hf_skew': 'high_freq_flow_skewness_D', 'hf_kurt': 'high_freq_flow_kurtosis_D', 'vpa_accel': 'VPA_ACCELERATION_13D',
            'ema_angle_55': 'ATAN_ANGLE_EMA_55_D', 'break_penalty': 'breakout_penalty_score_D', 'breakout_qual': 'breakout_quality_score_D',
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
            'shakeout_score': 'shakeout_score_D', 'dist_slope_13': 'SLOPE_13_distribution_score_D', 
            'dist_accel_13': 'ACCEL_13_distribution_score_D', 'dist_jerk_13': 'JERK_13_distribution_score_D',
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
            'ma_compression': 'MA_POTENTIAL_COMPRESSION_RATE_D', 'turnover_stability': 'TURNOVER_STABILITY_INDEX_D',
            'resistance_str': 'resistance_strength_D', 'flow_eff': 'flow_efficiency_D',
            'chip_flow_int': 'chip_flow_intensity_D', 'abs_change_str': 'absolute_change_strength_D',
            'flow_cluster': 'flow_cluster_intensity_D', 'breakout_chip': 'breakout_chip_score_D',
            'net_amt_ratio': 'net_amount_ratio_D', 'adx': 'ADX_14_D', 'tick_chip_bal': 'tick_chip_balance_ratio_D',
            'avg_net_13d': 'avg_daily_net_13d_D', 'avg_net_21d': 'avg_daily_net_21d_D',
            'avg_net_34d': 'avg_daily_net_34d_D', 'avg_net_55d': 'avg_daily_net_55d_D',
            'cfi_slope_13': 'SLOPE_13_chip_flow_intensity_D', 'adx_slope_13': 'SLOPE_13_ADX_14_D',
            'adx_accel_13': 'ACCEL_13_ADX_14_D', 'tc_bal_slope_13': 'SLOPE_13_tick_chip_balance_ratio_D'
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
        self._probe_tensors = {}
        raw = self._load_data(df)
        idx = df.index
        count = len(idx)
        if count < 5: return pd.Series(0.0, index=idx)
        self._probe_cache_raw = raw
        self._probe_cache_idx = idx
        hab_pool_flow = (raw['hab_inv_13'].values * 0.4 + raw['hab_inv_21'].values * 0.3 + raw['hab_inv_34'].values * 0.2 + raw['hab_inv_55'].values * 0.1)
        hab_pool_vol = (raw['vol_13d'].values * 0.4 + raw['vol_21d'].values * 0.3 + raw['vol_34d'].values * 0.2 + raw['vol_55d'].values * 0.1)
        hab_pool_net_avg = (raw['avg_net_13d'].values * 0.4 + raw['avg_net_21d'].values * 0.3 + raw['avg_net_34d'].values * 0.2 + raw['avg_net_55d'].values * 0.1)
        circ_mv_yi = np.maximum(raw['circ_mv'].values / 10000.0, 1.0)
        cap_discount = np.clip(np.log1p(1000.0) / np.log1p(circ_mv_yi), 0.5, 3.0)
        cap_factor = np.clip(circ_mv_yi / 100.0, 0.1, 10.0)
        hab_vol_impact = np.arcsinh(np.where(np.abs(hab_pool_vol) > 1.0, (raw['volume'].values * cap_discount) / (np.abs(hab_pool_vol) / 23.8 + 1e-9), 0.0))
        volatility_factor = 1.0 + np.tanh(np.maximum(0.0, raw['instability'].values) / 100.0)
        self._probe_tensors['hab_pool_flow'] = hab_pool_flow
        self._probe_tensors['hab_pool_net_avg'] = hab_pool_net_avg
        self._probe_tensors['cap_discount'] = cap_discount
        self._probe_tensors['cap_factor'] = cap_factor
        self._probe_tensors['vol_factor'] = volatility_factor
        self._probe_tensors['hab_vol_impact'] = hab_vol_impact
        thrust = self._calc_thrust_component(raw, idx)
        structure = self._calc_structure_component(raw, idx)
        drag = self._calc_drag_component(raw, idx)
        raw_intent = self._calc_tensor_synthesis(thrust, structure, drag, raw, idx)
        raw_intent_clean = np.clip(np.nan_to_num(raw_intent, nan=0.0), -1e9, 1e9)
        compressed_intent = np.sign(raw_intent_clean) * np.log1p(np.abs(raw_intent_clean))
        final_scores, z_scores = self._absolute_manifold_projection(compressed_intent)
        if self._is_probe_enabled(): 
            self._generate_probe_report(idx, raw, thrust, structure, drag, raw_intent_clean, compressed_intent, z_scores, final_scores)
        return pd.Series(final_scores, index=idx, dtype=np.float32)

    def _calc_thrust_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V34.2.0 · 动能极化清洗版】
        用途：计算主力多头/空头的初始物理推力。
        修改要点：
        1. 简化了 fused_net_flow 的代数开销。
        2. 修正 tick_chip_bal 比例极化强度：利用 abs(x - 0.5) * 2 将默认 0.5 的“死水状态”物理清洗为 0.0 推力，彻底阻断了平盘期的中性伪动能注入。
        """
        hab_pool_flow = self._probe_tensors['hab_pool_flow']
        hab_pool_net_avg = self._probe_tensors['hab_pool_net_avg']
        hab_vol_impact = self._probe_tensors['hab_vol_impact']
        cap_discount = self._probe_tensors['cap_discount']
        cap_factor = self._probe_tensors['cap_factor']
        vol_factor = self._probe_tensors['vol_factor']
        mf_net_buy = raw['mf_net_buy'].values
        tick_large_net = raw['tick_large_net'].values
        hm_synergy = raw['hm_synergy'].values
        net_energy = raw['net_energy'].values
        vol_burst = raw['vol_burst'].values
        cmf = raw['cmf'].values
        exp_flow = raw['exp_flow_1d'].values
        hf_skew = np.clip(raw['hf_skew'].values, -10.0, 10.0)
        hf_kurt = np.clip(raw['hf_kurt'].values, -5.0, 20.0)
        k_slope = self._kinematic_gate(raw['mf_slope_13'].values, np.array([500.0]) * cap_factor, np.array([5000.0]) * cap_factor, vol_factor)
        k_accel = self._kinematic_gate(raw['mf_accel_13'].values, np.array([200.0]) * cap_factor, np.array([2000.0]) * cap_factor, vol_factor)
        k_jerk = self._kinematic_gate(raw['mf_jerk_13'].values, np.array([50.0]) * cap_factor, np.array([500.0]) * cap_factor, vol_factor)
        tick_net_slope = self._kinematic_gate(raw['tick_net_slope_13'].values, np.array([50.0]) * cap_factor, np.array([2500.0]) * cap_factor, vol_factor)
        tick_net_accel = self._kinematic_gate(raw['tick_net_accel_13'].values, np.array([25.0]) * cap_factor, np.array([1250.0]) * cap_factor, vol_factor)
        tick_net_jerk = self._kinematic_gate(raw['tick_net_jerk_13'].values, np.array([5.0]) * cap_factor, np.array([250.0]) * cap_factor, vol_factor)
        push_slope = self._kinematic_gate(raw['pushing_slope_13'].values, np.array([0.5]), np.array([5.0]), vol_factor)
        push_accel = self._kinematic_gate(raw['pushing_accel_13'].values, np.array([0.25]), np.array([2.5]), vol_factor)
        push_jerk = self._kinematic_gate(raw['pushing_jerk_13'].values, np.array([0.1]), np.array([1.0]), vol_factor)
        cmf_slope = self._kinematic_gate(raw['cmf_slope_13'].values, np.array([0.02]), np.array([0.2]), vol_factor)
        cmf_accel = self._kinematic_gate(raw['cmf_accel_13'].values, np.array([0.01]), np.array([0.1]), vol_factor)
        cfi_slope = self._kinematic_gate(raw['cfi_slope_13'].values, np.array([0.5]), np.array([5.0]), vol_factor)
        tc_bal_slope = self._kinematic_gate(raw['tc_bal_slope_13'].values, np.array([0.05]), np.array([0.5]), vol_factor)
        # 优化点：直接标量相加，消除 np.sign(x)*np.abs(x) 冗余运算
        fused_net_flow = mf_net_buy + tick_large_net * 0.5 + exp_flow * 0.2
        hab_flow_impact = np.arcsinh(np.where(np.abs(hab_pool_flow) > 1.0, (fused_net_flow * cap_discount) / (np.abs(hab_pool_flow) / 23.8 + 1e-9), 0.0))
        net_shock_intensity = np.tanh(fused_net_flow / (np.abs(hab_pool_net_avg) + 1e-9))
        self._probe_tensors['hab_flow_impact'] = hab_flow_impact
        synergy_multiplier = 1.0 + np.maximum(0.0, hm_synergy / 100.0)
        energy_match = np.sign(fused_net_flow) * np.sign(net_energy)
        net_energy_amp = 1.0 + np.abs(np.tanh(net_energy / 100.0)) * np.where(energy_match > 0, 1.0, -0.5)
        # 优化点：提取极化失衡强度，防止在 0.5 中性默认值时注入伪动能
        tc_bal_intensity = np.abs(np.clip(raw['tick_chip_bal'].values, 0.0, 1.0) - 0.5) * 2.0
        micro_structure_bonus = 1.0 + np.clip(raw['chip_flow_int'].values / 100.0, 0.0, 1.0) * np.maximum(0.0, cfi_slope) + tc_bal_intensity * np.maximum(0.0, tc_bal_slope)
        macro_base = fused_net_flow * net_energy_amp * synergy_multiplier * (1.0 + np.tanh(np.abs(hab_flow_impact))) * (1.0 + np.abs(net_shock_intensity) * 0.5) * micro_structure_bonus
        norm_macro_base = np.sign(macro_base) * np.log1p(np.abs(macro_base) / (1000.0 * cap_factor + 1e-9))
        macro_damping = np.tanh(np.abs(fused_net_flow) / (10000.0 * cap_factor + 1e-9))
        tick_damping = np.tanh(np.abs(tick_large_net) / (5000.0 * cap_factor + 1e-9))
        push_damping = np.clip((raw['pushing_score'].values - 50.0) / 50.0, -1.0, 1.0)
        macro_kinematics = (k_slope + k_accel + k_jerk) * macro_damping
        tick_kinematics = (tick_net_slope + tick_net_accel + tick_net_jerk) * tick_damping
        push_kinematics = (push_slope + push_accel + push_jerk) * np.maximum(0.0, push_damping)
        cmf_kinematics = (cmf_slope + cmf_accel) * np.clip(cmf, -1.0, 1.0)
        coupling_field = np.tanh(macro_kinematics + tick_kinematics + push_kinematics + cmf_kinematics)
        kine_match = np.sign(norm_macro_base) * np.sign(coupling_field)
        kinematic_multiplier = np.maximum(1e-9, 1.0 + np.abs(coupling_field) * np.where(kine_match > 0, 1.0, -0.5))
        purity_multiplier = 1.0 + np.maximum(0.0, np.tanh(raw['buy_elg_rate'].values / 20.0))
        acc_conf_norm = np.clip((raw['intra_acc_conf'].values - 0.5) * 2.0, -1.0, 1.0)
        acc_confidence_multiplier = 1.0 + np.maximum(0.0, acc_conf_norm)
        norm_uptrend = np.clip(raw['uptrend_str'].values / 100.0, 0.0, 1.0)
        flow_z_norm = np.tanh(raw['flow_z'].values / 3.0)
        push_momentum = np.clip((raw['pushing_score'].values - 50.0) / 100.0, -0.5, 0.5)
        base_momentum = norm_macro_base + flow_z_norm * 0.5 + push_momentum
        macro_momentum = base_momentum * purity_multiplier * acc_confidence_multiplier * (1.0 + np.abs(coupling_field) * 0.5) * (1.0 + norm_uptrend)
        persistence_factor = np.tanh(raw['flow_persistence'].values / 120.0)
        tick_intensity = np.clip(tick_large_net / (np.abs(fused_net_flow) + 1e-9), -50.0, 50.0)
        tick_cluster_boost = 1.0 + np.clip((raw['tick_cluster'].values - 50.0) / 50.0, -0.5, 1.0)
        flow_cluster_boost = 1.0 + np.clip((raw['flow_cluster'].values - 50.0) / 50.0, 0.0, 1.0)
        abs_change_boost = 1.0 + np.clip((raw['abs_change_str'].values - 50.0) / 100.0, 0.0, 1.0)
        detonation_boost = (1.0 + np.tanh(np.maximum(0.0, raw['tick_abnormal_vol'].values - 1.0)) + np.clip(vol_burst - 1.0, 0.0, 5.0) * 0.2 * np.tanh(np.abs(hab_vol_impact)) + np.maximum(0.0, hf_kurt) * 0.05) * tick_cluster_boost * flow_cluster_boost * abs_change_boost
        norm_flow_consistency = np.clip(raw['flow_consistency'].values / 100.0, 0.0, 1.0)
        energy_dissipation = np.maximum(1e-9, 1.0 - norm_flow_consistency)
        aligned_jet = np.clip(raw['intra_accel'].values / 10.0, -5.0, 5.0) + tick_intensity + hf_skew * 0.2
        micro_jet_raw = aligned_jet * (raw['pushing_score'].values / 100.0) * np.maximum(1e-9, raw['mf_activity'].values) * persistence_factor * detonation_boost / energy_dissipation
        jet_exponent = np.tanh(micro_jet_raw) * (raw['breakout_flow'].values / 50.0) * np.maximum(1e-9, norm_flow_consistency)
        jet_match = np.sign(macro_momentum) * np.sign(micro_jet_raw)
        micro_multiplier = np.exp(np.abs(np.clip(jet_exponent, -2.0, 2.0)) * np.where(jet_match > 0, 1.0, -0.5))
        closing_amplifier = 1.0 + np.maximum(0.0, np.tanh((raw['closing_intensity'].values - 50.0) / 50.0))
        sentiment_amplifier = 1.0 + np.maximum(0.0, np.clip((raw['market_sentiment'].values - 50.0) / 50.0, -1.0, 1.0))
        industry_resonance = 1.0 + np.maximum(0.0, np.clip((raw['industry_markup'].values - 50.0) / 50.0, -1.0, 1.0))
        net_amt_ratio_mod = np.exp(np.clip(raw['net_amt_ratio'].values, -2.0, 2.0) * 0.5)
        w_macro = 1.0 / (1.0 + np.exp(-np.clip(macro_momentum, -10.0, 10.0) * 2.0))
        phase_alignment = w_macro * np.where(raw['industry_markup'].values > 50, 1.2, 1.0) + (1.0 - w_macro) * np.where(raw['industry_markup'].values <= 50, 1.2, 1.0)
        base_final_thrust = macro_momentum * micro_multiplier * kinematic_multiplier * sentiment_amplifier * closing_amplifier * industry_resonance * phase_alignment * net_amt_ratio_mod
        excess_kine = np.maximum(0.0, kinematic_multiplier - 1.0)
        excess_jet = np.maximum(0.0, micro_multiplier - 1.0)
        excess_beta = np.maximum(0.0, industry_resonance - 1.0)
        critical_resonance_index = ((excess_kine + excess_jet + excess_beta) / 3.0) * acc_confidence_multiplier * (1.0 + np.tanh(np.abs(hab_vol_impact)))
        nonlinear_gain = 1.0 + np.expm1(np.clip(np.power(np.maximum(0.0, critical_resonance_index), 1.618), 0.0, 5.0))
        self._probe_tensors['kine_mult'] = kinematic_multiplier
        self._probe_tensors['micro_mult'] = micro_multiplier
        return np.clip(np.nan_to_num(base_final_thrust * nonlinear_gain, nan=0.0), -1000.0, 1000.0)

    def _calc_structure_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        hab_vol_impact = self._probe_tensors['hab_vol_impact']
        vol_factor = self._probe_tensors['vol_factor']
        cost_avg_safe = np.maximum(raw['cost_avg'].values, 1e-9)
        cost_gap = (raw['close'].values - cost_avg_safe) / cost_avg_safe
        cost_rbf = np.exp(np.clip(-10.0 * (cost_gap - 0.05)**2, -20.0, 20.0))
        entropy_raw = np.maximum(0.01, raw['chip_entropy'].values)
        norm_intra_consolidation = 1.0 / (1.0 + np.exp(np.clip(-0.05 * (raw['intra_consolidation'].values - 50.0), -20.0, 20.0)))
        stability_raw = np.clip(raw['chip_stability'].values, 0.0, 1.0) + norm_intra_consolidation * 0.5
        entropy_penalty = np.maximum(1e-9, entropy_raw / (1.0 + stability_raw))
        price_entropy_penalty = np.exp(-np.maximum(0.0, raw['price_entropy'].values - 1.0))
        norm_coherence = 1.0 / (1.0 + np.exp(np.clip(-2.0 * raw['ma_coherence'].values, -10.0, 10.0)))
        lattice_orderliness = (stability_raw * np.maximum(1e-9, norm_coherence) * price_entropy_penalty) / entropy_penalty
        norm_convergence = np.clip(raw['chip_convergence'].values, 0.0, 1.0)
        convergence_factor = 1.0 + norm_convergence
        cost_range = np.clip(raw['cost_range'].values, 0.01, 1.0)
        cost_compression = np.exp(-5.0 * np.maximum(0.0, cost_range - 0.15))
        norm_tension = np.tanh(np.maximum(0.0, raw['ma_tension'].values) / 2.0)
        bbw_norm = np.clip(raw['bbw'].values / 50.0, 0.01, 2.0)
        bbw_squeeze = 1.0 / np.sqrt(bbw_norm + 1e-9)
        elastic_compression = np.maximum(0.0, norm_tension * norm_convergence) * (1.0 + cost_compression) * bbw_squeeze
        norm_peak_conc = np.clip((raw['peak_conc'].values * 0.5 + raw['vol_adj_conc'].values * 0.5) / 100.0, 0.0, 1.0)
        norm_breakout_chip = np.clip(raw['breakout_chip'].values / 100.0, 0.0, 1.0)
        peak_efficiency = np.maximum(0.01, norm_peak_conc) * np.maximum(0.01, np.clip(raw['winner_rate'].values / 100.0, 0.0, 1.0)) * convergence_factor * (1.0 + norm_breakout_chip)
        norm_control = np.clip(raw['control_solidity'].values, 0.0, 1.0)
        control_factor = 1.0 + norm_control * 0.5
        norm_acc = np.clip(raw['accumulation_score'].values / 100.0, 0.0, 1.0)
        norm_cons_accum = np.clip(raw['cons_accum'].values / 100.0, 0.0, 1.0)
        norm_behav_accum = np.clip(raw['behav_accum'].values / 100.0, 0.0, 1.0)
        acc_factor = 1.0 + norm_acc + norm_cons_accum * 0.5 + norm_behav_accum * 0.5
        angle_norm = np.clip(raw['ema_angle_55'].values / 45.0, -1.0, 1.0)
        adx_norm = np.clip(raw['adx'].values / 100.0, 0.0, 1.0)
        adx_slope = self._kinematic_gate(raw['adx_slope_13'].values, np.array([1.0]), np.array([10.0]), vol_factor)
        trend_bonus = 1.0 + np.clip(raw['uptrend_str'].values / 100.0, 0.0, 1.0) * 0.5 + np.maximum(0.0, angle_norm) * 0.5 + adx_norm * 0.5 + np.maximum(0.0, adx_slope) * 0.3
        vol_shrinkage_bonus = 1.0 + np.maximum(0.0, -hab_vol_impact * 0.5)
        static_lattice_energy = lattice_orderliness * peak_efficiency * cost_rbf * control_factor * acc_factor * trend_bonus * vol_shrinkage_bonus * 2.0
        inertia_bonus = 1.0 + np.maximum(0.0, (raw['hab_structure'].values - 0.6) * 1.5)
        ctrl_damping = np.maximum(1e-9, norm_control)
        raw_ctrl_kine = self._kinematic_gate(raw['ctrl_slope_13'].values, np.array([0.01]), np.array([0.1]), vol_factor) + self._kinematic_gate(raw['ctrl_accel_13'].values, np.array([0.005]), np.array([0.05]), vol_factor) + self._kinematic_gate(raw['ctrl_jerk_13'].values, np.array([0.001]), np.array([0.01]), vol_factor)
        effective_ctrl_kine = raw_ctrl_kine * ctrl_damping
        k_conc_slope = np.tanh(self._kinematic_gate(raw['conc_slope'].values, np.array([0.5]), np.array([5.0]), vol_factor) * 2.0)
        k_winner_accel = np.tanh(self._kinematic_gate(raw['winner_accel'].values, np.array([1.0]), np.array([10.0]), vol_factor) * 1.5)
        kine_vector = (k_conc_slope * 0.2) + (k_winner_accel * 0.15) + (np.tanh(effective_ctrl_kine) * 0.35)
        geom_multiplier = 1.0 + np.tanh(raw['geom_slope'].values) * 0.5
        turnover_int_boost = 1.0 + np.clip(raw['turnover_intensity'].values / 100.0, 0.0, 1.0) * 0.5
        up_slope = self._kinematic_gate(raw['up_slope_13'].values, np.array([1.0]), np.array([10.0]), vol_factor)
        up_accel = self._kinematic_gate(raw['up_accel_13'].values, np.array([0.5]), np.array([5.0]), vol_factor)
        trend_kinematics = np.tanh(up_slope + up_accel) * 0.5
        evolution_kinematics = np.maximum(0.1, 1.0 + kine_vector * (1.0 + elastic_compression * 2.0) * geom_multiplier * turnover_int_boost + trend_kinematics)
        norm_consolidation_conc = np.clip(raw['consolidation_chip_conc'].values / 100.0, 0.0, 1.0)
        consolidation_boost = 1.0 + norm_consolidation_conc
        consol_duration = np.clip(raw['consol_duration'].values, 0.0, 250.0)
        duration_bonus = np.log1p(consol_duration) / 4.0
        norm_platform = np.clip(raw['platform_quality'].values / 100.0, 0.0, 1.0)
        platform_factor = np.maximum(0.1, 1.0 + norm_platform * 0.6 * consolidation_boost * (1.0 + duration_bonus))
        sr_factor = np.exp(np.clip(np.tanh(raw['sr_ratio'].values - 1.0), -5.0, 5.0))
        norm_foundation = 1.0 / (1.0 + np.exp(np.clip(-0.1 * (raw['foundation_strength'].values - 50.0), -20.0, 20.0)))
        foundation_factor = 1.0 + norm_foundation * 0.4 * sr_factor
        pattern_bonus = np.maximum(0.1, 1.0 + (raw['rounding_bottom'].values * 0.3))
        base_structure = static_lattice_energy * inertia_bonus * evolution_kinematics * platform_factor * foundation_factor * pattern_bonus
        sri = (lattice_orderliness * norm_platform * raw['hab_structure'].values * np.maximum(1e-9, norm_tension))
        excitation_gain = 1.0 + np.expm1(np.clip(np.maximum(0.0, sri - 0.5), 0.0, 3.0)) * 1.5
        resonance_core = base_structure * excitation_gain
        excess_res = np.clip(np.maximum(0.0, resonance_core - 1.5), 0.0, 5.0)
        avalanche_gain = 1.0 + np.power(np.maximum(0.0, excess_res), 1.618) * 1.5
        return np.clip(np.nan_to_num(resonance_core * avalanche_gain, nan=1.0), 0.01, 1000.0)

    def _calc_drag_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V34.3.0 · 绝对奇点固化版】阻尼引力张量场
        修改要点：将相对论换手率的极限阈值提升至 0.999，强化天量震荡期的洛伦兹引力阻挡。
        """
        hab_pool_flow = self._probe_tensors['hab_pool_flow']
        cap_discount = self._probe_tensors['cap_discount']
        vol_factor = self._probe_tensors['vol_factor']
        dist_damping = np.clip((raw['dist_score'].values - 50.0) / 50.0, 0.0, 1.0)
        dist_slope_13 = self._kinematic_gate(raw['dist_slope_13'].values, np.array([1.0]), np.array([10.0]), vol_factor)
        dist_accel_13 = self._kinematic_gate(raw['dist_accel_13'].values, np.array([0.5]), np.array([5.0]), vol_factor)
        dist_jerk_13 = self._kinematic_gate(raw['dist_jerk_13'].values, np.array([0.2]), np.array([2.0]), vol_factor)
        vpa_accel_penalty = np.maximum(0.0, -np.tanh(self._kinematic_gate(raw['vpa_accel'].values, np.array([1.0]), np.array([10.0]), vol_factor)))
        raw_dist_kine = dist_slope_13 + dist_accel_13 + dist_jerk_13
        effective_dist_kine = raw_dist_kine * dist_damping
        kine_multiplier = 1.0 + np.maximum(0.0, np.tanh(effective_dist_kine) * 1.5) + vpa_accel_penalty
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool_flow * cap_discount / 50000.0, -20.0, 20.0))))
        hab_immunity = np.clip(hab_immunity, 0.0, 0.9)
        hab_drag_penalty = 1.0 + np.tanh(np.maximum(0.0, -hab_pool_flow * cap_discount) / 50000.0)
        norm_profit_pressure = np.expm1(np.clip(np.maximum(0.0, raw['profit_pressure'].values) / 50.0, 0.0, 5.0))
        norm_trapped_pressure = np.expm1(np.clip(np.maximum(0.0, raw['trapped_pressure'].values), 0.0, 1.0)) * 1.5
        norm_dist = np.clip((raw['dist_score'].values - 50.0) / 50.0, 0.0, 1.0)
        norm_intra_dist = np.clip((raw['intraday_dist'].values - 0.5) * 2.0, 0.0, 1.0)
        norm_instability = np.tanh(np.maximum(0.0, raw['instability'].values) / 100.0)
        angle_norm = np.clip(raw['ema_angle_55'].values / 45.0, -1.0, 1.0)
        macro_downtrend = np.maximum(0.0, -angle_norm)
        norm_downtrend = np.clip((raw['downtrend_str'].values - 50.0) / 50.0, 0.0, 1.0) + macro_downtrend * 0.5
        dump_quality_factor = 1.0 + np.clip((raw['outflow_qual'].values - 50.0) / 50.0, 0.0, 1.0) * 1.5
        energy_factor = 1.0 + np.clip((raw['dist_energy'].values - 50.0) / 50.0, 0.0, 1.0)
        norm_behav_dist = np.clip((raw['behav_dist'].values - 50.0) / 50.0, 0.0, 1.0)
        pf_div = np.clip(raw['pf_div'].values / 20.0, 0.0, 5.0)
        chip_div = np.clip(raw['chip_divergence'].values / 20.0, 0.0, 5.0)
        div_str = np.clip(raw['div_strength'].values / 20.0, 0.0, 5.0)
        chip_rsi_div = np.clip(raw['chip_rsi_div'].values / 20.0, 0.0, 5.0)
        break_pen = np.clip(raw['break_penalty'].values / 20.0, 0.0, 5.0)
        div_vector_l2 = np.sqrt(pf_div**2 + chip_div**2 + div_str**2 + chip_rsi_div**2 + break_pen**2)
        unified_div_penalty = 1.0 + np.sinh(np.clip(div_vector_l2, 0.0, 3.0))
        self._probe_tensors['uni_div'] = unified_div_penalty
        coupled_active_dump = (norm_dist + norm_intra_dist * 0.5) * dump_quality_factor * energy_factor * kine_multiplier * unified_div_penalty * (1.0 + norm_behav_dist)
        beta_headwind = 1.0 + np.clip((raw['ind_downtrend'].values - 0.5) * 2.0, 0.0, 1.0)
        flow_eff_penalty = np.cosh(np.clip((50.0 - raw['flow_eff'].values) / 25.0, 0.0, 2.0))
        friction_vpa = 1.0 + np.maximum(0.0, 1.0 - np.clip(raw['vpa_adj'].values / 100.0, -1.0, 1.0)) * flow_eff_penalty
        skew_penalty = 1.0 + np.maximum(0.0, -raw['intra_skew'].values) * 0.5
        range_penalty = 1.0 + np.clip(raw['price_range'].values / 10.0, 0.0, 2.0)
        vwap_dev_penalty = 1.0 + np.sinh(np.clip(np.abs(raw['vwap_dev'].values) / 2.0, 0.0, 3.0))
        coupled_viscosity = (1.0 + norm_instability) * beta_headwind * friction_vpa * skew_penalty * range_penalty * vwap_dev_penalty
        robust_trend_decay = 1.0 / (1.0 + np.clip(raw['robust_trend'].values, 0.0, 1.0) * 0.5)
        bias_val = raw['bias_21'].values
        bias_slope = self._kinematic_gate(raw['bias_slope_5'].values, np.array([0.5]), np.array([5.0]), vol_factor)
        bias_overbought = np.sinh(np.clip(np.maximum(0.0, bias_val) / 10.0, 0.0, 3.0))
        bias_oversold = np.sinh(np.clip(np.maximum(0.0, -bias_val) / 10.0, 0.0, 3.0))
        rsi_val = raw['rsi'].values
        rsi_overbought = np.maximum(0.0, rsi_val - 70.0) / 30.0
        rsi_oversold = np.maximum(0.0, 30.0 - rsi_val) / 30.0
        gravity_amplifier = (1.0 + rsi_overbought * 2.0 + bias_overbought) / (1.0 + rsi_oversold * 2.0 + bias_oversold)
        intra_support = np.clip(raw['intra_support'].values / 100.0, 0.0, 1.0)
        norm_resistance = 1.0 + np.clip((raw['resistance_str'].values - 50.0) / 50.0, 0.0, 1.0)
        coupled_gravity = (norm_profit_pressure + norm_trapped_pressure) * (1.0 + norm_downtrend) * robust_trend_decay * gravity_amplifier * (1.0 - intra_support * 0.5) * norm_resistance
        norm_release = np.clip(raw['pressure_release'].values, 0.0, 1.0)
        norm_shakeout = np.clip((raw['shakeout_score'].values - 50.0) / 50.0, 0.0, 1.0)
        relief_valve = np.maximum(0.1, 1.0 + norm_release * 1.5 + norm_shakeout * 1.0)
        hf_hidden_div = np.clip((raw['hf_flow_div'].values - 50.0) / 25.0, 0.0, 2.0)
        adj_turnover_threshold = 0.05 * cap_discount
        turnover_excess = np.maximum(0.0, (raw['turnover'].values / 100.0) - adj_turnover_threshold)
        # V34.3.0: 洛伦兹物理墙极化，将换手率速度壁垒极限提升至 0.999
        norm_turnover_rel = np.clip(raw['turnover'].values / 100.0, 0.0, 0.999)
        gamma_relativistic = 1.0 / np.sqrt(1.0 - norm_turnover_rel**2 + 1e-9)
        turnover_drag = np.expm1(np.clip(turnover_excess, 0.0, 5.0) * 10.0) * 0.5 * gamma_relativistic
        parabolic_penalty = 1.0 + np.clip(raw['parabolic_warn'].values, 0.0, 1.0) * 2.0
        core_drag_raw = np.clip(((coupled_gravity + coupled_active_dump) * coupled_viscosity * hab_drag_penalty) / relief_valve, 0.0, 1000.0)
        core_drag_shielded = np.clip(core_drag_raw * (1.0 - hab_immunity) + turnover_drag + hf_hidden_div, 0.0, 1000.0) * parabolic_penalty
        excess_drag = np.clip(np.maximum(0.0, core_drag_shielded - 1.5), 0.0, 10.0)
        avalanche_gain = 1.0 + np.power(np.maximum(0.0, excess_drag), 1.618) * 1.5
        final_drag = np.clip(np.nan_to_num(core_drag_shielded * avalanche_gain, nan=0.0), 0.0, 10000.0)
        return final_drag

    def _calc_tensor_synthesis(self, thrust: np.ndarray, structure: np.ndarray, drag: np.ndarray, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V34.3.0 · 绝对奇点固化版】拉格朗日-量子谐振子 (QHOIM) 意图引力投射
        修改要点：剥离冗余防爆代码 `np.maximum(..., 1e-9)`。因 `final_drag` 必然 >= 0，分母已自然绝对安全。代码洁癖至高体现。
        """
        hab_pool_flow = self._probe_tensors['hab_pool_flow']
        cap_discount = self._probe_tensors['cap_discount']
        cap_factor = self._probe_tensors['cap_factor']
        norm_theme = np.clip((raw['theme_hotness'].values - 50.0) / 50.0, 0.0, 1.0)
        norm_breakout_pot = (np.clip((raw['breakout_pot'].values - 50.0) / 50.0, 0.0, 1.0) + np.clip((raw['breakout_qual'].values - 50.0) / 50.0, 0.0, 1.0)) / 2.0
        norm_turnover_stab = np.clip((raw['turnover_stability'].values - 50.0) / 50.0, 0.0, 1.0)
        eco_premium = 1.0 + (raw['is_leader'].values * 0.8) + (raw['hm_top_tier'].values * 0.6) + (raw['breakout_conf'].values * 0.4) + (norm_theme * 0.3) + (np.clip((raw['trend_confirm'].values - 50.0) / 50.0, 0.0, 1.0) * 0.5)
        fractal_efficiency = np.exp(np.clip(1.5 - raw['fractal_dim'].values, -0.5, 0.5) * 2.0)
        w_thrust = 1.0 / (1.0 + np.exp(-np.clip(thrust, -10.0, 10.0) * 2.0))
        eff_structure = w_thrust * (np.sqrt(np.clip(structure, 0.01, 100.0)) * fractal_efficiency) + (1.0 - w_thrust) * (1.0 / np.clip(np.sqrt(np.clip(structure, 0.01, 100.0)) * fractal_efficiency, 0.2, 5.0))
        eff_eco_premium = w_thrust * eco_premium + (1.0 - w_thrust) * (1.0 / np.clip(eco_premium, 0.5, 2.0))
        eff_gap_mom = w_thrust * (1.0 + raw['gap_momentum'].values) + (1.0 - w_thrust) * (1.0 / np.clip(1.0 + raw['gap_momentum'].values, 0.5, 2.0))
        eff_breakout = w_thrust * (1.0 + norm_breakout_pot) + (1.0 - w_thrust) * (1.0 / np.clip(1.0 + norm_breakout_pot, 0.5, 2.0))
        eff_turnover = w_thrust * (1.0 + norm_turnover_stab * 0.5) + (1.0 - w_thrust) * (1.0 / np.clip(1.0 + norm_turnover_stab * 0.5, 0.5, 2.0))
        cmf = np.clip(raw['cmf'].values, -1.0, 1.0)
        closing_str_norm = np.clip((raw['closing_str'].values - 50.0) / 50.0, -1.0, 1.0)
        bbp = raw['bbp'].values
        roc_norm = np.clip(np.tanh(raw['roc_13'].values / 10.0), -1.0, 1.0)
        geom_r2_norm = np.clip(raw['geom_r2'].values, 0.0, 1.0)
        vwap_norm = np.clip(np.tanh(raw['vwap_dev'].values / 2.0), -1.0, 1.0)
        cmf_pos = np.clip(cmf, 0.0, 1.0)
        closing_pos = np.clip(closing_str_norm, 0.0, 1.0)
        bbp_pos = np.clip((bbp - 0.5) * 2.0, 0.0, 1.0)
        roc_pos = np.clip(roc_norm, 0.0, 1.0)
        vwap_pos = np.clip(vwap_norm, 0.0, 1.0)
        long_align = cmf_pos * 0.25 + closing_pos * 0.15 + bbp_pos * 0.15 + roc_pos * 0.15 + geom_r2_norm * 0.15 + vwap_pos * 0.15
        long_resonance = 1.0 + np.expm1(long_align * 2.5)
        cmf_neg = np.clip(-cmf, 0.0, 1.0)
        closing_neg = np.clip(-closing_str_norm, 0.0, 1.0)
        bbp_neg = np.clip((0.5 - bbp) * 2.0, 0.0, 1.0)
        roc_neg = np.clip(-roc_norm, 0.0, 1.0)
        vwap_neg = np.clip(-vwap_norm, 0.0, 1.0)
        short_align = cmf_neg * 0.25 + closing_neg * 0.15 + bbp_neg * 0.15 + roc_neg * 0.15 + geom_r2_norm * 0.15 + vwap_neg * 0.15
        short_resonance = 1.0 + np.expm1(short_align * 2.5)
        total_resonance = long_resonance * w_thrust + short_resonance * (1.0 - w_thrust)
        future_flow_premium = np.arcsinh(raw['exp_flow_1d'].values / (10000.0 * cap_factor + 1e-9)) * np.clip(raw['flow_conf'].values / 100.0, 0.0, 1.0)
        flow_geom_multiplier = np.exp(np.clip(future_flow_premium * np.sign(thrust) * 0.3, -2.0, 2.0))
        base_tensor = thrust * eff_structure * eff_gap_mom * eff_eco_premium * eff_breakout * eff_turnover * total_resonance * flow_geom_multiplier
        kinetic_energy = np.sign(base_tensor) * np.square(np.clip(base_tensor, -100.0, 100.0))
        potential_energy = np.square(np.clip(drag, 0.0, 100.0)) * eff_turnover
        lagrangian_action = kinetic_energy - potential_energy
        hamiltonian_energy = np.abs(kinetic_energy) + potential_energy
        qho_modulator = lagrangian_action / np.maximum(1.0, np.log1p(hamiltonian_energy))
        qho_match = np.sign(base_tensor) * np.sign(qho_modulator)
        qho_multiplier = np.exp(np.clip(qho_match * np.abs(qho_modulator) / 50.0, -2.0, 2.0))
        base_tensor = base_tensor * qho_multiplier
        norm_lock_ratio = np.clip(raw['lock_ratio'].values / 100.0, 0.0, 1.0)
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool_flow * cap_discount / 50000.0, -20.0, 20.0))))
        raw_effective_drag = drag * (1.0 - np.maximum(0.0, np.minimum(hab_immunity, 0.90)))
        exp_arg = np.clip(-2.0 * (base_tensor - 1.5 * raw_effective_drag), -50.0, 50.0)
        squeeze_transition = 1.0 / (1.0 + np.exp(exp_arg))
        norm_game_intensity = np.clip((raw['game_intensity'].values - 0.5) * 2.0, 0.0, 1.0) * 0.5 + np.clip((raw['intra_game'].values - 50.0) / 50.0, 0.0, 1.0) * 0.5
        trap_reversal_factor = 1.0 + (raw['golden_pit'].values * 2.0)
        norm_energy = np.clip((raw['energy_conc'].values - 0.5) * 2.0, 0.0, 1.0)
        absorption_norm = np.clip((raw['absorption_energy'].values - 50.0) / 50.0, 0.0, 1.0)
        w_base = 1.0 / (1.0 + np.exp(-np.clip(base_tensor, -10.0, 10.0) * 2.0))
        squeeze_bonus = w_base * squeeze_transition * raw_effective_drag * (1.0 + np.abs(raw['emotional_extreme'].values)) * norm_game_intensity * trap_reversal_factor * norm_energy * (1.0 + absorption_norm * 2.0)
        norm_reversal_prob = np.clip((raw['reversal_prob'].values - 50.0) / 50.0, 0.0, 1.0)
        final_drag = raw_effective_drag * (1.0 - squeeze_transition) * (1.0 - norm_reversal_prob) * (1.0 - norm_lock_ratio * 0.5)
        # V34.3.0 优化：移除分母防零冗余补丁，因为 1.0 + final_drag 必然 >= 1.0
        intent_pos = (base_tensor / (1.0 + final_drag)) + squeeze_bonus
        intent_neg = base_tensor * (1.0 + np.sqrt(np.clip(final_drag, 0.0, 10000.0)))
        raw_intent = intent_pos * w_base + intent_neg * (1.0 - w_base)
        t1_multiplier = np.exp(np.clip(np.tanh((raw['t1_premium'].values - 50.0) / 20.0), -1.0, 1.0))
        norm_compression = np.clip((raw['ma_compression'].values - 0.5) * 2.0, 0.0, 1.0)
        hri_pos = (base_tensor * (1.0 + squeeze_bonus)) / (1.0 + final_drag)
        hri_neg = base_tensor * (1.0 + np.sqrt(np.clip(final_drag, 0.0, 10000.0)))
        hri = hri_pos * w_base + hri_neg * (1.0 - w_base)
        hab_fuel = np.maximum(0.0, np.tanh(hab_pool_flow * cap_discount / 100000.0))
        hri_magnitude = np.abs(hri)
        hri_excess = np.clip(np.maximum(0.0, hri_magnitude - 3.0), 0.0, 10.0)
        exponent_gain = np.clip(hri_excess * t1_multiplier * (1.0 + norm_compression + hab_fuel), 0.0, 8.0)
        singularity_gain = 1.0 + np.expm1(np.clip(np.power(exponent_gain, 1.618), 0.0, 8.0))
        self._probe_tensors['long_align'] = long_align
        self._probe_tensors['short_align'] = short_align
        self._probe_tensors['total_res'] = total_resonance
        self._probe_tensors['w_thrust'] = w_thrust
        self._probe_tensors['qho_modulator'] = qho_modulator
        return np.clip(np.nan_to_num(raw_intent * singularity_gain, nan=0.0), -1e9, 1e9)

    def _generate_probe_report(self, idx, raw, thrust, structure, drag, raw_intent, compressed_intent, z_scores, final):
        """
        用途: 输出全息探针审计日志。
        修改要点: 升级探针表头版本号至 V34.3.0，标志引擎正式进入 The Absolute Singularity 态。
        """
        locs = self._get_probe_locs(idx, compressed_intent)
        kine_mult = self._probe_tensors['kine_mult']
        micro_mult = self._probe_tensors['micro_mult']
        uni_div = self._probe_tensors['uni_div']
        long_align = self._probe_tensors['long_align']
        short_align = self._probe_tensors['short_align']
        total_res = self._probe_tensors['total_res']
        w_thrust = self._probe_tensors['w_thrust']
        hab_pool_flow = self._probe_tensors['hab_pool_flow']
        cap_discount = self._probe_tensors['cap_discount']
        cap_factor = self._probe_tensors['cap_factor']
        vol_factor = self._probe_tensors['vol_factor']
        hab_flow_impact = self._probe_tensors['hab_flow_impact']
        qho_mod = self._probe_tensors['qho_modulator']
        for i in locs:
            ts = idx[i]
            net_buy = raw['mf_net_buy'].values[i]
            hab_pool = hab_pool_flow[i]
            circ_mv = raw['circ_mv'].values[i]
            cap_d = cap_discount[i]
            cap_f = cap_factor[i]
            v_fac = vol_factor[i]
            hab_impact = hab_flow_impact[i]
            k_burst = kine_mult[i]
            
            hab_imm = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool * cap_d / 50000.0, -10.0, 10.0))))
            eff_drag = drag[i] * (1.0 - hab_imm)
            
            energy_match = np.sign(net_buy) * np.sign(raw['net_energy'].values[i])
            net_energy_amp = 1.0 + np.abs(np.tanh(raw['net_energy'].values[i] / 100.0)) * (1.0 if energy_match > 0 else -0.5)
            roc_norm = np.clip(np.tanh(raw['roc_13'].values[i] / 10.0), -1.0, 1.0)
            
            report = [
                f"\n=== [PROBE V34.3.0] CalculateMainForceRallyIntent Full-Chain Audit (The Absolute Singularity) @ {ts.strftime('%Y-%m-%d')} ===",
                f"【0. Raw Data Overview (底层核心数据快照)】",
                f"   [Thrust] MF_NetBuy: {net_buy:.2f} | Tick_Large_Net: {raw['tick_large_net'].values[i]:.2f} | Tick_Balance: {raw['tick_chip_bal'].values[i]:.4f} | Net_Amt_Ratio: {raw['net_amt_ratio'].values[i]:.4f}",
                f"   [Struct] Close: {raw['close'].values[i]:.2f} | BehavAccum: {raw['behav_accum'].values[i]:.2f} | Control_Solidity: {raw['control_solidity'].values[i]:.4f} | ADX_14: {raw['adx'].values[i]:.2f}",
                f"   [Drag]   Profit_Pres: {raw['profit_pressure'].values[i]:.4f} | Div_Str: {raw['div_strength'].values[i]:.4f} | PF_Div: {raw['pf_div'].values[i]:.4f} | Break_Pen: {raw['break_penalty'].values[i]:.4f}",
                f"   [Eco]    Market_Sentiment: {raw['market_sentiment'].values[i]:.4f} | RSI_13: {raw['rsi'].values[i]:.2f} | BIAS_21: {raw['bias_21'].values[i]:.2f} | BBW_21: {raw['bbw'].values[i]:.2f}",
                f"---------------------------------------------------------------",
                f"【A. Kinematics (动态硬截断门限)】 Burst Kine: x{k_burst:.4f} | Micro Mult: x{micro_mult[i]:.4f} | VolFactor: {v_fac:.2f}",
                f"【B. HAB (市值加权对数缓冲体系)】 CircMV: {circ_mv:.0f}万 (Cap Factor: {cap_f:.2f}) | Fib Pool: {hab_pool:.0f} | Immunity: {hab_imm*100:.1f}% | Impact: {hab_impact:.4f}",
                f"【C. Ecosystem (脱敏与相空间惩罚)】 ROC13_Norm: {roc_norm:.4f} | NetEnergy_Amp: {net_energy_amp:.4f} | L2_UnifiedDiv: {uni_div[i]:.4f}",
                f"【E. Synthesis (量子谐振子能量场)】 Thrust(K): {thrust[i]:.4f} | Structure: {structure[i]:.4f} | EffectiveDrag(U): {eff_drag:.4f} | QHO_Mod: {qho_mod[i]:.4f}",
                f"                              LongAlign: {long_align[i]:.4f} | ShortAlign: {short_align[i]:.4f} | Total Resonance: {total_res[i]:.4f} (W_Thrust: {w_thrust[i]:.4f})",
                f"【F. Result (绝对流形投射基准)】 Raw Intent: {raw_intent[i]:.4f} | Compressed log1p: {compressed_intent[i]:.4f} | Z-Score: {z_scores[i]:.4f}",
                f"                           -> Final Normalized Score: {final[i]:.4f}",
                f"===============================================================\n"
            ]
            self._probe_cache.extend(report)
            for line in report: print(line)

    def _is_probe_enabled(self) -> bool:
        return False