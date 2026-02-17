# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
# 版本: V19.0 · 物理量纲统整与对数拓扑压缩版
# 说明: 1. 彻底修复变量断层引发的 NameError；2. 大规模修正底层因子的物理量纲映射 ([0,1]与[0,100]分制错位)；3. 引入 np.log1p 对数压缩层，解决抛压/拉升奇点畸变导致的 Z-Score 映射失效问题。
import pandas as pd
import numpy as np
import numba
from typing import Dict, List, Any
from strategies.trend_following.utils import get_params_block, get_param_value

class CalculateMainForceRallyIntent:
    """
    PROCESS_META_MAIN_FORCE_RALLY_INTENT
    【V19.0 · 物理域全息对数压缩版】
    基于 A 股“资金-结构-效率”三维物理场。
    核心方程：RallyIntent = (Kinetics * Control) / (1 + Resistance)
    """
    def __init__(self, strategy_instance, process_intelligence_helper_instance):
        self.strategy = strategy_instance
        self.helper = process_intelligence_helper_instance
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self._probe_cache = []

    def _get_neutral_defaults(self) -> Dict[str, float]:
        """
        【V19.0 · 绝对均衡态注入与量纲纠偏】
        针对不同因子的真实物理边界，精细化定义绝对中性填充值，杜绝量纲错位。
        """
        defaults = {k: 0.0 for k in self._get_required_column_map().keys()}
        score_keys = [
            'pushing_score', 'winner_rate', 'peak_conc', 'accumulation_score', 
            'platform_quality', 'dist_score', 'instability', 'shakeout_score', 
            'theme_hotness', 'lock_ratio', 'consolidation_chip_conc', 
            'downtrend_str', 't1_premium', 'industry_markup', 'breakout_flow', 
            'flow_consistency', 'closing_intensity', 'hf_flow_div', 
            'dist_energy', 'outflow_qual', 'reversal_prob'
        ]
        for k in score_keys:
            defaults[k] = 50.0
        small_keys = [
            'mf_activity', 'intra_consolidation', 'foundation_strength',
            'ind_downtrend', 'chip_convergence', 'control_solidity', 'chip_stability',
            'intra_acc_conf', 'intraday_dist', 'game_intensity', 'energy_conc', 
            'ma_compression', 'pressure_release', 'trapped_pressure', 
            'turnover_stability', 'breakout_pot', 'trend_confirm'
        ]
        for k in small_keys:
            defaults[k] = 0.5
        zero_keys = [
            'ma_coherence', 'ma_tension', 'profit_pressure', 'intra_skew', 
            'buy_elg_rate', 'chip_divergence', 'gap_momentum'
        ]
        for k in zero_keys:
            defaults[k] = 0.0
        defaults['market_sentiment'] = 5.0
        defaults['hab_structure'] = 0.6
        defaults['tick_abnormal_vol'] = 1.0
        defaults['sr_ratio'] = 1.0
        defaults['chip_entropy'] = 1.0
        defaults['vpa_efficiency'] = 0.0
        defaults['turnover'] = 5.0
        defaults['close'] = 1.0
        defaults['cost_avg'] = 1.0
        return defaults

    def _load_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V19.0 · 物理中立基底注入版】
        取代暴力的 fillna(0.0)，赋予缺失特征平滑的隐身态，消除 NaN 级联破坏。
        """
        data = {}
        col_map = self._get_required_column_map()
        defaults = self._get_neutral_defaults()
        for key, col_name in col_map.items():
            series = self.helper._get_safe_series(df, col_name, np.nan)
            if key not in ['close', 'cost_avg']:
                series = series.fillna(defaults.get(key, 0.0))
            data[key] = series.astype(np.float32)
        if 'close' in data:
            data['close'] = data['close'].ffill().bfill().fillna(1.0)
        if 'cost_avg' in data and 'close' in data:
            data['cost_avg'] = data['cost_avg'].replace(0.0, np.nan).fillna(data['close'])
        return data

    def _get_required_column_map(self) -> Dict[str, str]:
        return {
            'close': 'close_D',
            'cost_avg': 'cost_50pct_D',
            'mf_net_buy': 'net_mf_amount_D',
            'hab_inventory': 'total_net_amount_21d_D',
            'mf_slope_13': 'SLOPE_13_net_mf_amount_D',
            'mf_accel_13': 'ACCEL_13_net_mf_amount_D',
            'mf_jerk_13': 'JERK_13_net_mf_amount_D',
            'hm_synergy': 'HM_COORDINATED_ATTACK_D',
            'pushing_score': 'pushing_score_D',
            'market_sentiment': 'market_sentiment_score_D',
            'tick_large_net': 'tick_large_order_net_D',
            'intra_accel': 'flow_acceleration_intraday_D',
            'breakout_flow': 'breakout_fundflow_score_D',
            'mf_activity': 'intraday_main_force_activity_D',
            'energy_conc': 'energy_concentration_D',
            'winner_rate': 'winner_rate_D',
            'control_solidity': 'consolidation_chip_stability_D',
            'chip_entropy': 'chip_entropy_D',
            'chip_stability': 'chip_stability_D',
            'peak_conc': 'peak_concentration_D',
            'accumulation_score': 'accumulation_signal_score_D',
            'ma_coherence': 'MA_COHERENCE_RESONANCE_D',
            'hab_structure': 'long_term_chip_ratio_D',
            'conc_slope': 'SLOPE_5_peak_concentration_D',
            'winner_accel': 'ACCEL_5_winner_rate_D',
            'platform_quality': 'consolidation_quality_score_D',
            'foundation_strength': 'support_strength_D',
            'vpa_efficiency': 'VPA_EFFICIENCY_D',
            'profit_pressure': 'profit_pressure_D',
            'turnover': 'turnover_rate_D',
            'trapped_pressure': 'pressure_trapped_D',
            'dist_score': 'distribution_score_D',
            'intraday_dist': 'intraday_distribution_confidence_D',
            'instability': 'flow_volatility_21d_D',
            'pressure_release': 'pressure_release_index_D',
            'shakeout_score': 'shakeout_score_D',
            'chip_divergence': 'chip_divergence_ratio_D',
            'dist_slope': 'SLOPE_5_distribution_score_D',
            'dist_accel': 'ACCEL_5_distribution_score_D',
            'dist_jerk': 'JERK_5_distribution_score_D',
            'gap_momentum': 'GAP_MOMENTUM_STRENGTH_D',
            'emotional_extreme': 'STATE_EMOTIONAL_EXTREME_D',
            'reversal_prob': 'reversal_prob_D',
            'is_leader': 'STATE_MARKET_LEADER_D',
            'theme_hotness': 'THEME_HOTNESS_SCORE_D',
            'lock_ratio': 'high_position_lock_ratio_90_D',
            'trend_confirm': 'trend_confirmation_score_D',
            'flow_21d': 'total_net_amount_21d_D',
            'flow_55d': 'total_net_amount_55d_D',
            'buy_elg_rate': 'buy_elg_amount_rate_D',
            'flow_consistency': 'flow_consistency_D',
            'flow_persistence': 'flow_persistence_minutes_D',
            'closing_intensity': 'closing_flow_intensity_D',
            'industry_markup': 'industry_markup_score_D',
            'tick_abnormal_vol': 'tick_abnormal_volume_ratio_D',
            'intra_acc_conf': 'intraday_accumulation_confidence_D',
            'tick_net_slope_13': 'SLOPE_13_tick_large_order_net_D',
            'tick_net_accel_13': 'ACCEL_13_tick_large_order_net_D',
            'tick_net_jerk_13': 'JERK_13_tick_large_order_net_D',
            'pushing_slope_13': 'SLOPE_13_pushing_score_D',
            'pushing_accel_13': 'ACCEL_13_pushing_score_D',
            'pushing_jerk_13': 'JERK_13_pushing_score_D',
            'chip_convergence': 'chip_convergence_ratio_D',
            'intra_consolidation': 'intraday_chip_consolidation_degree_D',
            'ma_tension': 'MA_POTENTIAL_TENSION_INDEX_D',
            'consolidation_chip_conc': 'consolidation_chip_concentration_D',
            'rounding_bottom': 'STATE_ROUNDING_BOTTOM_D',
            'sr_ratio': 'support_resistance_ratio_D',
            'ctrl_slope_13': 'SLOPE_13_consolidation_chip_stability_D',
            'ctrl_accel_13': 'ACCEL_13_consolidation_chip_stability_D',
            'ctrl_jerk_13': 'JERK_13_consolidation_chip_stability_D',
            'outflow_qual': 'outflow_quality_D',
            'intra_skew': 'intraday_price_distribution_skewness_D',
            'ind_downtrend': 'industry_downtrend_score_D',
            'downtrend_str': 'downtrend_strength_D',
            'dist_energy': 'distribution_energy_D',
            'hf_flow_div': 'high_freq_flow_divergence_D',
            'dist_slope_13': 'SLOPE_13_distribution_score_D',
            'dist_accel_13': 'ACCEL_13_distribution_score_D',
            'dist_jerk_13': 'JERK_13_distribution_score_D',
            'game_intensity': 'game_intensity_D',
            'golden_pit': 'STATE_GOLDEN_PIT_D',
            'breakout_conf': 'STATE_BREAKOUT_CONFIRMED_D',
            'hm_top_tier': 'HM_ACTIVE_TOP_TIER_D',
            't1_premium': 'T1_PREMIUM_EXPECTATION_D',
            'breakout_pot': 'breakout_potential_D',
            'ma_compression': 'MA_POTENTIAL_COMPRESSION_RATE_D',
            'turnover_stability': 'TURNOVER_STABILITY_INDEX_D'
        }

    def _get_probe_locs(self, idx: pd.Index, target_tensor: np.ndarray = None) -> List[int]:
        """【V19.0 · 全息探针寻址器】自动捕获异常张量节点。"""
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
        """
        【V19.0 · 对数拓扑压缩版执行器】
        引入非线性压缩层 np.log1p(abs(x)) 将指数级膨胀的张量引力坍缩回高斯平原，
        彻底解决因长尾效应导致的 Median Absolute Deviation (MAD) 极化失效问题。
        """
        self._probe_cache = []
        raw = self._load_data(df)
        idx = df.index
        count = len(idx)
        if count < 5:
            return pd.Series(0.0, index=idx)
        self._probe_cache_raw = raw
        self._probe_cache_idx = idx
        thrust = self._calc_thrust_component(raw, idx)
        structure = self._calc_structure_component(raw, idx)
        drag = self._calc_drag_component(raw, idx)
        raw_intent = self._calc_tensor_synthesis(thrust, structure, drag, raw, idx)
        raw_intent_clean = np.clip(np.nan_to_num(raw_intent, nan=0.0), -1e6, 1e6)
        compressed_intent = np.sign(raw_intent_clean) * np.log1p(np.abs(raw_intent_clean))
        valid_mask = np.abs(compressed_intent) > 1e-4
        valid_intent = compressed_intent[valid_mask]
        if len(valid_intent) > 5:
            med = np.median(valid_intent)
            mad = np.median(np.abs(valid_intent - med))
        else:
            med = np.median(compressed_intent)
            mad = np.median(np.abs(compressed_intent - med))
        robust_mad = np.maximum(mad, 0.5)
        z_scores = (compressed_intent - med) / (robust_mad * 3.0)
        final_scores = 1.0 / (1.0 + np.exp(np.clip(-z_scores, -20.0, 20.0)))
        if self._is_probe_enabled():
            print(f"[PROBE-STAT-V19.0] Compressed Intent | Valid(>0): {len(valid_intent)} | Median: {med:.4f} | Raw MAD: {mad:.8f} | Robust MAD: {robust_mad:.4f}")
            self._generate_probe_report(idx, raw, thrust, structure, drag, raw_intent_clean, compressed_intent, z_scores, final_scores)
        return pd.Series(final_scores, index=idx, dtype=np.float32)

    def _calc_thrust_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V19.0 · 推力微观对齐纠偏版】
        修复微观射流方向“负负得正”导致的方向倒挂。
        """
        mf_net_buy = raw['mf_net_buy'].values
        hm_synergy = raw['hm_synergy'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        tick_large_net = raw['tick_large_net'].values
        intra_accel = raw['intra_accel'].values
        breakout_flow = raw['breakout_flow'].values
        pushing_score = raw['pushing_score'].values
        sentiment = raw['market_sentiment'].values
        mf_activity = raw['mf_activity'].values
        k_slope = raw['mf_slope_13'].values
        k_accel = raw['mf_accel_13'].values
        k_jerk = raw['mf_jerk_13'].values
        buy_elg_rate = raw['buy_elg_rate'].values
        flow_consistency = raw['flow_consistency'].values
        flow_persistence = raw['flow_persistence'].values
        closing_intensity = raw['closing_intensity'].values
        industry_markup = raw['industry_markup'].values
        tick_abnormal_vol = raw['tick_abnormal_vol'].values
        intra_acc_conf = raw['intra_acc_conf'].values
        tick_net_slope = raw['tick_net_slope_13'].values
        tick_net_accel = raw['tick_net_accel_13'].values
        tick_net_jerk = raw['tick_net_jerk_13'].values
        push_slope = raw['pushing_slope_13'].values
        push_accel = raw['pushing_accel_13'].values
        push_jerk = raw['pushing_jerk_13'].values
        hab_total_pool = (flow_21d * 0.6) + (flow_55d * 0.4)
        hab_cushion = np.where((mf_net_buy < 0) & (hab_total_pool > 0), np.clip(hab_total_pool / (np.abs(mf_net_buy) + 1e-9), 0.0, 1.0) * np.abs(mf_net_buy) * 0.8, 0.0)
        effective_net_buy = mf_net_buy + hab_cushion
        synergy_multiplier = 1.0 + np.maximum(0.0, hm_synergy / 100.0)
        macro_base = effective_net_buy * synergy_multiplier
        norm_macro_base = np.sign(macro_base) * np.log1p(np.abs(macro_base) / 1000.0)
        macro_damping = np.tanh(np.abs(effective_net_buy) / 10000.0)
        tick_damping = np.tanh(np.abs(tick_large_net) / 5000.0)
        push_damping = np.clip((pushing_score - 50.0) / 50.0, -1.0, 1.0)
        macro_kinematics = (k_slope + k_accel + k_jerk) * macro_damping
        tick_kinematics = (tick_net_slope + tick_net_accel + tick_net_jerk) * tick_damping
        push_kinematics = (push_slope + push_accel + push_jerk) * np.maximum(0.0, push_damping)
        coupling_field = np.tanh(macro_kinematics + tick_kinematics + push_kinematics)
        kinematic_multiplier = 1.0 + np.maximum(0.0, coupling_field)
        purity_multiplier = 1.0 + np.maximum(0.0, np.tanh(buy_elg_rate / 20.0))
        acc_conf_norm = np.clip((intra_acc_conf - 0.5) * 2.0, -1.0, 1.0)
        acc_confidence_multiplier = 1.0 + np.maximum(0.0, acc_conf_norm)
        macro_momentum = norm_macro_base * purity_multiplier * acc_confidence_multiplier * (1.0 + coupling_field * 0.5)
        persistence_factor = np.tanh(flow_persistence / 120.0)
        tick_intensity = np.clip(tick_large_net / (np.abs(effective_net_buy) + 1e-9), -50.0, 50.0)
        detonation_boost = 1.0 + np.tanh(np.maximum(0.0, tick_abnormal_vol - 1.0))
        norm_flow_consistency = np.clip(flow_consistency / 100.0, 0.0, 1.0)
        energy_dissipation = np.maximum(0.01, 1.0 - norm_flow_consistency)
        micro_jet_raw = (np.clip(intra_accel / 10.0, -5.0, 5.0) + tick_intensity) * (pushing_score / 100.0) * np.maximum(0.1, mf_activity) * persistence_factor * detonation_boost / energy_dissipation
        jet_exponent = np.tanh(micro_jet_raw) * (breakout_flow / 50.0) * np.maximum(0.1, norm_flow_consistency)
        micro_multiplier = np.exp(np.clip(jet_exponent, -2.0, 2.0))
        closing_amplifier = 1.0 + np.maximum(0.0, np.tanh((closing_intensity - 50.0) / 50.0))
        sentiment_amplifier = 1.0 + np.maximum(0.0, np.clip((sentiment - 5.0) / 5.0, -1.0, 1.0))
        industry_resonance = 1.0 + np.maximum(0.0, np.clip((industry_markup - 50.0) / 50.0, -1.0, 1.0))
        phase_alignment = np.where((macro_momentum > 0) & (industry_markup > 50), 1.2, 0.8)
        base_final_thrust = macro_momentum * micro_multiplier * kinematic_multiplier * sentiment_amplifier * closing_amplifier * industry_resonance * phase_alignment
        excess_kine = np.maximum(0.0, kinematic_multiplier - 1.0)
        excess_jet = np.maximum(0.0, micro_multiplier - 1.0)
        excess_beta = np.maximum(0.0, industry_resonance - 1.0)
        critical_resonance_index = excess_kine * excess_jet * excess_beta * acc_confidence_multiplier
        nonlinear_gain = 1.0 + np.expm1(np.clip(critical_resonance_index * 2.5, 0.0, 5.0))
        ultimate_thrust = np.clip(np.nan_to_num(base_final_thrust * nonlinear_gain, nan=0.0), -1000.0, 1000.0)
        return ultimate_thrust

    def _calc_structure_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V19.0 · 晶格相变量纲归一修复版】
        重设底层结构因子的自然物理界限激活映射；彻底修复 `ctrl_slope_13` 变量断层错误。
        """
        cost_avg = raw['cost_avg'].values
        close = raw['close'].values
        chip_entropy = raw['chip_entropy'].values
        chip_stability = raw['chip_stability'].values
        intra_consolidation = raw['intra_consolidation'].values
        ma_coherence = raw['ma_coherence'].values
        chip_convergence = raw['chip_convergence'].values
        ma_tension = raw['ma_tension'].values
        peak_conc = raw['peak_conc'].values
        winner_rate = raw['winner_rate'].values
        control_solidity = raw['control_solidity'].values
        accumulation_score = raw['accumulation_score'].values
        hab_structure = raw['hab_structure'].values
        platform_quality = raw['platform_quality'].values
        foundation_strength = raw['foundation_strength'].values
        conc_slope = raw['conc_slope'].values
        winner_accel = raw['winner_accel'].values
        consolidation_chip_conc = raw['consolidation_chip_conc'].values
        rounding_bottom = raw['rounding_bottom'].values
        sr_ratio = raw['sr_ratio'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        ctrl_slope_13 = raw['ctrl_slope_13'].values
        ctrl_accel_13 = raw['ctrl_accel_13'].values
        ctrl_jerk_13 = raw['ctrl_jerk_13'].values
        cost_gap = (close - cost_avg) / (cost_avg + 1e-9)
        cost_rbf = np.exp(np.clip(-10.0 * (cost_gap - 0.05)**2, -20.0, 20.0))
        entropy_raw = np.maximum(0.01, chip_entropy)
        norm_intra_consolidation = np.clip(intra_consolidation, 0.0, 1.0)
        stability_raw = np.clip(chip_stability, 0.0, 1.0) + norm_intra_consolidation * 0.5
        entropy_penalty = np.maximum(1e-4, entropy_raw / (1.0 + stability_raw))
        norm_coherence = 1.0 / (1.0 + np.exp(np.clip(-2.0 * ma_coherence, -10.0, 10.0)))
        lattice_orderliness = (stability_raw * np.maximum(0.1, norm_coherence)) / entropy_penalty
        norm_convergence = np.clip(chip_convergence, 0.0, 1.0)
        convergence_factor = 1.0 + norm_convergence
        norm_tension = np.tanh(np.maximum(0.0, ma_tension) / 2.0)
        elastic_compression = np.maximum(0.0, norm_tension * norm_convergence)
        norm_peak_conc = np.clip(peak_conc / 100.0, 0.0, 1.0)
        peak_efficiency = norm_peak_conc * np.clip(winner_rate / 100.0, 0.0, 1.0) * convergence_factor
        norm_control = np.clip(control_solidity, 0.0, 1.0)
        control_factor = 1.0 + norm_control * 0.5
        norm_acc = np.clip(accumulation_score / 100.0, 0.0, 1.0)
        acc_factor = 1.0 + norm_acc
        static_lattice_energy = lattice_orderliness * peak_efficiency * cost_rbf * control_factor * acc_factor * 5.0
        inertia_bonus = 1.0 + np.maximum(0.0, (hab_structure - 0.6) * 1.5)
        hab_pool = flow_21d * 0.618 + flow_55d * 0.382
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool / 50000.0, -20.0, 20.0))))
        hab_immunity = np.maximum(0.0, np.minimum(hab_immunity, 0.85))
        ctrl_damping = np.maximum(0.1, norm_control)
        raw_ctrl_kine = ctrl_slope_13 + ctrl_accel_13 + ctrl_jerk_13
        protected_ctrl_kine = np.where(raw_ctrl_kine < 0, raw_ctrl_kine * (1.0 - hab_immunity), raw_ctrl_kine)
        effective_ctrl_kine = protected_ctrl_kine * ctrl_damping
        k_conc_slope = np.tanh(conc_slope * 2.0)
        k_winner_accel = np.tanh(winner_accel * 1.5)
        kine_vector = (k_conc_slope * 0.2) + (k_winner_accel * 0.15) + (np.tanh(effective_ctrl_kine) * 0.35)
        evolution_kinematics = 1.0 + kine_vector * (1.0 + elastic_compression * 2.0)
        norm_consolidation_conc = np.clip(consolidation_chip_conc / 100.0, 0.0, 1.0)
        consolidation_boost = 1.0 + norm_consolidation_conc
        norm_platform = np.clip(platform_quality / 100.0, 0.0, 1.0)
        platform_factor = 1.0 + norm_platform * 0.6 * consolidation_boost
        sr_factor = np.exp(np.clip(np.tanh(sr_ratio - 1.0), -5.0, 5.0))
        norm_foundation = np.clip(foundation_strength, 0.0, 1.0)
        foundation_factor = 1.0 + norm_foundation * 0.4 * sr_factor
        pattern_bonus = 1.0 + (rounding_bottom * 0.3)
        base_structure = static_lattice_energy * inertia_bonus * evolution_kinematics * platform_factor * foundation_factor * pattern_bonus
        sri = (lattice_orderliness * norm_platform * hab_structure * np.maximum(0.1, norm_tension))
        excitation_gain = 1.0 + np.expm1(np.clip(np.maximum(0.0, sri - 0.5), 0.0, 3.0)) * 1.5
        resonance_core = base_structure * excitation_gain
        avalanche_threshold = 1.5
        excess_res = np.clip(np.maximum(0.0, resonance_core - avalanche_threshold), 0.0, 5.0)
        avalanche_gain = 1.0 + (excess_res ** 2) * 1.5
        final_structure = np.clip(np.nan_to_num(resonance_core * avalanche_gain, nan=1.0, posinf=100.0, neginf=0.01), 0.01, 100.0)
        return final_structure

    def _calc_drag_component(self, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V19.0 · 阻力引力场平滑限幅版】
        """
        profit_pressure = raw['profit_pressure'].values
        trapped_pressure = raw['trapped_pressure'].values
        dist_score = raw['dist_score'].values
        intraday_dist = raw['intraday_dist'].values
        instability = raw['instability'].values
        vpa_efficiency = raw['vpa_efficiency'].values
        turnover_rate = raw['turnover'].values
        pressure_release = raw['pressure_release'].values
        shakeout_score = raw['shakeout_score'].values
        outflow_qual = raw['outflow_qual'].values
        intra_skew = raw['intra_skew'].values
        ind_downtrend = raw['ind_downtrend'].values
        downtrend_str = raw['downtrend_str'].values
        dist_energy = raw['dist_energy'].values
        hf_flow_div = raw['hf_flow_div'].values
        dist_slope_13 = raw['dist_slope_13'].values
        dist_accel_13 = raw['dist_accel_13'].values
        dist_jerk_13 = raw['dist_jerk_13'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        dist_damping = np.clip(dist_score / 100.0, 0.0, 1.0)
        raw_dist_kine = dist_slope_13 + dist_accel_13 + dist_jerk_13
        effective_dist_kine = raw_dist_kine * dist_damping
        kine_multiplier = 1.0 + np.maximum(0.0, np.tanh(effective_dist_kine) * 1.5)
        hab_pool = flow_21d * 0.618 + flow_55d * 0.382
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(hab_pool / 50000.0, -20.0, 20.0))))
        hab_immunity = np.clip(hab_immunity, 0.0, 0.9)
        hab_burden = np.maximum(0.0, -hab_pool) / 50000.0
        hab_drag_penalty = 1.0 + np.tanh(hab_burden)
        norm_profit_pressure = np.expm1(np.clip(np.maximum(0.0, profit_pressure) / 50.0, 0.0, 5.0))
        norm_trapped_pressure = np.expm1(np.clip(np.maximum(0.0, trapped_pressure), 0.0, 1.0)) * 1.5
        norm_dist = np.clip(dist_score / 100.0, 0.0, 1.0)
        norm_intra_dist = np.clip(intraday_dist, 0.0, 1.0)
        norm_instability = np.tanh(np.maximum(0.0, instability) / 100.0)
        norm_downtrend = np.clip(downtrend_str / 100.0, 0.0, 1.0)
        dump_quality_factor = 1.0 + np.clip(outflow_qual / 100.0, 0.0, 1.0) * 1.5
        energy_factor = 1.0 + np.clip(dist_energy / 100.0, 0.0, 1.0)
        coupled_active_dump = (norm_dist + norm_intra_dist * 0.5) * dump_quality_factor * energy_factor * kine_multiplier
        beta_headwind = 1.0 + np.clip(ind_downtrend, 0.0, 1.0)
        friction_vpa = 1.0 + np.maximum(0.0, 1.0 - np.clip(vpa_efficiency, -1.0, 1.0))
        skew_penalty = 1.0 + np.maximum(0.0, -intra_skew) * 0.5
        coupled_viscosity = (1.0 + norm_instability) * beta_headwind * friction_vpa * skew_penalty
        coupled_gravity = (norm_profit_pressure + norm_trapped_pressure) * (1.0 + norm_downtrend)
        norm_release = np.clip(pressure_release, 0.0, 1.0)
        norm_shakeout = np.clip(shakeout_score / 100.0, 0.0, 1.0)
        relief_valve = np.maximum(0.1, 1.0 + norm_release * 1.5 + norm_shakeout * 1.0)
        hf_hidden_div = np.clip(hf_flow_div / 50.0, 0.0, 2.0)
        turnover_drag = np.expm1(np.clip((turnover_rate / 100.0) - 0.05, 0.0, 5.0) * 10.0) * 0.5
        core_drag_raw = np.clip(((coupled_gravity + coupled_active_dump) * coupled_viscosity * hab_drag_penalty) / relief_valve, 0.0, 1000.0)
        core_drag_shielded = np.clip(core_drag_raw * (1.0 - hab_immunity) + turnover_drag + hf_hidden_div, 0.0, 1000.0)
        avalanche_threshold = 1.5
        excess_drag = np.clip(np.maximum(0.0, core_drag_shielded - avalanche_threshold), 0.0, 10.0)
        avalanche_gain = 1.0 + (excess_drag ** 1.5) * 1.5
        final_drag = np.clip(np.nan_to_num(core_drag_shielded * avalanche_gain, nan=0.0, posinf=10000.0, neginf=0.0), 0.0, 10000.0)
        return final_drag

    def _calc_tensor_synthesis(self, thrust: np.ndarray, structure: np.ndarray, drag: np.ndarray, raw: Dict[str, np.ndarray], idx: pd.Index) -> np.ndarray:
        """
        【V19.0 · 奇点张量无极合成修复版】
        统一限制结构张量对于推力的放大/衰减倍数 `[0.2, 5.0]` 内，杜绝荒谬爆冲。
        """
        mf_net_buy = raw['mf_net_buy'].values
        pushing_score = raw['pushing_score'].values
        energy_conc = raw['energy_conc'].values
        mf_slope_13 = raw['mf_slope_13'].values
        mf_accel_13 = raw['mf_accel_13'].values
        mf_jerk_13 = raw['mf_jerk_13'].values
        push_slope_13 = raw['pushing_slope_13'].values
        push_accel_13 = raw['pushing_accel_13'].values
        push_jerk_13 = raw['pushing_jerk_13'].values
        flow_21d = raw['flow_21d'].values
        flow_55d = raw['flow_55d'].values
        theme_hotness = raw['theme_hotness'].values
        is_leader = raw['is_leader'].values
        gap_momentum = raw['gap_momentum'].values
        reversal_prob = raw['reversal_prob'].values
        lock_ratio = raw['lock_ratio'].values
        trend_confirm = raw['trend_confirm'].values
        emotional_extreme = raw['emotional_extreme'].values
        game_intensity = raw['game_intensity'].values
        golden_pit = raw['golden_pit'].values
        breakout_conf = raw['breakout_conf'].values
        hm_top_tier = raw['hm_top_tier'].values
        t1_premium = raw['t1_premium'].values
        breakout_pot = raw['breakout_pot'].values
        ma_compression = raw['ma_compression'].values
        turnover_stability = raw['turnover_stability'].values
        norm_push = np.clip(pushing_score / 100.0, 0.0, 1.0)
        kine_damping = np.tanh(np.abs(mf_net_buy) / 10000.0) * norm_push
        k_mf = np.tanh(mf_slope_13) * 0.3 + np.tanh(mf_accel_13) * 0.3 + np.tanh(mf_jerk_13) * 0.4
        k_push = np.tanh(push_slope_13) * 0.3 + np.tanh(push_accel_13) * 0.3 + np.tanh(push_jerk_13) * 0.4
        kinematic_burst = 1.0 + np.maximum(0.0, (k_mf + k_push * 0.5) * kine_damping)
        combined_inventory = (flow_21d * 0.618) + (flow_55d * 0.382)
        hab_immunity = 1.0 - (1.0 / (1.0 + np.exp(np.clip(combined_inventory / 50000.0, -20.0, 20.0))))
        hab_fuel = np.maximum(0.0, np.tanh(combined_inventory / 100000.0))
        norm_theme = np.clip(theme_hotness / 100.0, 0.0, 1.0)
        norm_breakout_pot = np.clip(breakout_pot, 0.0, 1.0)
        norm_turnover_stab = np.clip(turnover_stability, 0.0, 1.0)
        eco_premium = 1.0 + (is_leader * 0.8) + (hm_top_tier * 0.6) + (breakout_conf * 0.4) + (norm_theme * 0.3) + (np.clip(trend_confirm, 0.0, 1.0) * 0.5)
        eff_structure = np.where(thrust >= 0, np.sqrt(np.clip(structure, 0.01, 100.0)), 1.0 / np.clip(np.sqrt(np.clip(structure, 0.01, 100.0)), 0.2, 5.0))
        eff_eco_premium = np.where(thrust >= 0, eco_premium, 1.0 / np.clip(eco_premium, 0.5, 2.0))
        eff_gap_mom = np.where(thrust >= 0, 1.0 + gap_momentum, 1.0 / np.clip(1.0 + gap_momentum, 0.5, 2.0))
        eff_breakout = np.where(thrust >= 0, 1.0 + norm_breakout_pot, 1.0 / np.clip(1.0 + norm_breakout_pot, 0.5, 2.0))
        eff_turnover = np.where(thrust >= 0, 1.0 + norm_turnover_stab * 0.5, 1.0 / np.clip(1.0 + norm_turnover_stab * 0.5, 0.5, 2.0))
        base_tensor = thrust * eff_structure * eff_gap_mom * eff_eco_premium * kinematic_burst * eff_breakout * eff_turnover
        norm_lock_ratio = np.clip(lock_ratio / 100.0, 0.0, 1.0)
        raw_effective_drag = drag * (1.0 - np.maximum(0.0, np.minimum(hab_immunity, 0.90)))
        exp_arg = np.clip(-2.0 * (base_tensor - 1.5 * raw_effective_drag), -10.0, 10.0)
        squeeze_transition = 1.0 / (1.0 + np.exp(exp_arg))
        norm_game_intensity = np.clip(game_intensity, 0.0, 1.0)
        trap_reversal_factor = 1.0 + (golden_pit * 2.0)
        norm_energy = np.clip(energy_conc, 0.0, 1.0)
        squeeze_bonus = np.where(base_tensor >= 0, squeeze_transition * raw_effective_drag * (1.0 + np.abs(emotional_extreme)) * norm_game_intensity * kinematic_burst * trap_reversal_factor * norm_energy, 0.0)
        norm_reversal_prob = np.clip(reversal_prob / 100.0, 0.0, 1.0)
        final_drag = raw_effective_drag * (1.0 - squeeze_transition) * (1.0 - norm_reversal_prob) * (1.0 - norm_lock_ratio * 0.5)
        raw_intent = np.where(
            base_tensor >= 0,
            (base_tensor / np.maximum(1.0 + final_drag, 1.0)) + squeeze_bonus,
            base_tensor * (1.0 + np.sqrt(np.clip(final_drag, 0.0, 100.0)))
        )
        t1_multiplier = np.exp(np.clip(np.tanh((t1_premium - 50.0) / 20.0), -1.0, 1.0))
        norm_compression = np.clip(ma_compression, 0.0, 1.0)
        hri = np.where(
            base_tensor >= 0,
            (base_tensor * (1.0 + squeeze_bonus)) / np.maximum(1.0 + final_drag, 1.0),
            base_tensor * (1.0 + np.sqrt(np.clip(final_drag, 0.0, 100.0)))
        )
        hri_threshold = 3.0
        hri_magnitude = np.abs(hri)
        hri_excess = np.clip(np.maximum(0.0, hri_magnitude - hri_threshold), 0.0, 4.0)
        exponent_gain = np.clip(hri_excess * t1_multiplier * (1.0 + norm_compression + hab_fuel), 0.0, 4.0)
        singularity_gain = 1.0 + np.expm1(exponent_gain)
        final_intent = raw_intent * singularity_gain
        return final_intent

    def _generate_probe_report(self, idx, raw, thrust, structure, drag, raw_intent, compressed_intent, z_scores, final):
        """【V19.0 · 探针报告】增加了对数压缩与 Z-Score 映射的分布调试输出。"""
        locs = self._get_probe_locs(idx, compressed_intent)
        for i in locs:
            ts = idx[i]
            net_buy = raw['mf_net_buy'].values[i]
            energy_conc = raw['energy_conc'].values[i]
            energy_damping = np.tanh(np.abs(net_buy) / 10000.0) * energy_conc
            k_burst = 1.0 + ((np.tanh(raw['mf_slope_13'].values[i]) * 0.3 + np.tanh(raw['mf_accel_13'].values[i]) * 0.3 + np.tanh(raw['mf_jerk_13'].values[i]) * 0.4) * energy_damping)
            comb_inv = (raw['flow_21d'].values[i] * 0.6) + (raw['flow_55d'].values[i] * 0.4)
            hab_imm = 1.0 - (1.0 / (1.0 + np.exp(np.clip(comb_inv / 50000.0, -10.0, 10.0))))
            eff_drag = drag[i] * (1.0 - hab_imm)
            report = [
                f"\n=== [PROBE V19.0] CalculateMainForceRallyIntent Full-Chain Audit (对数映射版) @ {ts.strftime('%Y-%m-%d')} ===",
                f"【0. Raw Data Overview (底层核心数据快照)】",
                f"   [Thrust] MF_NetBuy: {raw['mf_net_buy'].values[i]:.2f} | Tick_Large_Net: {raw['tick_large_net'].values[i]:.2f} | Flow_21d: {raw['flow_21d'].values[i]:.2f} | Flow_55d: {raw['flow_55d'].values[i]:.2f}",
                f"   [Struct] Close: {raw['close'].values[i]:.2f} | Cost_Avg: {raw['cost_avg'].values[i]:.2f} | Chip_Entropy: {raw['chip_entropy'].values[i]:.4f} | Control_Solidity: {raw['control_solidity'].values[i]:.4f}",
                f"   [Drag]   Profit_Pres: {raw['profit_pressure'].values[i]:.4f} | Trapped_Pres: {raw['trapped_pressure'].values[i]:.4f} | Dist_Score: {raw['dist_score'].values[i]:.4f} | Turnover: {raw['turnover'].values[i]:.4f}",
                f"   [Eco]    Market_Sentiment: {raw['market_sentiment'].values[i]:.4f} | Is_Leader: {raw['is_leader'].values[i]:.1f} | Reversal_Prob: {raw['reversal_prob'].values[i]:.4f}",
                f"---------------------------------------------------------------",
                f"【A. Kinematics (动力学)】 Burst: x{k_burst:.4f} | Damping: {energy_damping:.4f} | Jerk: {raw['mf_jerk_13'].values[i]:.2f}",
                f"【B. HAB (存量意识)】 21d/55d Inv: {raw['flow_21d'].values[i]:.0f}/{raw['flow_55d'].values[i]:.0f} | Immunity: {hab_imm*100:.1f}%",
                f"【C. Ecosystem (生态)】 Leader: {raw['is_leader'].values[i]} | LockRatio: {raw['lock_ratio'].values[i]:.2f}% | Trend Confirm: {raw['trend_confirm'].values[i]:.2f}",
                f"【E. Synthesis (合成)】 Thrust: {thrust[i]:.4f} | Structure: {structure[i]:.4f} | EffectiveDrag: {eff_drag:.4f}",
                f"【F. Result (最终)】 Raw Intent: {raw_intent[i]:.4f} | Compressed log1p: {compressed_intent[i]:.4f} | Z-Score: {z_scores[i]:.4f}",
                f"                  -> Final Normalized Score: {final[i]:.4f}",
                f"===============================================================\n"
            ]
            self._probe_cache.extend(report)
            for line in report: print(line)

    def _is_probe_enabled(self) -> bool:
        return get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)







