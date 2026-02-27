# strategies\trend_following\intelligence\process\calculate_storm_eye_calm.py
# 【V58.0.2】 拆单吸筹强度 已完成
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score,
    normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
class CalculateStormEyeCalm:
    """
    【V2.0.0 · 风暴眼寂静 · 绝对非对称博弈与引力弹弓重构版】
    PROCESS_META_STORM_EYE_CALM
    - 核心修正: 彻底铲除 Look-ahead Bias，利用历史收益滞后对齐修复胜率测算。
    - 降噪方案: 引入 Threshold Gate 自适应门限函数，消除微积分操作产生的零基陷阱。
    - 时空记忆: 建立“增量/存量”占比的 HAB 时空缓冲系统，赋予资金流深度记忆。
    - 融合重构: 废除 .clip() 硬截断与 0值连乘，使用 Laplace 软连接与 Tanh 平滑映射张量。
    """
    def __init__(self, strategy_instance, helper: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        self.actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = "calculate_storm_eye_calm"
        self.last_df_index = df.index
        df_index = df.index
        params = self._get_storm_eye_calm_params(config)
        self._check_and_fill_data_existence(df, params)
        raw_data = self._get_raw_and_atomic_data(df, method_name, params)
        _temp_debug_values = {"原始信号值": {}, "能量压缩": {}, "量能枯竭": {}, "主力隐蔽意图": {}, "市场情绪低迷融合": {}, "突破准备度融合": {}, "最终融合": {}}
        energy_score = self._calculate_energy_compression_component(df_index, raw_data, {}, params['energy_compression_weights'], _temp_debug_values)
        volume_score = self._calculate_volume_exhaustion_component(df_index, raw_data, {}, params['volume_exhaustion_weights'], _temp_debug_values)
        intent_score, intent_dict = self._calculate_main_force_covert_intent_component(df_index, raw_data, {}, params['main_force_covert_intent_weights'], {}, _temp_debug_values)
        sentiment_score = self._calculate_subdued_market_sentiment_component(df_index, raw_data, params['subdued_market_sentiment_weights'], 21, 55, 1.0, 0.2, _temp_debug_values)
        readiness_score = self._calculate_breakout_readiness_component(df_index, raw_data, params['breakout_readiness_weights'], _temp_debug_values)
        dynamic_threshold = self._calculate_adaptive_phase_transition_threshold(df_index, raw_data)
        component_scores = {
            'energy': energy_score * self._calculate_fermi_dirac_gate(energy_score, threshold=dynamic_threshold, beta=12.0),
            'volume': volume_score * self._calculate_fermi_dirac_gate(volume_score, threshold=dynamic_threshold, beta=12.0),
            'intent': intent_score * self._calculate_fermi_dirac_gate(intent_score, threshold=dynamic_threshold, beta=12.0),
            'sentiment': sentiment_score * self._calculate_fermi_dirac_gate(sentiment_score, threshold=dynamic_threshold, beta=12.0),
            'readiness': readiness_score * self._calculate_fermi_dirac_gate(readiness_score, threshold=dynamic_threshold, beta=12.0)
        }
        final_fusion_score = self._perform_lp_norm_fusion(df_index, component_scores, params['final_fusion_weights'], raw_data, _temp_debug_values)
        raw_final_score = final_fusion_score * self._calculate_market_regulator_modulator(df_index, raw_data, params, _temp_debug_values)
        resonance_confirm = self._soft_clip((raw_final_score - 0.4) * 10.0) * self._soft_clip((self._calculate_consensus_entropy(component_scores) - 0.7) * 10.0)
        latch_multiplier = pd.Series(np.where(resonance_confirm.rolling(5, min_periods=1).sum() >= 2.5, 1.2, 1.0), index=df_index)
        latched_score = raw_final_score.rolling(3, min_periods=1).mean() * latch_multiplier
        veto_factor = self._calculate_kinetic_overflow_veto(df_index, raw_data, self._calculate_oversold_momentum_bipolarization(df_index, raw_data))
        reward_factor = self._calculate_spatio_temporal_asymmetric_reward(df_index, raw_data, resonance_confirm)
        mrkb_factor = self._calculate_mean_reversion_kinetic_bias(df_index, raw_data)
        tes_factor = self._calculate_trend_energy_shearing(df_index, raw_data)
        final_latched_score = self._soft_clip(latched_score * veto_factor * reward_factor * mrkb_factor * tes_factor)
        _temp_debug_values["最终融合"]["final_score"] = final_latched_score
        is_debug_enabled, probe_ts = self._get_debug_info(df, method_name)
        if is_debug_enabled and probe_ts is not None:
            self._print_debug_output_for_storm_eye_calm({}, _temp_debug_values, probe_ts, method_name, final_latched_score)
        return final_latched_score.astype(np.float32)
    def _check_and_fill_data_existence(self, df: pd.DataFrame, params: Dict):
        req_signals = self._get_required_signals(params)
        missing = [c for c in req_signals if c not in df.columns]
        if missing:
            print(f"【V2.0.0 探针警报】风暴眼基底特征断层，缺失列: {missing}。系统已启动拉普拉斯安全回退与零基填充保护机制！")
    def _soft_clip(self, series: pd.Series, min_val: float = 0.0, max_val: float = 1.0) -> pd.Series:
        mid = (max_val + min_val) / 2.0
        scale = (max_val - min_val) / 2.0
        return mid + scale * np.tanh((series - mid) / (scale + 1e-9))
    def _fermi_dirac_manifold(self, score_series: pd.Series, threshold: pd.Series | float = 0.5, beta: float = 10.0) -> pd.Series:
        if isinstance(threshold, pd.Series):
            threshold = threshold.reindex(score_series.index).fillna(0.5)
        return 1.0 / (1.0 + np.exp(beta * (threshold - score_series)))
    def _calculate_fermi_dirac_gate(self, score_series: pd.Series, threshold: float = 0.5, beta: float = 10.0) -> pd.Series:
        return 1.0 / (1.0 + np.exp(beta * (threshold - score_series)))
    def _lp_norm_fusion(self, scores: List[pd.Series], weights: List[float], p: float = 2.0) -> pd.Series:
        valid_scores, valid_weights = [], []
        for s, w in zip(scores, weights):
            valid_scores.append(s.fillna(0.0) + 0.05)
            valid_weights.append(w)
        weight_sum = sum(valid_weights) + 1e-9
        norm_weights = [w / weight_sum for w in valid_weights]
        sum_pow = pd.Series(0.0, index=valid_scores[0].index)
        for s, w in zip(valid_scores, norm_weights):
            sum_pow += w * (s ** p)
        return sum_pow ** (1.0 / p)
    def _apply_threshold_gate(self, series: pd.Series, window: int = 21) -> pd.Series:
        noise_floor = series.rolling(window=window, min_periods=5).std().fillna(0.0) * 0.5 + 1e-9
        return series * np.tanh((series / noise_floor)**2)
    def _calculate_custom_normalization(self, series: pd.Series, mode: str, sensitivity: float = 1.0, window: int = 55, denoise: bool = False, atr_series: Optional[pd.Series] = None) -> pd.Series:
        if not isinstance(series, pd.Series):
            series = pd.Series(float(series), index=getattr(self, 'last_df_index', []))
        if denoise and len(series) >= 21:
            if atr_series is not None:
                series = series / (atr_series + 1e-9)
            else:
                series = self._apply_threshold_gate(series, window=21)
        if mode == 'limit_low':
            return 1.0 - self._soft_clip(np.abs(series * sensitivity))
        elif mode == 'limit_high':
            return self._soft_clip(np.maximum(0, series * sensitivity))
        elif mode == 'zero_focus':
            return np.exp(- (series * sensitivity) ** 2)
        elif mode == 'relative_rank':
            roll_min = series.rolling(window=window, min_periods=1).min()
            roll_max = series.rolling(window=window, min_periods=1).max()
            norm = (series - roll_min) / (roll_max - roll_min + 1e-9)
            return self._soft_clip(norm * sensitivity)
        return pd.Series(0.0, index=series.index)
    def _get_debug_info(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp]]:
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        return is_debug_enabled_for_method, probe_ts
    def _print_debug_output_for_storm_eye_calm(self, debug_output: Dict, _temp_debug_values: Dict, probe_ts: pd.Timestamp, method_name: str, final_score: pd.Series):
        debug_output[f"  -- [风暴眼全链路量子探针] {method_name} @ {probe_ts.strftime('%Y-%m-%d')} --"] = ""
        for category, items in _temp_debug_values.items():
            if category == "原始信号值": continue
            debug_output[f"      [{category}]:"] = ""
            for key, series in items.items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
        debug_output[f"  -- 破局极值最终得分: {final_score.loc[probe_ts]:.4f}"] = ""
        for key, value in debug_output.items():
            if value: print(f"{key}: {value}")
            else: print(key)
    def _get_storm_eye_calm_params(self, config: Dict) -> Dict:
        params = get_param_value(config.get('storm_eye_calm_params'), {})
        return {
            'energy_compression_weights': get_param_value(params.get('energy_compression_weights'), {}),
            'volume_exhaustion_weights': get_param_value(params.get('volume_exhaustion_weights'), {}),
            'main_force_covert_intent_weights': get_param_value(params.get('main_force_covert_intent_weights'), {}),
            'subdued_market_sentiment_weights': get_param_value(params.get('subdued_market_sentiment_weights'), {}),
            'breakout_readiness_weights': get_param_value(params.get('breakout_readiness_weights'), {}),
            'mtf_cohesion_weights': get_param_value(params.get('mtf_cohesion_weights'), {"cohesion_score": 1.0}),
            'final_fusion_weights': get_param_value(params.get('final_fusion_weights'), {}),
            'price_calmness_modulator_params': get_param_value(params.get('price_calmness_modulator_params'), {}),
            'main_force_control_adjudicator_params': get_param_value(params.get('main_force_control_adjudicator'), {}),
            'mtf_slope_accel_weights': get_param_value(params.get('mtf_slope_accel_weights'), {}),
            'regime_modulator_params': get_param_value(params.get('regime_modulator_params'), {}),
            'mtf_cohesion_base_signals': get_param_value(params.get('mtf_cohesion_base_signals'), []),
            'sentiment_volatility_window': get_param_value(params.get('sentiment_volatility_window'), 21),
            'long_term_sentiment_window': get_param_value(params.get('long_term_sentiment_window'), 55),
            'main_force_flow_volatility_window': get_param_value(params.get('main_force_flow_volatility_window'), 21),
            'sentiment_neutral_range': get_param_value(params.get('sentiment_neutral_range'), 1.0),
            'sentiment_pendulum_neutral_range': get_param_value(params.get('sentiment_pendulum_neutral_range'), 0.2),
            'ambiguity_components_weights': get_param_value(params.get('ambiguity_components_weights'), {}),
        }
    def _get_required_signals(self, params: Dict) -> list[str]:
        required_signals = [
            'MA_POTENTIAL_TENSION_INDEX_D', 'MA_COHERENCE_RESONANCE_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'BBW_21_2.0_D', 'chip_concentration_ratio_D', 'concentration_entropy_D', 'PRICE_ENTROPY_D', 'GEOM_ARC_CURVATURE_D', 'dynamic_consolidation_duration_D', 'turnover_rate_f_D', 'volume_D', 'intraday_trough_filling_degree_D', 'tick_abnormal_volume_ratio_D', 'afternoon_flow_ratio_D', 'absorption_energy_D', 'stealth_flow_ratio_D', 'tick_clustering_index_D', 'accumulation_signal_score_D', 'SMART_MONEY_HM_NET_BUY_D', 'HM_ACTIVE_TOP_TIER_D', 'net_mf_amount_D', 'profit_ratio_D', 'winner_rate_D', 'market_sentiment_score_D', 'breakout_potential_D', 'breakout_confidence_D', 'breakout_penalty_score_D', 'resistance_strength_D', 'GEOM_REG_R2_D', 'GEOM_REG_SLOPE_D', 'ATR_14_D', 'chip_stability_D', 'ADX_14_D', 'flow_impact_ratio_D', 'industry_preheat_score_D', 'industry_rank_accel_D', 'industry_strength_rank_D', 'trend_confirmation_score_D', 'main_force_activity_index_D', 'intraday_cost_center_migration_D', 'migration_convergence_ratio_D', 'tick_chip_balance_ratio_D', 'VPA_EFFICIENCY_D', 'VPA_MF_ADJUSTED_EFF_D', 'VPA_ACCELERATION_13D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'OCH_ACCELERATION_D', 'OCH_D', 'PDI_14_D', 'NDI_14_D', 'price_vs_ma_21_ratio_D', 'price_vs_ma_55_ratio_D', 'HM_COORDINATED_ATTACK_D', 'TURNOVER_STABILITY_INDEX_D', 'amount_D', 'HM_ACTIVE_ANY_D', 'BIAS_55_D', 'MA_ACCELERATION_EMA_55_D', 'STATE_GOLDEN_PIT_D', 'BIAS_5_D', 'MA_FAN_EFFICIENCY_D', 'RSI_13_D', 'close', 'MA_144_D', 'chip_entropy_D', 'pressure_trapped_D', 'consolidation_quality_score_D', 'net_energy_flow_D', 'intraday_chip_game_index_D'
        ]
        return list(set(required_signals))
    def _safe_diff(self, series: pd.Series, period: int) -> pd.Series:
        return self._apply_threshold_gate(series.diff(period))
    def _get_raw_and_atomic_data(self, df: pd.DataFrame, method_name: str, params: Dict) -> Dict[str, pd.Series]:
        raw_data = {col: df.get(col, pd.Series(0.0, index=df.index)) for col in self._get_required_signals(params)}
        raw_data['close_D'] = df.get('close_D', df.get('close', pd.Series(0.0, index=df.index)))
        deriv_cols = [
            'VPA_ACCELERATION_13D', 'VPA_MF_ADJUSTED_EFF_D', 'tick_abnormal_volume_ratio_D', 'MA_ACCELERATION_EMA_55_D', 'PRICE_ENTROPY_D', 'STATE_GOLDEN_PIT_D', 'BIAS_55_D', 'NDI_14_D', 'PDI_14_D', 'breakout_penalty_score_D', 'RSI_13_D', 'OCH_D', 'ATR_14_D', 'MA_FAN_EFFICIENCY_D', 'HM_ACTIVE_ANY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'HM_COORDINATED_ATTACK_D', 'BIAS_5_D', 'market_sentiment_score_D', 'ADX_14_D', 'profit_ratio_D', 'chip_entropy_D', 'net_energy_flow_D'
        ]
        for col in deriv_cols:
            if col in raw_data:
                s13 = self._safe_diff(raw_data[col], 13)
                a8 = self._safe_diff(s13, 8)
                j5 = self._safe_diff(a8, 5)
                raw_data[f'SLOPE_13_{col}'] = s13
                raw_data[f'ACCEL_8_{col}'] = a8
                raw_data[f'JERK_5_{col}'] = j5
                raw_data[f'SLOPE_5_{col}'] = self._safe_diff(raw_data[col], 5)
        raw_data['price_vs_ma_144_ratio'] = raw_data['close_D'] / (raw_data['MA_144_D'] + 1e-9)
        raw_data['ACCEL_8_price_vs_ma_144_ratio'] = self._safe_diff(self._safe_diff(raw_data['price_vs_ma_144_ratio'], 13), 8)
        raw_data['pain_index_proxy'] = 1.0 - raw_data['profit_ratio_D']
        raw_data['JERK_5_pain_index_proxy'] = raw_data.get('JERK_5_profit_ratio_D', pd.Series(0.0, index=df.index)) * -1.0
        raw_data['price_slope_raw'] = raw_data.get('SLOPE_5_market_sentiment_score_D', pd.Series(0.0, index=df.index))
        return raw_data
    def _calculate_qho_historical_accumulation_buffer(self, daily_series: pd.Series, windows: list[int] = [13, 21, 34, 55]) -> pd.Series:
        buffers = []
        for w in windows:
            historical_stock = daily_series.abs().rolling(window=w, min_periods=w//2).mean() + 1e-9
            incremental_impact = daily_series / historical_stock
            buffer_factor = np.tanh(incremental_impact.abs() * 0.5)
            buffers.append(buffer_factor)
        return pd.concat(buffers, axis=1).mean(axis=1).fillna(0.0)
    def _calculate_energy_compression_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        fcc_factor = self._calculate_fan_curvature_collapse(df_index, raw_data)
        vvc_factor = self._calculate_volatility_vacuum_contraction(df_index, raw_data)
        lrf_score = self._calculate_linear_resonance_failure(df_index, raw_data)
        struct_quality = self._calculate_custom_normalization(raw_data['chip_stability_D'], mode='limit_high', sensitivity=1.5)
        entropy_gain = self._calculate_custom_normalization(raw_data.get('SLOPE_13_chip_entropy_D', pd.Series(0.0, index=df_index)), mode='limit_low', sensitivity=10.0, denoise=True)
        vpa_accel = raw_data.get('ACCEL_8_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        vpa_jerk = raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        phase_space_dist = np.sqrt(np.square(vpa_accel) + np.square(vpa_jerk))
        phase_attractor = np.exp(-phase_space_dist * 5.0)
        final_energy = self._lp_norm_fusion([fcc_factor, vvc_factor, lrf_score, entropy_gain, phase_attractor], [0.2, 0.2, 0.2, 0.2, 0.2], p=2.0) * (struct_quality + 0.1)
        _temp_debug_values["能量压缩"]["final"] = self._soft_clip(final_energy)
        return self._soft_clip(final_energy)
    def _calculate_volume_exhaustion_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        turnover_score = self._calculate_custom_normalization(raw_data['turnover_rate_f_D'], mode='limit_low', sensitivity=25.0)
        trough_fill = self._calculate_custom_normalization(raw_data['intraday_trough_filling_degree_D'], mode='limit_high', sensitivity=3.0)
        mdb_factor = self._calculate_momentum_dissipation_balance(df_index, raw_data)
        solid_factor = self._calculate_liquidity_solidification_threshold(df_index, raw_data)
        vpa_jerk = self._calculate_custom_normalization(raw_data.get('JERK_5_VPA_EFFICIENCY_D', pd.Series(0.0, index=df_index)), mode='limit_low', sensitivity=20.0, denoise=True)
        mf_eff = self._calculate_custom_normalization(raw_data['VPA_MF_ADJUSTED_EFF_D'], mode='limit_high', sensitivity=2.0)
        base_vac = self._lp_norm_fusion([turnover_score, trough_fill], [0.4, 0.6], p=2.0)
        final_vol = base_vac * (mdb_factor + 0.1) * (0.6 + 0.4 * solid_factor) * (0.8 + 0.2 * mf_eff) * (1.0 - 0.3 * (1.0 - vpa_jerk))
        _temp_debug_values["量能枯竭"]["final"] = self._soft_clip(final_vol)
        return self._soft_clip(final_vol)
    def _calculate_main_force_covert_intent_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, ambiguity_weights: Dict, _temp_debug_values: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        chf_base = raw_data['SMART_MONEY_HM_COORDINATED_ATTACK_D'].rolling(window=8, min_periods=1).mean()
        chf_jerk_score = self._calculate_custom_normalization(raw_data.get('JERK_5_SMART_MONEY_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=25.0, denoise=True)
        htc_factor = self._calculate_hunting_temporal_coherence(df_index, raw_data)
        stealth_score = self._calculate_custom_normalization(raw_data['stealth_flow_ratio_D'], mode='limit_high', sensitivity=4.0)
        migration_accel = self._calculate_custom_normalization(raw_data.get('ACCEL_8_intraday_cost_center_migration_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=20.0)
        mf_hab = self._calculate_qho_historical_accumulation_buffer(raw_data['net_mf_amount_D'], windows=[21, 34])
        energy_flow = self._calculate_custom_normalization(raw_data.get('net_energy_flow_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.5)
        base_intent = self._lp_norm_fusion([stealth_score, migration_accel, chf_base, chf_jerk_score, energy_flow], [0.2, 0.2, 0.2, 0.2, 0.2], p=2.0)
        final_intent = base_intent * (0.75 + 0.25 * htc_factor) * (0.6 + 0.4 * mf_hab)
        _temp_debug_values["主力隐蔽意图"]["final"] = self._soft_clip(final_intent)
        return self._soft_clip(final_intent), {"stealth_score": stealth_score, "htc_factor": htc_factor}
    def _calculate_subdued_market_sentiment_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, sentiment_volatility_window: int, long_term_sentiment_window: int, sentiment_neutral_range: float, sentiment_pendulum_neutral_range: float, _temp_debug_values: Dict) -> pd.Series:
        pain_score = self._calculate_custom_normalization(raw_data['pain_index_proxy'], mode='limit_high', sensitivity=3.0)
        despair_burst = self._calculate_custom_normalization(raw_data.get('JERK_5_pain_index_proxy', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=20.0, denoise=True)
        short_exhaustion = self._calculate_short_exhaustion_divergence(df_index, raw_data)
        bipolar_gain = self._calculate_oversold_momentum_bipolarization(df_index, raw_data)
        panic_resonance = self._calculate_extreme_panic_resonance(df_index, raw_data)
        order_gain = self._calculate_micro_order_gain(df_index, raw_data)
        cleanse_score = self._calculate_custom_normalization(raw_data['winner_rate_D'], mode='limit_low', sensitivity=15.0)
        trapped_pressure = self._calculate_custom_normalization(raw_data['pressure_trapped_D'], mode='limit_high', sensitivity=2.0)
        base_subdued = self._lp_norm_fusion([pain_score, cleanse_score, trapped_pressure], [0.4, 0.3, 0.3], p=2.0)
        final_sentiment = (base_subdued * (1.0 + 0.3 * order_gain) * self._lp_norm_fusion([short_exhaustion, bipolar_gain, panic_resonance], [0.3, 0.3, 0.4], p=2.0) + 0.25 * despair_burst)
        _temp_debug_values["市场情绪低迷融合"]["final"] = self._soft_clip(final_sentiment)
        return self._soft_clip(final_sentiment)
    def _calculate_breakout_readiness_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        grp_score = self._calculate_gravitational_regression_pull(df_index, raw_data)
        plr_score = self._calculate_phase_locked_resonance(df_index, raw_data)
        sed_score = self._calculate_short_exhaustion_divergence(df_index, raw_data)
        ssd_score = self._calculate_seat_scatter_decay(df_index, raw_data)
        egd_score = self._calculate_efficiency_gradient_dissipation(df_index, raw_data)
        sope_score = self._calculate_split_order_pulse_entropy(df_index, raw_data)
        aeo_score = self._calculate_abnormal_energy_overflow(df_index, raw_data)
        neutral_score = self._calculate_game_neutralization_modulator(df_index, raw_data)
        aded_score = self._calculate_amount_distribution_entropy_delta(df_index, raw_data)
        well_collapse = self._calculate_potential_well_collapse(df_index, raw_data)
        long_awakening = self._calculate_long_awakening_threshold(df_index, raw_data)
        stress_test = self._calculate_level_stress_test_modulator(df_index, raw_data)
        consolidation = self._calculate_custom_normalization(raw_data.get('consolidation_quality_score_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        momentum_part = self._lp_norm_fusion([grp_score, plr_score], [0.5, 0.5], p=2.0)
        friction_part = self._lp_norm_fusion([sed_score, ssd_score], [0.5, 0.5], p=2.0)
        quality_part = self._lp_norm_fusion([egd_score, sope_score, aeo_score], [0.3, 0.4, 0.3], p=2.0)
        state_part = self._lp_norm_fusion([well_collapse, long_awakening, aded_score, stress_test, neutral_score, consolidation], [0.15, 0.15, 0.15, 0.15, 0.2, 0.2], p=2.0)
        acc_hab = self._calculate_qho_historical_accumulation_buffer(raw_data['accumulation_signal_score_D'], windows=[21, 34])
        readiness = self._lp_norm_fusion([momentum_part, friction_part, quality_part, state_part], [0.15, 0.15, 0.3, 0.4], p=2.0) * (0.8 + 0.2 * acc_hab)
        _temp_debug_values["突破准备度融合"]["final"] = self._soft_clip(readiness)
        return self._soft_clip(readiness)
    def _calculate_market_regulator_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        sector_preheat = raw_data['industry_preheat_score_D']
        sector_hab = self._calculate_qho_historical_accumulation_buffer(sector_preheat, windows=[13, 21])
        sector_jerk = raw_data.get('JERK_5_industry_rank_accel_D', pd.Series(0.0, index=df_index))
        clean_sector_jerk = np.where(sector_jerk > sector_jerk.rolling(21).std().fillna(0), sector_jerk, 0.0)
        sector_ignite_score = self._calculate_custom_normalization(pd.Series(clean_sector_jerk, index=df_index), mode='limit_high', sensitivity=10.0)
        stock_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        macro_resonance = (sector_ignite_score * stock_calm) ** 0.5
        adx_raw = raw_data['ADX_14_D']
        adx_supp = 1.0 / (1.0 + np.exp(0.4 * (adx_raw - 28.0)))
        final_modulator = (0.7 + 0.3 * sector_hab) * (1.0 + 0.5 * macro_resonance) * adx_supp
        return self._soft_clip(final_modulator, min_val=0.2, max_val=1.5)
    def _perform_lp_norm_fusion(self, df_index: pd.Index, component_scores: dict[str, pd.Series], final_fusion_weights: dict, raw_data: dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        ewd_factor = self._calculate_consensus_entropy(component_scores)
        scores_list = [component_scores['energy'], component_scores['volume'], component_scores['intent'], component_scores['sentiment'], component_scores['readiness']]
        base_score = self._lp_norm_fusion(scores_list, [0.2, 0.2, 0.2, 0.15, 0.25], p=3.0)
        ext_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        struct_boost = self._calculate_custom_normalization(raw_data['accumulation_signal_score_D'], mode='limit_high', sensitivity=1.0)
        hunting_boost = 1.0 + 0.4 * (raw_data['SMART_MONEY_HM_COORDINATED_ATTACK_D'].rolling(8).mean() * 0.5).fillna(0)
        final_score = base_score * ewd_factor * (struct_boost + 0.1) * (1.0 + 0.5 * ext_calm) * hunting_boost
        return self._soft_clip(final_score)
    def _calculate_trend_energy_shearing(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        adx_raw = raw_data['ADX_14_D']
        adx_accel = raw_data.get('ACCEL_8_ADX_14_D', pd.Series(0.0, index=df_index))
        high_context = self._calculate_custom_normalization(adx_raw - 35.0, mode='limit_high', sensitivity=0.5)
        shearing_ignite = self._calculate_custom_normalization(adx_accel, mode='limit_low', sensitivity=15.0, denoise=True)
        shearing_factor = 1.0 + 0.2 * (high_context * shearing_ignite) ** 0.5
        return shearing_factor.fillna(1.0)
    def _calculate_consensus_entropy(self, scores_dict: dict[str, pd.Series]) -> pd.Series:
        df_scores = pd.concat(scores_dict.values(), axis=1)
        dispersion = df_scores.std(axis=1).fillna(1.0)
        corr_matrix = df_scores.rolling(window=5).corr()
        coherence = corr_matrix.groupby(level=0).mean().mean(axis=1).fillna(0.0)
        disp_decay = np.exp(- (dispersion * 2.5) ** 2)
        final_decay = disp_decay * (0.6 + 0.4 * coherence.clip(0, 1))
        return self._soft_clip(final_decay)
    def _calculate_pressure_backtest_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        penalty_raw = raw_data['breakout_penalty_score_D']
        penalty_slope = raw_data.get('SLOPE_13_breakout_penalty_score_D', pd.Series(0.0, index=df_index))
        penalty_hab = self._calculate_qho_historical_accumulation_buffer(penalty_raw, windows=[21])
        resistance_intensity = self._calculate_custom_normalization(penalty_raw * (1.0 + np.maximum(0, penalty_slope)), mode='limit_high', sensitivity=1.5)
        price_v = raw_data['price_slope_raw']
        backtest_factor = 1.0 - (resistance_intensity * np.tanh(np.maximum(0, price_v) * 10.0))
        final_modulator = (backtest_factor * (1.0 - penalty_hab)) + penalty_hab
        return self._soft_clip(final_modulator, min_val=0.2, max_val=1.0)
    def _calculate_level_stress_test_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        och_accel = raw_data['OCH_ACCELERATION_D']
        och_jerk = raw_data.get('JERK_5_OCH_ACCELERATION_D', pd.Series(0.0, index=df_index))
        och_jerk_score = self._calculate_custom_normalization(och_jerk, mode='limit_high', sensitivity=20.0, denoise=True)
        res_strength = raw_data['resistance_strength_D']
        ma21_proximity = 1.0 - np.minimum(1.0, np.abs(raw_data['price_vs_ma_21_ratio_D'] - 1.0) * 20.0)
        ma55_proximity = 1.0 - np.minimum(1.0, np.abs(raw_data['price_vs_ma_55_ratio_D'] - 1.0) * 20.0)
        level_weight = np.maximum(ma21_proximity, ma55_proximity)
        stress_test_score = och_jerk_score * level_weight * self._calculate_custom_normalization(res_strength, mode='limit_high', sensitivity=1.0)
        return self._soft_clip(stress_test_score)
    def _calculate_linear_resonance_failure(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        r2_raw = raw_data['GEOM_REG_R2_D']
        r2_accel = raw_data.get('ACCEL_8_GEOM_REG_R2_D', pd.Series(0.0, index=df_index))
        r2_hab = self._calculate_qho_historical_accumulation_buffer(r2_raw, windows=[21])
        failure_burst = self._calculate_custom_normalization(r2_accel.abs(), mode='limit_high', sensitivity=15.0, denoise=True)
        reg_slope = raw_data['GEOM_REG_SLOPE_D']
        slope_calm = self._calculate_custom_normalization(reg_slope, mode='zero_focus', sensitivity=40.0, denoise=True)
        failure_score = r2_hab * failure_burst * slope_calm
        return self._soft_clip(failure_score)
    def _calculate_micro_order_gain(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        entropy_raw = raw_data['PRICE_ENTROPY_D']
        entropy_slope = raw_data.get('SLOPE_13_PRICE_ENTROPY_D', pd.Series(0.0, index=df_index))
        game_index = raw_data.get('intraday_chip_game_index_D', pd.Series(50.0, index=df_index))
        orderly_score = self._calculate_custom_normalization(entropy_slope, mode='limit_low', sensitivity=15.0, denoise=True)
        entropy_hab = self._calculate_qho_historical_accumulation_buffer(entropy_raw, windows=[21])
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        game_intensity = self._calculate_custom_normalization(game_index, mode='limit_high', sensitivity=1.5)
        gain_score = orderly_score * price_calm * (1.0 - entropy_hab) * (game_intensity + 0.1)
        return self._soft_clip(gain_score)
    def _calculate_momentum_dissipation_balance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        vpa_accel = raw_data['VPA_ACCELERATION_13D']
        vpa_accel_jerk = raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        dissipation_focus = self._calculate_custom_normalization(vpa_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        jerk_silence = self._calculate_custom_normalization(vpa_accel_jerk, mode='zero_focus', sensitivity=60.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        mdb_score = dissipation_focus * jerk_silence * price_calm
        return self._soft_clip(mdb_score)
    def _calculate_hunting_temporal_coherence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        attack_raw = raw_data['HM_COORDINATED_ATTACK_D']
        attack_jerk = raw_data.get('JERK_5_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index))
        temporal_stability = 1.0 - self._calculate_custom_normalization(attack_raw.rolling(window=8).std(), mode='limit_high', sensitivity=2.0)
        rhythm_score = self._calculate_custom_normalization(attack_jerk, mode='zero_focus', sensitivity=50.0, denoise=True)
        top_tier_activity = self._calculate_custom_normalization(raw_data['HM_ACTIVE_TOP_TIER_D'], mode='limit_high', sensitivity=2.0)
        htc_score = temporal_stability * rhythm_score * (0.8 + 0.2 * top_tier_activity)
        return self._soft_clip(htc_score)
    def _calculate_liquidity_solidification_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        stability_raw = raw_data['TURNOVER_STABILITY_INDEX_D']
        stability_slope = raw_data.get('SLOPE_13_TURNOVER_STABILITY_INDEX_D', pd.Series(0.0, index=df_index))
        stability_score = self._calculate_custom_normalization(stability_raw, mode='limit_high', sensitivity=1.5)
        slope_growth = self._calculate_custom_normalization(stability_slope, mode='limit_high', sensitivity=10.0, denoise=True)
        stability_hab = self._calculate_qho_historical_accumulation_buffer(stability_raw, windows=[13, 21])
        turnover_low = self._calculate_custom_normalization(raw_data['turnover_rate_f_D'], mode='limit_low', sensitivity=25.0)
        solidification_factor = stability_score * (0.7 + 0.3 * slope_growth) * stability_hab * turnover_low
        return self._soft_clip(solidification_factor)
    def _calculate_amount_distribution_entropy_delta(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        entropy_raw = raw_data['concentration_entropy_D']
        entropy_slope = raw_data.get('SLOPE_13_concentration_entropy_D', pd.Series(0.0, index=df_index))
        interceptive_score = self._calculate_custom_normalization(entropy_slope, mode='limit_low', sensitivity=10.0, denoise=True)
        entropy_hab = self._calculate_qho_historical_accumulation_buffer(entropy_raw, windows=[21])
        final_score = interceptive_score * (1.0 - entropy_hab)
        return self._soft_clip(final_score)
    def _calculate_seat_scatter_decay(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        any_act = raw_data['HM_ACTIVE_ANY_D']
        top_act = raw_data['HM_ACTIVE_TOP_TIER_D']
        scatter_raw = np.maximum(0, any_act - top_act)
        scatter_jerk = pd.Series(scatter_raw, index=df_index).diff(5).diff(5).diff(5)
        decay_score = self._calculate_custom_normalization(scatter_jerk, mode='limit_low', sensitivity=15.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        final_decay = decay_score * price_calm
        return self._soft_clip(final_decay)
    def _calculate_gravitational_regression_pull(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        bias_raw = raw_data['BIAS_55_D']
        bias_accel = raw_data.get('ACCEL_8_BIAS_55_D', pd.Series(0.0, index=df_index))
        bias_hab = self._calculate_qho_historical_accumulation_buffer(np.minimum(0, bias_raw), windows=[21])
        gravity_ignite = self._calculate_custom_normalization(bias_accel, mode='limit_high', sensitivity=15.0, denoise=True)
        depth_score = self._calculate_custom_normalization(bias_raw, mode='limit_low', sensitivity=10.0)
        pull_score = depth_score * bias_hab * gravity_ignite
        return self._soft_clip(pull_score)
    def _calculate_short_exhaustion_divergence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        ndi_raw = raw_data['NDI_14_D']
        ndi_jerk = raw_data.get('JERK_5_NDI_14_D', pd.Series(0.0, index=df_index))
        exhaustion_score = self._calculate_custom_normalization(ndi_jerk, mode='limit_low', sensitivity=20.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        ndi_hab = self._calculate_qho_historical_accumulation_buffer(ndi_raw, windows=[21])
        divergence_score = exhaustion_score * price_calm * ndi_hab
        return self._soft_clip(divergence_score)
    def _calculate_long_awakening_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        pdi_raw = raw_data['PDI_14_D']
        pdi_slope = raw_data.get('SLOPE_13_PDI_14_D', pd.Series(0.0, index=df_index))
        pdi_jerk = raw_data.get('JERK_5_PDI_14_D', pd.Series(0.0, index=df_index))
        awakening_continuity = self._calculate_custom_normalization(pdi_slope, mode='limit_high', sensitivity=10.0, denoise=True)
        awakening_ignite = self._calculate_custom_normalization(pdi_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        pdi_hab = self._calculate_qho_historical_accumulation_buffer(pdi_raw, windows=[21])
        awakening_score = awakening_continuity * awakening_ignite * (1.0 - pdi_hab)
        return self._soft_clip(awakening_score)
    def _calculate_abnormal_energy_overflow(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        mf_eff = raw_data['VPA_MF_ADJUSTED_EFF_D']
        eff_jerk = raw_data.get('JERK_5_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        overflow_ignite = self._calculate_custom_normalization(eff_jerk, mode='limit_high', sensitivity=35.0, denoise=True)
        amount_calm = self._calculate_custom_normalization(raw_data['amount_D'], mode='limit_low', sensitivity=2.0)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        overflow_score = overflow_ignite * amount_calm * price_calm
        return self._soft_clip(overflow_score)
    def _calculate_phase_locked_resonance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        vpa_accel = raw_data['VPA_ACCELERATION_13D']
        price_accel = raw_data['MA_ACCELERATION_EMA_55_D']
        vpa_focus = self._calculate_custom_normalization(vpa_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        price_focus = self._calculate_custom_normalization(price_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        resonance_sim = (vpa_accel * price_accel).rolling(window=5).mean() / (vpa_accel.abs().rolling(window=5).mean() * price_accel.abs().rolling(window=5).mean() + 1e-9)
        vpa_jerk = self._calculate_custom_normalization(raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=60.0, denoise=True)
        plr_score = vpa_focus * price_focus * (0.5 + 0.5 * resonance_sim.clip(0, 1)) * vpa_jerk
        return self._soft_clip(plr_score)
    def _calculate_split_order_pulse_entropy(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        abnormal_raw = raw_data['tick_abnormal_volume_ratio_D']
        abnormal_jerk = raw_data.get('JERK_5_tick_abnormal_volume_ratio_D', pd.Series(0.0, index=df_index))
        jerk_std = abnormal_jerk.rolling(window=8).std()
        jerk_mean = abnormal_jerk.abs().rolling(window=8).mean()
        pulse_orderly = 1.0 / (1.0 + (jerk_std / (jerk_mean + 1e-9)))
        order_score = self._calculate_custom_normalization(pulse_orderly, mode='limit_high', sensitivity=5.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        sope_gain = order_score * price_calm
        return self._soft_clip(sope_gain)
    def _calculate_efficiency_gradient_dissipation(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        mf_eff = raw_data['VPA_MF_ADJUSTED_EFF_D']
        eff_slope = raw_data.get('SLOPE_13_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        eff_accel = raw_data.get('ACCEL_8_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        slope_stability = 1.0 - self._calculate_custom_normalization(eff_slope.rolling(window=8).std(), mode='limit_high', sensitivity=2.0)
        accel_lock = self._calculate_custom_normalization(eff_accel, mode='zero_focus', sensitivity=50.0, denoise=True)
        mf_activity = self._calculate_custom_normalization(raw_data['main_force_activity_index_D'], mode='limit_high', sensitivity=2.0)
        egd_score = slope_stability * accel_lock * (0.8 + 0.2 * mf_activity)
        return self._soft_clip(egd_score)
    def _calculate_potential_well_collapse(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        pit_state = raw_data['STATE_GOLDEN_PIT_D']
        pit_jerk = raw_data.get('JERK_5_STATE_GOLDEN_PIT_D', pd.Series(0.0, index=df_index))
        escape_ignite = self._calculate_custom_normalization(pit_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        trap_lock = self._calculate_custom_normalization(pit_jerk, mode='zero_focus', sensitivity=50.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        well_collapse_score = pit_state * (escape_ignite * 0.8 + (1.0 - trap_lock) * 0.2) * price_calm
        return self._soft_clip(well_collapse_score)
    def _calculate_high_freq_kinetic_gap_fill(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        bias5 = raw_data['BIAS_5_D']
        bias55 = raw_data['BIAS_55_D']
        b5_jerk = raw_data.get('JERK_5_BIAS_5_D', pd.Series(0.0, index=df_index))
        elasticity = self._calculate_custom_normalization(bias5, mode='limit_low', sensitivity=12.0)
        ignite = self._calculate_custom_normalization(b5_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        gap_score = self._calculate_custom_normalization(pd.Series(np.maximum(0, bias55 - bias5), index=df_index), mode='limit_high', sensitivity=5.0)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        final_fill_score = (elasticity * ignite * gap_score) * price_calm
        return self._soft_clip(final_fill_score)
    def _calculate_volatility_vacuum_contraction(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        atr_raw = raw_data['ATR_14_D']
        atr_slope = raw_data.get('SLOPE_13_ATR_14_D', pd.Series(0.0, index=df_index))
        atr_jerk = raw_data.get('JERK_5_ATR_14_D', pd.Series(0.0, index=df_index))
        atr_low_score = self._calculate_custom_normalization(atr_raw, mode='limit_low', sensitivity=1.5)
        decay_purity = self._calculate_custom_normalization(atr_slope, mode='limit_low', sensitivity=10.0, denoise=True)
        vacuum_silence = self._calculate_custom_normalization(atr_jerk, mode='zero_focus', sensitivity=80.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        vvc_score = atr_low_score * decay_purity * vacuum_silence * price_calm
        return self._soft_clip(vvc_score)
    def _calculate_fan_curvature_collapse(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        fan_raw = raw_data['MA_FAN_EFFICIENCY_D']
        fan_accel = raw_data.get('ACCEL_8_MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        fan_jerk = raw_data.get('JERK_5_MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        accel_focus = self._calculate_custom_normalization(fan_accel, mode='zero_focus', sensitivity=50.0, denoise=True)
        jerk_silence = self._calculate_custom_normalization(fan_jerk, mode='zero_focus', sensitivity=70.0, denoise=True)
        fan_high_score = self._calculate_custom_normalization(fan_raw, mode='limit_high', sensitivity=1.2)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        fcc_score = fan_high_score * accel_focus * jerk_silence * price_calm
        return self._soft_clip(fcc_score)
    def _calculate_game_neutralization_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        och_raw = raw_data['OCH_D']
        och_slope = raw_data.get('SLOPE_13_OCH_D', pd.Series(0.0, index=df_index))
        neutralization_focus = self._calculate_custom_normalization(och_slope, mode='zero_focus', sensitivity=45.0, denoise=True)
        och_intensity = self._calculate_custom_normalization(och_raw, mode='limit_high', sensitivity=1.0)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        neutral_score = och_intensity * neutralization_focus * price_calm
        return self._soft_clip(neutral_score)
    def _calculate_oversold_momentum_bipolarization(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        rsi_raw = raw_data['RSI_13_D']
        rsi_accel = raw_data.get('ACCEL_8_RSI_13_D', pd.Series(0.0, index=df_index))
        accel_rev_slope = rsi_accel.diff(5)
        vol = raw_data['volume_D']
        vol_consistency = 1.0 / (1.0 + vol.rolling(window=8).std() / (vol.rolling(window=8).mean() + 1e-9))
        oversold_lock = self._calculate_custom_normalization(rsi_raw, mode='limit_low', sensitivity=0.05)
        bipolar_ratio = self._calculate_custom_normalization(accel_rev_slope * vol_consistency, mode='limit_high', sensitivity=20.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        omb_score = oversold_lock * bipolar_ratio * price_calm
        return self._soft_clip(omb_score)
    def _calculate_kinetic_overflow_veto(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], bipolar_gain: pd.Series) -> pd.Series:
        rsi_raw = raw_data['RSI_13_D']
        rsi_slope = raw_data.get('SLOPE_5_RSI_13_D', pd.Series(0.0, index=df_index))
        veto_l1 = np.where((rsi_raw > 75) & (rsi_slope < 0), 0.8, 1.0)
        vol = raw_data['volume_D']
        vol_spike = vol / (vol.rolling(window=21).mean() + 1e-9)
        price_high = self._calculate_custom_normalization(raw_data['price_vs_ma_21_ratio_D'], mode='limit_high', sensitivity=5.0)
        veto_l2 = np.where((vol_spike > 2.5) & (price_high > 0.8), 0.7, 1.0)
        price_v = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='limit_high', sensitivity=5.0)
        veto_l3 = np.where((price_v > 0.6) & (bipolar_gain < 0.3), 0.6, 1.0)
        final_veto = pd.Series(veto_l1 * veto_l2 * veto_l3, index=df_index)
        return self._soft_clip(final_veto, min_val=0.3, max_val=1.0)
    def _calculate_spatio_temporal_asymmetric_reward(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], resonance_confirm: pd.Series) -> pd.Series:
        close = raw_data.get('close_D', pd.Series(1.0, index=df_index))
        past_ret = close / (close.shift(5) + 1e-9) - 1.0
        hist_hit_mask = resonance_confirm.shift(5).fillna(False)
        expected_gain = (past_ret * hist_hit_mask).rolling(window=120, min_periods=10).mean()
        reward_factor = 1.0 + self._calculate_custom_normalization(expected_gain.clip(lower=0), mode='limit_high', sensitivity=4.0)
        return reward_factor.fillna(1.0)
    def _calculate_extreme_panic_resonance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        pain_jerk = raw_data.get('JERK_5_pain_index_proxy', pd.Series(0.0, index=df_index))
        panic_burst = self._calculate_custom_normalization(pain_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        pit_state = raw_data['STATE_GOLDEN_PIT_D']
        resonance_score = panic_burst * pit_state
        return self._soft_clip(resonance_score)
    def _calculate_adaptive_phase_transition_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        price_v = raw_data['price_slope_raw']
        noise_cv = price_v.rolling(window=250, min_periods=60).std() / (price_v.rolling(window=250, min_periods=60).mean().abs() + 1e-9)
        adaptive_threshold = 0.45 * (0.8 + 0.5 * self._calculate_custom_normalization(noise_cv, mode='limit_high', sensitivity=2.0))
        return adaptive_threshold.fillna(0.45)
    def _calculate_mean_reversion_kinetic_bias(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        bias144 = raw_data['price_vs_ma_144_ratio']
        accel144 = raw_data.get('ACCEL_8_price_vs_ma_144_ratio', pd.Series(0.0, index=df_index))
        depth_reward = self._calculate_custom_normalization(bias144, mode='limit_low', sensitivity=5.0)
        slingshot_ignite = self._calculate_custom_normalization(accel144, mode='limit_high', sensitivity=15.0, denoise=True)
        bias_factor = 1.0 + 0.25 * np.sqrt(depth_reward * slingshot_ignite)
        return bias_factor.fillna(1.0)














