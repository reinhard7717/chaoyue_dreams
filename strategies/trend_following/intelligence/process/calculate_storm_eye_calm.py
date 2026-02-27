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
    【V11.0.0 · 风暴眼寂静 · 作用域修复与张量安全对齐终极版】
    PROCESS_META_STORM_EYE_CALM
    - 内存域固化: 修复 _get_raw_and_atomic_data 中 df_index 作用域越界泄漏引发的 NameError 熔断。
    - 张量安全对齐: _lp_norm_fusion 强制引入 df_index 进行 reindex 对齐，免疫标量与序列混算崩溃。
    - 级联污染切除: 噪音门限(Threshold Gate)引入底层下界 np.maximum(..., 1e-5)，治愈 0/0=NaN。
    - 量纲嗅探映射: 自动识别极大极小值的绝对数量级，执行动态相对映射，免疫指数爆炸归零。
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
        is_debug_enabled, probe_ts = self._get_debug_info(df, method_name)
        _probe_data = {}
        raw_data = self._get_raw_and_atomic_data(df, method_name, params, _probe_data, probe_ts)
        energy_score = self._calculate_energy_compression_component(df_index, raw_data, {}, params['energy_compression_weights'], _probe_data, probe_ts)
        volume_score = self._calculate_volume_exhaustion_component(df_index, raw_data, {}, params['volume_exhaustion_weights'], _probe_data, probe_ts)
        intent_score, intent_dict = self._calculate_main_force_covert_intent_component(df_index, raw_data, {}, params['main_force_covert_intent_weights'], {}, _probe_data, probe_ts)
        sentiment_score = self._calculate_subdued_market_sentiment_component(df_index, raw_data, params['subdued_market_sentiment_weights'], 21, 55, 1.0, 0.2, _probe_data, probe_ts)
        readiness_score = self._calculate_breakout_readiness_component(df_index, raw_data, params['breakout_readiness_weights'], _probe_data, probe_ts)
        dynamic_threshold = self._calculate_adaptive_phase_transition_threshold(df_index, raw_data, _probe_data, probe_ts)
        component_scores = {
            'energy': energy_score * self._calculate_fermi_dirac_gate(energy_score, threshold=dynamic_threshold, beta=12.0),
            'volume': volume_score * self._calculate_fermi_dirac_gate(volume_score, threshold=dynamic_threshold, beta=12.0),
            'intent': intent_score * self._calculate_fermi_dirac_gate(intent_score, threshold=dynamic_threshold, beta=12.0),
            'sentiment': sentiment_score * self._calculate_fermi_dirac_gate(sentiment_score, threshold=dynamic_threshold, beta=12.0),
            'readiness': readiness_score * self._calculate_fermi_dirac_gate(readiness_score, threshold=dynamic_threshold, beta=12.0)
        }
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Energy (能量门控分)", component_scores['energy'], probe_ts)
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Volume (量能门控分)", component_scores['volume'], probe_ts)
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Intent (意图门控分)", component_scores['intent'], probe_ts)
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Sentiment (情绪门控分)", component_scores['sentiment'], probe_ts)
        self._log_probe(_probe_data, "【05. 五大核心维度 (Domains)】", "Readiness (准备门控分)", component_scores['readiness'], probe_ts)
        final_fusion_score = self._perform_final_fusion(df_index, component_scores, raw_data, _probe_data, probe_ts)
        regulator_modulator = self._calculate_market_regulator_modulator(df_index, raw_data, params, _probe_data, probe_ts)
        raw_final_score = final_fusion_score * regulator_modulator
        ewd_factor = self._calculate_consensus_entropy(component_scores, _probe_data, probe_ts)
        resonance_confirm = pd.Series(np.clip((raw_final_score - 0.4) * 10.0, 0, 1) * np.clip((ewd_factor - 0.7) * 10.0, 0, 1), index=df_index)
        latch_multiplier = pd.Series(np.where(resonance_confirm.rolling(5, min_periods=1).sum() >= 2.5, 1.2, 1.0), index=df_index)
        latched_score = raw_final_score.rolling(3, min_periods=1).mean().fillna(raw_final_score) * latch_multiplier
        veto_factor = self._calculate_kinetic_overflow_veto(df_index, raw_data, self._calculate_oversold_momentum_bipolarization(df_index, raw_data, _probe_data, probe_ts), _probe_data, probe_ts)
        reward_factor = self._calculate_spatio_temporal_asymmetric_reward(df_index, raw_data, resonance_confirm, _probe_data, probe_ts)
        mrkb_factor = self._calculate_mean_reversion_kinetic_bias(df_index, raw_data, _probe_data, probe_ts)
        tes_factor = self._calculate_trend_energy_shearing(df_index, raw_data, _probe_data, probe_ts)
        final_latched_score = (latched_score * veto_factor * reward_factor * mrkb_factor * tes_factor).clip(0, 1)
        self._log_probe(_probe_data, "【08. 最终归一化输出 (Final)】", "Raw_Final_Score (原始分)", raw_final_score, probe_ts)
        self._log_probe(_probe_data, "【08. 最终归一化输出 (Final)】", "Latched_Score (锁存稳态分)", latched_score, probe_ts)
        self._log_probe(_probe_data, "【08. 最终归一化输出 (Final)】", "Final_StormEye_Score (最终破局点)", final_latched_score, probe_ts)
        if is_debug_enabled and probe_ts is not None:
            self._print_comprehensive_probe(_probe_data, probe_ts, method_name, final_latched_score)
        return final_latched_score.astype(np.float32)
    def _log_probe(self, _probe_data: Dict, category: str, key: str, value: Any, probe_ts: pd.Timestamp):
        if probe_ts is None: return
        if isinstance(value, pd.Series): val = value.loc[probe_ts] if probe_ts in value.index else np.nan
        else: val = value
        if category not in _probe_data: _probe_data[category] = {}
        _probe_data[category][key] = val
    def _print_comprehensive_probe(self, _probe_data: Dict, probe_ts: pd.Timestamp, method_name: str, final_score: pd.Series):
        print(f"\n{'='*20} [{method_name} 全链路量子探针] @ {probe_ts.strftime('%Y-%m-%d')} {'='*20}")
        for category in sorted(_probe_data.keys()):
            print(f"[{category}]")
            for k, v in _probe_data[category].items():
                if isinstance(v, (float, np.float32, np.float64)): print(f"  ├─ {k:<40}: {v:.4f}")
                else: print(f"  ├─ {k:<40}: {v}")
        print(f"{'-'*85}")
        print(f"  >>> 破局极值最终得分: {final_score.loc[probe_ts]:.4f} <<<")
        print(f"{'='*85}\n")
    def _check_and_fill_data_existence(self, df: pd.DataFrame, params: Dict):
        req_signals = self._get_required_signals(params)
        missing = [c for c in req_signals if c not in df.columns]
        if missing:
            print(f"【V11.0.0 探针警报】风暴眼基底特征断层，缺失列: {missing}。系统已启动拉普拉斯安全回退机制！")
    def _apply_threshold_gate(self, series: pd.Series, window: int = 21) -> pd.Series:
        noise_floor = np.maximum(series.rolling(window=window, min_periods=5).std().ffill().fillna(1e-5), 1e-5)
        return series * np.tanh((series / (noise_floor * 2.0))**2)
    def _safe_diff(self, series: pd.Series, period: int) -> pd.Series:
        return self._apply_threshold_gate(series.ffill().diff(period).fillna(0.0))
    def _lp_norm_fusion(self, df_index: pd.Index, scores: List[Any], weights: List[float], p: float = 2.0) -> pd.Series:
        valid_scores, valid_weights = [], []
        for s, w in zip(scores, weights):
            if isinstance(s, pd.Series):
                valid_scores.append(s.reindex(df_index).fillna(0.0).clip(0, 1) * 0.99 + 0.01)
            else:
                valid_scores.append(pd.Series(s, index=df_index).fillna(0.0).clip(0, 1) * 0.99 + 0.01)
            valid_weights.append(w)
        weight_sum = sum(valid_weights) + 1e-9
        norm_weights = [w / weight_sum for w in valid_weights]
        sum_pow = pd.Series(0.0, index=df_index)
        for s, w in zip(valid_scores, norm_weights): sum_pow += w * (s ** p)
        return (sum_pow ** (1.0 / p)).clip(0, 1)
    def _calculate_fermi_dirac_gate(self, score_series: pd.Series, threshold: float | pd.Series = 0.5, beta: float = 10.0) -> pd.Series:
        if isinstance(threshold, pd.Series): threshold = threshold.reindex(score_series.index).fillna(0.5)
        gate = 1.0 / (1.0 + np.exp(beta * (threshold - score_series)))
        return 0.5 + 0.5 * gate
    def _calculate_custom_normalization(self, series: pd.Series, mode: str, sensitivity: float = 1.0, window: int = 55, denoise: bool = False, atr_series: Optional[pd.Series] = None) -> pd.Series:
        if not isinstance(series, pd.Series): series = pd.Series(float(series), index=getattr(self, 'last_df_index', []))
        series = series.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if denoise and len(series) >= 21:
            if atr_series is not None: series = series / (atr_series + 1e-9)
            else: series = self._apply_threshold_gate(series, window=21)
        if mode == 'limit_high': return pd.Series(np.tanh(np.maximum(0, series * sensitivity)), index=series.index)
        elif mode == 'limit_low': return pd.Series(np.exp(-np.maximum(0, series * sensitivity)), index=series.index)
        elif mode == 'negative_extreme': return pd.Series(np.tanh(np.maximum(0, -series * sensitivity)), index=series.index)
        elif mode == 'zero_focus': return pd.Series(np.exp(- (series * sensitivity) ** 2), index=series.index)
        elif mode == 'relative_rank':
            roll_min = series.rolling(window=window, min_periods=1).min()
            roll_max = series.rolling(window=window, min_periods=1).max()
            return ((series - roll_min) / (roll_max - roll_min + 1e-9)).clip(0, 1)
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
            'MA_POTENTIAL_TENSION_INDEX_D', 'MA_COHERENCE_RESONANCE_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', 'BBW_21_2.0_D', 'chip_concentration_ratio_D', 'concentration_entropy_D', 'PRICE_ENTROPY_D', 'GEOM_ARC_CURVATURE_D', 'dynamic_consolidation_duration_D', 'turnover_rate_f_D', 'volume_D', 'intraday_trough_filling_degree_D', 'tick_abnormal_volume_ratio_D', 'afternoon_flow_ratio_D', 'absorption_energy_D', 'stealth_flow_ratio_D', 'tick_clustering_index_D', 'accumulation_signal_score_D', 'SMART_MONEY_HM_NET_BUY_D', 'HM_ACTIVE_TOP_TIER_D', 'net_mf_amount_D', 'profit_ratio_D', 'winner_rate_D', 'market_sentiment_score_D', 'breakout_potential_D', 'breakout_penalty_score_D', 'resistance_strength_D', 'GEOM_REG_R2_D', 'GEOM_REG_SLOPE_D', 'ATR_14_D', 'chip_stability_D', 'ADX_14_D', 'flow_impact_ratio_D', 'industry_preheat_score_D', 'industry_rank_accel_D', 'industry_strength_rank_D', 'trend_confirmation_score_D', 'main_force_activity_index_D', 'intraday_cost_center_migration_D', 'migration_convergence_ratio_D', 'tick_chip_balance_ratio_D', 'VPA_EFFICIENCY_D', 'VPA_MF_ADJUSTED_EFF_D', 'VPA_ACCELERATION_13D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'OCH_ACCELERATION_D', 'OCH_D', 'PDI_14_D', 'NDI_14_D', 'price_vs_ma_21_ratio_D', 'price_vs_ma_55_ratio_D', 'HM_COORDINATED_ATTACK_D', 'TURNOVER_STABILITY_INDEX_D', 'amount_D', 'HM_ACTIVE_ANY_D', 'BIAS_55_D', 'MA_ACCELERATION_EMA_55_D', 'STATE_GOLDEN_PIT_D', 'BIAS_5_D', 'MA_FAN_EFFICIENCY_D', 'RSI_13_D', 'close', 'MA_144_D', 'chip_entropy_D', 'pressure_trapped_D', 'consolidation_quality_score_D', 'net_energy_flow_D', 'intraday_chip_game_index_D', 'pattern_confidence_D', 'breakout_quality_score_D', 'breakout_chip_score_D', 'MA_55_D', 'MA_21_D', 'MA_5_D'
        ]
        return list(set(required_signals))
    def _get_raw_and_atomic_data(self, df: pd.DataFrame, method_name: str, params: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> Dict[str, pd.Series]:
        df_index = df.index
        raw_data = {col: df.get(col, pd.Series(0.0, index=df_index)).ffill().fillna(0.0) for col in self._get_required_signals(params)}
        raw_data['close_D'] = df.get('close_D', df.get('close', pd.Series(0.0, index=df_index))).ffill().fillna(0.0)
        close_base = raw_data['close_D'] + 1e-9
        amount_ma21 = raw_data['amount_D'].rolling(21, min_periods=1).mean() + 1e-9
        vol_ma21 = raw_data['volume_D'].rolling(21, min_periods=1).mean() + 1e-9
        raw_data['amount_D'] = raw_data['amount_D'] / amount_ma21
        raw_data['volume_D'] = raw_data['volume_D'] / vol_ma21
        raw_data['net_energy_flow_D'] = raw_data['net_energy_flow_D'] / amount_ma21
        raw_data['net_mf_amount_D'] = raw_data['net_mf_amount_D'] / amount_ma21
        raw_data['ATR_14_D'] = raw_data['ATR_14_D'] / close_base
        raw_data['GEOM_REG_SLOPE_D'] = raw_data['GEOM_REG_SLOPE_D'] / close_base
        raw_data['price_vs_ma_21_ratio_D'] = df.get('price_vs_ma_21_ratio_D', raw_data['close_D'] / (df.get('MA_21_D', raw_data['close_D']) + 1e-9)).ffill().fillna(1.0)
        raw_data['price_vs_ma_55_ratio_D'] = df.get('price_vs_ma_55_ratio_D', raw_data['close_D'] / (df.get('MA_55_D', raw_data['close_D']) + 1e-9)).ffill().fillna(1.0)
        raw_data['BIAS_55_D'] = (raw_data['close_D'] - df.get('MA_55_D', raw_data['close_D'])) / (df.get('MA_55_D', raw_data['close_D']) + 1e-9)
        raw_data['BIAS_5_D'] = (raw_data['close_D'] - df.get('MA_5_D', raw_data['close_D'])) / (df.get('MA_5_D', raw_data['close_D']) + 1e-9)
        scale_100_cols = ['turnover_rate_f_D', 'pattern_confidence_D', 'breakout_quality_score_D', 'breakout_chip_score_D', 'consolidation_quality_score_D', 'accumulation_signal_score_D', 'intraday_chip_game_index_D', 'tick_clustering_index_D', 'NDI_14_D', 'PDI_14_D', 'RSI_13_D', 'ADX_14_D', 'winner_rate_D', 'profit_ratio_D', 'HM_ACTIVE_ANY_D', 'HM_ACTIVE_TOP_TIER_D', 'STATE_GOLDEN_PIT_D', 'chip_stability_D', 'intraday_trough_filling_degree_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'HM_COORDINATED_ATTACK_D', 'pressure_trapped_D', 'breakout_penalty_score_D', 'market_sentiment_score_D']
        for col in scale_100_cols:
            if col in raw_data and len(raw_data[col]) > 0:
                if raw_data[col].abs().max() > 2.0: raw_data[col] = raw_data[col] / 100.0
        for k in ['close_D', 'pattern_confidence_D', 'breakout_quality_score_D', 'turnover_rate_f_D', 'ADX_14_D', 'VPA_EFFICIENCY_D', 'PRICE_ENTROPY_D', 'amount_D', 'ATR_14_D']:
            if k in raw_data: self._log_probe(_probe_data, "【01. 原始核心数据 (Raw Data)】", k, raw_data[k], probe_ts)
        deriv_cols = ['VPA_ACCELERATION_13D', 'VPA_MF_ADJUSTED_EFF_D', 'tick_abnormal_volume_ratio_D', 'MA_ACCELERATION_EMA_55_D', 'PRICE_ENTROPY_D', 'STATE_GOLDEN_PIT_D', 'BIAS_55_D', 'NDI_14_D', 'PDI_14_D', 'breakout_penalty_score_D', 'RSI_13_D', 'OCH_D', 'ATR_14_D', 'MA_FAN_EFFICIENCY_D', 'HM_ACTIVE_ANY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 'HM_COORDINATED_ATTACK_D', 'BIAS_5_D', 'market_sentiment_score_D', 'ADX_14_D', 'profit_ratio_D', 'chip_entropy_D', 'net_energy_flow_D', 'pattern_confidence_D', 'breakout_quality_score_D', 'consolidation_quality_score_D', 'OCH_ACCELERATION_D', 'VPA_EFFICIENCY_D', 'intraday_cost_center_migration_D', 'TURNOVER_STABILITY_INDEX_D', 'concentration_entropy_D', 'industry_rank_accel_D']
        for col in deriv_cols:
            if col in raw_data:
                s13 = self._safe_diff(raw_data[col], 13)
                a8 = self._safe_diff(s13, 8)
                j5 = self._safe_diff(a8, 5)
                raw_data[f'SLOPE_13_{col}'] = s13
                raw_data[f'ACCEL_8_{col}'] = a8
                raw_data[f'JERK_5_{col}'] = j5
                raw_data[f'SLOPE_5_{col}'] = self._safe_diff(raw_data[col], 5)
                if col in ['VPA_ACCELERATION_13D', 'PRICE_ENTROPY_D', 'pattern_confidence_D', 'OCH_ACCELERATION_D']:
                    self._log_probe(_probe_data, "【02. 微积分动力学 (Kinematics)】", f"SLOPE_13_{col}", s13, probe_ts)
                    self._log_probe(_probe_data, "【02. 微积分动力学 (Kinematics)】", f"JERK_5_{col}", j5, probe_ts)
        ma144 = df.get('MA_144_D', raw_data['close_D']).ffill().fillna(raw_data['close_D'])
        raw_data['price_vs_ma_144_ratio'] = raw_data['close_D'] / (ma144 + 1e-9)
        raw_data['ACCEL_8_price_vs_ma_144_ratio'] = self._safe_diff(self._safe_diff(raw_data['price_vs_ma_144_ratio'], 13), 8)
        raw_data['pain_index_proxy'] = pd.Series(1.0 - raw_data['profit_ratio_D'], index=df_index)
        raw_data['JERK_5_pain_index_proxy'] = pd.Series(raw_data.get('JERK_5_profit_ratio_D', pd.Series(0.0, index=df_index)) * -1.0, index=df_index)
        raw_data['price_slope_raw'] = raw_data['close_D'].pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        self._log_probe(_probe_data, "【02. 微积分动力学 (Kinematics)】", "price_slope_raw", raw_data['price_slope_raw'], probe_ts)
        self._log_probe(_probe_data, "【02. 微积分动力学 (Kinematics)】", "JERK_5_pain_index_proxy", raw_data['JERK_5_pain_index_proxy'], probe_ts)
        return raw_data
    def _calculate_qho_historical_accumulation_buffer(self, daily_series: pd.Series, windows: list[int] = [13, 21, 34, 55], name: str = "", _probe_data: Dict = None, probe_ts: pd.Timestamp = None) -> pd.Series:
        buffers = []
        for w in windows:
            historical_stock = daily_series.abs().rolling(window=w, min_periods=1).mean() + 1e-9
            incremental_impact = daily_series / historical_stock
            buffer_factor = pd.Series(np.tanh(np.maximum(0, incremental_impact) * 0.5), index=daily_series.index)
            buffers.append(buffer_factor)
        res = pd.concat(buffers, axis=1).mean(axis=1).fillna(0.0)
        if name and _probe_data is not None and probe_ts is not None:
            self._log_probe(_probe_data, "【03. 时空存量缓冲 (HAB)】", f"HAB_{name}", res, probe_ts)
        return res
    def _calculate_breakout_conviction_proxy(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        pattern_conf = self._calculate_custom_normalization(raw_data.get('pattern_confidence_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.5)
        pattern_slope = self._calculate_custom_normalization(raw_data.get('SLOPE_13_pattern_confidence_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=5.0, denoise=True)
        breakout_qual = self._calculate_custom_normalization(raw_data.get('breakout_quality_score_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.5)
        breakout_chip = self._calculate_custom_normalization(raw_data.get('breakout_chip_score_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        qual_hab = self._calculate_qho_historical_accumulation_buffer(raw_data.get('breakout_quality_score_D', pd.Series(0.0, index=df_index)), windows=[13, 21], name="BreakoutQual", _probe_data=_probe_data, probe_ts=probe_ts)
        final_conviction = self._lp_norm_fusion(df_index, [pattern_conf, breakout_qual, breakout_chip, pattern_slope, qual_hab], [0.3, 0.25, 0.15, 0.15, 0.15], p=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Conviction_Proxy (突破隧穿代理)", final_conviction, probe_ts)
        return final_conviction
    def _calculate_energy_compression_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        fcc_factor = self._calculate_fan_curvature_collapse(df_index, raw_data, _probe_data, probe_ts)
        vvc_factor = self._calculate_volatility_vacuum_contraction(df_index, raw_data, _probe_data, probe_ts)
        lrf_score = self._calculate_linear_resonance_failure(df_index, raw_data, _probe_data, probe_ts)
        struct_quality = self._calculate_custom_normalization(raw_data['chip_stability_D'], mode='limit_high', sensitivity=1.5)
        entropy_gain = self._calculate_custom_normalization(raw_data.get('SLOPE_13_chip_entropy_D', pd.Series(0.0, index=df_index)), mode='negative_extreme', sensitivity=10.0, denoise=True)
        vpa_accel = raw_data.get('ACCEL_8_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        vpa_jerk = raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        phase_space_dist = pd.Series(np.sqrt(np.square(vpa_accel) + np.square(vpa_jerk)), index=df_index)
        phase_attractor = pd.Series(np.exp(-phase_space_dist * 5.0), index=df_index)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Struct_Quality (筹码稳固度)", struct_quality, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Entropy_Gain (熵减红利)", entropy_gain, probe_ts)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Phase_Attractor (相空间吸引子)", phase_attractor, probe_ts)
        final_energy = self._lp_norm_fusion(df_index, [fcc_factor, vvc_factor, lrf_score, entropy_gain, phase_attractor, struct_quality], [0.2, 0.15, 0.15, 0.2, 0.15, 0.15], p=2.0)
        return final_energy
    def _calculate_volume_exhaustion_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        turnover_score = self._calculate_custom_normalization(raw_data['turnover_rate_f_D'], mode='limit_low', sensitivity=25.0)
        trough_fill = self._calculate_custom_normalization(raw_data['intraday_trough_filling_degree_D'], mode='limit_high', sensitivity=3.0)
        mdb_factor = self._calculate_momentum_dissipation_balance(df_index, raw_data, _probe_data, probe_ts)
        solid_factor = self._calculate_liquidity_solidification_threshold(df_index, raw_data, _probe_data, probe_ts)
        vpa_jerk = self._calculate_custom_normalization(raw_data.get('JERK_5_VPA_EFFICIENCY_D', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=20.0, denoise=True)
        mf_eff = self._calculate_custom_normalization(raw_data['VPA_MF_ADJUSTED_EFF_D'], mode='limit_high', sensitivity=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Turnover_Score (极小换手得分)", turnover_score, probe_ts)
        final_vol = self._lp_norm_fusion(df_index, [turnover_score, trough_fill, mdb_factor, solid_factor, vpa_jerk, mf_eff], [0.25, 0.2, 0.15, 0.15, 0.15, 0.1], p=2.0)
        return final_vol
    def _calculate_main_force_covert_intent_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, ambiguity_weights: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        chf_base = raw_data['SMART_MONEY_HM_COORDINATED_ATTACK_D'].rolling(window=8, min_periods=1).mean().fillna(0)
        chf_jerk_score = self._calculate_custom_normalization(raw_data.get('JERK_5_SMART_MONEY_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=25.0, denoise=True)
        htc_factor = self._calculate_hunting_temporal_coherence(df_index, raw_data, _probe_data, probe_ts)
        stealth_score = self._calculate_custom_normalization(raw_data['stealth_flow_ratio_D'], mode='limit_high', sensitivity=4.0)
        migration_accel = self._calculate_custom_normalization(raw_data.get('ACCEL_8_intraday_cost_center_migration_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=20.0)
        mf_hab = self._calculate_qho_historical_accumulation_buffer(raw_data['net_mf_amount_D'], windows=[21, 34], name="Net_MF", _probe_data=_probe_data, probe_ts=probe_ts)
        energy_flow = self._calculate_custom_normalization(raw_data.get('net_energy_flow_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=1.5)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Stealth_Score (隐秘潜行占比)", stealth_score, probe_ts)
        final_intent = self._lp_norm_fusion(df_index, [stealth_score, migration_accel, chf_base, chf_jerk_score, energy_flow, htc_factor, mf_hab], [0.15, 0.15, 0.1, 0.1, 0.15, 0.2, 0.15], p=2.0)
        return final_intent, {"stealth_score": stealth_score, "htc_factor": htc_factor}
    def _calculate_subdued_market_sentiment_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, sentiment_volatility_window: int, long_term_sentiment_window: int, sentiment_neutral_range: float, sentiment_pendulum_neutral_range: float, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        pain_score = self._calculate_custom_normalization(raw_data['pain_index_proxy'], mode='limit_high', sensitivity=3.0)
        despair_burst = self._calculate_custom_normalization(raw_data.get('JERK_5_pain_index_proxy', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=20.0, denoise=True)
        short_exhaustion = self._calculate_short_exhaustion_divergence(df_index, raw_data, _probe_data, probe_ts)
        bipolar_gain = self._calculate_oversold_momentum_bipolarization(df_index, raw_data, _probe_data, probe_ts)
        panic_resonance = self._calculate_extreme_panic_resonance(df_index, raw_data, _probe_data, probe_ts)
        order_gain = self._calculate_micro_order_gain(df_index, raw_data, _probe_data, probe_ts)
        cleanse_score = self._calculate_custom_normalization(raw_data['winner_rate_D'], mode='limit_low', sensitivity=15.0)
        trapped_pressure = self._calculate_custom_normalization(raw_data['pressure_trapped_D'], mode='limit_high', sensitivity=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Pain_Score (散户痛感释放)", pain_score, probe_ts)
        final_sentiment = self._lp_norm_fusion(df_index, [pain_score, cleanse_score, trapped_pressure, order_gain, short_exhaustion, bipolar_gain, panic_resonance, despair_burst], [0.15, 0.1, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1], p=2.0)
        return final_sentiment
    def _calculate_breakout_readiness_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        grp_score = self._calculate_gravitational_regression_pull(df_index, raw_data, _probe_data, probe_ts)
        plr_score = self._calculate_phase_locked_resonance(df_index, raw_data, _probe_data, probe_ts)
        sed_score = self._calculate_short_exhaustion_divergence(df_index, raw_data, _probe_data, probe_ts)
        ssd_score = self._calculate_seat_scatter_decay(df_index, raw_data, _probe_data, probe_ts)
        egd_score = self._calculate_efficiency_gradient_dissipation(df_index, raw_data, _probe_data, probe_ts)
        sope_score = self._calculate_split_order_pulse_entropy(df_index, raw_data, _probe_data, probe_ts)
        aeo_score = self._calculate_abnormal_energy_overflow(df_index, raw_data, _probe_data, probe_ts)
        neutral_score = self._calculate_game_neutralization_modulator(df_index, raw_data, _probe_data, probe_ts)
        aded_score = self._calculate_amount_distribution_entropy_delta(df_index, raw_data, _probe_data, probe_ts)
        well_collapse = self._calculate_potential_well_collapse(df_index, raw_data, _probe_data, probe_ts)
        long_awakening = self._calculate_long_awakening_threshold(df_index, raw_data, _probe_data, probe_ts)
        stress_test = self._calculate_level_stress_test_modulator(df_index, raw_data, _probe_data, probe_ts)
        consolidation = self._calculate_custom_normalization(raw_data.get('consolidation_quality_score_D', pd.Series(0.0, index=df_index)), mode='limit_high', sensitivity=2.0)
        conviction_proxy = self._calculate_breakout_conviction_proxy(df_index, raw_data, _probe_data, probe_ts)
        momentum_part = self._lp_norm_fusion(df_index, [grp_score, plr_score], [0.5, 0.5], p=2.0)
        friction_part = self._lp_norm_fusion(df_index, [sed_score, ssd_score], [0.5, 0.5], p=2.0)
        quality_part = self._lp_norm_fusion(df_index, [egd_score, sope_score, aeo_score], [0.35, 0.35, 0.3], p=2.0)
        state_part = self._lp_norm_fusion(df_index, [well_collapse, long_awakening, aded_score, stress_test, neutral_score, consolidation, conviction_proxy], [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3], p=2.0)
        acc_hab = self._calculate_qho_historical_accumulation_buffer(raw_data['accumulation_signal_score_D'], windows=[21, 34], name="Accum_Signal", _probe_data=_probe_data, probe_ts=probe_ts)
        readiness = self._lp_norm_fusion(df_index, [momentum_part, friction_part, quality_part, state_part, acc_hab], [0.15, 0.15, 0.2, 0.4, 0.1], p=2.0)
        return readiness
    def _perform_final_fusion(self, df_index: pd.Index, component_scores: dict[str, pd.Series], raw_data: dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        scores_list = [component_scores['energy'], component_scores['volume'], component_scores['intent'], component_scores['sentiment'], component_scores['readiness']]
        base_score = self._lp_norm_fusion(df_index, scores_list, [0.2, 0.2, 0.2, 0.15, 0.25], p=2.0)
        ext_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=30.0, denoise=True)
        struct_boost = self._calculate_custom_normalization(raw_data['accumulation_signal_score_D'], mode='limit_high', sensitivity=1.0)
        hunting_boost = self._calculate_custom_normalization(raw_data['SMART_MONEY_HM_COORDINATED_ATTACK_D'].rolling(8, min_periods=1).mean().fillna(0), mode='limit_high', sensitivity=2.0)
        multiplier = pd.Series(1.0 + 0.2 * struct_boost + 0.15 * ext_calm + 0.15 * hunting_boost, index=df_index)
        final_score = (base_score * multiplier).clip(0, 1)
        self._log_probe(_probe_data, "【06. 最终融合参数 (Final_Fusion_Params)】", "Lp_Base_Core_Score (基础域合分)", base_score, probe_ts)
        self._log_probe(_probe_data, "【06. 最终融合参数 (Final_Fusion_Params)】", "Additive_Multiplier (增强乘数)", multiplier, probe_ts)
        self._log_probe(_probe_data, "【06. 最终融合参数 (Final_Fusion_Params)】", "Pre_Modulator_Score (预调节分)", final_score, probe_ts)
        return final_score
    def _calculate_market_regulator_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], params: Dict, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        sector_preheat = raw_data['industry_preheat_score_D']
        sector_hab = self._calculate_qho_historical_accumulation_buffer(sector_preheat, windows=[13, 21], name="Sector_Preheat", _probe_data=_probe_data, probe_ts=probe_ts)
        sector_jerk = raw_data.get('JERK_5_industry_rank_accel_D', pd.Series(0.0, index=df_index))
        clean_sector_jerk = pd.Series(np.where(sector_jerk > sector_jerk.rolling(21, min_periods=1).std().fillna(0.0), sector_jerk, 0.0), index=df_index)
        sector_ignite_score = self._calculate_custom_normalization(clean_sector_jerk, mode='limit_high', sensitivity=10.0)
        stock_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=30.0, denoise=True)
        macro_resonance = self._lp_norm_fusion(df_index, [sector_ignite_score, stock_calm], [0.5, 0.5], p=2.0)
        adx_raw = raw_data['ADX_14_D']
        adx_supp = pd.Series(1.0 / (1.0 + np.exp(40.0 * (adx_raw - 0.28))), index=df_index)
        final_modulator = self._lp_norm_fusion(df_index, [sector_hab, macro_resonance, adx_supp], [0.3, 0.5, 0.2], p=2.0)
        adj_modulator = pd.Series(final_modulator * 1.5 + 0.5, index=df_index)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Market_Regulator (宏观起爆乘数)", adj_modulator, probe_ts)
        return adj_modulator
    def _calculate_trend_energy_shearing(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        adx_raw = raw_data['ADX_14_D']
        adx_accel = raw_data.get('ACCEL_8_ADX_14_D', pd.Series(0.0, index=df_index))
        high_context = self._calculate_custom_normalization(pd.Series(adx_raw - 0.35, index=df_index), mode='limit_high', sensitivity=50.0)
        shearing_ignite = self._calculate_custom_normalization(adx_accel, mode='negative_extreme', sensitivity=15.0, denoise=True)
        shearing_factor = pd.Series(1.0 + 0.2 * self._lp_norm_fusion(df_index, [high_context, shearing_ignite], [0.5, 0.5], p=2.0), index=df_index).fillna(1.0)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "TES_Factor (趋势能量剪切)", shearing_factor, probe_ts)
        return shearing_factor
    def _calculate_consensus_entropy(self, scores_dict: dict[str, pd.Series], _probe_data: Dict = None, probe_ts: pd.Timestamp = None) -> pd.Series:
        df_scores = pd.concat(scores_dict.values(), axis=1)
        dispersion = df_scores.std(axis=1).fillna(1.0)
        corr_matrix = df_scores.rolling(window=5, min_periods=1).corr()
        coherence = corr_matrix.groupby(level=0).mean().mean(axis=1).fillna(0.0)
        disp_decay = pd.Series(np.exp(- (dispersion * 2.5) ** 2), index=df_scores.index)
        final_decay = pd.Series(disp_decay * (0.6 + 0.4 * coherence.clip(0, 1)), index=df_scores.index).clip(0, 1)
        if _probe_data is not None and probe_ts is not None:
            self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "EWD_Consensus (共振互信息熵)", final_decay, probe_ts)
        return final_decay
    def _calculate_pressure_backtest_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        penalty_raw = raw_data['breakout_penalty_score_D']
        penalty_slope = raw_data.get('SLOPE_13_breakout_penalty_score_D', pd.Series(0.0, index=df_index))
        penalty_hab = self._calculate_qho_historical_accumulation_buffer(penalty_raw, windows=[21])
        resistance_intensity = self._calculate_custom_normalization(penalty_raw * (1.0 + np.maximum(0, penalty_slope)), mode='limit_high', sensitivity=1.5)
        price_v = raw_data['price_slope_raw']
        backtest_factor = pd.Series(1.0 - (resistance_intensity * np.tanh(np.maximum(0, price_v) * 10.0)), index=df_index)
        final_modulator = (backtest_factor * (1.0 - penalty_hab)) + penalty_hab
        return final_modulator.clip(0.2, 1.0)
    def _calculate_level_stress_test_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        och_jerk = raw_data.get('JERK_5_OCH_ACCELERATION_D', pd.Series(0.0, index=df_index))
        och_jerk_score = self._calculate_custom_normalization(och_jerk, mode='limit_high', sensitivity=20.0, denoise=True)
        res_strength = raw_data['resistance_strength_D']
        ma21_proximity = pd.Series(1.0 - np.minimum(1.0, np.abs(raw_data.get('price_vs_ma_21_ratio_D', pd.Series(1.0, index=df_index)) - 1.0) * 20.0), index=df_index)
        ma55_proximity = pd.Series(1.0 - np.minimum(1.0, np.abs(raw_data.get('price_vs_ma_55_ratio_D', pd.Series(1.0, index=df_index)) - 1.0) * 20.0), index=df_index)
        level_weight = pd.Series(np.maximum(ma21_proximity, ma55_proximity), index=df_index)
        stress_test_score = pd.Series((och_jerk_score * level_weight * self._calculate_custom_normalization(res_strength, mode='limit_high', sensitivity=1.0)).clip(0, 1), index=df_index)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Stress_Test (关键位极限测压)", stress_test_score, probe_ts)
        return stress_test_score
    def _calculate_linear_resonance_failure(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        r2_raw = raw_data['GEOM_REG_R2_D']
        r2_accel = raw_data.get('ACCEL_8_GEOM_REG_R2_D', pd.Series(0.0, index=df_index))
        r2_hab = self._calculate_qho_historical_accumulation_buffer(r2_raw, windows=[21], name="Geom_R2", _probe_data=_probe_data, probe_ts=probe_ts)
        failure_burst = self._calculate_custom_normalization(r2_accel.abs(), mode='limit_high', sensitivity=15.0, denoise=True)
        reg_slope = raw_data['GEOM_REG_SLOPE_D']
        slope_calm = self._calculate_custom_normalization(reg_slope, mode='zero_focus', sensitivity=40.0, denoise=True)
        failure_score = self._lp_norm_fusion(df_index, [r2_hab, failure_burst, slope_calm], [0.3, 0.4, 0.3], p=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "LRF_Score (线性死寂崩塌)", failure_score, probe_ts)
        return failure_score
    def _calculate_micro_order_gain(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        entropy_raw = raw_data['PRICE_ENTROPY_D']
        entropy_slope = raw_data.get('SLOPE_13_PRICE_ENTROPY_D', pd.Series(0.0, index=df_index))
        game_index = raw_data.get('intraday_chip_game_index_D', pd.Series(0.5, index=df_index))
        orderly_score = self._calculate_custom_normalization(entropy_slope, mode='negative_extreme', sensitivity=15.0, denoise=True)
        entropy_hab = self._calculate_qho_historical_accumulation_buffer(entropy_raw, windows=[21], name="Price_Entropy", _probe_data=_probe_data, probe_ts=probe_ts)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=30.0, denoise=True)
        game_intensity = self._calculate_custom_normalization(game_index, mode='limit_high', sensitivity=1.5)
        gain_score = self._lp_norm_fusion(df_index, [orderly_score, price_calm, pd.Series(1.0 - entropy_hab, index=df_index), game_intensity], [0.3, 0.3, 0.2, 0.2], p=2.0)
        return gain_score
    def _calculate_momentum_dissipation_balance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        vpa_accel = raw_data['VPA_ACCELERATION_13D']
        vpa_accel_jerk = raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index))
        dissipation_focus = self._calculate_custom_normalization(vpa_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        jerk_silence = self._calculate_custom_normalization(vpa_accel_jerk, mode='zero_focus', sensitivity=60.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=30.0, denoise=True)
        mdb_score = self._lp_norm_fusion(df_index, [dissipation_focus, jerk_silence, price_calm], [0.4, 0.3, 0.3], p=2.0)
        return mdb_score
    def _calculate_hunting_temporal_coherence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        attack_raw = raw_data['HM_COORDINATED_ATTACK_D']
        attack_jerk = raw_data.get('JERK_5_HM_COORDINATED_ATTACK_D', pd.Series(0.0, index=df_index))
        temporal_stability = pd.Series(1.0 - self._calculate_custom_normalization(attack_raw.rolling(window=8, min_periods=1).std().fillna(0.0), mode='limit_high', sensitivity=2.0), index=df_index)
        rhythm_score = self._calculate_custom_normalization(attack_jerk, mode='zero_focus', sensitivity=50.0, denoise=True)
        top_tier_activity = self._calculate_custom_normalization(raw_data['HM_ACTIVE_TOP_TIER_D'], mode='limit_high', sensitivity=2.0)
        htc_score = self._lp_norm_fusion(df_index, [temporal_stability, rhythm_score, top_tier_activity], [0.4, 0.4, 0.2], p=2.0)
        return htc_score
    def _calculate_liquidity_solidification_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        stability_raw = raw_data['TURNOVER_STABILITY_INDEX_D']
        stability_slope = raw_data.get('SLOPE_13_TURNOVER_STABILITY_INDEX_D', pd.Series(0.0, index=df_index))
        stability_score = self._calculate_custom_normalization(stability_raw, mode='limit_high', sensitivity=1.5)
        slope_growth = self._calculate_custom_normalization(stability_slope, mode='limit_high', sensitivity=10.0, denoise=True)
        stability_hab = self._calculate_qho_historical_accumulation_buffer(stability_raw, windows=[13, 21], name="Turnover_Stability", _probe_data=_probe_data, probe_ts=probe_ts)
        turnover_low = self._calculate_custom_normalization(raw_data['turnover_rate_f_D'], mode='limit_low', sensitivity=25.0)
        solidification_factor = self._lp_norm_fusion(df_index, [stability_score, slope_growth, stability_hab, turnover_low], [0.3, 0.2, 0.2, 0.3], p=2.0)
        return solidification_factor
    def _calculate_amount_distribution_entropy_delta(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        entropy_raw = raw_data['concentration_entropy_D']
        entropy_slope = raw_data.get('SLOPE_13_concentration_entropy_D', pd.Series(0.0, index=df_index))
        interceptive_score = self._calculate_custom_normalization(entropy_slope, mode='negative_extreme', sensitivity=10.0, denoise=True)
        entropy_hab = self._calculate_qho_historical_accumulation_buffer(entropy_raw, windows=[21], name="Amount_Entropy", _probe_data=_probe_data, probe_ts=probe_ts)
        final_score = self._lp_norm_fusion(df_index, [interceptive_score, pd.Series(1.0 - entropy_hab, index=df_index)], [0.6, 0.4], p=2.0)
        return final_score
    def _calculate_seat_scatter_decay(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        any_act = raw_data['HM_ACTIVE_ANY_D']
        top_act = raw_data['HM_ACTIVE_TOP_TIER_D']
        scatter_raw = pd.Series(np.maximum(0, any_act - top_act), index=df_index)
        scatter_jerk = self._safe_diff(self._safe_diff(self._safe_diff(scatter_raw, 5), 5), 5)
        decay_score = self._calculate_custom_normalization(scatter_jerk, mode='negative_extreme', sensitivity=15.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=30.0, denoise=True)
        final_decay = self._lp_norm_fusion(df_index, [decay_score, price_calm], [0.5, 0.5], p=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Scatter_Decay (跟风席位退潮)", final_decay, probe_ts)
        return final_decay
    def _calculate_gravitational_regression_pull(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        bias_raw = raw_data['BIAS_55_D']
        bias_accel = raw_data.get('ACCEL_8_BIAS_55_D', pd.Series(0.0, index=df_index))
        bias_hab = self._calculate_qho_historical_accumulation_buffer(pd.Series(np.maximum(0, -bias_raw), index=df_index), windows=[21], name="Bias_55_Neg", _probe_data=_probe_data, probe_ts=probe_ts)
        gravity_ignite = self._calculate_custom_normalization(bias_accel, mode='limit_high', sensitivity=15.0, denoise=True)
        depth_score = self._calculate_custom_normalization(bias_raw, mode='negative_extreme', sensitivity=10.0)
        pull_score = self._lp_norm_fusion(df_index, [depth_score, bias_hab, gravity_ignite], [0.4, 0.3, 0.3], p=2.0)
        return pull_score
    def _calculate_short_exhaustion_divergence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        ndi_raw = raw_data['NDI_14_D']
        ndi_jerk = raw_data.get('JERK_5_NDI_14_D', pd.Series(0.0, index=df_index))
        exhaustion_score = self._calculate_custom_normalization(ndi_jerk, mode='negative_extreme', sensitivity=20.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=30.0, denoise=True)
        ndi_hab = self._calculate_qho_historical_accumulation_buffer(ndi_raw, windows=[21], name="NDI_14", _probe_data=_probe_data, probe_ts=probe_ts)
        divergence_score = self._lp_norm_fusion(df_index, [exhaustion_score, price_calm, ndi_hab], [0.4, 0.3, 0.3], p=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Short_Exhaustion (空头抛压耗尽)", divergence_score, probe_ts)
        return divergence_score
    def _calculate_long_awakening_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        pdi_raw = raw_data['PDI_14_D']
        pdi_slope = raw_data.get('SLOPE_13_PDI_14_D', pd.Series(0.0, index=df_index))
        pdi_jerk = raw_data.get('JERK_5_PDI_14_D', pd.Series(0.0, index=df_index))
        awakening_continuity = self._calculate_custom_normalization(pdi_slope, mode='limit_high', sensitivity=10.0, denoise=True)
        awakening_ignite = self._calculate_custom_normalization(pdi_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        pdi_hab = self._calculate_qho_historical_accumulation_buffer(pdi_raw, windows=[21], name="PDI_14", _probe_data=_probe_data, probe_ts=probe_ts)
        pdi_suppressed = pd.Series(1.0 - pdi_hab, index=df_index).clip(0, 1)
        awakening_score = self._lp_norm_fusion(df_index, [awakening_continuity, awakening_ignite, pdi_suppressed], [0.35, 0.4, 0.25], p=2.0)
        return awakening_score
    def _calculate_abnormal_energy_overflow(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        eff_jerk = raw_data.get('JERK_5_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        overflow_ignite = self._calculate_custom_normalization(eff_jerk, mode='limit_high', sensitivity=35.0, denoise=True)
        amount_calm = self._calculate_custom_normalization(pd.Series(raw_data['amount_D'] - 1.0, index=df_index), mode='negative_extreme', sensitivity=2.0)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=30.0, denoise=True)
        overflow_score = self._lp_norm_fusion(df_index, [overflow_ignite, amount_calm, price_calm], [0.4, 0.3, 0.3], p=2.0)
        return overflow_score
    def _calculate_phase_locked_resonance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        vpa_accel = raw_data['VPA_ACCELERATION_13D']
        price_accel = raw_data['MA_ACCELERATION_EMA_55_D']
        vpa_focus = self._calculate_custom_normalization(vpa_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        price_focus = self._calculate_custom_normalization(price_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        resonance_sim = pd.Series((vpa_accel * price_accel).rolling(window=5, min_periods=1).mean() / (vpa_accel.abs().rolling(window=5, min_periods=1).mean() * price_accel.abs().rolling(window=5, min_periods=1).mean() + 1e-9), index=df_index)
        vpa_jerk = self._calculate_custom_normalization(raw_data.get('JERK_5_VPA_ACCELERATION_13D', pd.Series(0.0, index=df_index)), mode='zero_focus', sensitivity=60.0, denoise=True)
        plr_score = self._lp_norm_fusion(df_index, [vpa_focus, price_focus, pd.Series(0.5 + 0.5 * resonance_sim.clip(0, 1), index=df_index), vpa_jerk], [0.25, 0.25, 0.3, 0.2], p=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "Phase_Locked (量价加速度锁死)", plr_score, probe_ts)
        return plr_score
    def _calculate_split_order_pulse_entropy(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        abnormal_jerk = raw_data.get('JERK_5_tick_abnormal_volume_ratio_D', pd.Series(0.0, index=df_index))
        jerk_std = abnormal_jerk.rolling(window=8, min_periods=1).std().fillna(0.0)
        jerk_mean = abnormal_jerk.abs().rolling(window=8, min_periods=1).mean().fillna(0.0)
        pulse_orderly = pd.Series(1.0 / (1.0 + (jerk_std / (jerk_mean + 1e-9))), index=df_index)
        order_score = self._calculate_custom_normalization(pulse_orderly, mode='limit_high', sensitivity=5.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=30.0, denoise=True)
        sope_gain = self._lp_norm_fusion(df_index, [order_score, price_calm], [0.6, 0.4], p=2.0)
        return sope_gain
    def _calculate_efficiency_gradient_dissipation(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        eff_slope = raw_data.get('SLOPE_13_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        eff_accel = raw_data.get('ACCEL_8_VPA_MF_ADJUSTED_EFF_D', pd.Series(0.0, index=df_index))
        slope_stability = pd.Series(1.0 - self._calculate_custom_normalization(eff_slope.rolling(window=8, min_periods=1).std().fillna(0.0), mode='limit_high', sensitivity=2.0), index=df_index)
        accel_lock = self._calculate_custom_normalization(eff_accel, mode='zero_focus', sensitivity=50.0, denoise=True)
        mf_activity = self._calculate_custom_normalization(raw_data['main_force_activity_index_D'], mode='limit_high', sensitivity=2.0)
        egd_score = self._lp_norm_fusion(df_index, [slope_stability, accel_lock, mf_activity], [0.4, 0.4, 0.2], p=2.0)
        return egd_score
    def _calculate_potential_well_collapse(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        pit_state = raw_data['STATE_GOLDEN_PIT_D']
        pit_jerk = raw_data.get('JERK_5_STATE_GOLDEN_PIT_D', pd.Series(0.0, index=df_index))
        escape_ignite = self._calculate_custom_normalization(pit_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        trap_lock = self._calculate_custom_normalization(pit_jerk, mode='zero_focus', sensitivity=50.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=40.0, denoise=True)
        well_collapse_score = self._lp_norm_fusion(df_index, [pit_state, escape_ignite, pd.Series(1.0 - trap_lock, index=df_index), price_calm], [0.3, 0.3, 0.1, 0.3], p=2.0)
        return well_collapse_score
    def _calculate_high_freq_kinetic_gap_fill(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        bias5 = raw_data['BIAS_5_D']
        bias55 = raw_data['BIAS_55_D']
        b5_jerk = raw_data.get('JERK_5_BIAS_5_D', pd.Series(0.0, index=df_index))
        elasticity = self._calculate_custom_normalization(bias5, mode='negative_extreme', sensitivity=12.0)
        ignite = self._calculate_custom_normalization(b5_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        gap_score = self._calculate_custom_normalization(pd.Series(np.maximum(0, bias55 - bias5), index=df_index), mode='limit_high', sensitivity=5.0)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=40.0, denoise=True)
        final_fill_score = self._lp_norm_fusion(df_index, [elasticity, ignite, gap_score, price_calm], [0.3, 0.3, 0.2, 0.2], p=2.0)
        return final_fill_score
    def _calculate_volatility_vacuum_contraction(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        atr_raw = raw_data['ATR_14_D']
        atr_slope = raw_data.get('SLOPE_13_ATR_14_D', pd.Series(0.0, index=df_index))
        atr_jerk = raw_data.get('JERK_5_ATR_14_D', pd.Series(0.0, index=df_index))
        atr_low_score = self._calculate_custom_normalization(atr_raw, mode='limit_low', sensitivity=20.0)
        decay_purity = self._calculate_custom_normalization(atr_slope, mode='negative_extreme', sensitivity=30.0, denoise=True)
        vacuum_silence = self._calculate_custom_normalization(atr_jerk, mode='zero_focus', sensitivity=80.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=40.0, denoise=True)
        vvc_score = self._lp_norm_fusion(df_index, [atr_low_score, decay_purity, vacuum_silence, price_calm], [0.3, 0.25, 0.25, 0.2], p=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "VVC_Score (波动率真空态)", vvc_score, probe_ts)
        return vvc_score
    def _calculate_fan_curvature_collapse(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        fan_raw = raw_data['MA_FAN_EFFICIENCY_D']
        fan_accel = raw_data.get('ACCEL_8_MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        fan_jerk = raw_data.get('JERK_5_MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        accel_focus = self._calculate_custom_normalization(fan_accel, mode='zero_focus', sensitivity=50.0, denoise=True)
        jerk_silence = self._calculate_custom_normalization(fan_jerk, mode='zero_focus', sensitivity=70.0, denoise=True)
        fan_high_score = self._calculate_custom_normalization(fan_raw, mode='limit_high', sensitivity=1.2)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=40.0, denoise=True)
        fcc_score = self._lp_norm_fusion(df_index, [fan_high_score, accel_focus, jerk_silence, price_calm], [0.3, 0.25, 0.25, 0.2], p=2.0)
        self._log_probe(_probe_data, "【04. 组件计算节点 (Nodes)】", "FCC_Score (扇面曲率塌缩)", fcc_score, probe_ts)
        return fcc_score
    def _calculate_game_neutralization_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        och_raw = raw_data['OCH_D']
        och_slope = raw_data.get('SLOPE_13_OCH_D', pd.Series(0.0, index=df_index))
        neutralization_focus = self._calculate_custom_normalization(och_slope, mode='zero_focus', sensitivity=45.0, denoise=True)
        och_intensity = self._calculate_custom_normalization(och_raw, mode='limit_high', sensitivity=1.0)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=40.0, denoise=True)
        neutral_score = self._lp_norm_fusion(df_index, [och_intensity, neutralization_focus, price_calm], [0.4, 0.3, 0.3], p=2.0)
        return neutral_score
    def _calculate_oversold_momentum_bipolarization(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        rsi_raw = raw_data['RSI_13_D']
        rsi_accel = raw_data.get('ACCEL_8_RSI_13_D', pd.Series(0.0, index=df_index))
        accel_rev_slope = self._safe_diff(rsi_accel, 5)
        vol = raw_data['volume_D']
        vol_consistency = pd.Series(1.0 / (1.0 + vol.rolling(window=8, min_periods=1).std().fillna(0.0) / (vol.rolling(window=8, min_periods=1).mean() + 1e-9)), index=df_index)
        oversold_lock = self._calculate_custom_normalization(pd.Series(rsi_raw, index=df_index), mode='limit_low', sensitivity=5.0)
        bipolar_ratio = self._calculate_custom_normalization(accel_rev_slope * vol_consistency, mode='limit_high', sensitivity=20.0, denoise=True)
        price_calm = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=40.0, denoise=True)
        omb_score = self._lp_norm_fusion(df_index, [oversold_lock, bipolar_ratio, price_calm], [0.4, 0.4, 0.2], p=2.0)
        return omb_score
    def _calculate_kinetic_overflow_veto(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], bipolar_gain: pd.Series, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        rsi_raw = raw_data['RSI_13_D']
        rsi_slope = raw_data.get('SLOPE_5_RSI_13_D', pd.Series(0.0, index=df_index))
        veto_l1 = pd.Series(np.where((rsi_raw > 0.75) & (rsi_slope < 0), 0.8, 1.0), index=df_index)
        vol = raw_data['volume_D']
        vol_spike = vol / (vol.rolling(window=21, min_periods=1).mean() + 1e-9)
        price_high = self._calculate_custom_normalization(pd.Series(raw_data.get('price_vs_ma_21_ratio_D', pd.Series(1.0, index=df_index)) - 1.0, index=df_index), mode='limit_high', sensitivity=5.0)
        veto_l2 = pd.Series(np.where((vol_spike > 2.5) & (price_high > 0.8), 0.7, 1.0), index=df_index)
        price_v = self._calculate_custom_normalization(raw_data['price_slope_raw'], mode='limit_high', sensitivity=5.0)
        veto_l3 = pd.Series(np.where((price_v > 0.6) & (bipolar_gain < 0.3), 0.6, 1.0), index=df_index)
        final_veto = (veto_l1 * veto_l2 * veto_l3).clip(0.3, 1.0)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Veto_Factor (三级防爆熔断)", final_veto, probe_ts)
        return final_veto
    def _calculate_spatio_temporal_asymmetric_reward(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], resonance_confirm: pd.Series, _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        close = raw_data.get('close_D', pd.Series(1.0, index=df_index))
        past_ret = close / (close.shift(5).fillna(close) + 1e-9) - 1.0
        hist_hit_mask = resonance_confirm.shift(5).fillna(0.0)
        expected_gain = (past_ret * hist_hit_mask).rolling(window=120, min_periods=10).mean().fillna(0.0)
        reward_factor = pd.Series(1.0 + self._calculate_custom_normalization(expected_gain.clip(lower=0), mode='limit_high', sensitivity=10.0), index=df_index)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Reward_Factor (时空异步奖赏)", reward_factor, probe_ts)
        return reward_factor
    def _calculate_extreme_panic_resonance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        pain_jerk = raw_data.get('JERK_5_pain_index_proxy', pd.Series(0.0, index=df_index))
        panic_burst = self._calculate_custom_normalization(pain_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        pit_state = raw_data['STATE_GOLDEN_PIT_D']
        resonance_score = self._lp_norm_fusion(df_index, [panic_burst, pit_state], [0.7, 0.3], p=2.0)
        return resonance_score
    def _calculate_adaptive_phase_transition_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        price_v = raw_data['price_slope_raw']
        noise_cv = price_v.rolling(window=250, min_periods=60).std() / (price_v.rolling(window=250, min_periods=60).mean().abs() + 1e-9)
        adaptive_threshold = pd.Series(0.45 * (0.8 + 0.5 * self._calculate_custom_normalization(noise_cv, mode='limit_high', sensitivity=2.0)), index=df_index)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "Fermi_Threshold (动态软门限)", adaptive_threshold.fillna(0.45), probe_ts)
        return adaptive_threshold.fillna(0.45)
    def _calculate_mean_reversion_kinetic_bias(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], _probe_data: Dict, probe_ts: pd.Timestamp) -> pd.Series:
        bias144 = raw_data['price_vs_ma_144_ratio']
        accel144 = raw_data.get('ACCEL_8_price_vs_ma_144_ratio', pd.Series(0.0, index=df_index))
        depth_reward = self._calculate_custom_normalization(pd.Series(1.0 - bias144, index=df_index), mode='limit_high', sensitivity=5.0)
        slingshot_ignite = self._calculate_custom_normalization(accel144, mode='limit_high', sensitivity=15.0, denoise=True)
        bias_factor = pd.Series(1.0 + 0.25 * self._lp_norm_fusion(df_index, [depth_reward, slingshot_ignite], [0.5, 0.5], p=2.0), index=df_index)
        self._log_probe(_probe_data, "【07. 宏观环境调节 (Environment)】", "MRKB_Factor (均值引力弹弓)", bias_factor.fillna(1.0), probe_ts)
        return bias_factor.fillna(1.0)













