# strategies/trend_following/intelligence/process/calculate_price_volume_dynamics.py
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

class CalculatePriceVolumeDynamics:
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
        """
        校验所有必需的信号是否存在。
        """
        required_signals = [
            'close_D', 'volume_D', 'open_D', 'high_D', 'low_D',
            'main_force_conviction_index_D', 'wash_trade_intensity_D', 'retail_panic_surrender_index_D',
            'upward_impulse_purity_D', 'active_buying_support_D', 'deception_index_D', 'retail_fomo_premium_index_D',
            'net_sh_amount_calibrated_D', 'net_md_amount_calibrated_D', 'net_lg_amount_calibrated_D', 'net_xl_amount_calibrated_D',
            'winner_concentration_90pct_D', 'loser_concentration_90pct_D', 'chip_health_score_D', 'mf_vpoc_premium_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'trend_vitality_index_D',
            'volume_burstiness_index_D', 'main_force_flow_directionality_D', 'order_book_imbalance_D',
            'micro_price_impact_asymmetry_D', 'bid_side_liquidity_D', 'ask_side_liquidity_D',
            'vpin_score_D', 'loser_loss_margin_avg_D', 'total_winner_rate_D', 'total_loser_rate_D',
            'panic_selling_cascade_D', 'intraday_energy_density_D', 'price_reversion_velocity_D',
            'VOL_MA_21_D', # 用于量能萎缩代理
            'volume_profile_entropy_D', 'volume_structure_skew_D', 'VPA_EFFICIENCY_D', 'VPA_BUY_EFFICIENCY_D', 'VPA_SELL_EFFICIENCY_D',
            'turnover_rate_f_D', 'main_force_flow_gini_D', 'main_force_slippage_index_D', 'main_force_execution_alpha_D',
            'main_force_t0_buy_efficiency_D', 'main_force_t0_sell_efficiency_D', 'main_force_vwap_up_guidance_D', 'main_force_vwap_down_guidance_D',
            'order_flow_imbalance_score_D', 'liquidity_slope_D', 'order_book_clearing_rate_D', 'chip_fault_blockage_ratio_D',
            'winner_loser_momentum_D', 'FRACTAL_DIMENSION_89d_D', 'SAMPLE_ENTROPY_13d_D', 'micro_impact_elasticity_D',
            'structural_entropy_change_D',
            'volume_ratio_D', 'active_volume_price_efficiency_D', 'constructive_turnover_ratio_D',
            'main_force_cost_advantage_D', 'main_force_activity_ratio_D', 'main_force_posture_index_D',
            'main_force_on_peak_buy_flow_D', 'main_force_on_peak_sell_flow_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'order_book_liquidity_supply_D', 'closing_acceptance_type_D', 'auction_showdown_score_D',
            'chip_fatigue_index_D', 'loser_pain_index_D', 'profit_realization_quality_D',
            'structural_tension_index_D', 'upward_impulse_strength_D', 'price_thrust_divergence_D',
            'equilibrium_compression_index_D', 'microstructure_efficiency_index_D', 'liquidity_authenticity_score_D',
            'flow_credibility_index_D',
            'is_consolidating_D', 'dynamic_consolidation_duration_D', 'breakout_readiness_score_D',
            'trend_acceleration_score_D', 'trend_conviction_score_D', 'covert_accumulation_signal_D',
            'covert_distribution_signal_D', 'holistic_cmf_D', 'main_force_net_flow_calibrated_D',
            'reversal_power_index_D', 'reversal_recovery_rate_D', 'volatility_asymmetry_index_D',
            'mean_reversion_frequency_D',
            'SCORE_STRUCT_AXIOM_TENSION', 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 'SCORE_MICRO_STRATEGY_STEALTH_OPS',
            # 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', # 移除此行
            'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM',
            'SCORE_DYN_AXIOM_STABILITY', 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION',
            'SCORE_STRUCT_BREAKOUT_READINESS', 'SCORE_STRUCT_PLATFORM_FOUNDATION',
            'goodness_of_fit_score_D', 'platform_conviction_score_D',
            'ADX_14_D', 'hidden_accumulation_intensity_D' # 确保 hidden_accumulation_intensity_D 在列中
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        base_signals_for_mtf_raw = []
        for s in required_signals:
            if not s.startswith(('SLOPE_', 'ACCEL_')) and s.endswith('_D'):
                base_signals_for_mtf_raw.append(s.rsplit('_', 1)[0])
        for base_sig_name in base_signals_for_mtf_raw:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig_name}_D')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig_name}_D')
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled and probe_ts:
                print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return False
        return True

    def _get_raw_signals(self, df: pd.DataFrame, method_name: str) -> Dict[str, pd.Series]:
        """
        获取所有原始数据信号。
        """
        raw_signals = {}
        # Price and Volume
        raw_signals['close_D'] = self.helper._get_safe_series(df, 'close_D', method_name=method_name)
        raw_signals['volume_D'] = self.helper._get_safe_series(df, 'volume_D', method_name=method_name)
        raw_signals['open_D'] = self.helper._get_safe_series(df, 'open_D', method_name=method_name)
        raw_signals['high_D'] = self.helper._get_safe_series(df, 'high_D', method_name=method_name)
        raw_signals['low_D'] = self.helper._get_safe_series(df, 'low_D', method_name=method_name)
        raw_signals['pct_change_D'] = self.helper._get_safe_series(df, 'pct_change_D', method_name=method_name)
        # Main Force and Behavior
        raw_signals['main_force_conviction_index_D'] = self.helper._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        raw_signals['wash_trade_intensity_D'] = self.helper._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        raw_signals['retail_panic_surrender_index_D'] = self.helper._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        raw_signals['upward_impulse_purity_D'] = self.helper._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        raw_signals['active_buying_support_D'] = self.helper._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        raw_signals['deception_index_D'] = self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        raw_signals['retail_fomo_premium_index_D'] = self.helper._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name=method_name)
        raw_signals['net_sh_amount_calibrated_D'] = self.helper._get_safe_series(df, 'net_sh_amount_calibrated_D', 0.0, method_name=method_name)
        raw_signals['net_md_amount_calibrated_D'] = self.helper._get_safe_series(df, 'net_md_amount_calibrated_D', 0.0, method_name=method_name)
        raw_signals['net_lg_amount_calibrated_D'] = self.helper._get_safe_series(df, 'net_lg_amount_calibrated_D', 0.0, method_name=method_name)
        raw_signals['net_xl_amount_calibrated_D'] = self.helper._get_safe_series(df, 'net_xl_amount_calibrated_D', 0.0, method_name=method_name)
        raw_signals['upward_impulse_strength_D'] = self.helper._get_safe_series(df, 'upward_impulse_strength_D', 0.0, method_name=method_name)
        raw_signals['price_thrust_divergence_D'] = self.helper._get_safe_series(df, 'price_thrust_divergence_D', 0.0, method_name=method_name)
        # Chips and Health
        raw_signals['winner_concentration_90pct_D'] = self.helper._get_safe_series(df, 'winner_concentration_90pct_D', 0.0, method_name=method_name)
        raw_signals['loser_concentration_90pct_D'] = self.helper._get_safe_series(df, 'loser_concentration_90pct_D', 0.0, method_name=method_name)
        raw_signals['chip_health_score_D'] = self.helper._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name)
        raw_signals['mf_vpoc_premium_D'] = self.helper._get_safe_series(df, 'mf_vpoc_premium_D', 0.0, method_name=method_name)
        raw_signals['loser_loss_margin_avg_D'] = self.helper._get_safe_series(df, 'loser_loss_margin_avg_D', 0.0, method_name=method_name)
        raw_signals['total_winner_rate_D'] = self.helper._get_safe_series(df, 'total_winner_rate_D', 0.0, method_name=method_name)
        raw_signals['total_loser_rate_D'] = self.helper._get_safe_series(df, 'total_loser_rate_D', 0.0, method_name=method_name)
        raw_signals['chip_fault_blockage_ratio_D'] = self.helper._get_safe_series(df, 'chip_fault_blockage_ratio_D', 0.0, method_name=method_name)
        raw_signals['winner_loser_momentum_D'] = self.helper._get_safe_series(df, 'winner_loser_momentum_D', 0.0, method_name=method_name)
        raw_signals['chip_fatigue_index_D'] = self.helper._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name=method_name)
        raw_signals['hidden_accumulation_intensity_D'] = self.helper._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name) # 确保获取此原始信号
        # Market Context
        raw_signals['market_sentiment_score_D'] = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        raw_signals['VOLATILITY_INSTABILITY_INDEX_21d_D'] = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        raw_signals['trend_vitality_index_D'] = self.helper._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        raw_signals['panic_selling_cascade_D'] = self.helper._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name=method_name)
        raw_signals['price_reversion_velocity_D'] = self.helper._get_safe_series(df, 'price_reversion_velocity_D', 0.0, method_name=method_name)
        raw_signals['structural_entropy_change_D'] = self.helper._get_safe_series(df, 'structural_entropy_change_D', 0.0, method_name=method_name)
        raw_signals['equilibrium_compression_index_D'] = self.helper._get_safe_series(df, 'equilibrium_compression_index_D', 0.0, method_name=method_name)
        raw_signals['microstructure_efficiency_index_D'] = self.helper._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name=method_name)
        raw_signals['liquidity_authenticity_score_D'] = self.helper._get_safe_series(df, 'liquidity_authenticity_score_D', 0.0, method_name=method_name)
        raw_signals['flow_credibility_index_D'] = self.helper._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        raw_signals['is_consolidating_D'] = self.helper._get_safe_series(df, 'is_consolidating_D', 0.0, method_name=method_name)
        raw_signals['dynamic_consolidation_duration_D'] = self.helper._get_safe_series(df, 'dynamic_consolidation_duration_D', 0.0, method_name=method_name)
        raw_signals['breakout_readiness_score_D'] = self.helper._get_safe_series(df, 'breakout_readiness_score_D', 0.0, method_name=method_name)
        raw_signals['trend_acceleration_score_D'] = self.helper._get_safe_series(df, 'trend_acceleration_score_D', 0.0, method_name=method_name)
        raw_signals['trend_conviction_score_D'] = self.helper._get_safe_series(df, 'trend_conviction_score_D', 0.0, method_name=method_name)
        raw_signals['covert_accumulation_signal_D'] = self.helper._get_safe_series(df, 'covert_accumulation_signal_D', 0.0, method_name=method_name)
        raw_signals['covert_distribution_signal_D'] = self.helper._get_safe_series(df, 'covert_distribution_signal_D', 0.0, method_name=method_name)
        raw_signals['holistic_cmf_D'] = self.helper._get_safe_series(df, 'holistic_cmf_D', 0.0, method_name=method_name)
        raw_signals['main_force_net_flow_calibrated_D'] = self.helper._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        raw_signals['reversal_power_index_D'] = self.helper._get_safe_series(df, 'reversal_power_index_D', 0.0, method_name=method_name)
        raw_signals['reversal_recovery_rate_D'] = self.helper._get_safe_series(df, 'reversal_recovery_rate_D', 0.0, method_name=method_name)
        raw_signals['volatility_asymmetry_index_D'] = self.helper._get_safe_series(df, 'volatility_asymmetry_index_D', 0.0, method_name=method_name)
        raw_signals['mean_reversion_frequency_D'] = self.helper._get_safe_series(df, 'mean_reversion_frequency_D', 0.0, method_name=method_name)
        raw_signals['ADX_14_D'] = self.helper._get_safe_series(df, 'ADX_14_D', 0.0, method_name=method_name)
        raw_signals['VOL_MA_21_D'] = self.helper._get_safe_series(df, 'VOL_MA_21_D', 0.0, method_name=method_name)
        raw_signals['volume_profile_entropy_D'] = self.helper._get_safe_series(df, 'volume_profile_entropy_D', 0.0, method_name=method_name)
        raw_signals['volume_structure_skew_D'] = self.helper._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name=method_name)
        raw_signals['VPA_EFFICIENCY_D'] = self.helper._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.0, method_name=method_name)
        raw_signals['VPA_BUY_EFFICIENCY_D'] = self.helper._get_safe_series(df, 'VPA_BUY_EFFICIENCY_D', 0.0, method_name=method_name)
        raw_signals['VPA_SELL_EFFICIENCY_D'] = self.helper._get_safe_series(df, 'VPA_SELL_EFFICIENCY_D', 0.0, method_name=method_name)
        raw_signals['turnover_rate_f_D'] = self.helper._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name=method_name)
        raw_signals['main_force_flow_gini_D'] = self.helper._get_safe_series(df, 'main_force_flow_gini_D', 0.0, method_name=method_name)
        raw_signals['main_force_slippage_index_D'] = self.helper._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name=method_name)
        raw_signals['main_force_execution_alpha_D'] = self.helper._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name=method_name)
        raw_signals['main_force_t0_buy_efficiency_D'] = self.helper._get_safe_series(df, 'main_force_t0_buy_efficiency_D', 0.0, method_name=method_name)
        raw_signals['main_force_t0_sell_efficiency_D'] = self.helper._get_safe_series(df, 'main_force_t0_sell_efficiency_D', 0.0, method_name=method_name)
        raw_signals['main_force_vwap_up_guidance_D'] = self.helper._get_safe_series(df, 'main_force_vwap_up_guidance_D', 0.0, method_name=method_name)
        raw_signals['main_force_vwap_down_guidance_D'] = self.helper._get_safe_series(df, 'main_force_vwap_down_guidance_D', 0.0, method_name=method_name)
        raw_signals['order_flow_imbalance_score_D'] = self.helper._get_safe_series(df, 'order_flow_imbalance_score_D', 0.0, method_name=method_name)
        raw_signals['liquidity_slope_D'] = self.helper._get_safe_series(df, 'liquidity_slope_D', 0.0, method_name=method_name)
        raw_signals['order_book_clearing_rate_D'] = self.helper._get_safe_series(df, 'order_book_clearing_rate_D', 0.0, method_name=method_name)
        raw_signals['FRACTAL_DIMENSION_89d_D'] = self.helper._get_safe_series(df, 'FRACTAL_DIMENSION_89d_D', 0.0, method_name=method_name)
        raw_signals['SAMPLE_ENTROPY_13d_D'] = self.helper._get_safe_series(df, 'SAMPLE_ENTROPY_13d_D', 0.0, method_name=method_name)
        raw_signals['micro_impact_elasticity_D'] = self.helper._get_safe_series(df, 'micro_impact_elasticity_D', 0.0, method_name=method_name)
        raw_signals['volume_ratio_D'] = self.helper._get_safe_series(df, 'volume_ratio_D', 0.0, method_name=method_name)
        raw_signals['active_volume_price_efficiency_D'] = self.helper._get_safe_series(df, 'active_volume_price_efficiency_D', 0.0, method_name=method_name)
        raw_signals['constructive_turnover_ratio_D'] = self.helper._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name=method_name)
        raw_signals['main_force_cost_advantage_D'] = self.helper._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name=method_name)
        raw_signals['main_force_activity_ratio_D'] = self.helper._get_safe_series(df, 'main_force_activity_ratio_D', 0.0, method_name=method_name)
        raw_signals['main_force_posture_index_D'] = self.helper._get_safe_series(df, 'main_force_posture_index_D', 0.0, method_name=method_name)
        raw_signals['main_force_on_peak_buy_flow_D'] = self.helper._get_safe_series(df, 'main_force_on_peak_buy_flow_D', 0.0, method_name=method_name)
        raw_signals['main_force_on_peak_sell_flow_D'] = self.helper._get_safe_series(df, 'main_force_on_peak_sell_flow_D', 0.0, method_name=method_name)
        raw_signals['SMART_MONEY_HM_COORDINATED_ATTACK_D'] = self.helper._get_safe_series(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 0.0, method_name=method_name)
        raw_signals['order_book_liquidity_supply_D'] = self.helper._get_safe_series(df, 'order_book_liquidity_supply_D', 0.0, method_name=method_name)
        raw_signals['closing_acceptance_type_D'] = self.helper._get_safe_series(df, 'closing_acceptance_type_D', 0.0, method_name=method_name)
        raw_signals['auction_showdown_score_D'] = self.helper._get_safe_series(df, 'auction_showdown_score_D', 0.0, method_name=method_name)
        raw_signals['profit_realization_quality_D'] = self.helper._get_safe_series(df, 'profit_realization_quality_D', 0.0, method_name=method_name)
        raw_signals['structural_tension_index_D'] = self.helper._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name=method_name)
        raw_signals['turnover_rate_D'] = self.helper._get_safe_series(df, 'turnover_rate_D', 0.0, method_name=method_name)
        raw_signals['HURST_144d_D'] = self.helper._get_safe_series(df, 'HURST_144d_D', 0.0, method_name=method_name)
        raw_signals['BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D'] = self.helper._get_safe_series(df, 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D', 0.0, method_name=method_name)
        raw_signals['vwap_buy_control_strength_D'] = self.helper._get_safe_series(df, 'vwap_buy_control_strength_D', 0.0, method_name=method_name)
        raw_signals['vwap_sell_control_strength_D'] = self.helper._get_safe_series(df, 'vwap_sell_control_strength_D', 0.0, method_name=method_name)
        raw_signals['observed_large_order_size_avg_D'] = self.helper._get_safe_series(df, 'observed_large_order_size_avg_D', 0.0, method_name=method_name)
        raw_signals['market_impact_cost_D'] = self.helper._get_safe_series(df, 'market_impact_cost_D', 0.0, method_name=method_name)
        raw_signals['mean_reversion_frequency_D'] = self.helper._get_safe_series(df, 'mean_reversion_frequency_D', 0.0, method_name=method_name)
        raw_signals['trend_alignment_index_D'] = self.helper._get_safe_series(df, 'trend_alignment_index_D', 0.0, method_name=method_name)
        raw_signals['counterparty_exhaustion_index_D'] = self.helper._get_safe_series(df, 'counterparty_exhaustion_index_D', 0.0, method_name=method_name)
        raw_signals['sell_quote_exhaustion_rate_D'] = self.helper._get_safe_series(df, 'sell_quote_exhaustion_rate_D', 0.0, method_name=method_name)
        raw_signals['goodness_of_fit_score_D'] = self.helper._get_safe_series(df, 'goodness_of_fit_score_D', 0.0, method_name=method_name)
        raw_signals['platform_conviction_score_D'] = self.helper._get_safe_series(df, 'platform_conviction_score_D', 0.0, method_name=method_name)
        # Atomic Scores
        raw_signals['SCORE_STRUCT_AXIOM_TENSION'] = self.helper._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0)
        raw_signals['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = self.helper._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        raw_signals['SCORE_MICRO_STRATEGY_STEALTH_OPS'] = self.helper._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        # raw_signals['PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY'] = self.helper._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', np.nan) # 移除此行
        raw_signals['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'] = self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        raw_signals['SCORE_DYN_AXIOM_STABILITY'] = self.helper._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        raw_signals['SCORE_FOUNDATION_AXIOM_MARKET_TENSION'] = self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 0.0)
        raw_signals['SCORE_STRUCT_BREAKOUT_READINESS'] = self.helper._get_atomic_score(df, 'SCORE_STRUCT_BREAKOUT_READINESS', 0.0)
        raw_signals['SCORE_STRUCT_PLATFORM_FOUNDATION'] = self.helper._get_atomic_score(df, 'SCORE_STRUCT_PLATFORM_FOUNDATION', 0.0)
        raw_signals['SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE'] = self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 0.0)
        return raw_signals

    def _get_mtf_signals(self, df: pd.DataFrame, raw_signals: Dict[str, pd.Series], mtf_slope_accel_weights: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        生成所有多时间框架 (MTF) 信号。
        """
        mtf_signals = {}
        df_index = df.index
        # Price and Volume Momentum
        mtf_signals['mtf_price_momentum'] = self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_volume_momentum'] = self.helper._get_mtf_slope_accel_score(df, 'volume_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # Main Force and Behavior MTF
        mtf_signals['mtf_main_force_conviction'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_conviction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_wash_trade_intensity'] = self.helper._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_retail_panic_surrender'] = self.helper._get_mtf_slope_accel_score(df, 'retail_panic_surrender_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_upward_impulse_purity'] = self.helper._get_mtf_slope_accel_score(df, 'upward_impulse_purity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_active_buying_support'] = self.helper._get_mtf_slope_accel_score(df, 'active_buying_support_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_deception_index'] = self.helper._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_retail_fomo_premium_index'] = self.helper._get_mtf_slope_accel_score(df, 'retail_fomo_premium_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_main_force_flow_directionality'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_flow_directionality_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_upward_impulse_strength'] = self.helper._get_mtf_slope_accel_score(df, 'upward_impulse_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_price_thrust_divergence'] = self.helper._get_mtf_slope_accel_score(df, 'price_thrust_divergence_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # Power Transfer Proxy (PROCESS_META_POWER_TRANSFER's raw data implementation)
        effective_main_force_flow_proxy = (raw_signals['net_lg_amount_calibrated_D'] + raw_signals['net_xl_amount_calibrated_D']).diff(1).fillna(0)
        effective_retail_flow_proxy = (raw_signals['net_sh_amount_calibrated_D'] + raw_signals['net_md_amount_calibrated_D']).diff(1).fillna(0)
        power_transfer_raw_proxy = effective_main_force_flow_proxy - effective_retail_flow_proxy
        power_transfer_raw_proxy.name = 'power_transfer_raw_proxy_D'
        mtf_signals['mtf_power_transfer'] = self.helper._get_mtf_slope_accel_score(df.assign(power_transfer_raw_proxy_D=power_transfer_raw_proxy), 'power_transfer_raw_proxy_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # Lower Shadow Absorption Proxy (SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION's raw data implementation)
        total_range = (raw_signals['high_D'] - raw_signals['low_D']).replace(0, 1e-9)
        lower_shadow = np.minimum(raw_signals['open_D'], raw_signals['close_D']) - raw_signals['low_D']
        lower_shadow_ratio = (lower_shadow / total_range).fillna(0)
        lower_shadow_absorption_raw = (lower_shadow_ratio > 0.3).astype(float) * (raw_signals['close_D'] > raw_signals['open_D'] * 0.99).astype(float)
        lower_shadow_absorption_raw.name = 'lower_shadow_absorption_raw_D'
        mtf_signals['mtf_lower_shadow_absorption'] = self.helper._get_mtf_slope_accel_score(df.assign(lower_shadow_absorption_raw_D=lower_shadow_absorption_raw), 'lower_shadow_absorption_raw_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # Volume Atrophy Proxy (SCORE_BEHAVIOR_VOLUME_ATROPHY's raw data implementation)
        volume_atrophy_raw = (1 - (raw_signals['volume_D'] / raw_signals['VOL_MA_21_D'])).clip(0, 1)
        volume_atrophy_raw.name = 'volume_atrophy_raw_D'
        mtf_signals['mtf_volume_atrophy'] = self.helper._get_mtf_slope_accel_score(df.assign(volume_atrophy_raw_D=volume_atrophy_raw), 'volume_atrophy_raw_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # Chip Strategic Posture Proxy (SCORE_CHIP_STRATEGIC_POSTURE's raw data implementation)
        mtf_winner_concentration = self.helper._get_mtf_slope_accel_score(df, 'winner_concentration_90pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_chip_health = self.helper._get_mtf_slope_accel_score(df, 'chip_health_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_mf_vpoc_premium = self.helper._get_mtf_slope_accel_score(df, 'mf_vpoc_premium_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_chip_strategic_posture'] = (mtf_winner_concentration * 0.4 + mtf_chip_health * 0.3 + mtf_mf_vpoc_premium * 0.3).clip(-1, 1)
        # V12.0+ New MTF Signals
        mtf_signals['mtf_volume_burstiness'] = self.helper._get_mtf_slope_accel_score(df, 'volume_burstiness_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_order_book_imbalance'] = self.helper._get_mtf_slope_accel_score(df, 'order_book_imbalance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_micro_price_impact_asymmetry'] = self.helper._get_mtf_slope_accel_score(df, 'micro_price_impact_asymmetry_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_bid_side_liquidity'] = self.helper._get_mtf_slope_accel_score(df, 'bid_side_liquidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_ask_side_liquidity'] = self.helper._get_mtf_slope_accel_score(df, 'ask_side_liquidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_vpin_score'] = self.helper._get_mtf_slope_accel_score(df, 'vpin_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_loser_loss_margin_avg'] = self.helper._get_mtf_slope_accel_score(df, 'loser_loss_margin_avg_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_total_winner_rate'] = self.helper._get_mtf_slope_accel_score(df, 'total_winner_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_total_loser_rate'] = self.helper._get_mtf_slope_accel_score(df, 'total_loser_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_panic_selling_cascade'] = self.helper._get_mtf_slope_accel_score(df, 'panic_selling_cascade_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_intraday_energy_density'] = self.helper._get_mtf_slope_accel_score(df, 'intraday_energy_density_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_price_reversion_velocity'] = self.helper._get_mtf_slope_accel_score(df, 'price_reversion_velocity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # V13.0+ New MTF Signals
        mtf_signals['mtf_volume_profile_entropy'] = self.helper._get_mtf_slope_accel_score(df, 'volume_profile_entropy_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_volume_structure_skew'] = self.helper._get_mtf_slope_accel_score(df, 'volume_structure_skew_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_vpa_efficiency'] = self.helper._get_mtf_slope_accel_score(df, 'VPA_EFFICIENCY_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_vpa_buy_efficiency'] = self.helper._get_mtf_slope_accel_score(df, 'VPA_BUY_EFFICIENCY_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_vpa_sell_efficiency'] = self.helper._get_mtf_slope_accel_score(df, 'VPA_SELL_EFFICIENCY_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_turnover_rate_f'] = self.helper._get_mtf_slope_accel_score(df, 'turnover_rate_f_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_main_force_flow_gini'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_flow_gini_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_main_force_slippage_index'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_slippage_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_main_force_execution_alpha'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_execution_alpha_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_main_force_t0_buy_efficiency'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_t0_buy_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_main_force_t0_sell_efficiency'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_t0_sell_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_main_force_vwap_up_guidance'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_vwap_up_guidance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_main_force_vwap_down_guidance'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_vwap_down_guidance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_order_flow_imbalance_score'] = self.helper._get_mtf_slope_accel_score(df, 'order_flow_imbalance_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_liquidity_slope'] = self.helper._get_mtf_slope_accel_score(df, 'liquidity_slope_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_order_book_clearing_rate'] = self.helper._get_mtf_slope_accel_score(df, 'order_book_clearing_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_chip_fault_blockage_ratio'] = self.helper._get_mtf_slope_accel_score(df, 'chip_fault_blockage_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_winner_loser_momentum'] = self.helper._get_mtf_slope_accel_score(df, 'winner_loser_momentum_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_fractal_dimension'] = self.helper._get_mtf_slope_accel_score(df, 'FRACTAL_DIMENSION_89d_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_sample_entropy'] = self.helper._get_mtf_slope_accel_score(df, 'SAMPLE_ENTROPY_13d_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_micro_impact_elasticity'] = self.helper._get_mtf_slope_accel_score(df, 'micro_impact_elasticity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_structural_entropy_change'] = self.helper._get_mtf_slope_accel_score(df, 'structural_entropy_change_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # V14.0+ New MTF Signals
        mtf_signals['mtf_volume_ratio'] = self.helper._get_mtf_slope_accel_score(df, 'volume_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_active_volume_price_efficiency'] = self.helper._get_mtf_slope_accel_score(df, 'active_volume_price_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_constructive_turnover_ratio'] = self.helper._get_mtf_slope_accel_score(df, 'constructive_turnover_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_main_force_cost_advantage'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_main_force_activity_ratio'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_activity_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_main_force_posture_index'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_posture_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_main_force_on_peak_buy_flow'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_on_peak_buy_flow_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_main_force_on_peak_sell_flow'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_on_peak_sell_flow_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_smart_money_coordinated_attack'] = self.helper._get_mtf_slope_accel_score(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_order_book_liquidity_supply'] = self.helper._get_mtf_slope_accel_score(df, 'order_book_liquidity_supply_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_closing_acceptance_type'] = self.helper._get_mtf_slope_accel_score(df, 'closing_acceptance_type_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_auction_showdown_score'] = self.helper._get_mtf_slope_accel_score(df, 'auction_showdown_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_chip_fatigue_index'] = self.helper._get_mtf_slope_accel_score(df, 'chip_fatigue_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_loser_pain_index'] = self.helper._get_mtf_slope_accel_score(df, 'loser_pain_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_profit_realization_quality'] = self.helper._get_mtf_slope_accel_score(df, 'profit_realization_quality_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_structural_tension_index'] = self.helper._get_mtf_slope_accel_score(df, 'structural_tension_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # V15.0+ New MTF Signals
        mtf_signals['mtf_is_consolidating'] = self.helper._get_mtf_slope_accel_score(df, 'is_consolidating_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_dynamic_consolidation_duration'] = self.helper._get_mtf_slope_accel_score(df, 'dynamic_consolidation_duration_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_breakout_readiness_score'] = self.helper._get_mtf_slope_accel_score(df, 'breakout_readiness_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_trend_acceleration_score'] = self.helper._get_mtf_slope_accel_score(df, 'trend_acceleration_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_trend_conviction_score'] = self.helper._get_mtf_slope_accel_score(df, 'trend_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_covert_accumulation_signal'] = self.helper._get_mtf_slope_accel_score(df, 'covert_accumulation_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_covert_distribution_signal'] = self.helper._get_mtf_slope_accel_score(df, 'covert_distribution_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_holistic_cmf'] = self.helper._get_mtf_slope_accel_score(df, 'holistic_cmf_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_main_force_net_flow_calibrated'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_reversal_power_index'] = self.helper._get_mtf_slope_accel_score(df, 'reversal_power_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_reversal_recovery_rate'] = self.helper._get_mtf_slope_accel_score(df, 'reversal_recovery_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signals['mtf_volatility_asymmetry_index'] = self.helper._get_mtf_slope_accel_score(df, 'volatility_asymmetry_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals['mtf_mean_reversion_frequency'] = self.helper._get_mtf_slope_accel_score(df, 'mean_reversion_frequency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        return mtf_signals

    def _normalize_and_fuse_dimension(self, df_index: pd.Index, components: Dict[str, pd.Series], weights: Dict[str, float], method_name: str, bipolar: bool = False) -> pd.Series:
        """
        通用方法，用于归一化单个组件并使用几何平均融合。
        """
        normalized_components = {}
        for key, series in components.items():
            # 检查是否需要反向归一化 (例如，_inverted 结尾的信号)
            ascending = not ("_inverted" in key or "_negative" in key or "_low" in key)
            normalized_components[key] = self.helper._normalize_series(series, df_index, bipolar=bipolar, ascending=ascending)
        return _robust_geometric_mean(normalized_components, weights, df_index)

    def _calculate_energy_compression_dimension(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], weights: Dict[str, float], method_name: str) -> pd.Series:
        """
        计算能量压缩维度分数。
        """
        # 防御性地获取 'BBW_21_2.0_D'，避免 KeyError
        bbw_raw = raw_signals.get('BBW_21_2.0_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        if 'BBW_21_2.0_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'BBW_21_2.0_D' 在 raw_signals 中缺失。使用默认的零值 Series。")

        components = {
            'tension': raw_signals['SCORE_STRUCT_AXIOM_TENSION'],
            'bbw_inverted': bbw_raw, # 使用安全获取的值
            'vol_instability_inverted': raw_signals['VOLATILITY_INSTABILITY_INDEX_21d_D'],
            'equilibrium_compression': raw_signals['equilibrium_compression_index_D'],
            'bbw_slope_inverted': mtf_signals.get('bbw_slope_inverted_score', pd.Series(0.0, index=df_index)), # 同样防御性获取
            'vol_instability_slope_inverted': mtf_signals.get('vol_instability_slope_inverted_score', pd.Series(0.0, index=df_index)), # 同样防御性获取
            'dyn_stability': raw_signals['SCORE_DYN_AXIOM_STABILITY'],
            'market_tension': raw_signals['SCORE_FOUNDATION_AXIOM_MARKET_TENSION'],
            'price_sample_entropy_inverted': raw_signals['SAMPLE_ENTROPY_13d_D'],
            'price_volume_entropy_inverted': raw_signals['volume_profile_entropy_D'], # 原始信号中没有 price_volume_entropy_D，这里假设是 volume_profile_entropy_D
            'price_fractal_dimension_calm': (1 - (raw_signals['FRACTAL_DIMENSION_89d_D'] - 1.5).abs() / 0.5).clip(0, 1),
            'volume_structure_skew_inverted': raw_signals['volume_structure_skew_D'].abs(),
            'volume_profile_entropy_inverted': raw_signals['volume_profile_entropy_D']
        }
        return self._normalize_and_fuse_dimension(df_index, components, weights, method_name)

    def _calculate_volume_exhaustion_dimension(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], weights: Dict[str, float], method_name: str) -> pd.Series:
        """
        计算量能枯竭维度分数。
        """
        # 防御性地获取相关信号，避免 KeyError
        turnover_rate_f_raw = raw_signals.get('turnover_rate_f_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        turnover_rate_raw = raw_signals.get('turnover_rate_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        counterparty_exhaustion_raw = raw_signals.get('counterparty_exhaustion_index_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        order_book_liquidity_raw = raw_signals.get('order_book_liquidity_supply_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        buy_quote_exhaustion_raw = raw_signals.get('buy_quote_exhaustion_rate_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        sell_quote_exhaustion_raw = raw_signals.get('sell_quote_exhaustion_rate_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        order_book_imbalance_raw = raw_signals.get('order_book_imbalance_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        micro_price_impact_asymmetry_raw = raw_signals.get('micro_price_impact_asymmetry_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        bid_side_liquidity_raw = raw_signals.get('bid_side_liquidity_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        ask_side_liquidity_raw = raw_signals.get('ask_side_liquidity_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        vpin_score_raw = raw_signals.get('vpin_score_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        bid_liquidity_sample_entropy_raw = raw_signals.get('BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        volume_structure_skew_raw = raw_signals.get('volume_structure_skew_D', pd.Series(0.0, index=df_index, dtype=np.float32))
        volume_profile_entropy_raw = raw_signals.get('volume_profile_entropy_D', pd.Series(0.0, index=df_index, dtype=np.float32))

        # 打印警告信息
        if 'buy_quote_exhaustion_rate_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'buy_quote_exhaustion_rate_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'sell_quote_exhaustion_rate_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'sell_quote_exhaustion_rate_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'order_book_liquidity_supply_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'order_book_liquidity_supply_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'counterparty_exhaustion_index_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'counterparty_exhaustion_index_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'turnover_rate_f_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'turnover_rate_f_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'turnover_rate_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'turnover_rate_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'order_book_imbalance_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'order_book_imbalance_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'micro_price_impact_asymmetry_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'micro_price_impact_asymmetry_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'bid_side_liquidity_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'bid_side_liquidity_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'ask_side_liquidity_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'ask_side_liquidity_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'vpin_score_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'vpin_score_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'volume_structure_skew_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'volume_structure_skew_D' 在 raw_signals 中缺失。使用默认的零值 Series。")
        if 'volume_profile_entropy_D' not in raw_signals:
            print(f"DEBUG: {method_name} - 警告: 'volume_profile_entropy_D' 在 raw_signals 中缺失。使用默认的零值 Series。")


        turnover_rate_inverted_score = self.helper._normalize_series(turnover_rate_f_raw, target_index=df_index, ascending=False)
        turnover_rate_raw_inverted = self.helper._normalize_series(turnover_rate_raw, target_index=df_index, ascending=False)
        counterparty_exhaustion_score = self.helper._normalize_series(counterparty_exhaustion_raw, target_index=df_index, ascending=True)
        order_book_liquidity_inverted_score = self.helper._normalize_series(order_book_liquidity_raw, target_index=df_index, ascending=False)
        buy_quote_exhaustion_score = self.helper._normalize_series(buy_quote_exhaustion_raw, target_index=df_index, ascending=True)
        sell_quote_exhaustion_score = self.helper._normalize_series(sell_quote_exhaustion_raw, target_index=df_index, ascending=True)
        order_book_imbalance_inverted = self.helper._normalize_series(order_book_imbalance_raw.abs(), target_index=df_index, ascending=False)
        micro_price_impact_asymmetry_inverted = self.helper._normalize_series(micro_price_impact_asymmetry_raw.abs(), target_index=df_index, ascending=False)
        bid_side_liquidity_inverted = self.helper._normalize_series(bid_side_liquidity_raw, target_index=df_index, ascending=False)
        ask_side_liquidity_inverted = self.helper._normalize_series(ask_side_liquidity_raw, target_index=df_index, ascending=False)
        vpin_score_inverted = self.helper._normalize_series(vpin_score_raw, target_index=df_index, ascending=False)
        bid_liquidity_sample_entropy_inverted = self.helper._normalize_series(bid_liquidity_sample_entropy_raw, target_index=df_index, ascending=False)
        volume_structure_skew_inverted = self.helper._normalize_series(volume_structure_skew_raw.abs(), target_index=df_index, ascending=False)
        volume_profile_entropy_inverted = self.helper._normalize_series(volume_profile_entropy_raw, target_index=df_index, ascending=False)

        components = {
            'volume_atrophy': raw_signals['SCORE_BEHAVIOR_VOLUME_ATROPHY'],
            'turnover_rate_inverted': turnover_rate_inverted_score,
            'counterparty_exhaustion': counterparty_exhaustion_score,
            'order_book_liquidity_inverted': order_book_liquidity_inverted_score,
            'buy_quote_exhaustion': buy_quote_exhaustion_score,
            'sell_quote_exhaustion': sell_quote_exhaustion_score,
            'turnover_rate_slope_inverted': mtf_signals.get('turnover_rate_slope_inverted_score', pd.Series(0.0, index=df_index)), # 防御性获取
            'order_book_imbalance_inverted': order_book_imbalance_inverted,
            'micro_price_impact_asymmetry_inverted': micro_price_impact_asymmetry_inverted,
            'bid_side_liquidity_inverted': bid_side_liquidity_inverted,
            'ask_side_liquidity_inverted': ask_side_liquidity_inverted,
            'vpin_score_inverted': vpin_score_inverted,
            'bid_liquidity_sample_entropy_inverted': bid_liquidity_sample_entropy_inverted,
            'volume_structure_skew_inverted': volume_structure_skew_inverted,
            'volume_profile_entropy_inverted': volume_profile_entropy_inverted,
            'turnover_rate_raw_inverted': turnover_rate_raw_inverted
        }
        return self._normalize_and_fuse_dimension(df_index, components, weights, method_name)

    def _calculate_main_force_covert_intent_dimension(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], weights: Dict[str, float], ambiguity_weights: Dict[str, float], method_name: str) -> pd.Series:
        """
        计算主力隐蔽意图维度分数。
        """
        # Main Force Flow Ambiguity components
        main_force_flow_directionality_neutrality = 1 - self.helper._normalize_series(raw_signals['main_force_flow_directionality_D'].abs(), df_index, ascending=True)
        mf_net_flow_near_zero = 1 - self.helper._normalize_series(raw_signals['main_force_net_flow_calibrated_D'].abs(), df_index, ascending=True)
        deception_score = self.helper._normalize_series(raw_signals['deception_index_D'], df_index, ascending=True)
        wash_trade_score = self.helper._normalize_series(raw_signals['wash_trade_intensity_D'], df_index, ascending=True)
        mf_conviction_neutrality = 1 - self.helper._normalize_series(raw_signals['main_force_conviction_index_D'].abs(), df_index, ascending=True)
        deception_lure_neutrality = 1 - self.helper._normalize_series(raw_signals['deception_lure_long_intensity_D'].abs() + raw_signals['deception_lure_short_intensity_D'].abs(), df_index, ascending=True)
        covert_action_score = (1 - (self.helper._normalize_series(raw_signals['covert_accumulation_signal_D'], df_index, ascending=True) + self.helper._normalize_series(raw_signals['covert_distribution_signal_D'], df_index, ascending=True)).clip(0,1))
        main_force_slippage_inverted = self.helper._normalize_series(raw_signals['main_force_slippage_index_D'], df_index, ascending=False)
        main_force_flow_gini_inverted = self.helper._normalize_series(raw_signals['main_force_flow_gini_D'], df_index, ascending=False)
        micro_impact_elasticity_positive = self.helper._normalize_series(raw_signals['micro_impact_elasticity_D'], df_index, ascending=True)
        order_flow_imbalance_neutrality = 1 - self.helper._normalize_series(raw_signals['order_flow_imbalance_score_D'].abs(), df_index, ascending=True)
        liquidity_authenticity_positive = self.helper._normalize_series(raw_signals['liquidity_authenticity_score_D'], df_index, ascending=True)
        ambiguity_components = {
            'directionality_neutrality': main_force_flow_directionality_neutrality,
            'net_flow_near_zero': mf_net_flow_near_zero,
            'deception_score': deception_score,
            'wash_trade_score': wash_trade_score,
            'mf_conviction_neutrality': mf_conviction_neutrality,
            'deception_lure_neutrality': deception_lure_neutrality,
            'covert_action_score': covert_action_score,
            'main_force_slippage_inverted': main_force_slippage_inverted,
            'main_force_flow_gini_inverted': main_force_flow_gini_inverted,
            'micro_impact_elasticity_positive': micro_impact_elasticity_positive,
            'order_flow_imbalance_neutrality': order_flow_imbalance_neutrality,
            'liquidity_authenticity_positive': liquidity_authenticity_positive
        }
        main_force_flow_ambiguity = _robust_geometric_mean(ambiguity_components, ambiguity_weights, df_index)
        # 修正：split_order_accum_score 应使用原始指标 hidden_accumulation_intensity_D
        split_order_accum_score = self.helper._normalize_series(raw_signals['hidden_accumulation_intensity_D'], df_index, ascending=True)
        components = {
            'stealth_ops': raw_signals['SCORE_MICRO_STRATEGY_STEALTH_OPS'],
            'split_order_accum': split_order_accum_score, # 使用修正后的 split_order_accum_score
            'mf_net_flow_positive': raw_signals['main_force_net_flow_calibrated_D'].clip(lower=0),
            'mf_cost_advantage_positive': raw_signals['main_force_cost_advantage_D'].clip(lower=0),
            'mf_buy_ofi_positive': raw_signals['main_force_buy_ofi_D'],
            'mf_t0_buy_efficiency_positive': raw_signals['main_force_t0_buy_efficiency_D'],
            'mf_net_flow_slope_positive': mtf_signals['mf_net_flow_slope_positive'],
            'order_book_imbalance_positive': raw_signals['order_book_imbalance_D'].clip(lower=0),
            'micro_price_impact_asymmetry_positive': raw_signals['micro_price_impact_asymmetry_D'].clip(lower=0),
            'mf_vwap_guidance_neutrality': 1 - (raw_signals['main_force_vwap_up_guidance_D'] - raw_signals['main_force_vwap_down_guidance_D']).abs(),
            'vwap_control_neutrality': 1 - (raw_signals['vwap_buy_control_strength_D'] - raw_signals['vwap_sell_control_strength_D']).abs(),
            'observed_large_order_size_avg_inverted': raw_signals['observed_large_order_size_avg_D'],
            'market_impact_cost_inverted': raw_signals['market_impact_cost_D'],
            'main_force_net_flow_volatility_inverted': raw_signals['main_force_net_flow_calibrated_D'].rolling(window=21, min_periods=1).std(), # Assuming 21 is default
            'main_force_flow_ambiguity': main_force_flow_ambiguity
        }
        return self._normalize_and_fuse_dimension(df_index, components, weights, method_name)

    def _calculate_subdued_market_sentiment_dimension(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], weights: Dict[str, float], pvd_params: Dict, method_name: str) -> pd.Series:
        """
        计算市场情绪低迷维度分数。
        """
        sentiment_volatility_window = get_param_value(pvd_params.get('sentiment_volatility_window'), 21)
        long_term_sentiment_window = get_param_value(pvd_params.get('long_term_sentiment_window'), 55)
        sentiment_neutral_range = get_param_value(pvd_params.get('sentiment_neutral_range'), 1.0)
        sentiment_pendulum_neutral_range = get_param_value(pvd_params.get('sentiment_pendulum_neutral_range'), 0.2)
        market_sentiment_std_raw = raw_signals['market_sentiment_score_D'].rolling(window=sentiment_volatility_window, min_periods=1).std()
        sentiment_pendulum_std_raw = raw_signals['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'].rolling(window=sentiment_volatility_window, min_periods=1).std()
        market_sentiment_long_term_mean = raw_signals['market_sentiment_score_D'].rolling(window=long_term_sentiment_window, min_periods=1).mean()
        sentiment_pendulum_negative = self.helper._normalize_series(raw_signals['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'], df_index, bipolar=True).clip(upper=0).abs()
        market_sentiment_inverted = self.helper._normalize_series(raw_signals['market_sentiment_score_D'], df_index, ascending=False)
        retail_panic_inverted = self.helper._normalize_series(raw_signals['retail_panic_surrender_index_D'], df_index, ascending=False)
        retail_fomo_inverted = self.helper._normalize_series(raw_signals['retail_fomo_premium_index_D'], df_index, ascending=False)
        loser_pain_positive = self.helper._normalize_series(raw_signals['loser_pain_index_D'], df_index, ascending=True)
        liquidity_tide_calm = self.helper._normalize_series(raw_signals['SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE'].abs(), df_index, ascending=False)
        hurst_calm = (1 - (raw_signals['HURST_144d_D'] - 0.5).abs() / 0.5).clip(0, 1)
        sentiment_neutrality = 1 - self.helper._normalize_series(raw_signals['market_sentiment_score_D'].abs(), df_index, ascending=True)
        sentiment_pendulum_neutrality = 1 - self.helper._normalize_series(raw_signals['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'].abs(), df_index, bipolar=True).abs()
        sentiment_volatility_inverted = self.helper._normalize_series(market_sentiment_std_raw, df_index, ascending=False)
        sentiment_pendulum_volatility_inverted = self.helper._normalize_series(sentiment_pendulum_std_raw, df_index, ascending=False)
        long_term_sentiment_subdued = self.helper._normalize_series(market_sentiment_long_term_mean - raw_signals['market_sentiment_score_D'], df_index, ascending=True)
        sentiment_pendulum_not_extreme = (1 - (raw_signals['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'].abs() - sentiment_pendulum_neutral_range).clip(lower=0) / (raw_signals['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'].abs().max() - sentiment_pendulum_neutral_range + 1e-9)).clip(0, 1)
        market_sentiment_not_extreme = (1 - (raw_signals['market_sentiment_score_D'].abs() - sentiment_neutral_range).clip(lower=0) / (raw_signals['market_sentiment_score_D'].abs().max() - sentiment_neutral_range + 1e-9)).clip(0, 1)
        market_sentiment_boring_score = _robust_geometric_mean({'volatility_inverted': sentiment_volatility_inverted, 'not_extreme': market_sentiment_not_extreme}, {'volatility_inverted': 0.5, 'not_extreme': 0.5}, df_index)
        price_reversion_velocity_inverted = self.helper._normalize_series(raw_signals['price_reversion_velocity_D'], df_index, ascending=False)
        structural_entropy_change_inverted = self.helper._normalize_series(raw_signals['structural_entropy_change_D'].abs(), df_index, ascending=False)
        mean_reversion_frequency_inverted = self.helper._normalize_series(raw_signals['mean_reversion_frequency_D'], df_index, ascending=False)
        trend_alignment_positive = self.helper._normalize_series(raw_signals['trend_alignment_index_D'], df_index, ascending=True)
        components = {
            'sentiment_pendulum_negative': sentiment_pendulum_negative, 'market_sentiment_inverted': market_sentiment_inverted,
            'retail_panic_inverted': retail_panic_inverted, 'retail_fomo_inverted': retail_fomo_inverted,
            'loser_pain_positive': loser_pain_positive,
            'liquidity_tide_calm': liquidity_tide_calm, 'hurst_calm': hurst_calm,
            'sentiment_neutrality': sentiment_neutrality, 'sentiment_pendulum_neutrality': sentiment_pendulum_neutrality,
            'sentiment_volatility_inverted': sentiment_volatility_inverted,
            'sentiment_pendulum_volatility_inverted': sentiment_pendulum_volatility_inverted,
            'long_term_sentiment_subdued': long_term_sentiment_subdued,
            'market_sentiment_not_extreme': market_sentiment_not_extreme,
            'sentiment_pendulum_not_extreme': sentiment_pendulum_not_extreme,
            'market_sentiment_boring_score': market_sentiment_boring_score,
            'price_reversion_velocity_inverted': price_reversion_velocity_inverted,
            'structural_entropy_change_inverted': structural_entropy_change_inverted,
            'mean_reversion_frequency_inverted': mean_reversion_frequency_inverted,
            'trend_alignment_positive': trend_alignment_positive
        }
        return self._normalize_and_fuse_dimension(df_index, components, weights, method_name)

    def _calculate_breakout_readiness_dimension(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], weights: Dict[str, float], method_name: str) -> pd.Series:
        """
        计算突破准备度维度分数。
        """
        goodness_of_fit_score = self.helper._normalize_series(raw_signals['goodness_of_fit_score_D'], df_index, ascending=True)
        platform_conviction_score = self.helper._normalize_series(raw_signals['platform_conviction_score_D'], df_index, ascending=True)
        components = {
            'struct_breakout_readiness': raw_signals['SCORE_STRUCT_BREAKOUT_READINESS'],
            'struct_platform_foundation': raw_signals['SCORE_STRUCT_PLATFORM_FOUNDATION'],
            'goodness_of_fit': goodness_of_fit_score,
            'platform_conviction': platform_conviction_score
        }
        return self._normalize_and_fuse_dimension(df_index, components, weights, method_name)

    def _get_dynamic_weights(self, base_weights: Dict[str, float], context_modulator_score: pd.Series, dynamic_weight_sensitivity: float, df_index: pd.Index) -> Dict[str, pd.Series]:
        """
        根据情境动态调整权重。
        """
        dynamic_weights = {}
        for key, base_w in base_weights.items():
            # 简单示例：根据市场情绪调整权重，情绪越好，某些积极信号权重越高
            # 实际应用中可以根据具体信号和情境设计更复杂的调整逻辑
            # 这里使用一个简化的逻辑，积极信号在情境分数高时权重增加，消极信号在情境分数低时权重增加
            # context_modulator_score 范围 [0, 1]
            if "positive" in key or "inverted" in key or "high" in key or "calm" in key or "efficiency" in key or "purity" in key or "strength" in key or "quality" in key or "attack" in key or "readiness" in key or "recovery" in key or "accumulation" in key:
                dynamic_weights[key] = base_w * (1 + context_modulator_score * dynamic_weight_sensitivity) # 积极信号在好情境下权重增加
            elif "negative" in key or "low" in key or "fatigue" in key or "blockage" in key or "penalty" in key or "deception" in key or "wash_trade" in key or "slippage" in key or "pain" in key or "tension" in key or "distribution" in key:
                dynamic_weights[key] = base_w * (1 + (1 - context_modulator_score) * dynamic_weight_sensitivity) # 消极信号在坏情境下权重增加
            else:
                dynamic_weights[key] = pd.Series(base_w, index=df_index)
        # 归一化动态权重
        total_dynamic_weight = pd.Series(0.0, index=df_index, dtype=np.float32)
        for key in dynamic_weights:
            total_dynamic_weight += dynamic_weights[key]
        # 避免除以零
        total_dynamic_weight = total_dynamic_weight.replace(0, 1e-9)
        for key in dynamic_weights:
            dynamic_weights[key] = dynamic_weights[key] / total_dynamic_weight
        return dynamic_weights

    def _calculate_historical_context_factors(self, df: pd.DataFrame, df_index: pd.Index, mtf_signals: Dict[str, pd.Series], pvd_params: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        计算历史情境因子。
        """
        historical_context = {}
        historical_window_short = get_param_value(pvd_params.get('historical_window_short'), 5)
        historical_window_long = get_param_value(pvd_params.get('historical_window_long'), 21)
        # Quadrant Persistence Score (QPS) - 临时代理
        temp_q1_proxy = (mtf_signals['mtf_price_momentum'].clip(lower=0) * mtf_signals['mtf_volume_momentum'].clip(lower=0)).clip(0,1)
        temp_q2_proxy = (mtf_signals['mtf_price_momentum'].clip(lower=0) * mtf_signals['mtf_volume_momentum'].clip(upper=0).abs()).clip(0,1)
        temp_q3_proxy = (mtf_signals['mtf_price_momentum'].clip(upper=0).abs() * mtf_signals['mtf_volume_momentum'].clip(lower=0)).clip(0,1)
        temp_q4_proxy = (mtf_signals['mtf_price_momentum'].clip(upper=0).abs() * mtf_signals['mtf_volume_momentum'].clip(upper=0).abs()).clip(0,1)
        historical_context['quadrant_persistence_Q1_Q4'] = (temp_q1_proxy.rolling(window=historical_window_short).mean() + temp_q4_proxy.rolling(window=historical_window_short).mean()) / 2
        historical_context['quadrant_persistence_Q1_Q4'] = self.helper._normalize_series(historical_context['quadrant_persistence_Q1_Q4'], df_index, bipolar=False)
        historical_context['quadrant_persistence_Q2_Q3'] = (temp_q2_proxy.rolling(window=historical_window_short).mean() + temp_q3_proxy.rolling(window=historical_window_short).mean()) / 2
        historical_context['quadrant_persistence_Q2_Q3'] = self.helper._normalize_series(historical_context['quadrant_persistence_Q2_Q3'], df_index, bipolar=False)
        # Phase Transition Indicator (PTS)
        historical_context['phase_transition_Q4_to_Q1'] = ((temp_q4_proxy.shift(1) > 0.5) & (temp_q1_proxy > 0.5)).astype(float)
        historical_context['phase_transition_Q4_to_Q1'] = self.helper._normalize_series(historical_context['phase_transition_Q4_to_Q1'].rolling(window=historical_window_short).mean(), df_index, bipolar=False)
        # Cumulative Flow Balance (CFB)
        cumulative_flow_balance_raw = (mtf_signals['mtf_covert_accumulation_signal'] - mtf_signals['mtf_covert_distribution_signal'] + mtf_signals['mtf_holistic_cmf'] + mtf_signals['mtf_main_force_net_flow_calibrated']).rolling(window=historical_window_long).sum()
        historical_context['cumulative_flow_balance'] = self.helper._normalize_series(cumulative_flow_balance_raw, df_index, bipolar=True)
        # Market Regime Strength (MRS)
        market_regime_strength_raw = (1 - mtf_signals['mtf_is_consolidating']) * mtf_signals['mtf_trend_conviction_score'] + mtf_signals['mtf_breakout_readiness_score'] + mtf_signals['mtf_trend_acceleration_score']
        historical_context['market_regime_strength'] = self.helper._normalize_series(market_regime_strength_raw, df_index, bipolar=False)
        # Reversal Potential Score (RPS) - 用于动态指数
        historical_context['reversal_potential_score'] = self.helper._normalize_series(mtf_signals['mtf_reversal_power_index'] + mtf_signals['mtf_reversal_recovery_rate'], df_index, bipolar=False)
        return historical_context

    def _calculate_multi_level_resonance_factor(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], pvd_params: Dict, context_modulator_score_for_weights: pd.Series, method_name: str) -> pd.Series:
        """
        计算多层级共振因子。
        """
        multi_level_resonance_weights = get_param_value(pvd_params.get('multi_level_resonance_weights'), {})
        price_volume_resonance_components = get_param_value(pvd_params.get('price_volume_resonance_components'), {})
        main_chip_resonance_components = get_param_value(pvd_params.get('main_chip_resonance_components'), {})
        sentiment_liquidity_resonance_components = get_param_value(pvd_params.get('sentiment_liquidity_resonance_components'), {})
        micro_structure_resonance_components = get_param_value(pvd_params.get('micro_structure_resonance_components'), {})
        quality_efficiency_resonance_components = get_param_value(pvd_params.get('quality_efficiency_resonance_components'), {})
        dynamic_weight_sensitivity = get_param_value(pvd_params.get('dynamic_weight_sensitivity'), 0.3)

        # 3.1 价格-成交量共振
        dynamic_pv_resonance_weights = self._get_dynamic_weights(price_volume_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        price_volume_resonance_components_dict = {
            "lower_shadow_absorption": mtf_signals.get('mtf_lower_shadow_absorption', pd.Series(0.0, index=df_index)),
            "active_buying_support": mtf_signals.get('mtf_active_buying_support', pd.Series(0.0, index=df_index)),
            "volume_burstiness": mtf_signals.get('mtf_volume_burstiness', pd.Series(0.0, index=df_index)),
            "VPA_EFFICIENCY": mtf_signals.get('mtf_vpa_efficiency', pd.Series(0.0, index=df_index)).clip(lower=0),
            "volume_profile_entropy_inverted": (1 - mtf_signals.get('mtf_volume_profile_entropy', pd.Series(0.0, index=df_index))),
            "volume_ratio_positive": mtf_signals.get('mtf_volume_ratio', pd.Series(0.0, index=df_index)).clip(lower=0),
            "upward_impulse_strength": mtf_signals.get('mtf_upward_impulse_strength', pd.Series(0.0, index=df_index))
        }
        price_volume_resonance = _robust_geometric_mean(price_volume_resonance_components_dict, dynamic_pv_resonance_weights, df_index).clip(0, 1)

        # 3.2 主力-筹码共振
        dynamic_mc_resonance_weights = self._get_dynamic_weights(main_chip_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        main_chip_resonance_components_dict = {
            "power_transfer_positive": mtf_signals.get('mtf_power_transfer', pd.Series(0.0, index=df_index)).clip(lower=0),
            "main_force_conviction_positive": mtf_signals.get('mtf_main_force_conviction', pd.Series(0.0, index=df_index)).clip(lower=0),
            "main_force_flow_directionality_positive": mtf_signals.get('mtf_main_force_flow_directionality', pd.Series(0.0, index=df_index)).clip(lower=0),
            "chip_strategic_posture": mtf_signals.get('mtf_chip_strategic_posture', pd.Series(0.0, index=df_index)).clip(lower=0),
            "chip_fault_blockage_ratio_inverted": (1 - mtf_signals.get('mtf_chip_fault_blockage_ratio', pd.Series(0.0, index=df_index))),
            "main_force_cost_advantage_positive": mtf_signals.get('mtf_main_force_cost_advantage', pd.Series(0.0, index=df_index)).clip(lower=0),
            "SMART_MONEY_HM_COORDINATED_ATTACK": mtf_signals.get('mtf_smart_money_coordinated_attack', pd.Series(0.0, index=df_index))
        }
        main_chip_resonance = _robust_geometric_mean(main_chip_resonance_components_dict, dynamic_mc_resonance_weights, df_index).clip(0, 1)

        # 3.3 市场情绪-流动性共振
        dynamic_sl_resonance_weights = self._get_dynamic_weights(sentiment_liquidity_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        sentiment_liquidity_resonance_components_dict = {
            "market_sentiment_positive": raw_signals['market_sentiment_score_D'].clip(lower=0),
            "retail_panic_surrender_inverted": (1 - mtf_signals.get('mtf_retail_panic_surrender', pd.Series(0.0, index=df_index))),
            "bid_side_liquidity": mtf_signals.get('mtf_bid_side_liquidity', pd.Series(0.0, index=df_index)),
            "liquidity_slope_positive": mtf_signals.get('mtf_liquidity_slope', pd.Series(0.0, index=df_index)).clip(lower=0),
            "order_flow_imbalance_positive": mtf_signals.get('mtf_order_flow_imbalance_score', pd.Series(0.0, index=df_index)).clip(lower=0),
            "loser_pain_index_inverted": (1 - mtf_signals.get('mtf_loser_pain_index', pd.Series(0.0, index=df_index)))
        }
        sentiment_liquidity_resonance = _robust_geometric_mean(sentiment_liquidity_resonance_components_dict, dynamic_sl_resonance_weights, df_index).clip(0, 1)

        # 3.4 微观结构共振
        dynamic_ms_resonance_weights = self._get_dynamic_weights(micro_structure_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        micro_structure_resonance_components_dict = {
            "order_book_imbalance_positive": mtf_signals.get('mtf_order_book_imbalance', pd.Series(0.0, index=df_index)).clip(lower=0),
            "micro_price_impact_asymmetry_positive": mtf_signals.get('mtf_micro_price_impact_asymmetry', pd.Series(0.0, index=df_index)).clip(lower=0),
            "intraday_energy_density": mtf_signals.get('mtf_intraday_energy_density', pd.Series(0.0, index=df_index)),
            "vpin_score_inverted": (1 - mtf_signals.get('mtf_vpin_score', pd.Series(0.0, index=df_index))),
            "micro_impact_elasticity_positive": mtf_signals.get('mtf_micro_impact_elasticity', pd.Series(0.0, index=df_index)).clip(lower=0),
            "order_book_clearing_rate": mtf_signals.get('mtf_order_book_clearing_rate', pd.Series(0.0, index=df_index)),
            "closing_acceptance_type_positive": mtf_signals.get('mtf_closing_acceptance_type', pd.Series(0.0, index=df_index)).clip(lower=0)
        }
        micro_structure_resonance = _robust_geometric_mean(micro_structure_resonance_components_dict, dynamic_ms_resonance_weights, df_index).clip(0, 1)

        # 3.5 质量与效率共振
        dynamic_qe_resonance_weights = self._get_dynamic_weights(quality_efficiency_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        quality_efficiency_resonance_components_dict = {
            "upward_impulse_purity": mtf_signals.get('mtf_upward_impulse_purity', pd.Series(0.0, index=df_index)),
            "flow_credibility_index": mtf_signals.get('mtf_flow_credibility_index', pd.Series(0.0, index=df_index)), # 使用安全获取的值
            "profit_realization_quality": mtf_signals.get('mtf_profit_realization_quality', pd.Series(0.0, index=df_index)),
            "active_volume_price_efficiency": mtf_signals.get('mtf_active_volume_price_efficiency', pd.Series(0.0, index=df_index)).clip(lower=0),
            "constructive_turnover_ratio": mtf_signals.get('mtf_constructive_turnover_ratio', pd.Series(0.0, index=df_index))
        }
        quality_efficiency_resonance = _robust_geometric_mean(quality_efficiency_resonance_components_dict, dynamic_qe_resonance_weights, df_index).clip(0, 1)

        # 3.6 融合所有多层级共振因子
        multi_level_resonance_factor_dict = {
            "price_volume_resonance": price_volume_resonance,
            "main_chip_resonance": main_chip_resonance,
            "sentiment_liquidity_resonance": sentiment_liquidity_resonance,
            "micro_structure_resonance": micro_structure_resonance,
            "quality_efficiency_resonance": quality_efficiency_resonance
        }
        multi_level_resonance_factor = _robust_geometric_mean(multi_level_resonance_factor_dict, multi_level_resonance_weights, df_index).clip(0, 1)
        return multi_level_resonance_factor

    def _calculate_dynamic_thresholds(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], historical_context: Dict[str, pd.Series], pvd_params: Dict, method_name: str) -> Tuple[pd.Series, pd.Series]:
        """
        计算动态阈值。
        """
        dynamic_threshold_sensitivity = get_param_value(pvd_params.get('dynamic_threshold_sensitivity'), 0.05)
        price_volatility_norm = self.helper._normalize_series(raw_signals['close_D'].pct_change().rolling(self.std_window).std(), df_index, bipolar=False)
        volume_volatility_norm = self.helper._normalize_series(raw_signals['volume_D'].pct_change().rolling(self.std_window).std(), df_index, bipolar=False)
        # 动态阈值进一步考虑市场趋势强度、情绪和流动性真实性
        norm_trend_vitality = self.helper._normalize_series(raw_signals['trend_vitality_index_D'], df_index, bipolar=False)
        norm_market_sentiment = self.helper._normalize_series(raw_signals['market_sentiment_score_D'], df_index, bipolar=True)
        norm_liquidity_authenticity = self.helper._normalize_series(raw_signals['liquidity_authenticity_score_D'], df_index, bipolar=False)
        dynamic_threshold_modulator = (norm_trend_vitality * 0.3 + (1 - norm_market_sentiment.abs()) * 0.3 + norm_liquidity_authenticity * 0.4).clip(0.5, 1.5)
        dynamic_price_threshold = price_volatility_norm * dynamic_threshold_sensitivity * dynamic_threshold_modulator
        dynamic_volume_threshold = volume_volatility_norm * dynamic_threshold_sensitivity * dynamic_threshold_modulator
        return dynamic_price_threshold, dynamic_volume_threshold

    def _calculate_quadrant_scores(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], historical_context: Dict[str, pd.Series], pvd_params: Dict, context_modulator_score_for_weights: pd.Series, dynamic_price_threshold: pd.Series, dynamic_volume_threshold: pd.Series, method_name: str) -> Dict[str, pd.Series]:
        """
        计算四个象限的分数。
        """
        quadrant_scores = {}
        Q1_reward_weights = get_param_value(pvd_params.get('Q1_reward_weights'), {})
        Q1_penalty_weights = get_param_value(pvd_params.get('Q1_penalty_weights'), {})
        Q2_divergence_penalty_weights = get_param_value(pvd_params.get('Q2_divergence_penalty_weights'), {})
        Q3_reward_weights = get_param_value(pvd_params.get('Q3_reward_weights'), {})
        Q3_penalty_weights = get_param_value(pvd_params.get('Q3_penalty_weights'), {})
        Q4_reward_weights = get_param_value(pvd_params.get('Q4_reward_weights'), {})
        Q4_penalty_weights = get_param_value(pvd_params.get('Q4_penalty_weights'), {})
        context_impact_modulators = get_param_value(pvd_params.get('context_impact_modulators'), {})
        dynamic_weight_sensitivity = get_param_value(pvd_params.get('dynamic_weight_sensitivity'), 0.3)
        p_mom = mtf_signals['mtf_price_momentum']
        v_mom = mtf_signals['mtf_volume_momentum']
        # V16.0 Signal Impact Modulators
        accumulation_strength_modulator = (historical_context['cumulative_flow_balance'] + 1) / 2
        trend_strength_modulator = historical_context['market_regime_strength']
        bullish_persistence_modulator = historical_context['quadrant_persistence_Q1_Q4']
        bearish_persistence_modulator = historical_context['quadrant_persistence_Q2_Q3']
        # Q1: Healthy Rally (价涨量增)
        deception_impact_reduction = accumulation_strength_modulator * context_impact_modulators.get("deception_impact_reduction_factor", 0.5)
        trend_reward_enhancement = trend_strength_modulator * context_impact_modulators.get("trend_reward_enhancement_factor", 0.2)
        mtf_wash_trade_intensity_adjusted = mtf_signals['mtf_wash_trade_intensity'] * (1 - deception_impact_reduction)
        mtf_deception_index_adjusted = mtf_signals['mtf_deception_index'] * (1 - deception_impact_reduction)
        mtf_main_force_t0_sell_efficiency_adjusted = mtf_signals['mtf_main_force_t0_sell_efficiency'] * (1 - deception_impact_reduction)
        mtf_ask_side_liquidity_adjusted = mtf_signals['mtf_ask_side_liquidity'] * (1 - trend_reward_enhancement)
        mtf_profit_realization_quality_low_adjusted = (1 - mtf_signals['mtf_profit_realization_quality']) * (1 - trend_reward_enhancement)
        dynamic_Q1_reward_weights = self._get_dynamic_weights(Q1_reward_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        Q1_reward_components_dict = {
            "p_mom": p_mom.clip(lower=0),
            "v_mom": v_mom.clip(lower=0),
            "upward_purity": mtf_signals['mtf_upward_impulse_purity'] * (1 + trend_reward_enhancement),
            "main_force_conviction_positive": mtf_signals['mtf_main_force_conviction'].clip(lower=0) * (1 + trend_reward_enhancement),
            "main_force_flow_directionality_positive": mtf_signals['mtf_main_force_flow_directionality'].clip(lower=0) * (1 + trend_reward_enhancement),
            "VPA_BUY_EFFICIENCY": mtf_signals['mtf_vpa_buy_efficiency'] * (1 + trend_reward_enhancement),
            "main_force_execution_alpha": mtf_signals['mtf_main_force_execution_alpha'].clip(lower=0) * (1 + trend_reward_enhancement),
            "order_flow_imbalance_positive": mtf_signals['mtf_order_flow_imbalance_score'].clip(lower=0) * (1 + trend_reward_enhancement),
            "main_force_vwap_up_guidance_positive": mtf_signals['mtf_main_force_vwap_up_guidance'].clip(lower=0) * (1 + trend_reward_enhancement),
            "upward_impulse_strength": mtf_signals['mtf_upward_impulse_strength'] * (1 + trend_reward_enhancement),
            "flow_credibility_index": mtf_signals['mtf_flow_credibility_index'] * (1 + trend_reward_enhancement)
        }
        score1_reward = _robust_geometric_mean(Q1_reward_components_dict, dynamic_Q1_reward_weights, df_index).clip(0, 1)
        dynamic_Q1_penalty_weights = self._get_dynamic_weights(Q1_penalty_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        Q1_penalty_components_dict = {
            "wash_trade": mtf_wash_trade_intensity_adjusted,
            "deception_index": mtf_deception_index_adjusted,
            "main_force_t0_sell_efficiency": mtf_main_force_t0_sell_efficiency_adjusted,
            "ask_side_liquidity_high": mtf_ask_side_liquidity_adjusted,
            "profit_realization_quality_low": mtf_profit_realization_quality_low_adjusted
        }
        false_rally_penalty = _robust_geometric_mean(Q1_penalty_components_dict, dynamic_Q1_penalty_weights, df_index).clip(0, 1)
        quadrant_scores['score1'] = score1_reward * (1 - false_rally_penalty)
        # Q2: Bearish Divergence (价涨量缩)
        divergence_penalty_enhancement = bullish_persistence_modulator * context_impact_modulators.get("divergence_penalty_enhancement_factor", 0.3)
        deception_impact_reduction_Q2 = accumulation_strength_modulator * context_impact_modulators.get("deception_impact_reduction_factor", 0.5)
        mtf_retail_fomo_adjusted = mtf_signals['mtf_retail_fomo_premium_index'] * (1 - deception_impact_reduction_Q2)
        mtf_wash_trade_adjusted_Q2 = mtf_signals['mtf_wash_trade_intensity'] * (1 - deception_impact_reduction_Q2)
        mtf_deception_index_adjusted_Q2 = mtf_signals['mtf_deception_index'] * (1 - deception_impact_reduction_Q2)
        mtf_vpin_score_high_adjusted = mtf_signals['mtf_vpin_score'] * (1 + divergence_penalty_enhancement)
        mtf_winner_loser_momentum_negative_adjusted = mtf_signals['mtf_winner_loser_momentum'].clip(upper=0).abs() * (1 + divergence_penalty_enhancement)
        mtf_price_thrust_divergence_negative_adjusted = mtf_signals['mtf_price_thrust_divergence'].clip(upper=0).abs() * (1 + divergence_penalty_enhancement)
        price_up_volume_not_up = (p_mom > dynamic_price_threshold) & (v_mom < dynamic_volume_threshold) & (mtf_signals['mtf_main_force_flow_directionality'] < 0)
        dynamic_Q2_divergence_penalty_weights = self._get_dynamic_weights(Q2_divergence_penalty_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        Q2_divergence_penalty_components_dict = {
            "retail_fomo": mtf_retail_fomo_adjusted,
            "wash_trade": mtf_wash_trade_adjusted_Q2,
            "deception_index": mtf_deception_index_adjusted_Q2,
            "vpin_score_high": mtf_vpin_score_high_adjusted,
            "winner_loser_momentum_negative": mtf_winner_loser_momentum_negative_adjusted,
            "price_thrust_divergence_negative": mtf_price_thrust_divergence_negative_adjusted
        }
        divergence_penalty_factor = _robust_geometric_mean(Q2_divergence_penalty_components_dict, dynamic_Q2_divergence_penalty_weights, df_index).clip(0, 1)
        score2_magnitude = _robust_geometric_mean(
            {"p_mom_positive": p_mom.clip(lower=0), "v_mom_negative_abs": v_mom.clip(upper=0).abs()},
            {"p_mom_positive": 0.5, "v_mom_negative_abs": 0.5},
            df_index
        )
        score2_base = -(score2_magnitude * (1 + divergence_penalty_factor)).clip(0, 1)
        score2_base = score2_base.where(price_up_volume_not_up, 0.0)
        quadrant_scores['score2'] = score2_base - score2_base.abs() * bullish_persistence_modulator * 0.5
        # Q3: Panic Distribution (价跌量增)
        panic_impact_reduction = accumulation_strength_modulator * context_impact_modulators.get("panic_impact_reduction_factor", 0.4)
        absorption_reward_enhancement = historical_context['cumulative_flow_balance'].clip(lower=0) * context_impact_modulators.get("absorption_reward_enhancement_factor", 0.3)
        blockage_penalty_enhancement = bearish_persistence_modulator * context_impact_modulators.get("blockage_penalty_enhancement_factor", 0.3)
        mtf_retail_panic_surrender_adjusted = mtf_signals['mtf_retail_panic_surrender'] * (1 - panic_impact_reduction)
        mtf_panic_selling_cascade_adjusted = mtf_signals['mtf_panic_selling_cascade'] * (1 - panic_impact_reduction)
        mtf_chip_fault_blockage_ratio_adjusted = mtf_signals['mtf_chip_fault_blockage_ratio'] * (1 + blockage_penalty_enhancement)
        mtf_structural_tension_index_adjusted = mtf_signals['mtf_structural_tension_index'] * (1 + blockage_penalty_enhancement)
        Q3_panic_evidence_components = {
            "retail_panic_surrender": mtf_retail_panic_surrender_adjusted,
            "chip_strategic_posture_negative": mtf_signals['mtf_chip_strategic_posture'].clip(upper=0).abs(),
            "loser_loss_margin_avg_positive": mtf_signals['mtf_loser_loss_margin_avg'].clip(lower=0),
            "total_loser_rate_positive": mtf_signals['mtf_total_loser_rate'],
            "panic_selling_cascade": mtf_panic_selling_cascade_adjusted
        }
        Q3_panic_evidence_weights_internal = {"retail_panic_surrender": 0.25, "chip_strategic_posture_negative": 0.2, "loser_loss_margin_avg_positive": 0.2, "total_loser_rate_positive": 0.15, "panic_selling_cascade": 0.2}
        panic_evidence_factor = _robust_geometric_mean(Q3_panic_evidence_components, Q3_panic_evidence_weights_internal, df_index).clip(0, 1)
        score3_magnitude = _robust_geometric_mean(
            {"p_mom_negative_abs": p_mom.clip(upper=0).abs(), "v_mom_positive": v_mom.clip(lower=0)},
            {"p_mom_negative_abs": 0.5, "v_mom_positive": 0.5},
            df_index
        )
        score3_base = -(score3_magnitude * (1 + panic_evidence_factor)).clip(0, 1)
        dynamic_Q3_reward_weights = self._get_dynamic_weights(Q3_reward_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        Q3_reward_components_dict = {
            "lower_shadow_absorption": mtf_signals['mtf_lower_shadow_absorption'] * (1 + absorption_reward_enhancement),
            "active_buying_support": mtf_signals['mtf_active_buying_support'] * (1 + absorption_reward_enhancement),
            "main_force_flow_directionality_positive": mtf_signals['mtf_main_force_flow_directionality'].clip(lower=0) * (1 + absorption_reward_enhancement),
            "main_force_t0_buy_efficiency": mtf_signals['mtf_main_force_t0_buy_efficiency'] * (1 + absorption_reward_enhancement),
            "capitulation_absorption_index": self.helper._get_mtf_slope_accel_score(df, 'capitulation_absorption_index_D', pvd_params.get('mtf_slope_accel_weights', {}), df_index, method_name, bipolar=False) * (1 + absorption_reward_enhancement)
        }
        absorption_reward = _robust_geometric_mean(Q3_reward_components_dict, dynamic_Q3_reward_weights, df_index).clip(0, 1)
        dynamic_Q3_penalty_weights = self._get_dynamic_weights(Q3_penalty_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        Q3_penalty_components_dict = {
            "loser_loss_margin_avg_expanding": mtf_signals['mtf_loser_loss_margin_avg'].clip(lower=0),
            "panic_selling_cascade": mtf_panic_selling_cascade_adjusted,
            "chip_fault_blockage_ratio": mtf_chip_fault_blockage_ratio_adjusted,
            "downward_impulse_strength": mtf_signals['mtf_upward_impulse_strength'].clip(upper=0).abs(),
            "structural_tension_index": mtf_structural_tension_index_adjusted
        }
        blockage_penalty = _robust_geometric_mean(Q3_penalty_components_dict, dynamic_Q3_penalty_weights, df_index).clip(0, 1)
        quadrant_scores['score3'] = score3_base * (1 - absorption_reward) * (1 + blockage_penalty)
        quadrant_scores['score3'] -= quadrant_scores['score3'].abs() * bearish_persistence_modulator * 0.5
        # Q4: Selling Exhaustion (价跌量缩)
        exhaustion_reward_enhancement = bearish_persistence_modulator * context_impact_modulators.get("exhaustion_reward_enhancement_factor", 0.3)
        false_bottom_penalty_reduction = accumulation_strength_modulator * context_impact_modulators.get("false_bottom_penalty_reduction_factor", 0.4)
        mtf_volume_atrophy_adjusted = mtf_signals['mtf_volume_atrophy'] * (1 + exhaustion_reward_enhancement)
        mtf_retail_panic_surrender_adjusted_Q4 = mtf_signals['mtf_retail_panic_surrender'] * (1 + exhaustion_reward_enhancement)
        mtf_loser_pain_index_high_adjusted = mtf_signals['mtf_loser_pain_index'].clip(lower=0) * (1 + exhaustion_reward_enhancement)
        mtf_price_reversion_velocity_negative_adjusted = mtf_signals['mtf_price_reversion_velocity'].clip(upper=0).abs() * (1 - false_bottom_penalty_reduction)
        mtf_structural_entropy_change_positive_adjusted = mtf_signals['mtf_structural_entropy_change'].clip(lower=0) * (1 - false_bottom_penalty_reduction)
        mtf_main_force_vwap_down_guidance_positive_adjusted = mtf_signals['mtf_main_force_vwap_down_guidance'].clip(lower=0) * (1 - false_bottom_penalty_reduction)
        mtf_chip_fatigue_index_low_adjusted = (1 - mtf_signals['mtf_chip_fatigue_index']) * (1 - false_bottom_penalty_reduction)
        dynamic_Q4_reward_weights = self._get_dynamic_weights(Q4_reward_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        Q4_exhaustion_evidence_components_dict = {
            "volume_atrophy": mtf_volume_atrophy_adjusted,
            "retail_panic_surrender": mtf_retail_panic_surrender_adjusted_Q4,
            "lower_shadow_absorption": mtf_signals['mtf_lower_shadow_absorption'],
            "chip_health": mtf_signals['mtf_chip_strategic_posture'].clip(lower=0),
            "bid_side_liquidity": mtf_signals['mtf_bid_side_liquidity'],
            "vpin_score_low": (1 - mtf_signals['mtf_vpin_score']),
            "volume_profile_entropy_inverted": (1 - mtf_signals['mtf_volume_profile_entropy']),
            "FRACTAL_DIMENSION_calm": (1 - (mtf_signals['mtf_fractal_dimension'] - 1.5).abs() / 0.5).clip(0, 1),
            "loser_pain_index_high": mtf_loser_pain_index_high_adjusted,
            "equilibrium_compression_index": mtf_signals['mtf_equilibrium_compression_index']
        }
        exhaustion_evidence_factor = _robust_geometric_mean(Q4_exhaustion_evidence_components_dict, dynamic_Q4_reward_weights, df_index).clip(0, 1)
        score4_magnitude = _robust_geometric_mean(
            {"p_mom_negative_abs": p_mom.clip(upper=0).abs(), "v_mom_negative_abs": v_mom.clip(upper=0).abs()},
            {"p_mom_negative_abs": 0.5, "v_mom_negative_abs": 0.5},
            df_index
        )
        score4_base = (score4_magnitude * exhaustion_evidence_factor - score4_magnitude * (1 - exhaustion_evidence_factor)).clip(-1, 1)
        dynamic_Q4_penalty_weights = self._get_dynamic_weights(Q4_penalty_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity, df_index)
        Q4_penalty_components_dict = {
            "price_reversion_velocity_negative": mtf_price_reversion_velocity_negative_adjusted,
            "structural_entropy_change_positive": mtf_structural_entropy_change_positive_adjusted,
            "main_force_vwap_down_guidance_positive": mtf_main_force_vwap_down_guidance_positive_adjusted,
            "chip_fatigue_index_low": mtf_chip_fatigue_index_low_adjusted
        }
        false_bottom_penalty = _robust_geometric_mean(Q4_penalty_components_dict, dynamic_Q4_penalty_weights, df_index).clip(0, 1)
        quadrant_scores['score4'] = score4_base * (1 - false_bottom_penalty)
        quadrant_scores['score4'] += historical_context['phase_transition_Q4_to_Q1'] * 0.3
        quadrant_scores['score4'] += bullish_persistence_modulator * 0.2
        return quadrant_scores

    def _calculate_dynamic_modulators(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], historical_context: Dict[str, pd.Series], pvd_params: Dict, method_name: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算动态权重和指数调制器。
        """
        dynamic_context_modulator_weights = get_param_value(pvd_params.get('dynamic_context_modulator_weights'), {})
        dynamic_exponent_modulator_weights = get_param_value(pvd_params.get('dynamic_exponent_modulator_weights'), {})
        exponent_context_sensitivity = get_param_value(pvd_params.get('exponent_context_sensitivity'), 0.8)
        final_exponent_base = get_param_value(pvd_params.get('final_exponent_base'), 1.0)
        regime_modulator_params = get_param_value(pvd_params.get('regime_modulator_params'), {})
        norm_market_sentiment = self.helper._normalize_series(raw_signals['market_sentiment_score_D'], df_index, bipolar=True)
        norm_volatility_inverse = self.helper._normalize_series(raw_signals['VOLATILITY_INSTABILITY_INDEX_21d_D'], df_index, ascending=False)
        norm_trend_vitality = self.helper._normalize_series(raw_signals['trend_vitality_index_D'], df_index, bipolar=False)
        norm_liquidity_authenticity = self.helper._normalize_series(raw_signals['liquidity_authenticity_score_D'], df_index, bipolar=False)
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_inverse": norm_volatility_inverse,
            "trend_vitality": norm_trend_vitality,
            "liquidity_authenticity_score": norm_liquidity_authenticity,
            "market_regime_strength": historical_context['market_regime_strength']
        }
        context_modulator_score_for_weights = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in context_modulator_components.items()},
            dynamic_context_modulator_weights,
            df_index
        )
        # 防御性地获取 'mtf_microstructure_efficiency_index'，避免 KeyError
        mtf_microstructure_efficiency_index = mtf_signals.get('mtf_microstructure_efficiency_index', pd.Series(0.0, index=df_index, dtype=np.float32))
        if 'mtf_microstructure_efficiency_index' not in mtf_signals:
            print(f"DEBUG: {method_name} - 警告: 'mtf_microstructure_efficiency_index' 在 mtf_signals 中缺失。使用默认的零值 Series。")
        dynamic_final_exponent_components = {
            "volatility_inverse": norm_volatility_inverse,
            "trend_vitality": norm_trend_vitality,
            "market_sentiment": norm_market_sentiment.clip(lower=0), # 情绪积极时放大
            "liquidity_slope_positive": mtf_signals['mtf_liquidity_slope'].clip(lower=0), # 流动性斜率积极时放大
            "microstructure_efficiency_index": mtf_microstructure_efficiency_index, # 使用安全获取的值
            "reversal_potential_score": historical_context['reversal_potential_score']
        }
        dynamic_exponent_modulator = _robust_geometric_mean(dynamic_final_exponent_components, dynamic_exponent_modulator_weights, df_index)
        adjusted_final_exponent = final_exponent_base * (1 - dynamic_exponent_modulator * exponent_context_sensitivity)
        adjusted_final_exponent = adjusted_final_exponent.clip(0.1, 2.0)
        market_regime_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        if get_param_value(regime_modulator_params.get('enabled'), False):
            volatility_sensitivity = get_param_value(regime_modulator_params.get('volatility_sensitivity'), 0.5)
            trend_sensitivity = get_param_value(regime_modulator_params.get('trend_sensitivity'), 0.5)
            base_modulator_factor = get_param_value(regime_modulator_params.get('base_modulator_factor'), 1.0)
            min_modulator = get_param_value(regime_modulator_params.get('min_modulator'), 0.8)
            max_modulator = get_param_value(regime_modulator_params.get('max_modulator'), 1.2)
            volatility_norm = self.helper._normalize_series(raw_signals['VOLATILITY_INSTABILITY_INDEX_21d_D'], df_index, ascending=False)
            trend_norm = self.helper._normalize_series(raw_signals['ADX_14_D'], df_index, ascending=False)
            market_regime_modulator = (
                base_modulator_factor +
                (volatility_norm * volatility_sensitivity + trend_norm * trend_sensitivity) / (volatility_sensitivity + trend_sensitivity + 1e-9)
            ).clip(min_modulator, max_modulator)
        return context_modulator_score_for_weights, adjusted_final_exponent, market_regime_modulator

    def _fuse_final_score(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], quadrant_scores: Dict[str, pd.Series], multi_level_resonance_factor: pd.Series, context_modulator_score_for_weights: pd.Series, adjusted_final_exponent: pd.Series, market_regime_modulator: pd.Series, pvd_params: Dict, method_name: str) -> pd.Series:
        """
        最终融合所有分数并应用调节器。
        """
        quadrant_weights = get_param_value(pvd_params.get('quadrant_weights'), {})
        dynamic_weight_sensitivity = get_param_value(pvd_params.get('dynamic_weight_sensitivity'), 0.3)
        price_calmness_modulator_params = get_param_value(pvd_params.get('price_calmness_modulator_params'), {})
        main_force_control_adjudicator_params = get_param_value(pvd_params.get('main_force_control_adjudicator'), {})
        # Dynamic Quadrant Weights
        dynamic_quadrant_weights = {}
        for q_name, base_w in quadrant_weights.items():
            if "Q1" in q_name or "Q4" in q_name:
                dynamic_quadrant_weights[q_name] = base_w * (1 + context_modulator_score_for_weights * dynamic_weight_sensitivity)
            elif "Q2" in q_name or "Q3" in q_name:
                dynamic_quadrant_weights[q_name] = base_w * (1 + (1 - context_modulator_score_for_weights) * dynamic_weight_sensitivity)
            dynamic_quadrant_weights[q_name] = dynamic_quadrant_weights[q_name].clip(0.05, 0.5)
        total_dynamic_weight = pd.Series(0.0, index=df_index, dtype=np.float32)
        for key in dynamic_quadrant_weights:
            total_dynamic_weight += dynamic_quadrant_weights[key]
        total_dynamic_weight = total_dynamic_weight.replace(0, 1e-9)
        for key in dynamic_quadrant_weights:
            dynamic_quadrant_weights[key] = dynamic_quadrant_weights[key] / total_dynamic_weight
        # Final Fusion: Weighted Average
        final_score_raw = (
            quadrant_scores['score1'] * dynamic_quadrant_weights["Q1_healthy_rally"] +
            quadrant_scores['score2'] * dynamic_quadrant_weights["Q2_bearish_divergence"] +
            quadrant_scores['score3'] * dynamic_quadrant_weights["Q3_panic_distribution"] +
            quadrant_scores['score4'] * dynamic_quadrant_weights["Q4_selling_exhaustion"]
        )
        # Apply Multi-Level Resonance Factor and Non-Linear Exponent
        final_score = final_score_raw * (1 + multi_level_resonance_factor * 0.5)
        final_score = np.sign(final_score) * (final_score.abs().pow(adjusted_final_exponent))
        # Price Calmness Modulator
        price_slope_raw = self.helper._get_safe_series(df, f'SLOPE_{price_calmness_modulator_params.get("slope_period", 5)}_close_D', np.nan, method_name=method_name)
        pct_change_raw = self.helper._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        price_slope_norm_bipolar = self.helper._normalize_series(price_slope_raw, df_index, bipolar=True)
        pct_change_abs_norm_inverted = self.helper._normalize_series(pct_change_raw.abs(), df_index, ascending=False)
        price_calmness_modulator = (price_calmness_modulator_params.get('modulator_factor', 0.5) * (1 - price_slope_norm_bipolar.abs()) + (1 - price_calmness_modulator_params.get('modulator_factor', 0.5)) * pct_change_abs_norm_inverted).clip(0,1)
        price_calmness_amplifier = 1 + (price_calmness_modulator * price_calmness_modulator_params.get('modulator_factor', 0.5))
        final_score *= price_calmness_amplifier
        # Main Force Control Adjudicator
        control_solidity_raw = self.helper._get_safe_series(df, main_force_control_adjudicator_params.get('control_signal', 'control_solidity_index_D'), np.nan, method_name=method_name)
        mf_activity_ratio_raw = self.helper._get_safe_series(df, main_force_control_adjudicator_params.get('activity_signal', 'main_force_activity_ratio_D'), np.nan, method_name=method_name)
        control_solidity_score = self.helper._normalize_series(control_solidity_raw, df_index, bipolar=True)
        mf_activity_ratio_score = self.helper._normalize_series(mf_activity_ratio_raw, df_index, ascending=True)
        veto_threshold = main_force_control_adjudicator_params.get('veto_threshold', -0.2)
        amplifier_factor = main_force_control_adjudicator_params.get('amplifier_factor', 0.5)
        combined_control_score = (control_solidity_score * 0.7 + mf_activity_ratio_score * 0.3).clip(-1, 1)
        final_score = final_score.mask(combined_control_score < veto_threshold, 0.0)
        main_force_amplifier = 1 + (combined_control_score * amplifier_factor)
        final_score = (final_score * main_force_amplifier).clip(-1, 1).fillna(0.0)
        return final_score

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V17.0 · 信号强度自适应放大与情境敏感度增强】计算价量动态的专属分数。
        - 核心升级: 优化非线性指数变换，使其在有利情境下放大信号，并增强情境对动态权重的影响力，确保最终分数更准确反映市场行为。
        - 核心重构: 优化四象限的数学模型，特别是Q2和Q4，使其更符合价量分析的业务逻辑。
        - 引入多维共振因子和动态权重，增强信号的鲁棒性和情境感知能力。
        - 全面探针调试，输出所有关键计算节点。
        - 新增原始数据：is_consolidating_D, dynamic_consolidation_duration_D, breakout_readiness_score_D,
                        trend_acceleration_score_D, trend_conviction_score_D, covert_accumulation_signal_D,
                        covert_distribution_signal_D, holistic_cmf_D, reversal_power_index_D,
                        reversal_recovery_rate_D, volatility_asymmetry_index_D, mean_reversion_frequency_D。
        - 优化判定思路：引入动态象限边界，强化多层级共振因子，细化象限内智能逻辑，动态调整非线性指数。
        - 修复：MTF信号名称生成逻辑，确保正确处理包含'_D'子串的原始信号名称。
        - 修复：'norm_trend_vitality'等情境因子在被使用前未赋值的错误。
        """
        method_name = "calculate_price_volume_dynamics"
        is_debug_enabled_for_method, probe_ts, _temp_debug_values = self._setup_debug_info(df, method_name)
        df_index = df.index
        pvd_params = self._get_pvd_params(config)
        mtf_slope_accel_weights = get_param_value(pvd_params.get('mtf_slope_accel_weights'), {})
        ambiguity_components_weights = get_param_value(pvd_params.get('ambiguity_components_weights'), {})
        if not self._validate_all_required_signals(df, pvd_params, mtf_slope_accel_weights, method_name, is_debug_enabled_for_method, probe_ts):
            if is_debug_enabled_for_method and probe_ts:
                self._print_pvd_debug_output(_temp_debug_values, probe_ts, method_name, "价量动态诊断失败：缺少核心信号。")
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        raw_signals = self._get_raw_signals(df, method_name)
        _temp_debug_values["原始信号值"] = raw_signals
        mtf_signals = self._get_mtf_signals(df, raw_signals, mtf_slope_accel_weights, method_name)
        _temp_debug_values["MTF融合信号"] = mtf_signals
        # Calculate Dynamic Modulators
        context_modulator_score_for_weights, adjusted_final_exponent, market_regime_modulator = self._calculate_dynamic_modulators(df, df_index, raw_signals, mtf_signals, self._calculate_historical_context_factors(df, df_index, mtf_signals, pvd_params, method_name), pvd_params, method_name)
        _temp_debug_values["动态权重与情境调制"] = {
            "context_modulator_score_for_weights": context_modulator_score_for_weights,
            "adjusted_final_exponent": adjusted_final_exponent,
            "market_regime_modulator": market_regime_modulator
        }
        # Calculate Historical Context Factors
        historical_context = self._calculate_historical_context_factors(df, df_index, mtf_signals, pvd_params, method_name)
        _temp_debug_values["历史情境感知层"] = historical_context
        # Calculate Multi-Level Resonance Factor
        multi_level_resonance_factor = self._calculate_multi_level_resonance_factor(df, df_index, raw_signals, mtf_signals, pvd_params, context_modulator_score_for_weights, method_name)
        _temp_debug_values["多层级共振引擎"] = {"multi_level_resonance_factor": multi_level_resonance_factor}
        # Calculate Dynamic Thresholds
        dynamic_price_threshold, dynamic_volume_threshold = self._calculate_dynamic_thresholds(df, df_index, raw_signals, historical_context, pvd_params, method_name)
        _temp_debug_values["动态阈值"] = {
            "dynamic_price_threshold": dynamic_price_threshold,
            "dynamic_volume_threshold": dynamic_volume_threshold
        }
        # Calculate Dimension Scores
        energy_compression_score = self._calculate_energy_compression_dimension(df_index, raw_signals, mtf_signals, get_param_value(pvd_params.get('energy_compression_weights'), {}), method_name)
        volume_exhaustion_score = self._calculate_volume_exhaustion_dimension(df_index, raw_signals, mtf_signals, get_param_value(pvd_params.get('volume_exhaustion_weights'), {}), method_name)
        main_force_covert_intent_score = self._calculate_main_force_covert_intent_dimension(df_index, raw_signals, mtf_signals, get_param_value(pvd_params.get('main_force_covert_intent_weights'), {}), ambiguity_components_weights, method_name)
        subdued_market_sentiment_score = self._calculate_subdued_market_sentiment_dimension(df_index, raw_signals, mtf_signals, get_param_value(pvd_params.get('subdued_market_sentiment_weights'), {}), pvd_params, method_name)
        breakout_readiness_score = self._calculate_breakout_readiness_dimension(df_index, raw_signals, get_param_value(pvd_params.get('breakout_readiness_weights'), {}), method_name)
        _temp_debug_values["维度分数"] = {
            "energy_compression_score": energy_compression_score,
            "volume_exhaustion_score": volume_exhaustion_score,
            "main_force_covert_intent_score": main_force_covert_intent_score,
            "subdued_market_sentiment_score": subdued_market_sentiment_score,
            "breakout_readiness_score": breakout_readiness_score
        }
        # Calculate Quadrant Scores
        quadrant_scores = self._calculate_quadrant_scores(df, df_index, raw_signals, mtf_signals, historical_context, pvd_params, context_modulator_score_for_weights, dynamic_price_threshold, dynamic_volume_threshold, method_name)
        _temp_debug_values["四象限分数"] = quadrant_scores
        # Final Fusion
        final_score = self._fuse_final_score(df, df_index, raw_signals, mtf_signals, quadrant_scores, multi_level_resonance_factor, context_modulator_score_for_weights, adjusted_final_exponent, market_regime_modulator, pvd_params, method_name)
        _temp_debug_values["最终融合分数"] = {"final_score": final_score}
        if is_debug_enabled_for_method and probe_ts:
            self._print_pvd_debug_output(_temp_debug_values, probe_ts, method_name, "价量动态诊断完成")
        return final_score.astype(np.float32)





