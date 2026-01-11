# strategies\trend_following\intelligence\process\calculate_storm_eye_calm.py
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
    【V4.0.2 · 拆单吸筹强度 · 探针强化与问题暴露版】
    - 核心修正: 严格区分原始数据及其MTF衍生与原子信号。原子信号的趋势通过直接diff()计算，
                避免在df中查找不存在的MTF衍生列。
    - 核心升级: 引入动态效率基准线，增强价格行为捕捉，精细化欺诈意图识别，MTF核心信号增强，
                情境自适应权重调整，非线性融合强化，趋势动量diff()化。
    - 探针强化: 增加关键中间计算节点的详细探针，特别是针对_robust_geometric_mean的输入和输出，
                以及最终pow()操作的精确值，以暴露潜在的计算偏差或bug。
    """
    def __init__(self, strategy_instance, helper: ProcessIntelligenceHelper):
        """
        初始化拆单吸筹强度计算器。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
            helper: ProcessIntelligenceHelper 实例，用于访问辅助方法。
        """
        self.strategy = strategy_instance
        self.helper = helper
        # 从 helper 获取参数
        self.params = self.helper.params
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        self.actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})

    def _print_debug_output_for_storm_eye_calm(self, debug_output: Dict, _temp_debug_values: Dict, probe_ts: pd.Timestamp, method_name: str, final_score: pd.Series):
        """
        统一打印 _calculate_storm_eye_calm 方法的调试信息。
        """
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        for key, value in _temp_debug_values["原始信号值"].items():
            if isinstance(value, pd.Series):
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            else: # Handle non-Series values like dicts or raw numbers
                debug_output[f"        '{key}': {value}"] = ""
        debug_output[f"  -- [过程情报调试] {probe_ts.strftime('%Y-%m-%d')}: --- MTF斜率/加速度分数 ---"] = ""
        for key, series in _temp_debug_values["MTF斜率/加速度分数"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 能量压缩 ---"] = ""
        for key, series in _temp_debug_values["能量压缩"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 量能枯竭 ---"] = ""
        for key, series in _temp_debug_values["量能枯竭"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        
        # 修改此处：正确遍历主力隐蔽意图的组件
        if "主力隐蔽意图" in _temp_debug_values and isinstance(_temp_debug_values["主力隐蔽意图"], dict):
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力隐蔽意图 (组件) ---"] = ""
            for key, series in _temp_debug_values["主力隐蔽意图"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
        
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力隐蔽意图融合 ---"] = ""
        for key, series in _temp_debug_values["主力隐蔽意图融合"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场情绪低迷融合 ---"] = ""
        for key, series in _temp_debug_values["市场情绪低迷融合"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 突破准备度融合 ---"] = ""
        for key, series in _temp_debug_values["突破准备度融合"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场情境动态调节器 ---"] = ""
        for key, series in _temp_debug_values["市场情境动态调节器"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合 ---"] = ""
        for key, series in _temp_debug_values["最终融合"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            elif isinstance(series, dict): # Handle dicts within _temp_debug_values["最终融合"]
                debug_output[f"        {key}:"] = ""
                for sub_key, sub_value in series.items():
                    if isinstance(sub_value, pd.Series):
                        val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                        debug_output[f"          {sub_key}: {val:.4f}"] = ""
                    else:
                        debug_output[f"          {sub_key}: {sub_value}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 风暴眼中的寂静诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

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

    def _get_required_signals(self, params: Dict, mtf_slope_accel_weights: Dict, mtf_cohesion_base_signals: List) -> List[str]:
        required_signals = [
            'SCORE_STRUCT_AXIOM_TENSION', 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 'control_solidity_index_D',
            'BBW_21_2.0_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'turnover_rate_f_D',
            'counterparty_exhaustion_index_D', 'main_force_conviction_index_D',
            'SCORE_MICRO_STRATEGY_STEALTH_OPS', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY',
            'main_force_net_flow_calibrated_D', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM',
            'market_sentiment_score_D', 'SLOPE_5_close_D', 'pct_change_D',
            'equilibrium_compression_index_D',
            'order_book_liquidity_supply_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'main_force_cost_advantage_D', 'main_force_buy_ofi_D', 'main_force_t0_buy_efficiency_D',
            'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D', 'loser_pain_index_D',
            'SCORE_STRUCT_BREAKOUT_READINESS', 'SCORE_STRUCT_PLATFORM_FOUNDATION',
            'main_force_activity_ratio_D', 'order_book_imbalance_D', 'micro_price_impact_asymmetry_D', 'ADX_14_D',
            'SCORE_DYN_AXIOM_STABILITY', 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION',
            'SAMPLE_ENTROPY_13d_D', 'price_volume_entropy_D', 'FRACTAL_DIMENSION_89d_D',
            'bid_side_liquidity_D', 'ask_side_liquidity_D', 'vpin_score_D', 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D',
            'main_force_vwap_up_guidance_D', 'main_force_vwap_down_guidance_D', 'vwap_buy_control_strength_D', 'vwap_sell_control_strength_D',
            'observed_large_order_size_avg_D', 'market_impact_cost_D', 'main_force_flow_directionality_D',
            'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 'HURST_144d_D', 'turnover_rate_D',
            'volume_structure_skew_D', 'volume_profile_entropy_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'covert_accumulation_signal_D', 'covert_distribution_signal_D',
            'main_force_slippage_index_D', 'main_force_flow_gini_D',
            'price_reversion_velocity_D', 'structural_entropy_change_D',
            'micro_impact_elasticity_D', 'order_flow_imbalance_score_D', 'liquidity_authenticity_score_D',
            'mean_reversion_frequency_D', 'trend_alignment_index_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in mtf_cohesion_base_signals:
            for period_str in get_param_value(mtf_slope_accel_weights.get('slope_periods'), {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in get_param_value(mtf_slope_accel_weights.get('accel_periods'), {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        return required_signals

    def _get_raw_and_atomic_data(self, df: pd.DataFrame, method_name: str, params: Dict) -> Dict[str, pd.Series]:
        raw_data = {}
        # Energy Compression
        raw_data['tension_score'] = self.helper._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', np.nan)
        raw_data['bbw_raw'] = self.helper._get_safe_series(df, 'BBW_21_2.0_D', np.nan, method_name=method_name)
        raw_data['vol_instability_raw'] = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', np.nan, method_name=method_name)
        raw_data['equilibrium_compression_raw'] = self.helper._get_safe_series(df, 'equilibrium_compression_index_D', np.nan, method_name=method_name)
        raw_data['dyn_stability_score'] = self.helper._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', np.nan)
        raw_data['market_tension_score'] = self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', np.nan)
        raw_data['price_sample_entropy_raw'] = self.helper._get_safe_series(df, 'SAMPLE_ENTROPY_13d_D', np.nan, method_name=method_name)
        raw_data['price_volume_entropy_raw'] = self.helper._get_safe_series(df, 'price_volume_entropy_D', np.nan, method_name=method_name)
        raw_data['price_fractal_dimension_raw'] = self.helper._get_safe_series(df, 'FRACTAL_DIMENSION_89d_D', np.nan, method_name=method_name)
        raw_data['volume_structure_skew_raw'] = self.helper._get_safe_series(df, 'volume_structure_skew_D', np.nan, method_name=method_name)
        raw_data['volume_profile_entropy_raw'] = self.helper._get_safe_series(df, 'volume_profile_entropy_D', np.nan, method_name=method_name)
        # Volume Exhaustion
        raw_data['atrophy_score'] = self.helper._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', np.nan)
        raw_data['turnover_rate_f_raw'] = self.helper._get_safe_series(df, 'turnover_rate_f_D', np.nan, method_name=method_name)
        raw_data['turnover_rate_raw'] = self.helper._get_safe_series(df, 'turnover_rate_D', np.nan, method_name=method_name)
        raw_data['counterparty_exhaustion_raw'] = self.helper._get_safe_series(df, 'counterparty_exhaustion_index_D', np.nan, method_name=method_name)
        raw_data['order_book_liquidity_raw'] = self.helper._get_safe_series(df, 'order_book_liquidity_supply_D', np.nan, method_name=method_name)
        raw_data['buy_quote_exhaustion_raw'] = self.helper._get_safe_series(df, 'buy_quote_exhaustion_rate_D', np.nan, method_name=method_name)
        raw_data['sell_quote_exhaustion_raw'] = self.helper._get_safe_series(df, 'sell_quote_exhaustion_rate_D', np.nan, method_name=method_name)
        raw_data['order_book_imbalance_raw'] = self.helper._get_safe_series(df, 'order_book_imbalance_D', np.nan, method_name=method_name)
        raw_data['micro_price_impact_asymmetry_raw'] = self.helper._get_safe_series(df, 'micro_price_impact_asymmetry_D', np.nan, method_name=method_name)
        raw_data['bid_side_liquidity_raw'] = self.helper._get_safe_series(df, 'bid_side_liquidity_D', np.nan, method_name=method_name)
        raw_data['ask_side_liquidity_raw'] = self.helper._get_safe_series(df, 'ask_side_liquidity_D', np.nan, method_name=method_name)
        raw_data['vpin_score_raw'] = self.helper._get_safe_series(df, 'vpin_score_D', np.nan, method_name=method_name)
        raw_data['bid_liquidity_sample_entropy_raw'] = self.helper._get_safe_series(df, 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D', np.nan, method_name=method_name)
        # Main Force Covert Intent
        raw_data['stealth_ops_score'] = self.helper._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', np.nan)
        raw_data['split_order_accum_score'] = self.helper._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', np.nan)
        raw_data['mf_conviction_raw'] = self.helper._get_safe_series(df, 'main_force_conviction_index_D', np.nan, method_name=method_name)
        raw_data['mf_net_flow_raw'] = self.helper._get_safe_series(df, 'main_force_net_flow_calibrated_D', np.nan, method_name=method_name)
        raw_data['mf_cost_advantage_raw'] = self.helper._get_safe_series(df, 'main_force_cost_advantage_D', np.nan, method_name=method_name)
        raw_data['mf_buy_ofi_raw'] = self.helper._get_safe_series(df, 'main_force_buy_ofi_D', np.nan, method_name=method_name)
        raw_data['mf_t0_buy_efficiency_raw'] = self.helper._get_safe_series(df, 'main_force_t0_buy_efficiency_D', np.nan, method_name=method_name)
        raw_data['mf_vwap_up_guidance_raw'] = self.helper._get_safe_series(df, 'main_force_vwap_up_guidance_D', np.nan, method_name=method_name)
        raw_data['mf_vwap_down_guidance_raw'] = self.helper._get_safe_series(df, 'main_force_vwap_down_guidance_D', np.nan, method_name=method_name)
        raw_data['vwap_buy_control_raw'] = self.helper._get_safe_series(df, 'vwap_buy_control_strength_D', np.nan, method_name=method_name)
        raw_data['vwap_sell_control_raw'] = self.helper._get_safe_series(df, 'vwap_sell_control_strength_D', np.nan, method_name=method_name)
        raw_data['observed_large_order_size_avg_raw'] = self.helper._get_safe_series(df, 'observed_large_order_size_avg_D', np.nan, method_name=method_name)
        raw_data['market_impact_cost_raw'] = self.helper._get_safe_series(df, 'market_impact_cost_D', np.nan, method_name=method_name)
        raw_data['main_force_flow_directionality_raw'] = self.helper._get_safe_series(df, 'main_force_flow_directionality_D', np.nan, method_name=method_name)
        raw_data['mf_net_flow_std_raw'] = raw_data['mf_net_flow_raw'].rolling(window=params['main_force_flow_volatility_window'], min_periods=1).std()
        raw_data['deception_index_raw'] = self.helper._get_safe_series(df, 'deception_index_D', np.nan, method_name=method_name)
        raw_data['wash_trade_intensity_raw'] = self.helper._get_safe_series(df, 'wash_trade_intensity_D', np.nan, method_name=method_name)
        raw_data['deception_lure_long_raw'] = self.helper._get_safe_series(df, 'deception_lure_long_intensity_D', np.nan, method_name=method_name)
        raw_data['deception_lure_short_raw'] = self.helper._get_safe_series(df, 'deception_lure_short_intensity_D', np.nan, method_name=method_name)
        raw_data['covert_accumulation_raw'] = self.helper._get_safe_series(df, 'covert_accumulation_signal_D', np.nan, method_name=method_name)
        raw_data['covert_distribution_raw'] = self.helper._get_safe_series(df, 'covert_distribution_signal_D', np.nan, method_name=method_name)
        raw_data['main_force_slippage_raw'] = self.helper._get_safe_series(df, 'main_force_slippage_index_D', np.nan, method_name=method_name)
        raw_data['main_force_flow_gini_raw'] = self.helper._get_safe_series(df, 'main_force_flow_gini_D', np.nan, method_name=method_name)
        raw_data['micro_impact_elasticity_raw'] = self.helper._get_safe_series(df, 'micro_impact_elasticity_D', np.nan, method_name=method_name)
        raw_data['order_flow_imbalance_score_raw'] = self.helper._get_safe_series(df, 'order_flow_imbalance_score_D', np.nan, method_name=method_name)
        raw_data['liquidity_authenticity_score_raw'] = self.helper._get_safe_series(df, 'liquidity_authenticity_score_D', np.nan, method_name=method_name)
        # Subdued Market Sentiment
        raw_data['sentiment_pendulum_score'] = self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', np.nan)
        raw_data['market_sentiment_raw'] = self.helper._get_safe_series(df, 'market_sentiment_score_D', np.nan, method_name=method_name)
        raw_data['retail_panic_raw'] = self.helper._get_safe_series(df, 'retail_panic_surrender_index_D', np.nan, method_name=method_name)
        raw_data['retail_fomo_raw'] = self.helper._get_safe_series(df, 'retail_fomo_premium_index_D', np.nan, method_name=method_name)
        raw_data['loser_pain_raw'] = self.helper._get_safe_series(df, 'loser_pain_index_D', np.nan, method_name=method_name)
        raw_data['liquidity_tide_score'] = self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', np.nan)
        raw_data['hurst_raw'] = self.helper._get_safe_series(df, 'HURST_144d_D', np.nan, method_name=method_name)
        raw_data['market_sentiment_std_raw'] = raw_data['market_sentiment_raw'].rolling(window=params['sentiment_volatility_window'], min_periods=1).std()
        raw_data['sentiment_pendulum_std_raw'] = raw_data['sentiment_pendulum_score'].rolling(window=params['sentiment_volatility_window'], min_periods=1).std()
        raw_data['market_sentiment_long_term_mean'] = raw_data['market_sentiment_raw'].rolling(window=params['long_term_sentiment_window'], min_periods=1).mean()
        raw_data['price_reversion_velocity_raw'] = self.helper._get_safe_series(df, 'price_reversion_velocity_D', np.nan, method_name=method_name)
        raw_data['structural_entropy_change_raw'] = self.helper._get_safe_series(df, 'structural_entropy_change_D', np.nan, method_name=method_name)
        raw_data['mean_reversion_frequency_raw'] = self.helper._get_safe_series(df, 'mean_reversion_frequency_D', np.nan, method_name=method_name)
        raw_data['trend_alignment_index_raw'] = self.helper._get_safe_series(df, 'trend_alignment_index_D', np.nan, method_name=method_name)
        # Breakout Readiness
        raw_data['struct_breakout_readiness_score'] = self.helper._get_atomic_score(df, 'SCORE_STRUCT_BREAKOUT_READINESS', np.nan)
        raw_data['struct_platform_foundation_score'] = self.helper._get_atomic_score(df, 'SCORE_STRUCT_PLATFORM_FOUNDATION', np.nan)
        raw_data['goodness_of_fit_raw'] = self.helper._get_safe_series(df, 'goodness_of_fit_score_D', np.nan, method_name=method_name)
        raw_data['platform_conviction_raw'] = self.helper._get_safe_series(df, 'platform_conviction_score_D', np.nan, method_name=method_name)
        # Modulators
        raw_data['price_slope_raw'] = self.helper._get_safe_series(df, f'SLOPE_{params["price_calmness_modulator_params"].get("slope_period", 5)}_close_D', np.nan, method_name=method_name)
        raw_data['pct_change_raw'] = self.helper._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        raw_data['control_solidity_raw'] = self.helper._get_safe_series(df, params['main_force_control_adjudicator_params'].get('control_signal', 'control_solidity_index_D'), np.nan, method_name=method_name)
        raw_data['mf_activity_ratio_raw'] = self.helper._get_safe_series(df, params['main_force_control_adjudicator_params'].get('activity_signal', 'main_force_activity_ratio_D'), np.nan, method_name=method_name)
        raw_data['volatility_regime_raw'] = self.helper._get_safe_series(df, params['regime_modulator_params'].get('volatility_signal', 'VOLATILITY_INSTABILITY_INDEX_21d_D'), np.nan, method_name=method_name)
        raw_data['trend_regime_raw'] = self.helper._get_safe_series(df, params['regime_modulator_params'].get('trend_signal', 'ADX_14_D'), np.nan, method_name=method_name)
        return raw_data

    def _calculate_mtf_derived_scores(self, df: pd.DataFrame, df_index: pd.Index, mtf_slope_accel_weights: Dict, mtf_cohesion_base_signals: List, method_name: str) -> Dict[str, pd.Series]:
        mtf_derived_scores = {}
        mtf_derived_scores['bbw_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'BBW_21_2.0_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['vol_instability_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['turnover_rate_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'turnover_rate_f_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['mf_net_flow_slope_positive'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['mtf_cohesion_score'] = self.helper._get_mtf_cohesion_score(df, mtf_cohesion_base_signals, mtf_slope_accel_weights, df_index, method_name)
        return mtf_derived_scores

    def _calculate_energy_compression_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict) -> pd.Series:
        bbw_inverted_score = self.helper._normalize_series(raw_data['bbw_raw'], target_index=df_index, ascending=False)
        vol_instability_inverted_score = self.helper._normalize_series(raw_data['vol_instability_raw'], target_index=df_index, ascending=False)
        equilibrium_compression_score = self.helper._normalize_series(raw_data['equilibrium_compression_raw'], target_index=df_index, ascending=True)
        dyn_stability_norm = self.helper._normalize_series(raw_data['dyn_stability_score'], target_index=df_index, bipolar=False)
        market_tension_norm = self.helper._normalize_series(raw_data['market_tension_score'], target_index=df_index, bipolar=False)
        price_sample_entropy_inverted = self.helper._normalize_series(raw_data['price_sample_entropy_raw'], target_index=df_index, ascending=False)
        price_volume_entropy_inverted = self.helper._normalize_series(raw_data['price_volume_entropy_raw'], target_index=df_index, ascending=False)
        price_fractal_dimension_calm = (1 - (raw_data['price_fractal_dimension_raw'] - 1.5).abs() / 0.5).clip(0, 1)
        volume_structure_skew_inverted = self.helper._normalize_series(raw_data['volume_structure_skew_raw'].abs(), target_index=df_index, ascending=False)
        volume_profile_entropy_inverted = self.helper._normalize_series(raw_data['volume_profile_entropy_raw'], target_index=df_index, ascending=False)
        energy_compression_scores_dict = {
            'tension': raw_data['tension_score'], 'bbw_inverted': bbw_inverted_score, 'vol_instability_inverted': vol_instability_inverted_score,
            'equilibrium_compression': equilibrium_compression_score, 'bbw_slope_inverted': mtf_derived_scores['bbw_slope_inverted_score'],
            'vol_instability_slope_inverted': mtf_derived_scores['vol_instability_slope_inverted_score'],
            'dyn_stability': dyn_stability_norm, 'market_tension': market_tension_norm,
            'price_sample_entropy_inverted': price_sample_entropy_inverted, 'price_volume_entropy_inverted': price_volume_entropy_inverted,
            'price_fractal_dimension_calm': price_fractal_dimension_calm,
            'volume_structure_skew_inverted': volume_structure_skew_inverted, 'volume_profile_entropy_inverted': volume_profile_entropy_inverted
        }
        return _robust_geometric_mean(energy_compression_scores_dict, weights, df_index)

    def _calculate_volume_exhaustion_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict) -> pd.Series:
        turnover_rate_inverted_score = self.helper._normalize_series(raw_data['turnover_rate_f_raw'], target_index=df_index, ascending=False)
        turnover_rate_raw_inverted = self.helper._normalize_series(raw_data['turnover_rate_raw'], target_index=df_index, ascending=False)
        counterparty_exhaustion_score = self.helper._normalize_series(raw_data['counterparty_exhaustion_raw'], target_index=df_index, ascending=True)
        order_book_liquidity_inverted_score = self.helper._normalize_series(raw_data['order_book_liquidity_raw'], target_index=df_index, ascending=False)
        buy_quote_exhaustion_score = self.helper._normalize_series(raw_data['buy_quote_exhaustion_raw'], target_index=df_index, ascending=True)
        sell_quote_exhaustion_score = self.helper._normalize_series(raw_data['sell_quote_exhaustion_raw'], target_index=df_index, ascending=True)
        order_book_imbalance_inverted = self.helper._normalize_series(raw_data['order_book_imbalance_raw'].abs(), target_index=df_index, ascending=False)
        micro_price_impact_asymmetry_inverted = self.helper._normalize_series(raw_data['micro_price_impact_asymmetry_raw'].abs(), target_index=df_index, ascending=False)
        bid_side_liquidity_inverted = self.helper._normalize_series(raw_data['bid_side_liquidity_raw'], target_index=df_index, ascending=False)
        ask_side_liquidity_inverted = self.helper._normalize_series(raw_data['ask_side_liquidity_raw'], target_index=df_index, ascending=False)
        vpin_score_inverted = self.helper._normalize_series(raw_data['vpin_score_raw'], target_index=df_index, ascending=False)
        bid_liquidity_sample_entropy_inverted = self.helper._normalize_series(raw_data['bid_liquidity_sample_entropy_raw'], target_index=df_index, ascending=False)
        volume_structure_skew_inverted = self.helper._normalize_series(raw_data['volume_structure_skew_raw'].abs(), target_index=df_index, ascending=False)
        volume_profile_entropy_inverted = self.helper._normalize_series(raw_data['volume_profile_entropy_raw'], target_index=df_index, ascending=False)
        volume_exhaustion_scores_dict = {
            'volume_atrophy': raw_data['atrophy_score'], 'turnover_rate_inverted': turnover_rate_inverted_score,
            'counterparty_exhaustion': counterparty_exhaustion_score, 'order_book_liquidity_inverted': order_book_liquidity_inverted_score,
            'buy_quote_exhaustion': buy_quote_exhaustion_score, 'sell_quote_exhaustion': sell_quote_exhaustion_score,
            'turnover_rate_slope_inverted': mtf_derived_scores['turnover_rate_slope_inverted_score'],
            'order_book_imbalance_inverted': order_book_imbalance_inverted,
            'micro_price_impact_asymmetry_inverted': micro_price_impact_asymmetry_inverted,
            'bid_side_liquidity_inverted': bid_side_liquidity_inverted, 'ask_side_liquidity_inverted': ask_side_liquidity_inverted,
            'vpin_score_inverted': vpin_score_inverted, 'bid_liquidity_sample_entropy_inverted': bid_liquidity_sample_entropy_inverted,
            'volume_structure_skew_inverted': volume_structure_skew_inverted, 'volume_profile_entropy_inverted': volume_profile_entropy_inverted,
            'turnover_rate_raw_inverted': turnover_rate_raw_inverted
        }
        return _robust_geometric_mean(volume_exhaustion_scores_dict, weights, df_index)

    def _calculate_main_force_covert_intent_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, ambiguity_weights: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        stealth_ops_normalized = self.helper._normalize_series(raw_data['stealth_ops_score'], target_index=df_index, ascending=True)
        split_order_accum_normalized = self.helper._normalize_series(raw_data['split_order_accum_score'], target_index=df_index, ascending=True)
        mf_conviction_positive = self.helper._normalize_series(raw_data['mf_conviction_raw'], target_index=df_index, bipolar=True).clip(lower=0)
        mf_net_flow_positive = self.helper._normalize_series(raw_data['mf_net_flow_raw'], target_index=df_index, bipolar=True).clip(lower=0)
        mf_cost_advantage_positive = self.helper._normalize_series(raw_data['mf_cost_advantage_raw'], target_index=df_index, bipolar=True).clip(lower=0)
        mf_buy_ofi_positive = self.helper._normalize_series(raw_data['mf_buy_ofi_raw'], target_index=df_index, ascending=True)
        mf_t0_buy_efficiency_positive = self.helper._normalize_series(raw_data['mf_t0_buy_efficiency_raw'], target_index=df_index, ascending=True)
        order_book_imbalance_positive = self.helper._normalize_series(raw_data['order_book_imbalance_raw'].clip(lower=0), target_index=df_index, ascending=True)
        micro_price_impact_asymmetry_positive = self.helper._normalize_series(raw_data['micro_price_impact_asymmetry_raw'].clip(lower=0), target_index=df_index, ascending=True)
        mf_vwap_guidance_neutrality = 1 - self.helper._normalize_series((raw_data['mf_vwap_up_guidance_raw'] - raw_data['mf_vwap_down_guidance_raw']).abs(), target_index=df_index, ascending=True)
        vwap_control_neutrality = 1 - self.helper._normalize_series((raw_data['vwap_buy_control_raw'] - raw_data['vwap_sell_control_raw']).abs(), target_index=df_index, ascending=True)
        observed_large_order_size_avg_inverted = self.helper._normalize_series(raw_data['observed_large_order_size_avg_raw'], target_index=df_index, ascending=False)
        market_impact_cost_inverted = self.helper._normalize_series(raw_data['market_impact_cost_raw'], target_index=df_index, ascending=False)
        main_force_net_flow_volatility_inverted = self.helper._normalize_series(raw_data['mf_net_flow_std_raw'], target_index=df_index, ascending=False)
        main_force_flow_directionality_neutrality = 1 - self.helper._normalize_series(raw_data['main_force_flow_directionality_raw'].abs(), target_index=df_index, ascending=True)
        mf_net_flow_near_zero = 1 - self.helper._normalize_series(raw_data['mf_net_flow_raw'].abs(), target_index=df_index, ascending=True)
        deception_score = self.helper._normalize_series(raw_data['deception_index_raw'], target_index=df_index, ascending=True)
        wash_trade_score = self.helper._normalize_series(raw_data['wash_trade_intensity_raw'], target_index=df_index, ascending=True)
        mf_conviction_neutrality = 1 - self.helper._normalize_series(raw_data['mf_conviction_raw'].abs(), target_index=df_index, ascending=True)
        deception_lure_neutrality = 1 - self.helper._normalize_series(raw_data['deception_lure_long_raw'].abs() + raw_data['deception_lure_short_raw'].abs(), target_index=df_index, ascending=True)
        covert_accumulation_norm = self.helper._normalize_series(raw_data['covert_accumulation_raw'], target_index=df_index, ascending=True)
        covert_distribution_norm = self.helper._normalize_series(raw_data['covert_distribution_raw'], target_index=df_index, ascending=True)
        covert_action_score = (1 - (covert_accumulation_norm + covert_distribution_norm).clip(0,1))
        main_force_slippage_inverted = self.helper._normalize_series(raw_data['main_force_slippage_raw'], target_index=df_index, ascending=False)
        main_force_flow_gini_inverted = self.helper._normalize_series(raw_data['main_force_flow_gini_raw'], target_index=df_index, ascending=False)
        micro_impact_elasticity_positive = self.helper._normalize_series(raw_data['micro_impact_elasticity_raw'], target_index=df_index, ascending=True)
        order_flow_imbalance_neutrality = 1 - self.helper._normalize_series(raw_data['order_flow_imbalance_score_raw'].abs(), target_index=df_index, ascending=True)
        liquidity_authenticity_positive = self.helper._normalize_series(raw_data['liquidity_authenticity_score_raw'], target_index=df_index, ascending=True)
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
        main_force_covert_intent_scores_dict = {
            'stealth_ops': stealth_ops_normalized, 'split_order_accum': split_order_accum_normalized,
            'mf_net_flow_positive': mf_net_flow_positive,
            'mf_cost_advantage_positive': mf_cost_advantage_positive, 'mf_buy_ofi_positive': mf_buy_ofi_positive,
            'mf_t0_buy_efficiency_positive': mf_t0_buy_efficiency_positive, 'mf_net_flow_slope_positive': mtf_derived_scores['mf_net_flow_slope_positive'],
            'order_book_imbalance_positive': order_book_imbalance_positive,
            'micro_price_impact_asymmetry_positive': micro_price_impact_asymmetry_positive,
            'mf_vwap_guidance_neutrality': mf_vwap_guidance_neutrality, 'vwap_control_neutrality': vwap_control_neutrality,
            'observed_large_order_size_avg_inverted': observed_large_order_size_avg_inverted, 'market_impact_cost_inverted': market_impact_cost_inverted,
            'main_force_net_flow_volatility_inverted': main_force_net_flow_volatility_inverted,
            'main_force_flow_ambiguity': main_force_flow_ambiguity
        }
        fused_score = _robust_geometric_mean(main_force_covert_intent_scores_dict, weights, df_index)
        return fused_score, main_force_covert_intent_scores_dict

    def _calculate_subdued_market_sentiment_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, sentiment_volatility_window: int, long_term_sentiment_window: int, sentiment_neutral_range: float, sentiment_pendulum_neutral_range: float) -> pd.Series:
        sentiment_pendulum_negative = self.helper._normalize_series(raw_data['sentiment_pendulum_score'], target_index=df_index, bipolar=True).clip(upper=0).abs()
        market_sentiment_inverted = self.helper._normalize_series(raw_data['market_sentiment_raw'], target_index=df_index, ascending=False)
        retail_panic_inverted = self.helper._normalize_series(raw_data['retail_panic_raw'], target_index=df_index, ascending=False)
        retail_fomo_inverted = self.helper._normalize_series(raw_data['retail_fomo_raw'], target_index=df_index, ascending=False)
        loser_pain_positive = self.helper._normalize_series(raw_data['loser_pain_raw'], target_index=df_index, ascending=True)
        liquidity_tide_calm = self.helper._normalize_series(raw_data['liquidity_tide_score'].abs(), target_index=df_index, ascending=False)
        hurst_calm = (1 - (raw_data['hurst_raw'] - 0.5).abs() / 0.5).clip(0, 1)
        sentiment_neutrality = 1 - self.helper._normalize_series(raw_data['market_sentiment_raw'].abs(), target_index=df_index, ascending=True)
        sentiment_pendulum_neutrality = 1 - self.helper._normalize_series(raw_data['sentiment_pendulum_score'].abs(), target_index=df_index, bipolar=True).abs()
        sentiment_volatility_inverted = self.helper._normalize_series(raw_data['market_sentiment_std_raw'], target_index=df_index, ascending=False)
        sentiment_pendulum_volatility_inverted = self.helper._normalize_series(raw_data['sentiment_pendulum_std_raw'], target_index=df_index, ascending=False)
        long_term_sentiment_subdued = self.helper._normalize_series(raw_data['market_sentiment_long_term_mean'] - raw_data['market_sentiment_raw'], target_index=df_index, ascending=True)
        sentiment_pendulum_not_extreme = (1 - (raw_data['sentiment_pendulum_score'].abs() - sentiment_pendulum_neutral_range).clip(lower=0) / (raw_data['sentiment_pendulum_score'].abs().max() - sentiment_pendulum_neutral_range + 1e-9)).clip(0, 1)
        market_sentiment_not_extreme = (1 - (raw_data['market_sentiment_raw'].abs() - sentiment_neutral_range).clip(lower=0) / (raw_data['market_sentiment_raw'].abs().max() - sentiment_neutral_range + 1e-9)).clip(0, 1)
        market_sentiment_boring_score = _robust_geometric_mean({'volatility_inverted': sentiment_volatility_inverted, 'not_extreme': market_sentiment_not_extreme}, {'volatility_inverted': 0.5, 'not_extreme': 0.5}, df_index)
        price_reversion_velocity_inverted = self.helper._normalize_series(raw_data['price_reversion_velocity_raw'], target_index=df_index, ascending=False)
        structural_entropy_change_inverted = self.helper._normalize_series(raw_data['structural_entropy_change_raw'].abs(), target_index=df_index, ascending=False)
        mean_reversion_frequency_inverted = self.helper._normalize_series(raw_data['mean_reversion_frequency_raw'], target_index=df_index, ascending=False)
        trend_alignment_positive = self.helper._normalize_series(raw_data['trend_alignment_index_raw'], target_index=df_index, ascending=True)
        subdued_market_sentiment_scores_dict = {
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
        return _robust_geometric_mean(subdued_market_sentiment_scores_dict, weights, df_index)

    def _calculate_breakout_readiness_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict) -> pd.Series:
        goodness_of_fit_score = self.helper._normalize_series(raw_data['goodness_of_fit_raw'], target_index=df_index, ascending=True)
        platform_conviction_score = self.helper._normalize_series(raw_data['platform_conviction_raw'], target_index=df_index, ascending=True)
        breakout_readiness_scores_dict = {
            'struct_breakout_readiness': raw_data['struct_breakout_readiness_score'],
            'struct_platform_foundation': raw_data['struct_platform_foundation_score'],
            'goodness_of_fit': goodness_of_fit_score,
            'platform_conviction': platform_conviction_score
        }
        return _robust_geometric_mean(breakout_readiness_scores_dict, weights, df_index)

    def _calculate_market_regime_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], params: Dict) -> pd.Series:
        market_regime_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        regime_modulator_params = params['regime_modulator_params']
        if get_param_value(regime_modulator_params.get('enabled'), False):
            volatility_sensitivity = get_param_value(regime_modulator_params.get('volatility_sensitivity'), 0.5)
            trend_sensitivity = get_param_value(regime_modulator_params.get('trend_sensitivity'), 0.5)
            base_modulator_factor = get_param_value(regime_modulator_params.get('base_modulator_factor'), 1.0)
            min_modulator = get_param_value(regime_modulator_params.get('min_modulator'), 0.8)
            max_modulator = get_param_value(regime_modulator_params.get('max_modulator'), 1.2)
            volatility_norm = self.helper._normalize_series(raw_data['volatility_regime_raw'], target_index=df_index, ascending=False)
            trend_norm = self.helper._normalize_series(raw_data['trend_regime_raw'], target_index=df_index, ascending=False)
            market_regime_modulator = (
                base_modulator_factor +
                (volatility_norm * volatility_sensitivity + trend_norm * trend_sensitivity) / (volatility_sensitivity + trend_sensitivity + 1e-9)
            ).clip(min_modulator, max_modulator)
        return market_regime_modulator

    def _perform_final_fusion(self, df_index: pd.Index, component_scores: Dict[str, pd.Series], final_fusion_weights: Dict, price_calmness_params: Dict, main_force_control_params: Dict, raw_data: Dict[str, pd.Series]) -> pd.Series:
        base_calm_score = _robust_geometric_mean(component_scores, final_fusion_weights, df_index)
        price_slope_norm_bipolar = self.helper._normalize_series(raw_data['price_slope_raw'], target_index=df_index, bipolar=True)
        pct_change_abs_norm_inverted = self.helper._normalize_series(raw_data['pct_change_raw'].abs(), target_index=df_index, ascending=False)
        price_calmness_modulator = (price_calmness_params.get('modulator_factor', 0.5) * (1 - price_slope_norm_bipolar.abs()) + (1 - price_calmness_params.get('modulator_factor', 0.5)) * pct_change_abs_norm_inverted).clip(0,1)
        price_calmness_amplifier = 1 + (price_calmness_modulator * price_calmness_params.get('modulator_factor', 0.5))
        control_solidity_score = self.helper._normalize_series(raw_data['control_solidity_raw'], target_index=df_index, bipolar=True)
        mf_activity_ratio_score = self.helper._normalize_series(raw_data['mf_activity_ratio_raw'], target_index=df_index, ascending=True)
        veto_threshold = main_force_control_params.get('veto_threshold', -0.2)
        amplifier_factor = main_force_control_params.get('amplifier_factor', 0.5)
        final_score = base_calm_score * price_calmness_amplifier
        combined_control_score = (control_solidity_score * 0.7 + mf_activity_ratio_score * 0.3).clip(-1, 1)
        final_score = final_score.mask(combined_control_score < veto_threshold, 0.0)
        main_force_amplifier = 1 + (combined_control_score * amplifier_factor)
        final_score = (final_score * main_force_amplifier).clip(0, 1).fillna(0.0)
        return final_score

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = "calculate_storm_eye_calm"
        is_debug_enabled_for_method, probe_ts = self._get_debug_info(df, method_name)
        debug_output = {}
        _temp_debug_values = {}
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算风暴眼中的寂静..."] = ""
        df_index = df.index
        # 1. 获取所有参数
        params = self._get_storm_eye_calm_params(config)
        # 2. 校验所有必需的信号
        required_signals = self._get_required_signals(params, params['mtf_slope_accel_weights'], params['mtf_cohesion_base_signals'])
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output_for_storm_eye_calm(debug_output, _temp_debug_values, probe_ts, method_name, pd.Series(0.0, index=df.index, dtype=np.float32))
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 3. 获取原始数据和原子信号
        raw_data = self._get_raw_and_atomic_data(df, method_name, params)
        _temp_debug_values["原始信号值"] = raw_data
        # 4. 计算MTF斜率/加速度分数
        mtf_derived_scores = self._calculate_mtf_derived_scores(df, df_index, params['mtf_slope_accel_weights'], params['mtf_cohesion_base_signals'], method_name)
        _temp_debug_values["MTF斜率/加速度分数"] = mtf_derived_scores
        # 5. 归一化和计算各维度分数
        energy_compression_score = self._calculate_energy_compression_component(df_index, raw_data, mtf_derived_scores, params['energy_compression_weights'])
        _temp_debug_values["能量压缩"] = {"energy_compression_score": energy_compression_score}
        volume_exhaustion_score = self._calculate_volume_exhaustion_component(df_index, raw_data, mtf_derived_scores, params['volume_exhaustion_weights'])
        _temp_debug_values["量能枯竭"] = {"volume_exhaustion_score": volume_exhaustion_score}
        
        # 修改此处：接收元组并分别存储
        main_force_covert_intent_score, main_force_covert_intent_components = self._calculate_main_force_covert_intent_component(df_index, raw_data, mtf_derived_scores, params['main_force_covert_intent_weights'], params['ambiguity_components_weights'])
        _temp_debug_values["主力隐蔽意图"] = main_force_covert_intent_components # 存储组件
        _temp_debug_values["主力隐蔽意图融合"] = {"main_force_covert_intent_score": main_force_covert_intent_score} # 存储融合分数
        
        subdued_market_sentiment_score = self._calculate_subdued_market_sentiment_component(df_index, raw_data, params['subdued_market_sentiment_weights'], params['sentiment_volatility_window'], params['long_term_sentiment_window'], params['sentiment_neutral_range'], params['sentiment_pendulum_neutral_range'])
        _temp_debug_values["市场情绪低迷融合"] = {"subdued_market_sentiment_score": subdued_market_sentiment_score}
        breakout_readiness_score = self._calculate_breakout_readiness_component(df_index, raw_data, params['breakout_readiness_weights'])
        _temp_debug_values["突破准备度融合"] = {"breakout_readiness_score": breakout_readiness_score}
        # 6. 市场情境动态调节器
        market_regime_modulator = self._calculate_market_regime_modulator(df_index, raw_data, params)
        _temp_debug_values["市场情境动态调节器"] = {"market_regime_modulator": market_regime_modulator}
        # 7. 最终融合
        component_scores = {
            'energy_compression': energy_compression_score,
            'volume_exhaustion': volume_exhaustion_score,
            'main_force_covert_intent': main_force_covert_intent_score, # 使用融合分数
            'subdued_market_sentiment': subdued_market_sentiment_score,
            'breakout_readiness': breakout_readiness_score,
            'mtf_cohesion': mtf_derived_scores['mtf_cohesion_score']
        }
        # 调整最终融合权重
        adjusted_final_fusion_weights = {k: v * market_regime_modulator for k, v in params['final_fusion_weights'].items()}
        final_score = self._perform_final_fusion(df_index, component_scores, adjusted_final_fusion_weights, params['price_calmness_modulator_params'], params['main_force_control_adjudicator_params'], raw_data)
        _temp_debug_values["最终融合"] = {"final_score": final_score}
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            self._print_debug_output_for_storm_eye_calm(debug_output, _temp_debug_values, probe_ts, method_name, final_score)
        return final_score.astype(np.float32)

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

