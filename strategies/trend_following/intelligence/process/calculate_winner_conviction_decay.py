# strategies\trend_following\intelligence\process\calculate_winner_conviction_decay.py
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

class CalculateWinnerConvictionDecay:
    """
    【V4.1 · 全息动态审判版】“赢家信念衰减”专属计算引擎
    PROCESS_META_WINNER_CONVICTION_DECAY
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = helper_instance

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.1 · 全息动态审判版】“赢家信念衰减”专属计算引擎
        - 核心重构: 引入更多维度信号，深化对信念衰减、利润压力、派发确认和情境调制的感知。
        - 核心升级: 引入“买盘抵抗瓦解”证据，强化“诡道派发”识别，扩展情境调制器。
        - 核心优化: 引入“动态融合指数”，根据市场波动率和情绪动态调整最终融合的非线性指数。
        - 核心逻辑: 最终衰减分 = (核心衰减分 * (1 + 情境调制器))^动态非线性指数。
        """
        method_name = "calculate_winner_conviction_decay"
        # --- 调试信息构建 ---
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
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算赢家信念衰减..."] = ""
        df_index = df.index
        # 1. 获取所有参数和所需信号列表
        params_dict, all_required_signals = self._get_decay_params_and_signals(config, method_name)
        # 2. 校验所有必需的信号
        if not self.helper._validate_required_signals(df, all_required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self.helper._print_debug_output(debug_output)
            return pd.Series(dtype=np.float32)
        # 3. 获取原始数据
        raw_signals = self._get_raw_signals(df, df_index, params_dict, method_name)
        _temp_debug_values["原始信号值"] = raw_signals # 存储原始信号值用于调试
        # 4. 计算信念强度
        conviction_strength_score = self._calculate_conviction_strength(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        # 5. 计算压力韧性
        pressure_resilience_score = self._calculate_pressure_resilience(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values)
        # 6. 计算共振与背离因子
        synergy_factor = self._calculate_synergy_factor(conviction_strength_score, pressure_resilience_score, _temp_debug_values)
        # 7. 计算诡道过滤
        deception_filter = self._calculate_deception_filter(df, df_index, params_dict, method_name, _temp_debug_values)
        # 8. 计算情境调制
        context_modulator = self._calculate_contextual_modulator(df, df_index, raw_signals, params_dict, method_name, _temp_debug_values, is_debug_enabled_for_method, probe_ts)
        # 9. 最终融合
        final_score = self._perform_final_fusion(df_index, conviction_strength_score, pressure_resilience_score, synergy_factor, deception_filter, context_modulator, params_dict, _temp_debug_values)
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            self._collect_and_print_debug_info(method_name, probe_ts, debug_output, _temp_debug_values, final_score)
        return final_score.astype(np.float32)

    def _collect_and_print_debug_info(self, method_name: str, probe_ts: pd.Timestamp, debug_output: Dict, _temp_debug_values: Dict, final_score: pd.Series):
        """
        统一收集并打印 calculate_winner_conviction_decay 的调试信息。
        """
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        for key, value in _temp_debug_values["原始信号值"].items():
            if isinstance(value, pd.Series):
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            elif isinstance(value, dict): # 处理 _temp_debug_values["原始信号值"] 中的字典
                debug_output[f"        '{key}':"] = ""
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.Series):
                        val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                        debug_output[f"          {sub_key}: {val:.4f}"] = ""
                    else:
                        debug_output[f"          {sub_key}: {sub_value}"] = ""
            else:
                debug_output[f"        '{key}': {value}"] = ""
        sections = ["信念强度", "压力韧性", "共振与背离因子", "诡道过滤", "情境调制", "最终融合"]
        for section_name in sections:
            if section_name in _temp_debug_values:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section_name} ---"] = ""
                for key, series_or_val in _temp_debug_values[section_name].items():
                    if isinstance(series_or_val, pd.Series):
                        val = series_or_val.loc[probe_ts] if probe_ts in series_or_val.index else np.nan
                        debug_output[f"        {key}: {val:.4f}"] = ""
                    elif isinstance(series_or_val, dict):
                        debug_output[f"        {key}:"] = ""
                        for sub_key, sub_value in series_or_val.items():
                            if isinstance(sub_value, pd.Series):
                                val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                                debug_output[f"          {sub_key}: {val:.4f}"] = ""
                            else:
                                debug_output[f"          {sub_key}: {sub_value}"] = ""
                    else:
                        debug_output[f"        {key}: {series_or_val}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 赢家信念衰减诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        self.helper._print_debug_output(debug_output)

    def _prepare_norm_market_sentiment(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        准备归一化后的市场情绪信号。
        """
        market_sentiment_raw = raw_signals["market_sentiment_raw"]
        return self.helper._normalize_series(market_sentiment_raw, df_index, bipolar=True)

    def _prepare_norm_volatility_stability(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series], method_name: str, is_debug_enabled_for_method: bool, probe_ts: pd.Timestamp) -> pd.Series:
        """
        准备归一化后的波动率稳定性信号。
        """
        volatility_instability_raw = raw_signals["volatility_instability_raw"]
        volatility_stability_raw = 1 - normalize_score(
            volatility_instability_raw, 
            df_index, 
            21, 
            ascending=True,
            debug_info=(is_debug_enabled_for_method, probe_ts, f"{method_name}_volatility_stability_norm")
        )
        return self.helper._normalize_series(volatility_stability_raw, df_index, bipolar=False, ascending=True)

    def _prepare_norm_trend_vitality(self, df_index: pd.Index, raw_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        准备归一化后的趋势活力信号。
        """
        trend_vitality_raw = raw_signals["trend_vitality_raw"]
        return self.helper._normalize_series(trend_vitality_raw, df_index, bipolar=False)

    def _get_decay_params_and_signals(self, config: Dict, method_name: str) -> Tuple[Dict, List[str]]:
        """
        获取赢家信念衰减计算所需的所有参数和信号列表。
        """
        decay_params = get_param_value(config.get('winner_conviction_decay_params'), {})
        mtf_slope_accel_weights = get_param_value(decay_params.get('mtf_slope_accel_weights'), {"slope_periods": {"5": 0.4, "13": 0.3}, "accel_periods": {"5": 0.6}})
        belief_decay_components_weights = get_param_value(decay_params.get('belief_decay_components_weights'), {
            "winner_stability_mtf": 0.4, "winner_profit_margin_avg_inverted": 0.2,
            "total_winner_rate_inverted": 0.2, "chip_fatigue": 0.1,
            "chip_health_inverted": 0.1
        })
        profit_pressure_components_weights = get_param_value(decay_params.get('profit_pressure_components_weights'), {
            "profit_taking_flow_mtf": 0.3, "active_selling_pressure": 0.2,
            "rally_sell_distribution_intensity": 0.2, "main_force_t0_sell_efficiency": 0.15,
            "main_force_on_peak_sell_flow": 0.15
        })
        distribution_confirmation_components_weights = get_param_value(decay_params.get('distribution_confirmation_components_weights'), {
            "distribution_intent": 0.3, "chip_distribution_whisper": 0.2,
            "upper_shadow_selling_pressure": 0.2, "deception_lure_long": 0.15,
            "wash_trade_intensity": 0.15
        })
        buying_resistance_collapse_weights = get_param_value(decay_params.get('buying_resistance_collapse_weights'), {
            "pressure_rejection_strength_inverted": 0.25, "rally_buy_support_weakness": 0.25,
            "buy_quote_exhaustion": 0.2, "bid_side_liquidity_inverted": 0.15,
            "main_force_slippage": 0.15
        })
        contextual_modulator_weights = get_param_value(decay_params.get('contextual_modulator_weights'), {
            "market_sentiment": 0.1,
            "volatility_stability": 0.1,
            "trend_vitality": 0.1,
            "mtf_market_sentiment": 0.1,
            "mtf_volatility_stability": 0.1,
            "mtf_trend_vitality": 0.1,
            "upper_shadow_pressure": 0.1,
            "mtf_upper_shadow_pressure": 0.1,
            "context_resonance": 0.2
        })
        contextual_mtf_config = get_param_value(decay_params.get('contextual_mtf_config'), {})
        dynamic_fusion_exponent_params = get_param_value(decay_params.get('dynamic_fusion_exponent_params'), {"enabled": False, "base_exponent": 1.5})
        price_overextension_composite_weights = get_param_value(decay_params.get('price_overextension_composite_weights'), {"bias_13": 0.3, "bias_21": 0.2, "rsi_13": 0.3, "bbp_21": 0.2})
        relative_position_weights = get_param_value(decay_params.get('relative_position_weights'), {"winner_stability_high": 0.6, "profit_taking_flow_low": 0.4})
        final_fusion_gm_weights = get_param_value(decay_params.get('final_fusion_gm_weights'), {
            "conviction_magnitude": 0.3, "pressure_magnitude": 0.25,
            "synergy_factor": 0.2, "deception_filter": 0.15,
            "context_modulator": 0.1
        })
        final_exponent = get_param_value(decay_params.get('final_exponent'), 1.5)
        # 核心信号名称
        belief_signal_name = 'winner_stability_index_D'
        pressure_signal_name = 'profit_taking_flow_ratio_D'
        # 更新所有必需的DF列和原子信号
        required_df_columns = [
            belief_signal_name, pressure_signal_name,
            'upper_shadow_selling_pressure_D', 'retail_fomo_premium_index_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'BIAS_13_D', 'BIAS_21_D', 'RSI_13_D', 'BBP_21_2.0_D',
            'winner_profit_margin_avg_D', 'total_winner_rate_D', 'chip_fatigue_index_D',
            'active_selling_pressure_D', 'rally_sell_distribution_intensity_D',
            'main_force_t0_sell_efficiency_D', 'main_force_on_peak_sell_flow_D',
            'deception_lure_long_intensity_D', 'wash_trade_intensity_D',
            'pressure_rejection_strength_D', 'rally_buy_support_weakness_D',
            'buy_quote_exhaustion_rate_D', 'bid_side_liquidity_D', 'main_force_slippage_index_D',
            'structural_tension_index_D', 'volatility_expansion_ratio_D',
            'chip_health_score_D', 'market_impact_cost_D',
            'trend_vitality_index_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        # 原始信号列表，用于生成MTF衍生信号
        base_signals_for_mtf = [
            belief_signal_name, pressure_signal_name, 'winner_profit_margin_avg_D', 'total_winner_rate_D', 'chip_fatigue_index_D',
            'active_selling_pressure_D', 'rally_sell_distribution_intensity_D', 'main_force_t0_sell_efficiency_D',
            'main_force_on_peak_sell_flow_D', 'deception_lure_long_intensity_D', 'wash_trade_intensity_D',
            'pressure_rejection_strength_D', 'rally_buy_support_weakness_D', 'buy_quote_exhaustion_rate_D',
            'bid_side_liquidity_D', 'main_force_slippage_index_D', 'structural_tension_index_D',
            'volatility_expansion_ratio_D', 'chip_health_score_D', 'market_impact_cost_D', 'trend_vitality_index_D',
            # 新增用于复合信号的原始信号
            'market_sentiment_score_D', 'retail_fomo_premium_index_D', 'retail_panic_surrender_index_D',
        ]
        for base_sig in base_signals_for_mtf:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_df_columns.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_df_columns.append(f'ACCEL_{period_str}_{base_sig}')
        # 根据 contextual_mtf_config 动态添加情境调制器所需的 MTF 信号
        for config_key, config_val in contextual_mtf_config.items():
            if isinstance(config_val, dict):
                base_signal_name_for_mtf = config_val.get('base_signal_name')
                if base_signal_name_for_mtf:
                    for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                        required_df_columns.append(f'SLOPE_{period_str}_{base_signal_name_for_mtf}')
                    for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                        required_df_columns.append(f'ACCEL_{period_str}_{base_signal_name_for_mtf}')
        # 移除 required_atomic_signals
        all_required_signals = list(set(required_df_columns)) # 使用 set 去重
        params_dict = {
            'decay_params': decay_params,
            'mtf_slope_accel_weights': mtf_slope_accel_weights,
            'belief_decay_components_weights': belief_decay_components_weights,
            'profit_pressure_components_weights': profit_pressure_components_weights,
            'distribution_confirmation_components_weights': distribution_confirmation_components_weights,
            'buying_resistance_collapse_weights': buying_resistance_collapse_weights,
            'contextual_modulator_weights': contextual_modulator_weights,
            'contextual_mtf_config': contextual_mtf_config,
            'dynamic_fusion_exponent_params': dynamic_fusion_exponent_params,
            'price_overextension_composite_weights': price_overextension_composite_weights,
            'relative_position_weights': relative_position_weights,
            'final_fusion_gm_weights': final_fusion_gm_weights,
            'final_exponent': final_exponent,
            'belief_signal_name': belief_signal_name,
            'pressure_signal_name': pressure_signal_name,
            # 新增复合信号的权重配置，如果需要可配置化
            'composite_distribution_intent_weights': get_param_value(decay_params.get('composite_distribution_intent_weights'), {
                "active_selling_pressure": 0.25, "rally_sell_distribution_intensity": 0.25,
                "main_force_t0_sell_efficiency": 0.2, "main_force_on_peak_sell_flow": 0.15,
                "upper_shadow_selling_pressure": 0.15
            }),
            'composite_chip_distribution_whisper_weights': get_param_value(decay_params.get('composite_chip_distribution_whisper_weights'), {
                "buy_quote_exhaustion": 0.3, "bid_side_liquidity_inverted": 0.25,
                "main_force_slippage_inverted": 0.25, "market_impact_cost": 0.2
            }),
            'composite_market_tension_weights': get_param_value(decay_params.get('composite_market_tension_weights'), {
                "structural_tension": 0.4, "volatility_expansion": 0.3,
                "market_sentiment_inverted": 0.3
            }),
            'composite_sentiment_pendulum_weights': get_param_value(decay_params.get('composite_sentiment_pendulum_weights'), {
                "market_sentiment": 0.4, "retail_fomo_premium": 0.3,
                "retail_panic_surrender_inverted": 0.3
            }),
            'composite_deception_index_weights': get_param_value(decay_params.get('composite_deception_index_weights'), {
                "deception_lure_long": 0.6, "wash_trade_intensity": 0.4
            }),
        }
        return params_dict, all_required_signals

    def _get_raw_signals(self, df: pd.DataFrame, df_index: pd.Index, params_dict: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        获取所有原始数据和复合信号。
        """
        belief_signal_name = params_dict['belief_signal_name']
        pressure_signal_name = params_dict['pressure_signal_name']
        contextual_mtf_config = params_dict['contextual_mtf_config']
        mtf_slope_accel_weights = params_dict['mtf_slope_accel_weights']
        raw_signals = {
            "belief_signal_raw": self.helper._get_safe_series(df, belief_signal_name, 0.0, method_name=method_name),
            "pressure_signal_raw": self.helper._get_safe_series(df, pressure_signal_name, 0.0, method_name=method_name),
            "upper_shadow_pressure_raw": self.helper._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name=method_name),
            "retail_fomo_raw": self.helper._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name=method_name),
            "market_sentiment_raw": self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name),
            "volatility_instability_raw": self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name),
            "bias_13_raw": self.helper._get_safe_series(df, 'BIAS_13_D', 0.0, method_name=method_name),
            "bias_21_raw": self.helper._get_safe_series(df, 'BIAS_21_D', 0.0, method_name=method_name),
            "rsi_13_raw": self.helper._get_safe_series(df, 'RSI_13_D', 0.0, method_name=method_name),
            "bbp_21_raw": self.helper._get_safe_series(df, 'BBP_21_2.0_D', 0.0, method_name=method_name),
            # 替换原子信号为复合信号
            "distribution_intent_score": self._calculate_composite_distribution_intent(df, df_index, mtf_slope_accel_weights, params_dict['composite_distribution_intent_weights'], method_name),
            "chip_distribution_whisper_score": self._calculate_composite_chip_distribution_whisper(df, df_index, mtf_slope_accel_weights, params_dict['composite_chip_distribution_whisper_weights'], method_name),
            "market_tension_score": self._calculate_composite_market_tension(df, df_index, mtf_slope_accel_weights, params_dict['composite_market_tension_weights'], method_name),
            "sentiment_pendulum_score": self._calculate_composite_sentiment_pendulum(df, df_index, mtf_slope_accel_weights, params_dict['composite_sentiment_pendulum_weights'], method_name),
            # 原始信号
            "winner_profit_margin_avg_raw": self.helper._get_safe_series(df, 'winner_profit_margin_avg_D', 0.0, method_name=method_name),
            "total_winner_rate_raw": self.helper._get_safe_series(df, 'total_winner_rate_D', 0.0, method_name=method_name),
            "chip_fatigue_raw": self.helper._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name=method_name),
            "active_selling_pressure_raw": self.helper._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name=method_name),
            "rally_sell_distribution_intensity_raw": self.helper._get_safe_series(df, 'rally_sell_distribution_intensity_D', 0.0, method_name=method_name),
            "main_force_t0_sell_efficiency_raw": self.helper._get_safe_series(df, 'main_force_t0_sell_efficiency_D', 0.0, method_name=method_name),
            "main_force_on_peak_sell_flow_raw": self.helper._get_safe_series(df, 'main_force_on_peak_sell_flow_D', 0.0, method_name=method_name),
            "deception_lure_long_raw": self.helper._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name),
            "wash_trade_intensity_raw": self.helper._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name),
            "pressure_rejection_strength_raw": self.helper._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name=method_name),
            "rally_buy_support_weakness_raw": self.helper._get_safe_series(df, 'rally_buy_support_weakness_D', 0.0, method_name=method_name),
            "buy_quote_exhaustion_raw": self.helper._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name=method_name),
            "bid_side_liquidity_raw": self.helper._get_safe_series(df, 'bid_side_liquidity_D', 0.0, method_name=method_name),
            "main_force_slippage_raw": self.helper._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name=method_name),
            "structural_tension_raw": self.helper._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name=method_name),
            "volatility_expansion_raw": self.helper._get_safe_series(df, 'volatility_expansion_ratio_D', 0.0, method_name=method_name),
            "chip_health_raw": self.helper._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name),
            "market_impact_cost_raw": self.helper._get_safe_series(df, 'market_impact_cost_D', 0.0, method_name=method_name),
            "trend_vitality_raw": self.helper._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        }
        # 获取情境调制器所需的 MTF 信号 (保持不变，因为它们是基于原始信号的MTF)
        for config_key, config_val in contextual_mtf_config.items():
            if isinstance(config_val, dict):
                base_signal_name_for_mtf = config_val.get('base_signal_name')
                if base_signal_name_for_mtf:
                    for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                        slope_col = f'SLOPE_{period_str}_{base_signal_name_for_mtf}'
                        raw_signals[slope_col] = self.helper._get_safe_series(df, slope_col, 0.0, method_name=method_name)
                    for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                        accel_col = f'ACCEL_{period_str}_{base_signal_name_for_mtf}'
                        raw_signals[accel_col] = self.helper._get_safe_series(df, accel_col, 0.0, method_name=method_name)
        return raw_signals

    def _calculate_composite_distribution_intent(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的派发意图信号，替代 SCORE_BEHAVIOR_DISTRIBUTION_INTENT。
        """
        components = {
            "active_selling_pressure": {'signal': 'active_selling_pressure_D', 'bipolar': True, 'ascending': True},
            "rally_sell_distribution_intensity": {'signal': 'rally_sell_distribution_intensity_D', 'bipolar': True, 'ascending': True},
            "main_force_t0_sell_efficiency": {'signal': 'main_force_t0_sell_efficiency_D', 'bipolar': True, 'ascending': True},
            "main_force_on_peak_sell_flow": {'signal': 'main_force_on_peak_sell_flow_D', 'bipolar': True, 'ascending': True},
            "upper_shadow_selling_pressure": {'signal': 'upper_shadow_selling_pressure_D', 'bipolar': True, 'ascending': True},
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_composite_chip_distribution_whisper(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的筹码派发低语信号，替代 SCORE_CHIP_RISK_DISTRIBUTION_WHISPER。
        """
        components = {
            "buy_quote_exhaustion": {'signal': 'buy_quote_exhaustion_rate_D', 'bipolar': True, 'ascending': True},
            "bid_side_liquidity_inverted": {'signal': 'bid_side_liquidity_D', 'bipolar': True, 'ascending': False}, # 流动性低 -> 派发风险高
            "main_force_slippage_inverted": {'signal': 'main_force_slippage_index_D', 'bipolar': True, 'ascending': True}, # 滑点高 -> 派发成本高，但这里是反向，所以滑点低 -> 派发风险高
            "market_impact_cost": {'signal': 'market_impact_cost_D', 'bipolar': True, 'ascending': True},
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_composite_market_tension(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的市场张力信号，替代 SCORE_FOUNDATION_AXIOM_MARKET_TENSION。
        """
        components = {
            "structural_tension": {'signal': 'structural_tension_index_D', 'bipolar': True, 'ascending': True},
            "volatility_expansion": {'signal': 'volatility_expansion_ratio_D', 'bipolar': True, 'ascending': True},
            "market_sentiment_inverted": {'signal': 'market_sentiment_score_D', 'bipolar': True, 'ascending': False}, # 市场情绪低 -> 张力高
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_composite_sentiment_pendulum(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的情绪摆动信号，替代 SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM。
        """
        components = {
            "market_sentiment": {'signal': 'market_sentiment_score_D', 'bipolar': True, 'ascending': True},
            "retail_fomo_premium": {'signal': 'retail_fomo_premium_index_D', 'bipolar': True, 'ascending': True},
            "retail_panic_surrender_inverted": {'signal': 'retail_panic_surrender_index_D', 'bipolar': True, 'ascending': False}, # 散户恐慌低 -> 情绪好
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_composite_deception_index(self, df: pd.DataFrame, df_index: pd.Index, mtf_weights_config: Dict, composite_weights: Dict, method_name: str) -> pd.Series:
        """
        计算复合的欺骗指数，替代直接引用 'deception_index_D'。
        """
        components = {
            "deception_lure_long": {'signal': 'deception_lure_long_intensity_D', 'bipolar': True, 'ascending': True},
            "wash_trade_intensity": {'signal': 'wash_trade_intensity_D', 'bipolar': True, 'ascending': True},
        }
        fused_scores = []
        total_weight = 0.0
        for key, config in components.items():
            raw_signal = self.helper._get_safe_series(df, config['signal'], np.nan, method_name=method_name)
            if raw_signal.isnull().all():
                continue
            mtf_score = self.helper._get_mtf_slope_accel_score(
                df, config['signal'], mtf_weights_config, df_index, method_name,
                bipolar=config['bipolar'], ascending=config['ascending']
            )
            weight = composite_weights.get(key, 0.0)
            if pd.notna(mtf_score).any() and weight > 0:
                fused_scores.append(mtf_score * weight)
                total_weight += weight
        if not fused_scores or total_weight == 0:
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        composite_score = sum(fused_scores) / total_weight
        return composite_score.clip(-1, 1)

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        计算赢家信念强度（此处分数越高代表衰减越严重）。
        """
        belief_signal_name = params_dict['belief_signal_name']
        mtf_slope_accel_weights = params_dict['mtf_slope_accel_weights']
        # relative_position_weights = params_dict['relative_position_weights'] # 未使用，移除
        belief_decay_components_weights = params_dict['belief_decay_components_weights']
        # --- 原料数据存在性检查 ---
        required_raw_signals = [
            "belief_signal_raw", "chip_health_raw", "winner_profit_margin_avg_raw",
            "total_winner_rate_raw", "chip_fatigue_raw"
        ]
        for sig_name in required_raw_signals:
            if sig_name not in raw_signals or raw_signals[sig_name].empty or raw_signals[sig_name].isnull().all():
                print(f"    -> [过程情报警告] {method_name}: 缺少核心原始信号 '{sig_name}' 或其值全为NaN，信念强度计算可能不完整。")
                return pd.Series(np.nan, index=df_index, dtype=np.float32)
        belief_signal_raw = raw_signals["belief_signal_raw"]
        chip_health_raw = raw_signals["chip_health_raw"]
        winner_profit_margin_avg_raw = raw_signals["winner_profit_margin_avg_raw"]
        total_winner_rate_raw = raw_signals["total_winner_rate_raw"]
        chip_fatigue_raw = raw_signals["chip_fatigue_raw"]
        # 1. 计算MTF获利盘稳定性 (正值代表稳定性趋势向上，对衰减是负贡献)
        mtf_winner_stability = self.helper._get_mtf_slope_accel_score(df, belief_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 2. 归一化平均获利盘利润率 (正值代表利润率高，对衰减是负贡献)
        norm_profit_margin_avg = self.helper._normalize_series(winner_profit_margin_avg_raw, df_index, bipolar=True, ascending=True)
        # 3. 归一化总获利盘比例 (正值代表获利盘比例高，对衰减是负贡献)
        norm_total_winner_rate = self.helper._normalize_series(total_winner_rate_raw, df_index, bipolar=True, ascending=True)
        # 4. 归一化筹码疲劳指数 (正值代表疲劳度高，对衰减是正贡献)
        norm_chip_fatigue = self.helper._normalize_series(chip_fatigue_raw, df_index, bipolar=True, ascending=True)
        # 5. 归一化筹码健康度反向 (正值代表健康度低，对衰减是正贡献)
        norm_chip_health_inverted = self.helper._normalize_series(chip_health_raw, df_index, bipolar=True, ascending=False)
        # 提取相关权重
        w_mtf_stability = belief_decay_components_weights.get("winner_stability_mtf", 0.4)
        w_winner_profit_margin_inverted = belief_decay_components_weights.get("winner_profit_margin_avg_inverted", 0.2)
        w_total_winner_rate_inverted = belief_decay_components_weights.get("total_winner_rate_inverted", 0.2)
        w_chip_fatigue = belief_decay_components_weights.get("chip_fatigue", 0.1)
        w_chip_health_inverted = belief_decay_components_weights.get("chip_health_inverted", 0.1)
        # 计算总权重，避免除以零
        total_weights = (
            w_mtf_stability + w_winner_profit_margin_inverted + w_total_winner_rate_inverted +
            w_chip_fatigue + w_chip_health_inverted
        )
        if total_weights == 0:
            fused_conviction_score = pd.Series(np.nan, index=df_index, dtype=np.float32)
        else:
            # 融合信念强度组件 (分数越高代表衰减越严重)
            fused_conviction_score = (
                (-mtf_winner_stability * w_mtf_stability) + # 稳定性高 -> 衰减低
                (-norm_profit_margin_avg * w_winner_profit_margin_inverted) + # 利润率高 -> 衰减低
                (-norm_total_winner_rate * w_total_winner_rate_inverted) + # 获利盘高 -> 衰减低
                (norm_chip_fatigue * w_chip_fatigue) + # 疲劳度高 -> 衰减高
                (norm_chip_health_inverted * w_chip_health_inverted) # 健康度低 -> 衰减高
            ) / total_weights
        conviction_strength_score = fused_conviction_score.clip(-1, 1).fillna(np.nan)
        _temp_debug_values["信念强度"] = {
            "mtf_winner_stability": mtf_winner_stability,
            "norm_profit_margin_avg": norm_profit_margin_avg,
            "norm_total_winner_rate": norm_total_winner_rate,
            "norm_chip_fatigue": norm_chip_fatigue,
            "norm_chip_health_inverted": norm_chip_health_inverted,
            "conviction_strength_score": conviction_strength_score
        }
        return conviction_strength_score

    def _calculate_pressure_resilience(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        计算压力韧性。
        """
        pressure_signal_name = params_dict['pressure_signal_name']
        mtf_slope_accel_weights = params_dict['mtf_slope_accel_weights']
        relative_position_weights = params_dict['relative_position_weights']
        # --- 原料数据存在性检查 ---
        required_raw_signals = ["pressure_signal_raw"]
        for sig_name in required_raw_signals:
            if sig_name not in raw_signals or raw_signals[sig_name].empty or raw_signals[sig_name].isnull().all():
                print(f"    -> [过程情报警告] {method_name}: 缺少核心原始信号 '{sig_name}' 或其值全为NaN，压力韧性计算可能不完整。")
                return pd.Series(np.nan, index=df_index, dtype=np.float32)
        pressure_signal_raw = raw_signals["pressure_signal_raw"]
        mtf_profit_taking_flow = self.helper._get_mtf_slope_accel_score(df, pressure_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        profit_taking_flow_percentile = (1 - pressure_signal_raw.rank(pct=True)).fillna(np.nan) # fillna(np.nan)
        pressure_resilience_score = ((mtf_profit_taking_flow * -1) * relative_position_weights.get("profit_taking_flow_low", 0.4) + 
                                     (profit_taking_flow_percentile * 2 - 1) * (1 - relative_position_weights.get("profit_taking_flow_low", 0.4))).clip(-1, 1)
        _temp_debug_values["压力韧性"] = {
            "mtf_profit_taking_flow": mtf_profit_taking_flow,
            "profit_taking_flow_percentile": profit_taking_flow_percentile,
            "pressure_resilience_score": pressure_resilience_score
        }
        return pressure_resilience_score

    def _calculate_synergy_factor(self, conviction_strength_score: pd.Series, pressure_resilience_score: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        计算共振与背离因子。
        """
        norm_conviction = (conviction_strength_score + 1) / 2
        norm_resilience = (pressure_resilience_score + 1) / 2
        synergy_factor = (norm_conviction * norm_resilience + (1 - norm_conviction) * (1 - norm_resilience)).clip(0, 1)
        _temp_debug_values["共振与背离因子"] = {
            "norm_conviction": norm_conviction,
            "norm_resilience": norm_resilience,
            "synergy_factor": synergy_factor
        }
        return synergy_factor

    def _calculate_deception_filter(self, df: pd.DataFrame, df_index: pd.Index, params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        计算诡道过滤因子。
        """
        mtf_slope_accel_weights = params_dict['mtf_slope_accel_weights']
        composite_deception_index_weights = params_dict['composite_deception_index_weights']
        # 使用新计算的复合欺骗指数
        composite_deception_index = self._calculate_composite_deception_index(df, df_index, mtf_slope_accel_weights, composite_deception_index_weights, method_name)
        # 确保原始信号存在且有效，否则 _get_mtf_slope_accel_score 会返回NaN
        mtf_wash_trade_intensity = self.helper._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 如果任何一个MTF分数是NaN，则deception_penalty也会是NaN
        deception_penalty = (composite_deception_index * 0.6 + mtf_wash_trade_intensity * 0.4).clip(0, 1)
        deception_filter = (1 - deception_penalty).clip(0, 1)
        _temp_debug_values["诡道过滤"] = {
            "composite_deception_index": composite_deception_index,
            "mtf_wash_trade_intensity": mtf_wash_trade_intensity,
            "deception_penalty": deception_penalty,
            "deception_filter": deception_filter
        }
        return deception_filter

    def _calculate_contextual_modulator(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict, is_debug_enabled_for_method: bool, probe_ts: pd.Timestamp) -> pd.Series:
        """
        计算情境调制因子。
        """
        contextual_modulator_weights = params_dict['contextual_modulator_weights']
        contextual_mtf_config = params_dict['contextual_mtf_config']
        mtf_slope_accel_weights = params_dict['mtf_slope_accel_weights']

        # 辅助函数：将配置中的信号名映射到 raw_signals 字典中的实际键名
        def _map_config_signal_to_raw_signals_key(config_signal_name: str) -> str:
            if config_signal_name == 'market_sentiment_score_D':
                return 'market_sentiment_raw'
            elif config_signal_name == 'VOLATILITY_INSTABILITY_INDEX_21d_D':
                return 'volatility_instability_raw'
            elif config_signal_name == 'trend_vitality_index_D':
                return 'trend_vitality_raw'
            elif config_signal_name == 'upper_shadow_selling_pressure_D':
                return 'upper_shadow_pressure_raw'
            # 对于其他信号，假设配置名就是 raw_signals 中的键名
            return config_signal_name

        # --- 原料数据存在性检查 ---
        required_raw_signals = [
            "market_sentiment_raw", "volatility_instability_raw", "trend_vitality_raw",
            "upper_shadow_pressure_raw",
            # 新增复合信号的原始名称，确保它们在 raw_signals 中
            "market_tension_score", "sentiment_pendulum_score"
        ]
        for sig_name in required_raw_signals:
            if sig_name not in raw_signals or raw_signals[sig_name].empty or raw_signals[sig_name].isnull().all():
                print(f"    -> [过程情报警告] {method_name}: 缺少核心原始信号 '{sig_name}' 或其值全为NaN，情境调制计算可能不完整。")
                return pd.Series(np.nan, index=df_index, dtype=np.float32)
        # --- 1. 计算当前情境信号 ---
        norm_market_sentiment = self.helper._normalize_series(raw_signals["market_sentiment_raw"], df_index, bipolar=True)
        # 波动率不稳定性转换为稳定性，并归一化
        volatility_stability_raw = 1 - normalize_score(
            raw_signals["volatility_instability_raw"],
            df_index,
            21,
            ascending=True
        )
        norm_volatility_stability = self.helper._normalize_series(volatility_stability_raw, df_index, bipolar=False, ascending=True)
        norm_trend_vitality = self.helper._normalize_series(raw_signals["trend_vitality_raw"], df_index, bipolar=False)
        # 新增：上影线卖压归一化 (卖压越高，信念衰减越严重)
        # ascending=False，使高卖压对应低分数（负面影响）
        norm_upper_shadow_pressure = self.helper._normalize_series(raw_signals["upper_shadow_pressure_raw"], df_index, bipolar=True, ascending=False)
        # --- 2. 计算MTF增强情境信号 ---
        mtf_enhanced_signals = {}
        mtf_signals_for_resonance = [] # 用于计算共振的MTF信号
        for config_key, config_val in contextual_mtf_config.items():
            if isinstance(config_val, dict):
                base_signal_name_from_config = config_val.get('base_signal_name')
                if not base_signal_name_from_config:
                    continue

                # 使用映射函数获取 raw_signals 中的正确键名
                raw_signals_key = _map_config_signal_to_raw_signals_key(base_signal_name_from_config)

                raw_series_for_processing = raw_signals.get(raw_signals_key)
                if raw_series_for_processing is None or raw_series_for_processing.empty or raw_series_for_processing.isnull().all():
                    print(f"    -> [过程情报警告] {method_name}: 缺少MTF分析所需原始信号 '{base_signal_name_from_config}' (映射到 '{raw_signals_key}') 或其值为空。")
                    continue
                # 确定传递给 _get_mtf_score_from_series_slope_accel 的 ascending 参数
                ascending_param_for_mtf = True
                if config_val.get('inverted_for_decay', False) or config_val.get('inverted_for_stability', False):
                    ascending_param_for_mtf = False
                # 计算MTF斜率/加速度融合分数
                mtf_score = self.helper._get_mtf_score_from_series_slope_accel(
                    raw_series_for_processing, # 直接传入 Series
                    mtf_slope_accel_weights,
                    df_index,
                    method_name,
                    bipolar=config_val.get('bipolar', True),
                    ascending=ascending_param_for_mtf
                )
                mtf_enhanced_signals[f"mtf_{config_key}"] = mtf_score
                mtf_signals_for_resonance.append(mtf_score)
        # --- 3. 计算情境共振 ---
        context_mtf_resonance = pd.Series(np.nan, index=df_index, dtype=np.float32)
        if len(mtf_signals_for_resonance) >= 2:
            fused_scores_df = pd.DataFrame(mtf_enhanced_signals, index=df_index)
            if not fused_scores_df.empty and len(fused_scores_df.columns) >= 2:
                mean_scores = fused_scores_df.mean(axis=1)
                std_scores = fused_scores_df.std(axis=1).fillna(np.nan)
                max_possible_std = fused_scores_df.max(axis=1) - fused_scores_df.min(axis=1)
                max_possible_std = max_possible_std.replace(0, 1)
                normalized_std = (std_scores / max_possible_std).clip(0, 1)
                consistency_strength = (1 - normalized_std).fillna(np.nan)
                context_mtf_resonance = (mean_scores * consistency_strength).clip(-1, 1).astype(np.float32)
            else:
                print(f"    -> [过程情报警告] {method_name}: MTF共振计算至少需要2个有效MTF增强信号，当前只有 {len(mtf_enhanced_signals)} 个。共振分设置为np.nan。")
        # --- 4. 融合所有情境因子 ---
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_stability": norm_volatility_stability,
            "trend_vitality": norm_trend_vitality,
            "upper_shadow_pressure": norm_upper_shadow_pressure,
            **mtf_enhanced_signals,
            "context_resonance": context_mtf_resonance
        }
        # 确保输入 _robust_geometric_mean 的所有分数都是正值
        positive_components = {}
        for k, v in context_modulator_components.items():
            if v is None or v.empty or v.isnull().all(): # 增加对空或全NaN Series的检查
                continue
            if v.min() < 0:
                positive_components[k] = (v + 1) / 2
            else:
                positive_components[k] = v
        context_modulator_score = _robust_geometric_mean(
            positive_components,
            contextual_modulator_weights,
            df_index
        )
        context_modulator = 0.5 + context_modulator_score
        # --- 探针输出 ---
        _temp_debug_values["情境调制"] = {
            "norm_market_sentiment": norm_market_sentiment,
            "volatility_stability_raw": volatility_stability_raw,
            "norm_volatility_stability": norm_volatility_stability,
            "norm_trend_vitality": norm_trend_vitality,
            "norm_upper_shadow_pressure": norm_upper_shadow_pressure,
            "mtf_enhanced_signals": mtf_enhanced_signals,
            "context_mtf_resonance": context_mtf_resonance,
            "context_modulator_score": context_modulator_score,
            "context_modulator": context_modulator
        }
        return context_modulator

    def _perform_final_fusion(self, df_index: pd.Index, conviction_strength_score: pd.Series, pressure_resilience_score: pd.Series, synergy_factor: pd.Series, deception_filter: pd.Series, context_modulator: pd.Series, params_dict: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        执行最终的融合计算。
        """
        decay_params = params_dict['decay_params']
        final_fusion_gm_weights = params_dict['final_fusion_gm_weights']
        final_exponent = params_dict['final_exponent']
        direction_weight_conviction = get_param_value(decay_params.get('direction_weights', {}).get('conviction', 0.6), 0.6)
        direction_weight_pressure = get_param_value(decay_params.get('direction_weights', {}).get('pressure', 0.4), 0.4)
        overall_direction_raw = (conviction_strength_score * direction_weight_conviction + pressure_resilience_score * direction_weight_pressure)
        overall_direction = np.sign(overall_direction_raw)
        overall_direction = overall_direction.replace(0, np.nan) # 0方向改为NaN
        conviction_magnitude = (conviction_strength_score.abs() + 1) / 2
        pressure_magnitude = (pressure_resilience_score.abs() + 1) / 2
        fusion_components_for_gm = {
            "conviction_magnitude": conviction_magnitude,
            "pressure_magnitude": pressure_magnitude,
            "synergy_factor": synergy_factor,
            "deception_filter": deception_filter,
            "context_modulator": context_modulator
        }
        fused_magnitude = _robust_geometric_mean(
            fusion_components_for_gm,
            final_fusion_gm_weights,
            df_index
        )
        final_score = fused_magnitude * overall_direction
        final_score = np.sign(final_score) * (final_score.abs().pow(final_exponent))
        final_score = final_score.clip(-1, 1).fillna(np.nan) # fillna(np.nan)
        _temp_debug_values["最终融合"] = {
            "direction_weight_conviction": direction_weight_conviction,
            "direction_weight_pressure": direction_weight_pressure,
            "overall_direction_raw": overall_direction_raw,
            "overall_direction": overall_direction,
            "conviction_magnitude": conviction_magnitude,
            "pressure_magnitude": pressure_magnitude,
            "fused_magnitude": fused_magnitude,
            "final_score": final_score
        }
        return final_score







