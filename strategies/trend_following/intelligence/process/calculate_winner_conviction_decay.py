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
    def __init__(self, strategy_instance, helper):
        self.strategy = strategy_instance
        self.helper = helper
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

    def _get_decay_params_and_signals(self, config: Dict, method_name: str) -> Tuple[Dict, List[str]]:
        """
        获取赢家信念衰减计算所需的所有参数和信号列表。
        """
        decay_params = get_param_value(self.helper.params.get('winner_conviction_decay_params'), {})
        mtf_slope_accel_weights = get_param_value(decay_params.get('mtf_slope_accel_weights'), {"slope_periods": {"5": 0.4, "13": 0.3}, "accel_periods": {"5": 0.6}})
        belief_decay_components_weights = get_param_value(decay_params.get('belief_decay_components_weights'), {
            "winner_stability_mtf": 0.4, "winner_profit_margin_avg_inverted": 0.2,
            "total_winner_rate_inverted": 0.2, "chip_fatigue": 0.2
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
        # 修正此处的默认值，使其与 process.json 中的最新配置同步
        contextual_modulator_weights = get_param_value(decay_params.get('contextual_modulator_weights'), {
            "price_overextension_composite": 0.3,
            "retail_fomo": 0.2,
            "market_tension": 0.2,
            "sentiment_pendulum_negative": 0.3,
            "market_sentiment": 0.3,        # 新增
            "volatility_stability": 0.3,    # 新增
            "trend_vitality": 0.4           # 新增
        })
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
        for base_sig in [belief_signal_name, pressure_signal_name, 'winner_profit_margin_avg_D', 'total_winner_rate_D', 'chip_fatigue_index_D',
                         'active_selling_pressure_D', 'rally_sell_distribution_intensity_D', 'main_force_t0_sell_efficiency_D',
                         'main_force_on_peak_sell_flow_D', 'deception_lure_long_intensity_D', 'wash_trade_intensity_D',
                         'pressure_rejection_strength_D', 'rally_buy_support_weakness_D', 'buy_quote_exhaustion_rate_D',
                         'bid_side_liquidity_D', 'main_force_slippage_index_D', 'structural_tension_index_D',
                         'volatility_expansion_ratio_D', 'chip_health_score_D', 'market_impact_cost_D', 'trend_vitality_index_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_df_columns.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_df_columns.append(f'ACCEL_{period_str}_{base_sig}')
        required_atomic_signals = [
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER',
            'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'
        ]
        all_required_signals = required_df_columns + required_atomic_signals
        params_dict = {
            'decay_params': decay_params,
            'mtf_slope_accel_weights': mtf_slope_accel_weights,
            'belief_decay_components_weights': belief_decay_components_weights,
            'profit_pressure_components_weights': profit_pressure_components_weights,
            'distribution_confirmation_components_weights': distribution_confirmation_components_weights,
            'buying_resistance_collapse_weights': buying_resistance_collapse_weights,
            'contextual_modulator_weights': contextual_modulator_weights,
            'dynamic_fusion_exponent_params': dynamic_fusion_exponent_params,
            'price_overextension_composite_weights': price_overextension_composite_weights,
            'relative_position_weights': relative_position_weights,
            'final_fusion_gm_weights': final_fusion_gm_weights,
            'final_exponent': final_exponent,
            'belief_signal_name': belief_signal_name,
            'pressure_signal_name': pressure_signal_name
        }
        return params_dict, all_required_signals

    def _get_raw_signals(self, df: pd.DataFrame, df_index: pd.Index, params_dict: Dict, method_name: str) -> Dict[str, pd.Series]:
        """
        获取所有原始数据和原子信号。
        """
        belief_signal_name = params_dict['belief_signal_name']
        pressure_signal_name = params_dict['pressure_signal_name']
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
            "distribution_intent_score": self.helper._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0),
            "chip_distribution_whisper_score": self.helper._get_atomic_score(df, 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 0.0),
            "market_tension_score": self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 0.0),
            "sentiment_pendulum_score": self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0),
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
        return raw_signals

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, raw_signals: Dict[str, pd.Series], params_dict: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        计算赢家信念强度。
        """
        belief_signal_name = params_dict['belief_signal_name']
        mtf_slope_accel_weights = params_dict['mtf_slope_accel_weights']
        relative_position_weights = params_dict['relative_position_weights']
        belief_signal_raw = raw_signals["belief_signal_raw"]
        mtf_winner_stability = self.helper._get_mtf_slope_accel_score(df, belief_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        winner_stability_percentile = belief_signal_raw.rank(pct=True).fillna(0.5)
        conviction_strength_score = (mtf_winner_stability * relative_position_weights.get("winner_stability_high", 0.6) + 
                                     (winner_stability_percentile * 2 - 1) * (1 - relative_position_weights.get("winner_stability_high", 0.6))).clip(-1, 1)
        _temp_debug_values["信念强度"] = {
            "mtf_winner_stability": mtf_winner_stability,
            "winner_stability_percentile": winner_stability_percentile,
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
        pressure_signal_raw = raw_signals["pressure_signal_raw"]
        mtf_profit_taking_flow = self.helper._get_mtf_slope_accel_score(df, pressure_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        profit_taking_flow_percentile = (1 - pressure_signal_raw.rank(pct=True)).fillna(0.5)
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
        mtf_deception_index = self.helper._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_wash_trade_intensity = self.helper._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        deception_penalty = (mtf_deception_index * 0.6 + mtf_wash_trade_intensity * 0.4).clip(0, 1)
        deception_filter = (1 - deception_penalty).clip(0, 1)
        _temp_debug_values["诡道过滤"] = {
            "mtf_deception_index": mtf_deception_index,
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
        market_sentiment_raw = raw_signals["market_sentiment_raw"]
        volatility_instability_raw = raw_signals["volatility_instability_raw"]
        trend_vitality_raw = raw_signals["trend_vitality_raw"]
        norm_market_sentiment = self.helper._normalize_series(market_sentiment_raw, df_index, bipolar=True)
        volatility_stability_raw = 1 - normalize_score(
            volatility_instability_raw, 
            df_index, 
            21, 
            ascending=True,
            debug_info=(is_debug_enabled_for_method, probe_ts, f"{method_name}_volatility_stability_norm") # 传递debug_info
        )
        norm_volatility_stability = self.helper._normalize_series(volatility_stability_raw, df_index, bipolar=False, ascending=True)
        norm_trend_vitality = self.helper._normalize_series(trend_vitality_raw, df_index, bipolar=False)
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_stability": norm_volatility_stability,
            "trend_vitality": norm_trend_vitality
        }
        context_modulator_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in context_modulator_components.items()},
            contextual_modulator_weights,
            df_index,
            debug_info=(is_debug_enabled_for_method, probe_ts, f"{method_name}_context_modulator_score_gm") # 传递debug_info
        )
        context_modulator = 0.5 + context_modulator_score
        _temp_debug_values["情境调制"] = {
            "norm_market_sentiment": norm_market_sentiment,
            "volatility_stability_raw": volatility_stability_raw,
            "norm_volatility_stability": norm_volatility_stability,
            "norm_trend_vitality": norm_trend_vitality,
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
        overall_direction = overall_direction.replace(0, 1)
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
        final_score = final_score.clip(-1, 1).fillna(0.0)
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







