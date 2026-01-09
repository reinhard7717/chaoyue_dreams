# strategies\trend_following\intelligence\process\calculate_process_covert_accumulation.py
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

class CalculateProcessCovertAccumulation:
    def __init__(self, strategy_instance, helper):
        self.strategy = strategy_instance
        self.helper = helper

    def _print_debug_info(self, debug_output: Dict, _temp_debug_values: Dict, method_name: str, probe_ts: pd.Timestamp):
        """
        统一打印隐蔽吸筹计算的调试信息。
        """
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        for key, value in _temp_debug_values["原始信号值"].items():
            if isinstance(value, pd.Series):
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            else:
                debug_output[f"        '{key}': {value}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场背景 ---"] = ""
        for key, series in _temp_debug_values["市场背景"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 隐蔽行动 ---"] = ""
        for key, series in _temp_debug_values["隐蔽行动"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 筹码优化 ---"] = ""
        for key, series in _temp_debug_values["筹码优化"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终合成 ---"] = ""
        for key, series in _temp_debug_values["最终合成"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 隐蔽吸筹诊断完成，最终分值: {_temp_debug_values['final_score'].loc[probe_ts]:.4f}"] = ""
        self.helper._print_debug_output(debug_output)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.5 · 深度情境与多维隐蔽行动版】计算“隐蔽吸筹”的专属信号。
        - 核心升级: 优化 `market_context_score` 中的价格弱势判断，直接奖励价格弱势。
        - 【强化】优化 `covert_action_score` 中的欺诈信号融合，更侧重于正向的诱多欺诈。
        - 【调整】调整 `covert_action_weights` 中拆单吸筹的权重，使用原始指标的MTF融合版本。
        """
        method_name = "_calculate_process_covert_accumulation"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算隐蔽吸筹..."] = ""
        
        covert_accum_params = get_param_value(self.helper.params.get('covert_accumulation_params'), {})
        fusion_weights = get_param_value(covert_accum_params.get('fusion_weights'), {"market_context": 0.3, "covert_action": 0.4, "chip_optimization": 0.3})
        market_context_weights = get_param_value(covert_accum_params.get('market_context_weights'), {"retail_panic": 0.2, "price_weakness": 0.2, "low_volatility": 0.2, "sentiment_pendulum_inverted": 0.15, "tension_inverted": 0.1, "market_sentiment_inverted": 0.1, "volatility_instability_inverted": 0.05})
        covert_action_weights = get_param_value(covert_accum_params.get('covert_action_weights'), {"suppressive_accum": 0.15, "main_force_flow": 0.15, "deception_lure_long": 0.15, "stealth_ops": 0.15, "hidden_accumulation_intensity": 0.1, "chip_historical_potential": 0.1, "mf_buy_ofi": 0.05, "mf_cost_advantage": 0.05, "mf_flow_slope": 0.05, "suppressive_accum_slope": 0.05})
        chip_optimization_weights = get_param_value(covert_accum_params.get('chip_optimization_weights'), {"chip_fatigue": 0.25, "loser_pain": 0.25, "holder_sentiment_inverted": 0.2, "turnover_purity_cost_opt": 0.15, "floating_chip_cleansing": 0.1, "total_loser_rate": 0.05})
        price_weakness_slope_window = get_param_value(covert_accum_params.get('price_weakness_slope_window'), 5)
        low_volatility_bbw_window = get_param_value(covert_accum_params.get('low_volatility_bbw_window'), 21)
        
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        
        required_df_columns = [
            'retail_panic_surrender_index_D', f'SLOPE_{price_weakness_slope_window}_close_D', f'BBW_{low_volatility_bbw_window}_2.0_D',
            'suppressive_accumulation_intensity_D', 'main_force_net_flow_calibrated_D', 'deception_index_D',
            'chip_fatigue_index_D', 'loser_pain_index_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'hidden_accumulation_intensity_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'main_force_buy_ofi_D', 'main_force_cost_advantage_D',
            'floating_chip_cleansing_efficiency_D', 'total_loser_rate_D', 'loser_concentration_90pct_D'
        ]
        for base_sig in ['main_force_net_flow_calibrated_D', 'suppressive_accumulation_intensity_D',
                         'deception_lure_long_intensity_D', 'hidden_accumulation_intensity_D',
                         'main_force_buy_ofi_D', 'main_force_cost_advantage_D',
                         'retail_panic_surrender_index_D', 'BBW_21_2.0_D', 'market_sentiment_score_D',
                         'VOLATILITY_INSTABILITY_INDEX_21d_D', 'chip_fatigue_index_D', 'loser_pain_index_D',
                         'floating_chip_cleansing_efficiency_D', 'total_loser_rate_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_df_columns.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_df_columns.append(f'ACCEL_{period_str}_{base_sig}')
        required_atomic_signals = [
            'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 'SCORE_STRUCT_AXIOM_TENSION',
            'SCORE_MICRO_STRATEGY_STEALTH_OPS',
            'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL',
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 'SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION'
        ]
        all_required_signals = required_df_columns + required_atomic_signals
        if not self.helper._validate_required_signals(df, all_required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self.helper._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        retail_panic_raw = self.helper._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        price_weakness_slope_raw = self.helper._get_safe_series(df, f'SLOPE_{price_weakness_slope_window}_close_D', 0.0, method_name=method_name)
        bbw_raw = self.helper._get_safe_series(df, f'BBW_{low_volatility_bbw_window}_2.0_D', 0.0, method_name=method_name)
        suppressive_accum_raw = self.helper._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name=method_name)
        main_force_flow_raw = self.helper._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        deception_raw = self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        chip_fatigue_raw = self.helper._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name=method_name)
        loser_pain_raw = self.helper._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name=method_name)
        deception_lure_long_raw = self.helper._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name)
        deception_lure_short_raw = self.helper._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name)
        hidden_accumulation_intensity_raw = self.helper._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name)
        sentiment_pendulum_score = self.helper._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        tension_score = self.helper._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0)
        market_sentiment_raw = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        volatility_instability_raw = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        stealth_ops_score = self.helper._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        chip_historical_potential_score = self.helper._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        mf_buy_ofi_raw = self.helper._get_safe_series(df, 'main_force_buy_ofi_D', 0.0, method_name=method_name)
        mf_cost_advantage_raw = self.helper._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name=method_name)
        holder_sentiment_score = self.helper._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        turnover_purity_cost_opt_score = self.helper._get_atomic_score(df, 'SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION', 0.0)
        floating_chip_cleansing_raw = self.helper._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name=method_name)
        total_loser_rate_raw = self.helper._get_safe_series(df, 'total_loser_rate_D', 0.0, method_name=method_name)
        
        _temp_debug_values["原始信号值"] = {
            "retail_panic_surrender_index_D": retail_panic_raw,
            f"SLOPE_{price_weakness_slope_window}_close_D": price_weakness_slope_raw,
            f"BBW_{low_volatility_bbw_window}_2.0_D": bbw_raw,
            "suppressive_accumulation_intensity_D": suppressive_accum_raw,
            "main_force_net_flow_calibrated_D": main_force_flow_raw,
            "deception_index_D": deception_raw,
            "chip_fatigue_index_D": chip_fatigue_raw,
            "loser_pain_index_D": loser_pain_raw,
            "deception_lure_long_intensity_D": deception_lure_long_raw,
            "deception_lure_short_intensity_D": deception_lure_short_raw,
            "hidden_accumulation_intensity_D": hidden_accumulation_intensity_raw,
            "SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM": sentiment_pendulum_score,
            "SCORE_STRUCT_AXIOM_TENSION": tension_score,
            "market_sentiment_score_D": market_sentiment_raw,
            "VOLATILITY_INSTABILITY_INDEX_21d_D": volatility_instability_raw,
            "SCORE_MICRO_STRATEGY_STEALTH_OPS": stealth_ops_score,
            "SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL": chip_historical_potential_score,
            "main_force_buy_ofi_D": mf_buy_ofi_raw,
            "main_force_cost_advantage_D": mf_cost_advantage_raw,
            "SCORE_CHIP_AXIOM_HOLDER_SENTIMENT": holder_sentiment_score,
            "SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION": turnover_purity_cost_opt_score,
            "floating_chip_cleansing_efficiency_D": floating_chip_cleansing_raw,
            "total_loser_rate_D": total_loser_rate_raw
        }
        # --- 3. 维度一：市场背景 (Market Context) ---
        retail_panic_score = self.helper._normalize_series(retail_panic_raw, df_index, bipolar=False)
        mtf_price_weakness_score = self.helper._get_mtf_slope_accel_score(df, f'close_D', mtf_slope_accel_weights, df_index, method_name, ascending=False, bipolar=False)
        low_volatility_score = self.helper._normalize_series(bbw_raw, df_index, ascending=False)
        sentiment_pendulum_inverted_score = (1 - sentiment_pendulum_score.clip(lower=0))
        tension_inverted_score = (1 - tension_score.clip(lower=0))
        market_sentiment_inverted_score = self.helper._normalize_series(market_sentiment_raw, df_index, ascending=False)
        volatility_instability_inverted_score = self.helper._normalize_series(volatility_instability_raw, df_index, ascending=False)
        
        market_context_scores_dict = {
            "retail_panic": retail_panic_score,
            "price_weakness": mtf_price_weakness_score,
            "low_volatility": low_volatility_score,
            "sentiment_pendulum_inverted": sentiment_pendulum_inverted_score,
            "tension_inverted": tension_inverted_score,
            "market_sentiment_inverted": market_sentiment_inverted_score,
            "volatility_instability_inverted": volatility_instability_inverted_score
        }
        market_context_score = _robust_geometric_mean(market_context_scores_dict, market_context_weights, df_index)
        _temp_debug_values["市场背景"] = {
            "retail_panic_score": retail_panic_score,
            "mtf_price_weakness_score": mtf_price_weakness_score,
            "low_volatility_score": low_volatility_score,
            "sentiment_pendulum_inverted_score": sentiment_pendulum_inverted_score,
            "tension_inverted_score": tension_inverted_score,
            "market_sentiment_inverted_score": market_sentiment_inverted_score,
            "volatility_instability_inverted_score": volatility_instability_inverted_score,
            "market_context_score": market_context_score
        }
        # --- 4. 维度二：隐蔽行动 (Covert Action) ---
        mtf_suppressive_accum_score = self.helper._get_mtf_slope_accel_score(df, 'suppressive_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_flow_score = self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_deception_lure_long_score = self.helper._get_mtf_slope_accel_score(df, 'deception_lure_long_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        stealth_ops_normalized = self.helper._normalize_series(stealth_ops_score, df_index, bipolar=False)
        mtf_hidden_accumulation_intensity = self.helper._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        chip_historical_potential_normalized = self.helper._normalize_series(chip_historical_potential_score.clip(lower=0), df_index, bipolar=False)
        mtf_mf_buy_ofi_normalized = self.helper._get_mtf_slope_accel_score(df, 'main_force_buy_ofi_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_cost_advantage_normalized = self.helper._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_flow_slope_normalized = self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True).clip(lower=0)
        mtf_suppressive_accum_slope_normalized = self.helper._get_mtf_slope_accel_score(df, 'suppressive_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True).clip(lower=0)
        
        covert_action_scores_dict = {
            "suppressive_accum": mtf_suppressive_accum_score,
            "main_force_flow": mtf_main_force_flow_score,
            "deception_lure_long": mtf_deception_lure_long_score,
            "stealth_ops": stealth_ops_normalized,
            "hidden_accumulation_intensity": mtf_hidden_accumulation_intensity,
            "chip_historical_potential": chip_historical_potential_normalized,
            "mf_buy_ofi": mtf_mf_buy_ofi_normalized,
            "mf_cost_advantage": mtf_mf_cost_advantage_normalized,
            "mf_flow_slope": mtf_mf_flow_slope_normalized,
            "suppressive_accum_slope": mtf_suppressive_accum_slope_normalized
        }
        covert_action_score = _robust_geometric_mean(covert_action_scores_dict, covert_action_weights, df_index)
        _temp_debug_values["隐蔽行动"] = {
            "mtf_suppressive_accum_score": mtf_suppressive_accum_score,
            "mtf_main_force_flow_score": mtf_main_force_flow_score,
            "mtf_deception_lure_long_score": mtf_deception_lure_long_score,
            "stealth_ops_normalized": stealth_ops_normalized,
            "mtf_hidden_accumulation_intensity": mtf_hidden_accumulation_intensity,
            "chip_historical_potential_normalized": chip_historical_potential_normalized,
            "mtf_mf_buy_ofi_normalized": mtf_mf_buy_ofi_normalized,
            "mtf_mf_cost_advantage_normalized": mtf_mf_cost_advantage_normalized,
            "mtf_mf_flow_slope_normalized": mtf_mf_flow_slope_normalized,
            "mtf_suppressive_accum_slope_normalized": mtf_suppressive_accum_slope_normalized,
            "covert_action_score": covert_action_score
        }
        # --- 5. 维度三：筹码优化 (Chip Optimization) ---
        mtf_chip_fatigue_score = self.helper._get_mtf_slope_accel_score(df, 'chip_fatigue_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_loser_pain_score = self.helper._get_mtf_slope_accel_score(df, 'loser_pain_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        holder_sentiment_inverted_score = (1 - holder_sentiment_score).clip(0, 1)
        turnover_purity_cost_opt_normalized = self.helper._normalize_series(turnover_purity_cost_opt_score.clip(lower=0), df_index, bipolar=False)
        mtf_floating_chip_cleansing_normalized = self.helper._get_mtf_slope_accel_score(df, 'floating_chip_cleansing_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_total_loser_rate_normalized = self.helper._get_mtf_slope_accel_score(df, 'total_loser_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        
        chip_optimization_scores_dict = {
            "chip_fatigue": mtf_chip_fatigue_score,
            "loser_pain": mtf_loser_pain_score,
            "holder_sentiment_inverted": holder_sentiment_inverted_score,
            "turnover_purity_cost_opt": turnover_purity_cost_opt_normalized,
            "floating_chip_cleansing": mtf_floating_chip_cleansing_normalized,
            "total_loser_rate": mtf_total_loser_rate_normalized
        }
        chip_optimization_score = _robust_geometric_mean(chip_optimization_scores_dict, chip_optimization_weights, df_index)
        _temp_debug_values["筹码优化"] = {
            "mtf_chip_fatigue_score": mtf_chip_fatigue_score,
            "mtf_loser_pain_score": mtf_loser_pain_score,
            "holder_sentiment_inverted_score": holder_sentiment_inverted_score,
            "turnover_purity_cost_opt_normalized": turnover_purity_cost_opt_normalized,
            "mtf_floating_chip_cleansing_normalized": mtf_floating_chip_cleansing_normalized,
            "mtf_total_loser_rate_normalized": mtf_total_loser_rate_normalized,
            "chip_optimization_score": chip_optimization_score
        }
        # --- 6. 最终合成：三维融合 ---
        final_fusion_scores_dict = {
            "market_context": market_context_score,
            "covert_action": covert_action_score,
            "chip_optimization": chip_optimization_score
        }
        covert_accumulation_score = _robust_geometric_mean(final_fusion_scores_dict, fusion_weights, df_index)
        _temp_debug_values["最终合成"] = {
            "covert_accumulation_score": covert_accumulation_score
        }
        final_score = covert_accumulation_score.clip(0, 1).astype(np.float32)
        _temp_debug_values["final_score"] = final_score # 存储最终分数用于调试输出

        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            self._print_debug_info(debug_output, _temp_debug_values, method_name, probe_ts)
            
        return final_score
