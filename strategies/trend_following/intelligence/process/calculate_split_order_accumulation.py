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

class CalculateSplitOrderAccumulation:
    """
    【V4.0.0 · 拆单吸筹强度 · 全息动态校准版】
    - 核心升级: 引入动态效率基准线，增强价格行为捕捉，精细化欺诈意图识别，MTF核心信号增强，
                情境自适应权重调整，非线性融合强化，趋势动量MTF化。
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
        # 默认的MTF斜率/加速度权重，可在config中覆盖
        self.default_mtf_slope_accel_weights = {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}}

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.0.0 · 拆单吸筹强度 · 全息动态校准版】计算“拆单吸筹强度”的专属信号。
        - 核心升级: 引入动态效率基准线，增强价格行为捕捉，精细化欺诈意图识别，MTF核心信号增强，
                    情境自适应权重调整，非线性融合强化，趋势动量MTF化。
        """
        method_name = "calculate_split_order_accumulation"
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {}
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算拆单吸筹强度..."] = ""
        df_index = df.index
        # 获取配置中的MTF权重，如果config中没有，则使用默认值
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', self.default_mtf_slope_accel_weights)
        # 1. 获取并归一化所有信号
        raw_signals, normalized_signals, mtf_signals, context_signals = self._get_and_normalize_signals(df, mtf_slope_accel_weights, method_name)
        _temp_debug_values["原始信号值"] = raw_signals
        _temp_debug_values["归一化处理"] = normalized_signals
        _temp_debug_values["MTF信号"] = mtf_signals
        _temp_debug_values["情境信号"] = context_signals
        # 2. 动态效率基准线
        dynamic_efficiency_baseline, baseline_debug_values = self._calculate_dynamic_efficiency_baseline(context_signals, df_index, config)
        _temp_debug_values["动态效率基准线"] = baseline_debug_values
        _temp_debug_values["动态效率基准线"]["dynamic_efficiency_baseline"] = dynamic_efficiency_baseline
        # 3. 计算初步分数 (dynamic_preliminary_score)
        dynamic_preliminary_score, preliminary_debug_values = self._calculate_preliminary_score(normalized_signals, mtf_signals, context_signals, df_index, config)
        _temp_debug_values["初步计算"] = preliminary_debug_values
        _temp_debug_values["初步计算"]["dynamic_preliminary_score"] = dynamic_preliminary_score
        # 4. 计算全息验证分数 (holographic_validation_score)
        holographic_validation_score, holographic_debug_values = self._calculate_holographic_validation(df, normalized_signals, mtf_signals, context_signals, df_index, config)
        _temp_debug_values["全息验证"] = holographic_debug_values
        _temp_debug_values["全息验证"]["holographic_validation_score"] = holographic_validation_score
        # 5. 应用质效校准并计算最终分数
        final_score, final_score_debug_values = self._apply_quality_efficiency_calibration(dynamic_preliminary_score, holographic_validation_score, dynamic_efficiency_baseline)
        _temp_debug_values["最终分数"] = final_score_debug_values
        _temp_debug_values["最终分数"]["final_score"] = final_score
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            self._print_debug_info(method_name, probe_ts, debug_output, _temp_debug_values, final_score)
        return final_score.astype(np.float32)

    def _get_and_normalize_signals(self, df: pd.DataFrame, mtf_slope_accel_weights: Dict, method_name: str) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        获取并归一化所有拆单吸筹强度计算所需的原始信号。
        返回: (raw_signals, normalized_signals, mtf_signals, context_signals)
        """
        df_index = df.index
        required_signals = [
            'hidden_accumulation_intensity_D', 'SLOPE_5_close_D', 'deception_index_D',
            'upward_impulse_purity_D', 'PROCESS_META_POWER_TRANSFER',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_DYN_AXIOM_STABILITY',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'ADX_14_D', 'is_consolidating_D', 'dynamic_consolidation_duration_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'trend_vitality_index_D', 'close_D' # close_D用于MTF price trend
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        base_signals_for_mtf = [
            'hidden_accumulation_intensity_D', 'close_D', 'deception_index_D',
            'PROCESS_META_POWER_TRANSFER', 'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_DYN_AXIOM_STABILITY',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'trend_vitality_index_D'
        ]
        for base_sig in base_signals_for_mtf:
            # _get_mtf_slope_accel_score 会从df中查找SLOPE_X_SIGNAL_NAME，所以这里只需要确保原始信号名是正确的
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return {}, {}, {}, {}
        # 原始数据获取
        raw_intensity = self.helper._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name)
        price_trend_raw = self.helper._get_safe_series(df, 'SLOPE_5_close_D', 0.0, method_name=method_name)
        deception_index_raw = self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        upward_purity_raw = self.helper._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        flow_outcome_raw = self.helper._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        structure_outcome_raw = self.helper._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        potential_outcome_raw = self.helper._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        market_sentiment_raw = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        volatility_instability_raw = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        adx_raw = self.helper._get_safe_series(df, 'ADX_14_D', 0.0, method_name=method_name)
        is_consolidating_raw = self.helper._get_safe_series(df, 'is_consolidating_D', 0.0, method_name=method_name)
        dynamic_consolidation_duration_raw = self.helper._get_safe_series(df, 'dynamic_consolidation_duration_D', 0.0, method_name=method_name)
        deception_lure_long_raw = self.helper._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name)
        deception_lure_short_raw = self.helper._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name)
        trend_vitality_raw = self.helper._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        raw_signals = {
            "hidden_accumulation_intensity_D": raw_intensity,
            "SLOPE_5_close_D": price_trend_raw,
            "deception_index_D": deception_index_raw,
            "upward_impulse_purity_D": upward_purity_raw,
            "PROCESS_META_POWER_TRANSFER": flow_outcome_raw,
            "SCORE_CHIP_STRATEGIC_POSTURE": structure_outcome_raw,
            "SCORE_DYN_AXIOM_STABILITY": potential_outcome_raw,
            "market_sentiment_score_D": market_sentiment_raw,
            "VOLATILITY_INSTABILITY_INDEX_21d_D": volatility_instability_raw,
            "ADX_14_D": adx_raw,
            "is_consolidating_D": is_consolidating_raw,
            "dynamic_consolidation_duration_D": dynamic_consolidation_duration_raw,
            "deception_lure_long_intensity_D": deception_lure_long_raw,
            "deception_lure_short_intensity_D": deception_lure_short_raw,
            "trend_vitality_index_D": trend_vitality_raw
        }
        # 归一化处理
        normalized_score = (raw_intensity / 100).clip(0, 1)
        price_trend_norm = self.helper._normalize_series(price_trend_raw, df_index, bipolar=True)
        upward_purity = self.helper._normalize_series(upward_purity_raw, df_index, bipolar=False)
        deception_norm = self.helper._normalize_series(deception_index_raw, df_index, bipolar=True)
        flow_outcome = flow_outcome_raw # 已经是分数
        structure_outcome = structure_outcome_raw # 已经是分数
        potential_outcome = potential_outcome_raw # 已经是分数
        market_sentiment_norm = self.helper._normalize_series(market_sentiment_raw, df_index, bipolar=True)
        volatility_instability_norm = self.helper._normalize_series(volatility_instability_raw, df_index, bipolar=False)
        adx_norm = self.helper._normalize_series(adx_raw, df_index, bipolar=False)
        is_consolidating_norm = self.helper._normalize_series(is_consolidating_raw, df_index, bipolar=False)
        dynamic_consolidation_duration_norm = self.helper._normalize_series(dynamic_consolidation_duration_raw, df_index, bipolar=False)
        deception_lure_long_norm = self.helper._normalize_series(deception_lure_long_raw, df_index, bipolar=False)
        deception_lure_short_norm = self.helper._normalize_series(deception_lure_short_raw, df_index, bipolar=False)
        trend_vitality_norm = self.helper._normalize_series(trend_vitality_raw, df_index, bipolar=False)
        normalized_signals = {
            "normalized_score": normalized_score,
            "price_trend_norm": price_trend_norm,
            "upward_purity": upward_purity,
            "deception_norm": deception_norm,
            "flow_outcome": flow_outcome,
            "structure_outcome": structure_outcome,
            "potential_outcome": potential_outcome
        }
        # MTF信号
        mtf_hidden_accumulation_intensity = self.helper._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_price_trend = self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_deception_index = self.helper._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_flow_outcome_trend = self.helper._get_mtf_slope_accel_score(df, 'PROCESS_META_POWER_TRANSFER', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_structure_outcome_trend = self.helper._get_mtf_slope_accel_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_potential_outcome_trend = self.helper._get_mtf_slope_accel_score(df, 'SCORE_DYN_AXIOM_STABILITY', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_signals = {
            "mtf_hidden_accumulation_intensity": mtf_hidden_accumulation_intensity,
            "mtf_price_trend": mtf_price_trend,
            "mtf_deception_index": mtf_deception_index,
            "mtf_flow_outcome_trend": mtf_flow_outcome_trend,
            "mtf_structure_outcome_trend": mtf_structure_outcome_trend,
            "mtf_potential_outcome_trend": mtf_potential_outcome_trend
        }
        # 情境信号
        context_signals = {
            "market_sentiment_norm": market_sentiment_norm,
            "volatility_instability_norm": volatility_instability_norm,
            "adx_norm": adx_norm,
            "is_consolidating_norm": is_consolidating_norm,
            "dynamic_consolidation_duration_norm": dynamic_consolidation_duration_norm,
            "deception_lure_long_norm": deception_lure_long_norm,
            "deception_lure_short_norm": deception_lure_short_norm,
            "trend_vitality_norm": trend_vitality_norm
        }
        return raw_signals, normalized_signals, mtf_signals, context_signals

    def _calculate_dynamic_efficiency_baseline(self, context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        计算动态效率基准线。
        """
        baseline_debug_values = {}
        params = config.get('dynamic_efficiency_baseline_params', {})
        base_baseline = get_param_value(params.get('base_baseline'), 0.15)
        sentiment_impact = get_param_value(params.get('sentiment_impact'), 0.1)
        volatility_impact = get_param_value(params.get('volatility_impact'), 0.05)
        consolidation_impact = get_param_value(params.get('consolidation_impact'), 0.05)
        market_sentiment_norm = context_signals["market_sentiment_norm"]
        volatility_instability_norm = context_signals["volatility_instability_norm"]
        is_consolidating_norm = context_signals["is_consolidating_norm"]
        # 情绪越积极，基准线越高 (要求更高)
        # 波动率越低 (稳定性越高)，基准线越高 (要求更高)
        # 处于盘整期，基准线可以适当降低 (吸筹难度大)
        dynamic_baseline = base_baseline + \
                           (market_sentiment_norm.clip(lower=0) * sentiment_impact) - \
                           (volatility_instability_norm * volatility_impact) - \
                           (is_consolidating_norm * consolidation_impact)
        dynamic_baseline = dynamic_baseline.clip(0.05, 0.3) # 限制基准线范围
        baseline_debug_values["base_baseline"] = base_baseline
        baseline_debug_values["sentiment_impact_term"] = market_sentiment_norm.clip(lower=0) * sentiment_impact
        baseline_debug_values["volatility_impact_term"] = volatility_instability_norm * volatility_impact
        baseline_debug_values["consolidation_impact_term"] = is_consolidating_norm * consolidation_impact
        return dynamic_baseline, baseline_debug_values

    def _calculate_preliminary_score(self, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        计算初步的拆单吸筹强度分数。
        """
        preliminary_debug_values = {}
        params = config.get('preliminary_score_params', {})
        price_suppression_weights = get_param_value(params.get('price_suppression_weights'), {"price_trend_negative": 0.3, "is_consolidating": 0.3, "consolidation_duration": 0.2, "upward_purity_inverted": 0.2})
        strategic_context_weights = get_param_value(params.get('strategic_context_weights'), {"stability": 0.4, "deception_lure_long_inverted": 0.3, "deception_lure_short": 0.3})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"mtf_intensity": 0.4, "price_suppression": 0.3, "strategic_context": 0.3})
        momentum_weight = get_param_value(params.get('momentum_weight'), 0.3)
        normalized_score = normalized_signals["normalized_score"]
        price_trend_norm = normalized_signals["price_trend_norm"]
        upward_purity = normalized_signals["upward_purity"]
        potential_outcome = normalized_signals["potential_outcome"]
        deception_lure_long_norm = context_signals["deception_lure_long_norm"]
        deception_lure_short_norm = context_signals["deception_lure_short_norm"]
        mtf_hidden_accumulation_intensity = mtf_signals["mtf_hidden_accumulation_intensity"]
        is_consolidating_norm = context_signals["is_consolidating_norm"]
        dynamic_consolidation_duration_norm = context_signals["dynamic_consolidation_duration_norm"]
        # 增强价格行为捕捉 (price_suppression_factor)
        price_trend_negative = price_trend_norm.clip(upper=0).abs() # 价格下跌的强度
        upward_purity_inverted = 1 - upward_purity # 上涨纯度越低越好
        price_suppression_components = {
            "price_trend_negative": price_trend_negative,
            "is_consolidating": is_consolidating_norm,
            "consolidation_duration": dynamic_consolidation_duration_norm,
            "upward_purity_inverted": upward_purity_inverted
        }
        price_suppression_factor = _robust_geometric_mean(price_suppression_components, price_suppression_weights, df_index).clip(0, 1)
        preliminary_debug_values["price_suppression_factor_components"] = price_suppression_components
        preliminary_debug_values["price_suppression_factor"] = price_suppression_factor
        # 精细化欺诈意图识别 (strategic_context_factor)
        deception_lure_long_inverted = 1 - deception_lure_long_norm # 诱多越少越好
        deception_lure_short = deception_lure_short_norm # 诱空越多越好 (主力震仓)
        strategic_context_components = {
            "stability": potential_outcome,
            "deception_lure_long_inverted": deception_lure_long_inverted,
            "deception_lure_short": deception_lure_short
        }
        strategic_context_factor = _robust_geometric_mean(strategic_context_components, strategic_context_weights, df_index).clip(0, 1)
        preliminary_debug_values["strategic_context_factor_components"] = strategic_context_components
        preliminary_debug_values["strategic_context_factor"] = strategic_context_factor
        # MTF 核心信号增强
        preliminary_components = {
            "mtf_intensity": mtf_hidden_accumulation_intensity,
            "price_suppression": price_suppression_factor,
            "strategic_context": strategic_context_factor
        }
        preliminary_score = _robust_geometric_mean(preliminary_components, fusion_weights, df_index).fillna(0.0)
        preliminary_debug_values["preliminary_score_components"] = preliminary_components
        preliminary_debug_values["preliminary_score"] = preliminary_score
        tactical_momentum_score = self.helper._normalize_series(preliminary_score.diff(1).fillna(0), df_index, bipolar=False)
        preliminary_debug_values["tactical_momentum_score"] = tactical_momentum_score
        dynamic_preliminary_score = (preliminary_score * (1 - momentum_weight) + tactical_momentum_score * momentum_weight).clip(0, 1)
        return dynamic_preliminary_score, preliminary_debug_values

    def _calculate_holographic_validation(self, df: pd.DataFrame, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        计算全息验证分数。
        """
        holographic_debug_values = {}
        params = config.get('holographic_validation_params', {})
        state_fusion_weights = get_param_value(params.get('state_fusion_weights'), {"flow": 0.4, "structure": 0.3, "potential": 0.3})
        trend_fusion_weights = get_param_value(params.get('trend_fusion_weights'), {"flow_trend": 0.4, "structure_trend": 0.3, "potential_trend": 0.3})
        overall_fusion_weights = get_param_value(params.get('overall_fusion_weights'), {"state": 0.6, "trend": 0.4})
        flow_outcome = normalized_signals["flow_outcome"]
        structure_outcome = normalized_signals["structure_outcome"]
        potential_outcome = normalized_signals["potential_outcome"] # stability_score
        # 情境自适应权重调整
        market_sentiment_norm = context_signals["market_sentiment_norm"]
        volatility_instability_norm = context_signals["volatility_instability_norm"]
        adx_norm = context_signals["adx_norm"] # 趋势强度
        # 示例：在市场情绪积极且波动率低时，更看重结构和潜力；在趋势强劲时，更看重趋势
        sentiment_factor = (market_sentiment_norm + 1) / 2 # [0, 1]
        volatility_factor = 1 - volatility_instability_norm # [0, 1] (稳定性)
        trend_strength_factor = adx_norm # [0, 1]
        # 动态调整 state_fusion_weights
        dynamic_state_fusion_weights = state_fusion_weights.copy()
        dynamic_state_fusion_weights["flow"] = dynamic_state_fusion_weights["flow"] * (1 - sentiment_factor * 0.2) * (1 + volatility_factor * 0.1) * (1 - trend_strength_factor * 0.1)
        dynamic_state_fusion_weights["structure"] = dynamic_state_fusion_weights["structure"] * (1 + sentiment_factor * 0.2) * (1 + volatility_factor * 0.1) * (1 + trend_strength_factor * 0.1)
        dynamic_state_fusion_weights["potential"] = dynamic_state_fusion_weights["potential"] * (1 + sentiment_factor * 0.1) * (1 + volatility_factor * 0.05) * (1 + trend_strength_factor * 0.05)
        # 归一化动态权重
        total_dynamic_weight = sum(dynamic_state_fusion_weights.values())
        for k in dynamic_state_fusion_weights:
            dynamic_state_fusion_weights[k] /= total_dynamic_weight
        holographic_debug_values["dynamic_state_fusion_weights"] = dynamic_state_fusion_weights
        # 非线性融合 holographic_state_score
        holographic_state_components = {
            "flow_outcome": flow_outcome,
            "structure_outcome": structure_outcome,
            "potential_outcome": potential_outcome
        }
        holographic_state_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in holographic_state_components.items()}, # 确保输入为正
            dynamic_state_fusion_weights,
            df_index
        )
        holographic_debug_values["holographic_state_score_components"] = holographic_state_components
        holographic_debug_values["holographic_state_score"] = holographic_state_score
        # 趋势动量 MTF 化
        mtf_flow_outcome_trend = mtf_signals["mtf_flow_outcome_trend"]
        mtf_structure_outcome_trend = mtf_signals["mtf_structure_outcome_trend"]
        mtf_potential_outcome_trend = mtf_signals["mtf_potential_outcome_trend"]
        holographic_trend_components = {
            "flow_trend": mtf_flow_outcome_trend,
            "structure_trend": mtf_structure_outcome_trend,
            "potential_trend": mtf_potential_outcome_trend
        }
        holographic_trend_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in holographic_trend_components.items()}, # 确保输入为正
            trend_fusion_weights,
            df_index
        )
        holographic_debug_values["holographic_trend_score_components"] = holographic_trend_components
        holographic_debug_values["holographic_trend_score"] = holographic_trend_score
        # 整体融合
        overall_holographic_components = {
            "state": holographic_state_score,
            "trend": holographic_trend_score
        }
        holographic_validation_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in overall_holographic_components.items()}, # 确保输入为正
            overall_fusion_weights,
            df_index
        )
        # 将几何平均结果映射回 [-1, 1] 区间，通过乘以原始状态和趋势的平均符号
        # 这里需要更精确的符号判断，例如，如果flow_outcome, structure_outcome, potential_outcome 多数为负，则整体应为负
        # 简单取平均符号可能不够鲁棒，可以考虑加权平均符号或多数投票
        # 为了保持非线性融合的纯粹性，这里可以先不强制映射回[-1,1]，而是让其保持[0,1]的强度，
        # 并在最终校准时，根据原始信号的整体方向来赋予符号。
        # 但为了保持与原始逻辑的[-1,1]输出一致，我们仍然需要一个方向性。
        # 我们可以使用 flow_outcome, structure_outcome, potential_outcome 的加权平均作为方向性
        weighted_avg_direction = (flow_outcome * dynamic_state_fusion_weights["flow"] + 
                                  structure_outcome * dynamic_state_fusion_weights["structure"] + 
                                  potential_outcome * dynamic_state_fusion_weights["potential"])
        holographic_validation_score = holographic_validation_score * weighted_avg_direction.apply(np.sign).replace(0, 1) # 确保符号一致，0时视为正向
        return holographic_validation_score.clip(-1, 1), holographic_debug_values

    def _apply_quality_efficiency_calibration(self, dynamic_preliminary_score: pd.Series, holographic_validation_score: pd.Series, dynamic_efficiency_baseline: pd.Series) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        应用质效校准并计算最终分数。
        """
        final_score_debug_values = {}
        calibrated_holographic_score = holographic_validation_score - dynamic_efficiency_baseline
        final_score_debug_values["calibrated_holographic_score"] = calibrated_holographic_score
        # 确保 quality_efficiency_modulator 始终为正，且在合理范围内
        # 如果 calibrated_holographic_score 很高，modulator 应该小 (放大 preliminary_score)
        # 如果 calibrated_holographic_score 很低，modulator 应该大 (惩罚 preliminary_score)
        quality_efficiency_modulator = (1 - calibrated_holographic_score).clip(0.1, 2.0)
        final_score_debug_values["quality_efficiency_modulator"] = quality_efficiency_modulator
        final_score = dynamic_preliminary_score.pow(quality_efficiency_modulator).clip(0, 1).fillna(0.0)
        return final_score, final_score_debug_values

    def _print_debug_info(self, method_name: str, probe_ts: pd.Timestamp, debug_output: Dict, _temp_debug_values: Dict, final_score: pd.Series):
        """
        统一打印调试信息。
        """
        for section, values in _temp_debug_values.items():
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section} ---"] = ""
            for key, series_or_value in values.items():
                if isinstance(series_or_value, pd.Series):
                    val = series_or_value.loc[probe_ts] if probe_ts in series_or_value.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                elif isinstance(series_or_value, dict): # 处理嵌套字典，例如 preliminary_score_components
                    debug_output[f"        {key}:"] = ""
                    for sub_key, sub_series in series_or_value.items():
                        if isinstance(sub_series, pd.Series):
                            sub_val = sub_series.loc[probe_ts] if probe_ts in sub_series.index else np.nan
                            debug_output[f"          {sub_key}: {sub_val:.4f}"] = ""
                        else:
                            debug_output[f"          {sub_key}: {sub_series}"] = ""
                else:
                    debug_output[f"        {key}: {series_or_value}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 拆单吸筹强度诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        self.helper._print_debug_output(debug_output)
