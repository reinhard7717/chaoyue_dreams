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
        # 默认的MTF斜率/加速度权重，可在config中覆盖
        self.default_mtf_slope_accel_weights = {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}}

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.0.3 · 拆单吸筹强度 · 探针激活版】计算“拆单吸筹强度”的专属信号。
        - 核心修正: 确保调试信息 (is_debug_enabled_for_method, probe_ts) 被正确传递到
                    _calculate_holographic_validation 方法，从而激活 _robust_geometric_mean 内部的探针。
        - 核心升级: 引入动态效率基准线，增强价格行为捕捉，精细化欺诈意图识别，MTF核心信号增强，
                    情境自适应权重调整，非线性融合强化，趋势动量diff()化。
        - 探针强化: 增加关键中间计算节点的详细探针，特别是针对_robust_geometric_mean的输入和输出，
                    以及最终pow()操作的精确值，以暴露潜在的计算偏差或bug。
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
        # 传递 is_debug_enabled_for_method 和 probe_ts
        holographic_validation_score, holographic_debug_values = self._calculate_holographic_validation(df, raw_signals, normalized_signals, context_signals, df_index, config, is_debug_enabled_for_method, probe_ts)
        _temp_debug_values["全息验证"] = holographic_debug_values
        _temp_debug_values["全息验证"]["holographic_validation_score"] = holographic_validation_score

        # 5. 应用质效校准并计算最终分数
        final_score, final_score_debug_values = self._apply_quality_efficiency_calibration(dynamic_preliminary_score, holographic_validation_score, dynamic_efficiency_baseline, probe_ts)
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
        
        # 明确列出所有需要从 df 中获取的原始数据列
        raw_df_columns = [
            'hidden_accumulation_intensity_D', 'SLOPE_5_close_D', 'deception_index_D',
            'upward_impulse_purity_D', 'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'ADX_14_D', 'is_consolidating_D', 'dynamic_consolidation_duration_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'trend_vitality_index_D', 'close_D'
        ]
        
        # 明确列出所有需要从 atomic_states 中获取的原子信号
        atomic_state_signals = [
            'PROCESS_META_POWER_TRANSFER', 'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_DYN_AXIOM_STABILITY'
        ]

        # 构建 required_signals 列表，用于验证
        required_signals = list(raw_df_columns) + list(atomic_state_signals)

        # 动态添加数据层原始信号的MTF斜率和加速度到required_signals
        # 这些MTF信号是预期在df中作为预计算列存在的
        base_signals_for_mtf_from_df = [
            'hidden_accumulation_intensity_D', 'close_D', 'deception_index_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'trend_vitality_index_D'
        ]
        for base_sig in base_signals_for_mtf_from_df:
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

        # 归一化处理 (针对原始信号或其简单衍生)
        normalized_score = (raw_intensity / 100).clip(0, 1)
        price_trend_norm = self.helper._normalize_series(price_trend_raw, df_index, bipolar=True)
        upward_purity = self.helper._normalize_series(upward_purity_raw, df_index, bipolar=False)
        deception_norm = self.helper._normalize_series(deception_index_raw, df_index, bipolar=True)
        # 原子信号直接使用，它们已经是归一化分数
        flow_outcome = flow_outcome_raw
        structure_outcome = structure_outcome_raw
        potential_outcome = potential_outcome_raw
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

        # MTF信号 (仅针对数据层原始信号)
        mtf_hidden_accumulation_intensity = self.helper._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_price_trend = self.helper._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_deception_index = self.helper._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)

        mtf_signals = {
            "mtf_hidden_accumulation_intensity": mtf_hidden_accumulation_intensity,
            "mtf_price_trend": mtf_price_trend,
            "mtf_deception_index": mtf_deception_index
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
        preliminary_debug_values["price_trend_negative"] = price_trend_negative
        preliminary_debug_values["upward_purity_inverted"] = upward_purity_inverted
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
        preliminary_debug_values["deception_lure_long_inverted"] = deception_lure_long_inverted
        preliminary_debug_values["deception_lure_short"] = deception_lure_short
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

    def _calculate_holographic_validation(self, df: pd.DataFrame, raw_signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], context_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        【V4.0.4 · 拆单吸筹强度 · 键匹配修复版】计算全息验证分数。
        - 核心修复: 修正了传递给 _robust_geometric_mean 的 scores_dict 和 weights_dict 之间的键不匹配问题。
        - 核心升级: 引入动态效率基准线，增强价格行为捕捉，精细化欺诈意图识别，MTF核心信号增强，
                    情境自适应权重调整，非线性融合强化，趋势动量diff()化。
        - 探针强化: 增加关键中间计算节点的详细探针，特别是针对_robust_geometric_mean的输入和输出，
                    以及最终pow()操作的精确值，以暴露潜在的计算偏差或bug。
        """
        holographic_debug_values = {}
        params = config.get('holographic_validation_params', {})
        state_fusion_weights = get_param_value(params.get('state_fusion_weights'), {"flow": 0.4, "structure": 0.3, "potential": 0.3})
        trend_fusion_weights = get_param_value(params.get('trend_fusion_weights'), {"flow_trend": 0.4, "structure_trend": 0.3, "potential_trend": 0.3})
        overall_fusion_weights = get_param_value(params.get('overall_fusion_weights'), {"state": 0.6, "trend": 0.4})
        
        # 直接使用原始原子信号Series
        flow_outcome_raw = raw_signals["PROCESS_META_POWER_TRANSFER"]
        structure_outcome_raw = raw_signals["SCORE_CHIP_STRATEGIC_POSTURE"]
        potential_outcome_raw = raw_signals["SCORE_DYN_AXIOM_STABILITY"]

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
        # 注意：这里 dynamic_state_fusion_weights 的值会变成 Series
        dynamic_state_fusion_weights["flow"] = dynamic_state_fusion_weights["flow"] * (1 - sentiment_factor * 0.2) * (1 + volatility_factor * 0.1) * (1 - trend_strength_factor * 0.1)
        dynamic_state_fusion_weights["structure"] = dynamic_state_fusion_weights["structure"] * (1 + sentiment_factor * 0.2) * (1 + volatility_factor * 0.1) * (1 + trend_strength_factor * 0.1)
        dynamic_state_fusion_weights["potential"] = dynamic_state_fusion_weights["potential"] * (1 + sentiment_factor * 0.1) * (1 + volatility_factor * 0.05) * (1 + trend_strength_factor * 0.05)
        
        # 归一化动态权重
        # total_dynamic_weight 此时是一个 Series
        total_dynamic_weight = sum(dynamic_state_fusion_weights.values())
        # 避免除以零
        total_dynamic_weight = total_dynamic_weight.replace(0, 1e-9)
        for k in dynamic_state_fusion_weights:
            dynamic_state_fusion_weights[k] /= total_dynamic_weight
        holographic_debug_values["dynamic_state_fusion_weights"] = dynamic_state_fusion_weights

        # 非线性融合 holographic_state_score
        # 核心修复：修改键名以匹配 dynamic_state_fusion_weights 的键
        holographic_state_components = {
            "flow": flow_outcome_raw,         # 键从 "flow_outcome" 改为 "flow"
            "structure": structure_outcome_raw, # 键从 "structure_outcome" 改为 "structure"
            "potential": potential_outcome_raw  # 键从 "potential_outcome" 改为 "potential"
        }
        
        # --- 探针强化: 打印传递给 _robust_geometric_mean 的实际值 ---
        holographic_state_components_pre_gm = {k: (v + 1) / 2 if v.min() < 0 else v for k, v in holographic_state_components.items()}
        holographic_debug_values["holographic_state_components_pre_gm_values"] = holographic_state_components_pre_gm
        
        # 构建 debug_info_for_gm
        debug_info_for_state_gm = (is_debug_enabled_for_method, probe_ts, "holographic_state_score_GM")

        holographic_state_score = _robust_geometric_mean(
            holographic_state_components_pre_gm, # 确保输入为正
            dynamic_state_fusion_weights,
            df_index,
            debug_info=debug_info_for_state_gm # 传递调试信息
        )
        holographic_debug_values["holographic_state_score_components"] = holographic_state_components
        holographic_debug_values["holographic_state_score"] = holographic_state_score

        # 趋势动量 diff() 化 (针对原子信号)
        flow_trend_raw = flow_outcome_raw.diff(3).fillna(0)
        structure_trend_raw = structure_outcome_raw.diff(3).fillna(0)
        potential_trend_raw = potential_outcome_raw.diff(3).fillna(0)

        flow_trend = self.helper._normalize_series(flow_trend_raw, df_index, bipolar=True)
        structure_trend = self.helper._normalize_series(structure_trend_raw, df_index, bipolar=True)
        potential_trend = self.helper._normalize_series(potential_trend_raw, df_index, bipolar=True)

        holographic_debug_values["flow_trend_raw"] = flow_trend_raw
        holographic_debug_values["structure_trend_raw"] = structure_trend_raw
        holographic_debug_values["potential_trend_raw"] = potential_trend_raw

        # 核心修复：修改键名以匹配 trend_fusion_weights 的键
        holographic_trend_components = {
            "flow_trend": flow_trend,
            "structure_trend": structure_trend,
            "potential_trend": potential_trend
        }
        debug_info_for_trend_gm = (is_debug_enabled_for_method, probe_ts, "holographic_trend_score_GM")
        holographic_trend_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in holographic_trend_components.items()}, # 确保输入为正
            trend_fusion_weights,
            df_index,
            debug_info=debug_info_for_trend_gm # 传递调试信息
        )
        holographic_debug_values["holographic_trend_score_components"] = holographic_trend_components
        holographic_debug_values["holographic_trend_score"] = holographic_trend_score

        # 整体融合
        overall_holographic_components = {
            "state": holographic_state_score,
            "trend": holographic_trend_score
        }
        debug_info_for_overall_gm = (is_debug_enabled_for_method, probe_ts, "holographic_validation_score_GM")
        holographic_validation_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in overall_holographic_components.items()}, # 确保输入为正
            overall_fusion_weights,
            df_index,
            debug_info=debug_info_for_overall_gm # 传递调试信息
        )
        # 将几何平均结果映射回 [-1, 1] 区间，通过乘以原始状态和趋势的平均符号
        weighted_avg_direction = (flow_outcome_raw * dynamic_state_fusion_weights["flow"] + 
                                  structure_outcome_raw * dynamic_state_fusion_weights["structure"] + 
                                  potential_outcome_raw * dynamic_state_fusion_weights["potential"])
        holographic_validation_score = holographic_validation_score * weighted_avg_direction.apply(np.sign).replace(0, 1) # 确保符号一致，0时视为正向

        return holographic_validation_score.clip(-1, 1), holographic_debug_values

    def _apply_quality_efficiency_calibration(self, dynamic_preliminary_score: pd.Series, holographic_validation_score: pd.Series, dynamic_efficiency_baseline: pd.Series, probe_ts: Optional[pd.Timestamp]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
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

        # --- 探针强化: 打印pow()操作的精确值 ---
        if probe_ts is not None and probe_ts in dynamic_preliminary_score.index and probe_ts in quality_efficiency_modulator.index:
            final_score_debug_values["dynamic_preliminary_score_at_probe_ts"] = dynamic_preliminary_score.loc[probe_ts]
            final_score_debug_values["quality_efficiency_modulator_at_probe_ts"] = quality_efficiency_modulator.loc[probe_ts]
            final_score_debug_values["calculated_pow_result_at_probe_ts"] = dynamic_preliminary_score.loc[probe_ts] ** quality_efficiency_modulator.loc[probe_ts]
        
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
