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
    【V1.0 · 拆单吸筹强度计算器】
    - 核心职责: 封装拆单吸筹强度的计算逻辑，提高代码模块化和可维护性。
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

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.2 · 质效校准版】计算“拆单吸筹强度”的专属信号。
        - 核心升级: 引入“效率基准线”(efficiency_baseline)概念。在计算“质效调节指数”前，
                      先对“全息验证综合分”进行校准。这使得任何低于基准线的战果（即使为正）
                      都会被视为负向贡献，从而受到惩罚性抑制，为模型注入了赏罚分明的“主帅”逻辑。
        """
        method_name = "calculate_split_order_accumulation" # 更改方法名以在新类中更清晰
        # --- 调试信息构建 ---
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
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算拆单吸筹强度..."] = ""

        df_index = df.index
        efficiency_baseline = config.get('efficiency_baseline', 0.15)

        # 1. 获取并归一化所有信号
        raw_signals, normalized_signals = self._get_and_normalize_signals(df, method_name)
        _temp_debug_values["原始信号值"] = raw_signals
        _temp_debug_values["归一化处理"] = normalized_signals

        # 2. 计算初步分数 (dynamic_preliminary_score)
        dynamic_preliminary_score = self._calculate_preliminary_score(normalized_signals, df_index, config)
        _temp_debug_values["初步计算"] = {
            "dynamic_preliminary_score": dynamic_preliminary_score
        }

        # 3. 计算全息验证分数 (holographic_validation_score)
        holographic_validation_score = self._calculate_holographic_validation(df, normalized_signals, df_index, config)
        _temp_debug_values["全息验证"] = {
            "holographic_validation_score": holographic_validation_score
        }

        # 4. 应用质效校准并计算最终分数
        final_score = self._apply_quality_efficiency_calibration(dynamic_preliminary_score, holographic_validation_score, efficiency_baseline)
        _temp_debug_values["最终分数"] = {
            "final_score": final_score
        }

        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            for section, values in _temp_debug_values.items():
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {section} ---"] = ""
                for key, series_or_value in values.items():
                    if isinstance(series_or_value, pd.Series):
                        val = series_or_value.loc[probe_ts] if probe_ts in series_or_value.index else np.nan
                        debug_output[f"        {key}: {val:.4f}"] = ""
                    else:
                        debug_output[f"        {key}: {series_or_value}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 拆单吸筹强度诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self.helper._print_debug_output(debug_output) # 使用 helper 的打印方法

        return final_score.astype(np.float32)

    def _get_and_normalize_signals(self, df: pd.DataFrame, method_name: str) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        获取并归一化所有拆单吸筹强度计算所需的原始信号。
        """
        df_index = df.index
        required_signals = [
            'hidden_accumulation_intensity_D', 'SLOPE_5_close_D', 'deception_index_D',
            'upward_impulse_purity_D', 'PROCESS_META_POWER_TRANSFER',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_DYN_AXIOM_STABILITY'
        ]
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return {}, {} # 如果验证失败，返回空字典

        # 原始数据获取
        raw_intensity = self.helper._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name)
        price_trend_raw = self.helper._get_safe_series(df, 'SLOPE_5_close_D', 0.0, method_name=method_name)
        deception_index = self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        upward_purity_raw = self.helper._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        flow_outcome = self.helper._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        structure_outcome = self.helper._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        potential_outcome = self.helper._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)

        raw_signals = {
            "hidden_accumulation_intensity_D": raw_intensity,
            "SLOPE_5_close_D": price_trend_raw,
            "deception_index_D": deception_index,
            "upward_impulse_purity_D": upward_purity_raw,
            "PROCESS_META_POWER_TRANSFER": flow_outcome,
            "SCORE_CHIP_STRATEGIC_POSTURE": structure_outcome,
            "SCORE_DYN_AXIOM_STABILITY": potential_outcome
        }

        # 归一化处理
        normalized_score = (raw_intensity / 100).clip(0, 1)
        price_trend_norm = self.helper._normalize_series(price_trend_raw, df_index, bipolar=True)
        upward_purity = self.helper._normalize_series(upward_purity_raw, df_index, bipolar=False)
        deception_norm = self.helper._normalize_series(deception_index, df_index, bipolar=True)

        normalized_signals = {
            "normalized_score": normalized_score,
            "price_trend_norm": price_trend_norm,
            "upward_purity": upward_purity,
            "deception_norm": deception_norm,
            "flow_outcome": flow_outcome, # 这些已经是分数，但为了保持一致性，也放在这里
            "structure_outcome": structure_outcome,
            "potential_outcome": potential_outcome
        }
        return raw_signals, normalized_signals

    def _calculate_preliminary_score(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> pd.Series:
        """
        计算初步的拆单吸筹强度分数。
        """
        normalized_score = normalized_signals["normalized_score"]
        price_trend_norm = normalized_signals["price_trend_norm"]
        upward_purity = normalized_signals["upward_purity"]
        deception_norm = normalized_signals["deception_norm"]
        potential_outcome = normalized_signals["potential_outcome"]

        price_suppression_factor = (1 - price_trend_norm.clip(lower=0) * (1 - upward_purity)).clip(0, 1)
        strategic_context_factor = (potential_outcome * 0.5 + deception_norm.clip(lower=0) * 0.5).clip(0, 1)

        preliminary_score = (normalized_score * price_suppression_factor * strategic_context_factor).pow(1/3).fillna(0.0)
        tactical_momentum_score = self.helper._normalize_series(preliminary_score.diff(1).fillna(0), df_index, bipolar=False)
        dynamic_preliminary_score = (preliminary_score * 0.7 + tactical_momentum_score * 0.3).clip(0, 1)

        return dynamic_preliminary_score

    def _calculate_holographic_validation(self, df: pd.DataFrame, normalized_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> pd.Series:
        """
        计算全息验证分数。
        """
        flow_outcome = normalized_signals["flow_outcome"]
        structure_outcome = normalized_signals["structure_outcome"]
        potential_outcome = normalized_signals["potential_outcome"] # 这就是 stability_score

        stability_score = potential_outcome # 为清晰起见重命名，与原始代码保持一致
        weight_flow = 1 - stability_score
        weight_structure = stability_score
        total_weight = weight_flow + weight_structure + 0.2 # 0.2 是 potential_outcome 的权重
        # 确保 total_weight 不为零以避免除以零错误
        total_weight = total_weight.replace(0, 1e-9)

        w_f = weight_flow / total_weight
        w_s = weight_structure / total_weight
        w_p = 0.2 / total_weight # potential_outcome 的固定权重

        holographic_state_score = (flow_outcome * w_f + structure_outcome * w_s + potential_outcome * w_p)

        flow_trend = self.helper._normalize_series(flow_outcome.diff(3).fillna(0), df_index, bipolar=True)
        structure_trend = self.helper._normalize_series(structure_outcome.diff(3).fillna(0), df_index, bipolar=True)
        potential_trend = self.helper._normalize_series(potential_outcome.diff(3).fillna(0), df_index, bipolar=True)

        holographic_trend_score = (flow_trend * w_f + structure_trend * w_s + potential_trend * w_p)
        holographic_validation_score = (holographic_state_score * 0.6 + holographic_trend_score * 0.4).clip(-1, 1)

        return holographic_validation_score

    def _apply_quality_efficiency_calibration(self, dynamic_preliminary_score: pd.Series, holographic_validation_score: pd.Series, efficiency_baseline: float) -> pd.Series:
        """
        应用质效校准并计算最终分数。
        """
        calibrated_holographic_score = holographic_validation_score - efficiency_baseline
        quality_efficiency_modulator = (1 - calibrated_holographic_score).clip(0.1, 2.0)
        final_score = dynamic_preliminary_score.pow(quality_efficiency_modulator).clip(0, 1).fillna(0.0)

        return final_score
