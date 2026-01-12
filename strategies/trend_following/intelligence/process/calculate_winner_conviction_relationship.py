# strategies\trend_following\intelligence\process\calculate_winner_conviction_relationship.py
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

class CalculateWinnerConvictionRelationship:
    """
    【V4.0 · 韧性博弈与多维时空版】“赢家信念”专属关系计算引擎
    - 核心重构: 彻底废弃旧的“状态对抗”逻辑。引入“信念强度 × 压力韧性 × 诡道过滤 × 情境调制”的全新四维诊断框架。
    - 核心升级:
        1.  **多时间维度斜率与加速度融合：** 对“赢家稳定性”和“利润兑现压力”进行多时间维度（5, 13, 21, 34, 55日）斜率和加速度的融合，评估其趋势和动能。
        2.  **共振与背离判断：** 评估“赢家稳定性”和“利润兑现压力”在多时间维度上的共振（同向增强/减弱）或背离（一强一弱）。
        3.  **历史相对位置：** 引入信号相对于其历史区间的百分位，判断其是处于高位还是低位。
        4.  **诡道博弈特性：** 引入欺骗指数、对倒强度等信号，对虚假的信念增强或减弱进行惩罚。
        5.  **情境调制：** 引入市场情绪、波动率等情境因子进行动态调整。
        6.  **非线性融合：** 使用 _robust_geometric_mean 对所有强度/幅度组件进行融合，并结合整体方向。
    - 目标: 提供一个双极性分数，正值代表赢家信念坚定，负值代表信念动摇或面临风险。
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
        【V4.0 · 韧性博弈与多维时空版】“赢家信念”专属关系计算引擎
        - 核心重构: 彻底废弃旧的“状态对抗”逻辑。引入“信念强度 × 压力韧性 × 诡道过滤 × 情境调制”的全新四维诊断框架。
        - 核心升级:
            1.  **多时间维度斜率与加速度融合：** 对“赢家稳定性”和“利润兑现压力”进行多时间维度（5, 13, 21, 34, 55日）斜率和加速度的融合，评估其趋势和动能。
            2.  **共振与背离判断：** 评估“赢家稳定性”和“利润兑现压力”在多时间维度上的共振（同向增强/减弱）或背离（一强一弱）。
            3.  **历史相对位置：** 引入信号相对于其历史区间的百分位，判断其是处于高位还是低位。
            4.  **诡道博弈特性：** 引入欺骗指数、对倒强度等信号，对虚假的信念增强或减弱进行惩罚。
            5.  **情境调制：** 引入市场情绪、波动率等情境因子进行动态调整。
            6.  **非线性融合：** 使用 _robust_geometric_mean 对所有强度/幅度组件进行融合，并结合整体方向。
        - 目标: 提供一个双极性分数，正值代表赢家信念坚定，负值代表信念动摇或面临风险。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            config (Dict): 包含配置信息的字典。
        返回:
            pd.Series: 融合后的MTF共振分数 (范围 [-1, 1])。
        """
        method_name = "calculate_winner_conviction_relationship"
        df_index = df.index
        is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values = self._setup_debug_context(df, method_name)
        all_params = self._get_all_params(config)
        signals_data = self._get_and_validate_signals(df, df_index, method_name, all_params, _temp_debug_values)
        if signals_data is None:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self.helper._print_debug_output(debug_output)
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # 归一化处理
        normalized_signals = self._normalize_raw_data(df_index, signals_data, _temp_debug_values)
        # 1. 信念强度
        conviction_strength_score = self._calculate_conviction_strength(df, df_index, method_name, signals_data, all_params, _temp_debug_values)
        # 2. 压力韧性
        pressure_resilience_score = self._calculate_pressure_resilience(df, df_index, method_name, signals_data, all_params, _temp_debug_values)
        # 3. 共振与背离因子
        synergy_factor = self._calculate_synergy_factor(df_index, conviction_strength_score, pressure_resilience_score, _temp_debug_values)
        # 4. 诡道过滤
        deception_filter = self._calculate_deception_filter(df, df_index, method_name, signals_data, all_params, _temp_debug_values)
        # 5. 情境调制
        context_modulator = self._calculate_contextual_modulator(df_index, signals_data, all_params, _temp_debug_values)
        # 6. 最终融合
        final_score = self._perform_final_fusion(df_index, conviction_strength_score, pressure_resilience_score, synergy_factor, deception_filter, context_modulator, all_params, _temp_debug_values)
        self._print_debug_info(method_name, final_score, is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values)
        return final_score.astype(np.float32)

    def _setup_debug_context(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
        """
        【V1.0 · 调试上下文设置版】统一设置调试相关的变量。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            Tuple[bool, Optional[pd.Timestamp], Dict, Dict]:
            (is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values)
        """
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算赢家信念关系..."] = ""
        return is_debug_enabled_for_method, probe_ts, debug_output, _temp_debug_values

    def _print_debug_info(self, method_name: str, final_score: pd.Series, is_debug_enabled_for_method: bool, probe_ts: Optional[pd.Timestamp], debug_output: Dict, _temp_debug_values: Dict):
        """
        【V1.0 · 调试信息打印版】统一打印调试信息。
        参数:
            method_name (str): 调用此方法的名称，用于日志输出。
            final_score (pd.Series): 最终计算出的分数。
            is_debug_enabled_for_method (bool): 是否启用调试。
            probe_ts (Optional[pd.Timestamp]): 探针日期。
            debug_output (Dict): 调试输出字典。
            _temp_debug_values (Dict): 临时调试值字典。
        """
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name, series in _temp_debug_values.get("原始信号值", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values.get("归一化处理", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 信念强度 ---"] = ""
            for key, series in _temp_debug_values.get("信念强度", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 压力韧性 ---"] = ""
            for key, series in _temp_debug_values.get("压力韧性", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 共振与背离因子 ---"] = ""
            for key, series in _temp_debug_values.get("共振与背离因子", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道过滤 ---"] = ""
            for key, series in _temp_debug_values.get("诡道过滤", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 情境调制 ---"] = ""
            for key, series in _temp_debug_values.get("情境调制", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合 ---"] = ""
            for key, series in _temp_debug_values.get("最终融合", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 赢家信念关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self.helper._print_debug_output(debug_output)

    def _get_all_params(self, config: Dict) -> Dict[str, Any]:
        """
        【V1.0 · 参数获取版】从 config 中获取所有必要的参数。
        参数:
            config (Dict): 包含配置信息的字典。
        返回:
            Dict[str, Any]: 包含所有参数的字典。
        """
        params = get_param_value(config.get('winner_conviction_params'), {})
        mtf_slope_accel_weights = get_param_value(config.get('mtf_slope_accel_weights'), {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        relative_position_weights = get_param_value(params.get('relative_position_weights'), {"winner_stability_high": 0.6, "profit_taking_flow_low": 0.4})
        context_modulator_weights = get_param_value(params.get('context_modulator_weights'), {"market_sentiment": 0.4, "volatility_stability": 0.3, "trend_vitality": 0.3})
        final_exponent = get_param_value(params.get('final_exponent'), 1.5)
        final_fusion_gm_weights = get_param_value(params.get('final_fusion_gm_weights'), {
            "conviction_magnitude": 0.3,
            "pressure_magnitude": 0.2,
            "synergy_factor": 0.2,
            "deception_filter": 0.15,
            "context_modulator": 0.15
        })
        direction_weights = get_param_value(params.get('direction_weights'), {'conviction': 0.6, 'pressure': 0.4})
        return {
            "mtf_slope_accel_weights": mtf_slope_accel_weights,
            "relative_position_weights": relative_position_weights,
            "context_modulator_weights": context_modulator_weights,
            "final_exponent": final_exponent,
            "final_fusion_gm_weights": final_fusion_gm_weights,
            "direction_weights": direction_weights
        }

    def _get_and_validate_signals(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, params: Dict, _temp_debug_values: Dict) -> Optional[Dict[str, pd.Series]]:
        """
        【V1.1 · 信号补全版】获取所有原始信号数据，并进行有效性校验。
        - 核心修复: 确保 `flow_credibility_raw` 被正确添加到返回的信号字典中。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            Optional[Dict[str, pd.Series]]: 包含所有原始信号Series的字典，如果校验失败则返回None。
        """
        belief_signal_name = 'winner_stability_index_D'
        pressure_signal_name = 'profit_taking_flow_ratio_D'
        required_signals = [
            belief_signal_name, pressure_signal_name,
            'deception_index_D', 'wash_trade_intensity_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'trend_vitality_index_D', 'flow_credibility_index_D' # 确保 flow_credibility_index_D 在 required_signals 中
        ]
        mtf_slope_accel_weights = params["mtf_slope_accel_weights"]
        for base_sig in [belief_signal_name, pressure_signal_name, 'deception_index_D', 'wash_trade_intensity_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            return None
        winner_stability_raw = self.helper._get_safe_series(df, belief_signal_name, 0.0, method_name=method_name)
        profit_taking_flow_raw = self.helper._get_safe_series(df, pressure_signal_name, 0.0, method_name=method_name)
        deception_index_raw = self.helper._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        wash_trade_intensity_raw = self.helper._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        market_sentiment_raw = self.helper._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        volatility_instability_raw = self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        trend_vitality_raw = self.helper._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        flow_credibility_raw = self.helper._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name) # 获取 flow_credibility_raw
        _temp_debug_values["原始信号值"] = {
            belief_signal_name: winner_stability_raw,
            pressure_signal_name: profit_taking_flow_raw,
            'deception_index_D': deception_index_raw,
            'wash_trade_intensity_D': wash_trade_intensity_raw,
            'market_sentiment_score_D': market_sentiment_raw,
            'VOLATILITY_INSTABILITY_INDEX_21d_D': volatility_instability_raw,
            'trend_vitality_index_D': trend_vitality_raw,
            'flow_credibility_index_D': flow_credibility_raw # 添加到调试输出
        }
        return {
            "winner_stability_raw": winner_stability_raw,
            "profit_taking_flow_raw": profit_taking_flow_raw,
            "deception_index_raw": deception_index_raw,
            "wash_trade_intensity_raw": wash_trade_intensity_raw,
            "market_sentiment_raw": market_sentiment_raw,
            "volatility_instability_raw": volatility_instability_raw,
            "trend_vitality_raw": trend_vitality_raw,
            "flow_credibility_raw": flow_credibility_raw, # 确保 flow_credibility_raw 被返回
            "belief_signal_name": belief_signal_name,
            "pressure_signal_name": pressure_signal_name
        }

    def _normalize_raw_data(self, df_index: pd.Index, signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V1.0 · 原始数据归一化版】归一化原始数据。
        参数:
            df_index (pd.Index): DataFrame的索引。
            signals (Dict[str, pd.Series]): 包含原始信号Series的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            Dict[str, pd.Series]: 包含归一化信号Series的字典。
        """
        flow_credibility_norm = self.helper._normalize_series(signals["flow_credibility_raw"], df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {
            "flow_credibility_norm": flow_credibility_norm
        }
        return {
            "flow_credibility_norm": flow_credibility_norm
        }

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, signals: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0 · 信念强度计算版】计算赢家信念强度。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            signals (Dict[str, pd.Series]): 包含原始信号Series的字典。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 赢家信念强度分数。
        """
        mtf_slope_accel_weights = params["mtf_slope_accel_weights"]
        relative_position_weights = params["relative_position_weights"]
        mtf_winner_stability = self.helper._get_mtf_slope_accel_score(df, signals["belief_signal_name"], mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        winner_stability_percentile = signals["winner_stability_raw"].rank(pct=True).fillna(0.5)
        conviction_strength_score = (mtf_winner_stability * relative_position_weights.get("winner_stability_high", 0.6) +
                                     (winner_stability_percentile * 2 - 1) * (1 - relative_position_weights.get("winner_stability_high", 0.6))).clip(-1, 1)
        _temp_debug_values["信念强度"] = {
            "mtf_winner_stability": mtf_winner_stability,
            "winner_stability_percentile": winner_stability_percentile,
            "conviction_strength_score": conviction_strength_score
        }
        return conviction_strength_score

    def _calculate_pressure_resilience(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, signals: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0 · 压力韧性计算版】计算压力韧性。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            signals (Dict[str, pd.Series]): 包含原始信号Series的字典。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 压力韧性分数。
        """
        mtf_slope_accel_weights = params["mtf_slope_accel_weights"]
        relative_position_weights = params["relative_position_weights"]
        mtf_profit_taking_flow = self.helper._get_mtf_slope_accel_score(df, signals["pressure_signal_name"], mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        profit_taking_flow_percentile = (1 - signals["profit_taking_flow_raw"].rank(pct=True)).fillna(0.5)
        pressure_resilience_score = ((mtf_profit_taking_flow * -1) * relative_position_weights.get("profit_taking_flow_low", 0.4) +
                                     (profit_taking_flow_percentile * 2 - 1) * (1 - relative_position_weights.get("profit_taking_flow_low", 0.4))).clip(-1, 1)
        _temp_debug_values["压力韧性"] = {
            "mtf_profit_taking_flow": mtf_profit_taking_flow,
            "profit_taking_flow_percentile": profit_taking_flow_percentile,
            "pressure_resilience_score": pressure_resilience_score
        }
        return pressure_resilience_score

    def _calculate_synergy_factor(self, df_index: pd.Index, conviction_strength_score: pd.Series, pressure_resilience_score: pd.Series, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0 · 共振与背离因子计算版】计算共振与背离因子。
        参数:
            df_index (pd.Index): DataFrame的索引。
            conviction_strength_score (pd.Series): 赢家信念强度分数。
            pressure_resilience_score (pd.Series): 压力韧性分数。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 共振与背离因子分数。
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

    def _calculate_deception_filter(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, signals: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0 · 诡道过滤计算版】计算诡道过滤因子。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            signals (Dict[str, pd.Series]): 包含原始信号Series的字典。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 诡道过滤因子分数。
        """
        mtf_slope_accel_weights = params["mtf_slope_accel_weights"]
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

    def _calculate_contextual_modulator(self, df_index: pd.Index, signals: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0 · 情境调制计算版】计算情境调制因子。
        参数:
            df_index (pd.Index): DataFrame的索引。
            signals (Dict[str, pd.Series]): 包含原始信号Series的字典。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 情境调制因子分数。
        """
        context_modulator_weights = params["context_modulator_weights"]
        norm_market_sentiment = self.helper._normalize_series(signals["market_sentiment_raw"], df_index, bipolar=True)
        volatility_stability_raw = 1 - normalize_score(signals["volatility_instability_raw"], df_index, 21, ascending=True, debug_info=False)
        norm_volatility_stability = self.helper._normalize_series(volatility_stability_raw, df_index, bipolar=False, ascending=True)
        norm_trend_vitality = self.helper._normalize_series(signals["trend_vitality_raw"], df_index, bipolar=False)
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_stability": norm_volatility_stability,
            "trend_vitality": norm_trend_vitality
        }
        context_modulator_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in context_modulator_components.items()},
            context_modulator_weights,
            df_index
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

    def _perform_final_fusion(self, df_index: pd.Index, conviction_strength_score: pd.Series, pressure_resilience_score: pd.Series, synergy_factor: pd.Series, deception_filter: pd.Series, context_modulator: pd.Series, params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.0 · 最终融合版】执行最终的非线性融合。
        参数:
            df_index (pd.Index): DataFrame的索引。
            conviction_strength_score (pd.Series): 赢家信念强度分数。
            pressure_resilience_score (pd.Series): 压力韧性分数。
            synergy_factor (pd.Series): 共振与背离因子分数。
            deception_filter (pd.Series): 诡道过滤因子分数。
            context_modulator (pd.Series): 情境调制因子分数。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 最终融合分数。
        """
        final_exponent = params["final_exponent"]
        final_fusion_gm_weights = params["final_fusion_gm_weights"]
        direction_weights = params["direction_weights"]
        overall_direction_raw = (conviction_strength_score * direction_weights.get('conviction', 0.6) + pressure_resilience_score * direction_weights.get('pressure', 0.4))
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
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in fusion_components_for_gm.items()},
            final_fusion_gm_weights,
            df_index
        )
        final_score = fused_magnitude * overall_direction
        final_score = np.sign(final_score) * (final_score.abs().pow(final_exponent))
        final_score = final_score.clip(-1, 1).fillna(0.0)
        _temp_debug_values["最终融合"] = {
            "overall_direction_raw": overall_direction_raw,
            "overall_direction": overall_direction,
            "conviction_magnitude": conviction_magnitude,
            "pressure_magnitude": pressure_magnitude,
            "fused_magnitude": fused_magnitude,
            "final_score": final_score
        }
        return final_score







