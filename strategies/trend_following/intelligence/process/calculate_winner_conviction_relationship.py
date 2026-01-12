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
        【V4.1 · 参数修正版】“赢家信念”专属关系计算引擎
        - 核心修正: 修复 `_calculate_conviction_strength`, `_calculate_pressure_resilience`,
                    `_calculate_deception_filter`, `_calculate_contextual_modulator` 方法调用时缺少 `normalized_signals` 参数的错误。
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
        conviction_strength_score = self._calculate_conviction_strength(df, df_index, method_name, signals_data, normalized_signals, all_params, _temp_debug_values)
        # 2. 压力韧性
        pressure_resilience_score = self._calculate_pressure_resilience(df, df_index, method_name, signals_data, normalized_signals, all_params, _temp_debug_values)
        # 3. 共振与背离因子
        synergy_factor = self._calculate_synergy_factor(df_index, conviction_strength_score, pressure_resilience_score, _temp_debug_values)
        # 4. 诡道过滤
        deception_filter = self._calculate_deception_filter(df, df_index, method_name, signals_data, normalized_signals, all_params, _temp_debug_values)
        # 5. 情境调制
        context_modulator = self._calculate_contextual_modulator(df_index, signals_data, normalized_signals, all_params, _temp_debug_values)
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
        【V1.2 · 调试信息全面打印版】统一打印调试信息，新增累积上下文、趋势一致性、拐点信号的输出。
        参数:
            method_name (str): 调用此方法的名称，用于日志输出。
            final_score (pd.Series): 最终计算出的分数。
            is_debug_enabled_for_method (bool): 是否启用调试。
            probe_ts (Optional[pd.Timestamp]): 探针日期。
            debug_output (Dict): 调试输出字典。 (这是 calculate 方法的顶层 debug_output)
            _temp_debug_values (Dict): 临时调试值字典。
        """
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name, series in _temp_debug_values.get("原始信号值", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF信号值 ---"] = ""
            for key, series in _temp_debug_values.get("MTF信号值", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 累积上下文信号值 ---"] = ""
            for key, series in _temp_debug_values.get("累积上下文信号值", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            # 新增：打印 _get_and_validate_signals 内部收集的详细调试信息
            internal_get_signals_debug = _temp_debug_values.get("_get_and_validate_signals_debug_output", {})
            if internal_get_signals_debug:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 累积上下文计算详情 ---"] = ""
                for key, value in internal_get_signals_debug.items():
                    # 假设 value 已经是格式化好的字符串或空字符串
                    debug_output[key] = value # 直接添加到主 debug_output
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF趋势一致性信号值 ---"] = ""
            for key, series in _temp_debug_values.get("MTF趋势一致性信号值", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 拐点信号值 ---"] = ""
            for key, series in _temp_debug_values.get("拐点信号值", {}).items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
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
        【V1.5 · 参数全面扩展与累积流、趋势一致性、拐点参数版 - 压力信号更新】从 config 中获取所有必要的参数。
        新增了信念增强、压力韧性增强、诡道过滤增强和情境调制增强的权重，累积流参数，
        以及趋势一致性权重和拐点惩罚强度。
        核心修正：将 `profit_taking_flow_low` 替换为更通用的 `selling_pressure_low`，并更新相关权重。
        核心新增：在 `pressure_resilience_enhancement_weights` 中加入 `distribution_at_peak_intensity` 和 `upper_shadow_selling_pressure`。
        参数:
            config (Dict): 包含配置信息的字典。
        返回:
            Dict[str, Any]: 包含所有参数的字典。
        """
        params = get_param_value(config.get('winner_conviction_params'), {})
        # 更新MTF权重，加入5日周期
        mtf_slope_accel_weights = get_param_value(config.get('mtf_slope_accel_weights'), {
            "slope_periods": {"5": 0.3, "13": 0.3, "21": 0.2, "34": 0.1, "55": 0.1}, # 增加5日周期
            "accel_periods": {"5": 0.5, "13": 0.3, "21": 0.2} # 增加5日周期
        })
        # 修正：将 profit_taking_flow_low 替换为 selling_pressure_low
        relative_position_weights = get_param_value(params.get('relative_position_weights'), {"winner_stability_high": 0.6, "selling_pressure_low": 0.4})
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
        # 新增参数：信念增强因子权重
        conviction_enhancement_weights = get_param_value(params.get('conviction_enhancement_weights'), {
            "main_force_conviction": 0.2,
            "chip_health": 0.2,
            "winner_profit_margin_avg": 0.1,
            "loser_loss_margin_avg_inverse": 0.1,
            "winner_concentration_90pct": 0.1,
            "chip_fatigue_inverse": 0.1,
            "cost_gini_coefficient_inverse": 0.05,
            "winner_stability_trend_consistency": 0.1 # 新增
        })
        # 新增参数：压力韧性增强因子权重
        pressure_resilience_enhancement_weights = get_param_value(params.get('pressure_resilience_enhancement_weights'), {
            "main_force_buy_execution_alpha": 0.2,
            "bid_side_liquidity": 0.2,
            "absorption_strength_ma5": 0.2,
            "active_buying_support": 0.15,
            "large_order_support": 0.15,
            "dip_absorption_power": 0.1,
            "selling_pressure_trend_consistency": 0.1, # 修正：替换为更通用的 selling_pressure_trend_consistency
            "distribution_at_peak_intensity": 0.1, # 新增
            "upper_shadow_selling_pressure": 0.1 # 新增
        })
        # 新增参数：诡道过滤增强因子权重
        deception_enhancement_weights = get_param_value(params.get('deception_enhancement_weights'), {
            "smart_money_divergence": 0.3,
            "covert_accumulation_inverse": 0.2,
            "closing_auction_ambush_inverse": 0.1
        })
        # 新增参数：情境调制增强因子权重
        context_modulator_enhancement_weights = get_param_value(params.get('context_modulator_enhancement_weights'), {
            "theme_hotness": 0.2,
            "industry_leader_score": 0.1,
            "market_impact_cost_inverse": 0.05
        })
        # 累积流参数
        cumulative_flow_params = get_param_value(params.get('cumulative_flow_params'), {
            "flow_signals_for_cumulative_context": [ # 需要进行累积上下文调制的资金流信号 (原始D后缀)
                'main_force_conviction_index_D',
                'main_force_buy_execution_alpha_D',
                'bid_side_liquidity_D',
                'absorption_strength_ma5_D',
                'active_buying_support_D',
                'large_order_support_D',
                'covert_accumulation_signal_D',
                'closing_auction_ambush_D',
                'dip_absorption_power_D',
                'dispersal_by_distribution_D' # 新增：将新的压力信号加入累积上下文
            ],
            "cumulative_periods": [13, 21], # 累积周期
            "cumulative_weights": {13: 0.6, 21: 0.4}, # 累积周期权重
            "cumulative_modulation_strength": 0.7 # 累积上下文对MTF分数的调制强度 (0到1)
        })
        # 新增：拐点检测参数
        inflection_point_params = get_param_value(params.get('inflection_point_params'), {
            "signals_for_inflection_detection": [ # 需要检测拐点的MTF信号
                'mtf_winner_stability_index',
                'mtf_dispersal_by_distribution', # 修正：替换为新的压力信号
                'mtf_main_force_conviction_index',
                'mtf_chip_health_score'
            ],
            "inflection_detection_window": 5, # 拐点检测的平滑窗口
            "inflection_penalty_strength": 0.2 # 拐点惩罚强度 (0到1)
        })
        return {
            "mtf_slope_accel_weights": mtf_slope_accel_weights,
            "relative_position_weights": relative_position_weights,
            "context_modulator_weights": context_modulator_weights,
            "final_exponent": final_exponent,
            "final_fusion_gm_weights": final_fusion_gm_weights,
            "direction_weights": direction_weights,
            "conviction_enhancement_weights": conviction_enhancement_weights,
            "pressure_resilience_enhancement_weights": pressure_resilience_enhancement_weights,
            "deception_enhancement_weights": deception_enhancement_weights,
            "context_modulator_enhancement_weights": context_modulator_enhancement_weights,
            "cumulative_flow_params": cumulative_flow_params,
            "inflection_point_params": inflection_point_params # 添加拐点参数
        }

    def _get_and_validate_signals(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, params: Dict, _temp_debug_values: Dict) -> Optional[Dict[str, pd.Series]]:
        """
        【V2.1 · 原始信号、累积上下文、趋势一致性与拐点补充版 - 压力信号更新】获取所有原始信号数据及其多时间框架斜率/加速度，并进行有效性校验。
        - 核心修正: 替换 `profit_taking_flow_ratio_D` 为 `dispersal_by_distribution_D` 作为核心压力信号。
        - 核心新增: 补充 `distribution_at_peak_intensity_D` 和 `upper_shadow_selling_pressure_D` 原始信号的获取。
        - 核心新增: 计算并存储指定资金流信号的累积上下文分数。
        - 核心新增: 计算并存储指定信号的MTF趋势一致性分数。
        - 核心新增: 计算并存储指定MTF信号的拐点检测分数。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            Optional[Dict[str, pd.Series]]: 包含所有原始信号Series和MTF信号Series的字典，如果校验失败则返回None。
        """
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {} # 局部调试输出，最终会合并到_temp_debug_values
        belief_signal_name = 'winner_stability_index_D'
        pressure_signal_name = 'dispersal_by_distribution_D' # 修正：替换为新的压力信号
        # 所有需要进行MTF斜率和加速度分析的原始信号 (原始列名)
        mtf_base_signals_raw_names = [
            belief_signal_name, pressure_signal_name, # 修正：使用新的压力信号
            'deception_index_D', 'wash_trade_intensity_D',
            'winner_profit_margin_avg_D', 'loser_loss_margin_avg_D',
            'main_force_conviction_index_D', 'chip_health_score_D',
            'main_force_buy_execution_alpha_D', 'bid_side_liquidity_D',
            'absorption_strength_ma5_D', 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'winner_concentration_90pct_D', 'chip_fatigue_index_D',
            'active_buying_support_D', 'large_order_support_D',
            'covert_accumulation_signal_D', 'cost_gini_coefficient_D',
            'market_impact_cost_D', 'closing_auction_ambush_D',
            'distribution_at_peak_intensity_D', # 新增
            'upper_shadow_selling_pressure_D' # 新增
        ]
        # 所有非MTF的原始信号 (原始列名)
        non_mtf_raw_signals_raw_names = [
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'trend_vitality_index_D', 'flow_credibility_index_D',
            'THEME_HOTNESS_SCORE_D', 'industry_leader_score_D',
            'dip_absorption_power_D'
        ]
        # 所有需要获取原始数据的信号名称（带_D后缀）
        all_raw_signal_names_with_D_suffix = list(set(mtf_base_signals_raw_names + non_mtf_raw_signals_raw_names))
        # 生成用于校验的required_signals列表
        required_signals_for_validation = list(all_raw_signal_names_with_D_suffix)
        mtf_slope_accel_weights = params["mtf_slope_accel_weights"]
        # 动态添加所有MTF信号的列名到 required_signals_for_validation
        for base_sig in mtf_base_signals_raw_names:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals_for_validation.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals_for_validation.append(f'ACCEL_{period_str}_{base_sig}')
        # 执行信号存在性校验
        if not self.helper._validate_required_signals(df, required_signals_for_validation, method_name):
            return None
        # 获取所有原始信号数据，并显式定义键名
        signals_data = {
            "winner_stability_index_raw": self.helper._get_safe_series(df, 'winner_stability_index_D', np.nan, method_name=method_name),
            "dispersal_by_distribution_raw": self.helper._get_safe_series(df, 'dispersal_by_distribution_D', np.nan, method_name=method_name), # 修正：替换为新的压力信号
            "deception_index_raw": self.helper._get_safe_series(df, 'deception_index_D', np.nan, method_name=method_name),
            "wash_trade_intensity_raw": self.helper._get_safe_series(df, 'wash_trade_intensity_D', np.nan, method_name=method_name),
            "market_sentiment_score_raw": self.helper._get_safe_series(df, 'market_sentiment_score_D', np.nan, method_name=method_name),
            "volatility_instability_index_21d_raw": self.helper._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', np.nan, method_name=method_name),
            "trend_vitality_index_raw": self.helper._get_safe_series(df, 'trend_vitality_index_D', np.nan, method_name=method_name),
            "flow_credibility_index_raw": self.helper._get_safe_series(df, 'flow_credibility_index_D', np.nan, method_name=method_name),
            "winner_profit_margin_avg_raw": self.helper._get_safe_series(df, 'winner_profit_margin_avg_D', np.nan, method_name=method_name),
            "loser_loss_margin_avg_raw": self.helper._get_safe_series(df, 'loser_loss_margin_avg_D', np.nan, method_name=method_name),
            "main_force_conviction_index_raw": self.helper._get_safe_series(df, 'main_force_conviction_index_D', np.nan, method_name=method_name),
            "chip_health_score_raw": self.helper._get_safe_series(df, 'chip_health_score_D', np.nan, method_name=method_name),
            "main_force_buy_execution_alpha_raw": self.helper._get_safe_series(df, 'main_force_buy_execution_alpha_D', np.nan, method_name=method_name),
            "bid_side_liquidity_raw": self.helper._get_safe_series(df, 'bid_side_liquidity_D', np.nan, method_name=method_name),
            "absorption_strength_ma5_raw": self.helper._get_safe_series(df, 'absorption_strength_ma5_D', np.nan, method_name=method_name),
            "smart_money_divergence_hm_buy_inst_sell_raw": self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', np.nan, method_name=method_name),
            "theme_hotness_score_raw": self.helper._get_safe_series(df, 'THEME_HOTNESS_SCORE_D', np.nan, method_name=method_name),
            "winner_concentration_90pct_raw": self.helper._get_safe_series(df, 'winner_concentration_90pct_D', np.nan, method_name=method_name),
            "chip_fatigue_index_raw": self.helper._get_safe_series(df, 'chip_fatigue_index_D', np.nan, method_name=method_name),
            "active_buying_support_raw": self.helper._get_safe_series(df, 'active_buying_support_D', np.nan, method_name=method_name),
            "large_order_support_raw": self.helper._get_safe_series(df, 'large_order_support_D', np.nan, method_name=method_name),
            "covert_accumulation_signal_raw": self.helper._get_safe_series(df, 'covert_accumulation_signal_D', np.nan, method_name=method_name),
            "industry_leader_score_raw": self.helper._get_safe_series(df, 'industry_leader_score_D', np.nan, method_name=method_name),
            "cost_gini_coefficient_raw": self.helper._get_safe_series(df, 'cost_gini_coefficient_D', np.nan, method_name=method_name),
            "market_impact_cost_raw": self.helper._get_safe_series(df, 'market_impact_cost_D', np.nan, method_name=method_name),
            "closing_auction_ambush_raw": self.helper._get_safe_series(df, 'closing_auction_ambush_D', np.nan, method_name=method_name),
            "dip_absorption_power_raw": self.helper._get_safe_series(df, 'dip_absorption_power_D', np.nan, method_name=method_name),
            "distribution_at_peak_intensity_raw": self.helper._get_safe_series(df, 'distribution_at_peak_intensity_D', np.nan, method_name=method_name), # 新增
            "upper_shadow_selling_pressure_raw": self.helper._get_safe_series(df, 'upper_shadow_selling_pressure_D', np.nan, method_name=method_name) # 新增
        }
        # 获取所有MTF信号数据
        for base_sig in mtf_base_signals_raw_names:
            # MTF信号的键名保持mtf_前缀和原始信号名的小写形式
            mtf_key = f"mtf_{base_sig.replace('_D', '').lower()}"
            signals_data[mtf_key] = self.helper._get_mtf_slope_accel_score(
                df, base_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True
            )
        # 特殊处理一些MTF信号的bipolar/ascending，并统一键名格式
        # 注意：这里直接覆盖了上面循环中可能生成的mtf_key，以确保特殊处理的参数生效
        signals_data["mtf_deception_index"] = self.helper._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False, ascending=True)
        signals_data["mtf_wash_trade_intensity"] = self.helper._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False, ascending=True)
        signals_data["mtf_smart_money_divergence_hm_buy_inst_sell"] = self.helper._get_mtf_slope_accel_score(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False, ascending=True)
        signals_data["mtf_chip_fatigue_index"] = self.helper._get_mtf_slope_accel_score(df, 'chip_fatigue_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False, ascending=False)
        signals_data["mtf_covert_accumulation_signal"] = self.helper._get_mtf_slope_accel_score(df, 'covert_accumulation_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False, ascending=True)
        signals_data["mtf_closing_auction_ambush"] = self.helper._get_mtf_slope_accel_score(df, 'closing_auction_ambush_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False, ascending=True)
        signals_data["mtf_market_impact_cost"] = self.helper._get_mtf_slope_accel_score(df, 'market_impact_cost_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False, ascending=False)
        # 新增：为新的压力信号生成MTF分数，高派发/卖压是负面，所以 ascending=False
        signals_data["mtf_dispersal_by_distribution"] = self.helper._get_mtf_slope_accel_score(df, 'dispersal_by_distribution_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True, ascending=False)
        signals_data["mtf_distribution_at_peak_intensity"] = self.helper._get_mtf_slope_accel_score(df, 'distribution_at_peak_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True, ascending=False)
        signals_data["mtf_upper_shadow_selling_pressure"] = self.helper._get_mtf_slope_accel_score(df, 'upper_shadow_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True, ascending=False)

        # 计算并存储指定资金流信号的累积上下文分数
        cumulative_flow_params = params["cumulative_flow_params"]
        flow_signals_for_cumulative_context = cumulative_flow_params["flow_signals_for_cumulative_context"]
        cumulative_periods = cumulative_flow_params["cumulative_periods"]
        cumulative_weights = cumulative_flow_params["cumulative_weights"]
        for flow_sig_D in flow_signals_for_cumulative_context:
            raw_series = self.helper._get_safe_series(df, flow_sig_D, np.nan, method_name=method_name)
            if not raw_series.isnull().all():
                cumulative_score = self.helper._get_cumulative_context_score(
                    raw_series, df_index, cumulative_periods, cumulative_weights, bipolar=True,
                    signal_name=flow_sig_D, is_debug_enabled_for_method=is_debug_enabled_for_method, probe_ts=probe_ts, debug_output=debug_output # 传入调试参数
                )
                signals_data[f"cumulative_{flow_sig_D.replace('_D', '').lower()}_score"] = cumulative_score
        # 计算并存储指定信号的MTF趋势一致性分数
        signals_for_trend_consistency = [
            belief_signal_name,
            pressure_signal_name, # 修正：使用新的压力信号
            'main_force_conviction_index_D',
            'chip_health_score_D'
        ]
        for base_sig_D in signals_for_trend_consistency:
            trend_consistency_score = self.helper._get_mtf_trend_consistency_score(
                df, base_sig_D, mtf_slope_accel_weights, df_index, method_name
            )
            signals_data[f"mtf_trend_consistency_{base_sig_D.replace('_D', '').lower()}"] = trend_consistency_score
        # 计算并存储指定MTF信号的拐点检测分数
        inflection_point_params = params["inflection_point_params"]
        signals_for_inflection_detection = inflection_point_params["signals_for_inflection_detection"]
        inflection_detection_window = inflection_point_params["inflection_detection_window"]
        for mtf_sig_key in signals_for_inflection_detection:
            if mtf_sig_key in signals_data:
                inflection_score = self.helper._detect_inflection_point(
                    signals_data[mtf_sig_key], inflection_detection_window
                )
                signals_data[f"inflection_{mtf_sig_key}"] = inflection_score
        # 将 belief_signal_name 和 pressure_signal_name 也添加到返回字典
        signals_data["belief_signal_name"] = belief_signal_name
        signals_data["pressure_signal_name"] = pressure_signal_name
        # 更新调试输出
        _temp_debug_values["原始信号值"] = {k: v for k, v in signals_data.items() if not k.startswith('mtf_') and not k.startswith('cumulative_') and not k.startswith('inflection_') and not k.endswith('_name')}
        _temp_debug_values["MTF信号值"] = {k: v for k, v in signals_data.items() if k.startswith('mtf_') and not k.startswith('mtf_trend_consistency_')}
        _temp_debug_values["累积上下文信号值"] = {k: v for k, v in signals_data.items() if k.startswith('cumulative_')}
        _temp_debug_values["MTF趋势一致性信号值"] = {k: v for k, v in signals_data.items() if k.startswith('mtf_trend_consistency_')}
        _temp_debug_values["拐点信号值"] = {k: v for k, v in signals_data.items() if k.startswith('inflection_')}
        # 合并局部调试输出到_temp_debug_values
        if is_debug_enabled_for_method and probe_ts:
            _temp_debug_values["_get_and_validate_signals_debug_output"] = debug_output
        return signals_data

    def _normalize_raw_data(self, df_index: pd.Index, signals: Dict[str, pd.Series], _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V1.8 · 原始数据归一化键名全面修正版 - 压力信号更新】归一化原始数据，修正了访问signals字典时所有键名的大小写错误，并补充了新信号的归一化。
        此方法主要处理非MTF原始信号的归一化，MTF信号已在_get_and_validate_signals中通过_get_mtf_slope_accel_score内部归一化。
        核心修正：移除 `profit_taking_flow_ratio_norm`，新增 `dispersal_by_distribution_norm`、
                  `distribution_at_peak_intensity_norm` 和 `upper_shadow_selling_pressure_norm`。
        参数:
            df_index (pd.Index): DataFrame的索引。
            signals (Dict[str, pd.Series]): 包含原始信号Series的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            Dict[str, pd.Series]: 包含归一化信号Series的字典。
        """
        normalized_signals = {}
        # 归一化非MTF信号，使用显式定义的键名
        normalized_signals["flow_credibility_norm"] = self.helper._normalize_series(signals["flow_credibility_index_raw"], df_index, bipolar=False)
        normalized_signals["winner_profit_margin_avg_norm"] = self.helper._normalize_series(signals["winner_profit_margin_avg_raw"], df_index, bipolar=False, ascending=True)
        normalized_signals["loser_loss_margin_avg_norm"] = self.helper._normalize_series(signals["loser_loss_margin_avg_raw"], df_index, bipolar=False, ascending=False) # 输家亏损率越低越好
        normalized_signals["main_force_conviction_norm"] = self.helper._normalize_series(signals["main_force_conviction_index_raw"], df_index, bipolar=True, ascending=True)
        normalized_signals["chip_health_norm"] = self.helper._normalize_series(signals["chip_health_score_raw"], df_index, bipolar=True, ascending=True)
        normalized_signals["main_force_buy_execution_alpha_norm"] = self.helper._normalize_series(signals["main_force_buy_execution_alpha_raw"], df_index, bipolar=True, ascending=True)
        normalized_signals["bid_side_liquidity_norm"] = self.helper._normalize_series(signals["bid_side_liquidity_raw"], df_index, bipolar=False, ascending=True)
        normalized_signals["absorption_strength_ma5_norm"] = self.helper._normalize_series(signals["absorption_strength_ma5_raw"], df_index, bipolar=False, ascending=True)
        normalized_signals["smart_money_divergence_norm"] = self.helper._normalize_series(signals["smart_money_divergence_hm_buy_inst_sell_raw"], df_index, bipolar=True, ascending=False) # 聪明钱分歧越大越不好
        normalized_signals["theme_hotness_norm"] = self.helper._normalize_series(signals["theme_hotness_score_raw"], df_index, bipolar=False, ascending=True)
        normalized_signals["winner_concentration_90pct_norm"] = self.helper._normalize_series(signals["winner_concentration_90pct_raw"], df_index, bipolar=False, ascending=True)
        normalized_signals["chip_fatigue_norm"] = self.helper._normalize_series(signals["chip_fatigue_index_raw"], df_index, bipolar=False, ascending=False) # 筹码疲劳度越低越好
        normalized_signals["active_buying_support_norm"] = self.helper._normalize_series(signals["active_buying_support_raw"], df_index, bipolar=False, ascending=True)
        normalized_signals["large_order_support_norm"] = self.helper._normalize_series(signals["large_order_support_raw"], df_index, bipolar=False, ascending=True)
        normalized_signals["covert_accumulation_norm"] = self.helper._normalize_series(signals["covert_accumulation_signal_raw"], df_index, bipolar=False, ascending=False) # 隐蔽吸筹信号越低越好（作为欺骗的反向指标）
        normalized_signals["industry_leader_score_norm"] = self.helper._normalize_series(signals["industry_leader_score_raw"], df_index, bipolar=False, ascending=True)
        normalized_signals["cost_gini_coefficient_norm"] = self.helper._normalize_series(signals["cost_gini_coefficient_raw"], df_index, bipolar=False, ascending=False) # 基尼系数越低越好（筹码越均匀）
        normalized_signals["market_impact_cost_norm"] = self.helper._normalize_series(signals["market_impact_cost_raw"], df_index, bipolar=False, ascending=False) # 冲击成本越低越好
        normalized_signals["closing_auction_ambush_norm"] = self.helper._normalize_series(signals["closing_auction_ambush_raw"], df_index, bipolar=False, ascending=False) # 尾盘伏击越低越好（作为欺骗的反向指标）
        normalized_signals["dip_absorption_power_norm"] = self.helper._normalize_series(signals["dip_absorption_power_raw"], df_index, bipolar=False, ascending=True) # 新增：下跌吸筹能力归一化
        # 新增：压力信号的归一化，高值代表高压力，所以 ascending=False
        normalized_signals["dispersal_by_distribution_norm"] = self.helper._normalize_series(signals["dispersal_by_distribution_raw"], df_index, bipolar=False, ascending=False)
        normalized_signals["distribution_at_peak_intensity_norm"] = self.helper._normalize_series(signals["distribution_at_peak_intensity_raw"], df_index, bipolar=False, ascending=False)
        normalized_signals["upper_shadow_selling_pressure_norm"] = self.helper._normalize_series(signals["upper_shadow_selling_pressure_raw"], df_index, bipolar=False, ascending=False)
        _temp_debug_values["归一化处理"] = normalized_signals
        return normalized_signals

    def _calculate_conviction_strength(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.6 · 信念强度累积上下文、趋势一致性与拐点调制版】计算赢家信念强度。
        融入了更多MTF和归一化信号，修正了键名，新增累积上下文调制、趋势一致性增强和拐点惩罚。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            signals (Dict[str, pd.Series]): 包含原始信号Series和MTF信号Series的字典。
            normalized_signals (Dict[str, pd.Series]): 包含归一化信号Series的字典。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 赢家信念强度分数。
        """
        relative_position_weights = params["relative_position_weights"]
        conviction_enhancement_weights = params["conviction_enhancement_weights"]
        cumulative_modulation_strength = params["cumulative_flow_params"]["cumulative_modulation_strength"]
        inflection_penalty_strength = params["inflection_point_params"]["inflection_penalty_strength"]
        # 核心赢家稳定性MTF分数
        mtf_winner_stability = signals["mtf_winner_stability_index"]
        winner_stability_percentile = signals["winner_stability_index_raw"].rank(pct=True).fillna(0.5)
        core_conviction_component = (mtf_winner_stability * relative_position_weights.get("winner_stability_high", 0.6) +
                                     (winner_stability_percentile * 2 - 1) * (1 - relative_position_weights.get("winner_stability_high", 0.6)))
        # 获取MTF信号
        mtf_main_force_conviction_index = signals["mtf_main_force_conviction_index"]
        mtf_chip_health_score = signals["mtf_chip_health_score"]
        mtf_winner_profit_margin_avg = signals["mtf_winner_profit_margin_avg"]
        mtf_loser_loss_margin_avg = signals["mtf_loser_loss_margin_avg"]
        mtf_winner_concentration_90pct = signals["mtf_winner_concentration_90pct"]
        mtf_chip_fatigue_index = signals["mtf_chip_fatigue_index"]
        mtf_cost_gini_coefficient = signals["mtf_cost_gini_coefficient"]
        # 对相关MTF信号进行累积上下文调制
        if "cumulative_main_force_conviction_index_score" in signals:
            cumulative_score = signals["cumulative_main_force_conviction_index_score"]
            mtf_main_force_conviction_index = mtf_main_force_conviction_index + (cumulative_score - mtf_main_force_conviction_index) * cumulative_modulation_strength
            mtf_main_force_conviction_index = mtf_main_force_conviction_index.clip(-1, 1)
        if "cumulative_chip_health_score" in signals:
            cumulative_score = signals["cumulative_chip_health_score"]
            mtf_chip_health_score = mtf_chip_health_score + (cumulative_score - mtf_chip_health_score) * cumulative_modulation_strength
            mtf_chip_health_score = mtf_chip_health_score.clip(-1, 1)
        if "cumulative_winner_concentration_90pct_score" in signals:
            cumulative_score = signals["cumulative_winner_concentration_90pct_score"]
            mtf_winner_concentration_90pct = mtf_winner_concentration_90pct + (cumulative_score - mtf_winner_concentration_90pct) * cumulative_modulation_strength
            mtf_winner_concentration_90pct = mtf_winner_concentration_90pct.clip(-1, 1)
        if "cumulative_chip_fatigue_index_score" in signals:
            cumulative_score = signals["cumulative_chip_fatigue_index_score"]
            mtf_chip_fatigue_index = mtf_chip_fatigue_index + (cumulative_score - mtf_chip_fatigue_index) * cumulative_modulation_strength
            mtf_chip_fatigue_index = mtf_chip_fatigue_index.clip(-1, 1)
        if "cumulative_cost_gini_coefficient_score" in signals:
            cumulative_score = signals["cumulative_cost_gini_coefficient_score"]
            mtf_cost_gini_coefficient = mtf_cost_gini_coefficient + (cumulative_score - mtf_cost_gini_coefficient) * cumulative_modulation_strength
            mtf_cost_gini_coefficient = mtf_cost_gini_coefficient.clip(-1, 1)
        # 引入MTF趋势一致性分数
        mtf_trend_consistency_winner_stability = signals.get("mtf_trend_consistency_winner_stability_index", pd.Series(0.0, index=df_index))
        # 引入拐点惩罚
        inflection_mtf_winner_stability = signals.get("inflection_mtf_winner_stability_index", pd.Series(0.0, index=df_index))
        inflection_penalty = pd.Series(0.0, index=df_index, dtype=np.float32)
        # 当MTF分数 > 0 且 inflection_score > 0 (向上趋势减弱或向下趋势增强) -> 惩罚
        inflection_penalty = inflection_penalty.mask((mtf_winner_stability > 0) & (inflection_mtf_winner_stability > 0), inflection_mtf_winner_stability * inflection_penalty_strength)
        # 当MTF分数 < 0 且 inflection_score < 0 (向下趋势减弱或向上趋势增强) -> 奖励 (负负得正，减少负面影响)
        inflection_penalty = inflection_penalty.mask((mtf_winner_stability < 0) & (inflection_mtf_winner_stability < 0), inflection_mtf_winner_stability * inflection_penalty_strength) # 负值惩罚，实际是奖励
        # 将拐点惩罚应用于核心信念组件
        core_conviction_component = core_conviction_component - inflection_penalty
        all_conviction_components = {
            "core_conviction": core_conviction_component,
            "main_force_conviction": mtf_main_force_conviction_index, # 使用调制后的MTF版本
            "chip_health": mtf_chip_health_score, # 使用调制后的MTF版本
            "winner_profit_margin_avg": mtf_winner_profit_margin_avg,
            "loser_loss_margin_avg_inverse": mtf_loser_loss_margin_avg * -1,
            "winner_concentration_90pct": mtf_winner_concentration_90pct, # 使用调制后的MTF版本
            "chip_fatigue_inverse": mtf_chip_fatigue_index * -1, # 使用调制后的MTF版本
            "cost_gini_coefficient_inverse": mtf_cost_gini_coefficient * -1, # 使用调制后的MTF版本
            "winner_stability_trend_consistency": mtf_trend_consistency_winner_stability # 新增趋势一致性
        }
        # 融合权重
        conviction_fusion_weights = {
            "core_conviction": 0.4,
            "main_force_conviction": conviction_enhancement_weights.get("main_force_conviction", 0.2),
            "chip_health": conviction_enhancement_weights.get("chip_health", 0.2),
            "winner_profit_margin_avg": conviction_enhancement_weights.get("winner_profit_margin_avg", 0.1),
            "loser_loss_margin_avg_inverse": conviction_enhancement_weights.get("loser_loss_margin_avg_inverse", 0.1),
            "winner_concentration_90pct": conviction_enhancement_weights.get("winner_concentration_90pct", 0.1),
            "chip_fatigue_inverse": conviction_enhancement_weights.get("chip_fatigue_inverse", 0.1),
            "cost_gini_coefficient_inverse": conviction_enhancement_weights.get("cost_gini_coefficient_inverse", 0.05),
            "winner_stability_trend_consistency": conviction_enhancement_weights.get("winner_stability_trend_consistency", 0.1) # 新增权重
        }
        total_weight = sum(conviction_fusion_weights.values())
        if total_weight > 0:
            conviction_fusion_weights = {k: v / total_weight for k, v in conviction_fusion_weights.items()}
        else:
            conviction_fusion_weights = {k: 1/len(conviction_fusion_weights) for k in conviction_fusion_weights.keys()}
        fused_conviction_strength_0_1 = _robust_geometric_mean(
            all_conviction_components,
            conviction_fusion_weights,
            df_index
        )
        conviction_strength_score = (fused_conviction_strength_0_1 * 2 - 1).clip(-1, 1)
        _temp_debug_values["信念强度"] = {
            "mtf_winner_stability": mtf_winner_stability,
            "winner_stability_percentile": winner_stability_percentile,
            "core_conviction_component": core_conviction_component,
            "mtf_main_force_conviction_index": mtf_main_force_conviction_index,
            "mtf_chip_health_score": mtf_chip_health_score,
            "mtf_winner_profit_margin_avg": mtf_winner_profit_margin_avg,
            "mtf_loser_loss_margin_avg": mtf_loser_loss_margin_avg,
            "mtf_winner_concentration_90pct": mtf_winner_concentration_90pct,
            "mtf_chip_fatigue_index": mtf_chip_fatigue_index,
            "mtf_cost_gini_coefficient": mtf_cost_gini_coefficient,
            "mtf_trend_consistency_winner_stability": mtf_trend_consistency_winner_stability, # 调试输出
            "inflection_mtf_winner_stability_index": inflection_mtf_winner_stability, # 调试输出
            "inflection_penalty_conviction": inflection_penalty, # 调试输出
            "fused_conviction_strength_0_1": fused_conviction_strength_0_1,
            "conviction_strength_score": conviction_strength_score
        }
        return conviction_strength_score

    def _calculate_pressure_resilience(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.8 · 压力韧性累积上下文、趋势一致性与拐点调制版 - 压力信号更新】计算压力韧性。
        融入了更多MTF和归一化信号，修正了键名，新增累积上下文调制、趋势一致性增强和拐点惩罚。
        核心修正：将 `profit_taking_flow_ratio` 替换为 `dispersal_by_distribution` 作为核心压力信号。
                  引入 `distribution_at_peak_intensity` 和 `upper_shadow_selling_pressure` 作为增强压力信号。
                  根据业务逻辑，负的 `main_force_buy_execution_alpha` 代表主力积极买入推高股价，应正面增强压力韧性，
                  因此在融合时对其进行符号反转。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            signals (Dict[str, pd.Series]): 包含原始信号Series和MTF信号Series的字典。
            normalized_signals (Dict[str, pd.Series]): 包含归一化信号Series的字典。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 压力韧性分数。
        """
        relative_position_weights = params["relative_position_weights"]
        pressure_resilience_enhancement_weights = params["pressure_resilience_enhancement_weights"]
        cumulative_modulation_strength = params["cumulative_flow_params"]["cumulative_modulation_strength"]
        inflection_penalty_strength = params["inflection_point_params"]["inflection_penalty_strength"]
        # 核心压力信号MTF分数 (使用新的压力信号：dispersal_by_distribution)
        mtf_dispersal_by_distribution = signals["mtf_dispersal_by_distribution"]
        # 修正：使用新的压力信号的原始值计算百分位
        dispersal_by_distribution_percentile = (1 - signals["dispersal_by_distribution_raw"].rank(pct=True)).fillna(0.5)
        # 核心韧性组件：高派发/卖压是负面，所以 mtf_dispersal_by_distribution 乘以 -1，百分位也反转
        core_resilience_component = ((mtf_dispersal_by_distribution * -1) * relative_position_weights.get("selling_pressure_low", 0.4) +
                                     (dispersal_by_distribution_percentile * 2 - 1) * (1 - relative_position_weights.get("selling_pressure_low", 0.4)))
        # 获取MTF信号
        mtf_main_force_buy_execution_alpha = signals["mtf_main_force_buy_execution_alpha"]
        mtf_bid_side_liquidity = signals["mtf_bid_side_liquidity"]
        mtf_absorption_strength_ma5 = signals["mtf_absorption_strength_ma5"]
        mtf_active_buying_support = signals["mtf_active_buying_support"]
        mtf_large_order_support = signals["mtf_large_order_support"]
        mtf_distribution_at_peak_intensity = signals["mtf_distribution_at_peak_intensity"] # 新增
        mtf_upper_shadow_selling_pressure = signals["mtf_upper_shadow_selling_pressure"] # 新增

        # 对相关MTF信号进行累积上下文调制
        if "cumulative_main_force_buy_execution_alpha_score" in signals:
            cumulative_score = signals["cumulative_main_force_buy_execution_alpha_score"]
            # 调制时，保持原始符号，因为调制是基于原始信号的相对变化
            mtf_main_force_buy_execution_alpha = mtf_main_force_buy_execution_alpha + (cumulative_score - mtf_main_force_buy_execution_alpha) * cumulative_modulation_strength
            mtf_main_force_buy_execution_alpha = mtf_main_force_buy_execution_alpha.clip(-1, 1)
        if "cumulative_bid_side_liquidity_score" in signals:
            cumulative_score = signals["cumulative_bid_side_liquidity_score"]
            mtf_bid_side_liquidity = mtf_bid_side_liquidity + (cumulative_score - mtf_bid_side_liquidity) * cumulative_modulation_strength
            mtf_bid_side_liquidity = mtf_bid_side_liquidity.clip(-1, 1)
        if "cumulative_absorption_strength_ma5_score" in signals:
            cumulative_score = signals["cumulative_absorption_strength_ma5_score"]
            mtf_absorption_strength_ma5 = mtf_absorption_strength_ma5 + (cumulative_score - mtf_absorption_strength_ma5) * cumulative_modulation_strength
            mtf_absorption_strength_ma5 = mtf_absorption_strength_ma5.clip(-1, 1)
        if "cumulative_active_buying_support_score" in signals:
            cumulative_score = signals["cumulative_active_buying_support_score"]
            mtf_active_buying_support = mtf_active_buying_support + (cumulative_score - mtf_active_buying_support) * cumulative_modulation_strength
            mtf_active_buying_support = mtf_active_buying_support.clip(-1, 1)
        if "cumulative_large_order_support_score" in signals:
            cumulative_score = signals["cumulative_large_order_support_score"]
            mtf_large_order_support = mtf_large_order_support + (cumulative_score - mtf_large_order_support) * cumulative_modulation_strength
            mtf_large_order_support = mtf_large_order_support.clip(-1, 1)
        if "cumulative_dip_absorption_power_score" in signals:
            cumulative_score = signals["cumulative_dip_absorption_power_score"]
            normalized_signals["dip_absorption_power_norm"] = normalized_signals["dip_absorption_power_norm"] + (cumulative_score - normalized_signals["dip_absorption_power_norm"]) * cumulative_modulation_strength
            normalized_signals["dip_absorption_power_norm"] = normalized_signals["dip_absorption_power_norm"].clip(-1, 1)
        # 新增：对新的核心压力信号进行累积上下文调制
        if "cumulative_dispersal_by_distribution_score" in signals:
            cumulative_score = signals["cumulative_dispersal_by_distribution_score"]
            mtf_dispersal_by_distribution = mtf_dispersal_by_distribution + (cumulative_score - mtf_dispersal_by_distribution) * cumulative_modulation_strength
            mtf_dispersal_by_distribution = mtf_dispersal_by_distribution.clip(-1, 1)
        if "cumulative_distribution_at_peak_intensity_score" in signals:
            cumulative_score = signals["cumulative_distribution_at_peak_intensity_score"]
            mtf_distribution_at_peak_intensity = mtf_distribution_at_peak_intensity + (cumulative_score - mtf_distribution_at_peak_intensity) * cumulative_modulation_strength
            mtf_distribution_at_peak_intensity = mtf_distribution_at_peak_intensity.clip(-1, 1)
        if "cumulative_upper_shadow_selling_pressure_score" in signals:
            cumulative_score = signals["cumulative_upper_shadow_selling_pressure_score"]
            mtf_upper_shadow_selling_pressure = mtf_upper_shadow_selling_pressure + (cumulative_score - mtf_upper_shadow_selling_pressure) * cumulative_modulation_strength
            mtf_upper_shadow_selling_pressure = mtf_upper_shadow_selling_pressure.clip(-1, 1)

        # 引入MTF趋势一致性分数 (使用新的压力信号)
        mtf_trend_consistency_selling_pressure = signals.get("mtf_trend_consistency_dispersal_by_distribution", pd.Series(0.0, index=df_index))
        # 引入拐点惩罚 (使用新的压力信号)
        inflection_mtf_selling_pressure = signals.get("inflection_mtf_dispersal_by_distribution", pd.Series(0.0, index=df_index))
        inflection_penalty = pd.Series(0.0, index=df_index, dtype=np.float32)
        # 当MTF分数 > 0 (压力增加) 且 inflection_score > 0 (压力增加趋势减弱或压力减少趋势增强) -> 奖励 (减少惩罚)
        # 当MTF分数 < 0 (压力减少) 且 inflection_score < 0 (压力减少趋势减弱或压力增加趋势增强) -> 惩罚 (增加惩罚)
        # 这里的逻辑需要反转，因为压力信号是负面指标
        # 如果压力信号MTF为正（压力增加），且拐点信号为正（压力增加趋势减弱），则为奖励
        inflection_penalty = inflection_penalty.mask((mtf_dispersal_by_distribution > 0) & (inflection_mtf_selling_pressure > 0), inflection_mtf_selling_pressure * -1 * inflection_penalty_strength) # 压力增加趋势减弱是好事，所以是奖励
        # 如果压力信号MTF为负（压力减少），且拐点信号为负（压力减少趋势减弱），则为惩罚
        inflection_penalty = inflection_penalty.mask((mtf_dispersal_by_distribution < 0) & (inflection_mtf_selling_pressure < 0), inflection_mtf_selling_pressure * -1 * inflection_penalty_strength) # 压力减少趋势减弱是坏事，所以是惩罚
        # 将拐点惩罚应用于核心压力韧性组件
        core_resilience_component = core_resilience_component + inflection_penalty # 惩罚是负值，奖励是正值，所以用加法

        all_resilience_components = {
            "core_resilience": core_resilience_component,
            # 核心修正：将 mtf_main_force_buy_execution_alpha 反转符号，使其负值（积极买入）贡献正面效应
            "main_force_buy_execution_alpha": mtf_main_force_buy_execution_alpha * -1,
            "bid_side_liquidity": mtf_bid_side_liquidity, # 使用调制后的MTF版本
            "absorption_strength_ma5": mtf_absorption_strength_ma5, # 使用调制后的MTF版本
            "active_buying_support": mtf_active_buying_support, # 使用调制后的MTF版本
            "large_order_support": mtf_large_order_support, # 使用调制后的MTF版本
            "dip_absorption_power": normalized_signals["dip_absorption_power_norm"], # 使用调制后的归一化原始值
            "selling_pressure_trend_consistency": mtf_trend_consistency_selling_pressure * -1, # 修正：趋势一致性越高，压力越大，对韧性是负面
            "distribution_at_peak_intensity": mtf_distribution_at_peak_intensity * -1, # 新增，高派发强度对韧性是负面
            "upper_shadow_selling_pressure": mtf_upper_shadow_selling_pressure * -1 # 新增，高上影线压力对韧性是负面
        }
        # 融合权重
        resilience_fusion_weights = {
            "core_resilience": 0.4,
            "main_force_buy_execution_alpha": pressure_resilience_enhancement_weights.get("main_force_buy_execution_alpha", 0.2),
            "bid_side_liquidity": pressure_resilience_enhancement_weights.get("bid_side_liquidity", 0.2),
            "absorption_strength_ma5": pressure_resilience_enhancement_weights.get("absorption_strength_ma5", 0.2),
            "active_buying_support": pressure_resilience_enhancement_weights.get("active_buying_support", 0.15),
            "large_order_support": pressure_resilience_enhancement_weights.get("large_order_support", 0.15),
            "dip_absorption_power": pressure_resilience_enhancement_weights.get("dip_absorption_power", 0.1),
            "selling_pressure_trend_consistency": pressure_resilience_enhancement_weights.get("selling_pressure_trend_consistency", 0.1), # 修正
            "distribution_at_peak_intensity": pressure_resilience_enhancement_weights.get("distribution_at_peak_intensity", 0.1), # 新增
            "upper_shadow_selling_pressure": pressure_resilience_enhancement_weights.get("upper_shadow_selling_pressure", 0.1) # 新增
        }
        total_weight = sum(resilience_fusion_weights.values())
        if total_weight > 0:
            resilience_fusion_weights = {k: v / total_weight for k, v in resilience_fusion_weights.items()}
        else:
            resilience_fusion_weights = {k: 1/len(resilience_fusion_weights) for k in resilience_fusion_weights.keys()}
        fused_pressure_resilience_0_1 = _robust_geometric_mean(
            all_resilience_components,
            resilience_fusion_weights,
            df_index
        )
        pressure_resilience_score = (fused_pressure_resilience_0_1 * 2 - 1).clip(-1, 1)
        _temp_debug_values["压力韧性"] = {
            "mtf_dispersal_by_distribution": mtf_dispersal_by_distribution, # 修正
            "dispersal_by_distribution_percentile": dispersal_by_distribution_percentile, # 修正
            "core_resilience_component": core_resilience_component,
            "mtf_main_force_buy_execution_alpha": mtf_main_force_buy_execution_alpha, # 调试输出原始调制后的值
            "mtf_main_force_buy_execution_alpha_inverted_for_fusion": mtf_main_force_buy_execution_alpha * -1, # 调试输出融合前反转的值
            "mtf_bid_side_liquidity": mtf_bid_side_liquidity,
            "mtf_absorption_strength_ma5": mtf_absorption_strength_ma5,
            "mtf_active_buying_support": mtf_active_buying_support,
            "mtf_large_order_support": mtf_large_order_support,
            "dip_absorption_power_norm": normalized_signals["dip_absorption_power_norm"],
            "mtf_trend_consistency_selling_pressure": mtf_trend_consistency_selling_pressure, # 修正
            "inflection_mtf_selling_pressure": inflection_mtf_selling_pressure, # 修正
            "inflection_penalty_pressure": inflection_penalty, # 调试输出
            "mtf_distribution_at_peak_intensity": mtf_distribution_at_peak_intensity, # 新增
            "mtf_upper_shadow_selling_pressure": mtf_upper_shadow_selling_pressure, # 新增
            "fused_pressure_resilience_0_1": fused_pressure_resilience_0_1,
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

    def _calculate_deception_filter(self, df: pd.DataFrame, df_index: pd.Index, method_name: str, signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.4 · 诡道过滤累积上下文调制版】计算诡道过滤因子，融入了更多MTF和归一化信号，新增累积上下文调制。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            signals (Dict[str, pd.Series]): 包含原始信号Series和MTF信号Series的字典。
            normalized_signals (Dict[str, pd.Series]): 包含归一化信号Series的字典。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 诡道过滤因子分数。
        """
        deception_enhancement_weights = params["deception_enhancement_weights"]
        cumulative_modulation_strength = params["cumulative_flow_params"]["cumulative_modulation_strength"]
        # 核心欺骗指数和对倒强度MTF分数
        mtf_deception_index = signals["mtf_deception_index"]
        mtf_wash_trade_intensity = signals["mtf_wash_trade_intensity"]
        # 聪明钱分歧越大，欺骗惩罚越大 (mtf_smart_money_divergence_hm_buy_inst_sell 是 [0, 1]，高分歧 -> 高值)
        mtf_smart_money_divergence_hm_buy_inst_sell = signals["mtf_smart_money_divergence_hm_buy_inst_sell"]
        # 隐蔽吸筹，MTF分数越高代表隐蔽吸筹越强，惩罚越高
        mtf_covert_accumulation_signal = signals["mtf_covert_accumulation_signal"]
        # 集合竞价伏击，MTF分数越高代表伏击越强，惩罚越高
        mtf_closing_auction_ambush = signals["mtf_closing_auction_ambush"]
        # 对相关MTF信号进行累积上下文调制
        if "cumulative_covert_accumulation_signal_score" in signals:
            cumulative_score = signals["cumulative_covert_accumulation_signal_score"]
            mtf_covert_accumulation_signal = mtf_covert_accumulation_signal + (cumulative_score - mtf_covert_accumulation_signal) * cumulative_modulation_strength
            mtf_covert_accumulation_signal = mtf_covert_accumulation_signal.clip(-1, 1)
        if "cumulative_closing_auction_ambush_score" in signals:
            cumulative_score = signals["cumulative_closing_auction_ambush_score"]
            mtf_closing_auction_ambush = mtf_closing_auction_ambush + (cumulative_score - mtf_closing_auction_ambush) * cumulative_modulation_strength
            mtf_closing_auction_ambush = mtf_closing_auction_ambush.clip(-1, 1)
        # 基础欺骗惩罚 (范围 [0, 1])
        base_deception_penalty = (mtf_deception_index * 0.6 + mtf_wash_trade_intensity * 0.4).clip(0, 1)
        # 融合所有欺骗相关因子
        all_deception_components = {
            "base_deception_penalty": base_deception_penalty,
            "smart_money_divergence_penalty": mtf_smart_money_divergence_hm_buy_inst_sell, # 暂不调制，因为其含义是分歧，而非直接的累积流
            "covert_accumulation_penalty": mtf_covert_accumulation_signal, # 使用调制后的MTF版本
            "closing_auction_ambush_penalty": mtf_closing_auction_ambush # 使用调制后的MTF版本
        }
        deception_fusion_weights = {
            "base_deception_penalty": 0.7,
            "smart_money_divergence_penalty": deception_enhancement_weights.get("smart_money_divergence", 0.3),
            "covert_accumulation_penalty": deception_enhancement_weights.get("covert_accumulation_inverse", 0.2),
            "closing_auction_ambush_penalty": deception_enhancement_weights.get("closing_auction_ambush_inverse", 0.1)
        }
        total_weight = sum(deception_fusion_weights.values())
        if total_weight > 0:
            deception_fusion_weights = {k: v / total_weight for k, v in deception_fusion_weights.items()}
        else:
            deception_fusion_weights = {k: 1/len(deception_fusion_weights) for k in deception_fusion_weights.keys()}
        fused_deception_penalty_0_1 = _robust_geometric_mean(
            all_deception_components,
            deception_fusion_weights,
            df_index
        )
        deception_filter = (1 - fused_deception_penalty_0_1).clip(0, 1)
        _temp_debug_values["诡道过滤"] = {
            "mtf_deception_index": mtf_deception_index,
            "mtf_wash_trade_intensity": mtf_wash_trade_intensity,
            "base_deception_penalty": base_deception_penalty,
            "mtf_smart_money_divergence_hm_buy_inst_sell": mtf_smart_money_divergence_hm_buy_inst_sell,
            "mtf_covert_accumulation_signal": mtf_covert_accumulation_signal, # 调试输出调制后的值
            "mtf_closing_auction_ambush": mtf_closing_auction_ambush, # 调试输出调制后的值
            "fused_deception_penalty_0_1": fused_deception_penalty_0_1,
            "deception_filter": deception_filter
        }
        return deception_filter

    def _calculate_contextual_modulator(self, df_index: pd.Index, signals: Dict[str, pd.Series], normalized_signals: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V1.5 · 情境调制累积上下文调制版】计算情境调制因子，融入了更多MTF和归一化信号，并修正了键名，新增累积上下文调制。
        参数:
            df_index (pd.Index): DataFrame的索引。
            signals (Dict[str, pd.Series]): 包含原始信号Series和MTF信号Series的字典。
            normalized_signals (Dict[str, pd.Series]): 包含归一化信号Series的字典。
            params (Dict): 包含所有参数的字典。
            _temp_debug_values (Dict): 临时调试值字典。
        返回:
            pd.Series: 情境调制因子分数。
        """
        context_modulator_weights = params["context_modulator_weights"]
        context_modulator_enhancement_weights = params["context_modulator_enhancement_weights"]
        cumulative_modulation_strength = params["cumulative_flow_params"]["cumulative_modulation_strength"]
        norm_market_sentiment = self.helper._normalize_series(signals["market_sentiment_score_raw"], df_index, bipolar=True)
        volatility_stability_raw = 1 - normalize_score(signals["volatility_instability_index_21d_raw"], df_index, 21, ascending=True, debug_info=False)
        norm_volatility_stability = self.helper._normalize_series(volatility_stability_raw, df_index, bipolar=False, ascending=True)
        norm_trend_vitality = self.helper._normalize_series(signals["trend_vitality_index_raw"], df_index, bipolar=False)
        norm_theme_hotness = normalized_signals["theme_hotness_norm"]
        norm_industry_leader_score = normalized_signals["industry_leader_score_norm"]
        # 市场冲击成本MTF分数，MTF分数越高代表冲击成本越高，是负面影响
        mtf_market_impact_cost = signals["mtf_market_impact_cost"]
        # 对相关MTF信号进行累积上下文调制
        if "cumulative_market_impact_cost_score" in signals:
            cumulative_score = signals["cumulative_market_impact_cost_score"]
            mtf_market_impact_cost = mtf_market_impact_cost + (cumulative_score - mtf_market_impact_cost) * cumulative_modulation_strength
            mtf_market_impact_cost = mtf_market_impact_cost.clip(-1, 1)
        # 所有情境调制组件
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_stability": norm_volatility_stability,
            "trend_vitality": norm_trend_vitality,
            "theme_hotness": norm_theme_hotness,
            "industry_leader_score": norm_industry_leader_score,
            "market_impact_cost_inverse": mtf_market_impact_cost * -1 # 使用调制后的MTF版本
        }
        # 融合权重
        context_fusion_weights = {
            "market_sentiment": context_modulator_weights.get("market_sentiment", 0.4),
            "volatility_stability": context_modulator_weights.get("volatility_stability", 0.3),
            "trend_vitality": context_modulator_weights.get("trend_vitality", 0.3),
            "theme_hotness": context_modulator_enhancement_weights.get("theme_hotness", 0.2),
            "industry_leader_score": context_modulator_enhancement_weights.get("industry_leader_score", 0.1),
            "market_impact_cost_inverse": context_modulator_enhancement_weights.get("market_impact_cost_inverse", 0.05)
        }
        total_weight = sum(context_fusion_weights.values())
        if total_weight > 0:
            context_fusion_weights = {k: v / total_weight for k, v in context_fusion_weights.items()}
        else:
            context_fusion_weights = {k: 1/len(context_fusion_weights) for k in context_fusion_weights.keys()}
        context_modulator_score_0_1 = _robust_geometric_mean(
            context_modulator_components,
            context_fusion_weights,
            df_index
        )
        context_modulator = 0.5 + context_modulator_score_0_1
        _temp_debug_values["情境调制"] = {
            "norm_market_sentiment": norm_market_sentiment,
            "volatility_stability_raw": volatility_stability_raw,
            "norm_volatility_stability": norm_volatility_stability,
            "norm_trend_vitality": norm_trend_vitality,
            "norm_theme_hotness": norm_theme_hotness,
            "norm_industry_leader_score": norm_industry_leader_score,
            "mtf_market_impact_cost": mtf_market_impact_cost, # 调试输出调制后的值
            "context_modulator_score_0_1": context_modulator_score_0_1,
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







