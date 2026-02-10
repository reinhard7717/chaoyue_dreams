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
    PROCESS_META_STORM_EYE_CALM
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

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        V4.0.9: 计算“风暴眼中的寂静”信号。
        """
        method_name = "calculate_storm_eye_calm"
        is_debug_enabled_for_method, probe_ts = self._get_debug_info(df, method_name)
        debug_output = {}
        _temp_debug_values = {
            "能量压缩": {},
            "量能枯竭": {},
            "主力隐蔽意图": {}, # 用于存储组件
            "主力隐蔽意图融合": {}, # 用于存储融合分数
            "市场情绪低迷融合": {},
            "突破准备度融合": {},
            "市场情境动态调节器": {},
            "最终融合": {}
        }
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
        energy_compression_score = self._calculate_energy_compression_component(df_index, raw_data, mtf_derived_scores, params['energy_compression_weights'], _temp_debug_values)
        _temp_debug_values["能量压缩"]["energy_compression_score"] = energy_compression_score
        volume_exhaustion_score = self._calculate_volume_exhaustion_component(df_index, raw_data, mtf_derived_scores, params['volume_exhaustion_weights'], _temp_debug_values)
        _temp_debug_values["量能枯竭"]["volume_exhaustion_score"] = volume_exhaustion_score
        main_force_covert_intent_score, main_force_covert_intent_components = self._calculate_main_force_covert_intent_component(df_index, raw_data, mtf_derived_scores, params['main_force_covert_intent_weights'], params['ambiguity_components_weights'], _temp_debug_values)
        _temp_debug_values["主力隐蔽意图"].update(main_force_covert_intent_components) # 存储组件
        _temp_debug_values["主力隐蔽意图融合"]["main_force_covert_intent_score"] = main_force_covert_intent_score # 存储融合分数
        subdued_market_sentiment_score = self._calculate_subdued_market_sentiment_component(df_index, raw_data, params['subdued_market_sentiment_weights'], params['sentiment_volatility_window'], params['long_term_sentiment_window'], params['sentiment_neutral_range'], params['sentiment_pendulum_neutral_range'], _temp_debug_values)
        _temp_debug_values["市场情绪低迷融合"]["subdued_market_sentiment_score"] = subdued_market_sentiment_score
        breakout_readiness_score = self._calculate_breakout_readiness_component(df_index, raw_data, params['breakout_readiness_weights'], _temp_debug_values)
        _temp_debug_values["突破准备度融合"]["breakout_readiness_score"] = breakout_readiness_score
        # 6. 市场情境动态调节器
        market_regime_modulator = self._calculate_market_regulator_modulator(df_index, raw_data, params, _temp_debug_values)
        _temp_debug_values["市场情境动态调节器"]["market_regime_modulator"] = market_regime_modulator
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
        _temp_debug_values["最终融合"]["adjusted_final_fusion_weights"] = adjusted_final_fusion_weights # 记录调整后的权重
        final_score = self._perform_final_fusion(df_index, component_scores, adjusted_final_fusion_weights, params['price_calmness_modulator_params'], params['main_force_control_adjudicator_params'], raw_data, _temp_debug_values)
        _temp_debug_values["最终融合"]["final_score"] = final_score
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     self._print_debug_output_for_storm_eye_calm(debug_output, _temp_debug_values, probe_ts, method_name, final_score)
        return final_score.astype(np.float32)

    def _get_debug_info(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp]]:
        """
        V1.0: 集中获取调试信息（是否启用调试、探针日期等）。
        """
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        return is_debug_enabled_for_method, probe_ts

    def _print_debug_output_for_storm_eye_calm(self, debug_output: Dict, _temp_debug_values: Dict, probe_ts: pd.Timestamp, method_name: str, final_score: pd.Series):
        """
        V1.4: 统一打印 _calculate_storm_eye_calm 方法的调试信息。
        """
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        for key, value in _temp_debug_values["原始信号值"].items():
            if isinstance(value, pd.Series):
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            else: # Handle non-Series values like dicts or raw numbers
                debug_output[f"        '{key}': {value}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF斜率/加速度分数 ---"] = ""
        for key, series in _temp_debug_values["MTF斜率/加速度分数"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 能量压缩 (组件) ---"] = ""
        for key, series in _temp_debug_values["能量压缩"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 量能枯竭 (组件) ---"] = ""
        for key, series in _temp_debug_values["量能枯竭"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力隐蔽意图 (组件) ---"] = ""
        for key, series in _temp_debug_values["主力隐蔽意图"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"          {key}: {val:.4f}"] = ""
            else:
                debug_output[f"          {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力隐蔽意图融合 ---"] = ""
        for key, series in _temp_debug_values["主力隐蔽意图融合"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场情绪低迷融合 (组件) ---"] = ""
        for key, series in _temp_debug_values["市场情绪低迷融合"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 突破准备度融合 (组件) ---"] = ""
        for key, series in _temp_debug_values["突破准备度融合"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场情境动态调节器 (组件) ---"] = ""
        for key, series in _temp_debug_values["市场情境动态调节器"].items():
            if isinstance(series, pd.Series):
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            else:
                debug_output[f"        {key}: {series}"] = ""
        debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合 (组件) ---"] = ""
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
        """
        V1.0: 从配置中获取所有与“风暴眼中的寂静”相关的参数。
        """
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

    def _get_raw_and_atomic_data(self, df: pd.DataFrame, method_name: str, params: Dict) -> Dict[str, pd.Series]:
        """
        V6.0: 从DataFrame提取并在逻辑上映射原始数据。
        版本说明：
            - 实现 HAB (Historical Accumulation Buffer) 的 13/21/34 周期累积计算。
            - 提取所有动力学指标 (Slope, Accel, Jerk) 并规范化命名。
            - 提取调节器所需的新核心指标 (ATR, ATR Jerk)。
        """
        raw_data = {}
        # --- [Energy Compression] 能量压缩 ---
        raw_data['ma_potential_tension_raw'] = self.helper._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', np.nan, method_name=method_name)
        raw_data['bbw_raw'] = self.helper._get_safe_series(df, 'BBW_21_2.0_D', np.nan, method_name=method_name)
        raw_data['chip_stability_raw'] = self.helper._get_safe_series(df, 'chip_stability_D', np.nan, method_name=method_name)
        raw_data['chip_concentration_raw'] = self.helper._get_safe_series(df, 'chip_concentration_ratio_D', np.nan, method_name=method_name)
        raw_data['chip_convergence_raw'] = self.helper._get_safe_series(df, 'chip_convergence_ratio_D', np.nan, method_name=method_name)
        raw_data['price_entropy_raw'] = self.helper._get_safe_series(df, 'PRICE_ENTROPY_D', np.nan, method_name=method_name)
        raw_data['price_fractal_raw'] = self.helper._get_safe_series(df, 'PRICE_FRACTAL_DIM_D', np.nan, method_name=method_name)
        raw_data['chip_skewness_raw'] = self.helper._get_safe_series(df, 'chip_skewness_D', np.nan, method_name=method_name)
        raw_data['bias_21_raw'] = self.helper._get_safe_series(df, 'BIAS_21_D', np.nan, method_name=method_name)
        raw_data['ma_compression_raw'] = self.helper._get_safe_series(df, 'MA_POTENTIAL_COMPRESSION_RATE_D', np.nan, method_name=method_name)
        raw_data['concentration_entropy_raw'] = self.helper._get_safe_series(df, 'concentration_entropy_D', np.nan, method_name=method_name)
        raw_data['chip_stability_change_raw'] = self.helper._get_safe_series(df, 'chip_stability_change_5d_D', np.nan, method_name=method_name)
        # --- [Volume Exhaustion] 量能枯竭 ---
        raw_data['volume_raw'] = self.helper._get_safe_series(df, 'volume_D', np.nan, method_name=method_name)
        raw_data['turnover_rate_f_raw'] = self.helper._get_safe_series(df, 'turnover_rate_f_D', np.nan, method_name=method_name)
        raw_data['vpa_efficiency_raw'] = self.helper._get_safe_series(df, 'VPA_EFFICIENCY_D', np.nan, method_name=method_name)
        raw_data['turnover_stability_raw'] = self.helper._get_safe_series(df, 'turnover_stability_index_D', np.nan, method_name=method_name)
        raw_data['flow_impact_raw'] = self.helper._get_safe_series(df, 'flow_impact_ratio_D', np.nan, method_name=method_name)
        raw_data['flow_efficiency_raw'] = self.helper._get_safe_series(df, 'flow_efficiency_D', np.nan, method_name=method_name)
        raw_data['pressure_release_raw'] = self.helper._get_safe_series(df, 'pressure_release_index_D', np.nan, method_name=method_name)
        raw_data['tick_chip_flow_raw'] = self.helper._get_safe_series(df, 'tick_level_chip_flow_D', np.nan, method_name=method_name)
        raw_data['tick_abnormal_vol_raw'] = self.helper._get_safe_series(df, 'tick_abnormal_volume_ratio_D', np.nan, method_name=method_name)
        raw_data['afternoon_flow_ratio_raw'] = self.helper._get_safe_series(df, 'afternoon_flow_ratio_D', np.nan, method_name=method_name)
        # --- [Covert Intent] 主力隐蔽意图 ---
        raw_data['stealth_flow_ratio_raw'] = self.helper._get_safe_series(df, 'stealth_flow_ratio_D', np.nan, method_name=method_name)
        raw_data['mf_activity_raw'] = self.helper._get_safe_series(df, 'main_force_activity_index_D', np.nan, method_name=method_name)
        raw_data['net_mf_amount_raw'] = self.helper._get_safe_series(df, 'net_mf_amount_D', np.nan, method_name=method_name)
        raw_data['cost_advantage_raw'] = self.helper._get_safe_series(df, 'chip_cost_to_ma21_diff_D', np.nan, method_name=method_name)
        raw_data['tick_transfer_eff_raw'] = self.helper._get_safe_series(df, 'tick_chip_transfer_efficiency_D', np.nan, method_name=method_name)
        raw_data['buy_lg_rate_raw'] = self.helper._get_safe_series(df, 'buy_lg_amount_rate_D', np.nan, method_name=method_name)
        raw_data['tick_clustering_raw'] = self.helper._get_safe_series(df, 'tick_clustering_index_D', np.nan, method_name=method_name)
        raw_data['sm_divergence_raw'] = self.helper._get_safe_series(df, 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D', np.nan, method_name=method_name)
        # --- [HAB Data] 历史累积记忆 ---
        # 实时计算 13, 21, 34 日主力净买入累积额
        raw_data['net_mf_sum_13'] = raw_data['net_mf_amount_raw'].rolling(window=13, min_periods=1).sum()
        raw_data['net_mf_sum_21'] = raw_data['net_mf_amount_raw'].rolling(window=21, min_periods=1).sum()
        raw_data['net_mf_sum_34'] = raw_data['net_mf_amount_raw'].rolling(window=34, min_periods=1).sum()
        # --- [Subdued Sentiment] 市场情绪低迷 ---
        raw_data['market_sentiment_raw'] = self.helper._get_safe_series(df, 'market_sentiment_score_D', np.nan, method_name=method_name)
        raw_data['pressure_trapped_raw'] = self.helper._get_safe_series(df, 'pressure_trapped_D', np.nan, method_name=method_name)
        raw_data['profit_ratio_raw'] = self.helper._get_safe_series(df, 'profit_ratio_D', np.nan, method_name=method_name)
        raw_data['retail_buy_rate_raw'] = self.helper._get_safe_series(df, 'buy_sm_amount_rate_D', np.nan, method_name=method_name)
        raw_data['retail_sell_rate_raw'] = self.helper._get_safe_series(df, 'sell_sm_amount_rate_D', np.nan, method_name=method_name)
        raw_data['adx_raw'] = self.helper._get_safe_series(df, 'ADX_14_D', np.nan, method_name=method_name)
        raw_data['trend_confirm_raw'] = self.helper._get_safe_series(df, 'trend_confirmation_score_D', np.nan, method_name=method_name)
        raw_data['loser_pain_raw'] = self.helper._get_safe_series(df, 'loser_pain_index_D', np.nan, method_name=method_name)
        raw_data['winner_rate_raw'] = self.helper._get_safe_series(df, 'winner_rate_D', np.nan, method_name=method_name)
        # --- [Breakout Readiness] 突破准备度 ---
        raw_data['breakout_potential_raw'] = self.helper._get_safe_series(df, 'breakout_potential_D', np.nan, method_name=method_name)
        raw_data['breakout_confidence_raw'] = self.helper._get_safe_series(df, 'breakout_confidence_D', np.nan, method_name=method_name)
        raw_data['breakout_penalty_raw'] = self.helper._get_safe_series(df, 'breakout_penalty_score_D', np.nan, method_name=method_name)
        raw_data['resistance_strength_raw'] = self.helper._get_safe_series(df, 'resistance_strength_D', np.nan, method_name=method_name)
        raw_data['is_consolidating_raw'] = self.helper._get_safe_series(df, 'is_consolidating_D', np.nan, method_name=method_name)
        raw_data['consolidation_duration_raw'] = self.helper._get_safe_series(df, 'dynamic_consolidation_duration_D', np.nan, method_name=method_name)
        raw_data['geom_r2_raw'] = self.helper._get_safe_series(df, 'GEOM_REG_R2_D', np.nan, method_name=method_name)
        raw_data['consolidation_quality_raw'] = self.helper._get_safe_series(df, 'consolidation_quality_score_D', np.nan, method_name=method_name)
        raw_data['chip_structure_state_raw'] = self.helper._get_safe_series(df, 'chip_structure_state_D', np.nan, method_name=method_name)
        # --- [Regulator] 调节器 (新增) ---
        raw_data['atr_raw'] = self.helper._get_safe_series(df, 'ATR_14_D', np.nan, method_name=method_name)
        # --- [Modulators] 其他调节 ---
        raw_data['price_slope_raw'] = self.helper._get_safe_series(df, f'SLOPE_5_close_D', np.nan, method_name=method_name)
        raw_data['pct_change_raw'] = self.helper._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        # --- [Dynamics] 高阶动力学数据提取 (Slope/Accel/Jerk) ---
        fib_windows = {'SLOPE': '13', 'ACCEL': '8', 'JERK': '5'}
        dynamics_targets = [
            'MA_POTENTIAL_COMPRESSION_RATE_D', 'chip_concentration_ratio_D', 'PRICE_ENTROPY_D',
            'stealth_flow_ratio_D', 'turnover_rate_f_D', 'market_sentiment_score_D',
            'breakout_potential_D', 'ATR_14_D'
        ]
        for target in dynamics_targets:
            base_key = target.replace('_D', '') # 简化键名
            raw_data[f'{base_key}_slope'] = self.helper._get_safe_series(df, f"SLOPE_{fib_windows['SLOPE']}_{target}", np.nan, method_name=method_name)
            raw_data[f'{base_key}_accel'] = self.helper._get_safe_series(df, f"ACCEL_{fib_windows['ACCEL']}_{target}", np.nan, method_name=method_name)
            raw_data[f'{base_key}_jerk']  = self.helper._get_safe_series(df, f"JERK_{fib_windows['JERK']}_{target}", np.nan, method_name=method_name)
        # 其他零散的斜率 (用于MTF Cohesion等旧逻辑)
        raw_data['pressure_trapped_slope'] = self.helper._get_safe_series(df, f'SLOPE_5_pressure_trapped_D', np.nan, method_name=method_name)
        return raw_data

    def _calculate_physics_score(self, series: pd.Series, mode: str, sensitivity: float = 1.0, window: int = 55) -> pd.Series:
        """
        [物理归一化引擎]: 针对不同物理状态的专用归一化。
        参数:
            mode:
                - 'limit_low': 极限趋零 (Tanh)。用于检测'死寂' (如成交量、熵)。
                - 'limit_high': 极限饱和 (Sigmoid/Tanh)。用于检测'高压' (如集中度、隐蔽意图)。
                - 'zero_focus': 零度聚焦 (Gaussian)。用于检测'无变化' (如斜率)。
                - 'relative_rank': 相对分位。用于历史纵向对比。
            sensitivity: 敏感度。值越大，对极值的捕捉越敏锐。
        """
        if series.isnull().all():
            return pd.Series(0.0, index=series.index)
        # 1. 极限趋零 (Limit Low): 寻找绝对小值
        if mode == 'limit_low':
            # x越接近0，分越高。
            return 1 - np.tanh(series.abs() * sensitivity)
        # 2. 极限饱和 (Limit High): 寻找绝对大值
        elif mode == 'limit_high':
            # x越大，分越高。使用 Sigmoid 变体或 Tanh。
            # 假设输入已标准化或量级已知。这里使用 Tanh 映射正向无穷。
            return np.tanh(series * sensitivity)
        # 3. 零度聚焦 (Zero Focus): 寻找'不动'的状态
        elif mode == 'zero_focus':
            # 高斯径向基：x=0时为1。
            return np.exp(- (series * sensitivity) ** 2)
        # 4. 相对分位 (Relative Rank)
        elif mode == 'relative_rank':
            roll_min = series.rolling(window=window, min_periods=1).min()
            roll_max = series.rolling(window=window, min_periods=1).max()
            denom = (roll_max - roll_min).replace(0, 1.0)
            return (series - roll_min) / denom
        return pd.Series(0.0, index=series.index)

    def _calculate_mtf_derived_scores(self, df: pd.DataFrame, df_index: pd.Index, mtf_slope_accel_weights: Dict, mtf_cohesion_base_signals: List, method_name: str) -> Dict[str, pd.Series]:
        mtf_derived_scores = {}
        mtf_derived_scores['bbw_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'BBW_21_2.0_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['vol_instability_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['turnover_rate_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'turnover_rate_f_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['mf_net_flow_slope_positive'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['mtf_cohesion_score'] = self.helper._get_mtf_cohesion_score(df, mtf_cohesion_base_signals, mtf_slope_accel_weights, df_index, method_name)
        return mtf_derived_scores

    def _apply_energy_physics_norm(self, series: pd.Series, mode: str, sensitivity: float = 1.0) -> pd.Series:
        """
        [能量专用物理映射]: 专为能量压缩组件设计的物理相变归一化。
        区别于通用归一化，此处强调'绝对物理极限'。
        参数:
            mode:
                - 'limit_high': 极限饱和 (Tanh)。用于检测'高压' (如集中度、压缩率)。
                - 'limit_low': 极限趋零 (InvTanh)。用于检测'死寂' (如BBW、熵)。
                - 'zero_focus': 零度聚焦 (Gaussian)。用于检测'绝对稳态' (如分形稳定性)。
                - 'time_density': 时间密度 (SoftSat)。用于盘整时间积分。
            sensitivity: 敏感度系数。控制物理相变的陡峭程度。
        """
        if series.isnull().all():
            return pd.Series(0.0, index=series.index)
        if mode == 'limit_high':
            # 物理含义: 压力越大越好，直至饱和。
            # Tanh 映射: x * k -> 0~1
            return np.tanh(series * sensitivity).clip(0, 1)
        elif mode == 'limit_low':
            # 物理含义: 杂波越少越好，直至真空。
            # 1 - Tanh(|x| * k)
            return (1.0 - np.tanh(series.abs() * sensitivity)).clip(0, 1)
        elif mode == 'zero_focus':
            # 物理含义: 状态必须锁定在 0 点，任何扰动都是破坏。
            # Gaussian: exp(-(x*k)^2)
            return np.exp(- (series * sensitivity) ** 2)
        elif mode == 'time_density':
            # 物理含义: 时间越长，势能累积越大，但有边际递减。
            # 假设 input 是天数。Sensitivity=0.03 -> 33天达到 Tanh(1)=0.76。
            return np.tanh(series * sensitivity).clip(0, 1)
        return pd.Series(0.0, index=series.index)

    def _calculate_energy_compression_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V4.1 [物理极限高压模型 - 独立归一化版]: 计算能量压缩维度。
        版本说明：
            - 摒弃 helper 通用归一化，使用专用物理映射 (_apply_energy_physics_norm)。
            - 核心逻辑：
                1. 物理空间: 极限饱和的压缩率 + 极限趋零的带宽。
                2. 筹码相变: 极限饱和的集中度 + 极限趋零的熵。
                3. 混沌边缘: 零度聚焦的分形稳定性。
                4. 时空密度: 时间积分效应。
            - 拒绝 HAB：能量状态是瞬时相变，不可缓冲。
        """
        # --- 1. 物理空间压缩 (Physical Compression - Saturation) ---
        # 均线潜在压缩率: Limit High。越高越好。
        # Sensitivity=5.0: 假设 raw 约为 0.x，放大后进入饱和区。
        ma_comp_raw = raw_data['ma_compression_raw']
        ma_comp_score = self._apply_energy_physics_norm(ma_comp_raw, mode='limit_high', sensitivity=5.0)
        # 布林带宽 (BBW): Limit Low。带宽越小越好。
        # BBW通常在 0.1-0.3。Sensitivity=5.0 -> 0.1*5=0.5 -> 1-0.46=0.54; 0.05*5=0.25 -> 1-0.24=0.76.
        bbw_raw = raw_data['bbw_raw']
        bbw_score = self._apply_energy_physics_norm(bbw_raw, mode='limit_low', sensitivity=5.0)
        # 物理分：均线压缩为主(主动)，布林收口为辅(被动)
        physical_compression = (ma_comp_score * 0.6 + bbw_score * 0.4)
        # --- 2. 筹码相变 (Chip Phase Transition - Order) ---
        # 筹码集中度: Limit High。越高越好。
        chip_conc_raw = raw_data['chip_concentration_raw']
        chip_conc_score = self._apply_energy_physics_norm(chip_conc_raw, mode='limit_high', sensitivity=3.0)
        # 筹码熵: Limit Low。越低越好 (结构单一)。
        chip_entropy_raw = raw_data['concentration_entropy_raw']
        chip_entropy_score = self._apply_energy_physics_norm(chip_entropy_raw, mode='limit_low', sensitivity=3.0)
        # 筹码稳定性变化: Limit High。正向变化。
        # 这是一个动态指标，用来确认筹码是否正在沉淀。
        chip_stab_change_raw = raw_data['chip_stability_change_raw']
        chip_stab_score = self._apply_energy_physics_norm(chip_stab_change_raw, mode='limit_high', sensitivity=5.0)
        chip_final = (chip_conc_score * 0.4 + chip_entropy_score * 0.4 + chip_stab_score * 0.2)
        # --- 3. 混沌边缘 (Edge of Chaos - Silence) ---
        # 价格熵: Limit Low。越低代表价格运动越有序。
        price_ent_raw = raw_data['price_entropy_raw']
        price_ent_score = self._apply_energy_physics_norm(price_ent_raw, mode='limit_low', sensitivity=5.0)
        # 分形维数稳定性: Zero Focus。
        # 必须锁定在 0 点。任何分形维数的剧烈跳动(无论方向)都意味着性质改变。
        fractal_raw = raw_data['price_fractal_raw']
        fractal_diff = fractal_raw.diff(3).fillna(0) # 计算变化率
        # Sensitivity=20.0: 对微小波动极度敏感。
        fractal_stab = self._apply_energy_physics_norm(fractal_diff, mode='zero_focus', sensitivity=20.0)
        chaos_final = (price_ent_score * 0.6 + fractal_stab * 0.4)
        # --- 4. 动力学增益 (Dynamics - Pulse) ---
        # 压缩率加速度 (Accel): Limit High。
        # 如果压缩率正在加速上升 (Accel > 0)，说明均线正在以更快的速度粘合。
        comp_accel_raw = raw_data['MA_POTENTIAL_COMPRESSION_RATE_accel']
        comp_accel_score = self._apply_energy_physics_norm(comp_accel_raw, mode='limit_high', sensitivity=10.0)
        dynamics_mult = 1 + (0.2 * comp_accel_score)
        # --- 5. 时空密度 (Time-Space Density) ---
        # 盘整时间: Time Density。
        # 盘整越久，能量密度越大。
        # Sensitivity=0.03: 约 30-40 天进入高分区。
        duration_raw = raw_data['consolidation_duration_raw']
        duration_score = self._apply_energy_physics_norm(duration_raw, mode='time_density', sensitivity=0.03)
        # --- 6. 探针埋点 ---
        _temp_debug_values["能量压缩详细探针"] = {
            "Physical_Sat_Score": physical_compression,
            "Chip_Order_Score": chip_final,
            "Chaos_Silence_Score": chaos_final,
            "Dynamics_Mult": dynamics_mult,
            "Time_Density_Score": duration_score
        }
        # --- 7. 最终融合 ---
        # 物理公式：能量 = 物理压缩 * 筹码有序 * 混沌寂静 * 动力增益 * (时间密度^0.1)
        # 时间密度作为修正系数，不作为主导，防止死股得分过高。
        final_score = (
            physical_compression.pow(0.3) * chip_final.pow(0.4) * chaos_final.pow(0.3) * dynamics_mult * duration_score.pow(0.1)
        ).clip(0, 1).fillna(0)
        _temp_debug_values["能量压缩"]["energy_compression_final"] = final_score
        return final_score

    def _calculate_volume_exhaustion_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V5.0 [物理极限真空模型 + 压单吸收]: 计算量能枯竭维度。
        版本说明：
            - 基础模型: 物理极限真空 (Limit Low)。
            - 新增维度: [压单吸收] (Suppression & Absorption)。
        核心逻辑：
            1. 数量枯竭(Quantity): 换手 + 量比 (Limit Low)。
            2. 质量控制(Quality): 异常成交 + 午后比例 (Anomaly High)。
            3. [新增] 压单吸收(Absorption): 阻力测试 + 吸收能量 (Limit High)。
            4. 动力学(Dynamics): Jerk (Pulse)。
        """
        # --- 1. 数量枯竭 (Limit Low) ---
        turnover_raw = raw_data['turnover_rate_f_raw']
        turnover_score = self._calculate_physics_score(turnover_raw, mode='limit_low', sensitivity=20.0)
        vol_pct_change = raw_data['volume_raw'].pct_change(3).fillna(0)
        vol_slope_score = self._calculate_physics_score(vol_pct_change, mode='limit_low', sensitivity=5.0)
        quantity_score = (turnover_score * 0.7 + vol_slope_score * 0.3)
        # --- 2. 质量控制 (Anomaly High) ---
        abnormal_raw = raw_data['tick_abnormal_vol_raw']
        abnormal_score = self._calculate_physics_score(abnormal_raw, mode='limit_high', sensitivity=5.0)
        afternoon_raw = raw_data['afternoon_flow_ratio_raw']
        afternoon_centered = (afternoon_raw - 0.5).clip(lower=0)
        afternoon_score = self._calculate_physics_score(afternoon_centered, mode='limit_high', sensitivity=5.0)
        # --- 3. [新增] 压单吸收 (Suppression & Absorption - High Quality) ---
        # 阻力测试次数: 越多越好(Limit High)。主力反复触碰压单。
        # 假设 raw_data 已提取 intraday_resistance_test_count_D
        res_test_raw = self.helper._get_safe_series(raw_data, 'resistance_test_count_raw', np.nan)
        res_test_score = self._calculate_physics_score(res_test_raw, mode='limit_high', sensitivity=0.5) # 次数通常是个位数
        # 吸收能量: 越大越好(Limit High)。吃单动能。
        # 假设 raw_data 已提取 absorption_energy_D
        absorb_energy_raw = self.helper._get_safe_series(raw_data, 'absorption_energy_raw', np.nan)
        absorb_score = self._calculate_physics_score(absorb_energy_raw, mode='limit_high', sensitivity=3.0)
        # 压单吸收分: 这是极高质量的'假性枯竭'。
        absorption_quality = (res_test_score * 0.5 + absorb_score * 0.5)
        # 综合质量分: 异常成交 + 午后护盘 + 压单吸收
        # 压单吸收是最高级的质量特征，权重较高。
        quality_score = (abnormal_score * 0.3 + afternoon_score * 0.2 + absorption_quality * 0.5)
        # --- 4. 动力学 (Jerk Pulse) ---
        turnover_jerk_raw = raw_data['turnover_rate_f_jerk']
        jerk_score = self._calculate_physics_score(turnover_jerk_raw, mode='limit_high', sensitivity=50.0)
        dynamics_mult = 1 + (0.3 * jerk_score)
        # --- 5. 探针 ---
        _temp_debug_values["量能枯竭详细探针"] = {
            "Quantity_Score": quantity_score,
            "Quality_Score": quality_score,
            "Absorption_Score": absorption_quality,
            "Jerk_Signal": jerk_score
        }
        # --- 6. 融合 ---
        final_score = (quantity_score.pow(0.5) * quality_score.pow(0.5) * dynamics_mult).clip(0, 1).fillna(0)
        _temp_debug_values["量能枯竭"]["volume_exhaustion_final"] = final_score
        return final_score

    def _calculate_main_force_covert_intent_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, ambiguity_weights: Dict, _temp_debug_values: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        V5.0 [物理极限暗流模型 + 聪明钱协同]: 计算主力隐蔽意图维度。
        版本说明：
            - 基础模型: 物理极限暗流 (Limit High)。
            - 新增维度: [聪明钱协同] (Smart Money Synergy)。
        核心逻辑：
            1. 结构化隐蔽(Structure): 隐形资金 + 聚类 (Limit High)。
            2. 博弈分歧(Game): 内部筹码交换 (Limit High)。
            3. [新增] 聪明钱协同(Synergy): 北向 + 顶级机构 (Limit High)。
            4. 战略纵深(HAB): 历史累积。
            5. 动力学(Pulse): Jerk。
        """
        # --- 1. 结构化隐蔽 (Structure) ---
        stealth_raw = raw_data['stealth_flow_ratio_raw']
        stealth_score = self._calculate_physics_score(stealth_raw, mode='limit_high', sensitivity=5.0)
        clustering_raw = raw_data['tick_clustering_raw']
        clustering_score = self._calculate_physics_score(clustering_raw, mode='limit_high', sensitivity=5.0)
        transfer_eff_raw = raw_data['tick_transfer_eff_raw']
        transfer_score = self._calculate_physics_score(transfer_eff_raw, mode='limit_high', sensitivity=5.0)
        structural_intent = (stealth_score * 0.4 + clustering_score * 0.4 + transfer_score * 0.2)
        # --- 2. 博弈分歧 (Game) ---
        sm_divergence_raw = raw_data['sm_divergence_raw']
        divergence_score = self._calculate_physics_score(sm_divergence_raw, mode='limit_high', sensitivity=3.0)
        abnormal_raw = raw_data['tick_abnormal_vol_raw']
        abnormal_score = self._calculate_physics_score(abnormal_raw, mode='limit_high', sensitivity=5.0)
        game_multiplier = 1 + (0.3 * divergence_score + 0.2 * abnormal_score)
        # --- 3. [新增] 聪明钱协同 (Smart Money Synergy) ---
        # 北向净买入: Limit High。外资背书。
        # 假设 raw_data 已提取 hm_net_buy_raw (SMART_MONEY_HM_NET_BUY_D)
        hm_buy_raw = self.helper._get_safe_series(raw_data, 'hm_net_buy_raw', np.nan)
        # 资金量级较大，建议先 relative_rank 或 Log 处理。这里使用 relative_rank。
        hm_score = self._calculate_physics_score(hm_buy_raw, mode='relative_rank', window=55)
        # 顶级机构活跃: Limit High。内资机构背书。
        # 假设 raw_data 已提取 top_tier_active_raw (HM_ACTIVE_TOP_TIER_D)
        top_tier_raw = self.helper._get_safe_series(raw_data, 'top_tier_active_raw', np.nan)
        top_tier_score = self._calculate_physics_score(top_tier_raw, mode='limit_high', sensitivity=5.0)
        # 协同乘数: 如果聪明钱都在买，意图的可信度大幅提升。
        # Synergy = 1 + 0.5 * (HM * TopTier)^0.5
        synergy_multiplier = 1 + 0.5 * (hm_score * top_tier_score).pow(0.5)
        # --- 4. 战略纵深 (HAB) ---
        # (保持 V4.0 逻辑)
        daily_flow = raw_data['net_mf_amount_raw']
        def calculate_hab_physics_score(window_sum_col: str, window_days: int) -> pd.Series:
            hist_sum = raw_data[window_sum_col]
            base_strength = self._calculate_physics_score(hist_sum, mode='relative_rank', window=89)
            buffer_coef = pd.Series(1.0, index=df_index, dtype=np.float32)
            mask_buffer = (hist_sum > 0) & (daily_flow < 0)
            outflow_ratio = daily_flow.abs() / (hist_sum.replace(0, np.nan).abs() + 1e-9)
            current_buffer = (1.0 - np.tanh(outflow_ratio * 5.0)).clip(0, 1)
            return base_strength * buffer_coef.mask(mask_buffer, current_buffer)
        hab_13 = calculate_hab_physics_score('net_mf_sum_13', 13)
        hab_21 = calculate_hab_physics_score('net_mf_sum_21', 21)
        hab_34 = calculate_hab_physics_score('net_mf_sum_34', 34)
        strategic_depth_score = (hab_13 * 0.5 + hab_21 * 0.3 + hab_34 * 0.2)
        strategic_multiplier = 1 + (0.4 * strategic_depth_score)
        # --- 5. 动力学脉冲 (Pulse) ---
        stealth_jerk = raw_data['stealth_flow_ratio_jerk']
        jerk_score = self._calculate_physics_score(stealth_jerk, mode='limit_high', sensitivity=20.0)
        pulse_multiplier = 1 + (0.3 * jerk_score)
        # --- 6. 探针 ---
        _temp_debug_values["主力隐蔽意图详细探针"] = {
            "Structural_Intent": structural_intent,
            "Synergy_Multiplier": synergy_multiplier,
            "Game_Multiplier": game_multiplier,
            "HM_Score": hm_score,
            "TopTier_Score": top_tier_score
        }
        # --- 7. 融合 ---
        # 物理公式: 意图 = 结构 * (博弈 * 协同 * 战略 * 动力)
        fused_score = (structural_intent * game_multiplier * synergy_multiplier * strategic_multiplier * pulse_multiplier).clip(0, 1).fillna(0)
        components = {'structural_intent': structural_intent}
        _temp_debug_values["主力隐蔽意图融合"] = fused_score
        return fused_score, components

    def _calculate_subdued_market_sentiment_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, sentiment_volatility_window: int, long_term_sentiment_window: int, sentiment_neutral_range: float, sentiment_pendulum_neutral_range: float, _temp_debug_values: Dict) -> pd.Series:
        """
        V5.0 [动力学熵寂与痛楚悖论模型]: 计算市场情绪低迷维度。
        核心逻辑：
            1. 静态悖论: 痛楚(Limit High) * 零卖出(Limit Low)。越痛越不卖，代表极致麻木。
            2. 动态熵寂: Slope/Accel (Zero Focus)。情绪波动必须被物理锁定。
            3. 突变惩罚: Jerk (Limit Low)。任何情绪突变都意味着寂静的破坏。
            4. 拒绝HAB: 情绪是相变状态，一旦破位即终结，不可缓冲。
        """
        # --- 1. 痛楚麻木悖论 (Pain Numbness Paradox) ---
        # 输家痛楚指数: 使用 Limit High (物理极限饱和)。
        # 寻找市场中'最痛'的时刻，这是反转的势能基础。
        loser_pain_raw = self.helper._get_safe_series(raw_data, 'loser_pain_index_D', np.nan)
        pain_score = self._calculate_physics_score(loser_pain_raw, mode='limit_high', sensitivity=3.0)
        # 散户卖出率: 使用 Limit Low (物理极限趋零)。
        # 在高痛楚下，卖出率趋近于0，代表供给侧彻底枯竭(锁仓/绝望)。
        retail_sell_raw = raw_data['retail_sell_rate_raw']
        # Sensitivity=15.0: 对微小卖出极为敏感，必须是死一般的沉寂。
        retail_sell_score = self._calculate_physics_score(retail_sell_raw, mode='limit_low', sensitivity=15.0)
        # 痛楚麻木分
        pain_numbness = (pain_score * retail_sell_score).pow(0.5)
        # --- 2. 获利盘绝对清洗 (Winner Cleanse) ---
        # 获利盘比例: Limit Low。
        # 风暴眼前夕，获利盘应被清洗至物理极限(接近0)。
        winner_rate_raw = self.helper._get_safe_series(raw_data, 'winner_rate_D', np.nan)
        winner_cleanse = self._calculate_physics_score(winner_rate_raw, mode='limit_low', sensitivity=10.0)
        # --- 3. 情绪动力学熵寂 (Sentiment Entropy Silence) ---
        # 市场情绪分: Slope 和 Accel。
        # 使用 Zero Focus (高斯核) 寻找绝对静止。任何方向的波动(Slope!=0)或加速(Accel!=0)都是杂音。
        # 必须使用数据层提供的斐波拉契周期数据 (如 5日或8日)
        sent_slope = raw_data['market_sentiment_score_slope']
        sent_accel = raw_data['market_sentiment_score_accel']
        # Sensitivity=50.0: 高斯带宽极窄，只奖励极致的平滑。
        slope_silence = self._calculate_physics_score(sent_slope, mode='zero_focus', sensitivity=50.0)
        accel_silence = self._calculate_physics_score(sent_accel, mode='zero_focus', sensitivity=30.0)
        entropy_silence = (slope_silence * 0.6 + accel_silence * 0.4)
        # --- 4. 突变惩罚 (Jerk Penalty) ---
        # 加加速度 (Jerk): 代表情绪状态的突变。
        # 在'低迷'组件中，Jerk是破坏者。我们希望 Jerk 越小越好 (Limit Low)。
        # 如果 Jerk 很大，说明寂静正在崩塌(可能是起爆，也可能是崩盘)，此时'低迷分'应降低。
        # 假设 raw_data 中已提取 market_sentiment_jerk (需在 _get_raw 中补充提取 JERK_5_market_sentiment_score_D)
        sent_jerk = self.helper._get_safe_series(raw_data, 'market_sentiment_jerk', np.nan) 
        # Sensitivity=20.0: 对突变敏感。
        jerk_stability = self._calculate_physics_score(sent_jerk, mode='limit_low', sensitivity=20.0)
        # --- 5. 探针埋点 ---
        _temp_debug_values["市场情绪低迷详细探针"] = {
            "Pain_Score": pain_score,
            "Retail_Sell_Zero": retail_sell_score,
            "Winner_Cleanse": winner_cleanse,
            "Slope_Silence": slope_silence,
            "Accel_Silence": accel_silence,
            "Jerk_Stability": jerk_stability
        }
        # --- 6. 最终融合 ---
        # 物理公式: 低迷 = 麻木 * 清洗 * 熵寂 * (突变惩罚)
        # Jerk Stability 作为乘数，一旦突变发生，分数归零，不仅无 HAB，反而有熔断机制。
        final_score = (
            pain_numbness.pow(0.4) * winner_cleanse.pow(0.2) * entropy_silence.pow(0.2) * jerk_stability.pow(0.2)
        ).clip(0, 1).fillna(0)
        _temp_debug_values["市场情绪低迷融合"] = final_score
        return final_score

    def _calculate_breakout_readiness_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V5.1 [临界共振动力学模型]: 计算突破准备度维度。
        版本说明：
            - 引入CRD模型：静态共振 + 物理衰减 + 动力学点火。
            - 必须加入 Slope/Accel/Jerk：捕捉突破潜力的'动量矢量'和'点火脉冲'。
            - 拒绝 HAB：临界状态不具备历史缓冲性，一触即溃。
        数据依赖：
            - breakout_potential_D (及 Slope/Accel/Jerk)
            - breakout_confidence_D
            - breakout_penalty_score_D
            - resistance_strength_D
            - GEOM_REG_R2_D
        """
        # --- 1. 临界共振 (Critical Resonance) ---
        # 突破潜力: Limit High。越高越好。
        pot_raw = raw_data['breakout_potential_raw']
        pot_score = self._calculate_physics_score(pot_raw, mode='limit_high', sensitivity=5.0)
        # [cite_start]突破信心: Limit High。信心是资金合力的体现 [cite: 1]。
        conf_raw = raw_data['breakout_confidence_raw']
        conf_score = self._calculate_physics_score(conf_raw, mode='limit_high', sensitivity=5.0)
        # 共振分: 只有当潜力与信心双高时，Ready才有效。
        resonance_score = (pot_score * conf_score).pow(0.5)
        # --- 2. 物理衰减 (Physical Decay) ---
        # [cite_start]突破惩罚分: Limit Low。代表上行过程中的套牢抛压预期 [cite: 1]。
        penalty_raw = raw_data['breakout_penalty_raw']
        penalty_score = self._calculate_physics_score(penalty_raw, mode='limit_low', sensitivity=10.0)
        # [cite_start]阻力强度: Limit Low。明确的技术位阻力 [cite: 3]。
        # 只有当阻力趋近于0时，才是完美的突破点。
        res_raw = raw_data['resistance_strength_raw']
        res_decay_score = self._calculate_physics_score(res_raw, mode='limit_low', sensitivity=10.0)
        # 衰减分
        decay_score = (penalty_score * res_decay_score).pow(0.5)
        # --- 3. 几何结构 (Structure) ---
        # [cite_start]几何R2: Limit High。形态越标准，突破越可靠 [cite: 1]。
        geom_raw = raw_data['geom_r2_raw']
        geom_score = self._calculate_physics_score(geom_raw, mode='limit_high', sensitivity=10.0)
        # 盘整质量: Limit High。
        consol_qual_raw = raw_data['consolidation_quality_raw']
        qual_score = self._calculate_physics_score(consol_qual_raw, mode='limit_high', sensitivity=5.0)
        struct_score = (geom_score * 0.6 + qual_score * 0.4)
        # --- 4. 动力学点火 (Kinetic Ignition) ---
        # 突破潜力的动力学特征：Slope & Accel & Jerk
        # 必须处于上升通道 (Slope > 0) 且 加速 (Accel > 0)
        pot_slope = raw_data['breakout_potential_slope']
        pot_accel = raw_data['breakout_potential_accel']
        # Sensitivity=20.0: 对潜力的微小变化高度敏感
        dyn_slope_score = self._calculate_physics_score(pot_slope, mode='limit_high', sensitivity=20.0)
        dyn_accel_score = self._calculate_physics_score(pot_accel, mode='limit_high', sensitivity=20.0)
        # Jerk (加加速度): 点火脉冲。
        # 当 Jerk 出现正向脉冲时，代表潜力加速的'突变'，即主力按下了发射按钮。
        # 假设 raw_data 中已提取 breakout_potential_jerk
        pot_jerk = self.helper._get_safe_series(raw_data, 'breakout_potential_jerk', np.nan)
        ignition_pulse = self._calculate_physics_score(pot_jerk, mode='limit_high', sensitivity=30.0)
        # 动力学乘数：趋势(Slope) * 加速(Accel) * 脉冲(Jerk)
        # Jerk作为额外的强力加成
        dynamics_mult = 1 + (0.15 * dyn_slope_score + 0.15 * dyn_accel_score + 0.2 * ignition_pulse)
        # --- 5. 探针埋点 ---
        _temp_debug_values["突破准备度详细探针"] = {
            "Resonance_Score": resonance_score,
            "Decay_Score": decay_score,
            "Structure_Score": struct_score,
            "Dynamics_Mult": dynamics_mult,
            "Ignition_Pulse": ignition_pulse,
            "Penalty_Raw": penalty_raw
        }
        # --- 6. 最终融合 ---
        # 物理公式: 准备度 = (共振 * 衰减 * 结构)^(1/3) * 动力学
        fused_score = (resonance_score * decay_score * struct_score).pow(0.33) * dynamics_mult
        _temp_debug_values["突破准备度融合"] = fused_score.clip(0, 1).fillna(0)
        return fused_score.clip(0, 1).fillna(0)

    def _apply_regulator_penalty(self, series: pd.Series, mode: str, threshold: float = 0.5, steepness: float = 1.0, window: int = 89) -> pd.Series:
        """
        [调节器专用]: 环境惩罚函数库。
        用于计算环境对策略的'压制系数' (0.0 - 1.0)。
        参数:
            mode:
                - 'fermi_suppression': 费米抑制。超过阈值迅速衰减 (ADX, Impact)。
                - 'quadratic_decay': 二次衰减。针对历史分位 (ATR)。
                - 'gaussian_cutoff': 高斯熔断。针对突变 (Jerk)。
                - 'linear_activation': 线性激活。针对正向指标 (Stability)。
            threshold: 费米函数的拐点，或高斯的带宽。
            steepness: 衰减的陡峭程度。
        """
        if series.isnull().all():
            return pd.Series(1.0, index=series.index) # 默认不惩罚
        if mode == 'fermi_suppression':
            # 费米-狄拉克分布变体：在 threshold 处发生相变。
            # x < threshold 时接近 1; x > threshold 时迅速降为 0。
            # steepness 控制相变区的宽度。
            # 归一化输入量级：假设输入已是绝对值。
            return 1.0 / (1.0 + np.exp(steepness * (series - threshold)))
        elif mode == 'quadratic_decay':
            # 针对历史分位 (Ranking)。
            # 计算滚动百分位 Rank (0-1)。
            roll_rank = series.rolling(window=window, min_periods=1).rank(pct=True)
            # Rank越高(波动越大)，惩罚越重。使用平方函数加速高位衰减。
            # Rank=0.8 -> Score=1-0.64=0.36; Rank=0.2 -> Score=1-0.04=0.96.
            return (1.0 - roll_rank.pow(steepness)).clip(0, 1)
        elif mode == 'gaussian_cutoff':
            # 高斯函数：x=0 时为 1。
            # threshold 在此处作为标准差 sigma 的倒数代理。
            return np.exp(- (series * threshold) ** 2)
        elif mode == 'linear_activation':
            # 简单的线性映射，但增加了底噪过滤。
            # 假设输入 0-1。
            return ((series - threshold) * steepness).clip(0, 1)
        return pd.Series(1.0, index=series.index)

    def _calculate_market_regulator_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V7.0 [环境惩罚与宏观共振模型]: 计算市场情境动态调节器。
        版本说明：
            - 基础架构：环境惩罚函数 (V6.0)。
            - 新增维度：[板块预热] (Sector Preheat) 作为环境加分项。
        核心逻辑：
            1. 静态环境(Static): ATR + Stability + ADX (基础分, max 1.0)。
            2. 脆弱性(Stress): Impact (扣分项)。
            3. 熔断(Circuit): Jerk (一票否决)。
            4. 宏观共振(Resonance): 板块预热 + 排名加速 (加分项, 可突破 1.0)。
        """
        regime_params = params.get('regime_modulator_params', {})
        if not get_param_value(regime_params.get('enabled'), False):
            return pd.Series(1.0, index=df_index, dtype=np.float32)
        # --- 1. 静态环境相态 (Static Environment - Penalty) ---
        atr_raw = self.helper._get_safe_series(raw_data, 'atr_raw', np.nan)
        atr_penalty = self._apply_regulator_penalty(atr_raw, mode='quadratic_decay', steepness=2.0, window=89)
        chip_stab_raw = self.helper._get_safe_series(raw_data, 'chip_stability_raw', np.nan)
        stab_activation = self._apply_regulator_penalty(chip_stab_raw, mode='linear_activation', threshold=0.4, steepness=1.66)
        adx_raw = self.helper._get_safe_series(raw_data, 'adx_raw', np.nan)
        adx_suppression = self._apply_regulator_penalty(adx_raw, mode='fermi_suppression', threshold=25.0, steepness=0.2)
        static_env_score = (atr_penalty * stab_activation * adx_suppression).pow(0.33)
        # --- 2. 脆弱性与熔断 (Stress & Circuit - Penalty) ---
        impact_raw = self.helper._get_safe_series(raw_data, 'flow_impact_raw', np.nan)
        impact_suppression = self._apply_regulator_penalty(impact_raw, mode='fermi_suppression', threshold=0.05, steepness=100.0)
        atr_jerk_raw = self.helper._get_safe_series(raw_data, 'atr_jerk', np.nan)
        jerk_cutoff = self._apply_regulator_penalty(atr_jerk_raw, mode='gaussian_cutoff', threshold=10.0)
        # --- 3. [新增] 宏观共振 (Sector Preheat - Bonus) ---
        # 行业预热分: Limit High。板块越热，个股的风暴眼越有效(轮动预期)。
        # 假设 raw_data 已提取 industry_preheat_score_D
        sector_preheat_raw = self.helper._get_safe_series(raw_data, 'sector_preheat_raw', np.nan)
        sector_preheat_score = self._calculate_physics_score(sector_preheat_raw, mode='limit_high', sensitivity=3.0)
        # 行业排名加速度: Limit High。板块正在加速向上。
        sector_accel_raw = self.helper._get_safe_series(raw_data, 'sector_rank_accel_raw', np.nan)
        sector_accel_score = self._calculate_physics_score(sector_accel_raw, mode='limit_high', sensitivity=5.0)
        # 共振加成系数: 基础为1.0。如果板块环境好，最高可达 1.3 (+30%)。
        # Bonus = 1 + 0.3 * (Preheat * Accel)^0.5
        resonance_bonus = 1.0 + 0.3 * (sector_preheat_score * sector_accel_score).pow(0.5)
        # --- 4. 探针埋点 ---
        _temp_debug_values["市场调节器详细探针"] = {
            "Static_Env": static_env_score,
            "Impact_Suppress": impact_suppression,
            "Jerk_Cutoff": jerk_cutoff,
            "Sector_Preheat_Bonus": resonance_bonus
        }
        # --- 5. 最终融合 ---
        min_mod = regime_params.get('min_modulator', 0.2)
        max_mod = regime_params.get('max_modulator', 1.0)
        # 基础调节器 (0-1)
        base_modulator = (static_env_score * impact_suppression * jerk_cutoff)
        # 映射到 [min, max]
        scaled_modulator = min_mod + base_modulator * (max_mod - min_mod)
        # 应用宏观共振加成 (可突破 max_mod)
        final_modulator = scaled_modulator * resonance_bonus
        _temp_debug_values["市场情境动态调节器"]["market_regime_modulator"] = final_modulator
        return final_modulator

    def _apply_fusion_specific_norm(self, series: pd.Series, mode: str, threshold: float = 0.0, sensitivity: float = 1.0) -> pd.Series:
        """
        [决策级物理映射]: 专用于最终融合层的归一化逻辑。
        区别于通用归一化，此处强调'绝对阈值'和'决策边界'。
        参数:
            mode:
                - 'sigmoid_belief': 趋势置信度。将线性分数映射为两极分化的置信概率。
                - 'gaussian_zero': 绝对死寂。只奖励 0 值，偏离即惩罚。
                - 'gated_activation': 门控激活。低于阈值归零，高于阈值线性增长。
                - 'tanh_saturation': 极限饱和。用于捕捉高能脉冲。
        """
        if series.isnull().all():
            return pd.Series(0.0, index=series.index)
        if mode == 'sigmoid_belief':
            # 用于 Trend Score (0-100)。
            # Center at threshold (e.g., 50). Sensitivity 控制相变陡峭度。
            # x=50 -> 0.5; x=60 -> High; x=40 -> Low.
            return 1.0 / (1.0 + np.exp(-sensitivity * (series - threshold)))
        elif mode == 'gaussian_zero':
            # 用于 Price Slope。
            # 寻找绝对的 0。任何波动都是噪音。
            return np.exp(- (series * sensitivity) ** 2)
        elif mode == 'gated_activation':
            # 用于 Veto 指标 (如主力活跃度)。
            # 低于 threshold 直接为 0 (熔断)。高于 threshold 线性映射到 1。
            # values < threshold -> 0
            # values > threshold -> (x - threshold) * sensitivity
            return ((series - threshold) * sensitivity).clip(0, 1)
        elif mode == 'tanh_saturation':
            # 用于 Jerk 脉冲。
            # 捕捉正向的高能信号。
            return np.tanh(series * sensitivity).clip(0, 1)
        return pd.Series(0.0, index=series.index)

    def _perform_final_fusion(self, df_index: pd.Index, component_scores: Dict[str, pd.Series], final_fusion_weights: Dict, price_calmness_params: Dict, main_force_control_params: Dict, raw_data: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        V8.1 [临界压差与动态权重模型 - 独立归一化版]: 执行最终融合。
        版本说明：
            - 摒弃 helper 归一化，使用决策级物理映射 (_apply_fusion_specific_norm)。
            - 逻辑架构：
                1. 动态权重(Adaptive): 基于趋势置信度(Sigmoid Belief)切换 Bull/Bear 模式。
                2. 临界压差(Pressure): 外压(Gaussian Zero) vs 内压(Tanh Saturation)。
                3. 主力否决(Veto): 门控激活(Gated Activation)熔断风险。
        """
        # --- 1. 动态权重重配 (Adaptive Weighting) ---
        # 趋势确认分: 0-100。
        trend_raw = raw_data['trend_confirm_raw']
        # 使用 Sigmoid 映射：以 50 分为界。Sensitivity=0.1。
        # 50->0.5; 60->0.73; 40->0.26. 制造清晰的 牛/熊 倾向。
        trend_belief = self._apply_fusion_specific_norm(trend_raw, mode='sigmoid_belief', threshold=50.0, sensitivity=0.1)
        # 定义两套权重体系
        # Bear Mode (左侧): 侧重'情绪低迷'和'能量压缩'
        weights_bear = final_fusion_weights.copy()
        weights_bear['subdued_market_sentiment'] = weights_bear.get('subdued_market_sentiment', 0.1) * 2.5
        weights_bear['energy_compression'] = weights_bear.get('energy_compression', 0.2) * 1.5
        # Bull Mode (右侧): 侧重'突破准备'和'隐蔽意图'
        weights_bull = final_fusion_weights.copy()
        weights_bull['breakout_readiness'] = weights_bull.get('breakout_readiness', 0.2) * 2.5
        weights_bull['main_force_covert_intent'] = weights_bull.get('main_force_covert_intent', 0.2) * 1.5
        # 分别计算两种模式下的基础分
        score_bear = _robust_geometric_mean(component_scores, weights_bear, df_index)
        score_bull = _robust_geometric_mean(component_scores, weights_bull, df_index)
        # 线性插值: 根据趋势置信度平滑切换
        base_calm_score = score_bear * (1 - trend_belief) + score_bull * trend_belief
        # --- 2. 临界压差 (Pressure Differential) ---
        # [External Calmness]: 价格波动的死寂。
        # 使用 Gaussian Zero Focus。严格锁定 Slope=0。
        # Sensitivity=50.0: 只要 Slope 偏离 0.02，分数即衰减至 <0.3。
        price_slope = raw_data['price_slope_raw']
        ext_calmness = self._apply_fusion_specific_norm(price_slope, mode='gaussian_zero', sensitivity=50.0)
        # [Internal Boiling]: 意图动量的沸腾。
        # 使用 Tanh Saturation。捕捉 Jerk 的正向脉冲。
        # 隐形资金突变: stealth_flow_ratio_jerk
        stealth_jerk = raw_data['stealth_flow_ratio_jerk']
        int_boiling = self._apply_fusion_specific_norm(stealth_jerk, mode='tanh_saturation', sensitivity=20.0)
        # 压差乘数: Multiplier = 1 + 0.6 * (Ext_Calm * Int_Boil)
        # 物理含义: 外表极静(1.0) 且 内核极热(1.0) -> 1.6倍爆发力。
        pressure_diff_mult = 1 + (0.6 * ext_calmness * int_boiling)
        # --- 3. 主力一票否决 (Main Force Veto) ---
        # [Chip Structure]: 筹码结构状态。
        # 假设 chip_structure_state_D 是评分制 (0-100) 或 分类值。
        # 这里假设已映射为 0-1 或 0-100 的质量分。设门槛为 30 (0.3)。
        # 低于 30 直接熔断。
        struct_raw = raw_data['chip_structure_state_raw']
        # 假设 raw 是 0-100。Threshold=30, Sensitivity=0.02 (1/50)。
        # x=20 -> 0; x=80 -> (80-30)*0.02 = 1.0.
        struct_veto = self._apply_fusion_specific_norm(struct_raw, mode='gated_activation', threshold=30.0, sensitivity=0.02)
        # [Main Force Activity]: 主力活跃度。
        # 必须有主力在场。Threshold=10 (假设0-100), Sensitivity=0.05.
        activity_raw = raw_data['mf_activity_raw']
        activity_veto = self._apply_fusion_specific_norm(activity_raw, mode='gated_activation', threshold=10.0, sensitivity=0.05)
        # 综合否决系数
        control_veto = (struct_veto * activity_veto).pow(0.5)
        # --- 4. 探针埋点 ---
        _temp_debug_values["最终融合"]["Trend_Belief_Sigmoid"] = trend_belief
        _temp_debug_values["最终融合"]["Base_Calm_Adaptive"] = base_calm_score
        _temp_debug_values["最终融合"]["Ext_Calmness_Gaussian"] = ext_calmness
        _temp_debug_values["最终融合"]["Int_Boiling_Tanh"] = int_boiling
        _temp_debug_values["最终融合"]["Pressure_Diff_Mult"] = pressure_diff_mult
        _temp_debug_values["最终融合"]["Control_Veto_Gated"] = control_veto
        # --- 5. 计算最终分 ---
        # Final = Base * PressureDiff * Veto
        final_score = (base_calm_score * pressure_diff_mult * control_veto).clip(0, 1).fillna(0.0)
        return final_score
