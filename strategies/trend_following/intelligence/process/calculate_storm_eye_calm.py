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

    def _get_required_signals(self, params: Dict, mtf_slope_accel_weights: Dict, mtf_cohesion_base_signals: List) -> List[str]:
        """
        V11.0.0: 基于斐波那契动力学全量更新所需信号。
        """
        # 基础静态指标
        required_signals = [
            'MA_POTENTIAL_TENSION_INDEX_D', 'MA_COHERENCE_RESONANCE_D', 'MA_POTENTIAL_COMPRESSION_RATE_D',
            'BBW_21_2.0_D', 'chip_concentration_ratio_D', 'concentration_entropy_D',
            'PRICE_ENTROPY_D', 'GEOM_ARC_CURVATURE_D', 'dynamic_consolidation_duration_D',
            'turnover_rate_f_D', 'volume_D', 'intraday_trough_filling_degree_D',
            'tick_abnormal_volume_ratio_D', 'afternoon_flow_ratio_D', 'absorption_energy_D',
            'stealth_flow_ratio_D', 'tick_clustering_index_D', 'accumulation_signal_score_D',
            'SMART_MONEY_HM_NET_BUY_D', 'HM_ACTIVE_TOP_TIER_D', 'net_mf_amount_D',
            'loser_pain_index_D', 'sell_sm_amount_rate_D', 'winner_rate_D',
            'market_sentiment_score_D', 'breakout_potential_D', 'breakout_confidence_D',
            'breakout_penalty_score_D', 'resistance_strength_D', 'GEOM_REG_R2_D',
            'ATR_14_D', 'chip_stability_D', 'ADX_14_D', 'flow_impact_ratio_D',
            'industry_preheat_score_D', 'industry_rank_accel_D', 'trend_confirmation_score_D',
            'chip_structure_state_D', 'main_force_activity_index_D'
        ]
        # 引入物理导数维度: Slope(13), Accel(8), Jerk(5) 
        phys_map = {'SLOPE': '13', 'ACCEL': '8', 'JERK': '5'}
        targets = ['MA_POTENTIAL_COMPRESSION_RATE_D', 'turnover_rate_f_D', 'stealth_flow_ratio_D', 'breakout_potential_D', 'market_sentiment_score_D']
        for target in targets:
            for deriv, period in phys_map.items():
                required_signals.append(f'{deriv}_{period}_{target}')
        return required_signals

    def _get_raw_and_atomic_data(self, df: pd.DataFrame, method_name: str, params: Dict) -> Dict[str, pd.Series]:
        """
        V10.0.1: 映射军械库数据，加入强制探针。
        """
        raw_data = {}
        target_columns = self._get_required_signals(params, params.get('mtf_slope_accel_weights', {}), [])
        for col in target_columns:
            if col not in df.columns: print(f"[CRITICAL ERROR] 军械库缺失关键列: {col}")
            raw_data[col] = df[col]
        raw_data['net_mf_sum_13'] = raw_data['net_mf_amount_D'].rolling(window=13, min_periods=1).sum()
        raw_data['net_mf_sum_21'] = raw_data['net_mf_amount_D'].rolling(window=21, min_periods=1).sum()
        raw_data['price_slope_raw'] = df[f'SLOPE_5_market_sentiment_score_D']
        raw_data['stealth_jerk_raw'] = df[f'JERK_5_stealth_flow_ratio_D']
        return raw_data

    def _calculate_physics_score(self, series: pd.Series, mode: str, sensitivity: float = 1.0, window: int = 55, denoise: bool = False) -> pd.Series:
        """
        V11.0.1: 物理归一化引擎，包含降噪与零基陷阱过滤。
        """
        # 降噪预处理：计算指标的滚动变异系数，低于门限的视为无效背景 
        if denoise:
            noise_floor = series.rolling(window=21).std().fillna(0) * 0.1
            series = series.where(series.abs() > noise_floor, 0.0)
        if mode == 'limit_low':
            # 强化型 Tanh：1 - tanh(|x| * k)，专门用于量能枯竭 [cite: 2]
            return 1.0 - np.tanh(series.abs() * sensitivity)
        elif mode == 'limit_high':
            # 饱和映射：用于主力意图爆发 
            return np.tanh(series * sensitivity).clip(0, 1)
        elif mode == 'zero_focus':
            # 高斯核锁定：用于价格斜率死寂，物理映射 0 点为 1 
            return np.exp(- (series * sensitivity) ** 2)
        elif mode == 'relative_rank':
            roll_min = series.rolling(window=window, min_periods=1).min()
            roll_max = series.rolling(window=window, min_periods=1).max()
            return (series - roll_min) / (roll_max - roll_min).replace(0, 1.0)
        return pd.Series(0.0, index=series.index)

    def _calculate_mtf_derived_scores(self, df: pd.DataFrame, df_index: pd.Index, mtf_slope_accel_weights: Dict, mtf_cohesion_base_signals: List, method_name: str) -> Dict[str, pd.Series]:
        mtf_derived_scores = {}
        mtf_derived_scores['bbw_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'BBW_21_2.0_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['vol_instability_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['turnover_rate_slope_inverted_score'] = self.helper._get_mtf_slope_accel_score(df, 'turnover_rate_f_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['mf_net_flow_slope_positive'] = self.helper._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_derived_scores['mtf_cohesion_score'] = self.helper._get_mtf_cohesion_score(df, mtf_cohesion_base_signals, mtf_slope_accel_weights, df_index, method_name)
        return mtf_derived_scores

    def _calculate_energy_compression_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V12.0.1: 能量压缩组件升级版。
        新增：MA_POTENTIAL_COMPRESSION_RATE_D  的斜率与加速度判断。
        """
        print(f"--- [能量动力学探针] @ {df_index[-1]} ---")
        comp_rate = raw_data['MA_POTENTIAL_COMPRESSION_RATE_D']
        # 获取斐波那契周期下的动力学特征
        slope = raw_data.get('SLOPE_13_MA_POTENTIAL_COMPRESSION_RATE_D', comp_rate.diff(13))
        accel = raw_data.get('ACCEL_8_MA_POTENTIAL_COMPRESSION_RATE_D', slope.diff(8))
        # 降噪：零基陷阱过滤逻辑
        filtered_slope = slope.where(slope.abs() > slope.rolling(21).std() * 0.1, 0.0)
        # 物理映射：斜率趋近于0 且 压缩率维持在高位
        slope_score = np.exp(- (filtered_slope * 10.0) ** 2) # 高斯核锁定“静止收敛”
        accel_score = np.tanh(accel * 5.0).clip(-1, 1)
        # 结合均线共振度 
        resonance = self._calculate_physics_score(raw_data['MA_COHERENCE_RESONANCE_D'], mode='limit_high', sensitivity=2.0)
        print(f"  -- 压缩率: {comp_rate.iloc[-1]:.4f} | 斜率静止度: {slope_score.iloc[-1]:.4f} | 共振分: {resonance.iloc[-1]:.4f}")
        final_energy = (comp_rate * 0.4 + slope_score * 0.3 + resonance * 0.3)
        return final_energy.clip(0, 1)

    def _calculate_volume_exhaustion_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V12.0.2: 量能枯竭组件升级版。
        新增：turnover_rate_f_D  的 Jerk (加加速度) 以识别真空。
        """
        print(f"--- [量能真空探针] @ {df_index[-1]} ---")
        turnover = raw_data['turnover_rate_f_D']
        jerk = raw_data.get('JERK_5_turnover_rate_f_D', turnover.diff().diff().diff())
        # 采用逻辑门控去除零基噪音
        noise_floor = jerk.rolling(34).std() * 0.2
        clean_jerk = jerk.where(jerk.abs() > noise_floor, 0.0)
        # Jerk 评分：当加加速度出现负向脉冲后转为死寂，代表流动性彻底锁死
        jerk_score = 1.0 - np.tanh(clean_jerk.abs() * 50.0)
        # 结合地量填充度 
        trough_fill = self._calculate_physics_score(raw_data['intraday_trough_filling_degree_D'], mode='limit_high', sensitivity=3.0)
        print(f"  -- 换手率: {turnover.iloc[-1]:.4f} | Jerk死寂分: {jerk_score.iloc[-1]:.4f} | 地量填充: {trough_fill.iloc[-1]:.4f}")
        final_vol = (jerk_score * 0.5 + trough_fill * 0.5) * (1.0 - np.tanh(turnover * 10.0))
        return final_vol.clip(0, 1)

    def _calculate_main_force_covert_intent_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, ambiguity_weights: Dict, _temp_debug_values: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        V12.0.3: 主力意图组件 HAB 融合版。
        逻辑：当日净额 net_mf_amount_D  必须经过历史水库模型校验。
        """
        print(f"--- [主力意图 HAB 探针] @ {df_index[-1]} ---")
        mf_net = raw_data['net_mf_amount_D']
        # 调用 HAB 系统计算存量缓冲
        hab_buffer = self._calculate_historical_accumulation_buffer(mf_net, windows=[13, 21, 34])
        # 隐蔽收集率及其加速度 
        stealth = raw_data['stealth_flow_ratio_D']
        stealth_accel = raw_data.get('ACCEL_8_stealth_flow_ratio_D', stealth.diff().diff())
        # 存量修正逻辑：如果 HAB 缓冲极强，当日小幅负值不影响意图分
        intent_base = self._calculate_physics_score(stealth, mode='limit_high', sensitivity=4.0)
        accel_bonus = np.tanh(stealth_accel * 10.0).clip(0, 1)
        # 最终意图：(基础隐蔽分 + 加速奖赏) * 历史存量置信度
        fused_intent = (intent_base + accel_bonus * 0.3) * (0.7 + 0.3 * hab_buffer)
        print(f"  -- HAB缓冲系数: {hab_buffer.iloc[-1]:.4f} | 隐蔽加速度奖赏: {accel_bonus.iloc[-1]:.4f}")
        return fused_intent.clip(0, 1), {"intent_base": intent_base}

    def _calculate_subdued_market_sentiment_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, sentiment_volatility_window: int, long_term_sentiment_window: int, sentiment_neutral_range: float, sentiment_pendulum_neutral_range: float, _temp_debug_values: Dict) -> pd.Series:
        """
        V13.0.0: 基于动力学熵寂与痛楚悖论模型。
        新增：loser_pain_index_D 的 Jerk 捕捉与情绪 HAB 缓冲。
        """
        print(f"--- [市场情绪动力学探针] @ {df_index[-1]} ---")
        # 1. 痛楚麻木与绝望松动
        pain_raw = raw_data['loser_pain_index_D']
        pain_jerk = raw_data.get('JERK_5_loser_pain_index_D', pain_raw.diff().diff().diff())
        # 降噪处理：过滤微小心理波动
        pain_noise_floor = pain_jerk.rolling(21).std() * 0.1
        clean_pain_jerk = pain_jerk.where(pain_jerk.abs() > pain_noise_floor, 0.0)
        # 物理映射：高痛感 + 痛感突变(Jerk) = 筹码松动临界点
        pain_score = self._calculate_physics_score(pain_raw, mode='limit_high', sensitivity=3.0)
        despair_burst = self._calculate_physics_score(clean_pain_jerk, mode='limit_high', sensitivity=20.0)
        # 2. 情绪熵寂 (Entropy Silence)
        sent_slope = raw_data.get('SLOPE_13_market_sentiment_score_D', raw_data['market_sentiment_score_D'].diff(13))
        sent_accel = raw_data.get('ACCEL_8_market_sentiment_score_D', sent_slope.diff(8))
        # 使用高斯核锁定绝对静止：Slope/Accel 越接近 0，得分越高
        slope_silence = self._calculate_physics_score(sent_slope, mode='zero_focus', sensitivity=50.0, denoise=True)
        accel_silence = self._calculate_physics_score(sent_accel, mode='zero_focus', sensitivity=30.0, denoise=True)
        # 3. 情绪 HAB 缓冲 (存量意识)
        # 如果历史 21 日情绪极低，当日微幅反弹不应破坏“低迷”判定
        sent_hab = self._calculate_historical_accumulation_buffer(raw_data['market_sentiment_score_D'], windows=[21])
        # 4. 获利盘清洗 (Winner Cleanse)
        winner_rate = raw_data['winner_rate_D']
        cleanse_score = self._calculate_physics_score(winner_rate, mode='limit_low', sensitivity=15.0)
        print(f"  -- 痛感分: {pain_score.iloc[-1]:.4f} | 绝望突变(Jerk): {despair_burst.iloc[-1]:.4f} | 情绪静止度: {slope_silence.iloc[-1]:.4f}")
        # 5. 最终融合：(麻木度 * 熵寂度 * 清洗度) + 绝望突变补偿
        # 绝望突变 (Despair Burst) 是反转的强力加分项
        base_subdued = (pain_score.pow(0.4) * slope_silence.pow(0.3) * cleanse_score.pow(0.3))
        final_sentiment = (base_subdued * (0.8 + 0.2 * sent_hab) + 0.2 * despair_burst)
        _temp_debug_values["市场情绪低迷详细探针"] = {"pain_score": pain_score, "jerk_burst": despair_burst, "hab_factor": sent_hab}
        return final_sentiment.clip(0, 1)

    def _calculate_breakout_readiness_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V14.0.1: 突破准备度动力学升级版。
        新增：潜力加速度、阻力斜率衰减与累积信号 HAB 缓冲。
        """
        print(f"--- [突破临界探针] @ {df_index[-1]} ---")
        # 1. 突破潜力及其动力学 (SLOPE_13, ACCEL_8)
        pot_raw = raw_data['breakout_potential_D']
        pot_accel = raw_data.get('ACCEL_8_breakout_potential_D', pot_raw.diff().diff())
        pot_score = self._calculate_physics_score(pot_raw, mode='limit_high', sensitivity=5.0)
        accel_bonus = self._calculate_physics_score(pot_accel, mode='limit_high', sensitivity=10.0, denoise=True)
        # 2. 阻力衰减动力学 (SLOPE_13)
        res_raw = raw_data['resistance_strength_D']
        res_slope = raw_data.get('SLOPE_13_resistance_strength_D', res_raw.diff(13))
        # 物理含义：阻力在减小 (Slope < 0) 且当前值低
        res_decay = self._calculate_physics_score(res_slope, mode='limit_low', sensitivity=15.0)
        # 3. 筹码底蕴 HAB 缓冲
        acc_signal = raw_data['accumulation_signal_score_D']
        acc_hab = self._calculate_historical_accumulation_buffer(acc_signal, windows=[21, 34])
        print(f"  -- 潜力分: {pot_score.iloc[-1]:.4f} | 加速奖赏: {accel_bonus.iloc[-1]:.4f} | 阻力衰减度: {res_decay.iloc[-1]:.4f}")
        # 4. 最终准备度：共振分 * 动力学增益 * HAB 缓冲
        readiness = (pot_score * 0.4 + accel_bonus * 0.3 + res_decay * 0.3) * (0.8 + 0.2 * acc_hab)
        return readiness.clip(0, 1)

    def _calculate_market_regulator_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V10.0.5: 引入行业预热与Fermi环境抑制。
        """
        print(f"--- 环境调节节点探针 @ {df_index[-1]} ---")
        preheat_score = self._calculate_physics_score(raw_data['industry_preheat_score_D'], mode='limit_high', sensitivity=3.0)
        rank_accel_score = self._calculate_physics_score(raw_data['industry_rank_accel_D'], mode='limit_high', sensitivity=5.0)
        sector_bonus = 1.0 + 0.3 * (preheat_score * rank_accel_score).pow(0.5)
        adx_supp = 1.0 / (1.0 + np.exp(0.3 * (raw_data['ADX_14_D'] - 28.0)))
        atr_rank = raw_data['ATR_14_D'].rolling(window=89, min_periods=1).rank(pct=True)
        atr_penalty = (1.0 - atr_rank.pow(2.0))
        print(f"SectorBonus: {sector_bonus.iloc[-1]:.4f} | FermiADX: {adx_supp.iloc[-1]:.4f} | ATR_Penalty: {atr_penalty.iloc[-1]:.4f}")
        final_modulator = adx_supp * atr_penalty * sector_bonus
        return final_modulator.clip(0.2, 1.3)

    def _perform_final_fusion(self, df_index: pd.Index, component_scores: Dict[str, pd.Series], final_fusion_weights: Dict, price_calmness_params: Dict, main_force_control_params: Dict, raw_data: Dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        V14.0.2: 动力学压差与 HAB 全局融合模型。
        逻辑：利用资金 HAB 修正“流出误判”，通过物理压差确定点火置信度。
        """
        print(f"--- [最终融合 HAB 联动探针] @ {df_index[-1]} ---")
        # 1. 资金水库 HAB 缓冲
        mf_net = raw_data['net_mf_amount_D']
        mf_hab = self._calculate_historical_accumulation_buffer(mf_net, windows=[13, 21, 34])
        # 2. 物理压差：外静 (Price Slope) 内沸 (Intent Jerk)
        ext_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        int_boil = self._calculate_physics_score(raw_data.get('JERK_5_stealth_flow_ratio_D', 0), mode='limit_high', sensitivity=25.0)
        pressure_mult = 1.0 + 1.2 * (ext_calm * int_boil)
        # 3. 基础寂静分计算
        base_score = _robust_geometric_mean(component_scores, final_fusion_weights, df_index)
        # 4. HAB 修正：如果存量缓冲强，则放大 base_score
        final_score = base_score * (0.7 + 0.3 * mf_hab) * pressure_mult
        print(f"  -- 基础分: {base_score.iloc[-1]:.4f} | 资金存量缓冲: {mf_hab.iloc[-1]:.4f} | 最终输出: {final_score.iloc[-1]:.4f}")
        return final_score.clip(0, 1)

    def _calculate_historical_accumulation_buffer(self, daily_series: pd.Series, windows: list[int] = [13, 21, 34]) -> pd.Series:
        """
        V14.0.0: 历史累积记忆缓冲系统 (HAB)。
        逻辑：计算多周期累积存量对当日信号的缓冲能力，识别“大买小洗”特征。
        """
        print(f"--- [HAB 内存探针] 启动累积计算 ---")
        buffers = []
        for w in windows:
            acc_sum = daily_series.rolling(window=w, min_periods=w//2).sum()
            # 存量/流量比：当日流出占存量的比例。若占比极小，则缓冲系数接近 1.0
            # 采用 denoise 逻辑：分母加入极小项避免零基陷阱
            impact_ratio = daily_series / (acc_sum.abs() + 1e-9)
            # 使用 tanh 映射，当流出方向与存量相反且比例小时，返回高缓冲
            buffer_factor = 1.0 - np.tanh(impact_ratio.clip(lower=-1, upper=0).abs() * 5.0)
            buffers.append(buffer_factor)
            if not acc_sum.empty: print(f"  -- 周期 {w} 存量均值: {acc_sum.iloc[-1]:.2f} | 缓冲状态: {buffer_factor.iloc[-1]:.4f}")
        final_buffer = pd.concat(buffers, axis=1).mean(axis=1).fillna(1.0)
        return final_buffer








