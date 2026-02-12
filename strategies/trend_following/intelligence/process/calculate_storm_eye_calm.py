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
        V53.2.0: 全量整合版主入口逻辑。
        说明: 包含 Fermi-Dirac 自适应门控、三级熔断、STAR 奖励以及物理限幅自洽性校验。
        """
        method_name = "calculate_storm_eye_calm"
        print(f"--- [{method_name}] @ 拆单吸筹强度 探针开始 ---")
        self.last_df_index = df.index # 存储索引供辅助方法使用
        df_index = df.index
        params = self._get_storm_eye_calm_params(config)
        raw_data = self._get_raw_and_atomic_data(df, method_name, params)
        _temp_debug_values = {"raw": raw_data}
        energy_score = self._calculate_energy_compression_component(df_index, raw_data, {}, params['energy_compression_weights'], _temp_debug_values)
        volume_score = self._calculate_volume_exhaustion_component(df_index, raw_data, {}, params['volume_exhaustion_weights'], _temp_debug_values)
        intent_score, _ = self._calculate_main_force_covert_intent_component(df_index, raw_data, {}, params['main_force_covert_intent_weights'], {}, _temp_debug_values)
        sentiment_score = self._calculate_subdued_market_sentiment_component(df_index, raw_data, params['subdued_market_sentiment_weights'], 21, 55, 1.0, 0.2, _temp_debug_values)
        readiness_score = self._calculate_breakout_readiness_component(df_index, raw_data, params['breakout_readiness_weights'], _temp_debug_values)
        dynamic_threshold = self._calculate_adaptive_phase_transition_threshold(df_index, raw_data)
        gate_energy = self._calculate_fermi_dirac_gate(energy_score, threshold=dynamic_threshold, beta=12.0)
        gate_volume = self._calculate_fermi_dirac_gate(volume_score, threshold=dynamic_threshold, beta=12.0)
        gate_intent = self._calculate_fermi_dirac_gate(intent_score, threshold=dynamic_threshold, beta=12.0)
        gate_sentiment = self._calculate_fermi_dirac_gate(sentiment_score, threshold=dynamic_threshold, beta=12.0)
        gate_readiness = self._calculate_fermi_dirac_gate(readiness_score, threshold=dynamic_threshold, beta=12.0)
        component_scores = {
            'energy': energy_score * gate_energy, 'volume': volume_score * gate_volume,
            'intent': intent_score * gate_intent, 'sentiment': sentiment_score * gate_sentiment,
            'readiness': readiness_score * gate_readiness
        }
        final_fusion_score = self._perform_final_fusion(df_index, component_scores, params['final_fusion_weights'], {}, {}, raw_data, _temp_debug_values)
        regime_modulator = self._calculate_market_regulator_modulator(df_index, raw_data, params, _temp_debug_values)
        raw_final_score = final_fusion_score * regime_modulator
        ewd_factor = self._calculate_consensus_entropy(component_scores)
        resonance_confirm = (raw_final_score > 0.4) & (ewd_factor > 0.7)
        latch_count = resonance_confirm.rolling(window=5, min_periods=1).sum()
        latch_multiplier = np.where(latch_count >= 3, 1.2, 1.0)
        latched_score = raw_final_score.rolling(window=3, min_periods=1).mean() * latch_multiplier
        bipolar_gain = self._calculate_oversold_momentum_bipolarization(df_index, raw_data)
        veto_factor = self._calculate_kinetic_overflow_veto(df_index, raw_data, bipolar_gain)
        reward_factor = self._calculate_spatio_temporal_asymmetric_reward(df_index, raw_data, resonance_confirm)
        final_latched_score = (latched_score * veto_factor * reward_factor).clip(0, 1)
        print(f"  -- 最终输出: {final_latched_score.iloc[-1]:.4f} | 锁存次数: {latch_count.iloc[-1]} | 动态阈值: {dynamic_threshold.iloc[-1]:.4f}")
        return final_latched_score.astype(np.float32)

    def _calculate_fermi_dirac_gate(self, score_series: pd.Series, threshold: float = 0.5, beta: float = 10.0) -> pd.Series:
        """
        V46.0.1: Fermi-Dirac 博弈门控辅助方法。
        说明: 利用费米函数对组件分值进行非线性激活，模拟系统起爆占位效应。
        """
        gate_val = 1.0 / (1.0 + np.exp(beta * (threshold - score_series)))
        return gate_val

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

    def _get_required_signals(self, params: Dict, mtf_slope_accel_weights: Dict, mtf_cohesion_base_signals: List) -> list[str]:
        """
        V54.0.0: 军械库合规化原子信号清单。
        说明: 仅请求军械库中存在的原始指标，为动态导数引擎提供原子数据支撑。
        """
        required_signals = [
            'MA_POTENTIAL_TENSION_INDEX_D', 'MA_COHERENCE_RESONANCE_D', 'MA_POTENTIAL_COMPRESSION_RATE_D',
            'BBW_21_2.0_D', 'chip_concentration_ratio_D', 'concentration_entropy_D',
            'PRICE_ENTROPY_D', 'GEOM_ARC_CURVATURE_D', 'dynamic_consolidation_duration_D',
            'turnover_rate_f_D', 'volume_D', 'intraday_trough_filling_degree_D',
            'tick_abnormal_volume_ratio_D', 'afternoon_flow_ratio_D', 'absorption_energy_D',
            'stealth_flow_ratio_D', 'tick_clustering_index_D', 'accumulation_signal_score_D',
            'SMART_MONEY_HM_NET_BUY_D', 'HM_ACTIVE_TOP_TIER_D', 'net_mf_amount_D',
            'profit_ratio_D', 'winner_rate_D', 'market_sentiment_score_D', 
            'breakout_potential_D', 'breakout_confidence_D', 'breakout_penalty_score_D', 
            'resistance_strength_D', 'GEOM_REG_R2_D', 'GEOM_REG_SLOPE_D',
            'ATR_14_D', 'chip_stability_D', 'ADX_14_D', 'flow_impact_ratio_D',
            'industry_preheat_score_D', 'industry_rank_accel_D', 'industry_strength_rank_D',
            'trend_confirmation_score_D', 'chip_structure_state_D', 'main_force_activity_index_D',
            'intraday_cost_center_migration_D', 'migration_convergence_ratio_D', 'tick_chip_balance_ratio_D',
            'VPA_EFFICIENCY_D', 'VPA_MF_ADJUSTED_EFF_D', 'VPA_ACCELERATION_5D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'OCH_ACCELERATION_D', 'OCH_D', 'PDI_14_D', 'NDI_14_D', 'price_vs_ma_21_ratio_D', 
            'price_vs_ma_55_ratio_D', 'HM_COORDINATED_ATTACK_D', 'TURNOVER_STABILITY_INDEX_D', 
            'amount_D', 'price_grid_D', 'HM_ACTIVE_ANY_D', 'BIAS_55_D', 'MA_ACCELERATION_EMA_55_D',
            'STATE_GOLDEN_PIT_D', 'BIAS_5_D', 'MA_FAN_EFFICIENCY_D', 'RSI_13_D', 'close'
        ]
        return list(set(required_signals))

    def _get_raw_and_atomic_data(self, df: pd.DataFrame, method_name: str, params: Dict) -> Dict[str, pd.Series]:
        """
        V54.0.1: 具备动态导数生成与 3-Sigma 限幅的数据矩阵引擎。
        说明: 全量生成物理模型所需的 Slope/Accel/Jerk 衍生列并执行降噪处理。
        """
        raw_data = {}
        target_columns = self._get_required_signals(params, {}, [])
        for col in target_columns:
            if col in df.columns:
                raw_data[col] = df[col]
            else:
                print(f"[WARNING] 军械库清单中未发现列: {col}")
        # 1. 动态生成导数并执行物理限幅
        raw_data['JERK_5_VPA_ACCELERATION_5D'] = self._clip_physical_outliers(raw_data['VPA_ACCELERATION_5D'].diff().diff().diff())
        raw_data['SLOPE_13_VPA_MF_ADJUSTED_EFF_D'] = self._clip_physical_outliers(raw_data['VPA_MF_ADJUSTED_EFF_D'].diff(13))
        raw_data['ACCEL_8_VPA_MF_ADJUSTED_EFF_D'] = self._clip_physical_outliers(raw_data['SLOPE_13_VPA_MF_ADJUSTED_EFF_D'].diff(8))
        raw_data['JERK_5_VPA_MF_ADJUSTED_EFF_D'] = self._clip_physical_outliers(raw_data['VPA_MF_ADJUSTED_EFF_D'].diff().diff().diff())
        raw_data['JERK_5_tick_abnormal_volume_ratio_D'] = self._clip_physical_outliers(raw_data['tick_abnormal_volume_ratio_D'].diff().diff().diff())
        raw_data['JERK_5_MA_ACCELERATION_EMA_55_D'] = self._clip_physical_outliers(raw_data['MA_ACCELERATION_EMA_55_D'].diff().diff().diff())
        raw_data['SLOPE_13_PRICE_ENTROPY_D'] = self._clip_physical_outliers(raw_data['PRICE_ENTROPY_D'].diff(13))
        raw_data['ACCEL_8_PRICE_ENTROPY_D'] = self._clip_physical_outliers(raw_data['PRICE_ENTROPY_D'].diff().diff())
        raw_data['SLOPE_13_STATE_GOLDEN_PIT_D'] = self._clip_physical_outliers(raw_data['STATE_GOLDEN_PIT_D'].diff(13))
        raw_data['JERK_5_STATE_GOLDEN_PIT_D'] = self._clip_physical_outliers(raw_data['STATE_GOLDEN_PIT_D'].diff().diff().diff())
        raw_data['SLOPE_13_BIAS_55_D'] = self._clip_physical_outliers(raw_data['BIAS_55_D'].diff(13))
        raw_data['ACCEL_8_BIAS_55_D'] = self._clip_physical_outliers(raw_data['BIAS_55_D'].diff().diff())
        raw_data['JERK_5_NDI_14_D'] = self._clip_physical_outliers(raw_data['NDI_14_D'].diff().diff().diff())
        raw_data['SLOPE_13_PDI_14_D'] = self._clip_physical_outliers(raw_data['PDI_14_D'].diff(13))
        raw_data['JERK_5_PDI_14_D'] = self._clip_physical_outliers(raw_data['PDI_14_D'].diff().diff().diff())
        raw_data['SLOPE_13_breakout_penalty_score_D'] = self._clip_physical_outliers(raw_data['breakout_penalty_score_D'].diff(13))
        raw_data['SLOPE_13_RSI_13_D'] = self._clip_physical_outliers(raw_data['RSI_13_D'].diff(13))
        raw_data['ACCEL_8_RSI_13_D'] = self._clip_physical_outliers(raw_data['RSI_13_D'].diff().diff())
        raw_data['JERK_5_RSI_13_D'] = self._clip_physical_outliers(raw_data['RSI_13_D'].diff().diff().diff())
        raw_data['SLOPE_13_OCH_D'] = self._clip_physical_outliers(raw_data['OCH_D'].diff(13))
        raw_data['ACCEL_8_OCH_D'] = self._clip_physical_outliers(raw_data['OCH_D'].diff().diff())
        raw_data['SLOPE_13_ATR_14_D'] = self._clip_physical_outliers(raw_data['ATR_14_D'].diff(13))
        raw_data['ACCEL_8_ATR_14_D'] = self._clip_physical_outliers(raw_data['ATR_14_D'].diff().diff())
        raw_data['JERK_5_ATR_14_D'] = self._clip_physical_outliers(raw_data['ATR_14_D'].diff().diff().diff())
        raw_data['SLOPE_13_MA_FAN_EFFICIENCY_D'] = self._clip_physical_outliers(raw_data['MA_FAN_EFFICIENCY_D'].diff(13))
        raw_data['ACCEL_8_MA_FAN_EFFICIENCY_D'] = self._clip_physical_outliers(raw_data['MA_FAN_EFFICIENCY_D'].diff().diff())
        raw_data['JERK_5_MA_FAN_EFFICIENCY_D'] = self._clip_physical_outliers(raw_data['MA_FAN_EFFICIENCY_D'].diff().diff().diff())
        raw_data['JERK_5_HM_ACTIVE_ANY_D'] = self._clip_physical_outliers(raw_data['HM_ACTIVE_ANY_D'].diff().diff().diff())
        raw_data['JERK_5_SMART_MONEY_HM_COORDINATED_ATTACK_D'] = self._clip_physical_outliers(raw_data['SMART_MONEY_HM_COORDINATED_ATTACK_D'].diff().diff().diff())
        raw_data['SLOPE_13_HM_COORDINATED_ATTACK_D'] = self._clip_physical_outliers(raw_data['HM_COORDINATED_ATTACK_D'].diff(13))
        raw_data['JERK_5_HM_COORDINATED_ATTACK_D'] = self._clip_physical_outliers(raw_data['HM_COORDINATED_ATTACK_D'].diff().diff().diff())
        raw_data['JERK_5_BIAS_5_D'] = self._clip_physical_outliers(raw_data['BIAS_5_D'].diff().diff().diff())
        raw_data['ACCEL_8_BIAS_5_D'] = self._clip_physical_outliers(raw_data['BIAS_5_D'].diff().diff())
        raw_data['SLOPE_5_RSI_13_D'] = self._clip_physical_outliers(raw_data['RSI_13_D'].diff(5))
        raw_data['SLOPE_5_market_sentiment_score_D'] = self._clip_physical_outliers(raw_data['market_sentiment_score_D'].diff(5))
        raw_data['pain_index_proxy'] = 1.0 - raw_data['profit_ratio_D']
        raw_data['JERK_5_profit_ratio_D'] = self._clip_physical_outliers(raw_data['profit_ratio_D'].diff().diff().diff())
        raw_data['JERK_5_pain_index_proxy'] = raw_data['JERK_5_profit_ratio_D'] * -1.0
        raw_data['price_slope_raw'] = raw_data['SLOPE_5_market_sentiment_score_D']
        raw_data['net_mf_sum_13'] = raw_data['net_mf_amount_D'].rolling(window=13, min_periods=1).sum()
        raw_data['net_mf_sum_21'] = raw_data['net_mf_amount_D'].rolling(window=21, min_periods=1).sum()
        print(f"--- [物理导数自洽性探针] 原子化数据处理完成 @ {raw_data['close'].index[-1]} ---")
        return raw_data

    def _calculate_physics_score(self, series: pd.Series, mode: str, sensitivity: float = 1.0, window: int = 55, denoise: bool = False) -> pd.Series:
        """
        V53.1.0: 物理归一化引擎（合规增强版）。
        说明: 修复了输入非 Series 类型导致的 rolling 崩溃问题，增强了 denoise 逻辑的鲁棒性。
        """
        # 强制类型转换，防止 'int' object has no attribute 'rolling' 错误
        if not isinstance(series, pd.Series):
            series = pd.Series(float(series), index=getattr(self, 'last_df_index', series.index if hasattr(series, 'index') else []))
        if denoise:
            # 只有当 series 长度足以计算 rolling 时才进行降噪
            if len(series) >= 21:
                noise_floor = series.rolling(window=21).std().fillna(0) * 0.1
                series = series.where(series.abs() > noise_floor, 0.0)
        if mode == 'limit_low':
            return 1.0 - np.tanh(series.abs() * sensitivity)
        elif mode == 'limit_high':
            return np.tanh(series * sensitivity).clip(0, 1)
        elif mode == 'zero_focus':
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
        V24.0.2: 引入线性共振失效的能量压缩模型。
        新增：GEOM_REG_R2_D 失效因子对能量爆发临界点的动态补正。
        """
        print(f"--- [能量压缩与共振失效集成探针] @ {df_index[-1]} ---")
        # 1. 均线压缩动力学 
        resonance = self._calculate_physics_score(raw_data['MA_COHERENCE_RESONANCE_D'], mode='limit_high', sensitivity=2.0)
        # 2. 筹码集中动力学与 HAB [cite: 2, 4]
        conc_raw = raw_data['chip_concentration_ratio_D']
        conc_hab = self._calculate_historical_accumulation_buffer(conc_raw, windows=[13, 21])
        conc_slope = self._calculate_physics_score(raw_data.get('SLOPE_13_chip_concentration_ratio_D', conc_raw.diff(13)), mode='limit_high', sensitivity=10.0, denoise=True)
        # 3. 接入线性共振失效因子 
        lrf_score = self._calculate_linear_resonance_failure(df_index, raw_data)
        # 4. 结构质量校验 [cite: 2]
        struct_quality = self._calculate_physics_score(raw_data['chip_structure_state_D'], mode='limit_high', sensitivity=2.0)
        # 5. 最终合成：(均线共振 + 集中度斜率 + 线性共振失效) * 结构质量 * 存量缓冲
        # 权重分配：均线 30%，筹码 30%，共振失效 40% (作为主爆发点)
        final_energy = (resonance * 0.3 + conc_slope * 0.3 + lrf_score * 0.4) * struct_quality * (0.8 + 0.2 * conc_hab)
        print(f"  -- 均线共振分: {resonance.iloc[-1]:.4f} | 共振失效分: {lrf_score.iloc[-1]:.4f} | 最终能量分: {final_energy.iloc[-1]:.4f}")
        return final_energy.clip(0, 1)

    def _calculate_volume_exhaustion_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V53.1.1: 修复零基陷阱的量能真空组件。
        说明: 修正了 JERK_5_VPA_EFFICIENCY_D 缺失时的默认值类型，防止物理归一化引擎崩溃。
        """
        print(f"--- [量能真空与自洽性修复探针] @ {df_index[-1]} ---")
        turnover_score = self._calculate_physics_score(raw_data['turnover_rate_f_D'], mode='limit_low', sensitivity=25.0)
        trough_fill = self._calculate_physics_score(raw_data['intraday_trough_filling_degree_D'], mode='limit_high', sensitivity=3.0)
        mdb_factor = self._calculate_momentum_dissipation_balance(df_index, raw_data)
        solid_factor = self._calculate_liquidity_solidification_threshold(df_index, raw_data)
        # 修复：提供全零 Series 而非整数 0
        default_zero_series = pd.Series(0.0, index=df_index)
        vpa_jerk = self._calculate_physics_score(raw_data.get('JERK_5_VPA_EFFICIENCY_D', default_zero_series), mode='limit_low', sensitivity=20.0, denoise=True)
        mf_eff = self._calculate_physics_score(raw_data['VPA_MF_ADJUSTED_EFF_D'], mode='limit_high', sensitivity=2.0)
        base_vac = (turnover_score * 0.3 + trough_fill * 0.7)
        final_vol = base_vac * mdb_factor * (0.6 + 0.4 * solid_factor) * (0.8 + 0.2 * mf_eff) * (1.0 - 0.3 * (1.0 - vpa_jerk))
        print(f"  -- 基础地量分: {base_vac.iloc[-1]:.4f} | 量能锁存因子: {vpa_jerk.iloc[-1]:.4f}")
        return final_vol.clip(0, 1)

    def _calculate_main_force_covert_intent_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], mtf_derived_scores: Dict[str, pd.Series], weights: Dict, ambiguity_weights: Dict, _temp_debug_values: Dict) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        V27.0.2: 引入猎杀时间一致性的意图深度融合模型。
        逻辑：利用 HTC 校验协同攻击的战术纯度，过滤游资诱多杂音。 
        """
        print(f"--- [主力意图与猎杀一致性集成探针] @ {df_index[-1]} ---")
        # 1. 猎杀协同频率 (CHF) 与突变 Jerk 
        chf_base = raw_data['SMART_MONEY_HM_COORDINATED_ATTACK_D'].rolling(window=8, min_periods=1).mean()
        chf_jerk_score = self._calculate_physics_score(raw_data.get('JERK_5_SMART_MONEY_HM_COORDINATED_ATTACK_D', 0), mode='limit_high', sensitivity=25.0, denoise=True)
        # 2. 接入猎杀时间一致性因子 (HTC)
        htc_factor = self._calculate_hunting_temporal_coherence(df_index, raw_data)
        # 3. 隐蔽收集、迁移动力学与资金 HAB [cite: 3]
        stealth_score = self._calculate_physics_score(raw_data['stealth_flow_ratio_D'], mode='limit_high', sensitivity=4.0)
        migration_accel = self._calculate_physics_score(raw_data.get('ACCEL_8_intraday_cost_center_migration_D', 0), mode='limit_high', sensitivity=20.0)
        mf_hab = self._calculate_historical_accumulation_buffer(raw_data['net_mf_amount_D'], windows=[21, 34])
        # 4. 最终意图融合：(隐蔽 + 迁移 + 协同频率) * HTC 一致性因子 * 存量缓冲 
        # 权重分配：HTC 作为 25% 的核心战术质量修正项
        base_intent = (stealth_score * 0.3 + migration_accel * 0.4 + (chf_base * 0.4 + chf_jerk_score * 0.6) * 0.3)
        final_intent = base_intent * (0.75 + 0.25 * htc_factor) * (0.7 + 0.3 * mf_hab)
        print(f"  -- 基础意图分: {base_intent.iloc[-1]:.4f} | 时间一致性因子: {htc_factor.iloc[-1]:.4f} | 最终意图分: {final_intent.iloc[-1]:.4f}")
        return final_intent.clip(0, 1), {"stealth_score": stealth_score, "htc_factor": htc_factor}

    def _calculate_subdued_market_sentiment_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, sentiment_volatility_window: int, long_term_sentiment_window: int, sentiment_neutral_range: float, sentiment_pendulum_neutral_range: float, _temp_debug_values: Dict) -> pd.Series:
        """
        V50.0.2: 引入极值恐慌共振的情绪深度融合模型。
        说明: 利用 EPR 因子捕捉黄金坑末端的带血筹码收集行为，确证情绪底部的真实性。
        """
        print(f"--- [情绪与极值恐慌共振集成探针] @ {df_index[-1]} ---")
        # 1. 痛感代理与绝望脉冲 (1.0 - profit_ratio)
        pain_score = self._calculate_physics_score(raw_data['pain_index_proxy'], mode='limit_high', sensitivity=3.0)
        despair_burst = self._calculate_physics_score(raw_data.get('JERK_5_pain_index_proxy', 0), mode='limit_high', sensitivity=20.0, denoise=True)
        # 2. 存量情绪与趋势静止度
        sent_hab = self._calculate_historical_accumulation_buffer(raw_data['market_sentiment_score_D'], windows=[21])
        slope_silence = self._calculate_physics_score(raw_data.get('SLOPE_13_market_sentiment_score_D', 0), mode='zero_focus', sensitivity=50.0, denoise=True)
        # 3. 接入物理分量：空头力竭(SED)、二极化接管(OMB) 与 极值恐慌共振(EPR)
        short_exhaustion = self._calculate_short_exhaustion_divergence(df_index, raw_data)
        bipolar_gain = self._calculate_oversold_momentum_bipolarization(df_index, raw_data)
        panic_resonance = self._calculate_extreme_panic_resonance(df_index, raw_data)
        # 4. 微观有序化增益与获利盘清洗校验
        order_gain = self._calculate_micro_order_gain(df_index, raw_data)
        cleanse_score = self._calculate_physics_score(raw_data['winner_rate_D'], mode='limit_low', sensitivity=15.0)
        # 5. 非线性情绪融合合成：(基础底迷 * 锁仓有序性) * (1 + 物理力竭 + 二极化接管 + 恐慌共振) + 绝望补偿
        # 权重分配：EPR 作为 20% 的极值修正项，强化坑底反转信号
        base_subdued = (pain_score.pow(0.4) * slope_silence.pow(0.3) * sent_hab.pow(0.3) * cleanse_score.pow(0.2))
        final_sentiment = (base_subdued * (1.0 + 0.3 * order_gain) * (0.4 + 0.2 * short_exhaustion + 0.2 * bipolar_gain + 0.2 * panic_resonance) + 0.25 * despair_burst)
        print(f"  -- 恐慌共振分: {panic_resonance.iloc[-1]:.4f} | 最终情绪真空分: {final_sentiment.iloc[-1]:.4f}")
        _temp_debug_values["市场情绪矩阵"] = {"base": base_subdued, "panic_resonance": panic_resonance}
        return final_sentiment.clip(0, 1)

    def _calculate_breakout_readiness_component(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], weights: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V44.0.2: 引入博弈中性化因子的突破准备度终极矩阵。
        说明: 整合物理动能、摩擦衰减、执行质量与空间状态，实现起爆降维打击。
        """
        print(f"--- [突破准备度-终极矩阵探针] @ {df_index[-1]} ---")
        grp_score = self._calculate_gravitational_regression_pull(df_index, raw_data)
        plr_score = self._calculate_phase_locked_resonance(df_index, raw_data)
        sed_score = self._calculate_short_exhaustion_divergence(df_index, raw_data)
        ssd_score = self._calculate_seat_scatter_decay(df_index, raw_data)
        egd_score = self._calculate_efficiency_gradient_dissipation(df_index, raw_data)
        sope_score = self._calculate_split_order_pulse_entropy(df_index, raw_data)
        aeo_score = self._calculate_abnormal_energy_overflow(df_index, raw_data)
        neutral_score = self._calculate_game_neutralization_modulator(df_index, raw_data)
        aded_score = self._calculate_amount_distribution_entropy_delta(df_index, raw_data)
        well_collapse = self._calculate_potential_well_collapse(df_index, raw_data)
        long_awakening = self._calculate_long_awakening_threshold(df_index, raw_data)
        stress_test = self._calculate_level_stress_test_modulator(df_index, raw_data)
        momentum_part = (grp_score * 0.5 + plr_score * 0.5)
        friction_part = (sed_score * 0.5 + ssd_score * 0.5)
        quality_part = (egd_score * 0.3 + sope_score * 0.4 + aeo_score * 0.3)
        state_part = (well_collapse * 0.25 + long_awakening * 0.25 + aded_score * 0.15 + stress_test * 0.15 + neutral_score * 0.2)
        acc_hab = self._calculate_historical_accumulation_buffer(raw_data['accumulation_signal_score_D'], windows=[21, 34])
        readiness = (momentum_part * 0.15 + friction_part * 0.15 + quality_part * 0.2 + state_part * 0.4 + 0.1) * (0.8 + 0.2 * acc_hab)
        print(f"  -- 动能子层:{momentum_part.iloc[-1]:.4f} | 状态子层:{state_part.iloc[-1]:.4f} | 最终准备度:{readiness.iloc[-1]:.4f}")
        return readiness.clip(0, 1)

    def _calculate_market_regulator_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], params: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        V15.0.1: 行业轮动 Jerk 与宏观起爆调节器。
        逻辑：利用行业排名加速度的二级突变捕捉“个股寂静、板块点火”的临界瞬间。
        """
        print(f"--- [宏观起爆调节探针] @ {df_index[-1]} ---")
        # 1. 行业预热 HAB 缓冲：确保板块热度具备存量底蕴 
        sector_preheat = raw_data['industry_preheat_score_D']
        sector_hab = self._calculate_historical_accumulation_buffer(sector_preheat, windows=[13, 21])
        # 2. 板块轮动 Jerk：行业排名加速度的阶梯式跳升 
        sector_jerk = raw_data.get('JERK_5_industry_rank_accel_D', raw_data['industry_rank_accel_D'].diff().diff())
        # 降噪：过滤行业排名的细微波动，锁定真正的点火信号
        clean_sector_jerk = sector_jerk.where(sector_jerk > sector_jerk.rolling(21).std(), 0.0)
        sector_ignite_score = self._calculate_physics_score(clean_sector_jerk, mode='limit_high', sensitivity=10.0)
        # 3. 个股-板块背离度：个股越静止 (Slope->0)，板块越点火 (Jerk->High)，共振越强
        stock_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        macro_resonance = (sector_ignite_score * stock_calm).pow(0.5)
        # 4. 环境风险抑制 (Fermi-Dirac 函数) [cite: 1]
        adx_raw = raw_data['ADX_14_D']
        adx_supp = 1.0 / (1.0 + np.exp(0.4 * (adx_raw - 28.0)))
        print(f"  -- 行业HAB系数: {sector_hab.iloc[-1]:.4f} | 板块点火分: {sector_ignite_score.iloc[-1]:.4f} | 宏观共振: {macro_resonance.iloc[-1]:.4f}")
        # 最终调节系数：行业存量 * 宏观起爆奖赏 * 环境抑制
        # 基础系数 1.0，起爆瞬时最高可获得 1.5 倍加成
        final_modulator = (0.7 + 0.3 * sector_hab) * (1.0 + 0.5 * macro_resonance) * adx_supp
        return final_modulator.clip(0.2, 1.5)

    def _perform_final_fusion(self, df_index: pd.Index, component_scores: dict[str, pd.Series], final_fusion_weights: dict, price_calmness_params: dict, main_force_control_params: dict, raw_data: dict[str, pd.Series], _temp_debug_values: Dict) -> pd.Series:
        """
        V22.0.2: 融合猎杀协同频率与效率熵损的终极动力学模型。
        逻辑：利用协同猎杀因子作为最终爆破增益，确保在主力共识达成时快速响应。
        """
        print(f"--- [最终融合 协同猎杀动力学探针] @ {df_index[-1]} ---")
        # 1. 一致性熵权与基础几何分 
        ewd_factor = self._calculate_consensus_entropy(component_scores)
        base_score = _robust_geometric_mean(component_scores, final_fusion_weights, df_index)
        # 2. 动力学压差与资金 HAB 缓冲 
        ext_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        int_boil = self._calculate_physics_score(raw_data.get('ACCEL_8_intraday_cost_center_migration_D', 0), mode='limit_high', sensitivity=20.0)
        pressure_mult = 1.0 + 1.2 * (ext_calm * int_boil)
        mf_hab = self._calculate_historical_accumulation_buffer(raw_data['net_mf_amount_D'], windows=[13, 21, 34])
        # 3. 猎杀协同增益：从意图组件中提取共识因子 
        # 重新计算频率 Jerk 以确保逻辑自洽
        chf_base = raw_data['SMART_MONEY_HM_COORDINATED_ATTACK_D'].rolling(window=8, min_periods=1).mean()
        hunt_jerk = self._calculate_physics_score(raw_data.get('JERK_5_SMART_MONEY_HM_COORDINATED_ATTACK_D', 0), mode='limit_high', sensitivity=25.0, denoise=True)
        hunting_boost = 1.0 + 0.4 * (chf_base * hunt_jerk).pow(0.5)
        # 4. 环境抑制与熔断校验 
        pressure_backtest = self._calculate_pressure_backtest_modulator(df_index, raw_data)
        activity_veto = ((raw_data['main_force_activity_index_D'] - 5.0) * 0.1).clip(0, 1)
        struct_veto = self._calculate_physics_score(raw_data['chip_structure_state_D'], mode='limit_high', sensitivity=2.0)
        # 5. 最终合成 
        final_score = base_score * ewd_factor * (0.7 + 0.3 * mf_hab) * pressure_mult * pressure_backtest * activity_veto * struct_veto * hunting_boost
        print(f"  -- 基础分: {base_score.iloc[-1]:.4f} | 猎杀增益: {hunting_boost.iloc[-1]:.4f} | 最终输出: {final_score.iloc[-1]:.4f}")
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

    def _calculate_consensus_entropy(self, scores_dict: dict[str, pd.Series]) -> pd.Series:
        """
        V40.0.0: 基于多维时空互信息熵的共振校验器。
        说明: 通过计算各物理分量的离散度与时序耦合度，动态识别“伪共振”信号。
        """
        print(f"--- [时空互信息熵共振探针] 启动协同校验 ---")
        df_scores = pd.concat(scores_dict.values(), axis=1)
        dispersion = df_scores.std(axis=1).fillna(1.0)
        corr_matrix = df_scores.rolling(window=5).corr()
        coherence = corr_matrix.groupby(level=0).mean().mean(axis=1).fillna(0.0)
        disp_decay = np.exp(- (dispersion * 2.5) ** 2)
        final_decay = disp_decay * (0.6 + 0.4 * coherence.clip(0, 1))
        print(f"  -- 离散度均值: {dispersion.mean():.4f} | 时序耦合度: {coherence.iloc[-1]:.4f} | 最终熵权因子: {final_decay.iloc[-1]:.4f}")
        return final_decay.clip(0, 1)

    def _calculate_pressure_backtest_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V19.0.1: 压力回测模型 (Pressure Backtest)。
        逻辑：量化惩罚分对价格斜率的“衰减效应”，识别高压下的虚假突破。
        """
        print(f"--- [压力回测压测探针] @ {df_index[-1]} ---")
        # 1. 惩罚分强度与斜率
        penalty_raw = raw_data['breakout_penalty_score_D']
        penalty_slope = raw_data.get('SLOPE_13_breakout_penalty_score_D', penalty_raw.diff(13))
        # 2. 压力消化 HAB：如果长时间高压股价不跌，说明压力在被消化
        penalty_hab = self._calculate_historical_accumulation_buffer(penalty_raw, windows=[21])
        # 3. 衰减因子计算：当惩罚分高 且 斜率正在上升(>0) 时，产生强力压制
        # 物理含义：阻力正在变厚
        resistance_intensity = self._calculate_physics_score(penalty_raw * (1.0 + np.maximum(0, penalty_slope)), mode='limit_high', sensitivity=1.5)
        # 4. 动能衰减模拟：对比价格斜率与阻力强度
        price_v = raw_data['price_slope_raw']
        # 如果价格向上冲(v>0)但阻力极大，则进行非线性惩罚
        backtest_factor = 1.0 - (resistance_intensity * np.tanh(np.maximum(0, price_v) * 10.0))
        # 5. 存量修正：消化后的压力不再是阻碍
        final_modulator = (backtest_factor * (1.0 - penalty_hab)) + penalty_hab
        print(f"  -- 惩罚分强度: {penalty_raw.iloc[-1]:.4f} | 阻力系数: {resistance_intensity.iloc[-1]:.4f} | HAB消化度: {penalty_hab.iloc[-1]:.4f}")
        return final_modulator.clip(0.2, 1.0)

    def _calculate_level_stress_test_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V23.0.1: 盘口关键位压测模型 (Level Stress Test)。
        逻辑：量化主力在阻力位利用开收盘加速度进行“暴力测压”的物理强度。
        """
        print(f"--- [关键位压测探针] @ {df_index[-1]} ---")
        # 1. OCH 加速度突变捕捉 (Jerk) 
        och_accel = raw_data['OCH_ACCELERATION_D']
        och_jerk = raw_data.get('JERK_5_OCH_ACCELERATION_D', och_accel.diff().diff())
        # 降噪：过滤背景噪音，锁定真正的开收盘突变点火
        och_jerk_score = self._calculate_physics_score(och_jerk, mode='limit_high', sensitivity=20.0, denoise=True)
        # 2. 关键阻力位判定：阻力强度 * 价格位置 
        res_strength = raw_data['resistance_strength_D']
        # 计算价格相对于关键均线的压缩度 (MA21/MA55) 
        ma21_proximity = 1.0 - np.minimum(1.0, np.abs(raw_data['price_vs_ma_21_ratio_D'] - 1.0) * 20.0)
        ma55_proximity = 1.0 - np.minimum(1.0, np.abs(raw_data['price_vs_ma_55_ratio_D'] - 1.0) * 20.0)
        level_weight = np.maximum(ma21_proximity, ma55_proximity)
        # 3. 压测评分：仅在阻力位附近且有阻力时，OCH 加速度才被视为“压测”
        stress_test_score = och_jerk_score * level_weight * self._calculate_physics_score(res_strength, mode='limit_high', sensitivity=1.0)
        print(f"  -- OCH加速度: {och_accel.iloc[-1]:.4f} | 压测Jerk分: {och_jerk_score.iloc[-1]:.4f} | 均线临界度: {level_weight.iloc[-1]:.4f}")
        return stress_test_score.clip(0, 1)

    def _calculate_linear_resonance_failure(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V24.0.1: 线性共振失效模型 (Linear Resonance Failure)。
        逻辑：量化价格结构从“线性稳定”向“非线性爆发”转换的物理瞬间。
        """
        print(f"--- [线性共振失效探针] @ {df_index[-1]} ---")
        # 1. 拟合度及其结构瓦解加速度 
        r2_raw = raw_data['GEOM_REG_R2_D']
        r2_accel = raw_data.get('ACCEL_8_GEOM_REG_R2_D', r2_raw.diff().diff())
        # 2. 拟合度存量 HAB：判定此前是否处于极高的共振态 
        r2_hab = self._calculate_historical_accumulation_buffer(r2_raw, windows=[21])
        # 3. 失效判定：当加速度转负 (结构瓦解) 且此前线性度高
        # 物理含义：稳固的线性横盘结构正在出现裂纹
        failure_burst = self._calculate_physics_score(r2_accel.abs(), mode='limit_high', sensitivity=15.0, denoise=True)
        # 4. 结合价格静止：失效必须发生在价格尚未大幅偏离时
        reg_slope = raw_data['GEOM_REG_SLOPE_D']
        slope_calm = self._calculate_physics_score(reg_slope, mode='zero_focus', sensitivity=40.0, denoise=True)
        # 5. 最终失效分：历史高线性存量 * 当前瓦解加速度 * 价格斜率静止度
        failure_score = r2_hab * failure_burst * slope_calm
        print(f"  -- R2原始值: {r2_raw.iloc[-1]:.4f} | R2存量HAB: {r2_hab.iloc[-1]:.4f} | 失效突变分: {failure_burst.iloc[-1]:.4f}")
        return failure_score.clip(0, 1)

    def _calculate_micro_order_gain(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V25.0.1: 微观有序化增益模型 (Micro-Order Gain)。
        逻辑：量化价格熵的单调递减速率，识别主力高效率锁仓洗盘 。
        """
        print(f"--- [微观有序化探针] @ {df_index[-1]} ---")
        # 1. 价格熵原始值及其下降斜率 
        entropy_raw = raw_data['PRICE_ENTROPY_D']
        entropy_slope = raw_data.get('SLOPE_13_PRICE_ENTROPY_D', entropy_raw.diff(13))
        # 2. 有序化判定：斜率为负且越小（下降越快），有序化程度越高
        # 物理含义：系统熵减，代表主力控盘力增强 
        orderly_score = self._calculate_physics_score(entropy_slope, mode='limit_low', sensitivity=15.0, denoise=True)
        # 3. 价格熵存量 HAB：判定此前是否处于长期的有序化通道中 [cite: 4]
        entropy_hab = self._calculate_historical_accumulation_buffer(entropy_raw, windows=[21])
        # 4. 结合价格静止：有序化增益必须发生在“风暴眼”的静止期
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        # 5. 最终增益分：有序化趋势 * 价格静止度 * (1 - 历史高熵存量)
        # 如果历史存量熵极高，说明是单纯的沉寂；只有从高熵向低熵转化的过程才是增益 
        gain_score = orderly_score * price_calm * (1.0 - entropy_hab)
        print(f"  -- 价格熵: {entropy_raw.iloc[-1]:.4f} | 有序化斜率分: {orderly_score.iloc[-1]:.4f} | 存量熵缓冲: {entropy_hab.iloc[-1]:.4f}")
        return gain_score.clip(0, 1)

    def _calculate_momentum_dissipation_balance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V26.0.1: 动量耗散平衡模型 (Momentum Dissipation Balance)。
        逻辑：量化量价加速度向零轴收敛的对称性，识别“高能平衡”下的纯净静止。
        """
        print(f"--- [动量耗散平衡探针] @ {df_index[-1]} ---")
        # 1. 量价加速度及其收敛脉冲 (Jerk)
        vpa_accel = raw_data['VPA_ACCELERATION_5D']
        vpa_accel_jerk = raw_data.get('JERK_5_VPA_ACCELERATION_5D', vpa_accel.diff().diff())
        # 2. 耗散锁定：使用高斯核锁定加速度绝对零点
        # 物理含义：加速度越接近 0，系统动能耗散越彻底
        dissipation_focus = self._calculate_physics_score(vpa_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        # 3. 稳定性校验：排除在零轴附近的剧烈跳变
        # 物理含义：低 Jerk 代表耗散过程平滑、有序
        jerk_silence = self._calculate_physics_score(vpa_accel_jerk, mode='zero_focus', sensitivity=60.0, denoise=True)
        # 4. 结合价格静止环境
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        # 5. 最终合成：耗散锁定 * 稳定性因子 * 价格静止度
        mdb_score = dissipation_focus * jerk_silence * price_calm
        print(f"  -- VPA加速度: {vpa_accel.iloc[-1]:.4f} | 耗散锁定分: {dissipation_focus.iloc[-1]:.4f} | 耗散平衡分: {mdb_score.iloc[-1]:.4f}")
        return mdb_score.clip(0, 1)

    def _calculate_hunting_temporal_coherence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V27.0.1: 猎杀时间一致性模型 (Hunting Temporal Coherence)。
        逻辑：量化协同攻击信号的时域分布熵，识别主力部队的有序集结节奏。 
        """
        print(f"--- [猎杀一致性探针] @ {df_index[-1]} ---")
        # 1. 提取协同攻击强度及其突变 Jerk 
        attack_raw = raw_data['HM_COORDINATED_ATTACK_D']
        attack_jerk = raw_data.get('JERK_5_HM_COORDINATED_ATTACK_D', attack_raw.diff().diff())
        # 2. 计算时域一致性：信号强度的滚动标准差的倒数映射
        # 物理含义：攻击强度越平稳递增，代表资金集结越有计划 
        temporal_stability = 1.0 - self._calculate_physics_score(attack_raw.rolling(window=8).std(), mode='limit_high', sensitivity=2.0)
        # 3. 节奏感判定：捕捉加速度的有序性 (Jerk Silence)
        rhythm_score = self._calculate_physics_score(attack_jerk, mode='zero_focus', sensitivity=50.0, denoise=True)
        # 4. 顶级席位参与度补正 
        top_tier_activity = self._calculate_physics_score(raw_data['HM_ACTIVE_TOP_TIER_D'], mode='limit_high', sensitivity=2.0)
        # 5. 最终一致性系数：平稳性 * 节奏感 * 席位质量
        htc_score = temporal_stability * rhythm_score * (0.8 + 0.2 * top_tier_activity)
        print(f"  -- 攻击强度: {attack_raw.iloc[-1]:.4f} | 稳定性分: {temporal_stability.iloc[-1]:.4f} | 一致性最终分: {htc_score.iloc[-1]:.4f}")
        return htc_score.clip(0, 1)

    def _calculate_liquidity_solidification_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V28.0.1: 流动性固化阈值模型 (Liquidity Solidification Threshold)。
        逻辑：量化换手率在低位的结构性稳定趋势，识别筹码从浮动向固化相变的瞬间。
        """
        print(f"--- [流动性固化探针] @ {df_index[-1]} ---")
        # 1. 换手稳定性索引及其斜率
        stability_raw = raw_data['TURNOVER_STABILITY_INDEX_D']
        stability_slope = raw_data.get('SLOPE_13_TURNOVER_STABILITY_INDEX_D', stability_raw.diff(13))
        # 2. 固化判定：稳定性处于高位 (limit_high) 且 稳定性正在递增 (slope > 0)
        stability_score = self._calculate_physics_score(stability_raw, mode='limit_high', sensitivity=1.5)
        slope_growth = self._calculate_physics_score(stability_slope, mode='limit_high', sensitivity=10.0, denoise=True)
        # 3. 稳定性 HAB 缓冲：判定固化的时域深度
        stability_hab = self._calculate_historical_accumulation_buffer(stability_raw, windows=[13, 21])
        # 4. 结合地量环境：固化必须发生在低换手背景下
        turnover_low = self._calculate_physics_score(raw_data['turnover_rate_f_D'], mode='limit_low', sensitivity=25.0)
        # 5. 最终固化因子：稳定性 * 增长趋势 * 存量缓冲 * 地量背景
        solidification_factor = stability_score * (0.7 + 0.3 * slope_growth) * stability_hab * turnover_low
        print(f"  -- 稳定性值: {stability_raw.iloc[-1]:.4f} | 稳定性斜率分: {slope_growth.iloc[-1]:.4f} | 固化最终分: {solidification_factor.iloc[-1]:.4f}")
        return solidification_factor.clip(0, 1)

    def _calculate_amount_distribution_entropy_delta(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V29.0.1: 成交金额分布熵增模型 (Amount Distribution Entropy Delta)。
        逻辑：量化成交额在价格空间的分布变化，通过熵减识别主力的“拦截式吸筹” [cite: 1, 3]。
        """
        print(f"--- [成交分布熵增探针] @ {df_index[-1]} ---")
        # 1. 计算成交额的 13 日时空分布熵 (Proxy)
        amount = raw_data['amount_D']
        price_grid = raw_data['price_grid_D']
        # 联合计算金额与价格格点的互信息熵代理值
        # 物理含义：值越低，代表成交额越集中在特定的价格格点上
        dist_entropy = (amount / (price_grid + 1e-9)).rolling(window=13).std() / (amount.rolling(window=13).mean() + 1e-9)
        # 2. 计算熵增斜率 (Entropy Delta)
        # 负斜率代表熵减，即成交分布正在变得有序、集中
        entropy_delta = dist_entropy.diff(5)
        # 3. 映射为有序化得分：熵减越剧烈，得分越高
        interceptive_score = self._calculate_physics_score(entropy_delta, mode='limit_low', sensitivity=20.0, denoise=True)
        # 4. 结合 HAB 缓冲：判定有序化的持续性底蕴
        entropy_hab = self._calculate_historical_accumulation_buffer(dist_entropy, windows=[21])
        print(f"  -- 分布熵代理: {dist_entropy.iloc[-1]:.4f} | 熵增斜率: {entropy_delta.iloc[-1]:.4f} | 拦截吸筹分: {interceptive_score.iloc[-1]:.4f}")
        return (interceptive_score * (1.0 - entropy_hab)).clip(0, 1)

    def _calculate_seat_scatter_decay(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V30.0.1: 席位散点衰减模型 (Seat Scatter Decay)。
        说明: 计算任何席位  与顶级席位  活跃度的差值变化，通过 Jerk 识别噪音资金的加速离场。
        """
        print(f"--- [席位散点衰减探针] @ {df_index[-1]} ---")
        # 1. 定义席位散点：总活跃度 - 核心活跃度 
        any_act = raw_data['HM_ACTIVE_ANY_D']
        top_act = raw_data['HM_ACTIVE_TOP_TIER_D']
        scatter_raw = (any_act - top_act).clip(lower=0)
        # 2. 提取散点 Jerk：捕捉非线性消失瞬间
        scatter_jerk = scatter_raw.diff().diff().diff()
        # 3. 映射为衰减分：Jerk 越负（消失越快），得分越高
        decay_score = self._calculate_physics_score(scatter_jerk, mode='limit_low', sensitivity=15.0, denoise=True)
        # 4. 结合价格静止环境：噪音消失必须发生在横盘静止期 
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        # 5. 最终衰减分：消失速度 * 环境静止度
        final_decay = decay_score * price_calm
        print(f"  -- 散点活跃度: {scatter_raw.iloc[-1]:.4f} | 衰减Jerk分: {decay_score.iloc[-1]:.4f} | 最终衰减分: {final_decay.iloc[-1]:.4f}")
        return final_decay.clip(0, 1)

    def _calculate_gravitational_regression_pull(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V31.0.1: 引力回归拉力模型 (Gravitational Regression Pull)。
        说明: 量化股价偏离 55 日均线产生的物理拉力，识别超跌相变的引力爆发点。 
        """
        print(f"--- [引力回归拉力探针] @ {df_index[-1]} ---")
        # 1. 55日乖离率及其回归加速度 
        bias_raw = raw_data['BIAS_55_D']
        bias_accel = raw_data.get('ACCEL_8_BIAS_55_D', bias_raw.diff().diff())
        # 2. 势能存量 HAB：判定此前是否在深度负乖离区停留（积蓄能量）
        # 物理含义：负乖离越久，回归引力越稳定
        bias_hab = self._calculate_historical_accumulation_buffer(bias_raw.clip(upper=0), windows=[21])
        # 3. 引力激活：当加速度由正向（回归方向）爆发时得分
        # 注意：对于负乖离，加速度为正则代表正在向零轴（均线）靠拢
        gravity_ignite = self._calculate_physics_score(bias_accel, mode='limit_high', sensitivity=15.0, denoise=True)
        # 4. 深度修正：乖离越深，基础引力越强
        depth_score = self._calculate_physics_score(bias_raw, mode='limit_low', sensitivity=10.0)
        # 5. 最终引力拉力：深度权重 * 势能存量 * 激活加速度
        pull_score = depth_score * bias_hab * gravity_ignite
        print(f"  -- BIAS_55: {bias_raw.iloc[-1]:.4f} | 势能HAB: {bias_hab.iloc[-1]:.4f} | 引力激活分: {gravity_ignite.iloc[-1]:.4f}")
        return pull_score.clip(0, 1)

    def _calculate_short_exhaustion_divergence(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V32.0.1: 空头力竭背离模型 (Short Exhaustion Divergence)。
        说明: 量化空头趋向指标在价格静止期的加速崩塌，识别空头抛压的物理性终结。
        """
        print(f"--- [空头力竭背离探针] @ {df_index[-1]} ---")
        # 1. 提取空头指标及其加速崩塌脉冲 (Jerk)
        ndi_raw = raw_data['NDI_14_D']
        ndi_jerk = raw_data.get('JERK_5_NDI_14_D', ndi_raw.diff().diff().diff())
        # 2. 映射力竭分：Jerk 越负（NDI 下滑加速度越快），力竭感越强
        exhaustion_score = self._calculate_physics_score(ndi_jerk, mode='limit_low', sensitivity=20.0, denoise=True)
        # 3. 价格静止环境校验：背离必须发生在价格横盘期
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        # 4. NDI 高位存量校验：空头必须曾经很强大，其力竭才具备反转意义
        ndi_hab = self._calculate_historical_accumulation_buffer(ndi_raw, windows=[21])
        # 5. 最终背离分：力竭脉冲 * 环境静止度 * NDI存量意识
        divergence_score = exhaustion_score * price_calm * ndi_hab
        print(f"  -- NDI原始值: {ndi_raw.iloc[-1]:.4f} | 力竭Jerk分: {exhaustion_score.iloc[-1]:.4f} | 背离最终分: {divergence_score.iloc[-1]:.4f}")
        return divergence_score.clip(0, 1)

    def _calculate_long_awakening_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V33.0.1: 多头觉醒阈值模型 (Long Awakening Threshold)。
        说明: 量化多头趋向指标从低位钝化向主动进攻转化的动力学特征 。
        """
        print(f"--- [多头觉醒阈值探针] @ {df_index[-1]} ---")
        # 1. 提取多头指标及其点火脉冲 (Jerk)
        pdi_raw = raw_data['PDI_14_D']
        pdi_slope = raw_data.get('SLOPE_13_PDI_14_D', pdi_raw.diff(13))
        pdi_jerk = raw_data.get('JERK_5_PDI_14_D', pdi_raw.diff().diff().diff())
        # 2. 觉醒连续性判定：Slope 持续为正且递增
        awakening_continuity = self._calculate_physics_score(pdi_slope, mode='limit_high', sensitivity=10.0, denoise=True)
        # 3. 点火爆发力：Jerk 出现正向突变
        awakening_ignite = self._calculate_physics_score(pdi_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        # 4. 存量压抑校验：多头必须此前处于低位 (HAB)，其觉醒才具备爆发力
        # 使用 HAB 判定 PDI 此前 21 日的低迷程度 (1 - PDI_HAB)
        pdi_hab = self._calculate_historical_accumulation_buffer(pdi_raw, windows=[21])
        # 5. 最终觉醒分：连续性权重 * 点火爆发力 * 存量压抑系数
        awakening_score = awakening_continuity * awakening_ignite * (1.0 - pdi_hab)
        print(f"  -- PDI原始值: {pdi_raw.iloc[-1]:.4f} | 觉醒斜率分: {awakening_continuity.iloc[-1]:.4f} | 觉醒最终分: {awakening_score.iloc[-1]:.4f}")
        return awakening_score.clip(0, 1)

    def _calculate_abnormal_energy_overflow(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V34.0.1: 异常能量溢出模型 (Abnormal Energy Overflow)。
        说明: 量化主力修正效率相对于成交额的非线性突变，识别隐秘的高效率建仓行为 。
        """
        print(f"--- [异常能量溢出探针] @ {df_index[-1]} ---")
        # 1. 提取主力修正效率及其 Jerk 脉冲
        mf_eff = raw_data['VPA_MF_ADJUSTED_EFF_D']
        eff_jerk = raw_data.get('JERK_5_VPA_MF_ADJUSTED_EFF_D', mf_eff.diff().diff().diff())
        # 2. 映射溢出脉冲：Jerk 越高（效率提升加速度越快），能量溢出感越强 
        overflow_ignite = self._calculate_physics_score(eff_jerk, mode='limit_high', sensitivity=35.0, denoise=True)
        # 3. 能量背离判定：溢出必须发生在量能萎缩或平稳期
        amount_calm = self._calculate_physics_score(raw_data['amount_D'], mode='limit_low', sensitivity=2.0)
        # 4. 结合价格静止：此时的效率突变具有最高吸筹权重 [cite: 1]
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        # 5. 最终溢出分：效率脉冲 * 量能低位度 * 价格静止度
        overflow_score = overflow_ignite * amount_calm * price_calm
        print(f"  -- 主力修正效率: {mf_eff.iloc[-1]:.4f} | 效率Jerk分: {overflow_ignite.iloc[-1]:.4f} | 能量溢出分: {overflow_score.iloc[-1]:.4f}")
        return overflow_score.clip(0, 1)

    def _calculate_phase_locked_resonance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V35.0.1: 相位锁定共振模型 (Phase-Locked Resonance)。
        说明: 量化价格加速度与量价加速度的时序协同度，识别多维动能同步锁定的反转瞬间 。
        """
        print(f"--- [相位锁定共振探针] @ {df_index[-1]} ---")
        # 1. 提取量价加速度与价格加速度 
        vpa_accel = raw_data['VPA_ACCELERATION_5D']
        price_accel = raw_data['MA_ACCELERATION_EMA_55_D']
        # 2. 动能同步收敛锁定：使用高斯核锁定两者同时趋近于零的深度 
        vpa_focus = self._calculate_physics_score(vpa_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        price_focus = self._calculate_physics_score(price_accel, mode='zero_focus', sensitivity=40.0, denoise=True)
        # 3. 相位协同判定：计算两者的 5 日滚动余弦相似度 (代理相位差)
        # 物理含义：当两者同步波动或同步收敛时，余弦相似度趋于 1 
        resonance_sim = (vpa_accel * price_accel).rolling(window=5).mean() / (vpa_accel.abs().rolling(window=5).mean() * price_accel.abs().rolling(window=5).mean() + 1e-9)
        # 4. 结合 Jerk 稳定性：排除由于剧烈跳变产生的虚假共振 
        vpa_jerk = self._calculate_physics_score(raw_data.get('JERK_5_VPA_ACCELERATION_5D', 0), mode='zero_focus', sensitivity=60.0, denoise=True)
        # 5. 最终共振分：收敛锁定 * 相位协同 * 稳定性
        plr_score = vpa_focus * price_focus * (0.5 + 0.5 * resonance_sim.clip(0, 1)) * vpa_jerk
        print(f"  -- VPA加速度: {vpa_accel.iloc[-1]:.4f} | 价格加速度: {price_accel.iloc[-1]:.4f} | 相位共振分: {plr_score.iloc[-1]:.4f}")
        return plr_score.clip(0, 1)

    def _calculate_split_order_pulse_entropy(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V36.0.1: 拆单频率脉冲熵模型 (Split-Order Pulse Entropy)。
        说明: 量化异常成交比的 Jerk 稳定性，识别主力自动化建仓算法的规律性脉冲。 
        """
        print(f"--- [拆单脉冲熵探针] @ {df_index[-1]} ---")
        # 1. 提取分笔异常成交比及其 Jerk 导数 
        abnormal_raw = raw_data['tick_abnormal_volume_ratio_D']
        abnormal_jerk = raw_data.get('JERK_5_tick_abnormal_volume_ratio_D', abnormal_raw.diff().diff().diff())
        # 2. 计算时域有序度：Jerk 的滚动变异系数的倒数
        # 物理含义：Jerk 波动越稳定，代表脉冲节奏越有规律，算法特征越明显 
        jerk_std = abnormal_jerk.rolling(window=8).std()
        jerk_mean = abnormal_jerk.abs().rolling(window=8).mean()
        pulse_orderly = 1.0 / (1.0 + (jerk_std / (jerk_mean + 1e-9)))
        # 3. 映射为有序化得分：利用物理归一化引擎锁定规律脉冲 
        order_score = self._calculate_physics_score(pulse_orderly, mode='limit_high', sensitivity=5.0, denoise=True)
        # 4. 结合价格静止：算法吸筹必须发生在“风暴眼”静止期 
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        # 5. 最终拆单熵增益：有序度 * 价格静止度 
        sope_gain = order_score * price_calm
        print(f"  -- 异常成交比: {abnormal_raw.iloc[-1]:.4f} | 脉冲有序分: {order_score.iloc[-1]:.4f} | 拆单熵最终分: {sope_gain.iloc[-1]:.4f}")
        return sope_gain.clip(0, 1)

    def _calculate_efficiency_gradient_dissipation(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V37.0.1: 效率梯度耗散模型 (Efficiency Gradient Dissipation)。
        说明: 量化主力修正效率的斜率收敛度，识别“高能锁筹”下的平滑能量耗散特征 [cite: 1, 3, 4]。
        """
        print(f"--- [效率梯度耗散探针] @ {df_index[-1]} ---")
        # 1. 提取主力修正效率及其动力学梯度 (Slope, Accel)
        mf_eff = raw_data['VPA_MF_ADJUSTED_EFF_D']
        eff_slope = raw_data.get('SLOPE_13_VPA_MF_ADJUSTED_EFF_D', mf_eff.diff(13))
        eff_accel = raw_data.get('ACCEL_8_VPA_MF_ADJUSTED_EFF_D', eff_slope.diff(8))
        # 2. 耗散稳定性：斜率的波动率越低，代表能量耗散越有序
        slope_stability = 1.0 - self._calculate_physics_score(eff_slope.rolling(window=8).std(), mode='limit_high', sensitivity=2.0)
        # 3. 梯度锁定：使用高斯核锁定加速度向零轴回归的瞬间
        # 物理含义：加速度趋零代表效率变化进入稳态 [cite: 4]
        accel_lock = self._calculate_physics_score(eff_accel, mode='zero_focus', sensitivity=50.0, denoise=True)
        # 4. 结合主力活跃度：高效率耗散必须伴随主力基础活跃 [cite: 2]
        mf_activity = self._calculate_physics_score(raw_data['main_force_activity_index_D'], mode='limit_high', sensitivity=2.0)
        # 5. 最终耗散平衡分：稳定性 * 梯度锁定 * 活跃度补正
        egd_score = slope_stability * accel_lock * (0.8 + 0.2 * mf_activity)
        print(f"  -- 修正效率值: {mf_eff.iloc[-1]:.4f} | 梯度锁定分: {accel_lock.iloc[-1]:.4f} | 耗散平衡分: {egd_score.iloc[-1]:.4f}")
        return egd_score.clip(0, 1)

    def _calculate_potential_well_collapse(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V38.0.1: 势能阱塌缩模型 (Potential Well Collapse)。
        说明: 量化黄金坑状态的 Jerk 脉冲，识别股价从“引力囚禁”向“逃逸加速”转换的相变瞬间。
        """
        print(f"--- [势能阱塌缩探针] @ {df_index[-1]} ---")
        # 1. 提取黄金坑状态及其突变脉冲 (Jerk)
        pit_state = raw_data['STATE_GOLDEN_PIT_D']
        pit_jerk = raw_data.get('JERK_5_STATE_GOLDEN_PIT_D', pit_state.diff().diff().diff())
        # 2. 逃逸动能映射：仅在坑位激活时，Jerk 的正向爆发代表逃逸加速
        # 物理含义：Jerk 越高，摆脱坑底引力的力量越强
        escape_ignite = self._calculate_physics_score(pit_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        # 3. 引力囚禁判定：如果 Jerk 归零，代表系统处于稳态陷阱中
        trap_lock = self._calculate_physics_score(pit_jerk, mode='zero_focus', sensitivity=50.0, denoise=True)
        # 4. 价格静止环境校验
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        # 5. 最终塌缩分：状态活性 * (逃逸动能 * 0.8 + 囚禁抑制 * 0.2) * 环境静止度
        # 逻辑：处于黄金坑 且 产生逃逸脉冲 且 价格未大幅异动
        well_collapse_score = pit_state * (escape_ignite * 0.8 + (1.0 - trap_lock) * 0.2) * price_calm
        print(f"  -- 黄金坑状态: {pit_state.iloc[-1]:.4f} | 逃逸Jerk分: {escape_ignite.iloc[-1]:.4f} | 塌缩最终分: {well_collapse_score.iloc[-1]:.4f}")
        return well_collapse_score.clip(0, 1)

    def _calculate_high_freq_kinetic_gap_fill(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V41.0.1: 高频动能乖离回补模型 (High-Freq Kinetic Gap-Fill)。
        说明: 量化 5 日乖离率的弹性回补强度，识别短线极端超跌后的动力学“瞬时拉回” 。
        """
        print(f"--- [高频动能回补探针] @ {df_index[-1]} ---")
        # 1. 提取短线与长线乖离率原料 
        bias5 = raw_data['BIAS_5_D']
        bias55 = raw_data['BIAS_55_D']
        # 2. 计算 5 日乖离的 Jerk 脉冲：捕捉加速度的二次导数突变 
        b5_jerk = raw_data.get('JERK_5_BIAS_5_D', bias5.diff().diff().diff())
        # 3. 弹性激活判定：乖离越深且 Jerk 开始强力正向脉冲
        # 物理含义：向均值回补的冲击力正在爆发
        elasticity = self._calculate_physics_score(bias5, mode='limit_low', sensitivity=12.0)
        ignite = self._calculate_physics_score(b5_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        # 4. 空间差值补偿：短线乖离远深于长线乖离时，回补动能更强 
        gap_diff = (bias55 - bias5).clip(lower=0)
        gap_score = self._calculate_physics_score(gap_diff, mode='limit_high', sensitivity=5.0)
        # 5. 结合价格静止校验：确保回补发生在起爆前夜的静止态 
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        final_fill_score = (elasticity * ignite * gap_score) * price_calm
        print(f"  -- BIAS_5: {bias5.iloc[-1]:.4f} | 回补Jerk分: {ignite.iloc[-1]:.4f} | 最终回补分: {final_fill_score.iloc[-1]:.4f}")
        return final_fill_score.clip(0, 1)

    def _calculate_volatility_vacuum_contraction(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V42.0.1: 波动率真空收缩模型 (VVC)。
        说明: 量化 ATR 的高阶衰减特征，识别波动率进入绝对真空的临界瞬间。
        """
        print(f"--- [波动率真空收缩探针] @ {df_index[-1]} ---")
        atr_raw = raw_data['ATR_14_D']
        atr_slope = raw_data.get('SLOPE_13_ATR_14_D', pd.Series(0.0, index=df_index))
        atr_jerk = raw_data.get('JERK_5_ATR_14_D', pd.Series(0.0, index=df_index))
        atr_low_score = self._calculate_physics_score(atr_raw, mode='limit_low', sensitivity=1.5)
        decay_purity = self._calculate_physics_score(atr_slope, mode='limit_low', sensitivity=10.0, denoise=True)
        vacuum_silence = self._calculate_physics_score(atr_jerk, mode='zero_focus', sensitivity=80.0, denoise=True)
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        vvc_score = atr_low_score * decay_purity * vacuum_silence * price_calm
        print(f"  -- ATR值: {atr_raw.iloc[-1]:.4f} | 真空收缩评分: {vvc_score.iloc[-1]:.4f}")
        return vvc_score.clip(0, 1)

    def _calculate_fan_curvature_collapse(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V43.0.1: 扇面曲率坍塌模型 (FCC)。
        说明: 量化均线扇面效率的高阶衰减，捕捉几何结构向单点共振塌缩的瞬间。
        """
        print(f"--- [扇面曲率塌缩探针] @ {df_index[-1]} ---")
        fan_raw = raw_data['MA_FAN_EFFICIENCY_D']
        fan_accel = raw_data.get('ACCEL_8_MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        fan_jerk = raw_data.get('JERK_5_MA_FAN_EFFICIENCY_D', pd.Series(0.0, index=df_index))
        accel_focus = self._calculate_physics_score(fan_accel, mode='zero_focus', sensitivity=50.0, denoise=True)
        jerk_silence = self._calculate_physics_score(fan_jerk, mode='zero_focus', sensitivity=70.0, denoise=True)
        fan_high_score = self._calculate_physics_score(fan_raw, mode='limit_high', sensitivity=1.2)
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        fcc_score = fan_high_score * accel_focus * jerk_silence * price_calm
        print(f"  -- 扇面效率: {fan_raw.iloc[-1]:.4f} | 几何塌缩评分: {fcc_score.iloc[-1]:.4f}")
        return fcc_score.clip(0, 1)

    def _calculate_game_neutralization_modulator(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V44.0.1: 多维博弈中性化模型 (Multi-Dimensional Game Neutralization)。
        说明: 量化开收盘强度 OCH_D 的一阶导数归零度，识别多空博弈的力量对等失重态。
        """
        print(f"--- [博弈中性化探针] @ {df_index[-1]} ---")
        # 1. 提取开收盘强度及其一阶动力学梯度 (Slope)
        och_raw = raw_data['OCH_D']
        och_slope = raw_data.get('SLOPE_13_OCH_D', och_raw.diff(13))
        # 2. 梯度锁定：利用高斯核锁定 OCH 斜率绝对归零的瞬间
        # 物理含义：博弈强度停止变化，多空双方达成极致的力量对等
        neutralization_focus = self._calculate_physics_score(och_slope, mode='zero_focus', sensitivity=45.0, denoise=True)
        # 3. 强度基准校验：中性化必须发生在高强度的博弈背景下才有爆发意义
        och_intensity = self._calculate_physics_score(och_raw, mode='limit_high', sensitivity=1.0)
        # 4. 结合价格静止：博弈失重必须伴随价格斜率的彻底锁定
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        # 5. 最终中性化分：强度基准 * 梯度锁定 * 价格静止
        neutral_score = och_intensity * neutralization_focus * price_calm
        print(f"  -- OCH强度: {och_raw.iloc[-1]:.4f} | 梯度锁定分: {neutralization_focus.iloc[-1]:.4f} | 博弈中性分: {neutral_score.iloc[-1]:.4f}")
        return neutral_score.clip(0, 1)

    def _calculate_oversold_momentum_bipolarization(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V45.0.1: 超卖区动能二极化模型 (OMB)。
        说明: 量化 RSI 钝化后的加速度反转斜率与量能一致性，识别暴力接管行为。
        """
        print(f"--- [超卖动能二极化探针] @ {df_index[-1]} ---")
        rsi_raw = raw_data['RSI_13_D']
        rsi_accel = raw_data.get('ACCEL_8_RSI_13_D', pd.Series(0.0, index=df_index))
        accel_rev_slope = rsi_accel.diff(5)
        vol = raw_data['volume_D']
        vol_consistency = 1.0 / (1.0 + vol.rolling(window=8).std() / (vol.rolling(window=8).mean() + 1e-9))
        oversold_lock = self._calculate_physics_score(rsi_raw, mode='limit_low', sensitivity=0.05)
        bipolar_ratio = self._calculate_physics_score(accel_rev_slope * vol_consistency, mode='limit_high', sensitivity=20.0, denoise=True)
        price_calm = self._calculate_physics_score(raw_data['price_slope_raw'], mode='zero_focus', sensitivity=60.0, denoise=True)
        omb_score = oversold_lock * bipolar_ratio * price_calm
        print(f"  -- RSI值: {rsi_raw.iloc[-1]:.4f} | 二极化评分: {omb_score.iloc[-1]:.4f}")
        return omb_score.clip(0, 1)

    def _calculate_kinetic_overflow_veto(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], bipolar_gain: pd.Series) -> pd.Series:
        """
        V47.0.1: 三级动能溢出熔断模型。
        说明: 量化价格脉冲与动能二极化的非典型背离，防止情绪顶点的热点追高。
        """
        print(f"--- [三级动能溢出熔断探针] @ {df_index[-1]} ---")
        # 1. Level 1: RSI 超买耗尽 (Momentum Exhaustion)
        rsi_raw = raw_data['RSI_13_D']
        rsi_slope = raw_data.get('SLOPE_5_RSI_13_D', rsi_raw.diff(5))
        # 逻辑：RSI > 75 且 动能斜率转负，判定为 Level 1 溢出
        veto_l1 = np.where((rsi_raw > 75) & (rsi_slope < 0), 0.8, 1.0)
        # 2. Level 2: 量能末端脉冲 (Volume Blow-off)
        vol = raw_data['volume_D']
        vol_spike = vol / (vol.rolling(window=21).mean() + 1e-9)
        # 逻辑：价格在高位且成交量倍增 (> 2.5)，判定为 Level 2 诱多
        price_high = self._calculate_physics_score(raw_data['price_vs_ma_21_ratio_D'], mode='limit_high', sensitivity=5.0)
        veto_l2 = np.where((vol_spike > 2.5) & (price_high > 0.8), 0.7, 1.0)
        # 3. Level 3: 二极化特征瓦解 (Bipolarity Collapse)
        # 逻辑：当价格斜率极高 (> 0.6) 但 OMB (二极化分) 却低于 0.3，判定为动能脱轨
        price_v = self._calculate_physics_score(raw_data['price_slope_raw'], mode='limit_high', sensitivity=5.0)
        veto_l3 = np.where((price_v > 0.6) & (bipolar_gain < 0.3), 0.6, 1.0)
        # 4. 熔断合成：取三级熔断的最严厉压制值
        final_veto = pd.Series(veto_l1 * veto_l2 * veto_l3, index=df_index)
        print(f"  -- RSI状态: {rsi_raw.iloc[-1]:.4f} | L2量能脉冲: {vol_spike.iloc[-1]:.4f} | 最终熔断因子: {final_veto.iloc[-1]:.4f}")
        return final_veto.clip(0.3, 1.0)

    def _calculate_spatio_temporal_asymmetric_reward(self, df_index: pd.Index, raw_data: Dict[str, pd.Series], resonance_confirm: pd.Series) -> pd.Series:
        """
        V48.0.1: 多空时空非对称奖励模型 (STAR)。
        说明: 基于标的历史 120 日共振获利表现，提供个性化期望奖励。
        """
        print(f"--- [时空非对称奖励探针] @ {df_index[-1]} ---")
        close = raw_data.get('close', pd.Series(1.0, index=df_index))
        fwd_ret = close.shift(-5) / close - 1.0
        hist_hit_mask = resonance_confirm.shift(5).fillna(False)
        expected_gain = (fwd_ret * hist_hit_mask).rolling(window=120, min_periods=10).mean()
        reward_factor = 1.0 + self._calculate_physics_score(expected_gain.clip(lower=0), mode='limit_high', sensitivity=4.0)
        print(f"  -- 历史期望收益: {expected_gain.iloc[-1]:.4f} | 奖励系数: {reward_factor.iloc[-1]:.4f}")
        return reward_factor.fillna(1.0)

    def _calculate_extreme_panic_resonance(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V50.0.1: 极值恐慌共振模型 (Extreme Panic Resonance)。
        说明: 量化痛感代理在黄金坑内部的 Jerk 峰值，识别主力收集“带血筹码”的物理瞬间。
        """
        print(f"--- [极值恐慌共振探针] @ {df_index[-1]} ---")
        # 1. 提取痛感 Jerk (由 -JERK_5_profit_ratio_D 转换而来)
        pain_jerk = raw_data.get('JERK_5_pain_index_proxy', pd.Series(0.0, index=df_index))
        # 2. 映射恐慌爆发分：Jerk 越高，代表痛感增加的加速度越快，恐慌感越强
        panic_burst = self._calculate_physics_score(pain_jerk, mode='limit_high', sensitivity=25.0, denoise=True)
        # 3. 坑位限制：共振必须发生在黄金坑状态内 (STATE_GOLDEN_PIT_D)
        pit_state = raw_data['STATE_GOLDEN_PIT_D']
        # 4. 最终共振分：恐慌爆发脉冲 * 黄金坑状态激活
        resonance_score = panic_burst * pit_state
        print(f"  -- 痛感Jerk分: {panic_burst.iloc[-1]:.4f} | 黄金坑状态: {pit_state.iloc[-1]:.4f} | 共振最终分: {resonance_score.iloc[-1]:.4f}")
        return resonance_score.clip(0, 1)

    def _clip_physical_outliers(self, series: pd.Series, window: int = 55, sigma_multiplier: float = 3.0) -> pd.Series:
        """
        V52.0.0: 物理异常值限幅引擎。
        说明: 利用 55 日滚动 3 Sigma 原则对信号进行限幅，消除数据跳变引发的伪脉冲。
        """
        rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
        rolling_std = series.rolling(window=window, min_periods=window//2).std()
        upper_bound = rolling_mean + (rolling_std * sigma_multiplier)
        lower_bound = rolling_mean - (rolling_std * sigma_multiplier)
        # 针对导数序列，确保在统计极值范围内平滑波动
        clipped_series = series.clip(lower=lower_bound, upper=upper_bound).fillna(series)
        return clipped_series

    def _calculate_adaptive_phase_transition_threshold(self, df_index: pd.Index, raw_data: Dict[str, pd.Series]) -> pd.Series:
        """
        V53.0.0: 自适应相变阈值模型 (APTT)。
        说明: 量化标的历史信噪比，动态调优 Fermi 门控激活阈值。
        """
        print(f"--- [自适应阈值探针] @ {df_index[-1]} ---")
        price_v = raw_data['price_slope_raw']
        noise_cv = price_v.rolling(window=250, min_periods=60).std() / (price_v.rolling(window=250, min_periods=60).mean().abs() + 1e-9)
        adaptive_threshold = 0.45 * (0.8 + 0.5 * self._calculate_physics_score(noise_cv, mode='limit_high', sensitivity=2.0))
        adaptive_threshold = adaptive_threshold.fillna(0.45)
        print(f"  -- 噪音系数: {noise_cv.iloc[-1]:.4f} | 动态阈值: {adaptive_threshold.iloc[-1]:.4f}")
        return adaptive_threshold

















