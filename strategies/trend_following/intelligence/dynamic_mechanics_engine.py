import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, get_adaptive_mtf_normalized_score, bipolar_to_exclusive_unipolar

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [力学情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # 调整校验信息为“力学情报校验”
            print(f"    -> [力学情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_dynamic_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.3 · 探针增强版】动态力学引擎总指挥
        - 核心升级: 为“大统一力场”增加专属探针输出，使其内部的“矛”与“盾”的计算过程透明化，
                      便于持续监控和验证模型的顶层裁决逻辑。
        """
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        current_date_str = df.index[-1].strftime('%Y-%m-%d')
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 动态力学引擎在配置中被禁用，跳过分析。")
            return {}
        all_dynamic_states = {}
        axiom_momentum, momentum_tension = self._diagnose_axiom_momentum(df, False, current_date_str)
        axiom_inertia, inertia_tension = self._diagnose_axiom_inertia(df, False, current_date_str)
        axiom_stability, stability_tension = self._diagnose_axiom_stability(df, False, current_date_str)
        axiom_energy, energy_tension = self._diagnose_axiom_energy(df, False, current_date_str)
        axiom_ma_dynamics = self._diagnose_axiom_ma_dynamics(df)
        axiom_divergence = self._diagnose_axiom_divergence(df, axiom_momentum, axiom_inertia)
        all_tensions = [momentum_tension, inertia_tension, stability_tension, energy_tension]
        fused_tension_score = pd.concat(all_tensions, axis=1).mean(axis=1)
        all_dynamic_states['INTERNAL_DYN_MTF_CALIBRATION_TENSION_D'] = fused_tension_score.astype(np.float32)
        all_dynamic_states['SCORE_DYN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_dynamic_states['SCORE_DYN_AXIOM_MOMENTUM'] = axiom_momentum
        all_dynamic_states['SCORE_DYN_AXIOM_INERTIA'] = axiom_inertia
        all_dynamic_states['SCORE_DYN_AXIOM_STABILITY'] = axiom_stability
        all_dynamic_states['SCORE_DYN_AXIOM_ENERGY'] = axiom_energy
        all_dynamic_states['SCORE_DYN_AXIOM_MA_ACCELERATION'] = axiom_ma_dynamics
        grand_unification_score = self._synthesize_grand_unification_score(
            axiom_momentum, axiom_inertia, axiom_stability, axiom_energy
        )
        all_dynamic_states['SCORE_DYN_GRAND_UNIFICATION'] = grand_unification_score
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_dynamic_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, momentum_score: pd.Series, inertia_score: pd.Series) -> pd.Series:
        """
        【V1.3 · 参数清理版】力学公理五：诊断“力学背离”
        - 【修改】移除方法签名中未使用的 norm_window 参数。
        """
        divergence_score = (inertia_score - momentum_score).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_momentum(self, df: pd.DataFrame, is_probe_day: bool, current_date_str: str) -> (pd.Series, pd.Series):
        """
        【V3.4 · 纯力学品质版】力学公理一：诊断“动量品质”
        - 核心重构: 将品质调节器的输入替换为纯粹基于力学指标（ROC、MACDh）及其衍生（斜率、加速度、波动率）的力学特征。
        - 核心重构: 方法现在返回 (final_momentum_score, z_tension)，将计算出的“校准张力”向上传递。
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        quality_weights = get_param_value(p_conf_dyn.get('momentum_quality_weights'), {'trend_strength': 0.4, 'stability': 0.3, 'acceleration': 0.3})
        # 将ROC_12_D替换为ROC_13_D
        required_signals = [
            'ROC_13_D', 'MACDh_13_34_8_D',
            'SLOPE_5_ROC_13_D', 'SLOPE_5_MACDh_13_34_8_D',
            'ACCEL_5_ROC_13_D', 'ACCEL_5_MACDh_13_34_8_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_momentum"):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        # 将ROC_12_D替换为ROC_13_D
        roc = self._get_safe_series(df, 'ROC_13_D', 0.0, method_name="_diagnose_axiom_momentum")
        macd_h = self._get_safe_series(df, 'MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_momentum")
        # 正确获取MTF权重配置
        p_conf_struct_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_norm_weights_container = p_conf_struct_ultimate.get('mtf_normalization_weights', {})
        actual_mtf_weights = mtf_norm_weights_container.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 新增行：健壮性检查
        if not isinstance(actual_mtf_weights, dict):
            print(f"    -> [力学情报警告] 方法 '_diagnose_axiom_momentum' 的MTF权重配置格式错误，使用硬编码默认值。配置: {actual_mtf_weights}")
            actual_mtf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        # 将 default_weights 替换为 actual_mtf_weights
        roc_score = get_adaptive_mtf_normalized_bipolar_score(roc, df.index, actual_mtf_weights)
        macd_h_score = get_adaptive_mtf_normalized_bipolar_score(macd_h, df.index, actual_mtf_weights)
        raw_momentum_score = (roc_score * 0.6 + macd_h_score * 0.4).clip(-1, 1)
        # --- 新增纯力学品质因子计算 ---
        # 1. 动量趋势强度 (Momentum Trend Strength)
        roc_slope = self._get_safe_series(df, 'SLOPE_5_ROC_13_D', 0.0, method_name="_diagnose_axiom_momentum")
        macd_h_slope = self._get_safe_series(df, 'SLOPE_5_MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_momentum")
        momentum_trend_strength_score = (get_adaptive_mtf_normalized_bipolar_score(roc_slope, df.index, actual_mtf_weights) + \
                                         get_adaptive_mtf_normalized_bipolar_score(macd_h_slope, df.index, actual_mtf_weights)) / 2
        # 2. 动量稳定性 (Momentum Stability) - 波动率的倒数
        roc_std = roc.rolling(window=5, min_periods=1).std().fillna(0)
        macd_h_std = macd_h.rolling(window=5, min_periods=1).std().fillna(0)
        # 将标准差归一化，然后取1-score，表示波动越小分数越高
        roc_stability_score = 1 - get_adaptive_mtf_normalized_score(roc_std, df.index, actual_mtf_weights, ascending=True)
        macd_h_stability_score = 1 - get_adaptive_mtf_normalized_score(macd_h_std, df.index, actual_mtf_weights, ascending=True)
        momentum_stability_score = (roc_stability_score + macd_h_stability_score) / 2
        # 3. 动量加速度 (Momentum Acceleration)
        roc_accel = self._get_safe_series(df, 'ACCEL_5_ROC_13_D', 0.0, method_name="_diagnose_axiom_momentum")
        macd_h_accel = self._get_safe_series(df, 'ACCEL_5_MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_momentum")
        momentum_acceleration_score = (get_adaptive_mtf_normalized_bipolar_score(roc_accel, df.index, actual_mtf_weights) + \
                                       get_adaptive_mtf_normalized_bipolar_score(macd_h_accel, df.index, actual_mtf_weights)) / 2
        composite_quality_raw = (
            momentum_trend_strength_score * quality_weights.get('trend_strength', 0.4) +
            momentum_stability_score * quality_weights.get('stability', 0.3) +
            momentum_acceleration_score * quality_weights.get('acceleration', 0.3)
        )
        z_score, z_tension = self._get_mtf_calibrated_z_score(composite_quality_raw, p_conf_dyn)
        momentum_quality_modulator = (np.tanh(z_score) + 1) / 2
        final_momentum_score = (raw_momentum_score * momentum_quality_modulator).clip(-1, 1)
        if is_probe_day:
            last_values = {
                "原料-ROC": roc.iloc[-1], "原料-MACDh": macd_h.iloc[-1],
                "原料-ROC_Slope": roc_slope.iloc[-1], "原料-MACDh_Slope": macd_h_slope.iloc[-1],
                "原料-ROC_Std": roc_std.iloc[-1], "原料-MACDh_Std": macd_h_std.iloc[-1],
                "原料-ROC_Accel": roc_accel.iloc[-1], "原料-MACDh_Accel": macd_h_accel.iloc[-1],
                "节点-原始动量分": raw_momentum_score.iloc[-1],
                "节点-动量趋势强度分": momentum_trend_strength_score.iloc[-1],
                "节点-动量稳定性分": momentum_stability_score.iloc[-1],
                "节点-动量加速度分": momentum_acceleration_score.iloc[-1],
                "节点-品质综合分(原始)": composite_quality_raw.iloc[-1],
                "节点-品质ZScore": z_score.iloc[-1], "节点-品质调节器": momentum_quality_modulator.iloc[-1],
                "节点-校准张力": z_tension.iloc[-1],
                "结果-最终动量分": final_momentum_score.iloc[-1]
            }
            print(f"  > 动量品质探针: { {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in last_values.items()} }")
        return final_momentum_score.astype(np.float32), z_tension.astype(np.float32)

    def _diagnose_axiom_inertia(self, df: pd.DataFrame, is_probe_day: bool, current_date_str: str) -> (pd.Series, pd.Series):
        """
        【V3.5 · 校准张力版】力学公理二：诊断“结构化惯性”
        - 核心重构: 方法现在返回 (final_inertia_score, z_tension)，将计算出的“校准张力”向上传递。
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        inertia_weights = get_param_value(p_conf_dyn.get('inertia_structural_weights'), {'base_inertia': 0.7, 'structural_reinforcement': 0.3})
        ma_col_base = 'EMA_55'
        timeframe_key = 'D'
        # 将hurst_col直接指定为BID_LIQUIDITY_HURST_144d_D
        hurst_col = 'BID_LIQUIDITY_HURST_144d_D'
        fractal_col = next((col for col in df.columns if col.startswith('FRACTAL_DIMENSION_')), 'FRACTAL_DIMENSION_89d_D')
        # required_signals中包含新的hurst_col
        required_signals = [
            'ADX_14_D', hurst_col, fractal_col, f'MA_VELOCITY_{ma_col_base}_{timeframe_key}',
            f'MA_ACCELERATION_{ma_col_base}_{timeframe_key}', 'PDI_14_D', 'NDI_14_D',
            'trend_alignment_index_D', 'structural_leverage_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_inertia"):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        # 正确获取MTF权重配置
        p_conf_struct_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_norm_weights_container = p_conf_struct_ultimate.get('mtf_normalization_weights', {})
        actual_mtf_weights = mtf_norm_weights_container.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 新增行：健壮性检查
        if not isinstance(actual_mtf_weights, dict):
            print(f"    -> [力学情报警告] 方法 '_diagnose_axiom_inertia' 的MTF权重配置格式错误，使用硬编码默认值。配置: {actual_mtf_weights}")
            actual_mtf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        # 将 default_weights 替换为 actual_mtf_weights
        # 修正 get_adaptive_mtf_normalized_score 的参数顺序
        adx_strength = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'ADX_14_D', 0.0), df.index, actual_mtf_weights, True) # MODIFIED
        hurst = self._get_safe_series(df, hurst_col, 0.5).fillna(0.5)
        # hurst_quality的计算依赖于hurst，这里无需修改
        # 修正 get_adaptive_mtf_normalized_score 的参数顺序
        hurst_quality = get_adaptive_mtf_normalized_score(hurst, df.index, actual_mtf_weights, True) # MODIFIED
        fractal_dim = self._get_safe_series(df, fractal_col, 1.5).fillna(1.5)
        # 修正 get_adaptive_mtf_normalized_score 的参数顺序
        fractal_smoothness = get_adaptive_mtf_normalized_score(fractal_dim, df.index, actual_mtf_weights, False) # MODIFIED
        # 修正 get_adaptive_mtf_normalized_score 的参数顺序
        ma_velocity = get_adaptive_mtf_normalized_score(self._get_safe_series(df, f'MA_VELOCITY_{ma_col_base}_{timeframe_key}', 0.0), df.index, actual_mtf_weights, True) # MODIFIED
        # 修正 get_adaptive_mtf_normalized_score 的参数顺序
        ma_acceleration = get_adaptive_mtf_normalized_score(self._get_safe_series(df, f'MA_ACCELERATION_{ma_col_base}_{timeframe_key}', 0.0), df.index, actual_mtf_weights, True) # MODIFIED
        base_inertia_quality = (adx_strength * 0.3 + hurst_quality * 0.3 + fractal_smoothness * 0.1 + ma_velocity * 0.15 + ma_acceleration * 0.15).clip(0, 1)
        raw_alignment = self._get_safe_series(df, 'trend_alignment_index_D', 0.0)
        raw_leverage = self._get_safe_series(df, 'structural_leverage_D', 0.0)
        composite_reinforcement_raw = (raw_alignment * 0.5 + raw_leverage * 0.5)
        z_score, z_tension = self._get_mtf_calibrated_z_score(composite_reinforcement_raw, p_conf_dyn)
        structural_reinforcement_score = np.tanh(z_score)
        total_inertia_quality = (base_inertia_quality * inertia_weights.get('base_inertia', 0.7) + structural_reinforcement_score.clip(0, 1) * inertia_weights.get('structural_reinforcement', 0.3)).clip(0, 1)
        adx_direction = (self._get_safe_series(df, 'PDI_14_D', 0) > self._get_safe_series(df, 'NDI_14_D', 0)).astype(float) * 2 - 1
        final_inertia_score = (total_inertia_quality * adx_direction).clip(-1, 1)
        if is_probe_day:
            last_values = {
                "原料-ADX": self._get_safe_series(df, 'ADX_14_D').iloc[-1], "原料-Hurst": hurst.iloc[-1],
                "结构-排列": raw_alignment.iloc[-1], "结构-杠杆": raw_leverage.iloc[-1],
                "节点-基础惯性品质": base_inertia_quality.iloc[-1], "节点-结构加固分(原始)": composite_reinforcement_raw.iloc[-1],
                "节点-结构ZScore": z_score.iloc[-1], "节点-结构加固分": structural_reinforcement_score.iloc[-1],
                "节点-校准张力": z_tension.iloc[-1],
                "节点-总惯性品质": total_inertia_quality.iloc[-1], "结果-最终惯性分": final_inertia_score.iloc[-1]
            }
            print(f"  > 结构化惯性探针: { {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in last_values.items()} }")
        return final_inertia_score.astype(np.float32), z_tension.astype(np.float32)

    def _diagnose_axiom_stability(self, df: pd.DataFrame, is_probe_day: bool, current_date_str: str) -> (pd.Series, pd.Series):
        """
        【V3.5 · 校准张力版】力学公理三：诊断“势能稳定性”
        - 核心重构: 方法现在返回 (final_stability_score, z_tension)，将计算出的“校准张力”向上传递。
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        potential_weights = get_param_value(p_conf_dyn.get('stability_potential_weights'), {'ma_tension': 0.4, 'structural_tension': 0.4, 'breakout_readiness': 0.2})
        required_signals = [
            'BBW_21_2.0_D', 'ATR_14_D', 'close_D',
            'MA_POTENTIAL_TENSION_INDEX_D', 'structural_tension_index_D', 'breakout_readiness_score_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_stability"):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        # 正确获取MTF权重配置
        p_conf_struct_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_norm_weights_container = p_conf_struct_ultimate.get('mtf_normalization_weights', {})
        actual_mtf_weights = mtf_norm_weights_container.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 新增行：健壮性检查
        if not isinstance(actual_mtf_weights, dict):
            print(f"    -> [力学情报警告] 方法 '_diagnose_axiom_stability' 的MTF权重配置格式错误，使用硬编码默认值。配置: {actual_mtf_weights}")
            actual_mtf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        bbw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0)
        atr_pct = self._get_safe_series(df, 'ATR_14_D', 0.0) / self._get_safe_series(df, 'close_D', 1e-9).replace(0, np.nan)
        raw_volatility = (bbw + atr_pct).fillna(0)
        # 将 default_weights 替换为 actual_mtf_weights
        volatility_level_score = get_adaptive_mtf_normalized_score(raw_volatility, df.index, ascending=True, tf_weights=actual_mtf_weights)
        raw_stability_score = 1 - volatility_level_score
        raw_ma_tension = self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', 0.5)
        raw_structural_tension = self._get_safe_series(df, 'structural_tension_index_D', 0.5)
        raw_readiness = self._get_safe_series(df, 'breakout_readiness_score_D', 0.5)
        composite_potential_raw = (
            raw_ma_tension * potential_weights.get('ma_tension', 0.4) +
            raw_structural_tension * potential_weights.get('structural_tension', 0.4) +
            raw_readiness * potential_weights.get('breakout_readiness', 0.2)
        )
        z_score, z_tension = self._get_mtf_calibrated_z_score(composite_potential_raw, p_conf_dyn)
        potential_energy_score = (np.tanh(z_score) + 1) / 2
        final_stability_score = ((raw_stability_score * 2 - 1) * potential_energy_score).clip(-1, 1)
        if is_probe_day:
            last_values = {
                "原料-BBW": bbw.iloc[-1], "原料-ATR%": atr_pct.iloc[-1],
                "势能-均线张力": raw_ma_tension.iloc[-1], "势能-结构张力": raw_structural_tension.iloc[-1], "势能-准备度": raw_readiness.iloc[-1],
                "节点-波动率分": volatility_level_score.iloc[-1], "节点-原始稳定分": raw_stability_score.iloc[-1],
                "节点-势能综合分(原始)": composite_potential_raw.iloc[-1], "节点-势能ZScore": z_score.iloc[-1],
                "节点-校准张力": z_tension.iloc[-1],
                "节点-势能指数": potential_energy_score.iloc[-1], "结果-最终稳定分": final_stability_score.iloc[-1]
            }
            print(f"  > 势能稳定性探针: { {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in last_values.items()} }")
        return final_stability_score.astype(np.float32), z_tension.astype(np.float32)

    def _diagnose_axiom_energy(self, df: pd.DataFrame, is_probe_day: bool, current_date_str: str) -> (pd.Series, pd.Series):
        """
        【V2.5 · 校准张力版】力学公理四：诊断“能量真实性”
        - 核心重构: 方法现在返回 (final_energy_score, z_tension)，将计算出的“校准张力”向上传递。
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        quality_weights = get_param_value(p_conf_dyn.get('energy_quality_weights'), {'credibility': 0.5, 'directionality': 0.5})
        required_signals = [
            'VPA_EFFICIENCY_D', 'CMF_21_D',
            'flow_credibility_index_D', 'main_force_flow_directionality_D', 'wash_trade_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_energy"):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        vpa = self._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.5)
        cmf = self._get_safe_series(df, 'CMF_21_D', 0.0)
        vpa_bipolar = (vpa * 2 - 1).clip(-1, 1)
        # 正确获取MTF权重配置
        p_conf_struct_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_norm_weights_container = p_conf_struct_ultimate.get('mtf_normalization_weights', {})
        actual_mtf_weights = mtf_norm_weights_container.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 新增行：健壮性检查
        if not isinstance(actual_mtf_weights, dict):
            print(f"    -> [力学情报警告] 方法 '_diagnose_axiom_energy' 的MTF权重配置格式错误，使用硬编码默认值。配置: {actual_mtf_weights}")
            actual_mtf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        # 将 default_weights 替换为 actual_mtf_weights
        cmf_bipolar = get_adaptive_mtf_normalized_bipolar_score(cmf, df.index, actual_mtf_weights)
        raw_energy_score = (vpa_bipolar * 0.5 + cmf_bipolar * 0.5).clip(-1, 1)
        raw_credibility = self._get_safe_series(df, 'flow_credibility_index_D', 0.5)
        raw_directionality = self._get_safe_series(df, 'main_force_flow_directionality_D', 0.5)
        raw_wash_trade = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0)
        composite_quality_raw = (
            raw_credibility * quality_weights.get('credibility', 0.5) +
            raw_directionality * quality_weights.get('directionality', 0.5)
        )
        z_score, z_tension = self._get_mtf_calibrated_z_score(composite_quality_raw, p_conf_dyn)
        base_quality_score = (np.tanh(z_score) + 1) / 2
        # 将 default_weights 替换为 actual_mtf_weights
        # 修正 get_adaptive_mtf_normalized_score 的参数顺序
        wash_trade_penalty_factor = 1 - get_adaptive_mtf_normalized_score(raw_wash_trade, df.index, actual_mtf_weights, True).clip(0, 1) # MODIFIED
        energy_quality_modulator = (base_quality_score * wash_trade_penalty_factor).clip(0, 1)
        final_energy_score = (raw_energy_score * energy_quality_modulator).clip(-1, 1)
        if is_probe_day:
            last_values = {
                "原料-VPA": vpa.iloc[-1], "原料-CMF": cmf.iloc[-1],
                "品质-可信度": raw_credibility.iloc[-1], "品质-流向性": raw_directionality.iloc[-1], "品质-对倒": raw_wash_trade.iloc[-1],
                "节点-原始能量分": raw_energy_score.iloc[-1], "节点-品质综合分(原始)": composite_quality_raw.iloc[-1],
                "节点-品质ZScore": z_score.iloc[-1], "节点-惩罚因子": wash_trade_penalty_factor.iloc[-1],
                "节点-校准张力": z_tension.iloc[-1],
                "节点-品质调节器": energy_quality_modulator.iloc[-1], "结果-最终能量分": final_energy_score.iloc[-1]
            }
            print(f"  > 能量真实性探针: { {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in last_values.items()} }")
        return final_energy_score.astype(np.float32), z_tension.astype(np.float32)

    def _diagnose_axiom_ma_dynamics(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.5 · 参数清理版】力学公理六：诊断“均线动态”
        - 【修改】移除方法签名中未使用的 norm_window 参数。
        """
        ma_col_base = 'EMA_55'
        timeframe_key = 'D'
        velocity_col = f'MA_VELOCITY_{ma_col_base}_{timeframe_key}'
        acceleration_col = f'MA_ACCELERATION_{ma_col_base}_{timeframe_key}'
        required_signals = [velocity_col, acceleration_col]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_ma_dynamics"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        velocity_raw = self._get_safe_series(df, velocity_col, 0.0, method_name="_diagnose_axiom_ma_dynamics")
        acceleration_raw = self._get_safe_series(df, acceleration_col, 0.0, method_name="_diagnose_axiom_ma_dynamics")
        # 正确获取MTF权重配置
        p_conf_struct_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_norm_weights_container = p_conf_struct_ultimate.get('mtf_normalization_weights', {})
        actual_mtf_weights = mtf_norm_weights_container.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 新增行：健壮性检查
        if not isinstance(actual_mtf_weights, dict):
            print(f"    -> [力学情报警告] 方法 '_diagnose_axiom_ma_dynamics' 的MTF权重配置格式错误，使用硬编码默认值。配置: {actual_mtf_weights}")
            actual_mtf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        # 将 default_weights 替换为 actual_mtf_weights
        velocity_score = get_adaptive_mtf_normalized_bipolar_score(velocity_raw, df_index, actual_mtf_weights)
        acceleration_score = get_adaptive_mtf_normalized_bipolar_score(acceleration_raw, df_index, actual_mtf_weights)
        ma_dynamics_score = (velocity_score * 0.6 + acceleration_score * 0.4).clip(-1, 1)
        return ma_dynamics_score.astype(np.float32)

    def _synthesize_grand_unification_score(self, momentum: pd.Series, inertia: pd.Series, stability: pd.Series, energy: pd.Series) -> pd.Series:
        """
        【V1.0 · 矛与盾】大统一力场合成器
        - 核心逻辑: 基于“矛与盾”博弈模型，进行非线性融合，输出系统总合力。
                    矛 (进攻力量) = 动量 + 能量
                    盾 (结构完整性) = 惯性 + 稳定性
                    最终得分 = sign(矛) * sqrt(abs(矛 * 盾))
        """
        p_conf_dyn = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        unification_params = get_param_value(p_conf_dyn.get('grand_unification_params'), {})
        offensive_weights = get_param_value(unification_params.get('offensive_force_weights'), {'momentum': 0.6, 'energy': 0.4})
        structural_weights = get_param_value(unification_params.get('structural_integrity_weights'), {'inertia': 0.6, 'stability': 0.4})
        # 1. 计算进攻力量“矛”
        offensive_force = (
            momentum * offensive_weights.get('momentum', 0.6) +
            energy * offensive_weights.get('energy', 0.4)
        ).clip(-1, 1)
        # 2. 计算结构完整性“盾”
        structural_integrity = (
            inertia * structural_weights.get('inertia', 0.6) +
            stability * structural_weights.get('stability', 0.4)
        ).clip(-1, 1)
        # 3. 非线性融合
        # 使用 .values 可以确保 sign, sqrt, abs 操作在numpy层面高效执行
        product = offensive_force.values * structural_integrity.values
        grand_unification_score_raw = np.sign(offensive_force.values) * np.sqrt(np.abs(product))
        # 将numpy数组转换回带索引的pandas Series
        grand_unification_score = pd.Series(grand_unification_score_raw, index=momentum.index).fillna(0)
        return grand_unification_score.astype(np.float32)

    def _get_mtf_calibrated_z_score(self, raw_series: pd.Series, p_conf_dyn: Dict) -> (pd.Series, pd.Series):
        """
        【V2.0 · 张力诊断版】内部辅助方法，计算MTF融合Z-Score和“校准张力”。
        - 返回: (final_z_score, tension_score)
        - 新增: 计算短期Z-Score组与长期Z-Score组的加权平均值，并返回二者之差作为“张力分”。
        """
        norm_params = get_param_value(p_conf_dyn.get('calibrated_norm_params'), {})
        z_windows = norm_params.get('z_windows', [5, 13, 21, 55, 89, 144])
        z_weights = norm_params.get('z_weights', {"5": 0.3, "13": 0.25, "21": 0.2, "55": 0.15, "89": 0.05, "144": 0.05})
        z_sens = norm_params.get('sensitivity', 2.0)
        tension_windows = norm_params.get('tension_windows', {'short_term': [5, 13, 21], 'long_term': [55, 89, 144]})
        all_z_scores = {}
        for window in z_windows:
            mean = raw_series.rolling(window=window, min_periods=window // 2).mean()
            std = raw_series.rolling(window=window, min_periods=window // 2).std()
            z_score = pd.Series(0.0, index=raw_series.index)
            valid_std_mask = std > 1e-9
            z_score[valid_std_mask] = (raw_series[valid_std_mask] - mean[valid_std_mask]) / (std[valid_std_mask] * z_sens)
            all_z_scores[window] = z_score.fillna(0)
        weighted_z_scores = pd.Series(0.0, index=raw_series.index)
        total_weight = 0.0
        for window, z_score in all_z_scores.items():
            weight = z_weights.get(str(window), 0)
            if weight > 0:
                weighted_z_scores += z_score * weight
                total_weight += weight
        final_z_score = weighted_z_scores / total_weight if total_weight > 0 else pd.Series(0.0, index=raw_series.index)
        def get_group_avg(group_keys):
            group_scores = pd.Series(0.0, index=raw_series.index)
            group_weight = 0.0
            for window in group_keys:
                if window in all_z_scores:
                    weight = z_weights.get(str(window), 0)
                    if weight > 0:
                        group_scores += all_z_scores[window] * weight
                        group_weight += weight
            return group_scores / group_weight if group_weight > 0 else pd.Series(0.0, index=raw_series.index)
        short_term_z = get_group_avg(tension_windows.get('short_term', []))
        long_term_z = get_group_avg(tension_windows.get('long_term', []))
        tension_score = short_term_z - long_term_z
        return final_z_score, tension_score

