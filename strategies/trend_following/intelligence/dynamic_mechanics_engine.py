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
        【V5.9 · 上下文修复版】动态力学引擎总指挥
        - 【V5.9 修复】接收 df 参数作为统一的数据上下文，并移除内部对 self.strategy.df_indicators 的依赖。
        """
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 动态力学引擎在配置中被禁用，跳过分析。")
            return {}
        all_dynamic_states = {}
        # df = self.strategy.df_indicators # [代码删除]
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        axiom_momentum = self._diagnose_axiom_momentum(df, norm_window)
        axiom_inertia = self._diagnose_axiom_inertia(df, norm_window)
        axiom_stability = self._diagnose_axiom_stability(df, norm_window)
        axiom_energy = self._diagnose_axiom_energy(df, norm_window)
        axiom_ma_dynamics = self._diagnose_axiom_ma_dynamics(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        all_dynamic_states['SCORE_DYN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_dynamic_states['SCORE_DYN_AXIOM_MOMENTUM'] = axiom_momentum
        all_dynamic_states['SCORE_DYN_AXIOM_INERTIA'] = axiom_inertia
        all_dynamic_states['SCORE_DYN_AXIOM_STABILITY'] = axiom_stability
        all_dynamic_states['SCORE_DYN_AXIOM_ENERGY'] = axiom_energy
        all_dynamic_states['SCORE_DYN_AXIOM_MA_ACCELERATION'] = axiom_ma_dynamics
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_dynamic_states['SCORE_DYNAMIC_MECHANICS_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_dynamic_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.1 · 多时间维度归一化版】力学公理五：诊断“力学背离”
        - 核心逻辑: 诊断价格动量与惯性之间的背离。
          - 看涨背离：价格动量减弱（负）但惯性增强（正） -> 预示趋势可能反转向上。
          - 看跌背离：价格动量增强（正）但惯性减弱（负） -> 预示趋势可能反转向下。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `momentum_score` 和 `inertia_score` 的归一化方式改为多时间维度自适应归一化。
        """
        momentum_score = self._diagnose_axiom_momentum(df, norm_window)
        inertia_score = self._diagnose_axiom_inertia(df, norm_window)
        # 【优化】力学背离本身就是一种关系，其归一化应该基于其自身的动态，而不是简单地使用单一窗口。
        # 这里直接使用计算出的双极性分数，不再进行额外的 normalize_to_bipolar，因为 momentum_score 和 inertia_score 已经是双极性。
        divergence_score = (inertia_score - momentum_score).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.2 · 信号校验增强版】力学公理一：诊断“动量”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `roc_score` 和 `macd_h_score` 的归一化方式改为多时间维度自适应归一化。
        """
        required_signals = ['ROC_12_D', 'MACDh_13_34_8_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_momentum"):
            return pd.Series(0.0, index=df.index)
        roc = self._get_safe_series(df, 'ROC_12_D', 0.0, method_name="_diagnose_axiom_momentum")
        macd_h = self._get_safe_series(df, 'MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_momentum")
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        roc_score = get_adaptive_mtf_normalized_bipolar_score(roc, df.index, default_weights)
        macd_h_score = get_adaptive_mtf_normalized_bipolar_score(macd_h, df.index, default_weights)
        momentum_score = (roc_score * 0.6 + macd_h_score * 0.4).clip(-1, 1)
        return momentum_score.astype(np.float32)

    def _diagnose_axiom_inertia(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.7 · 信号校验增强版】力学公理二：诊断“惯性”
        - 核心重构: 废除脆弱的乘法融合模型，改为更稳健的加权平均模型。
        - 【新增】引入均线速度和加速度作为惯性判断的证据。
        - 【修复】修正了引用均线速度和加速度列名时，确保其与 `IndicatorService` 中 `merge_results` 方法添加后缀后的列名一致。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        """
        ma_col_base = 'EMA_55'
        timeframe_key = 'D'
        hurst_col = next((col for col in df.columns if col.startswith('hurst_')), 'hurst_144d_D')
        fractal_col = next((col for col in df.columns if col.startswith('FRACTAL_DIMENSION_')), 'FRACTAL_DIMENSION_100d_D')
        required_signals = [
            'ADX_14_D', hurst_col, fractal_col, f'MA_VELOCITY_{ma_col_base}_{timeframe_key}',
            f'MA_ACCELERATION_{ma_col_base}_{timeframe_key}', 'PDI_14_D', 'NDI_14_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_inertia"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        adx_strength = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'ADX_14_D', 0.0, method_name="_diagnose_axiom_inertia"), df.index, ascending=True, tf_weights=default_weights)
        hurst = self._get_safe_series(df, hurst_col, 0.5, method_name="_diagnose_axiom_inertia").fillna(0.5)
        hurst_quality = get_adaptive_mtf_normalized_score(hurst, df.index, ascending=True, tf_weights=default_weights)
        fractal_dim = self._get_safe_series(df, fractal_col, 1.5, method_name="_diagnose_axiom_inertia").fillna(1.5)
        fractal_smoothness = get_adaptive_mtf_normalized_score(fractal_dim, df.index, ascending=False, tf_weights=default_weights)
        ma_velocity = get_adaptive_mtf_normalized_score(self._get_safe_series(df, f'MA_VELOCITY_{ma_col_base}_{timeframe_key}', 0.0, method_name="_diagnose_axiom_inertia"), df.index, ascending=True, tf_weights=default_weights)
        ma_acceleration = get_adaptive_mtf_normalized_score(self._get_safe_series(df, f'MA_ACCELERATION_{ma_col_base}_{timeframe_key}', 0.0, method_name="_diagnose_axiom_inertia"), df.index, ascending=True, tf_weights=default_weights)
        inertia_quality = (
            adx_strength * 0.3 +
            hurst_quality * 0.3 +
            fractal_smoothness * 0.1 +
            ma_velocity * 0.15 +
            ma_acceleration * 0.15
        ).clip(0, 1)
        adx_direction = (self._get_safe_series(df, 'PDI_14_D', 0, method_name="_diagnose_axiom_inertia") > self._get_safe_series(df, 'NDI_14_D', 0, method_name="_diagnose_axiom_inertia")).astype(float) * 2 - 1
        inertia_score = (inertia_quality * adx_direction).clip(-1, 1)
        return inertia_score.astype(np.float32)

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.4 · 信号校验增强版】力学公理三：诊断“稳定性”
        - 核心重构: 废除脆弱的乘法融合模型，改为更稳健的加权平均模型。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `volatility_level_score` 和 `volatility_stability_score` 的归一化方式改为多时间维度自适应归一化。
        """
        vol_instability_col = next((col for col in df.columns if col.startswith('VOLATILITY_INSTABILITY_INDEX_')), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        required_signals = ['BBW_21_2.0_D', 'ATR_14_D', 'close_D', vol_instability_col]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_stability"):
            return pd.Series(0.0, index=df.index)
        bbw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name="_diagnose_axiom_stability")
        atr_pct = self._get_safe_series(df, 'ATR_14_D', 0.0, method_name="_diagnose_axiom_stability") / self._get_safe_series(df, 'close_D', 1e-9, method_name="_diagnose_axiom_stability").replace(0, np.nan)
        raw_volatility = (bbw + atr_pct).fillna(0)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        volatility_level_score = get_adaptive_mtf_normalized_score(raw_volatility, df.index, ascending=False, tf_weights=default_weights)
        vol_of_vol = self._get_safe_series(df, vol_instability_col, 0.0, method_name="_diagnose_axiom_stability")
        volatility_stability_score = get_adaptive_mtf_normalized_score(vol_of_vol, df.index, ascending=False, tf_weights=default_weights)
        raw_stability_score = (
            volatility_level_score * 0.6 +
            volatility_stability_score * 0.4
        ).clip(0, 1)
        stability_score = (raw_stability_score * 2 - 1).clip(-1, 1)
        return stability_score.astype(np.float32)

    def _diagnose_axiom_energy(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.3 · 信号校验增强版】力学公理四：诊断“能量”
        - 核心重构: 废除脆弱的乘法融合模型，改为更稳健的加权平均模型。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `cmf_bipolar` 的归一化方式改为多时间维度自适应归一化。
        """
        required_signals = ['VPA_EFFICIENCY_D', 'CMF_21_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_energy"):
            return pd.Series(0.0, index=df.index)
        vpa = self._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.5, method_name="_diagnose_axiom_energy")
        cmf = self._get_safe_series(df, 'CMF_21_D', 0.0, method_name="_diagnose_axiom_energy")
        vpa_bipolar = (vpa * 2 - 1).clip(-1, 1)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        cmf_bipolar = get_adaptive_mtf_normalized_bipolar_score(cmf, df.index, default_weights)
        energy_score = (
            vpa_bipolar * 0.5 +
            cmf_bipolar * 0.5
        ).clip(-1, 1)
        return energy_score.astype(np.float32)

    def _diagnose_axiom_ma_dynamics(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.4 · 信号校验增强版】力学公理六：诊断“均线动态”
        - 核心逻辑: 融合均线的速度和加速度，评估趋势的内在变化。
        - 【修复】修正了引用均线速度和加速度列名时，确保其与 `IndicatorService` 中 `merge_results` 方法添加后缀后的列名一致。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `velocity_score` 和 `acceleration_score` 的归一化方式改为多时间维度自适应归一化。
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
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        velocity_score = get_adaptive_mtf_normalized_bipolar_score(velocity_raw, df_index, default_weights)
        acceleration_score = get_adaptive_mtf_normalized_bipolar_score(acceleration_raw, df_index, default_weights)
        ma_dynamics_score = (velocity_score * 0.6 + acceleration_score * 0.4).clip(-1, 1)
        return ma_dynamics_score.astype(np.float32)

