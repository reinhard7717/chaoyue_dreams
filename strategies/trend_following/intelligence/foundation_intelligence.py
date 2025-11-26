import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, get_adaptive_mtf_normalized_score, bipolar_to_exclusive_unipolar

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [基础情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # [新增] 调整校验信息为“基础情报校验”
            print(f"    -> [基础情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_foundation_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.6 · 结构形态公理版】基础情报分析总指挥
        - 【V6.5 修复】接收 df 参数作为统一的数据上下文，并移除内部对 self.strategy.df_indicators 的依赖。
        - [新增] 调用新增的 _diagnose_axiom_structure_form 方法，引入结构形态公理。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("基础情报引擎已在配置中禁用，跳过。")
            return {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        axiom_trend = self._diagnose_axiom_trend(df, norm_window, p_conf)
        axiom_oscillator = self._diagnose_axiom_oscillator(df, norm_window)
        axiom_flow = self._diagnose_axiom_flow(df, norm_window)
        axiom_volatility = self._diagnose_axiom_volatility(df, norm_window)
        # 调用新增的结构形态公理诊断方法
        axiom_structure_form = self._diagnose_axiom_structure_form(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        all_states['SCORE_FOUNDATION_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FOUNDATION_AXIOM_TREND'] = axiom_trend
        all_states['SCORE_FOUNDATION_AXIOM_OSCILLATOR'] = axiom_oscillator
        all_states['SCORE_FOUNDATION_AXIOM_FLOW'] = axiom_flow
        all_states['SCORE_FOUNDATION_AXIOM_VOLATILITY'] = axiom_volatility
        # 将新的结构形态公理分数存入状态字典
        all_states['SCORE_FOUNDATION_AXIOM_STRUCTURE_FORM'] = axiom_structure_form
        context_trend_confirmed = self._diagnose_context_trend_confirmed(df, norm_window)
        all_states.update(context_trend_confirmed)
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_FOUNDATION_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FOUNDATION_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_states

    def _diagnose_context_trend_confirmed(self, df: pd.DataFrame, norm_window: int) -> Dict[str, pd.Series]:
        """
        【V1.2 · 信号校验增强版】诊断内部上下文信号：趋势确认分 (CONTEXT_TREND_CONFIRMED)
        - 核心逻辑: 融合趋势强度(ADX)、方向(PDI/NDI)和健康度(BIAS)，评估上升趋势的确认程度。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['ADX_14_D', 'PDI_14_D', 'NDI_14_D', 'SLOPE_5_PDI_14_D', 'BIAS_55_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_context_trend_confirmed"):
            return {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_mtf.get('long_term_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        adx_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'ADX_14_D', 0.0, method_name="_diagnose_context_trend_confirmed"), df.index, ascending=True, tf_weights=default_weights)
        pdi_gt_ndi = (self._get_safe_series(df, 'PDI_14_D', 0, method_name="_diagnose_context_trend_confirmed") > self._get_safe_series(df, 'NDI_14_D', 0, method_name="_diagnose_context_trend_confirmed")).astype(float)
        pdi_slope = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'SLOPE_5_PDI_14_D', 0.0, method_name="_diagnose_context_trend_confirmed"), df.index, ascending=True, tf_weights=default_weights)
        direction_score = (pdi_gt_ndi * pdi_slope).pow(0.5)
        bias_health_score = 1 - get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'BIAS_55_D', pd.Series(0.0, index=df.index), method_name="_diagnose_context_trend_confirmed").clip(lower=0), df.index, ascending=True, tf_weights=long_term_weights)
        trend_confirmed = (adx_score * direction_score * bias_health_score).pow(1/3).fillna(0.0)
        return {'CONTEXT_TREND_CONFIRMED': trend_confirmed.astype(np.float32)}

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.2 · 信号校验增强版】基础公理五：诊断“基础背离”
        - 核心逻辑: 诊断价格趋势与摆动指标（如RSI）之间的背离。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `price_trend` 和 `oscillator_trend` 的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['SLOPE_13_close_D', 'SLOPE_13_RSI_13_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'SLOPE_13_close_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, default_weights)
        oscillator_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'SLOPE_13_RSI_13_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, default_weights)
        divergence_score = (oscillator_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_trend(self, df: pd.DataFrame, norm_window: int, params: dict) -> pd.Series:
        """
        【V1.4 · 信号校验增强版】基础公理一：诊断“趋势”
        - 【新增】引入 DMA 指标的斜率作为趋势判断的辅助证据。
        - 【修复】修正了引用 DMA 斜率列名时，确保其与 `IndicatorService` 中 `merge_results` 方法添加后缀后的列名一致。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        ma_periods = [5, 13, 21, 55]
        required_signals = ['MACDh_13_34_8_D', 'SLOPE_5_DMA_D']
        required_signals.extend([f'EMA_{p}_D' for p in ma_periods])
        required_signals.extend([f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend"):
            return pd.Series(0.0, index=df.index)
        macd_h = self._get_safe_series(df, 'MACDh_13_34_8_D', 0.0, method_name="_diagnose_axiom_trend")
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        macd_score = get_adaptive_mtf_normalized_bipolar_score(macd_h, df.index, default_weights)
        fusion_weights = params.get('ma_health_fusion_weights', {'alignment': 0.5, 'slope': 0.5})
        bull_alignment_scores = [(self._get_safe_series(df, f'EMA_{ma_periods[i]}_D', method_name="_diagnose_axiom_trend") > self._get_safe_series(df, f'EMA_{ma_periods[i+1]}_D', method_name="_diagnose_axiom_trend")).astype(float).values for i in range(len(ma_periods) - 1)]
        alignment_score = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df.index), 0.5)
        alignment_bipolar = (pd.Series(alignment_score, index=df.index) - 0.5) * 2
        slope_scores = [get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, f'SLOPE_{p}_EMA_{p}_D', 0.0, method_name="_diagnose_axiom_trend"), df.index, default_weights).values for p in ma_periods]
        avg_slope_bipolar = pd.Series(np.mean(slope_scores, axis=0), index=df.index)
        dma_slope = self._get_safe_series(df, 'SLOPE_5_DMA_D', 0.0, method_name="_diagnose_axiom_trend")
        dma_slope_score = get_adaptive_mtf_normalized_bipolar_score(dma_slope, df.index, default_weights)
        structure_score = (
            alignment_bipolar * fusion_weights.get('alignment', 0.5) +
            avg_slope_bipolar * fusion_weights.get('slope', 0.5)
        ).clip(-1, 1)
        trend_score = (macd_score * 0.3 + structure_score * 0.5 + dma_slope_score * 0.2).clip(-1, 1)
        return trend_score.astype(np.float32)

    def _diagnose_axiom_oscillator(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.2 · 信号校验增强版】基础公理二：诊断“摆动”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `oscillator_score` 的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['RSI_13_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_oscillator"):
            return pd.Series(0.0, index=df.index)
        rsi = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name="_diagnose_axiom_oscillator")
        raw_bipolar_series = rsi - 50.0
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        oscillator_score = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df.index, default_weights, sensitivity=10.0)
        return oscillator_score.astype(np.float32)

    def _diagnose_axiom_flow(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.3 · 信号校验增强版】基础公理三：诊断“流体”
        - 核心升级: 增加调试探针，打印 CMF 原始值和归一化分数。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `flow_score` 的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['CMF_21_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        cmf = self._get_safe_series(df, 'CMF_21_D', 0.0, method_name="_diagnose_axiom_flow")
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        flow_score = get_adaptive_mtf_normalized_bipolar_score(cmf, df_index, default_weights, sensitivity=0.5) # 提高敏感度
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [基础流体探针] @ {probe_date_for_loop.date()}:")
                print(f"       - CMF_21_D: {cmf.loc[probe_date_for_loop]:.4f}")
                print(f"       - flow_score: {flow_score.loc[probe_date_for_loop]:.4f}")
        return flow_score.astype(np.float32)

    def _diagnose_axiom_volatility(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.2 · 信号校验增强版】基础公理四：诊断“波动”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `volatility_score` 的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['BBW_21_2.0_D', 'ATR_14_D', 'close_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_volatility"):
            return pd.Series(0.0, index=df.index)
        bbw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name="_diagnose_axiom_volatility")
        atr_pct = self._get_safe_series(df, 'ATR_14_D', 0.0, method_name="_diagnose_axiom_volatility") / self._get_safe_series(df, 'close_D', 1e-9, method_name="_diagnose_axiom_volatility")
        raw_volatility = bbw + atr_pct
        raw_bipolar_series = -raw_volatility
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        volatility_score = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df.index, default_weights, sensitivity=1.0)
        return volatility_score.astype(np.float32)

    def _diagnose_axiom_structure_form(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】基础公理六：诊断“结构形态” (平台与趋势线)
        - 核心逻辑: 融合平台势能与趋势动能，评估当前市场结构。
        - 平台势能: 基于 IS_HIGH_POTENTIAL_CONSOLIDATION_D 和 BBW_21_2.0_D。
        - 趋势动能: 基于 ATAN_ANGLE_EMA_55_D 和 ZIG_5_5.0_D。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'BBW_21_2.0_D', 'ATAN_ANGLE_EMA_55_D', 'ZIG_5_5.0_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_structure_form"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 计算平台势能分 (Potential Energy Score)
        is_platform = self._get_safe_series(df, 'IS_HIGH_POTENTIAL_CONSOLIDATION_D', 0.0, method_name="_diagnose_axiom_structure_form").astype(float)
        # BBW越小，squeeze程度越高，分数越高
        squeeze_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name="_diagnose_axiom_structure_form"), df_index, ascending=False, tf_weights=default_weights)
        platform_score = (is_platform * squeeze_score).clip(0, 1)
        # 2. 计算趋势动能分 (Kinetic Energy Score)
        angle_score = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'ATAN_ANGLE_EMA_55_D', 0.0, method_name="_diagnose_axiom_structure_form"), df_index, default_weights)
        zigzag_slope = self._get_safe_series(df, 'ZIG_5_5.0_D', 0.0, method_name="_diagnose_axiom_structure_form").diff().fillna(0)
        zigzag_score = get_adaptive_mtf_normalized_bipolar_score(zigzag_slope, df_index, default_weights)
        trend_score = (angle_score * 0.7 + zigzag_score * 0.3).clip(-1, 1)
        # 3. 融合势能与动能
        # 核心公式: final_score = trend_score * (1 - platform_score) + platform_score * 0.1
        # 释义: 当处于平台时(platform_score -> 1)，最终得分趋向于一个小的正值(0.1)，代表中性偏多的蓄势状态。
        #       当不处于平台时(platform_score -> 0)，最终得分完全由趋势决定。
        final_score = trend_score * (1 - platform_score) + platform_score * 0.1
        return final_score.clip(-1, 1).astype(np.float32)
