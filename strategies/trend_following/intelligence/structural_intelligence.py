import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar, is_limit_up, get_adaptive_mtf_normalized_score

class StructuralIntelligence:
    """
    【V3.0 · 三大公理重构版】
    - 核心升级: 废弃旧的复杂四支柱模型，引入基于结构本质的“趋势形态、多周期协同、结构稳定性”三大公理。
                使引擎更聚焦、逻辑更清晰、信号更纯粹。
    """
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化结构情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [结构情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def diagnose_structural_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.4 · 底分型集成版】结构情报分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出结构领域的原子公理信号和结构背离信号。
        - 移除信号: SCORE_STRUCTURE_BULLISH_RESONANCE, SCORE_STRUCTURE_BEARISH_RESONANCE, BIPOLAR_STRUCTURAL_DOMAIN_HEALTH, SCORE_STRUCTURE_BOTTOM_REVERSAL, SCORE_STRUCTURE_TOP_REVERSAL。
        - 【新增】调用 `_diagnose_bottom_fractal` 方法，将底分型信号添加到 `all_states` 中。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("结构情报引擎已在配置中禁用，跳过。")
            return {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 步骤一: 诊断三大公理 ---
        axiom_trend_form = self._diagnose_axiom_trend_form(df, norm_window)
        axiom_mtf_cohesion = self._diagnose_axiom_mtf_cohesion(df, norm_window, axiom_trend_form)
        axiom_stability = self._diagnose_axiom_stability(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        all_states['SCORE_STRUCT_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_STRUCT_AXIOM_TREND_FORM'] = axiom_trend_form
        all_states['SCORE_STRUCT_AXIOM_MTF_COHESION'] = axiom_mtf_cohesion
        all_states['SCORE_STRUCT_AXIOM_STABILITY'] = axiom_stability
        # 引入结构层面的看涨/看跌背离信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_STRUCTURE_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_STRUCTURE_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        # 【新增代码行】诊断底分型结构
        bottom_fractal_score = self._diagnose_bottom_fractal(df, n=5, min_depth_ratio=0.001)
        all_states['SCORE_STRUCT_BOTTOM_FRACTAL'] = bottom_fractal_score
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.1 · 多时间维度归一化版】结构公理四：诊断“结构背离”
        - 核心逻辑: 诊断价格行为与均线结构（如均线排列）的背离。
          - 看涨背离：价格下跌但均线排列开始收敛或转好。
          - 看跌背离：价格上涨但均线排列开始发散或恶化。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `price_trend` 和 `ma_structure_trend` 的归一化方式改为多时间维度自适应归一化。
        """
        ma_periods = [5, 13, 21, 55]
        required_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in required_cols):
            print("诊断结构背离失败：缺少必要的EMA列。")
            return pd.Series(0.0, index=df.index)
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        tf_weights_struct = get_param_value(p_conf_struct.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}) # 借用筹码的MTF权重配置
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, tf_weights_struct)
        ema_short_long_diff = self._get_safe_series(df, 'EMA_5_D', 0.0, method_name="_diagnose_axiom_divergence") - self._get_safe_series(df, 'EMA_55_D', 0.0, method_name="_diagnose_axiom_divergence")
        ma_structure_trend = get_adaptive_mtf_normalized_bipolar_score(ema_short_long_diff.diff(1), df.index, tf_weights_struct)
        divergence_score = (ma_structure_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_trend_form(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        print("    -> [结构情报] 正在诊断结构公理一：趋势形态...")
        df_index = df.index
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        tf_weights_struct = get_param_value(p_conf_struct.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        ema_periods = get_param_value(p_conf_struct.get('trend_form_ema_periods'), [5, 13, 21, 55])
        ma_periods = get_param_value(p_conf_struct.get('trend_form_ma_periods'), [5, 13, 21, 55])
        # --- EMA均线排列 (Alignment) ---
        bull_alignment_ema_raw = pd.Series(0.0, index=df_index)
        bear_alignment_ema_raw = pd.Series(0.0, index=df_index)
        ema_alignment_weights = [0.4, 0.3, 0.3]
        for i in range(len(ema_periods) - 1):
            ema_i = self._get_safe_series(df, f'EMA_{ema_periods[i]}_D', method_name="_diagnose_axiom_trend_form")
            ema_i_plus_1 = self._get_safe_series(df, f'EMA_{ema_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")
            bull_alignment_ema_raw += (ema_i > ema_i_plus_1).astype(float) * ema_alignment_weights[i]
            bear_alignment_ema_raw += (ema_i < ema_i_plus_1).astype(float) * ema_alignment_weights[i]
        price_above_all_ema = pd.Series(True, index=df_index)
        close_D = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_trend_form")
        for p in ema_periods:
            price_above_all_ema &= (close_D > self._get_safe_series(df, f'EMA_{p}_D', method_name="_diagnose_axiom_trend_form"))
        bull_alignment_ema_raw += price_above_all_ema.astype(float) * 0.5
        price_ema_distance = pd.Series(0.0, index=df_index)
        for p in ema_periods:
            price_ema_distance += (close_D - self._get_safe_series(df, f'EMA_{p}_D', method_name="_diagnose_axiom_trend_form")).clip(lower=0)
        price_ema_distance_score = get_adaptive_mtf_normalized_score(price_ema_distance, df_index, ascending=True, tf_weights=tf_weights_struct)
        bull_alignment_ema_raw += price_ema_distance_score * 0.5
        bull_alignment_ema = bull_alignment_ema_raw / (sum(ema_alignment_weights) + 1.0)
        bear_alignment_ema = bear_alignment_ema_raw / sum(ema_alignment_weights)
        # --- MA均线排列 (Alignment) ---
        bull_alignment_ma_raw = pd.Series(0.0, index=df_index)
        bear_alignment_ma_raw = pd.Series(0.0, index=df_index)
        ma_alignment_weights = [0.4, 0.3, 0.3]
        for i in range(len(ma_periods) - 1):
            ma_i = self._get_safe_series(df, f'MA_{ma_periods[i]}_D', method_name="_diagnose_axiom_trend_form")
            ma_i_plus_1 = self._get_safe_series(df, f'MA_{ma_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")
            bull_alignment_ma_raw += (ma_i > ma_i_plus_1).astype(float) * ma_alignment_weights[i]
            bear_alignment_ma_raw += (ma_i < ma_i_plus_1).astype(float) * ma_alignment_weights[i]
        price_above_all_ma = pd.Series(True, index=df_index)
        for p in ma_periods:
            price_above_all_ma &= (close_D > self._get_safe_series(df, f'MA_{p}_D', method_name="_diagnose_axiom_trend_form"))
        bull_alignment_ma_raw += price_above_all_ma.astype(float) * 0.5
        price_ma_distance = pd.Series(0.0, index=df_index)
        for p in ma_periods:
            price_ma_distance += (close_D - self._get_safe_series(df, f'MA_{p}_D', method_name="_diagnose_axiom_trend_form")).clip(lower=0)
        price_ma_distance_score = get_adaptive_mtf_normalized_score(price_ma_distance, df_index, ascending=True, tf_weights=tf_weights_struct)
        bull_alignment_ma_raw += price_ma_distance_score * 0.5
        bull_alignment_ma = bull_alignment_ma_raw / (sum(ma_alignment_weights) + 1.0)
        bear_alignment_ma = bear_alignment_ma_raw / sum(ma_alignment_weights)
        # 融合EMA和MA的排列分数
        alignment_fusion_weights = get_param_value(p_conf_struct.get('trend_form_alignment_fusion_weights'), {'ema': 0.6, 'ma': 0.4})
        bull_alignment = (bull_alignment_ema * alignment_fusion_weights['ema'] + bull_alignment_ma * alignment_fusion_weights['ma']).clip(0, 1)
        bear_alignment = (bear_alignment_ema * alignment_fusion_weights['ema'] + bear_alignment_ma * alignment_fusion_weights['ma']).clip(0, 1)
        # --- EMA均线速度 (Velocity) ---
        slope_ema_cols = [f'SLOPE_5_EMA_{p}_D' for p in ema_periods if f'SLOPE_5_EMA_{p}_D' in df.columns]
        bull_velocity_ema_raw = pd.Series(0.0, index=df_index)
        bear_velocity_ema_raw = pd.Series(0.0, index=df_index)
        if slope_ema_cols:
            for col in slope_ema_cols:
                bull_velocity_ema_raw += self._get_safe_series(df, col, method_name="_diagnose_axiom_trend_form").clip(lower=0)
                bear_velocity_ema_raw += self._get_safe_series(df, col, method_name="_diagnose_axiom_trend_form").clip(upper=0).abs()
        # --- MA均线速度 (Velocity) ---
        slope_ma_cols = [f'SLOPE_5_MA_{p}_D' for p in ma_periods if f'SLOPE_5_MA_{p}_D' in df.columns]
        bull_velocity_ma_raw = pd.Series(0.0, index=df_index)
        bear_velocity_ma_raw = pd.Series(0.0, index=df_index)
        if slope_ma_cols:
            for col in slope_ma_cols:
                bull_velocity_ma_raw += self._get_safe_series(df, col, method_name="_diagnose_axiom_trend_form").clip(lower=0)
                bear_velocity_ma_raw += self._get_safe_series(df, col, method_name="_diagnose_axiom_trend_form").clip(upper=0).abs()
        pct_change_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_trend_form").clip(lower=0), df_index, ascending=True, tf_weights=tf_weights_struct)
        bull_velocity_ema_raw += pct_change_score * 0.5
        bull_velocity_ma_raw += pct_change_score * 0.5
        # 融合EMA和MA的速度分数
        velocity_fusion_weights = get_param_value(p_conf_struct.get('trend_form_velocity_fusion_weights'), {'ema': 0.6, 'ma': 0.4})
        bull_velocity = get_adaptive_mtf_normalized_score(bull_velocity_ema_raw * velocity_fusion_weights['ema'] + bull_velocity_ma_raw * velocity_fusion_weights['ma'], df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        bear_velocity = get_adaptive_mtf_normalized_score(bear_velocity_ema_raw * velocity_fusion_weights['ema'] + bear_velocity_ma_raw * velocity_fusion_weights['ma'], df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        # --- 引入成交量爆裂度、上涨推力效能 ---
        volume_burstiness_raw = self._get_safe_series(df, 'volume_ratio_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form")
        upward_thrust_efficacy_raw = self._get_safe_series(df, 'upward_impulse_purity_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form")
        downward_absorption_efficacy_raw = self._get_safe_series(df, 'dip_absorption_power_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form")
        burstiness_score = get_adaptive_mtf_normalized_score(volume_burstiness_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        upward_efficacy_score = get_adaptive_mtf_normalized_score(upward_thrust_efficacy_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        downward_efficacy_score = get_adaptive_mtf_normalized_score(downward_absorption_efficacy_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        # --- 引入均线角度 ---
        ma_angle_raw = self._get_safe_series(df, f'ATAN_ANGLE_EMA_55_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form")
        ma_angle_score = get_adaptive_mtf_normalized_bipolar_score(ma_angle_raw, df_index, tf_weights_struct, sensitivity=10.0)
        # [代码修改开始] 使用 trend_vitality_index_D 和 closing_price_deviation_score_D 替代缺失的信号
        trend_quality_raw = self._get_safe_series(df, 'trend_vitality_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form")
        closing_momentum_raw = self._get_safe_series(df, 'closing_price_deviation_score_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form")
        # [代码修改结束]
        trend_quality_score = pd.Series(0.0, index=df_index)
        closing_momentum_score = pd.Series(0.0, index=df_index)
        if not trend_quality_raw.isnull().all() and not (trend_quality_raw == 0.0).all():
            trend_quality_score = get_adaptive_mtf_normalized_bipolar_score(trend_quality_raw, df.index, tf_weights_struct, sensitivity=0.5)
        if not closing_momentum_raw.isnull().all() and not (closing_momentum_raw == 0.0).all():
            closing_momentum_score = get_adaptive_mtf_normalized_bipolar_score(closing_momentum_raw, df.index, tf_weights_struct, sensitivity=0.5)
        # 融合牛熊分数
        bull_score = (
            bull_alignment * bull_velocity * (1 + burstiness_score * 0.2) +
            upward_efficacy_score * 0.1 +
            ma_angle_score.clip(lower=0) * 0.1 +
            trend_quality_score.clip(lower=0) * 0.1 +
            closing_momentum_score.clip(lower=0) * 0.1
        ).clip(0, 1)
        bear_score = (
            bear_alignment * bear_velocity * (1 + burstiness_score * 0.2) +
            downward_efficacy_score * 0.1 +
            ma_angle_score.clip(upper=0).abs() * 0.1 +
            trend_quality_score.clip(upper=0).abs() * 0.1 +
            closing_momentum_score.clip(upper=0).abs() * 0.1
        ).clip(0, 1)
        bull_score = bull_score.mask(is_limit_up_day, bull_score + 0.5).clip(0, 1)
        bear_score = bear_score.mask(is_limit_up_day, bear_score * 0.1).clip(0, 1)
        trend_form_score = (bull_score - bear_score).clip(-1, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [趋势形态探针] @ {probe_date_for_loop.date()}:")
                print(f"       - bull_alignment_ema_raw: {bull_alignment_ema_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_above_all_ema: {price_above_all_ema.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_ema_distance_score: {price_ema_distance_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_alignment_ema: {bull_alignment_ema.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_alignment_ma_raw: {bull_alignment_ma_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_above_all_ma: {price_above_all_ma.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_ma_distance_score: {price_ma_distance_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_alignment_ma: {bull_alignment_ma.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_alignment: {bull_alignment.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_velocity_ema_raw: {bull_velocity_ema_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_velocity_ma_raw: {bull_velocity_ma_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - pct_change_score: {pct_change_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_velocity: {bull_velocity.loc[probe_date_for_loop]:.4f}")
                print(f"       - burstiness_score: {burstiness_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - upward_efficacy_score: {upward_efficacy_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - ma_angle_score: {ma_angle_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_quality_raw: {trend_quality_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - closing_momentum_raw: {closing_momentum_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_quality_score: {trend_quality_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - closing_momentum_score: {closing_momentum_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_score: {bull_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bear_score: {bear_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_day: {is_limit_up_day.loc[probe_date_for_loop]}")
                print(f"       - trend_form_score: {trend_form_score.loc[probe_date_for_loop]:.4f}")
        return trend_form_score

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 微观结构博弈信号缺失处理与物理直观重构及共识增强版】结构公理三：诊断“结构稳定性”
        - 核心重构: 废除对间接且脆弱的 'MA_CONV_CV_SHORT_D' 的依赖。将“能量积蓄度”的评估核心完全聚焦于更直观、更稳健的布林带宽度(BBW)指标。
        - 诊断维度:
          1. 能量积蓄度 (Energy Accumulation): 直接使用BBW的收缩程度。
          2. 基石支撑度 (Foundation Support): 价格是否站稳在关键长期MA(55, 144)之上。
          3. 长期趋势健康度 (Long-term Trend Health): 关键长期MA自身的斜率方向。
        - 核心修复: 将 `raw_stability_score` 的计算从乘法改为加权平均，避免“一票否决”效应。
        - 增加探针，打印 `foundation_support_score` 和 `long_term_trend_health_score`。
        - 引入 `vpoc_consensus_strength_D` (VPOC共识强度) 和 `volatility_skew_index_D` (波动率偏度指数)
                   来增强对结构稳定性的判断。
        - 【修正】优化 `foundation_support_score` 和 `long_term_trend_health_score` 的计算，使其在涨停日能正确反映积极稳定性。
        - 【修复】移除 `normalize_score` 函数调用中的 `sensitivity` 参数，因为该函数不接受此参数。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        - 【新增】引入 `volume_structure_skew_D` (成交结构偏度) 作为判断结构稳定性的微观证据。
        - 【修正】当 `volume_structure_skew_D` 信号缺失时，不将其纳入融合计算。
        """
        df_index = df.index
        bbw_col = 'BBW_21_2.0_D'
        if bbw_col not in df.columns:
            print(f"    -> [结构情报警告] 方法 '_diagnose_axiom_stability' 缺少核心列 '{bbw_col}'。")
            return pd.Series(0.0, index=df_index)
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        tf_weights_struct = get_param_value(p_conf_struct.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        energy_accumulation_score = 1 - get_adaptive_mtf_normalized_score(self._get_safe_series(df, bbw_col, method_name="_diagnose_axiom_stability"), df_index, ascending=True, tf_weights=tf_weights_struct)
        energy_accumulation_score = energy_accumulation_score.fillna(0.5)
        long_term_ma_periods = [55, 144]
        required_ma_cols = [f'MA_{p}_D' for p in long_term_ma_periods]
        required_slope_cols = [f'SLOPE_5_MA_{p}_D' for p in long_term_ma_periods] # 默认使用5日斜率
        # 获取并归一化 vpoc_consensus_strength_D
        vpoc_consensus_raw = self._get_safe_series(df, 'mf_vpoc_premium_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_stability") # 替换为 mf_vpoc_premium_D
        vpoc_consensus_score = get_adaptive_mtf_normalized_score(vpoc_consensus_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        # 获取成交结构偏度，并检查是否缺失
        volume_structure_skew_raw = self._get_safe_series(df, 'volume_structure_skew_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_stability")
        volume_structure_skew_score = pd.Series(0.0, index=df_index)
        if not volume_structure_skew_raw.isnull().all() and not (volume_structure_skew_raw == 0.0).all():
            volume_structure_skew_score = get_adaptive_mtf_normalized_bipolar_score(volume_structure_skew_raw * -1, df.index, tf_weights_struct, sensitivity=0.5)
        if not all(col in df.columns for col in required_ma_cols + required_slope_cols):
            print("    -> [结构情报警告] 方法 '_diagnose_axiom_stability' 缺少必要的长期MA或其斜率列，长期结构评估将跳过。")
            foundation_health_score = pd.Series(0.5, index=df_index)
            foundation_support_score = pd.Series(0.5, index=df_index)
            long_term_trend_health_score = pd.Series(0.5, index=df_index)
        else:
            support_scores = []
            for p in long_term_ma_periods:
                support_score = get_adaptive_mtf_normalized_score((self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_stability") - self._get_safe_series(df, f'MA_{p}_D', method_name="_diagnose_axiom_stability")).clip(lower=0), df_index, ascending=True, tf_weights=tf_weights_struct).clip(0, 1)
                support_scores.append(support_score)
            foundation_support_score = pd.Series(np.mean(support_scores, axis=0), index=df_index)
            health_scores = []
            for p in long_term_ma_periods:
                slope_col_name = f'SLOPE_21_MA_{p}_D'
                if slope_col_name not in df.columns:
                    slope_col_name = f'SLOPE_5_MA_{p}_D'
                health_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, slope_col_name, pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_stability").clip(lower=0), df_index, ascending=True, tf_weights=tf_weights_struct).clip(0, 1)
                health_scores.append(health_score)
            long_term_trend_health_score = pd.Series(np.mean(health_scores, axis=0), index=df_index)
            foundation_health_score = (foundation_support_score * 0.5 + long_term_trend_health_score * 0.3 + vpoc_consensus_score * 0.2).clip(0, 1)
        raw_stability_score = (energy_accumulation_score * 0.3 + foundation_health_score * 0.7).fillna(0.5)
        volatility_skew_raw = self._get_safe_series(df, 'volatility_asymmetry_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_stability")
        volatility_skew_score = get_adaptive_mtf_normalized_bipolar_score(volatility_skew_raw, df_index, tf_weights_struct, sensitivity=0.5).fillna(0.0)
        # 融合所有分数，调整权重
        stability_score = (
            raw_stability_score * 0.7 +
            volatility_skew_score.clip(lower=0) * 0.15 -
            volatility_skew_score.clip(upper=0).abs() * 0.15
        )
        # 只有当 volume_structure_skew_score 有效时才加入融合
        if not volume_structure_skew_score.isnull().all() and not (volume_structure_skew_score == 0.0).all():
            stability_score += volume_structure_skew_score * 0.15
        stability_score = stability_score.clip(-1, 1)
        # --- Debugging output for probe date ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [结构稳定性探针] @ {probe_date_for_loop.date()}:")
                print(f"       - energy_accumulation_score: {energy_accumulation_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - foundation_support_score: {foundation_support_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - long_term_trend_health_score: {long_term_trend_health_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - vpoc_consensus_score: {vpoc_consensus_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - foundation_health_score: {foundation_health_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - raw_stability_score: {raw_stability_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - volatility_skew_raw: {volatility_skew_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - volatility_skew_score: {volatility_skew_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - volume_structure_skew_raw: {volume_structure_skew_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - volume_structure_skew_score: {volume_structure_skew_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - stability_score: {stability_score.loc[probe_date_for_loop]:.4f}")
        return stability_score

    def _diagnose_axiom_mtf_cohesion(self, df: pd.DataFrame, norm_window: int, daily_trend_form_score: pd.Series) -> pd.Series:
        """
        【V1.4 · 涨停日多周期协同增强与多时间维度归一化版】结构公理二：诊断“多周期协同”
        - 引入 `trend_efficiency_ratio_D` (趋势效率比) 来增强多周期协同的质量判断。
        - 【修正】优化 `cohesion_score` 的计算，使其在积极信号时贡献正分。
        - 【增强】在涨停日，大幅增强 `weekly_trend_form_score` 的贡献，并优化 `cohesion_score` 的融合逻辑。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        """
        df_index = df.index
        ma_periods_w = [5, 13, 21, 55]
        required_cols_w = [f'EMA_{p}_W' for p in ma_periods_w]
        if not all(col in df.columns for col in required_cols_w):
            print("诊断多周期协同失败：缺少必要的周线EMA列，将仅使用日线结构。")
            return pd.Series(0.0, index=df_index)
        # 判断是否为涨停日
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        tf_weights_struct = get_param_value(p_conf_struct.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 周线均线排列 (Weekly Alignment) ---
        bull_alignment_w_raw = pd.Series(0.0, index=df_index)
        bear_alignment_w_raw = pd.Series(0.0, index=df_index)
        for i in range(len(ma_periods_w) - 1):
            bull_alignment_w_raw += (self._get_safe_series(df, f'EMA_{ma_periods_w[i]}_W', method_name="_diagnose_axiom_mtf_cohesion") > self._get_safe_series(df, f'EMA_{ma_periods_w[i+1]}_W', method_name="_diagnose_axiom_mtf_cohesion")).astype(float)
            bear_alignment_w_raw += (self._get_safe_series(df, f'EMA_{ma_periods_w[i]}_W', method_name="_diagnose_axiom_mtf_cohesion") < self._get_safe_series(df, f'EMA_{ma_periods_w[i+1]}_W', method_name="_diagnose_axiom_mtf_cohesion")).astype(float)
        bull_alignment_w = bull_alignment_w_raw / (len(ma_periods_w) - 1)
        bear_alignment_w = bear_alignment_w_raw / (len(ma_periods_w) - 1)
        # --- 周线均线速度 (Weekly Velocity) ---
        slope_cols_w = [f'SLOPE_5_EMA_{p}_W' for p in ma_periods_w if f'SLOPE_5_EMA_{p}_W' in df.columns]
        if not slope_cols_w:
            return pd.Series(0.0, index=df_index)
        bull_velocity_w_raw = pd.Series(0.0, index=df_index)
        bear_velocity_w_raw = pd.Series(0.0, index=df_index)
        for col in slope_cols_w:
            bull_velocity_w_raw += self._get_safe_series(df, col, method_name="_diagnose_axiom_mtf_cohesion").clip(lower=0)
            bear_velocity_w_raw += self._get_safe_series(df, col, method_name="_diagnose_axiom_mtf_cohesion").clip(upper=0).abs()
        # 增强 bull_velocity_w: 引入 pct_change_W 的贡献
        pct_change_w_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'pct_change_W', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_mtf_cohesion").clip(lower=0), df_index, ascending=True, tf_weights=tf_weights_struct)
        bull_velocity_w_raw += pct_change_w_score * 0.5 # 额外权重
        bull_velocity_w = get_adaptive_mtf_normalized_score(bull_velocity_w_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        bear_velocity_w = get_adaptive_mtf_normalized_score(bear_velocity_w_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        weekly_trend_form_score = (pd.Series(bull_alignment_w * bull_velocity_w, index=df_index) - pd.Series(bear_alignment_w * bear_velocity_w, index=df_index)).clip(-1, 1)
        # 涨停日特殊处理：如果日线趋势形态为正，则周线趋势形态也应该至少有一个积极的基础分
        weekly_trend_form_score = weekly_trend_form_score.mask(is_limit_up_day & (daily_trend_form_score > 0), weekly_trend_form_score + 0.3).clip(-1, 1)
        # 获取并归一化 trend_efficiency_ratio_D
        trend_efficiency_raw = self._get_safe_series(df, 'VPA_EFFICIENCY_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_mtf_cohesion") # 替换为 VPA_EFFICIENCY_D
        efficiency_score = get_adaptive_mtf_normalized_score(trend_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        # 融合效率分数。效率分数作为乘数因子，增强协同的质量。
        # 确保在积极信号时贡献正分。
        # 优化 cohesion_score 融合逻辑，在涨停日，如果日线趋势形态为正，则协同分数更倾向于正向
        cohesion_score = (daily_trend_form_score.clip(lower=0) * weekly_trend_form_score.clip(lower=0) * (1 + efficiency_score * 0.5) -
                          daily_trend_form_score.clip(upper=0).abs() * weekly_trend_form_score.clip(upper=0).abs() * (1 + efficiency_score * 0.5)).fillna(0).clip(-1, 1)
        # 涨停日进一步增强协同分数
        cohesion_score = cohesion_score.mask(is_limit_up_day & (daily_trend_form_score > 0), cohesion_score + 0.2).clip(-1, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [多周期协同探针] @ {probe_date_for_loop.date()}:")
                print(f"       - bull_alignment_w: {bull_alignment_w.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_velocity_w: {bull_velocity_w.loc[probe_date_for_loop]:.4f}")
                print(f"       - pct_change_w_score: {pct_change_w_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - weekly_trend_form_score: {weekly_trend_form_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - efficiency_score: {efficiency_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - daily_trend_form_score: {daily_trend_form_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_day: {is_limit_up_day.loc[probe_date_for_loop]}")
                print(f"       - cohesion_score: {cohesion_score.loc[probe_date_for_loop]:.4f}")
        return cohesion_score

    def _diagnose_bottom_fractal(self, df: pd.DataFrame, n: int = 5, min_depth_ratio: float = 0.001) -> pd.Series:
        """
        【V1.0】结构公理五：诊断“底分型”结构
        - 核心逻辑: 识别底分型结构形态，并输出一个双极性分数 `SCORE_STRUCT_BOTTOM_FRACTAL`。
        - 底分型定义: 中间K线的最低价是其左右各 (n-1)/2 根K线中的最低价。
        - 信号输出: 识别到底分型时，输出1.0；否则输出0.0。
        - 引入 `min_depth_ratio` 过滤微小波动形成的底分型。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        bottom_fractal_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # 确保 n 是大于等于3的奇数
        if n % 2 == 0 or n < 3:
            print(f"    -> [结构情报警告] 底分型识别错误: 参数 n 必须是大于等于3的奇数，当前为 {n}。将使用默认值5。")
            n = 5
        half_n = n // 2
        # 确保 'low' 列存在
        if 'low_D' not in df.columns:
            print(f"    -> [结构情报警告] 诊断底分型失败: 缺少 'low_D' 列，返回默认分数。")
            return bottom_fractal_score
        low_series = self._get_safe_series(df, 'low_D', method_name="_diagnose_bottom_fractal")
        for i in range(half_n, len(df) - half_n):
            middle_low = low_series.iloc[i]
            is_bottom = True
            surrounding_lows = []
            for j in range(i - half_n, i + half_n + 1):
                if j == i:
                    continue
                current_low = low_series.iloc[j]
                surrounding_lows.append(current_low)
                if middle_low >= current_low:
                    is_bottom = False
                    break
            if is_bottom:
                # 进一步检查深度比例
                if min_depth_ratio > 0:
                    avg_surrounding_low = np.mean(surrounding_lows)
                    if avg_surrounding_low <= 0: # 避免除以零或负数
                        is_bottom = False
                    elif (avg_surrounding_low - middle_low) / avg_surrounding_low < min_depth_ratio:
                        is_bottom = False
            if is_bottom:
                bottom_fractal_score.iloc[i] = 1.0 # 识别到底分型，记为1.0
        # 调试信息
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [底分型探针] @ {probe_date_for_loop.date()}:")
                print(f"       - 底分型分数 (SCORE_STRUCT_BOTTOM_FRACTAL): {bottom_fractal_score.loc[probe_date_for_loop]:.4f}")
        return bottom_fractal_score


