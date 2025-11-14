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
        【V4.3 · 纯粹原子版】结构情报分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出结构领域的原子公理信号和结构背离信号。
        - 移除信号: SCORE_STRUCTURE_BULLISH_RESONANCE, SCORE_STRUCTURE_BEARISH_RESONANCE, BIPOLAR_STRUCTURAL_DOMAIN_HEALTH, SCORE_STRUCTURE_BOTTOM_REVERSAL, SCORE_STRUCTURE_TOP_REVERSAL。
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
        """
        【V1.9 · 涨停日趋势形态增强与多时间维度归一化版】结构公理一：诊断“趋势形态”
        - 引入 `volume_burstiness_index_D` (成交量爆裂度指数) 和 `upward_thrust_efficacy_D` (上涨推力效能)
                   来增强对趋势形态强度和质量的判断。
        - 【新增】引入均线角度（ATAN）作为趋势形态判断的证据。
        - 【修复】修正了引用均线角度列名时，确保其与 `IndicatorService` 中 `merge_results` 方法添加后缀后的列名一致。
        - 【修正】优化 `bull_alignment` 和 `bull_velocity` 的计算，使其在涨停日能更准确地反映积极趋势形态。
        - 【修复】移除 `normalize_score` 函数调用中的 `sensitivity` 参数，因为该函数不接受此参数。
        - 【增强】在涨停日，大幅增强 `bull_alignment` 和 `bull_velocity` 的贡献，并抑制 `bear_score`。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        """
        df_index = df.index
        ma_periods = [5, 13, 21, 55]
        required_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in required_cols):
            print("诊断趋势形态失败：缺少必要的EMA列。")
            return pd.Series(0.0, index=df_index)
        # 判断是否为涨停日
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        tf_weights_struct = get_param_value(p_conf_struct.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 均线排列 (Alignment) ---
        bull_alignment_raw = pd.Series(0.0, index=df_index)
        bear_alignment_raw = pd.Series(0.0, index=df_index)
        weights = [0.4, 0.3, 0.3] # 5-13, 13-21, 21-55 的权重
        # 基础均线交叉排列
        for i in range(len(ma_periods) - 1):
            bull_alignment_raw += (self._get_safe_series(df, f'EMA_{ma_periods[i]}_D', method_name="_diagnose_axiom_trend_form") > self._get_safe_series(df, f'EMA_{ma_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")).astype(float) * weights[i]
            bear_alignment_raw += (self._get_safe_series(df, f'EMA_{ma_periods[i]}_D', method_name="_diagnose_axiom_trend_form") < self._get_safe_series(df, f'EMA_{ma_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")).astype(float) * weights[i]
        # 增强 bull_alignment: 价格在所有均线之上
        price_above_all_ma = pd.Series(True, index=df_index)
        for p in ma_periods:
            price_above_all_ma &= (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_trend_form") > self._get_safe_series(df, f'EMA_{p}_D', method_name="_diagnose_axiom_trend_form"))
        bull_alignment_raw += price_above_all_ma.astype(float) * 0.5 # 额外权重
        # 价格与均线距离的积极贡献
        price_ma_distance = pd.Series(0.0, index=df_index)
        for p in ma_periods:
            price_ma_distance += (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_trend_form") - self._get_safe_series(df, f'EMA_{p}_D', method_name="_diagnose_axiom_trend_form")).clip(lower=0) # 只考虑价格在均线之上
        price_ma_distance_score = get_adaptive_mtf_normalized_score(price_ma_distance, df_index, ascending=True, tf_weights=tf_weights_struct)
        bull_alignment_raw += price_ma_distance_score * 0.5 # 额外权重
        bull_alignment = bull_alignment_raw / (sum(weights) + 1.0) # 归一化到 [0, 1]
        bear_alignment = bear_alignment_raw / sum(weights) # 归一化到 [0, 1]
        # --- 均线速度 (Velocity) ---
        slope_cols = [f'SLOPE_5_EMA_{p}_D' for p in ma_periods if f'SLOPE_5_EMA_{p}_D' in df.columns]
        if not slope_cols:
            return pd.Series(0.0, index=df_index)
        bull_velocity_raw = pd.Series(0.0, index=df_index)
        bear_velocity_raw = pd.Series(0.0, index=df_index)
        for col in slope_cols:
            bull_velocity_raw += self._get_safe_series(df, col, method_name="_diagnose_axiom_trend_form").clip(lower=0) # 只累加正向斜率
            bear_velocity_raw += self._get_safe_series(df, col, method_name="_diagnose_axiom_trend_form").clip(upper=0).abs() # 只累加负向斜率的绝对值
        # 增强 bull_velocity: 引入 pct_change_D 的贡献
        pct_change_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_trend_form").clip(lower=0), df_index, ascending=True, tf_weights=tf_weights_struct)
        bull_velocity_raw += pct_change_score * 0.5 # 额外权重
        # 调整 normalize_score 的敏感度，使其在涨停日能得到更高的分数
        bull_velocity = get_adaptive_mtf_normalized_score(bull_velocity_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        bear_velocity = get_adaptive_mtf_normalized_score(bear_velocity_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        # --- 引入成交量爆裂度、上涨推力效能 ---
        volume_burstiness_raw = self._get_safe_series(df, 'volume_ratio_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form") # 替换为 volume_ratio_D
        upward_thrust_efficacy_raw = self._get_safe_series(df, 'upward_impulse_purity_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form") # 替换为 upward_impulse_purity_D
        downward_absorption_efficacy_raw = self._get_safe_series(df, 'dip_absorption_power_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form") # 替换为 dip_absorption_power_D
        burstiness_score = get_adaptive_mtf_normalized_score(volume_burstiness_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        upward_efficacy_score = get_adaptive_mtf_normalized_score(upward_thrust_efficacy_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        downward_efficacy_score = get_adaptive_mtf_normalized_score(downward_absorption_efficacy_raw, df_index, ascending=True, tf_weights=tf_weights_struct).fillna(0.0)
        # --- 引入均线角度 ---
        ma_col_base = 'EMA_55'
        timeframe_key = 'D'
        ma_angle_raw = self._get_safe_series(df, f'ATAN_ANGLE_{ma_col_base}_{timeframe_key}', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_form")
        ma_angle_score = get_adaptive_mtf_normalized_bipolar_score(ma_angle_raw, df_index, tf_weights_struct, sensitivity=10.0)
        # --- 融合牛熊分数 ---
        bull_score = (bull_alignment * bull_velocity * (1 + burstiness_score * 0.2) + upward_efficacy_score * 0.1 + ma_angle_score.clip(lower=0) * 0.1).clip(0, 1)
        bear_score = (bear_alignment * bear_velocity * (1 + burstiness_score * 0.2) + downward_efficacy_score * 0.1 + ma_angle_score.clip(upper=0).abs() * 0.1).clip(0, 1)
        # 涨停日特殊处理：大幅增强牛分，抑制熊分
        bull_score = bull_score.mask(is_limit_up_day, bull_score + 0.5).clip(0, 1) # 涨停日额外加分
        bear_score = bear_score.mask(is_limit_up_day, bear_score * 0.1).clip(0, 1) # 涨停日熊分大幅削弱
        trend_form_score = (bull_score - bear_score).clip(-1, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [趋势形态探针] @ {probe_date_for_loop.date()}:")
                print(f"       - bull_alignment_raw: {bull_alignment_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_above_all_ma: {price_above_all_ma.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_ma_distance_score: {price_ma_distance_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_alignment: {bull_alignment.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_velocity_raw: {bull_velocity_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - pct_change_score: {pct_change_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_velocity: {bull_velocity.loc[probe_date_for_loop]:.4f}")
                print(f"       - burstiness_score: {burstiness_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - upward_efficacy_score: {upward_efficacy_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - ma_angle_score: {ma_angle_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bull_score: {bull_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - bear_score: {bear_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_day: {is_limit_up_day.loc[probe_date_for_loop]}")
                print(f"       - trend_form_score: {trend_form_score.loc[probe_date_for_loop]:.4f}")
        return trend_form_score

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

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.9 · 物理直观重构与共识增强及多时间维度归一化版】结构公理三：诊断“结构稳定性”
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
        if not all(col in df.columns for col in required_ma_cols + required_slope_cols):
            print("    -> [结构情报警告] 方法 '_diagnose_axiom_stability' 缺少必要的长期MA或其斜率列，长期结构评估将跳过。")
            foundation_health_score = pd.Series(0.5, index=df_index)
            foundation_support_score = pd.Series(0.5, index=df_index)
            long_term_trend_health_score = pd.Series(0.5, index=df_index)
        else:
            support_scores = []
            for p in long_term_ma_periods:
                # 价格高于MA越多，支撑越强，分数越高
                # 优化 support_score 的计算，使其更直接地反映价格在MA之上的强度
                # 调整 normalize_score 的敏感度，使其在涨停日能得到更高的分数
                # 【优化】使用多时间维度自适应归一化
                support_score = get_adaptive_mtf_normalized_score((self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_stability") - self._get_safe_series(df, f'MA_{p}_D', method_name="_diagnose_axiom_stability")).clip(lower=0), df_index, ascending=True, tf_weights=tf_weights_struct).clip(0, 1)
                support_scores.append(support_score)
            foundation_support_score = pd.Series(np.mean(support_scores, axis=0), index=df_index)
            health_scores = []
            for p in long_term_ma_periods:
                # MA斜率越大，趋势越健康，分数越高
                # 优化 health_score 的计算，使其更直接地反映MA斜率的正向强度
                # 考虑使用更长周期的斜率来评估长期均线的健康度，例如 SLOPE_21_MA_D
                slope_col_name = f'SLOPE_21_MA_{p}_D' # 使用21日斜率
                if slope_col_name not in df.columns:
                    slope_col_name = f'SLOPE_5_MA_{p}_D' # 回退到5日斜率
                # 【优化】使用多时间维度自适应归一化
                health_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, slope_col_name, pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_stability").clip(lower=0), df_index, ascending=True, tf_weights=tf_weights_struct).clip(0, 1)
                health_scores.append(health_score)
            long_term_trend_health_score = pd.Series(np.mean(health_scores, axis=0), index=df_index)
            # 融合 vpoc_consensus_strength_D 到 foundation_health_score
            foundation_health_score = (foundation_support_score * 0.5 + long_term_trend_health_score * 0.3 + vpoc_consensus_score * 0.2).clip(0, 1)
        # 将 raw_stability_score 的计算从乘法改为加权平均
        raw_stability_score = (energy_accumulation_score * 0.3 + foundation_health_score * 0.7).fillna(0.5) # 赋予基石支撑更高权重
        # 获取并归一化 volatility_skew_index_D
        volatility_skew_raw = self._get_safe_series(df, 'volatility_asymmetry_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_stability") # 替换为 volatility_asymmetry_index_D
        volatility_skew_score = get_adaptive_mtf_normalized_bipolar_score(volatility_skew_raw, df_index, tf_weights_struct, sensitivity=0.5).fillna(0.0) # 归一化到 [-1, 1]
        # 融合 volatility_skew_score 到最终的 stability_score
        stability_score = (raw_stability_score * 0.8 + volatility_skew_score.clip(lower=0) * 0.2 - volatility_skew_score.clip(upper=0).abs() * 0.2).clip(-1, 1)
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
                print(f"       - volatility_skew_score: {volatility_skew_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - stability_score: {stability_score.loc[probe_date_for_loop]:.4f}")
        return stability_score



