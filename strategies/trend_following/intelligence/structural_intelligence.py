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

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [结构情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

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
        """
        【V2.0 · 趋势能量与形态版】结构公理一：诊断“趋势能量与形态”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 将静态的均线排列与动态的日内能量指标相结合，评估趋势的综合质量。
        - 核心证据 (能量): 引入`intraday_energy_density`和`intraday_thrust_purity`，量化趋势形态背后的“动力强度”。
        """
        required_signals = [
            'intraday_energy_density_D', 'intraday_thrust_purity_D', 'close_D', 'pct_change_D'
        ]
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        ema_periods = get_param_value(p_conf_struct.get('trend_form_ema_periods'), [5, 13, 21, 55])
        required_signals.extend([f'EMA_{p}_D' for p in ema_periods])
        required_signals.extend([f'SLOPE_5_EMA_{p}_D' for p in ema_periods])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend_form"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        tf_weights_struct = get_param_value(p_conf_struct.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 形态 (Form) ---
        bull_alignment_raw = pd.Series(0.0, index=df_index)
        alignment_weights = [0.4, 0.3, 0.3]
        for i in range(len(ema_periods) - 1):
            ema_i = self._get_safe_series(df, f'EMA_{ema_periods[i]}_D', method_name="_diagnose_axiom_trend_form")
            ema_i_plus_1 = self._get_safe_series(df, f'EMA_{ema_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")
            bull_alignment_raw += (ema_i > ema_i_plus_1).astype(float) * alignment_weights[i]
        bull_alignment_score = bull_alignment_raw / sum(alignment_weights)
        slope_scores = [get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, f'SLOPE_5_EMA_{p}_D', 0.0, method_name="_diagnose_axiom_trend_form"), df_index, tf_weights_struct).values for p in ema_periods]
        avg_slope_bipolar = pd.Series(np.mean(slope_scores, axis=0), index=df_index)
        form_score = (bull_alignment_score * 0.5 + avg_slope_bipolar.clip(lower=0) * 0.5).clip(0, 1)
        # --- 2. 能量 (Energy) ---
        energy_density_raw = self._get_safe_series(df, 'intraday_energy_density_D', 0.0, method_name="_diagnose_axiom_trend_form")
        thrust_purity_raw = self._get_safe_series(df, 'intraday_thrust_purity_D', 0.0, method_name="_diagnose_axiom_trend_form")
        energy_density_score = get_adaptive_mtf_normalized_score(energy_density_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        thrust_purity_score = get_adaptive_mtf_normalized_score(thrust_purity_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        energy_score = (energy_density_score * 0.6 + thrust_purity_score * 0.4).clip(0, 1)
        # --- 3. 融合 ---
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_trend_form")
        trend_direction = np.sign(pct_change)
        # 只有在上涨时，形态和能量才有意义；下跌时，形态和能量越强，说明下跌趋势越猛烈
        bullish_score = form_score * energy_score
        bearish_score = (1 - form_score) * energy_score # 形态越差，能量越大，熊市得分越高
        trend_form_score = np.where(trend_direction >= 0, bullish_score, -bearish_score)
        return pd.Series(trend_form_score, index=df_index).clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.0 · 结构韧性与微观流动性版】结构公理三：诊断“结构韧性与微观流动性”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 将稳定性的评估深入到微观层面，引入流动性指标来验证宏观结构的真实支撑力。
        - 核心证据 (微观): `liquidity_authenticity_score`和`market_impact_cost`评估稳定性的“微观基础”。
        - 核心证据 (韧性): `pullback_depth_ratio`作为结构在压力测试下的直接表现。
        """
        required_signals = [
            'liquidity_authenticity_score_D', 'market_impact_cost_D', 'pullback_depth_ratio_D',
            'vpoc_consensus_strength_D', 'close_D'
        ]
        long_term_ma_periods = [55, 144]
        required_signals.extend([f'MA_{p}_D' for p in long_term_ma_periods])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_stability"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        tf_weights_struct = get_params_block(self.strategy, 'structural_ultimate_params', {}).get('tf_fusion_weights', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 宏观支撑 (Macro Support) ---
        foundation_support_scores = []
        for p in long_term_ma_periods:
            support_score = get_adaptive_mtf_normalized_score((self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_stability") - self._get_safe_series(df, f'MA_{p}_D', method_name="_diagnose_axiom_stability")).clip(lower=0), df_index, ascending=True, tf_weights=tf_weights_struct)
            foundation_support_scores.append(support_score)
        foundation_support_score = pd.Series(np.mean(foundation_support_scores, axis=0), index=df_index)
        vpoc_consensus_raw = self._get_safe_series(df, 'vpoc_consensus_strength_D', 0.0, method_name="_diagnose_axiom_stability")
        vpoc_consensus_score = get_adaptive_mtf_normalized_score(vpoc_consensus_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        macro_support_score = (foundation_support_score * 0.6 + vpoc_consensus_score * 0.4).clip(0, 1)
        # --- 2. 结构韧性 (Structural Resilience) ---
        pullback_depth_raw = self._get_safe_series(df, 'pullback_depth_ratio_D', 0.5, method_name="_diagnose_axiom_stability")
        resilience_score = get_adaptive_mtf_normalized_score(pullback_depth_raw, df_index, ascending=False, tf_weights=tf_weights_struct) # 回撤越浅越好
        # --- 3. 微观流动性 (Micro-Liquidity) ---
        liquidity_auth_raw = self._get_safe_series(df, 'liquidity_authenticity_score_D', 0.5, method_name="_diagnose_axiom_stability")
        market_impact_raw = self._get_safe_series(df, 'market_impact_cost_D', 0.1, method_name="_diagnose_axiom_stability")
        liquidity_auth_score = get_adaptive_mtf_normalized_score(liquidity_auth_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        market_impact_score = get_adaptive_mtf_normalized_score(market_impact_raw, df_index, ascending=False, tf_weights=tf_weights_struct) # 冲击成本越低越好
        micro_liquidity_score = (liquidity_auth_score * 0.6 + market_impact_score * 0.4).clip(0, 1)
        # --- 4. 融合 ---
        stability_score = (
            macro_support_score * 0.4 +
            resilience_score * 0.3 +
            micro_liquidity_score * 0.3
        ).clip(0, 1)
        return (stability_score * 2 - 1).astype(np.float32) # 转换为双极性

    def _diagnose_axiom_mtf_cohesion(self, df: pd.DataFrame, norm_window: int, daily_trend_form_score: pd.Series) -> pd.Series:
        """
        【V2.0 · 多周期协同与微观意图版】结构公理二：诊断“多周期协同与微观意图”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 引入微观结构指标，对宏观的多周期共振信号进行“意图验真”。
        - 核心证据 (意图): `order_flow_imbalance_score`和`buy_sweep_intensity`被用作“测谎仪”，验证共振背后的真实攻击意图。
        """
        required_signals = [
            'order_flow_imbalance_score_D', 'buy_sweep_intensity_D', 'sell_sweep_intensity_D', 'close_D'
        ]
        ma_periods_w = [5, 13, 21, 55]
        required_signals.extend([f'EMA_{p}_W' for p in ma_periods_w])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_mtf_cohesion"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        tf_weights_struct = get_params_block(self.strategy, 'structural_ultimate_params', {}).get('tf_fusion_weights', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 宏观共振 (Macro Cohesion) ---
        bull_alignment_w_raw = pd.Series(0.0, index=df_index)
        for i in range(len(ma_periods_w) - 1):
            bull_alignment_w_raw += (self._get_safe_series(df, f'EMA_{ma_periods_w[i]}_W', method_name="_diagnose_axiom_mtf_cohesion") > self._get_safe_series(df, f'EMA_{ma_periods_w[i+1]}_W', method_name="_diagnose_axiom_mtf_cohesion")).astype(float)
        bull_alignment_w_score = bull_alignment_w_raw / (len(ma_periods_w) - 1)
        macro_cohesion_score = (daily_trend_form_score.clip(lower=0) * bull_alignment_w_score).clip(0, 1)
        # --- 2. 微观意图 (Micro Intent) ---
        ofi_raw = self._get_safe_series(df, 'order_flow_imbalance_score_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        buy_sweep_raw = self._get_safe_series(df, 'buy_sweep_intensity_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        sell_sweep_raw = self._get_safe_series(df, 'sell_sweep_intensity_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        ofi_score = get_adaptive_mtf_normalized_bipolar_score(ofi_raw, df_index, tf_weights_struct)
        buy_sweep_score = get_adaptive_mtf_normalized_score(buy_sweep_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        sell_sweep_score = get_adaptive_mtf_normalized_score(sell_sweep_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        bullish_intent = (ofi_score.clip(lower=0) * 0.5 + buy_sweep_score * 0.5)
        bearish_intent = (ofi_score.clip(upper=0).abs() * 0.5 + sell_sweep_score * 0.5)
        micro_intent_score = (bullish_intent - bearish_intent).clip(-1, 1)
        # --- 3. 融合 ---
        # 只有当微观意图为正时，宏观共振才被认为是有效的
        cohesion_score = macro_cohesion_score * micro_intent_score.clip(lower=0)
        # 对于负向协同，也需要微观意图的确认
        bearish_cohesion = ((1 - daily_trend_form_score.clip(lower=0)) * (1 - bull_alignment_w_score)).clip(0, 1)
        final_score = cohesion_score - (bearish_cohesion * micro_intent_score.clip(upper=0).abs())
        return final_score.clip(-1, 1).astype(np.float32)

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


