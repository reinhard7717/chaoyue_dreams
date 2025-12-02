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
        # --- 新增代码开始 ---
        # 初始化探针参数
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        self.probe_dates = debug_params.get('probe_dates', [])
        self.is_probe_date = False
        # --- 新增代码结束 ---

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
        【V5.0 · 战略态势整合版】结构情报分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 输出结构领域的原子公理信号、结构背离信号，并新增顶层“战略态势”超级信号。
        - 移除信号: SCORE_STRUCTURE_BULLISH_RESONANCE, SCORE_STRUCTURE_BEARISH_RESONANCE, BIPOLAR_STRUCTURAL_DOMAIN_HEALTH, SCORE_STRUCTURE_BOTTOM_REVERSAL, SCORE_STRUCTURE_TOP_REVERSAL。
        - 【新增】调用 `_diagnose_bottom_fractal` 方法，将底分型信号添加到 `all_states` 中。
        - 【新增】调用 `_diagnose_strategic_posture` 方法，生成顶层战略信号。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("结构情报引擎已在配置中禁用，跳过。")
            return {}
        # --- 新增代码开始 ---
        # 检查是否为探针日期，并设置标志
        current_date_str = df.index[-1].strftime('%Y-%m-%d')
        self.is_probe_date = current_date_str in self.probe_dates
        if self.is_probe_date:
            print(f"\n--- [结构情报探针] @ {current_date_str} ---")
        # --- 新增代码结束 ---
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
        # 诊断底分型结构
        bottom_fractal_score = self._diagnose_bottom_fractal(df, n=5, min_depth_ratio=0.001)
        all_states['SCORE_STRUCT_BOTTOM_FRACTAL'] = bottom_fractal_score
        # --- 新增代码开始 ---
        # --- 步骤二: 诊断顶层战略态势 ---
        strategic_posture = self._diagnose_strategic_posture(
            axiom_trend_form, axiom_mtf_cohesion, axiom_stability
        )
        all_states['SCORE_STRUCT_STRATEGIC_POSTURE'] = strategic_posture
        # --- 新增代码结束 ---
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.3 · 探针植入版】结构公理四：诊断“结构背离”
        - 核心逻辑: 诊断价格行为与均线结构（如均线排列）的背离。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `price_trend` 和 `ma_structure_trend` 的归一化方式改为多时间维度自适应归一化。
        """
        required_signals = ['pct_change_D', 'EMA_5_D', 'EMA_55_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        ma_periods = [5, 13, 21, 55]
        required_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in required_cols):
            print("诊断结构背离失败：缺少必要的EMA列。")
            return pd.Series(0.0, index=df.index)
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        tf_weights_struct = get_param_value(p_conf_struct.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}) # 借用筹码的MTF权重配置
        price_trend_raw = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_divergence")
        price_trend_score = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df.index, tf_weights_struct)
        ema_short_long_diff = self._get_safe_series(df, 'EMA_5_D', 0.0, method_name="_diagnose_axiom_divergence") - self._get_safe_series(df, 'EMA_55_D', 0.0, method_name="_diagnose_axiom_divergence")
        ma_structure_trend_raw = ema_short_long_diff.diff(1)
        ma_structure_trend_score = get_adaptive_mtf_normalized_bipolar_score(ma_structure_trend_raw, df.index, tf_weights_struct)
        divergence_score = (ma_structure_trend_score - price_trend_score).clip(-1, 1)
        final_score = divergence_score.astype(np.float32)
        # --- 新增代码开始 ---
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 结构背离公理 (SCORE_STRUCT_AXIOM_DIVERGENCE): {today_score:.4f}")
            print(f"      - 原料: 价格趋势(原始)={price_trend_raw.iloc[-1]:.4f}, 均线结构趋势(原始)={ma_structure_trend_raw.iloc[-1]:.4f}")
            print(f"      - 计算: 价格趋势分={price_trend_score.iloc[-1]:.4f}, 均线结构趋势分={ma_structure_trend_score.iloc[-1]:.4f}")
        # --- 新增代码结束 ---
        return final_score

    def _diagnose_axiom_trend_form(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.2 · 几何形态增强版】结构公理一：诊断“趋势形态”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 废弃了旧版对“能量”的评估，更纯粹地聚焦于趋势的“几何形态品质”。
        - 核心证据: 融合均线排列的“有序度”、均线簇的“角度”以及均线本身的“斜率”。
        - 核心修复: 修复了硬编码EMA周期的BUG，改为从配置文件读取。
        """
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        # --- 修改代码开始 ---
        # 从配置文件读取EMA周期，修复硬编码BUG
        ema_periods = get_param_value(p_conf_struct.get('trend_form_ema_periods'), [5, 13, 21, 55])
        required_signals = [
            'MA_POTENTIAL_ORDERLINESS_SCORE_D', 'ATAN_ANGLE_EMA_55_D', # 新增几何指标
            'close_D', 'pct_change_D'
        ]
        # --- 修改代码结束 ---
        required_signals.extend([f'EMA_{p}_D' for p in ema_periods])
        # 动态检查所需斜率是否存在
        for p in ema_periods:
            # 假设斜率周期为5，实际应根据配置动态生成，此处为简化
            slope_col = f'SLOPE_5_EMA_{p}_D'
            if slope_col in self.strategy.slope_params['series_to_slope']:
                 required_signals.append(slope_col)
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend_form"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        tf_weights_struct = get_param_value(p_conf_struct.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 形态四维诊断 (Form) ---
        # 维度1: 排列 (Alignment)
        bull_alignment_raw = pd.Series(0.0, index=df_index)
        alignment_weights = np.linspace(0.5, 0.2, len(ema_periods) - 1)
        for i in range(len(ema_periods) - 1):
            ema_i = self._get_safe_series(df, f'EMA_{ema_periods[i]}_D', method_name="_diagnose_axiom_trend_form")
            ema_i_plus_1 = self._get_safe_series(df, f'EMA_{ema_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")
            bull_alignment_raw += (ema_i > ema_i_plus_1).astype(float) * alignment_weights[i]
        alignment_score = bull_alignment_raw / sum(alignment_weights)
        # 维度2: 斜率 (Slope)
        slope_scores = [get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, f'SLOPE_5_EMA_{p}_D', 0.0, method_name="_diagnose_axiom_trend_form"), df_index, tf_weights_struct).values for p in ema_periods if f'SLOPE_5_EMA_{p}_D' in df.columns]
        avg_slope_score = pd.Series(np.mean(slope_scores, axis=0), index=df_index)
        # --- 修改代码开始 ---
        # 维度3: 有序度 (Orderliness)
        orderliness_raw = self._get_safe_series(df, 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 0.0, method_name="_diagnose_axiom_trend_form")
        orderliness_score = get_adaptive_mtf_normalized_score(orderliness_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        # 维度4: 角度 (Angle)
        angle_raw = self._get_safe_series(df, 'ATAN_ANGLE_EMA_55_D', 0.0, method_name="_diagnose_axiom_trend_form")
        angle_score = get_adaptive_mtf_normalized_bipolar_score(angle_raw, df_index, tf_weights_struct)
        # --- 融合形态分 ---
        # 权重: 排列(0.3), 斜率(0.3), 有序度(0.2), 角度(0.2)
        bullish_form_score = (
            alignment_score * 0.3 +
            avg_slope_score.clip(lower=0) * 0.3 +
            orderliness_score * 0.2 +
            angle_score.clip(lower=0) * 0.2
        ).clip(0, 1)
        bearish_form_score = (
            (1 - alignment_score) * 0.3 +
            avg_slope_score.clip(upper=0).abs() * 0.3 +
            (1 - orderliness_score) * 0.2 +
            angle_score.clip(upper=0).abs() * 0.2
        ).clip(0, 1)
        # --- 最终裁决 ---
        trend_direction = np.sign(self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_trend_form"))
        trend_form_score = np.where(trend_direction >= 0, bullish_form_score, -bearish_form_score)
        final_score = pd.Series(trend_form_score, index=df_index).clip(-1, 1).astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 趋势形态公理 (SCORE_STRUCT_AXIOM_TREND_FORM): {today_score:.4f}")
            print(f"      - 原料: 排列分={alignment_score.iloc[-1]:.2f}, 平均斜率分={avg_slope_score.iloc[-1]:.2f}, 有序度(原始)={orderliness_raw.iloc[-1]:.2f}, 角度(原始)={angle_raw.iloc[-1]:.2f}")
            print(f"      - 计算: 有序度分={orderliness_score.iloc[-1]:.2f}, 角度分={angle_score.iloc[-1]:.2f}")
            print(f"      - 融合: 看涨形态分={bullish_form_score.iloc[-1]:.2f}, 看跌形态分={bearish_form_score.iloc[-1]:.2f}, 趋势方向={trend_direction.iloc[-1]:.0f}")
        return final_score
        # --- 修改代码结束 ---

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.2 · 结构杠杆增强版】结构公理三：诊断“结构稳定性”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 在原有三大支柱基础上，引入“结构杠杆”作为第四维度，评估结构效率。
        - 核心证据 (微观): `flow_credibility_index`和`main_force_slippage_index`评估稳定性的“微观基础”。
        - 核心证据 (韧性): `support_validation_strength`作为结构在压力测试下的直接表现。
        - 核心证据 (杠杆): `structural_leverage_D`评估结构对能量的转化效率。
        """
        required_signals = [
            'flow_credibility_index_D', 'main_force_slippage_index_D', 'support_validation_strength_D',
            'dominant_peak_solidity_D', 'close_D',
            'structural_leverage_D' # --- 新增信号 ---
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
        vpoc_consensus_raw = self._get_safe_series(df, 'dominant_peak_solidity_D', 0.0, method_name="_diagnose_axiom_stability")
        vpoc_consensus_score = get_adaptive_mtf_normalized_score(vpoc_consensus_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        macro_support_score = (foundation_support_score * 0.6 + vpoc_consensus_score * 0.4).clip(0, 1)
        # --- 2. 结构韧性 (Structural Resilience) ---
        pullback_depth_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.5, method_name="_diagnose_axiom_stability")
        resilience_score = get_adaptive_mtf_normalized_score(pullback_depth_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        # --- 3. 微观流动性 (Micro-Liquidity) ---
        liquidity_auth_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.5, method_name="_diagnose_axiom_stability")
        market_impact_raw = self._get_safe_series(df, 'main_force_slippage_index_D', 0.1, method_name="_diagnose_axiom_stability")
        liquidity_auth_score = get_adaptive_mtf_normalized_score(liquidity_auth_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        market_impact_score = get_adaptive_mtf_normalized_score(market_impact_raw, df_index, ascending=False, tf_weights=tf_weights_struct)
        micro_liquidity_score = (liquidity_auth_score * 0.6 + market_impact_score * 0.4).clip(0, 1)
        # --- 新增代码开始 ---
        # --- 4. 结构杠杆 (Structural Leverage) ---
        leverage_raw = self._get_safe_series(df, 'structural_leverage_D', 0.0, method_name="_diagnose_axiom_stability")
        leverage_score = get_adaptive_mtf_normalized_score(leverage_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        # --- 融合 ---
        # 权重: 宏观支撑(0.3), 结构韧性(0.3), 微观流动性(0.2), 结构杠杆(0.2)
        stability_score = (
            macro_support_score * 0.3 +
            resilience_score * 0.3 +
            micro_liquidity_score * 0.2 +
            leverage_score * 0.2
        ).clip(0, 1)
        final_score = (stability_score * 2 - 1).astype(np.float32) # 转换为双极性
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 结构稳定性公理 (SCORE_STRUCT_AXIOM_STABILITY): {today_score:.4f}")
            print(f"      - 原料: 支撑强度(原始)={pullback_depth_raw.iloc[-1]:.2f}, 杠杆(原始)={leverage_raw.iloc[-1]:.2f}")
            print(f"      - 计算: 宏观支撑分={macro_support_score.iloc[-1]:.2f}, 韧性分={resilience_score.iloc[-1]:.2f}, 流动性分={micro_liquidity_score.iloc[-1]:.2f}, 杠杆分={leverage_score.iloc[-1]:.2f}")
        return final_score
        # --- 新增代码结束 ---

    def _diagnose_axiom_mtf_cohesion(self, df: pd.DataFrame, norm_window: int, daily_trend_form_score: pd.Series) -> pd.Series:
        """
        【V2.2 · 探针植入版】结构公理二：诊断“多周期协同与微观意图”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 引入微观结构指标，对宏观的多周期共振信号进行“意图验真”。
        - 核心证据 (意图): `order_book_imbalance`和`buy_quote_exhaustion_rate`被用作“测谎仪”，验证共振背后的真实攻击意图。
        """
        required_signals = [
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D', 'close_D'
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
        ofi_raw = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        buy_sweep_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        sell_sweep_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        ofi_score = get_adaptive_mtf_normalized_bipolar_score(ofi_raw, df_index, tf_weights_struct)
        buy_sweep_score = get_adaptive_mtf_normalized_score(buy_sweep_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        sell_sweep_score = get_adaptive_mtf_normalized_score(sell_sweep_raw, df_index, ascending=True, tf_weights=tf_weights_struct)
        bullish_intent = (ofi_score.clip(lower=0) * 0.5 + buy_sweep_score * 0.5)
        bearish_intent = (ofi_score.clip(upper=0).abs() * 0.5 + sell_sweep_score * 0.5)
        micro_intent_score = (bullish_intent - bearish_intent).clip(-1, 1)
        # --- 3. 融合 ---
        cohesion_score = macro_cohesion_score * micro_intent_score.clip(lower=0)
        bearish_cohesion = ((1 - daily_trend_form_score.clip(lower=0)) * (1 - bull_alignment_w_score)).clip(0, 1)
        final_score_raw = cohesion_score - (bearish_cohesion * micro_intent_score.clip(upper=0).abs())
        final_score = final_score_raw.clip(-1, 1).astype(np.float32)
        # --- 新增代码开始 ---
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 多周期协同公理 (SCORE_STRUCT_AXIOM_MTF_COHESION): {today_score:.4f}")
            print(f"      - 宏观原料: 日线趋势形态分={daily_trend_form_score.iloc[-1]:.2f}, 周线排列分={bull_alignment_w_score.iloc[-1]:.2f}")
            print(f"      - 微观原料: OFI(原始)={ofi_raw.iloc[-1]:.2f}, 买盘消耗(原始)={buy_sweep_raw.iloc[-1]:.2f}, 卖盘消耗(原始)={sell_sweep_raw.iloc[-1]:.2f}")
            print(f"      - 计算节点: 宏观共振分={macro_cohesion_score.iloc[-1]:.2f}, 微观意图分={micro_intent_score.iloc[-1]:.2f}")
        # --- 新增代码结束 ---
        return final_score

    def _diagnose_bottom_fractal(self, df: pd.DataFrame, n: int = 5, min_depth_ratio: float = 0.001) -> pd.Series:
        """
        【V1.2 · 探针植入版】结构公理五：诊断“底分型”结构
        - 核心逻辑: 识别底分型结构形态，并输出一个双极性分数 `SCORE_STRUCT_BOTTOM_FRACTAL`。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        required_signals = ['low_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_bottom_fractal"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        bottom_fractal_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        if n % 2 == 0 or n < 3:
            print(f"    -> [结构情报警告] 底分型识别错误: 参数 n 必须是大于等于3的奇数，当前为 {n}。将使用默认值5。")
            n = 5
        half_n = n // 2
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
                if min_depth_ratio > 0:
                    avg_surrounding_low = np.mean(surrounding_lows)
                    if avg_surrounding_low <= 0:
                        is_bottom = False
                    elif (avg_surrounding_low - middle_low) / avg_surrounding_low < min_depth_ratio:
                        is_bottom = False
            if is_bottom:
                bottom_fractal_score.iloc[i] = 1.0
        # --- 新增代码开始 ---
        if self.is_probe_date:
            today_score = bottom_fractal_score.iloc[-1]
            print(f"    [探针] 底分型公理 (SCORE_STRUCT_BOTTOM_FRACTAL): {today_score:.4f}")
            if today_score > 0:
                probe_index = len(df) - 1
                middle_low_probe = low_series.iloc[probe_index]
                surrounding_lows_probe = [low_series.iloc[j] for j in range(probe_index - half_n, probe_index + half_n + 1) if j != probe_index]
                print(f"      - 结构确认: 中心Low={middle_low_probe:.2f}, 周围Lows={surrounding_lows_probe}")
        # --- 新增代码结束 ---
        return bottom_fractal_score

    def _diagnose_strategic_posture(self, axiom_trend_form: pd.Series, axiom_mtf_cohesion: pd.Series, axiom_stability: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】诊断顶层“结构战略态势”超级信号
        - 核心逻辑: 基于“矛与盾”博弈模型，对结构层的原子公理进行非线性战略融合。
        - 矛 (进攻力量): 融合“趋势形态”与“多周期协同”，代表结构的进攻潜力。
        - 盾 (防御基础): 基于“结构稳定性”，代表结构的防御强度。
        - 融合公式: sign(矛) * sqrt(abs(矛 * 盾))
        """
        # --- 1. 计算“矛” (Spear) - 进攻力量 ---
        # 权重: 趋势形态(0.6), 多周期协同(0.4)
        spear_score = (axiom_trend_form * 0.6 + axiom_mtf_cohesion * 0.4).clip(-1, 1)
        # --- 2. 计算“盾” (Shield) - 防御强度 ---
        # 将双极性的稳定性分数 [-1, 1] 映射为单极性的防御强度 [0, 1]
        shield_strength = (axiom_stability + 1) / 2
        # --- 3. 非线性融合 ---
        final_score_raw = np.sign(spear_score) * np.sqrt(np.abs(spear_score) * shield_strength)
        final_score = final_score_raw.fillna(0).clip(-1, 1).astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 结构战略态势 (SCORE_STRUCT_STRATEGIC_POSTURE): {today_score:.4f}")
            print(f"      - 原料: 趋势形态分={axiom_trend_form.iloc[-1]:.2f}, 多周期协同分={axiom_mtf_cohesion.iloc[-1]:.2f}, 结构稳定性分={axiom_stability.iloc[-1]:.2f}")
            print(f"      - 计算: 矛(进攻)分={spear_score.iloc[-1]:.2f}, 盾(防御)强度={shield_strength.iloc[-1]:.2f}")
        return final_score

