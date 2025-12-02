import pandas as pd
import numpy as np
import pandas_ta as ta
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
        【V5.4 · 情境感知版】结构情报分析总指挥
        - 核心升级: 引入“战场环境”公理，并基于此生成最终的“情境战略态势”，实现“天人合一”的决策。
        - 核心新增: 引入“结构张力”公理，作为势能压缩的先行指标。
        - 核心新增: 引入“二次启动”战术剧本识别，从公理组合中识别高阶意图。
        - 核心职责: 输出所有结构层信号，包括原子公理、内部态势、情境态势、动量和剧本。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("结构情报引擎已在配置中禁用，跳过。")
            return {}
        current_date_str = df.index[-1].strftime('%Y-%m-%d')
        self.is_probe_date = current_date_str in self.probe_dates
        if self.is_probe_date:
            print(f"\n--- [结构情报探针] @ {current_date_str} ---")
        # --- 步骤一: 诊断原子公理 ---
        axiom_trend_form = self._diagnose_axiom_trend_form(df)
        axiom_mtf_cohesion = self._diagnose_axiom_mtf_cohesion(df, axiom_trend_form)
        axiom_stability = self._diagnose_axiom_stability(df)
        axiom_divergence = self._diagnose_axiom_divergence(df)
        bottom_fractal_score = self._diagnose_bottom_fractal(df, n=5, min_depth_ratio=0.001)
        axiom_tension = self._diagnose_axiom_tension(df)
        # --- 新增代码开始 ---
        axiom_environment = self._diagnose_axiom_environment(df) # 新增：诊断战场环境
        all_states['SCORE_STRUCT_AXIOM_ENVIRONMENT'] = axiom_environment
        # --- 新增代码结束 ---
        all_states['SCORE_STRUCT_AXIOM_TENSION'] = axiom_tension
        all_states['SCORE_STRUCT_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_STRUCT_AXIOM_TREND_FORM'] = axiom_trend_form
        all_states['SCORE_STRUCT_AXIOM_MTF_COHESION'] = axiom_mtf_cohesion
        all_states['SCORE_STRUCT_AXIOM_STABILITY'] = axiom_stability
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_STRUCTURE_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_STRUCTURE_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        all_states['SCORE_STRUCT_BOTTOM_FRACTAL'] = bottom_fractal_score
        # --- 步骤二: 诊断内部战略态势 ---
        strategic_posture = self._diagnose_strategic_posture(
            axiom_trend_form, axiom_mtf_cohesion, axiom_stability, axiom_tension
        )
        all_states['SCORE_STRUCT_STRATEGIC_POSTURE'] = strategic_posture
        # --- 新增代码开始 ---
        # --- 步骤三: 融合战场环境，生成最终情境态势 ---
        env_factor = 0.5 # 环境对内部态势的影响系数
        env_modifier = (axiom_environment - 0.5) * env_factor
        contextual_posture = (strategic_posture * (1 + env_modifier)).clip(0, 1)
        all_states['SCORE_STRUCT_CONTEXTUAL_POSTURE'] = contextual_posture.astype(np.float32)
        if self.is_probe_date:
            print(f"    [探针] 情境战略态势 (SCORE_STRUCT_CONTEXTUAL_POSTURE): {contextual_posture.iloc[-1]:.4f}")
            print(f"      - 融合: 内部态势分={strategic_posture.iloc[-1]:.2f}, 环境调节器={env_modifier.iloc[-1]:.2f} -> 最终态势 = 内部态势 * (1 + 调节器)")
        # --- 新增代码结束 ---
        # --- 步骤四: 基于情境态势，诊断动量与剧本 ---
        # --- 修改代码开始 ---
        momentum_window = 5
        posture_slope_raw = ta.slope(contextual_posture, length=momentum_window) # 基于情境态势计算动量
        posture_slope_raw.fillna(0, inplace=True)
        mtf_weights_conf = get_param_value(p_conf.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        structural_momentum = get_adaptive_mtf_normalized_bipolar_score(posture_slope_raw, df.index, tf_weights)
        all_states['SCORE_STRUCT_MOMENTUM'] = structural_momentum.astype(np.float32)
        if self.is_probe_date:
            today_score = structural_momentum.iloc[-1]
            print(f"    [探针] 结构动量(势) (SCORE_STRUCT_MOMENTUM): {today_score:.4f}")
            print(f"      - 原料: 情境战略态势5日斜率(原始)={posture_slope_raw.iloc[-1]:.4f}")
        playbook_secondary_launch = self._diagnose_playbook_secondary_launch(
            df, axiom_stability, contextual_posture, structural_momentum # 使用情境态势进行剧本判断
        )
        all_states['SCORE_STRUCT_PLAYBOOK_SECONDARY_LAUNCH'] = playbook_secondary_launch
        # --- 修改代码结束 ---
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.4 · MTF重构版】结构公理四：诊断“结构背离”
        - 核心逻辑: 诊断价格行为与均线结构（如均线排列）的背离。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】使用专属的MTF权重配置进行归一化。
        """
        required_signals = ['pct_change_D', 'EMA_5_D', 'EMA_55_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        # --- 修改代码开始 ---
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 修改代码结束 ---
        price_trend_raw = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_divergence")
        price_trend_score = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df.index, tf_weights)
        ema_short_long_diff = self._get_safe_series(df, 'EMA_5_D', 0.0, method_name="_diagnose_axiom_divergence") - self._get_safe_series(df, 'EMA_55_D', 0.0, method_name="_diagnose_axiom_divergence")
        ma_structure_trend_raw = ema_short_long_diff.diff(1)
        ma_structure_trend_score = get_adaptive_mtf_normalized_bipolar_score(ma_structure_trend_raw, df.index, tf_weights)
        divergence_score = (ma_structure_trend_score - price_trend_score).clip(-1, 1)
        final_score = divergence_score.astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 结构背离公理 (SCORE_STRUCT_AXIOM_DIVERGENCE): {today_score:.4f}")
            print(f"      - MTF权重: default")
            print(f"      - 原料: 价格趋势(原始)={price_trend_raw.iloc[-1]:.4f}, 均线结构趋势(原始)={ma_structure_trend_raw.iloc[-1]:.4f}")
            print(f"      - 计算: 价格趋势分={price_trend_score.iloc[-1]:.4f}, 均线结构趋势分={ma_structure_trend_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_axiom_trend_form(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.5 · MTF重构版】结构公理一：诊断“趋势形态”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 废弃了旧版对“能量”的评估，更纯粹地聚焦于趋势的“几何形态品质”。
        - 核心证据: 融合均线排列的“有序度”、均线簇的“角度”以及均线本身的“斜率”。
        - 核心修复: 修复了因直接访问 `self.strategy.slope_params` 导致的 `AttributeError`。改为通过 `get_params_block` 安全地获取斜率配置。
        - 核心优化: 引入逻辑仲裁机制。当均线排列分极高时，将修正与之矛盾的“有序度”分，解决底层信号冲突问题。
        - 【优化】使用专属的 `short_term_geometry` MTF权重进行归一化。
        """
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        ema_periods = get_param_value(p_conf_struct.get('trend_form_ema_periods'), [5, 13, 21, 55])
        required_signals = [
            'MA_POTENTIAL_ORDERLINESS_SCORE_D', 'ATAN_ANGLE_EMA_55_D',
            'close_D', 'pct_change_D'
        ]
        required_signals.extend([f'EMA_{p}_D' for p in ema_periods])
        feature_eng_params = get_params_block(self.strategy, 'feature_engineering_params', {})
        slope_params = get_params_block(feature_eng_params, 'slope_params', {})
        series_to_slope_config = slope_params.get('series_to_slope', {})
        slope_cols_to_use = []
        for p in ema_periods:
            slope_col = f'SLOPE_5_EMA_{p}_D'
            base_col = f'EMA_{p}_D'
            if base_col in series_to_slope_config and 5 in series_to_slope_config.get(base_col, []):
                required_signals.append(slope_col)
                slope_cols_to_use.append(slope_col)
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend_form"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        # --- 修改代码开始 ---
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('short_term_geometry', {5: 0.5, 8: 0.3, 13: 0.2})
        # --- 修改代码结束 ---
        # 维度1: 排列 (Alignment)
        bull_alignment_raw = pd.Series(0.0, index=df_index)
        alignment_weights = np.linspace(0.5, 0.2, len(ema_periods) - 1)
        for i in range(len(ema_periods) - 1):
            ema_i = self._get_safe_series(df, f'EMA_{ema_periods[i]}_D', method_name="_diagnose_axiom_trend_form")
            ema_i_plus_1 = self._get_safe_series(df, f'EMA_{ema_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")
            bull_alignment_raw += (ema_i > ema_i_plus_1).astype(float) * alignment_weights[i]
        alignment_score = bull_alignment_raw / sum(alignment_weights)
        # 维度2: 斜率 (Slope)
        slope_scores = [get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, col, 0.0, method_name="_diagnose_axiom_trend_form"), df_index, tf_weights).values for col in slope_cols_to_use]
        avg_slope_score = pd.Series(np.mean(slope_scores, axis=0) if slope_scores else 0.0, index=df_index)
        # 维度3: 有序度 (Orderliness)
        orderliness_raw = self._get_safe_series(df, 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 0.0, method_name="_diagnose_axiom_trend_form")
        orderliness_score = get_adaptive_mtf_normalized_score(orderliness_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 逻辑仲裁
        corrected_orderliness_score = orderliness_score.copy()
        arbitration_triggered = (alignment_score > 0.9) & (orderliness_score < alignment_score)
        corrected_orderliness_score[arbitration_triggered] = alignment_score[arbitration_triggered]
        # 维度4: 角度 (Angle)
        angle_raw = self._get_safe_series(df, 'ATAN_ANGLE_EMA_55_D', 0.0, method_name="_diagnose_axiom_trend_form")
        angle_score = get_adaptive_mtf_normalized_bipolar_score(angle_raw, df_index, tf_weights)
        # --- 融合形态分 ---
        bullish_form_score = (
            alignment_score * 0.3 +
            avg_slope_score.clip(lower=0) * 0.3 +
            corrected_orderliness_score * 0.2 +
            angle_score.clip(lower=0) * 0.2
        ).clip(0, 1)
        bearish_form_score = (
            (1 - alignment_score) * 0.3 +
            avg_slope_score.clip(upper=0).abs() * 0.3 +
            (1 - corrected_orderliness_score) * 0.2 +
            angle_score.clip(upper=0).abs() * 0.2
        ).clip(0, 1)
        # --- 最终裁决 ---
        trend_direction = np.sign(self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_trend_form"))
        trend_form_score = np.where(trend_direction >= 0, bullish_form_score, -bearish_form_score)
        final_score = pd.Series(trend_form_score, index=df_index).clip(-1, 1).astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 趋势形态公理 (SCORE_STRUCT_AXIOM_TREND_FORM): {today_score:.4f}")
            print(f"      - MTF权重: short_term_geometry")
            print(f"      - 原料: 排列分={alignment_score.iloc[-1]:.2f}, 平均斜率分={avg_slope_score.iloc[-1]:.2f}, 有序度(原始)={orderliness_raw.iloc[-1]:.2f}, 角度(原始)={angle_raw.iloc[-1]:.2f}")
            print(f"      - 计算: 有序度分(原始)={orderliness_score.iloc[-1]:.2f}, 角度分={angle_score.iloc[-1]:.2f}")
            if arbitration_triggered.iloc[-1]:
                print(f"      - 仲裁: [触发] 排列分(0.9+) > 有序度分, 已修正有序度分 => {corrected_orderliness_score.iloc[-1]:.2f}")
            print(f"      - 融合: 看涨形态分={bullish_form_score.iloc[-1]:.2f}, 看跌形态分={bearish_form_score.iloc[-1]:.2f}, 趋势方向={trend_direction.iloc[-1]:.0f}")
        return final_score

    def _diagnose_axiom_stability(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.4 · 纯粹防御重构版】结构公理三：诊断“结构稳定性”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 移除了进攻性的“结构杠杆”指标，回归“纯粹防御”本质。现在完全由宏观支撑、结构韧性、微观流动性三大支柱构成。
        - 核心证据 (韧性): `support_validation_strength`作为结构在压力测试下的直接表现。
        - 【优化】使用专属的 `long_term_stability` MTF权重进行归一化。
        """
        # --- 修改代码开始 ---
        # 移除 structural_leverage_D
        required_signals = [
            'flow_credibility_index_D', 'main_force_slippage_index_D', 'support_validation_strength_D',
            'dominant_peak_solidity_D', 'close_D'
        ]
        # --- 修改代码结束 ---
        long_term_ma_periods = [55, 144]
        required_signals.extend([f'MA_{p}_D' for p in long_term_ma_periods])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_stability"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('long_term_stability', {13: 0.2, 21: 0.3, 55: 0.4, 89: 0.1})
        # --- 1. 宏观支撑 (Macro Support) ---
        foundation_support_scores = []
        for p in long_term_ma_periods:
            support_score = get_adaptive_mtf_normalized_score((self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_stability") - self._get_safe_series(df, f'MA_{p}_D', method_name="_diagnose_axiom_stability")).clip(lower=0), df_index, ascending=True, tf_weights=tf_weights)
            foundation_support_scores.append(support_score)
        foundation_support_score = pd.Series(np.mean(foundation_support_scores, axis=0), index=df_index)
        vpoc_consensus_raw = self._get_safe_series(df, 'dominant_peak_solidity_D', 0.0, method_name="_diagnose_axiom_stability")
        vpoc_consensus_score = get_adaptive_mtf_normalized_score(vpoc_consensus_raw, df_index, ascending=True, tf_weights=tf_weights)
        macro_support_score = (foundation_support_score * 0.6 + vpoc_consensus_score * 0.4).clip(0, 1)
        # --- 2. 结构韧性 (Structural Resilience) ---
        pullback_depth_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.5, method_name="_diagnose_axiom_stability")
        resilience_score = get_adaptive_mtf_normalized_score(pullback_depth_raw, df_index, ascending=True, tf_weights=tf_weights)
        # --- 3. 微观流动性 (Micro-Liquidity) ---
        liquidity_auth_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.5, method_name="_diagnose_axiom_stability")
        market_impact_raw = self._get_safe_series(df, 'main_force_slippage_index_D', 0.1, method_name="_diagnose_axiom_stability")
        liquidity_auth_score = get_adaptive_mtf_normalized_score(liquidity_auth_raw, df_index, ascending=True, tf_weights=tf_weights)
        market_impact_score = get_adaptive_mtf_normalized_score(market_impact_raw, df_index, ascending=False, tf_weights=tf_weights)
        micro_liquidity_score = (liquidity_auth_score * 0.6 + market_impact_score * 0.4).clip(0, 1)
        # --- 4. 融合 ---
        # --- 修改代码开始 ---
        # 移除杠杆分，重新分配权重: 宏观支撑(0.4), 韧性(0.4), 流动性(0.2)
        stability_score = (
            macro_support_score * 0.4 +
            resilience_score * 0.4 +
            micro_liquidity_score * 0.2
        ).clip(0, 1)
        # --- 修改代码结束 ---
        final_score = (stability_score * 2 - 1).astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 结构稳定性公理 (SCORE_STRUCT_AXIOM_STABILITY): {today_score:.4f}")
            print(f"      - MTF权重: long_term_stability")
            # --- 修改代码开始 ---
            print(f"      - 原料: 支撑强度(原始)={pullback_depth_raw.iloc[-1]:.2f}, 流动性(原始)={liquidity_auth_raw.iloc[-1]:.2f}")
            print(f"      - 计算: 宏观支撑分={macro_support_score.iloc[-1]:.2f}, 韧性分={resilience_score.iloc[-1]:.2f}, 流动性分={micro_liquidity_score.iloc[-1]:.2f}")
            # --- 修改代码结束 ---
        return final_score

    def _diagnose_axiom_mtf_cohesion(self, df: pd.DataFrame, daily_trend_form_score: pd.Series) -> pd.Series:
        """
        【V2.7 · 自适应通道风险版】结构公理二：诊断“宏观趋势健康度”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 使用布林带百分比(BBP)替代BIAS作为核心风险标尺，解决了BIAS静态、不自适应的根本缺陷。模型现在能更好地区分“健康的趋势”与“高风险的极端行情”。
        - 核心修复: 修复了逻辑上的不对称性。模型现在能同时识别上升趋势中的“过热”风险和下降趋势中的“超跌”状态（趋势衰竭信号），并对两者进行对称的降权处理。
        - 核心融合: 继续采用“和谐度”模型，将经过风险调整后的“宏观健康度分”与“微观意图分”进行加权融合。
        """
        short_periods = [5, 13, 21]
        long_periods = [55, 89, 144]
        required_signals = [
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'close_D', 'BBP_21_2.0_D' # 替换 BIAS_55_D
        ]
        required_signals.extend([f'EMA_{p}_D' for p in short_periods])
        required_signals.extend([f'EMA_{p}_D' for p in long_periods])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_mtf_cohesion"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 宏观趋势健康度 (Macro Trend Health) ---
        # 1a. 伪多周期排列分 (Pseudo-MTF Alignment)
        fastest_short_ma = self._get_safe_series(df, f'EMA_{min(short_periods)}_D', method_name="_diagnose_axiom_mtf_cohesion")
        slowest_long_ma = self._get_safe_series(df, f'EMA_{max(long_periods)}_D', method_name="_diagnose_axiom_mtf_cohesion")
        alignment_score = (fastest_short_ma > slowest_long_ma).astype(float)
        # --- 修改代码开始 ---
        # 1b. 自适应风险感知：基于布林带的 过热惩罚 与 超跌缓和
        bbp_raw = self._get_safe_series(df, 'BBP_21_2.0_D', 0.5, method_name="_diagnose_axiom_mtf_cohesion")
        # 过热惩罚：当价格进入布林带上轨的最后5%区间(BBP>0.95)时开始惩罚，突破上轨越多惩罚越大
        overheat_penalty = ((bbp_raw - 0.95).clip(lower=0) / 0.2).clip(upper=1.0) # 在BBP=1.15时惩罚达到最大
        # 超跌缓和：当价格进入布林带下轨的最初5%区间(BBP<0.05)时开始缓和，突破下轨越多缓和越大
        oversold_mitigation = (((0.05 - bbp_raw)).clip(lower=0) / 0.2).clip(upper=1.0) # 在BBP=-0.15时缓和达到最大
        # 1c. 风险调整后的宏观分
        bullish_macro_health = alignment_score * (1 - overheat_penalty)
        bearish_macro_health = (1 - alignment_score) * (1 - oversold_mitigation)
        # --- 修改代码结束 ---
        # --- 2. 微观意图 (Micro Intent) ---
        ofi_raw = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        buy_sweep_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        sell_sweep_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_mtf_cohesion")
        ofi_score = get_adaptive_mtf_normalized_bipolar_score(ofi_raw, df_index, tf_weights)
        buy_sweep_score = get_adaptive_mtf_normalized_score(buy_sweep_raw, df_index, ascending=True, tf_weights=tf_weights)
        sell_sweep_score = get_adaptive_mtf_normalized_score(sell_sweep_raw, df_index, ascending=True, tf_weights=tf_weights)
        bullish_intent = (ofi_score.clip(lower=0) * 0.5 + buy_sweep_score * 0.5)
        bearish_intent = (ofi_score.clip(upper=0).abs() * 0.5 + sell_sweep_score * 0.5)
        micro_intent_score = (bullish_intent - bearish_intent).clip(-1, 1)
        # --- 3. 和谐度融合 ---
        # 权重: 宏观(0.7), 微观(0.3)
        bullish_harmony = bullish_macro_health * 0.7 + micro_intent_score.clip(lower=0) * 0.3
        bearish_harmony = bearish_macro_health * 0.7 + micro_intent_score.clip(upper=0).abs() * 0.3
        final_score_raw = bullish_harmony - bearish_harmony
        final_score = final_score_raw.clip(-1, 1).astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 宏观趋势健康度公理 (SCORE_STRUCT_AXIOM_MTF_COHESION): {today_score:.4f}")
            print(f"      - MTF权重: default (for micro)")
            # --- 修改代码开始 ---
            print(f"      - 宏观原料: 排列分={alignment_score.iloc[-1]:.2f}, BBP_21(原始)={bbp_raw.iloc[-1]:.2f}")
            print(f"      - 宏观计算: 过热惩罚={overheat_penalty.iloc[-1]:.2f}, 超跌缓和={oversold_mitigation.iloc[-1]:.2f}")
            print(f"      - 宏观计算: 看涨健康度分={bullish_macro_health.iloc[-1]:.2f}, 看跌健康度分={bearish_macro_health.iloc[-1]:.2f}")
            # --- 修改代码结束 ---
            print(f"      - 微观原料: OFI(原始)={ofi_raw.iloc[-1]:.2f}, 买盘消耗(原始)={buy_sweep_raw.iloc[-1]:.2f}")
            print(f"      - 微观计算: 微观意图分={micro_intent_score.iloc[-1]:.2f}")
            print(f"      - 和谐度融合: (宏观健康度*0.7 + 微观意图*0.3)")
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

    def _diagnose_strategic_posture(self, axiom_trend_form: pd.Series, axiom_mtf_cohesion: pd.Series, axiom_stability: pd.Series, axiom_tension: pd.Series) -> pd.Series:
        """
        【V2.3 · 张力催化版】诊断顶层“战略态势”
        - 核心升级: 将“结构张力”作为“进攻催化剂”整合进“矛”的计算中，旨在放大从高势能状态发起的突破的战略价值。
        - 核心逻辑:
          - 矛 (进攻): (趋势形态 + 宏观健康度 + 结构杠杆) * (1 + 张力催化)
          - 盾 (防御): 直接由纯粹的结构稳定性公理决定。
        - 输出: 一个综合了进攻与防御的顶层战略分数。
        """
        required_signals = ['structural_leverage_D']
        if not self._validate_required_signals(self.strategy.df_indicators, required_signals, "_diagnose_strategic_posture"):
            return pd.Series(0.0, index=axiom_trend_form.index)
        df_index = axiom_trend_form.index
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('long_term_stability', {13: 0.2, 21: 0.3, 55: 0.4, 89: 0.1})
        leverage_raw = self._get_safe_series(self.strategy.df_indicators, 'structural_leverage_D', 0.0, method_name="_diagnose_strategic_posture")
        leverage_score = get_adaptive_mtf_normalized_score(leverage_raw, df_index, ascending=True, tf_weights=tf_weights)
        # --- 1. 矛 (Offense) ---
        # 1a. 基础进攻分
        base_offense_score = (
            axiom_trend_form.clip(lower=0) * 0.4 +
            axiom_mtf_cohesion.clip(lower=0) * 0.4 +
            leverage_score * 0.2
        ).clip(0, 1)
        # --- 修改代码开始 ---
        # 1b. 张力催化
        tension_catalyst_factor = 0.5 # 张力对进攻的催化系数
        tension_amplifier = 1 + (axiom_tension * tension_catalyst_factor)
        # 1c. 最终进攻分
        offense_score = (base_offense_score * tension_amplifier).clip(0, 1)
        # --- 修改代码结束 ---
        # --- 2. 盾 (Defense) ---
        defense_strength = ((axiom_stability + 1) / 2).clip(0, 1)
        # --- 3. 协同信念融合 ---
        conviction_factor = 0.5
        defense_modifier = (defense_strength - 0.5) * conviction_factor
        strategic_posture = (offense_score * (1 + defense_modifier)).clip(0, 1)
        final_score = strategic_posture.astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 结构战略态势 (SCORE_STRUCT_STRATEGIC_POSTURE): {today_score:.4f}")
            # --- 修改代码开始 ---
            print(f"      - 原料: 趋势形态分={axiom_trend_form.iloc[-1]:.2f}, 宏观健康度分={axiom_mtf_cohesion.iloc[-1]:.2f}, 结构稳定性分={axiom_stability.iloc[-1]:.2f}")
            print(f"      - 新增原料: 杠杆(原始)={leverage_raw.iloc[-1]:.2f} -> 杠杆分={leverage_score.iloc[-1]:.2f}")
            print(f"      - 新增原料: 结构张力分={axiom_tension.iloc[-1]:.2f}")
            print(f"      - 计算: 基础矛分={base_offense_score.iloc[-1]:.2f}, 张力放大器={tension_amplifier.iloc[-1]:.2f} -> 最终矛分={offense_score.iloc[-1]:.2f}")
            print(f"      - 计算: 盾(防御)强度={defense_strength.iloc[-1]:.2f}")
            print(f"      - 协同融合: 防御调节器={defense_modifier.iloc[-1]:.2f} -> 最终态势 = 最终矛分 * (1 + 调节器)")
            # --- 修改代码结束 ---
        return final_score

    def _diagnose_axiom_tension(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 势能压缩版】结构公理六：诊断“结构张力”
        - 核心逻辑: 量化系统内部能量的压缩程度，作为潜在状态突变的先行指标。
        - 核心维度: 融合价格空间压缩(BBW)、均线结构压缩(EMA标准差)和量能压缩(成交量均线比)。
        """
        ema_periods = [5, 13, 21, 34]
        required_signals = ['BBW_21_2.0_D', 'VOL_MA_5_D', 'VOL_MA_55_D']
        required_signals.extend([f'EMA_{p}_D' for p in ema_periods])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_tension"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 价格空间压缩 ---
        price_compression_raw = self._get_safe_series(df, 'BBW_21_2.0_D', 1.0, method_name="_diagnose_axiom_tension")
        price_compression_score = get_adaptive_mtf_normalized_score(price_compression_raw, df_index, ascending=False, tf_weights=tf_weights)
        # --- 2. 均线结构压缩 ---
        ema_cluster = df[[f'EMA_{p}_D' for p in ema_periods]]
        structure_compression_raw = ema_cluster.std(axis=1) / df['close_D'] # 标准差归一化，避免股价本身大小的影响
        structure_compression_score = get_adaptive_mtf_normalized_score(structure_compression_raw, df_index, ascending=False, tf_weights=tf_weights)
        # --- 3. 量能压缩 ---
        vol_ma_short = self._get_safe_series(df, 'VOL_MA_5_D', 1.0, method_name="_diagnose_axiom_tension")
        vol_ma_long = self._get_safe_series(df, 'VOL_MA_55_D', 1.0, method_name="_diagnose_axiom_tension")
        volume_compression_raw = vol_ma_short / vol_ma_long
        volume_compression_score = get_adaptive_mtf_normalized_score(volume_compression_raw, df_index, ascending=False, tf_weights=tf_weights)
        # --- 4. 融合 ---
        # 权重: 价格(0.4), 结构(0.4), 量能(0.2)
        tension_score = (
            price_compression_score * 0.4 +
            structure_compression_score * 0.4 +
            volume_compression_score * 0.2
        ).clip(0, 1)
        final_score = tension_score.astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 结构张力公理 (SCORE_STRUCT_AXIOM_TENSION): {today_score:.4f}")
            print(f"      - 价格压缩: BBW(原始)={price_compression_raw.iloc[-1]:.3f} -> 分数={price_compression_score.iloc[-1]:.2f}")
            print(f"      - 结构压缩: EMA标准差(原始)={structure_compression_raw.iloc[-1]:.4f} -> 分数={structure_compression_score.iloc[-1]:.2f}")
            print(f"      - 量能压缩: 均量比(原始)={volume_compression_raw.iloc[-1]:.3f} -> 分数={volume_compression_score.iloc[-1]:.2f}")
        return final_score

    def _diagnose_playbook_secondary_launch(self, df: pd.DataFrame, axiom_stability: pd.Series, strategic_posture: pd.Series, structural_momentum: pd.Series) -> pd.Series:
        """
        【V1.0 · 战术剧本识别】识别“暴力洗盘后二次启动”剧本
        - 核心逻辑: 在时间序列上匹配一个完整的战术行为模式。
        - 剧本序列: [前期稳定蓄势] -> [短暂暴力洗盘+主力吸筹] -> [当日强势启动]
        """
        required_signals = ['capitulation_absorption_index_D', 'close_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_playbook_secondary_launch"):
            return pd.Series(0.0, index=df.index)
        playbook_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        absorption_signal = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_playbook_secondary_launch")
        # 为了效率，我们只在最近的K天内寻找模式
        lookback_days = 60
        start_index = max(10, len(df) - lookback_days) # 至少需要10天历史数据
        for i in range(start_index, len(df)):
            # --- 条件3: 当日强势启动 ---
            is_launch_day = strategic_posture.iloc[i] > 0.6 and structural_momentum.iloc[i] > 0.6
            if not is_launch_day:
                continue
            # --- 回溯寻找洗盘和蓄势阶段 ---
            washout_found = False
            # 洗盘窗口: 启动日前1-3天
            for j in range(max(0, i - 3), i):
                # --- 条件2: 暴力洗盘 + 主力吸筹 ---
                price_dropped = df['close_D'].iloc[j] < df['close_D'].iloc[j-1]
                strong_absorption = absorption_signal.iloc[j] > 0.7
                if price_dropped and strong_absorption:
                    # --- 条件1: 前期稳定蓄势 ---
                    # 蓄势窗口: 洗盘日前5天
                    accumulation_period_end = j - 1
                    accumulation_period_start = max(0, accumulation_period_end - 5)
                    if accumulation_period_start < accumulation_period_end:
                        avg_stability = axiom_stability.iloc[accumulation_period_start:accumulation_period_end].mean()
                        if avg_stability > 0.2:
                            washout_found = True
                            break # 找到符合条件的洗盘日，即可停止内层循环
            if washout_found:
                playbook_score.iloc[i] = 1.0
        if self.is_probe_date and playbook_score.iloc[-1] > 0:
            print(f"    [探针] 结构剧本-二次启动 (SCORE_STRUCT_PLAYBOOK_SECONDARY_LAUNCH): {playbook_score.iloc[-1]:.4f}")
            print(f"      - 剧本确认: [蓄势]->[洗盘吸筹]->[启动] 模式在今日识别成功!")
        elif self.is_probe_date:
            print(f"    [探针] 结构剧本-二次启动 (SCORE_STRUCT_PLAYBOOK_SECONDARY_LAUNCH): 0.0000")
        return playbook_score

    def _diagnose_axiom_environment(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 审时度势版】结构公理七：诊断“战场环境”
        - 核心逻辑: 评估个股所处的外部宏观环境，融合板块强度与主题热度。
        """
        required_signals = ['industry_strength_rank_D', 'THEME_HOTNESS_SCORE_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_environment"):
            return pd.Series(0.5, index=df.index) # 环境未知时返回中性分
        df_index = df.index
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 地利 (Sector Strength) ---
        sector_rank_raw = self._get_safe_series(df, 'industry_strength_rank_D', 0.5, method_name="_diagnose_axiom_environment")
        # 排名越小越好，因此 ascending=False
        sector_strength_score = get_adaptive_mtf_normalized_score(sector_rank_raw, df_index, ascending=False, tf_weights=tf_weights)
        # --- 2. 人和 (Thematic Resonance) ---
        theme_hotness_raw = self._get_safe_series(df, 'THEME_HOTNESS_SCORE_D', 0.5, method_name="_diagnose_axiom_environment")
        theme_hotness_score = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=tf_weights)
        # --- 3. 融合 ---
        # 权重: 板块(0.6), 主题(0.4)
        environment_score = (
            sector_strength_score * 0.6 +
            theme_hotness_score * 0.4
        ).clip(0, 1)
        final_score = environment_score.astype(np.float32)
        if self.is_probe_date:
            today_score = final_score.iloc[-1]
            print(f"    [探针] 战场环境公理 (SCORE_STRUCT_AXIOM_ENVIRONMENT): {today_score:.4f}")
            print(f"      - 地利: 板块排名(原始)={sector_rank_raw.iloc[-1]:.2f} -> 分数={sector_strength_score.iloc[-1]:.2f}")
            print(f"      - 人和: 主题热度(原始)={theme_hotness_raw.iloc[-1]:.2f} -> 分数={theme_hotness_score.iloc[-1]:.2f}")
        return final_score













