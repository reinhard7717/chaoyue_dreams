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
        【V8.0 · 荣耀代价版】结构情报分析总指挥
        - 核心升级: 引入“荣耀的代价”协议。当“龙头潜力”激活时，其分值将作为“豁免系数”，
                      部分抵消环境惩罚，而非简单叠加奖励。最终得分同时体现逆势的荣耀与代价。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("结构情报引擎已在配置中禁用，跳过。")
            return {}
        # --- 移除代码开始 ---
        # current_date_str = df.index[-1].strftime('%Y-%m-%d')
        # self.is_probe_date = current_date_str in self.probe_dates
        # if self.is_probe_date:
        #     print(f"\n--- [结构情报探针] @ {current_date_str} ---")
        # --- 移除代码结束 ---
        # --- 步骤一: 诊断原子公理 ---
        axiom_trend_form = self._diagnose_axiom_trend_form(df)
        axiom_mtf_cohesion = self._diagnose_axiom_mtf_cohesion(df, axiom_trend_form)
        axiom_stability = self._diagnose_axiom_stability(df)
        axiom_divergence = self._diagnose_axiom_divergence(df)
        bottom_fractal_score = self._diagnose_bottom_fractal(df, n=5, min_depth_ratio=0.001)
        axiom_tension = self._diagnose_axiom_tension(df)
        axiom_environment = self._diagnose_axiom_environment(df)
        platform_quality, dynamic_high, dynamic_low, vpoc = self._diagnose_platform_foundation(df)
        breakout_readiness = self._diagnose_breakout_readiness(df, axiom_tension)
        all_states['SCORE_STRUCT_BREAKOUT_READINESS'] = breakout_readiness
        all_states['SCORE_STRUCT_PLATFORM_FOUNDATION'] = platform_quality
        all_states['STRUCT_PLATFORM_DYNAMIC_HIGH'] = dynamic_high
        all_states['STRUCT_PLATFORM_DYNAMIC_LOW'] = dynamic_low
        all_states['STRUCT_PLATFORM_VPOC'] = vpoc
        all_states['SCORE_STRUCT_AXIOM_ENVIRONMENT'] = axiom_environment
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
        strategic_posture, defense_strength = self._diagnose_strategic_posture(
            axiom_trend_form, axiom_mtf_cohesion, axiom_stability, axiom_tension, platform_quality, breakout_readiness
        )
        all_states['SCORE_STRUCT_STRATEGIC_POSTURE'] = strategic_posture
        # --- 步骤三: 生成原始环境调节器 ---
        env_factor = 0.5
        env_modifier = (axiom_environment - 0.5) * env_factor
        # --- 步骤四: 基于“基础”情境态势，诊断原始动量 ---
        # 暂时使用原始调节器计算基础态势和动量，供龙头潜力判断
        contextual_posture_base_for_momentum = (strategic_posture * (1 + env_modifier)).clip(0, 1)
        momentum_window = 5
        posture_slope_raw = ta.slope(contextual_posture_base_for_momentum, length=momentum_window)
        posture_slope_raw.fillna(0, inplace=True)
        mtf_weights_conf = get_param_value(p_conf.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        structural_momentum = get_adaptive_mtf_normalized_bipolar_score(posture_slope_raw, df.index, tf_weights)
        # --- 步骤五: 龙头潜力裁决 ---
        leadership_potential = self._diagnose_leadership_potential(
            strategic_posture, axiom_environment, structural_momentum, axiom_tension
        )
        all_states['SCORE_STRUCT_LEADERSHIP_POTENTIAL'] = leadership_potential
        # --- 修改代码开始 ---
        # --- 步骤六: “荣耀的代价” -> 计算最终情境态势 ---
        waiver_coefficient = leadership_potential
        effective_env_modifier = env_modifier * (1 - waiver_coefficient)
        contextual_posture = (strategic_posture * (1 + effective_env_modifier)).clip(0, 1)
        all_states['SCORE_STRUCT_CONTEXTUAL_POSTURE'] = contextual_posture.astype(np.float32)
        # --- 步骤七: 基于最终情境态势，更新动量 ---
        final_posture_slope_raw = ta.slope(contextual_posture, length=momentum_window)
        final_posture_slope_raw.fillna(0, inplace=True)
        final_structural_momentum = get_adaptive_mtf_normalized_bipolar_score(final_posture_slope_raw, df.index, tf_weights)
        all_states['SCORE_STRUCT_MOMENTUM'] = final_structural_momentum.astype(np.float32)
        # --- 步骤八: 诊断剧本 ---
        playbook_secondary_launch = self._diagnose_playbook_secondary_launch(
            df, axiom_stability, contextual_posture, final_structural_momentum
        )
        all_states['SCORE_STRUCT_PLAYBOOK_SECONDARY_LAUNCH'] = playbook_secondary_launch
        # --- 步骤九: 终极裁决 ---
        final_judgment = self._diagnose_final_judgment(
            contextual_posture, defense_strength, final_structural_momentum
        )
        all_states['SCORE_STRUCT_FINAL_JUDGMENT'] = final_judgment
        return all_states
        # --- 修改代码结束 ---

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
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_trend_raw = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_divergence")
        price_trend_score = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df.index, tf_weights)
        ema_short_long_diff = self._get_safe_series(df, 'EMA_5_D', 0.0, method_name="_diagnose_axiom_divergence") - self._get_safe_series(df, 'EMA_55_D', 0.0, method_name="_diagnose_axiom_divergence")
        ma_structure_trend_raw = ema_short_long_diff.diff(1)
        ma_structure_trend_score = get_adaptive_mtf_normalized_bipolar_score(ma_structure_trend_raw, df.index, tf_weights)
        divergence_score = (ma_structure_trend_score - price_trend_score).clip(-1, 1)
        final_score = divergence_score.astype(np.float32)
        return final_score

    def _diagnose_axiom_trend_form(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.7 · 几何形态深度进化版】结构公理一：诊断“趋势形态”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 废弃了旧版对“能量”的评估，更纯粹地聚焦于趋势的“几何形态品质”。
        - 核心证据: 融合均线排列的“有序度”、“均线簇的角度”、“均线本身的“斜率”、“加速度”和“共振”。
        - 核心修复: 修复了因直接访问 `self.strategy.slope_params` 导致的 `AttributeError`。改为通过 `get_params_block` 安全地获取斜率配置。
        - 核心优化: 引入逻辑仲裁机制。当均线排列分极高时，将修正与之矛盾的“有序度”分，解决底层信号冲突问题。
        - 【优化】使用专属的 `short_term_geometry` MTF权重进行归一化。
        - 【V3.0 核心升级】移除了对当日价格涨跌幅（pct_change_D）的直接依赖来判断最终方向。现在，最终的双极性分数直接由看涨形态分减去看跌形态分得出, 更纯粹地反映了趋势形态本身的内在方向和强度，避免了当日价格波动对趋势形态判断的干扰。
        - 【V3.0 探针植入】增加了详细的探针输出，以便于检查和调试。
        - 【V3.0 参数化】融合权重现在从配置文件中读取。
        - 【V3.1 探针增强】增强了斜率和有序度维度的探针输出，以诊断其归一化结果。
        - 【V3.2 核心升级】引入多级别时间维度斜率和加速度，并将其纳入融合计算。
        - 【V3.3 核心升级】引入“共振”维度，评估多级别斜率和加速度的方向一致性与强度稳定性。
        - 【V3.5 核心修正】严格限定为纯粹的趋势几何形态分析，移除“结构健康度”和“市场博弈”维度，将这些更高层级的融合交由上层逻辑处理。
        - 【V3.6 核心进化】
            - 引入**动态权重调整**：根据市场波动率动态调整各维度融合权重。
            - 引入**均线粘合度**维度：量化均线簇的紧密程度，作为趋势形态健康的新指标。
            - **波动率调整的斜率/角度敏感度**：使斜率、加速度和角度指标在高波动率时更稳健，低波动率时更灵敏。
        - 【V3.7 核心进化】
            - 引入**价格均线乖离 (Price-MA Gap)** 维度：量化价格相对于均线的延伸度。
            - 引入**布林带动态 (Bollinger Band Dynamics)** 维度：评估布林带宽度和价格在带内的位置。
            - 引入**趋势效率 (Trend Efficiency)** 维度：衡量趋势运动的平滑度和直接性。
        """
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        ema_periods = get_param_value(p_conf_struct.get('trend_form_ema_periods'), [5, 13, 21, 34, 55])
        ma_periods = get_param_value(p_conf_struct.get('trend_form_ma_periods'), [5, 13, 21, 34, 55])
        # 修改开始：定义一个完整的默认融合权重字典，新增价格均线乖离、布林带动态、趋势效率维度
        default_fusion_weights = {
            'alignment': 0.15,
            'slope': 0.1,
            'acceleration': 0.1,
            'orderliness': 0.1,
            'angle': 0.1,
            'resonance': 0.1,
            'ma_cluster_cohesion': 0.1,
            'price_ma_gap': 0.1,
            'bb_dynamics': 0.1,
            'trend_efficiency': 0.05
        }
        # 从配置中获取融合权重，如果配置中没有，则使用空字典
        configured_fusion_weights = get_param_value(p_conf_struct.get('trend_form_fusion_weights'), {})
        # 将默认权重与配置权重合并，确保所有键都存在，并优先使用配置中的值
        fusion_weights = default_fusion_weights.copy()
        fusion_weights.update(configured_fusion_weights)
        # 过滤非数值类型的融合权重
        fusion_weights = {k: v for k, v in fusion_weights.items() if isinstance(v, (int, float))}
        # 修改结束

        # 获取共振参数
        resonance_params = get_param_value(p_conf_struct.get('trend_form_resonance_params'), {
            'enabled': True,
            'slope_consistency_weight': 0.4,
            'accel_consistency_weight': 0.4,
            'slope_accel_alignment_weight': 0.2
        })
        # 修改开始：获取动态权重、均线粘合度、斜率/角度波动率调整、价格均线乖离、布林带动态、趋势效率参数
        dynamic_weights_params = get_param_value(p_conf_struct.get('trend_form_dynamic_weights_params'), {})
        ma_cluster_cohesion_params = get_param_value(p_conf_struct.get('ma_cluster_cohesion_params'), {})
        slope_angle_volatility_adjustment_params = get_param_value(p_conf_struct.get('slope_angle_volatility_adjustment_params'), {})
        price_ma_gap_params = get_param_value(p_conf_struct.get('price_ma_gap_params'), {})
        bb_dynamics_params = get_param_value(p_conf_struct.get('bb_dynamics_params'), {})
        trend_efficiency_params = get_param_value(p_conf_struct.get('trend_efficiency_params'), {})
        # 修改结束

        # 修改开始：更新required_signals，新增ATR、波动率指标、布林带指标、价格均线乖离指标、趋势效率指标
        required_signals = [
            'MA_POTENTIAL_ORDERLINESS_SCORE_D', 'ATAN_ANGLE_EMA_55_D', 'close_D',
            'ATR_14_D', # 用于均线粘合度和斜率/角度波动率调整
            'VOLATILITY_INSTABILITY_INDEX_21d_D', # 用于动态权重
            'BBW_21_2.0_D', 'BBP_21_2.0_D', # 布林带动态
            'trend_efficiency_ratio_D' # 趋势效率
        ]
        required_signals.extend([f'EMA_{p}_D' for p in ema_periods])
        required_signals.extend([f'MA_{p}_D' for p in ma_periods])
        # 价格均线乖离指标
        price_ma_gap_periods = get_param_value(price_ma_gap_params.get('ma_periods'), [5, 13, 21])
        required_signals.extend([f'price_vs_ma_{p}_ratio_D' for p in price_ma_gap_periods])
        # 修改结束

        feature_eng_params = get_params_block(self.strategy, 'feature_engineering_params', {})
        slope_params = feature_eng_params.get('slope_params', {})
        accel_params = feature_eng_params.get('accel_params', {})
        series_to_slope_config = slope_params.get('series_to_slope', {})
        series_to_accel_config = accel_params.get('series_to_accel', {})
        all_slope_cols = []
        all_accel_cols = []

        # 收集EMA的斜率和加速度
        for p in ema_periods:
            base_ema_col = f'EMA_{p}_D'
            configured_slope_periods = series_to_slope_config.get(base_ema_col, [])
            for sp in configured_slope_periods:
                slope_col = f'SLOPE_{sp}_{base_ema_col}'
                required_signals.append(slope_col)
                all_slope_cols.append(slope_col)
            configured_accel_periods = series_to_accel_config.get(base_ema_col, [])
            for ap in configured_accel_periods:
                accel_col = f'ACCEL_{ap}_{base_ema_col}'
                required_signals.append(accel_col)
                all_accel_cols.append(accel_col)
        # 收集MA的斜率和加速度
        for p in ma_periods:
            base_ma_col = f'MA_{p}_D'
            configured_slope_periods = series_to_slope_config.get(base_ma_col, [])
            for sp in configured_slope_periods:
                slope_col = f'SLOPE_{sp}_{base_ma_col}'
                required_signals.append(slope_col)
                all_slope_cols.append(slope_col)
            configured_accel_periods = series_to_accel_config.get(base_ma_col, [])
            for ap in configured_accel_periods:
                accel_col = f'ACCEL_{ap}_{base_ma_col}'
                required_signals.append(accel_col)
                all_accel_cols.append(accel_col)

        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend_form"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('short_term_geometry', {5: 0.5, 8: 0.3, 13: 0.2})
        debug_config = get_params_block(self.strategy, 'debug_params', {})
        current_processing_date_str = df.index[-1].strftime('%Y-%m-%d') if not df.empty else ""

        is_probe_date = debug_config.get('should_probe', False) and \
                        current_processing_date_str in debug_config.get('probe_dates', [])
        if is_probe_date:
            print(f"\n--- [结构公理] 趋势形态探针 @ {current_processing_date_str} ---")
            print(f"  -> 原始输入:")
            print(f"    MA_POTENTIAL_ORDERLINESS_SCORE_D: {df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].iloc[-1]:.4f}")
            print(f"    ATAN_ANGLE_EMA_55_D: {df['ATAN_ANGLE_EMA_55_D'].iloc[-1]:.4f}")
            for p in ema_periods:
                print(f"    EMA_{p}_D: {df[f'EMA_{p}_D'].iloc[-1]:.4f}")
            for p in ma_periods:
                print(f"    MA_{p}_D: {df[f'MA_{p}_D'].iloc[-1]:.4f}")
            # 修改开始：探针输出新增的原始数据
            print(f"    ATR_14_D: {df['ATR_14_D'].iloc[-1]:.4f}")
            print(f"    VOLATILITY_INSTABILITY_INDEX_21d_D: {df['VOLATILITY_INSTABILITY_INDEX_21d_D'].iloc[-1]:.4f}")
            print(f"    BBW_21_2.0_D: {df['BBW_21_2.0_D'].iloc[-1]:.4f}")
            print(f"    BBP_21_2.0_D: {df['BBP_21_2.0_D'].iloc[-1]:.4f}")
            for p in price_ma_gap_periods:
                print(f"    price_vs_ma_{p}_ratio_D: {df[f'price_vs_ma_{p}_ratio_D'].iloc[-1]:.4f}")
            print(f"    trend_efficiency_ratio_D: {df['trend_efficiency_ratio_D'].iloc[-1]:.4f}")
            # 修改结束
            print(f"  -> MTF归一化权重 (tf_weights): {tf_weights}")
            print(f"  -> 融合权重 (fusion_weights): {fusion_weights}")

        # 动态权重计算
        if dynamic_weights_params.get('enabled', False):
            volatility_source_col = dynamic_weights_params.get('volatility_source', 'VOLATILITY_INSTABILITY_INDEX_21d_D')
            volatility_series = self._get_safe_series(df, volatility_source_col, 0.0, method_name="_diagnose_axiom_trend_form")
            volatility_sensitivity = dynamic_weights_params.get('volatility_sensitivity', 2.0)
            volatility_threshold = dynamic_weights_params.get('volatility_threshold', 0.5)
            adjustment_factor_range = dynamic_weights_params.get('adjustment_factor_range', 0.5)

            norm_volatility = get_adaptive_mtf_normalized_score(volatility_series, df_index, tf_weights, ascending=True)
            current_volatility_adjustment_factor = (1 + adjustment_factor_range * (volatility_threshold - norm_volatility.iloc[-1]) * volatility_sensitivity)
            current_volatility_adjustment_factor = np.clip(current_volatility_adjustment_factor, 1 - adjustment_factor_range, 1 + adjustment_factor_range)

            for key in fusion_weights:
                fusion_weights[key] = fusion_weights[key] * current_volatility_adjustment_factor
            total_dynamic_weight = sum(fusion_weights.values())
            if total_dynamic_weight > 0:
                fusion_weights = {k: v / total_dynamic_weight for k, v in fusion_weights.items()}

            if is_probe_date:
                print(f"  -> 动态权重调整:")
                print(f"    {volatility_source_col} (末值): {volatility_series.iloc[-1]:.4f}")
                print(f"    norm_volatility (末值): {norm_volatility.iloc[-1]:.4f}")
                print(f"    current_volatility_adjustment_factor (末值): {current_volatility_adjustment_factor:.4f}")
                print(f"    调整后的融合权重 (fusion_weights): {fusion_weights}")

        # 维度1: 排列 (Alignment) - 融合EMA和MA
        bull_alignment_raw_ema = pd.Series(0.0, index=df_index)
        alignment_weights_internal = np.linspace(0.5, 0.2, len(ema_periods) - 1)
        for i in range(len(ema_periods) - 1):
            ema_i = self._get_safe_series(df, f'EMA_{ema_periods[i]}_D', method_name="_diagnose_axiom_trend_form")
            ema_i_plus_1 = self._get_safe_series(df, f'EMA_{ema_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")
            bull_alignment_raw_ema += (ema_i > ema_i_plus_1).astype(float) * alignment_weights_internal[i]
        alignment_score_ema = bull_alignment_raw_ema / sum(alignment_weights_internal)

        bull_alignment_raw_ma = pd.Series(0.0, index=df_index)
        alignment_weights_internal_ma = np.linspace(0.5, 0.2, len(ma_periods) - 1)
        for i in range(len(ma_periods) - 1):
            ma_i = self._get_safe_series(df, f'MA_{ma_periods[i]}_D', method_name="_diagnose_axiom_trend_form")
            ma_i_plus_1 = self._get_safe_series(df, f'MA_{ma_periods[i+1]}_D', method_name="_diagnose_axiom_trend_form")
            bull_alignment_raw_ma += (ma_i > ma_i_plus_1).astype(float) * alignment_weights_internal_ma[i]
        alignment_score_ma = bull_alignment_raw_ma / sum(alignment_weights_internal_ma)

        alignment_fusion_weights = get_param_value(p_conf_struct.get('trend_form_alignment_fusion_weights'), {'ema': 0.6, 'ma': 0.4})
        alignment_score = (alignment_score_ema * alignment_fusion_weights.get('ema', 0.5) +
                           alignment_score_ma * alignment_fusion_weights.get('ma', 0.5))

        if is_probe_date:
            print(f"  -> 维度1: 排列 (Alignment)")
            print(f"    alignment_score_ema (末值): {alignment_score_ema.iloc[-1]:.4f}")
            print(f"    alignment_score_ma (末值): {alignment_score_ma.iloc[-1]:.4f}")
            print(f"    alignment_score (融合末值): {alignment_score.iloc[-1]:.4f}")

        # 波动率调整斜率和角度的敏感度
        adjusted_sensitivity_series = pd.Series(1.0, index=df_index)
        if slope_angle_volatility_adjustment_params.get('enabled', False):
            volatility_source_col = slope_angle_volatility_adjustment_params.get('volatility_source', 'ATR_14_D')
            volatility_series = self._get_safe_series(df, volatility_source_col, 0.0, method_name="_diagnose_axiom_trend_form")
            adjustment_strength = slope_angle_volatility_adjustment_params.get('adjustment_strength', 0.5)

            norm_volatility = get_adaptive_mtf_normalized_score(volatility_series, df_index, tf_weights, ascending=True)
            sensitivity_adjustment_factor = 1 - adjustment_strength * (norm_volatility - 0.5) * 2
            adjusted_sensitivity_series = sensitivity_adjustment_factor.clip(0.1, 2.0)

            if is_probe_date:
                print(f"  -> 斜率/角度波动率调整:")
                print(f"    {volatility_source_col} (末值): {volatility_series.iloc[-1]:.4f}")
                print(f"    norm_volatility (末值): {norm_volatility.iloc[-1]:.4f}")
                print(f"    adjusted_sensitivity_series (末值): {adjusted_sensitivity_series.iloc[-1]:.4f}")

        # 维度2: 斜率 (Slope) - 使用调整后的敏感度
        individual_slope_scores_list = []
        for col in all_slope_cols:
            raw_slope_series = self._get_safe_series(df, col, 0.0, method_name="_diagnose_axiom_trend_form")
            normalized_slope_score = get_adaptive_mtf_normalized_bipolar_score(raw_slope_series, df_index, tf_weights, sensitivity=adjusted_sensitivity_series)
            individual_slope_scores_list.append(normalized_slope_score)
        avg_slope_score = pd.Series(np.mean([s.values for s in individual_slope_scores_list], axis=0) if individual_slope_scores_list else 0.0, index=df_index)
        if is_probe_date:
            print(f"  -> 维度2: 斜率 (Slope)")
            print(f"    avg_slope_score (末值): {avg_slope_score.iloc[-1]:.4f}")

        # 维度3: 加速度 (Acceleration) - 同样使用调整后的敏感度
        individual_accel_scores_list = []
        for col in all_accel_cols:
            raw_accel_series = self._get_safe_series(df, col, 0.0, method_name="_diagnose_axiom_trend_form")
            normalized_accel_score = get_adaptive_mtf_normalized_bipolar_score(raw_accel_series, df_index, tf_weights, sensitivity=adjusted_sensitivity_series)
            individual_accel_scores_list.append(normalized_accel_score)
        avg_accel_score = pd.Series(np.mean([s.values for s in individual_accel_scores_list], axis=0) if individual_accel_scores_list else 0.0, index=df_index)
        if is_probe_date:
            print(f"  -> 维度3: 加速度 (Acceleration)")
            print(f"    avg_accel_score (末值): {avg_accel_score.iloc[-1]:.4f}")

        # 维度4: 共振 (Resonance)
        slope_consistency_score = pd.Series(0.0, index=df_index)
        if len(individual_slope_scores_list) > 1:
            concatenated_slopes = pd.concat(individual_slope_scores_list, axis=1).fillna(0)
            std_norm_slopes = concatenated_slopes.std(axis=1)
            slope_consistency_factor = (1 - (std_norm_slopes / 2.0)).clip(0, 1)
            slope_consistency_score = avg_slope_score * slope_consistency_factor
        accel_consistency_score = pd.Series(0.0, index=df_index)
        if len(individual_accel_scores_list) > 1:
            concatenated_accels = pd.concat(individual_accel_scores_list, axis=1).fillna(0)
            std_norm_accels = concatenated_accels.std(axis=1)
            accel_consistency_factor = (1 - (std_norm_accels / 2.0)).clip(0, 1)
            accel_consistency_score = avg_accel_score * accel_consistency_factor
        slope_accel_directional_alignment_score = pd.Series(0.0, index=df_index)
        if not avg_slope_score.empty and not avg_accel_score.empty:
            slope_accel_directional_alignment_score = (avg_slope_score * avg_accel_score).clip(-1, 1)
        overall_resonance_score = (
            slope_consistency_score * resonance_params['slope_consistency_weight'] +
            accel_consistency_score * resonance_params['accel_consistency_weight'] +
            slope_accel_directional_alignment_score * resonance_params['slope_accel_alignment_weight']
        ).clip(-1, 1)
        if is_probe_date:
            print(f"  -> 维度4: 共振 (Resonance)")
            print(f"    slope_consistency_score (末值): {slope_consistency_score.iloc[-1]:.4f}")
            print(f"    accel_consistency_score (末值): {accel_consistency_score.iloc[-1]:.4f}")
            print(f"    slope_accel_directional_alignment_score (末值): {slope_accel_directional_alignment_score.iloc[-1]:.4f}")
            print(f"    overall_resonance_score (末值): {overall_resonance_score.iloc[-1]:.4f}")

        # 维度5: 有序度 (Orderliness)
        orderliness_raw = self._get_safe_series(df, 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 0.0, method_name="_diagnose_axiom_trend_form")
        orderliness_score = get_adaptive_mtf_normalized_score(orderliness_raw, df_index, tf_weights, ascending=True)
        corrected_orderliness_score = orderliness_score.copy()
        arbitration_triggered = (alignment_score > 0.9) & (orderliness_score < alignment_score)
        corrected_orderliness_score[arbitration_triggered] = alignment_score[arbitration_triggered]
        if is_probe_date:
            print(f"  -> 维度5: 有序度 (Orderliness)")
            print(f"    orderliness_raw (末值): {orderliness_raw.iloc[-1]:.4f}")
            print(f"    orderliness_score (归一化末值): {orderliness_score.iloc[-1]:.4f}")
            print(f"    corrected_orderliness_score (仲裁后末值): {corrected_orderliness_score.iloc[-1]:.4f}")

        # 维度6: 角度 (Angle) - 使用调整后的敏感度
        angle_raw_ema = self._get_safe_series(df, 'ATAN_ANGLE_EMA_55_D', 0.0, method_name="_diagnose_axiom_trend_form")
        angle_score_ema = get_adaptive_mtf_normalized_bipolar_score(angle_raw_ema, df_index, tf_weights, sensitivity=adjusted_sensitivity_series)
        angle_fusion_weights = get_param_value(p_conf_struct.get('trend_form_angle_fusion_weights'), {'ema': 1.0, 'ma': 0.0})
        angle_score = (angle_score_ema * angle_fusion_weights.get('ema', 1.0))
        if is_probe_date:
            print(f"  -> 维度6: 角度 (Angle)")
            print(f"    angle_raw_ema (末值): {angle_raw_ema.iloc[-1]:.4f}")
            print(f"    angle_score (融合末值): {angle_score.iloc[-1]:.4f}")

        # 维度7: 均线粘合度 (MA Cluster Cohesion)
        ma_cluster_cohesion_score = pd.Series(0.0, index=df_index)
        if ma_cluster_cohesion_params.get('enabled', False):
            # normalization_period = ma_cluster_cohesion_params.get('normalization_period', 55) # Not directly used in this calculation
            all_ma_series = []
            for p in ema_periods:
                all_ma_series.append(self._get_safe_series(df, f'EMA_{p}_D', method_name="_diagnose_axiom_trend_form"))
            for p in ma_periods:
                all_ma_series.append(self._get_safe_series(df, f'MA_{p}_D', method_name="_diagnose_axiom_trend_form"))

            if all_ma_series:
                ma_df_for_std = pd.concat(all_ma_series, axis=1).fillna(df['close_D'])
                ma_std = ma_df_for_std.std(axis=1)

                atr_series = self._get_safe_series(df, 'ATR_14_D', 1.0, method_name="_diagnose_axiom_trend_form").replace(0, 1e-9)
                normalized_ma_std = (ma_std / atr_series).replace([np.inf, -np.inf], np.nan).fillna(0)
                ma_cluster_cohesion_score = get_adaptive_mtf_normalized_score(normalized_ma_std, df_index, tf_weights, ascending=False)

            if is_probe_date:
                print(f"  -> 维度7: 均线粘合度 (MA Cluster Cohesion)")
                print(f"    ma_std (末值): {ma_std.iloc[-1]:.4f}")
                print(f"    atr_series (末值): {atr_series.iloc[-1]:.4f}")
                print(f"    normalized_ma_std (末值): {normalized_ma_std.iloc[-1]:.4f}")
                print(f"    ma_cluster_cohesion_score (末值): {ma_cluster_cohesion_score.iloc[-1]:.4f}")

        # 修改开始：新增维度8: 价格均线乖离 (Price-MA Gap)
        price_ma_gap_score = pd.Series(0.0, index=df_index)
        if price_ma_gap_params.get('enabled', False):
            price_ma_gap_periods = get_param_value(price_ma_gap_params.get('ma_periods'), [5, 13, 21])
            price_ma_ratios = []
            for p in price_ma_gap_periods:
                col_name = f'price_vs_ma_{p}_ratio_D'
                price_ma_ratios.append(self._get_safe_series(df, col_name, 0.0, method_name="_diagnose_axiom_trend_form"))
            if price_ma_ratios:
                avg_price_ma_ratio = pd.concat(price_ma_ratios, axis=1).mean(axis=1)
                price_ma_gap_score = get_adaptive_mtf_normalized_bipolar_score(avg_price_ma_ratio, df_index, tf_weights)
            if is_probe_date:
                print(f"  -> 维度8: 价格均线乖离 (Price-MA Gap)")
                print(f"    avg_price_ma_ratio (末值): {avg_price_ma_ratio.iloc[-1]:.4f}")
                print(f"    price_ma_gap_score (末值): {price_ma_gap_score.iloc[-1]:.4f}")
        # 修改结束

        # 修改开始：新增维度9: 布林带动态 (Bollinger Band Dynamics)
        bb_dynamics_score = pd.Series(0.0, index=df_index)
        if bb_dynamics_params.get('enabled', False):
            bbw_raw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name="_diagnose_axiom_trend_form")
            bbp_raw = self._get_safe_series(df, 'BBP_21_2.0_D', 0.5, method_name="_diagnose_axiom_trend_form") # BBP默认值0.5

            # BBW分数：宽度扩张通常是趋势的标志，宽度收缩是盘整的标志。这里我们奖励扩张。
            bbw_score = get_adaptive_mtf_normalized_score(bbw_raw, df_index, tf_weights, ascending=True)
            # BBP分数：价格在布林带中的位置，高值看涨，低值看跌
            bbp_score = get_adaptive_mtf_normalized_bipolar_score(bbp_raw, df_index, tf_weights)

            bb_dynamics_score = (
                bbw_score * bb_dynamics_params.get('bbw_weight', 0.5) +
                bbp_score * bb_dynamics_params.get('bbp_weight', 0.5)
            ).clip(-1, 1) # 融合后裁剪到[-1, 1]

            if is_probe_date:
                print(f"  -> 维度9: 布林带动态 (Bollinger Band Dynamics)")
                print(f"    bbw_raw (末值): {bbw_raw.iloc[-1]:.4f}")
                print(f"    bbp_raw (末值): {bbp_raw.iloc[-1]:.4f}")
                print(f"    bbw_score (末值): {bbw_score.iloc[-1]:.4f}")
                print(f"    bbp_score (末值): {bbp_score.iloc[-1]:.4f}")
                print(f"    bb_dynamics_score (融合末值): {bb_dynamics_score.iloc[-1]:.4f}")
        # 修改结束

        # 修改开始：新增维度10: 趋势效率 (Trend Efficiency)
        trend_efficiency_score = pd.Series(0.0, index=df_index)
        if trend_efficiency_params.get('enabled', False):
            trend_efficiency_raw = self._get_safe_series(df, 'trend_efficiency_ratio_D', 0.0, method_name="_diagnose_axiom_trend_form")
            trend_efficiency_score = get_adaptive_mtf_normalized_score(trend_efficiency_raw, df_index, tf_weights, ascending=True)
            if is_probe_date:
                print(f"  -> 维度10: 趋势效率 (Trend Efficiency)")
                print(f"    trend_efficiency_raw (末值): {trend_efficiency_raw.iloc[-1]:.4f}")
                print(f"    trend_efficiency_score (末值): {trend_efficiency_score.iloc[-1]:.4f}")
        # 修改结束

        # --- 融合形态分 ---
        bullish_alignment_contrib = alignment_score * fusion_weights['alignment']
        bullish_slope_contrib = avg_slope_score.clip(lower=0) * fusion_weights['slope']
        bullish_accel_contrib = avg_accel_score.clip(lower=0) * fusion_weights['acceleration']
        bullish_resonance_contrib = overall_resonance_score.clip(lower=0) * fusion_weights['resonance']
        bullish_orderliness_contrib = corrected_orderliness_score * fusion_weights['orderliness']
        bullish_angle_contrib = angle_score.clip(lower=0) * fusion_weights['angle']
        bullish_ma_cluster_cohesion_contrib = ma_cluster_cohesion_score * fusion_weights.get('ma_cluster_cohesion', 0.0)
        # 修改开始：新增价格均线乖离、布林带动态、趋势效率的看涨贡献
        bullish_price_ma_gap_contrib = price_ma_gap_score.clip(lower=0) * fusion_weights.get('price_ma_gap', 0.0)
        bullish_bb_dynamics_contrib = bb_dynamics_score.clip(lower=0) * fusion_weights.get('bb_dynamics', 0.0)
        bullish_trend_efficiency_contrib = trend_efficiency_score * fusion_weights.get('trend_efficiency', 0.0)
        # 修改结束
        bullish_form_score = (
            bullish_alignment_contrib +
            bullish_slope_contrib +
            bullish_accel_contrib +
            bullish_resonance_contrib +
            bullish_orderliness_contrib +
            bullish_angle_contrib +
            bullish_ma_cluster_cohesion_contrib +
            bullish_price_ma_gap_contrib + # 修改：加入价格均线乖离贡献
            bullish_bb_dynamics_contrib + # 修改：加入布林带动态贡献
            bullish_trend_efficiency_contrib # 修改：加入趋势效率贡献
        ).clip(0, 1)

        bearish_alignment_contrib = (1 - alignment_score) * fusion_weights['alignment']
        bearish_slope_contrib = avg_slope_score.clip(upper=0).abs() * fusion_weights['slope']
        bearish_accel_contrib = avg_accel_score.clip(upper=0).abs() * fusion_weights['acceleration']
        bearish_resonance_contrib = overall_resonance_score.clip(upper=0).abs() * fusion_weights['resonance']
        bearish_orderliness_contrib = (1 - corrected_orderliness_score) * fusion_weights['orderliness']
        bearish_angle_contrib = angle_score.clip(upper=0).abs() * fusion_weights['angle']
        bearish_ma_cluster_cohesion_contrib = (1 - ma_cluster_cohesion_score) * fusion_weights.get('ma_cluster_cohesion', 0.0)
        # 修改开始：新增价格均线乖离、布林带动态、趋势效率的看跌贡献
        bearish_price_ma_gap_contrib = price_ma_gap_score.clip(upper=0).abs() * fusion_weights.get('price_ma_gap', 0.0)
        bearish_bb_dynamics_contrib = bb_dynamics_score.clip(upper=0).abs() * fusion_weights.get('bb_dynamics', 0.0)
        bearish_trend_efficiency_contrib = (1 - trend_efficiency_score) * fusion_weights.get('trend_efficiency', 0.0) # 效率越低，看跌贡献越高
        # 修改结束
        bearish_form_score = (
            bearish_alignment_contrib +
            bearish_slope_contrib +
            bearish_accel_contrib +
            bearish_resonance_contrib +
            bearish_orderliness_contrib +
            bearish_angle_contrib +
            bearish_ma_cluster_cohesion_contrib +
            bearish_price_ma_gap_contrib + # 修改：加入价格均线乖离贡献
            bearish_bb_dynamics_contrib + # 修改：加入布林带动态贡献
            bearish_trend_efficiency_contrib # 修改：加入趋势效率贡献
        ).clip(0, 1)

        trend_form_score = bullish_form_score - bearish_form_score
        final_score = pd.Series(trend_form_score, index=df_index).clip(-1, 1).astype(np.float32)
        if is_probe_date:
            print(f"  -> 最终融合:")
            print(f"    bullish_form_score (末值): {bullish_form_score.iloc[-1]:.4f}")
            print(f"    bearish_form_score (末值): {bearish_form_score.iloc[-1]:.4f}")
            print(f"    SCORE_STRUCT_AXIOM_TREND_FORM (最终分数末值): {final_score.iloc[-1]:.4f}")
            print(f"--- [结构公理] 趋势形态探针结束 ---")
        return final_score

    def _diagnose_axiom_stability(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.4 · 纯粹防御重构版】结构公理三：诊断“结构稳定性”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 移除了进攻性的“结构杠杆”指标，回归“纯粹防御”本质。现在完全由宏观支撑、结构韧性、微观流动性三大支柱构成。
        - 核心证据 (韧性): `support_validation_strength`作为结构在压力测试下的直接表现。
        - 【优化】使用专属的 `long_term_stability` MTF权重进行归一化。
        """
        # 移除 structural_leverage_D
        required_signals = [
            'flow_credibility_index_D', 'main_force_slippage_index_D', 'support_validation_strength_D',
            'dominant_peak_solidity_D', 'close_D'
        ]
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
        # 移除杠杆分，重新分配权重: 宏观支撑(0.4), 韧性(0.4), 流动性(0.2)
        stability_score = (
            macro_support_score * 0.4 +
            resilience_score * 0.4 +
            micro_liquidity_score * 0.2
        ).clip(0, 1)
        final_score = (stability_score * 2 - 1).astype(np.float32)
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
        # 1b. 自适应风险感知：基于布林带的 过热惩罚 与 超跌缓和
        bbp_raw = self._get_safe_series(df, 'BBP_21_2.0_D', 0.5, method_name="_diagnose_axiom_mtf_cohesion")
        # 过热惩罚：当价格进入布林带上轨的最后5%区间(BBP>0.95)时开始惩罚，突破上轨越多惩罚越大
        overheat_penalty = ((bbp_raw - 0.95).clip(lower=0) / 0.2).clip(upper=1.0) # 在BBP=1.15时惩罚达到最大
        # 超跌缓和：当价格进入布林带下轨的最初5%区间(BBP<0.05)时开始缓和，突破下轨越多缓和越大
        oversold_mitigation = (((0.05 - bbp_raw)).clip(lower=0) / 0.2).clip(upper=1.0) # 在BBP=-0.15时缓和达到最大
        # 1c. 风险调整后的宏观分
        bullish_macro_health = alignment_score * (1 - overheat_penalty)
        bearish_macro_health = (1 - alignment_score) * (1 - oversold_mitigation)
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
        return bottom_fractal_score

    def _diagnose_strategic_posture(self, axiom_trend_form: pd.Series, axiom_mtf_cohesion: pd.Series, axiom_stability: pd.Series, axiom_tension: pd.Series, platform_foundation: pd.Series, breakout_readiness: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        【V3.0 · 双通道防御版】诊断顶层“战略态势”
        - 核心升级: 重铸“静态盾”的定义，使其成为“平台基石品质”与“突破准备度”的双通道最大值。
                      这使得模型能提前感知到正在高质量构筑中的防御工事，解决了“平台基石”的认知延迟问题。
        - 核心逻辑:
          - 矛 (进攻): (趋势形态 + 宏观健康度 + 结构杠杆) * (1 + 张力催化)
          - 盾 (防御): 动态防御 * 0.6 + Max(平台品质, 突破准备度) * 0.4
        - 输出: (战略态势分数, 最终防御强度)
        """
        required_signals = ['structural_leverage_D']
        if not self._validate_required_signals(self.strategy.df_indicators, required_signals, "_diagnose_strategic_posture"):
            return pd.Series(0.0, index=axiom_trend_form.index), pd.Series(0.5, index=axiom_trend_form.index)
        df_index = axiom_trend_form.index
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('long_term_stability', {13: 0.2, 21: 0.3, 55: 0.4, 89: 0.1})
        leverage_raw = self._get_safe_series(self.strategy.df_indicators, 'structural_leverage_D', 0.0, method_name="_diagnose_strategic_posture")
        leverage_score = get_adaptive_mtf_normalized_score(leverage_raw, df_index, ascending=True, tf_weights=tf_weights)
        base_offense_score = (
            axiom_trend_form.clip(lower=0) * 0.4 +
            axiom_mtf_cohesion.clip(lower=0) * 0.4 +
            leverage_score * 0.2
        ).clip(0, 1)
        tension_catalyst_factor = 0.5
        tension_amplifier = 1 + (axiom_tension * tension_catalyst_factor)
        offense_score = (base_offense_score * tension_amplifier).clip(0, 1)
        dynamic_defense = ((axiom_stability + 1) / 2).clip(0, 1)
        # --- 修改代码开始 ---
        # 静态盾现在是“认证工程师”和“首席质量官”报告中的最大值
        static_defense = pd.concat([platform_foundation, breakout_readiness], axis=1).max(axis=1)
        # --- 修改代码结束 ---
        defense_strength = (dynamic_defense * 0.6 + static_defense * 0.4).clip(0, 1)
        conviction_factor = 0.5
        defense_modifier = (defense_strength - 0.5) * conviction_factor
        strategic_posture = (offense_score * (1 + defense_modifier)).clip(0, 1)
        final_score = strategic_posture.astype(np.float32)
        return final_score, defense_strength

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
        return final_score

    def _diagnose_leadership_potential(self, strategic_posture: pd.Series, axiom_environment: pd.Series, structural_momentum: pd.Series, axiom_tension: pd.Series) -> pd.Series:
        """
        【V1.0 · 逆势王者版】裁决“龙头潜力”
        - 核心逻辑: 在“个体强，环境弱”的特定情境下，通过寻找额外证据（动能、张力），
                      来判断标的是“真龙头”还是“补跌陷阱”。
        """
        # --- 1. 定义情境激活条件 ---
        posture_threshold = 0.7  # 个体态势足够强的阈值
        env_threshold = 0.4      # 环境足够弱的阈值
        is_conflict_zone = (strategic_posture > posture_threshold) & (axiom_environment < env_threshold)
        # --- 2. 融合裁决证据 ---
        # 证据权重: 动量(0.6), 张力(0.4)
        leadership_evidence_score = (
            structural_momentum.clip(lower=0) * 0.6 +
            axiom_tension * 0.4
        ).clip(0, 1)
        # --- 3. 输出最终裁决 ---
        # 只有在矛盾区域内，才输出龙头潜力的证据分
        final_score = (leadership_evidence_score * is_conflict_zone).astype(np.float32)
        return final_score

    def _diagnose_platform_foundation(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        【V3.0 · 法医鉴定版】对平台进行法医级鉴定并勘探其战场边界
        - 核心逻辑: 从“结构形态”、“筹码状态”、“主力行为”、“市场情绪”四大维度，
                      对平台进行全方位品质鉴定，并输出基于主力意图的动态边界。
        - 输出: (品质分, 动态高点, 动态低点, VPOC)
        """
        required_signals = [
            'BBW_21_2.0_D', 'VOL_MA_5_D', 'VOL_MA_55_D', 'high_D', 'low_D', 'close_D', 'volume_D', 'open_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'PRICE_VOLUME_ENTROPY_D', # 结构形态
            'dominant_peak_solidity_D', 'peak_separation_ratio_D', 'chip_fatigue_index_D', # 筹码状态
            'main_force_vpoc_D', 'mf_cost_zone_defense_intent_D', 'control_solidity_index_D', # 主力行为
            'counterparty_exhaustion_index_D', 'retail_panic_surrender_index_D', 'turnover_rate_f_D' # 市场情绪
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_platform_foundation"):
            nan_series = pd.Series(np.nan, index=df.index)
            return pd.Series(0.0, index=df.index), nan_series, nan_series, nan_series
        # 获取 tf_weights
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 步骤一: 识别平台状态 ---
        # 传递 tf_weights 参数
        price_stability_score = get_adaptive_mtf_normalized_score(df['BBW_21_2.0_D'], df.index, tf_weights, ascending=False)
        supply_exhaustion_score = get_adaptive_mtf_normalized_score(df['VOL_MA_5_D'] / df['VOL_MA_55_D'], df.index, tf_weights, ascending=False)
        is_in_platform_state = (price_stability_score > 0.6) & (supply_exhaustion_score > 0.6)
        min_duration = 5
        platform_group = is_in_platform_state.ne(is_in_platform_state.shift()).cumsum()
        duration_counts = platform_group.groupby(platform_group).transform('size')
        is_valid_platform_day = is_in_platform_state & (duration_counts >= min_duration)
        # --- 步骤二: 对有效平台进行法医级鉴定 ---
        platform_quality = pd.Series(0.0, index=df.index, dtype=np.float32)
        dynamic_high = pd.Series(np.nan, index=df.index, dtype=np.float32)
        dynamic_low = pd.Series(np.nan, index=df.index, dtype=np.float32)
        vpoc = pd.Series(np.nan, index=df.index, dtype=np.float32)
        # --- 新增代码开始 ---
        # 预计算所有维度的分数
        # 传递 tf_weights 参数
        s_structure = (
            get_adaptive_mtf_normalized_score(df['VOLATILITY_INSTABILITY_INDEX_21d_D'], df.index, tf_weights, ascending=False) * 0.5 +
            get_adaptive_mtf_normalized_score(df['PRICE_VOLUME_ENTROPY_D'], df.index, tf_weights, ascending=False) * 0.5
        )
        s_chips = (
            get_adaptive_mtf_normalized_score(df['dominant_peak_solidity_D'], df.index, tf_weights, ascending=True) * 0.5 +
            get_adaptive_mtf_normalized_score(df['peak_separation_ratio_D'], df.index, tf_weights, ascending=True) * 0.3 +
            get_adaptive_mtf_normalized_score(df['chip_fatigue_index_D'], df.index, tf_weights, ascending=False) * 0.2 # 疲劳指数低代表筹码稳定
        )
        s_main_force = (
            get_adaptive_mtf_normalized_score(df['mf_cost_zone_defense_intent_D'], df.index, tf_weights, ascending=True) * 0.5 +
            get_adaptive_mtf_normalized_score(df['control_solidity_index_D'], df.index, tf_weights, ascending=True) * 0.5
        )
        s_sentiment = (
            get_adaptive_mtf_normalized_score(df['counterparty_exhaustion_index_D'], df.index, tf_weights, ascending=True) * 0.5 +
            get_adaptive_mtf_normalized_score(df['retail_panic_surrender_index_D'], df.index, tf_weights, ascending=True) * 0.3 +
            get_adaptive_mtf_normalized_score(df['turnover_rate_f_D'], df.index, tf_weights, ascending=False) * 0.2
        )
        # 最终品质分权重: 主力(0.4), 筹码(0.3), 情绪(0.2), 形态(0.1)
        final_quality_score = (s_main_force * 0.4 + s_chips * 0.3 + s_sentiment * 0.2 + s_structure * 0.1).clip(0, 1)
        # --- 新增代码结束 ---
        for group_id in platform_group[is_valid_platform_day].unique():
            platform_indices = platform_group[platform_group == group_id].index
            platform_df = df.loc[platform_indices]
            # 使用最可靠的信号定义边界
            current_vpoc = platform_df['main_force_vpoc_D'].iloc[-1]
            # 简化的边界，未来可引入更复杂的试探性K线逻辑
            platform_range = (platform_df['high_D'].max() - platform_df['low_D'].min())
            current_dyn_high = current_vpoc + platform_range / 2
            current_dyn_low = current_vpoc - platform_range / 2
            # 将计算结果填充回整个平台期
            platform_quality.loc[platform_indices] = final_quality_score.loc[platform_indices]
            dynamic_high.loc[platform_indices] = current_dyn_high
            dynamic_low.loc[platform_indices] = current_dyn_low
            vpoc.loc[platform_indices] = current_vpoc
        dynamic_high.ffill(inplace=True)
        dynamic_low.ffill(inplace=True)
        vpoc.ffill(inplace=True)
        return platform_quality, dynamic_high, dynamic_low, vpoc

    def _diagnose_final_judgment(self, contextual_posture: pd.Series, defense_strength: pd.Series, structural_momentum: pd.Series) -> pd.Series:
        """
        【V1.0 · 总司令版】执行终极裁决
        - 核心逻辑: 识别并否决高风险的“力竭滞涨陷阱”模式。
        - 否决模式: 高态势分 + 弱防御 + 低动量
        """
        # --- 1. 识别“力竭滞涨陷阱” (Stagnation Trap) ---
        # 1a. 触发条件: 表面上的进攻机会
        is_trap_candidate = contextual_posture > 0.6
        # 1b. 否决证据: 防御脆弱且动能衰竭
        is_defense_weak = defense_strength < 0.4
        is_momentum_stalled = structural_momentum < 0.1
        is_veto_triggered = is_trap_candidate & is_defense_weak & is_momentum_stalled
        # --- 2. 计算否决惩罚 ---
        # 惩罚力度与防御脆弱程度和动能停滞程度相关
        defense_weakness = (0.4 - defense_strength).clip(lower=0) / 0.4
        momentum_weakness = (0.1 - structural_momentum).clip(lower=0) / 0.1
        # 惩罚基数，这是一个超参数，决定了否决的力度
        veto_penalty_base = 1.2
        veto_penalty = (defense_weakness * 0.6 + momentum_weakness * 0.4) * veto_penalty_base
        # 只在触发时施加惩罚
        final_penalty = veto_penalty * is_veto_triggered
        # --- 3. 做出最终裁决 ---
        final_judgment_score = (contextual_posture - final_penalty).clip(-1, 1)
        final_score = final_judgment_score.astype(np.float32)
        return final_score

    def _diagnose_breakout_readiness(self, df: pd.DataFrame, axiom_tension: pd.Series) -> pd.Series:
        """
        【V2.0 · 无条件监理版】诊断“突破准备度”
        - 核心升级: 废除对`is_consolidating_D`的依赖，使其成为一个无条件的、连续性的质量评估信号。
        - 评估维度: 供应枯竭度 + 主力控盘度 + 势能积蓄度
        """
        required_signals = [
            'counterparty_exhaustion_index_D', 'turnover_rate_f_D',
            'control_solidity_index_D', 'mf_cost_zone_defense_intent_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_breakout_readiness"):
            return pd.Series(0.0, index=df.index)
        # 获取 tf_weights
        p_conf_struct = get_params_block(self.strategy, 'structural_ultimate_params', {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 评估三大维度 (无条件执行) ---
        # 1a. 供应枯竭度
        # 传递 tf_weights 参数
        supply_exhaustion_score = (
            get_adaptive_mtf_normalized_score(df['counterparty_exhaustion_index_D'], df.index, tf_weights, ascending=True) * 0.7 +
            get_adaptive_mtf_normalized_score(df['turnover_rate_f_D'], df.index, tf_weights, ascending=False) * 0.3
        )
        # 1b. 主力控盘度
        main_force_control_score = (
            get_adaptive_mtf_normalized_score(df['control_solidity_index_D'], df.index, tf_weights, ascending=True) * 0.5 +
            get_adaptive_mtf_normalized_score(df['mf_cost_zone_defense_intent_D'], df.index, tf_weights, ascending=True) * 0.5
        )
        # 1c. 势能积蓄度 (直接复用结构张力公理)
        energy_accumulation_score = axiom_tension
        # --- 2. 融合输出 (无条件执行) ---
        # 权重: 主力(0.4), 供应(0.4), 势能(0.2)
        readiness_score = (
            main_force_control_score * 0.4 +
            supply_exhaustion_score * 0.4 +
            energy_accumulation_score * 0.2
        ).clip(0, 1)
        # 废除 is_consolidating 开关，直接输出连续性分数
        final_score = readiness_score.astype(np.float32)
        return final_score









