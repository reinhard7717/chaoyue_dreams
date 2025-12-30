# strategies\trend_following\intelligence\structural_intelligence.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Any
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, 
    bipolar_to_exclusive_unipolar, is_limit_up, get_adaptive_mtf_normalized_score
)

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
        # 加载结构情报模块的专属配置
        self.structural_ultimate_params = self._load_structural_config().get('structural_ultimate_params', {})
        if not self.structural_ultimate_params:
            print("    -> [结构情报警告] 未能加载 'structural_ultimate_params' 配置块，结构情报引擎可能无法正常工作。")

    def _load_structural_config(self) -> Dict:
        """
        加载 config/intelligence/structural.json 配置文件。
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 从 strategies/trend_following/intelligence/structural_intelligence.py 到 project_root/config/intelligence/structural.json
        # 向上三级到 project_root，然后进入 config/intelligence
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        config_path = os.path.join(project_root, 'config', 'intelligence', 'structural.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"    -> [结构情报] 成功加载专属配置: {config_path}")
            return config
        except FileNotFoundError:
            print(f"    -> [结构情报警告] 专属配置文件未找到: {config_path}。使用空配置。")
            return {}
        except json.JSONDecodeError:
            print(f"    -> [结构情报警告] 专属配置文件 '{config_path}' 解析失败。使用空配置。")
            return {}

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

    def get_dynamic_normalized_score(self, series: pd.Series, df_index: pd.Index, tf_weights: Dict[int, float],
                                     ascending: bool = True, method: str = "mtf_adaptive",
                                     clip_range: Tuple[float, float] = None,
                                     mapping_func: str = None) -> pd.Series:
        """
        【V1.1 · 动态归一化 - 映射函数支持】根据配置的归一化方法和参数，对Series进行归一化。
        - 支持多种归一化方法，目前主要实现 'mtf_adaptive' (原 get_adaptive_mtf_normalized_score 逻辑) 和 'quantile'。
        - 增加了裁剪功能。
        - 新增 mapping_func 参数，允许对归一化后的分数进行二次映射。
        :param series: 待归一化的Series。
        :param df_index: DataFrame的索引，用于对齐。
        :param tf_weights: 多时间框架权重，用于 'mtf_adaptive' 方法。
        :param ascending: True表示值越大分数越高，False表示值越小分数越高。
        :param method: 归一化方法，可选 'mtf_adaptive', 'quantile'。
        :param clip_range: (min_val, max_val) 元组，用于裁剪原始Series的值。
        :param mapping_func: 应用于最终分数的映射函数名称（字符串）。
        :return: 归一化后的Series，范围 [0, 1]。
        """
        if series.empty:
            return pd.Series(0.0, index=df_index)
        processed_series = series.copy()
        if clip_range and len(clip_range) == 2:
            processed_series = processed_series.clip(lower=clip_range[0], upper=clip_range[1])
        score = pd.Series(0.0, index=df_index)
        if method == "mtf_adaptive":
            score = get_adaptive_mtf_normalized_score(processed_series, df_index, tf_weights, ascending=ascending)
        elif method == "quantile":
            window_sizes = sorted(tf_weights.keys(), reverse=True)
            quantile_scores = pd.Series(0.0, index=df_index)
            for window_val in window_sizes:
                _window = int(window_val)
                _min_periods = int(window_val)
                # 确保有足够的数据进行滚动窗口计算
                if len(processed_series) < _window:
                    # 如果数据不足，则该窗口的贡献为0，或者可以填充为NaN，最终由fillna处理
                    continue 
                rank = processed_series.rolling(window=_window, min_periods=_min_periods).apply(
                    lambda x: x.rank(pct=True).iloc[-1], raw=False
                )
                if not ascending:
                    rank = 1 - rank
                quantile_scores += rank * tf_weights.get(window_val, 0)
            total_weight = sum(tf_weights.values())
            if total_weight > 0:
                score = quantile_scores / total_weight
            else:
                score = pd.Series(0.5, index=df_index) # 默认中性分数
            score = score.fillna(0.5) # 填充滚动窗口计算可能产生的NaN
        else:
            print(f"    -> [结构情报警告] 未知归一化方法 '{method}'，回退到 'mtf_adaptive'。")
            score = get_adaptive_mtf_normalized_score(processed_series, df_index, tf_weights, ascending=ascending)
        score = score.clip(0, 1) # 确保分数在 [0, 1] 范围内
        # 应用映射函数
        if mapping_func:
            if mapping_func == "hurst_mapping":
                # 赫斯特指数的自定义映射函数示例：
                # 如果赫斯特指数原始值较高（例如 > 0.75，表示强趋势），但其归一化分数较低（例如 < 0.5），
                # 则适当提升其分数，以更好地反映其趋势性。
                # 这是一个示例，实际映射逻辑需要根据业务需求和回测结果进行精细调整。
                def _hurst_mapping_func_impl(s, raw_series):
                    # 只有当原始赫斯特指数较高且当前分数相对较低时才进行提升
                    boost_condition = (raw_series > 0.75) & (s < 0.6)
                    # 提升分数，使其更接近0.75，但不超过1
                    s[boost_condition] = s[boost_condition] + (0.75 - s[boost_condition]) * 0.5 
                    return s.clip(0, 1)
                score = _hurst_mapping_func_impl(score, series) # 注意这里传入的是原始series，以便映射函数可以访问原始值
            # 未来可以添加其他映射函数
            else:
                print(f"    -> [结构情报警告] 未知映射函数 '{mapping_func}'，跳过应用。")
        return score

    def diagnose_structural_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 荣耀代价版】结构情报分析总指挥
        - 核心升级: 引入“荣耀的代价”协议。当“龙头潜力”激活时，其分值将作为“豁免系数”，
                      部分抵消环境惩罚，而非简单叠加奖励。最终得分同时体现逆势的荣耀与代价。
        - 【V6.0 · 结构动量深度进化版】更新结构动量计算逻辑，调用新的 `_diagnose_structural_momentum` 方法。
        - 【V6.0.1 · 探针清晰化】在调用 `_diagnose_structural_momentum` 时，增加 `posture_type` 参数，使探针输出更明确。
        """
        all_states = {}
        p_conf = self.structural_ultimate_params
        if not get_param_value(p_conf.get('enabled'), True):
            print("结构情报引擎已在配置中禁用，跳过。")
            return {}
        debug_config = get_params_block(self.strategy, 'debug_params', {})
        current_processing_date_str = df.index[-1].strftime('%Y-%m-%d') if not df.empty else ""
        self.is_probe_date = debug_config.get('should_probe', False) and \
                             current_processing_date_str in debug_config.get('probe_dates', [])
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
        # 修改开始：新增 posture_type 参数
        structural_momentum = self._diagnose_structural_momentum(
            df, contextual_posture_base_for_momentum, axiom_tension, breakout_readiness, axiom_stability, axiom_mtf_cohesion, posture_type="base_for_leadership"
        )
        # --- 步骤五: 龙头潜力裁决 ---
        leadership_potential = self._diagnose_leadership_potential(
            strategic_posture, axiom_environment, structural_momentum, axiom_tension
        )
        all_states['SCORE_STRUCT_LEADERSHIP_POTENTIAL'] = leadership_potential
        # --- 步骤六: “荣耀的代价” -> 计算最终情境态势 ---
        waiver_coefficient = leadership_potential
        effective_env_modifier = env_modifier * (1 - waiver_coefficient)
        contextual_posture = (strategic_posture * (1 + effective_env_modifier)).clip(0, 1)
        all_states['SCORE_STRUCT_CONTEXTUAL_POSTURE'] = contextual_posture.astype(np.float32)
        # --- 步骤七: 基于最终情境态势，更新动量 ---
        # 修改开始：新增 posture_type 参数
        final_structural_momentum = self._diagnose_structural_momentum(
            df, contextual_posture, axiom_tension, breakout_readiness, axiom_stability, axiom_mtf_cohesion, posture_type="final_score"
        )
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
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"\n  [结构情报探针] -> 总指挥诊断结果 ({current_date}):")
            print(f"    -> SCORE_STRUCT_AXIOM_TREND_FORM: {axiom_trend_form.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_AXIOM_MTF_COHESION: {axiom_mtf_cohesion.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_AXIOM_STABILITY: {axiom_stability.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_AXIOM_DIVERGENCE: {axiom_divergence.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_BOTTOM_FRACTAL: {bottom_fractal_score.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_AXIOM_TENSION: {axiom_tension.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_AXIOM_ENVIRONMENT: {axiom_environment.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_PLATFORM_FOUNDATION: {platform_quality.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_BREAKOUT_READINESS: {breakout_readiness.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_STRATEGIC_POSTURE: {strategic_posture.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_LEADERSHIP_POTENTIAL: {leadership_potential.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_CONTEXTUAL_POSTURE: {contextual_posture.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_MOMENTUM: {final_structural_momentum.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_PLAYBOOK_SECONDARY_LAUNCH: {playbook_secondary_launch.iloc[-1]:.4f}")
            print(f"    -> SCORE_STRUCT_FINAL_JUDGMENT: {final_judgment.iloc[-1]:.4f}")
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.5 · 结构动量背离版】结构公理四：诊断“结构背离”
        - 核心逻辑: 诊断价格结构动量与均线结构动量之间的背离。
        - 核心升级: 将价格趋势从 `pct_change_D` 替换为 `SLOPE_5_close_D`，将均线结构趋势从 `EMA_5_D` 和 `EMA_55_D` 的差值变化率替换为 `SLOPE_5_EMA_55_D`。
                    这使得背离的判断更纯粹地基于结构性动量。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】使用专属的MTF权重配置进行归一化。
        """
        method_name = "_diagnose_axiom_divergence"
        required_signals = ['SLOPE_5_close_D', 'SLOPE_5_EMA_55_D'] # 更新依赖信号
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        p_conf_struct = self.structural_ultimate_params
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 价格结构动量
        price_structural_momentum_raw = self._get_safe_series(df, 'SLOPE_5_close_D', 0.0, method_name=method_name)
        price_structural_momentum_score = get_adaptive_mtf_normalized_bipolar_score(price_structural_momentum_raw, df.index, tf_weights)
        # 均线结构动量
        ma_structural_momentum_raw = self._get_safe_series(df, 'SLOPE_5_EMA_55_D', 0.0, method_name=method_name)
        ma_structural_momentum_score = get_adaptive_mtf_normalized_bipolar_score(ma_structural_momentum_raw, df.index, tf_weights)
        # 计算结构动量背离
        divergence_score = (ma_structural_momentum_score - price_structural_momentum_score).clip(-1, 1)
        final_score = divergence_score.astype(np.float32)
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 价格结构动量原始值 (SLOPE_5_close_D): {price_structural_momentum_raw.iloc[-1]:.4f}")
            print(f"    -> 价格结构动量分数: {price_structural_momentum_score.iloc[-1]:.4f}")
            print(f"    -> 均线结构动量原始值 (SLOPE_5_EMA_55_D): {ma_structural_momentum_raw.iloc[-1]:.4f}")
            print(f"    -> 均线结构动量分数: {ma_structural_momentum_score.iloc[-1]:.4f}")
            print(f"    -> 最终结构动量背离分数: {final_score.iloc[-1]:.4f}")
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
            - 引入**动态权重调整**：根据市场波动率动态调整趋势形态的最终融合分数。
            - 引入**均线粘合度**维度：量化均线簇的紧密程度，作为趋势形态健康的新指标。
            - **波动率调整的斜率/角度敏感度**：使斜率、加速度和角度指标在高波动率时更稳健，低波动率时更灵敏。
        - 【V3.7 核心进化】
            - 引入**价格均线乖离 (Price-MA Gap)** 维度：量化价格相对于均线的延伸度。
            - 引入**布林带动态 (Bollinger Band Dynamics)** 维度：评估布林带宽度和价格在带内的位置。
            - 引入**趋势效率 (Trend Efficiency)** 维度：衡量趋势运动的平滑度和直接性。
        """
        method_name = "_diagnose_axiom_trend_form"
        p_conf_struct = self.structural_ultimate_params
        ema_periods = get_param_value(p_conf_struct.get('trend_form_ema_periods'), [5, 13, 21, 34, 55])
        ma_periods = get_param_value(p_conf_struct.get('trend_form_ma_periods'), [5, 13, 21, 34, 55])
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
        configured_fusion_weights = get_param_value(p_conf_struct.get('trend_form_fusion_weights'), {})
        fusion_weights = default_fusion_weights.copy()
        fusion_weights.update(configured_fusion_weights)
        fusion_weights = {k: v for k, v in fusion_weights.items() if isinstance(v, (int, float))}
        resonance_params = get_param_value(p_conf_struct.get('trend_form_resonance_params'), {
            'enabled': True,
            'slope_consistency_weight': 0.4,
            'accel_consistency_weight': 0.4,
            'slope_accel_alignment_weight': 0.2
        })
        dynamic_weights_params = get_param_value(p_conf_struct.get('trend_form_dynamic_weights_params'), {})
        ma_cluster_cohesion_params = get_param_value(p_conf_struct.get('ma_cluster_cohesion_params'), {})
        slope_angle_volatility_adjustment_params = get_param_value(p_conf_struct.get('slope_angle_volatility_adjustment_params'), {})
        price_ma_gap_params = get_param_value(p_conf_struct.get('price_ma_gap_params'), {})
        bb_dynamics_params = get_param_value(p_conf_struct.get('bb_dynamics_params'), {})
        trend_efficiency_params = get_param_value(p_conf_struct.get('trend_efficiency_params'), {})
        required_signals = [
            'MA_POTENTIAL_ORDERLINESS_SCORE_D', 'ATAN_ANGLE_EMA_55_D', 'close_D',
            'ATR_14_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'BBW_21_2.0_D', 'BBP_21_2.0_D',
            'trend_efficiency_ratio_D', 'MA_POTENTIAL_COMPRESSION_RATE_D', # 新增
            'structural_entropy_change_D', 'impulse_quality_ratio_D' # 新增
        ]
        required_signals.extend([f'EMA_{p}_D' for p in ema_periods])
        required_signals.extend([f'MA_{p}_D' for p in ma_periods])
        price_ma_gap_periods = get_param_value(price_ma_gap_params.get('ma_periods'), [5, 13, 21])
        required_signals.extend([f'price_vs_ma_{p}_ratio_D' for p in price_ma_gap_periods])
        feature_eng_params = get_params_block(self.strategy, 'feature_engineering_params', {})
        slope_params = feature_eng_params.get('slope_params', {})
        accel_params = feature_eng_params.get('accel_params', {})
        series_to_slope_config = slope_params.get('series_to_slope', {})
        series_to_accel_config = accel_params.get('series_to_accel', {})
        all_slope_cols = []
        all_accel_cols = []
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
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_struct = self.structural_ultimate_params
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('short_term_geometry', {5: 0.5, 8: 0.3, 13: 0.2})
        # 动态权重计算逻辑
        volatility_adjustment_factor_series = pd.Series(1.0, index=df_index)
        if dynamic_weights_params.get('enabled', False):
            volatility_source_col = dynamic_weights_params.get('volatility_source', 'VOLATILITY_INSTABILITY_INDEX_21d_D')
            volatility_series = self._get_safe_series(df, volatility_source_col, 0.0, method_name=method_name)
            volatility_sensitivity = dynamic_weights_params.get('volatility_sensitivity', 2.0)
            volatility_threshold = dynamic_weights_params.get('volatility_threshold', 0.5)
            adjustment_factor_range = dynamic_weights_params.get('adjustment_factor_range', 0.5)
            norm_volatility = get_adaptive_mtf_normalized_score(volatility_series, df_index, tf_weights, ascending=True)
            volatility_adjustment_factor_series = 1 + adjustment_factor_range * (volatility_threshold - norm_volatility) * volatility_sensitivity
            volatility_adjustment_factor_series = volatility_adjustment_factor_series.clip(1 - adjustment_factor_range, 1 + adjustment_factor_range)
        # 维度1: 排列 (Alignment) - 融合EMA和MA
        bull_alignment_raw_ema = pd.Series(0.0, index=df_index)
        alignment_weights_internal = np.linspace(0.5, 0.2, len(ema_periods) - 1)
        for i in range(len(ema_periods) - 1):
            ema_i = self._get_safe_series(df, f'EMA_{ema_periods[i]}_D', method_name=method_name)
            ema_i_plus_1 = self._get_safe_series(df, f'EMA_{ema_periods[i+1]}_D', method_name=method_name)
            bull_alignment_raw_ema += (ema_i > ema_i_plus_1).astype(float) * alignment_weights_internal[i]
        alignment_score_ema = bull_alignment_raw_ema / sum(alignment_weights_internal)
        bull_alignment_raw_ma = pd.Series(0.0, index=df_index)
        alignment_weights_internal_ma = np.linspace(0.5, 0.2, len(ma_periods) - 1)
        for i in range(len(ma_periods) - 1):
            ma_i = self._get_safe_series(df, f'MA_{ma_periods[i]}_D', method_name=method_name)
            ma_i_plus_1 = self._get_safe_series(df, f'MA_{ma_periods[i+1]}_D', method_name=method_name)
            bull_alignment_raw_ma += (ma_i > ma_i_plus_1).astype(float) * alignment_weights_internal_ma[i]
        alignment_score_ma = bull_alignment_raw_ma / sum(alignment_weights_internal_ma)
        alignment_fusion_weights = get_param_value(p_conf_struct.get('trend_form_alignment_fusion_weights'), {'ema': 0.6, 'ma': 0.4})
        alignment_score = (alignment_score_ema * alignment_fusion_weights.get('ema', 0.5) +
                           alignment_score_ma * alignment_fusion_weights.get('ma', 0.5))
        # 波动率调整斜率和角度的敏感度
        adjusted_sensitivity_series = pd.Series(1.0, index=df_index)
        if slope_angle_volatility_adjustment_params.get('enabled', False):
            volatility_source_col = slope_angle_volatility_adjustment_params.get('volatility_source', 'ATR_14_D')
            volatility_series = self._get_safe_series(df, volatility_source_col, 0.0, method_name=method_name)
            adjustment_strength = slope_angle_volatility_adjustment_params.get('adjustment_strength', 0.5)
            norm_volatility = get_adaptive_mtf_normalized_score(volatility_series, df_index, tf_weights, ascending=True)
            sensitivity_adjustment_factor = 1 - adjustment_strength * (norm_volatility - 0.5) * 2
            adjusted_sensitivity_series = sensitivity_adjustment_factor.clip(0.1, 2.0)
        # 维度2: 斜率 (Slope) - 使用调整后的敏感度
        individual_slope_scores_list = []
        for col in all_slope_cols:
            raw_slope_series = self._get_safe_series(df, col, 0.0, method_name=method_name)
            normalized_slope_score = get_adaptive_mtf_normalized_bipolar_score(raw_slope_series, df_index, tf_weights, sensitivity=adjusted_sensitivity_series)
            individual_slope_scores_list.append(normalized_slope_score)
        avg_slope_score = pd.Series(np.mean([s.values for s in individual_slope_scores_list], axis=0) if individual_slope_scores_list else 0.0, index=df_index)
        # 维度3: 加速度 (Acceleration) - 同样使用调整后的敏感度
        individual_accel_scores_list = []
        for col in all_accel_cols:
            raw_accel_series = self._get_safe_series(df, col, 0.0, method_name=method_name)
            normalized_accel_score = get_adaptive_mtf_normalized_bipolar_score(raw_accel_series, df_index, tf_weights, sensitivity=adjusted_sensitivity_series)
            individual_accel_scores_list.append(normalized_accel_score)
        avg_accel_score = pd.Series(np.mean([s.values for s in individual_accel_scores_list], axis=0) if individual_accel_scores_list else 0.0, index=df_index)
        # 维度4: 共振 (Resonance)
        # 引入 structural_entropy_change_D
        structural_entropy_change_raw = self._get_safe_series(df, 'structural_entropy_change_D', 0.0, method_name=method_name)
        structural_entropy_change_score = get_adaptive_mtf_normalized_score(structural_entropy_change_raw, df_index, tf_weights, ascending=False) # 熵变越小越好
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
            slope_accel_directional_alignment_score * resonance_params['slope_accel_alignment_weight'] +
            structural_entropy_change_score * 0.1 # 增加熵变贡献
        ).clip(-1, 1)
        # 维度5: 有序度 (Orderliness)
        orderliness_raw = self._get_safe_series(df, 'MA_POTENTIAL_ORDERLINESS_SCORE_D', 0.0, method_name=method_name)
        orderliness_score = get_adaptive_mtf_normalized_score(orderliness_raw, df_index, tf_weights, ascending=True)
        corrected_orderliness_score = orderliness_score.copy()
        arbitration_triggered = (alignment_score > 0.9) & (orderliness_score < alignment_score)
        corrected_orderliness_score[arbitration_triggered] = alignment_score[arbitration_triggered]
        # 维度6: 角度 (Angle) - 使用调整后的敏感度
        angle_raw_ema = self._get_safe_series(df, 'ATAN_ANGLE_EMA_55_D', 0.0, method_name=method_name)
        angle_score_ema = get_adaptive_mtf_normalized_bipolar_score(angle_raw_ema, df_index, tf_weights, sensitivity=adjusted_sensitivity_series)
        angle_fusion_weights = get_param_value(p_conf_struct.get('trend_form_angle_fusion_weights'), {'ema': 1.0, 'ma': 0.0})
        angle_score = (angle_score_ema * angle_fusion_weights.get('ema', 1.0))
        # 维度7: 均线粘合度 (MA Cluster Cohesion)
        ma_cluster_cohesion_score = pd.Series(0.0, index=df_index)
        if ma_cluster_cohesion_params.get('enabled', False):
            all_ma_series = []
            for p in ema_periods:
                all_ma_series.append(self._get_safe_series(df, f'EMA_{p}_D', method_name=method_name))
            for p in ma_periods:
                all_ma_series.append(self._get_safe_series(df, f'MA_{p}_D', method_name=method_name))
            if all_ma_series:
                ma_df_for_std = pd.concat(all_ma_series, axis=1).fillna(df['close_D'])
                ma_std = ma_df_for_std.std(axis=1)
                atr_series = self._get_safe_series(df, 'ATR_14_D', 1.0, method_name=method_name).replace(0, 1e-9)
                normalized_ma_std = (ma_std / atr_series).replace([np.inf, -np.inf], np.nan).fillna(0)
                ma_cluster_cohesion_score_std = get_adaptive_mtf_normalized_score(normalized_ma_std, df_index, tf_weights, ascending=False)
                # 引入 MA_POTENTIAL_COMPRESSION_RATE_D
                ma_compression_raw = self._get_safe_series(df, 'MA_POTENTIAL_COMPRESSION_RATE_D', 0.0, method_name=method_name)
                ma_compression_score = get_adaptive_mtf_normalized_score(ma_compression_raw, df_index, tf_weights, ascending=True)
                ma_cluster_cohesion_score = (ma_cluster_cohesion_score_std * 0.7 + ma_compression_score * 0.3).clip(0,1)
        # 维度8: 价格均线乖离 (Price-MA Gap)
        price_ma_gap_score = pd.Series(0.0, index=df_index)
        if price_ma_gap_params.get('enabled', False):
            price_ma_gap_periods = get_param_value(price_ma_gap_params.get('ma_periods'), [5, 13, 21])
            price_ma_ratios = []
            for p in price_ma_gap_periods:
                col_name = f'price_vs_ma_{p}_ratio_D'
                price_ma_ratios.append(self._get_safe_series(df, col_name, 0.0, method_name=method_name))
            if price_ma_ratios:
                avg_price_ma_ratio = pd.concat(price_ma_ratios, axis=1).mean(axis=1)
                price_ma_gap_score = get_adaptive_mtf_normalized_bipolar_score(avg_price_ma_ratio, df_index, tf_weights)
        # 维度9: 布林带动态 (Bollinger Band Dynamics)
        bb_dynamics_score = pd.Series(0.0, index=df_index)
        if bb_dynamics_params.get('enabled', False):
            bbw_raw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name=method_name)
            bbp_raw = self._get_safe_series(df, 'BBP_21_2.0_D', 0.5, method_name=method_name)
            bbw_score = get_adaptive_mtf_normalized_score(bbw_raw, df_index, tf_weights, ascending=True)
            bbp_score = get_adaptive_mtf_normalized_bipolar_score(bbp_raw, df_index, tf_weights)
            bb_dynamics_score = (
                bbw_score * bb_dynamics_params.get('bbw_weight', 0.5) +
                bbp_score * bb_dynamics_params.get('bbp_weight', 0.5)
            ).clip(-1, 1)
        # 维度10: 趋势效率 (Trend Efficiency)
        trend_efficiency_score = pd.Series(0.0, index=df_index)
        if trend_efficiency_params.get('enabled', False):
            trend_efficiency_raw = self._get_safe_series(df, 'trend_efficiency_ratio_D', 0.0, method_name=method_name)
            impulse_quality_raw = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name=method_name)
            trend_efficiency_score_base = get_adaptive_mtf_normalized_score(trend_efficiency_raw, df_index, tf_weights, ascending=True)
            impulse_quality_score = get_adaptive_mtf_normalized_score(impulse_quality_raw, df_index, tf_weights, ascending=True)
            trend_efficiency_score = (trend_efficiency_score_base * 0.7 + impulse_quality_score * 0.3).clip(0,1)
        # --- 融合形态分 ---
        bullish_alignment_contrib = alignment_score * fusion_weights['alignment']
        bullish_slope_contrib = avg_slope_score.clip(lower=0) * fusion_weights['slope']
        bullish_accel_contrib = avg_accel_score.clip(lower=0) * fusion_weights['acceleration']
        bullish_resonance_contrib = overall_resonance_score.clip(lower=0) * fusion_weights['resonance']
        bullish_orderliness_contrib = corrected_orderliness_score * fusion_weights['orderliness']
        bullish_angle_contrib = angle_score.clip(lower=0) * fusion_weights['angle']
        bullish_ma_cluster_cohesion_contrib = ma_cluster_cohesion_score * fusion_weights.get('ma_cluster_cohesion', 0.0)
        bullish_price_ma_gap_contrib = price_ma_gap_score.clip(lower=0) * fusion_weights.get('price_ma_gap', 0.0)
        bullish_bb_dynamics_contrib = bb_dynamics_score.clip(lower=0) * fusion_weights.get('bb_dynamics', 0.0)
        bullish_trend_efficiency_contrib = trend_efficiency_score * fusion_weights.get('trend_efficiency', 0.0)
        bullish_form_score = (
            bullish_alignment_contrib +
            bullish_slope_contrib +
            bullish_accel_contrib +
            bullish_resonance_contrib +
            bullish_orderliness_contrib +
            bullish_angle_contrib +
            bullish_ma_cluster_cohesion_contrib +
            bullish_price_ma_gap_contrib +
            bullish_bb_dynamics_contrib +
            bullish_trend_efficiency_contrib
        ).clip(0, 1)
        bearish_alignment_contrib = (1 - alignment_score) * fusion_weights['alignment']
        bearish_slope_contrib = avg_slope_score.clip(upper=0).abs() * fusion_weights['slope']
        bearish_accel_contrib = avg_accel_score.clip(upper=0).abs() * fusion_weights['acceleration']
        bearish_resonance_contrib = overall_resonance_score.clip(upper=0).abs() * fusion_weights['resonance']
        bearish_orderliness_contrib = (1 - corrected_orderliness_score) * fusion_weights['orderliness']
        bearish_angle_contrib = angle_score.clip(upper=0).abs() * fusion_weights['angle']
        bearish_ma_cluster_cohesion_contrib = (1 - ma_cluster_cohesion_score) * fusion_weights.get('ma_cluster_cohesion', 0.0)
        bearish_price_ma_gap_contrib = price_ma_gap_score.clip(upper=0).abs() * fusion_weights.get('price_ma_gap', 0.0)
        bearish_bb_dynamics_contrib = bb_dynamics_score.clip(upper=0).abs() * fusion_weights.get('bb_dynamics', 0.0)
        bearish_trend_efficiency_contrib = (1 - trend_efficiency_score) * fusion_weights.get('trend_efficiency', 0.0)
        bearish_form_score = (
            bearish_alignment_contrib +
            bearish_slope_contrib +
            bearish_accel_contrib +
            bearish_resonance_contrib +
            bearish_orderliness_contrib +
            bearish_angle_contrib +
            bearish_ma_cluster_cohesion_contrib +
            bearish_price_ma_gap_contrib +
            bearish_bb_dynamics_contrib +
            bearish_trend_efficiency_contrib
        ).clip(0, 1)
        trend_form_score = bullish_form_score - bearish_form_score
        if dynamic_weights_params.get('enabled', False):
            trend_form_score = trend_form_score * volatility_adjustment_factor_series
        final_score = pd.Series(trend_form_score, index=df_index).clip(-1, 1).astype(np.float32)
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 排列分数 (Alignment): {alignment_score.iloc[-1]:.4f}")
            print(f"    -> 平均斜率分数 (Avg Slope): {avg_slope_score.iloc[-1]:.4f}")
            print(f"    -> 平均加速度分数 (Avg Accel): {avg_accel_score.iloc[-1]:.4f}")
            print(f"    -> 结构熵变原始值: {structural_entropy_change_raw.iloc[-1]:.4f}")
            print(f"    -> 结构熵变分数: {structural_entropy_change_score.iloc[-1]:.4f}")
            print(f"    -> 共振分数 (Resonance): {overall_resonance_score.iloc[-1]:.4f}")
            print(f"    -> 有序度分数 (Orderliness): {corrected_orderliness_score.iloc[-1]:.4f}")
            print(f"    -> 角度分数 (Angle): {angle_score.iloc[-1]:.4f}")
            print(f"    -> 均线压缩率原始值: {ma_compression_raw.iloc[-1]:.4f}")
            print(f"    -> 均线压缩率分数: {ma_compression_score.iloc[-1]:.4f}")
            print(f"    -> 均线粘合度分数 (MA Cluster Cohesion): {ma_cluster_cohesion_score.iloc[-1]:.4f}")
            print(f"    -> 价格均线乖离分数 (Price-MA Gap): {price_ma_gap_score.iloc[-1]:.4f}")
            print(f"    -> 布林带动态分数 (BB Dynamics): {bb_dynamics_score.iloc[-1]:.4f}")
            print(f"    -> 趋势效率原始值: {trend_efficiency_raw.iloc[-1]:.4f}")
            print(f"    -> 脉冲质量原始值: {impulse_quality_raw.iloc[-1]:.4f}")
            print(f"    -> 趋势效率分数 (Trend Efficiency): {trend_efficiency_score.iloc[-1]:.4f}")
            print(f"    -> 看涨形态分 (Bullish Form): {bullish_form_score.iloc[-1]:.4f}")
            print(f"    -> 看跌形态分 (Bearish Form): {bearish_form_score.iloc[-1]:.4f}")
            print(f"    -> 最终趋势形态分数: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_axiom_stability(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.9.3 · 纯结构深度进化版 - 变量名修正】结构公理三：诊断“结构稳定性”
        - 核心升级: 彻底重构为六大核心支柱：结构支撑强度、结构形态坚固性、波动率秩序性、结构运动效率、结构突破强度、结构回撤效率。
                    严格限定在纯粹的【结构】类原始数据范畴内，移除筹码、资金等其他维度的数据。
        - 核心证据:
            - 结构支撑强度: 支撑验证强度、压力拒绝强度、下影线吸收强度、防御坚实度、开盘跳空防御强度。
            - 结构形态坚固性: 均衡压缩、平台信念分数、价值区域重叠度、拟合优度分数、结构节点数量。
            - 波动率秩序性: 布林带宽度、波动率不稳定性、分形维度、样本熵、赫斯特指数。
            - 结构运动效率: 趋势效率比、非对称摩擦指数、脉冲质量比、均线有序度。
            - 结构突破强度: 突破成交量比率、突破区间扩张、突破回踩成功率、突破持续时间。
            - 结构回撤效率: 回撤深度百分比、回撤速度比率、回撤成交量衰减、回撤均线粘附。
        - 【优化】引入 `get_dynamic_normalized_score` 函数，支持分位数归一化、裁剪和映射函数，解决部分指标归一化不符预期的问题。
        - 【探针植入】增加了详细的探针输出，以便于检查和调试。
        - 【数据鲁棒性】对分形维度和赫斯特指数进行合理性裁剪，避免异常值影响。
        - 【V5.9 信号替换】根据数据层实际提供的信号，替换了结构突破强度和结构回撤效率的原始信号。
        - 【V5.9.1 权重匹配修正】统一了子维度权重字典的键名和实际获取权重时的键名，确保与JSON配置中的概念名称一致。
        - 【V5.9.2 信号映射】引入集中的信号映射配置，提高代码的可读性和可维护性。
        - 【V5.9.3 变量名修正】修正了结构支撑强度融合计算中错误的变量名。
        """
        method_name = "_diagnose_axiom_stability"
        # required_signals 使用概念名称，并通过映射获取实际信号名称
        concept_signals = [
            'support_validation_strength_D', 'pressure_rejection_strength_D', 'lower_shadow_absorption_strength_D',
            'defense_solidity_score_D', 'opening_gap_defense_strength_D',
            'equilibrium_compression_index_D', 'value_area_overlap_pct_D',
            'structural_node_count_D', 'platform_conviction_score_D', 'goodness_of_fit_score_D',
            'BBW_21_2.0_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'FRACTAL_DIMENSION_89d_D',
            'SAMPLE_ENTROPY_13d_D', 'HURST_144d_D',
            'trend_efficiency_ratio_D', 'asymmetric_friction_index_D', 'impulse_quality_ratio_D',
            'MA_POTENTIAL_ORDERLINESS_SCORE_D', 'close_D',
            'breakout_volume_ratio_D', 'breakout_range_expansion_D', 'breakout_retest_success_D', 'breakout_duration_D',
            'retracement_depth_pct_D', 'retracement_speed_ratio_D', 'retracement_volume_decay_D', 'retracement_MA_adherence_D'
        ]
        p_conf_struct = self.structural_ultimate_params
        stability_params = get_param_value(p_conf_struct.get('stability_params'), {})
        signal_mapping = get_param_value(stability_params.get('signal_mapping'), {})
        # 根据映射配置生成实际的 required_signals
        required_signals = []
        for concept_signal in concept_signals:
            actual_signal = signal_mapping.get(concept_signal, concept_signal)
            required_signals.append(actual_signal)
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('long_term_stability', {13: 0.2, 21: 0.3, 55: 0.4, 89: 0.1})
        raw_stability_fusion_weights = get_param_value(p_conf_struct.get('stability_fusion_weights'), {
            "structural_support_strength": 0.25,
            "structural_form_solidity": 0.25,
            "volatility_orderliness": 0.2,
            "structural_movement_efficiency": 0.15,
            "structural_break_strength": 0.075,
            "structural_retracement_efficiency": 0.075
        })
        stability_fusion_weights = {k: v for k, v in raw_stability_fusion_weights.items() if isinstance(v, (int, float))}
        structural_support_strength_weights = get_param_value(stability_params.get('structural_support_strength_weights'), {
            "support_validation_strength": 0.3, "pressure_rejection_strength": 0.3, "lower_shadow_absorption_strength": 0.2,
            "defense_solidity_score": 0.1, "opening_gap_defense_strength": 0.1
        })
        structural_form_solidity_weights = get_param_value(stability_params.get('structural_form_solidity_weights'), {
            "equilibrium_compression_index": 0.3, "platform_conviction_score": 0.25, "value_area_overlap_pct": 0.2,
            "goodness_of_fit_score": 0.15, "structural_node_count": 0.1
        })
        volatility_orderliness_weights = get_param_value(stability_params.get('volatility_orderliness_weights'), {
            "BBW_21_2.0": 0.3, "VOLATILITY_INSTABILITY_INDEX_21d": 0.3, "FRACTAL_DIMENSION_89d": 0.2,
            "SAMPLE_ENTROPY_13d": 0.1, "HURST_144d": 0.1
        })
        structural_movement_efficiency_weights = get_param_value(stability_params.get('structural_movement_efficiency_weights'), {
            "trend_efficiency_ratio": 0.3, "asymmetric_friction_index": 0.3, "impulse_quality_ratio": 0.2,
            "MA_POTENTIAL_ORDERLINESS_SCORE": 0.2
        })
        structural_break_strength_weights = get_param_value(stability_params.get('structural_break_strength_weights'), {
            "breakout_volume_ratio": 0.3, "breakout_range_expansion": 0.3, "breakout_retest_success": 0.2, "breakout_duration": 0.2
        })
        structural_retracement_efficiency_weights = get_param_value(stability_params.get('structural_retracement_efficiency_weights'), {
            "retracement_depth_pct": 0.3, "retracement_speed_ratio": 0.3, "retracement_volume_decay": 0.2, "retracement_MA_adherence": 0.2
        })
        normalization_configs = get_param_value(stability_params.get('normalization_configs'), {})
        # Helper to get normalization config for a signal
        # get_norm_config 接受概念信号名称，并查找其对应的实际信号名称来获取配置
        def get_norm_config(concept_signal_name, default_method="mtf_adaptive", default_ascending=True, default_clip_range=None):
            actual_signal_name = signal_mapping.get(concept_signal_name, concept_signal_name)
            config = normalization_configs.get(actual_signal_name, {})
            return {
                "method": config.get("method", default_method),
                "ascending": config.get("ascending", default_ascending),
                "clip_range": config.get("clip_range", default_clip_range),
                "mapping_func": config.get("mapping_func", None)
            }
        # --- 1. 结构支撑强度 (Structural Support Strength) ---
        # 所有 _get_safe_series 调用都通过 signal_mapping 获取实际信号名称
        support_validation_strength_raw = self._get_safe_series(df, signal_mapping.get('support_validation_strength_D', 'support_validation_strength_D'), 0.0, method_name=method_name)
        pressure_rejection_strength_raw = self._get_safe_series(df, signal_mapping.get('pressure_rejection_strength_D', 'pressure_rejection_strength_D'), 0.0, method_name=method_name)
        lower_shadow_absorption_strength_raw = self._get_safe_series(df, signal_mapping.get('lower_shadow_absorption_strength_D', 'lower_shadow_absorption_strength_D'), 0.0, method_name=method_name)
        defense_solidity_score_raw = self._get_safe_series(df, signal_mapping.get('defense_solidity_score_D', 'defense_solidity_score_D'), 0.0, method_name=method_name)
        opening_gap_defense_strength_raw = self._get_safe_series(df, signal_mapping.get('opening_gap_defense_strength_D', 'opening_gap_defense_strength_D'), 0.0, method_name=method_name)
        support_validation_strength_score = self.get_dynamic_normalized_score(
            support_validation_strength_raw, df_index, tf_weights, **get_norm_config('support_validation_strength_D', default_ascending=True))
        pressure_rejection_strength_score = self.get_dynamic_normalized_score(
            pressure_rejection_strength_raw, df_index, tf_weights, **get_norm_config('pressure_rejection_strength_D', default_ascending=True))
        lower_shadow_absorption_strength_score = self.get_dynamic_normalized_score(
            lower_shadow_absorption_strength_raw, df_index, tf_weights, **get_norm_config('lower_shadow_absorption_strength_D', default_ascending=True))
        defense_solidity_score = self.get_dynamic_normalized_score(
            defense_solidity_score_raw, df_index, tf_weights, **get_norm_config('defense_solidity_score_D', default_ascending=True))
        opening_gap_defense_strength_score = self.get_dynamic_normalized_score(
            opening_gap_defense_strength_raw, df_index, tf_weights, **get_norm_config('opening_gap_defense_strength_D', default_ascending=True))
        structural_support_strength_score = (
            support_validation_strength_score * structural_support_strength_weights.get('support_validation_strength', 0.3) +
            pressure_rejection_strength_score * structural_support_strength_weights.get('pressure_rejection_strength', 0.3) +
            lower_shadow_absorption_strength_score * structural_support_strength_weights.get('lower_shadow_absorption_strength', 0.2) +
            defense_solidity_score * structural_support_strength_weights.get('defense_solidity_score', 0.1) +
            opening_gap_defense_strength_score * structural_support_strength_weights.get('opening_gap_defense_strength', 0.1)
        ).clip(0, 1)
        # --- 2. 结构形态坚固性 (Structural Form Solidity) ---
        equilibrium_compression_raw = self._get_safe_series(df, signal_mapping.get('equilibrium_compression_index_D', 'equilibrium_compression_index_D'), 0.0, method_name=method_name)
        # platform_conviction_score_D 和 goodness_of_fit_score_D 现在由 _get_safe_series 处理，如果缺失则返回0.0
        platform_conviction_score_raw = self._get_safe_series(df, signal_mapping.get('platform_conviction_score_D', 'platform_conviction_score_D'), 0.0, method_name=method_name)
        value_area_overlap_raw = self._get_safe_series(df, signal_mapping.get('value_area_overlap_pct_D', 'value_area_overlap_pct_D'), 0.0, method_name=method_name)
        goodness_of_fit_score_raw = self._get_safe_series(df, signal_mapping.get('goodness_of_fit_score_D', 'goodness_of_fit_score_D'), 0.0, method_name=method_name)
        structural_node_count_raw = self._get_safe_series(df, signal_mapping.get('structural_node_count_D', 'structural_node_count_D'), 0.0, method_name=method_name)
        equilibrium_compression_score = self.get_dynamic_normalized_score(
            equilibrium_compression_raw, df_index, tf_weights, **get_norm_config('equilibrium_compression_index_D', default_ascending=True))
        platform_conviction_score = self.get_dynamic_normalized_score(
            platform_conviction_score_raw, df_index, tf_weights, **get_norm_config('platform_conviction_score_D', default_ascending=True))
        value_area_overlap_score = self.get_dynamic_normalized_score(
            value_area_overlap_raw, df_index, tf_weights, **get_norm_config('value_area_overlap_pct_D', default_ascending=True))
        goodness_of_fit_score = self.get_dynamic_normalized_score(
            goodness_of_fit_score_raw, df_index, tf_weights, **get_norm_config('goodness_of_fit_score_D', default_ascending=True))
        structural_node_count_score = self.get_dynamic_normalized_score(
            structural_node_count_raw, df_index, tf_weights, **get_norm_config('structural_node_count_D', default_ascending=True))
        structural_form_solidity_score = (
            equilibrium_compression_score * structural_form_solidity_weights.get('equilibrium_compression_index', 0.3) +
            platform_conviction_score * structural_form_solidity_weights.get('platform_conviction_score', 0.25) +
            value_area_overlap_score * structural_form_solidity_weights.get('value_area_overlap_pct', 0.2) +
            goodness_of_fit_score * structural_form_solidity_weights.get('goodness_of_fit_score', 0.15) +
            structural_node_count_score * structural_form_solidity_weights.get('structural_node_count', 0.1)
        ).clip(0, 1)
        # --- 3. 波动率秩序性 (Volatility Orderliness) ---
        bbw_raw = self._get_safe_series(df, signal_mapping.get('BBW_21_2.0_D', 'BBW_21_2.0_D'), 1.0, method_name=method_name)
        volatility_instability_raw = self._get_safe_series(df, signal_mapping.get('VOLATILITY_INSTABILITY_INDEX_21d_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D'), 1.0, method_name=method_name)
        fractal_dimension_raw = self._get_safe_series(df, signal_mapping.get('FRACTAL_DIMENSION_89d_D', 'FRACTAL_DIMENSION_89d_D'), 1.5, method_name=method_name)
        sample_entropy_raw = self._get_safe_series(df, signal_mapping.get('SAMPLE_ENTROPY_13d_D', 'SAMPLE_ENTROPY_13d_D'), 1.0, method_name=method_name)
        hurst_raw = self._get_safe_series(df, signal_mapping.get('HURST_144d_D', 'HURST_144d_D'), 0.5, method_name=method_name)
        bbw_norm_config = get_norm_config('BBW_21_2.0_D', default_ascending=False)
        bbw_score = self.get_dynamic_normalized_score(bbw_raw, df_index, tf_weights, **bbw_norm_config)
        volatility_instability_norm_config = get_norm_config('VOLATILITY_INSTABILITY_INDEX_21d_D', default_ascending=False)
        volatility_instability_score = self.get_dynamic_normalized_score(volatility_instability_raw, df_index, tf_weights, **volatility_instability_norm_config)
        fd_norm_config = get_norm_config('FRACTAL_DIMENSION_89d_D', default_ascending=False, default_clip_range=[1.0, 2.0])
        fractal_dimension_score = self.get_dynamic_normalized_score(fractal_dimension_raw, df_index, tf_weights, **fd_norm_config)
        sample_entropy_norm_config = get_norm_config('SAMPLE_ENTROPY_13d_D', default_ascending=False)
        sample_entropy_score = self.get_dynamic_normalized_score(sample_entropy_raw, df_index, tf_weights, **sample_entropy_norm_config)
        hurst_norm_config = get_norm_config('HURST_144d_D', default_ascending=True, default_clip_range=[0.0, 1.0])
        hurst_score = self.get_dynamic_normalized_score(hurst_raw, df_index, tf_weights, **hurst_norm_config)
        volatility_orderliness_score = (
            bbw_score * volatility_orderliness_weights.get('BBW_21_2.0', 0.3) +
            volatility_instability_score * volatility_orderliness_weights.get('VOLATILITY_INSTABILITY_INDEX_21d', 0.3) +
            fractal_dimension_score * volatility_orderliness_weights.get('FRACTAL_DIMENSION_89d', 0.2) +
            sample_entropy_score * volatility_orderliness_weights.get('SAMPLE_ENTROPY_13d', 0.1) +
            hurst_score * volatility_orderliness_weights.get('HURST_144d', 0.1)
        ).clip(0, 1)
        # --- 4. 结构运动效率 (Structural Movement Efficiency) ---
        trend_efficiency_ratio_raw = self._get_safe_series(df, signal_mapping.get('trend_efficiency_ratio_D', 'trend_efficiency_ratio_D'), 0.0, method_name=method_name)
        asymmetric_friction_index_raw = self._get_safe_series(df, signal_mapping.get('asymmetric_friction_index_D', 'asymmetric_friction_index_D'), 0.0, method_name=method_name)
        impulse_quality_ratio_raw = self._get_safe_series(df, signal_mapping.get('impulse_quality_ratio_D', 'impulse_quality_ratio_D'), 0.0, method_name=method_name)
        ma_potential_orderliness_score_raw = self._get_safe_series(df, signal_mapping.get('MA_POTENTIAL_ORDERLINESS_SCORE_D', 'MA_POTENTIAL_ORDERLINESS_SCORE_D'), 0.0, method_name=method_name)
        trend_efficiency_ratio_norm_config = get_norm_config('trend_efficiency_ratio_D', default_ascending=True)
        trend_efficiency_ratio_score = self.get_dynamic_normalized_score(trend_efficiency_ratio_raw, df_index, tf_weights, **trend_efficiency_ratio_norm_config)
        afi_norm_config = get_norm_config('asymmetric_friction_index_D', default_ascending=False)
        asymmetric_friction_index_score = self.get_dynamic_normalized_score(asymmetric_friction_index_raw, df_index, tf_weights, **afi_norm_config)
        impulse_quality_ratio_norm_config = get_norm_config('impulse_quality_ratio_D', default_ascending=True)
        impulse_quality_ratio_score = self.get_dynamic_normalized_score(impulse_quality_ratio_raw, df_index, tf_weights, **impulse_quality_ratio_norm_config)
        ma_potential_orderliness_score_norm_config = get_norm_config('MA_POTENTIAL_ORDERLINESS_SCORE_D', default_ascending=True)
        ma_potential_orderliness_score = self.get_dynamic_normalized_score(ma_potential_orderliness_score_raw, df_index, tf_weights, **ma_potential_orderliness_score_norm_config)
        structural_movement_efficiency_score = (
            trend_efficiency_ratio_score * structural_movement_efficiency_weights.get('trend_efficiency_ratio', 0.3) +
            asymmetric_friction_index_score * structural_movement_efficiency_weights.get('asymmetric_friction_index', 0.3) +
            impulse_quality_ratio_score * structural_movement_efficiency_weights.get('impulse_quality_ratio', 0.2) +
            ma_potential_orderliness_score * structural_movement_efficiency_weights.get('MA_POTENTIAL_ORDERLINESS_SCORE', 0.2)
        ).clip(0, 1)
        # --- 5. 结构突破强度 (Structural Break Strength) - 使用代理信号 ---
        breakout_volume_ratio_raw = self._get_safe_series(df, signal_mapping.get('breakout_volume_ratio_D', 'volume_burstiness_index_D'), 0.0, method_name=method_name)
        breakout_range_expansion_raw = self._get_safe_series(df, signal_mapping.get('breakout_range_expansion_D', 'volatility_expansion_ratio_D'), 0.0, method_name=method_name)
        breakout_retest_success_raw = self._get_safe_series(df, signal_mapping.get('breakout_retest_success_D', 'breakout_quality_score_D'), 0.0, method_name=method_name)
        breakout_duration_raw = self._get_safe_series(df, signal_mapping.get('breakout_duration_D', 'duration_D'), 0.0, method_name=method_name) # 映射到 duration_D
        breakout_volume_ratio_score = self.get_dynamic_normalized_score(breakout_volume_ratio_raw, df_index, tf_weights, **get_norm_config('breakout_volume_ratio_D', default_ascending=True))
        breakout_range_expansion_score = self.get_dynamic_normalized_score(breakout_range_expansion_raw, df_index, tf_weights, **get_norm_config('breakout_range_expansion_D', default_ascending=True))
        breakout_retest_success_score = self.get_dynamic_normalized_score(breakout_retest_success_raw, df_index, tf_weights, **get_norm_config('breakout_retest_success_D', default_ascending=True))
        breakout_duration_score = self.get_dynamic_normalized_score(breakout_duration_raw, df_index, tf_weights, **get_norm_config('breakout_duration_D', default_ascending=True))
        structural_break_strength_score = (
            breakout_volume_ratio_score * structural_break_strength_weights.get('breakout_volume_ratio', 0.3) +
            breakout_range_expansion_score * structural_break_strength_weights.get('breakout_range_expansion', 0.3) +
            breakout_retest_success_score * structural_break_strength_weights.get('breakout_retest_success', 0.2) +
            breakout_duration_score * structural_break_strength_weights.get('breakout_duration', 0.2)
        ).clip(0, 1)
        # --- 6. 结构回撤效率 (Structural Retracement Efficiency) - 使用代理信号 ---
        retracement_depth_pct_raw = self._get_safe_series(df, signal_mapping.get('retracement_depth_pct_D', 'pullback_depth_ratio_D'), 0.0, method_name=method_name)
        retracement_speed_ratio_raw = self._get_safe_series(df, signal_mapping.get('retracement_speed_ratio_D', 'price_reversion_velocity_D'), 0.0, method_name=method_name)
        retracement_volume_decay_raw = self._get_safe_series(df, signal_mapping.get('retracement_volume_decay_D', 'volume_burstiness_index_D'), 0.0, method_name=method_name) # 映射到 volume_burstiness_index_D
        retracement_MA_adherence_raw = self._get_safe_series(df, signal_mapping.get('retracement_MA_adherence_D', 'MA_POTENTIAL_ORDERLINESS_SCORE_D'), 0.0, method_name=method_name) # 映射到 MA_POTENTIAL_ORDERLINESS_SCORE_D
        # 回撤深度和速度通常是越小越好，对手盘枯竭和控盘坚实度是越大越好
        retracement_depth_pct_score = self.get_dynamic_normalized_score(retracement_depth_pct_raw, df_index, tf_weights, **get_norm_config('retracement_depth_pct_D', default_ascending=False))
        retracement_speed_ratio_score = self.get_dynamic_normalized_score(retracement_speed_ratio_raw, df_index, tf_weights, **get_norm_config('retracement_speed_ratio_D', default_ascending=False))
        retracement_volume_decay_score = self.get_dynamic_normalized_score(retracement_volume_decay_raw, df_index, tf_weights, **get_norm_config('retracement_volume_decay_D', default_ascending=False)) # 爆发性越低越好
        retracement_MA_adherence_score = self.get_dynamic_normalized_score(retracement_MA_adherence_raw, df_index, tf_weights, **get_norm_config('retracement_MA_adherence_D', default_ascending=False)) # 有序度越低越好
        structural_retracement_efficiency_score = (
            retracement_depth_pct_score * structural_retracement_efficiency_weights.get('retracement_depth_pct', 0.3) +
            retracement_speed_ratio_score * structural_retracement_efficiency_weights.get('retracement_speed_ratio', 0.3) +
            retracement_volume_decay_score * structural_retracement_efficiency_weights.get('retracement_volume_decay', 0.2) +
            retracement_MA_adherence_score * structural_retracement_efficiency_weights.get('retracement_MA_adherence', 0.2)
        ).clip(0, 1)
        # --- 7. 最终融合 ---
        stability_score = (
            structural_support_strength_score * stability_fusion_weights.get('structural_support_strength', 0.25) +
            structural_form_solidity_score * stability_fusion_weights.get('structural_form_solidity', 0.25) +
            volatility_orderliness_score * stability_fusion_weights.get('volatility_orderliness', 0.2) +
            structural_movement_efficiency_score * stability_fusion_weights.get('structural_movement_efficiency', 0.15) +
            structural_break_strength_score * stability_fusion_weights.get('structural_break_strength', 0.075) +
            structural_retracement_efficiency_score * stability_fusion_weights.get('structural_retracement_efficiency', 0.075)
        ).clip(0, 1)
        final_score = (stability_score * 2 - 1).astype(np.float32)
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 结构支撑强度分数: {structural_support_strength_score.iloc[-1]:.4f}")
            print(f"    -> 结构形态坚固性分数: {structural_form_solidity_score.iloc[-1]:.4f}")
            print(f"    -> 波动率秩序性分数: {volatility_orderliness_score.iloc[-1]:.4f}")
            print(f"    -> 结构运动效率分数: {structural_movement_efficiency_score.iloc[-1]:.4f}")
            print(f"    -> 突破量比原始值: {breakout_volume_ratio_raw.iloc[-1]:.4f}")
            print(f"    -> 突破量比分数: {breakout_volume_ratio_score.iloc[-1]:.4f}")
            print(f"    -> 突破区间扩张原始值: {breakout_range_expansion_raw.iloc[-1]:.4f}")
            print(f"    -> 突破区间扩张分数: {breakout_range_expansion_score.iloc[-1]:.4f}")
            print(f"    -> 突破回踩成功原始值: {breakout_retest_success_raw.iloc[-1]:.4f}")
            print(f"    -> 突破回踩成功分数: {breakout_retest_success_score.iloc[-1]:.4f}")
            print(f"    -> 突破持续时间原始值: {breakout_duration_raw.iloc[-1]:.4f}")
            print(f"    -> 突破持续时间分数: {breakout_duration_score.iloc[-1]:.4f}")
            print(f"    -> 结构突破强度分数: {structural_break_strength_score.iloc[-1]:.4f}")
            print(f"    -> 回撤深度原始值: {retracement_depth_pct_raw.iloc[-1]:.4f}")
            print(f"    -> 回撤深度分数: {retracement_depth_pct_score.iloc[-1]:.4f}")
            print(f"    -> 回撤速度原始值: {retracement_speed_ratio_raw.iloc[-1]:.4f}")
            print(f"    -> 回撤速度分数: {retracement_speed_ratio_score.iloc[-1]:.4f}")
            print(f"    -> 回撤量能衰减原始值: {retracement_volume_decay_raw.iloc[-1]:.4f}")
            print(f"    -> 回撤量能衰减分数: {retracement_volume_decay_score.iloc[-1]:.4f}")
            print(f"    -> 回撤均线粘附原始值: {retracement_MA_adherence_raw.iloc[-1]:.4f}")
            print(f"    -> 回撤均线粘附分数: {retracement_MA_adherence_score.iloc[-1]:.4f}")
            print(f"    -> 结构回撤效率分数: {structural_retracement_efficiency_score.iloc[-1]:.4f}")
            print(f"    -> 最终稳定性分数: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_axiom_mtf_cohesion(self, df: pd.DataFrame, daily_trend_form_score: pd.Series) -> pd.Series:
        """
        【V2.7 · 自适应通道风险版】结构公理二：诊断“宏观趋势健康度”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 使用布林带百分比(BBP)替代BIAS作为核心风险标尺，解决了BIAS静态、不自适应的根本缺陷。模型现在能更好地区分“健康的趋势”与“高风险的极端行情”。
        - 核心修复: 修复了逻辑上的不对称性。模型现在能同时识别上升趋势中的“过热”风险和下降趋势中的“超跌”状态（趋势衰竭信号），并对两者进行对称的降权处理。
        - 核心融合: 继续采用“和谐度”模型，将经过风险调整后的“宏观健康度分”与“微观意图分”进行加权融合。
        """
        method_name = "_diagnose_axiom_mtf_cohesion"
        short_periods = [5, 13, 21]
        long_periods = [55, 89, 144]
        required_signals = [
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'close_D', 'BBP_21_2.0_D' # 替换 BIAS_55_D
        ]
        required_signals.extend([f'EMA_{p}_D' for p in short_periods])
        required_signals.extend([f'EMA_{p}_D' for p in long_periods])
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_struct = self.structural_ultimate_params
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 宏观趋势健康度 (Macro Trend Health) ---
        # 1a. 伪多周期排列分 (Pseudo-MTF Alignment)
        fastest_short_ma = self._get_safe_series(df, f'EMA_{min(short_periods)}_D', method_name=method_name)
        slowest_long_ma = self._get_safe_series(df, f'EMA_{max(long_periods)}_D', method_name=method_name)
        alignment_score = (fastest_short_ma > slowest_long_ma).astype(float)
        # 1b. 自适应风险感知：基于布林带的 过热惩罚 与 超跌缓和
        bbp_raw = self._get_safe_series(df, 'BBP_21_2.0_D', 0.5, method_name=method_name)
        # 过热惩罚：当价格进入布林带上轨的最后5%区间(BBP>0.95)时开始惩罚，突破上轨越多惩罚越大
        overheat_penalty = ((bbp_raw - 0.95).clip(lower=0) / 0.2).clip(upper=1.0) # 在BBP=1.15时惩罚达到最大
        # 超跌缓和：当价格进入布林带下轨的最初5%区间(BBP<0.05)时开始缓和，突破下轨越多缓和越大
        oversold_mitigation = (((0.05 - bbp_raw)).clip(lower=0) / 0.2).clip(upper=1.0) # 在BBP=-0.15时缓和达到最大
        # 1c. 风险调整后的宏观分
        bullish_macro_health = alignment_score * (1 - overheat_penalty)
        bearish_macro_health = (1 - alignment_score) * (1 - oversold_mitigation)
        # --- 2. 微观意图 (Micro Intent) ---
        ofi_raw = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name=method_name)
        buy_sweep_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name=method_name)
        sell_sweep_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', 0.0, method_name=method_name)
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
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 伪多周期排列分数: {alignment_score.iloc[-1]:.4f}")
            print(f"    -> BBP原始值: {bbp_raw.iloc[-1]:.4f}")
            print(f"    -> 过热惩罚: {overheat_penalty.iloc[-1]:.4f}")
            print(f"    -> 超跌缓和: {oversold_mitigation.iloc[-1]:.4f}")
            print(f"    -> 看涨宏观健康度: {bullish_macro_health.iloc[-1]:.4f}")
            print(f"    -> 看跌宏观健康度: {bearish_macro_health.iloc[-1]:.4f}")
            print(f"    -> OFI分数: {ofi_score.iloc[-1]:.4f}")
            print(f"    -> 买盘扫盘分数: {buy_sweep_score.iloc[-1]:.4f}")
            print(f"    -> 卖盘扫盘分数: {sell_sweep_score.iloc[-1]:.4f}")
            print(f"    -> 微观意图分数: {micro_intent_score.iloc[-1]:.4f}")
            print(f"    -> 最终多周期协同分数: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_bottom_fractal(self, df: pd.DataFrame, n: int = 5, min_depth_ratio: float = 0.001) -> pd.Series:
        """
        【V1.2 · 探针植入版】结构公理五：诊断“底分型”结构
        - 核心逻辑: 识别底分型结构形态，并输出一个双极性分数 `SCORE_STRUCT_BOTTOM_FRACTAL`。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        method_name = "_diagnose_bottom_fractal"
        required_signals = ['low_D']
        if not self._validate_required_signals(df, required_signals, method_name):
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
        low_series = self._get_safe_series(df, 'low_D', method_name=method_name)
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
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 当前低点 (low_D): {low_series.iloc[-1]:.4f}")
            print(f"    -> 最终底分型分数: {bottom_fractal_score.iloc[-1]:.4f}")
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
        method_name = "_diagnose_strategic_posture"
        required_signals = ['structural_leverage_D']
        if not self._validate_required_signals(self.strategy.df_indicators, required_signals, method_name):
            return pd.Series(0.0, index=axiom_trend_form.index), pd.Series(0.5, index=axiom_trend_form.index)
        df_index = axiom_trend_form.index
        p_conf_struct = self.structural_ultimate_params
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('long_term_stability', {13: 0.2, 21: 0.3, 55: 0.4, 89: 0.1})
        leverage_raw = self._get_safe_series(self.strategy.df_indicators, 'structural_leverage_D', 0.0, method_name=method_name)
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
        if self.is_probe_date and not df_index.empty:
            current_date = df_index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 结构杠杆原始值: {leverage_raw.iloc[-1]:.4f}")
            print(f"    -> 结构杠杆分数: {leverage_score.iloc[-1]:.4f}")
            print(f"    -> 基础进攻分数: {base_offense_score.iloc[-1]:.4f}")
            print(f"    -> 张力催化因子: {tension_amplifier.iloc[-1]:.4f}")
            print(f"    -> 进攻分数: {offense_score.iloc[-1]:.4f}")
            print(f"    -> 动态防御分数: {dynamic_defense.iloc[-1]:.4f}")
            print(f"    -> 静态防御分数 (Max(平台, 突破)): {static_defense.iloc[-1]:.4f}")
            print(f"    -> 最终防御强度: {defense_strength.iloc[-1]:.4f}")
            print(f"    -> 最终战略态势分数: {final_score.iloc[-1]:.4f}")
        return final_score, defense_strength

    def _diagnose_axiom_tension(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 势能压缩版】结构公理六：诊断“结构张力”
        - 核心逻辑: 量化系统内部能量的压缩程度，作为潜在状态突变的先行指标。
        - 核心维度: 融合价格空间压缩(BBW)、均线结构压缩(EMA标准差)和量能压缩(成交量均线比)。
        """
        method_name = "_diagnose_axiom_tension"
        ema_periods = [5, 13, 21, 34]
        required_signals = ['BBW_21_2.0_D', 'VOL_MA_5_D', 'VOL_MA_55_D', 'MA_POTENTIAL_TENSION_INDEX_D'] # 新增
        required_signals.extend([f'EMA_{p}_D' for p in ema_periods])
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_struct = self.structural_ultimate_params
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 价格空间压缩 ---
        price_compression_raw = self._get_safe_series(df, 'BBW_21_2.0_D', 1.0, method_name=method_name)
        price_compression_score = get_adaptive_mtf_normalized_score(price_compression_raw, df_index, tf_weights, ascending=False)
        # --- 2. 均线结构压缩 ---
        ema_cluster = df[[f'EMA_{p}_D' for p in ema_periods]]
        structure_compression_raw_std = ema_cluster.std(axis=1) / df['close_D'] # 标准差归一化，避免股价本身大小的影响
        structure_compression_score_std = get_adaptive_mtf_normalized_score(structure_compression_raw_std, df_index, tf_weights, ascending=False)
        # 引入 MA_POTENTIAL_TENSION_INDEX_D
        ma_tension_raw = self._get_safe_series(df, 'MA_POTENTIAL_TENSION_INDEX_D', 0.0, method_name=method_name)
        ma_tension_score = get_adaptive_mtf_normalized_score(ma_tension_raw, df_index, tf_weights, ascending=True) # 张力越大越好 # 修正此处
        structure_compression_score = (structure_compression_score_std * 0.6 + ma_tension_score * 0.4).clip(0,1)
        # --- 3. 量能压缩 ---
        vol_ma_short = self._get_safe_series(df, 'VOL_MA_5_D', 1.0, method_name=method_name)
        vol_ma_long = self._get_safe_series(df, 'VOL_MA_55_D', 1.0, method_name=method_name)
        volume_compression_raw = vol_ma_short / vol_ma_long
        volume_compression_score = get_adaptive_mtf_normalized_score(volume_compression_raw, df_index, tf_weights, ascending=False)
        # --- 4. 融合 ---
        # 权重: 价格(0.4), 结构(0.4), 量能(0.2)
        tension_score = (
            price_compression_score * 0.4 +
            structure_compression_score * 0.4 +
            volume_compression_score * 0.2
        ).clip(0, 1)
        final_score = tension_score.astype(np.float32)
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 价格空间压缩原始值 (BBW_21_2.0_D): {price_compression_raw.iloc[-1]:.4f}")
            print(f"    -> 价格空间压缩分数: {price_compression_score.iloc[-1]:.4f}")
            print(f"    -> 均线结构压缩原始值 (Std): {structure_compression_raw_std.iloc[-1]:.4f}")
            print(f"    -> 均线结构压缩分数 (Std): {structure_compression_score_std.iloc[-1]:.4f}")
            print(f"    -> 均线势能张力原始值 (MA_POTENTIAL_TENSION_INDEX_D): {ma_tension_raw.iloc[-1]:.4f}")
            print(f"    -> 均线势能张力分数: {ma_tension_score.iloc[-1]:.4f}")
            print(f"    -> 融合均线结构压缩分数: {structure_compression_score.iloc[-1]:.4f}")
            print(f"    -> 量能压缩原始值 (VOL_MA_5_D / VOL_MA_55_D): {volume_compression_raw.iloc[-1]:.4f}")
            print(f"    -> 量能压缩分数: {volume_compression_score.iloc[-1]:.4f}")
            print(f"    -> 最终结构张力分数: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_playbook_secondary_launch(self, df: pd.DataFrame, axiom_stability: pd.Series, strategic_posture: pd.Series, structural_momentum: pd.Series) -> pd.Series:
        """
        【V1.0 · 战术剧本识别】识别“暴力洗盘后二次启动”剧本
        - 核心逻辑: 在时间序列上匹配一个完整的战术行为模式。
        - 剧本序列: [前期稳定蓄势] -> [短暂暴力洗盘+主力吸筹] -> [当日强势启动]
        """
        method_name = "_diagnose_playbook_secondary_launch"
        required_signals = ['capitulation_absorption_index_D', 'close_D']
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        playbook_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        absorption_signal = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name=method_name)
        # 为了效率，我们只在最近的K天内寻找模式
        lookback_days = 60
        start_index = max(10, len(df) - lookback_days) # 至少需要10天历史数据
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 当前战略态势: {strategic_posture.iloc[-1]:.4f}")
            print(f"    -> 当前结构动量: {structural_momentum.iloc[-1]:.4f}")
        for i in range(start_index, len(df)):
            # --- 条件3: 当日强势启动 ---
            is_launch_day = strategic_posture.iloc[i] > 0.6 and structural_momentum.iloc[i] > 0.6
            if not is_launch_day:
                if self.is_probe_date and i == len(df) - 1:
                    print(f"    -> 当日非强势启动 (战略态势={strategic_posture.iloc[i]:.4f}, 结构动量={structural_momentum.iloc[i]:.4f})")
                continue
            # --- 回溯寻找洗盘和蓄势阶段 ---
            washout_found = False
            # 洗盘窗口: 启动日前1-3天
            for j in range(max(0, i - 3), i):
                # 确保 j-1 索引有效
                if j - 1 < 0:
                    continue
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
                            if self.is_probe_date and i == len(df) - 1:
                                print(f"    -> 发现洗盘日 {df.index[j].strftime('%Y-%m-%d')} (价格下跌={price_dropped}, 吸收强度={absorption_signal.iloc[j]:.4f})")
                                print(f"    -> 前期蓄势稳定性 ({df.index[accumulation_period_start].strftime('%Y-%m-%d')}~{df.index[accumulation_period_end].strftime('%Y-%m-%d')}): {avg_stability:.4f}")
                            break # 找到符合条件的洗盘日，即可停止内层循环
            if washout_found:
                playbook_score.iloc[i] = 1.0
                if self.is_probe_date and i == len(df) - 1:
                    print(f"    -> 剧本 '暴力洗盘后二次启动' 激活！")
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"    -> 最终剧本分数: {playbook_score.iloc[-1]:.4f}")
        return playbook_score

    def _diagnose_axiom_environment(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 审时度势版】结构公理七：诊断“战场环境”
        - 核心逻辑: 评估个股所处的外部宏观环境，融合板块强度与主题热度。
        """
        method_name = "_diagnose_axiom_environment"
        required_signals = ['industry_strength_rank_D', 'THEME_HOTNESS_SCORE_D']
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.5, index=df.index) # 环境未知时返回中性分
        df_index = df.index
        p_conf_struct = self.structural_ultimate_params
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 地利 (Sector Strength) ---
        sector_rank_raw = self._get_safe_series(df, 'industry_strength_rank_D', 0.5, method_name=method_name)
        # 排名越小越好，因此 ascending=False
        sector_strength_score = get_adaptive_mtf_normalized_score(sector_rank_raw, df_index, ascending=False, tf_weights=tf_weights)
        # --- 2. 人和 (Thematic Resonance) ---
        theme_hotness_raw = self._get_safe_series(df, 'THEME_HOTNESS_SCORE_D', 0.5, method_name=method_name)
        theme_hotness_score = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=tf_weights)
        # --- 3. 融合 ---
        # 权重: 板块(0.6), 主题(0.4)
        environment_score = (
            sector_strength_score * 0.6 +
            theme_hotness_score * 0.4
        ).clip(0, 1)
        final_score = environment_score.astype(np.float32)
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 行业强度排名原始值: {sector_rank_raw.iloc[-1]:.4f}")
            print(f"    -> 板块强度分数: {sector_strength_score.iloc[-1]:.4f}")
            print(f"    -> 主题热度原始值: {theme_hotness_raw.iloc[-1]:.4f}")
            print(f"    -> 主题热度分数: {theme_hotness_score.iloc[-1]:.4f}")
            print(f"    -> 最终环境分数: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_leadership_potential(self, strategic_posture: pd.Series, axiom_environment: pd.Series, structural_momentum: pd.Series, axiom_tension: pd.Series) -> pd.Series:
        """
        【V1.0 · 逆势王者版】裁决“龙头潜力”
        - 核心逻辑: 在“个体强，环境弱”的特定情境下，通过寻找额外证据（动能、张力），
                      来判断标的是“真龙头”还是“补跌陷阱”。
        """
        method_name = "_diagnose_leadership_potential"
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
        if self.is_probe_date and not strategic_posture.empty:
            current_date = strategic_posture.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 战略态势: {strategic_posture.iloc[-1]:.4f} (阈值 > {posture_threshold})")
            print(f"    -> 环境分数: {axiom_environment.iloc[-1]:.4f} (阈值 < {env_threshold})")
            print(f"    -> 是否处于冲突区: {is_conflict_zone.iloc[-1]}")
            print(f"    -> 结构动量: {structural_momentum.iloc[-1]:.4f}")
            print(f"    -> 结构张力: {axiom_tension.iloc[-1]:.4f}")
            print(f"    -> 领导力证据分数 (动量*0.6 + 张力*0.4): {leadership_evidence_score.iloc[-1]:.4f}")
            print(f"    -> 最终龙头潜力分数: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_platform_foundation(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        【V3.0 · 法医鉴定版】对平台进行法医级鉴定并勘探其战场边界
        - 核心逻辑: 从“结构形态”、“筹码状态”、“主力行为”、“市场情绪”四大维度，
                      对平台进行全方位品质鉴定，并输出基于主力意图的动态边界。
        - 输出: (品质分, 动态高点, 动态低点, VPOC)
        """
        method_name = "_diagnose_platform_foundation"
        required_signals = [
            'BBW_21_2.0_D', 'VOL_MA_5_D', 'VOL_MA_55_D', 'high_D', 'low_D', 'close_D', 'volume_D', 'open_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'price_volume_entropy_D', # 结构形态
            'dominant_peak_solidity_D', 'peak_separation_ratio_D', 'chip_fatigue_index_D', # 筹码状态
            'main_force_vpoc_D', 'mf_cost_zone_defense_intent_D', 'control_solidity_index_D', # 主力行为
            'counterparty_exhaustion_index_D', 'retail_panic_surrender_index_D', 'turnover_rate_f_D' # 市场情绪
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            nan_series = pd.Series(np.nan, index=df.index)
            return pd.Series(0.0, index=df.index), nan_series, nan_series, nan_series
        # 获取 tf_weights
        p_conf_struct = self.structural_ultimate_params
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 步骤一: 识别平台状态 ---
        # 传递 tf_weights 参数
        bbw_raw = self._get_safe_series(df, 'BBW_21_2.0_D', 1.0, method_name=method_name)
        vol_ma_5_raw = self._get_safe_series(df, 'VOL_MA_5_D', 1.0, method_name=method_name)
        vol_ma_55_raw = self._get_safe_series(df, 'VOL_MA_55_D', 1.0, method_name=method_name)
        price_stability_score = get_adaptive_mtf_normalized_score(bbw_raw, df.index, tf_weights, ascending=False)
        supply_exhaustion_score = get_adaptive_mtf_normalized_score(vol_ma_5_raw / vol_ma_55_raw, df.index, tf_weights, ascending=False)
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
        # 预计算所有维度的分数
        # 结构形态品质
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 1.0, method_name=method_name)
        price_volume_entropy_raw = self._get_safe_series(df, 'price_volume_entropy_D', 0.0, method_name=method_name)
        s_structure = (
            get_adaptive_mtf_normalized_score(volatility_instability_raw, df.index, tf_weights, ascending=False) * 0.5 +
            get_adaptive_mtf_normalized_score(price_volume_entropy_raw, df.index, tf_weights, ascending=False) * 0.5
        )
        # 筹码状态品质
        dominant_peak_solidity_raw = self._get_safe_series(df, 'dominant_peak_solidity_D', 0.0, method_name=method_name)
        peak_separation_ratio_raw = self._get_safe_series(df, 'peak_separation_ratio_D', 0.0, method_name=method_name)
        chip_fatigue_index_raw = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name=method_name)
        s_chips = (
            get_adaptive_mtf_normalized_score(dominant_peak_solidity_raw, df.index, tf_weights, ascending=True) * 0.5 +
            get_adaptive_mtf_normalized_score(peak_separation_ratio_raw, df.index, tf_weights, ascending=True) * 0.3 +
            get_adaptive_mtf_normalized_score(chip_fatigue_index_raw, df.index, tf_weights, ascending=False) * 0.2 # 疲劳指数低代表筹码稳定
        )
        # 主力行为品质
        mf_cost_zone_defense_intent_raw = self._get_safe_series(df, 'mf_cost_zone_defense_intent_D', 0.0, method_name=method_name)
        control_solidity_index_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name=method_name)
        s_main_force = (
            get_adaptive_mtf_normalized_score(mf_cost_zone_defense_intent_raw, df.index, tf_weights, ascending=True) * 0.5 +
            get_adaptive_mtf_normalized_score(control_solidity_index_raw, df.index, tf_weights, ascending=True) * 0.5
        )
        # 市场情绪品质
        counterparty_exhaustion_index_raw = self._get_safe_series(df, 'counterparty_exhaustion_index_D', 0.0, method_name=method_name)
        retail_panic_surrender_index_raw = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        turnover_rate_f_raw = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name=method_name)
        s_sentiment = (
            get_adaptive_mtf_normalized_score(counterparty_exhaustion_index_raw, df.index, tf_weights, ascending=True) * 0.5 +
            get_adaptive_mtf_normalized_score(retail_panic_surrender_index_raw, df.index, tf_weights, ascending=True) * 0.3 +
            get_adaptive_mtf_normalized_score(turnover_rate_f_raw, df.index, tf_weights, ascending=False) * 0.2
        )
        # 最终品质分权重: 主力(0.4), 筹码(0.3), 情绪(0.2), 形态(0.1)
        final_quality_score = (s_main_force * 0.4 + s_chips * 0.3 + s_sentiment * 0.2 + s_structure * 0.1).clip(0, 1)
        for group_id in platform_group[is_valid_platform_day].unique():
            platform_indices = platform_group[platform_group == group_id].index
            platform_df = df.loc[platform_indices]
            # 使用最可靠的信号定义边界
            current_vpoc = self._get_safe_series(platform_df, 'main_force_vpoc_D', np.nan, method_name=method_name).iloc[-1]
            # 简化的边界，未来可引入更复杂的试探性K线逻辑
            platform_range = (self._get_safe_series(platform_df, 'high_D', np.nan, method_name=method_name).max() - self._get_safe_series(platform_df, 'low_D', np.nan, method_name=method_name).min())
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
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 价格稳定性原始值 (BBW_21_2.0_D): {bbw_raw.iloc[-1]:.4f}")
            print(f"    -> 价格稳定性分数: {price_stability_score.iloc[-1]:.4f}")
            print(f"    -> 供应枯竭原始值 (VOL_MA_5_D / VOL_MA_55_D): {(vol_ma_5_raw / vol_ma_55_raw).iloc[-1]:.4f}")
            print(f"    -> 供应枯竭分数: {supply_exhaustion_score.iloc[-1]:.4f}")
            print(f"    -> 是否处于有效平台状态: {is_valid_platform_day.iloc[-1]}")
            print(f"    -> 结构形态原始值 (VOLATILITY_INSTABILITY_INDEX_21d_D): {volatility_instability_raw.iloc[-1]:.4f}")
            print(f"    -> 结构形态原始值 (price_volume_entropy_D): {price_volume_entropy_raw.iloc[-1]:.4f}")
            print(f"    -> 结构品质分: {s_structure.iloc[-1]:.4f}")
            print(f"    -> 筹码状态原始值 (dominant_peak_solidity_D): {dominant_peak_solidity_raw.iloc[-1]:.4f}")
            print(f"    -> 筹码状态原始值 (peak_separation_ratio_D): {peak_separation_ratio_raw.iloc[-1]:.4f}")
            print(f"    -> 筹码状态原始值 (chip_fatigue_index_D): {chip_fatigue_index_raw.iloc[-1]:.4f}")
            print(f"    -> 筹码品质分: {s_chips.iloc[-1]:.4f}")
            print(f"    -> 主力行为原始值 (mf_cost_zone_defense_intent_D): {mf_cost_zone_defense_intent_raw.iloc[-1]:.4f}")
            print(f"    -> 主力行为原始值 (control_solidity_index_D): {control_solidity_index_raw.iloc[-1]:.4f}")
            print(f"    -> 主力品质分: {s_main_force.iloc[-1]:.4f}")
            print(f"    -> 市场情绪原始值 (counterparty_exhaustion_index_D): {counterparty_exhaustion_index_raw.iloc[-1]:.4f}")
            print(f"    -> 市场情绪原始值 (retail_panic_surrender_index_D): {retail_panic_surrender_index_raw.iloc[-1]:.4f}")
            print(f"    -> 市场情绪原始值 (turnover_rate_f_D): {turnover_rate_f_raw.iloc[-1]:.4f}")
            print(f"    -> 情绪品质分: {s_sentiment.iloc[-1]:.4f}")
            print(f"    -> 最终平台品质分: {platform_quality.iloc[-1]:.4f}")
            print(f"    -> 动态高点: {dynamic_high.iloc[-1]:.4f}")
            print(f"    -> 动态低点: {dynamic_low.iloc[-1]:.4f}")
            print(f"    -> VPOC: {vpoc.iloc[-1]:.4f}")
        return platform_quality, dynamic_high, dynamic_low, vpoc

    def _diagnose_final_judgment(self, contextual_posture: pd.Series, defense_strength: pd.Series, structural_momentum: pd.Series) -> pd.Series:
        """
        【V1.0 · 总司令版】执行终极裁决
        - 核心逻辑: 识别并否决高风险的“力竭滞涨陷阱”模式。
        - 否决模式: 高态势分 + 弱防御 + 低动量
        """
        method_name = "_diagnose_final_judgment"
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
        if self.is_probe_date and not contextual_posture.empty:
            current_date = contextual_posture.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 情境态势: {contextual_posture.iloc[-1]:.4f}")
            print(f"    -> 防御强度: {defense_strength.iloc[-1]:.4f}")
            print(f"    -> 结构动量: {structural_momentum.iloc[-1]:.4f}")
            print(f"    -> 是否为陷阱候选 (情境态势 > 0.6): {is_trap_candidate.iloc[-1]}")
            print(f"    -> 防御是否脆弱 (防御强度 < 0.4): {is_defense_weak.iloc[-1]}")
            print(f"    -> 动量是否停滞 (结构动量 < 0.1): {is_momentum_stalled.iloc[-1]}")
            print(f"    -> 是否触发否决 (陷阱候选 & 脆弱防御 & 停滞动量): {is_veto_triggered.iloc[-1]}")
            print(f"    -> 防御脆弱度: {defense_weakness.iloc[-1]:.4f}")
            print(f"    -> 动量停滞度: {momentum_weakness.iloc[-1]:.4f}")
            print(f"    -> 最终惩罚: {final_penalty.iloc[-1]:.4f}")
            print(f"    -> 最终裁决分数: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_breakout_readiness(self, df: pd.DataFrame, axiom_tension: pd.Series) -> pd.Series:
        """
        【V2.0 · 无条件监理版】诊断“突破准备度”
        - 核心升级: 废除对`is_consolidating_D`的依赖，使其成为一个无条件的、连续性的质量评估信号。
        - 评估维度: 供应枯竭度 + 主力控盘度 + 势能积蓄度
        """
        method_name = "_diagnose_breakout_readiness"
        required_signals = [
            'counterparty_exhaustion_index_D', 'turnover_rate_f_D',
            'control_solidity_index_D', 'mf_cost_zone_defense_intent_D',
            'equilibrium_compression_index_D' # 新增
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        # 获取 tf_weights
        p_conf_struct = self.structural_ultimate_params
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights = mtf_weights_conf.get('default', {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 1. 评估三大维度 (无条件执行) ---
        # 1a. 供应枯竭度
        counterparty_exhaustion_raw = self._get_safe_series(df, 'counterparty_exhaustion_index_D', 0.0, method_name=method_name)
        turnover_rate_f_raw = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name=method_name)
        supply_exhaustion_score = (
            get_adaptive_mtf_normalized_score(counterparty_exhaustion_raw, df.index, tf_weights, ascending=True) * 0.7 +
            get_adaptive_mtf_normalized_score(turnover_rate_f_raw, df.index, tf_weights, ascending=False) * 0.3
        )
        # 1b. 主力控盘度
        control_solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name=method_name)
        mf_cost_zone_defense_intent_raw = self._get_safe_series(df, 'mf_cost_zone_defense_intent_D', 0.0, method_name=method_name)
        main_force_control_score = (
            get_adaptive_mtf_normalized_score(control_solidity_raw, df.index, tf_weights, ascending=True) * 0.5 +
            get_adaptive_mtf_normalized_score(mf_cost_zone_defense_intent_raw, df.index, tf_weights, ascending=True) * 0.5
        )
        # 1c. 势能积蓄度 (复用结构张力公理并引入均衡压缩)
        equilibrium_compression_raw = self._get_safe_series(df, 'equilibrium_compression_index_D', 0.0, method_name=method_name)
        equilibrium_compression_score = get_adaptive_mtf_normalized_score(equilibrium_compression_raw, df.index, tf_weights, ascending=True)
        energy_accumulation_score = (axiom_tension * 0.7 + equilibrium_compression_score * 0.3).clip(0,1)
        # --- 2. 融合输出 (无条件执行) ---
        # 权重: 主力(0.4), 供应(0.4), 势能(0.2)
        readiness_score = (
            main_force_control_score * 0.4 +
            supply_exhaustion_score * 0.4 +
            energy_accumulation_score * 0.2
        ).clip(0, 1)
        # 废除 is_consolidating 开关，直接输出连续性分数
        final_score = readiness_score.astype(np.float32)
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} ({current_date})")
            print(f"    -> 对手盘枯竭原始值: {counterparty_exhaustion_raw.iloc[-1]:.4f}")
            print(f"    -> 换手率原始值: {turnover_rate_f_raw.iloc[-1]:.4f}")
            print(f"    -> 供应枯竭度分数: {supply_exhaustion_score.iloc[-1]:.4f}")
            print(f"    -> 控盘坚实度原始值: {control_solidity_raw.iloc[-1]:.4f}")
            print(f"    -> 主力成本区防御意图原始值: {mf_cost_zone_defense_intent_raw.iloc[-1]:.4f}")
            print(f"    -> 主力控盘度分数: {main_force_control_score.iloc[-1]:.4f}")
            print(f"    -> 均衡压缩原始值: {equilibrium_compression_raw.iloc[-1]:.4f}")
            print(f"    -> 均衡压缩分数: {equilibrium_compression_score.iloc[-1]:.4f}")
            print(f"    -> 势能积蓄度分数 (来自结构张力与均衡压缩): {energy_accumulation_score.iloc[-1]:.4f}")
            print(f"    -> 最终突破准备度分数: {final_score.iloc[-1]:.4f}")
        return final_score

    def _diagnose_structural_momentum(self, df: pd.DataFrame, input_contextual_posture: pd.Series, axiom_tension: pd.Series, breakout_readiness: pd.Series, axiom_stability: pd.Series, axiom_mtf_cohesion: pd.Series, posture_type: str = "unknown") -> pd.Series:
        """
        【V6.0 · 结构动量深度进化版 - 势能与惯性融合】诊断“结构动量”
        - 核心升级: 不再是简单地计算结构战略态势的斜率，而是融合了“结构速度”、“结构加速度”、“动量持续性”和“动量情境品质”四大维度。
        - 核心证据:
            - 结构速度: 结构战略态势的短期变化率。
            - 结构加速度: 结构战略态态势变化率的变化率。
            - 动量持续性: 当前动量方向的持续时间。
            - 动量情境品质: 动量发生时所处的结构环境（如是否突破压缩，是否加速进入阻力）。
        - 旨在提供一个更全面、更具前瞻性的结构动量信号，区分“虚假动量”与“真实势能”。
        - 【V6.0.1 · 探针清晰化】新增 `posture_type` 参数，用于在探针输出中明确当前情境态势的类型。
        """
        method_name = "_diagnose_structural_momentum"
        df_index = df.index
        p_conf_struct = self.structural_ultimate_params
        momentum_params = get_param_value(p_conf_struct.get('structural_momentum_params'), {})
        mtf_weights_conf = get_param_value(p_conf_struct.get('mtf_normalization_weights'), {})
        tf_weights_momentum = mtf_weights_conf.get('structural_momentum', {3: 0.4, 5: 0.3, 8: 0.2, 13: 0.1})
        velocity_window = get_param_value(momentum_params.get('velocity_window'), 5)
        acceleration_window = get_param_value(momentum_params.get('acceleration_window'), 3)
        persistence_window = get_param_value(momentum_params.get('persistence_window'), 10)
        fusion_weights = get_param_value(momentum_params.get('fusion_weights'), {
            "velocity": 0.4, "acceleration": 0.3, "persistence": 0.15, "context_quality": 0.15
        })
        context_weights = get_param_value(momentum_params.get('context_weights'), {
            "favorable_tension_inverse": 0.3, "favorable_breakout_readiness": 0.3,
            "unfavorable_stability_inverse": 0.2, "unfavorable_cohesion_inverse": 0.2
        })
        non_linear_exponent = get_param_value(momentum_params.get('non_linear_exponent'), 1.5)
        # 1. 结构速度 (Structural Velocity)
        velocity_raw = ta.slope(input_contextual_posture, length=velocity_window)
        velocity_raw.fillna(0, inplace=True)
        velocity_score = get_adaptive_mtf_normalized_bipolar_score(velocity_raw, df_index, tf_weights_momentum)
        # 2. 结构加速度 (Structural Acceleration)
        acceleration_raw = ta.slope(velocity_raw, length=acceleration_window)
        acceleration_raw.fillna(0, inplace=True)
        acceleration_score = get_adaptive_mtf_normalized_bipolar_score(acceleration_raw, df_index, tf_weights_momentum)
        # 3. 动量持续性 (Momentum Persistence)
        # 统计连续正向或负向的速度周期数
        positive_velocity_count = (velocity_raw > 0).astype(int).rolling(window=persistence_window, min_periods=1).sum()
        negative_velocity_count = (velocity_raw < 0).astype(int).rolling(window=persistence_window, min_periods=1).sum()
        persistence_raw = positive_velocity_count - negative_velocity_count
        persistence_score = get_adaptive_mtf_normalized_bipolar_score(persistence_raw, df_index, tf_weights_momentum)
        # 4. 动量情境品质 (Momentum Context Quality)
        # 有利情境：低张力（张力越低越好），高突破准备度
        favorable_context_raw = (1 - axiom_tension) * context_weights.get('favorable_tension_inverse', 0.3) + \
                                breakout_readiness * context_weights.get('favorable_breakout_readiness', 0.3)
        favorable_context_score = get_adaptive_mtf_normalized_score(favorable_context_raw, df_index, tf_weights_momentum, ascending=True)
        # 不利情境：低稳定性（稳定性越低越差），低协同度（协同度越低越差）
        unfavorable_context_raw = (1 - ((axiom_stability + 1) / 2)) * context_weights.get('unfavorable_stability_inverse', 0.2) + \
                                  (1 - ((axiom_mtf_cohesion + 1) / 2)) * context_weights.get('unfavorable_cohesion_inverse', 0.2)
        unfavorable_context_score = get_adaptive_mtf_normalized_score(unfavorable_context_raw, df_index, tf_weights_momentum, ascending=True)
        context_quality_score = (favorable_context_score - unfavorable_context_score).clip(-1, 1)
        # 5. 最终融合
        fused_momentum = (
            velocity_score * fusion_weights.get('velocity', 0.4) +
            acceleration_score * fusion_weights.get('acceleration', 0.3) +
            persistence_score * fusion_weights.get('persistence', 0.15) +
            context_quality_score * fusion_weights.get('context_quality', 0.15)
        ).clip(-1, 1)
        # 应用非线性放大
        final_score = pd.Series(np.sign(fused_momentum) * (np.abs(fused_momentum) ** non_linear_exponent), index=df_index).clip(-1, 1).astype(np.float32)
        if self.is_probe_date and not df.empty:
            current_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  [结构情报探针] -> 方法: {method_name} (类型: {posture_type}) ({current_date})")
            print(f"    -> 输入情境态势: {input_contextual_posture.iloc[-1]:.4f}")
            print(f"    -> 结构速度原始值: {velocity_raw.iloc[-1]:.4f}")
            print(f"    -> 结构速度分数: {velocity_score.iloc[-1]:.4f}")
            print(f"    -> 结构加速度原始值: {acceleration_raw.iloc[-1]:.4f}")
            print(f"    -> 结构加速度分数: {acceleration_score.iloc[-1]:.4f}")
            print(f"    -> 动量持续性原始值: {persistence_raw.iloc[-1]:.4f}")
            print(f"    -> 动量持续性分数: {persistence_score.iloc[-1]:.4f}")
            print(f"    -> 有利情境原始值: {favorable_context_raw.iloc[-1]:.4f}")
            print(f"    -> 不利情境原始值: {unfavorable_context_raw.iloc[-1]:.4f}")
            print(f"    -> 动量情境品质分数: {context_quality_score.iloc[-1]:.4f}")
            print(f"    -> 融合动量 (非线性前): {fused_momentum.iloc[-1]:.4f}")
            print(f"    -> 最终结构动量分数: {final_score.iloc[-1]:.4f}")
        return final_score








