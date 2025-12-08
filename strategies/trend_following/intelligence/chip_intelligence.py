import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Union
from strategies.trend_following import utils
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    get_adaptive_mtf_normalized_bipolar_score, get_robust_bipolar_normalized_score, normalize_score
)

class ChipIntelligence:
    def __init__(self, strategy_instance):
        """
        【V2.1 · 依赖注入版】
        - 核心升级: 新增 self.bipolar_sensitivity 属性，从策略配置中读取归一化所需的敏感度参数。
                     解决了在调用外部归一化工具时缺少依赖参数的问题。
        """
        self.strategy = strategy_instance
        self.params = get_params_block(self.strategy, 'chip_intelligence_params', {})
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})
        # 注入双极归一化所需的敏感度参数
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        self.bipolar_sensitivity = get_param_value(process_params.get('bipolar_sensitivity'), 1.0)

    def _get_safe_series(self, df: pd.DataFrame, data_source: Union[pd.DataFrame, Dict[str, pd.Series]], column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        【V2.0 · 上下文修复版】安全地从DataFrame或字典中获取Series，如果不存在则打印警告并返回默认Series。
        - 核心修复: 接收 df 参数，并使用其索引创建默认 Series，确保上下文一致。
        """
        df_index = df.index # 使用传入的 df.index
        if isinstance(data_source, pd.DataFrame):
            if column_name not in data_source.columns:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            return data_source[column_name]
        elif isinstance(data_source, dict):
            if column_name not in data_source:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少字典数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            series = data_source[column_name]
            if isinstance(series, pd.Series):
                return series.reindex(df_index, fill_value=default_value)
            else:
                return pd.Series(series, index=df_index)
        else:
            print(f"    -> [筹码情报警告] 方法 '{method_name}' 接收到未知数据源类型 {type(data_source)}，无法获取 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df_index)

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V18.0 · 破晓版】筹码情报总指挥
        - 核心升维: 新增对“和谐拐点”的诊断。在生成“战略战术和谐度”后，立即对其进行二阶求导分析，
                      旨在捕捉其从冲突转向协同的关键反转点，为模型提供预判战局“破晓”时刻的能力。
        """
        print("启动【V18.0 · 破晓版】筹码情报分析...") # [修改代码行]
        all_chip_states = {}
        periods = [5, 13, 21, 55]
        holder_sentiment_scores = self._diagnose_axiom_holder_sentiment(df, periods)
        divergence_scores = self._diagnose_axiom_divergence(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = holder_sentiment_scores
        all_chip_states['SCORE_CHIP_AXIOM_DIVERGENCE'] = divergence_scores
        strategic_posture = self._diagnose_strategic_posture(df)
        all_chip_states['SCORE_CHIP_STRATEGIC_POSTURE'] = strategic_posture
        battlefield_geography = self._diagnose_battlefield_geography(df)
        all_chip_states['SCORE_CHIP_BATTLEFIELD_GEOGRAPHY'] = battlefield_geography
        chip_trend_momentum_scores = self._diagnose_axiom_trend_momentum(df, periods, strategic_posture, battlefield_geography, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_AXIOM_TREND_MOMENTUM'] = chip_trend_momentum_scores
        historical_potential = self._diagnose_axiom_historical_potential(df)
        all_chip_states['SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL'] = historical_potential
        absorption_echo = self._diagnose_absorption_echo(df, divergence_scores)
        all_chip_states['SCORE_CHIP_OPP_ABSORPTION_ECHO'] = absorption_echo
        distribution_whisper = self._diagnose_distribution_whisper(df, divergence_scores)
        all_chip_states['SCORE_CHIP_RISK_DISTRIBUTION_WHISPER'] = distribution_whisper
        coherent_drive = self._diagnose_structural_consensus(df, battlefield_geography, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_COHERENT_DRIVE'] = coherent_drive
        tactical_exchange = self._diagnose_tactical_exchange(df, battlefield_geography)
        all_chip_states['SCORE_CHIP_TACTICAL_EXCHANGE'] = tactical_exchange
        strategic_tactical_harmony = self._diagnose_strategic_tactical_harmony(df, strategic_posture, tactical_exchange)
        all_chip_states['SCORE_CHIP_STRATEGIC_TACTICAL_HARMONY'] = strategic_tactical_harmony
        # [新增代码块] 调用新增的和谐拐点诊断方法
        harmony_inflection = self._diagnose_harmony_inflection(df, strategic_tactical_harmony)
        all_chip_states['SCORE_CHIP_HARMONY_INFLECTION'] = harmony_inflection
        print(f"【V18.0 · 破晓版】分析完成，生成 {len(all_chip_states)} 个筹码原子信号。") # [修改代码行]
        return all_chip_states

    def _diagnose_strategic_posture(self, df: pd.DataFrame) -> pd.Series:
        """
        【V7.0 · 诡道时序增强版】诊断主力的综合战略态势 (大一统信号)
        - 核心升级1: 细化“指挥官决心”维度中的诡道类型，将单一欺骗指数拆分为“压价吸筹（诱空）”、“拉高出货（诱多）”和“对倒”三种，并进行加权融合。
        - 核心升级2: 对基础战略态态得分进行时间序列分析，计算其“速度”和“加速度”，并将其与基础得分进行融合，增强信号的前瞻性。
        """
        # 移除调试探针
        # print("    -> [筹码层] 正在诊断“战略态态 (V7.0 · 诡道时序增强版)”...")
        required_signals = [
            'cost_gini_coefficient_D', 'covert_accumulation_signal_D', 'peak_exchange_purity_D',
            'main_force_cost_advantage_D', 'control_solidity_index_D', 'SLOPE_5_main_force_conviction_index_D',
            'floating_chip_cleansing_efficiency_D', 'dominant_peak_solidity_D',
            'deception_index_D', # 原始欺骗指数，用于拆分
            'wash_trade_intensity_D' # 新增对倒强度依赖
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_strategic_posture"):
            return pd.Series(0.0, index=df.index)

        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        sp_params = get_param_value(p_conf.get('strategic_posture_params'), {})
        deception_fusion_weights = get_param_value(sp_params.get('deception_fusion_weights'), {"bear_trap_positive": 0.6, "bull_trap_negative": 0.2, "wash_trade_negative": 0.2})
        dynamic_fusion_weights = get_param_value(sp_params.get('dynamic_fusion_weights'), {'base_score': 0.6, 'velocity': 0.2, 'acceleration': 0.2})
        smoothing_ema_span = get_param_value(sp_params.get('smoothing_ema_span'), 5)

        df_index = df.index

        # --- 维度1: 阵型部署 (Formation Deployment) ---
        concentration_level = 1 - self._get_safe_series(df, df, 'cost_gini_coefficient_D', 0.5)
        covert_accumulation = self._get_safe_series(df, df, 'covert_accumulation_signal_D', 0.0)
        peak_purity = self._get_safe_series(df, df, 'peak_exchange_purity_D', 0.0)

        level_score = get_adaptive_mtf_normalized_bipolar_score(concentration_level, df_index, tf_weights)
        efficiency_score = (
            get_adaptive_mtf_normalized_bipolar_score(covert_accumulation, df_index, tf_weights).add(1)/2 *
            get_adaptive_mtf_normalized_bipolar_score(peak_purity, df_index, tf_weights).add(1)/2
        ).pow(0.5) * 2 - 1
        formation_deployment_score = (level_score.add(1)/2 * efficiency_score.add(1)/2).pow(0.5) * 2 - 1

        # --- 维度2: 指挥官决心 (Commander's Resolve) ---
        cost_advantage = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0)
        control_solidity = self._get_safe_series(df, df, 'control_solidity_index_D', 0.0)
        conviction_slope = self._get_safe_series(df, df, 'SLOPE_5_main_force_conviction_index_D', 0.0)
        deception_index = self._get_safe_series(df, df, 'deception_index_D', 0.0)
        wash_trade_intensity = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0)

        deception_bear_trap_raw = deception_index.clip(lower=0)
        deception_bull_trap_raw = deception_index.clip(upper=0).abs()

        bear_trap_score = get_adaptive_mtf_normalized_score(deception_bear_trap_raw, df_index, ascending=True, tf_weights=tf_weights)
        bull_trap_score = get_adaptive_mtf_normalized_score(deception_bull_trap_raw, df_index, ascending=True, tf_weights=tf_weights)
        wash_trade_score = get_adaptive_mtf_normalized_score(wash_trade_intensity, df_index, ascending=True, tf_weights=tf_weights)

        deception_impact_score = (
            bear_trap_score * deception_fusion_weights.get('bear_trap_positive', 0.6)
            - bull_trap_score * deception_fusion_weights.get('bull_trap_negative', 0.2)
            - wash_trade_score * deception_fusion_weights.get('wash_trade_negative', 0.2)
        ).clip(-1, 1)

        advantage_score = get_adaptive_mtf_normalized_bipolar_score(cost_advantage, df_index, tf_weights)
        solidity_score = get_adaptive_mtf_normalized_bipolar_score(control_solidity, df_index, tf_weights)
        intent_score = get_adaptive_mtf_normalized_bipolar_score(conviction_slope, df_index, tf_weights)

        commanders_resolve_score = (
            (advantage_score.add(1)/2) * (solidity_score.add(1)/2) *
            (intent_score.clip(lower=-1, upper=1).add(1)/2) * (deception_impact_score.add(1)/2)
        ).pow(1/4) * 2 - 1

        # --- 维度3: 战场控制 (Battlefield Control) ---
        cleansing_efficiency = self._get_safe_series(df, df, 'floating_chip_cleansing_efficiency_D', 0.0)
        peak_solidity = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5)

        cleansing_score = get_adaptive_mtf_normalized_bipolar_score(cleansing_efficiency, df_index, tf_weights)
        peak_solidity_score = get_adaptive_mtf_normalized_bipolar_score(peak_solidity, df_index, tf_weights)
        battlefield_control_score = (cleansing_score.add(1)/2 * peak_solidity_score.add(1)/2).pow(0.5) * 2 - 1

        # --- 基础融合 (不含时间序列动态) ---
        base_strategic_posture_score = (
            (commanders_resolve_score.add(1)/2).pow(0.5) *
            (formation_deployment_score.add(1)/2).pow(0.3) *
            (battlefield_control_score.add(1)/2).pow(0.2)
        ).pow(1/(0.5+0.3+0.2)) * 2 - 1

        # --- 时间序列分析 (Strategic Dynamics) ---
        smoothed_base_score = base_strategic_posture_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)

        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights)

        final_score = (
            (base_strategic_posture_score.add(1)/2).pow(dynamic_fusion_weights.get('base_score', 0.6)) *
            (norm_velocity.add(1)/2).pow(dynamic_fusion_weights.get('velocity', 0.2)) *
            (norm_acceleration.add(1)/2).pow(dynamic_fusion_weights.get('acceleration', 0.2))
        ).pow(1 / sum(dynamic_fusion_weights.values())) * 2 - 1
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_battlefield_geography(self, df: pd.DataFrame) -> pd.Series:
        """
        【V7.0 · 动态演化版】诊断筹码的战场地形 (大一统信号)
        - 核心升级: 引入“动态演化”因子。通过融合支撑强度斜率与阻力强度斜率，量化战场地形的
                      有利度是在改善还是在恶化，从而对静态地形分进行动态调节，使信号更具前瞻性。
        - 核心升级: 植入标准化的“真理探针”，输出所有原始数据、关键计算过程及最终结果。
        """
        print("    -> [筹码层] 正在诊断“战场地形 (V7.0 · 动态演化版)”...")
        required_signals = [
            'dominant_peak_solidity_D', 'support_validation_strength_D', 'chip_fault_blockage_ratio_D',
            'pressure_rejection_strength_D', 'vacuum_zone_magnitude_D', 'vacuum_traversal_efficiency_D',
            'SLOPE_5_support_validation_strength_D', 'SLOPE_5_pressure_rejection_strength_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_battlefield_geography"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # --- 维度1: 下方支撑 (Support Strength) ---
        peak_solidity = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5)
        support_validation = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0)
        solidity_score = get_adaptive_mtf_normalized_score(peak_solidity, df_index, tf_weights)
        validation_score = get_adaptive_mtf_normalized_score(support_validation, df_index, tf_weights)
        support_strength_score = (solidity_score * validation_score).pow(0.5)
        # --- 维度2: 上方阻力 (Resistance Strength) ---
        fault_blockage = self._get_safe_series(df, df, 'chip_fault_blockage_ratio_D', 0.5)
        pressure_rejection = self._get_safe_series(df, df, 'pressure_rejection_strength_D', 0.5)
        blockage_score = get_adaptive_mtf_normalized_score(fault_blockage, df_index, tf_weights)
        rejection_score = get_adaptive_mtf_normalized_score(pressure_rejection, df_index, tf_weights)
        resistance_strength_score = (blockage_score * rejection_score).pow(0.5)
        # --- 维度3: 最小阻力路径 (Path Score) ---
        vacuum_magnitude = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0)
        vacuum_efficiency = self._get_safe_series(df, df, 'vacuum_traversal_efficiency_D', 0.0)
        magnitude_score = get_adaptive_mtf_normalized_score(vacuum_magnitude, df_index, tf_weights)
        efficiency_score = get_adaptive_mtf_normalized_score(vacuum_efficiency, df_index, tf_weights)
        path_score = (magnitude_score * efficiency_score).pow(0.5)
        # --- 静态融合 ---
        base_score_raw = support_strength_score * (1 - resistance_strength_score)
        static_final_score = np.sign(base_score_raw) * (base_score_raw.abs() * path_score).pow(0.5)
        # --- 维度4: 动态演化 (Dynamic Evolution) ---
        support_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_support_validation_strength_D', 0.0)
        resistance_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_pressure_rejection_strength_D', 0.0)
        support_trend_score = get_adaptive_mtf_normalized_bipolar_score(support_trend_raw, df_index, tf_weights)
        resistance_trend_score = get_adaptive_mtf_normalized_bipolar_score(resistance_trend_raw, df_index, tf_weights)
        terrain_advantage_change = support_trend_score - resistance_trend_score
        dynamic_evolution_factor = 1.0 + (terrain_advantage_change * 0.25) # 变化趋势的影响力设为25%
        # --- 最终动态融合 ---
        final_score = static_final_score * dynamic_evolution_factor
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V7.8 · 最终优化版】筹码公理三：诊断“持仓信念韧性”
        - 核心升级1: 动态信念权重。引入“筹码趋势情境”（赢家集中度趋势）来动态调整 `winner_stability` 和 `loser_pain` 在 `belief_core_score` 中的权重。
                      在赢家集中度上升趋势中，更看重赢家稳定性；在下降趋势中，更看重输家痛苦指数。
        - 核心升级2: 恐慌奖励动态敏感度。`capitulation_bonus` 的乘数不再是固定值，而是根据“筹码疲劳指数”进行动态调整。
                      筹码疲劳度越高，恐慌吸收的奖励可能越大。
        - 核心升级3: 情绪纯度非线性动态调制。`impurity_score` 对 `conviction_base` 的削弱作用，将通过一个非线性函数（如 `tanh`）进行调制，
                      并引入一个动态敏感度，该敏感度可以根据筹码情绪的绝对强度进行调整。
        - 核心升级4: 诡道因子融入压力测试。引入一个“筹码故障幅度”（负向部分，代表筹码诱空）来调节 `pressure_test_score`。
                      如果存在筹码诱空，即使承接和防守分数不高，也可能被视为一种“策略性”的压力测试。
        - 核心升级5: 欺骗的非对称影响。引入正向“筹码故障幅度”（诱多）作为对 `conviction_base` 的直接惩罚，以反映主力诱多派发的风险。
        - 核心升级6: 情绪杂质的上下文敏感性。根据“筹码健康度”（如过热/过冷）动态调整 `impurity_score` 的非线性调制强度。
        - 核心升级7: 调制器信号MTF增强。`sentiment_trend_modulator`、`panic_reward_modulator`和`impurity_context_modulator`的归一化方式升级为多时间框架（MTF）自适应归一化，使其更鲁棒和情境感知。
        - 核心升级8: 信念内核与压力测试动态融合。`conviction_base`的计算引入动态权重，根据筹码压力信号（如筹码疲劳指数）调整信念内核和压力测试的融合权重，使模型更具情境适应性。
        - 核心升级9: 杂质独立调制与乘法融合。`fomo_score`和`profit_taking_score`将独立进行情境敏感的非线性调制，然后以乘法形式融合其对`conviction_base`的削弱作用，以更精细地捕捉不同杂质的非对称影响。
        - 核心升级10: 杂质协同调制。引入可调的“杂质融合指数”，并使其根据信念强度动态调整，以更灵活地控制`fomo_effect`和`profit_taking_effect`融合时的协同强度，避免过度惩罚。
        - 核心升级11: 杂质双向感知与阈值化。`fomo_score`（基于赢家集中度）改造为双向感知杂质，偏离“最优集中度目标”越大，杂质越高；`profit_taking_score`（基于赢家平均利润率）引入阈值，低于阈值不计入杂质。
        - 代码优化: 移除所有调试探针，优化部分计算逻辑，提高运行效率。
        """
        required_signals = [
            'winner_stability_index_D', 'loser_pain_index_D', 'active_buying_support_D',
            'support_validation_strength_D', 'winner_concentration_90pct_D',
            'winner_profit_margin_avg_D', 'capitulation_absorption_index_D',
            'SLOPE_55_winner_concentration_90pct_D',
            'chip_fatigue_index_D',
            'chip_fault_magnitude_D',
            'chip_health_score_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_holder_sentiment"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        holder_sentiment_params = get_param_value(p_conf.get('holder_sentiment_params'), {})
        sentiment_trend_modulator_signal_name = get_param_value(holder_sentiment_params.get('sentiment_trend_modulator_signal_name'), 'SLOPE_55_winner_concentration_90pct_D')
        sentiment_trend_mod_factor = get_param_value(holder_sentiment_params.get('sentiment_trend_mod_factor'), 0.5)
        panic_reward_modulator_signal_name = get_param_value(holder_sentiment_params.get('panic_reward_modulator_signal_name'), 'chip_fatigue_index_D')
        panic_reward_mod_tanh_factor = get_param_value(holder_sentiment_params.get('panic_reward_mod_tanh_factor'), 1.0)
        panic_reward_mod_factor = get_param_value(holder_sentiment_params.get('panic_reward_mod_factor'), 1.0)
        capitulation_base_reward_multiplier = get_param_value(holder_sentiment_params.get('capitulation_base_reward_multiplier'), 0.3)
        impurity_non_linear_enabled = get_param_value(holder_sentiment_params.get('impurity_non_linear_enabled'), True)
        fomo_tanh_factor = get_param_value(holder_sentiment_params.get('fomo_tanh_factor'), 1.0)
        fomo_sentiment_sensitivity = get_param_value(holder_sentiment_params.get('fomo_sentiment_sensitivity'), 0.5)
        profit_taking_tanh_factor = get_param_value(holder_sentiment_params.get('profit_taking_tanh_factor'), 1.0)
        profit_taking_sentiment_sensitivity = get_param_value(holder_sentiment_params.get('profit_taking_sentiment_sensitivity'), 0.5)
        deception_factor_enabled = get_param_value(holder_sentiment_params.get('deception_factor_enabled'), True)
        deception_signal_name = get_param_value(holder_sentiment_params.get('deception_signal_name'), 'chip_fault_magnitude_D')
        deception_impact_factor = get_param_value(holder_sentiment_params.get('deception_impact_factor'), 0.2)
        positive_deception_penalty_enabled = get_param_value(holder_sentiment_params.get('positive_deception_penalty_enabled'), True)
        positive_deception_impact_factor = get_param_value(holder_sentiment_params.get('positive_deception_impact_factor'), 0.15)
        impurity_context_modulation_enabled = get_param_value(holder_sentiment_params.get('impurity_context_modulation_enabled'), True)
        impurity_context_modulator_signal_name = get_param_value(holder_sentiment_params.get('impurity_context_modulator_signal_name'), 'chip_health_score_D')
        impurity_context_overbought_amp_factor = get_param_value(holder_sentiment_params.get('impurity_context_overbought_amp_factor'), 0.5)
        impurity_context_oversold_damp_factor = get_param_value(holder_sentiment_params.get('impurity_context_oversold_damp_factor'), 0.2)
        dynamic_fusion_enabled = get_param_value(holder_sentiment_params.get('dynamic_fusion_enabled'), True)
        min_pressure_weight = get_param_value(holder_sentiment_params.get('min_pressure_weight'), 0.3)
        max_pressure_weight = get_param_value(holder_sentiment_params.get('max_pressure_weight'), 0.7)
        impurity_fusion_exponent_base = get_param_value(holder_sentiment_params.get('impurity_fusion_exponent_base'), 0.7)
        impurity_fusion_exponent_sensitivity = get_param_value(holder_sentiment_params.get('impurity_fusion_exponent_sensitivity'), 0.5)
        fomo_concentration_optimal_target = get_param_value(holder_sentiment_params.get('fomo_concentration_optimal_target'), 0.5)
        profit_taking_threshold = get_param_value(holder_sentiment_params.get('profit_taking_threshold'), 5.0)
        df_index = df.index
        winner_stability = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        loser_pain = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        stability_score = get_adaptive_mtf_normalized_bipolar_score(winner_stability, df_index, tf_weights)
        pain_score = get_adaptive_mtf_normalized_bipolar_score(loser_pain, df_index, tf_weights)
        sentiment_trend_raw = self._get_safe_series(df, df, sentiment_trend_modulator_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
        normalized_sentiment_trend = get_adaptive_mtf_normalized_score(sentiment_trend_raw, df_index, tf_weights=tf_weights, ascending=True)
        x = (normalized_sentiment_trend * sentiment_trend_mod_factor).clip(-0.4, 0.4)
        dynamic_stability_weight = 0.5 + x
        dynamic_pain_weight = 0.5 - x
        belief_core_score = (
            (stability_score.add(1)/2).pow(dynamic_stability_weight) * 
            (pain_score.add(1)/2).pow(dynamic_pain_weight)
        ) * 2 - 1 # [修改代码行] 移除冗余的 .pow(1 / (dynamic_stability_weight + dynamic_pain_weight))
        absorption_power = self._get_safe_series(df, df, 'active_buying_support_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        defense_intent = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        capitulation_absorption = self._get_safe_series(df, df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        absorption_score = get_adaptive_mtf_normalized_bipolar_score(absorption_power, df_index, tf_weights)
        defense_score = get_adaptive_mtf_normalized_bipolar_score(defense_intent, df_index, tf_weights)
        capitulation_score = get_adaptive_mtf_normalized_score(capitulation_absorption, df_index, tf_weights)
        base_pressure_score = ((absorption_score.add(1)/2 * defense_score.add(1)/2).pow(0.5) * 2 - 1)
        panic_modulator_raw = self._get_safe_series(df, df, panic_reward_modulator_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
        normalized_panic_modulator = get_adaptive_mtf_normalized_score(panic_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
        panic_reward_adjustment_factor = np.tanh(normalized_panic_modulator * panic_reward_mod_tanh_factor) * panic_reward_mod_factor
        dynamic_capitulation_reward_multiplier = capitulation_base_reward_multiplier * (1 + panic_reward_adjustment_factor)
        dynamic_capitulation_reward_multiplier = dynamic_capitulation_reward_multiplier.clip(0.1, 0.8)
        capitulation_bonus = capitulation_score * dynamic_capitulation_reward_multiplier
        deception_impact = pd.Series(0.0, index=df.index)
        deception_raw = self._get_safe_series(df, df, deception_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
        if deception_factor_enabled:
            negative_deception = deception_raw.clip(upper=0).abs()
            normalized_negative_deception = get_adaptive_mtf_normalized_score(negative_deception, df_index, tf_weights)
            deception_impact = normalized_negative_deception * deception_impact_factor
        pressure_test_score = base_pressure_score * (1 + capitulation_bonus + deception_impact)
        pressure_test_score = pressure_test_score.clip(-1, 1)
        s_belief_core = belief_core_score.add(1)/2
        s_pressure_test = pressure_test_score.add(1)/2
        dynamic_belief_core_weight = pd.Series(0.5, index=df.index)
        dynamic_pressure_test_weight = pd.Series(0.5, index=df.index)
        if dynamic_fusion_enabled:
            dynamic_pressure_test_weight = min_pressure_weight + (max_pressure_weight - min_pressure_weight) * normalized_panic_modulator
            dynamic_belief_core_weight = 1.0 - dynamic_pressure_test_weight
        conviction_base = (s_belief_core.pow(dynamic_belief_core_weight) * s_pressure_test.pow(dynamic_pressure_test_weight))
        positive_deception_penalty = pd.Series(0.0, index=df.index)
        if positive_deception_penalty_enabled:
            positive_deception_raw = deception_raw.clip(lower=0)
            normalized_positive_deception = get_adaptive_mtf_normalized_score(positive_deception_raw, df_index, tf_weights)
            positive_deception_penalty = normalized_positive_deception * positive_deception_impact_factor
            conviction_base = conviction_base * (1 - positive_deception_penalty)
            conviction_base = conviction_base.clip(0, 1)
        fomo_index_raw = self._get_safe_series(df, df, 'winner_concentration_90pct_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        profit_taking_quality_raw = self._get_safe_series(df, df, 'winner_profit_margin_avg_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        fomo_deviation = (fomo_index_raw - fomo_concentration_optimal_target).abs()
        fomo_score = get_adaptive_mtf_normalized_score(fomo_deviation, df_index, ascending=True, tf_weights=tf_weights)
        profit_taking_quality_thresholded = (profit_taking_quality_raw - profit_taking_threshold).clip(lower=0) # [修改代码行] 优化矢量化操作
        profit_taking_score = get_adaptive_mtf_normalized_score(profit_taking_quality_thresholded, df_index, ascending=True, tf_weights=tf_weights)
        fomo_effect = pd.Series(0.0, index=df.index)
        profit_taking_effect = pd.Series(0.0, index=df.index)
        final_impurity_effect = pd.Series(0.0, index=df.index)
        if impurity_non_linear_enabled:
            current_sentiment_strength = conviction_base.abs()
            normalized_sentiment_strength = normalize_score(current_sentiment_strength, df_index, window=21, ascending=True)
            context_adjustment_factor = pd.Series(1.0, index=df.index)
            if impurity_context_modulation_enabled:
                context_modulator_raw = self._get_safe_series(df, df, impurity_context_modulator_signal_name, 0.0, method_name="_diagnose_axiom_holder_sentiment")
                normalized_context_modulator = get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
                overbought_mask = normalized_context_modulator > 0.7
                oversold_mask = normalized_context_modulator < 0.3
                context_adjustment_factor.loc[overbought_mask] = 1 + (normalized_context_modulator.loc[overbought_mask] - 0.7) * impurity_context_overbought_amp_factor / 0.3
                context_adjustment_factor.loc[oversold_mask] = 1 - (0.3 - normalized_context_modulator.loc[oversold_mask]) * impurity_context_oversold_damp_factor / 0.3
            dynamic_fomo_tanh_factor = fomo_tanh_factor * (1 + normalized_sentiment_strength * fomo_sentiment_sensitivity)
            dynamic_fomo_tanh_factor = dynamic_fomo_tanh_factor * context_adjustment_factor
            dynamic_fomo_tanh_factor = dynamic_fomo_tanh_factor.clip(0.5, 3.0)
            fomo_effect = np.tanh(fomo_score * dynamic_fomo_tanh_factor)
            dynamic_profit_taking_tanh_factor = profit_taking_tanh_factor * (1 + normalized_sentiment_strength * profit_taking_sentiment_sensitivity)
            dynamic_profit_taking_tanh_factor = dynamic_profit_taking_tanh_factor * context_adjustment_factor
            dynamic_profit_taking_tanh_factor = dynamic_profit_taking_tanh_factor.clip(0.5, 3.0)
            profit_taking_effect = np.tanh(profit_taking_score * dynamic_profit_taking_tanh_factor)
            dynamic_impurity_fusion_exponent = impurity_fusion_exponent_base * (1 - normalized_sentiment_strength * impurity_fusion_exponent_sensitivity)
            dynamic_impurity_fusion_exponent = dynamic_impurity_fusion_exponent.clip(0.1, 1.0)
            final_impurity_effect = 1 - ((1 - fomo_effect) * (1 - profit_taking_effect)).pow(dynamic_impurity_fusion_exponent)
        else:
            final_impurity_effect = pd.Series(0.0, index=df.index)
        final_score = (conviction_base * (1 - final_impurity_effect)) * 2 - 1
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_trend_momentum(self, df: pd.DataFrame, periods: list, strategic_posture: pd.Series, battlefield_geography: pd.Series, holder_sentiment: pd.Series) -> pd.Series:
        """
        【V7.0 · 战略推力引擎版】筹码公理六：诊断“结构性推力”
        - 核心升级1: 引擎功率动态权重。引入筹码健康度趋势作为调制器，动态调整静态基础分与动态变化率的融合权重。
        - 核心升级2: 燃料品质诡道调制。引入筹码故障幅度作为负向调制器，削弱被“诱多”等诡道污染的燃料品质，并使协同奖励情境感知。
        - 核心升级3: 喷管效率多维深化。融合真空区大小、真空区趋势和真空穿越效率，更全面评估最小阻力路径。
        - 核心升级4: 最终融合动态权重。引入战略态势作为情境调制器，动态调整引擎功率、燃料品质、喷管效率的融合权重。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        - 修复: 解决 SCORE_CHIP_STRATEGIC_POSTURE 作为调制器时因重复查找导致的数据缺失警告。
        - 修复: 修正 get_adaptive_mtf_normalized_score 函数的参数传递错误。
        """
        print("    -> [筹码层] 正在诊断“结构性推力”公理 (V7.0 · 战略推力引擎版)...")
        required_signals = [
            'main_force_conviction_index_D', 'vacuum_zone_magnitude_D', 'upward_impulse_purity_D',
            'chip_health_score_D', 'chip_fault_magnitude_D', 'SLOPE_5_vacuum_zone_magnitude_D',
            'vacuum_traversal_efficiency_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend_momentum"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        trend_momentum_params = get_param_value(p_conf.get('trend_momentum_params'), {})
        health_weights = get_param_value(trend_momentum_params.get('health_weights'), {'posture': 0.4, 'geography': 0.4, 'sentiment': 0.2})
        engine_power_dynamic_weight_modulator_signal_name = get_param_value(trend_momentum_params.get('engine_power_dynamic_weight_modulator_signal_name'), 'SLOPE_5_chip_health_score_D')
        engine_power_dynamic_weight_sensitivity = get_param_value(trend_momentum_params.get('engine_power_dynamic_weight_sensitivity'), 0.5)
        static_engine_power_base_weight = get_param_value(trend_momentum_params.get('static_engine_power_base_weight'), 0.5)
        dynamic_engine_power_base_weight = get_param_value(trend_momentum_params.get('dynamic_engine_power_base_weight'), 0.5)
        fuel_purity_deception_penalty_factor = get_param_value(trend_momentum_params.get('fuel_purity_deception_penalty_factor'), 0.3)
        synergy_bonus_base = get_param_value(trend_momentum_params.get('synergy_bonus_base'), 0.25)
        synergy_bonus_context_modulator_signal_name = get_param_value(trend_momentum_params.get('synergy_bonus_context_modulator_signal_name'), 'chip_health_score_D')
        synergy_bonus_context_sensitivity = get_param_value(trend_momentum_params.get('synergy_bonus_context_sensitivity'), 0.5)
        nozzle_efficiency_weights = get_param_value(trend_momentum_params.get('nozzle_efficiency_weights'), {'magnitude': 0.5, 'trend': 0.3, 'traversal': 0.2})
        final_fusion_dynamic_weights_enabled = get_param_value(trend_momentum_params.get('final_fusion_dynamic_weights_enabled'), True)
        final_fusion_weights_base = get_param_value(trend_momentum_params.get('final_fusion_weights_base'), {'engine': 0.33, 'fuel': 0.33, 'nozzle': 0.34})
        final_fusion_weights_sensitivity = get_param_value(trend_momentum_params.get('final_fusion_weights_sensitivity'), {'engine': 0.5, 'fuel': 0.5, 'nozzle': 0.5})
        df_index = df.index
        static_engine_power = (
            strategic_posture * health_weights['posture'] +
            battlefield_geography * health_weights['geography'] +
            holder_sentiment * health_weights['sentiment']
        )
        health_score_slope_raw = self._get_safe_series(df, df, engine_power_dynamic_weight_modulator_signal_name, 0.0, method_name="_diagnose_axiom_trend_momentum")
        norm_health_score_slope = get_adaptive_mtf_normalized_bipolar_score(health_score_slope_raw, df_index, tf_weights)
        dynamic_weight_mod = (norm_health_score_slope * engine_power_dynamic_weight_sensitivity).clip(-0.5, 0.5)
        current_static_weight = (static_engine_power_base_weight - dynamic_weight_mod).clip(0.1, 0.9)
        current_dynamic_weight = (dynamic_engine_power_base_weight + dynamic_weight_mod).clip(0.1, 0.9)
        sum_current_weights = current_static_weight + current_dynamic_weight
        current_static_weight = current_static_weight / sum_current_weights
        current_dynamic_weight = current_dynamic_weight / sum_current_weights
        slope = static_engine_power.diff(1).fillna(0)
        accel = slope.diff(1).fillna(0)
        norm_slope = get_adaptive_mtf_normalized_bipolar_score(slope, df_index, tf_weights)
        norm_accel = get_adaptive_mtf_normalized_bipolar_score(accel, df_index, tf_weights)
        dynamic_engine_power = (norm_slope.add(1)/2 * norm_accel.clip(lower=-1, upper=1).add(1)/2).pow(0.5) * 2 - 1
        engine_power_score = static_engine_power * current_static_weight + dynamic_engine_power * current_dynamic_weight
        conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        impulse_purity_raw = self._get_safe_series(df, df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(conviction_raw, df_index, tf_weights)
        purity_score = get_adaptive_mtf_normalized_bipolar_score(impulse_purity_raw, df_index, tf_weights)
        base_fuel_quality = ((conviction_score.add(1)/2) * (purity_score.add(1)/2)).pow(0.5) * 2 - 1
        chip_fault_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        norm_chip_fault = get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights)
        deception_penalty = pd.Series(0.0, index=df.index)
        positive_fault_mask = chip_fault_raw > 0
        deception_penalty.loc[positive_fault_mask] = norm_chip_fault.loc[positive_fault_mask] * fuel_purity_deception_penalty_factor
        fuel_quality_score_after_deception = base_fuel_quality * (1 - deception_penalty.clip(0, 1))
        synergy_context_raw = self._get_safe_series(df, df, synergy_bonus_context_modulator_signal_name, 0.0, method_name="_diagnose_axiom_trend_momentum")
        norm_synergy_context = get_adaptive_mtf_normalized_score(synergy_context_raw, df_index, ascending=True, tf_weights=tf_weights)
        dynamic_synergy_bonus_factor = synergy_bonus_base * (1 + norm_synergy_context * synergy_bonus_context_sensitivity)
        dynamic_synergy_bonus_factor = dynamic_synergy_bonus_factor.clip(0.1, 0.5)
        synergy_bonus = (conviction_score.clip(lower=0) * purity_score.clip(lower=0)).pow(0.5) * dynamic_synergy_bonus_factor
        fuel_quality_score = fuel_quality_score_after_deception + synergy_bonus
        vacuum_magnitude_raw = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        vacuum_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_vacuum_zone_magnitude_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        vacuum_traversal_raw = self._get_safe_series(df, df, 'vacuum_traversal_efficiency_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        norm_vacuum_magnitude = get_adaptive_mtf_normalized_bipolar_score(vacuum_magnitude_raw, df_index, tf_weights)
        norm_vacuum_trend = get_adaptive_mtf_normalized_bipolar_score(vacuum_trend_raw, df_index, tf_weights)
        norm_traversal_efficiency = get_adaptive_mtf_normalized_bipolar_score(vacuum_traversal_raw, df_index, tf_weights)
        nozzle_efficiency_score = (
            norm_vacuum_magnitude * nozzle_efficiency_weights.get('magnitude', 0.5) +
            norm_vacuum_trend * nozzle_efficiency_weights.get('trend', 0.3) +
            norm_traversal_efficiency * nozzle_efficiency_weights.get('traversal', 0.2)
        ).clip(-1, 1)
        engine_score_normalized = engine_power_score.add(1)/2
        fuel_score_normalized = fuel_quality_score.clip(-1, 1).add(1)/2
        nozzle_score_normalized = nozzle_efficiency_score.add(1)/2
        final_engine_weight = pd.Series(final_fusion_weights_base.get('engine', 0.33), index=df_index)
        final_fuel_weight = pd.Series(final_fusion_weights_base.get('fuel', 0.33), index=df_index)
        final_nozzle_weight = pd.Series(final_fusion_weights_base.get('nozzle', 0.34), index=df_index)
        if final_fusion_dynamic_weights_enabled:
            fusion_modulator_raw = strategic_posture
            normalized_fusion_modulator = get_adaptive_mtf_normalized_bipolar_score(fusion_modulator_raw, df_index, tf_weights)
            engine_mod = normalized_fusion_modulator * final_fusion_weights_sensitivity.get('engine', 0.5)
            fuel_mod = normalized_fusion_modulator * final_fusion_weights_sensitivity.get('fuel', 0.5)
            nozzle_mod = -normalized_fusion_modulator * final_fusion_weights_sensitivity.get('nozzle', 0.5)
            final_engine_weight = (final_fusion_weights_base.get('engine', 0.33) + engine_mod).clip(0.1, 0.6)
            final_fuel_weight = (final_fusion_weights_base.get('fuel', 0.33) + fuel_mod).clip(0.1, 0.6)
            final_nozzle_weight = (final_fusion_weights_base.get('nozzle', 0.34) + nozzle_mod).clip(0.1, 0.6)
            sum_dynamic_fusion_weights = final_engine_weight + final_fuel_weight + final_nozzle_weight
            final_engine_weight = final_engine_weight / sum_dynamic_fusion_weights
            final_fuel_weight = final_fuel_weight / sum_dynamic_fusion_weights
            final_nozzle_weight = final_nozzle_weight / sum_dynamic_fusion_weights
        final_score = (
            engine_score_normalized.pow(final_engine_weight) *
            fuel_score_normalized.pow(final_fuel_weight) *
            nozzle_score_normalized.pow(final_nozzle_weight)
        ).pow(1 / (final_engine_weight + final_fuel_weight + final_nozzle_weight)) * 2 - 1
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"       - 内部调试: final_score (raw): {final_score.loc[probe_date]}")
                print(f"    -> [结构性推力探针] @ {probe_date.date()}:")
                print(f"       - 参数: health_weights: {health_weights}")
                print(f"       - 参数: engine_power_dynamic_weight_modulator_signal_name: {engine_power_dynamic_weight_modulator_signal_name}, engine_power_dynamic_weight_sensitivity: {engine_power_dynamic_weight_sensitivity:.2f}")
                print(f"       - 参数: static_engine_power_base_weight: {static_engine_power_base_weight:.2f}, dynamic_engine_power_base_weight: {dynamic_engine_power_base_weight:.2f}")
                print(f"       - 参数: fuel_purity_deception_penalty_factor: {fuel_purity_deception_penalty_factor:.2f}")
                print(f"       - 参数: synergy_bonus_base: {synergy_bonus_base:.2f}, synergy_bonus_context_modulator_signal_name: {synergy_bonus_context_modulator_signal_name}, synergy_bonus_context_sensitivity: {synergy_bonus_context_sensitivity:.2f}")
                print(f"       - 参数: nozzle_efficiency_weights: {nozzle_efficiency_weights}")
                print(f"       - 参数: final_fusion_dynamic_weights_enabled: {final_fusion_dynamic_weights_enabled}")
                print(f"       - 参数: final_fusion_weights_base: {final_fusion_weights_base}, final_fusion_weights_sensitivity: {final_fusion_weights_sensitivity}")
                print(f"       - 原料: strategic_posture: {strategic_posture.loc[probe_date]:.4f}, battlefield_geography: {battlefield_geography.loc[probe_date]:.4f}, holder_sentiment: {holder_sentiment.loc[probe_date]:.4f}")
                print(f"       - 原料: {engine_power_dynamic_weight_modulator_signal_name}: {health_score_slope_raw.loc[probe_date]:.4f}, norm_health_score_slope: {norm_health_score_slope.loc[probe_date]:.4f}")
                print(f"       - 原料: main_force_conviction_index_D: {conviction_raw.loc[probe_date]:.4f}, upward_impulse_purity_D: {impulse_purity_raw.loc[probe_date]:.4f}")
                print(f"       - 原料: chip_fault_magnitude_D: {chip_fault_raw.loc[probe_date]:.4f}, norm_chip_fault: {norm_chip_fault.loc[probe_date]:.4f}")
                print(f"       - 原料: {synergy_bonus_context_modulator_signal_name}: {synergy_context_raw.loc[probe_date]:.4f}, norm_synergy_context: {norm_synergy_context.loc[probe_date]:.4f}")
                print(f"       - 原料: vacuum_zone_magnitude_D: {vacuum_magnitude_raw.loc[probe_date]:.4f}, SLOPE_5_vacuum_zone_magnitude_D: {vacuum_trend_raw.loc[probe_date]:.4f}, vacuum_traversal_efficiency_D: {vacuum_traversal_raw.loc[probe_date]:.4f}")
                if final_fusion_dynamic_weights_enabled:
                    print(f"       - 原料: fusion_modulator_raw (from strategic_posture): {fusion_modulator_raw.loc[probe_date]:.4f}, normalized_fusion_modulator: {normalized_fusion_modulator.loc[probe_date]:.4f}")
                print(f"       - 过程: static_engine_power: {static_engine_power.loc[probe_date]:.4f}")
                print(f"       - 过程: dynamic_weight_mod: {dynamic_weight_mod.loc[probe_date]:.4f}, current_static_weight: {current_static_weight.loc[probe_date]:.4f}, current_dynamic_weight: {current_dynamic_weight.loc[probe_date]:.4f}")
                print(f"       - 过程: norm_slope (health): {norm_slope.loc[probe_date]:.4f}, norm_accel (health): {norm_accel.loc[probe_date]:.4f}, dynamic_engine_power: {dynamic_engine_power.loc[probe_date]:.4f}")
                print(f"       - 过程: engine_power_score: {engine_power_score.loc[probe_date]:.4f}")
                print(f"       - 过程: conviction_score: {conviction_score.loc[probe_date]:.4f}, purity_score: {purity_score.loc[probe_date]:.4f}, base_fuel_quality: {base_fuel_quality.loc[probe_date]:.4f}")
                print(f"       - 过程: deception_penalty: {deception_penalty.loc[probe_date]:.4f}, fuel_quality_score_after_deception: {fuel_quality_score_after_deception.loc[probe_date]:.4f}")
                print(f"       - 过程: dynamic_synergy_bonus_factor: {dynamic_synergy_bonus_factor.loc[probe_date]:.4f}, synergy_bonus: {synergy_bonus.loc[probe_date]:.4f}")
                print(f"       - 过程: fuel_quality_score: {fuel_quality_score.loc[probe_date]:.4f}")
                print(f"       - 过程: norm_vacuum_magnitude: {norm_vacuum_magnitude.loc[probe_date]:.4f}, norm_vacuum_trend: {norm_vacuum_trend.loc[probe_date]:.4f}, norm_traversal_efficiency: {norm_traversal_efficiency.loc[probe_date]:.4f}")
                print(f"       - 过程: nozzle_efficiency_score: {nozzle_efficiency_score.loc[probe_date]:.4f}")
                if final_fusion_dynamic_weights_enabled:
                    print(f"       - 过程: normalized_fusion_modulator: {normalized_fusion_modulator.loc[probe_date]:.4f}")
                    print(f"       - 过程: final_engine_weight: {final_engine_weight.loc[probe_date]:.4f}, final_fuel_weight: {final_fuel_weight.loc[probe_date]:.4f}, final_nozzle_weight: {final_nozzle_weight.loc[probe_date]:.4f}")
                print(f"       - 结果: final_score: {final_score.loc[probe_date]:.4f}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V7.2 · 情境自适应张力版】筹码公理五：诊断“价筹张力”
        - 核心数学升级1: 将“主力共谋验证”从依赖资金流信号升级为更纯粹、更稳健的“主力筹码意图验证”模型。
                          该模型直接评估1)主力筹码信念是否与背离方向一致(同谋), 2)主力信念强度是否足够大(兵力)。
                          只有当两者都满足时，才确认为一次高置信度的“战术性背离”，并给予显著加成。
        - 核心数学升级2: “筹码趋势”的多元化解读。引入赢家集中度与赢家/输家动量共同构建复合筹码趋势，更全面捕捉筹码结构与价格的分歧。
        - 核心数学升级3: “持续性”的优化。将持续性量化为分歧方向的一致性累积，而非波动性，更准确反映张力积蓄。
        - 核心数学升级4: “能量注入”的筹码化。替换通用成交量为建设性换手率，更精准反映筹码层面的活跃度与质量。
        - 核心数学升级5: “诡道双向调制”。引入筹码故障幅度对分歧强度进行情境调制，根据故障与分歧方向的匹配关系，动态地放大或削弱价筹张力信号。
        - 核心数学升级6: “情境自适应放大器”。引入筹码健康度作为情境调制器，动态调整张力强度和主力意图验证的放大倍数。
        - 核心数学升级7: “非线性放大控制”。对放大项引入tanh变换，使其增长更平滑，并有饱和上限，防止过度放大。
        - 核心数学升级8: “动态复合筹码趋势权重”。引入筹码波动不稳定性指数作为调制器，自适应调整复合筹码趋势中动量和集中度的权重。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        - 修复: 修正 get_adaptive_mtf_normalized_score 函数的参数传递错误。
        """
        print("    -> [筹码层] 正在诊断“价筹张力”公理 (V7.2 · 情境自适应张力版)...")
        required_signals = [
            'winner_loser_momentum_D', 'winner_concentration_90pct_D', 'SLOPE_5_close_D',
            'constructive_turnover_ratio_D', 'main_force_conviction_index_D', 'chip_fault_magnitude_D',
            'chip_health_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        divergence_params = get_param_value(p_conf.get('divergence_params'), {})
        chip_trend_momentum_weight_base = get_param_value(divergence_params.get('chip_trend_momentum_weight'), 0.6)
        chip_trend_concentration_weight_base = get_param_value(divergence_params.get('chip_trend_concentration_weight'), 0.4)
        tension_magnitude_amplifier_base = get_param_value(divergence_params.get('tension_magnitude_amplifier'), 1.5)
        chip_intent_factor_amplifier_base = get_param_value(divergence_params.get('chip_intent_factor_amplifier'), 0.5)
        deception_modulator_impact_clip = get_param_value(divergence_params.get('deception_modulator_impact_clip'), 0.5)
        conflict_bonus = get_param_value(divergence_params.get('conflict_bonus'), 0.5)
        deception_modulator_reinforce_factor = get_param_value(divergence_params.get('deception_modulator_reinforce_factor'), 0.5)
        contextual_amplification_enabled = get_param_value(divergence_params.get('contextual_amplification_enabled'), True)
        context_modulator_signal_name = get_param_value(divergence_params.get('context_modulator_signal_name'), 'chip_health_score_D')
        context_sensitivity_tension = get_param_value(divergence_params.get('context_sensitivity_tension'), 0.5)
        context_sensitivity_intent = get_param_value(divergence_params.get('context_sensitivity_intent'), 0.5)
        non_linear_amplification_enabled = get_param_value(divergence_params.get('non_linear_amplification_enabled'), True)
        non_linear_amp_tanh_factor = get_param_value(divergence_params.get('non_linear_amp_tanh_factor'), 1.0)
        dynamic_chip_trend_weights_enabled = get_param_value(divergence_params.get('dynamic_chip_trend_weights_enabled'), True)
        chip_trend_weight_modulator_signal_name = get_param_value(divergence_params.get('chip_trend_weight_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_trend_weight_mod_sensitivity = get_param_value(divergence_params.get('chip_trend_weight_mod_sensitivity'), 0.5)
        df_index = df.index
        dynamic_momentum_weight = pd.Series(chip_trend_momentum_weight_base, index=df_index)
        dynamic_concentration_weight = pd.Series(chip_trend_concentration_weight_base, index=df_index)
        if dynamic_chip_trend_weights_enabled:
            chip_trend_modulator_raw = self._get_safe_series(df, df, chip_trend_weight_modulator_signal_name, 0.0, method_name="_diagnose_axiom_divergence")
            normalized_chip_trend_modulator = get_adaptive_mtf_normalized_score(chip_trend_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
            dynamic_momentum_weight = chip_trend_momentum_weight_base * (1 + normalized_chip_trend_modulator * chip_trend_weight_mod_sensitivity)
            dynamic_concentration_weight = chip_trend_concentration_weight_base * (1 - normalized_chip_trend_modulator * chip_trend_weight_mod_sensitivity)
            sum_dynamic_weights = dynamic_momentum_weight + dynamic_concentration_weight
            dynamic_momentum_weight = (dynamic_momentum_weight / sum_dynamic_weights).clip(0.1, 0.9)
            dynamic_concentration_weight = (dynamic_concentration_weight / sum_dynamic_weights).clip(0.1, 0.9)
        chip_momentum_raw = self._get_safe_series(df, df, 'winner_loser_momentum_D', 0.0, method_name="_diagnose_axiom_divergence")
        chip_concentration_raw = self._get_safe_series(df, df, 'winner_concentration_90pct_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_chip_momentum = get_adaptive_mtf_normalized_bipolar_score(chip_momentum_raw, df_index, tf_weights)
        norm_chip_concentration = get_adaptive_mtf_normalized_bipolar_score(chip_concentration_raw, df_index, tf_weights)
        composite_chip_trend = (
            norm_chip_momentum * dynamic_momentum_weight +
            norm_chip_concentration * dynamic_concentration_weight
        )
        price_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_close_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, tf_weights)
        disagreement_vector = composite_chip_trend - norm_price_trend
        persistence_raw = np.sign(disagreement_vector).rolling(window=13, min_periods=5).sum().fillna(0)
        norm_persistence = get_adaptive_mtf_normalized_score(persistence_raw.abs(), df_index, tf_weights=tf_weights)
        constructive_turnover_raw = self._get_safe_series(df, df, 'constructive_turnover_ratio_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_constructive_turnover = get_adaptive_mtf_normalized_score(constructive_turnover_raw, df_index, tf_weights=tf_weights)
        energy_injection = norm_constructive_turnover * disagreement_vector.abs()
        tension_magnitude = (norm_persistence * energy_injection).pow(0.5)
        mf_chip_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0)
        norm_mf_chip_conviction = get_adaptive_mtf_normalized_bipolar_score(mf_chip_conviction_raw, df_index, tf_weights)
        is_aligned = (np.sign(disagreement_vector) * np.sign(norm_mf_chip_conviction)) > 0
        intent_strength = norm_mf_chip_conviction.abs()
        chip_intent_verification_score = is_aligned * intent_strength
        dynamic_tension_amplifier = pd.Series(tension_magnitude_amplifier_base, index=df_index)
        dynamic_chip_intent_factor_amplifier = pd.Series(chip_intent_factor_amplifier_base, index=df_index)
        if contextual_amplification_enabled:
            context_modulator_raw = self._get_safe_series(df, df, context_modulator_signal_name, 0.0, method_name="_diagnose_axiom_divergence")
            normalized_context = get_adaptive_mtf_normalized_score(context_modulator_raw, df_index, tf_weights=tf_weights, ascending=True)
            dynamic_tension_amplifier = tension_magnitude_amplifier_base * (1 + normalized_context * context_sensitivity_tension)
            dynamic_chip_intent_factor_amplifier = chip_intent_factor_amplifier_base * (1 + normalized_context * context_sensitivity_intent)
            dynamic_tension_amplifier = dynamic_tension_amplifier.clip(tension_magnitude_amplifier_base * 0.5, tension_magnitude_amplifier_base * 2.0)
            dynamic_chip_intent_factor_amplifier = dynamic_chip_intent_factor_amplifier.clip(chip_intent_factor_amplifier_base * 0.5, chip_intent_factor_amplifier_base * 2.0)
        tension_amplification_term = tension_magnitude * dynamic_tension_amplifier
        chip_intent_amplification_term = chip_intent_verification_score * dynamic_chip_intent_factor_amplifier
        if non_linear_amplification_enabled:
            tension_amplification_term = np.tanh(tension_amplification_term * non_linear_amp_tanh_factor)
            chip_intent_amplification_term = np.tanh(chip_intent_amplification_term * non_linear_amp_tanh_factor)
        chip_intent_factor = 1.0 + chip_intent_amplification_term
        chip_fault_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', 0.0)
        norm_chip_fault = get_adaptive_mtf_normalized_score(chip_fault_raw.abs(), df_index, ascending=True, tf_weights=tf_weights) # [修改代码行] 修正参数传递
        divergence_sign = np.sign(disagreement_vector)
        fault_sign = np.sign(chip_fault_raw)
        deception_modulator_factor = pd.Series(1.0, index=df_index)
        align_mask = (divergence_sign == fault_sign)
        deception_modulator_factor.loc[align_mask] = 1 - norm_chip_fault.loc[align_mask] * deception_modulator_impact_clip
        oppose_mask = (divergence_sign != fault_sign)
        deception_modulator_factor.loc[oppose_mask] = 1 + norm_chip_fault.loc[oppose_mask] * deception_modulator_reinforce_factor
        deception_modulator_factor = deception_modulator_factor.clip(0.1, 2.0)
        base_final_score = disagreement_vector * (1 + tension_amplification_term) * chip_intent_factor * deception_modulator_factor
        conflict_mask = (np.sign(composite_chip_trend) * np.sign(norm_price_trend) < 0)
        conflict_amplifier = pd.Series(1.0, index=df_index)
        conflict_amplifier.loc[conflict_mask] = 1.0 + conflict_bonus
        safe_base_score = base_final_score.clip(-0.999, 0.999)
        final_score = np.tanh(np.arctanh(safe_base_score) * conflict_amplifier)
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_structural_consensus(self, df: pd.DataFrame, cost_structure_scores: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series:
        """
        【V7.18 · 最终分数敏感度动态版 (生产就绪版)】诊断筹码同调驱动力
        一个基于“引擎-传动”思想的终极信号，旨在量化筹码结构对上涨意愿的真实转化效率。
        它将“持股心态”视为提供上涨意愿的引擎，将“成本结构”视为决定能量损耗的传动系统。
        核心升级:
        - 筹码健康度 `chip_health_score_D` 作为非线性调制参数（amplification_power, dampening_power）的动态调节器。
        - 筹码健康度对幂指数的敏感度根据另一个筹码层面的信号（例如 `VOLATILITY_INSTABILITY_INDEX_21d_D` 筹码波动性）进行动态调整。
        - 筹码结构分数 `cost_structure_scores` 对情绪驱动力的调制强度，根据持股心态 `holder_sentiment_scores` 的正负方向，进行非对称的非线性动态调整。
        - 情绪与筹码结构之间的耦合强度也实现了动态调整。
        - 筹码健康度调制敏感度引入了非对称性。
        - 动态中性阈值使得判断情绪和筹码结构是看涨/看跌或顺风/逆风的“中性”界限，将根据筹码健康度动态调整。
        - 情绪激活阈值使得持股心态的原始强度在参与驱动力计算之前，会根据其与动态中性阈值的相对关系进行“激活”或“去激活”处理。
        - 情绪强度对筹码结构调制效果的动态缩放，激活后的情绪强度将动态缩放筹码结构分数对驱动力的最终影响。
        - 结构强度对幂指数的自适应调整，amplification_power 和 dampening_power 将根据最终用于调制的筹码结构分数的绝对强度进行进一步的动态调整。
        - 结构强度对幂指数自适应调整的敏感度动态调制，使得模型在不同市场环境下对筹码结构信号的反应更加精细和智能。
        - 结构强度对幂指数自适应调整的非对称非线性映射，为正向和负向结构强度引入独立的 tanh 因子和可选的偏移量。
        - 最终分数敏感度的动态调整，final_score 的饱和速度将根据市场环境进行动态调整。
        高分代表市场不仅想涨，而且其内部筹码结构健康且具备高效转化这种意愿的能力。
        """
        print("    -> [筹码层] 正在诊断“同调驱动力 (V7.18 · 最终分数敏感度动态版 (生产就绪版))”...") # [修改代码行]
        
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        coherent_drive_params = get_param_value(p_conf.get('coherent_drive_params'), {})
        
        base_amplification_power = get_param_value(coherent_drive_params.get('amplification_power'), 1.2)
        base_dampening_power = get_param_value(coherent_drive_params.get('dampening_power'), 1.5)

        chip_health_modulation_enabled = get_param_value(coherent_drive_params.get('chip_health_modulation_enabled'), True)
        default_chip_health_sensitivity_amp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp'), 0.5)
        default_chip_health_sensitivity_damp = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp'), 0.5)
        
        chip_health_mtf_norm_params = get_param_value(coherent_drive_params.get('chip_health_mtf_norm_params'), {})
        chip_health_tanh_factor_amp = get_param_value(coherent_drive_params.get('chip_health_tanh_factor_amp'), 1.0)
        chip_health_tanh_factor_damp = get_param_value(coherent_drive_params.get('chip_health_tanh_factor_damp'), 1.0)

        chip_health_sensitivity_modulation_enabled = get_param_value(coherent_drive_params.get('chip_health_sensitivity_modulation_enabled'), False)
        chip_sensitivity_modulator_signal_name = get_param_value(coherent_drive_params.get('chip_sensitivity_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        chip_sensitivity_mod_norm_window = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_norm_window'), 21)
        chip_sensitivity_mod_factor_amp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_factor_amp'), 1.0)
        chip_sensitivity_mod_factor_damp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_factor_damp'), 1.0)
        chip_sensitivity_mod_tanh_factor_amp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_tanh_factor_amp'), 1.0)
        chip_sensitivity_mod_tanh_factor_damp = get_param_value(coherent_drive_params.get('chip_sensitivity_mod_tanh_factor_damp'), 1.0)

        cost_structure_asymmetric_impact_enabled = get_param_value(coherent_drive_params.get('cost_structure_asymmetric_impact_enabled'), False)
        cost_structure_impact_base_factor_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_base_factor_bullish'), 1.0)
        cost_structure_impact_base_factor_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_base_factor_bearish'), 1.0)
        cost_structure_impact_sentiment_sensitivity_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_sensitivity_bullish'), 1.0)
        cost_structure_impact_sentiment_sensitivity_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_sensitivity_bearish'), 1.0)
        cost_structure_impact_sentiment_tanh_factor_bullish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_tanh_factor_bullish'), 1.0)
        cost_structure_impact_sentiment_tanh_factor_bearish = get_param_value(coherent_drive_params.get('cost_structure_impact_sentiment_tanh_factor_bearish'), 1.0)

        sentiment_cost_structure_coupling_enabled = get_param_value(coherent_drive_params.get('sentiment_cost_structure_coupling_enabled'), False)
        sentiment_coupling_base_factor = get_param_value(coherent_drive_params.get('sentiment_coupling_base_factor'), 1.0)
        sentiment_coupling_tanh_factor = get_param_value(coherent_drive_params.get('sentiment_coupling_tanh_factor'), 1.0)
        sentiment_coupling_sensitivity = get_param_value(coherent_drive_params.get('sentiment_coupling_sensitivity'), 1.0)

        chip_health_asymmetric_sensitivity_enabled = get_param_value(coherent_drive_params.get('chip_health_asymmetric_sensitivity_enabled'), False)
        chip_health_sensitivity_amp_positive_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp_positive_health'), 0.5)
        chip_health_sensitivity_amp_negative_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_amp_negative_health'), 0.5)
        chip_health_sensitivity_damp_positive_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp_positive_health'), 0.5)
        chip_health_sensitivity_damp_negative_health = get_param_value(coherent_drive_params.get('chip_health_sensitivity_damp_negative_health'), 0.5)

        dynamic_neutrality_thresholds_enabled = get_param_value(coherent_drive_params.get('dynamic_neutrality_thresholds_enabled'), False)
        sentiment_neutrality_base_threshold = get_param_value(coherent_drive_params.get('sentiment_neutrality_base_threshold'), 0.0)
        sentiment_neutrality_chip_health_sensitivity = get_param_value(coherent_drive_params.get('sentiment_neutrality_chip_health_sensitivity'), 0.1)
        cost_structure_neutrality_base_threshold = get_param_value(coherent_drive_params.get('cost_structure_neutrality_base_threshold'), 0.0)
        cost_structure_neutrality_chip_health_sensitivity = get_param_value(coherent_drive_params.get('cost_structure_neutrality_chip_health_sensitivity'), 0.1)

        sentiment_activation_enabled = get_param_value(coherent_drive_params.get('sentiment_activation_enabled'), False)
        sentiment_activation_tanh_factor = get_param_value(coherent_drive_params.get('sentiment_activation_tanh_factor'), 1.0)
        sentiment_activation_strength = get_param_value(coherent_drive_params.get('sentiment_activation_strength'), 1.0)

        structure_modulation_strength_enabled = get_param_value(coherent_drive_params.get('structure_modulation_strength_enabled'), False)
        structure_modulation_base_strength = get_param_value(coherent_drive_params.get('structure_modulation_base_strength'), 1.0)
        structure_modulation_sentiment_tanh_factor = get_param_value(coherent_drive_params.get('structure_modulation_sentiment_tanh_factor'), 1.0)
        structure_modulation_sentiment_sensitivity = get_param_value(coherent_drive_params.get('structure_modulation_sentiment_sensitivity'), 1.0)

        structural_power_adjustment_enabled = get_param_value(coherent_drive_params.get('structural_power_adjustment_enabled'), False)
        default_structural_power_sensitivity_amp = get_param_value(coherent_drive_params.get('structural_power_sensitivity_amp'), 0.5)
        default_structural_power_sensitivity_damp = get_param_value(coherent_drive_params.get('structural_power_sensitivity_damp'), 0.5)
        default_structural_power_tanh_factor_amp = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_amp'), 1.0)
        default_structural_power_tanh_factor_damp = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_damp'), 1.0)

        structural_power_sensitivity_modulation_enabled = get_param_value(coherent_drive_params.get('structural_power_sensitivity_modulation_enabled'), False)
        structural_power_modulator_signal_name = get_param_value(coherent_drive_params.get('structural_power_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        structural_power_mod_norm_window = get_param_value(coherent_drive_params.get('structural_power_mod_norm_window'), 21)
        structural_power_mod_factor_amp = get_param_value(coherent_drive_params.get('structural_power_mod_factor_amp'), 1.0)
        structural_power_mod_factor_damp = get_param_value(coherent_drive_params.get('structural_power_mod_factor_damp'), 1.0)
        structural_power_mod_tanh_factor_amp = get_param_value(coherent_drive_params.get('structural_power_mod_tanh_factor_amp'), 1.0)
        structural_power_mod_tanh_factor_damp = get_param_value(coherent_drive_params.get('structural_power_mod_tanh_factor_damp'), 1.0)

        structural_power_asymmetric_tanh_enabled = get_param_value(coherent_drive_params.get('structural_power_asymmetric_tanh_enabled'), False)
        structural_power_tanh_factor_positive_structure = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_positive_structure'), 1.0)
        structural_power_tanh_factor_negative_structure = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_negative_structure'), 1.0)
        structural_power_offset_positive_structure = get_param_value(coherent_drive_params.get('structural_power_offset_positive_structure'), 0.0)
        structural_power_offset_negative_structure = get_param_value(coherent_drive_params.get('structural_power_offset_negative_structure'), 0.0)

        final_score_sensitivity_modulation_enabled = get_param_value(coherent_drive_params.get('final_score_sensitivity_modulation_enabled'), False)
        final_score_modulator_signal_name = get_param_value(coherent_drive_params.get('final_score_modulator_signal_name'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        final_score_mod_norm_window = get_param_value(coherent_drive_params.get('final_score_mod_norm_window'), 21)
        final_score_mod_factor = get_param_value(coherent_drive_params.get('final_score_mod_factor'), 1.0)
        final_score_mod_tanh_factor = get_param_value(coherent_drive_params.get('final_score_mod_tanh_factor'), 1.0)
        final_score_base_sensitivity_multiplier = get_param_value(coherent_drive_params.get('final_score_base_sensitivity_multiplier'), 2.0)

        amplification_power = pd.Series(base_amplification_power, index=df.index)
        dampening_power = pd.Series(base_dampening_power, index=df.index)
        modulation_factor = pd.Series(1.0, index=df.index)

        current_chip_health_score_raw = pd.Series(0.0, index=df.index)
        normalized_chip_health = pd.Series(0.0, index=df.index)
        dynamic_chip_health_sensitivity_amp = pd.Series(default_chip_health_sensitivity_amp, index=df.index)
        dynamic_chip_health_sensitivity_damp = pd.Series(default_chip_health_sensitivity_damp, index=df.index)
        dynamic_cost_structure_impact_factor_bullish = pd.Series(cost_structure_impact_base_factor_bullish, index=df.index)
        dynamic_cost_structure_impact_factor_bearish = pd.Series(cost_structure_impact_base_factor_bearish, index=df.index)
        
        dynamic_coupling_factor = pd.Series(sentiment_coupling_base_factor, index=df.index)
        final_cost_structure_for_modulation = pd.Series(0.0, index=df.index)

        dynamic_sentiment_neutrality_threshold = pd.Series(sentiment_neutrality_base_threshold, index=df.index)
        dynamic_cost_structure_neutrality_threshold = pd.Series(cost_structure_neutrality_base_threshold, index=df.index)

        activated_holder_sentiment_scores = holder_sentiment_scores.copy()
        dynamic_structure_modulation_strength = pd.Series(structure_modulation_base_strength, index=df.index)
        final_cost_structure_for_modulation_scaled = pd.Series(0.0, index=df.index)
        dynamic_structural_power_sensitivity_amp = pd.Series(default_structural_power_sensitivity_amp, index=df.index)
        dynamic_structural_power_sensitivity_damp = pd.Series(default_structural_power_sensitivity_damp, index=df.index)
        dynamic_final_score_sensitivity_multiplier = pd.Series(final_score_base_sensitivity_multiplier, index=df.index)
        if chip_health_modulation_enabled:
            current_chip_health_score_raw = self._get_safe_series(df, df, 'chip_health_score_D', 0.0, method_name="_diagnose_structural_consensus")
            normalized_chip_health = get_adaptive_mtf_normalized_bipolar_score(
                current_chip_health_score_raw, 
                df.index, 
                tf_weights=chip_health_mtf_norm_params.get('weights', {}),
                sensitivity=chip_health_mtf_norm_params.get('sensitivity', 2.0)
            )
            base_amp_sensitivity_series = pd.Series(default_chip_health_sensitivity_amp, index=df.index)
            base_damp_sensitivity_series = pd.Series(default_chip_health_sensitivity_damp, index=df.index)
            if chip_health_asymmetric_sensitivity_enabled:
                positive_health_mask = normalized_chip_health > 0
                negative_health_mask = normalized_chip_health < 0
                base_amp_sensitivity_series.loc[positive_health_mask] = chip_health_sensitivity_amp_positive_health
                base_amp_sensitivity_series.loc[negative_health_mask] = chip_health_sensitivity_amp_negative_health
                base_damp_sensitivity_series.loc[positive_health_mask] = chip_health_sensitivity_damp_positive_health
                base_damp_sensitivity_series.loc[negative_health_mask] = chip_health_sensitivity_damp_negative_health
            if chip_health_sensitivity_modulation_enabled:
                modulator_signal_raw = self._get_safe_series(df, df, chip_sensitivity_modulator_signal_name, 0.0, method_name="_diagnose_structural_consensus")
                normalized_modulator_signal = normalize_score(
                    modulator_signal_raw,
                    df.index,
                    window=chip_sensitivity_mod_norm_window,
                    ascending=True
                )
                
                modulator_bipolar = (normalized_modulator_signal * 2) - 1
                non_linear_modulator_effect_amp = np.tanh(modulator_bipolar * chip_sensitivity_mod_tanh_factor_amp)
                non_linear_modulator_effect_damp = np.tanh(modulator_bipolar * chip_sensitivity_mod_tanh_factor_damp)

                dynamic_chip_health_sensitivity_amp = base_amp_sensitivity_series * (1 + non_linear_modulator_effect_amp * chip_sensitivity_mod_factor_amp)
                dynamic_chip_health_sensitivity_damp = base_damp_sensitivity_series * (1 + non_linear_modulator_effect_damp * chip_sensitivity_mod_factor_damp)

                dynamic_chip_health_sensitivity_amp = dynamic_chip_health_sensitivity_amp.clip(base_amp_sensitivity_series * 0.1, base_amp_sensitivity_series * 2.0)
                dynamic_chip_health_sensitivity_damp = dynamic_chip_health_sensitivity_damp.clip(base_damp_sensitivity_series * 0.1, base_damp_sensitivity_series * 2.0) # [修改代码行] 修正变量名
            else:
                dynamic_chip_health_sensitivity_amp = base_amp_sensitivity_series
                dynamic_chip_health_sensitivity_damp = base_damp_sensitivity_series
            modulated_chip_health_amp = np.tanh(normalized_chip_health * chip_health_tanh_factor_amp)
            modulated_chip_health_damp = np.tanh(normalized_chip_health * chip_health_tanh_factor_damp)
            amplification_power = base_amplification_power * (1 + modulated_chip_health_amp * dynamic_chip_health_sensitivity_amp)
            dampening_power = base_dampening_power * (1 - modulated_chip_health_damp * dynamic_chip_health_sensitivity_damp)
            amplification_power = amplification_power.clip(0.5, 2.0) 
            dampening_power = dampening_power.clip(0.5, 2.0) 
        if dynamic_neutrality_thresholds_enabled:
            dynamic_sentiment_neutrality_threshold = sentiment_neutrality_base_threshold + (normalized_chip_health * sentiment_neutrality_chip_health_sensitivity)
            dynamic_cost_structure_neutrality_threshold = cost_structure_neutrality_base_threshold + (normalized_chip_health * cost_structure_neutrality_chip_health_sensitivity)
            dynamic_sentiment_neutrality_threshold = dynamic_sentiment_neutrality_threshold.clip(-0.2, 0.2)
            dynamic_cost_structure_neutrality_threshold = dynamic_cost_structure_neutrality_threshold.clip(-0.2, 0.2)
        if sentiment_activation_enabled:
            positive_active_mask = holder_sentiment_scores > dynamic_sentiment_neutrality_threshold
            negative_active_mask = holder_sentiment_scores < -dynamic_sentiment_neutrality_threshold
            neutral_mask = ~(positive_active_mask | negative_active_mask)
            activated_holder_sentiment_scores.loc[positive_active_mask] = \
                holder_sentiment_scores.loc[positive_active_mask] - dynamic_sentiment_neutrality_threshold.loc[positive_active_mask]
            activated_holder_sentiment_scores.loc[negative_active_mask] = \
                holder_sentiment_scores.loc[negative_active_mask] + dynamic_sentiment_neutrality_threshold.loc[negative_active_mask]
            activated_holder_sentiment_scores.loc[neutral_mask] = 0.0
            activated_holder_sentiment_scores = np.tanh(activated_holder_sentiment_scores * sentiment_activation_tanh_factor) * sentiment_activation_strength
        if cost_structure_asymmetric_impact_enabled:
            positive_sentiment_mask = holder_sentiment_scores > 0
            if positive_sentiment_mask.any():
                positive_sentiment_strength = holder_sentiment_scores[positive_sentiment_mask]
                normalized_positive_sentiment_tanh = np.tanh(positive_sentiment_strength * cost_structure_impact_sentiment_tanh_factor_bullish)
                dynamic_cost_structure_impact_factor_bullish.loc[positive_sentiment_mask] = \
                    cost_structure_impact_base_factor_bullish * (1 + (normalized_positive_sentiment_tanh - 0.5) * cost_structure_impact_sentiment_sensitivity_bullish)
                dynamic_cost_structure_impact_factor_bullish = dynamic_cost_structure_impact_factor_bullish.clip(0.1, 2.0)
            negative_sentiment_mask = holder_sentiment_scores < 0
            if negative_sentiment_mask.any():
                negative_sentiment_strength = holder_sentiment_scores[negative_sentiment_mask].abs()
                normalized_negative_sentiment_tanh = np.tanh(negative_sentiment_strength * cost_structure_impact_sentiment_tanh_factor_bearish)
                dynamic_cost_structure_impact_factor_bearish.loc[negative_sentiment_mask] = \
                    cost_structure_impact_base_factor_bearish * (1 + (normalized_negative_sentiment_tanh - 0.5) * cost_structure_impact_sentiment_sensitivity_bearish)
                dynamic_cost_structure_impact_factor_bearish = dynamic_cost_structure_impact_factor_bearish.clip(0.1, 2.0)
        selected_dynamic_cost_structure_impact_factor = pd.Series(1.0, index=df.index)
        selected_dynamic_cost_structure_impact_factor.loc[holder_sentiment_scores > 0] = dynamic_cost_structure_impact_factor_bullish.loc[holder_sentiment_scores > 0]
        selected_dynamic_cost_structure_impact_factor.loc[holder_sentiment_scores < 0] = dynamic_cost_structure_impact_factor_bearish.loc[holder_sentiment_scores < 0]
        adjusted_cost_structure_scores = cost_structure_scores * selected_dynamic_cost_structure_impact_factor
        if sentiment_cost_structure_coupling_enabled:
            abs_holder_sentiment = holder_sentiment_scores.abs()
            sentiment_tanh_modulated = np.tanh(abs_holder_sentiment * sentiment_coupling_tanh_factor)
            dynamic_coupling_factor = sentiment_coupling_base_factor * (1 + sentiment_tanh_modulated * sentiment_coupling_sensitivity)
            dynamic_coupling_factor = dynamic_coupling_factor.clip(0.1, 2.0)
        final_cost_structure_for_modulation = adjusted_cost_structure_scores * dynamic_coupling_factor
        if structure_modulation_strength_enabled:
            abs_activated_sentiment = activated_holder_sentiment_scores.abs()
            sentiment_tanh_modulated_for_structure = np.tanh(abs_activated_sentiment * structure_modulation_sentiment_tanh_factor)
            dynamic_structure_modulation_strength = structure_modulation_base_strength * (1 + sentiment_tanh_modulated_for_structure * structure_modulation_sentiment_sensitivity)
            dynamic_structure_modulation_strength = dynamic_structure_modulation_strength.clip(0.1, 2.0)
        final_cost_structure_for_modulation_scaled = final_cost_structure_for_modulation * dynamic_structure_modulation_strength
        if structural_power_sensitivity_modulation_enabled:
            structural_power_modulator_signal_raw = self._get_safe_series(df, df, structural_power_modulator_signal_name, 0.0, method_name="_diagnose_structural_consensus")
            structural_power_normalized_modulator_signal = normalize_score(
                structural_power_modulator_signal_raw,
                df.index,
                window=structural_power_mod_norm_window,
                ascending=True
            )
            structural_power_modulator_bipolar = (structural_power_normalized_modulator_signal * 2) - 1
            structural_power_non_linear_modulator_effect_amp = np.tanh(structural_power_modulator_bipolar * structural_power_mod_tanh_factor_amp)
            structural_power_non_linear_modulator_effect_damp = np.tanh(structural_power_modulator_bipolar * structural_power_mod_tanh_factor_damp)
            dynamic_structural_power_sensitivity_amp = default_structural_power_sensitivity_amp * (1 + structural_power_non_linear_modulator_effect_amp * structural_power_mod_factor_amp)
            dynamic_structural_power_sensitivity_damp = default_structural_power_sensitivity_damp * (1 + structural_power_non_linear_modulator_effect_damp * structural_power_mod_factor_damp)
            dynamic_structural_power_sensitivity_amp = dynamic_structural_power_sensitivity_amp.clip(default_structural_power_sensitivity_amp * 0.1, default_structural_power_sensitivity_amp * 2.0)
            dynamic_structural_power_sensitivity_damp = dynamic_structural_power_sensitivity_damp.clip(default_structural_power_sensitivity_damp * 0.1, default_structural_power_sensitivity_damp * 2.0)
        else:
            dynamic_structural_power_sensitivity_amp = pd.Series(default_structural_power_sensitivity_amp, index=df.index)
            dynamic_structural_power_sensitivity_damp = pd.Series(default_structural_power_sensitivity_damp, index=df.index)
        if structural_power_adjustment_enabled:
            positive_structure_mask = final_cost_structure_for_modulation_scaled > 0
            negative_structure_mask = final_cost_structure_for_modulation_scaled < 0
            if positive_structure_mask.any():
                positive_structure_strength = final_cost_structure_for_modulation_scaled[positive_structure_mask]
                if structural_power_asymmetric_tanh_enabled:
                    boost_amp = np.tanh((positive_structure_strength + structural_power_offset_positive_structure) * structural_power_tanh_factor_positive_structure) * dynamic_structural_power_sensitivity_amp.loc[positive_structure_mask]
                else:
                    boost_amp = np.tanh(positive_structure_strength * default_structural_power_tanh_factor_amp) * dynamic_structural_power_sensitivity_amp.loc[positive_structure_mask]
                amplification_power.loc[positive_structure_mask] = amplification_power.loc[positive_structure_mask] * (1 + boost_amp)
            if negative_structure_mask.any():
                negative_structure_strength = final_cost_structure_for_modulation_scaled[negative_structure_mask].abs()
                if structural_power_asymmetric_tanh_enabled:
                    boost_damp = np.tanh((negative_structure_strength + structural_power_offset_negative_structure) * structural_power_tanh_factor_negative_structure) * dynamic_structural_power_sensitivity_damp.loc[negative_structure_mask]
                else:
                    boost_damp = np.tanh(negative_structure_strength * default_structural_power_tanh_factor_damp) * dynamic_structural_power_sensitivity_damp.loc[negative_structure_mask]
                dampening_power.loc[negative_structure_mask] = dampening_power.loc[negative_structure_mask] * (1 + boost_damp)
            amplification_power = amplification_power.clip(0.5, 3.0)
            dampening_power = dampening_power.clip(0.5, 3.0)
        bullish_mask = holder_sentiment_scores > dynamic_sentiment_neutrality_threshold
        bearish_mask = holder_sentiment_scores < -dynamic_sentiment_neutrality_threshold
        bullish_tailwind_mask = bullish_mask & (final_cost_structure_for_modulation_scaled > dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bullish_tailwind_mask] = (1 + final_cost_structure_for_modulation_scaled.loc[bullish_tailwind_mask]) ** amplification_power.loc[bullish_tailwind_mask]
        bullish_headwind_mask = bullish_mask & (final_cost_structure_for_modulation_scaled < -dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bullish_headwind_mask] = (1 - final_cost_structure_for_modulation_scaled.loc[bullish_headwind_mask].abs()) ** dampening_power.loc[bullish_headwind_mask]
        bearish_tailwind_mask = bearish_mask & (final_cost_structure_for_modulation_scaled < -dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bearish_tailwind_mask] = (1 + final_cost_structure_for_modulation_scaled.loc[bearish_tailwind_mask].abs()) ** amplification_power.loc[bearish_tailwind_mask]
        bearish_headwind_mask = bearish_mask & (final_cost_structure_for_modulation_scaled > dynamic_cost_structure_neutrality_threshold)
        modulation_factor.loc[bearish_headwind_mask] = (1 - final_cost_structure_for_modulation_scaled.loc[bearish_headwind_mask]) ** dampening_power.loc[bearish_headwind_mask]
        coherent_drive_raw = activated_holder_sentiment_scores * modulation_factor
        if final_score_sensitivity_modulation_enabled:
            final_score_modulator_signal_raw = self._get_safe_series(df, df, final_score_modulator_signal_name, 0.0, method_name="_diagnose_structural_consensus")
            final_score_normalized_modulator_signal = normalize_score(
                final_score_modulator_signal_raw,
                df.index,
                window=final_score_mod_norm_window,
                ascending=True
            )
            final_score_modulator_bipolar = (final_score_normalized_modulator_signal * 2) - 1
            final_score_non_linear_modulator_effect = np.tanh(final_score_modulator_bipolar * final_score_mod_tanh_factor)
            dynamic_final_score_sensitivity_multiplier = final_score_base_sensitivity_multiplier * (1 + final_score_non_linear_modulator_effect * final_score_mod_factor)
            dynamic_final_score_sensitivity_multiplier = dynamic_final_score_sensitivity_multiplier.clip(final_score_base_sensitivity_multiplier * 0.5, final_score_base_sensitivity_multiplier * 2.0)
        else:
            dynamic_final_score_sensitivity_multiplier = pd.Series(final_score_base_sensitivity_multiplier, index=df.index)
        final_score = np.tanh(coherent_drive_raw * (self.bipolar_sensitivity * dynamic_final_score_sensitivity_multiplier))
        return final_score.astype(np.float32)

    def _diagnose_absorption_echo(self, df: pd.DataFrame, divergence_score: pd.Series) -> pd.Series:
        """
        【V1.2 · 零点门控版】诊断“吸筹回声”信号
        - 核心修复: 解决“中性点悖论”。增加“零点门控”，当价格趋势为0时，强制将恐慌背景分置为0，
                      确保“无趋势则无信号”，根除因归一化映射产生的幽灵信号。
        """
        print("    -> [筹码层] 正在诊断“吸筹回声” (V1.2 · 零点门控版)...") # [修改代码行]
        required_signals = ['SLOPE_1_close_D', 'main_force_net_flow_calibrated_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_absorption_echo"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        price_trend = self._get_safe_series(df, df, 'SLOPE_1_close_D', 0.0)
        is_panic_context = price_trend < 0
        # 要素1: 恐慌声源 (Panic Source)
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend, df_index, tf_weights)
        panic_source_score_raw = (1 - norm_price_trend.add(1)/2).clip(0, 1)
        panic_source_score = panic_source_score_raw.where(price_trend != 0, 0) # [修改代码行] 增加零点门控
        # 要素2: 逆流介质 (Counter Flow Medium)
        counter_flow_medium_score = divergence_score.clip(0, 1)
        # 要素3: 主力回声 (Main Force Echo)
        mf_inflow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0)
        main_force_echo_score = get_adaptive_mtf_normalized_score(mf_inflow, df_index, tf_weights)
        final_score = (panic_source_score * counter_flow_medium_score * main_force_echo_score) * is_panic_context
        return final_score.clip(0, 1).fillna(0.0).astype(np.float32)

    def _diagnose_distribution_whisper(self, df: pd.DataFrame, divergence_score: pd.Series) -> pd.Series:
        """
        【V1.2 · 零点门控版】诊断“派发诡影”信号
        - 核心修复: 解决“中性点悖论”。增加“零点门控”，当价格趋势为0时，强制将狂热背景分置为0，
                      确保“无趋势则无信号”，实现完美的逻辑自洽。
        """
        print("    -> [筹码层] 正在诊断“派发诡影” (V1.2 · 零点门控版)...") # [修改代码行]
        required_signals = ['SLOPE_1_close_D', 'main_force_net_flow_calibrated_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_distribution_whisper"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        price_trend = self._get_safe_series(df, df, 'SLOPE_1_close_D', 0.0)
        is_fomo_context = price_trend > 0
        # 要素1: 狂热背景 (FOMO Backdrop)
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend, df_index, tf_weights)
        fomo_backdrop_score_raw = (norm_price_trend.add(1)/2).clip(0, 1)
        fomo_backdrop_score = fomo_backdrop_score_raw.where(price_trend != 0, 0) # [修改代码行] 增加零点门控
        # 要素2: 背离诡影 (Divergence Shadow)
        divergence_shadow_score = divergence_score.abs().clip(0, 1)
        # 要素3: 主力抽离 (Main Force Retreat)
        mf_inflow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0)
        main_force_retreat_score = get_adaptive_mtf_normalized_score(mf_inflow, df_index, ascending=False, tf_weights=tf_weights)
        final_score = (fomo_backdrop_score * divergence_shadow_score * main_force_retreat_score) * is_fomo_context
        return final_score.clip(0, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_historical_potential(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.4 · 信号换代版】筹码公理六：诊断“筹码势能”
        - 核心升级: 将评估长期筹码趋势的`SLOPE_55_winner_concentration_90pct_D`替换为更综合、更稳健的`chip_health_score_D`。
                      这使得对长期势能的评估从单一的“集中趋势”升级为对“结构健康度”的全面诊断。
        """
        print("    -> [筹码层] 正在诊断“筹码势能”公理 (V1.4 · 信号换代版)...")
        required_signals = [
            'main_force_net_flow_calibrated_D', 'chip_health_score_D',
            'dominant_peak_solidity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_historical_potential"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {21: 0.5, 55: 0.3, 89: 0.2})
        long_window = 250
        mf_net_flow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0)
        long_term_flow_accumulation = mf_net_flow.clip(lower=0).rolling(window=long_window, min_periods=55).sum()
        flow_score = get_adaptive_mtf_normalized_score(long_term_flow_accumulation, df_index, ascending=True, tf_weights=tf_weights)
        chip_health = self._get_safe_series(df, df, 'chip_health_score_D', 0.0)
        concentration_score_unipolar = get_adaptive_mtf_normalized_score(chip_health, df_index, ascending=True, tf_weights=tf_weights)
        peak_solidity = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5)
        stability_score = get_adaptive_mtf_normalized_score(peak_solidity, df_index, ascending=True, tf_weights=tf_weights)
        potential_score = (flow_score * 0.5 + concentration_score_unipolar * 0.3 + stability_score * 0.2)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [筹码势能探针] @ {probe_date_for_loop.date()}:")
                # 更新探针输出
                print(f"       - 原料: long_term_flow_accum: {long_term_flow_accumulation.loc[probe_date_for_loop]:.2f}, chip_health_score: {chip_health.loc[probe_date_for_loop]:.4f}, peak_solidity: {peak_solidity.loc[probe_date_for_loop]:.4f}")
                print(f"       - 过程: flow_score: {flow_score.loc[probe_date_for_loop]:.4f}, concentration_score_unipolar: {concentration_score_unipolar.loc[probe_date_for_loop]:.4f}, stability_score: {stability_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - 结果: final_potential_score: {potential_score.loc[probe_date_for_loop]:.4f}")
        return potential_score.clip(0, 1).astype(np.float32)

    def _diagnose_tactical_exchange(self, df: pd.DataFrame, battlefield_geography: pd.Series) -> pd.Series:
        """
        【V2.2 · 最终裁定 · 连续仲裁版】诊断战术换手博弈的质量与意图
        - 核心裁定: 将“诡道仲裁”模型从“硬阈值触发”升级为“连续非线性仲裁”。废除硬阈值，
                      仲裁的“话语权”由“诡道欺骗指数”的强度连续地、非线性地决定。这使得模型能更平滑、
                      更真实地融合常规意图与诡道意图，做出大师级的精妙裁决。
        """
        print("    -> [筹码层] 正在诊断“战术换手博弈 (V2.2 · 连续仲裁版)”...") # [修改代码行]
        required_signals = [
            'main_force_net_flow_calibrated_D', 'retail_net_flow_calibrated_D', 'turnover_rate_f_D',
            'peak_control_transfer_D', 'floating_chip_cleansing_efficiency_D', 'capitulation_absorption_index_D',
            'profit_realization_quality_D', 'BIAS_55_D', 'is_consolidating_D',
            'upward_impulse_purity_D', 'SLOPE_1_close_D', 'deception_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_tactical_exchange"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # 维度1: 换手意图 (Exchange Intent)
        power_transfer = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D') - self._get_safe_series(df, df, 'retail_net_flow_calibrated_D')
        turnover = self._get_safe_series(df, df, 'turnover_rate_f_D')
        control_transfer = self._get_safe_series(df, df, 'peak_control_transfer_D')
        deception_index = self._get_safe_series(df, df, 'deception_index_D')
        norm_power_transfer = get_adaptive_mtf_normalized_bipolar_score(power_transfer, df_index, tf_weights)
        norm_turnover = get_adaptive_mtf_normalized_score(turnover, df_index, tf_weights)
        norm_control_transfer = get_adaptive_mtf_normalized_bipolar_score(control_transfer, df_index, tf_weights)
        norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_index, df_index, tf_weights)
        # [修改代码块] 升级为连续非线性仲裁逻辑
        base_intent_score = (
            norm_power_transfer * 0.7 +
            norm_control_transfer * 0.3
        )
        # 仲裁权重由诡道指数的绝对强度非线性决定（平方使其对强信号更敏感）
        arbitration_weight = norm_deception.abs().pow(2)
        # 最终意图分 = (1 - 仲裁权重) * 基础分 + 仲裁权重 * 诡道分
        intent_score = base_intent_score * (1 - arbitration_weight) + norm_deception * arbitration_weight
        # 维度2: 换手质量 (Exchange Quality) - 零值门控
        price_trend = self._get_safe_series(df, df, 'SLOPE_1_close_D', 0.0)
        is_up_day = price_trend > 0
        absorption_idx = self._get_safe_series(df, df, 'capitulation_absorption_index_D')
        impulse_purity = self._get_safe_series(df, df, 'upward_impulse_purity_D')
        profit_quality = self._get_safe_series(df, df, 'profit_realization_quality_D')
        norm_absorption = get_adaptive_mtf_normalized_score(absorption_idx, df_index, tf_weights).where(absorption_idx != 0, 0)
        norm_impulse_purity = get_adaptive_mtf_normalized_score(impulse_purity, df_index, tf_weights).where(impulse_purity != 0, 0)
        bullish_quality = pd.Series(np.where(is_up_day, norm_impulse_purity, norm_absorption), index=df_index)
        bearish_quality = get_adaptive_mtf_normalized_score(profit_quality, df_index, tf_weights)
        quality_score = bullish_quality - bearish_quality
        # 维度3: 换手环境 (Exchange Context)
        geography = battlefield_geography
        bias = self._get_safe_series(df, df, 'BIAS_55_D')
        is_consolidating = self._get_safe_series(df, df, 'is_consolidating_D')
        norm_bias_risk = (bias / 0.3).clip(-1, 1)
        context_score = (geography * 0.6 - norm_bias_risk * 0.4) * (1 + is_consolidating * 0.2)
        weights = {'intent': 0.4, 'quality': 0.4, 'context': 0.2}
        final_score = (
            intent_score.clip(-1, 1) * weights['intent'] +
            quality_score.clip(-1, 1) * weights['quality'] +
            context_score.clip(-1, 1) * weights['context']
        )
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_strategic_tactical_harmony(self, df: pd.DataFrame, strategic_posture: pd.Series, tactical_exchange: pd.Series) -> pd.Series:
        """
        【V1.0 · 天人合一版】诊断战略与战术的和谐度
        - 核心算法: 融合主力的长期战略意图(strategic_posture)与当日战术执行(tactical_exchange)。
                      1. 计算以战略为重的“基础意图分”。
                      2. 计算衡量两者一致性的“和谐因子”。
                      3. 最终得分 = 基础意图分 × 和谐因子，以此量化两者之间的协同或冲突。
        """
        print("    -> [筹码层] 正在诊断“战略战术和谐度 (V1.0 · 天人合一版)”...")
        df_index = df.index
        # 1. 基础意图分 (战略权重更高)
        base_intent_score = strategic_posture * 0.6 + tactical_exchange * 0.4
        # 2. 和谐因子 (1:完全和谐, 0:完全冲突)
        harmony_factor = (1 - abs(strategic_posture - tactical_exchange) / 2).clip(lower=0)
        # 3. 最终裁决
        final_score = base_intent_score * harmony_factor
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_harmony_inflection(self, df: pd.DataFrame, harmony_score: pd.Series) -> pd.Series:
        """
        【V1.1 · 神笔版】诊断和谐度的反转拐点
        - 核心裁定: 引入“双正天条”。通过增加一道 `(velocity > 0) & (acceleration > 0)` 的逻辑门，
                      确保只有当和谐度的原始速度和加速度同时为正时，才计算拐点分数。
                      此举彻底修复了因归一化导致的逻辑漏洞，使信号只在最真实的“破晓”时刻触发。
        """
        print("    -> [筹码层] 正在诊断“和谐拐点 (V1.1 · 神笔版)”...") # [修改代码行]
        df_index = df.index
        # 1. 计算原始速度与加速度
        harmony_velocity = harmony_score.diff(1).fillna(0)
        harmony_acceleration = harmony_velocity.diff(1).fillna(0)
        # [修改代码块] 引入“双正天条”逻辑门
        is_inflection_gate = (harmony_velocity > 0) & (harmony_acceleration > 0)
        # 2. 归一化并计算“反转动能”
        norm_velocity = get_adaptive_mtf_normalized_score(harmony_velocity, df_index)
        norm_acceleration = get_adaptive_mtf_normalized_score(harmony_acceleration, df_index)
        reversal_momentum = (norm_velocity * norm_acceleration).pow(0.5)
        # 3. 计算“位置惩罚因子” (和谐度越低，因子越接近1)
        position_factor = (1 - harmony_score.clip(lower=0, upper=1))
        # 4. 最终融合，并应用“天条”门控
        final_score = reversal_momentum * position_factor * is_inflection_gate
        return final_score.clip(0, 1).fillna(0.0).astype(np.float32)



