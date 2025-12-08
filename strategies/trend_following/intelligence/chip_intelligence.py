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
        【V6.1 · 奖励机制版】筹码公理三：诊断“持仓信念韧性”
        - 核心升级: 将“恐慌盘吸收”的融合方式从“加权平均”升级为“奖励因子”。基础压力测试分由常规承接与防守构成，
                      只有当出现高质量的恐慌盘吸收时，才给予额外奖励。这更符合其“事件驱动”的本质，避免了在无恐慌日惩罚分数的问题。
        """
        print("    -> [筹码层] 正在诊断“持仓信念”公理 (V6.1 · 奖励机制版)...") # [修改代码行]
        required_signals = [
            'winner_stability_index_D', 'loser_pain_index_D', 'dip_absorption_power_D',
            'mf_cost_zone_defense_intent_D', 'retail_fomo_premium_index_D',
            'profit_realization_quality_D', 'capitulation_absorption_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_holder_sentiment"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        winner_stability = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        loser_pain = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        stability_score = get_adaptive_mtf_normalized_bipolar_score(winner_stability, df_index, tf_weights)
        pain_score = get_adaptive_mtf_normalized_bipolar_score(loser_pain, df_index, tf_weights)
        belief_core_score = (stability_score.add(1)/2 * pain_score.add(1)/2).pow(0.5) * 2 - 1
        absorption_power = self._get_safe_series(df, df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        defense_intent = self._get_safe_series(df, df, 'mf_cost_zone_defense_intent_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        capitulation_absorption = self._get_safe_series(df, df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        absorption_score = get_adaptive_mtf_normalized_bipolar_score(absorption_power, df_index, tf_weights)
        defense_score = get_adaptive_mtf_normalized_bipolar_score(defense_intent, df_index, tf_weights)
        capitulation_score = get_adaptive_mtf_normalized_score(capitulation_absorption, df_index, tf_weights)
        # [修改代码块] 升级为奖励机制
        base_pressure_score = ((absorption_score.add(1)/2 * defense_score.add(1)/2).pow(0.5) * 2 - 1)
        # 只有当恐慌吸收发生时，才给予奖励，奖励幅度由吸收质量决定
        capitulation_bonus = capitulation_score * 0.3 # 恐慌吸收的最大奖励为30%
        pressure_test_score = base_pressure_score * (1 + capitulation_bonus)
        fomo_index = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        profit_taking_quality = self._get_safe_series(df, df, 'profit_realization_quality_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        fomo_score = get_adaptive_mtf_normalized_score(fomo_index, df_index, ascending=True, tf_weights=tf_weights)
        profit_taking_score = get_adaptive_mtf_normalized_score(profit_taking_quality, df_index, ascending=True, tf_weights=tf_weights)
        impurity_score = (fomo_score * profit_taking_score).pow(0.5)
        conviction_base = ((belief_core_score.add(1)/2) * (pressure_test_score.clip(-1, 1).add(1)/2)).pow(0.5)
        final_score = (conviction_base * (1 - impurity_score)) * 2 - 1
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_trend_momentum(self, df: pd.DataFrame, periods: list, strategic_posture: pd.Series, battlefield_geography: pd.Series, holder_sentiment: pd.Series) -> pd.Series:
        """
        【V6.2 · 最终裁定 · 韧性引擎版】筹码公理六：诊断“结构性推力”
        - 核心裁定: 将“引擎功率”的基础健康分(health_score)计算模型从“几何平均”升级为“加权算术平均”。
                      此举旨在修复几何平均过于严苛的“一票否决”特性，使引擎健康度的评估更具韧性，
                      能够更均衡地反映多维度战况，避免因单一情绪指标的短期失真而导致战略误判。
        """
        print("    -> [筹码层] 正在诊断“结构性推力”公理 (V6.2 · 韧性引擎版)...") # [修改代码行]
        required_signals = [
            'main_force_conviction_index_D', 'vacuum_zone_magnitude_D', 'upward_impulse_purity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend_momentum"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # [修改代码块] 升级为加权算术平均，提高鲁棒性
        health_weights = {'posture': 0.4, 'geography': 0.4, 'sentiment': 0.2}
        health_score = (
            strategic_posture * health_weights['posture'] +
            battlefield_geography * health_weights['geography'] +
            holder_sentiment * health_weights['sentiment']
        )
        slope = health_score.diff(1).fillna(0)
        accel = slope.diff(1).fillna(0)
        norm_slope = get_adaptive_mtf_normalized_bipolar_score(slope, df_index, tf_weights)
        norm_accel = get_adaptive_mtf_normalized_bipolar_score(accel, df_index, tf_weights)
        dynamic_engine_power = (norm_slope.add(1)/2 * norm_accel.clip(lower=-1, upper=1).add(1)/2).pow(0.5) * 2 - 1
        static_engine_power = health_score # 算术平均结果本身就在[-1, 1]区间，无需再转换
        static_weight, dynamic_weight = 0.5, 0.5
        engine_power_score = static_engine_power * static_weight + dynamic_engine_power * dynamic_weight
        conviction = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        impulse_purity = self._get_safe_series(df, df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(conviction, df_index, tf_weights)
        purity_score = get_adaptive_mtf_normalized_bipolar_score(impulse_purity, df_index, tf_weights)
        base_fuel_quality = ((conviction_score.add(1)/2) * (purity_score.add(1)/2)).pow(0.5) * 2 - 1
        synergy_bonus = (conviction_score.clip(lower=0) * purity_score.clip(lower=0)).pow(0.5) * 0.25
        fuel_quality_score = base_fuel_quality + synergy_bonus
        vacuum = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        nozzle_efficiency_score = get_adaptive_mtf_normalized_bipolar_score(vacuum, df_index, tf_weights)
        final_score = (
            (engine_power_score.add(1)/2) *
            (fuel_quality_score.clip(-1, 1).add(1)/2) *
            (nozzle_efficiency_score.add(1)/2)
        ).pow(1/3) * 2 - 1
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V6.1 · 主力意图验证版】筹码公理五：诊断“价筹张力”
        - 核心数学升级: 将“主力共谋”验证从相关性分析升级为更稳健的“主力意图验证”模型。
                          该模型直接评估1)主力资金流向是否与背离方向一致(同谋), 2)主力资金流强度是否足够大(兵力)。
                          只有当两者都满足时，才确认为一次高置信度的“战术性背离”，并给予显著加成。
        """
        print("    -> [筹码层] 正在诊断“价筹张力”公理 (V6.1 · 主力意图验证版)...") # [修改代码行]
        required_signals = ['winner_loser_momentum_D', 'SLOPE_5_close_D', 'volume_D', 'main_force_net_flow_calibrated_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        conflict_bonus = get_param_value(p_conf.get('divergence_conflict_bonus'), 0.5)
        df_index = df.index
        chip_momentum = self._get_safe_series(df, df, 'winner_loser_momentum_D', 0.0, method_name="_diagnose_axiom_divergence")
        price_trend = self._get_safe_series(df, df, 'SLOPE_5_close_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_chip_momentum = get_adaptive_mtf_normalized_bipolar_score(chip_momentum, df_index, tf_weights)
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend, df_index, tf_weights)
        disagreement_vector = norm_chip_momentum - norm_price_trend
        persistence = disagreement_vector.rolling(window=13, min_periods=5).std().fillna(0)
        norm_persistence = get_adaptive_mtf_normalized_score(persistence, df_index, tf_weights=tf_weights)
        volume = self._get_safe_series(df, df, 'volume_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_volume = get_adaptive_mtf_normalized_score(volume, df_index, tf_weights=tf_weights)
        energy_injection = norm_volume * disagreement_vector.abs()
        tension_magnitude = (norm_persistence * energy_injection).pow(0.5)
        # [修改代码块] 升级为“主力意图验证”模型
        mf_flow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0)
        # 1. 方向同谋验证
        is_conspiracy = np.sign(disagreement_vector) * np.sign(mf_flow) > 0
        # 2. 兵力强度验证
        norm_mf_flow_strength = get_adaptive_mtf_normalized_score(mf_flow.abs(), df_index, tf_weights)
        # 意图验证得分 = 方向一致 * 兵力强度
        conviction_score = is_conspiracy * norm_mf_flow_strength
        # 将意图验证得分转化为放大因子
        conviction_factor = 1.0 + conviction_score * 0.5 # 最大放大1.5倍
        base_final_score = disagreement_vector * (1 + tension_magnitude * 1.5) * conviction_factor
        conflict_mask = (np.sign(norm_chip_momentum) * np.sign(norm_price_trend) < 0)
        conflict_amplifier = pd.Series(1.0, index=df_index)
        conflict_amplifier.loc[conflict_mask] = 1.0 + conflict_bonus
        safe_base_score = base_final_score.clip(-0.999, 0.999)
        final_score = np.tanh(np.arctanh(safe_base_score) * conflict_amplifier)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_structural_consensus(self, df: pd.DataFrame, cost_structure_scores: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series:
        """
        【V7.15 · 结构强度幂指数自适应版】诊断筹码同调驱动力
        - 核心升级1: 引入结构强度对幂指数的自适应调整。amplification_power 和 dampening_power 将根据
                      最终用于调制的筹码结构分数（final_cost_structure_for_modulation_scaled）的绝对强度
                      进行进一步的动态调整，使得传动系统在面对极端顺风或逆风情况时，能够自适应地增强其放大或削弱驱动力的能力。
        - 核心升级2: 增强真理探针。详细输出新的结构强度幂指数自适应参数和中间计算结果。
        - 修复: 修正了探针输出中 `abs_activated_sentiment_val` 的计算错误，确保在标量值上正确调用绝对值函数。
        """
        print("    -> [筹码层] 正在诊断“同调驱动力 (V7.15 · 结构强度幂指数自适应版)”...") # [修改代码行]
        
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

        # [新增代码块] 结构强度幂指数自适应参数
        structural_power_adjustment_enabled = get_param_value(coherent_drive_params.get('structural_power_adjustment_enabled'), False)
        structural_power_sensitivity_amp = get_param_value(coherent_drive_params.get('structural_power_sensitivity_amp'), 0.5)
        structural_power_sensitivity_damp = get_param_value(coherent_drive_params.get('structural_power_sensitivity_damp'), 0.5)
        structural_power_tanh_factor_amp = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_amp'), 1.0)
        structural_power_tanh_factor_damp = get_param_value(coherent_drive_params.get('structural_power_tanh_factor_damp'), 1.0)

        amplification_power = pd.Series(base_amplification_power, index=df.index)
        dampening_power = pd.Series(base_dampening_power, index=df.index)
        modulation_factor = pd.Series(1.0, index=df.index)

        current_chip_health_score_raw = pd.Series(0.0, index=df.index)
        normalized_chip_health = pd.Series(0.0, index=df.index)
        modulated_chip_health_amp = pd.Series(0.0, index=df.index)
        modulated_chip_health_damp = pd.Series(0.0, index=df.index)

        dynamic_chip_health_sensitivity_amp = pd.Series(default_chip_health_sensitivity_amp, index=df.index)
        dynamic_chip_health_sensitivity_damp = pd.Series(default_chip_health_sensitivity_damp, index=df.index)
        modulator_signal_raw = pd.Series(0.0, index=df.index)
        normalized_modulator_signal = pd.Series(0.5, index=df.index)
        non_linear_modulator_effect_amp = pd.Series(0.0, index=df.index)
        non_linear_modulator_effect_damp = pd.Series(0.0, index=df.index)

        dynamic_cost_structure_impact_factor_bullish = pd.Series(cost_structure_impact_base_factor_bullish, index=df.index)
        dynamic_cost_structure_impact_factor_bearish = pd.Series(cost_structure_impact_base_factor_bearish, index=df.index)
        
        dynamic_coupling_factor = pd.Series(sentiment_coupling_base_factor, index=df.index)
        final_cost_structure_for_modulation = pd.Series(0.0, index=df.index)

        dynamic_sentiment_neutrality_threshold = pd.Series(sentiment_neutrality_base_threshold, index=df.index)
        dynamic_cost_structure_neutrality_threshold = pd.Series(cost_structure_neutrality_base_threshold, index=df.index)

        activated_holder_sentiment_scores = holder_sentiment_scores.copy()
        dynamic_structure_modulation_strength = pd.Series(structure_modulation_base_strength, index=df.index)
        final_cost_structure_for_modulation_scaled = pd.Series(0.0, index=df.index)

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
                dynamic_chip_health_sensitivity_damp = dynamic_chip_health_sensitivity_damp.clip(base_damp_sensitivity_series * 0.1, base_damp_sensitivity_series * 2.0)
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

        # [新增代码块] 结构强度对幂指数的自适应调整
        if structural_power_adjustment_enabled:
            # 记录调整前的幂指数，用于探针
            amplification_power_before_structural_adj = amplification_power.copy()
            dampening_power_before_structural_adj = dampening_power.copy()

            positive_structure_mask = final_cost_structure_for_modulation_scaled > 0
            negative_structure_mask = final_cost_structure_for_modulation_scaled < 0

            if positive_structure_mask.any():
                positive_structure_strength = final_cost_structure_for_modulation_scaled[positive_structure_mask]
                boost_amp = np.tanh(positive_structure_strength * structural_power_tanh_factor_amp) * structural_power_sensitivity_amp
                amplification_power.loc[positive_structure_mask] = amplification_power.loc[positive_structure_mask] * (1 + boost_amp)
            
            if negative_structure_mask.any():
                negative_structure_strength = final_cost_structure_for_modulation_scaled[negative_structure_mask].abs()
                boost_damp = np.tanh(negative_structure_strength * structural_power_tanh_factor_damp) * structural_power_sensitivity_damp
                dampening_power.loc[negative_structure_mask] = dampening_power.loc[negative_structure_mask] * (1 + boost_damp)
            
            # 再次裁剪，确保幂指数在合理范围内
            amplification_power = amplification_power.clip(0.5, 3.0) # [修改代码行] 扩大裁剪范围以允许更大的动态调整
            dampening_power = dampening_power.clip(0.5, 3.0) # [修改代码行] 扩大裁剪范围以允许更大的动态调整

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
        final_score = np.tanh(coherent_drive_raw * (self.bipolar_sensitivity * 2))
        
        # 植入标准化探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [同调驱动力探针] @ {probe_date.date()}:")
                print(f"       - 基础参数: base_amplification_power: {base_amplification_power:.2f}, base_dampening_power: {base_dampening_power:.2f}")
                print(f"       - 筹码健康度默认敏感度: default_chip_health_sensitivity_amp: {default_chip_health_sensitivity_amp:.2f}, default_chip_health_sensitivity_damp: {default_chip_health_sensitivity_damp:.2f}")
                print(f"       - 筹码健康度MTF归一化参数: {chip_health_mtf_norm_params}")
                print(f"       - 筹码健康度非线性参数: chip_health_tanh_factor_amp: {chip_health_tanh_factor_amp:.2f}, chip_health_tanh_factor_damp: {chip_health_tanh_factor_damp:.2f}")
                print(f"       - 筹码健康度动态敏感度调制: enabled: {chip_health_sensitivity_modulation_enabled}, modulator: '{chip_sensitivity_modulator_signal_name}', norm_window: {chip_sensitivity_mod_norm_window}, mod_factor_amp: {chip_sensitivity_mod_factor_amp:.2f}, mod_factor_damp: {chip_sensitivity_mod_factor_damp:.2f}")
                print(f"       - 动态敏感度非线性参数: chip_sensitivity_mod_tanh_factor_amp: {chip_sensitivity_mod_tanh_factor_amp:.2f}, chip_sensitivity_mod_tanh_factor_damp: {chip_sensitivity_mod_tanh_factor_damp:.2f}")
                print(f"       - 筹码结构情绪非对称影响参数: enabled: {cost_structure_asymmetric_impact_enabled}")
                print(f"         - 看涨: base_factor: {cost_structure_impact_base_factor_bullish:.2f}, sentiment_sensitivity: {cost_structure_impact_sentiment_sensitivity_bullish:.2f}, sentiment_tanh_factor: {cost_structure_impact_sentiment_tanh_factor_bullish:.2f}")
                print(f"         - 看跌: base_factor: {cost_structure_impact_base_factor_bearish:.2f}, sentiment_sensitivity: {cost_structure_impact_sentiment_sensitivity_bearish:.2f}, sentiment_tanh_factor: {cost_structure_impact_sentiment_tanh_factor_bearish:.2f}")
                print(f"       - 情绪结构动态耦合参数: enabled: {sentiment_cost_structure_coupling_enabled}, base_factor: {sentiment_coupling_base_factor:.2f}, tanh_factor: {sentiment_coupling_tanh_factor:.2f}, sensitivity: {sentiment_coupling_sensitivity:.2f}")
                print(f"       - 筹码健康度敏感度非对称参数: enabled: {chip_health_asymmetric_sensitivity_enabled}")
                print(f"         - 积极健康度: amp_sens: {chip_health_sensitivity_amp_positive_health:.2f}, damp_sens: {chip_health_sensitivity_damp_positive_health:.2f}")
                print(f"         - 消极健康度: amp_sens: {chip_health_sensitivity_amp_negative_health:.2f}, damp_sens: {chip_health_sensitivity_damp_negative_health:.2f}")
                print(f"       - 动态中性阈值参数: enabled: {dynamic_neutrality_thresholds_enabled}")
                print(f"         - 情绪: base_threshold: {sentiment_neutrality_base_threshold:.2f}, chip_health_sensitivity: {sentiment_neutrality_chip_health_sensitivity:.2f}")
                print(f"         - 筹码结构: base_threshold: {cost_structure_neutrality_base_threshold:.2f}, chip_health_sensitivity: {cost_structure_neutrality_chip_health_sensitivity:.2f}")
                print(f"       - 情绪激活阈值参数: enabled: {sentiment_activation_enabled}, tanh_factor: {sentiment_activation_tanh_factor:.2f}, strength: {sentiment_activation_strength:.2f}")
                print(f"       - 情绪强度结构调制参数: enabled: {structure_modulation_strength_enabled}, base_strength: {structure_modulation_base_strength:.2f}, sentiment_tanh_factor: {structure_modulation_sentiment_tanh_factor:.2f}, sentiment_sensitivity: {structure_modulation_sentiment_sensitivity:.2f}")
                # [新增代码块] 打印结构强度幂指数自适应参数
                print(f"       - 结构强度幂指数自适应参数: enabled: {structural_power_adjustment_enabled}")
                print(f"         - 放大: sensitivity: {structural_power_sensitivity_amp:.2f}, tanh_factor: {structural_power_tanh_factor_amp:.2f}")
                print(f"         - 削弱: sensitivity: {structural_power_sensitivity_damp:.2f}, tanh_factor: {structural_power_tanh_factor_damp:.2f}")

                if chip_health_modulation_enabled:
                    print(f"       - 筹码健康度信号 (原始): chip_health_score_D: {current_chip_health_score_raw.loc[probe_date]:.4f}")
                    print(f"       - 筹码健康度信号 (归一化): normalized_chip_health: {normalized_chip_health.loc[probe_date]:.4f}")
                    if chip_health_asymmetric_sensitivity_enabled:
                        selected_base_amp_sens = base_amp_sensitivity_series.loc[probe_date]
                        selected_base_damp_sens = base_damp_sensitivity_series.loc[probe_date]
                        print(f"       - 筹码健康度选定基础敏感度: amp_sens: {selected_base_amp_sens:.2f}, damp_sens: {selected_base_damp_sens:.2f}")
                    print(f"       - 筹码健康度信号 (非线性调制-放大): modulated_chip_health_amp: {modulated_chip_health_amp.loc[probe_date]:.4f}")
                    print(f"       - 筹码健康度信号 (非线性调制-削弱): modulated_chip_health_damp: {modulated_chip_health_damp.loc[probe_date]:.4f}")
                    if chip_health_sensitivity_modulation_enabled:
                        print(f"       - 敏感度调制信号 (原始): {chip_sensitivity_modulator_signal_name}: {modulator_signal_raw.loc[probe_date]:.4f}")
                        print(f"       - 敏感度调制信号 (归一化): normalized_modulator_signal: {normalized_modulator_signal.loc[probe_date]:.4f}")
                        print(f"       - 敏感度调制信号 (双极性): modulator_bipolar: {((normalized_modulator_signal.loc[probe_date] * 2) - 1):.4f}")
                        print(f"       - 敏感度调制信号 (非线性-放大): non_linear_modulator_effect_amp: {non_linear_modulator_effect_amp.loc[probe_date]:.4f}")
                        print(f"       - 敏感度调制信号 (非线性-削弱): non_linear_modulator_effect_damp: {non_linear_modulator_effect_damp.loc[probe_date]:.4f}")
                    print(f"       - 动态敏感度: dynamic_chip_health_sensitivity_amp: {dynamic_chip_health_sensitivity_amp.loc[probe_date]:.2f}, dynamic_chip_health_sensitivity_damp: {dynamic_chip_health_sensitivity_damp.loc[probe_date]:.2f}")
                    print(f"       - 动态幂指数 (筹码健康度调制后): amplification_power: {amplification_power_before_structural_adj.loc[probe_date]:.2f}, dampening_power: {dampening_power_before_structural_adj.loc[probe_date]:.2f}") # [修改代码行]
                else:
                    print(f"       - 静态幂指数: amplification_power: {amplification_power.loc[probe_date]:.2f}, dampening_power: {dampening_power.loc[probe_date]:.2f}")
                
                if dynamic_neutrality_thresholds_enabled:
                    print(f"       - 动态情绪中性阈值: {dynamic_sentiment_neutrality_threshold.loc[probe_date]:.4f}")
                    print(f"       - 动态筹码结构中性阈值: {dynamic_cost_structure_neutrality_threshold.loc[probe_date]:.4f}")

                print(f"       - 原料: base_drive (holder_sentiment): {holder_sentiment_scores.loc[probe_date]:.4f}")
                if sentiment_activation_enabled:
                    print(f"       - 原料: activated_holder_sentiment_scores: {activated_holder_sentiment_scores.loc[probe_date]:.4f}")

                if cost_structure_asymmetric_impact_enabled:
                    current_holder_sentiment = holder_sentiment_scores.loc[probe_date]
                    if current_holder_sentiment > 0:
                        positive_sentiment_strength_val = current_holder_sentiment
                        normalized_positive_sentiment_tanh_val = np.tanh(positive_sentiment_strength_val * cost_structure_impact_sentiment_tanh_factor_bullish)
                        print(f"       - 原料: holder_sentiment_scores (看涨强度): {positive_sentiment_strength_val:.4f}")
                        print(f"       - 原料: holder_sentiment_scores (看涨强度非线性): {normalized_positive_sentiment_tanh_val:.4f}")
                        print(f"       - 筹码结构动态影响因子 (看涨): dynamic_cost_structure_impact_factor_bullish: {dynamic_cost_structure_impact_factor_bullish.loc[probe_date]:.4f}")
                        print(f"       - 选定筹码结构动态影响因子: {selected_dynamic_cost_structure_impact_factor.loc[probe_date]:.4f}")
                    elif current_holder_sentiment < 0:
                        negative_sentiment_strength_val = np.abs(current_holder_sentiment)
                        normalized_negative_sentiment_tanh_val = np.tanh(negative_sentiment_strength_val * cost_structure_impact_sentiment_tanh_factor_bearish)
                        print(f"       - 原料: holder_sentiment_scores (看跌强度): {negative_sentiment_strength_val:.4f}")
                        print(f"       - 原料: holder_sentiment_scores (看跌强度非线性): {normalized_negative_sentiment_tanh_val:.4f}")
                        print(f"       - 筹码结构动态影响因子 (看跌): dynamic_cost_structure_impact_factor_bearish: {dynamic_cost_structure_impact_factor_bearish.loc[probe_date]:.4f}")
                        print(f"       - 选定筹码结构动态影响因子: {selected_dynamic_cost_structure_impact_factor.loc[probe_date]:.4f}")
                    else:
                        print(f"       - 选定筹码结构动态影响因子: {selected_dynamic_cost_structure_impact_factor.loc[probe_date]:.4f}")
                    print(f"       - 原料: cost_structure_scores (原始): {cost_structure_scores.loc[probe_date]:.4f}")
                    print(f"       - 原料: cost_structure_scores (调整后): {adjusted_cost_structure_scores.loc[probe_date]:.4f}")
                else:
                    print(f"       - 原料: cost_structure_scores: {cost_structure_scores.loc[probe_date]:.4f}")
                
                if sentiment_cost_structure_coupling_enabled:
                    abs_holder_sentiment_val = np.abs(holder_sentiment_scores.loc[probe_date])
                    sentiment_tanh_modulated_val = np.tanh(abs_holder_sentiment_val * sentiment_coupling_tanh_factor)
                    print(f"       - 情绪强度 (绝对值): {abs_holder_sentiment_val:.4f}")
                    print(f"       - 情绪强度 (tanh调制): {sentiment_tanh_modulated_val:.4f}")
                    print(f"       - 动态耦合因子: {dynamic_coupling_factor.loc[probe_date]:.4f}")
                    print(f"       - 最终用于调制的筹码结构分数 (耦合前): {final_cost_structure_for_modulation.loc[probe_date]:.4f}")
                
                if structure_modulation_strength_enabled:
                    abs_activated_sentiment_val = np.abs(activated_holder_sentiment_scores.loc[probe_date])
                    sentiment_tanh_modulated_for_structure_val = np.tanh(abs_activated_sentiment_val * structure_modulation_sentiment_tanh_factor)
                    print(f"       - 激活情绪强度 (绝对值): {abs_activated_sentiment_val:.4f}")
                    print(f"       - 激活情绪强度 (tanh调制-结构): {sentiment_tanh_modulated_for_structure_val:.4f}")
                    print(f"       - 动态结构调制强度: {dynamic_structure_modulation_strength.loc[probe_date]:.4f}")
                    print(f"       - 最终用于调制的筹码结构分数 (缩放后): {final_cost_structure_for_modulation_scaled.loc[probe_date]:.4f}")
                else:
                    print(f"       - 最终用于调制的筹码结构分数: {final_cost_structure_for_modulation_scaled.loc[probe_date]:.4f}")

                # [新增代码块] 打印结构强度幂指数自适应的中间结果
                if structural_power_adjustment_enabled:
                    current_final_cost_structure_for_modulation_scaled = final_cost_structure_for_modulation_scaled.loc[probe_date]
                    current_amplification_power_before_structural_adj = amplification_power_before_structural_adj.loc[probe_date]
                    current_dampening_power_before_structural_adj = dampening_power_before_structural_adj.loc[probe_date]
                    
                    boost_amp_val = 0.0
                    boost_damp_val = 0.0
                    if current_final_cost_structure_for_modulation_scaled > 0:
                        boost_amp_val = np.tanh(current_final_cost_structure_for_modulation_scaled * structural_power_tanh_factor_amp) * structural_power_sensitivity_amp
                        print(f"       - 结构强度对放大幂指数的增强因子: {boost_amp_val:.4f}")
                    elif current_final_cost_structure_for_modulation_scaled < 0:
                        boost_damp_val = np.tanh(np.abs(current_final_cost_structure_for_modulation_scaled) * structural_power_tanh_factor_damp) * structural_power_sensitivity_damp
                        print(f"       - 结构强度对削弱幂指数的增强因子: {boost_damp_val:.4f}")
                    
                    print(f"       - 动态幂指数 (结构强度自适应后): amplification_power: {amplification_power.loc[probe_date]:.2f}, dampening_power: {dampening_power.loc[probe_date]:.2f}")


                current_raw_sentiment = holder_sentiment_scores.loc[probe_date]
                current_activated_sentiment = activated_holder_sentiment_scores.loc[probe_date]
                current_cost_structure_for_mod_scaled = final_cost_structure_for_modulation_scaled.loc[probe_date]
                current_modulation_factor = modulation_factor.loc[probe_date]
                current_amp_power = amplification_power.loc[probe_date]
                current_damp_power = dampening_power.loc[probe_date]
                current_dynamic_sentiment_threshold = dynamic_sentiment_neutrality_threshold.loc[probe_date]
                current_dynamic_cost_structure_threshold = dynamic_cost_structure_neutrality_threshold.loc[probe_date]

                print(f"       - 过程: raw_sentiment > {current_dynamic_sentiment_threshold:.4f}: {current_raw_sentiment > current_dynamic_sentiment_threshold}, cost_structure_for_mod_scaled > {current_dynamic_cost_structure_threshold:.4f}: {current_cost_structure_for_mod_scaled > current_dynamic_cost_structure_threshold}")
                
                if current_raw_sentiment > current_dynamic_sentiment_threshold:
                    if current_cost_structure_for_mod_scaled > current_dynamic_cost_structure_threshold:
                        expected_mod_factor = (1 + current_cost_structure_for_mod_scaled) ** current_amp_power
                        print(f"         - 逻辑: 牛市情绪顺风 (1 + {current_cost_structure_for_mod_scaled:.4f})^{current_amp_power:.2f} = {expected_mod_factor:.4f}")
                    elif current_cost_structure_for_mod_scaled < -current_dynamic_cost_structure_threshold:
                        expected_mod_factor = (1 - abs(current_cost_structure_for_mod_scaled)) ** current_damp_power
                        print(f"         - 逻辑: 牛市情绪逆风 (1 - |{current_cost_structure_for_mod_scaled:.4f}|)^{current_damp_power:.2f} = {expected_mod_factor:.4f}")
                    else:
                        expected_mod_factor = 1.0
                        print(f"         - 逻辑: 牛市情绪，筹码结构中性，调制因子保持为 {expected_mod_factor:.4f}")
                elif current_raw_sentiment < -current_dynamic_sentiment_threshold:
                    if current_cost_structure_for_mod_scaled < -current_dynamic_cost_structure_threshold:
                        expected_mod_factor = (1 + abs(current_cost_structure_for_mod_scaled)) ** current_amp_power
                        print(f"         - 逻辑: 熊市情绪顺风 (1 + |{current_cost_structure_for_mod_scaled:.4f}|)^{current_amp_power:.2f} = {expected_mod_factor:.4f}")
                    elif current_cost_structure_for_mod_scaled > current_dynamic_cost_structure_threshold:
                        expected_mod_factor = (1 - current_cost_structure_for_mod_scaled) ** current_damp_power
                        print(f"         - 逻辑: 熊市情绪逆风 (1 - {current_cost_structure_for_mod_scaled:.4f})^{current_damp_power:.2f} = {expected_mod_factor:.4f}")
                    else:
                        expected_mod_factor = 1.0
                        print(f"         - 逻辑: 熊市情绪，筹码结构中性，调制因子保持为 {expected_mod_factor:.4f}")
                else:
                    expected_mod_factor = 1.0
                    print(f"         - 逻辑: 情绪中性，调制因子保持为 {expected_mod_factor:.4f}")
                print(f"       - 过程: modulation_factor (实际): {current_modulation_factor:.4f}")
                print(f"       - 过程: coherent_drive_raw (pre-tanh): {current_activated_sentiment * current_modulation_factor:.4f}")
                print(f"       - 结果: final_score: {final_score.loc[probe_date]:.4f}")
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



