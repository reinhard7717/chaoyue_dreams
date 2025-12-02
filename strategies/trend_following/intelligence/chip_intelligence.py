import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Union
from strategies.trend_following import utils
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    get_adaptive_mtf_normalized_bipolar_score, is_limit_up
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
        【V17.0 · 天人合一版】筹码情报总指挥
        - 核心升维: 新增对“战略战术和谐度”的诊断。通过调用 `_diagnose_strategic_tactical_harmony` 方法，
                      模型现在能够评估主力的长期战略意图与当日战术执行之间的协同性，
                      从而在更高维度上对趋势的健康度与风险进行裁决。
        """
        print("启动【V17.0 · 天人合一版】筹码情报分析...") # [修改代码行]
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
        # [新增代码块] 调用新增的和谐度诊断方法
        strategic_tactical_harmony = self._diagnose_strategic_tactical_harmony(df, strategic_posture, tactical_exchange)
        all_chip_states['SCORE_CHIP_STRATEGIC_TACTICAL_HARMONY'] = strategic_tactical_harmony
        print(f"【V17.0 · 天人合一版】分析完成，生成 {len(all_chip_states)} 个筹码原子信号。") # [修改代码行]
        return all_chip_states

    def _run_integrity_probe(self, df: pd.DataFrame, required_signals: list, probe_name: str):
        """
        【V2.4 · 物证探针版】
        - 核心升级: 不再进行条件判断，而是无条件打印所有依赖信号在探针日期的值和近期标准差，
                      以获取关于“幻影信号”的决定性物证。
        """
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if not probe_dates_str:
            return
        probe_date_naive = pd.to_datetime(probe_dates_str[0])
        probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
        if probe_date in df.index:
            print(f"    -> [筹码公理-{probe_name}-物证探针] 正在检查数据值...")
            for s in required_signals:
                if s not in df.columns:
                    print(f"        - [失败] 信号 '{s}' 列不存在。")
                    continue
                val = df.loc[probe_date, s]
                std_dev = df[s].loc[:probe_date].tail(21).std()
                print(f"        - [物证] 信号: {s:<45} | 当日值: {val:<10.4f} | 近期标准差: {std_dev:.4f}")

    def _diagnose_strategic_posture(self, df: pd.DataFrame) -> pd.Series:
        """
        【V6.4 · 诡道增强版】诊断主力的综合战略态势 (大一统信号)
        - 核心升级: 在“指挥官决心”维度中，引入“博弈欺骗指数”作为第四大融合因子，
                      旨在识别并量化主力在部署战略态势时所采用的欺骗战术，
                      从而更精准地评估其真实意图的强度与决心。
        - 核心升级: 植入标准化的“真理探针”，输出所有原始数据、关键计算过程及最终结果。
        """
        print("    -> [筹码层] 正在诊断“战略态势 (V6.4 · 诡道增强版)”...")
        required_signals = [
            'cost_gini_coefficient_D', 'covert_accumulation_signal_D', 'peak_exchange_purity_D',
            'main_force_cost_advantage_D', 'control_solidity_index_D', 'SLOPE_5_main_force_conviction_index_D',
            'floating_chip_cleansing_efficiency_D', 'dominant_peak_solidity_D',
            'deception_index_D' # 新增依赖信号
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_strategic_posture"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        concentration_level = 1 - self._get_safe_series(df, df, 'cost_gini_coefficient_D', 0.5)
        covert_accumulation = self._get_safe_series(df, df, 'covert_accumulation_signal_D', 0.0)
        peak_purity = self._get_safe_series(df, df, 'peak_exchange_purity_D', 0.0)
        level_score = get_adaptive_mtf_normalized_bipolar_score(concentration_level, df_index, tf_weights)
        efficiency_score = (
            get_adaptive_mtf_normalized_bipolar_score(covert_accumulation, df_index, tf_weights).add(1)/2 *
            get_adaptive_mtf_normalized_bipolar_score(peak_purity, df_index, tf_weights).add(1)/2
        ).pow(0.5) * 2 - 1
        formation_deployment_score = (level_score.add(1)/2 * efficiency_score.add(1)/2).pow(0.5) * 2 - 1
        cost_advantage = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0)
        control_solidity = self._get_safe_series(df, df, 'control_solidity_index_D', 0.0)
        conviction_slope = self._get_safe_series(df, df, 'SLOPE_5_main_force_conviction_index_D', 0.0)
        deception_index = self._get_safe_series(df, df, 'deception_index_D', 0.0) # 获取欺骗指数
        advantage_score = get_adaptive_mtf_normalized_bipolar_score(cost_advantage, df_index, tf_weights)
        solidity_score = get_adaptive_mtf_normalized_bipolar_score(control_solidity, df_index, tf_weights)
        intent_score = get_adaptive_mtf_normalized_bipolar_score(conviction_slope, df_index, tf_weights)
        deception_score = get_adaptive_mtf_normalized_bipolar_score(deception_index, df_index, tf_weights) # 归一化欺骗指数
        commanders_resolve_score = ( # 将欺骗指数得分融合进指挥官决心
            (advantage_score.add(1)/2) * (solidity_score.add(1)/2) *
            (intent_score.clip(lower=-1, upper=1).add(1)/2) * (deception_score.add(1)/2)
        ).pow(1/4) * 2 - 1
        cleansing_efficiency = self._get_safe_series(df, df, 'floating_chip_cleansing_efficiency_D', 0.0)
        peak_solidity = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5)
        cleansing_score = get_adaptive_mtf_normalized_bipolar_score(cleansing_efficiency, df_index, tf_weights)
        peak_solidity_score = get_adaptive_mtf_normalized_bipolar_score(peak_solidity, df_index, tf_weights)
        battlefield_control_score = (cleansing_score.add(1)/2 * peak_solidity_score.add(1)/2).pow(0.5) * 2 - 1
        final_score = (
            (commanders_resolve_score.add(1)/2).pow(0.5) *
            (formation_deployment_score.add(1)/2).pow(0.3) *
            (battlefield_control_score.add(1)/2).pow(0.2)
        ) * 2 - 1
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [战略态势探针] @ {probe_date.date()}:")
                print(f"       - 维度1: 阵型部署 (Formation Deployment)")
                print(f"         - 原料: cost_gini: {self._get_safe_series(df, df, 'cost_gini_coefficient_D').loc[probe_date]:.4f}, covert_accum: {covert_accumulation.loc[probe_date]:.4f}, peak_purity: {peak_purity.loc[probe_date]:.4f}")
                print(f"         - 过程: level_score: {level_score.loc[probe_date]:.4f}, efficiency_score: {efficiency_score.loc[probe_date]:.4f}")
                print(f"         - 结果: formation_deployment_score: {formation_deployment_score.loc[probe_date]:.4f}")
                print(f"       - 维度2: 指挥官决心 (Commander's Resolve)")
                # 更新探针输出
                print(f"         - 原料: cost_adv: {cost_advantage.loc[probe_date]:.4f}, ctrl_solidity: {control_solidity.loc[probe_date]:.4f}, conviction_slope: {conviction_slope.loc[probe_date]:.4f}, deception_idx: {deception_index.loc[probe_date]:.4f}")
                print(f"         - 过程: advantage_score: {advantage_score.loc[probe_date]:.4f}, solidity_score: {solidity_score.loc[probe_date]:.4f}, intent_score: {intent_score.loc[probe_date]:.4f}, deception_score: {deception_score.loc[probe_date]:.4f}")
                print(f"         - 结果: commanders_resolve_score: {commanders_resolve_score.loc[probe_date]:.4f}")
                print(f"       - 维度3: 战场控制 (Battlefield Control)")
                print(f"         - 原料: cleansing_eff: {cleansing_efficiency.loc[probe_date]:.4f}, peak_solidity: {peak_solidity.loc[probe_date]:.4f}")
                print(f"         - 过程: cleansing_score: {cleansing_score.loc[probe_date]:.4f}, peak_solidity_score: {peak_solidity_score.loc[probe_date]:.4f}")
                print(f"         - 结果: battlefield_control_score: {battlefield_control_score.loc[probe_date]:.4f}")
                print(f"       - 最终融合结果: final_score: {final_score.loc[probe_date]:.4f}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_battlefield_geography(self, df: pd.DataFrame) -> pd.Series:
        """
        【V7.0 · 动态演化版】诊断筹码的战场地形 (大一统信号)
        - 核心升级: 引入“动态演化”因子。通过融合支撑强度斜率与阻力强度斜率，量化战场地形的
                      有利度是在改善还是在恶化，从而对静态地形分进行动态调节，使信号更具前瞻性。
        - 核心升级: 植入标准化的“真理探针”，输出所有原始数据、关键计算过程及最终结果。
        """
        print("    -> [筹码层] 正在诊断“战场地形 (V7.0 · 动态演化版)”...") # [修改代码行]
        required_signals = [
            'dominant_peak_solidity_D', 'support_validation_strength_D', 'chip_fault_blockage_ratio_D',
            'pressure_rejection_strength_D', 'vacuum_zone_magnitude_D', 'vacuum_traversal_efficiency_D',
            'SLOPE_5_support_validation_strength_D', 'SLOPE_5_pressure_rejection_strength_D' # [修改代码行] 新增斜率依赖
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_battlefield_geography"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        peak_solidity = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5)
        support_validation = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0)
        solidity_score = get_adaptive_mtf_normalized_score(peak_solidity, df_index, tf_weights)
        validation_score = get_adaptive_mtf_normalized_score(support_validation, df_index, tf_weights)
        support_strength_score = (solidity_score * validation_score).pow(0.5)
        fault_blockage = self._get_safe_series(df, df, 'chip_fault_blockage_ratio_D', 0.5)
        pressure_rejection = self._get_safe_series(df, df, 'pressure_rejection_strength_D', 0.5)
        blockage_score = get_adaptive_mtf_normalized_score(fault_blockage, df_index, tf_weights)
        rejection_score = get_adaptive_mtf_normalized_score(pressure_rejection, df_index, tf_weights)
        resistance_strength_score = (blockage_score * rejection_score).pow(0.5)
        vacuum_magnitude = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0)
        vacuum_efficiency = self._get_safe_series(df, df, 'vacuum_traversal_efficiency_D', 0.0)
        magnitude_score = get_adaptive_mtf_normalized_score(vacuum_magnitude, df_index, tf_weights)
        efficiency_score = get_adaptive_mtf_normalized_score(vacuum_efficiency, df_index, tf_weights)
        path_score = (magnitude_score * efficiency_score).pow(0.5)
        base_score = support_strength_score * (1 - resistance_strength_score)
        static_final_score = np.sign(base_score) * (base_score.abs() * path_score).pow(0.5)
        # [新增代码块] 计算动态演化因子
        support_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_support_validation_strength_D', 0.0)
        resistance_trend_raw = self._get_safe_series(df, df, 'SLOPE_5_pressure_rejection_strength_D', 0.0)
        support_trend_score = get_adaptive_mtf_normalized_bipolar_score(support_trend_raw, df_index, tf_weights)
        resistance_trend_score = get_adaptive_mtf_normalized_bipolar_score(resistance_trend_raw, df_index, tf_weights)
        # 地形优势变化 = 支撑趋势 - 阻力趋势 (支撑增强、阻力减弱为最佳)
        terrain_advantage_change = support_trend_score - resistance_trend_score
        # 将变化趋势转化为一个调节因子 (1.0为中性, >1.0为增强, <1.0为削弱)
        dynamic_evolution_factor = 1.0 + (terrain_advantage_change * 0.25) # 变化趋势的影响力设为25%
        final_score = static_final_score * dynamic_evolution_factor
        # 植入标准化探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [战场地形探针] @ {probe_date.date()}:")
                print(f"       - 维度1: 下方支撑 (Support)")
                print(f"         - 原料: peak_solidity: {peak_solidity.loc[probe_date]:.4f}, support_validation: {support_validation.loc[probe_date]:.4f}")
                print(f"         - 结果: support_strength_score: {support_strength_score.loc[probe_date]:.4f}")
                print(f"       - 维度2: 上方阻力 (Resistance)")
                print(f"         - 原料: fault_blockage: {fault_blockage.loc[probe_date]:.4f}, pressure_rejection: {pressure_rejection.loc[probe_date]:.4f}")
                print(f"         - 结果: resistance_strength_score: {resistance_strength_score.loc[probe_date]:.4f}")
                print(f"       - 维度3: 最小阻力路径 (Path)")
                print(f"         - 原料: vacuum_magnitude: {vacuum_magnitude.loc[probe_date]:.4f}, vacuum_efficiency: {vacuum_efficiency.loc[probe_date]:.4f}")
                print(f"         - 结果: path_score: {path_score.loc[probe_date]:.4f}")
                print(f"       - 静态融合结果: static_final_score: {static_final_score.loc[probe_date]:.4f}")
                # [新增代码块] 探针输出动态演化维度
                print(f"       - 维度4: 动态演化 (Dynamic Evolution)")
                print(f"         - 原料: support_trend_raw: {support_trend_raw.loc[probe_date]:.4f}, resistance_trend_raw: {resistance_trend_raw.loc[probe_date]:.4f}")
                print(f"         - 过程: terrain_advantage_change: {terrain_advantage_change.loc[probe_date]:.4f}")
                print(f"         - 结果: dynamic_evolution_factor: {dynamic_evolution_factor.loc[probe_date]:.4f}")
                print(f"       - 最终动态融合结果: final_score: {final_score.loc[probe_date]:.4f}")
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
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [持仓信念探针] @ {probe_date.date()}:")
                print(f"       - 维度1: 信念内核 (Belief Core)")
                print(f"         - 原料: winner_stability: {winner_stability.loc[probe_date]:.4f}, loser_pain: {loser_pain.loc[probe_date]:.4f}")
                print(f"         - 结果: belief_core_score: {belief_core_score.loc[probe_date]:.4f}")
                print(f"       - 维度2: 压力测试 (Pressure Test)")
                # [修改代码块] 更新探针输出
                print(f"         - 原料: absorption_power: {absorption_power.loc[probe_date]:.4f}, defense_intent: {defense_intent.loc[probe_date]:.4f}, capitulation_absorption: {capitulation_absorption.loc[probe_date]:.4f}")
                print(f"         - 过程: base_pressure_score: {base_pressure_score.loc[probe_date]:.4f}, capitulation_bonus: {capitulation_bonus.loc[probe_date]:.4f}")
                print(f"         - 结果: pressure_test_score (with bonus): {pressure_test_score.loc[probe_date]:.4f}")
                print(f"       - 维度3: 情绪纯度 (Impurity)")
                print(f"         - 原料: fomo_index: {fomo_index.loc[probe_date]:.4f}, profit_taking_quality: {profit_taking_quality.loc[probe_date]:.4f}")
                print(f"         - 过程: fomo_score (corrected): {fomo_score.loc[probe_date]:.4f}, profit_taking_score: {profit_taking_score.loc[probe_date]:.4f}")
                print(f"         - 结果: impurity_score: {impurity_score.loc[probe_date]:.4f}")
                print(f"       - 最终融合结果: final_score: {final_score.loc[probe_date]:.4f}")
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
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [结构性推力探针] @ {probe_date.date()}:")
                print(f"       - 维度1: 引擎功率 (Engine Power)")
                print(f"         - 原料 (上游信号): posture: {strategic_posture.loc[probe_date]:.4f}, geography: {battlefield_geography.loc[probe_date]:.4f}, sentiment: {holder_sentiment.loc[probe_date]:.4f}")
                # [修改代码块] 更新探针输出
                print(f"         - 原料 (计算过程): health_score (arithmetic): {health_score.loc[probe_date]:.4f}, slope: {slope.loc[probe_date]:.4f}, accel: {accel.loc[probe_date]:.4f}")
                print(f"         - 过程 (融合): static_power: {static_engine_power.loc[probe_date]:.4f}, dynamic_power: {dynamic_engine_power.loc[probe_date]:.4f}")
                print(f"         - 结果: engine_power_score: {engine_power_score.loc[probe_date]:.4f}")
                print(f"       - 维度2: 燃料品质 (Fuel Quality)")
                print(f"         - 原料: conviction_index: {conviction.loc[probe_date]:.4f}, impulse_purity: {impulse_purity.loc[probe_date]:.4f}")
                print(f"         - 过程: conviction_score: {conviction_score.loc[probe_date]:.4f}, purity_score: {purity_score.loc[probe_date]:.4f}")
                print(f"         - 过程: base_fuel_quality: {base_fuel_quality.loc[probe_date]:.4f}, synergy_bonus: {synergy_bonus.loc[probe_date]:.4f}")
                print(f"         - 结果: fuel_quality_score (with bonus): {fuel_quality_score.loc[probe_date]:.4f}")
                print(f"       - 维度3: 喷管效率 (Nozzle Efficiency)")
                print(f"         - 原料: vacuum_magnitude: {vacuum.loc[probe_date]:.4f}")
                print(f"         - 结果: nozzle_efficiency_score: {nozzle_efficiency_score.loc[probe_date]:.4f}")
                print(f"       - 最终融合结果: final_score: {final_score.loc[probe_date]:.4f}")
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
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [价筹张力探针] @ {probe_date.date()}:")
                print(f"       - 维度1: 分歧向量 (Disagreement Vector)")
                print(f"         - 原料: chip_momentum: {chip_momentum.loc[probe_date]:.4f}, price_trend: {price_trend.loc[probe_date]:.4f}")
                print(f"         - 过程: norm_chip_momentum: {norm_chip_momentum.loc[probe_date]:.4f}, norm_price_trend: {norm_price_trend.loc[probe_date]:.4f}")
                print(f"         - 结果: disagreement_vector: {disagreement_vector.loc[probe_date]:.4f}")
                print(f"       - 维度2: 张力强度 (Tension Magnitude)")
                print(f"         - 原料: volume: {volume.loc[probe_date]:.0f}")
                print(f"         - 过程: persistence: {persistence.loc[probe_date]:.4f}, energy_injection: {energy_injection.loc[probe_date]:.4f}")
                print(f"         - 结果: tension_magnitude: {tension_magnitude.loc[probe_date]:.4f}")
                # [修改代码块] 更新探针输出
                print(f"       - 维度3: 主力意图验证 (Main Force Intent)")
                print(f"         - 原料: mf_flow: {mf_flow.loc[probe_date]:.2f}")
                print(f"         - 过程: is_conspiracy: {is_conspiracy.loc[probe_date]}, norm_mf_flow_strength: {norm_mf_flow_strength.loc[probe_date]:.4f}")
                print(f"         - 过程: conviction_score: {conviction_score.loc[probe_date]:.4f}, conviction_factor: {conviction_factor.loc[probe_date]:.4f}")
                print(f"       - 最终融合 (渐进放大):")
                print(f"         - 过程: base_final_score (with conviction): {base_final_score.loc[probe_date]:.4f}, conflict_amplifier: {conflict_amplifier.loc[probe_date]:.4f}")
                print(f"         - 过程: uncompressed_score: {np.arctanh(safe_base_score).loc[probe_date]:.4f}, amplified_uncompressed: {np.arctanh(safe_base_score).loc[probe_date] * conflict_amplifier.loc[probe_date]:.4f}")
                print(f"         - 结果: final_score: {final_score.loc[probe_date]:.4f}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_structural_consensus(self, df: pd.DataFrame, cost_structure_scores: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series:
        """
        【V4.5 · 探针植入版】诊断筹码同调驱动力
        - 核心升级: 植入标准化的“真理探针”，输出所有原始数据、关键计算过程及最终结果。
        """
        print("    -> [筹码层] 正在诊断“同调驱动力”...")
        base_drive = holder_sentiment_scores
        modulation_factor = pd.Series(1.0, index=df.index)
        bullish_mask = base_drive > 0
        bullish_tailwind_mask = bullish_mask & (cost_structure_scores > 0)
        modulation_factor.loc[bullish_tailwind_mask] = 1 + cost_structure_scores.loc[bullish_tailwind_mask]
        bullish_headwind_mask = bullish_mask & (cost_structure_scores < 0)
        modulation_factor.loc[bullish_headwind_mask] = (1 - cost_structure_scores.loc[bullish_headwind_mask].abs()).clip(lower=0.1)
        bearish_mask = base_drive < 0
        bearish_tailwind_mask = bearish_mask & (cost_structure_scores < 0)
        modulation_factor.loc[bearish_tailwind_mask] = 1 + cost_structure_scores.loc[bearish_tailwind_mask].abs()
        bearish_headwind_mask = bearish_mask & (cost_structure_scores > 0)
        modulation_factor.loc[bearish_headwind_mask] = (1 - cost_structure_scores.loc[bearish_headwind_mask]).clip(lower=0.1)
        coherent_drive_raw = base_drive * modulation_factor
        final_score = np.tanh(coherent_drive_raw * (self.bipolar_sensitivity * 2))
        # 植入标准化探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [同调驱动力探针] @ {probe_date.date()}:")
                print(f"       - 原料: base_drive (holder_sentiment): {base_drive.loc[probe_date]:.4f}")
                print(f"       - 原料: cost_structure_scores: {cost_structure_scores.loc[probe_date]:.4f}")
                print(f"       - 过程: modulation_factor: {modulation_factor.loc[probe_date]:.4f}")
                print(f"       - 过程: coherent_drive_raw (pre-tanh): {coherent_drive_raw.loc[probe_date]:.4f}")
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
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [吸筹回声探针] @ {probe_date.date()}:")
                print(f"       - 状态门控: is_panic_context (price_trend < 0): {is_panic_context.loc[probe_date]}")
                print(f"       - 要素1: 恐慌声源 (Panic Source)")
                print(f"         - 原料: price_trend: {price_trend.loc[probe_date]:.4f}")
                # [修改代码行] 更新探针以显示门控效果
                print(f"         - 过程: panic_source_raw: {panic_source_score_raw.loc[probe_date]:.4f}")
                print(f"         - 结果: panic_source_score (gated): {panic_source_score.loc[probe_date]:.4f}")
                print(f"       - 要素2: 逆流介质 (Counter Flow Medium)")
                print(f"         - 原料: divergence_score: {divergence_score.loc[probe_date]:.4f}")
                print(f"         - 结果: counter_flow_medium_score: {counter_flow_medium_score.loc[probe_date]:.4f}")
                print(f"       - 要素3: 主力回声 (Main Force Echo)")
                print(f"         - 原料: mf_inflow: {mf_inflow.loc[probe_date]:.2f}")
                print(f"         - 结果: main_force_echo_score: {main_force_echo_score.loc[probe_date]:.4f}")
                print(f"       - 最终融合结果: final_score: {final_score.loc[probe_date]:.4f}")
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
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [派发诡影探针] @ {probe_date.date()}:")
                print(f"       - 状态门控: is_fomo_context (price_trend > 0): {is_fomo_context.loc[probe_date]}")
                print(f"       - 要素1: 狂热背景 (FOMO Backdrop)")
                print(f"         - 原料: price_trend: {price_trend.loc[probe_date]:.4f}")
                # [修改代码行] 更新探针以显示门控效果
                print(f"         - 过程: fomo_backdrop_raw: {fomo_backdrop_score_raw.loc[probe_date]:.4f}")
                print(f"         - 结果: fomo_backdrop_score (gated): {fomo_backdrop_score.loc[probe_date]:.4f}")
                print(f"       - 要素2: 背离诡影 (Divergence Shadow)")
                print(f"         - 原料: divergence_score: {divergence_score.loc[probe_date]:.4f}")
                print(f"         - 结果: divergence_shadow_score: {divergence_shadow_score.loc[probe_date]:.4f}")
                print(f"       - 要素3: 主力抽离 (Main Force Retreat)")
                print(f"         - 原料: mf_inflow: {mf_inflow.loc[probe_date]:.2f}")
                print(f"         - 结果: main_force_retreat_score: {main_force_retreat_score.loc[probe_date]:.4f}")
                print(f"       - 最终融合结果: final_score: {final_score.loc[probe_date]:.4f}")
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
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [战术换手博弈探针] @ {probe_date.date()}:")
                print(f"       - 维度1: 换手意图 (Intent)")
                # [修改代码块] 更新探针输出
                print(f"         - 原料: power_transfer: {power_transfer.loc[probe_date]:.2f}, control_transfer: {control_transfer.loc[probe_date]:.4f}, deception_idx: {deception_index.loc[probe_date]:.4f}")
                print(f"         - 过程: base_intent_score: {base_intent_score.loc[probe_date]:.4f}, norm_deception: {norm_deception.loc[probe_date]:.4f}")
                print(f"         - 过程(连续仲裁): arbitration_weight: {arbitration_weight.loc[probe_date]:.4f}")
                print(f"         - 结果: intent_score (arbitrated): {intent_score.loc[probe_date]:.4f}")
                print(f"       - 维度2: 换手质量 (Quality)")
                print(f"         - 原料: absorption: {absorption_idx.loc[probe_date]:.4f}, impulse_purity: {impulse_purity.loc[probe_date]:.4f}, profit_taking: {profit_quality.loc[probe_date]:.4f}, is_up_day: {is_up_day.loc[probe_date]}")
                print(f"         - 过程: norm_absorption(gated): {norm_absorption.loc[probe_date]:.4f}, norm_impulse_purity(gated): {norm_impulse_purity.loc[probe_date]:.4f}")
                print(f"         - 过程: bullish_quality (dynamic): {bullish_quality.loc[probe_date]:.4f}, bearish_quality: {bearish_quality.loc[probe_date]:.4f}")
                print(f"         - 结果: quality_score: {quality_score.loc[probe_date]:.4f}")
                print(f"       - 维度3: 换手环境 (Context)")
                print(f"         - 原料: geography (injected): {geography.loc[probe_date]:.4f}, bias55: {bias.loc[probe_date]:.4f}, is_consolidating: {is_consolidating.loc[probe_date]}")
                print(f"         - 过程: norm_bias_risk: {norm_bias_risk.loc[probe_date]:.4f}")
                print(f"         - 结果: context_score: {context_score.loc[probe_date]:.4f}")
                print(f"       - 最终融合 (加权算术平均):")
                print(f"         - 贡献: Intent({weights['intent']}): {intent_score.clip(-1, 1).loc[probe_date] * weights['intent']:.4f}, Quality({weights['quality']}): {quality_score.clip(-1, 1).loc[probe_date] * weights['quality']:.4f}, Context({weights['context']}): {context_score.clip(-1, 1).loc[probe_date] * weights['context']:.4f}")
                print(f"         - 结果: final_score: {final_score.loc[probe_date]:.4f}")
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
        # 植入标准化探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [战略战术和谐度探针] @ {probe_date.date()}:")
                print(f"       - 原料: strategic_posture: {strategic_posture.loc[probe_date]:.4f}, tactical_exchange: {tactical_exchange.loc[probe_date]:.4f}")
                print(f"       - 过程: base_intent_score: {base_intent_score.loc[probe_date]:.4f}")
                print(f"       - 过程: harmony_factor: {harmony_factor.loc[probe_date]:.4f}")
                print(f"       - 结果: final_score: {final_score.loc[probe_date]:.4f}")
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)




