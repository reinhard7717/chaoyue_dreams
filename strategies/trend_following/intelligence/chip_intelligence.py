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
        # [新增] 注入双极归一化所需的敏感度参数
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
        【V14.1 · 探针清理版】筹码情报总指挥
        - 核心清理: 移除了方法末尾的调试探针逻辑，净化日志输出。
        """
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
        df['SCORE_CHIP_STRATEGIC_POSTURE'] = strategic_posture
        df['SCORE_CHIP_BATTLEFIELD_GEOGRAPHY'] = battlefield_geography
        df['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = holder_sentiment_scores
        chip_trend_momentum_scores = self._diagnose_axiom_trend_momentum(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_TREND_MOMENTUM'] = chip_trend_momentum_scores
        absorption_echo = self._diagnose_absorption_echo(df, divergence_scores)
        all_chip_states['SCORE_CHIP_OPP_ABSORPTION_ECHO'] = absorption_echo
        distribution_whisper = self._diagnose_distribution_whisper(df, divergence_scores)
        all_chip_states['SCORE_CHIP_RISK_DISTRIBUTION_WHISPER'] = distribution_whisper
        coherent_drive = self._diagnose_structural_consensus(df, battlefield_geography, holder_sentiment_scores)
        all_chip_states['SCORE_CHIP_COHERENT_DRIVE'] = coherent_drive
        # [删除] 移除了方法末尾的调试探针逻辑
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
        【V6.2 · 探针清理版】诊断主力的综合战略态势 (大一统信号)
        - 核心清理: 移除了方法内的“真理探针”逻辑，净化日志输出。
        """
        required_signals = [
            'cost_gini_coefficient_D', 'covert_accumulation_signal_D', 'peak_exchange_purity_D',
            'main_force_cost_advantage_D', 'control_solidity_index_D', 'SLOPE_5_main_force_conviction_index_D',
            'floating_chip_cleansing_efficiency_D', 'dominant_peak_solidity_D'
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
        advantage_score = get_adaptive_mtf_normalized_bipolar_score(cost_advantage, df_index, tf_weights)
        solidity_score = get_adaptive_mtf_normalized_bipolar_score(control_solidity, df_index, tf_weights)
        intent_score = get_adaptive_mtf_normalized_bipolar_score(conviction_slope, df_index, tf_weights)
        commanders_resolve_score = (
            (advantage_score.add(1)/2) * (solidity_score.add(1)/2) * (intent_score.clip(lower=-1, upper=1).add(1)/2)
        ).pow(1/3) * 2 - 1
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
        # [删除] 移除了方法内的“真理探针”逻辑
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_battlefield_geography(self, df: pd.DataFrame) -> pd.Series:
        """
        【V6.0 · 战场地形学版】诊断筹码的战场地形 (大一统信号)
        - 核心思想: 统一并取代旧的“结构防御工事”、“战略节点功能”和“发射台品质”，消除概念重叠。旨在评估当前成本结构地形的攻防属性。
        - 诊断维度:
          1. 下方支撑强度 (Support Strength): 下方支撑带的厚度、质量与主力维护意愿。
          2. 上方阻力强度 (Resistance Strength): 上方套牢盘的压力、密度与松动迹象。
          3. 最小阻力路径 (Path of Least Resistance): “真空区”的大小、纯净度。
        - 非线性合成: 地形分 = (支撑分 * (1 - 阻力分) * 路径分)^1/3，形成一个综合的攻防评估。
        """
        required_signals = [
            'dominant_peak_solidity_D', 'support_validation_strength_D', 'chip_fault_blockage_ratio_D',
            'pressure_rejection_strength_D', 'vacuum_zone_magnitude_D', 'vacuum_traversal_efficiency_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_battlefield_geography"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # 维度一: 下方支撑强度
        peak_solidity = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5)
        support_validation = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0)
        solidity_score = get_adaptive_mtf_normalized_score(peak_solidity, df_index, tf_weights)
        validation_score = get_adaptive_mtf_normalized_score(support_validation, df_index, tf_weights)
        support_strength_score = (solidity_score * validation_score).pow(0.5)
        # 维度二: 上方阻力强度 (风险项)
        fault_blockage = self._get_safe_series(df, df, 'chip_fault_blockage_ratio_D', 0.5)
        pressure_rejection = self._get_safe_series(df, df, 'pressure_rejection_strength_D', 0.5)
        blockage_score = get_adaptive_mtf_normalized_score(fault_blockage, df_index, tf_weights)
        rejection_score = get_adaptive_mtf_normalized_score(pressure_rejection, df_index, tf_weights)
        resistance_strength_score = (blockage_score * rejection_score).pow(0.5)
        # 维度三: 最小阻力路径
        vacuum_magnitude = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0)
        vacuum_efficiency = self._get_safe_series(df, df, 'vacuum_traversal_efficiency_D', 0.0)
        magnitude_score = get_adaptive_mtf_normalized_score(vacuum_magnitude, df_index, tf_weights)
        efficiency_score = get_adaptive_mtf_normalized_score(vacuum_efficiency, df_index, tf_weights)
        path_score = (magnitude_score * efficiency_score).pow(0.5)
        # 最终非线性合成
        # (支撑分 * (1-阻力分)) 形成基础攻防分，再乘以路径分进行调制
        base_score = support_strength_score * (1 - resistance_strength_score)
        final_score = np.sign(base_score) * (base_score.abs() * path_score).pow(0.5)
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V5.0 · 信念韧性版】筹码公理三：诊断“持仓信念韧性”
        - 核心思想: 废弃对静态“默契”的评估，引入“压力测试下的信念三体模型”，旨在测量信念能抵抗多大压力。
        - 诊断维度:
          1. 信念内核 (Belief Core): 赢家是否稳定，输家是否绝望？(静态基础)
          2. 压力测试 (Pressure Test): 下跌时承接力量与主力防守意图如何？(动态验证)
          3. 情绪纯度 (Emotional Purity): 当前信念是否被散户FOMO情绪污染？(风险对冲)
        - 非线性合成: 信念韧性 = (内核 * 测试)^0.5 * (1 - 杂质)，旨在识别经历过考验的、高纯度的、真正坚固的持仓信念。
        """
        required_signals = [
            'winner_stability_index_D', 'loser_pain_index_D', 'dip_absorption_power_D',
            'mf_cost_zone_defense_intent_D', 'retail_fomo_premium_index_D', 'profit_taking_flow_ratio_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_holder_sentiment"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # 维度一: 信念内核
        winner_stability = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        loser_pain = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        stability_score = get_adaptive_mtf_normalized_bipolar_score(winner_stability, df_index, tf_weights)
        pain_score = get_adaptive_mtf_normalized_bipolar_score(loser_pain, df_index, tf_weights)
        belief_core_score = (stability_score.add(1)/2 * pain_score.add(1)/2).pow(0.5) * 2 - 1
        # 维度二: 压力测试
        absorption_power = self._get_safe_series(df, df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        defense_intent = self._get_safe_series(df, df, 'mf_cost_zone_defense_intent_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        absorption_score = get_adaptive_mtf_normalized_bipolar_score(absorption_power, df_index, tf_weights)
        defense_score = get_adaptive_mtf_normalized_bipolar_score(defense_intent, df_index, tf_weights)
        pressure_test_score = (absorption_score.add(1)/2 * defense_score.add(1)/2).pow(0.5) * 2 - 1
        # 维度三: 情绪纯度 (风险项)
        fomo_index = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        profit_taking = self._get_safe_series(df, df, 'profit_taking_flow_ratio_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        fomo_score = get_adaptive_mtf_normalized_score(fomo_index, df_index, ascending=True, tf_weights=tf_weights) # 单极性
        profit_taking_score = get_adaptive_mtf_normalized_score(profit_taking, df_index, ascending=True, tf_weights=tf_weights) # 单极性
        impurity_score = (fomo_score * profit_taking_score).pow(0.5)
        # 最终非线性合成
        conviction_base = ((belief_core_score.add(1)/2) * (pressure_test_score.add(1)/2)).pow(0.5)
        final_score = (conviction_base * (1 - impurity_score)) * 2 - 1
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_trend_momentum(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V5.0 · 结构性推力版】筹码公理六：诊断“结构性推力”
        - 核心思想: 废弃简单的“速度计”模型，引入“火箭推力三元模型”，旨在评估筹码趋势的质量、意图与可持续性。
        - 诊断维度:
          1. 引擎功率 (Engine Power): 整体筹码健康度的斜率与加速度。
          2. 燃料品质 (Fuel Quality): 推力是否由“主力信念”这一高能燃料驱动？
          3. 喷管效率 (Nozzle Efficiency): 前方是否存在“真空区”以最高效地传导推力？
        - 非线性合成: 结构性推力 = 功率 * 品质 * 效率。任何一环的缺失都将导致推力瓦解，旨在识别真正由主力主导的、具备持续性的高质量趋势。
        """
        # 更新 required_signals 以匹配新的“大一统”信号
        required_signals = [
            'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT',
            'main_force_conviction_index_D', 'vacuum_zone_magnitude_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_trend_momentum"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # 使用新的“大一统”信号重构整体健康度计算
        health_score = (
            (df['SCORE_CHIP_STRATEGIC_POSTURE'].add(1)/2) *
            (df['SCORE_CHIP_BATTLEFIELD_GEOGRAPHY'].add(1)/2) *
            (df['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'].add(1)/2)
        ).pow(1/3)
        # 维度一: 引擎功率
        slope = health_score.diff(1).fillna(0)
        accel = slope.diff(1).fillna(0)
        norm_slope = get_adaptive_mtf_normalized_bipolar_score(slope, df_index, tf_weights)
        norm_accel = get_adaptive_mtf_normalized_bipolar_score(accel, df_index, tf_weights)
        engine_power_score = (norm_slope.add(1)/2 * norm_accel.clip(lower=-1, upper=1).add(1)/2).pow(0.5) * 2 - 1
        # 维度二: 燃料品质
        conviction = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        fuel_quality_score = get_adaptive_mtf_normalized_bipolar_score(conviction, df_index, tf_weights)
        # 维度三: 喷管效率
        vacuum = self._get_safe_series(df, df, 'vacuum_zone_magnitude_D', 0.0, method_name="_diagnose_axiom_trend_momentum")
        nozzle_efficiency_score = get_adaptive_mtf_normalized_bipolar_score(vacuum, df_index, tf_weights)
        # 最终非线性合成
        final_score = (
            (engine_power_score.add(1)/2) *
            (fuel_quality_score.add(1)/2) *
            (nozzle_efficiency_score.add(1)/2)
        ).pow(1/3) * 2 - 1
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V5.0 · 价筹张力版】筹码公理五：诊断“价筹张力”
        - 核心思想: 废弃线性的差值计算，引入“弹性势能模型”，将价筹背离视为一个积蓄能量的弹性系统。
        - 诊断维度:
          1. 分歧向量 (Disagreement Vector): 价格与筹码趋势分离的方向与距离。
          2. 张力强度 (Tension Magnitude): 分歧的持续性与期间注入的能量（成交量）共同决定了势能的大小。
        - 非线性合成: 张力 = 向量方向 * 向量距离 * (1 + 张力强度)。旨在识别那些持续的、由巨量推动的、蕴含巨大反转势能的高质量背离。
        """
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # 1. 分歧向量 (Disagreement Vector)
        concentration_trend = self._get_safe_series(df, df, 'SLOPE_5_winner_concentration_90pct_D', 0.0, method_name="_diagnose_axiom_divergence")
        price_trend = self._get_safe_series(df, df, 'SLOPE_5_close_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_concentration_trend = get_adaptive_mtf_normalized_bipolar_score(concentration_trend, df_index, tf_weights)
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend, df_index, tf_weights)
        disagreement_vector = norm_concentration_trend - norm_price_trend
        # 2. 张力强度 (Tension Magnitude)
        # 2.1 持续性: 使用分歧向量的滚动标准差来衡量其稳定性与持续性
        persistence = disagreement_vector.rolling(window=13, min_periods=5).std().fillna(0)
        norm_persistence = get_adaptive_mtf_normalized_score(persistence, df_index, tf_weights=tf_weights)
        # 2.2 能量注入: 衡量分歧期间的成交量水平
        volume = self._get_safe_series(df, df, 'volume_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_volume = get_adaptive_mtf_normalized_score(volume, df_index, tf_weights=tf_weights)
        # 能量注入只在有分歧时才重要
        energy_injection = norm_volume * disagreement_vector.abs()
        tension_magnitude = (norm_persistence * energy_injection).pow(0.5)
        # 3. 最终非线性合成
        # 基础分歧向量的大小，由张力强度进行非线性放大
        final_score = disagreement_vector * (1 + tension_magnitude * 1.5) # 1.5是放大系数
        return final_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _diagnose_chip_lockdown_degree(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · 逻辑修复版】诊断“筹码锁定度”
        - 核心修复: 增加了对 `winner_stability_index_D` 的归一化处理。此前代码错误地假设其为[0,1]区间的值，
                      导致计算结果饱和。现在，盈利锁定和亏损锁定都在归一化后进行加权，确保逻辑的正确性。
        - 探针逻辑: 在指定的探针日期，输出所有原料数据和关键计算过程值。
        """
        # 1. 获取原料数据
        locked_profit_raw = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_chip_lockdown_degree")
        loser_pain_raw = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="_diagnose_chip_lockdown_degree")
        # 2. 关键计算过程
        # 对盈利锁定原始值进行归一化
        locked_profit_score = utils.normalize_score(locked_profit_raw, df.index, 55, ascending=True)
        locked_loss_score = utils.normalize_score(loser_pain_raw, df.index, 55, ascending=True)
        # 使用归一化后的分数进行计算
        lockdown_degree = (locked_profit_score * 0.6 + locked_loss_score * 0.4).clip(0, 1).fillna(0.0)
        # 3. 植入探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date in df.index:
                print(f"    -> [探针: _diagnose_chip_lockdown_degree] @ {probe_date.date()}:")
                print(f"       - 原料: winner_stability_index_D: {locked_profit_raw.loc[probe_date]:.4f}")
                print(f"       - 过程: locked_profit_score (盈利锁定归一化): {locked_profit_score.loc[probe_date]:.4f}") # 更新探针输出
                print(f"       - 原料: loser_pain_index_D: {loser_pain_raw.loc[probe_date]:.4f}")
                print(f"       - 过程: locked_loss_score (亏损锁定归一化): {locked_loss_score.loc[probe_date]:.4f}")
                print(f"       - 结果: final lockdown_degree: {lockdown_degree.loc[probe_date]:.4f}")
        return lockdown_degree.astype(np.float32)

    def _diagnose_structural_consensus(self, df: pd.DataFrame, cost_structure_scores: pd.Series, holder_sentiment_scores: pd.Series) -> pd.Series:
        """
        【V4.4 · 探针清理版】诊断筹码同调驱动力
        - 核心清理: 移除了方法内的调试探针逻辑，净化日志输出。
        """
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
        # [删除] 移除了方法内的调试探针逻辑
        return final_score.astype(np.float32)

    def _diagnose_control_sovereignty(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 主力控盘权版】诊断主力对筹码的控制权
        - 核心重构: 废弃中性的“锁定度”概念，引入直指博弈核心的“主力控盘权”三维诊断模型。
        - 诊断维度:
          1. 静态锁定度 (Static Lockdown): 盘面有多稳定？(基础)
          2. 主力控制力 (Sovereign's Grip): 这种稳定是否由一个强大的主力所主导？(核心)
          3. 趋势脆弱性 (Trend Fragility): 当前的统治是否存在潜在的崩溃风险？(风险对冲)
        - 非线性合成: 控盘权 = (静态锁定度 * 主力控制力) / (1 + 趋势脆弱性)，旨在识别由强大主力主导的、内部结构稳固的、高质量控盘状态。
        """
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 1. 静态锁定度维度
        winner_stability = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_control_sovereignty")
        loser_pain = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="_diagnose_control_sovereignty")
        static_lockdown_score = (
            get_adaptive_mtf_normalized_score(winner_stability, df.index, ascending=True, tf_weights=tf_weights) * 0.6 +
            get_adaptive_mtf_normalized_score(loser_pain, df.index, ascending=True, tf_weights=tf_weights) * 0.4
        ).fillna(0.0)
        # 2. 主力控制力维度
        control_solidity = self._get_safe_series(df, df, 'control_solidity_index_D', 0.0, method_name="_diagnose_control_sovereignty")
        main_force_conviction = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_control_sovereignty")
        peak_control_transfer = self._get_safe_series(df, df, 'peak_control_transfer_D', 0.0, method_name="_diagnose_control_sovereignty")
        sovereign_grip_score = (
            get_adaptive_mtf_normalized_score(control_solidity, df.index, ascending=True, tf_weights=tf_weights) * 0.5 +
            get_adaptive_mtf_normalized_score(main_force_conviction, df.index, ascending=True, tf_weights=tf_weights) * 0.3 +
            get_adaptive_mtf_normalized_score(peak_control_transfer.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights) * 0.2
        ).fillna(0.0)
        # 3. 趋势脆弱性维度 (风险对冲项)
        chip_fatigue = self._get_safe_series(df, df, 'chip_fatigue_index_D', 0.5, method_name="_diagnose_control_sovereignty")
        vol_asymmetry = self._get_safe_series(df, df, 'volatility_asymmetry_index_D', 0.0, method_name="_diagnose_control_sovereignty")
        fragility_score = (
            get_adaptive_mtf_normalized_score(chip_fatigue, df.index, ascending=True, tf_weights=tf_weights) * 0.5 +
            get_adaptive_mtf_normalized_score(vol_asymmetry.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights) * 0.5
        ).fillna(0.0)
        # 4. 三维非线性合成
        # 核心驱动力是 (静态锁定 * 主力控制)，脆弱性作为分母进行惩罚
        control_sovereignty = (static_lockdown_score * sovereign_grip_score) / (1 + fragility_score)
        return control_sovereignty.clip(0, 1).fillna(0.0).astype(np.float32)

    def _diagnose_absorption_echo(self, df: pd.DataFrame, divergence_scores: pd.Series) -> pd.Series:
        """
        【V5.0 · 吸筹回声版】诊断看涨背离的战术意图
        - 核心思想: 看涨背离是一个“主力借恐慌吸筹”的战术事件。本函数旨在捕捉该事件留下的“回声”。
        - 诊断三要素:
          1. 恐慌声源: 存在明确的价格快速下跌。
          2. 逆流介质: 存在价跌筹码集中的基础背离。
          3. 主力回声: 存在主力资金逆势净流入的直接证据。
        - 非线性合成: 回声强度 = 声源 * 介质 * 主力回声。三者缺一不可。
        """
        # 将 main_force_net_inflow_rate_D 替换为 main_force_net_flow_calibrated_D
        required_signals = ['SLOPE_5_close_D', 'main_force_net_flow_calibrated_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_absorption_echo"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # 1. 恐慌声源 (价格下跌)
        price_trend = self._get_safe_series(df, df, 'SLOPE_5_close_D', 0.0, method_name="_diagnose_absorption_echo")
        panic_source_score = get_adaptive_mtf_normalized_score(price_trend.abs(), df_index, tf_weights) * (price_trend < 0)
        # 2. 逆流介质 (基础背离)
        counter_flow_medium_score = divergence_scores.clip(lower=0)
        # 3. 主力回声 (主力逆势流入)
        # 将 main_force_net_inflow_rate_D 替换为 main_force_net_flow_calibrated_D
        mf_inflow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_absorption_echo")
        main_force_echo_score = get_adaptive_mtf_normalized_score(mf_inflow, df_index, tf_weights) * (mf_inflow > 0) * (price_trend < 0)
        # 最终合成
        final_score = (panic_source_score * counter_flow_medium_score * main_force_echo_score).pow(1/3)
        return final_score.clip(0, 1).fillna(0.0).astype(np.float32)

    def _diagnose_distribution_whisper(self, df: pd.DataFrame, divergence_scores: pd.Series) -> pd.Series:
        """
        【V5.0 · 派发诡影版】诊断看跌背离的战术意图
        - 核心思想: 看跌背离是一个“主力借狂热派发”的战术事件。本函数旨在捕捉该事件留下的“诡影”。
        - 诊断三要素:
          1. 狂热背景: 存在价格上涨引发的市场狂热。
          2. 背离诡影: 存在价涨筹码分散的基础背离。
          3. 主力抽离: 存在主力资金顺势净流出的直接证据。
        - 非线性合成: 诡影风险 = 背景 * 诡影 * 主力抽离。三者缺一不可。
        """
        # 将 main_force_net_inflow_rate_D 替换为 main_force_net_flow_calibrated_D
        required_signals = ['SLOPE_5_close_D', 'main_force_net_flow_calibrated_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_distribution_whisper"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        df_index = df.index
        # 1. 狂热背景 (价格上涨)
        price_trend = self._get_safe_series(df, df, 'SLOPE_5_close_D', 0.0, method_name="_diagnose_distribution_whisper")
        fomo_backdrop_score = get_adaptive_mtf_normalized_score(price_trend.abs(), df_index, tf_weights) * (price_trend > 0)
        # 2. 背离诡影 (基础背离)
        divergence_shadow_score = divergence_scores.clip(upper=0).abs()
        # 3. 主力抽离 (主力顺势流出)
        # 将 main_force_net_inflow_rate_D 替换为 main_force_net_flow_calibrated_D
        mf_inflow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_distribution_whisper")
        main_force_retreat_score = get_adaptive_mtf_normalized_score(mf_inflow.abs(), df_index, tf_weights) * (mf_inflow < 0) * (price_trend > 0)
        # 最终合成
        final_score = (fomo_backdrop_score * divergence_shadow_score * main_force_retreat_score).pow(1/3)
        return final_score.clip(0, 1).fillna(0.0).astype(np.float32)







