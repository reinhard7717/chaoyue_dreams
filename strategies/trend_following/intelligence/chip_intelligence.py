import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Union
from strategies.trend_following import utils
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, get_adaptive_mtf_normalized_score, bipolar_to_exclusive_unipolar

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

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
        【V10.1 · 调用修复版】筹码情报总指挥
        - 核心修复: 修正了方法内部对 _get_safe_series 的调用，确保正确传递 data_source 参数，
                      解决了因参数错位导致的“未知数据源类型”警告。
        """
        all_chip_states = {}
        periods = [5, 13, 21, 55]
        # 步骤一: 诊断四大公理，生成纯粹的筹码原子信号
        concentration_scores = self._diagnose_axiom_concentration(df, periods)
        cost_structure_scores = self._diagnose_axiom_cost_structure(df, periods)
        holder_sentiment_scores = self._diagnose_axiom_holder_sentiment(df, periods)
        peak_integrity_scores = self._diagnose_axiom_peak_integrity(df, periods)
        divergence_scores = self._diagnose_axiom_divergence(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_DIVERGENCE'] = divergence_scores
        all_chip_states['SCORE_CHIP_AXIOM_CONCENTRATION'] = concentration_scores
        all_chip_states['SCORE_CHIP_AXIOM_COST_STRUCTURE'] = cost_structure_scores
        all_chip_states['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = holder_sentiment_scores
        all_chip_states['SCORE_CHIP_AXIOM_PEAK_INTEGRITY'] = peak_integrity_scores
        chip_trend_momentum_scores = self._diagnose_axiom_trend_momentum(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_TREND_MOMENTUM'] = chip_trend_momentum_scores
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(divergence_scores)
        all_chip_states['SCORE_CHIP_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_chip_states['SCORE_CHIP_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        # 步骤二: 工程化超级原子信号
        # 信号1: 筹码干净度 (SCORE_CHIP_CLEANLINESS)
        chip_fault = self._get_safe_series(df, df, 'chip_fault_blockage_ratio_D', 0.5, method_name="run_chip_intelligence_command") # [代码修改]
        # 使用(1 - 获利盘稳定度)作为短期获利盘压力的代理
        winner_stability = self._get_safe_series(df, df, 'winner_stability_index_D', 0.5, method_name="run_chip_intelligence_command") # [代码修改]
        profit_pressure = 1 - winner_stability
        cleanliness_raw_score = ((1 - chip_fault) * (1 - profit_pressure)).pow(0.5).fillna(0.5)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        cleanliness_score = get_adaptive_mtf_normalized_score(cleanliness_raw_score, df.index, ascending=True, tf_weights=tf_weights)
        all_chip_states['SCORE_CHIP_CLEANLINESS'] = cleanliness_score.astype(np.float32)
        # 信号2: 筹码锁定度 (SCORE_CHIP_LOCKDOWN_DEGREE)
        # 使用获利盘稳定度代表盈利锁定，使用套牢盘痛苦指数代表亏损锁定
        locked_profit = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="run_chip_intelligence_command") # [代码修改]
        loser_pain = self._get_safe_series(df, df, 'loser_pain_index_D', 0.0, method_name="run_chip_intelligence_command") # [代码修改]
        # 痛苦指数越高，越不愿卖出，锁定度越高，因此直接归一化
        locked_loss = utils.normalize_score(loser_pain, df.index, 55, ascending=True)
        lockdown_degree = (locked_profit * 0.6 + locked_loss * 0.4).clip(0, 1).fillna(0.0) # 盈利锁定权重更高
        all_chip_states['SCORE_CHIP_LOCKDOWN_DEGREE'] = lockdown_degree.astype(np.float32)
        # 信号3: 结构共识分 (SCORE_CHIP_STRUCTURAL_CONSENSUS)
        bullish_structure_score = cost_structure_scores.clip(lower=0)
        positive_sentiment_score = holder_sentiment_scores.clip(lower=0)
        structural_consensus_score = (bullish_structure_score * positive_sentiment_score).pow(0.5)
        all_chip_states['SCORE_CHIP_STRUCTURAL_CONSENSUS'] = structural_consensus_score.astype(np.float32)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [ChipIntelligence Debug] @ {probe_date_for_loop.date()}: Signals returned by ChipIntelligence:")
                for k, v in all_chip_states.items():
                    if isinstance(v, pd.Series) and probe_date_for_loop in v.index:
                        print(f"       - {k}: {v.loc[probe_date_for_loop]:.4f}")
                    else:
                        print(f"       - {k}: {v}")
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

    def _diagnose_axiom_concentration(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V4.0 · 集结效率与纯度版】筹码公理一：诊断“集结效率与纯度”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 引入高频指标，将对“集中”的评估从静态数量提升到动态质量。
        - 核心证据 (纯度): `peak_exchange_purity`，高纯度的交换意味着真实的、具有攻击性的筹码集结。
        - 辅助证据 (效率): `impulse_quality_ratio`，高质量的上涨脉冲验证了集结过程的健康度。
        """
        required_signals = [
            'cost_gini_coefficient_D', 'structural_node_count_D', 'peak_separation_ratio_D',
            'peak_exchange_purity_D', 'impulse_quality_ratio_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_concentration"):
            return pd.Series(0.0, index=df.index)
        concentration_level_raw = 1 - self._get_safe_series(df, df, 'cost_gini_coefficient_D', 0.5, method_name="_diagnose_axiom_concentration")
        node_count_raw = self._get_safe_series(df, df, 'structural_node_count_D', 3.0, method_name="_diagnose_axiom_concentration")
        separation_raw = self._get_safe_series(df, df, 'peak_separation_ratio_D', 50.0, method_name="_diagnose_axiom_concentration")
        peak_fusion_raw = (1 - utils.normalize_score(node_count_raw, df.index, 55, ascending=True)) * \
                          (1 - utils.normalize_score(separation_raw, df.index, 55, ascending=True))
        peak_exchange_purity_raw = self._get_safe_series(df, df, 'peak_exchange_purity_D', 0.0, method_name="_diagnose_axiom_concentration")
        impulse_quality_raw = self._get_safe_series(df, df, 'impulse_quality_ratio_D', 0.0, method_name="_diagnose_axiom_concentration")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        level_score = get_adaptive_mtf_normalized_bipolar_score(concentration_level_raw, df.index, tf_weights, sensitivity=0.2)
        fusion_score = get_adaptive_mtf_normalized_bipolar_score(peak_fusion_raw, df.index, tf_weights, sensitivity=0.5)
        purity_score = get_adaptive_mtf_normalized_bipolar_score(peak_exchange_purity_raw, df.index, tf_weights, sensitivity=0.5)
        quality_score = get_adaptive_mtf_normalized_bipolar_score(impulse_quality_raw, df.index, tf_weights, sensitivity=0.5)
        final_score = (
            purity_score * 0.4 +             # 交换纯度是核心
            level_score * 0.25 +             # 集中度水平是基础
            fusion_score * 0.2 +             # 峰形融合是过程
            quality_score * 0.15             # 脉冲品质是验证
        ).clip(-1, 1)
        return final_score.astype(np.float32)

    def _diagnose_axiom_cost_structure(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V4.0 · 攻防阵地有效性版】筹码公理二：诊断“攻防阵地有效性”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 从分析静态的成本结构，升级为通过高频数据验证这些结构在实战中的攻防有效性。
        - 核心证据 (验证): `support_validation_strength` 和 `pressure_rejection_strength` 直接量化了支撑和压力的真实强度。
        - 核心证据 (利用): `vacuum_traversal_efficiency` 衡量了主力利用结构优势（真空区）的能力。
        """
        required_signals = [
            'support_validation_strength_D', 'pressure_rejection_strength_D', 'vacuum_traversal_efficiency_D',
            'structural_tension_index_D', 'structural_leverage_D', 'chip_fault_magnitude_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_cost_structure"):
            return pd.Series(0.0, index=df.index)
        support_validation_raw = self._get_safe_series(df, df, 'support_validation_strength_D', 0.0, method_name="_diagnose_axiom_cost_structure")
        pressure_rejection_raw = self._get_safe_series(df, df, 'pressure_rejection_strength_D', 0.0, method_name="_diagnose_axiom_cost_structure")
        vacuum_efficiency_raw = self._get_safe_series(df, df, 'vacuum_traversal_efficiency_D', 0.0, method_name="_diagnose_axiom_cost_structure")
        structural_tension_raw = self._get_safe_series(df, df, 'structural_tension_index_D', 0.0, method_name="_diagnose_axiom_cost_structure")
        structural_leverage_raw = self._get_safe_series(df, df, 'structural_leverage_D', 0.0, method_name="_diagnose_axiom_cost_structure")
        fault_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', 0.0, method_name="_diagnose_axiom_cost_structure")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        support_score = get_adaptive_mtf_normalized_bipolar_score(support_validation_raw, df.index, tf_weights, sensitivity=0.5)
        rejection_score = get_adaptive_mtf_normalized_bipolar_score(pressure_rejection_raw, df.index, tf_weights, sensitivity=0.5)
        vacuum_score = get_adaptive_mtf_normalized_bipolar_score(vacuum_efficiency_raw, df.index, tf_weights, sensitivity=0.5)
        tension_score = get_adaptive_mtf_normalized_bipolar_score(structural_tension_raw, df.index, tf_weights, sensitivity=0.5)
        leverage_score = get_adaptive_mtf_normalized_bipolar_score(structural_leverage_raw, df.index, tf_weights, sensitivity=0.5)
        fault_penalty = get_adaptive_mtf_normalized_score(fault_raw, df.index, ascending=False, tf_weights=tf_weights)
        final_score = (
            (support_score * 0.4 + vacuum_score * 0.3 - rejection_score * 0.3).clip(-1, 1) * 0.6 +
            (tension_score * 0.5 + leverage_score * 0.5).clip(-1, 1) * 0.4
        ) * fault_penalty
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V4.0 · 持仓稳定性与博弈意图版】筹码公理三：诊断“持仓稳定性与博弈意图”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 通过高频日内行为指标，从侧面推断和验证持股者的真实心态和稳定性。
        - 核心证据 (稳定化): `floating_chip_cleansing_efficiency`，衡量主力主动优化持仓结构的意图。
        - 核心证据 (承压测试): `profit_realization_quality` 和 `capitulation_absorption_index`，评估在压力下持仓结构的真实稳定性。
        """
        required_signals = [
            'winner_stability_index_D', 'conviction_flow_index_D', 'floating_chip_cleansing_efficiency_D',
            'profit_realization_quality_D', 'capitulation_absorption_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_holder_sentiment"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        winner_stability_raw = self._get_safe_series(df, df, 'winner_stability_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        conviction_flow_raw = self._get_safe_series(df, df, 'conviction_flow_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        cleansing_efficiency_raw = self._get_safe_series(df, df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        profit_quality_raw = self._get_safe_series(df, df, 'profit_realization_quality_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        absorption_index_raw = self._get_safe_series(df, df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_axiom_holder_sentiment")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        winner_stability_score = get_adaptive_mtf_normalized_bipolar_score(winner_stability_raw, df_index, tf_weights, sensitivity=0.5)
        conviction_flow_score = get_adaptive_mtf_normalized_bipolar_score(conviction_flow_raw, df_index, tf_weights, sensitivity=0.5)
        cleansing_score = get_adaptive_mtf_normalized_bipolar_score(cleansing_efficiency_raw, df_index, tf_weights, sensitivity=0.5)
        profit_quality_score = get_adaptive_mtf_normalized_bipolar_score(profit_quality_raw, df_index, tf_weights, sensitivity=0.5)
        absorption_score = get_adaptive_mtf_normalized_bipolar_score(absorption_index_raw, df_index, tf_weights, sensitivity=0.5)
        final_score = (
            (cleansing_score * 0.5 + absorption_score * 0.5).clip(-1, 1) * 0.4 + # 主动行为
            (winner_stability_score * 0.5 + profit_quality_score * 0.5).clip(-1, 1) * 0.4 + # 状态结果
            conviction_flow_score * 0.2 # 跨日信念流转
        ).clip(-1, 1)
        return final_score.astype(np.float32)

    def _diagnose_axiom_peak_integrity(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V4.0 · 主峰控制权与有效性版】筹码公理四：诊断“主峰控制权与有效性”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 从静态的形态分析，升级为对主峰的动态控制权和日内有效性的综合评估。
        - 核心证据 (控制权): `peak_control_transfer`，衡量主力是否在巩固其核心阵地。
        - 核心证据 (有效性): `intraday_posture_score`，验证主峰在日内实战中是支撑平台还是压力区。
        """
        required_signals = [
            'dominant_peak_cost_D', 'dominant_peak_solidity_D', 'peak_control_transfer_D',
            'intraday_posture_score_D', 'close_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_peak_integrity"):
            return pd.Series(0.0, index=df.index)
        price_vs_peak_raw = self._get_safe_series(df, df, 'close_D', method_name="_diagnose_axiom_peak_integrity") - self._get_safe_series(df, df, 'dominant_peak_cost_D', self._get_safe_series(df, df, 'close_D', method_name="_diagnose_axiom_peak_integrity"), method_name="_diagnose_axiom_peak_integrity")
        peak_solidity_raw = self._get_safe_series(df, df, 'dominant_peak_solidity_D', 0.5, method_name="_diagnose_axiom_peak_integrity")
        control_transfer_raw = self._get_safe_series(df, df, 'peak_control_transfer_D', 0.0, method_name="_diagnose_axiom_peak_integrity")
        intraday_posture_raw = self._get_safe_series(df, df, 'intraday_posture_score_D', 0.0, method_name="_diagnose_axiom_peak_integrity")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_vs_peak_score = get_adaptive_mtf_normalized_bipolar_score(price_vs_peak_raw, df.index, tf_weights, sensitivity=1.2)
        peak_solidity_score = get_adaptive_mtf_normalized_score(peak_solidity_raw, df.index, ascending=True, tf_weights=tf_weights)
        control_transfer_score = get_adaptive_mtf_normalized_bipolar_score(control_transfer_raw, df.index, tf_weights, sensitivity=0.5)
        intraday_posture_score = get_adaptive_mtf_normalized_bipolar_score(intraday_posture_raw, df.index, tf_weights, sensitivity=0.5)
        # 价格突破主峰，且主峰稳固，是基础看涨信号
        base_bullish_signal = (price_vs_peak_score * peak_solidity_score).clip(0, 1)
        # 日内姿态和控制权转移是强力验证
        validation_score = (intraday_posture_score * 0.6 + control_transfer_score * 0.4).clip(-1, 1)
        final_score = (base_bullish_signal * 0.5 + validation_score * 0.5).clip(-1, 1)
        return final_score.astype(np.float32)

    def _diagnose_axiom_trend_momentum(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.1 · 上下文修复版】筹码公理六：诊断“筹码趋势动量”
        - 【V2.1 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        df_index = df.index
        required_signals = ['structural_entropy_change_D', 'peak_mass_transfer_rate_D', 'constructive_turnover_ratio_D']
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码趋势动量探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df_index)
        entropy_change_raw = self._get_safe_series(df, df, 'structural_entropy_change_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_momentum")
        mass_transfer_raw = self._get_safe_series(df, df, 'peak_mass_transfer_rate_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_momentum")
        constructive_turnover_raw = self._get_safe_series(df, df, 'constructive_turnover_ratio_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_momentum")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        entropy_change_score = get_adaptive_mtf_normalized_bipolar_score(entropy_change_raw * -1, df_index, tf_weights, sensitivity=0.1)
        mass_transfer_score = get_adaptive_mtf_normalized_bipolar_score(mass_transfer_raw, df_index, tf_weights, sensitivity=0.1)
        constructive_turnover_score = get_adaptive_mtf_normalized_bipolar_score(constructive_turnover_raw, df_index, tf_weights, sensitivity=0.1)
        chip_trend_momentum_score = (
            entropy_change_score * 0.4 +
            mass_transfer_score * 0.3 +
            constructive_turnover_score * 0.3
        ).clip(-1, 1)
        return chip_trend_momentum_score.astype(np.float32)

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.2 · 上下文修复版】筹码公理五：诊断筹码“背离”动态
        - 【V1.2 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        required_signals = ['pct_change_D', 'SLOPE_5_short_term_concentration_90pct_D']
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, df, 'pct_change_D', method_name="_diagnose_axiom_divergence"), df.index, tf_weights)
        concentration_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, df, 'SLOPE_5_short_term_concentration_90pct_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_divergence"), df.index, tf_weights)
        divergence_score = (concentration_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)



