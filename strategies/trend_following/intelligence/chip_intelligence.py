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
        df_index = df.index # [代码修改] 使用传入的 df.index
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
        # 修改代码行：使用(1 - 获利盘稳定度)作为短期获利盘压力的代理
        winner_stability = self._get_safe_series(df, df, 'winner_stability_index_D', 0.5, method_name="run_chip_intelligence_command") # [代码修改]
        profit_pressure = 1 - winner_stability
        cleanliness_raw_score = ((1 - chip_fault) * (1 - profit_pressure)).pow(0.5).fillna(0.5)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        cleanliness_score = get_adaptive_mtf_normalized_score(cleanliness_raw_score, df.index, ascending=True, tf_weights=tf_weights)
        all_chip_states['SCORE_CHIP_CLEANLINESS'] = cleanliness_score.astype(np.float32)
        # 信号2: 筹码锁定度 (SCORE_CHIP_LOCKDOWN_DEGREE)
        # 修改代码行：使用获利盘稳定度代表盈利锁定，使用套牢盘痛苦指数代表亏损锁定
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
        【V3.2 · 上下文修复版】筹码公理一：诊断筹码“聚散”动态
        - 【V3.2 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        required_signals = [
            'cost_gini_coefficient_D', 'structural_node_count_D', 'peak_separation_ratio_D',
            'winner_concentration_90pct_D', 'ZIG_5_5.0_D', 'peak_exchange_purity_D'
        ] + [f'SLOPE_{p}_winner_concentration_90pct_D' for p in periods if f'SLOPE_{p}_winner_concentration_90pct_D' in df.columns]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码集中度探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        concentration_level_raw = 1 - self._get_safe_series(df, df, 'cost_gini_coefficient_D', 0.5, method_name="_diagnose_axiom_concentration")
        concentration_trend_raw = pd.Series(0.0, index=df.index)
        for p in periods:
            slope_col = f'SLOPE_{p}_winner_concentration_90pct_D'
            concentration_trend_raw += self._get_safe_series(df, df, slope_col, 0.0, method_name="_diagnose_axiom_concentration")
        concentration_trend_raw /= len(periods)
        node_count_raw = self._get_safe_series(df, df, 'structural_node_count_D', 3.0, method_name="_diagnose_axiom_concentration")
        separation_raw = self._get_safe_series(df, df, 'peak_separation_ratio_D', 50.0, method_name="_diagnose_axiom_concentration")
        peak_fusion_raw = (1 - utils.normalize_score(node_count_raw, df.index, 55, ascending=True)) * \
                          (1 - utils.normalize_score(separation_raw, df.index, 55, ascending=True))
        zigzag_trend_raw = self._get_safe_series(df, df, 'ZIG_5_5.0_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_concentration")
        peak_exchange_purity_raw = self._get_safe_series(df, df, 'peak_exchange_purity_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_concentration")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        level_score = get_adaptive_mtf_normalized_bipolar_score(concentration_level_raw, df.index, tf_weights, sensitivity=0.2)
        trend_score = get_adaptive_mtf_normalized_bipolar_score(concentration_trend_raw, df.index, tf_weights, sensitivity=1.0)
        fusion_score = get_adaptive_mtf_normalized_bipolar_score(peak_fusion_raw, df.index, tf_weights, sensitivity=0.5)
        zigzag_score = get_adaptive_mtf_normalized_bipolar_score(zigzag_trend_raw, df.index, tf_weights, sensitivity=0.05)
        peak_exchange_purity_score = get_adaptive_mtf_normalized_bipolar_score(peak_exchange_purity_raw, df.index, tf_weights, sensitivity=0.5)
        final_score = (level_score * 0.30 + trend_score * 0.20 + fusion_score * 0.20 + zigzag_score * 0.10 + peak_exchange_purity_score * 0.20).clip(-1, 1)
        return final_score

    def _diagnose_axiom_cost_structure(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V3.2 · 上下文修复版】筹码公理二：诊断“成本结构”动态
        - 【V3.2 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        required_signals = [
            'winner_loser_momentum_D', 'chip_fault_magnitude_D', 'cost_structure_skewness_D',
            'pressure_validation_score_D', 'support_validation_score_D',
            'cost_gini_coefficient_D', 'structural_tension_index_D', 'structural_leverage_D'
        ]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码成本结构探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        momentum_raw = self._get_safe_series(df, df, 'winner_loser_momentum_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        fault_raw = self._get_safe_series(df, df, 'chip_fault_magnitude_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        skewness_raw = self._get_safe_series(df, df, 'cost_structure_skewness_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        pressure_validation_raw = self._get_safe_series(df, df, 'pressure_validation_score_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        support_validation_raw = self._get_safe_series(df, df, 'support_validation_score_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        cost_gini_raw = self._get_safe_series(df, df, 'cost_gini_coefficient_D', pd.Series(0.5, index=df.index), method_name="_diagnose_axiom_cost_structure")
        structural_tension_raw = self._get_safe_series(df, df, 'structural_tension_index_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        structural_leverage_raw = self._get_safe_series(df, df, 'structural_leverage_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        momentum_score = get_adaptive_mtf_normalized_bipolar_score(momentum_raw, df.index, tf_weights, sensitivity=1.0)
        fault_score = get_adaptive_mtf_normalized_bipolar_score(fault_raw, df.index, tf_weights, sensitivity=0.5) * -1
        skewness_score = get_adaptive_mtf_normalized_bipolar_score(skewness_raw, df.index, tf_weights, sensitivity=0.5)
        pressure_validation_score = get_adaptive_mtf_normalized_bipolar_score(pressure_validation_raw, df.index, tf_weights, sensitivity=0.5)
        support_validation_score = get_adaptive_mtf_normalized_bipolar_score(support_validation_raw, df.index, tf_weights, sensitivity=0.5)
        cost_gini_score = get_adaptive_mtf_normalized_bipolar_score(1 - cost_gini_raw, df.index, tf_weights, sensitivity=0.2)
        structural_tension_score = get_adaptive_mtf_normalized_bipolar_score(structural_tension_raw, df.index, tf_weights, sensitivity=0.5)
        structural_leverage_score = get_adaptive_mtf_normalized_bipolar_score(structural_leverage_raw, df.index, tf_weights, sensitivity=0.5)
        final_score = (
            momentum_score * 0.20 +
            skewness_score * 0.10 +
            support_validation_score * 0.10 +
            cost_gini_score * 0.20 +
            structural_tension_score * 0.15 +
            structural_leverage_score * 0.15 +
            fault_score * 0.05 -
            pressure_validation_score * 0.05
        ).clip(-1, 1)
        return final_score

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V3.2 · 上下文修复版】筹码公理三：诊断“持股心态”动态
        - 【V3.2 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        df_index = df.index
        required_signals = [
            'winner_stability_index_D', 'conviction_flow_index_D', 'loser_pain_index_D', 'chip_fatigue_index_D',
            'covert_accumulation_signal_D'
        ]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [持股心态探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        winner_stability_raw = self._get_safe_series(df, df, 'winner_stability_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        conviction_flow_raw = self._get_safe_series(df, df, 'conviction_flow_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        pain_raw = self._get_safe_series(df, df, 'loser_pain_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        fatigue_raw = self._get_safe_series(df, df, 'chip_fatigue_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        covert_accumulation_raw = self._get_safe_series(df, df, 'covert_accumulation_signal_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        winner_stability_score = get_adaptive_mtf_normalized_bipolar_score(winner_stability_raw, df_index, tf_weights, sensitivity=0.5)
        conviction_flow_score = get_adaptive_mtf_normalized_bipolar_score(conviction_flow_raw, df_index, tf_weights, sensitivity=0.5)
        pain_score = get_adaptive_mtf_normalized_bipolar_score(pain_raw, df_index, tf_weights, sensitivity=5.0)
        fatigue_score = get_adaptive_mtf_normalized_bipolar_score(fatigue_raw, df_index, tf_weights, sensitivity=5.0)
        covert_accumulation_score = get_adaptive_mtf_normalized_bipolar_score(covert_accumulation_raw, df_index, tf_weights, sensitivity=0.5)
        final_score = (
            winner_stability_score * 0.35 +
            conviction_flow_score * 0.35 +
            covert_accumulation_score * 0.10 -
            pain_score * 0.10 -
            fatigue_score * 0.10
        ).clip(-1, 1)
        return final_score

    def _diagnose_axiom_peak_integrity(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V3.1 · 上下文修复版】筹码公理四：诊断“筹码峰形态”
        - 【V3.1 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        required_signals = [
            'dominant_peak_cost_D', 'dominant_peak_solidity_D', 'price_volume_entropy_D',
            'primary_peak_kurtosis_D'
        ]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        price_vs_peak_raw = self._get_safe_series(df, df, 'close_D', method_name="_diagnose_axiom_peak_integrity") - self._get_safe_series(df, df, 'dominant_peak_cost_D', self._get_safe_series(df, df, 'close_D', method_name="_diagnose_axiom_peak_integrity"), method_name="_diagnose_axiom_peak_integrity")
        peak_solidity_raw = self._get_safe_series(df, df, 'dominant_peak_solidity_D', pd.Series(0.5, index=df.index), method_name="_diagnose_axiom_peak_integrity")
        price_volume_entropy_raw = self._get_safe_series(df, df, 'price_volume_entropy_D', pd.Series(0.5, index=df.index), method_name="_diagnose_axiom_peak_integrity")
        primary_peak_kurtosis_raw = self._get_safe_series(df, df, 'primary_peak_kurtosis_D', pd.Series(3.0, index=df.index), method_name="_diagnose_axiom_peak_integrity")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_vs_peak_score = get_adaptive_mtf_normalized_bipolar_score(price_vs_peak_raw, df.index, tf_weights, sensitivity=1.2)
        peak_solidity_score = get_adaptive_mtf_normalized_score(peak_solidity_raw, df.index, ascending=True, tf_weights=tf_weights)
        price_volume_entropy_score = get_adaptive_mtf_normalized_bipolar_score(price_volume_entropy_raw * -1, df.index, tf_weights, sensitivity=0.5)
        primary_peak_kurtosis_score = get_adaptive_mtf_normalized_bipolar_score(primary_peak_kurtosis_raw, df.index, tf_weights, sensitivity=2.0)
        final_score = (
            price_vs_peak_score * peak_solidity_score * 0.5 +
            price_volume_entropy_score * 0.25 +
            primary_peak_kurtosis_score * 0.25
        ).clip(-1, 1)
        return final_score

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



