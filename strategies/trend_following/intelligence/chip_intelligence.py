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

    def _get_safe_series(self, data_source: Union[pd.DataFrame, Dict[str, pd.Series]], column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame或字典中获取Series，如果不存在则打印警告并返回默认Series。
        - 核心修复: 兼容处理 pd.DataFrame 和 Dict[str, pd.Series] 两种数据源。
        """
        df_index = self.strategy.df_indicators.index # 获取全局的DataFrame索引
        if isinstance(data_source, pd.DataFrame):
            if column_name not in data_source.columns:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            return data_source[column_name]
        elif isinstance(data_source, dict):
            if column_name not in data_source:
                print(f"    -> [筹码情报警告] 方法 '{method_name}' 缺少字典数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            # 确保从字典中取出的也是Series，并且索引对齐
            series = data_source[column_name]
            if isinstance(series, pd.Series):
                return series.reindex(df_index, fill_value=default_value)
            else: # 如果字典中存储的不是Series，则转换为Series
                return pd.Series(series, index=df_index)
        else:
            print(f"    -> [筹码情报警告] 方法 '{method_name}' 接收到未知数据源类型 {type(data_source)}，无法获取 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df_index)

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.9 · 结构共识信号版】筹码情报总指挥
        - 核心新增: 引入超级原子信号 `SCORE_CHIP_STRUCTURAL_CONSENSUS`。该信号通过融合“成本结构”和“持股心态”两大公理的正面得分，量化“健康结构”与“坚定信念”之间的共振强度。
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出筹码领域的原子公理信号和筹码背离信号。
        - 移除信号: SCORE_CHIP_BULLISH_RESONANCE, SCORE_CHIP_BEARISH_RESONANCE, BIPOLAR_CHIP_DOMAIN_HEALTH, SCORE_CHIP_BOTTOM_REVERSAL, SCORE_CHIP_TOP_REVERSAL。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【修正】移除 `OCH_D` 的计算，因为它现在已在数据层提前计算并添加到 `df` 中。
        - 【新增】添加调试打印，显示返回的 `all_chip_states` 内容。
        - 【修复】对 `SCORE_CHIP_CLEANLINESS` 进行多时间维度自适应归一化。
        """
        all_chip_states = {}
        periods = [5, 13, 21, 55]
        # 步骤一: 诊断四大公理，生成纯粹的筹码原子信号
        concentration_scores = self._diagnose_axiom_concentration(df, periods)
        cost_structure_scores = self._diagnose_axiom_cost_structure(df, periods)
        holder_sentiment_scores = self._diagnose_axiom_holder_sentiment(df, periods)
        peak_integrity_scores = self._diagnose_axiom_peak_integrity(df, periods)
        # 诊断筹码背离公理
        divergence_scores = self._diagnose_axiom_divergence(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_DIVERGENCE'] = divergence_scores
        # 将公理的诊断结果存入原子状态，供上层追溯
        all_chip_states['SCORE_CHIP_AXIOM_CONCENTRATION'] = concentration_scores
        all_chip_states['SCORE_CHIP_AXIOM_COST_STRUCTURE'] = cost_structure_scores
        all_chip_states['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = holder_sentiment_scores
        all_chip_states['SCORE_CHIP_AXIOM_PEAK_INTEGRITY'] = peak_integrity_scores
        # 移除 OCH_D 的计算，因为它现在已在数据层提前计算并添加到 df 中
        # 诊断筹码趋势动量公理 (需要依赖 OCH_D，所以放在后面)
        chip_trend_momentum_scores = self._diagnose_axiom_trend_momentum(df, periods)
        all_chip_states['SCORE_CHIP_AXIOM_TREND_MOMENTUM'] = chip_trend_momentum_scores
        # 引入筹码层面的看涨/看跌背离信号 (保持不变)
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(divergence_scores)
        all_chip_states['SCORE_CHIP_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_chip_states['SCORE_CHIP_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        # 步骤二: 工程化超级原子信号
        # 信号1: 筹码干净度 (SCORE_CHIP_CLEANLINESS)
        chip_fault = self._get_safe_series(df, 'chip_fault_blockage_ratio_D', 0.5, method_name="run_chip_intelligence_command")
        profit_pressure = self._get_safe_series(df, 'imminent_profit_taking_supply_D', 0.5, method_name="run_chip_intelligence_command")
        cleanliness_raw_score = ((1 - chip_fault) * (1 - profit_pressure)).pow(0.5).fillna(0.5)
        # 对 cleanliness_raw_score 进行多时间维度自适应归一化
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        cleanliness_score = get_adaptive_mtf_normalized_score(cleanliness_raw_score, df.index, ascending=True, tf_weights=tf_weights)
        all_chip_states['SCORE_CHIP_CLEANLINESS'] = cleanliness_score.astype(np.float32)
        # 信号2: 筹码锁定度 (SCORE_CHIP_LOCKDOWN_DEGREE)
        locked_profit = self._get_safe_series(df, 'locked_profit_rate_D', 0.0, method_name="run_chip_intelligence_command")
        locked_loss = self._get_safe_series(df, 'locked_loss_rate_D', 0.0, method_name="run_chip_intelligence_command")
        lockdown_degree = (locked_profit + locked_loss).clip(0, 1).fillna(0.0)
        all_chip_states['SCORE_CHIP_LOCKDOWN_DEGREE'] = lockdown_degree.astype(np.float32)
        # 新增代码块: 信号3: 结构共识分 (SCORE_CHIP_STRUCTURAL_CONSENSUS)
        # 逻辑: 只有当成本结构健康(分数为正)且持股心态积极(分数为正)时，共振才会发生。
        # 我们将两个分数的正值部分相乘，再开方以平滑结果。
        bullish_structure_score = cost_structure_scores.clip(lower=0)
        positive_sentiment_score = holder_sentiment_scores.clip(lower=0)
        structural_consensus_score = (bullish_structure_score * positive_sentiment_score).pow(0.5)
        all_chip_states['SCORE_CHIP_STRUCTURAL_CONSENSUS'] = structural_consensus_score.astype(np.float32)
        # --- Debugging output ---
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
        # --- End Debugging output ---
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
        【V3.0 · 微观筹码博弈增强与峰融合增强及列名引用修复版】筹码公理一：诊断筹码“聚散”动态
        - 核心修复: 遵循“先归一，后融合”原则。不再对原始值进行武断缩放，而是先将“集中度水平”和“集中趋势”
                      分别归一化为[-1, 1]的双极性分数，然后再进行加权融合，确保模型在不同市场环境下的健壮性。
        - 引入 `peak_fusion_indicator` (筹码峰融合指标) 作为判断筹码集中度的重要证据。
        - 【新增】引入 ZIGZAG 趋势作为辅助证据，增强对集中度有效性的判断。
        - 【修复】修正了引用 ZIGZAG 列名时，确保其与 `IndicatorService` 中 `merge_results` 方法添加后缀后的列名一致。
        - 【修正】调整 `zigzag_score` 的计算逻辑，直接使用 `ZIG_5_5.0_D` 本身，并调整 `normalize_to_bipolar` 的敏感度，避免极端值。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `level_score`, `trend_score`, `fusion_score`, `zigzag_score` 的归一化方式改为多时间维度自适应归一化。
        - 【新增】引入 `peak_exchange_purity_D` (主峰交换纯度) 作为判断筹码集中度的微观证据。
        """
        required_signals = [
            'short_term_concentration_90pct_D', 'long_term_concentration_90pct_D', 'winner_concentration_90pct_D',
            'peak_fusion_indicator_D',
            'ZIG_5_5.0_D', # 修正为 merge_results 后的列名
            'peak_exchange_purity_D' # 新增微观筹码博弈指标
        ] + [f'SLOPE_{p}_winner_concentration_90pct_D' for p in periods if f'SLOPE_{p}_winner_concentration_90pct_D' in df.columns]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码集中度探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        concentration_level_raw = (
            self._get_safe_series(df, 'short_term_concentration_90pct_D', 50.0, method_name="_diagnose_axiom_concentration") +
            self._get_safe_series(df, 'long_term_concentration_90pct_D', 50.0, method_name="_diagnose_axiom_concentration") +
            self._get_safe_series(df, 'winner_concentration_90pct_D', 50.0, method_name="_diagnose_axiom_concentration")
        ) / 3.0 - 50.0
        concentration_trend_raw = pd.Series(0.0, index=df.index)
        for p in periods:
            slope_col = f'SLOPE_{p}_winner_concentration_90pct_D'
            concentration_trend_raw += self._get_safe_series(df, slope_col, 0.0, method_name="_diagnose_axiom_concentration")
        concentration_trend_raw /= len(periods)
        peak_fusion_raw = self._get_safe_series(df, 'peak_fusion_indicator_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_concentration")
        zigzag_trend_raw = self._get_safe_series(df, 'ZIG_5_5.0_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_concentration")
        # 获取主峰交换纯度
        peak_exchange_purity_raw = self._get_safe_series(df, 'peak_exchange_purity_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_concentration")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        level_score = get_adaptive_mtf_normalized_bipolar_score(concentration_level_raw, df.index, tf_weights, sensitivity=10.0)
        trend_score = get_adaptive_mtf_normalized_bipolar_score(concentration_trend_raw, df.index, tf_weights, sensitivity=1.0)
        fusion_score = get_adaptive_mtf_normalized_bipolar_score(peak_fusion_raw, df.index, tf_weights, sensitivity=50.0)
        zigzag_score = get_adaptive_mtf_normalized_bipolar_score(zigzag_trend_raw, df.index, tf_weights, sensitivity=0.05)
        # 归一化主峰交换纯度，越高越好
        peak_exchange_purity_score = get_adaptive_mtf_normalized_bipolar_score(peak_exchange_purity_raw, df.index, tf_weights, sensitivity=0.5)
        # 融合所有分数，调整权重
        final_score = (level_score * 0.20 + trend_score * 0.30 + fusion_score * 0.20 + zigzag_score * 0.10 + peak_exchange_purity_score * 0.20).clip(-1, 1) # 调整权重并加入新信号
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [筹码集中度探针] @ {probe_date_for_loop.date()}:")
                print(f"       - level_score: {level_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_score: {trend_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - fusion_score: {fusion_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - zigzag_score: {zigzag_score.loc[probe_date_for_loop]:.4f}")
                # 打印主峰交换纯度分数
                print(f"       - peak_exchange_purity_score: {peak_exchange_purity_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_score: {final_score.loc[probe_date_for_loop]:.4f}")
        return final_score

    def _diagnose_axiom_cost_structure(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V3.0 · 结构动力学增强版】筹码公理二：诊断“成本结构”动态
        - 核心增强: 引入 `cost_gini_coefficient_D`, `structural_tension_index_D`, `structural_leverage_D` 三大新一代结构指标，
                      从成本分布公平性、结构内应力和潜在能量等物理学视角，深度量化成本结构的健康度与势能。
        - 核心修复: 遵循“先归一，后融合”原则。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        required_signals = [
            'winner_loser_momentum_D', 'cost_divergence_normalized_D', 'cost_structure_skewness_D',
            'pressure_validation_score_D', 'support_validation_score_D',
            'cost_gini_coefficient_D', 'structural_tension_index_D', 'structural_leverage_D' # 新增代码行：引入新指标
        ]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码成本结构探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        momentum_raw = self._get_safe_series(df, 'winner_loser_momentum_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        divergence_raw = self._get_safe_series(df, 'cost_divergence_normalized_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        skewness_raw = self._get_safe_series(df, 'cost_structure_skewness_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        pressure_validation_raw = self._get_safe_series(df, 'pressure_validation_score_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        support_validation_raw = self._get_safe_series(df, 'support_validation_score_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        # 新增代码行：获取新一代结构指标
        cost_gini_raw = self._get_safe_series(df, 'cost_gini_coefficient_D', pd.Series(0.5, index=df.index), method_name="_diagnose_axiom_cost_structure")
        structural_tension_raw = self._get_safe_series(df, 'structural_tension_index_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        structural_leverage_raw = self._get_safe_series(df, 'structural_leverage_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_cost_structure")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        momentum_score = get_adaptive_mtf_normalized_bipolar_score(momentum_raw, df.index, tf_weights, sensitivity=1.0)
        divergence_score = get_adaptive_mtf_normalized_bipolar_score(divergence_raw, df.index, tf_weights, sensitivity=1.0)
        skewness_score = get_adaptive_mtf_normalized_bipolar_score(skewness_raw, df.index, tf_weights, sensitivity=0.5)
        pressure_validation_score = get_adaptive_mtf_normalized_bipolar_score(pressure_validation_raw, df.index, tf_weights, sensitivity=0.5)
        support_validation_score = get_adaptive_mtf_normalized_bipolar_score(support_validation_raw, df.index, tf_weights, sensitivity=0.5)
        # 新增代码行：归一化新指标
        # 基尼系数越小越好，因此用 1-gini 进行归一化
        cost_gini_score = get_adaptive_mtf_normalized_bipolar_score(1 - cost_gini_raw, df.index, tf_weights, sensitivity=0.2)
        structural_tension_score = get_adaptive_mtf_normalized_bipolar_score(structural_tension_raw, df.index, tf_weights, sensitivity=0.5)
        structural_leverage_score = get_adaptive_mtf_normalized_bipolar_score(structural_leverage_raw, df.index, tf_weights, sensitivity=0.5)
        # 修改代码行：融合所有分数，调整权重以纳入新指标
        final_score = (
            momentum_score * 0.20 +
            skewness_score * 0.10 +
            support_validation_score * 0.10 +
            cost_gini_score * 0.20 +           # 新增：成本基尼系数
            structural_tension_score * 0.15 +  # 新增：结构张力
            structural_leverage_score * 0.15 - # 新增：结构杠杆
            divergence_score * 0.05 -
            pressure_validation_score * 0.05
        ).clip(-1, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [筹码成本结构探针] @ {probe_date_for_loop.date()}:")
                print(f"       - momentum_score: {momentum_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - divergence_score: {divergence_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - skewness_score: {skewness_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - pressure_validation_score: {pressure_validation_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - support_validation_score: {support_validation_score.loc[probe_date_for_loop]:.4f}")
                # 新增代码行：打印新指标分数
                print(f"       - cost_gini_score: {cost_gini_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - structural_tension_score: {structural_tension_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - structural_leverage_score: {structural_leverage_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_score: {final_score.loc[probe_date_for_loop]:.4f}")
        return final_score

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V3.0 · 结构动力学增强版】筹码公理三：诊断“持股心态”动态
        - 核心增强: 引入 `winner_stability_index_D` (获利盘稳定度)，更直接地量化获利盘的持股信心与决心，
                      为诊断持股心态提供关键的微观结构证据。
        - 核心修复: 遵循“先归一，后融合”原则。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        required_signals = [
            'winner_conviction_index_D', 'loser_pain_index_D', 'chip_fatigue_index_D',
            'locked_profit_rate_D', 'locked_loss_rate_D',
            'covert_accumulation_signal_D',
            'winner_stability_index_D' # 新增代码行：引入新指标
        ]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [持股心态探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        conviction_raw = self._get_safe_series(df, 'winner_conviction_index_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_holder_sentiment")
        pain_raw = self._get_safe_series(df, 'loser_pain_index_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_holder_sentiment")
        fatigue_raw = self._get_safe_series(df, 'chip_fatigue_index_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_holder_sentiment")
        locked_profit_raw = self._get_safe_series(df, 'locked_profit_rate_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        locked_loss_raw = self._get_safe_series(df, 'locked_loss_rate_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        covert_accumulation_raw = self._get_safe_series(df, 'covert_accumulation_signal_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        # 新增代码行：获取获利盘稳定度
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_holder_sentiment")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(conviction_raw, df_index, tf_weights, sensitivity=0.5)
        pain_score = get_adaptive_mtf_normalized_bipolar_score(pain_raw, df_index, tf_weights, sensitivity=5.0)
        fatigue_score = get_adaptive_mtf_normalized_bipolar_score(fatigue_raw, df_index, tf_weights, sensitivity=5.0)
        locked_profit_score = get_adaptive_mtf_normalized_bipolar_score(locked_profit_raw, df_index, tf_weights, sensitivity=20.0)
        locked_loss_score = get_adaptive_mtf_normalized_bipolar_score(locked_loss_raw, df_index, tf_weights, sensitivity=20.0)
        covert_accumulation_score = get_adaptive_mtf_normalized_bipolar_score(covert_accumulation_raw, df_index, tf_weights, sensitivity=0.5)
        # 新增代码行：归一化获利盘稳定度
        winner_stability_score = get_adaptive_mtf_normalized_bipolar_score(winner_stability_raw, df_index, tf_weights, sensitivity=0.5)
        # 修改代码行：调整权重并融合新指标
        final_score = (
            conviction_score * 0.25 +
            locked_profit_score * 0.10 +
            covert_accumulation_score * 0.10 +
            winner_stability_score * 0.25 - # 新增：获利盘稳定度
            pain_score * 0.15 -
            fatigue_score * 0.1 -
            locked_loss_score * 0.05
        ).clip(-1, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [持股心态探针] @ {probe_date_for_loop.date()}:")
                print(f"       - conviction_score: {conviction_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - pain_score: {pain_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - fatigue_score: {fatigue_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - locked_profit_score: {locked_profit_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - locked_loss_score: {locked_loss_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - covert_accumulation_score: {covert_accumulation_score.loc[probe_date_for_loop]:.4f}")
                # 新增代码行：打印新指标分数
                print(f"       - winner_stability_score: {winner_stability_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_score: {final_score.loc[probe_date_for_loop]:.4f}")
        return final_score

    def _diagnose_axiom_peak_integrity(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V3.0 · 结构动力学增强版】筹码公理四：诊断“筹码峰形态”
        - 核心增强: 引入 `primary_peak_kurtosis_D` (主峰峰态系数) 和 `price_volume_entropy_D` (价格成交量熵)，
                      从统计学和信息论角度，更精确地刻画筹码峰的形态（尖锐/平缓）与内部结构的有序性。
        - 核心修复: 遵循“先归一，后融合”原则。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        required_signals = [
            'dominant_peak_cost_D', 'dominant_peak_solidity_D', 'price_volume_entropy_D',
            'primary_peak_kurtosis_D' # 新增代码行：引入新指标
        ]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        price_vs_peak_raw = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_peak_integrity") - self._get_safe_series(df, 'dominant_peak_cost_D', self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_peak_integrity"), method_name="_diagnose_axiom_peak_integrity")
        peak_solidity_raw = self._get_safe_series(df, 'dominant_peak_solidity_D', pd.Series(0.5, index=df.index), method_name="_diagnose_axiom_peak_integrity")
        price_volume_entropy_raw = self._get_safe_series(df, 'price_volume_entropy_D', pd.Series(0.5, index=df.index), method_name="_diagnose_axiom_peak_integrity")
        # 新增代码行：获取主峰峰态系数
        primary_peak_kurtosis_raw = self._get_safe_series(df, 'primary_peak_kurtosis_D', pd.Series(3.0, index=df.index), method_name="_diagnose_axiom_peak_integrity")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_vs_peak_score = get_adaptive_mtf_normalized_bipolar_score(price_vs_peak_raw, df.index, tf_weights, sensitivity=1.2)
        peak_solidity_score = get_adaptive_mtf_normalized_score(peak_solidity_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 熵值越低（结构越有序）越好，因此乘以-1
        price_volume_entropy_score = get_adaptive_mtf_normalized_bipolar_score(price_volume_entropy_raw * -1, df.index, tf_weights, sensitivity=0.5)
        # 新增代码行：归一化主峰峰态系数，越高越好
        primary_peak_kurtosis_score = get_adaptive_mtf_normalized_bipolar_score(primary_peak_kurtosis_raw, df.index, tf_weights, sensitivity=2.0)
        # 修改代码行：融合所有分数，调整权重
        final_score = (
            price_vs_peak_score * peak_solidity_score * 0.5 +
            price_volume_entropy_score * 0.25 +
            primary_peak_kurtosis_score * 0.25
        ).clip(-1, 1)
        return final_score

    def _diagnose_axiom_trend_momentum(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.4 · OCH数据层获取与多时间维度归一化版】筹码公理六：诊断“筹码趋势动量”
        - 核心逻辑: 衡量整体筹码健康度的变化速度和方向。
          - 整体筹码健康度 (OCH): 从 df 中直接获取已在数据层计算的 OCH_D。
          - OCH的短期 (5日) 和中期 (21日) 斜率，反映筹码健康度的动量和趋势。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `overall_chip_health` 的归一化方式改为多时间维度自适应归一化。
        - 【新增】引入 `structural_resilience_index_D` (结构韧性指数) 作为判断筹码趋势动量的微观证据。
        """
        df_index = df.index
        required_signals = ['OCH_D', 'SLOPE_5_OCH_D', 'SLOPE_21_OCH_D', 'structural_resilience_index_D'] # 新增微观筹码博弈指标
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码趋势动量探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df_index)
        overall_chip_health_raw = self._get_safe_series(df, 'OCH_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_momentum")
        slope_5_och = self._get_safe_series(df, 'SLOPE_5_OCH_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_momentum")
        slope_21_och = self._get_safe_series(df, 'SLOPE_21_OCH_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_momentum")
        # 获取结构韧性指数
        structural_resilience_raw = self._get_safe_series(df, 'structural_resilience_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_trend_momentum")
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        overall_chip_health_score = get_adaptive_mtf_normalized_bipolar_score(overall_chip_health_raw, df_index, tf_weights, sensitivity=0.05)
        slope_5_och_score = get_adaptive_mtf_normalized_bipolar_score(slope_5_och, df_index, tf_weights, sensitivity=0.005)
        slope_21_och_score = get_adaptive_mtf_normalized_bipolar_score(slope_21_och, df_index, tf_weights, sensitivity=0.002)
        # 归一化结构韧性指数，越高越好
        structural_resilience_score = get_adaptive_mtf_normalized_bipolar_score(structural_resilience_raw, df_index, tf_weights, sensitivity=0.5)
        # 融合OCH的当前值和其动量，并加入结构韧性指数
        chip_trend_momentum_score = (
            overall_chip_health_score * 0.3 +
            slope_5_och_score * 0.3 +
            slope_21_och_score * 0.2 +
            structural_resilience_score * 0.2 # 新增结构韧性指数
        ).clip(-1, 1) # 调整权重并加入新信号
        # --- Debugging output for probe date ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [筹码趋势动量探针] @ {probe_date_for_loop.date()}:")
                print(f"       - overall_chip_health: {overall_chip_health_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - slope_5_och: {slope_5_och.loc[probe_date_for_loop]:.6f}")
                print(f"       - slope_21_och: {slope_21_och.loc[probe_date_for_loop]:.6f}")
                print(f"       - structural_resilience_raw: {structural_resilience_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - overall_chip_health_score: {overall_chip_health_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - slope_5_och_score: {slope_5_och_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - slope_21_och_score: {slope_21_och_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - structural_resilience_score: {structural_resilience_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_trend_momentum_score: {chip_trend_momentum_score.loc[probe_date_for_loop]:.4f}")
        return chip_trend_momentum_score.astype(np.float32)

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.1】筹码公理五：诊断筹码“背离”动态
        - 核心逻辑: 诊断价格行为与筹码集中度之间的背离。
          - 看涨背离：价格下跌但筹码集中度上升（主力吸筹）。
          - 看跌背离：价格上涨但筹码集中度下降（主力派发）。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `price_trend` 和 `concentration_trend` 的归一化方式改为多时间维度自适应归一化。
        """
        required_signals = ['pct_change_D', 'SLOPE_5_short_term_concentration_90pct_D']
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_divergence"), df.index, tf_weights)
        concentration_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'SLOPE_5_short_term_concentration_90pct_D', pd.Series(0.0, index=df.index), method_name="_diagnose_axiom_divergence"), df.index, tf_weights)
        divergence_score = (concentration_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)



