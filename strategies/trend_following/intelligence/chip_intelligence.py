import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following import utils
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.3 · 纯粹原子版】筹码情报总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出筹码领域的原子公理信号、筹码背离信号和超级原子信号。
        - 移除信号: SCORE_CHIP_BULLISH_RESONANCE, SCORE_CHIP_BEARISH_RESONANCE, BIPOLAR_CHIP_DOMAIN_HEALTH, SCORE_CHIP_BOTTOM_REVERSAL, SCORE_CHIP_TOP_REVERSAL。
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
        # 引入筹码层面的看涨/看跌背离信号 (保持不变)
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(divergence_scores)
        all_chip_states['SCORE_CHIP_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_chip_states['SCORE_CHIP_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        # 步骤二: 工程化超级原子信号
        # 信号1: 筹码干净度 (SCORE_CHIP_CLEANLINESS)
        chip_fault = df.get('chip_fault_blockage_ratio_D', 0.5)
        profit_pressure = df.get('imminent_profit_taking_supply_D', 0.5)
        cleanliness_score = ((1 - chip_fault) * (1 - profit_pressure)).pow(0.5).fillna(0.5)
        all_chip_states['SCORE_CHIP_CLEANLINESS'] = cleanliness_score.astype(np.float32)
        # 信号2: 筹码锁定度 (SCORE_CHIP_LOCKDOWN_DEGREE)
        locked_profit = df.get('locked_profit_rate_D', 0.0)
        locked_loss = df.get('locked_loss_rate_D', 0.0)
        lockdown_degree = (locked_profit + locked_loss).clip(0, 1).fillna(0.0)
        all_chip_states['SCORE_CHIP_LOCKDOWN_DEGREE'] = lockdown_degree.astype(np.float32)
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
        【V2.8 · 数学重构与峰融合增强及列名引用修复版】筹码公理一：诊断筹码“聚散”动态
        - 核心修复: 遵循“先归一，后融合”原则。不再对原始值进行武断缩放，而是先将“集中度水平”和“集中趋势”
                      分别归一化为[-1, 1]的双极性分数，然后再进行加权融合，确保模型在不同市场环境下的健壮性。
        - 引入 `peak_fusion_indicator` (筹码峰融合指标) 作为判断筹码集中度的重要证据。
        - 【新增】引入 ZIGZAG 趋势作为辅助证据，增强对集中度有效性的判断。
        - 【修复】修正了引用 ZIGZAG 列名时，确保其与 `IndicatorService` 中 `merge_results` 方法添加后缀后的列名一致。
        - 【修正】调整 `zigzag_score` 的计算逻辑，直接使用 `ZIG_5_5.0_D` 本身，并调整 `normalize_to_bipolar` 的敏感度，避免极端值。
        """
        required_signals = [
            'short_term_concentration_90pct_D', 'long_term_concentration_90pct_D', 'winner_concentration_90pct_D',
            'peak_fusion_indicator_D',
            'ZIG_5_5.0_D' # 修正为 merge_results 后的列名
        ] + [f'SLOPE_{p}_winner_concentration_90pct_D' for p in periods if f'SLOPE_{p}_winner_concentration_90pct_D' in df.columns]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码集中度探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        concentration_level_raw = (
            df.get('short_term_concentration_90pct_D', 50.0) +
            df.get('long_term_concentration_90pct_D', 50.0) +
            df.get('winner_concentration_90pct_D', 50.0)
        ) / 3.0 - 50.0
        concentration_trend_raw = pd.Series(0.0, index=df.index)
        for p in periods:
            slope_col = f'SLOPE_{p}_winner_concentration_90pct_D'
            if slope_col in df.columns:
                concentration_trend_raw += df.get(slope_col, 0.0)
        concentration_trend_raw /= len(periods)
        peak_fusion_raw = df.get('peak_fusion_indicator_D', pd.Series(0.0, index=df.index))
        # 新增 ZIGZAG 趋势作为辅助证据
        # 修正：直接使用 ZIG_5_5.0_D 本身，它代表了价格的ZIGZAG趋势，向上为正，向下为负
        zigzag_trend_raw = df.get('ZIG_5_5.0_D', pd.Series(0.0, index=df.index))
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        level_score = utils.get_adaptive_mtf_normalized_bipolar_score(concentration_level_raw, df.index, tf_weights, sensitivity=10.0)
        trend_score = utils.get_adaptive_mtf_normalized_bipolar_score(concentration_trend_raw, df.index, tf_weights, sensitivity=1.0)
        fusion_score = utils.get_adaptive_mtf_normalized_bipolar_score(peak_fusion_raw, df.index, tf_weights, sensitivity=50.0)
        # 归一化 ZIGZAG 趋势，向上为正，向下为负
        # 调整敏感度，避免微小波动被过度放大到极端值
        zigzag_score = utils.get_adaptive_mtf_normalized_bipolar_score(zigzag_trend_raw, df.index, tf_weights, sensitivity=0.05) # 调整敏感度
        # 融合 fusion_score 和 zigzag_score
        final_score = (level_score * 0.25 + trend_score * 0.35 + fusion_score * 0.25 + zigzag_score * 0.15).clip(-1, 1) # 调整权重
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
                print(f"       - final_score: {final_score.loc[probe_date_for_loop]:.4f}")
        return final_score

    def _diagnose_axiom_cost_structure(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.7 · 数学重构与偏度增强版】筹码公理二：诊断“成本结构”动态
        - 核心修复: 遵循“先归一，后融合”原则。将尺度差异巨大的 'winner_loser_momentum_D' 和 'cost_divergence_normalized_D'
                      分别进行自适应双极性归一化，然后再进行相减，避免了信号被单一指标主导的问题。
        - 引入 `cost_structure_skewness` (成本结构偏度) 作为判断成本结构健康度的重要证据。
        - 【修正】调整 `skewness_score` 的 `normalize_to_bipolar` 敏感度，避免极端值。
        """
        required_signals = ['winner_loser_momentum_D', 'cost_divergence_normalized_D', 'cost_structure_skewness_D'] # 增加 cost_structure_skewness_D
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [筹码成本结构探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        momentum_raw = df.get('winner_loser_momentum_D', pd.Series(0.0, index=df.index))
        divergence_raw = df.get('cost_divergence_normalized_D', pd.Series(0.0, index=df.index))
        skewness_raw = df.get('cost_structure_skewness_D', pd.Series(0.0, index=df.index)) # 新增行
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        momentum_score = utils.get_adaptive_mtf_normalized_bipolar_score(momentum_raw, df.index, tf_weights, sensitivity=1.0)
        divergence_score = utils.get_adaptive_mtf_normalized_bipolar_score(divergence_raw, df.index, tf_weights, sensitivity=1.0)
        # [代码修改开始]
        # 归一化偏度，正偏度为正分。调整敏感度，避免极端值。
        skewness_score = utils.get_adaptive_mtf_normalized_bipolar_score(skewness_raw, df.index, tf_weights, sensitivity=0.5) # 调整敏感度从0.1到0.5
        # [代码修改结束]
        # 融合 skewness_score
        # 融合逻辑：动量（正向）- 发散（负向）+ 偏度（正向）
        final_score = (momentum_score * 0.4 + skewness_score * 0.3 - divergence_score * 0.3).clip(-1, 1) # 调整权重
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
                print(f"       - final_score: {final_score.loc[probe_date_for_loop]:.4f}")
        return final_score

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.7 · 数学重构与敏感度优化版】筹码公理三：诊断“持股心态”动态
        - 核心修复: 遵循“先归一，后融合”原则。将三个尺度不同的心态指标分别归一化，再进行融合。
                      'winner_conviction' 为正向贡献，'loser_pain' 和 'chip_fatigue' 为负向贡献。
        - 核心优化: 调整 `pain_score` 和 `fatigue_score` 的 `sensitivity`，避免在积极行情下过度惩罚。
        - 引入 `locked_profit_rate` 和 `locked_loss_rate` 作为判断持股心态的重要证据。
        - 【修正】调整 `conviction_score` 的 `normalize_to_bipolar` 敏感度，确保在涨停日能正确反映积极心态。
        """
        df_index = df.index
        required_signals = [
            'winner_conviction_index_D', 'loser_pain_index_D', 'chip_fatigue_index_D',
            'locked_profit_rate_D', 'locked_loss_rate_D' # 增加 locked_profit_rate_D, locked_loss_rate_D
        ]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [持股心态探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df.index)
        conviction_raw = df.get('winner_conviction_index_D', pd.Series(0.0, index=df.index))
        pain_raw = df.get('loser_pain_index_D', pd.Series(0.0, index=df.index))
        fatigue_raw = df.get('chip_fatigue_index_D', pd.Series(0.0, index=df.index))
        locked_profit_raw = df.get('locked_profit_rate_D', pd.Series(0.0, index=df.index)) # 新增行
        locked_loss_raw = df.get('locked_loss_rate_D', pd.Series(0.0, index=df.index)) # 新增行
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 赢家信念：越高越好，正向贡献。调整敏感度，确保在涨停日能正确反映积极心态。
        conviction_score = utils.get_adaptive_mtf_normalized_bipolar_score(conviction_raw, df_index, tf_weights, sensitivity=0.5) # 调整敏感度
        # 输家痛苦：越高越差，负向贡献。降低敏感度，避免微小痛苦被放大。
        pain_score = utils.get_adaptive_mtf_normalized_bipolar_score(pain_raw, df_index, tf_weights, sensitivity=5.0)
        # 筹码疲劳：越高越差，负向贡献。降低敏感度。
        fatigue_score = utils.get_adaptive_mtf_normalized_bipolar_score(fatigue_raw, df_index, tf_weights, sensitivity=5.0)
        # 锁定利润盘：越高越好，正向贡献
        locked_profit_score = utils.get_adaptive_mtf_normalized_bipolar_score(locked_profit_raw, df_index, tf_weights, sensitivity=20.0) # 新增行
        # 锁定亏损盘：越高越差，负向贡献
        locked_loss_score = utils.get_adaptive_mtf_normalized_bipolar_score(locked_loss_raw, df_index, tf_weights, sensitivity=20.0) # 新增行
        # 调整权重并融合 locked_profit_score 和 locked_loss_score
        final_score = (
            conviction_score * 0.4 +
            locked_profit_score * 0.2 -
            pain_score * 0.15 -
            fatigue_score * 0.15 -
            locked_loss_score * 0.1
        ).clip(-1, 1) # 调整权重
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [持股心态探针] @ {probe_date_for_loop.date()}:")
                print(f"       - conviction_raw: {conviction_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - pain_raw: {pain_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - fatigue_raw: {fatigue_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - locked_profit_raw: {locked_profit_raw.loc[probe_date_for_loop]:.4f}") # 新增行
                print(f"       - locked_loss_raw: {locked_loss_raw.loc[probe_date_for_loop]:.4f}") # 新增行
                print(f"       - conviction_score: {conviction_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - pain_score: {pain_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - fatigue_score: {fatigue_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - locked_profit_score: {locked_profit_score.loc[probe_date_for_loop]:.4f}") # 新增行
                print(f"       - locked_loss_score: {locked_loss_score.loc[probe_date_for_loop]:.4f}") # 新增行
                print(f"       - final_score: {final_score.loc[probe_date_for_loop]:.4f}")
        return final_score

    def _diagnose_axiom_peak_integrity(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.4 · 数学重构版】筹码公理四：诊断“筹码峰形态”
        - 核心修复: 遵循“先归一，后融合”原则。将“价格与筹码峰的距离”和“筹码峰的坚实度”分别归一化，
                      然后相乘。相乘的逻辑是合理的，代表“坚实度”对“价格偏离”信号的确认或证伪。
        """
        required_signals = ['dominant_peak_cost_D', 'dominant_peak_solidity_D']
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        price_vs_peak_raw = df['close_D'] - df.get('dominant_peak_cost_D', df['close_D'])
        peak_solidity_raw = df.get('dominant_peak_solidity_D', pd.Series(0.5, index=df.index))
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_vs_peak_score = utils.get_adaptive_mtf_normalized_bipolar_score(price_vs_peak_raw, df.index, tf_weights, sensitivity=1.2)
        peak_solidity_score = utils.get_adaptive_mtf_normalized_score(peak_solidity_raw, df.index, ascending=True, tf_weights=tf_weights)
        final_score = (price_vs_peak_score * peak_solidity_score).clip(-1, 1)
        return final_score

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.0】筹码公理五：诊断筹码“背离”动态
        - 核心逻辑: 诊断价格行为与筹码集中度之间的背离。
          - 看涨背离：价格下跌但筹码集中度上升（主力吸筹）。
          - 看跌背离：价格上涨但筹码集中度下降（主力派发）。
        """
        required_signals = ['pct_change_D', 'SLOPE_5_short_term_concentration_90pct_D']
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        price_trend = normalize_to_bipolar(df['pct_change_D'], df.index, window=55)
        concentration_trend = normalize_to_bipolar(df.get('SLOPE_5_short_term_concentration_90pct_D', pd.Series(0.0, index=df.index)), df.index, window=55)
        divergence_score = (concentration_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

