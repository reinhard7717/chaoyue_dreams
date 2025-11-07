# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following import utils
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.5 · 真理探针植入版】筹码情报分析总指挥
        - 核心升级: 植入三层“真理探针”，深度追踪数据在引擎内部的流转，以诊断信号归零的根本原因。
        """
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 筹码情报引擎在配置中被禁用，跳过分析。")
            return {}
        print("-> [指挥覆盖探针] 筹码情报引擎已启用，开始分析...")
        all_states = {}
        periods = get_param_value(p_conf.get('periods'), [5, 13, 21, 55])
        # --- 真理探针 V2.5 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            if df.index.tz:
                probe_date = probe_date_naive.tz_localize(df.index.tz)
            else:
                probe_date = probe_date_naive
            print(f"    -> [CHIP引擎探针] @ {probe_date.date()} 启动...")
            
            # --- 探针一: 输入层检查 ---
            print("      -> [探针 1/3] 检查所有依赖的底层信号当日值:")
            required_signals_for_probe = [
                'short_term_concentration_90pct_D', 'long_term_concentration_90pct_D', 'winner_concentration_90pct_D',
                'winner_loser_momentum_D', 'cost_divergence_normalized_D',
                'winner_conviction_index_D', 'loser_pain_index_D', 'chip_fatigue_index_D',
                'dominant_peak_cost_D', 'dominant_peak_solidity_D', 'close_D'
            ] + [f'SLOPE_{p}_winner_concentration_90pct_D' for p in periods]
            
            for signal in required_signals_for_probe:
                if signal in df.columns and probe_date in df.index:
                    val = df.loc[probe_date, signal]
                    print(f"         - 信号: {signal:<45s} | 当日值: {val:.4f}")
                else:
                    print(f"         - 信号: {signal:<45s} | 状态: 未找到")

        axiom_concentration = self._diagnose_axiom_concentration(df, periods)
        axiom_cost_structure = self._diagnose_axiom_cost_structure(df, periods)
        axiom_holder_sentiment = self._diagnose_axiom_holder_sentiment(df, periods)
        axiom_peak_integrity = self._diagnose_axiom_peak_integrity(df, periods)
        all_states['SCORE_CHIP_AXIOM_CONCENTRATION'] = axiom_concentration
        all_states['SCORE_CHIP_AXIOM_COST_STRUCTURE'] = axiom_cost_structure
        all_states['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = axiom_holder_sentiment
        all_states['SCORE_CHIP_AXIOM_PEAK_INTEGRITY'] = axiom_peak_integrity
        # --- 探针二: 公理层检查 ---
        if probe_date:
            print("      -> [探针 2/3] 检查各公理诊断引擎输出的当日分数:")
            print(f"         - 公理: 聚散 (Concentration) | 当日分数: {axiom_concentration.loc[probe_date]:.4f}")
            print(f"         - 公理: 成本 (Cost)         | 当日分数: {axiom_cost_structure.loc[probe_date]:.4f}")
            print(f"         - 公理: 心态 (Sentiment)    | 当日分数: {axiom_holder_sentiment.loc[probe_date]:.4f}")
            print(f"         - 公理: 形态 (Integrity)    | 当日分数: {axiom_peak_integrity.loc[probe_date]:.4f}")

        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'concentration': 0.3, 'cost_structure': 0.3, 'holder_sentiment': 0.2, 'peak_integrity': 0.2
        })
        bipolar_health = (
            axiom_concentration * axiom_weights['concentration'] +
            axiom_cost_structure * axiom_weights['cost_structure'] +
            axiom_holder_sentiment * axiom_weights['holder_sentiment'] +
            axiom_peak_integrity * axiom_weights['peak_integrity']
        ).clip(-1, 1)
        # --- 探针三: 融合层检查 ---
        if probe_date:
            print("      -> [探针 3/3] 检查最终融合前的双极性健康分:")
            print(f"         - 融合健康分 (bipolar_health) | 当日值: {bipolar_health.loc[probe_date]:.4f}")

        bullish_resonance, bearish_resonance = utils.bipolar_to_exclusive_unipolar(bipolar_health)
        all_states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance
        all_states['SCORE_CHIP_BEARISH_RESONANCE'] = bearish_resonance
        return all_states

    def _synthesize_ultimate_signals(self, df: pd.DataFrame, concentration: pd.Series, cost_structure: pd.Series, holder_sentiment: pd.Series, peak_integrity: pd.Series) -> Dict[str, pd.Series]:
        """
        【V8.2 · 逻辑修复版】终极信号合成器
        - 核心修复: 修正了终极信号合成逻辑。不再对双极性公理分进行错误的 clip 操作，
                      而是先加权融合成一个总体的双极性健康分，然后使用标准工具分裂为
                      正确的、互斥的看涨/看跌共振分。同时简化了后续信号的生成逻辑。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {'concentration': 0.3, 'cost_structure': 0.3, 'holder_sentiment': 0.2, 'peak_integrity': 0.2})
        # 步骤一：计算双极性“全息筹码健康分”
        bipolar_health = (
            concentration * axiom_weights['concentration'] +
            cost_structure * axiom_weights['cost_structure'] +
            holder_sentiment * axiom_weights['holder_sentiment'] +
            peak_integrity * axiom_weights['peak_integrity']
        ).clip(-1, 1)
        # 步骤二：使用标准工具将双极性健康分分裂为互斥的看涨和看跌共振分
        from strategies.trend_following.utils import bipolar_to_exclusive_unipolar
        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_health)
        states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance.fillna(0).astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE'] = bearish_resonance.fillna(0).astype(np.float32)
        # 步骤三：简化其他终极信号的生成
        bullish_momentum = bullish_resonance.diff().fillna(0)
        bearish_momentum = bearish_resonance.diff().fillna(0)
        norm_bullish_momentum = normalize_score(bullish_momentum, df.index, 21, ascending=True)
        norm_bearish_momentum = normalize_score(bearish_momentum, df.index, 21, ascending=True)
        states['SCORE_CHIP_BULLISH_ACCELERATION'] = norm_bullish_momentum.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BEARISH_ACCELERATION'] = norm_bearish_momentum.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = (1.0 - norm_bearish_momentum).clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_TOP_REVERSAL'] = (1.0 - norm_bullish_momentum).clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_TACTICAL_PULLBACK'] = (bullish_resonance * states['SCORE_CHIP_TOP_REVERSAL']).clip(0, 1).astype(np.float32)
        return states

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
        【V2.4 · 数学重构版】筹码公理一：诊断筹码“聚散”动态
        - 核心修复: 遵循“先归一，后融合”原则。不再对原始值进行武断缩放，而是先将“集中度水平”和“集中趋势”
                      分别归一化为[-1, 1]的双极性分数，然后再进行加权融合，确保模型在不同市场环境下的健壮性。
        """
        required_signals = [
            'short_term_concentration_90pct_D', 'long_term_concentration_90pct_D', 'winner_concentration_90pct_D'
        ] + [f'SLOPE_{p}_winner_concentration_90pct_D' for p in periods if f'SLOPE_{p}_winner_concentration_90pct_D' in df.columns]
        self._run_integrity_probe(df, required_signals, "聚散")
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        # 证据1: 集中度水平。将原始的0-100值，通过减去50构造一个原始双极性序列。
        concentration_level_raw = (
            df.get('short_term_concentration_90pct_D', 50.0) +
            df.get('long_term_concentration_90pct_D', 50.0) +
            df.get('winner_concentration_90pct_D', 50.0)
        ) / 3.0 - 50.0
        # 证据2: 集中度趋势。直接使用斜率作为原始序列。
        concentration_trend_raw = pd.Series(0.0, index=df.index)
        for p in periods:
            slope_col = f'SLOPE_{p}_winner_concentration_90pct_D'
            if slope_col in df.columns:
                concentration_trend_raw += df.get(slope_col, 0.0)
        concentration_trend_raw /= len(periods)
        # 分别对“水平”和“趋势”进行多时间框架自适应双极性归一化
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        level_score = utils.get_adaptive_mtf_normalized_bipolar_score(concentration_level_raw, df.index, tf_weights, sensitivity=10.0)
        trend_score = utils.get_adaptive_mtf_normalized_bipolar_score(concentration_trend_raw, df.index, tf_weights, sensitivity=1.0)
        # 融合归一化后的分数
        final_score = (level_score * 0.3 + trend_score * 0.7).clip(-1, 1)
        return final_score

    def _diagnose_axiom_cost_structure(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.4 · 数学重构版】筹码公理二：诊断“成本结构”动态
        - 核心修复: 遵循“先归一，后融合”原则。将尺度差异巨大的 'winner_loser_momentum_D' 和 'cost_divergence_normalized_D'
                      分别进行自适应双极性归一化，然后再进行相减，避免了信号被单一指标主导的问题。
        """
        required_signals = ['winner_loser_momentum_D', 'cost_divergence_normalized_D']
        self._run_integrity_probe(df, required_signals, "成本")
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        # 证据1: 获利盘与亏损盘的动量
        momentum_raw = df.get('winner_loser_momentum_D', pd.Series(0.0, index=df.index))
        # 证据2: 成本发散度
        divergence_raw = df.get('cost_divergence_normalized_D', pd.Series(0.0, index=df.index))
        # 分别对两个证据进行多时间框架自适应双极性归一化
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        momentum_score = utils.get_adaptive_mtf_normalized_bipolar_score(momentum_raw, df.index, tf_weights, sensitivity=1.0)
        divergence_score = utils.get_adaptive_mtf_normalized_bipolar_score(divergence_raw, df.index, tf_weights, sensitivity=1.0)
        # 融合归一化后的分数：我们期望动量为正（获利盘强），发散为负（成本集中），所以是 momentum_score - divergence_score
        final_score = (momentum_score - divergence_score).clip(-1, 1)
        return final_score

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.4 · 数学重构版】筹码公理三：诊断“持股心态”动态
        - 核心修复: 遵循“先归一，后融合”原则。将三个尺度不同的心态指标分别归一化，再进行融合。
                      'winner_conviction' 为正向贡献，'loser_pain' 和 'chip_fatigue' 为负向贡献。
        """
        required_signals = ['winner_conviction_index_D', 'loser_pain_index_D', 'chip_fatigue_index_D']
        self._run_integrity_probe(df, required_signals, "心态")
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        # 证据1: 赢家信念 (正向)
        conviction_raw = df.get('winner_conviction_index_D', pd.Series(0.0, index=df.index))
        # 证据2: 输家痛苦 (负向)
        pain_raw = df.get('loser_pain_index_D', pd.Series(0.0, index=df.index))
        # 证据3: 筹码疲劳 (负向)
        fatigue_raw = df.get('chip_fatigue_index_D', pd.Series(0.0, index=df.index))
        # 分别归一化
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        conviction_score = utils.get_adaptive_mtf_normalized_bipolar_score(conviction_raw, df.index, tf_weights, sensitivity=1.0)
        pain_score = utils.get_adaptive_mtf_normalized_bipolar_score(pain_raw, df.index, tf_weights, sensitivity=1.0)
        fatigue_score = utils.get_adaptive_mtf_normalized_bipolar_score(fatigue_raw, df.index, tf_weights, sensitivity=1.0)
        # 融合归一化后的分数
        final_score = (conviction_score - pain_score - fatigue_score).clip(-1, 1)
        return final_score

    def _diagnose_axiom_peak_integrity(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.4 · 数学重构版】筹码公理四：诊断“筹码峰形态”
        - 核心修复: 遵循“先归一，后融合”原则。将“价格与筹码峰的距离”和“筹码峰的坚实度”分别归一化，
                      然后相乘。相乘的逻辑是合理的，代表“坚实度”对“价格偏离”信号的确认或证伪。
        """
        required_signals = ['dominant_peak_cost_D', 'dominant_peak_solidity_D']
        self._run_integrity_probe(df, required_signals, "形态")
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            return pd.Series(0.0, index=df.index)
        # 证据1: 价格与主筹码峰的偏离度
        price_vs_peak_raw = df['close_D'] - df.get('dominant_peak_cost_D', df['close_D'])
        # 证据2: 主筹码峰的坚实度
        peak_solidity_raw = df.get('dominant_peak_solidity_D', pd.Series(0.5, index=df.index))
        # 分别归一化
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_vs_peak_score = utils.get_adaptive_mtf_normalized_bipolar_score(price_vs_peak_raw, df.index, tf_weights, sensitivity=1.2)
        # 坚实度是正向指标，值越大越好，所以使用单极性归一化
        peak_solidity_score = utils.get_adaptive_mtf_normalized_score(peak_solidity_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 融合：价格偏离度 * 坚实度确认。坚实度越高，价格偏离的信号越可信。
        final_score = (price_vs_peak_score * peak_solidity_score).clip(-1, 1)
        return final_score


