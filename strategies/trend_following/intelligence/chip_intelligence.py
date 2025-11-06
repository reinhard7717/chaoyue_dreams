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

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.0 · 超级原子信号版】筹码情报总指挥
        - 核心升级: 新增“超级原子信号工程化”步骤，负责精炼和铸造更具实战意义的复合信号，
                      以支撑认知层更复杂的战术剧本推演。
        - 新增信号:
          - SCORE_CHIP_CLEANLINESS: 筹码干净度，衡量上方套牢盘和短期获利盘的综合压力。
          - SCORE_CHIP_LOCKDOWN_DEGREE: 筹码锁定度，衡量市场中总的不愿交易的筹码比例。
        """
        print("启动【V9.0 · 超级原子信号版】筹码情报分析...")
        all_chip_states = {}
        periods = [5, 13, 21, 55] # 筹码分析更侧重中长周期
        # 步骤一: 诊断四大公理，生成纯粹的筹码原子信号
        print("工序一: 正在诊断四大筹码公理...")
        concentration_scores = self._diagnose_axiom_concentration(df, periods)
        cost_structure_scores = self._diagnose_axiom_cost_structure(df, periods)
        holder_sentiment_scores = self._diagnose_axiom_holder_sentiment(df, periods)
        peak_integrity_scores = self._diagnose_axiom_peak_integrity(df, periods)
        # 将公理的诊断结果存入原子状态，供上层追溯
        all_chip_states['SCORE_CHIP_AXIOM_CONCENTRATION'] = concentration_scores
        all_chip_states['SCORE_CHIP_AXIOM_COST_STRUCTURE'] = cost_structure_scores
        all_chip_states['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT'] = holder_sentiment_scores
        all_chip_states['SCORE_CHIP_AXIOM_PEAK_INTEGRITY'] = peak_integrity_scores
        # 步骤二: 合成筹码领域的终极信号
        print("工序二: 正在合成终极筹码信号...")
        ultimate_signals = self._synthesize_ultimate_signals(
            df,
            concentration_scores,
            cost_structure_scores,
            holder_sentiment_scores,
            peak_integrity_scores
        )
        all_chip_states.update(ultimate_signals)
        # 步骤三 (新增): 工程化超级原子信号
        print("工序三: 正在工程化超级原子信号...")
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
        print(f"【V9.0 · 超级原子信号版】筹码情报分析完成，新增2个超级原子信号。")
        return all_chip_states

    def _synthesize_ultimate_signals(self, df: pd.DataFrame, concentration: pd.Series, cost_structure: pd.Series, holder_sentiment: pd.Series, peak_integrity: pd.Series) -> Dict[str, pd.Series]:
        """
        【V8.1 · 单一信号输入版】终极信号合成器
        - 核心修改: 调整了方法签名，现在接收的是单一的、已融合的公理分数Series，而不是字典。
        """
        states = {}
        # 移除了对 periods 的依赖，因为输入已经是融合后的Series
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        # tf_weights 在公理层融合时使用，这里不再需要
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {'concentration': 0.3, 'cost_structure': 0.3, 'holder_sentiment': 0.2, 'peak_integrity': 0.2})
        
        # 步骤一：计算双极性“全息筹码健康分”
        # 直接使用传入的已融合的公理分数Series
        bipolar_health = (
            concentration * axiom_weights['concentration'] +
            cost_structure * axiom_weights['cost_structure'] +
            holder_sentiment * axiom_weights['holder_sentiment'] +
            peak_integrity * axiom_weights['peak_integrity']
        ).clip(-1, 1)
        
        # 步骤二：分离为纯粹的看涨/看跌健康分
        bullish_resonance = bipolar_health.clip(0, 1)
        bearish_resonance = (bipolar_health.clip(-1, 0) * -1)
        
        states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE'] = bearish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        
        # 步骤三：计算动态信号 (一阶导数)
        bullish_momentum = bullish_resonance.diff().fillna(0)
        bearish_momentum = bearish_resonance.diff().fillna(0)
        
        # 归一化动态信号
        norm_bullish_momentum = normalize_score(bullish_momentum, df.index, 21, ascending=True)
        norm_bearish_momentum = normalize_score(bearish_momentum, df.index, 21, ascending=True)
        
        # 步骤四：赋值给命名准确的终极信号
        states['SCORE_CHIP_BULLISH_ACCELERATION'] = norm_bullish_momentum.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BEARISH_ACCELERATION'] = norm_bearish_momentum.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = (1.0 - norm_bearish_momentum).clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_TOP_REVERSAL'] = (1.0 - norm_bullish_momentum).clip(0, 1).astype(np.float32)
        
        # 步骤五：重铸战术回调信号
        states['SCORE_CHIP_TACTICAL_PULLBACK'] = (bullish_resonance * states['SCORE_CHIP_TOP_REVERSAL']).clip(0, 1).astype(np.float32)
        return states

    def _diagnose_axiom_concentration(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.1 · 融合输出版】筹码公理一：诊断筹码“聚散”动态
        - 核心修复: 不再返回字典，而是将多周期分数进行加权融合，返回一个单一的 pd.Series。
        """
        scores_by_period = {}
        concentration_level = (
            df.get('short_term_concentration_90pct_D', pd.Series(50.0, index=df.index)) +
            df.get('long_term_concentration_90pct_D', pd.Series(50.0, index=df.index)) +
            df.get('winner_concentration_90pct_D', pd.Series(50.0, index=df.index))
        ) / 3.0
        for p in periods:
            concentration_trend = df.get(f'SLOPE_{p}_winner_concentration_90pct_D', pd.Series(0.0, index=df.index))
            raw_bipolar_series = (concentration_level - 50) + (concentration_trend * 20)
            scores_by_period[p] = normalize_to_bipolar(raw_bipolar_series, df.index, window=p, sensitivity=1.0)
        
        # 进行多周期融合
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        final_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        if total_weight > 0:
            for p, weight in tf_weights.items():
                if p in scores_by_period:
                    final_score += scores_by_period[p] * (weight / total_weight)
        return final_score.clip(-1, 1)

    def _diagnose_axiom_cost_structure(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.1 · 融合输出版】筹码公理二：诊断“成本结构”动态
        - 核心修复: 不再返回字典，而是将多周期分数进行加权融合，返回一个单一的 pd.Series。
        """
        scores_by_period = {}
        raw_bipolar_series = (
            df.get('winner_loser_momentum_D', pd.Series(0.0, index=df.index)) -
            df.get('cost_divergence_normalized_D', pd.Series(0.0, index=df.index))
        )
        for p in periods:
            scores_by_period[p] = normalize_to_bipolar(raw_bipolar_series, df.index, window=p, sensitivity=1.0)
            
        # 进行多周期融合
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        final_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        if total_weight > 0:
            for p, weight in tf_weights.items():
                if p in scores_by_period:
                    final_score += scores_by_period[p] * (weight / total_weight)
        return final_score.clip(-1, 1)

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.1 · 融合输出版】筹码公理三：诊断“持股心态”动态
        - 核心修复: 不再返回字典，而是将多周期分数进行加权融合，返回一个单一的 pd.Series。
        """
        scores_by_period = {}
        raw_bipolar_series = (
            df.get('winner_conviction_index_D', pd.Series(0.0, index=df.index)) -
            df.get('loser_pain_index_D', pd.Series(0.0, index=df.index)) -
            df.get('chip_fatigue_index_D', pd.Series(0.0, index=df.index))
        )
        for p in periods:
            scores_by_period[p] = normalize_to_bipolar(raw_bipolar_series, df.index, window=p, sensitivity=0.8)
            
        # 进行多周期融合
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        final_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        if total_weight > 0:
            for p, weight in tf_weights.items():
                if p in scores_by_period:
                    final_score += scores_by_period[p] * (weight / total_weight)
        return final_score.clip(-1, 1)

    def _diagnose_axiom_peak_integrity(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.1 · 融合输出版】筹码公理四：诊断“筹码峰形态”
        - 核心修复: 不再返回字典，而是将多周期分数进行加权融合，返回一个单一的 pd.Series。
        """
        scores_by_period = {}
        price_vs_peak_raw = df['close_D'] - df.get('dominant_peak_cost_D', df['close_D'])
        peak_solidity = df.get('dominant_peak_solidity_D', pd.Series(0.5, index=df.index))
        raw_bipolar_series = price_vs_peak_raw * peak_solidity
        for p in periods:
            scores_by_period[p] = normalize_to_bipolar(raw_bipolar_series, df.index, window=p, sensitivity=1.2)
            
        # 进行多周期融合
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        final_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        if total_weight > 0:
            for p, weight in tf_weights.items():
                if p in scores_by_period:
                    final_score += scores_by_period[p] * (weight / total_weight)
        return final_score.clip(-1, 1)


