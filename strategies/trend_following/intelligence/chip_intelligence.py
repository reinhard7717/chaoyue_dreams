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

    def _synthesize_ultimate_signals(self, df: pd.DataFrame, concentration: Dict[int, pd.Series], cost_structure: Dict[int, pd.Series], holder_sentiment: Dict[int, pd.Series], peak_integrity: Dict[int, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V8.0 · 四大公理融合版】终极信号合成器
        - 核心重构: 基于“四大公理”的诊断结果，进行终极的、多时间周期的融合。
        """
        states = {}
        periods = sorted(concentration.keys())
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {'concentration': 0.3, 'cost_structure': 0.3, 'holder_sentiment': 0.2, 'peak_integrity': 0.2})
        # 步骤一：计算各周期的双极性“全息筹码健康分”
        bipolar_health_by_period = {}
        for p in periods:
            conc_score = concentration.get(p, pd.Series(0.0, index=df.index))
            cost_score = cost_structure.get(p, pd.Series(0.0, index=df.index))
            sentiment_score = holder_sentiment.get(p, pd.Series(0.0, index=df.index))
            peak_score = peak_integrity.get(p, pd.Series(0.0, index=df.index))
            bipolar_health_by_period[p] = (
                conc_score * axiom_weights['concentration'] +
                cost_score * axiom_weights['cost_structure'] +
                sentiment_score * axiom_weights['holder_sentiment'] +
                peak_score * axiom_weights['peak_integrity']
            ).clip(-1, 1)
        # 步骤二：分离为纯粹的看涨/看跌健康分
        bullish_scores_by_period = {p: score.clip(0, 1) for p, score in bipolar_health_by_period.items()}
        bearish_scores_by_period = {p: (score.clip(-1, 0) * -1) for p, score in bipolar_health_by_period.items()}
        # 步骤三：计算静态的共振信号 (零阶动态)
        bullish_resonance = pd.Series(0.0, index=df.index)
        bearish_resonance = pd.Series(0.0, index=df.index)
        numeric_weights = {int(k): v for k, v in tf_weights.items() if isinstance(v, (int, float))}
        total_weight = sum(numeric_weights.values())
        if total_weight > 0:
            for p, weight in numeric_weights.items():
                normalized_weight = weight / total_weight
                bullish_resonance += bullish_scores_by_period.get(p, 0.0) * normalized_weight
                bearish_resonance += bearish_scores_by_period.get(p, 0.0) * normalized_weight
        states['SCORE_CHIP_BULLISH_RESONANCE'] = bullish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BEARISH_RESONANCE'] = bearish_resonance.fillna(0).clip(0, 1).astype(np.float32)
        # 步骤四：计算动态信号 (一阶导数)
        bullish_momentum = bullish_resonance.diff().fillna(0)
        bearish_momentum = bearish_resonance.diff().fillna(0)
        # 归一化动态信号
        norm_bullish_momentum = normalize_score(bullish_momentum, df.index, 21, ascending=True)
        norm_bearish_momentum = normalize_score(bearish_momentum, df.index, 21, ascending=True)
        # 步骤五：赋值给命名准确的终极信号
        states['SCORE_CHIP_BULLISH_ACCELERATION'] = norm_bullish_momentum.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BEARISH_ACCELERATION'] = norm_bearish_momentum.clip(0, 1).astype(np.float32)
        states['SCORE_CHIP_BOTTOM_REVERSAL'] = (1.0 - norm_bearish_momentum).clip(0, 1).astype(np.float32) # 看跌动能减弱即为底部反转
        states['SCORE_CHIP_TOP_REVERSAL'] = (1.0 - norm_bullish_momentum).clip(0, 1).astype(np.float32) # 看涨动能减弱即为顶部反转
        # 步骤六：重铸战术回调信号
        states['SCORE_CHIP_TACTICAL_PULLBACK'] = (bullish_resonance * states['SCORE_CHIP_TOP_REVERSAL']).clip(0, 1).astype(np.float32)
        return states

    def _diagnose_axiom_concentration(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.0 · 双极性重构版】筹码公理一：诊断筹码“聚散”动态
        - 核心重构: 采用 normalize_to_bipolar 进行归一化，输出[-1, 1]的双极性分数。
                      正分代表筹码趋向集中，负分代表趋向分散。
        """
        scores = {}
        # 构造一个能反映集中度绝对水平的原始序列
        concentration_level = (
            df.get('short_term_concentration_90pct_D', pd.Series(50.0, index=df.index)) +
            df.get('long_term_concentration_90pct_D', pd.Series(50.0, index=df.index)) +
            df.get('winner_concentration_90pct_D', pd.Series(50.0, index=df.index))
        ) / 3.0
        for p in periods:
            # 构造一个能反映集中度变化趋势的原始序列
            concentration_trend = df.get(f'SLOPE_{p}_winner_concentration_90pct_D', pd.Series(0.0, index=df.index))
            # 融合静态水平和动态趋势，形成一个原始双极性序列
            raw_bipolar_series = (concentration_level - 50) + (concentration_trend * 20) # 减50使其中性化，趋势乘以权重
            # 使用双极归一化引擎进行最终裁决
            scores[p] = normalize_to_bipolar(raw_bipolar_series, df.index, window=p, sensitivity=1.0)
        return scores

    def _diagnose_axiom_cost_structure(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.0 · 双极性重构版】筹码公理二：诊断“成本结构”动态
        - 核心重构: 采用 normalize_to_bipolar 进行归一化，输出[-1, 1]的双极性分数。
                      正分代表成本结构健康（获利盘稳固），负分代表恶化。
        """
        scores = {}
        # 构造一个能反映成本结构健康度的核心原始序列
        # winner_loser_momentum: 正数代表获利盘优势扩大，本身就是优秀双极性信号源
        # cost_divergence_normalized: 成本发散度，越小越好，因此用负号
        raw_bipolar_series = (
            df.get('winner_loser_momentum_D', pd.Series(0.0, index=df.index)) -
            df.get('cost_divergence_normalized_D', pd.Series(0.0, index=df.index))
        )
        for p in periods:
            # 使用双极归一化引擎进行最终裁决
            scores[p] = normalize_to_bipolar(raw_bipolar_series, df.index, window=p, sensitivity=1.0)
        return scores

    def _diagnose_axiom_holder_sentiment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.0 · 双极性重构版】筹码公理三：诊断“持股心态”动态
        - 核心重构: 采用 normalize_to_bipolar 进行归一化，输出[-1, 1]的双极性分数。
                      正分代表心态稳定（惜售、躺平），负分代表心态不稳（恐慌、疲劳）。
        """
        scores = {}
        # 构造一个能反映持股心态的原始双极性序列
        # winner_conviction: 赢家信念，越高越好
        # loser_pain_index: 输家痛苦，越高越差，用负号
        # chip_fatigue_index: 筹码疲劳，越高越差，用负号
        raw_bipolar_series = (
            df.get('winner_conviction_index_D', pd.Series(0.0, index=df.index)) -
            df.get('loser_pain_index_D', pd.Series(0.0, index=df.index)) -
            df.get('chip_fatigue_index_D', pd.Series(0.0, index=df.index))
        )
        for p in periods:
            # 使用双极归一化引擎进行最终裁决
            scores[p] = normalize_to_bipolar(raw_bipolar_series, df.index, window=p, sensitivity=0.8)
        return scores

    def _diagnose_axiom_peak_integrity(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V2.0 · 双极性重构版】筹码公理四：诊断“筹码峰形态”
        - 核心重构: 采用 normalize_to_bipolar 进行归一化，输出[-1, 1]的双极性分数。
                      正分代表筹码峰提供有效支撑，负分代表构成压力。
        """
        scores = {}
        # 构造一个能反映价格与主峰关系的原始双极性序列
        # 价格高于主峰成本为正，低于为负
        price_vs_peak_raw = df['close_D'] - df.get('dominant_peak_cost_D', df['close_D'])
        # 用主峰稳固度作为权重，调节上述信号的强度
        peak_solidity = df.get('dominant_peak_solidity_D', pd.Series(0.5, index=df.index))
        raw_bipolar_series = price_vs_peak_raw * peak_solidity
        for p in periods:
            # 使用双极归一化引擎进行最终裁决
            scores[p] = normalize_to_bipolar(raw_bipolar_series, df.index, window=p, sensitivity=1.2)
        return scores


