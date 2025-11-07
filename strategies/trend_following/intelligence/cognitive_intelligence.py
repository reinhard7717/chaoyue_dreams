import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar

class CognitiveIntelligence:
    """
    【V20.0 · 贝叶斯战术推演引擎】
    - 核心重构: 废弃旧的、分散的信号合成方法，引入统一的“贝叶斯战术推演”框架。
    - 核心思想: 将A股的复杂博弈场景抽象为一系列“战术剧本”。引擎不再是简单地叠加信号，
                  而是基于融合层提供的“战场态势”（先验信念），结合原子层的“微观证据”（似然度），
                  通过贝叶斯推演，计算出每个战术剧本上演的“后验概率”（最终信号分）。
    - 收益: 使认知层的每一个判断都有清晰的数学逻辑和博弈论基础，直指A股本质。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        # 认知层不再需要依赖其他引擎的内部方法，实现完全解耦

    def _get_fused_score(self, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.1 · 真理探针版】安全地从原子状态库中获取由融合层提供的态势分数。
        - 核心升级: 植入真理探针。如果获取不到信号，将打印明确的警告信息。
        """
        # 认知层只消费融合层和原子层的信号
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        else:
            print(f"    -> [认知层警告] 融合态势信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=self.strategy.df_indicators.index)

    def _get_atomic_score(self, name: str, default: float = 0.0) -> pd.Series:
        """
        【V2.1 · 真理探针版】安全地从原子状态库或主数据帧中获取信号。
        - 核心升级: 植入真理探针。如果获取不到信号，将打印明确的警告信息。
        """
        # 优先从原子状态库获取，因为那里可能有经过处理的信号
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        # 如果没有，再从主数据帧获取原始指标
        elif name in self.strategy.df_indicators.columns:
            return self.strategy.df_indicators[name]
        else:
            print(f"    -> [认知层警告] 原子信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=self.strategy.df_indicators.index)

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V25.0 · 状态归位版】总指挥
        - 核心重构: 废弃了最终的融合步骤 `_fuse_and_adjudicate_playbooks`。
        - 新流程: 引擎的最终输出是所有独立的、经过动态证据锻造的剧本后验概率。
        - 状态归位: 不再污染 atomic_states，而是将所有剧本信号直接存入专属的 self.strategy.playbook_states。
        """
        # 清空旧的剧本状态，确保每次运行都是全新的
        self.strategy.playbook_states = {}
        
        # --- 步骤一: 建立先验信念 (逻辑不变) ---
        priors = self._establish_prior_beliefs()
        # 先验概率作为中间计算结果，可以放入原子状态供调试
        self.strategy.atomic_states.update(priors)
        
        # --- 步骤二: 并行推演所有战术剧本，直接获得“动态后验概率” ---
        playbook_scores = {}
        playbook_scores.update(self._deduce_suppressive_accumulation(priors))
        playbook_scores.update(self._deduce_chasing_accumulation(priors))
        playbook_scores.update(self._deduce_capitulation_reversal(priors))
        playbook_scores.update(self._deduce_distribution_at_high(priors))
        playbook_scores.update(self._deduce_leading_dragon_awakening(priors))
        playbook_scores.update(self._deduce_sector_rotation_vanguard(priors))
        playbook_scores.update(self._deduce_trend_exhaustion_risk(priors))
        playbook_scores.update(self._deduce_energy_compression_breakout(priors))
        
        # 将所有计算出的剧本信号存入其专属的状态库
        self.strategy.playbook_states.update(playbook_scores)
        
        # --- 步骤三: (已废弃) 不再进行融合，直接输出独立剧本信号 ---
        print(f"【V25.0 · 状态归位版】分析完成，生成 {len(self.strategy.playbook_states)} 个剧本信号并存入专属状态库。")
        # 返回剧本信号，供 intelligence_layer 可能的日志记录或其他用途，但不用于更新 atomic_states
        return self.strategy.playbook_states

    def _establish_prior_beliefs(self) -> Dict[str, pd.Series]:
        """
        【V1.1 · 信念融合重构版】建立先验信念
        - 核心重构: 废弃了脆弱的乘法融合模型 (A*B)，改为更健壮的加权平均模型 (A*w1 + B*w2)。
                      这避免了因单一维度评分为零或负值而导致先验概率被“一票否决”的问题，
                      使模型能更好地适应A股充满噪声和矛盾信号的实战环境。
        - 探针植入: 新增探针，打印计算先验概率所依赖的两个核心态势分及其最终结果。
        """
        states = {}
        market_regime = self._get_fused_score('FUSION_BIPOLAR_MARKET_REGIME', 0.0)
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        # --- 废弃脆弱的乘法模型 ---
        # prior_trend = (market_regime.clip(lower=0) * trend_quality.clip(lower=0)).pow(0.5)
        # --- 采用更健壮的加权平均模型 ---
        # 权重可以根据策略的偏好进行配置
        regime_weight = 0.5
        quality_weight = 0.5
        prior_trend = (market_regime.clip(lower=0) * regime_weight + trend_quality.clip(lower=0) * quality_weight)
        # --- 新增探针，监控先验概率的计算过程 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(self.strategy.df_indicators.index.tz) if self.strategy.df_indicators.index.tz else probe_date_naive
            if probe_date in market_regime.index:
                print(f"    -> [先验信念探针] @ {probe_date.date()}:")
                print(f"       - 市场政权分 (market_regime): {market_regime.loc[probe_date]:.4f}")
                print(f"       - 趋势质量分 (trend_quality): {trend_quality.loc[probe_date]:.4f}")
                print(f"       - 最终趋势先验概率 (prior_trend): {prior_trend.loc[probe_date]:.4f}")
        states['COGNITIVE_PRIOR_TREND_PROB'] = prior_trend.astype(np.float32)
        market_pressure = self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0)
        prior_reversal = (market_pressure.abs() * (1 - market_regime.abs())).pow(0.5)
        states['COGNITIVE_PRIOR_REVERSAL_PROB'] = prior_reversal.astype(np.float32)
        return states

    def _fuse_and_adjudicate_playbooks(self, playbook_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.2 · 指挥链修复版】融合与裁决模块
        - 核心修复: 将引用的信号名称从废弃的 'COGNITIVE_FORGED_...' 修正为 'COGNITIVE_PLAYBOOK_...' 和 'COGNITIVE_RISK_...'，
                      重新连接了信号融合的指挥链。
        """
        states = {}
        df_index = self.strategy.df_indicators.index
        
        # 融合所有看涨剧本的分数
        bullish_playbooks = [
            'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION',
            'COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION',
            'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL',
            'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING',
            'COGNITIVE_PLAYBOOK_SECTOR_ROTATION_VANGUARD',
            'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION',
        ]
        bullish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bullish_playbooks]
        # 取所有看涨剧本中的最高分作为当天的看涨总分
        cognitive_bullish_score = np.maximum.reduce([s.values for s in bullish_scores])
        states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df_index, dtype=np.float32)
        
        # 融合所有看跌剧本的分数
        bearish_playbooks = [
            'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH',
            'COGNITIVE_RISK_TREND_EXHAUSTION',
        ]
        bearish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bearish_playbooks]
        # 取所有看跌剧本中的最高分作为当天的看跌总分
        cognitive_bearish_score = np.maximum.reduce([s.values for s in bearish_scores])
        states['COGNITIVE_BEARISH_SCORE'] = pd.Series(cognitive_bearish_score, index=df_index, dtype=np.float32)
        
        return states

    def _deduce_suppressive_accumulation(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.1 · 动态证据版】贝叶斯推演：“主力打压吸筹”剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        """
        print("    -- [剧本推演] 主力打压吸筹 (动态证据)...")
        # --- 1. 收集并锻造所有相关证据 ---
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_falling_evidence = self._forge_dynamic_evidence(price_change_bipolar.clip(upper=0).abs())
        efficiency_evidence = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('dip_absorption_power_D'), self.strategy.df_indicators.index, 55))
        process_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_STEALTH_ACCUMULATION', 0.0).clip(lower=0))
        chip_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        # --- 2. 计算似然度 P(证据 | 打压吸筹) ---
        evidence_scores = np.stack([
            capital_confrontation.values, price_falling_evidence.values, efficiency_evidence.values,
            process_evidence.values, chip_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 P(反转) ---
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 (最终信号分) ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_chasing_accumulation(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.2 · 级联探针版】贝叶斯推演：“主力拉升抢筹”剧本
        - 探针植入: 打印本剧本所依赖的先验概率和计算出的似然度，以诊断后验概率为零的原因。
        """
        print("    -- [剧本推演] 主力拉升抢筹 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_rising_evidence = self._forge_dynamic_evidence(price_change_bipolar.clip(lower=0))
        efficiency_evidence = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55))
        urgency_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_MAIN_FORCE_URGENCY', 0.0).clip(lower=0))
        conviction_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION', 0.0).clip(lower=0))
        process_evidence = (urgency_evidence * conviction_evidence).pow(0.5)
        chip_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        evidence_scores = np.stack([
            capital_confrontation.values, price_rising_evidence.values, efficiency_evidence.values,
            process_evidence.values, chip_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        
        # --- 级联探针: 认知层 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date = pd.to_datetime(probe_dates_str[0])
            if self.strategy.df_indicators.index.tz:
                probe_date = probe_date.tz_localize(self.strategy.df_indicators.index.tz)
            if probe_date in likelihood.index:
                print(f"      -> [认知层探针] @ {probe_date.date()} for '主力拉升抢筹':")
                print(f"         - 先验概率 (P(Trend)): {prior_prob.loc[probe_date]:.4f}")
                print(f"         - 似然度 (P(证据|剧本)): {likelihood.loc[probe_date]:.4f}")

        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_capitulation_reversal(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.3 · 命名协议修复版】贝叶斯推演：“恐慌投降反转”剧本
        - 核心修复: 将证据 'SCORE_MICRO_BULLISH_RESONANCE' 修正为 'SCORE_MICRO_BEHAVIOR_BULLISH_RESONANCE'，
                      以符合统一命名协议。
        """
        print("    -- [剧本推演] 恐慌投降反转 (动态证据)...")
        upward_pressure = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(lower=0))
        price_rebound_evidence = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('dip_absorption_power_D'), self.strategy.df_indicators.index, 55))
        vol_ma55 = self._get_atomic_score('VOL_MA_55_D', 1.0)
        volume_spike = (self._get_atomic_score('volume_D') / vol_ma55.replace(0, 1.0)).fillna(1.0)
        volume_evidence = self._forge_dynamic_evidence(normalize_score(volume_spike, self.strategy.df_indicators.index, 55))
        process_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_LOSER_CAPITULATION', 0.0).clip(lower=0))
        # 修正信号名称以符合统一命名协议
        micro_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_MICRO_BEHAVIOR_BULLISH_RESONANCE', 0.0).clip(lower=0))
        evidence_scores = np.stack([
            upward_pressure.values, price_rebound_evidence.values, volume_evidence.values,
            process_evidence.values, micro_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.1, 0.2, 0.2, 0.3, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL': posterior_prob.astype(np.float32)}

    def _deduce_distribution_at_high(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.1 · 动态证据版】贝叶斯推演：“高位派发”风险剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        """
        print("    -- [剧本推演] 高位派发风险 (动态证据)...")
        # --- 1. 收集并锻造所有相关证据 ---
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        price_at_high = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('BIAS_55_D'), self.strategy.df_indicators.index, 55))
        efficiency_evidence = self._forge_dynamic_evidence((1 - normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55)).clip(0, 1))
        process_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(lower=0))
        chip_evidence = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        # --- 2. 计算似然度 P(证据 | 高位派发) ---
        evidence_scores = np.stack([
            capital_confrontation.values, price_at_high.values, efficiency_evidence.values,
            process_evidence.values, chip_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.2, 0.1, 0.2, 0.2, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 P(趋势) ---
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 (最终风险分) ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH': posterior_prob.astype(np.float32)}

    def _deduce_leading_dragon_awakening(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.2 · 军备换装版】贝叶斯推演：“龙头苏醒”剧本
        - 核心修复: 将证据 'relative_strength_vs_index_D' 替换为更精准的 'industry_strength_rank_D'。
        """
        print("    -- [剧本推演] 龙头苏醒 (动态证据)...")
        # --- 1. 收集并锻造所有相关证据 ---
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        breakout_quality = self._forge_dynamic_evidence(self._get_atomic_score('breakout_quality_score_D', 0.0))
        sector_sync = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_STOCK_SECTOR_SYNC', 0.0).clip(lower=0))
        # 使用 'industry_strength_rank_D' 替换不存在的 'relative_strength_vs_index_D'
        relative_strength = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('industry_strength_rank_D', 0.5), self.strategy.df_indicators.index, 55))
        # --- 2. 计算似然度 ---
        evidence_scores = np.stack([capital_confrontation.values, breakout_quality.values, sector_sync.values, relative_strength.values], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 ---
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING': posterior_prob.astype(np.float32)}

    def _deduce_trend_exhaustion_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.2 · 动态证据版】贝叶斯推演：“趋势衰竭”风险剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        """
        print("    -- [剧本推演] 趋势衰竭风险 (动态证据)...")
        # --- 1. 收集并锻造所有相关证据 ---
        divergence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0).clip(lower=0))
        winner_conviction_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        stagnation = self._forge_dynamic_evidence((1 - normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55)).clip(0, 1))
        concentration_slope = self._get_atomic_score('SLOPE_5_long_term_concentration_90pct_D', 0.0)
        chip_distribution_evidence = self._forge_dynamic_evidence(normalize_score(concentration_slope.clip(upper=0).abs(), self.strategy.df_indicators.index, 55))
        # --- 2. 计算似然度 ---
        evidence_scores = np.stack([divergence.values, winner_conviction_decay.values, stagnation.values, chip_distribution_evidence.values], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.1, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 ---
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_TREND_EXHAUSTION': posterior_prob.astype(np.float32)}

    def _deduce_sector_rotation_vanguard(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.4 · 信号源修复版】贝叶斯推演：“板块轮动先锋”剧本
        - 核心修复: 将证据 'SCORE_FF_BULLISH_RESONANCE' 的获取方式从 _get_fused_score 修正为正确的 _get_atomic_score，
                      因为它是一个原子/基础情报信号，而非融合层信号。
        """
        print("    -- [剧本推演] 板块轮动先锋 (动态证据)...")
        # 使用 _get_atomic_score 获取基础情报信号，而不是 _get_fused_score
        sector_flow = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FUND_FLOW_BULLISH_RESONANCE', 0.0).clip(lower=0))
        price_position = self._forge_dynamic_evidence(1 - normalize_score(self._get_atomic_score('BIAS_144_D', 0.0), self.strategy.df_indicators.index, 144))
        chip_cleanliness = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_CLEANLINESS', 0.0))
        hot_sector_cooling = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_HOT_SECTOR_COOLING', 0.0))
        evidence_scores = np.stack([sector_flow.values, price_position.values, chip_cleanliness.values, hot_sector_cooling.values], axis=0)
        evidence_weights = np.array([0.4, 0.2, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_SECTOR_ROTATION_VANGUARD': posterior_prob.astype(np.float32)}

    def _deduce_energy_compression_breakout(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 动态证据版】贝叶斯推演：“能量压缩爆发”剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        """
        print("    -- [剧本推演] 能量压缩爆发 (动态证据)...")
        # --- 1. 收集并锻造所有相关证据 ---
        bbw = self._get_atomic_score('BBW_21_2.0_D', 0.1)
        volatility_compression = self._forge_dynamic_evidence(1 - normalize_score(bbw, self.strategy.df_indicators.index, 144))
        volume_atrophy = self._forge_dynamic_evidence(1 - normalize_score(self._get_atomic_score('volume_D'), self.strategy.df_indicators.index, 144))
        entropy_col = next((col for col in self.strategy.df_indicators.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            entropy = self._get_atomic_score(entropy_col, 1.0)
            orderliness_score = self._forge_dynamic_evidence(1 - normalize_score(entropy, self.strategy.df_indicators.index, 144))
        else:
            orderliness_score = pd.Series(0.5, index=self.strategy.df_indicators.index)
        # --- 2. 计算似然度 ---
        evidence_scores = np.stack([volatility_compression.values, volume_atrophy.values, orderliness_score.values], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.4])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 ---
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION': posterior_prob.astype(np.float32)}

    def _forge_dynamic_evidence(self, raw_evidence: pd.Series, is_bipolar: bool = False) -> pd.Series:
        """
        【V1.0 · 新增】动态证据锻造厂
        - 核心职责: 将一个静态的原始证据信号，锻造成一个融合了“状态-速度-加速度”的动态证据。
        - 数学逻辑: DynamicEvidence = w_s * State + w_v * Velocity + w_a * Acceleration
        """
        if not isinstance(raw_evidence, pd.Series) or raw_evidence.empty:
            return pd.Series(0.0, index=self.strategy.df_indicators.index)
        # 统一转换为[-1, 1]的双极性信号进行处理
        if not is_bipolar:
            # 假设输入是[0, 1]的单极信号
            bipolar_evidence = (raw_evidence.fillna(0.5) * 2 - 1).clip(-1, 1)
        else:
            bipolar_evidence = raw_evidence.clip(-1, 1)
        # 计算速度（一阶导数）和加速度（二阶导数）
        velocity = bipolar_evidence.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        # 归一化
        norm_window = 55
        state_score = bipolar_evidence
        velocity_score = normalize_to_bipolar(velocity, self.strategy.df_indicators.index, norm_window)
        acceleration_score = normalize_to_bipolar(acceleration, self.strategy.df_indicators.index, norm_window)
        # 加权融合
        w_state, w_velocity, w_acceleration = 0.3, 0.4, 0.3
        dynamic_force = (state_score * w_state + velocity_score * w_velocity + acceleration_score * w_acceleration)
        # 将最终的力转换回[0, 1]区间的证据强度
        forged_evidence = (dynamic_force.clip(-1, 1) + 1) / 2.0
        return forged_evidence
