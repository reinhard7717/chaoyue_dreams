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

    def _get_fused_score(self, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.1 · 真理探针版】安全地从原子状态库中获取由融合层提供的态势分数。
        - 核心升级: 植入真理探针。如果获取不到信号，将打印明确的警告信息。
        """
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
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in self.strategy.df_indicators.columns:
            return self.strategy.df_indicators[name]
        else:
            print(f"    -> [认知层警告] 原子信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=self.strategy.df_indicators.index)

    def _get_playbook_score(self, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.0】安全地从剧本状态库中获取已计算的剧本分数。
        - 核心职责: 统一剧本信号获取路径，优先从 self.strategy.playbook_states 获取，
                      若无则返回默认值，确保数据流的稳定性。
        - 预警机制: 如果信号不存在，打印明确的警告信息。
        """
        if name in self.strategy.playbook_states: # 新增行
            return self.strategy.playbook_states[name] # 新增行
        else: # 新增行
            print(f"    -> [认知层警告] 剧本信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。") # 新增行
            return pd.Series(default, index=self.strategy.df_indicators.index) # 新增行

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V25.3 · 剧本依赖优化版】总指挥
        - 核心升级: 优化剧本推演顺序，确保被依赖的剧本在其依赖者之前被计算。
        - 状态归位: 不再污染 atomic_states，而是将所有剧本信号直接存入专属的 self.strategy.playbook_states。
        """
        self.strategy.playbook_states = {}
        priors = self._establish_prior_beliefs()
        self.strategy.atomic_states.update(priors)
        playbook_scores = {}
        # 优先计算被其他剧本依赖的风险信号
        playbook_scores.update(self._deduce_distribution_at_high(priors)) # 修改行: 提前计算
        playbook_scores.update(self._deduce_retail_fomo_retreat_risk(priors)) # 新增行: 提前计算
        playbook_scores.update(self._deduce_trend_exhaustion_risk(priors)) # 修改行: 提前计算

        # 计算其他机会和风险剧本
        playbook_scores.update(self._deduce_suppressive_accumulation(priors))
        playbook_scores.update(self._deduce_chasing_accumulation(priors))
        playbook_scores.update(self._deduce_capitulation_reversal(priors))
        playbook_scores.update(self._deduce_leading_dragon_awakening(priors))
        playbook_scores.update(self._deduce_sector_rotation_vanguard(priors))
        playbook_scores.update(self._deduce_energy_compression_breakout(priors))
        playbook_scores.update(self._deduce_divergence_reversal(priors))
        playbook_scores.update(self._deduce_long_term_profit_distribution_risk(priors))
        playbook_scores.update(self._deduce_market_uncertainty_risk(priors))
        playbook_scores.update(self._deduce_harvest_confirmation_risk(priors))
        playbook_scores.update(self._deduce_bull_trap_distribution_risk(priors))
        playbook_scores.update(self._deduce_liquidity_trap_risk(priors))
        playbook_scores.update(self._deduce_t0_arbitrage_pressure_risk(priors))
        playbook_scores.update(self._deduce_key_support_break_risk(priors))
        playbook_scores.update(self._deduce_high_level_structural_collapse_risk(priors))

        self.strategy.playbook_states.update(playbook_scores)
        print(f"【V25.3 · 剧本依赖优化版】分析完成，生成 {len(self.strategy.playbook_states)} 个剧本信号并存入专属状态库。") # 修改行
        return self.strategy.playbook_states

    def _deduce_suppressive_accumulation(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.2 · 背离证据增强版】贝叶斯推演：“主力打压吸筹”剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        - 【新增】引入市场矛盾（看涨背离）作为证据。
        """
        print("    -- [剧本推演] 主力打压吸筹 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_falling_evidence = self._forge_dynamic_evidence(price_change_bipolar.clip(upper=0).abs())
        efficiency_evidence = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('dip_absorption_power_D'), self.strategy.df_indicators.index, 55))
        process_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_STEALTH_ACCUMULATION', 0.0).clip(lower=0))
        chip_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        market_contradiction_bullish = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0).clip(lower=0))
        evidence_scores = np.stack([
            capital_confrontation.values, price_falling_evidence.values, efficiency_evidence.values,
            process_evidence.values, chip_evidence.values,
            market_contradiction_bullish.values
        ], axis=0)
        evidence_weights = np.array([0.2, 0.1, 0.1, 0.2, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_distribution_at_high(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.4 · 风险信号重构版】贝叶斯推演：“高位派发”风险剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        - 【重构】修正先验概率为 `COGNITIVE_PRIOR_REVERSAL_PROB`，更符合风险预警的本质。
        - 【增强】引入更多直接反映主力派发和筹码分散的证据，提高信号区分度。
        """
        print("    -- [剧本推演] 高位派发风险 (动态证据)...")
        capital_confrontation_bearish = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        price_overextension_risk = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_RISK_PRICE_OVEREXTENSION', 0.0))
        low_upward_efficiency = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5)).clip(0, 1))
        profit_vs_flow_bearish = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        chip_dispersion_evidence = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        market_contradiction_bearish = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0).clip(upper=0).abs())
        upper_shadow_pressure = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE', 0.0))
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        chip_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_BEARISH_DIVERGENCE', 0.0))
        evidence_scores = np.stack([
            capital_confrontation_bearish.values,
            price_overextension_risk.values,
            low_upward_efficiency.values,
            profit_vs_flow_bearish.values,
            chip_dispersion_evidence.values,
            market_contradiction_bearish.values,
            upper_shadow_pressure.values,
            fund_flow_bearish_divergence.values,
            chip_bearish_divergence.values
        ], axis=0)
        evidence_weights = np.array([0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.15, 0.05, 0.05])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH': posterior_prob.astype(np.float32)}

    def _deduce_trend_exhaustion_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.5 · 风险信号重构版】贝叶斯推演：“趋势衰竭”风险剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        - 【重构】修正先验概率为 `COGNITIVE_PRIOR_REVERSAL_PROB`，更符合风险预警的本质。
        - 【增强】引入多维度背离信号，更准确地捕捉趋势内在动能的衰竭。
        - 【优化】引入“趋势质量”和“新高强度”作为反向证据，抑制上涨日的误报。
        - 【新增】引入“周期顶风险”作为强力证据。
        """
        print("    -- [剧本推演] 趋势衰竭风险 (动态证据)...")
        price_momentum_divergence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0).clip(lower=0))
        winner_conviction_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        stagnation = self._forge_dynamic_evidence((1 - normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55)).clip(0, 1))
        concentration_slope = self._get_atomic_score('SLOPE_5_long_term_concentration_90pct_D', 0.0)
        chip_distribution_evidence = self._forge_dynamic_evidence(normalize_score(concentration_slope.clip(upper=0).abs(), self.strategy.df_indicators.index, 55))
        foundation_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FOUNDATION_BEARISH_DIVERGENCE', 0.0))
        dynamic_mechanics_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_DYNAMIC_MECHANICS_BEARISH_DIVERGENCE', 0.0))
        behavior_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_BEARISH_DIVERGENCE', 0.0))
        trend_quality_inverse = self._forge_dynamic_evidence(1 - self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0).clip(lower=0))
        new_high_strength_inverse = self._forge_dynamic_evidence(1 - self._get_atomic_score('CONTEXT_NEW_HIGH_STRENGTH', 0.0))
        cyclical_top_risk = self._forge_dynamic_evidence(self._get_atomic_score('COGNITIVE_RISK_CYCLICAL_TOP', 0.0))
        evidence_scores = np.stack([
            price_momentum_divergence.values,
            winner_conviction_decay.values,
            stagnation.values,
            chip_distribution_evidence.values,
            foundation_bearish_divergence.values,
            dynamic_mechanics_bearish_divergence.values,
            behavior_bearish_divergence.values,
            trend_quality_inverse.values,
            new_high_strength_inverse.values,
            cyclical_top_risk.values
        ], axis=0)
        evidence_weights = np.array([0.15, 0.1, 0.05, 0.1, 0.15, 0.15, 0.1, -0.05, -0.05, 0.15])
        evidence_weights /= np.sum(np.abs(evidence_weights))
        safe_scores = np.maximum(evidence_scores, 1e-9)
        weighted_evidence_sum = np.sum(safe_scores * evidence_weights[:, np.newaxis], axis=0)
        likelihood = pd.Series((weighted_evidence_sum + 1) / 2, index=self.strategy.df_indicators.index).clip(0.01, 0.99)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        self.strategy.atomic_states['COGNITIVE_RISK_TREND_EXHAUSTION'] = posterior_prob.astype(np.float32)
        return {'COGNITIVE_RISK_TREND_EXHAUSTION': posterior_prob.astype(np.float32)}

    def _establish_prior_beliefs(self) -> Dict[str, pd.Series]:
        """
        【V1.4 · 反转概率优化版】建立先验信念
        - 核心重构: 修正了双极性分数到概率的映射方式。将 `clip(lower=0)` 替换为 `(score + 1) / 2`，
                      确保中性信号贡献 0.5 的概率，避免了因错误映射导致先验概率被过度压制为零的问题。
        - 核心修复: 重构了 `prior_reversal` 的计算逻辑，废弃了 `(1 - market_regime.abs())` 的“一票否决”机制。
                      改为将市场压力和市场政权强度都视为对反转先验概率的正面贡献，并进行加权平均。
        - 【优化】在计算 `prior_reversal` 时，引入 `CONTEXT_TREND_CONFIRMED` 作为抑制因子。
        """
        states = {}
        market_regime = self._get_fused_score('FUSION_BIPOLAR_MARKET_REGIME', 0.0)
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        regime_weight = 0.5
        quality_weight = 0.5
        market_regime_prob = (market_regime + 1) / 2
        trend_quality_prob = (trend_quality + 1) / 2
        prior_trend = (market_regime_prob * regime_weight + trend_quality_prob * quality_weight).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(self.strategy.df_indicators.index.tz) if self.strategy.df_indicators.index.tz else probe_date_naive
            if probe_date in market_regime.index:
                print(f"    -> [先验信念探针] @ {probe_date.date()}:")
                print(f"       - 市场政权分 (market_regime): {market_regime.loc[probe_date]:.4f}")
                print(f"       - 趋势质量分 (trend_quality): {trend_quality.loc[probe_date]:.4f}")
                print(f"       - 市场政权概率 (market_regime_prob): {market_regime_prob.loc[probe_date]:.4f}")
                print(f"       - 趋势质量概率 (trend_quality_prob): {trend_quality_prob.loc[probe_date]:.4f}")
                print(f"       - 最终趋势先验概率 (prior_trend): {prior_trend.loc[probe_date]:.4f}")
        states['COGNITIVE_PRIOR_TREND_PROB'] = prior_trend.astype(np.float32)
        market_pressure = self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0)
        reversal_pressure_weight = 0.6
        reversal_regime_strength_weight = 0.4
        trend_confirmed = self._get_atomic_score('CONTEXT_TREND_CONFIRMED', 0.0)
        suppression_factor = (1 - trend_confirmed).clip(0, 1)
        prior_reversal_raw = (market_pressure.abs() * reversal_pressure_weight + market_regime.abs() * reversal_regime_strength_weight).clip(0, 1)
        prior_reversal = (prior_reversal_raw * suppression_factor).clip(0, 1)
        states['COGNITIVE_PRIOR_REVERSAL_PROB'] = prior_reversal.astype(np.float32)
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(self.strategy.df_indicators.index.tz) if self.strategy.df_indicators.index.tz else probe_date_naive
            if probe_date in market_pressure.index:
                print(f"       - 市场压力分 (market_pressure): {market_pressure.loc[probe_date]:.4f}")
                print(f"       - 市场政权绝对值 (market_regime.abs()): {market_regime.abs().loc[probe_date]:.4f}")
                print(f"       - 趋势确认分 (trend_confirmed): {trend_confirmed.loc[probe_date]:.4f}")
                print(f"       - 抑制因子 (suppression_factor): {suppression_factor.loc[probe_date]:.4f}")
                print(f"       - 最终反转先验概率 (prior_reversal): {prior_reversal.loc[probe_date]:.4f}")
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
        【V3.4 · 纯粹原子版】贝叶斯推演：“恐慌投降反转”剧本
        - 核心修复: 将证据 'SCORE_MICRO_BULLISH_RESONANCE' 修正为 'PROCESS_META_MICRO_BEHAVIOR_BOTTOM_REVERSAL'，
                      以符合新的信号生成职责。
        """
        print("    -- [剧本推演] 恐慌投降反转 (动态证据)...")
        upward_pressure = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(lower=0))
        price_rebound_evidence = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('dip_absorption_power_D'), self.strategy.df_indicators.index, 55))
        vol_ma55 = self._get_atomic_score('VOL_MA_55_D', 1.0)
        volume_spike = (self._get_atomic_score('volume_D') / vol_ma55.replace(0, 1.0)).fillna(1.0)
        volume_evidence = self._forge_dynamic_evidence(normalize_score(volume_spike, self.strategy.df_indicators.index, 55))
        process_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_LOSER_CAPITULATION', 0.0).clip(lower=0))
        micro_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_MICRO_BEHAVIOR_BOTTOM_REVERSAL', 0.0).clip(lower=0))
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

    def _deduce_leading_dragon_awakening(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.2 · 军备换装版】贝叶斯推演：“龙头苏醒”剧本
        - 核心修复: 将证据 'relative_strength_vs_index_D' 替换为更精准的 'industry_strength_rank_D'。
        """
        print("    -- [剧本推演] 龙头苏醒 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        breakout_quality = self._forge_dynamic_evidence(self._get_atomic_score('breakout_quality_score_D', 0.0))
        sector_sync = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_STOCK_SECTOR_SYNC', 0.0).clip(lower=0))
        relative_strength = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('industry_strength_rank_D', 0.5), self.strategy.df_indicators.index, 55))
        evidence_scores = np.stack([capital_confrontation.values, breakout_quality.values, sector_sync.values, relative_strength.values], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING': posterior_prob.astype(np.float32)}

    def _deduce_divergence_reversal(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.1 · 信号源修复版】贝叶斯推演：“背离反转”剧本
        - 核心逻辑: 专门利用融合层的“市场矛盾”信号来推演趋势反转的可能性。
        - 【修复】通过 `_get_atomic_score` 正确获取 `COGNITIVE_RISK_TREND_EXHAUSTION`。
        """
        print("    -- [剧本推演] 背离反转 (动态证据)...")
        market_contradiction = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0))
        market_pressure = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).abs())
        trend_exhaustion_risk = self._forge_dynamic_evidence(self._get_atomic_score('COGNITIVE_RISK_TREND_EXHAUSTION', 0.0))
        bullish_contradiction_evidence = market_contradiction.clip(lower=0)
        evidence_scores = np.stack([
            bullish_contradiction_evidence.values,
            market_pressure.values,
            trend_exhaustion_risk.values
        ], axis=0)
        evidence_weights = np.array([0.5, 0.3, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_DIVERGENCE_REVERSAL': posterior_prob.astype(np.float32)}

    def _deduce_sector_rotation_vanguard(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 纯粹原子版】贝叶斯推演：“板块轮动先锋”剧本
        - 核心修复: 将证据 'SCORE_FF_BULLISH_RESONANCE' 的获取方式从 _get_fused_score 修正为正确的 _get_atomic_score，
                      并更改为使用 'PROCESS_META_FUND_FLOW_BOTTOM_REVERSAL'，以符合新的信号生成职责。
        """
        print("    -- [剧本推演] 板块轮动先锋 (动态证据)...")
        sector_flow = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_FUND_FLOW_BOTTOM_REVERSAL', 0.0).clip(lower=0))
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
        bbw = self._get_atomic_score('BBW_21_2.0_D', 0.1)
        volatility_compression = self._forge_dynamic_evidence(1 - normalize_score(bbw, self.strategy.df_indicators.index, 144))
        volume_atrophy = self._forge_dynamic_evidence(1 - normalize_score(self._get_atomic_score('volume_D'), self.strategy.df_indicators.index, 144))
        entropy_col = next((col for col in self.strategy.df_indicators.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            entropy = self._get_atomic_score(entropy_col, 1.0)
            orderliness_score = self._forge_dynamic_evidence(1 - normalize_score(entropy, self.strategy.df_indicators.index, 144))
        else:
            orderliness_score = pd.Series(0.5, index=self.strategy.df_indicators.index)
        evidence_scores = np.stack([volatility_compression.values, volume_atrophy.values, orderliness_score.values], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.4])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION': posterior_prob.astype(np.float32)}

    def _forge_dynamic_evidence(self, raw_evidence: pd.Series, is_bipolar: bool = False) -> pd.Series:
        """
        【V1.0】动态证据锻造厂
        - 核心职责: 将一个静态的原始证据信号，锻造成一个融合了“状态-速度-加速度”的动态证据。
        - 数学逻辑: DynamicEvidence = w_s * State + w_v * Velocity + w_a * Acceleration
        """
        if not isinstance(raw_evidence, pd.Series) or raw_evidence.empty:
            return pd.Series(0.0, index=self.strategy.df_indicators.index)
        if not is_bipolar:
            bipolar_evidence = (raw_evidence.fillna(0.5) * 2 - 1).clip(-1, 1)
        else:
            bipolar_evidence = raw_evidence.clip(-1, 1)
        velocity = bipolar_evidence.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_window = 55
        state_score = bipolar_evidence
        velocity_score = normalize_to_bipolar(velocity, self.strategy.df_indicators.index, norm_window)
        acceleration_score = normalize_to_bipolar(acceleration, self.strategy.df_indicators.index, norm_window)
        w_state, w_velocity, w_acceleration = 0.3, 0.4, 0.3
        dynamic_force = (state_score * w_state + velocity_score * w_velocity + acceleration_score * w_acceleration)
        forged_evidence = (dynamic_force.clip(-1, 1) + 1) / 2.0
        return forged_evidence

    # [代码新增开始]
    def _deduce_long_term_profit_distribution_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0】贝叶斯推演：“长期获利盘派发”风险剧本
        - 核心逻辑: 识别长期获利盘的派发迹象。
        """
        print("    -- [剧本推演] 长期获利盘派发风险 (动态证据)...")
        # 证据1: 长期获利盘比例下降 (total_winner_rate_D 的衰减)
        long_term_profit_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        # 证据2: 筹码集中度下降 (SCORE_CHIP_AXIOM_CONCENTRATION 的负向)
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        # 证据3: 资金流出 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION 的负向)
        capital_outflow = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        evidence_scores = np.stack([
            long_term_profit_decay.values,
            chip_dispersion.values,
            capital_outflow.values
        ], axis=0)
        evidence_weights = np.array([0.4, 0.3, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION': posterior_prob.astype(np.float32)}

    def _deduce_market_uncertainty_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0】贝叶斯推演：“市场方向不明”风险剧本
        - 核心逻辑: 量化市场的不可预测性，通常表现为混沌、无趋势。
        """
        print("    -- [剧本推演] 市场方向不明风险 (动态证据)...")
        # 证据1: 市场政权处于震荡 (FUSION_BIPOLAR_MARKET_REGIME 接近0)
        regime_neutrality = self._forge_dynamic_evidence(1 - self._get_fused_score('FUSION_BIPOLAR_MARKET_REGIME', 0.0).abs())
        # 证据2: 趋势质量低下 (FUSION_BIPOLAR_TREND_QUALITY 接近0)
        low_trend_quality = self._forge_dynamic_evidence(1 - self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0).abs())
        # 证据3: 混沌度高 (SAMPLE_ENTROPY_D)
        entropy_col = next((col for col in self.strategy.df_indicators.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            high_entropy = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score(entropy_col, 0.5), self.strategy.df_indicators.index, 55))
        else:
            high_entropy = pd.Series(0.5, index=self.strategy.df_indicators.index)
        evidence_scores = np.stack([
            regime_neutrality.values,
            low_trend_quality.values,
            high_entropy.values
        ], axis=0)
        evidence_weights = np.array([0.3, 0.4, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index)) # 不确定性也可能导致反转
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_MARKET_UNCERTAINTY': posterior_prob.astype(np.float32)}

    def _deduce_retail_fomo_retreat_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.1 · 风险剧本】贝叶斯推演：“散户狂热主力撤退”风险剧本
        - 核心逻辑: 识别经典的牛市陷阱，散户Fomo情绪高涨，但主力资金却在悄然撤退。
        """
        print("    -- [剧本推演] 散户狂热主力撤退风险 (动态证据)...")
        # 证据1: 散户资金净流入 (SCORE_FF_AXIOM_CONSENSUS 负向)
        retail_inflow = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(upper=0).abs())
        # 证据2: 主力资金净流出 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION 负向)
        main_force_outflow = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        # 证据3: 价格上涨 (pct_change_D 正向)
        price_rising = self._forge_dynamic_evidence(normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21).clip(lower=0))
        # 证据4: 筹码分散 (SCORE_CHIP_AXIOM_CONCENTRATION 负向)
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        evidence_scores = np.stack([
            retail_inflow.values,
            main_force_outflow.values,
            price_rising.values,
            chip_dispersion.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.3, 0.2, 0.25])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_RETAIL_FOMO_RETREAT': posterior_prob.astype(np.float32)}

    def _deduce_harvest_confirmation_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.1 · 风险剧本】贝叶斯推演：“收割确认”风险剧本
        - 核心逻辑: 确认上冲派发是否伴随真实利润兑现，即主力在高位出货。
        - 【修复】从 playbook_states 获取 COGNITIVE_RISK_DISTRIBUTION_AT_HIGH。
        """
        print("    -- [剧本推演] 收割确认风险 (动态证据)...")
        # 证据1: 高位派发风险 (COGNITIVE_RISK_DISTRIBUTION_AT_HIGH)
        high_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0)) # 修改行
        # 证据2: 主力T+0效率高 (PROCESS_META_PROFIT_VS_FLOW 负向)
        high_t0_efficiency = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        # 证据3: 赢家信念衰减 (PROCESS_META_WINNER_CONVICTION_DECAY)
        winner_conviction_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        evidence_scores = np.stack([
            high_distribution_risk.values,
            high_t0_efficiency.values,
            winner_conviction_decay.values
        ], axis=0)
        evidence_weights = np.array([0.4, 0.3, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_HARVEST_CONFIRMATION': posterior_prob.astype(np.float32)}

    def _deduce_bull_trap_distribution_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0】贝叶斯推演：“主力诱多派发”风险剧本
        - 核心逻辑: 识别筹码派发背景下的诱多收割行为。
        """
        print("    -- [剧本推演] 主力诱多派发风险 (动态证据)...")
        # 证据1: 微观欺骗 (SCORE_MICRO_AXIOM_DECEPTION 负向，即伪装派发)
        micro_deception_bearish = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0).clip(upper=0).abs())
        # 证据2: 上影线抛压 (SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE)
        upper_shadow_pressure = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE', 0.0))
        # 证据3: 筹码分散 (SCORE_CHIP_AXIOM_CONCENTRATION 负向)
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        # 证据4: 价格上涨 (pct_change_D 正向)
        price_rising = self._forge_dynamic_evidence(normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21).clip(lower=0))
        evidence_scores = np.stack([
            micro_deception_bearish.values,
            upper_shadow_pressure.values,
            chip_dispersion.values,
            price_rising.values
        ], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION': posterior_prob.astype(np.float32)}

    def _deduce_liquidity_trap_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0】贝叶斯推演：“流动性陷阱”风险剧本
        - 核心逻辑: 资金流出与流动性枯竭的共振。
        """
        print("    -- [剧本推演] 流动性陷阱风险 (动态证据)...")
        # 证据1: 资金流出 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION 负向)
        capital_outflow = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        # 证据2: 成交量冷漠 (SCORE_BEHAVIOR_VOLUME_APATHY)
        volume_apathy = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_APATHY', 0.0))
        # 证据3: 波动率收缩 (1 - SCORE_FOUNDATION_AXIOM_VOLATILITY)
        volatility_contraction = self._forge_dynamic_evidence(1 - self._get_atomic_score('SCORE_FOUNDATION_AXIOM_VOLATILITY', 0.0).clip(lower=0))
        evidence_scores = np.stack([
            capital_outflow.values,
            volume_apathy.values,
            volatility_contraction.values
        ], axis=0)
        evidence_weights = np.array([0.4, 0.3, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_LIQUIDITY_TRAP': posterior_prob.astype(np.float32)}

    def _deduce_t0_arbitrage_pressure_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0】贝叶斯推演：“T+0套利压力”风险剧本
        - 核心逻辑: 主力短期派发行为，通过高频T+0操作获利。
        """
        print("    -- [剧本推演] T+0套利压力风险 (动态证据)...")
        # 证据1: 主力T+0效率高 (PROCESS_META_PROFIT_VS_FLOW 负向)
        high_t0_efficiency = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        # 证据2: 资金流出 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION 负向)
        capital_outflow = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        # 证据3: 微观欺骗 (SCORE_MICRO_AXIOM_DECEPTION 负向，即伪装派发)
        micro_deception_bearish = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0).clip(upper=0).abs())
        evidence_scores = np.stack([
            high_t0_efficiency.values,
            capital_outflow.values,
            micro_deception_bearish.values
        ], axis=0)
        evidence_weights = np.array([0.4, 0.3, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE': posterior_prob.astype(np.float32)}

    def _deduce_key_support_break_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0】贝叶斯推演：“关键支撑破位”风险剧本
        - 核心逻辑: 恐慌抛售击穿所有关键支撑的系统性风险。
        """
        print("    -- [剧本推演] 关键支撑破位风险 (动态证据)...")
        # 证据1: 市场压力大 (FUSION_BIPOLAR_MARKET_PRESSURE 负向)
        downward_pressure = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(upper=0).abs())
        # 证据2: 结构稳定性差 (SCORE_STRUCT_AXIOM_STABILITY 负向)
        low_structural_stability = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_STRUCT_AXIOM_STABILITY', 0.0).clip(upper=0).abs())
        # 证据3: 基础趋势弱 (SCORE_FOUNDATION_AXIOM_TREND 负向)
        weak_foundation_trend = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FOUNDATION_AXIOM_TREND', 0.0).clip(upper=0).abs())
        # 证据4: 恐慌抛售信号 (PROCESS_META_LOSER_CAPITULATION)
        loser_capitulation = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_LOSER_CAPITULATION', 0.0))
        evidence_scores = np.stack([
            downward_pressure.values,
            low_structural_stability.values,
            weak_foundation_trend.values,
            loser_capitulation.values
        ], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_KEY_SUPPORT_BREAK': posterior_prob.astype(np.float32)}

    def _deduce_high_level_structural_collapse_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.1 · 风险剧本】贝叶斯推演：“高位结构瓦解”风险剧本
        - 核心逻辑: 结构顶部与主力派发、散户接盘的共振。
        - 【修复】从 playbook_states 获取 COGNITIVE_RISK_DISTRIBUTION_AT_HIGH 和 COGNITIVE_RISK_RETAIL_FOMO_RETREAT。
        """
        print("    -- [剧本推演] 高位结构瓦解风险 (动态证据)...")
        # 证据1: 高位派发风险 (COGNITIVE_RISK_DISTRIBUTION_AT_HIGH)
        high_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0)) # 修改行
        # 证据2: 结构趋势形态恶化 (SCORE_STRUCT_AXIOM_TREND_FORM 负向)
        structural_trend_deterioration = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0).clip(upper=0).abs())
        # 证据3: 散户狂热主力撤退风险 (COGNITIVE_RISK_RETAIL_FOMO_RETREAT)
        retail_fomo_retreat = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0)) # 修改行
        # 证据4: 筹码分散 (SCORE_CHIP_AXIOM_CONCENTRATION 负向)
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        evidence_scores = np.stack([
            high_distribution_risk.values,
            structural_trend_deterioration.values,
            retail_fomo_retreat.values,
            chip_dispersion.values
        ], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE': posterior_prob.astype(np.float32)}

    def _deduce_market_uncertainty_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0】贝叶斯推演：“市场方向不明”风险剧本
        - 核心逻辑: 量化市场的不可预测性，通常表现为混沌、无趋势。
        """
        print("    -- [剧本推演] 市场方向不明风险 (动态证据)...")
        # 证据1: 市场政权处于震荡 (FUSION_BIPOLAR_MARKET_REGIME 接近0)
        regime_neutrality = self._forge_dynamic_evidence(1 - self._get_fused_score('FUSION_BIPOLAR_MARKET_REGIME', 0.0).abs())
        # 证据2: 趋势质量低下 (FUSION_BIPOLAR_TREND_QUALITY 接近0)
        low_trend_quality = self._forge_dynamic_evidence(1 - self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0).abs())
        # 证据3: 混沌度高 (SAMPLE_ENTROPY_D)
        entropy_col = next((col for col in self.strategy.df_indicators.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            high_entropy = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score(entropy_col, 0.5), self.strategy.df_indicators.index, 55))
        else:
            high_entropy = pd.Series(0.5, index=self.strategy.df_indicators.index)
        evidence_scores = np.stack([
            regime_neutrality.values,
            low_trend_quality.values,
            high_entropy.values
        ], axis=0)
        evidence_weights = np.array([0.3, 0.4, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index)) # 不确定性也可能导致反转
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_MARKET_UNCERTAINTY': posterior_prob.astype(np.float32)}
    # [代码新增结束]
