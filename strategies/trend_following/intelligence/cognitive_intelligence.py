# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, is_limit_up

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
        if name in self.strategy.playbook_states:
            return self.strategy.playbook_states[name]
        else:
            print(f"    -> [认知层警告] 剧本信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=self.strategy.df_indicators.index)

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V25.7 · 隐秘筑底背离与风险剧本顺序优化版】总指挥
        - 核心升级: 引入新的认知剧本 `COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE` 和 `COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE`。
        - 核心修复: 调整风险剧本的调用顺序，确保在所有机会剧本之后进行推演。
        """
        print("启动【V25.7 · 隐秘筑底背离与风险剧本顺序优化版】认知情报分析...")
        self.strategy.playbook_states = {} # 初始化剧本状态库
        priors = self._establish_prior_beliefs()
        self.strategy.atomic_states.update(priors)
        # 计算所有机会剧本，并立即更新到 self.strategy.playbook_states
        self.strategy.playbook_states.update(self._deduce_suppressive_accumulation(priors))
        self.strategy.playbook_states.update(self._deduce_chasing_accumulation(priors))
        self.strategy.playbook_states.update(self._deduce_capitulation_reversal(priors))
        self.strategy.playbook_states.update(self._deduce_leading_dragon_awakening(priors))
        self.strategy.playbook_states.update(self._deduce_sector_rotation_vanguard(priors))
        self.strategy.playbook_states.update(self._deduce_energy_compression_breakout(priors))
        self.strategy.playbook_states.update(self._deduce_divergence_reversal(priors))
        self.strategy.playbook_states.update(self._deduce_stealth_bottoming_divergence(priors))
        self.strategy.playbook_states.update(self._deduce_micro_absorption_divergence(priors))
        # 优先计算所有风险信号，并立即更新到 self.strategy.playbook_states
        self.strategy.playbook_states.update(self._deduce_distribution_at_high(priors))
        self.strategy.playbook_states.update(self._deduce_retail_fomo_retreat_risk(priors))
        self.strategy.playbook_states.update(self._deduce_long_term_profit_distribution_risk(priors))
        self.strategy.playbook_states.update(self._deduce_trend_exhaustion_risk(priors))
        self.strategy.playbook_states.update(self._deduce_market_uncertainty_risk(priors))
        self.strategy.playbook_states.update(self._deduce_harvest_confirmation_risk(priors))
        self.strategy.playbook_states.update(self._deduce_bull_trap_distribution_risk(priors))
        self.strategy.playbook_states.update(self._deduce_liquidity_trap_risk(priors))
        self.strategy.playbook_states.update(self._deduce_t0_arbitrage_pressure_risk(priors))
        self.strategy.playbook_states.update(self._deduce_key_support_break_risk(priors))
        self.strategy.playbook_states.update(self._deduce_high_level_structural_collapse_risk(priors))
        print(f"【V25.7 · 隐秘筑底背离与风险剧本顺序优化版】分析完成，生成 {len(self.strategy.playbook_states)} 个剧本信号并存入专属状态库。")
        return self.strategy.playbook_states


    def _deduce_suppressive_accumulation(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.2 · 背离证据增强版】贝叶斯推演：“主力打压吸筹”剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        - 引入市场矛盾（看涨背离）作为证据。
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
        【V3.6 · 风险信号重构版】贝叶斯推演：“高位派发”风险剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        - 【重构】修正先验概率为 `COGNITIVE_PRIOR_REVERSAL_PROB`，更符合风险预警的本质。
        - 【增强】引入更多直接反映主力派发和筹码分散的证据，提高信号区分度。
        - 【修复】将 `SCORE_BEHAVIOR_RISK_PRICE_OVEREXTENSION` 替换为融合层信号 `FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT` 的负向部分。
        - 【修复】将 `SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE` 替换为融合层信号 `FUSION_BIPOLAR_UPPER_SHADOW_INTENT` 的负向部分。
        """
        print("    -- [剧本推演] 高位派发风险 (动态证据)...")
        capital_confrontation_bearish = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        price_overextension_risk = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0).clip(upper=0).abs())
        low_upward_efficiency = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5)).clip(0, 1))
        profit_vs_flow_bearish = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        chip_dispersion_evidence = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        market_contradiction_bearish = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0).clip(upper=0).abs())
        upper_shadow_pressure = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0).clip(upper=0).abs()) # 修改行
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
        【V2.9 · 深度博弈版证据链优化版】贝叶斯推演：“趋势衰竭”风险剧本
        - 核心升级: 不再直接使用原始证据，而是先通过 `_forge_dynamic_evidence` 进行动态锻造。
        - 【重构】修正先验概率为 `COGNITIVE_PRIOR_REVERSAL_PROB`，更符合风险预警的本质。
        - 【增强】引入多维度背离信号、资金流、筹码、结构、微观行为等证据，更准确地捕捉趋势内在动能的衰竭。
        - 【优化】引入“趋势质量”和“新高强度”作为反向证据，抑制上涨日的误报。
        - 引入“周期顶风险”作为强力证据。
        - 【修复】修正 `trend_quality_inverse` 和 `new_high_strength_inverse` 的权重为正。
        - 【优化】调整证据权重，降低 `stagnation_evidence` 和 `price_overextension_risk` 在涨停日的风险贡献，
                  提高 `trend_quality_inverse` 和 `new_high_strength_inverse` 的权重，以更好地抑制误报。
        - 【修复】移除之前为强制极小值而添加的 `.mask` 逻辑，现在由 `_forge_dynamic_evidence` 统一处理零值输入。
        """
        print("    -- [剧本推演] 趋势衰竭风险 (动态证据)...")
        # 1. 价格动能与效率的衰减
        price_momentum_divergence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0).clip(lower=0))
        stagnation_evidence = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5)).clip(0, 1))
        # 价格超买意图负向是风险
        raw_price_overextension_score = self._get_fused_score('FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0)
        price_overextension_risk = self._forge_dynamic_evidence(raw_price_overextension_score.clip(upper=0).abs())
        # 2. 主力资金的撤退与意图
        winner_conviction_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        # 资本对抗为负是资金撤退
        raw_capital_confrontation_score = self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        capital_retreat_evidence = self._forge_dynamic_evidence(raw_capital_confrontation_score.clip(upper=0).abs())
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        retail_fomo_retreat_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0))
        # 3. 筹码的派发与分散
        chip_dispersion_evidence = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        chip_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_BEARISH_DIVERGENCE', 0.0))
        long_term_profit_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 0.0))
        # 4. 结构与形态的恶化
        # 结构趋势形态为负是恶化
        raw_structural_trend_form_score = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        structural_deterioration = self._forge_dynamic_evidence(raw_structural_trend_form_score.clip(upper=0).abs())
        # 市场矛盾为负是风险
        raw_market_contradiction_score = self._get_fused_score('FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0)
        market_contradiction_bearish = self._forge_dynamic_evidence(raw_market_contradiction_score.clip(upper=0).abs())
        # 上影线意图为负是抛压
        raw_upper_shadow_intent_score = self._get_fused_score('FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0)
        upper_shadow_pressure_risk = self._forge_dynamic_evidence(raw_upper_shadow_intent_score.clip(upper=0).abs())
        # 5. 宏观周期与风险
        cyclical_top_risk = self._forge_dynamic_evidence(self._get_atomic_score('COGNITIVE_RISK_CYCLICAL_TOP', 0.0))
        # 6. 趋势质量的反向证据 (低趋势质量是风险证据)
        trend_quality_inverse = self._forge_dynamic_evidence(1 - self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0).clip(lower=0))
        # 7. 新高强度的反向证据 (新高强度低是风险证据)
        new_high_strength_inverse = self._forge_dynamic_evidence(1 - self._get_atomic_score('CONTEXT_NEW_HIGH_STRENGTH', 0.0))
        evidence_scores = np.stack([
            price_momentum_divergence.values,
            winner_conviction_decay.values,
            stagnation_evidence.values,
            chip_dispersion_evidence.values,
            fund_flow_bearish_divergence.values,
            structural_deterioration.values,
            capital_retreat_evidence.values,
            cyclical_top_risk.values,
            price_overextension_risk.values,
            upper_shadow_pressure_risk.values,
            market_contradiction_bearish.values,
            retail_fomo_retreat_risk.values,
            chip_bearish_divergence.values,
            long_term_profit_distribution_risk.values,
            trend_quality_inverse.values,
            new_high_strength_inverse.values
        ], axis=0)
        # 重新分配权重，确保所有权重为正，且总和为1
        evidence_weights = np.array([
            0.10, # price_momentum_divergence
            0.08, # winner_conviction_decay
            0.03, # stagnation_evidence (降低权重，避免涨停日误报)
            0.08, # chip_dispersion_evidence
            0.07, # fund_flow_bearish_divergence
            0.06, # structural_deterioration
            0.10, # capital_retreat_evidence
            0.07, # cyclical_top_risk
            0.03, # price_overextension_risk (降低权重，避免涨停日误报)
            0.04, # upper_shadow_pressure_risk
            0.04, # market_contradiction_bearish
            0.04, # retail_fomo_retreat_risk
            0.03, # chip_bearish_divergence
            0.03, # long_term_profit_distribution_risk
            0.10, # trend_quality_inverse (提高权重，强调趋势健康度对风险的抑制)
            0.10  # new_high_strength_inverse (提高权重，强调新高强度对风险的抑制)
        ])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        self.strategy.atomic_states['COGNITIVE_RISK_TREND_EXHAUSTION'] = posterior_prob.astype(np.float32)
        return {'COGNITIVE_RISK_TREND_EXHAUSTION': posterior_prob.astype(np.float32)}

    def _establish_prior_beliefs(self) -> Dict[str, pd.Series]:
        """
        【V1.5 · 趋势结构强化版】建立先验信念
        - 核心升级: 将融合层的“趋势结构分” (FUSION_BIPOLAR_TREND_STRUCTURE_SCORE) 融入到
                      “趋势先验概率” (COGNITIVE_PRIOR_TREND_PROB) 的计算中，以提供更稳定、更具结构性的背景判断。
        """
        states = {}
        market_regime = self._get_fused_score('FUSION_BIPOLAR_MARKET_REGIME', 0.0)
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        # 获取融合层的趋势结构分
        trend_structure_score = self._get_fused_score('FUSION_BIPOLAR_TREND_STRUCTURE_SCORE', 0.0)
        # 调整趋势先验概率的权重，引入趋势结构分
        # 示例权重，需要根据回测优化
        regime_weight = 0.3
        quality_weight = 0.3
        # 趋势结构分的权重
        structure_weight = 0.4
        market_regime_prob = (market_regime + 1) / 2
        trend_quality_prob = (trend_quality + 1) / 2
        # 趋势结构分转换为概率
        trend_structure_prob = (trend_structure_score + 1) / 2
        # 融合趋势结构分到趋势先验概率中
        prior_trend = (
            market_regime_prob * regime_weight +
            trend_quality_prob * quality_weight +
            trend_structure_prob * structure_weight # 融合趋势结构分
        ).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(self.strategy.df_indicators.index.tz) if self.strategy.df_indicators.index.tz else probe_date_naive
            if probe_date in market_regime.index:
                print(f"    -> [先验信念探针] @ {probe_date.date()}:")
                print(f"       - 市场政权分 (market_regime): {market_regime.loc[probe_date]:.4f}")
                print(f"       - 趋势质量分 (trend_quality): {trend_quality.loc[probe_date]:.4f}")
                print(f"       - 趋势结构分 (trend_structure_score): {trend_structure_score.loc[probe_date]:.4f}")
                print(f"       - 市场政权概率 (market_regime_prob): {market_regime_prob.loc[probe_date]:.4f}")
                print(f"       - 趋势质量概率 (trend_quality_prob): {trend_quality_prob.loc[probe_date]:.4f}")
                print(f"       - 趋势结构概率 (trend_structure_prob): {trend_structure_prob.loc[probe_date]:.4f}")
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
        【V3.4 · 指挥链修复与新剧本集成版】融合与裁决模块
        - 核心升级: 将新的认知剧本 `COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE` 集成到看涨剧本列表中。
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
            'COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE',
            'COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE', # 新增行
        ]
        bullish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bullish_playbooks]
        # 取所有看涨剧本中的最高分作为当天的看涨总分
        cognitive_bullish_score = np.maximum.reduce([s.values for s in bullish_scores])
        states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df_index, dtype=np.float32)
        # 融合所有看跌剧本的分数
        bearish_playbooks = [
            'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH',
            'COGNITIVE_RISK_TREND_EXHAUSTION',
            'COGNITIVE_RISK_MARKET_UNCERTAINTY',
            'COGNITIVE_RISK_RETAIL_FOMO_RETREAT',
            'COGNITIVE_RISK_HARVEST_CONFIRMATION',
            'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION',
            'COGNITIVE_RISK_LIQUIDITY_TRAP',
            'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE',
            'COGNITIVE_RISK_KEY_SUPPORT_BREAK',
            'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE',
            'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION'
        ]
        bearish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bearish_playbooks]
        # 取所有看跌剧本中的最高分作为当天的看跌总分
        cognitive_bearish_score = np.maximum.reduce([s.values for s in bearish_scores])
        states['COGNITIVE_BEARISH_SCORE'] = pd.Series(cognitive_bearish_score, index=df_index, dtype=np.float32)
        return states


    def _deduce_chasing_accumulation(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.3 · 级联探针版】贝叶斯推演：“主力拉升抢筹”剧本
        - 探针植入: 打印本剧本所依赖的先验概率和计算出的似然度，以诊断后验概率为零的原因。
        - 【修正】更新 `urgency_evidence` 的信号名称，从 `PROCESS_META_MAIN_FORCE_URGENCY` 更改为 `PROCESS_META_MAIN_FORCE_RALLY_INTENT`。
        """
        print("    -- [剧本推演] 主力拉升抢筹 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_rising_evidence = self._forge_dynamic_evidence(price_change_bipolar.clip(lower=0))
        efficiency_evidence = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55))
        # 更新信号名称
        rally_intent_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0).clip(lower=0))
        conviction_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION', 0.0).clip(lower=0))
        process_evidence = (rally_intent_evidence * conviction_evidence).pow(0.5) # 使用新的拉升意图信号
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
        【V3.5 · 纯粹原子与WW1模式版】贝叶斯推演：“恐慌投降反转”剧本
        - 核心修复: 将证据 'SCORE_MICRO_BULLISH_RESONANCE' 修正为 'PROCESS_META_MICRO_BEHAVIOR_BOTTOM_REVERSAL'，
                      以符合新的信号生成职责。
        - 【新增】引入“WW1”模式作为恐慌投降反转的证据。
        """
        print("    -- [剧本推演] 恐慌投降反转 (动态证据)...")
        upward_pressure = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(lower=0))
        price_rebound_evidence = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('dip_absorption_power_D'), self.strategy.df_indicators.index, 55))
        vol_ma55 = self._get_atomic_score('VOL_MA_55_D', 1.0)
        volume_spike = (self._get_atomic_score('volume_D') / vol_ma55.replace(0, 1.0)).fillna(1.0)
        volume_evidence = self._forge_dynamic_evidence(normalize_score(volume_spike, self.strategy.df_indicators.index, 55))
        process_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_LOSER_CAPITULATION', 0.0).clip(lower=0))
        micro_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_MICRO_BEHAVIOR_BOTTOM_REVERSAL', 0.0).clip(lower=0))
        ww1_mode = self._forge_dynamic_evidence(self._get_atomic_score('IS_WW1_D', 0.0).astype(float))
        evidence_scores = np.stack([
            upward_pressure.values, price_rebound_evidence.values, volume_evidence.values,
            process_evidence.values, micro_evidence.values, ww1_mode.values # 修改行
        ], axis=0)
        evidence_weights = np.array([0.1, 0.2, 0.15, 0.25, 0.15, 0.15]) # 调整权重
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL': posterior_prob.astype(np.float32)}

    def _deduce_leading_dragon_awakening(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 军备换装与霸占模式版】贝叶斯推演：“龙头苏醒”剧本
        - 核心修复: 将证据 'relative_strength_vs_index_D' 替换为更精准的 'industry_strength_rank_D'。
        - 【新增】引入“霸占”模式作为龙头苏醒的证据。
        """
        print("    -- [剧本推演] 龙头苏醒 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        breakout_quality = self._forge_dynamic_evidence(self._get_atomic_score('breakout_quality_score_D', 0.0))
        sector_sync = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_STOCK_SECTOR_SYNC', 0.0).clip(lower=0))
        relative_strength = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('industry_strength_rank_D', 0.5), self.strategy.df_indicators.index, 55))
        bazhan_mode = self._forge_dynamic_evidence(self._get_atomic_score('IS_BAZHAN_D', 0.0).astype(float))
        evidence_scores = np.stack([capital_confrontation.values, breakout_quality.values, sector_sync.values, relative_strength.values, bazhan_mode.values], axis=0) # 修改行
        evidence_weights = np.array([0.25, 0.25, 0.15, 0.15, 0.2]) # 调整权重
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
        trend_exhaustion_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_TREND_EXHAUSTION', 0.0)) # 修改行
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
        【V1.6 · 能量压缩爆发增强版 - 量能萎缩信号升级】贝叶斯推演：“能量压缩爆发”剧本
        - 核心升级: 将 `volume_atrophy` 证据替换为更精确的 `SCORE_BEHAVIOR_VOLUME_ATROPHY` 信号。
        - 【增强】引入价格变化率和成交量爆发作为“爆发”的直接证据，并调整证据权重，使其在爆发当天能更积极地反映剧本。
        - 【修正】在涨停日，对“压缩”证据（波动率压缩、成交量萎缩）更侧重其“状态”而非“动态”，并对最终似然度进行额外加成。
        """
        print("    -- [剧本推演] 能量压缩爆发 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        is_limit_up_day = self.strategy.df_indicators.apply(lambda row: is_limit_up(row), axis=1)
        # 证据1: 波动率压缩 (BBW收缩)
        bbw = self._get_atomic_score('BBW_21_2.0_D', 0.1)
        volatility_compression_raw_score = normalize_score(1 - bbw, df_index, 144, ascending=True) # 1-BBW，越高越压缩
        volatility_compression = self._forge_dynamic_evidence(1 - normalize_score(bbw, df_index, 144))
        # 证据2: 成交量萎缩 (使用新的 SCORE_BEHAVIOR_VOLUME_ATROPHY 信号)
        volume_atrophy_raw_score = self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0) # 直接获取新的萎缩信号
        volume_atrophy = self._forge_dynamic_evidence(volume_atrophy_raw_score) # 对新的萎缩信号进行动态锻造
        # 证据3: 有序性 (熵值低)
        entropy_col = next((col for col in self.strategy.df_indicators.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            entropy = self._get_atomic_score(entropy_col, 1.0)
            orderliness_score = self._forge_dynamic_evidence(1 - normalize_score(entropy, df_index, 144))
        else:
            orderliness_score = pd.Series(0.5, index=df_index)
        # 证据4: 价格变化率 (直接爆发证据)
        pct_change_raw = self._get_atomic_score('pct_change_D', 0.0)
        price_burst_evidence = self._forge_dynamic_evidence(pct_change_raw.clip(lower=0), is_bipolar=False) # 只关注上涨爆发
        # 证据5: 成交量爆发 (直接爆发证据)
        volume_burst_raw = self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_BURST', 0.0)
        volume_burst_evidence = self._forge_dynamic_evidence(volume_burst_raw, is_bipolar=False)
        # 涨停日特殊处理：如果当天是涨停，则压缩证据直接取其“状态”分，并给予高权重
        volatility_compression_final = volatility_compression.mask(is_limit_up_day, volatility_compression_raw_score)
        volume_atrophy_final = volume_atrophy.mask(is_limit_up_day, volume_atrophy_raw_score) # 对新的萎缩信号也进行涨停日特殊处理
        evidence_scores = np.stack([
            volatility_compression_final.values, # 使用处理后的压缩证据
            volume_atrophy_final.values,         # 使用处理后的萎缩证据
            orderliness_score.values,
            price_burst_evidence.values,
            volume_burst_evidence.values
        ], axis=0)
        # 调整权重，增加爆发证据的权重，并略微降低压缩证据的权重，但确保其基础贡献
        evidence_weights = np.array([0.15, 0.15, 0.15, 0.3, 0.25]) # 调整权重
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 涨停日额外加成：在涨停日，能量压缩爆发的似然度应该更高
        likelihood = likelihood.mask(is_limit_up_day, likelihood + 0.3).clip(0, 1) # 涨停日额外加分
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION': posterior_prob.astype(np.float32)}

    def _forge_dynamic_evidence(self, raw_evidence: pd.Series, is_bipolar: bool = False) -> pd.Series:
        """
        【V1.4 · 动态证据锻造厂 - 零值修正与归一化增强版】
        - 核心职责: 将一个静态的原始证据信号，锻造成一个融合了“状态-速度-加速度”的动态证据。
        - 核心修正:
            1. 如果输入的 `raw_evidence` 全为零（或接近零），则直接返回全零的 `forged_evidence`，
               避免将“无证据”错误地转换为“中性证据 (0.5)”。
            2. 在计算 `bipolar_evidence` 之前，对 `raw_evidence` 进行归一化到 `[0, 1]`，
               确保 `(normalized_raw_evidence * 2 - 1)` 的映射逻辑正确。
            3. 调整 `forged_evidence` 的最终映射，使其在 `dynamic_force` 为 `0` 或负时返回 `0`，
               只有当 `dynamic_force` 为正时才贡献正值。
        - 数学逻辑: DynamicEvidence = w_s * State + w_v * Velocity + w_a * Acceleration
        - 【修正】调整 `forged_evidence` 的最终映射，使其在 `dynamic_force` 为负时，
                  如果 `state_score` 为正，仍能保留 `state_score` 的一部分贡献，避免过度惩罚。
        """
        if not isinstance(raw_evidence, pd.Series) or raw_evidence.empty:
            return pd.Series(0.0, index=self.strategy.df_indicators.index)
        # 修正1: 如果原始证据全为零，则直接返回全零的证据
        # 使用一个小的阈值来判断是否为零，避免浮点数精度问题
        if (raw_evidence.abs() < 1e-9).all():
            return pd.Series(0.0, index=self.strategy.df_indicators.index)
        norm_window = 55 # 默认值，可以从配置中获取
        # 修正2: 在转换为双极性之前，先将 raw_evidence 归一化到 [0, 1]
        # 如果 raw_evidence 已经是双极性，则不需要再次归一化到 [0, 1]
        if not is_bipolar:
            # 假设 raw_evidence 已经是代表“强度”的非负值，或者通过 clip().abs() 转换为非负值
            # normalize_score 将其缩放到 [0, 1]
            normalized_raw_evidence = normalize_score(raw_evidence, self.strategy.df_indicators.index, window=norm_window, ascending=True)
            normalized_raw_evidence = normalized_raw_evidence.fillna(0.0) # 确保没有NaN
            # 将 [0, 1] 映射到 [-1, 1]
            bipolar_evidence = (normalized_raw_evidence * 2 - 1).clip(-1, 1)
        else:
            # 如果输入已经是双极性，则直接使用
            bipolar_evidence = raw_evidence.clip(-1, 1)
        velocity = bipolar_evidence.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        state_score = bipolar_evidence
        velocity_score = normalize_to_bipolar(velocity, self.strategy.df_indicators.index, norm_window)
        acceleration_score = normalize_to_bipolar(acceleration, self.strategy.df_indicators.index, norm_window)
        w_state, w_velocity, w_acceleration = 0.3, 0.4, 0.3
        dynamic_force = (state_score * w_state + velocity_score * w_velocity + acceleration_score * w_acceleration)
        # 修正3: 调整 forged_evidence 的最终映射
        # 我们希望：
        # 1. dynamic_force > 0 时，贡献正向证据强度。
        # 2. dynamic_force <= 0 但 state_score > 0 时，仍能保留 state_score 的一部分贡献，避免过度惩罚。
        # 3. dynamic_force <= 0 且 state_score <= 0 时，贡献为 0。
        forged_evidence = pd.Series(0.0, index=self.strategy.df_indicators.index)
        # 情况1: dynamic_force > 0，直接使用 (dynamic_force + 1) / 2.0 映射到 [0.5, 1]
        positive_dynamic_force_mask = dynamic_force > 0
        forged_evidence[positive_dynamic_force_mask] = (dynamic_force[positive_dynamic_force_mask] + 1) / 2.0
        # 情况2: dynamic_force <= 0 但 state_score > 0，保留 state_score 的一部分贡献
        # 映射 state_score 从 [-1, 1] 到 [0, 1]，并乘以一个衰减因子 (例如 0.3)
        # 这样即使动态力为负，只要状态本身是积极的，也能有基础分
        negative_dynamic_force_positive_state_mask = (dynamic_force <= 0) & (state_score > 0)
        forged_evidence[negative_dynamic_force_positive_state_mask] = (state_score[negative_dynamic_force_positive_state_mask] + 1) / 2.0 * 0.3
        # 确保最终输出在 [0, 1] 范围内
        return forged_evidence.clip(0, 1)

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
        【V1.4 · 风险剧本证据链优化版】贝叶斯推演：“散户狂热主力撤退”风险剧本
        - 核心逻辑: 识别经典的牛市陷阱，散户Fomo情绪高涨，但主力资金却在悄然撤退。
        - 【优化】调整证据权重，降低 `retail_inflow` 的权重，提高 `main_force_outflow` 和 `chip_dispersion` 的权重，
                  以更准确地捕捉主力撤退和筹码分散的核心风险。
        - 【修复】移除之前为强制极小值而添加的 `.mask` 逻辑，现在由 `_forge_dynamic_evidence` 统一处理零值输入。
        """
        print("    -- [剧本推演] 散户狂热主力撤退风险 (动态证据)...")
        # 证据1: 散户资金净流入 (SCORE_FF_AXIOM_CONSENSUS 正向)
        # 风险剧本需要散户狂热（即散户净流入为正）。如果散户净流入为负或零，则该证据强度应为0。
        raw_retail_inflow_score = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        retail_inflow = self._forge_dynamic_evidence(raw_retail_inflow_score.clip(lower=0))
        # 证据2: 主力资金净流出 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION 负向)
        # 风险剧本需要主力撤退（即资本对抗为负）。如果资本对抗为正或零，则该证据强度应为0。
        raw_mf_confrontation_score = self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        main_force_outflow = self._forge_dynamic_evidence(raw_mf_confrontation_score.clip(upper=0).abs())
        # 证据3: 价格上涨 (pct_change_D 正向) - 作为背景条件，权重可以适当降低
        raw_price_rising_score = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_rising = self._forge_dynamic_evidence(raw_price_rising_score.clip(lower=0))
        # 证据4: 筹码分散 (SCORE_CHIP_AXIOM_CONCENTRATION 负向) - 核心风险证据
        # 筹码分散是风险，所以 (1 - 集中度) 越高，证据越强
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        evidence_scores = np.stack([
            retail_inflow.values,
            main_force_outflow.values,
            price_rising.values,
            chip_dispersion.values
        ], axis=0)
        # 权重分配：降低散户流入的权重，提高主力流出和筹码分散的权重
        evidence_weights = np.array([0.15, 0.35, 0.1, 0.4]) # 调整权重
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
        high_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0))
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
        【V1.5 · 深度博弈版证据链优化版】贝叶斯推演：“主力诱多派发”风险剧本
        - 核心逻辑: 识别筹码派发背景下的诱多收割行为。
        - 【增强】引入更多维度证据，包括主力资金流出、赢家信念衰减、零售狂热等，以更全面地捕捉诱多派发本质。
        - 【优化】调整证据权重，降低 `price_rising` 作为风险证据的权重，提高 `main_force_outflow` 和 `chip_dispersion` 等核心派发证据的权重。
        - 【修复】移除之前为强制极小值而添加的 `.mask` 逻辑，现在由 `_forge_dynamic_evidence` 统一处理零值输入。
        """
        print("    -- [剧本推演] 主力诱多派发风险 (动态证据)...")
        # 证据1: 价格上涨 (诱多表象) - 作为背景条件，权重可以适当降低
        price_rising = self._forge_dynamic_evidence(normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21).clip(lower=0))
        # 证据2: 微观欺骗 (伪装派发) - 负向欺骗是风险
        raw_micro_deception_score = self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        micro_deception_bearish = self._forge_dynamic_evidence(raw_micro_deception_score.clip(upper=0).abs())
        # 证据3: 上影线抛压 (真实抛压) - 负向上影线意图是抛压
        raw_upper_shadow_intent_score = self._get_fused_score('FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0)
        upper_shadow_pressure = self._forge_dynamic_evidence(raw_upper_shadow_intent_score.clip(upper=0).abs())
        # 证据4: 筹码分散 (派发核心) - 筹码分散是风险
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        # 证据5: 主力资金净流出 (直接派发) - 资本对抗为负是流出
        raw_mf_confrontation_score = self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        main_force_outflow = self._forge_dynamic_evidence(raw_mf_confrontation_score.clip(upper=0).abs())
        # 证据6: 赚钱卖出 (主力T+0效率负向，即赚钱卖出) - 负向是风险
        raw_profit_vs_flow_score = self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0)
        profit_vs_flow_bearish = self._forge_dynamic_evidence(raw_profit_vs_flow_score.clip(upper=0).abs())
        # 证据7: 赢家信念衰减 (获利盘动摇) - 赢家信念衰减是风险
        winner_conviction_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        # 证据8: 散户狂热主力撤退 (直接捕捉牛市陷阱) - 依赖于另一个风险剧本
        retail_fomo_retreat_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0))
        # 证据9: 价格超买意图负向 (价格高但缺乏真实支撑) - 负向超买意图是风险
        raw_price_overextension_score = self._get_fused_score('FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0)
        price_overextension_risk = self._forge_dynamic_evidence(raw_price_overextension_score.clip(upper=0).abs())
        # 证据10: 长期获利盘派发风险 (更深层次的派发确认) - 依赖于另一个风险剧本
        long_term_profit_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 0.0))
        evidence_scores = np.stack([
            price_rising.values,
            micro_deception_bearish.values,
            upper_shadow_pressure.values,
            chip_dispersion.values,
            main_force_outflow.values,
            profit_vs_flow_bearish.values,
            winner_conviction_decay.values,
            retail_fomo_retreat_risk.values,
            price_overextension_risk.values,
            long_term_profit_distribution_risk.values
        ], axis=0)
        # 权重分配：降低 price_rising 的权重，提高核心派发证据的权重
        evidence_weights = np.array([
            0.03, # price_rising (作为背景条件，权重低，但必须存在)
            0.10, # micro_deception_bearish
            0.10, # upper_shadow_pressure
            0.15, # chip_dispersion
            0.15, # main_force_outflow
            0.07, # profit_vs_flow_bearish
            0.07, # winner_conviction_decay
            0.10, # retail_fomo_retreat_risk
            0.08, # price_overextension_risk
            0.15  # long_term_profit_distribution_risk
        ])
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
        # 证据2: 成交量冷漠 (SCORE_BEHAVIOR_VOLUME_ATROPHY)
        volume_apathy = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
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
        high_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0))
        # 证据2: 结构趋势形态恶化 (SCORE_STRUCT_AXIOM_TREND_FORM 负向)
        structural_trend_deterioration = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0).clip(upper=0).abs())
        # 证据3: 散户狂热主力撤退风险 (COGNITIVE_RISK_RETAIL_FOMO_RETREAT)
        retail_fomo_retreat = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0))
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

    def _deduce_stealth_bottoming_divergence(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0 · 隐秘筑底背离版】贝叶斯推演：“隐秘筑底背离”剧本
        - 核心逻辑: 识别在股价下跌趋势趋缓、成交量萎缩的情况下，资金或筹码出现向好迹象的底部背离。
        - 证据链:
          1. 价格下跌趋势趋缓/底部反转迹象 (行为层价格下跌动能衰减、价格加速度转正、行为底部反转过程信号)
          2. 成交量萎缩 (行为层成交量萎缩信号)
          3. 资金向好迹象 (资金流共识、权力转移过程信号)
          4. 筹码向好迹象 (筹码集中度、隐秘吸筹过程信号、主力成本优势趋势、输家投降过程信号)
        """
        print("    -- [剧本推演] 隐秘筑底背离 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 1. 价格下跌趋势趋缓/底部反转迹象
        # 价格下跌动能衰减 (1 - SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM)
        downward_momentum_decay = self._forge_dynamic_evidence(1 - self._get_atomic_score('SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0))
        # 价格加速度转正 (ACCEL_5_close_D 的正向部分)
        price_accel_positive = self._forge_dynamic_evidence(self._get_atomic_score('ACCEL_5_close_D', 0.0).clip(lower=0))
        # 行为底部反转过程信号
        behavior_bottom_reversal = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_BEHAVIOR_BOTTOM_REVERSAL', 0.0))
        # 2. 成交量萎缩
        # 行为层成交量萎缩信号 (SCORE_BEHAVIOR_VOLUME_ATROPHY)
        volume_atrophy_strong = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        # 3. 资金向好迹象
        # 资金流共识 (SCORE_FF_AXIOM_CONSENSUS 的正向部分)
        fund_flow_consensus_positive = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(lower=0))
        # 权力转移过程信号 (PROCESS_META_POWER_TRANSFER 的正向部分)
        power_transfer_positive = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0))
        # 4. 筹码向好迹象
        # 筹码集中度 (SCORE_CHIP_AXIOM_CONCENTRATION 的正向部分)
        chip_concentration_positive = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        # 隐秘吸筹过程信号 (PROCESS_META_STEALTH_ACCUMULATION)
        stealth_accumulation_process = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_STEALTH_ACCUMULATION', 0.0))
        # 主力成本优势趋势 (PROCESS_META_COST_ADVANTAGE_TREND 的正向部分)
        cost_advantage_trend_positive = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_COST_ADVANTAGE_TREND', 0.0).clip(lower=0))
        # 输家投降过程信号 (PROCESS_META_LOSER_CAPITULATION)
        loser_capitulation_process = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_LOSER_CAPITULATION', 0.0))
        evidence_scores = np.stack([
            downward_momentum_decay.values,
            price_accel_positive.values,
            behavior_bottom_reversal.values,
            volume_atrophy_strong.values,
            fund_flow_consensus_positive.values,
            power_transfer_positive.values,
            chip_concentration_positive.values,
            stealth_accumulation_process.values,
            cost_advantage_trend_positive.values,
            loser_capitulation_process.values
        ], axis=0)
        # 证据权重分配 (需要根据回测优化，这里给出初始示例)
        evidence_weights = np.array([
            0.10, # downward_momentum_decay
            0.10, # price_accel_positive
            0.05, # behavior_bottom_reversal
            0.20, # volume_atrophy_strong (核心证据)
            0.15, # fund_flow_consensus_positive
            0.05, # power_transfer_positive
            0.15, # chip_concentration_positive
            0.10, # stealth_accumulation_process
            0.05, # cost_advantage_trend_positive
            0.05  # loser_capitulation_process
        ])
        evidence_weights /= evidence_weights.sum() # 归一化权重
        safe_scores = np.maximum(evidence_scores, 1e-9) # 避免log(0)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 先验概率：底部背离通常是趋势反转的一种形式
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE': posterior_prob.astype(np.float32)}

    def _deduce_micro_absorption_divergence(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0 · 微观承接背离版】贝叶斯推演：“微观承接背离”剧本
        - 核心逻辑: 识别在价格弱势或横盘、量能萎缩的背景下，微观层面的主动卖压衰竭，同时主动买盘或承接力量增强的底部背离。
        - 证据链:
          1. 价格弱势/稳定：价格下跌动能高或价格变化小。
          2. 量能萎缩：行为层成交量萎缩信号。
          3. 卖压衰竭：对手盘耗尽指数高，主动卖压斜率为负（下降）。
          4. 买盘承接：抄底承接力量高，主动买盘斜率为正（上升）。
        """
        print("    -- [剧本推演] 微观承接背离 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 1. 价格弱势/稳定上下文
        # 价格下跌动能高，表示价格仍在弱势
        price_down_momentum_high = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0))
        # 价格变化小，表示价格趋于稳定
        price_stabilization = self._forge_dynamic_evidence(1 - self._get_atomic_score('pct_change_D', 0.0).abs())
        # 价格弱势或稳定，作为背景条件
        price_weak_or_stable_context = np.maximum(price_down_momentum_high, price_stabilization)
        # 2. 量能萎缩上下文
        volume_atrophy_context = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        # 3. 卖压衰竭证据
        counterparty_exhaustion = self._forge_dynamic_evidence(self._get_atomic_score('counterparty_exhaustion_index_D', 0.0))
        # 主动卖压的5日斜率，负值表示卖压减弱，取绝对值作为证据强度
        selling_pressure_decreasing = self._forge_dynamic_evidence(self._get_atomic_score('SLOPE_5_active_selling_pressure_D', 0.0).clip(upper=0).abs())
        # 4. 买盘承接证据
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        # 主动买盘的5日斜率，正值表示买盘增强
        buying_support_increasing = self._forge_dynamic_evidence(self._get_atomic_score('SLOPE_5_active_buying_support_D', 0.0).clip(lower=0))
        evidence_scores = np.stack([
            price_weak_or_stable_context.values,
            volume_atrophy_context.values,
            counterparty_exhaustion.values,
            selling_pressure_decreasing.values,
            dip_absorption_power.values,
            buying_support_increasing.values
        ], axis=0)
        # 证据权重分配
        evidence_weights = np.array([
            0.10, # price_weak_or_stable_context (背景)
            0.10, # volume_atrophy_context (背景)
            0.25, # counterparty_exhaustion (核心证据)
            0.15, # selling_pressure_decreasing (确认证据)
            0.25, # dip_absorption_power (核心证据)
            0.15  # buying_support_increasing (确认证据)
        ])
        evidence_weights /= evidence_weights.sum() # 归一化权重
        safe_scores = np.maximum(evidence_scores, 1e-9) # 避免log(0)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 先验概率：微观承接背离通常是趋势反转的一种形式
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE': posterior_prob.astype(np.float32)}

