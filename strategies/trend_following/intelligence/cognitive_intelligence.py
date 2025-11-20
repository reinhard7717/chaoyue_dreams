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
        self.min_evidence_threshold = 1e-9 # 最小证据阈值，避免对数运算错误
        self.norm_window = 55 # 统一归一化窗口，可根据需要调整

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        【V27.1 · 返回值修复版】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        if column_name not in df.columns:
            print(f"    -> [CognitiveIntelligence情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index) # [代码修改] 移除了末尾的 .index
        return df[column_name]

    def _get_fused_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.4 · 返回值修复版】安全地从原子状态库中获取由融合层提供的态势分数。
        - 【V1.4 修复】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        if name in self.strategy.atomic_states:
            score = self.strategy.atomic_states[name]
            debug_params = get_params_block(self.strategy, 'debug_params', {})
            probe_dates_str = debug_params.get('probe_dates', [])
            if probe_dates_str and name == 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION':
                probe_date_naive = pd.to_datetime(probe_dates_str[0])
                probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
                if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                    if isinstance(score, pd.Series):
                        print(f"    -> [DEBUG _get_fused_score] 获取融合信号 '{name}' 原始值: {score.loc[probe_date_for_loop]:.4f}")
                    else:
                        print(f"    -> [DEBUG _get_fused_score] 获取融合信号 '{name}' 原始值: {score:.4f}")
            return score
        else:
            print(f"    -> [认知层警告] 融合态势信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=df.index) # [代码修改] 移除了末尾的 .index

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V2.3 · 返回值修复版】安全地从原子状态库或主数据帧中获取信号。
        - 【V2.3 修复】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in df.columns:
            return df[name]
        else:
            print(f"    -> [认知层警告] 原子信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=df.index) # [代码修改] 移除了末尾的 .index

    def _get_playbook_score(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        安全地从 playbook_states 获取剧本信号分数。
        【V27.1 · 返回值修复版】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        score = self.strategy.playbook_states.get(signal_name)
        if score is None:
            print(f"    -> [认知层警告] 剧本信号 '{signal_name}' 不存在，无法作为证据！返回默认值 {default_value}。")
            return pd.Series(default_value, index=df.index) # [代码修改] 移除了末尾的 .index
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                if isinstance(score, pd.Series):
                    print(f"    -> [DEBUG _get_playbook_score] 信号 '{signal_name}' 原始值: {score.loc[probe_date_for_loop]:.4f}")
                else:
                    print(f"    -> [DEBUG _get_playbook_score] 信号 '{signal_name}' 原始值: {score:.4f}")
        return score

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V27.5 · 清理版】总指挥
        - 核心职责: 编排所有认知剧本的推演，并合成最终的剧本信号。
        - 清理: 移除了用于调试的“总指挥探针”相关代码。
        """
        # [代码修改开始] 移除了探针相关的print语句
        playbook_states = {}
        priors = self._establish_prior_beliefs(df)
        self.strategy.atomic_states.update(priors)
        # --- 剧本计算与状态更新 ---
        # 第1批：机会剧本 (通常无内部依赖)
        playbook_states.update(self._deduce_suppressive_accumulation(df, priors))
        playbook_states.update(self._deduce_chasing_accumulation(df, priors))
        playbook_states.update(self._deduce_capitulation_reversal(df, priors))
        playbook_states.update(self._deduce_leading_dragon_awakening(df, priors))
        playbook_states.update(self._deduce_sector_rotation_vanguard(df, priors))
        playbook_states.update(self._deduce_energy_compression_breakout(df, priors))
        playbook_states.update(self._deduce_stealth_bottoming_divergence(df, priors))
        playbook_states.update(self._deduce_micro_absorption_divergence(df, priors))
        # 第2批：无内部依赖的风险剧本
        playbook_states.update(self._deduce_distribution_at_high(df, priors))
        playbook_states.update(self._deduce_retail_fomo_retreat_risk(df, priors))
        playbook_states.update(self._deduce_long_term_profit_distribution_risk(df, priors))
        playbook_states.update(self._deduce_market_uncertainty_risk(df, priors))
        playbook_states.update(self._deduce_liquidity_trap_risk(df, priors))
        playbook_states.update(self._deduce_t0_arbitrage_pressure_risk(df, priors))
        playbook_states.update(self._deduce_key_support_break_risk(df, priors))
        # 在调用依赖剧本之前，将当前已生成的剧本更新到 self.strategy.playbook_states 以供 _get_playbook_score 使用
        self.strategy.playbook_states.update(playbook_states)
        # 第3批：依赖第2批剧本的风险剧本
        playbook_states.update(self._deduce_trend_exhaustion_risk(df, priors))
        self.strategy.playbook_states.update(playbook_states)
        playbook_states.update(self._deduce_harvest_confirmation_risk(df, priors))
        self.strategy.playbook_states.update(playbook_states)
        playbook_states.update(self._deduce_bull_trap_distribution_risk(df, priors))
        self.strategy.playbook_states.update(playbook_states)
        playbook_states.update(self._deduce_high_level_structural_collapse_risk(df, priors))
        self.strategy.playbook_states.update(playbook_states)
        # 第4批：依赖第3批剧本的机会剧本
        playbook_states.update(self._deduce_divergence_reversal(df, priors))
        # [代码修改开始] 移除了整个“总指挥探针”的调试代码块
        return playbook_states

    def _deduce_suppressive_accumulation(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.5 · 清理版】贝叶斯推演：“主力打压吸筹”剧本
        - 清理: 移除了用于调试的“生产线探针”相关代码。
        """
        print("    -- [剧本推演] 主力打压吸筹 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score(df, 'pct_change_D'), df.index, 21)
        price_falling_evidence = self._forge_dynamic_evidence(df, price_change_bipolar.clip(upper=0).abs())
        efficiency_evidence = self._forge_dynamic_evidence(df, normalize_score(self._get_atomic_score(df, 'dip_absorption_power_D'), df.index, 55))
        process_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_STEALTH_ACCUMULATION', 0.0).clip(lower=0))
        chip_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        market_contradiction_bullish = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0).clip(lower=0))
        evidence_scores = np.stack([
            capital_confrontation.values, price_falling_evidence.values, efficiency_evidence.values,
            process_evidence.values, chip_evidence.values,
            market_contradiction_bullish.values
        ], axis=0)
        evidence_weights = np.array([0.2, 0.1, 0.1, 0.2, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # [代码修改开始] 移除了整个“生产线探针”的调试代码块
        return {'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_distribution_at_high(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.11 · 数据帧上下文修复版】贝叶斯推演：“高位派发”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【V3.11 修复】接收并使用 df 参数，确保索引上下文统一。
        """
        print("    -- [剧本推演] 高位派发风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_distribution_at_high").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        capital_confrontation_bearish = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        price_overextension_risk = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0).clip(upper=0).abs())
        low_upward_efficiency = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5)).clip(0, 1))
        profit_vs_flow_bearish = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        chip_dispersion_evidence = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        market_contradiction_bearish = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0).clip(upper=0).abs())
        upper_shadow_pressure = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0).clip(upper=0).abs())
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        chip_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_BEARISH_DIVERGENCE', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            capital_confrontation_bearish.values,
            price_overextension_risk.values,
            low_upward_efficiency.values,
            profit_vs_flow_bearish.values,
            chip_dispersion_evidence.values,
            market_contradiction_bearish.values,
            upper_shadow_pressure.values,
            fund_flow_bearish_divergence.values,
            chip_bearish_divergence.values,
            dip_absorption_inverse.values,
            main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.12, 0.08, 0.08, 0.12, 0.12, 0.08, 0.12, 0.04, 0.04, 0.10, 0.10])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [高位派发风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - capital_confrontation_bearish: {capital_confrontation_bearish.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_overextension_risk: {price_overextension_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - low_upward_efficiency: {low_upward_efficiency.loc[probe_date_for_loop]:.4f}")
                print(f"       - profit_vs_flow_bearish: {profit_vs_flow_bearish.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_dispersion_evidence: {chip_dispersion_evidence.loc[probe_date_for_loop]:.4f}")
                print(f"       - market_contradiction_bearish: {market_contradiction_bearish.loc[probe_date_for_loop]:.4f}")
                print(f"       - upper_shadow_pressure: {upper_shadow_pressure.loc[probe_date_for_loop]:.4f}")
                print(f"       - fund_flow_bearish_divergence: {fund_flow_bearish_divergence.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_bearish_divergence: {chip_bearish_divergence.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_holding_inverse: {main_force_holding_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH': posterior_prob.astype(np.float32)}

    def _deduce_trend_exhaustion_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.4 · 数据帧上下文修复版】贝叶斯推演：“趋势衰竭”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【V3.4 修复】接收并使用 df 参数，确保索引上下文统一。
        """
        print("    -- [剧本推演] 趋势衰竭风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_trend_exhaustion_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        price_momentum_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0).clip(lower=0))
        stagnation_evidence = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5)).clip(0, 1))
        raw_price_overextension_score = self._get_fused_score(df, 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0)
        price_overextension_risk = self._forge_dynamic_evidence(df, raw_price_overextension_score.clip(upper=0).abs())
        winner_conviction_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        raw_capital_confrontation_score = self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        capital_retreat_evidence = self._forge_dynamic_evidence(df, raw_capital_confrontation_score.clip(upper=0).abs())
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        retail_fomo_retreat_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        chip_dispersion_evidence = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        chip_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_BEARISH_DIVERGENCE', 0.0))
        long_term_profit_distribution_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 0.0), is_probability=True)
        raw_structural_trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        structural_deterioration = self._forge_dynamic_evidence(df, raw_structural_trend_form_score.clip(upper=0).abs())
        raw_market_contradiction_score = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0)
        market_contradiction_bearish = self._forge_dynamic_evidence(df, raw_market_contradiction_score.clip(upper=0).abs())
        raw_upper_shadow_intent_score = self._get_fused_score(df, 'FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0)
        upper_shadow_pressure_risk = self._forge_dynamic_evidence(df, raw_upper_shadow_intent_score.clip(upper=0).abs())
        cyclical_top_risk = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'COGNITIVE_RISK_CYCLICAL_TOP', 0.0))
        trend_quality_inverse = self._forge_dynamic_evidence(df, 1 - self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0).clip(lower=0))
        new_high_strength_inverse = self._forge_dynamic_evidence(df, 1 - self._get_atomic_score(df, 'CONTEXT_NEW_HIGH_STRENGTH', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            price_momentum_divergence.values, winner_conviction_decay.values, stagnation_evidence.values,
            chip_dispersion_evidence.values, fund_flow_bearish_divergence.values, structural_deterioration.values,
            capital_retreat_evidence.values, cyclical_top_risk.values, price_overextension_risk.values,
            upper_shadow_pressure_risk.values, market_contradiction_bearish.values, retail_fomo_retreat_risk.values,
            chip_bearish_divergence.values, long_term_profit_distribution_risk.values, trend_quality_inverse.values,
            new_high_strength_inverse.values, dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([
            0.07, 0.06, 0.02, 0.06, 0.05, 0.04, 0.07, 0.05, 0.02, 0.03, 0.03, 0.03, 0.02, 0.02, 0.08, 0.08, 0.07, 0.09
        ])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [趋势衰竭风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_TREND_EXHAUSTION': posterior_prob.astype(np.float32)}

    def _establish_prior_beliefs(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.8 · 数据帧上下文修复版】建立先验信念
        - 核心升级: 将 `SCORE_CHIP_STRUCTURAL_CONSENSUS` 信号融入到“趋势先验概率” (COGNITIVE_PRIOR_TREND_PROB) 的计算中。
        - 【V1.8 修复】接收并使用 df 参数，确保索引上下文统一。
        """
        states = {}
        df_index = df.index
        market_regime = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_REGIME', 0.0)
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        trend_structure_score = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_STRUCTURE_SCORE', 0.0)
        fund_flow_trend = self._get_fused_score(df, 'FUSION_BIPOLAR_FUND_FLOW_TREND', 0.0)
        chip_trend = self._get_fused_score(df, 'FUSION_BIPOLAR_CHIP_TREND', 0.0)
        structural_consensus = self._get_atomic_score(df, 'SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0)
        market_regime_prob = (market_regime + 1) / 2
        trend_quality_prob = (trend_quality + 1) / 2
        trend_structure_prob = (trend_structure_score + 1) / 2
        fund_flow_trend_prob = (fund_flow_trend + 1) / 2
        chip_trend_prob = (chip_trend + 1) / 2
        structural_consensus_prob = structural_consensus
        regime_weight = 0.15
        quality_weight = 0.15
        structure_weight = 0.15
        fund_flow_weight = 0.15
        chip_trend_weight = 0.15
        structural_consensus_weight = 0.25
        prior_trend = (
            market_regime_prob * regime_weight +
            trend_quality_prob * quality_weight +
            trend_structure_prob * structure_weight +
            fund_flow_trend_prob * fund_flow_weight +
            chip_trend_prob * chip_trend_weight +
            structural_consensus_prob * structural_consensus_weight
        ).clip(0, 1)
        states['COGNITIVE_PRIOR_TREND_PROB'] = prior_trend.astype(np.float32)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date in df_index:
                print(f"    -> [先验信念探针] @ {probe_date.date()}:")
                print(f"       - 最终趋势先验概率 (prior_trend): {prior_trend.loc[probe_date]:.4f}")
        market_pressure = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_PRESSURE', 0.0)
        reversal_pressure_weight = 0.6
        reversal_regime_strength_weight = 0.4
        trend_confirmed = self._get_atomic_score(df, 'CONTEXT_TREND_CONFIRMED', 0.0)
        suppression_factor = (1 - trend_confirmed).clip(0, 1)
        prior_reversal_raw = (market_pressure.abs() * reversal_pressure_weight + market_regime.abs() * reversal_regime_strength_weight).clip(0, 1)
        prior_reversal = (prior_reversal_raw * suppression_factor).clip(0, 1)
        states['COGNITIVE_PRIOR_REVERSAL_PROB'] = prior_reversal.astype(np.float32)
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date in df_index:
                print(f"       - 最终反转先验概率 (prior_reversal): {prior_reversal.loc[probe_date]:.4f}")
        return states

    def _fuse_and_adjudicate_playbooks(self, df: pd.DataFrame, playbook_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.6 · 数据帧上下文修复版】融合与裁决模块
        - 核心升级: 将新的认知剧本 `COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE` 集成到看涨剧本列表中。
        - 【V3.6 修复】接收并使用 df 参数，确保索引上下文统一。
        """
        states = {}
        df_index = df.index
        bullish_playbooks = [
            'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION', 'COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION',
            'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL', 'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING',
            'COGNITIVE_PLAYBOOK_SECTOR_ROTATION_VANGUARD', 'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION',
            'COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE', 'COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE',
        ]
        bullish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bullish_playbooks]
        cognitive_bullish_score = np.maximum.reduce([s.values for s in bullish_scores])
        states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df_index, dtype=np.float32)
        bearish_playbooks = [
            'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 'COGNITIVE_RISK_TREND_EXHAUSTION',
            'COGNITIVE_RISK_MARKET_UNCERTAINTY', 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT',
            'COGNITIVE_RISK_HARVEST_CONFIRMATION', 'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION',
            'COGNITIVE_RISK_LIQUIDITY_TRAP', 'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE',
            'COGNITIVE_RISK_KEY_SUPPORT_BREAK', 'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE',
            'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION'
        ]
        bearish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bearish_playbooks]
        cognitive_bearish_score = np.maximum.reduce([s.values for s in bearish_scores])
        states['COGNITIVE_BEARISH_SCORE'] = pd.Series(cognitive_bearish_score, index=df_index, dtype=np.float32)
        return states

    def _deduce_chasing_accumulation(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.8 · 调用修复版】贝叶斯推演：“主力拉升抢筹”剧本
        - 核心升级: 引入 `SCORE_CHIP_STRUCTURAL_CONSENSUS` 作为强有力证据。
        - 【V3.8 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 主力拉升抢筹 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score(df, 'pct_change_D'), df.index, 21)
        price_rising_evidence = self._forge_dynamic_evidence(df, price_change_bipolar.clip(lower=0))
        efficiency_evidence = self._forge_dynamic_evidence(df, normalize_score(self._get_atomic_score(df, 'VPA_EFFICIENCY_D'), df.index, 55))
        rally_intent_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0).clip(lower=0))
        conviction_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION', 0.0).clip(lower=0))
        process_evidence = (rally_intent_evidence * conviction_evidence).pow(0.5)
        chip_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        structural_consensus_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0))
        pullback_confirmation_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_PATTERN_PULLBACK_CONFIRMATION', 0.0))
        duofangpao_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_PATTERN_DUOFANGPAO', 0.0))
        evidence_scores = np.stack([
            capital_confrontation.values, price_rising_evidence.values, efficiency_evidence.values,
            process_evidence.values, chip_evidence.values, structural_consensus_evidence.values,
            pullback_confirmation_evidence.values, duofangpao_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.12, 0.08, 0.08, 0.18, 0.12, 0.12, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_capitulation_reversal(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.7 · 调用修复版】贝叶斯推演：“恐慌投降反转”剧本
        - 核心修复: 修正信号名称。
        - 【V3.7 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 恐慌投降反转 (动态证据)...")
        upward_pressure = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(lower=0))
        price_rebound_evidence = self._forge_dynamic_evidence(df, normalize_score(self._get_atomic_score(df, 'dip_absorption_power_D'), df.index, 55))
        vol_ma55 = self._get_atomic_score(df, 'VOL_MA_55_D', 1.0)
        volume_spike = (self._get_atomic_score(df, 'volume_D') / vol_ma55.replace(0, 1.0)).fillna(1.0)
        volume_evidence = self._forge_dynamic_evidence(df, normalize_score(volume_spike, df.index, 55))
        process_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_LOSER_CAPITULATION', 0.0).clip(lower=0))
        micro_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_MICRO_BEHAVIOR_BOTTOM_REVERSAL', 0.0).clip(lower=0))
        ww1_mode = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'IS_WW1_D', 0.0).astype(float))
        evidence_scores = np.stack([
            upward_pressure.values, price_rebound_evidence.values, volume_evidence.values,
            process_evidence.values, micro_evidence.values, ww1_mode.values
        ], axis=0)
        evidence_weights = np.array([0.1, 0.2, 0.15, 0.25, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL': posterior_prob.astype(np.float32)}

    def _deduce_leading_dragon_awakening(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 调用修复版】贝叶斯推演：“龙头苏醒”剧本
        - 核心修复: 修正信号名称。
        - 【V1.6 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 龙头苏醒 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        breakout_quality = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'breakout_quality_score_D', 0.0))
        sector_sync = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_STOCK_SECTOR_SYNC', 0.0).clip(lower=0))
        relative_strength = self._forge_dynamic_evidence(df, normalize_score(self._get_atomic_score(df, 'industry_strength_rank_D', 0.5), df.index, 55))
        bazhan_mode = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'IS_BAZHAN_D', 0.0).astype(float))
        structural_consensus_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0))
        evidence_scores = np.stack([
            capital_confrontation.values, breakout_quality.values, sector_sync.values,
            relative_strength.values, bazhan_mode.values,
            structural_consensus_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=df.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING': posterior_prob.astype(np.float32)}

    def _deduce_divergence_reversal(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.2 · 调用修复版】贝叶斯推演：“背离反转”剧本
        - 核心逻辑: 捕捉价格与关键指标的背离。
        - 【V2.2 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 背离反转 (动态证据)...")
        df_index = df.index
        price_momentum_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0))
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        chip_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_BEARISH_DIVERGENCE', 0.0))
        trend_exhaustion_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_TREND_EXHAUSTION', 0.0), is_probability=True)
        raw_market_contradiction_score = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0)
        market_contradiction_bearish = self._forge_dynamic_evidence(df, raw_market_contradiction_score.clip(upper=0).abs())
        winner_conviction_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        evidence_scores = np.stack([
            price_momentum_divergence.values, fund_flow_bearish_divergence.values,
            chip_bearish_divergence.values, trend_exhaustion_risk.values,
            market_contradiction_bearish.values, winner_conviction_decay.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_DIVERGENCE_REVERSAL': posterior_prob.astype(np.float32)}

    def _deduce_sector_rotation_vanguard(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.7 · 调用修复版】贝叶斯推演：“板块轮动先锋”剧本
        - 核心修复: 修正信号名称。
        - 【V1.7 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 板块轮动先锋 (动态证据)...")
        sector_flow = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_FUND_FLOW_BOTTOM_REVERSAL', 0.0).clip(lower=0))
        price_position = self._forge_dynamic_evidence(df, 1 - normalize_score(self._get_atomic_score(df, 'BIAS_144_D', 0.0), df.index, 144))
        chip_cleanliness = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_CLEANLINESS', 0.0))
        hot_sector_cooling = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_HOT_SECTOR_COOLING', 0.0))
        evidence_scores = np.stack([sector_flow.values, price_position.values, chip_cleanliness.values, hot_sector_cooling.values], axis=0)
        evidence_weights = np.array([0.4, 0.2, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=df.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_SECTOR_ROTATION_VANGUARD': posterior_prob.astype(np.float32)}

    def _deduce_energy_compression_breakout(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.9 · 调用修复版】贝叶斯推演：“能量压缩爆发”剧本
        - 核心升级: 替换为更精确的信号。
        - 【V1.9 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 能量压缩爆发 (动态证据)...")
        df_index = df.index
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        bbw = self._get_atomic_score(df, 'BBW_21_2.0_D', 0.1)
        volatility_compression_raw_score = normalize_score(1 - bbw, df_index, 144, ascending=True)
        volatility_compression = self._forge_dynamic_evidence(df, 1 - normalize_score(bbw, df_index, 144))
        volume_atrophy_raw_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        volume_atrophy = self._forge_dynamic_evidence(df, volume_atrophy_raw_score)
        entropy_col = next((col for col in df.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            entropy = self._get_atomic_score(df, entropy_col, 1.0)
            orderliness_score = self._forge_dynamic_evidence(df, 1 - normalize_score(entropy, df_index, 144))
        else:
            orderliness_score = pd.Series(0.5, index=df_index)
        pct_change_raw = self._get_atomic_score(df, 'pct_change_D', 0.0)
        price_burst_evidence = self._forge_dynamic_evidence(df, pct_change_raw.clip(lower=0), is_probability=False)
        volume_burst_raw = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_BURST', 0.0)
        volume_burst_evidence = self._forge_dynamic_evidence(df, volume_burst_raw, is_probability=False)
        volatility_compression_final = volatility_compression.mask(is_limit_up_day, volatility_compression_raw_score)
        volume_atrophy_final = volume_atrophy.mask(is_limit_up_day, volume_atrophy_raw_score)
        evidence_scores = np.stack([
            volatility_compression_final.values, volume_atrophy_final.values,
            orderliness_score.values, price_burst_evidence.values, volume_burst_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.15, 0.15, 0.15, 0.3, 0.25])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood.mask(is_limit_up_day, likelihood + 0.3).clip(0, 1)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION': posterior_prob.astype(np.float32)}

    def _forge_dynamic_evidence(self, df: pd.DataFrame, evidence: pd.Series, is_probability: bool = False) -> pd.Series:
        """
        【V2.2 · 返回值修复版】动态证据锻造
        - 【V2.2 修复】修复了方法内部多处错误返回索引而非Series的问题，确保返回值始终是数值型Series。
        """
        if not isinstance(evidence, pd.Series):
            evidence = pd.Series(evidence, index=df.index) # [代码修改] 移除了末尾的 .index
        evidence = evidence.fillna(self.min_evidence_threshold)
        evidence = evidence.mask(evidence < self.min_evidence_threshold, self.min_evidence_threshold)
        if not is_probability:
            evidence = normalize_score(evidence, df.index, window=self.norm_window, ascending=True) # [代码修改] 移除了末尾的 .index
        return evidence

    def _deduce_long_term_profit_distribution_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 调用修复版】贝叶斯推演：“长期获利盘派发”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.5 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 长期获利盘派发风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_long_term_profit_distribution_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        long_term_profit_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        chip_dispersion = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        capital_outflow = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            long_term_profit_decay.values, chip_dispersion.values, capital_outflow.values,
            dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION': posterior_prob.astype(np.float32)}

    def _deduce_market_uncertainty_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 调用修复版】贝叶斯推演：“市场方向不明”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.5 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 市场方向不明风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_market_uncertainty_risk").shift(1).fillna(False)
        regime_neutrality = self._forge_dynamic_evidence(df, 1 - self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_REGIME', 0.0).abs())
        low_trend_quality = self._forge_dynamic_evidence(df, 1 - self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0).abs())
        entropy_col = next((col for col in df.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            high_entropy = self._forge_dynamic_evidence(df, self._get_atomic_score(df, entropy_col, 0.5))
        else:
            high_entropy = pd.Series(0.5, index=df.index)
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            regime_neutrality.values, low_trend_quality.values,
            high_entropy.values, dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_MARKET_UNCERTAINTY': posterior_prob.astype(np.float32)}

    def _deduce_retail_fomo_retreat_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.10 · 调用修复版】贝叶斯推演：“散户狂热主力撤退”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.10 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 散户狂热主力撤退风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_retail_fomo_retreat_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        raw_retail_inflow_score = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0)
        retail_inflow = self._forge_dynamic_evidence(df, raw_retail_inflow_score.clip(lower=0))
        raw_mf_confrontation_score = self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        main_force_outflow = self._forge_dynamic_evidence(df, raw_mf_confrontation_score.clip(upper=0).abs())
        raw_price_rising_score = normalize_to_bipolar(self._get_atomic_score(df, 'pct_change_D'), df.index, 21)
        price_rising = self._forge_dynamic_evidence(df, raw_price_rising_score.clip(lower=0))
        chip_dispersion = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            retail_inflow.values, main_force_outflow.values, price_rising.values,
            chip_dispersion.values, dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.10, 0.25, 0.07, 0.25, 0.18, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_RETAIL_FOMO_RETREAT': posterior_prob.astype(np.float32)}

    def _deduce_harvest_confirmation_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 调用修复版】贝叶斯推演：“收割确认”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.6 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 收割确认风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_harvest_confirmation_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        high_distribution_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0), is_probability=True)
        high_t0_efficiency = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        winner_conviction_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_distribution_risk.values, high_t0_efficiency.values, winner_conviction_decay.values,
            dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_HARVEST_CONFIRMATION': posterior_prob.astype(np.float32)}

    def _deduce_bull_trap_distribution_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.10 · 调用修复版】贝叶斯推演：“主力诱多派发”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.10 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 主力诱多派发风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_bull_trap_distribution_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        price_rising = self._forge_dynamic_evidence(df, normalize_to_bipolar(self._get_atomic_score(df, 'pct_change_D'), df.index, 21).clip(lower=0))
        raw_micro_deception_score = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        micro_deception_bearish = self._forge_dynamic_evidence(df, raw_micro_deception_score.clip(upper=0).abs())
        raw_upper_shadow_intent_score = self._get_fused_score(df, 'FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0)
        upper_shadow_pressure = self._forge_dynamic_evidence(df, raw_upper_shadow_intent_score.clip(upper=0).abs())
        chip_dispersion = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        raw_mf_confrontation_score = self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        main_force_outflow = self._forge_dynamic_evidence(df, raw_mf_confrontation_score.clip(upper=0).abs())
        raw_profit_vs_flow_score = self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0)
        profit_vs_flow_bearish = self._forge_dynamic_evidence(df, raw_profit_vs_flow_score.clip(upper=0).abs())
        winner_conviction_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        retail_fomo_retreat_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        raw_price_overextension_score = self._get_fused_score(df, 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0)
        price_overextension_risk = self._forge_dynamic_evidence(df, raw_price_overextension_score.clip(upper=0).abs())
        long_term_profit_distribution_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 0.0), is_probability=True)
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            price_rising.values, micro_deception_bearish.values, upper_shadow_pressure.values,
            chip_dispersion.values, main_force_outflow.values, profit_vs_flow_bearish.values,
            winner_conviction_decay.values, retail_fomo_retreat_risk.values, price_overextension_risk.values,
            long_term_profit_distribution_risk.values, dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.02, 0.08, 0.08, 0.12, 0.12, 0.05, 0.05, 0.08, 0.07, 0.10, 0.12, 0.11])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION': posterior_prob.astype(np.float32)}

    def _deduce_liquidity_trap_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 调用修复版】贝叶斯推演：“流动性陷阱”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.5 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 流动性陷阱风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_liquidity_trap_risk").shift(1).fillna(False)
        capital_outflow = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        volume_apathy = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        volatility_contraction = self._forge_dynamic_evidence(df, 1 - self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_VOLATILITY', 0.0).clip(lower=0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            capital_outflow.values, volume_apathy.values,
            volatility_contraction.values, dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_LIQUIDITY_TRAP': posterior_prob.astype(np.float32)}

    def _deduce_t0_arbitrage_pressure_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 调用修复版】贝叶斯推演：“T+0套利压力”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.5 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] T+0套利压力风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_t0_arbitrage_pressure_risk").shift(1).fillna(False)
        high_t0_efficiency = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        capital_outflow = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        micro_deception_bearish = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_DECEPTION', 0.0).clip(upper=0).abs())
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_t0_efficiency.values, capital_outflow.values,
            micro_deception_bearish.values, dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE': posterior_prob.astype(np.float32)}

    def _deduce_key_support_break_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 调用修复版】贝叶斯推演：“关键支撑破位”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.5 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 关键支撑破位风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_key_support_break_risk").shift(1).fillna(False)
        downward_pressure = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(upper=0).abs())
        low_structural_stability = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_STABILITY', 0.0).clip(upper=0).abs())
        weak_foundation_trend = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_TREND', 0.0).clip(upper=0).abs())
        loser_capitulation = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_LOSER_CAPITULATION', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        close_price = self._get_safe_series(df, 'close_D', method_name="_deduce_key_support_break_risk")
        ema21 = self._get_safe_series(df, 'EMA_21_D', method_name="_deduce_key_support_break_risk")
        ema55 = self._get_safe_series(df, 'EMA_55_D', method_name="_deduce_key_support_break_risk")
        price_above_ma_score = normalize_score(
            (close_price - ema21).clip(lower=0) + (close_price - ema55).clip(lower=0),
            df_index, window=55, ascending=True
        )
        price_above_ma_inverse = self._forge_dynamic_evidence(df, 1 - price_above_ma_score)
        evidence_scores = np.stack([
            downward_pressure.values, low_structural_stability.values, weak_foundation_trend.values,
            loser_capitulation.values, dip_absorption_inverse.values, price_above_ma_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_KEY_SUPPORT_BREAK': posterior_prob.astype(np.float32)}

    def _deduce_high_level_structural_collapse_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 调用修复版】贝叶斯推演：“高位结构瓦解”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.6 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 高位结构瓦解风险 (动态证据)...")
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_high_level_structural_collapse_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        high_distribution_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0), is_probability=True)
        structural_trend_deterioration = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0).clip(upper=0).abs())
        retail_fomo_retreat = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        chip_dispersion = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_distribution_risk.values, structural_trend_deterioration.values, retail_fomo_retreat.values,
            chip_dispersion.values, dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE': posterior_prob.astype(np.float32)}

    def _deduce_stealth_bottoming_divergence(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 调用修复版】贝叶斯推演：“隐秘筑底背离”剧本
        - 核心逻辑: 识别底部背离信号。
        - 【V1.3 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 隐秘筑底背离 (动态证据)...")
        df_index = df.index
        downward_momentum_decay = self._forge_dynamic_evidence(df, 1 - self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0))
        price_accel_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'ACCEL_5_close_D', 0.0).clip(lower=0))
        behavior_bottom_reversal = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_BEHAVIOR_BOTTOM_REVERSAL', 0.0))
        volume_atrophy_strong = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        fund_flow_consensus_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(lower=0))
        power_transfer_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0))
        chip_concentration_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        stealth_accumulation_process = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_STEALTH_ACCUMULATION', 0.0))
        cost_advantage_trend_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_COST_ADVANTAGE_TREND', 0.0).clip(lower=0))
        loser_capitulation_process = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_LOSER_CAPITULATION', 0.0))
        structural_consensus_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0))
        evidence_scores = np.stack([
            downward_momentum_decay.values, price_accel_positive.values, behavior_bottom_reversal.values,
            volume_atrophy_strong.values, fund_flow_consensus_positive.values, power_transfer_positive.values,
            chip_concentration_positive.values, stealth_accumulation_process.values, cost_advantage_trend_positive.values,
            loser_capitulation_process.values, structural_consensus_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.08, 0.08, 0.04, 0.15, 0.12, 0.04, 0.12, 0.08, 0.04, 0.05, 0.20])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE': posterior_prob.astype(np.float32)}

    def _deduce_micro_absorption_divergence(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 调用修复版】贝叶斯推演：“微观承接背离”剧本
        - 核心逻辑: 识别微观层面的底部背离。
        - 【V1.3 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 微观承接背离 (动态证据)...")
        df_index = df.index
        price_down_momentum_high = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0))
        price_stabilization = self._forge_dynamic_evidence(df, 1 - self._get_atomic_score(df, 'pct_change_D', 0.0).abs())
        price_weak_or_stable_context = np.maximum(price_down_momentum_high, price_stabilization)
        volume_atrophy_context = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        counterparty_exhaustion = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'counterparty_exhaustion_index_D', 0.0))
        selling_pressure_decreasing = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SLOPE_5_active_selling_pressure_D', 0.0).clip(upper=0).abs())
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        buying_support_increasing = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SLOPE_5_active_buying_support_D', 0.0).clip(lower=0))
        structural_consensus_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0))
        evidence_scores = np.stack([
            price_weak_or_stable_context.values, volume_atrophy_context.values, counterparty_exhaustion.values,
            selling_pressure_decreasing.values, dip_absorption_power.values, buying_support_increasing.values,
            structural_consensus_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.08, 0.08, 0.20, 0.12, 0.20, 0.12, 0.20])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE': posterior_prob.astype(np.float32)}

    def _get_main_force_holding_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · 数据帧上下文修复版】计算主力持仓信念强度。
        - 核心逻辑: 融合筹码集中度、资金流信念、主力控盘和成本优势趋势，评估主力当前对股票的持有信念。
        - 【V1.1 修复】接收并使用 df 参数，确保索引上下文统一。
        """
        df_index = df.index
        chip_concentration = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0)
        fund_flow_conviction = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONVICTION', 0.0).clip(lower=0)
        main_force_control = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_CONTROL', 0.0).clip(lower=0)
        cost_advantage_trend = self._get_atomic_score(df, 'PROCESS_META_COST_ADVANTAGE_TREND', 0.0).clip(lower=0)
        components = [chip_concentration, fund_flow_conviction, main_force_control, cost_advantage_trend]
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        main_force_holding_strength = (
            aligned_components[0] * weights[0] +
            aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] +
            aligned_components[3] * weights[3]
        ).clip(0, 1)
        return main_force_holding_strength

