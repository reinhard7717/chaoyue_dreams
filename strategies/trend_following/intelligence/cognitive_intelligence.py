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
        """
        if column_name not in df.columns:
            print(f"    -> [CognitiveIntelligence情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _get_fused_score(self, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.2 · 真理探针版】安全地从原子状态库中获取由融合层提供的态势分数。
        - 核心升级: 植入真理探针。如果获取不到信号，将打印明确的警告信息。
        - 【新增】增加探针输出，打印获取到的融合信号原始值，特别是针对 FUSION_BIPOLAR_CAPITAL_CONFRONTATION。
        """
        if name in self.strategy.atomic_states:
            score = self.strategy.atomic_states[name]
            debug_params = get_params_block(self.strategy, 'debug_params', {})
            probe_dates_str = debug_params.get('probe_dates', [])
            if probe_dates_str and name == 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION':
                probe_date_naive = pd.to_datetime(probe_dates_str[0])
                probe_date_for_loop = probe_date_naive.tz_localize(self.strategy.df_indicators.index.tz) if self.strategy.df_indicators.index.tz else probe_date_naive
                if probe_date_for_loop is not None and probe_date_for_loop in self.strategy.df_indicators.index:
                    if isinstance(score, pd.Series):
                        print(f"    -> [DEBUG _get_fused_score] 获取融合信号 '{name}' 原始值: {score.loc[probe_date_for_loop]:.4f}")
                    else:
                        print(f"    -> [DEBUG _get_fused_score] 获取融合信号 '{name}' 原始值: {score:.4f}")
            return score
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

    def _get_playbook_score(self, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        安全地从 playbook_states 获取剧本信号分数。
        如果信号不存在，则返回默认值，并打印警告。
        """
        score = self.strategy.playbook_states.get(signal_name)
        if score is None:
            print(f"    -> [认知层警告] 剧本信号 '{signal_name}' 不存在，无法作为证据！返回默认值 {default_value}。")
            # 创建一个与 df_indicators 索引对齐的 Series
            return pd.Series(default_value, index=self.strategy.df_indicators.index)
        # 打印实际获取到的值，以便调试
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(self.strategy.df_indicators.index.tz) if self.strategy.df_indicators.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in self.strategy.df_indicators.index:
                if isinstance(score, pd.Series):
                    print(f"    -> [DEBUG _get_playbook_score] 信号 '{signal_name}' 原始值: {score.loc[probe_date_for_loop]:.4f}")
                else:
                    print(f"    -> [DEBUG _get_playbook_score] 信号 '{signal_name}' 原始值: {score:.4f}")
        return score

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V25.9 · 剧本调用顺序优化版】总指挥
        - 核心修复: 调整风险剧本的调用顺序，确保依赖的剧本信号在被使用前已计算。
        """
        print("启动【V25.9 · 剧本调用顺序优化版】认知情报分析...")
        playbook_states = {} # [代码修改] 使用局部变量，不再直接修改 self.strategy.playbook_states
        priors = self._establish_prior_beliefs()
        self.strategy.atomic_states.update(priors)
        # 计算所有机会剧本，并立即更新到 playbook_states
        playbook_states.update(self._deduce_suppressive_accumulation(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_chasing_accumulation(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_capitulation_reversal(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_leading_dragon_awakening(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_sector_rotation_vanguard(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_energy_compression_breakout(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_stealth_bottoming_divergence(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_micro_absorption_divergence(priors)) # [代码修改] 更新局部变量
        # 优先计算所有风险信号，并立即更新到 playbook_states
        # 第一批风险剧本：无内部剧本依赖
        playbook_states.update(self._deduce_distribution_at_high(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_retail_fomo_retreat_risk(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_long_term_profit_distribution_risk(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_market_uncertainty_risk(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_liquidity_trap_risk(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_t0_arbitrage_pressure_risk(priors)) # [代码修改] 更新局部变量
        playbook_states.update(self._deduce_key_support_break_risk(priors)) # [代码修改] 更新局部变量
        # 第二批风险剧本：依赖第一批剧本
        # [代码修改] 在调用依赖剧本之前，将当前已生成的剧本更新到 self.strategy.playbook_states 以供 _get_playbook_score 使用
        self.strategy.playbook_states.update(playbook_states)
        playbook_states.update(self._deduce_trend_exhaustion_risk(priors)) # 依赖 RETAIL_FOMO_RETREAT, LONG_TERM_PROFIT_DISTRIBUTION
        self.strategy.playbook_states.update(playbook_states) # [代码修改] 再次更新
        playbook_states.update(self._deduce_harvest_confirmation_risk(priors)) # 依赖 DISTRIBUTION_AT_HIGH
        self.strategy.playbook_states.update(playbook_states) # [代码修改] 再次更新
        playbook_states.update(self._deduce_bull_trap_distribution_risk(priors)) # 依赖 RETAIL_FOMO_RETREAT, LONG_TERM_PROFIT_DISTRIBUTION
        self.strategy.playbook_states.update(playbook_states) # [代码修改] 再次更新
        playbook_states.update(self._deduce_high_level_structural_collapse_risk(priors)) # 依赖 DISTRIBUTION_AT_HIGH, RETAIL_FOMO_RETREAT
        self.strategy.playbook_states.update(playbook_states) # [代码修改] 再次更新
        # 第三批风险剧本：依赖第二批剧本
        playbook_states.update(self._deduce_divergence_reversal(priors)) # 依赖 TREND_EXHAUSTION
        self.strategy.playbook_states.update(playbook_states) # [代码修改] 最终更新
        print(f"【V25.9 · 剧本调用顺序优化版】分析完成，生成 {len(playbook_states)} 个剧本信号并存入专属状态库。")
        return playbook_states # [代码修改] 返回局部变量

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
        【V3.10 · 风险信号重构与趋势背景调制版】贝叶斯推演：“高位派发”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【新增】引入主力持仓信念强度作为逆向证据，主力信念强则派发风险低。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        - 【增强】提高 `trend_modulator` 的风险削弱能力。
        """
        print("    -- [剧本推演] 高位派发风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        # 提高调制因子从 0.5 到 1.0，使风险削弱更显著
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_distribution_at_high").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength()
        main_force_holding_inverse = self._forge_dynamic_evidence(1 - main_force_holding_strength)
        capital_confrontation_bearish = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        price_overextension_risk = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0).clip(upper=0).abs())
        low_upward_efficiency = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5)).clip(0, 1))
        profit_vs_flow_bearish = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        chip_dispersion_evidence = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        market_contradiction_bearish = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0).clip(upper=0).abs())
        upper_shadow_pressure = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0).clip(upper=0).abs())
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        chip_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_BEARISH_DIVERGENCE', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
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

    def _deduce_trend_exhaustion_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.2 · 深度博弈版证据链优化与趋势背景调制版】贝叶斯推演：“趋势衰竭”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【新增】引入主力持仓信念强度作为逆向证据，主力信念强则趋势衰竭风险低。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] 趋势衰竭风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0) # 降低50%的风险权重
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_trend_exhaustion_risk").shift(1).fillna(False)
        # 主力持仓信念强度 (逆向证据)
        main_force_holding_strength = self._get_main_force_holding_strength()
        main_force_holding_inverse = self._forge_dynamic_evidence(1 - main_force_holding_strength) # 主力信念越强，这个证据越低
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
        retail_fomo_retreat_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        # 3. 筹码的派发与分散
        chip_dispersion_evidence = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        chip_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_BEARISH_DIVERGENCE', 0.0))
        long_term_profit_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 0.0), is_probability=True)
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
        # 8. 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1) # 吸收能力越强，这个值越低，风险越低
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
            new_high_strength_inverse.values,
            dip_absorption_inverse.values,
            main_force_holding_inverse.values
        ], axis=0)
        # 重新分配权重，确保所有权重为正，且总和为1
        evidence_weights = np.array([
            0.07, # price_momentum_divergence
            0.06, # winner_conviction_decay
            0.02, # stagnation_evidence (降低权重，避免涨停日误报)
            0.06, # chip_dispersion_evidence
            0.05, # fund_flow_bearish_divergence
            0.04, # structural_deterioration
            0.07, # capital_retreat_evidence
            0.05, # cyclical_top_risk
            0.02, # price_overextension_risk (降低权重，避免涨停日误报)
            0.03, # upper_shadow_pressure_risk
            0.03, # market_contradiction_bearish
            0.03, # retail_fomo_retreat_risk
            0.02, # chip_bearish_divergence
            0.02, # long_term_profit_distribution_risk
            0.08, # trend_quality_inverse (提高权重，强调趋势健康度对风险的抑制)
            0.08, # new_high_strength_inverse (提高权重，强调新高强度对风险的抑制)
            0.07, # dip_absorption_inverse (下跌吸收能力的反向证据，权重较高)
            0.09  # main_force_holding_inverse (主力持仓信念逆向证据)
        ])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        self.strategy.playbook_states['COGNITIVE_RISK_TREND_EXHAUSTION'] = posterior_prob.astype(np.float32)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [趋势衰竭风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - price_momentum_divergence: {price_momentum_divergence.loc[probe_date_for_loop]:.4f}")
                print(f"       - winner_conviction_decay: {winner_conviction_decay.loc[probe_date_for_loop]:.4f}")
                print(f"       - stagnation_evidence: {stagnation_evidence.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_dispersion_evidence: {chip_dispersion_evidence.loc[probe_date_for_loop]:.4f}")
                print(f"       - fund_flow_bearish_divergence: {fund_flow_bearish_divergence.loc[probe_date_for_loop]:.4f}")
                print(f"       - structural_deterioration: {structural_deterioration.loc[probe_date_for_loop]:.4f}")
                print(f"       - capital_retreat_evidence: {capital_retreat_evidence.loc[probe_date_for_loop]:.4f}")
                print(f"       - cyclical_top_risk: {cyclical_top_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_overextension_risk: {price_overextension_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - upper_shadow_pressure_risk: {upper_shadow_pressure_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - market_contradiction_bearish: {market_contradiction_bearish.loc[probe_date_for_loop]:.4f}")
                print(f"       - retail_fomo_retreat_risk: {retail_fomo_retreat_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_bearish_divergence: {chip_bearish_divergence.loc[probe_date_for_loop]:.4f}")
                print(f"       - long_term_profit_distribution_risk: {long_term_profit_distribution_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_quality_inverse: {trend_quality_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - new_high_strength_inverse: {new_high_strength_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_holding_inverse: {main_force_holding_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_TREND_EXHAUSTION': posterior_prob.astype(np.float32)}

    def _establish_prior_beliefs(self) -> Dict[str, pd.Series]:
        """
        【V1.7 · 结构共识强化版】建立先验信念
        - 核心升级: 将 `SCORE_CHIP_STRUCTURAL_CONSENSUS` 信号融入到“趋势先验概率” (COGNITIVE_PRIOR_TREND_PROB) 的计算中，以提供更稳定、更具结构性的背景判断。
        """
        states = {}
        df_index = self.strategy.df_indicators.index
        market_regime = self._get_fused_score('FUSION_BIPOLAR_MARKET_REGIME', 0.0)
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        trend_structure_score = self._get_fused_score('FUSION_BIPOLAR_TREND_STRUCTURE_SCORE', 0.0)
        fund_flow_trend = self._get_fused_score('FUSION_BIPOLAR_FUND_FLOW_TREND', 0.0)
        chip_trend = self._get_fused_score('FUSION_BIPOLAR_CHIP_TREND', 0.0)
        # 获取结构共识分
        structural_consensus = self._get_atomic_score('SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0)
        # 转换为概率 (0-1范围)
        market_regime_prob = (market_regime + 1) / 2
        trend_quality_prob = (trend_quality + 1) / 2
        trend_structure_prob = (trend_structure_score + 1) / 2
        fund_flow_trend_prob = (fund_flow_trend + 1) / 2
        chip_trend_prob = (chip_trend + 1) / 2
        # 结构共识分本身就是 [0,1] 范围，直接使用
        structural_consensus_prob = structural_consensus
        # 调整趋势先验概率的权重，引入趋势结构分、资金趋势、筹码趋势和结构共识分
        # 示例权重，需要根据回测优化，确保总和为1
        regime_weight = 0.15
        quality_weight = 0.15
        structure_weight = 0.15
        fund_flow_weight = 0.15
        chip_trend_weight = 0.15
        # 结构共识分的权重
        structural_consensus_weight = 0.25
        prior_trend = (
            market_regime_prob * regime_weight +
            trend_quality_prob * quality_weight +
            trend_structure_prob * structure_weight +
            fund_flow_trend_prob * fund_flow_weight +
            chip_trend_prob * chip_trend_weight +
            # 将结构共识分纳入先验信念
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
                print(f"       - 市场政权分 (market_regime): {market_regime.loc[probe_date]:.4f}")
                print(f"       - 趋势质量分 (trend_quality): {trend_quality.loc[probe_date]:.4f}")
                print(f"       - 趋势结构分 (trend_structure_score): {trend_structure_score.loc[probe_date]:.4f}")
                print(f"       - 资金趋势分 (fund_flow_trend): {fund_flow_trend.loc[probe_date]:.4f}")
                print(f"       - 筹码趋势分 (chip_trend): {chip_trend.loc[probe_date]:.4f}")
                # 打印结构共识分
                print(f"       - 结构共识分 (structural_consensus): {structural_consensus.loc[probe_date]:.4f}")
                print(f"       - 市场政权概率 (market_regime_prob): {market_regime_prob.loc[probe_date]:.4f}")
                print(f"       - 趋势质量概率 (trend_quality_prob): {trend_quality_prob.loc[probe_date]:.4f}")
                print(f"       - 趋势结构概率 (trend_structure_prob): {trend_structure_prob.loc[probe_date]:.4f}")
                print(f"       - 资金趋势概率 (fund_flow_trend_prob): {fund_flow_trend_prob.loc[probe_date]:.4f}")
                print(f"       - 筹码趋势概率 (chip_trend_prob): {chip_trend_prob.loc[probe_date]:.4f}")
                # 打印结构共识概率
                print(f"       - 结构共识概率 (structural_consensus_prob): {structural_consensus_prob.loc[probe_date]:.4f}")
                print(f"       - 最终趋势先验概率 (prior_trend): {prior_trend.loc[probe_date]:.4f}")
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
            probe_date = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date in df_index:
                print(f"       - 市场压力分 (market_pressure): {market_pressure.loc[probe_date]:.4f}")
                print(f"       - 市场政权绝对值 (market_regime.abs()): {market_regime.abs().loc[probe_date]:.4f}")
                print(f"       - 趋势确认分 (trend_confirmed): {trend_confirmed.loc[probe_date]:.4f}")
                print(f"       - 抑制因子 (suppression_factor): {suppression_factor.loc[probe_date]:.4f}")
                print(f"       - 最终反转先验概率 (prior_reversal): {prior_reversal.loc[probe_date]:.4f}")
        return states

    def _fuse_and_adjudicate_playbooks(self, playbook_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.5 · 微观承接背离剧本集成版】融合与裁决模块
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
            'COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE',
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
        【V3.6 · 多方炮强化版】贝叶斯推演：“主力拉升抢筹”剧本
        - 核心升级: 引入 `SCORE_CHIP_STRUCTURAL_CONSENSUS` 作为主力拉升抢筹的强有力证据。
        - 探针植入: 打印本剧本所依赖的先验概率和计算出的似然度，以诊断后验概率为零的原因。
        - 【修正】更新 `urgency_evidence` 的信号名称，从 `PROCESS_META_MAIN_FORCE_URGENCY` 更改为 `PROCESS_META_MAIN_FORCE_RALLY_INTENT`。
        - 【新增】引入“回踩确认二次启动”形态和“多方炮”形态作为主力拉升抢筹的证据。
        """
        print("    -- [剧本推演] 主力拉升抢筹 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_rising_evidence = self._forge_dynamic_evidence(price_change_bipolar.clip(lower=0))
        efficiency_evidence = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55))
        rally_intent_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0).clip(lower=0))
        conviction_evidence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION', 0.0).clip(lower=0))
        process_evidence = (rally_intent_evidence * conviction_evidence).pow(0.5)
        chip_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        structural_consensus_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0))
        pullback_confirmation_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_PATTERN_PULLBACK_CONFIRMATION', 0.0))
        # 新增行: 获取多方炮证据
        duofangpao_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_PATTERN_DUOFANGPAO', 0.0))
        evidence_scores = np.stack([
            capital_confrontation.values, price_rising_evidence.values, efficiency_evidence.values,
            process_evidence.values, chip_evidence.values,
            structural_consensus_evidence.values,
            pullback_confirmation_evidence.values,
            # 新增行: 将多方炮证据加入堆栈
            duofangpao_evidence.values
        ], axis=0)
        # 调整权重，为结构共识分、回踩确认二次启动形态和多方炮分配适当权重
        # 确保总和为1
        evidence_weights = np.array([0.12, 0.08, 0.08, 0.18, 0.12, 0.12, 0.15, 0.15]) # 调整权重
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
                print(f"         - capital_confrontation: {capital_confrontation.loc[probe_date]:.4f}")
                print(f"         - price_rising_evidence: {price_rising_evidence.loc[probe_date]:.4f}")
                print(f"         - efficiency_evidence: {efficiency_evidence.loc[probe_date]:.4f}")
                print(f"         - process_evidence: {process_evidence.loc[probe_date]:.4f}")
                print(f"         - chip_evidence: {chip_evidence.loc[probe_date]:.4f}")
                print(f"         - structural_consensus_evidence: {structural_consensus_evidence.loc[probe_date]:.4f}")
                print(f"         - pullback_confirmation_evidence: {pullback_confirmation_evidence.loc[probe_date]:.4f}")
                # 新增行: 打印多方炮证据
                print(f"         - duofangpao_evidence: {duofangpao_evidence.loc[probe_date]:.4f}")
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
        【V1.4 · 结构共识增强版】贝叶斯推演：“龙头苏醒”剧本
        - 核心修复: 将证据 'relative_strength_vs_index_D' 替换为更精准的 'industry_strength_rank_D'。
        - 【新增】引入“霸占”模式作为龙头苏醒的证据。
        - 核心升级: 引入 `SCORE_CHIP_STRUCTURAL_CONSENSUS` 作为龙头苏醒的强有力证据。
        """
        print("    -- [剧本推演] 龙头苏醒 (动态证据)...")
        capital_confrontation = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        breakout_quality = self._forge_dynamic_evidence(self._get_atomic_score('breakout_quality_score_D', 0.0))
        sector_sync = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_STOCK_SECTOR_SYNC', 0.0).clip(lower=0))
        relative_strength = self._forge_dynamic_evidence(normalize_score(self._get_atomic_score('industry_strength_rank_D', 0.5), self.strategy.df_indicators.index, 55))
        bazhan_mode = self._forge_dynamic_evidence(self._get_atomic_score('IS_BAZHAN_D', 0.0).astype(float))
        # 获取结构共识分并锻造成动态证据
        structural_consensus_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0))
        evidence_scores = np.stack([
            capital_confrontation.values, breakout_quality.values, sector_sync.values,
            relative_strength.values, bazhan_mode.values,
            # 将结构共识分作为证据
            structural_consensus_evidence.values
        ], axis=0)
        # 调整权重，为结构共识分分配适当权重
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING': posterior_prob.astype(np.float32)}

    def _deduce_divergence_reversal(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.0 · 深度博弈版证据链优化版】贝叶斯推演：“背离反转”剧本
        - 核心逻辑: 捕捉价格与关键动能、资金、筹码、结构等指标的背离，预示趋势可能反转。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        """
        print("    -- [剧本推演] 背离反转 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 证据1: 价格动能背离 (PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE)
        price_momentum_divergence = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0))
        # 证据2: 资金流背离 (SCORE_FUND_FLOW_BEARISH_DIVERGENCE)
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        # 证据3: 筹码背离 (SCORE_CHIP_BEARISH_DIVERGENCE)
        chip_bearish_divergence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_BEARISH_DIVERGENCE', 0.0))
        # 证据4: 趋势衰竭风险 (COGNITIVE_RISK_TREND_EXHAUSTION)
        trend_exhaustion_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_TREND_EXHAUSTION', 0.0), is_probability=True)
        # 证据5: 市场矛盾 (FUSION_BIPOLAR_MARKET_CONTRADICTION 负向)
        raw_market_contradiction_score = self._get_fused_score('FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0)
        market_contradiction_bearish = self._forge_dynamic_evidence(raw_market_contradiction_score.clip(upper=0).abs())
        # 证据6: 赢家信念衰减 (PROCESS_META_WINNER_CONVICTION_DECAY)
        winner_conviction_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        evidence_scores = np.stack([
            price_momentum_divergence.values,
            fund_flow_bearish_divergence.values,
            chip_bearish_divergence.values,
            trend_exhaustion_risk.values,
            market_contradiction_bearish.values,
            winner_conviction_decay.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [背离反转探针] @ {probe_date_for_loop.date()}:")
                print(f"       - price_momentum_divergence: {price_momentum_divergence.loc[probe_date_for_loop]:.4f}")
                print(f"       - fund_flow_bearish_divergence: {fund_flow_bearish_divergence.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_bearish_divergence: {chip_bearish_divergence.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_exhaustion_risk: {trend_exhaustion_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - market_contradiction_bearish: {market_contradiction_bearish.loc[probe_date_for_loop]:.4f}")
                print(f"       - winner_conviction_decay: {winner_conviction_decay.loc[probe_date_for_loop]:.4f}")
                print(f"       - likelihood: {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
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
        【V1.7 · 能量压缩爆发增强版 - 量能萎缩信号升级】贝叶斯推演：“能量压缩爆发”剧本
        - 核心升级: 将 `volume_atrophy` 证据替换为更精确的 `SCORE_BEHAVIOR_VOLUME_ATROPHY` 信号。
        - 【增强】引入价格变化率和成交量爆发作为“爆发”的直接证据，并调整证据权重，使其在爆发当天能更积极地反映剧本。
        - 【修正】在涨停日，对“压缩”证据（波动率压缩、成交量萎缩）更侧重其“状态”而非“动态”，并对最终似然度进行额外加成。
        - 【修复】修正 `_forge_dynamic_evidence` 方法调用时，将 `is_bipolar` 参数改为 `is_probability`。
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
        price_burst_evidence = self._forge_dynamic_evidence(pct_change_raw.clip(lower=0), is_probability=False) # 只关注上涨爆发
        # 证据5: 成交量爆发 (直接爆发证据)
        volume_burst_raw = self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_BURST', 0.0)
        volume_burst_evidence = self._forge_dynamic_evidence(volume_burst_raw, is_probability=False)
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

    def _forge_dynamic_evidence(self, evidence: pd.Series, is_probability: bool = False) -> pd.Series:
        """
        【V2.0 · 动态证据锻造】
        - 核心逻辑: 对原始证据进行动态处理，确保其有效性并转换为适合贝叶斯推断的格式。
        - 修复: 确保证据值不会为零，避免对数运算错误。
        - 优化: 引入动态归一化，使证据强度更具可比性。
        - 新增: 针对已是概率值的证据，跳过归一化步骤。
        """
        # 确保证据是 Series 类型
        if not isinstance(evidence, pd.Series):
            evidence = pd.Series(evidence, index=self.strategy.df_indicators.index)
        # 填充NaN值
        evidence = evidence.fillna(self.min_evidence_threshold)
        # 确保证据值不会低于最小阈值，避免对数运算错误
        evidence = evidence.mask(evidence < self.min_evidence_threshold, self.min_evidence_threshold)
        # 如果证据已经是概率值，则不进行归一化
        if not is_probability:
            evidence = normalize_score(evidence, self.strategy.df_indicators.index, window=self.norm_window, ascending=True)
        return evidence

    def _deduce_long_term_profit_distribution_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 风险剧本与趋势背景调制版】贝叶斯推演：“长期获利盘派发”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【新增】引入主力持仓信念强度作为逆向证据，主力信念强则派发风险低。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] 长期获利盘派发风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_long_term_profit_distribution_risk").shift(1).fillna(False)
        # 主力持仓信念强度 (逆向证据)
        main_force_holding_strength = self._get_main_force_holding_strength()
        main_force_holding_inverse = self._forge_dynamic_evidence(1 - main_force_holding_strength) # 主力信念越强，这个证据越低
        # 证据1: 长期获利盘比例下降 (total_winner_rate_D 的衰减)
        long_term_profit_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        # 证据2: 筹码集中度下降 (SCORE_CHIP_AXIOM_CONCENTRATION 的负向)
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        # 证据3: 资金流出 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION 的负向)
        capital_outflow = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        # 证据4: 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            long_term_profit_decay.values,
            chip_dispersion.values,
            capital_outflow.values,
            dip_absorption_inverse.values,
            main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [长期获利盘派发风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - long_term_profit_decay: {long_term_profit_decay.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_dispersion: {chip_dispersion.loc[probe_date_for_loop]:.4f}")
                print(f"       - capital_outflow: {capital_outflow.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_holding_inverse: {main_force_holding_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION': posterior_prob.astype(np.float32)}

    def _deduce_market_uncertainty_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 风险剧本与趋势背景调制版】贝叶斯推演：“市场方向不明”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] 市场方向不明风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_market_uncertainty_risk").shift(1).fillna(False)
        # 证据1: 市场政权处于震荡 (FUSION_BIPOLAR_MARKET_REGIME 接近0)
        regime_neutrality = self._forge_dynamic_evidence(1 - self._get_fused_score('FUSION_BIPOLAR_MARKET_REGIME', 0.0).abs())
        # 证据2: 趋势质量低下 (FUSION_BIPOLAR_TREND_QUALITY 接近0)
        low_trend_quality = self._forge_dynamic_evidence(1 - self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0).abs())
        # 证据3: 混沌度高 (SAMPLE_ENTROPY_D)
        entropy_col = next((col for col in self.strategy.df_indicators.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            high_entropy = self._forge_dynamic_evidence(self._get_atomic_score(entropy_col, 0.5)) # 这里不需要 is_probability=True，因为 SAMPLE_ENTROPY_D 不是概率
        else:
            high_entropy = pd.Series(0.5, index=self.strategy.df_indicators.index)
        # 证据4: 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            regime_neutrality.values,
            low_trend_quality.values,
            high_entropy.values,
            dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index)) # 不确定性也可能导致反转
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [市场方向不明风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - regime_neutrality: {regime_neutrality.loc[probe_date_for_loop]:.4f}")
                print(f"       - low_trend_quality: {low_trend_quality.loc[probe_date_for_loop]:.4f}")
                print(f"       - high_entropy: {high_entropy.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_MARKET_UNCERTAINTY': posterior_prob.astype(np.float32)}

    def _deduce_retail_fomo_retreat_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.8 · 风险剧本证据链优化与趋势背景调制版】贝叶斯推演：“散户狂热主力撤退”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【新增】引入主力持仓信念强度作为逆向证据，主力信念强则撤退风险低。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        - 【增强】提高 `trend_modulator` 的风险削弱能力。
        """
        print("    -- [剧本推演] 散户狂热主力撤退风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        # 提高调制因子从 0.5 到 1.0，使风险削弱更显著
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_retail_fomo_retreat_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength()
        main_force_holding_inverse = self._forge_dynamic_evidence(1 - main_force_holding_strength)
        raw_retail_inflow_score = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        retail_inflow = self._forge_dynamic_evidence(raw_retail_inflow_score.clip(lower=0))
        raw_mf_confrontation_score = self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        main_force_outflow = self._forge_dynamic_evidence(raw_mf_confrontation_score.clip(upper=0).abs())
        raw_price_rising_score = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_rising = self._forge_dynamic_evidence(raw_price_rising_score.clip(lower=0))
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            retail_inflow.values,
            main_force_outflow.values,
            price_rising.values,
            chip_dispersion.values,
            dip_absorption_inverse.values,
            main_force_holding_inverse.values
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
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [散户狂热主力撤退风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - retail_inflow: {retail_inflow.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_outflow: {main_force_outflow.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_rising: {price_rising.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_dispersion: {chip_dispersion.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_holding_inverse: {main_force_holding_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_RETAIL_FOMO_RETREAT': posterior_prob.astype(np.float32)}

    def _deduce_harvest_confirmation_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.4 · 风险剧本与趋势背景调制版】贝叶斯推演：“收割确认”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【新增】引入主力持仓信念强度作为逆向证据，主力信念强则收割风险低。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] 收割确认风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_harvest_confirmation_risk").shift(1).fillna(False)
        # 主力持仓信念强度 (逆向证据)
        main_force_holding_strength = self._get_main_force_holding_strength()
        main_force_holding_inverse = self._forge_dynamic_evidence(1 - main_force_holding_strength) # 主力信念越强，这个证据越低
        # 证据1: 高位派发风险 (COGNITIVE_RISK_DISTRIBUTION_AT_HIGH)
        high_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0), is_probability=True)
        # 证据2: 主力T+0效率高 (PROCESS_META_PROFIT_VS_FLOW 负向)
        high_t0_efficiency = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        # 证据3: 赢家信念衰减 (PROCESS_META_WINNER_CONVICTION_DECAY)
        winner_conviction_decay = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        # 证据4: 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_distribution_risk.values,
            high_t0_efficiency.values,
            winner_conviction_decay.values,
            dip_absorption_inverse.values,
            main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [收割确认风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - high_distribution_risk: {high_distribution_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - high_t0_efficiency: {high_t0_efficiency.loc[probe_date_for_loop]:.4f}")
                print(f"       - winner_conviction_decay: {winner_conviction_decay.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_holding_inverse: {main_force_holding_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_HARVEST_CONFIRMATION': posterior_prob.astype(np.float32)}

    def _deduce_bull_trap_distribution_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.8 · 深度博弈版证据链优化与趋势背景调制版】贝叶斯推演：“主力诱多派发”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【新增】引入主力持仓信念强度作为逆向证据，主力信念强则派发风险低。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] 主力诱多派发风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_bull_trap_distribution_risk").shift(1).fillna(False)
        # 主力持仓信念强度 (逆向证据)
        main_force_holding_strength = self._get_main_force_holding_strength()
        main_force_holding_inverse = self._forge_dynamic_evidence(1 - main_force_holding_strength) # 主力信念越强，这个证据越低
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
        retail_fomo_retreat_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        # 证据9: 价格超买意图负向 (价格高但缺乏真实支撑) - 负向超买意图是风险
        raw_price_overextension_score = self._get_fused_score('FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0)
        price_overextension_risk = self._forge_dynamic_evidence(raw_price_overextension_score.clip(upper=0).abs())
        # 证据10: 长期获利盘派发风险 (更深层次的派发确认) - 依赖于另一个风险剧本
        long_term_profit_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 0.0), is_probability=True)
        # 证据11: 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
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
            long_term_profit_distribution_risk.values,
            dip_absorption_inverse.values,
            main_force_holding_inverse.values
        ], axis=0)
        # 权重分配：降低 price_rising 的权重，提高核心派发证据的权重
        evidence_weights = np.array([
            0.02, # price_rising (作为背景条件，权重低，但必须存在)
            0.08, # micro_deception_bearish
            0.08, # upper_shadow_pressure
            0.12, # chip_dispersion
            0.12, # main_force_outflow
            0.05, # profit_vs_flow_bearish
            0.05, # winner_conviction_decay
            0.08, # retail_fomo_retreat_risk
            0.07, # price_overextension_risk
            0.10, # long_term_profit_distribution_risk
            0.12, # dip_absorption_inverse (下跌吸收能力的反向证据，权重较高)
            0.11  # main_force_holding_inverse (主力持仓信念逆向证据)
        ])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [主力诱多派发风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - price_rising: {price_rising.loc[probe_date_for_loop]:.4f}")
                print(f"       - micro_deception_bearish: {micro_deception_bearish.loc[probe_date_for_loop]:.4f}")
                print(f"       - upper_shadow_pressure: {upper_shadow_pressure.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_dispersion: {chip_dispersion.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_outflow: {main_force_outflow.loc[probe_date_for_loop]:.4f}")
                print(f"       - profit_vs_flow_bearish: {profit_vs_flow_bearish.loc[probe_date_for_loop]:.4f}")
                print(f"       - winner_conviction_decay: {winner_conviction_decay.loc[probe_date_for_loop]:.4f}")
                print(f"       - retail_fomo_retreat_risk: {retail_fomo_retreat_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_overextension_risk: {price_overextension_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - long_term_profit_distribution_risk: {long_term_profit_distribution_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_holding_inverse: {main_force_holding_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION': posterior_prob.astype(np.float32)}

    def _deduce_liquidity_trap_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 风险剧本与趋势背景调制版】贝叶斯推演：“流动性陷阱”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] 流动性陷阱风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_liquidity_trap_risk").shift(1).fillna(False)
        # 证据1: 资金流出 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION 负向)
        capital_outflow = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        # 证据2: 成交量冷漠 (SCORE_BEHAVIOR_VOLUME_ATROPHY)
        volume_apathy = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        # 证据3: 波动率收缩 (1 - SCORE_FOUNDATION_AXIOM_VOLATILITY)
        volatility_contraction = self._forge_dynamic_evidence(1 - self._get_atomic_score('SCORE_FOUNDATION_AXIOM_VOLATILITY', 0.0).clip(lower=0))
        # 证据4: 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            capital_outflow.values,
            volume_apathy.values,
            volatility_contraction.values,
            dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [流动性陷阱风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - capital_outflow: {capital_outflow.loc[probe_date_for_loop]:.4f}")
                print(f"       - volume_apathy: {volume_apathy.loc[probe_date_for_loop]:.4f}")
                print(f"       - volatility_contraction: {volatility_contraction.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_LIQUIDITY_TRAP': posterior_prob.astype(np.float32)}

    def _deduce_t0_arbitrage_pressure_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 风险剧本与趋势背景调制版】贝叶斯推演：“T+0套利压力”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] T+0套利压力风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_t0_arbitrage_pressure_risk").shift(1).fillna(False)
        # 证据1: 主力T+0效率高 (PROCESS_META_PROFIT_VS_FLOW 负向)
        high_t0_efficiency = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        # 证据2: 资金流出 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION 负向)
        capital_outflow = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        # 证据3: 微观欺骗 (SCORE_MICRO_AXIOM_DECEPTION 负向，即伪装派发)
        micro_deception_bearish = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0).clip(upper=0).abs())
        # 证据4: 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_t0_efficiency.values,
            capital_outflow.values,
            micro_deception_bearish.values,
            dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [T+0套利压力风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - high_t0_efficiency: {high_t0_efficiency.loc[probe_date_for_loop]:.4f}")
                print(f"       - capital_outflow: {capital_outflow.loc[probe_date_for_loop]:.4f}")
                print(f"       - micro_deception_bearish: {micro_deception_bearish.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE': posterior_prob.astype(np.float32)}

    def _deduce_key_support_break_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 风险剧本与趋势背景调制版】贝叶斯推演：“关键支撑破位”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【新增】引入价格与关键均线距离作为逆向证据，价格远离支撑则破位风险低。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] 关键支撑破位风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_key_support_break_risk").shift(1).fillna(False)
        # 证据1: 市场压力大 (FUSION_BIPOLAR_MARKET_PRESSURE 负向)
        downward_pressure = self._forge_dynamic_evidence(self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(upper=0).abs())
        # 证据2: 结构稳定性差 (SCORE_STRUCT_AXIOM_STABILITY 负向)
        low_structural_stability = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_STRUCT_AXIOM_STABILITY', 0.0).clip(upper=0).abs())
        # 证据3: 基础趋势弱 (SCORE_FOUNDATION_AXIOM_TREND 负向)
        weak_foundation_trend = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FOUNDATION_AXIOM_TREND', 0.0).clip(upper=0).abs())
        # 证据4: 恐慌抛售信号 (PROCESS_META_LOSER_CAPITULATION)
        loser_capitulation = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_LOSER_CAPITULATION', 0.0))
        # 证据5: 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        # 证据6: 价格与关键均线距离 (逆向证据，价格远离支撑则破位风险低)
        # 考虑 EMA21 和 EMA55 作为关键支撑
        close_price = self._get_safe_series(self.strategy.df_indicators, 'close_D', method_name="_deduce_key_support_break_risk")
        ema21 = self._get_safe_series(self.strategy.df_indicators, 'EMA_21_D', method_name="_deduce_key_support_break_risk")
        ema55 = self._get_safe_series(self.strategy.df_indicators, 'EMA_55_D', method_name="_deduce_key_support_break_risk")
        # 价格高于均线越多，支撑越强，风险越低
        price_above_ma_score = normalize_score(
            (close_price - ema21).clip(lower=0) + (close_price - ema55).clip(lower=0),
            df_index, window=55, ascending=True
        )
        price_above_ma_inverse = self._forge_dynamic_evidence(1 - price_above_ma_score) # 价格远离支撑，这个证据越低
        evidence_scores = np.stack([
            downward_pressure.values,
            low_structural_stability.values,
            weak_foundation_trend.values,
            loser_capitulation.values,
            dip_absorption_inverse.values,
            price_above_ma_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [关键支撑破位风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - downward_pressure: {downward_pressure.loc[probe_date_for_loop]:.4f}")
                print(f"       - low_structural_stability: {low_structural_stability.loc[probe_date_for_loop]:.4f}")
                print(f"       - weak_foundation_trend: {weak_foundation_trend.loc[probe_date_for_loop]:.4f}")
                print(f"       - loser_capitulation: {loser_capitulation.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_above_ma_inverse: {price_above_ma_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_KEY_SUPPORT_BREAK': posterior_prob.astype(np.float32)}

    def _deduce_high_level_structural_collapse_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.4 · 风险剧本与趋势背景调制版】贝叶斯推演：“高位结构瓦解”风险剧本
        - 核心升级: 引入“趋势背景调制因子”，当趋势质量和结构形态良好时，降低风险证据的权重。
        - 【增强】引入下跌吸收能力作为反向证据，抑制误报。
        - 【优化】对涨停日后的回调，降低风险权重。
        - 【新增】引入主力持仓信念强度作为逆向证据，主力信念强则结构瓦解风险低。
        - 【修复】修正 `_forge_dynamic_evidence` 对已是概率值的信号进行二次归一化的问题。
        - 【优化】将 `trend_modulator` 应用到 `likelihood` 计算中。
        - 【新增】增加探针输出，检查各组成部分的贡献。
        """
        print("    -- [剧本推演] 高位结构瓦解风险 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 趋势背景调制因子
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        # 涨停日后的回调特殊处理
        is_limit_up_yesterday = self._get_safe_series(self.strategy.df_indicators, 'IS_LIMIT_UP_D', False, method_name="_deduce_high_level_structural_collapse_risk").shift(1).fillna(False)
        # 主力持仓信念强度 (逆向证据)
        main_force_holding_strength = self._get_main_force_holding_strength()
        main_force_holding_inverse = self._forge_dynamic_evidence(1 - main_force_holding_strength) # 主力信念越强，这个证据越低
        # 证据1: 高位派发风险 (COGNITIVE_RISK_DISTRIBUTION_AT_HIGH)
        high_distribution_risk = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0), is_probability=True)
        # 证据2: 结构趋势形态恶化 (SCORE_STRUCT_AXIOM_TREND_FORM 负向)
        structural_trend_deterioration = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0).clip(upper=0).abs())
        # 证据3: 散户狂热主力撤退风险 (COGNITIVE_RISK_RETAIL_FOMO_RETREAT)
        retail_fomo_retreat = self._forge_dynamic_evidence(self._get_playbook_score('COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        # 证据4: 筹码分散 (SCORE_CHIP_AXIOM_CONCENTRATION 负向)
        chip_dispersion = self._forge_dynamic_evidence((1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1))
        # 证据5: 下跌吸收能力 (反向证据，吸收能力强则风险低)
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_distribution_risk.values,
            structural_trend_deterioration.values,
            retail_fomo_retreat.values,
            chip_dispersion.values,
            dip_absorption_inverse.values,
            main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        # 将 trend_modulator 应用到 likelihood
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        # 涨停日后的回调，进一步降低风险
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [高位结构瓦解风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - high_distribution_risk: {high_distribution_risk.loc[probe_date_for_loop]:.4f}")
                print(f"       - structural_trend_deterioration: {structural_trend_deterioration.loc[probe_date_for_loop]:.4f}")
                print(f"       - retail_fomo_retreat: {retail_fomo_retreat.loc[probe_date_for_loop]:.4f}")
                print(f"       - chip_dispersion: {chip_dispersion.loc[probe_date_for_loop]:.4f}")
                print(f"       - dip_absorption_inverse: {dip_absorption_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - main_force_holding_inverse: {main_force_holding_inverse.loc[probe_date_for_loop]:.4f}")
                print(f"       - trend_modulator: {trend_modulator.loc[probe_date_for_loop]:.4f}")
                print(f"       - is_limit_up_yesterday: {is_limit_up_yesterday.loc[probe_date_for_loop]}")
                print(f"       - likelihood (modulated): {likelihood.loc[probe_date_for_loop]:.4f}")
                print(f"       - prior_prob: {prior_prob.loc[probe_date_for_loop]:.4f}")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE': posterior_prob.astype(np.float32)}

    def _deduce_stealth_bottoming_divergence(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.1 · 结构共识增强版】贝叶斯推演：“隐秘筑底背离”剧本
        - 核心逻辑: 识别在股价下跌趋势趋缓、成交量萎缩的情况下，资金或筹码出现向好迹象的底部背离。
        - 核心升级: 引入 `SCORE_CHIP_STRUCTURAL_CONSENSUS` 作为隐秘筑底背离的强有力证据。
        - 证据链:
          1. 价格下跌趋势趋缓/底部反转迹象 (行为层价格下跌动能衰减、价格加速度转正、行为底部反转过程信号)
          2. 成交量萎缩 (行为层成交量萎缩信号)
          3. 资金向好迹象 (资金流共识、权力转移过程信号)
          4. 筹码向好迹象 (筹码集中度、隐秘吸筹过程信号、主力成本优势趋势、输家投降过程信号)
          5. 结构共识 (SCORE_CHIP_STRUCTURAL_CONSENSUS)
        """
        print("    -- [剧本推演] 隐秘筑底背离 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 1. 价格下跌趋势趋缓/底部反转迹象
        downward_momentum_decay = self._forge_dynamic_evidence(1 - self._get_atomic_score('SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0))
        price_accel_positive = self._forge_dynamic_evidence(self._get_atomic_score('ACCEL_5_close_D', 0.0).clip(lower=0))
        behavior_bottom_reversal = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_BEHAVIOR_BOTTOM_REVERSAL', 0.0))
        # 2. 成交量萎缩
        volume_atrophy_strong = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        # 3. 资金向好迹象
        fund_flow_consensus_positive = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(lower=0))
        power_transfer_positive = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0))
        # 4. 筹码向好迹象
        chip_concentration_positive = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0))
        stealth_accumulation_process = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_STEALTH_ACCUMULATION', 0.0))
        cost_advantage_trend_positive = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_COST_ADVANTAGE_TREND', 0.0).clip(lower=0))
        loser_capitulation_process = self._forge_dynamic_evidence(self._get_atomic_score('PROCESS_META_LOSER_CAPITULATION', 0.0))
        # 获取结构共识分并锻造成动态证据
        structural_consensus_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0))
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
            loser_capitulation_process.values,
            # 将结构共识分作为证据
            structural_consensus_evidence.values
        ], axis=0)
        # 证据权重分配 (需要根据回测优化，这里给出初始示例)
        evidence_weights = np.array([
            0.08, # downward_momentum_decay
            0.08, # price_accel_positive
            0.04, # behavior_bottom_reversal
            0.15, # volume_atrophy_strong (核心证据)
            0.12, # fund_flow_consensus_positive
            0.04, # power_transfer_positive
            0.12, # chip_concentration_positive
            0.08, # stealth_accumulation_process
            0.04, # cost_advantage_trend_positive
            0.05, # loser_capitulation_process
            # 结构共识分的权重
            0.20
        ])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE': posterior_prob.astype(np.float32)}

    def _deduce_micro_absorption_divergence(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.1 · 结构共识增强版】贝叶斯推演：“微观承接背离”剧本
        - 核心逻辑: 识别在价格弱势或横盘、量能萎缩的背景下，微观层面的主动卖压衰竭，同时主动买盘或承接力量增强的底部背离。
        - 核心升级: 引入 `SCORE_CHIP_STRUCTURAL_CONSENSUS` 作为微观承接背离的强有力证据。
        - 证据链:
          1. 价格弱势/稳定：价格下跌动能高或价格变化小。
          2. 量能萎缩：行为层成交量萎缩信号。
          3. 卖压衰竭：对手盘耗尽指数高，主动卖压斜率为负（下降）。
          4. 买盘承接：抄底承接力量高，主动买盘斜率为正（上升）。
          5. 结构共识 (SCORE_CHIP_STRUCTURAL_CONSENSUS)
        """
        print("    -- [剧本推演] 微观承接背离 (动态证据)...")
        df_index = self.strategy.df_indicators.index
        # 1. 价格弱势/稳定上下文
        price_down_momentum_high = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0))
        price_stabilization = self._forge_dynamic_evidence(1 - self._get_atomic_score('pct_change_D', 0.0).abs())
        price_weak_or_stable_context = np.maximum(price_down_momentum_high, price_stabilization)
        # 2. 量能萎缩上下文
        volume_atrophy_context = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        # 3. 卖压衰竭证据
        counterparty_exhaustion = self._forge_dynamic_evidence(self._get_atomic_score('counterparty_exhaustion_index_D', 0.0))
        selling_pressure_decreasing = self._forge_dynamic_evidence(self._get_atomic_score('SLOPE_5_active_selling_pressure_D', 0.0).clip(upper=0).abs())
        # 4. 买盘承接证据
        dip_absorption_power = self._forge_dynamic_evidence(self._get_atomic_score('dip_absorption_power_D', 0.0))
        buying_support_increasing = self._forge_dynamic_evidence(self._get_atomic_score('SLOPE_5_active_buying_support_D', 0.0).clip(lower=0))
        # 获取结构共识分并锻造成动态证据
        structural_consensus_evidence = self._forge_dynamic_evidence(self._get_atomic_score('SCORE_CHIP_STRUCTURAL_CONSENSUS', 0.0))
        evidence_scores = np.stack([
            price_weak_or_stable_context.values,
            volume_atrophy_context.values,
            counterparty_exhaustion.values,
            selling_pressure_decreasing.values,
            dip_absorption_power.values,
            buying_support_increasing.values,
            # 将结构共识分作为证据
            structural_consensus_evidence.values
        ], axis=0)
        # 证据权重分配
        evidence_weights = np.array([
            0.08, # price_weak_or_stable_context (背景)
            0.08, # volume_atrophy_context (背景)
            0.20, # counterparty_exhaustion (核心证据)
            0.12, # selling_pressure_decreasing (确认证据)
            0.20, # dip_absorption_power (核心证据)
            0.12, # buying_support_increasing (确认证据)
            # 结构共识分的权重
            0.20
        ])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE': posterior_prob.astype(np.float32)}

    def _get_main_force_holding_strength(self) -> pd.Series:
        """
        【V1.0】计算主力持仓信念强度。
        - 核心逻辑: 融合筹码集中度、资金流信念、主力控盘和成本优势趋势，评估主力当前对股票的持有信念。
        - 输出: [0, 1] 的分数，越高代表主力持仓信念越强。
        """
        df_index = self.strategy.df_indicators.index
        # 筹码集中度 (正向部分)
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0)
        # 资金流信念 (正向部分)
        fund_flow_conviction = self._get_atomic_score('SCORE_FF_AXIOM_CONVICTION', 0.0).clip(lower=0)
        # 主力控盘 (正向部分)
        main_force_control = self._get_atomic_score('PROCESS_META_MAIN_FORCE_CONTROL', 0.0).clip(lower=0)
        # 成本优势趋势 (正向部分)
        cost_advantage_trend = self._get_atomic_score('PROCESS_META_COST_ADVANTAGE_TREND', 0.0).clip(lower=0)
        # 融合这些证据，使用加权平均
        # 权重可以根据实际回测效果调整
        components = [chip_concentration, fund_flow_conviction, main_force_control, cost_advantage_trend]
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        # 确保所有分量都是 Series，并且索引对齐
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        main_force_holding_strength = (
            aligned_components[0] * weights[0] +
            aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] +
            aligned_components[3] * weights[3]
        ).clip(0, 1)
        return main_force_holding_strength

