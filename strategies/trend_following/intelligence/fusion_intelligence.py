# 文件: strategies/trend_following/intelligence/fusion_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, get_adaptive_mtf_normalized_score, normalize_score, normalize_to_bipolar

class FusionIntelligence:
    """
    【V3.0 · 战场态势引擎】
    - 核心重构: 遵循“联合情报部”职责，废弃所有旧方法。不再消费原始指标，
                  只消费各原子情报层输出的“公理级”信号。
    - 核心职责: 将各领域情报“冶炼”成四大客观战场态势：
                  1. 市场政权 (Market Regime): 判断趋势市 vs 震荡市。
                  2. 趋势质量 (Trend Quality): 评估趋势的健康度与共识度。
                  3. 市场压力 (Market Pressure): 衡量向上与向下的反转压力。
                  4. 资本对抗 (Capital Confrontation): 洞察主力与散户的博弈格局。
    - 定位: 连接“感知”与“认知”的关键桥梁，为认知层提供决策依据。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [融合情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.4 · 上下文修复版】安全地从原子状态库或主数据帧中获取分数。
        - 【V1.4 修复】接收 df 参数，并使用其索引创建默认 Series，确保上下文一致。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in self.strategy.df_indicators.columns:
            return self.strategy.df_indicators[name]
        else:
            print(f"    -> [融合层-原子信号警告] 预期原子信号 '{name}' 在 atomic_states 和 df_indicators 中均不存在，使用默认值 {default}。")
            return pd.Series(default, index=df.index)

    def run_fusion_diagnostics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.4 · 上下文修复版】融合情报分析总指挥
        - 【V3.4 修复】接收 df 参数并将其传递给所有 _synthesize_* 方法，确保上下文统一。
        """
        print("启动【V3.4 · 上下文修复版】融合情报分析...")
        all_fusion_states = {}
        regime_states = self._synthesize_market_regime(df)
        all_fusion_states.update(regime_states)
        self.strategy.atomic_states.update(regime_states)
        quality_states = self._synthesize_trend_quality(df)
        all_fusion_states.update(quality_states)
        self.strategy.atomic_states.update(quality_states)
        pressure_states = self._synthesize_market_pressure(df)
        all_fusion_states.update(pressure_states)
        self.strategy.atomic_states.update(pressure_states)
        confrontation_states = self._synthesize_capital_confrontation(df)
        all_fusion_states.update(confrontation_states)
        self.strategy.atomic_states.update(confrontation_states)
        contradiction_states = self._synthesize_market_contradiction(df)
        all_fusion_states.update(contradiction_states)
        self.strategy.atomic_states.update(contradiction_states)
        overextension_intent_states = self._synthesize_price_overextension_intent(df)
        all_fusion_states.update(overextension_intent_states)
        self.strategy.atomic_states.update(overextension_intent_states)
        upper_shadow_intent_states = self._synthesize_upper_shadow_intent(df)
        all_fusion_states.update(upper_shadow_intent_states)
        self.strategy.atomic_states.update(upper_shadow_intent_states)
        stagnation_risk_states = self._synthesize_stagnation_risk(df)
        all_fusion_states.update(stagnation_risk_states)
        self.strategy.atomic_states.update(stagnation_risk_states)
        trend_structure_states = self._synthesize_trend_structure_score(df)
        all_fusion_states.update(trend_structure_states)
        self.strategy.atomic_states.update(trend_structure_states)
        fund_flow_trend_states = self._synthesize_fund_flow_trend(df)
        all_fusion_states.update(fund_flow_trend_states)
        self.strategy.atomic_states.update(fund_flow_trend_states)
        chip_trend_states = self._synthesize_chip_trend(df)
        all_fusion_states.update(chip_trend_states)
        self.strategy.atomic_states.update(chip_trend_states)
        chip_potential_states = self._synthesize_chip_structural_potential(df)
        all_fusion_states.update(chip_potential_states)
        self.strategy.atomic_states.update(chip_potential_states)
        accumulation_inflection_states = self._synthesize_accumulation_inflection(df)
        all_fusion_states.update(accumulation_inflection_states)
        self.strategy.atomic_states.update(accumulation_inflection_states)
        print(f"【V3.4 · 上下文修复版】分析完成，生成 {len(all_fusion_states)} 个融合态势信号。")
        return all_fusion_states

    def _synthesize_market_contradiction(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 上下文修复版】冶炼“市场矛盾” (Market Contradiction)
        - 【V1.1 修复】接收 df 参数并在调用 _get_atomic_score 时传递。
        """
        print("  -- [融合层] 正在冶炼“市场矛盾”...")
        states = {}
        divergence_sources = [
            'FOUNDATION', 'STRUCTURE', 'PATTERN', 'DYNAMIC_MECHANICS',
            'CHIP', 'FUND_FLOW', 'MICRO_BEHAVIOR', 'BEHAVIOR'
        ]
        bullish_divergence_scores = []
        bearish_divergence_scores = []
        for source in divergence_sources:
            bull_signal_name = f'SCORE_{source}_BULLISH_DIVERGENCE'
            bear_signal_name = f'SCORE_{source}_BEARISH_DIVERGENCE'
            bullish_divergence_scores.append(self._get_atomic_score(df, bull_signal_name, 0.0).values)
            bearish_divergence_scores.append(self._get_atomic_score(df, bear_signal_name, 0.0).values)
        net_bullish_divergence = np.maximum.reduce(bullish_divergence_scores)
        net_bearish_divergence = np.maximum.reduce(bearish_divergence_scores)
        bipolar_contradiction = (pd.Series(net_bullish_divergence, index=df.index) -
                                 pd.Series(net_bearish_divergence, index=df.index)).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_CONTRADICTION'] = bipolar_contradiction.astype(np.float32)
        print(f"  -- [融合层] “市场矛盾”冶炼完成，最新分值: {bipolar_contradiction.iloc[-1]:.4f}")
        return states

    def _synthesize_market_regime(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 上下文修复版】冶炼“市场政权” (Market Regime)
        - 【V1.2 修复】接收 df 参数并在调用 _get_atomic_score 时传递。
        """
        print("  -- [融合层] 正在冶炼“市场政权”...")
        states = {}
        hurst_memory = self._get_atomic_score(df, 'SCORE_CYCLICAL_HURST_MEMORY', 0.0)
        inertia = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_INERTIA', 0.0)
        stability = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        trend_evidence_weights = {'hurst': 0.4, 'inertia': 0.4, 'stability': 0.2}
        trend_evidence = (
            hurst_memory * trend_evidence_weights['hurst'] +
            inertia * trend_evidence_weights['inertia'] +
            stability * trend_evidence_weights['stability']
        ).clip(-1, 1)
        reversion_evidence = (hurst_memory.clip(upper=0).abs() * inertia.clip(upper=0).abs()).pow(0.5)
        bipolar_regime = (trend_evidence - reversion_evidence).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_REGIME'] = bipolar_regime.astype(np.float32)
        return states

    def _synthesize_market_pressure(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.3 · 上下文修复版】冶炼“市场压力” (Market Pressure)
        - 【V1.3 修复】接收 df 参数并在调用 _get_atomic_score 时传递，并使用 df.index 创建 Series。
        """
        print("  -- [融合层] 正在冶炼“市场压力”...")
        states = {}
        df_index = df.index
        reversal_sources = [
            'PROCESS_META_FOUNDATION_BOTTOM_REVERSAL', 'PROCESS_META_FOUNDATION_TOP_REVERSAL',
            'PROCESS_META_STRUCTURE_BOTTOM_REVERSAL', 'PROCESS_META_STRUCTURE_TOP_REVERSAL',
            'PROCESS_META_PATTERN_BOTTOM_REVERSAL', 'PROCESS_META_PATTERN_TOP_REVERSAL',
            'PROCESS_META_DYNAMIC_MECHANICS_BOTTOM_REVERSAL', 'PROCESS_META_DYNAMIC_MECHANICS_TOP_REVERSAL',
            'PROCESS_META_CHIP_BOTTOM_REVERSAL', 'PROCESS_META_CHIP_TOP_REVERSAL',
            'PROCESS_META_FUND_FLOW_BOTTOM_REVERSAL', 'PROCESS_META_FUND_FLOW_TOP_REVERSAL',
            'PROCESS_META_MICRO_BEHAVIOR_BOTTOM_REVERSAL', 'PROCESS_META_MICRO_BEHAVIOR_TOP_REVERSAL',
            'PROCESS_META_BEHAVIOR_BOTTOM_REVERSAL', 'PROCESS_META_BEHAVIOR_TOP_REVERSAL'
        ]
        upward_pressure_scores = []
        downward_pressure_scores = []
        for signal_name in reversal_sources:
            if 'BOTTOM_REVERSAL' in signal_name:
                upward_pressure_scores.append(self._get_atomic_score(df, signal_name, 0.0).values)
            elif 'TOP_REVERSAL' in signal_name:
                downward_pressure_scores.append(self._get_atomic_score(df, signal_name, 0.0).values)
        structural_bottom_fractal = self._get_atomic_score(df, 'SCORE_STRUCT_BOTTOM_FRACTAL', 0.0)
        upward_pressure_scores.append(structural_bottom_fractal.values)
        net_upward_pressure = np.maximum.reduce(upward_pressure_scores)
        net_downward_pressure = np.maximum.reduce(downward_pressure_scores)
        bipolar_pressure = (pd.Series(net_upward_pressure, index=df_index) -
                            pd.Series(net_downward_pressure, index=df_index)).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_PRESSURE'] = bipolar_pressure.astype(np.float32)
        print(f"  -- [融合层] “市场压力”冶炼完成，最新分值: {bipolar_pressure.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_quality(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 上下文修复版】冶炼“趋势质量” (Trend Quality)
        - 【V2.1 修复】接收 df 参数并在调用 _get_atomic_score 和 _get_safe_series 时传递。
        """
        print("  -- [融合层] 正在冶炼“趋势质量”...")
        states = {}
        df_index = df.index
        foundation_trend = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_TREND', 0.0)
        foundation_oscillator = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_OSCILLATOR', 0.0)
        foundation_flow = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_FLOW', 0.0)
        foundation_volatility = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_VOLATILITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        structural_mtf_cohesion = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_MTF_COHESION', 0.0)
        structural_stability = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_STABILITY', 0.0)
        dynamic_momentum = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_MOMENTUM', 0.0)
        dynamic_inertia = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_INERTIA', 0.0)
        dynamic_stability = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        dynamic_energy = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_ENERGY', 0.0)
        dynamic_ma_acceleration = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_MA_ACCELERATION', 0.0)
        fund_flow_consensus = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0)
        fund_flow_conviction = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONVICTION', 0.0)
        fund_flow_increment = self._get_atomic_score(df, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0)
        chip_concentration = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        chip_cost_structure = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_COST_STRUCTURE', 0.0)
        chip_holder_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        chip_peak_integrity = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_PEAK_INTEGRITY', 0.0)
        micro_deception = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        micro_probe = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_PROBE', 0.0)
        micro_efficiency = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_EFFICIENCY', 0.0)
        behavior_upward_efficiency = self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.0)
        behavior_downward_resistance = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE', 0.0)
        behavior_intraday_bull_control = self._get_atomic_score(df, 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', 0.0)
        pattern_reversal = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_REVERSAL', 0.0)
        pattern_breakout = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_BREAKOUT', 0.0)
        main_force_on_peak_flow = self._get_atomic_score(df, 'main_force_on_peak_flow_D', 0.0)
        aaa_raw = self._get_safe_series(df, 'AAA_D', 0.0, method_name="_synthesize_trend_quality")
        aaa_score = normalize_to_bipolar(aaa_raw * -1, df_index, window=55)
        pdi_raw = self._get_safe_series(df, 'PDI_14_D', 0.0, method_name="_synthesize_trend_quality")
        pdi_score = normalize_to_bipolar(pdi_raw, df_index, window=55)
        structural_bottom_fractal = self._get_atomic_score(df, 'SCORE_STRUCT_BOTTOM_FRACTAL', 0.0)
        breakout_readiness_raw = self._get_safe_series(df, 'breakout_readiness_score_D', 0.0, method_name="_synthesize_trend_quality")
        breakout_readiness_score = normalize_to_bipolar(breakout_readiness_raw, df_index, window=55, sensitivity=20)
        trend_vitality_raw = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name="_synthesize_trend_quality")
        trend_vitality_score = normalize_to_bipolar(trend_vitality_raw, df_index, window=55, sensitivity=0.5)
        components_and_weights = {
            'foundation_trend': (foundation_trend, 0.08), 'foundation_oscillator': (foundation_oscillator, -0.02),
            'foundation_flow': (foundation_flow, 0.03), 'foundation_volatility': (foundation_volatility, 0.02),
            'structural_trend_form': (structural_trend_form, 0.10), 'structural_mtf_cohesion': (structural_mtf_cohesion, 0.05),
            'structural_stability': (structural_stability, 0.05), 'dynamic_momentum': (dynamic_momentum, 0.08),
            'dynamic_inertia': (dynamic_inertia, 0.05), 'dynamic_stability': (dynamic_stability, 0.02),
            'dynamic_energy': (dynamic_energy, 0.02), 'dynamic_ma_acceleration': (dynamic_ma_acceleration, 0.03),
            'fund_flow_consensus': (fund_flow_consensus, 0.03), 'fund_flow_conviction': (fund_flow_conviction, 0.03),
            'fund_flow_increment': (fund_flow_increment, 0.03), 'chip_concentration': (chip_concentration, 0.03),
            'chip_cost_structure': (chip_cost_structure, 0.03), 'chip_holder_sentiment': (chip_holder_sentiment, 0.03),
            'chip_peak_integrity': (chip_peak_integrity, 0.03), 'micro_deception': (micro_deception, 0.01),
            'micro_probe': (micro_probe, 0.01), 'micro_efficiency': (micro_efficiency, 0.01),
            'behavior_upward_efficiency': (behavior_upward_efficiency, 0.02), 'behavior_downward_resistance': (behavior_downward_resistance, 0.02),
            'behavior_intraday_bull_control': (behavior_intraday_bull_control, 0.01), 'pattern_reversal': (pattern_reversal, 0.01),
            'pattern_breakout': (pattern_breakout, 0.02), 'main_force_on_peak_flow': (main_force_on_peak_flow, 0.01),
            'aaa_score': (aaa_score, 0.02), 'pdi_score': (pdi_score, 0.03),
            'structural_bottom_fractal': (structural_bottom_fractal, 0.02), 'breakout_readiness_score': (breakout_readiness_score, 0.10),
            'trend_vitality_score': (trend_vitality_score, 0.10)
        }
        bipolar_quality = pd.Series(0.0, index=df_index)
        total_weight = sum(w for _, w in components_and_weights.values())
        for name, (series, weight) in components_and_weights.items():
            if not series.empty:
                contribution = series * (weight / total_weight)
                bipolar_quality += contribution
        bipolar_quality = bipolar_quality.clip(-1, 1)
        states['FUSION_BIPOLAR_TREND_QUALITY'] = bipolar_quality.astype(np.float32)
        print(f"  -- [融合层] “趋势质量”冶炼完成，最新分值: {bipolar_quality.iloc[-1]:.4f}")
        return states

    def _synthesize_stagnation_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 上下文修复版】冶炼“滞涨风险” (FUSION_RISK_STAGNATION)
        - 【V1.1 修复】接收 df 参数并在调用 _get_atomic_score 和 _get_safe_series 时传递。
        """
        print("  -- [融合层] 正在冶炼“滞涨风险”...")
        states = {}
        df_index = df.index
        stagnation_evidence_raw = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0)
        price_overextension_risk = self._get_atomic_score(df, 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0).clip(upper=0).abs()
        upper_shadow_pressure_risk = self._get_atomic_score(df, 'FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0).clip(upper=0).abs()
        fund_flow_bearish_risk = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(upper=0).abs()
        chip_dispersion_risk = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(upper=0).abs()
        profit_taking_supply_risk = normalize_score(self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_synthesize_stagnation_risk"), df_index, window=55, ascending=True).clip(0, 1)
        trend_confirmation_risk = (1 - self._get_atomic_score(df, 'CONTEXT_TREND_CONFIRMED', 0.0)).clip(0, 1)
        retail_fomo_risk = normalize_score(self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name="_synthesize_stagnation_risk"), df_index, window=55, ascending=True).clip(0, 1)
        is_price_stagnant_or_rising = (self._get_safe_series(df, 'pct_change_D', method_name="_synthesize_stagnation_risk") >= -0.005).astype(float)
        risk_components = [
            stagnation_evidence_raw, price_overextension_risk, upper_shadow_pressure_risk,
            fund_flow_bearish_risk, chip_dispersion_risk, profit_taking_supply_risk,
            trend_confirmation_risk, retail_fomo_risk
        ]
        weights = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05, 0.10])
        aligned_risk_components = [comp.reindex(df_index, fill_value=0.0) for comp in risk_components]
        safe_risk_components = [comp + 1e-9 for comp in aligned_risk_components]
        stagnation_risk_score = pd.Series(np.prod([comp.values ** w for comp, w in zip(safe_risk_components, weights)], axis=0), index=df_index)
        final_stagnation_risk = (stagnation_risk_score * is_price_stagnant_or_rising).clip(0, 1)
        states['FUSION_RISK_STAGNATION'] = final_stagnation_risk.astype(np.float32)
        print(f"  -- [融合层] “滞涨风险”冶炼完成，最新分值: {final_stagnation_risk.iloc[-1]:.4f}")
        return states

    def _synthesize_capital_confrontation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.3 · 上下文修复版】冶炼“资本对抗” (Capital Confrontation)
        - 【V1.3 修复】接收 df 参数并在调用 _get_atomic_score 时传递。
        """
        print("  -- [融合层] 正在冶炼“资本对抗”...")
        states = {}
        flow_confrontation = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0)
        chip_transfer = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        deception = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        bipolar_confrontation = (flow_confrontation * 0.5 + chip_transfer * 0.3 + deception * 0.2).clip(-1, 1)
        states['FUSION_BIPOLAR_CAPITAL_CONFRONTATION'] = bipolar_confrontation.astype(np.float32)
        print(f"  -- [融合层] “资本对抗”冶炼完成，最新分值: {bipolar_confrontation.iloc[-1]:.4f}")
        return states

    def _synthesize_price_overextension_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.1 · 上下文修复版】冶炼“价格超买意图” (Price Overextension Intent)
        - 【V3.1 修复】接收 df 参数并在调用 _get_atomic_score 和 _get_safe_series 时传递。
        """
        print("  -- [融合层] 正在冶炼“价格超买意图”...")
        states = {}
        df_index = df.index
        norm_window = 55
        overextension_raw_bipolar = (self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.5) * 2 - 1).clip(-1, 1)
        bias_raw = self._get_safe_series(df, 'BIAS_21_D', pd.Series(0.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        bias_score = normalize_to_bipolar(bias_raw, df_index, window=norm_window, sensitivity=0.05)
        winner_rate_raw = self._get_safe_series(df, 'total_winner_rate_D', pd.Series(0.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        winner_rate_score = normalize_to_bipolar(winner_rate_raw, df_index, window=norm_window, sensitivity=0.1, default_value=-1.0)
        rsi_raw = self._get_safe_series(df, 'RSI_13_D', pd.Series(50.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        rsi_score = normalize_to_bipolar(rsi_raw, df_index, window=norm_window, sensitivity=10.0)
        core_overextension_sum = (
            overextension_raw_bipolar * 0.2 + bias_score * 0.2 +
            rsi_score * 0.15 + winner_rate_score * 0.15
        )
        fund_flow_consensus = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0)
        chip_concentration = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        micro_efficiency = (self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_EFFICIENCY', 0.5) * 2 - 1).clip(-1, 1)
        body_ratio_raw = self._get_safe_series(df, 'closing_price_deviation_score_D', pd.Series(0.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        body_score = normalize_to_bipolar(body_ratio_raw, df_index, window=norm_window, sensitivity=0.2)
        upper_shadow_ratio_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', pd.Series(0.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        upper_shadow_score = normalize_to_bipolar(upper_shadow_ratio_raw, df_index, window=norm_window, sensitivity=0.2) * -1
        health_sum = (
            fund_flow_consensus * 0.1 + chip_concentration * 0.05 +
            structural_trend_form * 0.05 + micro_efficiency * 0.03 +
            body_score * 0.04 + upper_shadow_score * 0.03
        )
        final_overextension_intent = (core_overextension_sum - health_sum).clip(-1, 1)
        states['FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT'] = final_overextension_intent.astype(np.float32)
        print(f"  -- [融合层] “价格超买意图”冶炼完成，最新分值: {final_overextension_intent.iloc[-1]:.4f}")
        return states

    def _synthesize_upper_shadow_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 上下文修复版】冶炼“上影线意图” (Upper Shadow Intent)
        - 【V2.1 修复】接收 df 参数并在调用 _get_atomic_score 和 _get_safe_series 时传递。
        """
        print("  -- [融合层] 正在冶炼“上影线意图” (深度博弈版)...")
        states = {}
        df_index = df.index
        norm_window = 55
        upper_shadow_normalized = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW', 0.0)
        pct_change = self._get_safe_series(df, 'pct_change_D', pd.Series(0.0, index=df_index), method_name="_synthesize_upper_shadow_intent")
        main_force_flow = normalize_to_bipolar(self._get_safe_series(df, 'main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index), method_name="_synthesize_upper_shadow_intent"), df_index, norm_window)
        chip_concentration = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        micro_deception = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        upward_efficiency = (self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5) * 2 - 1).clip(-1, 1)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        volume_burst = (self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_BURST', 0.5) * 2 - 1).clip(-1, 1)
        is_up_day = (pct_change > 0).astype(float)
        is_down_day = (pct_change < 0).astype(float)
        is_flat_day = ((pct_change == 0) | (pct_change.abs() < 0.005)).astype(float)
        bullish_intent_components = [
            main_force_flow.clip(lower=0), chip_concentration.clip(lower=0),
            micro_deception.clip(lower=0), upward_efficiency.clip(lower=0),
            structural_trend_form.clip(lower=0)
        ]
        bullish_weights = np.array([0.35, 0.25, 0.2, 0.1, 0.1])
        bullish_intent_score = sum(w * s for w, s in zip(bullish_weights, bullish_intent_components)).clip(0, 1)
        bearish_intent_components = [
            main_force_flow.clip(upper=0).abs(), chip_concentration.clip(upper=0).abs(),
            micro_deception.clip(upper=0).abs(), (1 - upward_efficiency.clip(lower=0)),
            structural_trend_form.clip(upper=0).abs()
        ]
        bearish_weights = np.array([0.35, 0.25, 0.2, 0.1, 0.1])
        bearish_intent_score = sum(w * s for w, s in zip(bearish_weights, bearish_intent_components)).clip(0, 1)
        net_intent_direction = (bullish_intent_score - bearish_intent_score).clip(-1, 1)
        volume_conviction_multiplier = (volume_burst + 1) / 2
        base_intent_score = upper_shadow_normalized * net_intent_direction * volume_conviction_multiplier
        final_intent = pd.Series(0.0, index=df_index)
        up_day_adjusted_intent = base_intent_score * (1 + base_intent_score.abs() * 0.5)
        final_intent = final_intent.mask(is_up_day.astype(bool), up_day_adjusted_intent)
        down_day_adjusted_intent = (
            -upper_shadow_normalized * 0.8
            - upper_shadow_normalized * bearish_intent_score * 0.5
            + upper_shadow_normalized * bullish_intent_score * 0.2
        ) * volume_conviction_multiplier
        final_intent = final_intent.mask(is_down_day.astype(bool), down_day_adjusted_intent)
        flat_day_adjusted_intent = base_intent_score * 0.5
        final_intent = final_intent.mask(is_flat_day.astype(bool), flat_day_adjusted_intent)
        final_intent = final_intent.clip(-1, 1)
        states['FUSION_BIPOLAR_UPPER_SHADOW_INTENT'] = final_intent.astype(np.float32)
        print(f"  -- [融合层] “上影线意图”冶炼完成，最新分值: {final_intent.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_structure_score(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.5 · 上下文修复版】冶炼“趋势结构分” (FUSION_BIPOLAR_TREND_STRUCTURE_SCORE)
        - 【V1.5 修复】接收 df 参数并在调用 _get_safe_series 时传递。
        """
        print("  -- [融合层] 正在冶炼“趋势结构分”...")
        states = {}
        df_index = df.index
        norm_window = 55
        ema5 = self._get_safe_series(df, 'EMA_5_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        ema21 = self._get_safe_series(df, 'EMA_21_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        if ema5.isnull().all() or ema21.isnull().all():
            alignment_score = pd.Series(0.0, index=df_index)
        else:
            raw_alignment = (ema5 - ema21) / (ema21.abs().replace(0, 1e-9))
            alignment_score = normalize_to_bipolar(raw_alignment, df_index, window=norm_window, sensitivity=5.0)
        slope_ema5 = self._get_safe_series(df, 'SLOPE_5_EMA_5_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        slope_ema21 = self._get_safe_series(df, 'SLOPE_5_EMA_21_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        if slope_ema5.isnull().all() or slope_ema21.isnull().all():
            slope_score = pd.Series(0.0, index=df_index)
        else:
            norm_slope_ema5 = normalize_to_bipolar(slope_ema5, df_index, window=norm_window, sensitivity=0.005)
            norm_slope_ema21 = normalize_to_bipolar(slope_ema21, df_index, window=norm_window, sensitivity=0.005)
            slope_score = (norm_slope_ema5 * 0.6 + norm_slope_ema21 * 0.4).clip(-1, 1)
        if ema5.isnull().all() or ema21.isnull().all():
            divergence_score = pd.Series(0.0, index=df_index)
        else:
            ma_bias_raw = (ema5 - ema21) / (ema21.abs().replace(0, 1e-9))
            ma_bias_slope_raw = ma_bias_raw.diff(1).fillna(0)
            norm_ma_bias = normalize_to_bipolar(ma_bias_raw, df_index, window=norm_window, sensitivity=0.02)
            norm_ma_bias_slope = normalize_to_bipolar(ma_bias_slope_raw, df_index, window=norm_window, sensitivity=0.001)
            divergence_score = (norm_ma_bias * 0.7 + norm_ma_bias_slope * 0.3).clip(-1, 1)
        dma_raw = self._get_safe_series(df, 'DMA_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        dma_score = normalize_to_bipolar(dma_raw, df_index, window=norm_window)
        zigzag_raw = self._get_safe_series(df, 'ZIG_5_5.0_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        zigzag_score = normalize_to_bipolar(zigzag_raw, df_index, window=norm_window, sensitivity=0.05)
        weights = np.array([0.3, 0.3, 0.15, 0.15, 0.1])
        components = [alignment_score, slope_score, divergence_score, dma_score, zigzag_score]
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        final_trend_structure_score = (
            aligned_components[0] * weights[0] + aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] + aligned_components[3] * weights[3] +
            aligned_components[4] * weights[4]
        ).clip(-1, 1)
        states['FUSION_BIPOLAR_TREND_STRUCTURE_SCORE'] = final_trend_structure_score.astype(np.float32)
        print(f"  -- [融合层] “趋势结构分”冶炼完成，最新分值: {final_trend_structure_score.iloc[-1]:.4f}")
        return states

    def _synthesize_fund_flow_trend(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 上下文修复版】冶炼“资金趋势” (FUSION_BIPOLAR_FUND_FLOW_TREND)
        - 【V1.2 修复】接收 df 参数并在调用 _get_atomic_score 时传递。
        """
        print("  -- [融合层] 正在冶炼“资金趋势”...")
        states = {}
        df_index = df.index
        ff_consensus = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0)
        ff_conviction = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONVICTION', 0.0)
        ff_flow_momentum = self._get_atomic_score(df, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0)
        ff_divergence = self._get_atomic_score(df, 'SCORE_FF_AXIOM_DIVERGENCE', 0.0)
        components = [ff_consensus, ff_conviction, ff_flow_momentum, ff_divergence]
        weights = np.array([0.35, 0.30, 0.25, 0.10])
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        fund_flow_trend_score = (
            aligned_components[0] * weights[0] + aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] + aligned_components[3] * weights[3]
        ).clip(-1, 1)
        states['FUSION_BIPOLAR_FUND_FLOW_TREND'] = fund_flow_trend_score.astype(np.float32)
        print(f"  -- [融合层] “资金趋势”冶炼完成，最新分值: {fund_flow_trend_score.iloc[-1]:.4f}")
        return states

    def _synthesize_chip_trend(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.7 · 上下文修复版】冶炼“筹码趋势” (FUSION_BIPOLAR_CHIP_TREND)
        - 【V1.7 修复】接收 df 参数并在调用 _get_atomic_score 时传递。
        """
        print("  -- [融合层] 正在冶炼“筹码趋势”...")
        states = {}
        df_index = df.index
        required_chip_signals_meta = {
            'SCORE_CHIP_AXIOM_CONCENTRATION': {'polarity': 0, 'default_on_missing': 0.0},
            'SCORE_CHIP_AXIOM_COST_STRUCTURE': {'polarity': 0, 'default_on_missing': 0.0},
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT': {'polarity': 0, 'default_on_missing': 0.0},
            'SCORE_CHIP_AXIOM_PEAK_INTEGRITY': {'polarity': 0, 'default_on_missing': 0.0},
            'SCORE_CHIP_AXIOM_DIVERGENCE': {'polarity': 0, 'default_on_missing': 0.0},
            'SCORE_CHIP_CLEANLINESS': {'polarity': 1, 'default_on_missing': 0.0},
            'SCORE_CHIP_LOCKDOWN_DEGREE': {'polarity': 1, 'default_on_missing': 0.0},
            'SCORE_CHIP_AXIOM_TREND_MOMENTUM': {'polarity': 0, 'default_on_missing': 0.0}
        }
        missing_signals_in_atomic_states = [sig for sig in required_chip_signals_meta.keys() if sig not in self.strategy.atomic_states]
        if missing_signals_in_atomic_states:
            print(f"    -> [融合层-筹码趋势警告] 缺少以下关键筹码信号: {', '.join(missing_signals_in_atomic_states)}")
        chip_concentration = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', required_chip_signals_meta['SCORE_CHIP_AXIOM_CONCENTRATION']['default_on_missing'])
        chip_cost_structure = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_COST_STRUCTURE', required_chip_signals_meta['SCORE_CHIP_AXIOM_COST_STRUCTURE']['default_on_missing'])
        chip_holder_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', required_chip_signals_meta['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT']['default_on_missing'])
        chip_peak_integrity = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_PEAK_INTEGRITY', required_chip_signals_meta['SCORE_CHIP_AXIOM_PEAK_INTEGRITY']['default_on_missing'])
        chip_divergence = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_DIVERGENCE', required_chip_signals_meta['SCORE_CHIP_AXIOM_DIVERGENCE']['default_on_missing'])
        chip_cleanliness = self._get_atomic_score(df, 'SCORE_CHIP_CLEANLINESS', required_chip_signals_meta['SCORE_CHIP_CLEANLINESS']['default_on_missing'])
        chip_lockdown_degree = self._get_atomic_score(df, 'SCORE_CHIP_LOCKDOWN_DEGREE', required_chip_signals_meta['SCORE_CHIP_LOCKDOWN_DEGREE']['default_on_missing'])
        chip_trend_momentum = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_TREND_MOMENTUM', required_chip_signals_meta['SCORE_CHIP_AXIOM_TREND_MOMENTUM']['default_on_missing'])
        components = [
            chip_concentration, chip_cost_structure, chip_holder_sentiment,
            chip_peak_integrity, chip_divergence, chip_cleanliness, chip_lockdown_degree,
            chip_trend_momentum
        ]
        weights = np.array([0.12, 0.12, 0.12, 0.08, 0.08, 0.05, 0.05, 0.38])
        aligned_components = []
        for comp, meta_key in zip(components, required_chip_signals_meta.keys()):
            aligned_components.append(comp.reindex(df_index, fill_value=required_chip_signals_meta[meta_key]['default_on_missing']))
        chip_trend_score = pd.Series(0.0, index=df_index)
        for i, (comp_series, weight) in enumerate(zip(aligned_components, weights)):
            chip_trend_score += comp_series * weight
        chip_trend_score = chip_trend_score.clip(-1, 1)
        states['FUSION_BIPOLAR_CHIP_TREND'] = chip_trend_score.astype(np.float32)
        print(f"  -- [融合层] “筹码趋势”冶炼完成，最新分值: {chip_trend_score.iloc[-1]:.4f}")
        return states

    def _synthesize_accumulation_inflection(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 上下文修复版】冶炼“吸筹拐点信号” (FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT)
        - 【V1.1 修复】接收 df 参数并在调用 _get_atomic_score 时传递。
        """
        print("  -- [融合层] 正在冶炼“吸筹拐点信号”...")
        states = {}
        df_index = df.index
        inflection_score = pd.Series(0.0, index=df_index)
        ff_inflection_intent = self._get_atomic_score(df, 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT', 0.0)
        chip_concentration = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        chip_cost_structure = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_COST_STRUCTURE', 0.0)
        chip_holder_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        chip_trend_momentum = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_TREND_MOMENTUM', 0.0)
        p_conf_fusion_inflection = get_params_block(self.strategy, 'fusion_accumulation_inflection_params', {})
        ff_inflection_threshold = get_param_value(p_conf_fusion_inflection.get('ff_inflection_threshold'), 0.6)
        chip_concentration_threshold = get_param_value(p_conf_fusion_inflection.get('chip_concentration_threshold'), 0.5)
        chip_cost_structure_favorable_threshold = get_param_value(p_conf_fusion_inflection.get('chip_cost_structure_favorable_threshold'), 0.0)
        chip_holder_sentiment_strong_threshold = get_param_value(p_conf_fusion_inflection.get('chip_holder_sentiment_strong_threshold'), 0.5)
        chip_trend_momentum_positive_threshold = get_param_value(p_conf_fusion_inflection.get('chip_trend_momentum_positive_threshold'), 0.0)
        cond_ff_inflection_strong = (ff_inflection_intent > ff_inflection_threshold)
        cond_chip_supportive = (chip_concentration > chip_concentration_threshold) & \
                               (chip_cost_structure > chip_cost_structure_favorable_threshold) & \
                               (chip_holder_sentiment > chip_holder_sentiment_strong_threshold) & \
                               (chip_trend_momentum > chip_trend_momentum_positive_threshold)
        inflection_score.loc[cond_ff_inflection_strong] = ff_inflection_intent.loc[cond_ff_inflection_strong] * 0.6
        inflection_score.loc[cond_ff_inflection_strong & cond_chip_supportive] += \
            (chip_concentration.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.1 +
             chip_cost_structure.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.1 +
             chip_holder_sentiment.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.1 +
             chip_trend_momentum.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.1)
        inflection_score = inflection_score.clip(0, 1)
        tf_weights_fusion_inflection = get_param_value(p_conf_fusion_inflection.get('tf_fusion_weights'), {5: 0.6, 13: 0.3, 21: 0.1})
        inflection_score_normalized = get_adaptive_mtf_normalized_score(inflection_score, df_index, ascending=True, tf_weights=tf_weights_fusion_inflection).clip(0, 1)
        states['FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT'] = inflection_score_normalized
        print(f"  -- [融合层] “吸筹拐点信号”冶炼完成，最新分值: {inflection_score_normalized.iloc[-1]:.4f}")
        return states

    def _synthesize_chip_structural_potential(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 上下文修复版】冶炼“筹码结构势能” (FUSION_BIPOLAR_CHIP_STRUCTURAL_POTENTIAL)
        - 【V1.1 修复】接收 df 参数并在调用 _get_safe_series 时传递。
        """
        print("  -- [融合层] 正在冶炼“筹码结构势能”...")
        states = {}
        df_index = df.index
        potential_score_raw = self._get_safe_series(df, 'structural_potential_score_D', 50.0, method_name="_synthesize_chip_structural_potential")
        breakout_readiness_raw = self._get_safe_series(df, 'breakout_readiness_score_D', 50.0, method_name="_synthesize_chip_structural_potential")
        tension_raw = self._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name="_synthesize_chip_structural_potential")
        leverage_raw = self._get_safe_series(df, 'structural_leverage_D', 0.0, method_name="_synthesize_chip_structural_potential")
        vacuum_raw = self._get_safe_series(df, 'vacuum_zone_magnitude_D', 0.0, method_name="_synthesize_chip_structural_potential")
        potential_score = normalize_to_bipolar(potential_score_raw, df_index, window=55, sensitivity=20)
        breakout_readiness_score = normalize_to_bipolar(breakout_readiness_raw, df_index, window=55, sensitivity=20)
        tension_score = normalize_to_bipolar(tension_raw, df_index, window=55, sensitivity=0.5)
        leverage_score = normalize_to_bipolar(leverage_raw, df_index, window=55, sensitivity=0.5)
        vacuum_score = normalize_to_bipolar(vacuum_raw, df_index, window=55, sensitivity=0.5)
        components = [potential_score, breakout_readiness_score, tension_score, leverage_score, vacuum_score]
        weights = np.array([0.3, 0.3, 0.15, 0.15, 0.1])
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        structural_potential_score = (
            aligned_components[0] * weights[0] + aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] + aligned_components[3] * weights[3] +
            aligned_components[4] * weights[4]
        ).clip(-1, 1)
        states['FUSION_BIPOLAR_CHIP_STRUCTURAL_POTENTIAL'] = structural_potential_score.astype(np.float32)
        print(f"  -- [融合层] “筹码结构势能”冶炼完成，最新分值: {structural_potential_score.iloc[-1]:.4f}")
        return states

