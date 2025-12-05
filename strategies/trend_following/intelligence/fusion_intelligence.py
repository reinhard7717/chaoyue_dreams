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
        【V6.4 · 终极时序版】融合情报分析总指挥
        - 核心修复: 进行最终时序修正，确保`价格超买意图`在`滞涨风险`之前计算，
                    彻底解决所有融合信号之间的依赖问题，使法阵调度圆满无缺。
        """
        print("启动【V6.4 · 终极时序版】融合情报分析...")
        all_fusion_states = {}
        micro_conviction_states = self._synthesize_micro_conviction(df)
        all_fusion_states.update(micro_conviction_states)
        self.strategy.atomic_states.update(micro_conviction_states)
        regime_states = self._synthesize_market_regime(df)
        all_fusion_states.update(regime_states)
        self.strategy.atomic_states.update(regime_states)
        quality_states = self._synthesize_trend_quality(df)
        all_fusion_states.update(quality_states)
        self.strategy.atomic_states.update(quality_states)
        # [修改] 将“价格超买意图”的计算提前，作为“滞涨风险”的原料
        overextension_intent_states = self._synthesize_price_overextension_intent(df)
        all_fusion_states.update(overextension_intent_states)
        self.strategy.atomic_states.update(overextension_intent_states)
        # [修改] 确保“滞涨风险”在其所有原料计算完毕后执行
        stagnation_risk_states = self._synthesize_stagnation_risk(df)
        all_fusion_states.update(stagnation_risk_states)
        self.strategy.atomic_states.update(stagnation_risk_states)
        trend_exhaustion_states = self._synthesize_trend_exhaustion_syndrome(df)
        all_fusion_states.update(trend_exhaustion_states)
        self.strategy.atomic_states.update(trend_exhaustion_states)
        contested_accumulation_states = self._synthesize_contested_accumulation(df)
        all_fusion_states.update(contested_accumulation_states)
        self.strategy.atomic_states.update(contested_accumulation_states)
        pressure_states = self._synthesize_market_pressure(df)
        all_fusion_states.update(pressure_states)
        self.strategy.atomic_states.update(pressure_states)
        confrontation_states = self._synthesize_capital_confrontation(df)
        all_fusion_states.update(confrontation_states)
        self.strategy.atomic_states.update(confrontation_states)
        contradiction_states = self._synthesize_market_contradiction(df)
        all_fusion_states.update(contradiction_states)
        self.strategy.atomic_states.update(contradiction_states)
        trend_structure_states = self._synthesize_trend_structure_score(df)
        all_fusion_states.update(trend_structure_states)
        self.strategy.atomic_states.update(trend_structure_states)
        fund_flow_trend_states = self._synthesize_fund_flow_trend(df)
        all_fusion_states.update(fund_flow_trend_states)
        self.strategy.atomic_states.update(fund_flow_trend_states)
        chip_trend_states = self._synthesize_chip_trend(df)
        all_fusion_states.update(chip_trend_states)
        self.strategy.atomic_states.update(chip_trend_states)
        accumulation_inflection_states = self._synthesize_accumulation_inflection(df)
        all_fusion_states.update(accumulation_inflection_states)
        self.strategy.atomic_states.update(accumulation_inflection_states)
        accumulation_playbook_states = self._synthesize_accumulation_playbook(df)
        all_fusion_states.update(accumulation_playbook_states)
        self.strategy.atomic_states.update(accumulation_playbook_states)
        print(f"【V6.4 · 终极时序版】分析完成，生成 {len(all_fusion_states)} 个融合态势信号。")
        return all_fusion_states

    def _synthesize_market_contradiction(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 矛盾共振版】冶炼“市场矛盾” (Market Contradiction)
        - 核心重构: 废弃V2.x基于max()的“赢家通吃”模型，引入“矛盾共振”加权融合模型。
        - 诡道哲学: 采用加权求和，使得多个微弱但同向的背离信号能形成“共振”，其合力
                      可以超越单一的强烈信号，旨在“于无声处听惊雷”。
        - 信号赋权: 为不同情报域的信号赋予不同权重，体现其情报价值的差异。
        """
        print("  -- [融合层] 正在冶炼“市场矛盾”...")
        states = {}
        df_index = df.index
        # 1. [信号赋权] 定义各领域背离信号及其权重
        divergence_sources = {
            # 核心博弈层，权重最高
            'CHIP': 0.30,
            'FUND_FLOW': 0.25,
            # 战术执行层，权重次之
            'BEHAVIOR': 0.15,
            'DYNAMIC_MECHANICS': 0.10,
            # 表象结构层，权重较低
            'STRUCTURE': 0.10,
            'PATTERN': 0.05,
            'MICRO_BEHAVIOR': 0.05,
        }
        # 2. [共振融合] 核心数学逻辑 - 加权求和
        total_bullish_score = pd.Series(0.0, index=df_index)
        total_bearish_score = pd.Series(0.0, index=df_index)
        # 特殊处理双极性的筹码背离信号
        chip_divergence = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_DIVERGENCE', 0.0)
        chip_weight = divergence_sources.pop('CHIP')
        total_bullish_score += chip_divergence.clip(lower=0) * chip_weight
        total_bearish_score += chip_divergence.clip(upper=0).abs() * chip_weight
        # 处理其他单极性背离信号
        for source, weight in divergence_sources.items():
            # 行为层信号名称特殊
            if source == 'BEHAVIOR':
                bull_signal_name = 'SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY'
                bear_signal_name = 'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY'
            else:
                bull_signal_name = f'SCORE_{source}_BULLISH_DIVERGENCE'
                bear_signal_name = f'SCORE_{source}_BEARISH_DIVERGENCE'
            total_bullish_score += self._get_atomic_score(df, bull_signal_name, 0.0) * weight
            total_bearish_score += self._get_atomic_score(df, bear_signal_name, 0.0) * weight
        # 3. 最终裁决
        bipolar_contradiction = (total_bullish_score - total_bearish_score).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_CONTRADICTION'] = bipolar_contradiction.astype(np.float32)
        print(f"  -- [融合层] “市场矛盾”冶炼完成，最新分值: {bipolar_contradiction.iloc[-1]:.4f}")
        return states

    def _synthesize_market_regime(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 上下文修复版】冶炼“市场政权” (Market Regime)
        - 【V1.2 修复】接收 df 参数并在调用 _get_atomic_score 时传递。
        """
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

    def _synthesize_stagnation_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 依赖净化版】冶炼“滞涨风险” (FUSION_RISK_STAGNATION)
        - 核心修复: 将对已废弃的 `FUSION_BIPOLAR_UPPER_SHADOW_INTENT` 的依赖，
                    升级为对行为层更权威的 `SCORE_BEHAVIOR_DISTRIBUTION_INTENT` 信号的依赖。
        """
        states = {}
        df_index = df.index
        stagnation_evidence_raw = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0)
        price_overextension_risk = self._get_atomic_score(df, 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0).clip(upper=0).abs()
        # 替换为更权威的派发意图信号
        distribution_intent_risk = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0)
        fund_flow_bearish_risk = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(upper=0).abs()
        chip_dispersion_risk = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(upper=0).abs()
        profit_taking_supply_risk = normalize_score(self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_synthesize_stagnation_risk"), df_index, window=55, ascending=True).clip(0, 1)
        trend_confirmation_risk = (1 - self._get_atomic_score(df, 'CONTEXT_TREND_CONFIRMED', 0.0)).clip(0, 1)
        retail_fomo_risk = normalize_score(self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name="_synthesize_stagnation_risk"), df_index, window=55, ascending=True).clip(0, 1)
        is_price_stagnant_or_rising = (self._get_safe_series(df, 'pct_change_D', method_name="_synthesize_stagnation_risk") >= -0.005).astype(float)
        risk_components = [
            stagnation_evidence_raw, price_overextension_risk, distribution_intent_risk,
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
        【V2.1 · 代际同步版】冶炼“资本对抗” (Capital Confrontation)
        - 核心修复: 将对废弃信号 `SCORE_MICRO_AXIOM_DECEPTION` 的依赖，替换为对
                    新信号 `SCORE_MICRO_STRATEGY_STEALTH_OPS` 的依赖，完成情报代际同步。
        """
        states = {}
        flow_confrontation = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0)
        chip_transfer = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        # 替换为新的微观信号
        stealth_ops = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        bipolar_confrontation = (flow_confrontation * 0.5 + chip_transfer * 0.3 + stealth_ops * 0.2).clip(-1, 1)
        states['FUSION_BIPOLAR_CAPITAL_CONFRONTATION'] = bipolar_confrontation.astype(np.float32)
        print(f"  -- [融合层] “资本对抗”冶炼完成，最新分值: {bipolar_confrontation.iloc[-1]:.4f}")
        return states

    def _synthesize_price_overextension_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 状态与意图审判版】冶炼“价格超买意图” (Price Overextension Intent)
        - 核心重构: 废弃V1.x仅依赖价格振荡器的线性模型，引入“状态 × 意图”的非线性乘法模型。
        - 诡道哲学: 最终意图 = 超涨状态 × 反转意图。只有当价格达到极限状态，且市场
                      出现明确的反转意图时，信号才会爆发，旨在过滤“高位钝化”等陷阱。
        """
        print("  -- [融合层] 正在冶炼“价格超买意图”...")
        states = {}
        df_index = df.index
        # 1. 定义“超涨状态”的原料 (客观的伸展程度)
        overbought_state_sources = {
            'SCORE_DYN_AXIOM_OSCILLATOR': 0.6,  # 力学层-振荡器公理
            'SCORE_STRUCT_AXIOM_DEVIATION': 0.4, # 结构层-偏离度公理
        }
        # 2. 定义“反转意图”的原料 (主观的博弈动机)
        bearish_intent_sources = {
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT': 0.4, # 行为层-派发意图
            'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY': 0.3, # 行为层-看跌背离品质
            'FUSION_RISK_STAGNATION': 0.3, # 融合层-滞涨风险
        }
        bullish_intent_sources = {
            'SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK': 0.4, # 行为层-伏击式反攻
            'SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY': 0.3, # 行为层-看涨背离品质
            'FUSION_OPPORTUNITY_CONTESTED_ACCUMULATION': 0.3, # 融合层-争夺性吸筹
        }
        # 3. 核心数学逻辑 - 状态与意图的非线性审判
        # 3.1 计算“超涨状态”分
        oscillator_score = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_OSCILLATOR', 0.0)
        deviation_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_DEVIATION', 0.0)
        overbought_state = (oscillator_score.clip(lower=0) * overbought_state_sources['SCORE_DYN_AXIOM_OSCILLATOR'] +
                            deviation_score.clip(lower=0) * overbought_state_sources['SCORE_STRUCT_AXIOM_DEVIATION'])
        oversold_state = (oscillator_score.clip(upper=0).abs() * overbought_state_sources['SCORE_DYN_AXIOM_OSCILLATOR'] +
                          deviation_score.clip(upper=0).abs() * overbought_state_sources['SCORE_STRUCT_AXIOM_DEVIATION'])
        # 3.2 计算“反转意图”分
        bearish_intent = pd.Series(0.0, index=df_index)
        for signal, weight in bearish_intent_sources.items():
            bearish_intent += self._get_atomic_score(df, signal, 0.0) * weight
        bullish_intent = pd.Series(0.0, index=df_index)
        for signal, weight in bullish_intent_sources.items():
            bullish_intent += self._get_atomic_score(df, signal, 0.0) * weight
        # 3.3 非线性融合: 状态 × 意图
        overextension_bearish = (overbought_state * bearish_intent).clip(0, 1)
        overextension_bullish = (oversold_state * bullish_intent).clip(0, 1)
        # 4. 最终裁决
        bipolar_intent = (overextension_bullish - overextension_bearish).clip(-1, 1)
        states['FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT'] = bipolar_intent.astype(np.float32)
        # 5. [新增] 植入究极探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = debug_params.get('probe_dates', [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [价格超买意图究极探针 V2.0 · 状态与意图审判版] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料 - 超买状态源 (Overbought State)]:")
            print(f"    - SCORE_DYN_AXIOM_OSCILLATOR (Bearish Part): {oscillator_score.clip(lower=0).iloc[last_date_index]:.4f}")
            print(f"    - SCORE_STRUCT_AXIOM_DEVIATION (Bearish Part): {deviation_score.clip(lower=0).iloc[last_date_index]:.4f}")
            print("  [输入原料 - 看跌意图源 (Bearish Intent)]:")
            for s in bearish_intent_sources:
                print(f"    - {s}: {self._get_atomic_score(df, s, 0.0).iloc[last_date_index]:.4f}")
            print("  ---")
            print("  [输入原料 - 超卖状态源 (Oversold State)]:")
            print(f"    - SCORE_DYN_AXIOM_OSCILLATOR (Bullish Part): {oscillator_score.clip(upper=0).abs().iloc[last_date_index]:.4f}")
            print(f"    - SCORE_STRUCT_AXIOM_DEVIATION (Bullish Part): {deviation_score.clip(upper=0).abs().iloc[last_date_index]:.4f}")
            print("  [输入原料 - 看涨意图源 (Bullish Intent)]:")
            for s in bullish_intent_sources:
                print(f"    - {s}: {self._get_atomic_score(df, s, 0.0).iloc[last_date_index]:.4f}")
            print("  [关键计算节点]:")
            print(f"    - 综合超买状态分: {overbought_state.iloc[last_date_index]:.4f}")
            print(f"    - 综合看跌意图分: {bearish_intent.iloc[last_date_index]:.4f}")
            print(f"    - 综合超卖状态分: {oversold_state.iloc[last_date_index]:.4f}")
            print(f"    - 综合看涨意图分: {bullish_intent.iloc[last_date_index]:.4f}")
            print(f"    - [状态×意图] 看跌意图最终得分: {overextension_bearish.iloc[last_date_index]:.4f}")
            print(f"    - [状态×意图] 看涨意图最终得分: {overextension_bullish.iloc[last_date_index]:.4f}")
            print("  [最终裁决]:")
            print(f"    - 价格超买意图分 (FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT): {bipolar_intent.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        print(f"  -- [融合层] “价格超买意图”冶炼完成，最新分值: {bipolar_intent.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_structure_score(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.7 · 工具修正版】冶炼“趋势结构分” (FUSION_BIPOLAR_TREND_STRUCTURE_SCORE)
        - 核心修复: 修正了因误用 `normalize_to_bipolar` 处理多时间框架权重而导致的 TypeError。
                      全面换装为正确的 `get_adaptive_mtf_normalized_bipolar_score` 函数，
                      确保“变焦透镜”的战略意图得以正确执行。
        """
        states = {}
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        short_term_weights = get_param_value(p_mtf.get('short_term_weights'), {'weights': {3: 0.5, 5: 0.3, 8: 0.2}})
        ema5 = self._get_safe_series(df, 'EMA_5_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        ema21 = self._get_safe_series(df, 'EMA_21_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        if ema5.isnull().all() or ema21.isnull().all():
            alignment_score = pd.Series(0.0, index=df_index)
        else:
            raw_alignment = (ema5 - ema21) / (ema21.abs().replace(0, 1e-9))
            # 换用正确的多时间框架归一化函数
            alignment_score = get_adaptive_mtf_normalized_bipolar_score(raw_alignment, df_index, tf_weights=default_weights, sensitivity=5.0)
        slope_ema5 = self._get_safe_series(df, 'SLOPE_5_EMA_5_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        slope_ema21 = self._get_safe_series(df, 'SLOPE_5_EMA_21_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        if slope_ema5.isnull().all() or slope_ema21.isnull().all():
            slope_score = pd.Series(0.0, index=df_index)
        else:
            # 换用正确的多时间框架归一化函数
            norm_slope_ema5 = get_adaptive_mtf_normalized_bipolar_score(slope_ema5, df_index, tf_weights=short_term_weights, sensitivity=0.005)
            norm_slope_ema21 = get_adaptive_mtf_normalized_bipolar_score(slope_ema21, df_index, tf_weights=short_term_weights, sensitivity=0.005)
            slope_score = (norm_slope_ema5 * 0.6 + norm_slope_ema21 * 0.4).clip(-1, 1)
        if ema5.isnull().all() or ema21.isnull().all():
            divergence_score = pd.Series(0.0, index=df_index)
        else:
            ma_bias_raw = (ema5 - ema21) / (ema21.abs().replace(0, 1e-9))
            ma_bias_slope_raw = ma_bias_raw.diff(1).fillna(0)
            # 换用正确的多时间框架归一化函数
            norm_ma_bias = get_adaptive_mtf_normalized_bipolar_score(ma_bias_raw, df_index, tf_weights=default_weights, sensitivity=0.02)
            norm_ma_bias_slope = get_adaptive_mtf_normalized_bipolar_score(ma_bias_slope_raw, df_index, tf_weights=default_weights, sensitivity=0.001)
            divergence_score = (norm_ma_bias * 0.7 + norm_ma_bias_slope * 0.3).clip(-1, 1)
        dma_raw = self._get_safe_series(df, 'DMA_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        # 换用正确的多时间框架归一化函数
        dma_score = get_adaptive_mtf_normalized_bipolar_score(dma_raw, df_index, tf_weights=default_weights)
        zigzag_raw = self._get_safe_series(df, 'ZIG_5_5.0_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        # 换用正确的多时间框架归一化函数
        zigzag_score = get_adaptive_mtf_normalized_bipolar_score(zigzag_raw, df_index, tf_weights=default_weights, sensitivity=0.05)
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
        【V2.0 · 诡道博弈版】冶炼“资金趋势” (FUSION_BIPOLAR_FUND_FLOW_TREND)
        - 核心重构: 废弃V1.x的线性加权模型，引入基于“攻防力量博弈”的非线性数学模型。
        - 攻方力量: 融合主力共识、信念与动能，并以资金纯度(对倒强度)和隐蔽吸筹行为进行修正，量化真实、健康的多头攻击力。
        - 守方弱点: 聚合空头力量、派发压力与散户FOMO陷阱等风险因子，通过“木桶短板”逻辑识别最主要的威胁。
        - 诡道融合: 将“攻防力量差”与“价资背离”信号进行非线性融合，旨在穿透数据表象，揭示博弈格局与趋势的可持续性。
        """
        states = {}
        df_index = df.index
        # 1. 获取核心原子信号
        ff_consensus = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0)
        ff_conviction = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONVICTION', 0.0)
        ff_flow_momentum = self._get_atomic_score(df, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0)
        ff_divergence = self._get_atomic_score(df, 'SCORE_FF_AXIOM_DIVERGENCE', 0.0)
        # 2. 获取用于深度博弈分析的原始数据
        hidden_accumulation_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_synthesize_fund_flow_trend")
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_synthesize_fund_flow_trend")
        distribution_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_synthesize_fund_flow_trend")
        retail_fomo_raw = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name="_synthesize_fund_flow_trend")
        # 3. 数据归一化处理
        norm_window = 55
        hidden_accumulation_score = normalize_score(hidden_accumulation_raw, df_index, window=norm_window, ascending=True).clip(0, 1)
        wash_trade_score = normalize_score(wash_trade_raw, df_index, window=norm_window, ascending=True).clip(0, 1)
        distribution_pressure_score = normalize_score(distribution_pressure_raw, df_index, window=norm_window, ascending=True).clip(0, 1)
        retail_fomo_score = normalize_score(retail_fomo_raw, df_index, window=norm_window, ascending=True).clip(0, 1)
        # 4. 核心数学逻辑 - 攻防力量模型
        # 4.1 定义“攻方”力量 (多头力量)
        bullish_base_force = (ff_consensus.clip(lower=0) * ff_conviction.clip(lower=0)).pow(0.5)
        bullish_momentum_amplifier = ff_flow_momentum.clip(lower=0) * (1 + hidden_accumulation_score * 0.5)
        purity_factor = 1 - wash_trade_score
        bullish_power = (bullish_base_force * bullish_momentum_amplifier).pow(0.5) * purity_factor
        # 4.2 定义“守方”弱点 (空头力量与风险)
        bearish_explicit_force = np.maximum.reduce([
            ff_consensus.clip(upper=0).abs(),
            ff_conviction.clip(upper=0).abs(),
            ff_flow_momentum.clip(upper=0).abs()
        ])
        bearish_implicit_risk = np.maximum(distribution_pressure_score, retail_fomo_score)
        bearish_weakness = np.maximum(bearish_explicit_force, bearish_implicit_risk)
        # 5. 融合攻防力量与“诡道”因子(背离)
        power_balance = (bullish_power - bearish_weakness).clip(-1, 1)
        fund_flow_trend_score = np.tanh(power_balance * 1.5 + ff_divergence * 0.5)
        states['FUSION_BIPOLAR_FUND_FLOW_TREND'] = fund_flow_trend_score.astype(np.float32)
        print(f"  -- [融合层] “资金趋势”冶炼完成，最新分值: {fund_flow_trend_score.iloc[-1]:.4f}")
        return states

    def _synthesize_chip_trend(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 大一统重构版】冶炼“筹码趋势” (FUSION_BIPOLAR_CHIP_TREND)
        - 核心重构: 废弃旧的、基于冗余信号的线性模型。引入基于两大“大一统”信号的全新非线性融合模型。
        - 核心逻辑:
          1. 静态健康度: 由“战略态势”与“战场地形学”共同决定。有利的态势+有利的地形=健康的静态基础。
          2. 动态修正项: 由“持股心态”、“结构性推力(动量)”和“价筹张力(背离)”共同构成，对静态健康度进行动态修正。
          3. 最终融合: 趋势分 = tanh(静态健康度 * 1.5 + 动态修正项 * 0.5)，旨在输出一个能反映筹码综合战局的终极信号。
        """
        states = {}
        # 1. 获取全新的核心筹码信号
        strategic_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        battlefield_geography = self._get_atomic_score(df, 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 0.0)
        holder_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        trend_momentum = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_TREND_MOMENTUM', 0.0)
        divergence = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_DIVERGENCE', 0.0)
        # 2. 核心数学逻辑 - 静态与动态融合
        # 2.1 静态健康度 (Static Health)
        # 只有当态势和地形都为正时，才认为是健康的。负向信号作为风险项。
        static_health_base = (strategic_posture.clip(lower=0) * battlefield_geography.clip(lower=0)).pow(0.5)
        static_risk = np.maximum(strategic_posture.clip(upper=0).abs(), battlefield_geography.clip(upper=0).abs())
        static_health_score = (static_health_base - static_risk).clip(-1, 1)
        # 2.2 动态修正项 (Dynamic Modulator)
        # 融合心态、动量和背离
        dynamic_modulator = (
            holder_sentiment * 0.4 +
            trend_momentum * 0.4 +
            divergence * 0.2
        ).clip(-1, 1)
        # 3. 非线性融合
        chip_trend_score = np.tanh(
            static_health_score * 1.5 +       # 静态健康度是核心
            dynamic_modulator * 0.5           # 动态因子进行修正
        )
        states['FUSION_BIPOLAR_CHIP_TREND'] = chip_trend_score.astype(np.float32)
        print(f"  -- [融合层] “筹码趋势”冶炼完成，最新分值: {chip_trend_score.iloc[-1]:.4f}")
        return states

    def _synthesize_accumulation_inflection(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 大一统同步版】冶炼“吸筹拐点信号” (FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT)
        - 核心重构: 将判断筹码支撑环境的条件，从旧的多个原子公理升级为两大“大一统”战略信号，
                    使其对“吸筹末端”的判断更具战略深度。
        """
        states = {}
        df_index = df.index
        inflection_score = pd.Series(0.0, index=df_index)
        ff_inflection_intent = self._get_atomic_score(df, 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT', 0.0)
        # 引入新的大一统筹码信号
        strategic_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        battlefield_geography = self._get_atomic_score(df, 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 0.0)
        holder_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        trend_momentum = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_TREND_MOMENTUM', 0.0)
        p_conf_fusion_inflection = get_params_block(self.strategy, 'fusion_accumulation_inflection_params', {})
        ff_inflection_threshold = get_param_value(p_conf_fusion_inflection.get('ff_inflection_threshold'), 0.6)
        # 更新阈值参数，适配新信号
        strategic_posture_threshold = get_param_value(p_conf_fusion_inflection.get('strategic_posture_threshold'), 0.5)
        battlefield_geography_favorable_threshold = get_param_value(p_conf_fusion_inflection.get('battlefield_geography_favorable_threshold'), 0.0)
        holder_sentiment_strong_threshold = get_param_value(p_conf_fusion_inflection.get('holder_sentiment_strong_threshold'), 0.5)
        trend_momentum_positive_threshold = get_param_value(p_conf_fusion_inflection.get('trend_momentum_positive_threshold'), 0.0)
        cond_ff_inflection_strong = (ff_inflection_intent > ff_inflection_threshold)
        # 重构筹码支撑条件
        cond_chip_supportive = (strategic_posture > strategic_posture_threshold) & \
                               (battlefield_geography > battlefield_geography_favorable_threshold) & \
                               (holder_sentiment > holder_sentiment_strong_threshold) & \
                               (trend_momentum > trend_momentum_positive_threshold)
        inflection_score.loc[cond_ff_inflection_strong] = ff_inflection_intent.loc[cond_ff_inflection_strong] * 0.6
        # 更新加分项
        inflection_score.loc[cond_ff_inflection_strong & cond_chip_supportive] += \
            (strategic_posture.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.15 +
             battlefield_geography.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.15 +
             holder_sentiment.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.05 +
             trend_momentum.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.05)
        inflection_score = inflection_score.clip(0, 1)
        tf_weights_fusion_inflection = get_param_value(p_conf_fusion_inflection.get('tf_fusion_weights'), {5: 0.6, 13: 0.3, 21: 0.1})
        inflection_score_normalized = get_adaptive_mtf_normalized_score(inflection_score, df_index, ascending=True, tf_weights=tf_weights_fusion_inflection).clip(0, 1)
        states['FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT'] = inflection_score_normalized
        print(f"  -- [融合层] “吸筹拐点信号”冶炼完成，最新分值: {inflection_score_normalized.iloc[-1]:.4f}")
        return states

    def _synthesize_contested_accumulation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 诡道博弈版】冶炼“博弈吸筹” (Contested Accumulation)
        - 核心目标: 识别并量化“高位换手洗盘”这一高级战术形态。
        - 博弈逻辑: 当微观层“隐秘吸筹”与行为层“派发意图”同时强烈时，这并非矛盾，而是
                      “新主力利用老主力的卖盘进行权力交接”的信号。
        - 数学公式: 博弈吸筹分 = (隐秘行动分 * 派发意图分)^0.5 * 趋势质量分
        """
        states = {}
        # 1. 获取矛盾双方的信号
        stealth_ops = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        distribution_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0)
        # 2. 获取场景过滤器
        trend_quality = self._get_atomic_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0).clip(lower=0)
        # 3. 战术融合
        # 只有当隐秘吸筹和派发意图同时存在时，信号才被激活
        contested_evidence = (stealth_ops * distribution_intent).pow(0.5).fillna(0.0)
        # 只有在健康的上升趋势中，这种换手才有积极意义
        final_score = (contested_evidence * trend_quality).clip(0, 1)
        states['FUSION_OPPORTUNITY_CONTESTED_ACCUMULATION'] = final_score.astype(np.float32)
        print(f"  -- [融合层] “博弈吸筹”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_micro_conviction(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 意图确认版】冶炼“微观信念” (Micro Conviction)
        - 核心重构: 废弃V1.0的线性加权模型，引入“意图-确认”非线性融合模型。
        - 核心公式: 微观信念 = 瞬时意图 × (1 + 意图趋势 × 确认系数)
        - 融合逻辑: 以“瞬时意图”为基础，用“意图趋势”作为确认或否定的调节器。
                      此模型旨在放大“共振”信号，并揭示“意图与趋势相悖”的诡道陷阱。
        """
        print("  -- [融合层] 正在冶炼“微观信念”...")
        states = {}
        # 1. 获取核心微观信号
        micro_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0)
        micro_divergence = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_DIVERGENCE', 0.0)
        # 2. 核心数学逻辑 - “意图-确认”模型
        confirmation_factor = 0.5 # 确认系数，控制意图趋势的影响力
        # 确认调节器：当意图趋势与瞬时意图同向时 > 1 (放大)，反向时 < 1 (抑制)
        confirmation_modulator = (1 + micro_divergence * confirmation_factor)
        # 非线性融合
        micro_conviction_score = (micro_intent * confirmation_modulator).clip(-1, 1)
        output_name = 'FUSION_BIPOLAR_MICRO_CONVICTION'
        states[output_name] = micro_conviction_score.astype(np.float32)
        print(f"  -- [融合层] “微观信念”冶炼完成，最新分值: {micro_conviction_score.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_quality(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 · 四象支柱版】冶炼“趋势质量” (Trend Quality)
        - 核心重构: 废弃V3.x的线性加权“大杂烩”模型，引入基于“四象支柱”的非线性融合模型。
        - 核心公式: 趋势质量 = (结构支柱 × 动能支柱 × 信念支柱 × 根基支柱)^(1/4)
        - 诡道哲学: 基于“木桶效应”，任何一根支柱的崩塌都会导致整体质量的急剧下降，
                      旨在暴露趋势的“最短板”，而非被平庸信号所平均。
        - 信号净化: 严格遵守融合层戒律，只消费各情报层提炼的“公理级”或“超级原子”信号。
        """
        print("  -- [融合层] 正在冶炼“趋势质量”...")
        states = {}
        df_index = df.index
        # --- 1. 信号原料库 (严格筛选公理级信号) ---
        # 支柱一：结构 (Structure Pillar) - 趋势的骨架是否坚固
        struct_posture = self._get_atomic_score(df, 'SCORE_STRUCT_STRATEGIC_POSTURE', 0.0)
        struct_geography = self._get_atomic_score(df, 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 0.0) # 筹码地形归于结构
        struct_stability = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_STABILITY', 0.0)
        # 支柱二：动能 (Momentum Pillar) - 趋势的引擎是否强劲
        dyn_momentum = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_MOMENTUM', 0.0)
        dyn_inertia = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_INERTIA', 0.0)
        behavior_upward_momentum = self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM', 0.0)
        # 支柱三：信念 (Conviction Pillar) - 趋势的灵魂是否坚定
        ff_posture = self._get_atomic_score(df, 'SCORE_FF_STRATEGIC_POSTURE', 0.0)
        chip_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        chip_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        # 支柱四：根基 (Foundation Pillar) - 趋势的土壤是否肥沃
        found_posture = self._get_atomic_score(df, 'SCORE_FOUNDATION_STRATEGIC_POSTURE', 0.0)
        found_constitution = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION', 0.0)
        found_tide = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 0.0)
        # 最终调节器
        micro_conviction = self._get_atomic_score(df, 'FUSION_BIPOLAR_MICRO_CONVICTION', 0.0)
        # --- 2. 核心数学逻辑 - 四象支柱模型 ---
        # 分别计算各支柱得分 (将所有输入信号转换为[0,2]区间，便于几何平均)
        pillar_structure = ((struct_posture + 1) * 0.5 + (struct_geography + 1) * 0.3 + (struct_stability + 1) * 0.2)
        pillar_momentum = ((dyn_momentum + 1) * 0.4 + (dyn_inertia + 1) * 0.3 + (behavior_upward_momentum + 1) * 0.3)
        pillar_conviction = ((ff_posture + 1) * 0.4 + (chip_posture + 1) * 0.4 + (chip_sentiment + 1) * 0.2)
        pillar_foundation = ((found_posture + 1) * 0.5 + (found_constitution + 1) * 0.3 + (found_tide + 1) * 0.2)
        # 非线性融合：几何平均体现“木桶效应”，任何支柱为0则整体为0
        # 为避免负数开方，先在[0,2]区间计算，再映射回[-1,1]
        raw_quality_score_positive = (pillar_structure * pillar_momentum * pillar_conviction * pillar_foundation).pow(1/4)
        # 映射回[-1, 1]
        bipolar_quality = (raw_quality_score_positive - 1).clip(-1, 1)
        # 应用微观信念作为最终的真实性检验器
        micro_conviction_regulator = (1 + micro_conviction * 0.3).clip(0.7, 1.3)
        final_bipolar_quality = (bipolar_quality * micro_conviction_regulator).clip(-1, 1)
        states['FUSION_BIPOLAR_TREND_QUALITY'] = final_bipolar_quality.astype(np.float32)
        print(f"  -- [融合层] “趋势质量”冶炼完成，最新分值: {final_bipolar_quality.iloc[-1]:.4f} (原始分: {bipolar_quality.iloc[-1]:.4f}, 微观调节器: {micro_conviction_regulator.iloc[-1]:.4f})")
        return states

    def _synthesize_market_pressure(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 压力共振版】冶炼“市场压力” (Market Pressure)
        - 核心重构: 废弃V1.x基于max()的“赢家通吃”模型，引入“压力共振”非线性融合模型。
        - 信号升维: 引用最高阶的战术级机会与风险信号，取代底层的过程信号。
        - 诡道哲学: 多个同向压力信号同时出现时，其合力将通过“共振放大器”被非线性放大，
                      以体现风险或机会的“集群效应”。
        """
        print("  -- [融合层] 正在冶炼“市场压力”...")
        states = {}
        df_index = df.index
        # 1. [信号升维] 引用最高阶的战术信号
        opportunity_signals = {
            'SCORE_CHIP_HARMONY_INFLECTION': 0.3,
            'SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK': 0.25,
            'SCORE_OPPORTUNITY_SELLING_EXHAUSTION': 0.2,
            'SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY': 0.15,
            'FUSION_OPPORTUNITY_CONTESTED_ACCUMULATION': 0.1,
        }
        risk_signals = {
            'SCORE_RISK_BREAKOUT_FAILURE_CASCADE': 0.3,
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT': 0.25,
            'PROCESS_FUSION_TREND_EXHAUSTION_SYNDROME': 0.2,
            'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY': 0.15,
            'FUSION_RISK_STAGNATION': 0.1,
        }
        # 2. 核心数学逻辑 - 压力共振模型
        # 2.1 计算基础压力 (加权融合)
        base_upward_pressure = pd.Series(0.0, index=df_index)
        for signal, weight in opportunity_signals.items():
            base_upward_pressure += self._get_atomic_score(df, signal, 0.0) * weight
        base_downward_pressure = pd.Series(0.0, index=df_index)
        for signal, weight in risk_signals.items():
            base_downward_pressure += self._get_atomic_score(df, signal, 0.0) * weight
        # 2.2 计算共振放大器
        resonance_threshold = 0.5
        resonance_bonus_factor = 0.2
        resonant_upward_signals = sum(self._get_atomic_score(df, s, 0.0) > resonance_threshold for s in opportunity_signals)
        resonant_downward_signals = sum(self._get_atomic_score(df, s, 0.0) > resonance_threshold for s in risk_signals)
        upward_resonance_modulator = 1 + (resonant_upward_signals * resonance_bonus_factor)
        downward_resonance_modulator = 1 + (resonant_downward_signals * resonance_bonus_factor)
        # 2.3 应用放大器
        amplified_upward_pressure = (base_upward_pressure * upward_resonance_modulator).clip(0, 1)
        amplified_downward_pressure = (base_downward_pressure * downward_resonance_modulator).clip(0, 1)
        # 3. 最终裁决
        bipolar_pressure = (amplified_upward_pressure - amplified_downward_pressure).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_PRESSURE'] = bipolar_pressure.astype(np.float32)
        print(f"  -- [融合层] “市场压力”冶炼完成，最新分值: {bipolar_pressure.iloc[-1]:.4f}")
        return states

    def _synthesize_accumulation_playbook(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 吸筹剧本】融合心法
        - 核心目标: 将不同战术的吸筹信号融合成统一的“吸筹剧本”分。
        - 融合模型: 吸筹剧本分 = max(隐秘吸筹, 恐慌吸筹, 诡道吸筹) × (1 + 筹码态势改善度)
        - 核心诡道: 1. max()体现战术互斥性。 2. 筹码态势改善度作为品质调节器。
        """
        print("  -- [融合层] 正在推演“吸筹剧本”...")
        states = {}
        df_index = df.index
        # 1. 获取原料信号
        stealth_accumulation = self._get_atomic_score(df, 'PROCESS_META_STEALTH_ACCUMULATION', 0.0)
        panic_accumulation = self._get_atomic_score(df, 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 0.0)
        deceptive_accumulation = self._get_atomic_score(df, 'PROCESS_META_DECEPTIVE_ACCUMULATION', 0.0)
        chip_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        # 2. 核心数学逻辑
        # 2.1 识别主导战术 (max体现互斥性)
        max_accumulation_tactic = np.maximum.reduce([
            stealth_accumulation.values,
            panic_accumulation.values,
            deceptive_accumulation.values
        ])
        max_accumulation_tactic = pd.Series(max_accumulation_tactic, index=df_index)
        # 2.2 计算品质调节器 (筹码态势改善度)
        chip_posture_change = chip_posture.diff(1).fillna(0)
        chip_improvement_factor = chip_posture_change.clip(lower=0) # 只关注正向改善
        quality_modulator = 1 + chip_improvement_factor
        # 2.3 融合
        playbook_score = (max_accumulation_tactic * quality_modulator).clip(0, 1)
        output_name = 'PROCESS_FUSION_ACCUMULATION_PLAYBOOK'
        states[output_name] = playbook_score.astype(np.float32)
        print(f"  -- [融合层] “吸筹剧本”推演完成，最新分值: {playbook_score.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_exhaustion_syndrome(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 趋势衰竭综合征】融合心法
        - 核心目标: 将独立的顶部风险征兆进行非线性融合，识别风险“共振”。
        - 融合模型: 衰竭综合征 = (w1*背离 + w2*衰减 + w3*冷却) × (1 + 共振奖励)
        - 核心诡道: 引入“共振奖励”机制，体现风险的非线性叠加效应。
        """
        print("  -- [融合层] 正在诊断“趋势衰竭综合征”...")
        states = {}
        df_index = df.index
        # 1. 从配置文件获取参数
        p_conf = get_params_block(self.strategy, 'fusion_playbook_params', {})
        p_exhaustion = get_param_value(p_conf.get('trend_exhaustion_syndrome'), {})
        weights = get_param_value(p_exhaustion.get('weights'), {'divergence': 0.4, 'conviction_decay': 0.4, 'cooling': 0.2})
        resonance_threshold = get_param_value(p_exhaustion.get('resonance_threshold'), 0.5)
        resonance_bonus_factor = get_param_value(p_exhaustion.get('resonance_bonus_factor'), 0.3)
        # 2. 获取原料信号
        divergence = self._get_atomic_score(df, 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0)
        conviction_decay = self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0)
        sector_cooling = self._get_atomic_score(df, 'PROCESS_META_HOT_SECTOR_COOLING', 0.0)
        # 3. 核心数学逻辑
        # 3.1 计算加权基础分
        total_weight = sum(weights.values())
        if total_weight == 0: total_weight = 1.0 # 防止除零
        weighted_sum = (
            divergence * weights.get('divergence', 0) +
            conviction_decay * weights.get('conviction_decay', 0) +
            sector_cooling * weights.get('cooling', 0)
        ) / total_weight
        # 3.2 计算共振奖励
        signals_above_threshold = (
            (divergence > resonance_threshold).astype(int) +
            (conviction_decay > resonance_threshold).astype(int) +
            (sector_cooling > resonance_threshold).astype(int)
        )
        resonance_bonus = (signals_above_threshold >= 2).astype(float) * resonance_bonus_factor
        resonance_modulator = 1 + resonance_bonus
        # 3.3 融合
        syndrome_score = (weighted_sum * resonance_modulator).clip(0, 1)
        output_name = 'PROCESS_FUSION_TREND_EXHAUSTION_SYNDROME'
        states[output_name] = syndrome_score.astype(np.float32)
        print(f"  -- [融合层] “趋势衰竭综合征”诊断完成，最新分值: {syndrome_score.iloc[-1]:.4f}")
        return states









