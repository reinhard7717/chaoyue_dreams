# 文件: strategies/trend_following/intelligence/fusion_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_adaptive_mtf_normalized_bipolar_score, normalize_score, normalize_to_bipolar

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

    def _get_atomic_score(self, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.1 · 默认值修复版】安全地从原子状态库中获取分数，处理缺失情况。
        - 核心修复: 将默认值从 0.5 改为 0.0。0.5代表中性，而0.0代表无信号/无贡献，
                      这在几何平均中是更安全的选择，避免了中性信号对结果的污染。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in self.strategy.df_indicators.columns:
            return self.strategy.df_indicators[name]
        else:
            # 默认值从 0.5 改为 0.0
            return pd.Series(default, index=self.strategy.df_indicators.index)

    def run_fusion_diagnostics(self) -> Dict[str, pd.Series]:
        print("启动【V3.0 · 战场态势引擎】融合情报分析...")
        all_fusion_states = {}
        regime_states = self._synthesize_market_regime()
        all_fusion_states.update(regime_states)
        quality_states = self._synthesize_trend_quality()
        all_fusion_states.update(quality_states)
        pressure_states = self._synthesize_market_pressure()
        all_fusion_states.update(pressure_states)
        confrontation_states = self._synthesize_capital_confrontation()
        all_fusion_states.update(confrontation_states)
        contradiction_states = self._synthesize_market_contradiction()
        all_fusion_states.update(contradiction_states)
        overextension_intent_states = self._synthesize_price_overextension_intent()
        all_fusion_states.update(overextension_intent_states)
        upper_shadow_intent_states = self._synthesize_upper_shadow_intent()
        all_fusion_states.update(upper_shadow_intent_states)
        # 新增行: 冶炼“滞涨风险”
        stagnation_risk_states = self._synthesize_stagnation_risk()
        all_fusion_states.update(stagnation_risk_states)
        self.strategy.atomic_states.update(all_fusion_states)
        print(f"【V3.0 · 战场态势引擎】分析完成，生成 {len(all_fusion_states)} 个融合态势信号。")
        return all_fusion_states

    def _synthesize_market_contradiction(self) -> Dict[str, pd.Series]:
        """
        【V1.0】冶炼“市场矛盾” (Market Contradiction)
        - 核心思想: 融合各情报领域（行为、筹码、资金流、结构、力学、形态、微观）的背离信号。
        - 证据链: 收集所有领域的看涨背离和看跌背离信号，进行加权融合。
        """
        print("  -- [融合层] 正在冶炼“市场矛盾”...")
        states = {}
        # 定义所有领域的背离信号源
        divergence_sources = [
            'FOUNDATION', 'STRUCTURE', 'PATTERN', 'DYNAMIC_MECHANICS',
            'CHIP', 'FUND_FLOW', 'MICRO_BEHAVIOR', 'BEHAVIOR'
        ]
        bullish_divergence_scores = []
        bearish_divergence_scores = []
        for source in divergence_sources:
            bull_signal_name = f'SCORE_{source}_BULLISH_DIVERGENCE'
            bear_signal_name = f'SCORE_{source}_BEARISH_DIVERGENCE'
            bullish_divergence_scores.append(self._get_atomic_score(bull_signal_name, 0.0).values)
            bearish_divergence_scores.append(self._get_atomic_score(bear_signal_name, 0.0).values)
        net_bullish_divergence = np.maximum.reduce(bullish_divergence_scores)
        net_bearish_divergence = np.maximum.reduce(bearish_divergence_scores)
        bipolar_contradiction = (pd.Series(net_bullish_divergence, index=self.strategy.df_indicators.index) -
                                 pd.Series(net_bearish_divergence, index=self.strategy.df_indicators.index)).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_CONTRADICTION'] = bipolar_contradiction.astype(np.float32)
        print(f"  -- [融合层] “市场矛盾”冶炼完成，最新分值: {bipolar_contradiction.iloc[-1]:.4f}")
        return states

    def _synthesize_market_regime(self) -> Dict[str, pd.Series]:
        """
        【V1.1 · 柔性评估重构版】冶炼“市场政权” (Market Regime)
        - 核心重构: 废弃了对 `hurst_memory`, `inertia`, `stability` 进行 `clip(lower=0)` 的“一票否决”逻辑。
                      改为直接使用它们的原始双极性分数进行加权平均，得到一个更柔性的 `trend_evidence`。
                      这避免了在趋势酝酿期，因单一公理暂时为负而导致整体趋势证据归零的问题。
        - 探针植入: 新增探针，打印 `hurst_memory`, `inertia`, `stability` 的原始值，以及计算出的 `trend_evidence` 和 `reversion_evidence`。
        """
        print("  -- [融合层] 正在冶炼“市场政权”...")
        df = self.strategy.df_indicators
        states = {}
        hurst_memory = self._get_atomic_score('SCORE_CYCLICAL_HURST_MEMORY', 0.0)
        inertia = self._get_atomic_score('SCORE_DYN_AXIOM_INERTIA', 0.0)
        stability = self._get_atomic_score('SCORE_DYN_AXIOM_STABILITY', 0.0)
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

    def _synthesize_trend_quality(self) -> Dict[str, pd.Series]:
        """
        【V1.3 · 纯粹原子版】冶炼“趋势质量” (Trend Quality)
        - 核心修复: 不再消费原子层的“共振”信号，而是直接消费各原子情报模块的**公理信号**。
        - 核心逻辑: 融合各领域公理的双极性分数，形成一个整体的趋势质量判断。
        """
        print("  -- [融合层] 正在冶炼“趋势质量”...")
        states = {}
        # 收集各领域公理信号
        # 基础层公理
        foundation_trend = self._get_atomic_score('SCORE_FOUNDATION_AXIOM_TREND', 0.0)
        foundation_oscillator = self._get_atomic_score('SCORE_FOUNDATION_AXIOM_OSCILLATOR', 0.0)
        foundation_flow = self._get_atomic_score('SCORE_FOUNDATION_AXIOM_FLOW', 0.0)
        foundation_volatility = self._get_atomic_score('SCORE_FOUNDATION_AXIOM_VOLATILITY', 0.0)
        # 结构层公理
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        structural_mtf_cohesion = self._get_atomic_score('SCORE_STRUCT_AXIOM_MTF_COHESION', 0.0)
        structural_stability = self._get_atomic_score('SCORE_STRUCT_AXIOM_STABILITY', 0.0)
        # 力学层公理
        dynamic_momentum = self._get_atomic_score('SCORE_DYN_AXIOM_MOMENTUM', 0.0)
        dynamic_inertia = self._get_atomic_score('SCORE_DYN_AXIOM_INERTIA', 0.0)
        dynamic_stability = self._get_atomic_score('SCORE_DYN_AXIOM_STABILITY', 0.0)
        dynamic_energy = self._get_atomic_score('SCORE_DYN_AXIOM_ENERGY', 0.0)
        # 资金流层公理
        fund_flow_consensus = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        fund_flow_conviction = self._get_atomic_score('SCORE_FF_AXIOM_CONVICTION', 0.0)
        fund_flow_increment = self._get_atomic_score('SCORE_FF_AXIOM_INCREMENT', 0.0)
        # 筹码层公理
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        chip_cost_structure = self._get_atomic_score('SCORE_CHIP_AXIOM_COST_STRUCTURE', 0.0)
        chip_holder_sentiment = self._get_atomic_score('SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        chip_peak_integrity = self._get_atomic_score('SCORE_CHIP_AXIOM_PEAK_INTEGRITY', 0.0)
        # 微观行为层公理
        micro_deception = self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        micro_probe = self._get_atomic_score('SCORE_MICRO_AXIOM_PROBE', 0.0)
        micro_efficiency = self._get_atomic_score('SCORE_MICRO_AXIOM_EFFICIENCY', 0.0)
        # 行为层原子信号 (部分作为公理使用)
        behavior_upward_efficiency = self._get_atomic_score('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.0)
        behavior_downward_resistance = self._get_atomic_score('SCORE_BEHAVIOR_DOWNWARD_RESISTANCE', 0.0)
        behavior_intraday_bull_control = self._get_atomic_score('SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', 0.0)
        # 形态层公理
        pattern_reversal = self._get_atomic_score('SCORE_PATTERN_AXIOM_REVERSAL', 0.0)
        pattern_breakout = self._get_atomic_score('SCORE_PATTERN_AXIOM_BREAKOUT', 0.0)
        # 融合所有公理，形成一个整体的双极性趋势质量分
        # 这里需要根据每个公理的性质和对趋势质量的贡献进行加权
        # 这是一个示例性的加权融合，实际权重需要通过回测和优化确定
        bipolar_quality = (
            # 基础层
            foundation_trend * 0.1 +
            foundation_oscillator * -0.05 + # 摆动指标超买/超卖对趋势质量有负面影响
            foundation_flow * 0.05 +
            foundation_volatility * 0.05 + # 低波动对趋势有利
            # 结构层
            structural_trend_form * 0.15 +
            structural_mtf_cohesion * 0.1 +
            structural_stability * 0.1 +
            # 力学层
            dynamic_momentum * 0.1 +
            dynamic_inertia * 0.1 +
            dynamic_stability * 0.05 +
            dynamic_energy * 0.05 +
            # 资金流层
            fund_flow_consensus * 0.05 +
            fund_flow_conviction * 0.05 +
            fund_flow_increment * 0.05 +
            # 筹码层
            chip_concentration * 0.05 +
            chip_cost_structure * 0.05 +
            chip_holder_sentiment * 0.05 +
            chip_peak_integrity * 0.05 +
            # 微观行为层
            micro_deception * 0.02 + # 伪装吸筹对趋势质量有正面影响
            micro_probe * 0.02 + # 试探确认对趋势质量有正面影响
            micro_efficiency * 0.02 + # 高效率对趋势质量有正面影响
            # 行为层
            behavior_upward_efficiency * 0.03 +
            behavior_downward_resistance * 0.03 +
            behavior_intraday_bull_control * 0.02 +
            # 形态层
            pattern_reversal * 0.02 + # 反转信号对趋势质量有影响
            pattern_breakout * 0.03 # 突破信号对趋势质量有影响
        ).clip(-1, 1)
        states['FUSION_BIPOLAR_TREND_QUALITY'] = bipolar_quality.astype(np.float32)
        print(f"  -- [融合层] “趋势质量”冶炼完成，最新分值: {bipolar_quality.iloc[-1]:.4f}")
        return states

    def _synthesize_market_pressure(self) -> Dict[str, pd.Series]:
        """
        【V1.1 · 纯粹原子版】冶炼“市场压力” (Market Pressure)
        - 核心思想: 衡量市场中“向上反转”与“向下回调”两股力量的净压力。
        - 证据链: 融合所有原子情报层和过程情报层的“底部反转”和“顶部反转”信号。
        """
        print("  -- [融合层] 正在冶炼“市场压力”...")
        states = {}
        # 定义所有领域的反转信号源 (现在从 ProcessIntelligence 获取)
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
                upward_pressure_scores.append(self._get_atomic_score(signal_name, 0.0).values)
            elif 'TOP_REVERSAL' in signal_name:
                downward_pressure_scores.append(self._get_atomic_score(signal_name, 0.0).values)
        net_upward_pressure = np.maximum.reduce(upward_pressure_scores)
        net_downward_pressure = np.maximum.reduce(downward_pressure_scores)
        bipolar_pressure = (pd.Series(net_upward_pressure, index=self.strategy.df_indicators.index) -
                            pd.Series(net_downward_pressure, index=self.strategy.df_indicators.index)).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_PRESSURE'] = bipolar_pressure.astype(np.float32)
        print(f"  -- [融合层] “市场压力”冶炼完成，最新分值: {bipolar_pressure.iloc[-1]:.4f}")
        return states

    def _synthesize_stagnation_risk(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 深度博弈版】冶炼“滞涨风险” (FUSION_RISK_STAGNATION)
        - 核心思想: 综合多维度证据，精确识别多头力量衰竭与空头隐秘积蓄的滞涨风险。
        - 证据链:
          1. 行为层滞涨证据原始分 (INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW)
          2. 价格超买意图 (FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT 的负向部分)
          3. 上影线抛压意图 (FUSION_BIPOLAR_UPPER_SHADOW_INTENT 的负向部分)
          4. 主力资金流出背离 (SCORE_FF_AXIOM_CONSENSUS 的负向部分)
          5. 筹码集中度下降 (SCORE_CHIP_AXIOM_CONCENTRATION 的负向部分)
          6. 获利盘供给压力 (imminent_profit_taking_supply_D)
          7. 趋势确认度下降 (CONTEXT_TREND_CONFIRMED 的反向)
          8. 市场情绪亢奋 (retail_fomo_premium_index_D)
        - 输出: [0, 1] 的风险分数，0表示无风险，1表示最大风险。
        """
        print("  -- [融合层] 正在冶炼“滞涨风险”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        # 1. 行为层滞涨证据原始分 (来自行为层)
        stagnation_evidence_raw = self._get_atomic_score('INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0)
        # 2. 价格超买意图 (来自融合层，负向部分代表风险)
        price_overextension_risk = self._get_atomic_score('FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0).clip(upper=0).abs()
        # 3. 上影线抛压意图 (来自融合层，负向部分代表风险)
        upper_shadow_pressure_risk = self._get_atomic_score('FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0).clip(upper=0).abs()
        # 4. 主力资金流出背离 (来自资金流层，负向共识代表风险)
        fund_flow_bearish_risk = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(upper=0).abs()
        # 5. 筹码集中度下降 (来自筹码层，负向集中度代表风险)
        chip_dispersion_risk = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(upper=0).abs()
        # 6. 获利盘供给压力 (来自筹码高级指标，需要归一化)
        profit_taking_supply_risk = normalize_score(self.strategy.df_indicators.get('imminent_profit_taking_supply_D', 0.0), df_index, window=55, ascending=True).clip(0, 1)
        # 7. 趋势确认度下降 (来自基础层上下文，反向代表风险)
        trend_confirmation_risk = (1 - self._get_atomic_score('CONTEXT_TREND_CONFIRMED', 0.0)).clip(0, 1)
        # 8. 市场情绪亢奋 (来自资金流高级指标，需要归一化)
        retail_fomo_risk = normalize_score(self.strategy.df_indicators.get('retail_fomo_premium_index_D', 0.0), df_index, window=55, ascending=True).clip(0, 1)
        # 9. 价格上涨或横盘的前提条件 (滞涨风险的前提)
        is_price_stagnant_or_rising = (self.strategy.df_indicators['pct_change_D'] >= -0.005).astype(float)
        # 融合所有风险证据 (加权几何平均)
        # 权重分配需要根据回测结果进行优化，这里给出示例权重
        risk_components = [
            stagnation_evidence_raw,        # 行为层原始证据 (权重高)
            price_overextension_risk,       # 价格超买意图 (权重中高)
            upper_shadow_pressure_risk,     # 上影线抛压意图 (权重高)
            fund_flow_bearish_risk,         # 主力资金流出背离 (权重高)
            chip_dispersion_risk,           # 筹码集中度下降 (权重高)
            profit_taking_supply_risk,      # 获利盘供给压力 (权重中)
            trend_confirmation_risk,        # 趋势确认度下降 (权重中低)
            retail_fomo_risk                # 市场情绪亢奋 (权重中)
        ]
        weights = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05, 0.10])
        # 确保所有风险分量都是 Series，并且索引对齐
        aligned_risk_components = [comp.reindex(df_index, fill_value=0.0) for comp in risk_components]
        # 避免 log(0) 错误，将所有分量加上一个极小值
        safe_risk_components = [comp + 1e-9 for comp in aligned_risk_components]
        # 计算加权几何平均
        stagnation_risk_score = pd.Series(np.prod([comp.values ** w for comp, w in zip(safe_risk_components, weights)], axis=0), index=df_index)
        # 最终风险分只在价格上涨或横盘时有效，否则为0
        final_stagnation_risk = (stagnation_risk_score * is_price_stagnant_or_rising).clip(0, 1)
        states['FUSION_RISK_STAGNATION'] = final_stagnation_risk.astype(np.float32)
        print(f"  -- [融合层] “滞涨风险”冶炼完成，最新分值: {final_stagnation_risk.iloc[-1]:.4f}")
        return states

    def _synthesize_capital_confrontation(self) -> Dict[str, pd.Series]:
        """
        【V1.0】冶炼“资本对抗” (Capital Confrontation)
        - 核心思想: 深度洞察A股的博弈核心——主力与散户的对抗。
        - 证据链:
          1. 资金流对抗 (FundFlow): 主力与散户的资金流方向是否相反。
          2. 筹码转移 (Chip): 筹码是在集中还是在发散。
          3. 微观欺骗 (MicroBehavior): 是否存在“伪装成散户吸筹”等欺骗行为。
        """
        print("  -- [融合层] 正在冶炼“资本对抗”...")
        states = {}
        # 证据1: 资金流对抗 (来自资金流层)
        flow_confrontation = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        # 证据2: 筹码转移 (来自筹码层)
        chip_transfer = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        # 证据3: 微观欺骗 (来自微观行为层)
        deception = self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        # 融合三大博弈证据
        # 正分代表主力占优（吸筹、集中、欺骗性买入）
        # 负分代表散户占优（接盘、筹码发散）
        bipolar_confrontation = (flow_confrontation * 0.5 + chip_transfer * 0.3 + deception * 0.2).clip(-1, 1)
        states['FUSION_BIPOLAR_CAPITAL_CONFRONTATION'] = bipolar_confrontation.astype(np.float32)
        print(f"  -- [融合层] “资本对抗”冶炼完成，最新分值: {bipolar_confrontation.iloc[-1]:.4f}")
        return states

    def _synthesize_price_overextension_intent(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 深度博弈版】冶炼“价格超买意图” (Price Overextension Intent)
        - 核心思想: 综合判断价格偏离均线是强力进攻还是真实超买风险。
        - 证据链:
          1. 行为层价格超买原始分 (INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW)
          2. 资金流共识 (SCORE_FF_AXIOM_CONSENSUS)
          3. 筹码集中度 (SCORE_CHIP_AXIOM_CONCENTRATION)
          4. 结构趋势形态 (SCORE_STRUCT_AXIOM_TREND_FORM)
          5. 微观效率 (SCORE_MICRO_AXIOM_EFFICIENCY)
        """
        print("  -- [融合层] 正在冶炼“价格超买意图”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        # 1. 获取行为层价格超买原始分 (0到1，越高越超买)
        overextension_raw = self._get_atomic_score('INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.0)
        # 2. 获取其他维度的支持/抑制信号 (均为双极性 [-1, 1])
        fund_flow_consensus = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        micro_efficiency = self._get_atomic_score('SCORE_MICRO_AXIOM_EFFICIENCY', 0.0)
        # 3. 综合判断逻辑
        # 将 overextension_raw 从 [0, 1] 映射到 [-1, 1]，0.5为中性，1为极度超买
        overextension_bipolar = (overextension_raw * 2 - 1).clip(-1, 1)
        # 综合意图分数
        # 当 overextension_bipolar 为正（超买）时，如果支持信号强，则修正因子为正；如果压力信号强，则修正因子为负。
        # 最终意图 = overextension_bipolar * 基础权重 + 各维度修正
        overextension_intent = (
            overextension_bipolar * 0.4 + # 基础超买程度
            fund_flow_consensus * 0.2 + # 资金流向
            chip_concentration * 0.15 + # 筹码结构
            structural_trend_form * 0.15 + # 结构形态
            micro_efficiency * 0.1 # 微观效率
        ).clip(-1, 1)
        states['FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT'] = overextension_intent.astype(np.float32)
        print(f"  -- [融合层] “价格超买意图”冶炼完成，最新分值: {overextension_intent.iloc[-1]:.4f}")
        return states

    def _synthesize_upper_shadow_intent(self) -> Dict[str, pd.Series]: # 修改方法
        """
        【V2.0 · 深度博弈版】冶炼“上影线意图” (Upper Shadow Intent)
        - 核心思想: 综合判断上影线是真抛压（空头意图）还是洗盘/试探（多头意图）。
        - 证据链:
          1. 行为层上影线原始分 (INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW)
          2. 价格涨跌 (pct_change_D)
          3. 主力资金流向 (main_force_net_flow_calibrated_D)
          4. 筹码集中度 (SCORE_CHIP_AXIOM_CONCENTRATION)
          5. 微观欺骗 (SCORE_MICRO_AXIOM_DECEPTION)
          6. 上涨效率 (SCORE_BEHAVIOR_UPWARD_EFFICIENCY)
          7. 结构趋势形态 (SCORE_STRUCT_AXIOM_TREND_FORM)
          8. 成交量爆发 (SCORE_BEHAVIOR_VOLUME_BURST)
        - 输出: [-1, 1] 的双极性分数，负分代表真抛压风险，正分代表偏向多头意图（如洗盘、试探）。
        """
        print("  -- [融合层] 正在冶炼“上影线意图” (深度博弈版)...")
        states = {}
        df = self.strategy.df_indicators
        df_index = df.index
        norm_window = 55 # 统一归一化窗口

        # 1. 获取行为层上影线原始分 (0到1，越高上影线越强)
        upper_shadow_normalized = self._get_atomic_score('INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW', 0.0)

        # 2. 获取其他维度的支持/抑制信号 (转换为 [-1, 1] 双极性或 [0, 1] 单极性)
        pct_change = df.get('pct_change_D', pd.Series(0.0, index=df_index))
        main_force_flow = normalize_to_bipolar(df.get('main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index)), df_index, norm_window)
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0) # [-1, 1]
        micro_deception = self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0) # [-1, 1]
        upward_efficiency = (self._get_atomic_score('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5) * 2 - 1).clip(-1, 1) # 转换为双极性
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0) # [-1, 1]
        volume_burst = (self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_BURST', 0.5) * 2 - 1).clip(-1, 1) # 转换为双极性

        # 3. 价格背景判断
        is_up_day = (pct_change > 0).astype(float)
        is_down_day = (pct_change < 0).astype(float)
        is_flat_day = ((pct_change == 0) | (pct_change.abs() < 0.005)).astype(float) # 定义平盘日

        # 4. 计算多头意图分数 (Bullish Intent Score) - 范围 [0, 1]
        # 这些因素，当为正时，表明上影线背后是多头意图（洗盘、试探）
        bullish_intent_components = [
            main_force_flow.clip(lower=0), # 主力净流入
            chip_concentration.clip(lower=0), # 筹码集中
            micro_deception.clip(lower=0), # 微观欺骗为伪装派发（实为吸筹）
            upward_efficiency.clip(lower=0), # 上涨效率高
            structural_trend_form.clip(lower=0) # 结构趋势向上
        ]
        # 权重分配：主力资金和筹码集中度最为关键
        bullish_weights = np.array([0.35, 0.25, 0.2, 0.1, 0.1]) # 权重和为1
        bullish_intent_score = sum(w * s for w, s in zip(bullish_weights, bullish_intent_components)).clip(0, 1)

        # 5. 计算空头意图分数 (Bearish Intent Score) - 范围 [0, 1]
        # 这些因素，当为正时，表明上影线背后是空头意图（派发、诱多）
        bearish_intent_components = [
            main_force_flow.clip(upper=0).abs(), # 主力净流出
            chip_concentration.clip(upper=0).abs(), # 筹码分散
            micro_deception.clip(upper=0).abs(), # 微观欺骗为伪装吸筹（实为派发）
            (1 - upward_efficiency.clip(lower=0)), # 上涨效率低 (1-效率高)
            structural_trend_form.clip(upper=0).abs() # 结构趋势向下
        ]
        # 权重分配：同样主力资金和筹码分散度最为关键
        bearish_weights = np.array([0.35, 0.25, 0.2, 0.1, 0.1]) # 权重和为1
        bearish_intent_score = sum(w * s for w, s in zip(bearish_weights, bearish_intent_components)).clip(0, 1)

        # 6. 综合净意图方向 (Net Intent Direction) - 范围 [-1, 1]
        # 正值代表多头意图占优，负值代表空头意图占优
        net_intent_direction = (bullish_intent_score - bearish_intent_score).clip(-1, 1)

        # 7. 成交量信念乘数 (Volume Conviction Multiplier) - 范围 [0, 1]
        # 高量能 (volume_burst=1) 意味着信念乘数为1，低量能 (volume_burst=-1) 意味着信念乘数为0
        # 这样，低量能的上影线，无论意图如何，其最终信号强度都会被削弱
        volume_conviction_multiplier = (volume_burst + 1) / 2 # 将 [-1, 1] 映射到 [0, 1]

        # 8. 基础意图分数 (Base Intent Score)
        # 原始上影线强度 * 净意图方向 * 成交量信念乘数
        base_intent_score = upper_shadow_normalized * net_intent_direction * volume_conviction_multiplier

        # 9. 价格背景的条件调整
        final_intent = pd.Series(0.0, index=df_index)

        # 调整1: 上涨日 (is_up_day)
        # 上涨日的上影线，如果净意图为正（洗盘/试探），则放大其积极性；如果净意图为负（诱多/派发），则放大其消极性。
        up_day_adjusted_intent = base_intent_score * (1 + base_intent_score.abs() * 0.5) # 意图越明确，放大效果越强
        final_intent = final_intent.mask(is_up_day.astype(bool), up_day_adjusted_intent)

        # 调整2: 下跌日 (is_down_day)
        # 下跌日的上影线，通常是抛压。即使有少量多头证据，也应大幅削弱其积极性，并强化其消极性。
        # 基础惩罚：-upper_shadow_normalized * 0.8 (即使没有其他空头证据，也至少是较强的抛压)
        # 叠加空头意图：-upper_shadow_normalized * bearish_intent_score * 0.5
        # 少量多头缓解：+upper_shadow_normalized * bullish_intent_score * 0.2 (缓解作用有限)
        down_day_adjusted_intent = (
            -upper_shadow_normalized * 0.8 # 基础抛压
            - upper_shadow_normalized * bearish_intent_score * 0.5 # 叠加空头意图
            + upper_shadow_normalized * bullish_intent_score * 0.2 # 少量多头缓解
        ) * volume_conviction_multiplier # 依然受量能信念影响
        final_intent = final_intent.mask(is_down_day.astype(bool), down_day_adjusted_intent)

        # 调整3: 平盘日 (is_flat_day)
        # 平盘日的上影线，意图相对模糊，信号强度减半。
        flat_day_adjusted_intent = base_intent_score * 0.5
        final_intent = final_intent.mask(is_flat_day.astype(bool), flat_day_adjusted_intent)

        final_intent = final_intent.clip(-1, 1)
        states['FUSION_BIPOLAR_UPPER_SHADOW_INTENT'] = final_intent.astype(np.float32)
        print(f"  -- [融合层] “上影线意图”冶炼完成，最新分值: {final_intent.iloc[-1]:.4f}")
        return states
