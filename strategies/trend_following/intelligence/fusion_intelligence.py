import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_adaptive_mtf_normalized_bipolar_score

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
        """
        【V3.2 · 市场矛盾增强版】运行所有融合诊断任务。
        - 核心流程: 依次冶炼四大战场态势，并发布到原子状态库。
        - 【新增】冶炼“市场矛盾”态势。
        """
        print("启动【V3.0 · 战场态势引擎】融合情报分析...")
        all_fusion_states = {}
        # 步骤一: 冶炼“市场政权”
        regime_states = self._synthesize_market_regime()
        all_fusion_states.update(regime_states)
        # 步骤二: 冶炼“趋势质量”
        quality_states = self._synthesize_trend_quality()
        all_fusion_states.update(quality_states)
        # 步骤三: 冶炼“市场压力”
        pressure_states = self._synthesize_market_pressure()
        all_fusion_states.update(pressure_states)
        # 步骤四: 冶炼“资本对抗”
        confrontation_states = self._synthesize_capital_confrontation()
        all_fusion_states.update(confrontation_states)
        # 步骤五: 冶炼“市场矛盾”
        contradiction_states = self._synthesize_market_contradiction()
        all_fusion_states.update(contradiction_states)
        # 步骤六: 将新生成的融合信号立即发布，供后续认知层使用
        self.strategy.atomic_states.update(all_fusion_states)
        print(f"【V3.0 · 战场态势引擎】分析完成，生成 {len(all_fusion_states)} 个融合态势信号。")
        return all_fusion_states

    def _synthesize_market_contradiction(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】冶炼“市场矛盾” (Market Contradiction)
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
        print("  -- [融合层] 正在冶炼“趋势质量”...") # 新增行
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
        print(f"  -- [融合层] “趋势质量”冶炼完成，最新分值: {bipolar_quality.iloc[-1]:.4f}") # 新增行
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
        for signal_name in reversal_sources: # 修改行
            if 'BOTTOM_REVERSAL' in signal_name: # 修改行
                upward_pressure_scores.append(self._get_atomic_score(signal_name, 0.0).values) # 修改行
            elif 'TOP_REVERSAL' in signal_name: # 修改行
                downward_pressure_scores.append(self._get_atomic_score(signal_name, 0.0).values) # 修改行
        net_upward_pressure = np.maximum.reduce(upward_pressure_scores)
        net_downward_pressure = np.maximum.reduce(downward_pressure_scores)
        bipolar_pressure = (pd.Series(net_upward_pressure, index=self.strategy.df_indicators.index) -
                            pd.Series(net_downward_pressure, index=self.strategy.df_indicators.index)).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_PRESSURE'] = bipolar_pressure.astype(np.float32)
        print(f"  -- [融合层] “市场压力”冶炼完成，最新分值: {bipolar_pressure.iloc[-1]:.4f}")
        return states

    def _synthesize_capital_confrontation(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】冶炼“资本对抗” (Capital Confrontation)
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
