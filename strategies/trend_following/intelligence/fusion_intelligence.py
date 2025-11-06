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
            # [代码修改开始]
            # 默认值从 0.5 改为 0.0
            return pd.Series(default, index=self.strategy.df_indicators.index)
            # [代码修改结束]

    def run_fusion_diagnostics(self) -> Dict[str, pd.Series]:
        """
        【V3.0 · 战场态势引擎】运行所有融合诊断任务。
        - 核心流程: 依次冶炼四大战场态势，并发布到原子状态库。
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
        # 步骤五: 将新生成的融合信号立即发布，供后续认知层使用
        self.strategy.atomic_states.update(all_fusion_states)
        print(f"【V3.0 · 战场态势引擎】分析完成，生成 {len(all_fusion_states)} 个融合态势信号。")
        return all_fusion_states

    def _synthesize_market_regime(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】冶炼“市场政权” (Market Regime)
        - 核心思想: 融合“序列记忆性”和“趋势惯性”，判断市场是趋势主导还是均值回归主导。
        - 证据链:
          1. 记忆性 (Hurst): 价格序列的数学本质是趋势延续还是反转。
          2. 惯性 (ADX): 市场当前是否存在可见的、有力量的趋势。
          3. 稳定性 (Volatility): 稳定的市场环境更有利于趋势形成。
        """
        print("  -- [融合层] 正在冶炼“市场政权”...")
        states = {}
        # 证据1: 序列记忆性 (来自周期层)
        hurst_memory = self._get_atomic_score('SCORE_CYCLICAL_HURST_MEMORY', 0.0)
        # 证据2: 趋势惯性 (来自力学层)
        inertia = self._get_atomic_score('SCORE_DYN_AXIOM_INERTIA', 0.0)
        # 证据3: 市场稳定性 (来自力学层)
        stability = self._get_atomic_score('SCORE_DYN_AXIOM_STABILITY', 0.0)
        # 融合趋势证据: 记忆性为正(趋势) * 惯性为正(有趋势) * 市场稳定
        trend_evidence = (hurst_memory.clip(lower=0) * inertia.clip(lower=0) * stability.clip(lower=0)).pow(1/3)
        # 融合震荡证据: 记忆性为负(回归) * 惯性为负(无趋势)
        reversion_evidence = (hurst_memory.clip(upper=0).abs() * inertia.clip(upper=0).abs()).pow(1/2)
        # 最终裁决: 生成双极性市场政权分
        bipolar_regime = (trend_evidence - reversion_evidence).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_REGIME'] = bipolar_regime.astype(np.float32)
        print(f"  -- [融合层] “市场政权”冶炼完成，最新分值: {bipolar_regime.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_quality(self) -> Dict[str, pd.Series]:
        """
        【V1.1 · 级联探针版】冶炼“趋势质量” (Trend Quality)
        - 探针植入: 打印其依赖的所有领域共振分，以诊断融合结果为中性的根源。
        """
        print("  -- [融合层] 正在冶炼“趋势质量”...")
        states = {}
        resonance_sources = [
            'FOUNDATION', 'STRUCTURE', 'PATTERN', 'DYNAMIC_MECHANICS', 
            'CHIP', 'FUND_FLOW', 'MICRO_BEHAVIOR'
        ]
        bullish_scores = []
        bearish_scores = []
        
        # [代码新增开始]
        # --- 级联探针: 融合层 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            if self.strategy.df_indicators.index.tz:
                probe_date = probe_date_naive.tz_localize(self.strategy.df_indicators.index.tz)
            else:
                probe_date = probe_date_naive
            print(f"    -> [融合层探针] @ {probe_date.date()} 检查趋势质量的证据链:")
        # [代码新增结束]

        for source in resonance_sources:
            bull_signal_name = f'SCORE_{source}_BULLISH_RESONANCE'
            bear_signal_name = f'SCORE_{source}_BEARISH_RESONANCE'
            bull_score_series = self._get_atomic_score(bull_signal_name, 0.5)
            bear_score_series = self._get_atomic_score(bear_signal_name, 0.5)
            bullish_scores.append(bull_score_series.values)
            bearish_scores.append(bear_score_series.values)
            
            # [代码新增开始]
            if probe_date and probe_date in bull_score_series.index:
                bull_val = bull_score_series.loc[probe_date]
                bear_val = bear_score_series.loc[probe_date]
                print(f"       - {source:<18s} | 看涨共振: {bull_val:.4f} | 看跌共振: {bear_val:.4f}")
            # [代码新增结束]

        safe_bullish_scores = np.maximum(np.stack(bullish_scores), 1e-9)
        holistic_bullish_consensus = np.exp(np.mean(np.log(safe_bullish_scores), axis=0))
        safe_bearish_scores = np.maximum(np.stack(bearish_scores), 1e-9)
        holistic_bearish_consensus = np.exp(np.mean(np.log(safe_bearish_scores), axis=0))
        bipolar_quality = (pd.Series(holistic_bullish_consensus, index=self.strategy.df_indicators.index) - 
                           pd.Series(holistic_bearish_consensus, index=self.strategy.df_indicators.index)).clip(-1, 1)
        states['FUSION_BIPOLAR_TREND_QUALITY'] = bipolar_quality.astype(np.float32)
        print(f"  -- [融合层] “趋势质量”冶炼完成，最新分值: {bipolar_quality.iloc[-1]:.4f}")
        return states

    def _synthesize_market_pressure(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】冶炼“市场压力” (Market Pressure)
        - 核心思想: 衡量市场中“向上反转”与“向下回调”两股力量的净压力。
        - 证据链: 融合所有原子情报层的“底部反转”和“顶部反转”信号。
        """
        print("  -- [融合层] 正在冶炼“市场压力”...")
        states = {}
        # 定义所有领域的反转信号源
        reversal_sources = [
            'FOUNDATION', 'STRUCTURE', 'PATTERN', 'DYNAMIC_MECHANICS', 
            'CHIP', 'FUND_FLOW', 'MICRO_BEHAVIOR'
        ]
        upward_pressure_scores = []
        downward_pressure_scores = []
        for source in reversal_sources:
            # 动态构建信号名，例如 'SCORE_CHIP_BOTTOM_REVERSAL'
            bottom_signal_name = f'SCORE_{source}_BOTTOM_REVERSAL'
            top_signal_name = f'SCORE_{source}_TOP_REVERSAL'
            upward_pressure_scores.append(self._get_atomic_score(bottom_signal_name, 0.0).values)
            downward_pressure_scores.append(self._get_atomic_score(top_signal_name, 0.0).values)
        # 在各类压力内部，取最大值，代表最强的压力信号
        net_upward_pressure = np.maximum.reduce(upward_pressure_scores)
        net_downward_pressure = np.maximum.reduce(downward_pressure_scores)
        # 最终裁决: 生成双极性市场压力分
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
