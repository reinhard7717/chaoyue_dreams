# strategies\trend_following\forensic_probes\process_probes.py

# [代码新增开始]
import pandas as pd
import numpy as np
import pandas_ta as ta
from strategies.trend_following.utils import normalize_to_bipolar

class ProcessProbes:
    """
    【探针模块 V1.0】过程情报专属探针
    - 核心职责: 提供对 ProcessIntelligence 引擎输出的元信号进行深度解剖和诊断的能力。
    """
    def __init__(self, intel_layer):
        self.intel_layer = intel_layer
        self.strategy = intel_layer.strategy

    def _deploy_cost_advantage_probe(self, probe_date: pd.Timestamp, signal_name: str):
        """
        【探针】主力成本优势趋势探针
        - 诊断信号: PROCESS_META_COST_ADVANTAGE_TREND
        - 核心逻辑: 解剖“主力成本优势”与“价格”的共识关系元分析过程。
        """
        print(f"--- 探针启动: 解剖信号 '{signal_name}' 在 {probe_date.date()} 的状态 ---")
        config = next((c for c in self.strategy.params['process_intelligence_params']['diagnostics'] if c.get('name') == signal_name), None)
        if not config:
            print(f"    [错误] 在配置中未找到信号 '{signal_name}' 的定义。")
            return
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        # 提取所需参数
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        std_window = self.strategy.params['process_intelligence_params']['std_window']
        bipolar_sensitivity = self.strategy.params['process_intelligence_params']['bipolar_sensitivity']
        meta_window = self.strategy.params['process_intelligence_params']['meta_window']
        norm_window = self.strategy.params['process_intelligence_params']['norm_window']
        trend_weight, accel_weight = self.strategy.params['process_intelligence_params']['meta_score_weights']
        # 链路层 1: 提取原始信号
        signal_a = df.get(signal_a_name)
        signal_b = df.get(signal_b_name)
        if signal_a is None or signal_b is None:
            print(f"    [错误] 无法提取原始信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return
        print(f"\n[链路层 1] 原始信号值:")
        print(f"  - {signal_a_name}: {signal_a.get(probe_date):.4f}")
        print(f"  - {signal_b_name}: {signal_b.get(probe_date):.4f}")
        # 链路层 2: 计算瞬时关系分
        change_a = ta.percent_return(signal_a, length=1).fillna(0)
        change_b = signal_b.diff(1).fillna(0)
        momentum_a = normalize_to_bipolar(change_a, df.index, std_window, bipolar_sensitivity)
        thrust_b = normalize_to_bipolar(change_b, df.index, std_window, bipolar_sensitivity)
        relationship_score = (momentum_a + thrust_b) / 2
        relationship_score = relationship_score.clip(-1, 1)
        print(f"\n[链路层 2] 瞬时关系计算:")
        print(f"  - {signal_a_name} 变化率: {change_a.get(probe_date):.4f} -> 动量分: {momentum_a.get(probe_date):.4f}")
        print(f"  - {signal_b_name} 变化值: {change_b.get(probe_date):.4f} -> 推力分: {thrust_b.get(probe_date):.4f}")
        print(f"  - 瞬时关系分 (共识): {relationship_score.get(probe_date):.4f}")
        # 链路层 3: 动态元分析
        relationship_trend = ta.linreg(relationship_score, length=meta_window).fillna(0)
        relationship_accel = ta.linreg(relationship_trend, length=meta_window).fillna(0)
        bipolar_trend_strength = normalize_to_bipolar(relationship_trend, df.index, norm_window, bipolar_sensitivity)
        bipolar_accel_strength = normalize_to_bipolar(relationship_accel, df.index, norm_window, bipolar_sensitivity)
        print(f"\n[链路层 3] 关系元分析:")
        print(f"  - 关系分趋势 (原始斜率): {relationship_trend.get(probe_date):.4f} -> 归一化趋势强度: {bipolar_trend_strength.get(probe_date):.4f}")
        print(f"  - 关系分加速度 (趋势斜率): {relationship_accel.get(probe_date):.4f} -> 归一化加速度强度: {bipolar_accel_strength.get(probe_date):.4f}")
        # 链路层 4: 最终得分
        meta_score = (bipolar_trend_strength * trend_weight + bipolar_accel_strength * accel_weight)
        final_score = atomic_states.get(signal_name, pd.Series(np.nan, index=df.index)).get(probe_date)
        print(f"\n[链路层 4] 最终得分合成:")
        print(f"  - 融合公式: (趋势强度 * {trend_weight}) + (加速度强度 * {accel_weight})")
        print(f"  - 计算过程: ({bipolar_trend_strength.get(probe_date):.4f} * {trend_weight}) + ({bipolar_accel_strength.get(probe_date):.4f} * {accel_weight}) = {meta_score.get(probe_date):.4f}")
        print(f"  - 最终原子状态分 ({signal_name}): {final_score:.4f}")
        print(f"--- 探针结束: '{signal_name}' 解剖完毕 ---\n")














