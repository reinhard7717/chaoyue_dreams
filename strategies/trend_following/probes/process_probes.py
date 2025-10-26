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

    def _deploy_themis_scales_probe(self, probe_date: pd.Timestamp, signal_to_probe: str):
        """
        【V1.1 · 动态同步版】“忒弥斯的天平”探针 - 过程情报引擎深度解剖
        - 核心升级: 探针不再硬编码变化类型，而是从信号配置中动态读取。
        """
        print("\n" + "="*35 + f" [过程探针] 正在启用 ⚖️【过程引擎解剖 V1.1】⚖️ " + "="*35)
        print(f"  [目标信号]: {signal_to_probe}")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.process_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        target_config = next((c for c in engine.diagnostics_config if c.get('name') == signal_to_probe), None)
        if not target_config:
            print(f"  [错误] 在过程引擎配置中未找到信号 '{signal_to_probe}' 的定义。")
            return
        change_type_a = target_config.get('change_type_A', 'pct')
        change_type_b = target_config.get('change_type_B', 'pct')
        final_score_actual = get_val(atomic.get(signal_to_probe), probe_date, 0.0)
        print(f"    - 【最终得分】: {final_score_actual:.4f}")
        signal_a_name = target_config.get('signal_A')
        signal_b_name = target_config.get('signal_B')
        k = target_config.get('signal_b_factor_k', 1.0)
        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if series is None: return pd.Series(dtype=float)
            if change_type == 'diff':
                return series.diff(1).fillna(0)
            return ta.percent_return(series, length=1).fillna(0)
        signal_a_series = df.get(signal_a_name)
        signal_b_series = df.get(signal_b_name)
        change_a = get_change_series(signal_a_series, change_type_a)
        change_b = get_change_series(signal_b_series, change_type_b)
        momentum_a = normalize_to_bipolar(change_a, df.index, engine.std_window, engine.bipolar_sensitivity)
        thrust_b = normalize_to_bipolar(change_b, df.index, engine.std_window, engine.bipolar_sensitivity)
        momentum_a_val = get_val(momentum_a, probe_date)
        thrust_b_val = get_val(thrust_b, probe_date)
        recalc_score_unclipped = (k * thrust_b_val - momentum_a_val) / (k + 1)
        recalc_score_clipped = np.clip(recalc_score_unclipped, -1, 1)
        print(f"    - [探针重算]: {recalc_score_clipped:.4f}")
        print("\n--- “过程引擎探针”解剖完毕 ---")

    def _deploy_process_sync_probe(self, probe_date: pd.Timestamp, signal_to_probe: str):
        """
        【V1.0】过程同步探针
        - 核心使命: 深度解剖 'strategy_sync' 类型的过程信号。
        """
        print("\n" + "="*35 + f" [过程探针] 正在启用 🔗【过程同步探针 V1.0】🔗 " + "="*35)
        print(f"  [目标信号]: {signal_to_probe}")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.process_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        target_config = next((c for c in engine.diagnostics_config if c.get('name') == signal_to_probe), None)
        if not target_config:
            print(f"  [错误] 在过程引擎配置中未找到信号 '{signal_to_probe}' 的定义。")
            return
        actual_final_score = get_val(atomic.get(signal_to_probe), probe_date, 0.0)
        print(f"    - 【最终同步分】: {actual_final_score:.4f}")
        relationship_series = engine._calculate_strategy_sync_relationship(df, target_config)
        relationship_trend = ta.linreg(relationship_series, length=engine.meta_window).fillna(0)
        relationship_accel = ta.linreg(relationship_trend, length=engine.meta_window).fillna(0)
        bipolar_trend_strength = normalize_to_bipolar(relationship_trend, df.index, engine.norm_window, engine.bipolar_sensitivity)
        bipolar_accel_strength = normalize_to_bipolar(relationship_accel, df.index, engine.norm_window, engine.bipolar_sensitivity)
        trend_weight, accel_weight = engine.meta_score_weights
        recalc_meta_score = (get_val(bipolar_trend_strength, probe_date) * trend_weight + get_val(bipolar_accel_strength, probe_date) * accel_weight)
        recalc_meta_score = np.clip(recalc_meta_score, -1, 1)
        # 根据信号分裂逻辑重算
        recalc_final_score = max(0, -recalc_meta_score) # DECAY信号取负值部分
        print(f"    - [元分析得分]: {recalc_meta_score:.4f} -> [最终风险分]: {recalc_final_score:.4f}")
        print("\n--- “过程同步探针”解剖完毕 ---")












