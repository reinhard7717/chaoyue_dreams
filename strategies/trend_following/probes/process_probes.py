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
        【探针 V1.1 · 配置访问修复版】主力成本优势趋势探针
        - 核心修复: 废除错误的硬编码参数路径，改为使用标准的 get_params_block 工具函数获取配置，
                      解决了因配置结构理解错误导致的 KeyError。
        """
        print(f"--- 探针启动: 解剖信号 '{signal_name}' 在 {probe_date.date()} 的状态 ---")
        # 使用 get_params_block 安全地获取参数块
        from strategies.trend_following.utils import get_params_block
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        config = next((c for c in process_params.get('diagnostics', []) if c.get('name') == signal_name), None)
        if not config:
            print(f"    [错误] 在配置中未找到信号 '{signal_name}' 的定义。")
            return
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        # 从正确的参数块中获取参数
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        std_window = process_params.get('std_window')
        bipolar_sensitivity = process_params.get('bipolar_sensitivity')
        meta_window = process_params.get('meta_window')
        norm_window = process_params.get('norm_window')
        trend_weight, accel_weight = process_params.get('meta_score_weights')
        signal_a = df.get(signal_a_name)
        signal_b = df.get(signal_b_name)
        if signal_a is None or signal_b is None:
            print(f"    [错误] 无法提取原始信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return
        print(f"\n[链路层 1] 原始信号值:")
        print(f"  - {signal_a_name}: {signal_a.get(probe_date):.4f}")
        print(f"  - {signal_b_name}: {signal_b.get(probe_date):.4f}")
        change_a = ta.percent_return(signal_a, length=1).fillna(0)
        change_b = signal_b.diff(1).fillna(0)
        momentum_a = normalize_to_bipolar(change_a, df.index, std_window, bipolar_sensitivity)
        thrust_b = normalize_to_bipolar(change_b, df.index, std_window, bipolar_sensitivity)
        relationship_score = (momentum_a + thrust_b) / 2
        relationship_score = relationship_score.clip(-1, 1)
        print(f"\n[链路层 2] 瞬时关系计算:")
        print(f"  - 瞬时关系分 (共识): {relationship_score.get(probe_date):.4f}")
        relationship_trend = ta.linreg(relationship_score, length=meta_window).fillna(0)
        relationship_accel = ta.linreg(relationship_trend, length=meta_window).fillna(0)
        bipolar_trend_strength = normalize_to_bipolar(relationship_trend, df.index, norm_window, bipolar_sensitivity)
        bipolar_accel_strength = normalize_to_bipolar(relationship_accel, df.index, norm_window, bipolar_sensitivity)
        print(f"\n[链路层 3] 关系元分析:")
        print(f"  - 归一化趋势强度: {bipolar_trend_strength.get(probe_date):.4f}")
        print(f"  - 归一化加速度强度: {bipolar_accel_strength.get(probe_date):.4f}")
        meta_score = (bipolar_trend_strength * trend_weight + bipolar_accel_strength * accel_weight)
        final_score = atomic_states.get(signal_name, pd.Series(np.nan, index=df.index)).get(probe_date)
        print(f"\n[链路层 4] 最终得分合成:")
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
    def _deploy_winner_conviction_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V2.1 · 精度校准版】赢家信念探针
        - 核心升级: 将最终对质的逻辑从直接比较升级为使用 numpy.isclose，
                      以兼容浮点数计算中产生的微小精度差异，避免误报。
        """
        signal_name = 'PROCESS_META_WINNER_CONVICTION'
        print("\n" + "="*25 + f" [过程探针] 正在启用 🙏【赢家信念探针 V2.1】🙏 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.intel_layer.process_intel
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        config = next((c for c in engine.diagnostics_config if c.get('name') == signal_name), None)
        if not config:
            print(f"    [错误] 在配置中未找到信号 '{signal_name}' 的定义。")
            return
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终信号分】: {actual_score:.4f}")
        print("\n  [链路层 2] 原始信号与动量计算 (Tri-Variable)")
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        antidote_signal_name = config.get('antidote_signal')
        signal_a_series = df.get(signal_a_name)
        signal_b_series = df.get(signal_b_name)
        antidote_signal_series = df.get(antidote_signal_name)
        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if series is None: return pd.Series(dtype=np.float32)
            if change_type == 'diff': return series.diff(1).fillna(0)
            return ta.percent_return(series, length=1).fillna(0)
        change_a = get_change_series(signal_a_series, config.get('change_type_A'))
        change_b = get_change_series(signal_b_series, config.get('change_type_B'))
        change_antidote = get_change_series(antidote_signal_series, config.get('antidote_change_type'))
        momentum_a = normalize_to_bipolar(change_a, df.index, engine.std_window, engine.bipolar_sensitivity)
        momentum_b_raw = normalize_to_bipolar(change_b, df.index, engine.std_window, engine.bipolar_sensitivity)
        momentum_antidote = normalize_to_bipolar(change_antidote, df.index, engine.std_window, engine.bipolar_sensitivity)
        print(f"    - [信号A: {signal_a_name}] 值: {get_val(signal_a_series, probe_date):.4f} -> 动量: {get_val(momentum_a, probe_date):.4f} (紧迫度)")
        print(f"    - [信号B: {signal_b_name}] 值: {get_val(signal_b_series, probe_date):.4f} -> 动量: {get_val(momentum_b_raw, probe_date):.4f} (原始利润)")
        print(f"    - [解毒剂: {antidote_signal_name}] 值: {get_val(antidote_signal_series, probe_date):.4f} -> 动量: {get_val(momentum_antidote, probe_date):.4f} (新赢家流入)")
        print("\n  [链路层 3] 解毒剂协议 (Antidote Protocol)")
        antidote_k = config.get('antidote_k', 1.0)
        momentum_b_corrected = momentum_b_raw + antidote_k * momentum_antidote
        print(f"    - 【修正后利润动量】: {get_val(momentum_b_raw, probe_date):.4f} + {antidote_k:.1f} * {get_val(momentum_antidote, probe_date):.4f} = {get_val(momentum_b_corrected, probe_date):.4f}")
        print("\n  [链路层 4] 瞬时关系分计算 (Instantaneous Relationship)")
        k = config.get('signal_b_factor_k', 1.0)
        relationship_score = (k * momentum_b_corrected - momentum_a) / (k + 1)
        relationship_score_val = get_val(relationship_score, probe_date)
        print(f"    - 【瞬时关系分 (背离)】: ({k:.1f} * {get_val(momentum_b_corrected, probe_date):.4f} - {get_val(momentum_a, probe_date):.4f}) / {k+1:.1f} = {relationship_score_val:.4f}")
        print("\n  [链路层 5] 关系元分析 (Meta-Analysis)")
        relationship_trend = ta.linreg(relationship_score, length=engine.meta_window).fillna(0)
        relationship_accel = ta.linreg(relationship_trend, length=engine.meta_window).fillna(0)
        bipolar_trend_strength = normalize_to_bipolar(relationship_trend, df.index, engine.norm_window, engine.bipolar_sensitivity)
        bipolar_accel_strength = normalize_to_bipolar(relationship_accel, df.index, engine.norm_window, engine.bipolar_sensitivity)
        print(f"    - [关系分趋势]: {get_val(relationship_trend, probe_date):.4f} -> 归一化趋势强度: {get_val(bipolar_trend_strength, probe_date):.4f}")
        print(f"    - [关系分加速度]: {get_val(relationship_accel, probe_date):.4f} -> 归一化加速度强度: {get_val(bipolar_accel_strength, probe_date):.4f}")
        print("\n  [链路层 6] 最终裁决与对质 (Final Adjudication & Verdict)")
        trend_weight, accel_weight = engine.meta_score_weights
        recalc_score = (get_val(bipolar_trend_strength, probe_date) * trend_weight + get_val(bipolar_accel_strength, probe_date) * accel_weight)
        recalc_score = np.clip(recalc_score, -1, 1)
        print(f"    - 【探针重算-最终分】: {get_val(bipolar_trend_strength, probe_date):.4f} * {trend_weight} + {get_val(bipolar_accel_strength, probe_date):.4f} * {accel_weight} = {recalc_score:.4f}")
        # 使用 isclose 替代直接比较，以容忍浮点数精度误差
        match = np.isclose(actual_score, recalc_score, atol=1e-4)
        print(f"    - [对比]: 系统最终值 {actual_score:.4f} vs. 探针正确值 {recalc_score:.4f} -> {'✅ 一致' if match else '❌ 不一致'}")
        print("\n--- “赢家信念探针”解剖完毕 ---")












