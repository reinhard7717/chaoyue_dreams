# 文件: strategies/trend_following/intelligence/intraday_behavior_engine.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional, Any
# 导入 get_params_block 工具
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_robust_bipolar_normalized_score, 
    get_adaptive_mtf_normalized_bipolar_score, is_limit_up, get_adaptive_mtf_normalized_score, 
    normalize_score
)

class IntradayBehaviorEngine:
    """
    【V4.0 · 日内诡道引擎版】
    - 核心升级: 在“日内叙事”基础上，引入基于主力诡道博弈的“伏击与侧翼”、“终末强袭”、“VWAP攻防”三大全新公理。
                旨在穿透全天战果的表象，深度解读主力资金在日内的完整战术剧本，为T+1决策提供更高维度的博弈洞察。
    """
    def __init__(self, strategy_instance):
        """初始化时加载专属配置，并获取指标计算器的引用"""
        self.strategy = strategy_instance
        self.calculator = strategy_instance.orchestrator.indicator_service.calculator
        self.params = get_params_block(self.strategy, 'intraday_behavior_engine_params', {})

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [日内行为情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # 调整校验信息为“日内行为情报校验”
            print(f"    -> [日内行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_intraday_diagnostics(self, df: pd.DataFrame) -> Dict[str, pd.Series]: # 移除 async
        """
        【V4.3 · 同步执行版】日内诊断总指挥
        - 核心重构: 移除所有 async/await 关键字，改为同步执行，以符合CPU密集型任务的最佳实践并与其他情报引擎保持一致。
        """
        # --- 引擎启动探针 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        processed_date_str = "未知日期"
        if not df.empty:
            start_date_str = df.index.min().strftime('%Y-%m-%d')
            end_date_str = df.index.max().strftime('%Y-%m-%d')
            processed_date_str = f"{start_date_str} to {end_date_str}"
        if is_debug_enabled and probe_dates:
            print(f"  [日内行为引擎探针] run_intraday_diagnostics @ {processed_date_str}")
            print(f"    - 引擎已启动。日线数据是否为空: {df.empty}")
            if not df.empty:
                print(f"    - 日线数据行数: {len(df)}")
        # --- 探针结束 ---
        print("启动【V4.3 · 同步执行版】日内行为诊断...") # 更新版本号和描述
        # 移除对 _prepare_intraday_indicators 的调用
        if df is None or df.empty:
            print("日线数据为空，无法进行日内行为诊断。")
            return {
                "SCORE_INTRADAY_OFFENSIVE_PURITY": pd.Series(dtype=np.float64),
                "SCORE_INTRADAY_DOMINANCE_CONSENSUS": pd.Series(dtype=np.float64),
                "SCORE_INTRADAY_CONVICTION_REVERSAL": pd.Series(dtype=np.float64),
                "SCORE_INTRADAY_TACTICAL_ARC": pd.Series(dtype=np.float64),
                "SCORE_INTRADAY_AUCTION_INTENT": pd.Series(dtype=np.float64),
                "SCORE_INTRADAY_RECOVERY_QUALITY": pd.Series(dtype=np.float64),
                "SCORE_INTRADAY_AMBUSH_AND_FLANK": pd.Series(dtype=np.float64),
                "SCORE_INTRADAY_FINAL_ASSAULT": pd.Series(dtype=np.float64),
                "SCORE_INTRADAY_VWAP_BATTLEFIELD": pd.Series(dtype=np.float64),
            }
        diagnostics_to_run = [
            self._diagnose_offensive_purity,
            self._diagnose_dominance_consensus,
            self._diagnose_conviction_reversal,
            self._diagnose_tactical_arc,
            self._diagnose_auction_intent,
            self._diagnose_recovery_quality,
            self._diagnose_ambush_and_flank,
            self._diagnose_final_assault,
            self._diagnose_vwap_battlefield,
       ]
        final_scores = {}
        for diagnostic_func in diagnostics_to_run:
            result = diagnostic_func(df)
            final_scores.update(result)
        print(f"日内行为诊断完成，生成 {len(final_scores)} 个信号序列。")
        return final_scores

    def _diagnose_offensive_purity(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.1 · Production Ready版】日内战报之一：诊断“进攻纯度”
        - 核心重构: 引入“裁决放大”机制。品质分不再是简单的调节器，而是放大器。
                      1. 将“核心进攻得分”向量化至[-1, 1]，明确胜负方向。
                      2. 将“品质调节器”转换为“品质放大器”，值域如[0.5, 1.5]。
                      3. 最终向量 = 核心向量 * 品质放大器。此举能精准识别“胜者愈胜”，
                         并深刻洞察“在弱抵抗下失败是更大的失败”这一高级博弈逻辑。
        """
        signal_name = "SCORE_INTRADAY_OFFENSIVE_PURITY"
        required_signals = [
            'opening_battle_result_D',
            'vwap_control_strength_D',
            'upper_shadow_selling_pressure_D',
            'closing_strength_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_offensive_purity"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        # --- 获取参数 ---
        parent_params = get_params_block(self.strategy, 'intraday_behavior_params', {})
        params = get_param_value(parent_params.get('offensive_purity_params'), {})
        axis_weights = get_param_value(params.get('primary_axis_weights'), {'opening': 0.2, 'control': 0.5, 'closing': 0.3})
        # --- 反脆弱归一化层 ---
        mtf_params = get_params_block(self.strategy, 'behavioral_dynamics_params', {}).get('mtf_normalization_params', {})
        default_weights = mtf_params.get('default_weights')
        raw_opening_intent = self._get_safe_series(df, 'opening_battle_result_D', 0.0, "_diagnose_offensive_purity")
        raw_midday_control = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, "_diagnose_offensive_purity")
        raw_upper_shadow_pressure = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, "_diagnose_offensive_purity")
        raw_closing_power = self._get_safe_series(df, 'closing_strength_index_D', 0.0, "_diagnose_offensive_purity")
        norm_opening_intent = get_adaptive_mtf_normalized_bipolar_score(raw_opening_intent, df.index, default_weights)
        norm_midday_control = get_adaptive_mtf_normalized_bipolar_score(raw_midday_control, df.index, default_weights)
        norm_pressure_suppression = get_adaptive_mtf_normalized_score(raw_upper_shadow_pressure, df.index, ascending=False, tf_weights=default_weights)
        norm_closing_power = get_adaptive_mtf_normalized_bipolar_score(raw_closing_power, df.index, default_weights)
        # --- 裁决放大逻辑 ---
        # 1. 计算“核心进攻得分” ([0, 1] 区间)
        opening_score = (norm_opening_intent + 1) / 2
        control_score = (norm_midday_control + 1) / 2
        closing_score = (norm_closing_power + 1) / 2
        primary_axis_score = (
            opening_score * axis_weights.get('opening', 0.2) +
            control_score * axis_weights.get('control', 0.5) +
            closing_score * axis_weights.get('closing', 0.3)
        )
        # 2. 将核心得分向量化至 [-1, 1]
        primary_axis_vector = (primary_axis_score - 0.5) * 2
        # 3. 将品质分转换为 [0.5, 1.5] 区间的放大器
        quality_amplifier = 0.5 + norm_pressure_suppression
        # 4. 最终裁决：核心向量 * 品质放大器
        final_vector = (primary_axis_vector * quality_amplifier).fillna(0.0)
        # 5. 将最终向量映射回 [0, 1] 区间作为最终得分
        final_score = (final_vector / 2) + 0.5
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        return {signal_name: final_score.clip(0, 1)}

    def _diagnose_dominance_consensus(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.4 · Production Ready版】日内战报之二：诊断“支配共识”
        - 核心重构: 沿用V4.3的“预测向量”模型，并调用经过“临界点豁免”最终淬火的
                      V1.1版`get_robust_bipolar_normalized_score`工具，确保系统绝对鲁棒。
        """
        signal_name = "SCORE_INTRADAY_DOMINANCE_CONSENSUS"
        required_signals = ['vwap_control_strength_D', 'SLOPE_5_main_force_conviction_index_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_dominance_consensus"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        # --- 获取参数 ---
        parent_params = get_params_block(self.strategy, 'intraday_behavior_params', {})
        params = get_param_value(parent_params.get('dominance_consensus_params'), {})
        weights = get_param_value(params.get('vector_weights'), {'state': 0.6, 'trend': 0.4})
        # --- [核心逻辑] ---
        # 1. 获取原料信号
        dominance_state_vector = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, "_diagnose_dominance_consensus")
        raw_conviction_trend = self._get_safe_series(df, 'SLOPE_5_main_force_conviction_index_D', 0.0, "_diagnose_dominance_consensus")
        # 2. 标准化趋势原料，生成[-1, 1]区间的“共识趋势向量”
        conviction_trend_vector = get_robust_bipolar_normalized_score(raw_conviction_trend, df.index, window=55, sensitivity=1.0)
        # 3. 向量加权合成
        final_score = (
            dominance_state_vector * weights.get('state', 0.6) +
            conviction_trend_vector * weights.get('trend', 0.4)
        ).fillna(0.0)
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        return {signal_name: final_score.clip(-1, 1)}

    def _diagnose_conviction_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 非对称战争版】日内战报之三：诊断“信念反转”
        - 核心重构: 废弃V1.5的几何平均模型和对主力Alpha的错误解读。
                      进化为基于“非对称博弈”的全新诊断框架。
                      1. 看涨(底部): 诊断“恐慌下的优质承接”，采用加法模型，由正Alpha作为可靠性裁决。
                      2. 看跌(顶部): 诊断“信念崩溃下的从容派发”，彻底修正逻辑，将“正Alpha派发”作为最危险的顶部信号。
        """
        signal_name = "SCORE_INTRADAY_CONVICTION_REVERSAL"
        required_signals = [
            'panic_selling_cascade_D', 'capitulation_absorption_index_D',
            'main_force_execution_alpha_D', 'rally_distribution_pressure_D',
            'SLOPE_5_main_force_conviction_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_conviction_reversal"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        # --- 获取参数与归一化配置 ---
        mtf_params = get_params_block(self.strategy, 'behavioral_dynamics_params', {}).get('mtf_normalization_params', {})
        default_weights = mtf_params.get('default_weights')
        # --- [核心逻辑] V2.0 ---
        # 1. 获取所有原料信号
        raw_panic = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, "_diagnose_conviction_reversal")
        raw_absorption = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, "_diagnose_conviction_reversal")
        raw_mf_alpha = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, "_diagnose_conviction_reversal")
        raw_distribution = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, "_diagnose_conviction_reversal")
        raw_conviction_slope = self._get_safe_series(df, 'SLOPE_5_main_force_conviction_index_D', 0.0, "_diagnose_conviction_reversal")
        # 2. 信号归一化处理
        norm_panic = get_adaptive_mtf_normalized_score(raw_panic, df.index, tf_weights=default_weights)
        norm_absorption = get_adaptive_mtf_normalized_score(raw_absorption, df.index, tf_weights=default_weights)
        norm_distribution = get_adaptive_mtf_normalized_score(raw_distribution, df.index, tf_weights=default_weights)
        norm_conviction_decay = get_adaptive_mtf_normalized_score(-raw_conviction_slope.clip(upper=0), df.index, tf_weights=default_weights)
        # 对双极性的主力Alpha进行归一化
        norm_mf_alpha = get_robust_bipolar_normalized_score(raw_mf_alpha, df.index, window=55)
        # 3. 构建非对称模型
        # 3.1 看涨诊断: 恐慌下的优质承接
        base_bullish_score = (norm_panic * 0.5 + norm_absorption * 0.5)
        bullish_alpha_amplifier = (1 + norm_mf_alpha.clip(lower=0) * 0.5) # 正Alpha作为放大器
        bullish_reversal_score = (base_bullish_score * bullish_alpha_amplifier).fillna(0.0)
        # 3.2 看跌诊断: 信念崩溃下的从容派发
        base_bearish_score = (norm_distribution * 0.5 + norm_conviction_decay * 0.5)
        bearish_alpha_amplifier = (1 + norm_mf_alpha.clip(lower=0) * 1.0) # 正Alpha作为更强的放大器
        bearish_reversal_score = (base_bearish_score * bearish_alpha_amplifier).fillna(0.0)
        # 4. 最终裁决
        final_score = (bullish_reversal_score - bearish_reversal_score).fillna(0.0)
        # --- [探针逻辑] 暴露所有计算节点 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        print(f"      [日内行为探针 V2.0] _diagnose_conviction_reversal @ {probe_date_str}")
                        # --- 看涨诊断 ---
                        p_raw_panic = raw_panic.get(probe_date, 0.0)
                        p_norm_panic = norm_panic.get(probe_date, 0.0)
                        p_raw_absorption = raw_absorption.get(probe_date, 0.0)
                        p_norm_absorption = norm_absorption.get(probe_date, 0.0)
                        p_base_bullish = base_bullish_score.get(probe_date, 0.0)
                        p_bullish_amp = bullish_alpha_amplifier.get(probe_date, 1.0)
                        p_bullish_final = bullish_reversal_score.get(probe_date, 0.0)
                        print(f"        --- [看涨诊断：恐慌下的优质承接] ---")
                        print(f"          - 原料-恐慌: {p_raw_panic:.4f} -> 归一化: {p_norm_panic:.4f}")
                        print(f"          - 原料-承接: {p_raw_absorption:.4f} -> 归一化: {p_norm_absorption:.4f}")
                        print(f"          - 关键节点-基础分 (恐慌+承接): {p_base_bullish:.4f}")
                        print(f"          - 关键节点-Alpha放大器: {p_bullish_amp:.4f}")
                        print(f"          - 结果-看涨分: {p_bullish_final:.4f}")
                        # --- 看跌诊断 ---
                        p_raw_dist = raw_distribution.get(probe_date, 0.0)
                        p_norm_dist = norm_distribution.get(probe_date, 0.0)
                        p_raw_conv_slope = raw_conviction_slope.get(probe_date, 0.0)
                        p_norm_conv_decay = norm_conviction_decay.get(probe_date, 0.0)
                        p_base_bearish = base_bearish_score.get(probe_date, 0.0)
                        p_bearish_amp = bearish_alpha_amplifier.get(probe_date, 1.0)
                        p_bearish_final = bearish_reversal_score.get(probe_date, 0.0)
                        print(f"        --- [看跌诊断：信念崩溃下的从容派发] ---")
                        print(f"          - 原料-派发: {p_raw_dist:.4f} -> 归一化: {p_norm_dist:.4f}")
                        print(f"          - 原料-信念斜率: {p_raw_conv_slope:.4f} -> 归一化信念衰减: {p_norm_conv_decay:.4f}")
                        print(f"          - 关键节点-基础分 (派发+衰减): {p_base_bearish:.4f}")
                        print(f"          - 关键节点-Alpha放大器: {p_bearish_amp:.4f}")
                        print(f"          - 结果-看跌分: {p_bearish_final:.4f}")
                        # --- Alpha裁决者 ---
                        p_raw_alpha = raw_mf_alpha.get(probe_date, 0.0)
                        p_norm_alpha = norm_mf_alpha.get(probe_date, 0.0)
                        print(f"        --- [Alpha裁决者] ---")
                        print(f"          - 原料-主力Alpha: {p_raw_alpha:.4f} -> 归一化: {p_norm_alpha:.4f}")
                        # --- 最终裁决 ---
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"        --- [最终裁决] ---")
                        print(f"        - 最终信念反转分 (看涨分 - 看跌分): {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_conviction_reversal 处理日期 {probe_date_str} 失败: {e}")
        return {signal_name: final_score.clip(-1, 1)}

    def _diagnose_tactical_arc(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 代理信号重构版】日内叙事之一：诊断“战术弧线”
        - 核心重构: 由于预计算信号不存在，本方法重构为使用 `closing_strength_index_D` 作为核心代理，
                      因为它反映了全天力量博弈的最终结果。
        """
        signal_name = "SCORE_INTRADAY_TACTICAL_ARC"
        # 使用可用的代理信号
        raw_signal_name = "closing_strength_index_D"
        if not self._validate_required_signals(df, [raw_signal_name], "_diagnose_tactical_arc"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        
        final_score = self._get_safe_series(df, raw_signal_name, 0.0, "_diagnose_tactical_arc")
        
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        score_on_date = final_score.get(probe_date, 'N/A')
                        print(f"      [日内行为探针] _diagnose_tactical_arc @ {probe_date_str}")
                        print(f"        - [代理] 收盘强度指数 ({raw_signal_name}): {score_on_date:.4f}")
                        print(f"        - 最终战术弧线分: {score_on_date:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_tactical_arc 处理日期 {probe_date_str} 失败: {e}")
        
        return {signal_name: final_score}

    def _diagnose_auction_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 向量化重构版】日内叙事之二：诊断“竞价意图”
        - 核心重构: 对整个方法进行向量化重构，使其能够处理完整的日线数据DataFrame并为每一天生成分数。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        if df.empty:
            return {"SCORE_INTRADAY_AUCTION_INTENT": pd.Series(dtype=np.float64)}
        # 向量化计算
        # 从日线数据中获取开盘和收盘的博弈信号Series
        opening_intent = self._get_safe_series(df, 'opening_battle_result_D', 0.0, "_diagnose_auction_intent")
        closing_intent = self._get_safe_series(df, 'closing_auction_ambush_D', 0.0, "_diagnose_auction_intent")
        # 从配置中获取权重
        params = get_params_block(self.strategy, 'intraday_narrative_engine_params', {})
        weights = get_param_value(params.get('auction_intent_weights'), {'opening': 0.4, 'closing': 0.6})
        # 加权融合
        final_score = (opening_intent * weights.get('opening', 0.4) +
                       closing_intent * weights.get('closing', 0.6))
        # --- 重构探针逻辑 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_opening_intent = opening_intent.get(probe_date, 0.0)
                        p_closing_intent = closing_intent.get(probe_date, 0.0)
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"      [日内行为探针] _diagnose_auction_intent @ {probe_date_str}")
                        print(f"        - 开盘博弈结果分: {p_opening_intent:.4f}")
                        print(f"        - 收盘竞价偷袭分: {p_closing_intent:.4f}")
                        print(f"        - 最终竞价意图分: {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_auction_intent 处理日期 {probe_date_str} 失败: {e}")
        return {"SCORE_INTRADAY_AUCTION_INTENT": final_score.clip(-1, 1)}

    def _diagnose_recovery_quality(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 代理信号重构版】日内叙事之三：诊断“修复质量”
        - 核心重构: 由于预计算信号不存在，本方法重构为融合“下影线承接强度”和“逢低吸筹力量”
                      来综合评估日内下跌后的修复质量。
        """
        signal_name = "SCORE_INTRADAY_RECOVERY_QUALITY"
        # 使用可用的代理信号进行计算
        required_signals = ['lower_shadow_absorption_strength_D', 'dip_absorption_power_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_recovery_quality"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        
        lower_shadow = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, "_diagnose_recovery_quality")
        dip_absorption = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, "_diagnose_recovery_quality")
        
        # 融合逻辑：高质量的修复体现在下影线和整体低位区都有强力承接
        final_score = (lower_shadow * 0.5 + dip_absorption * 0.5).fillna(0.0)
        
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_lower_shadow = lower_shadow.get(probe_date, 0.0)
                        p_dip_absorption = dip_absorption.get(probe_date, 0.0)
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"      [日内行为探针] _diagnose_recovery_quality @ {probe_date_str}")
                        print(f"        - [代理] 下影线承接强度: {p_lower_shadow:.4f}")
                        print(f"        - [代理] 逢低吸筹力量: {p_dip_absorption:.4f}")
                        print(f"        - 最终修复质量分: {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_recovery_quality 处理日期 {probe_date_str} 失败: {e}")
        
        return {signal_name: final_score.clip(0, 1)}

    def _diagnose_ambush_and_flank(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 向量化重构版】日内诡道之一：诊断“伏击与侧翼”
        - 核心重构: 对整个方法进行向量化重构，使其能够处理完整的日线数据DataFrame并为每一天生成分数。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        if df.empty:
            return {"SCORE_INTRADAY_AMBUSH_AND_FLANK": pd.Series(dtype=np.float64)}
        # 向量化计算
        params = get_params_block(self.strategy, 'intraday_gambit_engine_params', {}).get('ambush_flank_params', {})
        weights = params.get('fusion_weights', {'panic_evidence': 0.2, 'absorption_power': 0.4, 'recovery_strength': 0.4})
        min_dip_pct = params.get('min_dip_to_open_pct', 0.03)
        daily_open = self._get_safe_series(df, 'open_D', 0.0, "_diagnose_ambush_and_flank")
        daily_low = self._get_safe_series(df, 'low_D', 0.0, "_diagnose_ambush_and_flank")
        # 门控条件：当日必须有足够的下探幅度
        gate_condition = (daily_open > 0) & (((daily_open - daily_low) / daily_open) >= min_dip_pct)
        # 直接从日线信号获取三大核心证据
        panic_evidence = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, "_diagnose_ambush_and_flank")
        absorption_power = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, "_diagnose_ambush_and_flank")
        recovery_strength = self._get_safe_series(df, 'closing_strength_index_D', 0.0, "_diagnose_ambush_and_flank")
        # 融合计算
        final_score = (panic_evidence.pow(weights.get('panic_evidence', 0.2)) *
                       absorption_power.pow(weights.get('absorption_power', 0.4)) *
                       recovery_strength.pow(weights.get('recovery_strength', 0.4))).where(gate_condition, 0.0).fillna(0.0)
        # --- 重构探针逻辑 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_gate = gate_condition.get(probe_date, False)
                        if not p_gate:
                            print(f"      [日内诡道探针] _diagnose_ambush_and_flank @ {probe_date_str} -> 未触发 (下探幅度不足)")
                            continue
                        p_panic_evidence = panic_evidence.get(probe_date, 0.0)
                        p_absorption_power = absorption_power.get(probe_date, 0.0)
                        p_recovery_strength = recovery_strength.get(probe_date, 0.0)
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"      [日内诡道探针] _diagnose_ambush_and_flank @ {probe_date_str}")
                        print(f"        - [证据] 恐慌抛售级联 (panic_selling_cascade_D): {p_panic_evidence:.4f}")
                        print(f"        - [证据] 逢低吸筹力量 (dip_absorption_power_D): {p_absorption_power:.4f}")
                        print(f"        - [证据] 收盘强度指数 (closing_strength_index_D): {p_recovery_strength:.4f}")
                        print(f"        - 最终伏击与侧翼分: {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_ambush_and_flank 处理日期 {probe_date_str} 失败: {e}")
        return {"SCORE_INTRADAY_AMBUSH_AND_FLANK": final_score.clip(0, 1)}

    def _diagnose_final_assault(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 代理信号重构版】日内诡道之二：诊断“终末强袭”
        - 核心重构: 由于预计算信号不存在，本方法重构为融合“收盘竞价偷袭”和“收盘前动态”这两个信号。
        """
        signal_name = "SCORE_INTRADAY_FINAL_ASSAULT"
        # 使用可用的代理信号
        required_signals = ['closing_auction_ambush_D', 'pre_closing_posturing_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_final_assault"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        
        params = get_params_block(self.strategy, 'intraday_gambit_engine_params', {}).get('final_assault_params', {})
        # 调整权重以适应新的信号
        weights = params.get('fusion_weights', {'closing_auction': 0.6, 'pre_closing': 0.4})
        
        closing_auction_intent = self._get_safe_series(df, 'closing_auction_ambush_D', 0.0, "_diagnose_final_assault")
        pre_closing_posturing = self._get_safe_series(df, 'pre_closing_posturing_D', 0.0, "_diagnose_final_assault")
        
        final_score = (closing_auction_intent * weights.get('closing_auction', 0.6) +
                       pre_closing_posturing * weights.get('pre_closing', 0.4)).fillna(0.0)
        
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_auction = closing_auction_intent.get(probe_date, 0.0)
                        p_posturing = pre_closing_posturing.get(probe_date, 0.0)
                        p_final = final_score.get(probe_date, 0.0)
                        print(f"      [日内诡道探针] _diagnose_final_assault @ {probe_date_str}")
                        print(f"        - [代理] 收盘竞价偷袭 (closing_auction_ambush_D): {p_auction:.4f}")
                        print(f"        - [代理] 收盘前动态 (pre_closing_posturing_D): {p_posturing:.4f}")
                        print(f"        - 最终终末强袭分: {p_final:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_final_assault 处理日期 {probe_date_str} 失败: {e}")
        
        return {signal_name: final_score.clip(-1, 1)}

    def _diagnose_vwap_battlefield(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.3 · 代理信号重构版】日内诡道之三：诊断“VWAP攻防”
        - 核心重构: 由于 `vwap_D` 信号不可用，无法进行攻防计算。本方法重构为直接采纳
                      `vwap_control_strength_D` 作为最终裁决，它本身就是对VWAP攻防的总结。
        """
        signal_name = "SCORE_INTRADAY_VWAP_BATTLEFIELD"
        # 直接使用 vwap_control_strength_D
        required_signals = ['vwap_control_strength_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_vwap_battlefield"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        
        control_strength = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, "_diagnose_vwap_battlefield")
        final_score = control_strength
        
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_control_strength = control_strength.get(probe_date, 0.0)
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"      [日内诡道探针] _diagnose_vwap_battlefield @ {probe_date_str}")
                        print(f"        - [代理] VWAP控制强度 (vwap_control_strength_D): {p_control_strength:.4f}")
                        print(f"        - 最终VWAP攻防分: {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_vwap_battlefield 处理日期 {probe_date_str} 失败: {e}")
        
        return {signal_name: final_score.clip(-1, 1)}


