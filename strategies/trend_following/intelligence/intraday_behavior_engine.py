# 文件: strategies/trend_following/intelligence/intraday_behavior_engine.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional, Any
# 导入 get_params_block 工具
from strategies.trend_following.utils import (
    get_params_block, get_param_value, normalize_to_bipolar, 
    get_adaptive_mtf_normalized_score, is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
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

    def run_intraday_diagnostics(self, df: pd.DataFrame) -> Dict[str, pd.Series]: # [代码修改] 移除 async
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
        print("启动【V4.3 · 同步执行版】日内行为诊断...") # [代码修改] 更新版本号和描述
        # [代码修改] 移除对 _prepare_intraday_indicators 的调用
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
        【V2.0 · 日线信号重构版】日内战报之一：诊断“进攻纯度”
        - 核心重构: 废除所有分钟级计算，转为直接使用数据层提供的、预计算好的日线级信号 `intraday_offensive_purity_D`。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        signal_name = "SCORE_INTRADAY_OFFENSIVE_PURITY"
        # [代码修改开始] 重构为直接读取日线级预计算信号
        raw_signal_name = "intraday_offensive_purity_D"
        if not self._validate_required_signals(df, [raw_signal_name], "_diagnose_offensive_purity"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        final_score = self._get_safe_series(df, raw_signal_name, 0.0, "_diagnose_offensive_purity")
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑以适配向量化计算 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        score_on_date = final_score.get(probe_date, 'N/A')
                        print(f"      [日内行为探针] _diagnose_offensive_purity @ {probe_date_str}")
                        print(f"        - 读取预计算信号 '{raw_signal_name}': {score_on_date:.4f}")
                        print(f"        - 最终进攻纯度分: {score_on_date:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_offensive_purity 处理日期 {probe_date_str} 失败: {e}")
        # [代码修改结束]
        return {signal_name: final_score}

    def _diagnose_dominance_consensus(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 日线信号重构版】日内战报之二：诊断“支配共识”
        - 核心重构: 废除所有分钟级计算，转为直接使用数据层提供的、预计算好的日线级信号 `intraday_dominance_consensus_D`。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        signal_name = "SCORE_INTRADAY_DOMINANCE_CONSENSUS"
        # [代码修改开始] 重构为直接读取日线级预计算信号
        raw_signal_name = "intraday_dominance_consensus_D"
        if not self._validate_required_signals(df, [raw_signal_name], "_diagnose_dominance_consensus"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        final_score = self._get_safe_series(df, raw_signal_name, 0.0, "_diagnose_dominance_consensus")
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        score_on_date = final_score.get(probe_date, 'N/A')
                        print(f"      [日内行为探针] _diagnose_dominance_consensus @ {probe_date_str}")
                        print(f"        - 读取预计算信号 '{raw_signal_name}': {score_on_date:.4f}")
                        print(f"        - 最终支配共识分: {score_on_date:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_dominance_consensus 处理日期 {probe_date_str} 失败: {e}")
        # [代码修改结束]
        return {signal_name: final_score}

    def _diagnose_conviction_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.5 · 向量化重构版】日内战报之三：诊断“信念反转”
        - 核心重构: 对整个方法进行向量化重构，使其能够处理完整的日线数据DataFrame并为每一天生成分数，而不是仅处理第一天。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        # [代码修改] 此方法现在需要从分钟数据中找到对应的日线数据，但实际上传入的已经是日线数据df
        if df.empty:
            return {"SCORE_INTRADAY_CONVICTION_REVERSAL": pd.Series(dtype=np.float64)}
        # [代码修改开始] 将所有计算向量化，使其能处理整个DataFrame
        panic_score = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, "_diagnose_conviction_reversal")
        absorption_score = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, "_diagnose_conviction_reversal")
        mf_alpha_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, "_diagnose_conviction_reversal")
        # 使用tanh进行柔性映射，k=2.0表示对alpha的正负较为敏感
        bullish_alpha_score = (np.tanh(mf_alpha_raw * 2.0) + 1) / 2
        bullish_reversal_evidence = (panic_score * absorption_score * bullish_alpha_score).pow(1/3).fillna(0)
        distribution_score = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, "_diagnose_conviction_reversal")
        conviction_slope_5d = self._get_safe_series(df, 'SLOPE_5_main_force_conviction_index_D', np.nan, "_diagnose_conviction_reversal")
        # 当5日斜率无效时，明确将衰减设为0
        conviction_decay = -conviction_slope_5d.clip(upper=0).fillna(0)
        mf_alpha_bearish = mf_alpha_raw.clip(upper=0).abs()
        bearish_reversal_evidence = (distribution_score * conviction_decay * mf_alpha_bearish).pow(1/3).fillna(0)
        final_score = (bullish_reversal_evidence - bearish_reversal_evidence).fillna(0)
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑以适配历史回溯和向量化计算 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            # 遍历所有探针日期
            for probe_date_str in probe_dates:
                try:
                    # 尝试将字符串日期转换为与DataFrame索引时区匹配的Timestamp
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        # 从Series中获取探针日期的具体值
                        p_panic_score = panic_score.get(probe_date, 0.0)
                        p_absorption_score = absorption_score.get(probe_date, 0.0)
                        p_bullish_alpha_score = bullish_alpha_score.get(probe_date, 0.0)
                        p_mf_alpha_raw = mf_alpha_raw.get(probe_date, 0.0)
                        p_bullish_reversal_evidence = bullish_reversal_evidence.get(probe_date, 0.0)
                        p_distribution_score = distribution_score.get(probe_date, 0.0)
                        p_conviction_decay = conviction_decay.get(probe_date, 0.0)
                        p_conviction_slope_5d = conviction_slope_5d.get(probe_date, np.nan)
                        conviction_decay_source = "FALLBACK_ZERO" if pd.isna(p_conviction_slope_5d) else "5D_SLOPE"
                        p_mf_alpha_bearish = mf_alpha_bearish.get(probe_date, 0.0)
                        p_bearish_reversal_evidence = bearish_reversal_evidence.get(probe_date, 0.0)
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"      [日内行为探针] _diagnose_conviction_reversal @ {probe_date_str}")
                        print(f"        - 看涨证据: 恐慌={p_panic_score:.2f}, 承接={p_absorption_score:.2f}, 主力Alpha分(新)={p_bullish_alpha_score:.2f} (原始Alpha={p_mf_alpha_raw:.2f}) -> 综合={p_bullish_reversal_evidence:.4f}")
                        print(f"        - 看跌证据: 派发={p_distribution_score:.2f}, 信念衰减={p_conviction_decay:.2f} (来源: {conviction_decay_source}), 主力Alpha-={p_mf_alpha_bearish:.2f} -> 综合={p_bearish_reversal_evidence:.4f}")
                        print(f"        - 最终信念反转分: {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_conviction_reversal 处理日期 {probe_date_str} 失败: {e}")
        # [代码修改结束]
        return {"SCORE_INTRADAY_CONVICTION_REVERSAL": final_score.clip(-1, 1)}

    def _diagnose_tactical_arc(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 日线信号重构版】日内叙事之一：诊断“战术弧线”
        - 核心重构: 废除所有分钟级计算，转为直接使用数据层提供的、预计算好的日线级信号 `intraday_tactical_arc_D`。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        signal_name = "SCORE_INTRADAY_TACTICAL_ARC"
        # [代码修改开始] 重构为直接读取日线级预计算信号
        raw_signal_name = "intraday_tactical_arc_D"
        if not self._validate_required_signals(df, [raw_signal_name], "_diagnose_tactical_arc"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        final_score = self._get_safe_series(df, raw_signal_name, 0.0, "_diagnose_tactical_arc")
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑 ---
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
                        print(f"        - 读取预计算信号 '{raw_signal_name}': {score_on_date:.4f}")
                        print(f"        - 最终战术弧线分: {score_on_date:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_tactical_arc 处理日期 {probe_date_str} 失败: {e}")
        # [代码修改结束]
        return {signal_name: final_score}

    def _diagnose_auction_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 向量化重构版】日内叙事之二：诊断“竞价意图”
        - 核心重构: 对整个方法进行向量化重构，使其能够处理完整的日线数据DataFrame并为每一天生成分数。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        if df.empty:
            return {"SCORE_INTRADAY_AUCTION_INTENT": pd.Series(dtype=np.float64)}
        # [代码修改开始] 向量化计算
        # 从日线数据中获取开盘和收盘的博弈信号Series
        opening_intent = self._get_safe_series(df, 'opening_battle_result_D', 0.0, "_diagnose_auction_intent")
        closing_intent = self._get_safe_series(df, 'closing_auction_ambush_D', 0.0, "_diagnose_auction_intent")
        # 从配置中获取权重
        params = get_params_block(self.strategy, 'intraday_narrative_engine_params', {})
        weights = get_param_value(params.get('auction_intent_weights'), {'opening': 0.4, 'closing': 0.6})
        # 加权融合
        final_score = (opening_intent * weights.get('opening', 0.4) +
                       closing_intent * weights.get('closing', 0.6))
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑 ---
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
        # [代码修改结束]
        return {"SCORE_INTRADAY_AUCTION_INTENT": final_score.clip(-1, 1)}

    def _diagnose_recovery_quality(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 日线信号重构版】日内叙事之三：诊断“修复质量”
        - 核心重构: 废除所有分钟级计算，转为直接使用数据层提供的、预计算好的日线级信号 `intraday_recovery_quality_D`。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        signal_name = "SCORE_INTRADAY_RECOVERY_QUALITY"
        # [代码修改开始] 重构为直接读取日线级预计算信号
        raw_signal_name = "intraday_recovery_quality_D"
        if not self._validate_required_signals(df, [raw_signal_name], "_diagnose_recovery_quality"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        final_score = self._get_safe_series(df, raw_signal_name, 0.0, "_diagnose_recovery_quality")
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        score_on_date = final_score.get(probe_date, 'N/A')
                        print(f"      [日内行为探针] _diagnose_recovery_quality @ {probe_date_str}")
                        print(f"        - 读取预计算信号 '{raw_signal_name}': {score_on_date:.4f}")
                        print(f"        - 最终修复质量分: {score_on_date:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_recovery_quality 处理日期 {probe_date_str} 失败: {e}")
        # [代码修改结束]
        return {signal_name: final_score.clip(0, 1)}

    def _diagnose_ambush_and_flank(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 向量化重构版】日内诡道之一：诊断“伏击与侧翼”
        - 核心重构: 对整个方法进行向量化重构，使其能够处理完整的日线数据DataFrame并为每一天生成分数。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        if df.empty:
            return {"SCORE_INTRADAY_AMBUSH_AND_FLANK": pd.Series(dtype=np.float64)}
        # [代码修改开始] 向量化计算
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
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑 ---
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
        # [代码修改结束]
        return {"SCORE_INTRADAY_AMBUSH_AND_FLANK": final_score.clip(0, 1)}

    def _diagnose_final_assault(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 日线信号重构版】日内诡道之二：诊断“终末强袭”
        - 核心重构: 废除所有分钟级计算，转为融合数据层提供的三个核心日线级信号：尾盘攻击强度、加速度和收盘竞价意图。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        signal_name = "SCORE_INTRADAY_FINAL_ASSAULT"
        required_signals = ['final_assault_strength_D', 'final_assault_accel_D', 'closing_auction_ambush_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_final_assault"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        # [代码修改开始] 重构为融合多个日线级预计算信号
        params = get_params_block(self.strategy, 'intraday_gambit_engine_params', {}).get('final_assault_params', {})
        weights = params.get('fusion_weights', {'assault_strength': 0.5, 'assault_accel': 0.2, 'closing_auction': 0.3})
        # 获取三个核心证据的Series
        assault_strength = self._get_safe_series(df, 'final_assault_strength_D', 0.0, "_diagnose_final_assault")
        assault_accel = self._get_safe_series(df, 'final_assault_accel_D', 0.0, "_diagnose_final_assault")
        closing_auction_intent = self._get_safe_series(df, 'closing_auction_ambush_D', 0.0, "_diagnose_final_assault")
        # 加权融合
        final_score = (assault_strength * weights.get('assault_strength', 0.5) +
                       assault_accel * weights.get('assault_accel', 0.2) +
                       closing_auction_intent * weights.get('closing_auction', 0.3)).fillna(0.0)
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_strength = assault_strength.get(probe_date, 0.0)
                        p_accel = assault_accel.get(probe_date, 0.0)
                        p_auction = closing_auction_intent.get(probe_date, 0.0)
                        p_final = final_score.get(probe_date, 0.0)
                        print(f"      [日内诡道探针] _diagnose_final_assault @ {probe_date_str}")
                        print(f"        - [证据] 尾盘攻击强度 (final_assault_strength_D): {p_strength:.4f}")
                        print(f"        - [证据] 尾盘攻击加速 (final_assault_accel_D): {p_accel:.4f}")
                        print(f"        - [证据] 收盘竞价偷袭 (closing_auction_ambush_D): {p_auction:.4f}")
                        print(f"        - 最终终末强袭分: {p_final:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_final_assault 处理日期 {probe_date_str} 失败: {e}")
        # [代码修改结束]
        return {signal_name: final_score.clip(-1, 1)}

    def _diagnose_vwap_battlefield(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 向量化重构版】日内诡道之三：诊断“VWAP攻防”
        - 核心重构: 对整个方法进行向量化重构，使其能够处理完整的日线数据DataFrame并为每一天生成分数。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        required_signals = ['close_D', 'low_D', 'high_D', 'vwap_D', 'volume_D', 'vwap_control_strength_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_vwap_battlefield"):
            return {"SCORE_INTRADAY_VWAP_BATTLEFIELD": pd.Series(0.0, index=df.index)}
        # [代码修改开始] 向量化计算
        params = get_params_block(self.strategy, 'intraday_gambit_engine_params', {}).get('vwap_battlefield_params', {})
        weights = params.get('fusion_weights', {'net_battle_score': 0.6, 'control_strength': 0.4})
        # 使用日线级信号进行攻防测试判断
        vwap = self._get_safe_series(df, 'vwap_D', np.nan, "_diagnose_vwap_battlefield")
        # VWAP支撑测试: 最低价触及或低于VWAP，但收盘价高于VWAP
        support_mask = (self._get_safe_series(df, 'low_D') <= vwap) & (self._get_safe_series(df, 'close_D') > vwap)
        # VWAP压制测试: 最高价触及或高于VWAP，但收盘价低于VWAP
        suppression_mask = (self._get_safe_series(df, 'high_D') >= vwap) & (self._get_safe_series(df, 'close_D') < vwap)
        volume = self._get_safe_series(df, 'volume_D', 0.0, "_diagnose_vwap_battlefield")
        # 以当日成交量作为支撑或压制的“分数”
        support_score = volume.where(support_mask, 0.0)
        suppression_score = volume.where(suppression_mask, 0.0)
        total_battle_volume = support_score + suppression_score + 1e-9
        net_battle_score = (support_score - suppression_score) / total_battle_volume
        # 直接使用日线级的VWAP控制信号
        control_strength = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, "_diagnose_vwap_battlefield")
        final_score = (net_battle_score * weights.get('net_battle_score', 0.6) +
                       control_strength * weights.get('control_strength', 0.4)).fillna(0.0)
        # [代码修改结束]
        # --- [代码修改开始] 重构探针逻辑 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_support_score = support_score.get(probe_date, 0.0)
                        p_suppression_score = suppression_score.get(probe_date, 0.0)
                        p_net_battle_score = net_battle_score.get(probe_date, 0.0)
                        p_control_strength = control_strength.get(probe_date, 0.0)
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"      [日内诡道探针] _diagnose_vwap_battlefield @ {probe_date_str}")
                        print(f"        - [计算] VWAP支撑总量: {p_support_score:.2f}")
                        print(f"        - [计算] VWAP压制总量: {p_suppression_score:.2f}")
                        print(f"        - [计算] 净战斗分: {p_net_battle_score:.4f}")
                        print(f"        - [证据] VWAP控制强度 (vwap_control_strength_D): {p_control_strength:.4f}")
                        print(f"        - 最终VWAP攻防分: {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_vwap_battlefield 处理日期 {probe_date_str} 失败: {e}")
        # [代码修改结束]
        return {"SCORE_INTRADAY_VWAP_BATTLEFIELD": final_score.clip(-1, 1)}


