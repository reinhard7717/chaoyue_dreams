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
                "SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL": pd.Series(dtype=np.float64), # 新增信号的空序列
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
            self._diagnose_intraday_bull_control, # 新增对日内多头控制力诊断方法的调用
        ]
        final_scores = {}
        for diagnostic_func in diagnostics_to_run:
            result = diagnostic_func(df)
            final_scores.update(result)
        print(f"日内行为诊断完成，生成 {len(final_scores)} 个信号序列。")
        return final_scores

    def _diagnose_intraday_bull_control(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.1 · Antifragile Resilience版】行为层原子信号：诊断“日内多头控制力”
        - 核心逻辑: 引入“反脆弱”裁决层。在使用主力信念指数前，通过clip(-1, 1)强制约束其范围，
                    确保模型在面对上游极端数据时具备韧性，不会因数学错误而失效。
        """
        signal_name = "SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL"
        required_signals = [
            'vwap_control_strength_D',
            'upward_impulse_purity_D',
            'lower_shadow_absorption_strength_D',
            'dip_absorption_power_D',
            'main_force_conviction_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_intraday_bull_control"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        # --- 获取参数 ---
        parent_params = get_params_block(self.strategy, 'intraday_behavior_params', {})
        params = get_param_value(parent_params.get('bull_control_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {'asymmetric_quality': 0.7, 'belief_modulator': 0.3})
        # 1. 获取战略位置分 (核心基石)
        position_score = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, "_diagnose_intraday_bull_control")
        # 2. 计算战术品质分
        # 2.1 获取进攻与防守的原料信号并归一化
        offensive_quality = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, "_diagnose_intraday_bull_control")
        mtf_params = get_params_block(self.strategy, 'behavioral_dynamics_params', {}).get('mtf_normalization_params', {})
        default_weights = mtf_params.get('default_weights')
        raw_lower_shadow = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, "_diagnose_intraday_bull_control")
        raw_dip_absorption = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, "_diagnose_intraday_bull_control")
        lower_shadow_strength = get_adaptive_mtf_normalized_score(raw_lower_shadow, df.index, True, default_weights)
        dip_absorption_power = get_adaptive_mtf_normalized_score(raw_dip_absorption, df.index, True, default_weights)
        defensive_quality = (lower_shadow_strength + dip_absorption_power) / 2
        # 2.2 构建基于战略位置的非对称权重
        weight_offensive = (1 + position_score) / 2
        weight_defensive = (1 - position_score) / 2
        asymmetric_action_quality = offensive_quality * weight_offensive + defensive_quality * weight_defensive
        # 2.3 构建信念调节器
        raw_conviction_index = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, "_diagnose_intraday_bull_control")
        # [代码修改] 增加“裁决层”，确保信念指数在[-1, 1]范围内，增强模型韧性
        conviction_index = raw_conviction_index.clip(-1, 1)
        belief_modulator = (conviction_index + 1) / 2
        # 2.4 非线性融合战术品质与信念
        comprehensive_quality_modulator = (
            asymmetric_action_quality.pow(fusion_weights.get('asymmetric_quality', 0.7)) *
            belief_modulator.pow(fusion_weights.get('belief_modulator', 0.3))
        ).fillna(0.0)
        # 3. 最终融合：战略位置分 * 综合品质调节器
        final_score = (position_score * comprehensive_quality_modulator).fillna(0.0)
        # --- 探针逻辑 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_position = position_score.get(probe_date, 0.0)
                        p_offensive = offensive_quality.get(probe_date, 0.0)
                        p_defensive = defensive_quality.get(probe_date, 0.0)
                        p_w_off = weight_offensive.get(probe_date, 0.5)
                        p_w_def = weight_defensive.get(probe_date, 0.5)
                        p_asymmetric_quality = asymmetric_action_quality.get(probe_date, 0.0)
                        # [代码修改] 升级探针，展示裁决过程
                        p_raw_conviction = raw_conviction_index.get(probe_date, 0.0)
                        p_clipped_conviction = conviction_index.get(probe_date, 0.0)
                        p_belief_mod = belief_modulator.get(probe_date, 0.0)
                        p_comp_quality = comprehensive_quality_modulator.get(probe_date, 0.0)
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"      [日内行为探针 V3.1] _diagnose_intraday_bull_control @ {probe_date_str}")
                        print(f"        - [基石] 战略位置 (vwap_control_strength_D): {p_position:.4f}")
                        print(f"        - [原料] 进攻品质: {p_offensive:.4f}, 防守品质: {p_defensive:.4f}")
                        print(f"        - [计算节点] 非对称权重 (攻/防): {p_w_off:.2f} / {p_w_def:.2f}")
                        print(f"        - [计算节点] 非对称战术品质分: {p_asymmetric_quality:.4f}")
                        print(f"        - [原料] 主力信念指数 (原始): {p_raw_conviction:.4f} -> (裁决后): {p_clipped_conviction:.4f} -> 信念调节器: {p_belief_mod:.4f}")
                        print(f"        - [计算节点] 综合品质调节器 (品质^0.7 * 信念^0.3): {p_comp_quality:.4f}")
                        print(f"        - [结果] 最终日内多头控制力分: {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_intraday_bull_control 处理日期 {probe_date_str} 失败: {e}")
        return {signal_name: final_score.clip(-1, 1)}

    def _diagnose_offensive_purity(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 代理信号重构版】日内战报之一：诊断“进攻纯度”
        - 核心重构: 由于预计算信号不存在，本方法重构为使用可用的日线级代理信号进行融合计算。
                      通过融合“上涨脉冲纯度”和“主力信念指数”来评估进攻质量。
        """
        signal_name = "SCORE_INTRADay_OFFENSIVE_PURITY"
        # 使用可用的代理信号进行计算
        required_signals = ['upward_impulse_purity_D', 'main_force_conviction_index_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_offensive_purity"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        
        purity = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, "_diagnose_offensive_purity")
        conviction = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, "_diagnose_offensive_purity")
        
        # 融合逻辑：高质量的进攻是既纯粹又有信念的
        final_score = (purity.pow(0.6) * ((conviction + 1) / 2).pow(0.4)).fillna(0.0)
        
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        p_purity = purity.get(probe_date, 0.0)
                        p_conviction = conviction.get(probe_date, 0.0)
                        p_final_score = final_score.get(probe_date, 0.0)
                        print(f"      [日内行为探针] _diagnose_offensive_purity @ {probe_date_str}")
                        print(f"        - [代理] 上涨脉冲纯度 (upward_impulse_purity_D): {p_purity:.4f}")
                        print(f"        - [代理] 主力信念指数 (main_force_conviction_index_D): {p_conviction:.4f}")
                        print(f"        - 最终进攻纯度分: {p_final_score:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_offensive_purity 处理日期 {probe_date_str} 失败: {e}")
        
        return {signal_name: final_score}

    def _diagnose_dominance_consensus(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 代理信号重构版】日内战报之二：诊断“支配共识”
        - 核心重构: 由于预计算信号不存在，本方法重构为直接采用 `vwap_control_strength_D` 作为核心代理信号。
        """
        signal_name = "SCORE_INTRADAY_DOMINANCE_CONSENSUS"
        # 使用可用的代理信号
        raw_signal_name = "vwap_control_strength_D"
        if not self._validate_required_signals(df, [raw_signal_name], "_diagnose_dominance_consensus"):
            return {signal_name: pd.Series(0.0, index=df.index)}
        
        final_score = self._get_safe_series(df, raw_signal_name, 0.0, "_diagnose_dominance_consensus")
        
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
                        print(f"        - [代理] VWAP控制强度 ({raw_signal_name}): {score_on_date:.4f}")
                        print(f"        - 最终支配共识分: {score_on_date:.4f}")
                except Exception as e:
                    print(f"    -> [日内行为探针错误] _diagnose_dominance_consensus 处理日期 {probe_date_str} 失败: {e}")
        
        return {signal_name: final_score}

    def _diagnose_conviction_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.5 · 向量化重构版】日内战报之三：诊断“信念反转”
        - 核心重构: 对整个方法进行向量化重构，使其能够处理完整的日线数据DataFrame并为每一天生成分数，而不是仅处理第一天。
        - 核心修复: 修复了探针逻辑，使其能够正确地在指定的 probe_dates 循环并打印每日的详细诊断信息。
        """
        # 此方法现在需要从分钟数据中找到对应的日线数据，但实际上传入的已经是日线数据df
        if df.empty:
            return {"SCORE_INTRADAY_CONVICTION_REVERSAL": pd.Series(dtype=np.float64)}
        # 将所有计算向量化，使其能处理整个DataFrame
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
        # --- 重构探针逻辑以适配历史回溯和向量化计算 ---
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
        return {"SCORE_INTRADAY_CONVICTION_REVERSAL": final_score.clip(-1, 1)}

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


