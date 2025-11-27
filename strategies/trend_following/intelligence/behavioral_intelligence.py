# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional, List, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_score, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar, get_adaptive_mtf_normalized_score

class BehavioralIntelligence:
    """
    【V28.0 · 结构升维版】
    - 核心升级: 废弃了旧的 _calculate_price_health, _calculate_volume_health, _calculate_kline_pattern_health 方法。
                所有健康度计算已统一由全新的 _calculate_structural_behavior_health 引擎负责。
    """
    def __init__(self, strategy_instance):
        """
        初始化行为与模式识别模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        self.pattern_recognizer = strategy_instance.pattern_recognizer

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [行为情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_behavioral_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V29.0 · 结构指标升维版】行为情报模块总指挥
        - 核心升级: 新增对 _diagnose_microstructure_intent 方法的调用，引入微观结构意图信号。
        """
        all_behavioral_states = {}
        atomic_signals = self._diagnose_behavioral_axioms(df)
        self.strategy.atomic_states.update(atomic_signals)
        all_behavioral_states.update(atomic_signals)
        # [新增代码块] 调用微观结构意图诊断
        micro_intent_signals = self._diagnose_microstructure_intent(df)
        self.strategy.atomic_states.update(micro_intent_signals)
        all_behavioral_states.update(micro_intent_signals)
        context_new_high_strength = self._diagnose_context_new_high_strength(df)
        self.strategy.atomic_states.update(context_new_high_strength)
        all_behavioral_states.update(context_new_high_strength)
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(atomic_signals.get('SCORE_BEHAVIOR_PRICE_VS_VOLUME_DIVERGENCE', pd.Series(0.0, index=df.index)))
        all_behavioral_states['SCORE_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_behavioral_states['SCORE_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        for k, v in atomic_signals.items():
            if k not in df.columns:
                df[k] = v
        df_with_dynamics = self._calculate_signal_dynamics(df)
        dynamic_cols = [c for c in df_with_dynamics.columns if c.startswith(('MOMENTUM_', 'POTENTIAL_', 'THRUST_', 'RESONANCE_'))]
        self.strategy.atomic_states.update(df_with_dynamics[dynamic_cols])
        all_behavioral_states.update(df_with_dynamics[dynamic_cols])
        return all_behavioral_states

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.0】安全地从原子状态库或主数据帧中获取分数。
        - 核心职责: 统一信号获取路径，优先从 self.strategy.atomic_states 获取，
                      若无则从主数据帧 df 获取，最后提供默认值，确保数据流的稳定性。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in df.columns:
            return df[name]
        else:
            print(f"     -> [行为情报引擎警告] 信号 '{name}' 不存在，使用默认值 {default}。")
            return pd.Series(default, index=df.index)

    def _get_signal(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        【V1.0】信号获取哨兵方法
        - 核心职责: 安全地从DataFrame获取信号。
        - 预警机制: 如果信号不存在，打印明确的警告信息，并返回一个包含默认值的Series，以防止程序崩溃。
        """
        if signal_name not in df.columns:
            print(f"    -> [行为情报引擎警告] 依赖信号 '{signal_name}' 在数据帧中不存在，将使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[signal_name]

    def _generate_all_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 职责净化版】原子信号中心
        - 核心升级: 遵循“三层金字塔”架构，本方法不再计算跨领域的“趋势健康度”和“绝望度”。
                      这些高级融合逻辑已迁移至 FusionIntelligence。
                      新增对纯净版“行为K线质量分”的计算和发布。
        """
        atomic_signals = {}
        atomic_signals.update(self._diagnose_behavioral_axioms(df))
        day_quality_score = self._calculate_behavioral_day_quality(df)
        atomic_signals['BIPOLAR_BEHAVIORAL_DAY_QUALITY'] = day_quality_score
        battlefield_momentum = day_quality_score.ewm(span=5, adjust=False).mean()
        atomic_signals['SCORE_BEHAVIORAL_BATTLEFIELD_MOMENTUM'] = battlefield_momentum.astype(np.float32)
        self.strategy.atomic_states.update(atomic_signals)
        atomic_signals.update(self._diagnose_upper_shadow_intent(df))
        return atomic_signals

    def _calculate_signal_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.3 · 上涨衰竭动态增强与多时间维度归一化版】信号动态计算引擎
        - 核心错误修复: 彻底剥离了对其他情报层终极共振信号的依赖，解决了因执行时序错乱导致的信号获取失败问题。
        - 核心逻辑重构: 遵循“职责分离”原则，本方法现在只聚焦于为【本模块生产的】纯粹行为原子信号注入动态因子（动量、潜力、推力）。
                        不再计算跨领域的 RESONANCE_HEALTH_D 等信号。
        - 【修改】移除对 `SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE` 的动态增强。
        - 【优化】将 `momentum`, `potential`, `thrust` 的归一化方式改为多时间维度自适应归一化。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_dyn = get_param_value(p_conf.get('signal_dynamics_params'), {})
        momentum_span = get_param_value(p_dyn.get('momentum_span'), 5)
        potential_window = get_param_value(p_dyn.get('potential_window'), 120)
        dynamics_df = pd.DataFrame(index=df.index)
        atomic_signals_to_enhance = [
            'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM',
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM',
            'SCORE_BEHAVIOR_VOLUME_BURST',
            'SCORE_BEHAVIOR_VOLUME_ATROPHY',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION',
            'SCORE_OPPORTUNITY_LOCKUP_RALLY',
            'SCORE_OPPORTUNITY_SELLING_EXHAUSTION',
            'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', # 新增上涨衰竭原始分的动态增强
            'SCORE_RISK_LIQUIDITY_DRAIN'
        ]
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        for signal_name in atomic_signals_to_enhance:
            if signal_name in self.strategy.atomic_states:
                signal_series = self.strategy.atomic_states[signal_name]
                momentum = signal_series.diff(momentum_span).fillna(0)
                # 【优化】使用多时间维度自适应归一化
                norm_momentum = get_adaptive_mtf_normalized_score(momentum, df.index, ascending=True, tf_weights=default_weights)
                dynamics_df[f'MOMENTUM_{signal_name}'] = norm_momentum.astype(np.float32)
                potential = signal_series.rolling(window=potential_window).mean().fillna(signal_series)
                # 【优化】使用多时间维度自适应归一化
                norm_potential = get_adaptive_mtf_normalized_score(potential, df.index, ascending=True, tf_weights=default_weights)
                dynamics_df[f'POTENTIAL_{signal_name}'] = norm_potential.astype(np.float32)
                thrust = momentum.diff(1).fillna(0)
                # 【优化】使用多时间维度自适应归一化
                norm_thrust = get_adaptive_mtf_normalized_score(thrust, df.index, ascending=True, tf_weights=default_weights)
                dynamics_df[f'THRUST_{signal_name}'] = norm_thrust.astype(np.float32)
            else:
                print(f"     - [警告] 信号 '{signal_name}' 在原子状态库中不存在，跳过动态因子计算。")
        final_df = pd.concat([df, dynamics_df], axis=1)
        return final_df

    def _calculate_behavioral_day_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.2 · 信号校验增强版】行为K线质量分计算引擎
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        required_signals = [
            'closing_price_deviation_score_D', 'real_body_vs_range_ratio_D', 'shadow_dominance_D',
            'VPA_EFFICIENCY_D', 'vwap_control_strength_D', 'intraday_trend_purity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_behavioral_day_quality"):
            return pd.Series(0.0, index=df.index)
        print("开始执行【V1.0 · 纯净版】行为K线质量分计算...")
        outcome_core = (self._get_safe_series(df, 'closing_price_deviation_score_D', 0.5, method_name="_calculate_behavioral_day_quality") * 2 - 1).clip(-1, 1)
        body_dominance = self._get_safe_series(df, 'real_body_vs_range_ratio_D', 0.0, method_name="_calculate_behavioral_day_quality")
        shadow_dominance = self._get_safe_series(df, 'shadow_dominance_D', 0.0, method_name="_calculate_behavioral_day_quality")
        pillar1_outcome_score = (outcome_core * 0.7 + outcome_core * body_dominance * 0.1 + shadow_dominance * 0.2).clip(-1, 1)
        vpa_eff_bipolar = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'VPA_EFFICIENCY_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index)
        vwap_ctrl_bipolar = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'vwap_control_strength_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index)
        trend_purity_bipolar = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'intraday_trend_purity_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index)
        bullish_execution = (((vpa_eff_bipolar + 1)/2) * ((vwap_ctrl_bipolar + 1)/2) * ((trend_purity_bipolar + 1)/2)).pow(1/3)
        pillar2_execution_score = (bullish_execution * 2 - 1).clip(-1, 1)
        day_quality_score = (
            pillar1_outcome_score * 0.4 +
            pillar2_execution_score * 0.6
        ).clip(-1, 1)
        print("【纯净版行为K线质量分】计算完成。")
        return day_quality_score.astype(np.float32)

    def _diagnose_behavioral_axioms(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V29.9 · 战术信号增补版】原子信号中心
        - 核心增补: 新增两个高阶战术博弈信号，用于识别关键的战场转折点。
          - 新增机会信号: SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK (伏击式反攻)，识别主力利用恐慌完成吸筹后的反攻。
          - 新增风险信号: SCORE_RISK_BREAKOUT_FAILURE_CASCADE (突破失败风险)，识别“牛市陷阱”式的假突破。
        """
        required_signals = [
            'pct_change_D', 'BIAS_55_D', 'volume_ratio_D', 'microstructure_efficiency_index_D',
            'dip_absorption_power_D', 'vwap_control_strength_D', 'lower_shadow_absorption_strength_D',
            'upper_shadow_selling_pressure_D', 'volume_D', 'rally_distribution_pressure_D',
            'closing_price_deviation_score_D', 'intraday_posture_score_D', 'main_force_net_flow_calibrated_D',
            'amount_D', 'BIAS_21_D', 'ACCEL_5_pct_change_D', 'RSI_13_D', 'total_winner_rate_D',
            'winner_stability_index_D', 'control_solidity_index_D', 'trend_vitality_index_D',
            'upward_impulse_purity_D', 'vacuum_traversal_efficiency_D', 'support_validation_strength_D',
            'impulse_quality_ratio_D', 'floating_chip_cleansing_efficiency_D',
            #  为新增战术信号补充所需信号
            'panic_selling_cascade_D', 'main_force_execution_alpha_D', 'covert_accumulation_signal_D', 'high_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_behavioral_axioms"):
            return {}
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_conf.get('long_term_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        # --- 基础信号计算 (维持V29.8版本) ---
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_behavioral_axioms")
        closing_deviation = self._get_safe_series(df, 'closing_price_deviation_score_D', 0.5, method_name="_diagnose_behavioral_axioms")
        intraday_posture = self._get_safe_series(df, 'intraday_posture_score_D', 0.0, method_name="_diagnose_behavioral_axioms")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_behavioral_axioms")
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_diagnose_behavioral_axioms").replace(0, 1e-9)
        bias_21 = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name="_diagnose_behavioral_axioms")
        magnitude_factor_up = get_adaptive_mtf_normalized_score(pct_change.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        internal_strength_score = closing_deviation.clip(0, 1)
        path_efficiency_score = get_adaptive_mtf_normalized_score(intraday_posture, df.index, ascending=True, tf_weights=default_weights)
        flow_ratio = main_force_flow / amount
        norm_flow_ratio = get_adaptive_mtf_normalized_bipolar_score(flow_ratio, df.index, default_weights, sensitivity=0.05)
        smart_money_confirmation_score = (np.tanh(norm_flow_ratio * 2.0) + 1) / 2
        quality_factor = (internal_strength_score * path_efficiency_score * smart_money_confirmation_score).pow(1/3).fillna(0.0)
        overextension_risk_up = get_adaptive_mtf_normalized_score(bias_21.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        sustainability_factor = (1 - overextension_risk_up).clip(0, 1)
        upward_momentum_score = (magnitude_factor_up * quality_factor * sustainability_factor).clip(0, 1)
        states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = upward_momentum_score.astype(np.float32)
        price_accel = self._get_safe_series(df, 'ACCEL_5_pct_change_D', 0.0, method_name="_diagnose_behavioral_axioms")
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_diagnose_behavioral_axioms")
        magnitude_factor_down = get_adaptive_mtf_normalized_score(pct_change.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        internal_panic_score = (1 - closing_deviation).clip(0, 1)
        path_weakness_score = get_adaptive_mtf_normalized_score(intraday_posture.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        smart_money_intent_score = (1 - norm_flow_ratio) / 2
        authenticity_factor = (internal_panic_score * path_weakness_score * smart_money_intent_score).pow(1/3).fillna(0.0)
        volume_burst_score_down = get_adaptive_mtf_normalized_score(volume_ratio, df.index, ascending=True, tf_weights=default_weights)
        price_accel_score = get_adaptive_mtf_normalized_score(price_accel.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        panic_score = (volume_burst_score_down * price_accel_score).pow(0.5)
        panic_amplifier = 1 + panic_score
        downward_momentum_score = (magnitude_factor_down * authenticity_factor * panic_amplifier).clip(0, 1)
        states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = downward_momentum_score.astype(np.float32)
        rsi = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name="_diagnose_behavioral_axioms")
        winner_rate = self._get_safe_series(df, 'total_winner_rate_D', 50.0, method_name="_diagnose_behavioral_axioms")
        winner_stability = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_diagnose_behavioral_axioms")
        control_solidity = self._get_safe_series(df, 'control_solidity_index_D', 0.5, method_name="_diagnose_behavioral_axioms")
        trend_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name="_diagnose_behavioral_axioms")
        norm_bias = get_adaptive_mtf_normalized_score(bias_21, df.index, ascending=True, tf_weights=default_weights)
        norm_rsi = get_adaptive_mtf_normalized_score(rsi, df.index, ascending=True, tf_weights=default_weights)
        norm_winner_rate = get_adaptive_mtf_normalized_score(winner_rate, df.index, ascending=True, tf_weights=default_weights)
        excitement_force = (norm_bias * norm_rsi * norm_winner_rate).pow(1/3).fillna(0.0)
        norm_winner_stability = get_adaptive_mtf_normalized_score(winner_stability, df.index, ascending=True, tf_weights=long_term_weights)
        norm_control_solidity = get_adaptive_mtf_normalized_score(control_solidity, df.index, ascending=True, tf_weights=long_term_weights)
        norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality, df.index, ascending=True, tf_weights=long_term_weights)
        conviction_force = (norm_winner_stability * norm_control_solidity * norm_trend_vitality).pow(1/3).fillna(0.0)
        overextension_raw_score = (excitement_force - conviction_force).clip(lower=0)
        states['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = overextension_raw_score.astype(np.float32)
        base_efficiency_raw = self._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name="_diagnose_behavioral_axioms")
        path_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_behavioral_axioms")
        structural_confirm_raw = self._get_safe_series(df, 'vacuum_traversal_efficiency_D', 0.0, method_name="_diagnose_behavioral_axioms")
        base_efficiency_score = get_adaptive_mtf_normalized_score(base_efficiency_raw, df.index, ascending=True, tf_weights=default_weights)
        path_purity_score = get_adaptive_mtf_normalized_score(path_purity_raw, df.index, ascending=True, tf_weights=default_weights)
        structural_confirm_score = get_adaptive_mtf_normalized_score(structural_confirm_raw, df.index, ascending=True, tf_weights=default_weights)
        upward_efficiency_score = (base_efficiency_score.pow(0.4) * path_purity_score.pow(0.3) * structural_confirm_score.pow(0.3)).fillna(0.0)
        states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = upward_efficiency_score.astype(np.float32)
        passive_absorption_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_behavioral_axioms")
        active_defense_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name="_diagnose_behavioral_axioms")
        counter_attack_raw = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, method_name="_diagnose_behavioral_axioms")
        passive_absorption_score = get_adaptive_mtf_normalized_score(passive_absorption_raw, df.index, ascending=True, tf_weights=default_weights)
        active_defense_score = get_adaptive_mtf_normalized_score(active_defense_raw, df.index, ascending=True, tf_weights=default_weights)
        counter_attack_score = get_adaptive_mtf_normalized_score(counter_attack_raw, df.index, ascending=True, tf_weights=default_weights)
        downward_resistance_score = (passive_absorption_score.pow(0.2) * active_defense_score.pow(0.4) * counter_attack_score.pow(0.4)).fillna(0.0)
        states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = downward_resistance_score.astype(np.float32)
        strategic_position_raw = self._get_safe_series(df, 'vwap_control_strength_D', 0.5, method_name="_diagnose_behavioral_axioms")
        offensive_capability_raw = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name="_diagnose_behavioral_axioms")
        defensive_resilience_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_behavioral_axioms")
        strategic_position_score = get_adaptive_mtf_normalized_score(strategic_position_raw, df.index, ascending=True, tf_weights=default_weights)
        offensive_capability_score = get_adaptive_mtf_normalized_score(offensive_capability_raw, df.index, ascending=True, tf_weights=default_weights)
        defensive_resilience_score = get_adaptive_mtf_normalized_score(defensive_resilience_raw, df.index, ascending=True, tf_weights=default_weights)
        intraday_bull_control_score = (strategic_position_score.pow(0.4) * offensive_capability_score.pow(0.3) * defensive_resilience_score.pow(0.3)).fillna(0.0)
        states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = intraday_bull_control_score.astype(np.float32)
        lower_shadow_quality = self._diagnose_lower_shadow_quality(df)
        upper_shadow_risk = self._diagnose_upper_shadow_risk(df)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_quality
        states['INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW'] = upper_shadow_risk
        #  调用新增的战术信号诊断方法
        states['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK'] = self._diagnose_ambush_counterattack(df, lower_shadow_quality)
        states['SCORE_RISK_BREAKOUT_FAILURE_CASCADE'] = self._diagnose_breakout_failure_risk(df, upper_shadow_risk)
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_diagnose_behavioral_axioms"), df.index, ascending=True, tf_weights=default_weights)
        states['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = self._calculate_volume_atrophy(df, default_weights).astype(np.float32)
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_behavioral_axioms"), df.index, default_weights)
        volume_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'volume_D', method_name="_diagnose_behavioral_axioms").diff(1), df.index, default_weights)
        divergence_score = (volume_trend - price_trend).clip(-1, 1)
        states['SCORE_BEHAVIOR_PRICE_VS_VOLUME_DIVERGENCE'] = divergence_score.astype(np.float32)
        is_rising = (self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_behavioral_axioms") > 0).astype(float)
        is_falling = (self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_behavioral_axioms") < 0).astype(float)
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = (is_rising * states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] * states['SCORE_BEHAVIOR_VOLUME_ATROPHY']).pow(1/3).astype(np.float32)
        #  --- 开始重构 SCORE_OPPORTUNITY_SELLING_EXHAUSTION ---
        # 1. 获取额外所需原始数据
        price_accel = self._get_safe_series(df, 'ACCEL_5_pct_change_D', 0.0, method_name="_diagnose_behavioral_axioms")
        capitulation_raw = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_behavioral_axioms")
        # 2. 计算新维度得分
        # 卖出动能衰减分：负向加速度的绝对值越小，得分越高
        selling_deceleration_score = (1 - get_adaptive_mtf_normalized_score(price_accel.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)).clip(0, 1)
        # 投降确认分
        capitulation_confirm_score = get_adaptive_mtf_normalized_score(capitulation_raw, df.index, ascending=True, tf_weights=default_weights)
        # 3. 融合新旧维度
        selling_exhaustion_score = (
            states['SCORE_BEHAVIOR_VOLUME_ATROPHY'].pow(0.3) *
            states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'].pow(0.3) *
            selling_deceleration_score.pow(0.2) *
            capitulation_confirm_score.pow(0.2)
        ).fillna(0.0)
        states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = (is_falling * selling_exhaustion_score).astype(np.float32)
        # --- 结束重构 ---
        states['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'] = self._diagnose_stagnation_evidence(df, states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'])
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_BURST'] * states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM']).pow(1/2).astype(np.float32)
        distribution_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_behavioral_axioms")
        states['SCORE_BEHAVIOR_DISTRIBUTION_PRESSURE'] = get_adaptive_mtf_normalized_score(distribution_pressure_raw, df.index, ascending=True, tf_weights=default_weights)
        return states

    def _calculate_volume_atrophy(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V1.1 · 量能萎缩证据增强版】计算 SCORE_BEHAVIOR_VOLUME_ATROPHY 信号。
        - 核心逻辑: 融合三种量能萎缩的强力证据，并引入成交量均线空头排列和萎缩持续性。
          1. 量能一直低于ma_vol_21 (持续性萎缩)
          2. ma_vol_5已经低于ma_vol_21 (结构性萎缩)
          3. 当日量能 < ma_vol_5 < ma_vol_21 (冰点萎缩)
          4. 成交量均线空头排列 (VOL_MA_5_D < VOL_MA_13_D < VOL_MA_21_D)
        - 输出: [0, 1] 的单极性分数，分数越高代表量能萎缩越严重。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        # 获取必要的成交量均线数据
        vol = self._get_safe_series(df, 'volume_D', 0.0, method_name="_calculate_volume_atrophy")
        vol_ma5 = self._get_safe_series(df, 'VOL_MA_5_D', 0.0, method_name="_calculate_volume_atrophy")
        vol_ma13 = self._get_safe_series(df, 'VOL_MA_13_D', 0.0, method_name="_calculate_volume_atrophy") # 新增13日均量
        vol_ma21 = self._get_safe_series(df, 'VOL_MA_21_D', 0.0, method_name="_calculate_volume_atrophy")
        # 确保数据存在，否则返回默认值
        if vol.isnull().all() or vol_ma5.isnull().all() or vol_ma13.isnull().all() or vol_ma21.isnull().all():
            print("    -> [行为情报引擎警告] 缺少成交量或成交量均线数据，无法计算 SCORE_BEHAVIOR_VOLUME_ATROPHY。")
            return pd.Series(0.0, index=df_index)
        # 避免除以零
        vol_ma5_safe = vol_ma5.replace(0, 1e-9)
        vol_ma13_safe = vol_ma13.replace(0, 1e-9)
        vol_ma21_safe = vol_ma21.replace(0, 1e-9)
        # 证据1: 量能一直低于ma_vol_21 (持续性萎缩)
        # 使用 (1 - volume_D / VOL_MA_21_D) 衡量萎缩程度，比值越小，分数越高
        vol_below_ma21_raw = (1 - (vol / vol_ma21_safe)).clip(0, 1)
        vol_below_ma21_score = get_adaptive_mtf_normalized_score(vol_below_ma21_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 证据2: ma_vol_5已经低于ma_vol_21 (结构性萎缩)
        # 使用 (1 - VOL_MA_5_D / VOL_MA_21_D) 衡量萎缩程度
        vol_ma5_below_ma21_raw = (1 - (vol_ma5 / vol_ma21_safe)).clip(0, 1)
        vol_ma5_below_ma21_score = get_adaptive_mtf_normalized_score(vol_ma5_below_ma21_raw, df_index, ascending=True, tf_weights=tf_weights)
        # 证据3: 当日量能 < ma_vol_5 < ma_vol_21 (冰点萎缩)
        # 这是一个布尔条件，直接转换为分数，并可以考虑其持续性
        is_ice_point_atrophy = ((vol < vol_ma5) & (vol_ma5 < vol_ma21)).astype(float)
        # 冰点萎缩的持续性
        ice_point_atrophy_persistence = is_ice_point_atrophy.rolling(window=5, min_periods=1).sum() / 5
        ice_point_atrophy_score = is_ice_point_atrophy * (0.5 + ice_point_atrophy_persistence * 0.5) # 持续越久，分数越高
        # 证据4: 成交量均线空头排列 (VOL_MA_5_D < VOL_MA_13_D < VOL_MA_21_D)
        # 这是一个结构性萎缩的强力证据
        is_ma_bearish_alignment = ((vol_ma5 < vol_ma13) & (vol_ma13 < vol_ma21)).astype(float)
        # 均线空头排列的强度，可以根据均线之间的距离来衡量
        ma_bearish_alignment_strength_raw = (vol_ma21 - vol_ma5) / vol_ma21_safe.replace(0, np.nan)
        ma_bearish_alignment_score = get_adaptive_mtf_normalized_score(ma_bearish_alignment_strength_raw.clip(lower=0), df_index, ascending=True, tf_weights=tf_weights)
        # 融合布尔条件和强度
        ma_bearish_alignment_final_score = is_ma_bearish_alignment * ma_bearish_alignment_score
        # 融合所有证据
        # 赋予冰点萎缩和均线空头排列更高的权重，因为它代表了最强的萎缩信号
        # 使用加权几何平均，确保所有证据都存在时分数才高
        evidence_components = [
            vol_below_ma21_score,
            vol_ma5_below_ma21_score,
            ice_point_atrophy_score, # 使用增强后的冰点萎缩分数
            ma_bearish_alignment_final_score # 新增均线空头排列分数
        ]
        # 调整权重，冰点萎缩和均线空头排列权重更高
        weights = np.array([0.2, 0.2, 0.3, 0.3]) # 调整权重
        aligned_evidence_components = [comp.reindex(df_index, fill_value=0.0) for comp in evidence_components]
        safe_evidence_components = [comp + 1e-9 for comp in aligned_evidence_components] # 避免log(0)
        volume_atrophy_score = pd.Series(np.prod([comp.values ** w for comp, w in zip(safe_evidence_components, weights)], axis=0), index=df_index)
        return volume_atrophy_score.clip(0, 1)

    def _diagnose_context_new_high_strength(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 信号校验增强版】诊断内部上下文信号：新高强度 (CONTEXT_NEW_HIGH_STRENGTH)
        - 核心逻辑: 融合价格突破、均线斜率和BIAS健康度，评估新高的综合质量。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        """
        required_signals = ['pct_change_D', 'SLOPE_5_EMA_55_D', 'BIAS_55_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_context_new_high_strength"):
            return {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_mtf.get('long_term_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        price_breakthrough_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_context_new_high_strength").clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        ma_slope_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'SLOPE_5_EMA_55_D', pd.Series(0.0, index=df.index), method_name="_diagnose_context_new_high_strength"), df.index, ascending=True, tf_weights=default_weights)
        bias_health_score = 1 - get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'BIAS_55_D', pd.Series(0.0, index=df.index), method_name="_diagnose_context_new_high_strength").clip(lower=0), df.index, ascending=True, tf_weights=long_term_weights)
        new_high_strength = (price_breakthrough_score * ma_slope_score * bias_health_score).pow(1/3).fillna(0.0)
        return {'CONTEXT_NEW_HIGH_STRENGTH': new_high_strength.astype(np.float32)}

    def _resolve_pressure_absorption_dynamics(self, provisional_pressure: pd.Series, intent_diagnosis: pd.Series) -> Dict[str, pd.Series]:
        """
        【V3.3 · 情报校验加固版】压力-承接能量转化模型
        - 核心修改: 调用从 utils.py 导入的公共归一化工具。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `absorption_efficiency` 和 `absorption_control` 的归一化方式改为多时间维度自适应归一化。
        - 【新增】增加战前情报校验，确保所有依赖信号存在。
        """
        states = {}
        df = self.strategy.df_indicators
        # [新增代码块] 战前情报校验
        required_signals = ['VPA_EFFICIENCY_D', 'vwap_control_strength_D']
        if not self._validate_required_signals(df, required_signals, "_resolve_pressure_absorption_dynamics"):
            return {
                'SCORE_RISK_UNRESOLVED_PRESSURE': pd.Series(0.0, index=df.index),
                'SCORE_OPPORTUNITY_PRESSURE_ABSORPTION': pd.Series(0.0, index=df.index)
            }
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        absorption_efficiency = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'VPA_EFFICIENCY_D', pd.Series(0.5, index=df.index), method_name="_resolve_pressure_absorption_dynamics"), df.index, ascending=True, tf_weights=default_weights)
        absorption_control = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'vwap_control_strength_D', pd.Series(0.5, index=df.index), method_name="_resolve_pressure_absorption_dynamics"), df.index, ascending=True, tf_weights=default_weights)
        absorption_intent_factor = (intent_diagnosis.clip(-1, 1) + 1) / 2.0
        absorption_quality_score = (absorption_efficiency * absorption_control * absorption_intent_factor).pow(1/3)
        daily_net_force = absorption_quality_score - provisional_pressure
        battlefield_momentum_score = daily_net_force.ewm(span=3, adjust=False).mean().fillna(0)
        base_risk = provisional_pressure * (1.0 - absorption_quality_score)
        risk_amplifier = 1.0 - battlefield_momentum_score.clip(upper=0)
        final_risk_score = (base_risk * risk_amplifier).clip(0, 1)
        base_opportunity = provisional_pressure * absorption_quality_score
        opportunity_amplifier = 1.0 + battlefield_momentum_score.clip(lower=0)
        trend_health = self.strategy.atomic_states.get('SCORE_TREND_HEALTH', pd.Series(0.5, index=df.index))
        context_modulator = 1.0 + trend_health * 0.5
        final_opportunity_score = (base_opportunity * opportunity_amplifier * context_modulator).clip(0, 1)
        states['SCORE_RISK_UNRESOLVED_PRESSURE'] = final_risk_score.astype(np.float32)
        states['SCORE_OPPORTUNITY_PRESSURE_ABSORPTION'] = final_opportunity_score.astype(np.float32)
        return states

    def _diagnose_microstructure_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 信号校验增强版】微观结构意图诊断引擎
        - 核心职责: 融合订单流失衡(OFI)与扫单强度，生成一个全新的原子信号 `SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT`，
                    用于捕捉最细微的主力攻击或撤退意图。
        - 数学思想: OFI反映了挂单册上的力量变化，扫单强度反映了直接吃单的决心。两者结合，可以区分是“温和吸筹”还是“暴力抢筹”。
        """
        required_signals = ['order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_microstructure_intent"):
            return {}
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        ofi_raw = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_microstructure_intent")
        buy_sweep_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_microstructure_intent")
        sell_sweep_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_microstructure_intent")
        ofi_score = get_adaptive_mtf_normalized_bipolar_score(ofi_raw, df.index, default_weights)
        buy_sweep_score = get_adaptive_mtf_normalized_score(buy_sweep_raw, df.index, ascending=True, tf_weights=default_weights)
        sell_sweep_score = get_adaptive_mtf_normalized_score(sell_sweep_raw, df.index, ascending=True, tf_weights=default_weights)
        bullish_intent = (ofi_score.clip(lower=0) * 0.5 + buy_sweep_score * 0.5)
        bearish_intent = (ofi_score.clip(upper=0).abs() * 0.5 + sell_sweep_score * 0.5)
        micro_intent_score = (bullish_intent - bearish_intent).clip(-1, 1)
        states['SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT'] = micro_intent_score.astype(np.float32)
        return states

    def _diagnose_stagnation_evidence(self, df: pd.DataFrame, upward_efficiency: pd.Series) -> pd.Series:
        """
        【V3.0 · 宏观共振版】诊断内部行为信号：滞涨证据
        - 核心重构: 在V2.0“多头衰竭 vs 空头伏击”模型基础上，嫁接一个“宏观风险放大器”，实现微观行为与宏观势能的风险共振。
        - 微观增强: 在“多头衰竭”维度中，新增“筹码疲劳度”证据。
        - 宏观放大: 引入“获利盘压力”和“趋势活力衰减”作为风险放大器，一波巨大上涨后的滞涨，其风险将被指数级放大。
        - 最终证据 = (微观冲突) * (宏观风险放大器)，旨在捕捉最危险的、发生在趋势末端的滞涨信号。
        """
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 数据准备
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_stagnation_evidence")
        price_accel = self._get_safe_series(df, 'ACCEL_5_pct_change_D', 0.0, method_name="_diagnose_stagnation_evidence")
        upper_shadow = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_stagnation_evidence")
        closing_deviation = self._get_safe_series(df, 'closing_price_deviation_score_D', 0.5, method_name="_diagnose_stagnation_evidence")
        distribution_pressure = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_stagnation_evidence")
        active_selling = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name="_diagnose_stagnation_evidence")
        chip_fatigue = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name="_diagnose_stagnation_evidence") # 新增
        winner_rate = self._get_safe_series(df, 'total_winner_rate_D', 50.0, method_name="_diagnose_stagnation_evidence") # 新增
        trend_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name="_diagnose_stagnation_evidence") # 新增
        # 2. 计算“微观滞涨证据”
        # 2.1 “多头衰竭 (Bullish Exhaustion)” - 增强版
        inefficiency_score = (1 - upward_efficiency).clip(0, 1)
        momentum_decay_score = get_adaptive_mtf_normalized_score(price_accel.clip(upper=0).abs(), df_index, ascending=True, tf_weights=default_weights)
        intraday_failure_score = get_adaptive_mtf_normalized_score(upper_shadow * (1 - closing_deviation), df_index, ascending=True, tf_weights=default_weights)
        chip_fatigue_score = get_adaptive_mtf_normalized_score(chip_fatigue, df_index, ascending=True, tf_weights=default_weights) # 新增
        bullish_exhaustion = (inefficiency_score * momentum_decay_score * intraday_failure_score * chip_fatigue_score).pow(1/4).fillna(0.0)
        # 2.2 “空头伏击 (Bearish Ambush)”
        distribution_score = get_adaptive_mtf_normalized_score(distribution_pressure, df_index, ascending=True, tf_weights=default_weights)
        active_selling_score = get_adaptive_mtf_normalized_score(active_selling, df_index, ascending=True, tf_weights=default_weights)
        bearish_ambush = np.maximum(distribution_score, active_selling_score)
        micro_conflict_score = (bullish_exhaustion * bearish_ambush).pow(0.5)
        # 3. 构建“宏观风险放大器”
        profit_pressure_score = get_adaptive_mtf_normalized_score(winner_rate, df_index, ascending=True, tf_weights=default_weights)
        vitality_decay_raw = trend_vitality.diff(3).clip(upper=0).abs() # 趋势活力3日衰减量
        vitality_decay_score = get_adaptive_mtf_normalized_score(vitality_decay_raw, df_index, ascending=True, tf_weights=default_weights)
        macro_amplifier = 1 + (profit_pressure_score * vitality_decay_score).pow(0.5)
        # 4. 非线性合成最终证据
        stagnation_evidence = micro_conflict_score * macro_amplifier
        is_rising_or_flat = (pct_change >= -0.005).astype(float)
        final_stagnation_evidence = (stagnation_evidence * is_rising_or_flat).clip(0, 1)
        return final_stagnation_evidence.astype(np.float32)

    def _diagnose_lower_shadow_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 反击品质版】诊断下影线承接品质
        - 核心重构: 将单一的“承接强度”升级为“反击品质”三维诊断模型，旨在识别真实的、由主力主导的强势反击，过滤诱多陷阱。
        - 1. 反击幅度 (Magnitude): 基础的下影线强度。
        - 2. 主力意图 (Intent): 主力资金当天是否净流入？这是识别真伪的核心。
        - 3. 战场位置 (Location): 反击是否发生在关键支撑位？
        - 非线性合成: 品质分 = (幅度^0.3 * 意图^0.5 * 位置^0.2)，主力意图具备最高的权重和一票否决能力。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 获取三维度原始数据
        magnitude_raw = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_diagnose_lower_shadow_quality").replace(0, 1e-9)
        location_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        # 2. 计算各维度得分
        magnitude_score = get_adaptive_mtf_normalized_score(magnitude_raw, df.index, ascending=True, tf_weights=default_weights)
        # 主力意图得分：只有主力净流入才算有效意图
        flow_ratio = main_force_flow / amount
        intent_score = get_adaptive_mtf_normalized_score(flow_ratio.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        location_score = get_adaptive_mtf_normalized_score(location_raw, df.index, ascending=True, tf_weights=default_weights)
        # 3. 非线性合成
        lower_shadow_quality_score = (magnitude_score.pow(0.3) * intent_score.pow(0.5) * location_score.pow(0.2)).fillna(0.0)
        return lower_shadow_quality_score.astype(np.float32)

    def _diagnose_upper_shadow_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 派发风险版】诊断上影线派发风险
        - 核心重构: 将单一的“上影线压力”升级为“派发风险”三维诊断模型，旨在识别真实的、由主力主导的拉高派发行为。
        - 1. 压力强度 (Magnitude): 基础的上影线抛压强度。
        - 2. 主力行为 (Behavior): 主力资金当天是否净流出？
        - 3. 派发过程 (Process): 是否伴随了“拉高出货”的特征？
        - 非线性合成: 风险分 = (强度^0.3 * 行为^0.4 * 过程^0.3)，精准锁定“天线杀”风险。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 获取三维度原始数据
        magnitude_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_upper_shadow_risk")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_upper_shadow_risk")
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_diagnose_upper_shadow_risk").replace(0, 1e-9)
        process_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_upper_shadow_risk")
        # 2. 计算各维度得分
        magnitude_score = get_adaptive_mtf_normalized_score(magnitude_raw, df.index, ascending=True, tf_weights=default_weights)
        # 主力行为得分：主力净流出额越大，得分越高
        flow_ratio = main_force_flow / amount
        behavior_score = get_adaptive_mtf_normalized_score(flow_ratio.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        process_score = get_adaptive_mtf_normalized_score(process_raw, df.index, ascending=True, tf_weights=default_weights)
        # 3. 非线性合成
        upper_shadow_risk_score = (magnitude_score.pow(0.3) * behavior_score.pow(0.4) * process_score.pow(0.3)).fillna(0.0)
        return upper_shadow_risk_score.astype(np.float32)

    def _diagnose_ambush_counterattack(self, df: pd.DataFrame, lower_shadow_quality: pd.Series) -> pd.Series:
        """
        【V1.0 · 伏击战术版】诊断伏击式反攻信号
        - 核心目标: 识别主力利用日内恐慌进行大规模隐蔽吸筹后发动的强势反攻。
        - 战术要素:
          1. 反击形态 (Morphology): 必须有高质量的长下影线。
          2. 恐慌环境 (Panic): 日内必须出现过恐慌性抛售。
          3. 主力伏击 (Ambush): 主力展现出精准的低吸高抛能力。
          4. 隐蔽吸筹 (Covert Ops): 存在隐蔽吸筹的微观证据。
        - 非线性合成: 信号分 = (形态^0.3 * 环境^0.2 * 伏击^0.4 * 隐蔽^0.1)，赋予主力行为最高权重。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 获取战术要素原始数据
        panic_evidence_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_ambush_counterattack")
        main_force_alpha_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_ambush_counterattack")
        covert_ops_raw = self._get_safe_series(df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_ambush_counterattack")
        # 2. 归一化各要素
        morphology_score = lower_shadow_quality # 直接使用传入的高质量下影线分数
        panic_score = get_adaptive_mtf_normalized_score(panic_evidence_raw, df.index, ascending=True, tf_weights=default_weights)
        ambush_score = get_adaptive_mtf_normalized_score(main_force_alpha_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        covert_ops_score = get_adaptive_mtf_normalized_score(covert_ops_raw, df.index, ascending=True, tf_weights=default_weights)
        # 3. 战术合成
        ambush_counterattack_score = (morphology_score.pow(0.3) * panic_score.pow(0.2) * ambush_score.pow(0.4) * covert_ops_score.pow(0.1)).fillna(0.0)
        return ambush_counterattack_score.astype(np.float32)

    def _diagnose_breakout_failure_risk(self, df: pd.DataFrame, upper_shadow_risk: pd.Series) -> pd.Series:
        """
        【V1.0 · 牛市陷阱版】诊断突破失败级联风险
        - 核心目标: 识别并量化“假突破”或“牛市陷阱”形态的风险，其核心是大量跟风盘在高位被套牢。
        - 风险要素:
          1. 突破尝试 (Attempt): 当天必须有向上突破近期高点的行为。
          2. 快速溃败 (Collapse): 突破后快速回落，形成高风险上影线。
          3. 高位换手 (Volume): 突破时成交量显著放大，意味着大量资金被套。
        - 非线性合成: 风险分 = 突破尝试(布尔) * (快速溃败^0.6 * 高位换手^0.4)，只有在突破失败时才激活。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 定义突破尝试
        high_price = self._get_safe_series(df, 'high_D', 0.0, method_name="_diagnose_breakout_failure_risk")
        recent_high = high_price.rolling(window=21, min_periods=21).max().shift(1)
        is_breakout_attempt = (high_price > recent_high).astype(float)
        # 2. 获取其他风险要素
        collapse_score = upper_shadow_risk # 直接使用传入的高风险上影线分数
        volume_raw = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_diagnose_breakout_failure_risk")
        volume_score = get_adaptive_mtf_normalized_score(volume_raw, df.index, ascending=True, tf_weights=default_weights)
        # 3. 风险合成
        breakout_failure_risk = is_breakout_attempt * (collapse_score.pow(0.6) * volume_score.pow(0.4)).fillna(0.0)
        return breakout_failure_risk.astype(np.float32)

    def _diagnose_divergence_quality(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        【V2.0 · 背离品质版】诊断高品质价量/价资背离
        - 核心重构: 废弃简单的线性差值，引入“背离品质”三维诊断模型，旨在识别由主力意图确认的、发生在关键位置的、真正有效的背离信号。
        - 诊断维度:
          1. 背离形态 (Morphology): 价格趋势与资金流趋势的明确“背道而驰”。
          2. 主力意图 (Intent): 主力资金流向是背离的核心驱动力。
          3. 战场位置 (Location): 背离是否发生在超卖/超买的极端区域。
        - 输出: 分别输出独立的、[0, 1]区间的牛市背离品质分和熊市背离品质分。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 获取原始数据
        price = self._get_safe_series(df, 'close_D', 0.0, method_name="_diagnose_divergence_quality")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_divergence_quality")
        bias = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name="_diagnose_divergence_quality")
        # 2. 计算各指标的趋势 (使用短期EMA的斜率来表示)
        price_trend = price.ewm(span=5, adjust=False).mean().diff().fillna(0)
        flow_trend = main_force_flow.ewm(span=5, adjust=False).mean().diff().fillna(0)
        # 3. 归一化趋势和位置指标
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend, df.index, default_weights)
        norm_flow_trend = get_adaptive_mtf_normalized_bipolar_score(flow_trend, df.index, default_weights)
        norm_bias = get_adaptive_mtf_normalized_bipolar_score(bias, df.index, default_weights)
        # 4. 计算牛市背离品质 (价格跌，资金涨，位置超卖)
        bullish_morphology_evidence = (norm_flow_trend.clip(lower=0) - norm_price_trend.clip(upper=0).abs()).clip(lower=0)
        oversold_location_evidence = norm_bias.clip(upper=0).abs()
        bullish_divergence_quality = (bullish_morphology_evidence * oversold_location_evidence).pow(0.5).fillna(0.0)
        # 5. 计算熊市背离品质 (价格涨，资金跌，位置超买)
        bearish_morphology_evidence = (norm_price_trend.clip(lower=0) - norm_flow_trend.clip(upper=0).abs()).clip(lower=0)
        overbought_location_evidence = norm_bias.clip(lower=0)
        bearish_divergence_quality = (bearish_morphology_evidence * overbought_location_evidence).pow(0.5).fillna(0.0)
        return bullish_divergence_quality.astype(np.float32), bearish_divergence_quality.astype(np.float32)









