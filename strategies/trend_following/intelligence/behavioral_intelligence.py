# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional, List, Any
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    get_adaptive_mtf_normalized_score, is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score
)

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
        【V30.0 · 背离品质重构版】行为情报模块总指挥
        - 核心重构: 废弃了旧的、基于单一双极性信号 `SCORE_BEHAVIOR_PRICE_VS_VOLUME_DIVERGENCE` 的背离处理逻辑。
                    现在，高质量的牛熊背离信号由 `_diagnose_behavioral_axioms` 内部的 `_diagnose_divergence_quality` 方法直接生成，
                    确保了信号的独立性和诊断的深度。
        """
        all_behavioral_states = {}
        atomic_signals = self._diagnose_behavioral_axioms(df)
        # 如果核心公理诊断失败，则提前返回，防止后续错误
        if not atomic_signals:
            print("    -> [行为情报引擎] 核心公理诊断失败，行为分析中止。")
            return {}
        self.strategy.atomic_states.update(atomic_signals)
        all_behavioral_states.update(atomic_signals)
        micro_intent_signals = self._diagnose_microstructure_intent(df)
        self.strategy.atomic_states.update(micro_intent_signals)
        all_behavioral_states.update(micro_intent_signals)
        context_new_high_strength = self._diagnose_context_new_high_strength(df)
        self.strategy.atomic_states.update(context_new_high_strength)
        all_behavioral_states.update(context_new_high_strength)
        # 移除旧的、基于 bipolar_to_exclusive_unipolar 的背离处理逻辑
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
        【V1.3 · 意图解读重构版】行为K线质量分计算引擎
        - 核心修改: 废弃了基础的K线形态指标，全面转向使用更能反映主力意图和过程质量的微观结构信号。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        # 更新依赖信号列表，使用新一代的意图解读型信号
        required_signals = [
            'intraday_posture_score_D', 'microstructure_efficiency_index_D', 'impulse_quality_ratio_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_behavioral_day_quality"):
            return pd.Series(0.0, index=df.index)
        print("开始执行【V1.3 · 意图解读重构版】行为K线质量分计算...")
        # 结果评估：使用“日内姿态分”作为对全天博弈结果的评估，它比单纯的收盘位置更全面
        outcome_score = self._get_safe_series(df, 'intraday_posture_score_D', 0.0, method_name="_calculate_behavioral_day_quality").clip(-1, 1)
        # 过程质量评估：融合“微观结构效率”和“脉冲质量”，评估日内走势的含金量
        micro_efficiency = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'microstructure_efficiency_index_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index)
        impulse_quality = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'impulse_quality_ratio_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index)
        # 将过程质量分转化为[-1, 1]的双极性分数
        process_quality_score = ((micro_efficiency * impulse_quality).pow(0.5) * 2 - 1).clip(-1, 1)
        # 最终质量分 = 结果 * 40% + 过程 * 60%
        day_quality_score = (
            outcome_score * 0.4 +
            process_quality_score * 0.6
        ).clip(-1, 1)
        print("【意图解读重构版行为K线质量分】计算完成。")
        return day_quality_score.astype(np.float32)

    def _diagnose_behavioral_axioms(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V33.3 · 信号链修复版】原子信号中心
        - 核心修复: 修复了 `_diagnose_offensive_absorption_intent` 的调用逻辑，
                    通过直接传递 `lower_shadow_quality` 参数，解决了信号链断裂的BUG。
        - 核心升级: 新增对“进攻性承接意图”和“博弈欺骗指数”两大诡道博弈信号的诊断，提升对主力真实意图的洞察力。
        """
        required_signals = [
            'close_D', 'high_D', 'low_D', 'open_D', 'volume_D', 'amount_D', 'pct_change_D',
            'volume_ratio_D', 'turnover_rate_f_D', 'main_force_net_flow_calibrated_D',
            'retail_net_flow_calibrated_D', 'net_mf_amount_D', 'buy_elg_amount_D', 'buy_lg_amount_D',
            'dip_absorption_power_D', 'lower_shadow_absorption_strength_D',
            'rally_distribution_pressure_D', 'upper_shadow_selling_pressure_D',
            'profit_taking_flow_ratio_D', 'main_force_execution_alpha_D',
            'SLOPE_5_main_force_conviction_index_D', 'breakout_quality_score_D',
            'SLOPE_5_breakout_quality_score_D', 'total_winner_rate_D', 'winner_stability_index_D',
            'control_solidity_index_D', 'trend_vitality_index_D', 'BIAS_21_D', 'RSI_13_D',
            'ACCEL_5_pct_change_D', 'closing_strength_index_D', 'active_selling_pressure_D',
            'chip_fatigue_index_D', 'main_force_ofi_D', 'retail_ofi_D', 'buy_quote_exhaustion_rate_D',
            'sell_quote_exhaustion_rate_D',
            'microstructure_efficiency_index_D', 'upward_impulse_purity_D', 'vacuum_traversal_efficiency_D',
            'support_validation_strength_D', 'impulse_quality_ratio_D', 'floating_chip_cleansing_efficiency_D',
            'panic_selling_cascade_D', 'capitulation_absorption_index_D', 'covert_accumulation_signal_D',
            'VOL_MA_5_D', 'VOL_MA_13_D', 'VOL_MA_21_D', 'loser_pain_index_D',
            'deception_index_D', 'wash_trade_intensity_D', 'closing_auction_ambush_D', 'mf_retail_battle_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_behavioral_axioms"):
            print("    -> [行为情报引擎] 核心公理诊断失败，行为分析中止。")
            return {}
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        long_term_weights = get_param_value(p_conf.get('long_term_weights'), {'weights': {21: 0.5, 55: 0.3, 89: 0.2}})
        # --- 基础信号计算 ---
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_behavioral_axioms")
        closing_deviation = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_diagnose_behavioral_axioms")
        intraday_posture = self._get_safe_series(df, 'intraday_posture_score_D', 0.0, method_name="_diagnose_behavioral_axioms")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_behavioral_axioms")
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_diagnose_behavioral_axioms").replace(0, 1e-9)
        bias_21 = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name="_diagnose_behavioral_axioms")
        if 'ACCEL_5_pct_change_D' in df.columns:
            price_accel = self._get_safe_series(df, 'ACCEL_5_pct_change_D', 0.0, method_name="_diagnose_behavioral_axioms")
        else:
            print("    -> [行为情报兼容模式] _diagnose_behavioral_axioms: 未找到 'ACCEL_5_pct_change_D'，使用 'pct_change_D' 的5日差分作为代理。")
            price_accel = pct_change.diff(5).fillna(0.0)
        # --- 动能信号 ---
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
        # --- 超买信号 ---
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
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_diagnose_behavioral_axioms")
        volume_amplifier = (1 + get_adaptive_mtf_normalized_score(volume_ratio, df.index, ascending=True, tf_weights=default_weights)).clip(lower=1)
        final_overextension_score = (overextension_raw_score * volume_amplifier).clip(0, 1)
        states['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = final_overextension_score.astype(np.float32)
        # --- 行为铁三角 ---
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
        # --- 影线、战术与派发意图信号 ---
        lower_shadow_quality = self._diagnose_lower_shadow_quality(df)
        distribution_intent = self._calculate_distribution_intent(df, default_weights)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_quality
        states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'] = distribution_intent
        states['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK'] = self._diagnose_ambush_counterattack(df, lower_shadow_quality)
        states['SCORE_RISK_BREAKOUT_FAILURE_CASCADE'] = self._diagnose_breakout_failure_risk(df, distribution_intent)
        # --- 量能与博弈信号 ---
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = self._calculate_volume_burst_quality(df, default_weights)
        states['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = self._calculate_volume_atrophy(df, default_weights)
        states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'] = self._calculate_absorption_strength(df, default_weights)
        states['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION'] = self._diagnose_shakeout_confirmation(
            df,
            states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'],
            states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'],
            states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT']
        )
        # --- 开始重构背离信号 ---
        bullish_divergence_quality, bearish_divergence_quality = self._diagnose_divergence_quality(df)
        states['SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY'] = bullish_divergence_quality
        states['SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY'] = bearish_divergence_quality
        # --- 机会与风险信号 ---
        is_rising = (pct_change > 0).astype(float)
        is_falling = (pct_change < 0).astype(float)
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = (is_rising * states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] * states['SCORE_BEHAVIOR_VOLUME_ATROPHY']).pow(1/3).astype(np.float32)
        capitulation_raw = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_behavioral_axioms")
        selling_deceleration_score = (1 - get_adaptive_mtf_normalized_score(price_accel.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)).clip(0, 1)
        capitulation_confirm_score = get_adaptive_mtf_normalized_score(capitulation_raw, df.index, ascending=True, tf_weights=default_weights)
        selling_exhaustion_score = (
            states['SCORE_BEHAVIOR_VOLUME_ATROPHY'].pow(0.3) *
            states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'].pow(0.3) *
            selling_deceleration_score.pow(0.2) *
            capitulation_confirm_score.pow(0.2)
        ).fillna(0.0)
        states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = (is_falling * selling_exhaustion_score).astype(np.float32)
        states['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'] = self._diagnose_stagnation_evidence(df, states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'])
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_BURST'] * states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM']).pow(1/2).astype(np.float32)
        # 将计算好的 lower_shadow_quality 直接作为参数传递，修复信号链
        states['SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT'] = self._diagnose_offensive_absorption_intent(df, lower_shadow_quality)
        states['SCORE_BEHAVIOR_DECEPTION_INDEX'] = self._diagnose_deception_index(df)
        return states

    def _calculate_volume_atrophy(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.2 · 探针逻辑重构版】计算 SCORE_BEHAVIOR_VOLUME_ATROPHY 信号。
        - 核心重构: 彻底重构了探针逻辑，使其不再依赖于数据集的最后一天。现在探针会遍历
                      `probe_dates` 配置，并为每个在数据集中找到的日期精确打印当日的详细信息，
                      完美适配历史区间调试。
        """
        df_index = df.index
        vol = self._get_safe_series(df, 'volume_D', 0.0, method_name="_calculate_volume_atrophy")
        vol_ma5 = self._get_safe_series(df, 'VOL_MA_5_D', 0.0, method_name="_calculate_volume_atrophy")
        vol_ma13 = self._get_safe_series(df, 'VOL_MA_13_D', 0.0, method_name="_calculate_volume_atrophy")
        vol_ma21 = self._get_safe_series(df, 'VOL_MA_21_D', 0.0, method_name="_calculate_volume_atrophy")
        if vol.isnull().all() or vol_ma5.isnull().all() or vol_ma13.isnull().all() or vol_ma21.isnull().all():
            print("    -> [行为情报引擎警告] 缺少成交量或成交量均线数据，无法计算 SCORE_BEHAVIOR_VOLUME_ATROPHY。")
            return pd.Series(0.0, index=df_index)
        vol_ma21_safe = vol_ma21.replace(0, 1e-9)
        vol_below_ma21_raw = (1 - (vol / vol_ma21_safe)).clip(0, 1)
        vol_below_ma21_score = get_adaptive_mtf_normalized_score(vol_below_ma21_raw, df_index, ascending=True, tf_weights=tf_weights)
        vol_ma5_below_ma21_raw = (1 - (vol_ma5 / vol_ma21_safe)).clip(0, 1)
        vol_ma5_below_ma21_score = get_adaptive_mtf_normalized_score(vol_ma5_below_ma21_raw, df_index, ascending=True, tf_weights=tf_weights)
        is_ice_point_atrophy = ((vol < vol_ma5) & (vol_ma5 < vol_ma21)).astype(float)
        ice_point_atrophy_persistence = is_ice_point_atrophy.rolling(window=5, min_periods=1).sum() / 5
        ice_point_atrophy_score = is_ice_point_atrophy * (0.5 + ice_point_atrophy_persistence * 0.5)
        is_ma_bearish_alignment = ((vol_ma5 < vol_ma13) & (vol_ma13 < vol_ma21)).astype(float)
        ma_bearish_alignment_strength_raw = (vol_ma21 - vol_ma5) / vol_ma21_safe.replace(0, np.nan)
        ma_bearish_alignment_score = get_adaptive_mtf_normalized_score(ma_bearish_alignment_strength_raw.clip(lower=0), df_index, ascending=True, tf_weights=tf_weights)
        ma_bearish_alignment_final_score = is_ma_bearish_alignment * ma_bearish_alignment_score
        evidence_components = [vol_below_ma21_score, vol_ma5_below_ma21_score, ice_point_atrophy_score, ma_bearish_alignment_final_score]
        weights = np.array([0.2, 0.2, 0.3, 0.3])
        aligned_evidence_components = [comp.reindex(df_index, fill_value=0.0) for comp in evidence_components]
        safe_evidence_components = [comp + 1e-9 for comp in aligned_evidence_components]
        base_atrophy_score = pd.Series(np.prod([comp.values ** w for comp, w in zip(safe_evidence_components, weights)], axis=0), index=df_index)
        lockup_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_calculate_volume_atrophy")
        exhaustion_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_calculate_volume_atrophy")
        cleansing_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_calculate_volume_atrophy")
        lockup_factor = get_adaptive_mtf_normalized_score(lockup_raw, df.index, ascending=True, tf_weights=tf_weights)
        exhaustion_factor = get_adaptive_mtf_normalized_score(exhaustion_raw, df.index, ascending=True, tf_weights=tf_weights)
        cleansing_factor = get_adaptive_mtf_normalized_score(cleansing_raw, df.index, ascending=True, tf_weights=tf_weights)
        context_modulator = (lockup_factor * exhaustion_factor * cleansing_factor).pow(1/3).fillna(0.0)
        high_quality_atrophy_score = (base_atrophy_score * context_modulator).pow(0.5)
        # --- 彻底重构探针逻辑以适配历史回溯 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _calculate_volume_atrophy @ {probe_date_str}")
                print(f"        - 基础萎缩分: {base_atrophy_score.loc[probe_ts]:.4f}")
                print(f"        - 环境调节器原始值: 锁定度={lockup_raw.loc[probe_ts]:.2f}, 枯竭度={exhaustion_raw.loc[probe_ts]:.2f}, 清洗度={cleansing_raw.loc[probe_ts]:.2f}")
                print(f"        - 环境调节器因子: 锁定={lockup_factor.loc[probe_ts]:.4f}, 枯竭={exhaustion_factor.loc[probe_ts]:.4f}, 清洗={cleansing_factor.loc[probe_ts]:.4f} -> 综合={context_modulator.loc[probe_ts]:.4f}")
                print(f"        - 最终高品质萎缩分: {high_quality_atrophy_score.loc[probe_ts]:.4f}")
        return high_quality_atrophy_score.clip(0, 1)

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
        # 战前情报校验
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
        【V1.3 · 主力意图聚焦版】微观结构意图诊断引擎
        - 核心升级: 将通用的订单流失衡(OFI)替换为“主力订单流失衡(main_force_ofi_D)”，
                    从而更精准地聚焦于市场主导力量的真实意图，排除散户噪音。
        """
        # 将依赖信号从通用OFI升级为主力OFI
        required_signals = ['main_force_ofi_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_microstructure_intent"):
            return {}
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 获取主力OFI原始数据
        ofi_raw = self._get_safe_series(df, 'main_force_ofi_D', 0.0, method_name="_diagnose_microstructure_intent")
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
        【V3.6 · 或门逻辑版】诊断内部行为信号：滞涨证据
        - 核心重构: 修复了“微观冲突”计算中“与门”逻辑过于严苛的盲点。
                      新逻辑采用 `max(多头衰竭, 空头伏击)`，将“与”门改为“或”门，
                      确保只要任何一方出现极端风险，信号就能被准确捕捉。
        - ... (其他注释保持不变)
        """
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 数据准备
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_stagnation_evidence")
        if 'ACCEL_5_pct_change_D' in df.columns:
            price_accel = self._get_safe_series(df, 'ACCEL_5_pct_change_D', 0.0, method_name="_diagnose_stagnation_evidence")
        else:
            print("    -> [行为情报兼容模式] _diagnose_stagnation_evidence: 未找到 'ACCEL_5_pct_change_D'，使用 'pct_change_D' 的5日差分作为代理。")
            price_accel = pct_change.diff(5).fillna(0.0)
        closing_deviation = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_diagnose_stagnation_evidence")
        distribution_pressure = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_stagnation_evidence")
        active_selling = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name="_diagnose_stagnation_evidence")
        chip_fatigue = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name="_diagnose_stagnation_evidence")
        winner_rate = self._get_safe_series(df, 'total_winner_rate_D', 50.0, method_name="_diagnose_stagnation_evidence")
        trend_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name="_diagnose_stagnation_evidence")
        # 2. 计算“微观滞涨证据”
        inefficiency_score = (1 - upward_efficiency).clip(0, 1)
        momentum_decay_score = get_adaptive_mtf_normalized_score(price_accel.clip(upper=0).abs(), df_index, ascending=True, tf_weights=default_weights)
        intraday_failure_score = (1 - closing_deviation).clip(0, 1)
        chip_fatigue_score = get_adaptive_mtf_normalized_score(chip_fatigue, df_index, ascending=True, tf_weights=default_weights)
        bullish_exhaustion = (inefficiency_score * momentum_decay_score * intraday_failure_score * chip_fatigue_score).pow(1/4).fillna(0.0)
        distribution_score = get_adaptive_mtf_normalized_score(distribution_pressure, df_index, ascending=True, tf_weights=default_weights)
        active_selling_score = get_adaptive_mtf_normalized_score(active_selling, df_index, ascending=True, tf_weights=default_weights)
        bullish_failure_score = (1 - closing_deviation).clip(0, 1)
        process_evidence = (distribution_score * active_selling_score).pow(0.5)
        outcome_evidence = bullish_failure_score
        bearish_ambush = (process_evidence * 0.7 + outcome_evidence * 0.3)
        # 将“与门”逻辑改为“或门”逻辑
        micro_conflict_score = np.maximum(bullish_exhaustion, bearish_ambush)
        # 3. 构建“宏观风险放大器”
        profit_pressure_score = get_adaptive_mtf_normalized_score(winner_rate, df_index, ascending=True, tf_weights=default_weights)
        vitality_decay_raw = trend_vitality.diff(3).clip(upper=0).abs()
        vitality_decay_score = get_adaptive_mtf_normalized_score(vitality_decay_raw, df_index, ascending=True, tf_weights=default_weights)
        macro_amplifier = 1 + (profit_pressure_score * vitality_decay_score).pow(0.5)
        # 4. 非线性合成最终证据
        stagnation_evidence = micro_conflict_score * macro_amplifier
        is_rising_or_flat = (pct_change >= -0.005).astype(float)
        final_stagnation_evidence = (stagnation_evidence * is_rising_or_flat).clip(0, 1)
        # --- 彻底重构探针逻辑以适配历史回溯 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _diagnose_stagnation_evidence @ {probe_date_str}")
                print(f"        - 多头衰竭分: {bullish_exhaustion.loc[probe_ts]:.4f}")
                print(f"        - 空头伏击分 (新): {bearish_ambush.loc[probe_ts]:.4f} (过程分={process_evidence.loc[probe_ts]:.2f}, 结果分={outcome_evidence.loc[probe_ts]:.2f})")
                print(f"        - 微观冲突分(或门): {micro_conflict_score.loc[probe_ts]:.4f}")
                print(f"        - 宏观放大器: {macro_amplifier.loc[probe_ts]:.4f}")
                print(f"        - 最终滞涨证据分: {final_stagnation_evidence.loc[probe_ts]:.4f}")
        return final_stagnation_evidence.astype(np.float32)

    def _diagnose_lower_shadow_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.6 · 双极性校准版】诊断下影线承接品质。
        - 核心修复: 使用专属的 `_get_calibrated_adaptive_mtf_bipolar_score` 对主力执行Alpha进行归一化，
                      代码更简洁，并彻底修复了“归一化悖论”BUG。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 1. 计算基础K线品质分 (逻辑不变) ---
        magnitude_raw = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_diagnose_lower_shadow_quality").replace(0, 1e-9)
        location_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        magnitude_score = get_adaptive_mtf_normalized_score(magnitude_raw, df.index, ascending=True, tf_weights=default_weights)
        flow_ratio = main_force_flow / amount
        intent_score = get_adaptive_mtf_normalized_score(flow_ratio.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        location_score = get_adaptive_mtf_normalized_score(location_raw, df.index, ascending=True, tf_weights=default_weights)
        base_quality_score = (magnitude_score.pow(0.3) * intent_score.pow(0.5) * location_score.pow(0.2)).fillna(0.0)
        # --- 2. 构建双向奖惩机制和新的放大器 ---
        panic_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        capitulation_raw = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        ambush_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        # 使用专属的双极性校准函数直接处理主力Alpha
        ambush_intent_score = self._get_calibrated_adaptive_mtf_bipolar_score(ambush_raw, default_weights)
        modulated_quality_score = base_quality_score * (1 + ambush_intent_score * 0.5).clip(0, 2)
        panic_absorption_score = get_adaptive_mtf_normalized_score((panic_raw * capitulation_raw).pow(0.5), df.index, ascending=True, tf_weights=default_weights)
        context_amplifier = 1 + (panic_absorption_score * modulated_quality_score).pow(0.5)
        # --- 3. 非线性合成 ---
        final_lower_shadow_quality = (modulated_quality_score * context_amplifier).clip(0, 1)
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _diagnose_lower_shadow_quality @ {probe_date_str}")
                print(f"        - 基础品质分: {base_quality_score.loc[probe_ts]:.4f} (幅度={magnitude_score.loc[probe_ts]:.2f}, 意图={intent_score.loc[probe_ts]:.2f}, 位置={location_score.loc[probe_ts]:.2f})")
                print(f"        - 伏击意图(校准后): {ambush_intent_score.loc[probe_ts]:.4f} (原始Alpha={ambush_raw.loc[probe_ts]:.2f}) -> 调制后品质分: {modulated_quality_score.loc[probe_ts]:.4f}")
                print(f"        - 智能放大器: 恐慌承接度={panic_absorption_score.loc[probe_ts]:.4f} -> 放大倍数={context_amplifier.loc[probe_ts]:.4f}")
                print(f"        - 最终下影线品质分: {final_lower_shadow_quality.loc[probe_ts]:.4f}")
        return final_lower_shadow_quality.astype(np.float32)

    def _calculate_distribution_intent(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V1.7 · 零值校准版】计算派发意图
        - 核心修复: 使用 `_get_calibrated_adaptive_mtf_score` 对正向主力Alpha进行归一化，
                      修复了0 Alpha被错误解读为强控盘能力，从而不当放大派发风险的BUG。
        """
        required_signals = [
            'rally_distribution_pressure_D', 'upper_shadow_selling_pressure_D',
            'profit_taking_flow_ratio_D', 'main_force_execution_alpha_D',
            'SLOPE_5_main_force_conviction_index_D', 'closing_strength_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_distribution_intent"):
            return pd.Series(0.0, index=df.index)
        # --- 五维证据链融合 (逻辑不变) ---
        rally_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_calculate_distribution_intent")
        process_evidence = get_adaptive_mtf_normalized_score(rally_pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_calculate_distribution_intent")
        outcome_evidence = get_adaptive_mtf_normalized_score(upper_shadow_pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        profit_taking_raw = self._get_safe_series(df, 'profit_taking_flow_ratio_D', 0.0, method_name="_calculate_distribution_intent")
        flow_evidence = get_adaptive_mtf_normalized_score(profit_taking_raw, df.index, ascending=True, tf_weights=tf_weights)
        mf_alpha_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name="_calculate_distribution_intent")
        mf_alpha_bearish = abs(mf_alpha_raw.clip(upper=0))
        main_force_evidence = get_adaptive_mtf_normalized_score(mf_alpha_bearish, df.index, ascending=True, tf_weights=tf_weights)
        conviction_slope_raw = self._get_safe_series(df, 'SLOPE_5_main_force_conviction_index_D', 0.0, method_name="_calculate_distribution_intent")
        conviction_decay = abs(conviction_slope_raw.clip(upper=0))
        conviction_evidence = get_adaptive_mtf_normalized_score(conviction_decay, df.index, ascending=True, tf_weights=tf_weights)
        base_distribution_intent = (
            process_evidence.pow(0.3) *
            outcome_evidence.pow(0.2) *
            flow_evidence.pow(0.2) *
            main_force_evidence.pow(0.15) *
            conviction_evidence.pow(0.15)
        ).fillna(0.0)
        # --- 市场接受度放大器 (逻辑不变) ---
        market_acceptance_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_calculate_distribution_intent")
        market_acceptance_normalized = normalize_score(market_acceptance_raw, df.index, 55)
        acceptance_amplifier = 1 + (market_acceptance_normalized * 0.5)
        # --- 主力控盘能力调节器 ---
        # 使用校准后的归一化函数处理正Alpha
        positive_alpha_score = self._get_calibrated_adaptive_mtf_score(mf_alpha_raw.clip(lower=0), tf_weights)
        control_modulator = 1 + (positive_alpha_score * 0.5)
        # --- 最终合成 ---
        distribution_intent_score = (base_distribution_intent * acceptance_amplifier * control_modulator).clip(0, 1)
        # --- 更新探针逻辑 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _calculate_distribution_intent @ {probe_date_str}")
                print(f"        - 过程证据(新): rally_distribution_pressure_D = {rally_pressure_raw.loc[probe_ts]:.2f} -> 归一化分 = {process_evidence.loc[probe_ts]:.4f}")
                print(f"        - 基础派发意图分: {base_distribution_intent.loc[probe_ts]:.4f}")
                print(f"        - 市场接受度放大器: {acceptance_amplifier.loc[probe_ts]:.4f} (收盘偏离度(原始)={market_acceptance_raw.loc[probe_ts]:.2f}, 净化后={market_acceptance_normalized.loc[probe_ts]:.4f})")
                # 更新探针，明确展示校准后的逻辑
                print(f"        - 控盘能力调节器(校准后): {control_modulator.loc[probe_ts]:.4f} (原始Alpha={mf_alpha_raw.loc[probe_ts]:.2f}, 正Alpha分={positive_alpha_score.loc[probe_ts]:.4f})")
                print(f"        - 最终派发意图分: {distribution_intent_score.loc[probe_ts]:.4f}")
        return distribution_intent_score.astype(np.float32)

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

    def _diagnose_breakout_failure_risk(self, df: pd.DataFrame, distribution_intent: pd.Series) -> pd.Series:
        """
        【V1.1 · 意图驱动版】诊断突破失败级联风险
        - 核心目标: 识别并量化“假突破”或“牛市陷阱”形态的风险，其核心是大量跟风盘在高位被套牢。
        - 风险要素:
          1. 突破尝试 (Attempt): 当天必须有向上突破近期高点的行为。
          2. 快速溃败 (Collapse): 突破后快速回落，其核心证据由全新的“派发意图”信号提供。
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
        collapse_score = distribution_intent # 直接使用传入的、更强大的派发意图分数
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

    def _calculate_volume_burst_quality(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V1.4 · 绝对裁决版】计算高品质看涨量能爆发信号。
        - 核心重构: 修复了“守门员”因归一化逻辑而失职的漏洞。引入基于原始主力资金流的
                      “绝对裁决”机制，确保只有在主力净流入为正时，驱动分才有效，
                      彻底杜绝了主力流出日仍被判为“看涨爆发”的逻辑矛盾。
        - ... (其他注释保持不变)
        """
        # --- 1. 获取四维度原始数据 ---
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_calculate_volume_burst_quality")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_volume_burst_quality")
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_calculate_volume_burst_quality").replace(0, 1e-9)
        efficiency_raw = self._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name="_calculate_volume_burst_quality")
        urgency_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_calculate_volume_burst_quality")
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_volume_burst_quality")
        # --- 2. 计算各维度得分 ---
        magnitude_score = get_adaptive_mtf_normalized_score(volume_ratio, df.index, ascending=True, tf_weights=tf_weights)
        # 实施“绝对裁决”
        flow_ratio = main_force_flow / amount
        # 归一化前先不clip，保留原始信息用于裁决
        driver_score_normalized = get_adaptive_mtf_normalized_score(flow_ratio.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        # 绝对裁决：只有当原始主力资金流为正时，驱动分才有效
        is_positive_flow = (main_force_flow > 0).astype(float)
        driver_score = driver_score_normalized * is_positive_flow
        efficiency_score = get_adaptive_mtf_normalized_score(efficiency_raw, df.index, ascending=True, tf_weights=tf_weights)
        urgency_score = get_adaptive_mtf_normalized_score(urgency_raw, df.index, ascending=True, tf_weights=tf_weights)
        # --- 3. 非线性合成与情景过滤 ---
        is_rising = (pct_change > 0).astype(float)
        other_factors_quality = (magnitude_score * efficiency_score * urgency_score).pow(1/3)
        volume_burst_quality = (driver_score * other_factors_quality * is_rising).fillna(0.0)
        # --- 彻底重构探针逻辑以适配历史回溯 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _calculate_volume_burst_quality @ {probe_date_str}")
                print(f"        - 原始值: 量比={volume_ratio.loc[probe_ts]:.2f}, 主力流={main_force_flow.loc[probe_ts]:.2f}, 效率={efficiency_raw.loc[probe_ts]:.2f}, 紧迫性={urgency_raw.loc[probe_ts]:.2f}")
                print(f"        - 归一化分: 幅度={magnitude_score.loc[probe_ts]:.4f}, 驱动(裁决后)={driver_score.loc[probe_ts]:.4f}, 效率={efficiency_score.loc[probe_ts]:.4f}, 紧迫性={urgency_score.loc[probe_ts]:.4f}")
                print(f"        - 最终爆发品质分: {volume_burst_quality.loc[probe_ts]:.4f} (上涨日: {is_rising.loc[probe_ts]})")
        return volume_burst_quality.clip(0, 1).astype(np.float32)

    def _calculate_absorption_strength(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V1.5 · 三维诊断版】计算下跌吸筹强度
        - 核心重构: 升级为“事件-位置-形态”三维诊断模型，全面提升信号的可靠性和战术价值。
                      - 事件: 引入“投降承接指数”，聚焦于最具反转意义的恐慌盘吸收事件。
                      - 位置: 引入“支撑验证强度”，评估承接是否发生在关键战略位置。
                      - 形态: 保留“形态显著性”修正的下影线，作为最终的K线确认。
        """
        # 更新依赖信号，引入投降承接和支撑验证
        required_signals = ['capitulation_absorption_index_D', 'support_validation_strength_D', 'lower_shadow_absorption_strength_D', 'high_D', 'low_D', 'open_D', 'close_D']
        if not self._validate_required_signals(df, required_signals, "_calculate_absorption_strength"):
            return pd.Series(0.0, index=df.index)
        # --- 形态显著性过滤器 (逻辑不变) ---
        high = self._get_safe_series(df, 'high_D', method_name="_calculate_absorption_strength")
        low = self._get_safe_series(df, 'low_D', method_name="_calculate_absorption_strength")
        open_price = self._get_safe_series(df, 'open_D', method_name="_calculate_absorption_strength")
        close_price = self._get_safe_series(df, 'close_D', method_name="_calculate_absorption_strength")
        total_range = (high - low).replace(0, 1e-9)
        lower_shadow_length = (np.minimum(open_price, close_price) - low).clip(lower=0)
        significance_factor = (lower_shadow_length / total_range).clip(0, 1).fillna(0.0)
        # --- 计算三维得分 ---
        # 1. 事件驱动分
        event_raw = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_calculate_absorption_strength")
        event_score = get_adaptive_mtf_normalized_score(event_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 2. 位置确认分
        location_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name="_calculate_absorption_strength")
        location_score = get_adaptive_mtf_normalized_score(location_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 3. 形态确认分
        morphology_raw = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', 0.0, method_name="_calculate_absorption_strength")
        morphology_score_normalized = get_adaptive_mtf_normalized_score(morphology_raw, df.index, ascending=True, tf_weights=tf_weights)
        morphology_score_adjusted = morphology_score_normalized * significance_factor
        # 最终融合，赋予事件和位置更高权重
        final_score = (event_score * 0.4 + location_score * 0.4 + morphology_score_adjusted * 0.2).clip(0, 1)
        # --- 探针逻辑更新 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _calculate_absorption_strength @ {probe_date_str}")
                # 更新探针输出
                print(f"        - 事件分 (投降承接): {event_score.loc[probe_ts]:.4f} (原始值: {event_raw.loc[probe_ts]:.2f})")
                print(f"        - 位置分 (支撑验证): {location_score.loc[probe_ts]:.4f} (原始值: {location_raw.loc[probe_ts]:.2f})")
                print(f"        - 形态分 (下影线): {morphology_score_adjusted.loc[probe_ts]:.4f} (修正前: {morphology_score_normalized.loc[probe_ts]:.4f}, 显著性: {significance_factor.loc[probe_ts]:.4f})")
                print(f"        - 最终下跌吸筹强度分: {final_score.loc[probe_ts]:.4f}")
        return final_score.astype(np.float32)

    def _diagnose_shakeout_confirmation(self, df: pd.DataFrame, downward_resistance: pd.Series, absorption_strength: pd.Series, distribution_intent: pd.Series) -> pd.Series:
        """
        【V1.2 · 结果验证版】诊断震荡洗盘确认信号
        - 核心升级: 引入“浮筹清洗效率”作为对洗盘“结果”的验证。
                      模型升级为“过程(防守反击) + 意图(无派发) + 结果(清洗效率)”的三维确认模型，
                      确保信号只在真正成功的洗盘后发出。
        """
        # --- 1. 获取核心输入 ---
        # [新增代码行] 获取浮筹清洗效率
        cleansing_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_shakeout_confirmation")
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        cleansing_efficiency_score = get_adaptive_mtf_normalized_score(cleansing_raw, df.index, ascending=True, tf_weights=default_weights)
        # --- 2. 计算防守反击强度 (逻辑不变) ---
        defense_counter_attack_strength = (downward_resistance * absorption_strength).pow(0.5).fillna(0.0)
        # --- 3. 计算非线性的“确定性放大器” (逻辑不变) ---
        certainty_amplifier = (1 - distribution_intent).pow(2).clip(0, 1)
        # --- 4. 战术合成 (三维融合) ---
        # 引入清洗效率作为第三个核心确认维度
        shakeout_confirmation_score = (defense_counter_attack_strength * certainty_amplifier * cleansing_efficiency_score).pow(1/3).clip(0, 1)
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _diagnose_shakeout_confirmation @ {probe_date_str}")
                print(f"        - 防守反击强度: {defense_counter_attack_strength.loc[probe_ts]:.4f} (抵抗={downward_resistance.loc[probe_ts]:.2f}, 吸筹={absorption_strength.loc[probe_ts]:.2f})")
                print(f"        - 确定性放大器: {certainty_amplifier.loc[probe_ts]:.4f} (派发意图={distribution_intent.loc[probe_ts]:.2f})")
                # 在探针中增加新维度的输出
                print(f"        - 清洗效率分(新): {cleansing_efficiency_score.loc[probe_ts]:.4f} (原始值={cleansing_raw.loc[probe_ts]:.2f})")
                print(f"        - 最终洗盘确认分: {shakeout_confirmation_score.loc[probe_ts]:.4f}")
        return shakeout_confirmation_score.astype(np.float32)

    def _diagnose_offensive_absorption_intent(self, df: pd.DataFrame, lower_shadow_quality: pd.Series) -> pd.Series:
        """
        【V1.3 · 鲁棒融合版】诊断进攻性承接意图
        - 核心修复: 将融合算法从几何平均升级为加权算术平均，避免了“零值污染”问题，
                      使得信号在部分证据缺失时依然能提供有效评估，逻辑更具鲁棒性。
        - 核心修复: 全面采用校准后的归一化函数，确保0值输入得到0分输出。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 获取四维评估的原始数据
        morphology_score = lower_shadow_quality
        execution_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_offensive_absorption_intent")
        covert_ops_raw = self._get_safe_series(df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_offensive_absorption_intent")
        context_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_offensive_absorption_intent")
        # 2. 归一化各维度证据
        execution_score = self._get_calibrated_adaptive_mtf_score(execution_raw.clip(lower=0), default_weights)
        covert_ops_score = self._get_calibrated_adaptive_mtf_score(covert_ops_raw, default_weights)
        context_score = get_adaptive_mtf_normalized_score(context_raw, df.index, ascending=True, tf_weights=default_weights)
        # 3. 升级为加权算术平均融合，避免零值污染
        offensive_absorption_intent = (
            morphology_score * 0.20 +
            execution_score * 0.35 +
            covert_ops_score * 0.25 +
            context_score * 0.20
        ).fillna(0.0)
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _diagnose_offensive_absorption_intent @ {probe_date_str}")
                print(f"        - 形态分 (下影线品质): {morphology_score.loc[probe_ts]:.4f}")
                print(f"        - 执行分 (正Alpha-校准后): {execution_score.loc[probe_ts]:.4f} (原始值: {execution_raw.loc[probe_ts]:.2f})")
                print(f"        - 隐蔽分 (隐蔽吸筹-校准后): {covert_ops_score.loc[probe_ts]:.4f} (原始值: {covert_ops_raw.loc[probe_ts]:.2f})")
                print(f"        - 环境分 (恐慌级联): {context_score.loc[probe_ts]:.4f} (原始值: {context_raw.loc[probe_ts]:.2f})")
                print(f"        - 最终进攻性承接意图分: {offensive_absorption_intent.loc[probe_ts]:.4f}")
        return offensive_absorption_intent.clip(0, 1).astype(np.float32)

    def _diagnose_deception_index(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · 净化裁决版】诊断博弈欺骗指数
        - 核心修复: 修复了因直接使用原始 `closing_strength_index_D` 而导致计算结果为负数的BUG。
                      引入 `normalize_score` 对收盘强弱度进行净化，确保“强势”与“弱势”的评估基于
                      历史相对位置，使计算逻辑更严谨。
        - 核心目标: 量化主力的欺骗性操盘行为，识别“压价吸筹”与“拉高出货”。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 获取原始数据
        base_deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_diagnose_deception_index")
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_deception_index")
        # 获取原始的收盘强弱度，不再直接使用
        closing_strength_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_diagnose_deception_index")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_deception_index")
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_diagnose_deception_index").replace(0, 1e-9)
        closing_ambush_raw = self._get_safe_series(df, 'closing_auction_ambush_D', 0.0, method_name="_diagnose_deception_index")
        # 2. 归一化和计算组件
        deception_evidence_score = get_adaptive_mtf_normalized_score((base_deception_raw + wash_trade_raw).pow(0.5), df.index, ascending=True, tf_weights=default_weights)
        flow_ratio = main_force_flow / amount
        flow_direction_score = get_adaptive_mtf_normalized_bipolar_score(flow_ratio, df.index, default_weights)
        # [新增代码行] 对收盘强弱度进行净化，得到[0,1]区间的相对强势度
        normalized_strength = normalize_score(closing_strength_raw, df.index, 55)
        # 3. 计算看涨/看跌欺骗
        # 看涨欺骗 = 弱势表象 * 欺骗行为 * 主力净流入
        bullish_deception_score = (1 - normalized_strength) * deception_evidence_score * flow_direction_score.clip(lower=0)
        # 看跌欺骗 = 强势表象 * 欺骗行为 * 主力净流出
        bearish_deception_score = normalized_strength * deception_evidence_score * flow_direction_score.clip(upper=0).abs()
        # 4. 合成基础欺骗指数并融合尾盘偷袭
        base_deception_index = (bullish_deception_score - bearish_deception_score).clip(-1, 1)
        closing_ambush_score = get_adaptive_mtf_normalized_bipolar_score(closing_ambush_raw, df.index, default_weights)
        final_deception_index = (base_deception_index * 0.7 + closing_ambush_score * 0.3).clip(-1, 1)
        # --- 探针监测 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _diagnose_deception_index @ {probe_date_str}")
                print(f"        - 欺骗证据分: {deception_evidence_score.loc[probe_ts]:.4f} (基础欺骗={base_deception_raw.loc[probe_ts]:.2f}, 对倒={wash_trade_raw.loc[probe_ts]:.2f})")
                print(f"        - 主力流向分: {flow_direction_score.loc[probe_ts]:.4f} (原始流比率={flow_ratio.loc[probe_ts]:.4f})")
                # 更新探针输出，展示净化前后的值
                print(f"        - 看涨欺骗分: {bullish_deception_score.loc[probe_ts]:.4f} (收盘弱势={1 - normalized_strength.loc[probe_ts]:.2f}, 原始强弱={closing_strength_raw.loc[probe_ts]:.2f})")
                print(f"        - 看跌欺骗分: {bearish_deception_score.loc[probe_ts]:.4f} (收盘强势={normalized_strength.loc[probe_ts]:.2f})")
                print(f"        - 基础欺骗指数: {base_deception_index.loc[probe_ts]:.4f}")
                print(f"        - 尾盘偷袭分: {closing_ambush_score.loc[probe_ts]:.4f} (原始值={closing_ambush_raw.loc[probe_ts]:.2f})")
                print(f"        - 最终博弈欺骗指数: {final_deception_index.loc[probe_ts]:.4f}")
        return final_deception_index.astype(np.float32)

    def _get_calibrated_adaptive_mtf_score(self, series: pd.Series, tf_weights: Dict, tolerance: float = 1e-6) -> pd.Series:
        """
        【V4.0 · Index-Loc 终极修复版】校准后的自适应MTF归一化分数
        - 核心修复: 采用“先完整归一化，再根据索引.loc精准覆写”的策略，彻底根除“零值归一化悖论”。
        - 校准逻辑:
          1. 对完整的Series进行标准归一化，保证正确的历史上下文。
          2. 从原始Series中提取出值接近于零的索引。
          3. 使用.loc索引器，将归一化结果中对应索引位置的值强制覆写为0.0。
        """
        # 1. 对完整的、未经修改的Series进行归一化
        normalized_series = get_adaptive_mtf_normalized_score(
            series, series.index, ascending=True, tf_weights=tf_weights
        )
        # 2. 从原始Series中获取值接近于零的索引
        zero_indices = series.index[series.abs() < tolerance]
        # 3. 如果存在这样的索引，则使用.loc将归一化结果中对应位置的值强制覆写为0.0
        if not zero_indices.empty:
            normalized_series.loc[zero_indices] = 0.0
        return normalized_series

    def _get_calibrated_adaptive_mtf_bipolar_score(self, series: pd.Series, tf_weights: Dict, tolerance: float = 1e-6) -> pd.Series:
        """
        【V3.0 · Index-Loc 终极修复版】校准后的自适应MTF双极性归一化分数
        - 核心修复: 采用“先完整归一化，再根据索引.loc精准覆写”的策略，修复双极性信号的“零值归一化悖论”。
        """
        # 1. 对完整的Series进行双极性归一化
        normalized_series = get_adaptive_mtf_normalized_bipolar_score(
            series, series.index, tf_weights
        )
        # 2. 从原始Series中获取值接近于零的索引
        zero_indices = series.index[series.abs() < tolerance]
        # 3. 如果存在这样的索引，则使用.loc将归一化结果中对应位置的值强制覆写为0.0
        if not zero_indices.empty:
            normalized_series.loc[zero_indices] = 0.0
        return normalized_series



