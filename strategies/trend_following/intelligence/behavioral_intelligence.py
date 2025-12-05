# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional, List, Any
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
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
        【V34.5 · 攻防一体化适配版】原子信号中心
        - 核心升级: 适配了 V5.0 "派发罪证链" 和 V3.0 "战略反击许可" 模型，
                      并调整了内部调用顺序以确保逻辑依赖的正确性。
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
            'deception_index_D', 'wash_trade_intensity_D', 'closing_auction_ambush_D', 'mf_retail_battle_intensity_D',
            'main_force_conviction_index_D', 'SLOPE_5_loser_pain_index_D',
            'pressure_rejection_strength_D', 'active_buying_support_D', 'vwap_control_strength_D',
            'SLOPE_5_winner_stability_index_D'
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
        if 'ACCEL_5_pct_change_D' in df.columns:
            price_accel = self._get_safe_series(df, 'ACCEL_5_pct_change_D', 0.0, method_name="_diagnose_behavioral_axioms")
        else:
            print("    -> [行为情报兼容模式] _diagnose_behavioral_axioms: 未找到 'ACCEL_5_pct_change_D'，使用 'pct_change_D' 的5日差分作为代理。")
            price_accel = pct_change.diff(5).fillna(0.0)
        # --- 动能信号 ---
        upward_momentum_score = self._diagnose_upward_momentum(df, default_weights)
        states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = upward_momentum_score.astype(np.float32)
        downward_momentum_score = self._diagnose_downward_momentum(df, default_weights)
        states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = downward_momentum_score.astype(np.float32)
        # --- 超买信号 ---
        final_overextension_score = self._diagnose_price_overextension(df, default_weights, long_term_weights)
        states['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = final_overextension_score.astype(np.float32)
        # --- 行为铁三角 ---
        upward_efficiency_score = self._diagnose_upward_efficiency(df, default_weights)
        states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = upward_efficiency_score.astype(np.float32)
        downward_resistance_score = self._diagnose_downward_resistance(df, default_weights)
        states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = downward_resistance_score.astype(np.float32)
        intraday_bull_control_score = self._diagnose_intraday_bull_control(df, default_weights)
        states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = intraday_bull_control_score.astype(np.float32)
        stagnation_evidence = self._diagnose_stagnation_evidence(df, states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'])
        states['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'] = stagnation_evidence
        lower_shadow_quality = self._diagnose_lower_shadow_quality(df, stagnation_evidence)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_quality
        # [修改的代码行] 调整调用顺序，确保“派发意图”先于“进攻性承接”计算
        distribution_intent = self._diagnose_distribution_intent(df, default_weights)
        states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'] = distribution_intent
        offensive_absorption_intent = self._diagnose_offensive_absorption_intent(df, lower_shadow_quality, distribution_intent)
        states['SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT'] = offensive_absorption_intent
        states['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK'] = self._diagnose_ambush_counterattack(df, offensive_absorption_intent)
        states['SCORE_RISK_BREAKOUT_FAILURE_CASCADE'] = self._diagnose_breakout_failure_risk(df, distribution_intent)
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = self._calculate_volume_burst_quality(df, default_weights)
        states['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = self._calculate_volume_atrophy(df, default_weights)
        states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'] = self._calculate_absorption_strength(df, default_weights)
        states['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION'] = self._diagnose_shakeout_confirmation(
            df,
            states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'],
            states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT']
        )
        bullish_divergence_quality, bearish_divergence_quality = self._diagnose_divergence_quality(
            df,
            states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'],
            states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT']
        )
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
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_BURST'] * states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM']).pow(1/2).astype(np.float32)
        states['SCORE_BEHAVIOR_DECEPTION_INDEX'] = self._diagnose_deception_index(df)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                self._probe_raw_material_diagnostics(df, probe_ts)
        return states

    def _diagnose_upward_momentum(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】诊断高品质上涨动能。
        - 核心重构: 废弃了基于“表观强度幻觉”的 V1.0 模型。引入基于“闪电战三要素”
                      （攻击力度-战略指挥-后勤支撑）的全新品质诊断模型。
        - 闪电战三要素:
          1. 攻击力度 (Offensive Force): 审判攻击的纯净度与效率。采用 `upward_impulse_purity_D`
                                         和 `impulse_quality_ratio_D`。
          2. 战略指挥 (Strategic Command): 审判攻击背后的主力真实信念。采用 `main_force_conviction_index_D`。
          3. 后勤支撑 (Sustainability): 审判动能的可持续性，即内部获利盘的稳固程度。
                                        采用 `winner_stability_index_D`。
        - 数学模型: 动能分 = (攻击力度分 * 战略指挥分 * 后勤支撑分) ^ (1/3)
        """
        # --- 1. 获取三要素原始数据 ---
        impulse_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_upward_momentum")
        impulse_quality_raw = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name="_diagnose_upward_momentum")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_upward_momentum")
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_diagnose_upward_momentum")
        # --- 2. 计算各要素得分 ---
        # 要素一：攻击力度分
        purity_score = get_adaptive_mtf_normalized_score(impulse_purity_raw, df.index, ascending=True, tf_weights=tf_weights)
        quality_score = get_adaptive_mtf_normalized_score(impulse_quality_raw, df.index, ascending=True, tf_weights=tf_weights)
        offensive_force_score = (purity_score * quality_score).pow(0.5)
        # 要素二：战略指挥分
        strategic_command_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        # 要素三：后勤支撑分
        sustainability_score = get_adaptive_mtf_normalized_score(winner_stability_raw, df.index, ascending=True, tf_weights=tf_weights)
        # --- 3. “闪电战”三要素合成 ---
        upward_momentum_score = (
            (offensive_force_score + 1e-9) *
            (strategic_command_score + 1e-9) *
            (sustainability_score + 1e-9)
        ).pow(1/3).fillna(0.0)
        return upward_momentum_score.clip(0, 1).astype(np.float32)

    def _diagnose_downward_momentum(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】诊断高品质下跌动能。
        - 核心重构: 废弃了基于“恐慌幻觉”的 V1.0 模型。引入基于“斩首行动三要素”
                      （打击力度-战略意图-心理战果）的全新品质诊断模型。
        - 斩首行动三要素:
          1. 打击力度 (Overwhelming Force): 审判卖压的主动性与持续性。采用 `active_selling_pressure_D`
                                            和 `rally_distribution_pressure_D`。
          2. 战略意图 (Strategic Intent): 审判主力是否“佯退”还是“真撤”。采用 `main_force_conviction_index_D` 的负值。
          3. 心理战果 (Psychological Warfare): 审判多头阵营的士气崩溃程度。采用 `SLOPE_5_loser_pain_index_D`。
        - 数学模型: 动能分 = (打击力度分 * 战略意图分 * 心理战果分) ^ (1/3)
        """
        # --- 1. 获取三要素原始数据 ---
        active_selling_raw = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name="_diagnose_downward_momentum")
        distribution_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_downward_momentum")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_downward_momentum")
        loser_pain_slope_raw = self._get_safe_series(df, 'SLOPE_5_loser_pain_index_D', 0.0, method_name="_diagnose_downward_momentum")
        # --- 2. 计算各要素得分 ---
        # 要素一：打击力度分
        active_selling_score = get_adaptive_mtf_normalized_score(active_selling_raw, df.index, ascending=True, tf_weights=tf_weights)
        distribution_pressure_score = get_adaptive_mtf_normalized_score(distribution_pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        overwhelming_force_score = (active_selling_score * distribution_pressure_score).pow(0.5)
        # 要素二：战略意图分 (只考虑主力负向信念)
        strategic_intent_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=tf_weights)
        # 要素三：心理战果分 (套牢盘痛苦加剧)
        psychological_warfare_score = get_adaptive_mtf_normalized_score(loser_pain_slope_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        # --- 3. “斩首行动”三要素合成 ---
        downward_momentum_score = (
            (overwhelming_force_score + 1e-9) *
            (strategic_intent_score + 1e-9) *
            (psychological_warfare_score + 1e-9)
        ).pow(1/3).fillna(0.0)
        return downward_momentum_score.clip(0, 1).astype(np.float32)

    def _diagnose_price_overextension(self, df: pd.DataFrame, tf_weights: Dict, long_term_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】诊断价格过热风险。
        - 核心重构: 废弃了基于“静态热度谬误”和“粗暴音量陷阱”的 V1.0 模型。引入基于
                      “泡沫脆弱度”思想的全新对抗性诊断模型。
        - 核心博弈: 脆弱度 = 内部压力 (市场狂热) / 结构完整性 (主力信念与控制力)
          1. 内部压力 (Internal Pressure): 审判市场狂热的加速度。由 `total_winner_rate_D`,
                                           `ACCEL_5_pct_change_D`, `turnover_rate_f_D` 构成。
          2. 结构完整性 (Structural Integrity): 审判泡沫壁的坚固度。由 `winner_stability_index_D`,
                                                `control_solidity_index_D`, `main_force_conviction_index_D` 构成。
        - 数学模型: 脆弱度分 = 内部压力分 / (结构完整性分 + ε)，并废弃成交量放大器。
        """
        # --- 1. 获取两大维度原始数据 ---
        # 内部压力维度
        winner_rate_raw = self._get_safe_series(df, 'total_winner_rate_D', 50.0, method_name="_diagnose_price_overextension")
        price_accel_raw = self._get_safe_series(df, 'ACCEL_5_pct_change_D', 0.0, method_name="_diagnose_price_overextension")
        turnover_raw = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name="_diagnose_price_overextension")
        # 结构完整性维度
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_diagnose_price_overextension")
        control_solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.5, method_name="_diagnose_price_overextension")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_price_overextension")
        # --- 2. 计算各维度得分 ---
        # 维度一：内部压力分 (市场狂热)
        winner_rate_score = get_adaptive_mtf_normalized_score(winner_rate_raw, df.index, ascending=True, tf_weights=tf_weights)
        price_accel_score = get_adaptive_mtf_normalized_score(price_accel_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        turnover_score = get_adaptive_mtf_normalized_score(turnover_raw, df.index, ascending=True, tf_weights=tf_weights)
        internal_pressure_score = (winner_rate_score * price_accel_score * turnover_score).pow(1/3)
        # 维度二：结构完整性分 (主力信念与控制力)
        winner_stability_score = get_adaptive_mtf_normalized_score(winner_stability_raw, df.index, ascending=True, tf_weights=long_term_weights)
        control_solidity_score = get_adaptive_mtf_normalized_score(control_solidity_raw, df.index, ascending=True, tf_weights=long_term_weights)
        conviction_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=long_term_weights)
        structural_integrity_score = (winner_stability_score * control_solidity_score * conviction_score).pow(1/3)
        # --- 3. “泡沫脆弱度”合成 ---
        bubble_fragility_score = (internal_pressure_score / (structural_integrity_score + 1e-9)).fillna(0.0)
        # 对结果进行非线性放大和归一化，使得中低风险区差异不大，高风险区被显著放大
        final_overextension_score = np.tanh(bubble_fragility_score * 0.5)
        return final_overextension_score.clip(0, 1).astype(np.float32)

    def _diagnose_upward_efficiency(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】诊断高品质上涨效率。
        - 核心重构: 废弃了基于宽泛概念的 V1.0 模型。引入基于“闪击战三要素”
                      （突破纯净度-进攻性价比-卖压压制力）的全新诊断模型。
        - 闪击战三要素:
          1. 突破纯净度 (Breach Purity): 审判向上攻击的流畅性与直接性。采用 `upward_impulse_purity_D`。
          2. 进攻性价比 (Offensive Efficiency): 审判攻击的能量转换效率。采用 `impulse_quality_ratio_D`。
          3. 卖压压制力 (Pressure Suppression): 审判巩固战果、压制敌方反扑的能力。采用 `pressure_rejection_strength_D`。
        - 数学模型: 效率分 = (纯净度分^0.4 * 性价比分^0.3 * 压制力分^0.3)
        """
        # --- 1. 获取三要素原始数据 ---
        purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_upward_efficiency")
        offensive_efficiency_raw = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name="_diagnose_upward_efficiency")
        suppression_raw = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name="_diagnose_upward_efficiency")
        # --- 2. 计算各要素得分 ---
        purity_score = get_adaptive_mtf_normalized_score(purity_raw, df.index, ascending=True, tf_weights=tf_weights)
        offensive_efficiency_score = get_adaptive_mtf_normalized_score(offensive_efficiency_raw, df.index, ascending=True, tf_weights=tf_weights)
        suppression_score = get_adaptive_mtf_normalized_score(suppression_raw, df.index, ascending=True, tf_weights=tf_weights)
        # --- 3. “闪击战”三要素合成 ---
        upward_efficiency_score = (
            (purity_score + 1e-9).pow(0.4) *
            (offensive_efficiency_score + 1e-9).pow(0.3) *
            (suppression_score + 1e-9).pow(0.3)
        ).fillna(0.0)
        return upward_efficiency_score.clip(0, 1).astype(np.float32)

    def _diagnose_downward_resistance(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】诊断高品质下跌抵抗。
        - 核心重构: 废弃了基于“被动挨打”视角的 V1.0 模型。引入基于“纵深防御三要素”
                      （被动承接-主动防御-积极反击）的全新诊断模型。
        - 纵深防御三要素:
          1. 被动承接 (Passive Absorption): 衡量市场对下跌的初步、自然消化能力。采用 `dip_absorption_power_D`。
          2. 主动防御 (Active Defense): 审判在关键支撑位置的主动防守强度。采用 `support_validation_strength_D`。
          3. 积极反击 (Proactive Counterattack): 审判多头主动出击、夺回失地的能力。采用 `active_buying_support_D`。
        - 数学模型: 抵抗分 = (被动分^0.2 * 主动分^0.4 * 反击分^0.4)
        """
        # --- 1. 获取三要素原始数据 ---
        passive_absorption_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_downward_resistance")
        active_defense_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name="_diagnose_downward_resistance")
        counter_attack_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_diagnose_downward_resistance")
        # --- 2. 计算各要素得分 ---
        passive_absorption_score = get_adaptive_mtf_normalized_score(passive_absorption_raw, df.index, ascending=True, tf_weights=tf_weights)
        active_defense_score = get_adaptive_mtf_normalized_score(active_defense_raw, df.index, ascending=True, tf_weights=tf_weights)
        counter_attack_score = get_adaptive_mtf_normalized_score(counter_attack_raw, df.index, ascending=True, tf_weights=tf_weights)
        # --- 3. “纵深防御”三要素合成 ---
        downward_resistance_score = (
            (passive_absorption_score + 1e-9).pow(0.2) *
            (active_defense_score + 1e-9).pow(0.4) *
            (counter_attack_score + 1e-9).pow(0.4)
        ).fillna(0.0)
        return downward_resistance_score.clip(0, 1).astype(np.float32)

    def _diagnose_intraday_bull_control(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】诊断高品质日内多头控制力。
        - 核心重构: 废弃了缺乏“灵魂”的 V1.0 模型。引入基于“战区司令部三要素”
                      （战略位置-战术火力-司令意志）的全新诊断模型。
        - 战区司令部三要素:
          1. 战略位置 (Strategic Position): 审判对日内核心战场(VWAP)的控制权。采用 `vwap_control_strength_D`。
          2. 战术火力 (Tactical Firepower): 审判发动有效攻击的能力。采用 `impulse_quality_ratio_D`。
          3. 司令意志 (Commander's Will): 审判所有战术行动背后的主力真实信念。采用 `main_force_conviction_index_D`。
        - 数学模型: 控制力分 = (位置分^0.3 * 火力分^0.3 * 意志分^0.4)
        """
        # --- 1. 获取三要素原始数据 ---
        strategic_position_raw = self._get_safe_series(df, 'vwap_control_strength_D', 0.5, method_name="_diagnose_intraday_bull_control")
        tactical_firepower_raw = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name="_diagnose_intraday_bull_control")
        commanders_will_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_intraday_bull_control")
        # --- 2. 计算各要素得分 ---
        strategic_position_score = get_adaptive_mtf_normalized_score(strategic_position_raw, df.index, ascending=True, tf_weights=tf_weights)
        tactical_firepower_score = get_adaptive_mtf_normalized_score(tactical_firepower_raw, df.index, ascending=True, tf_weights=tf_weights)
        commanders_will_score = get_adaptive_mtf_normalized_score(commanders_will_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        # --- 3. “战区司令部”三要素合成 ---
        intraday_bull_control_score = (
            (strategic_position_score + 1e-9).pow(0.3) *
            (tactical_firepower_score + 1e-9).pow(0.3) *
            (commanders_will_score + 1e-9).pow(0.4)
        ).fillna(0.0)
        return intraday_bull_control_score.clip(0, 1).astype(np.float32)

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
        【V4.1 · 生产版】诊断内部行为信号：滞涨证据
        - 核心重构: 废弃了基于“战术僵化”的 V3.9 模型。引入基于“信念危机”思想的全新
                      双维度诊断模型，旨在区分“良性蓄势”与“恶性派发”的滞涨。
        - 信念危机双维度:
          1. 微观战局僵持 (Micro-Battlefield Stalemate): 审判前线战况的胶着程度。
          2. 宏观信念动摇 (Macro-Conviction Erosion): 审判主力司令部的真实意图与筹码结构的稳定性。
        """
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        p_thresholds = get_param_value(p_conf.get('neutral_zone_thresholds'), {})
        alpha_threshold = get_param_value(p_thresholds.get('main_force_execution_alpha_D'), 0.0)
        # --- 1. 获取原始数据 ---
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_stagnation_evidence")
        price_accel = self._get_safe_series(df, 'ACCEL_5_pct_change_D', 0.0, method_name="_diagnose_stagnation_evidence")
        chip_fatigue_raw = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name="_diagnose_stagnation_evidence")
        rally_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_stagnation_evidence").clip(lower=0)
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_stagnation_evidence")
        mf_alpha_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_stagnation_evidence")
        winner_rate_raw = self._get_safe_series(df, 'total_winner_rate_D', 50.0, method_name="_diagnose_stagnation_evidence")
        conviction_slope_raw = self._get_safe_series(df, 'SLOPE_5_main_force_conviction_index_D', 0.0, method_name="_diagnose_stagnation_evidence")
        winner_stability_slope_raw = self._get_safe_series(df, 'SLOPE_5_winner_stability_index_D', 0.0, method_name="_diagnose_stagnation_evidence")
        # --- 2. 维度一：微观战局僵持 (Micro-Battlefield Stalemate) ---
        inefficiency_score = (1 - upward_efficiency).clip(0, 1)
        momentum_decay_score = get_adaptive_mtf_normalized_score(price_accel.clip(upper=0).abs(), df_index, ascending=True, tf_weights=default_weights)
        chip_fatigue_score = get_adaptive_mtf_normalized_score(chip_fatigue_raw, df_index, ascending=True, tf_weights=default_weights)
        bullish_exhaustion_score = (inefficiency_score * momentum_decay_score * chip_fatigue_score).pow(1/3).fillna(0.0)
        rally_pressure_score = get_adaptive_mtf_normalized_score(rally_pressure_raw, df_index, ascending=True, tf_weights=default_weights)
        upper_shadow_score = get_adaptive_mtf_normalized_score(upper_shadow_pressure_raw, df_index, ascending=True, tf_weights=default_weights)
        mf_alpha_filtered = self._apply_neutral_zone_filter(mf_alpha_raw, alpha_threshold)
        mf_distribution_evidence = get_adaptive_mtf_normalized_score(mf_alpha_filtered.clip(upper=0).abs(), df_index, ascending=True, tf_weights=default_weights)
        bearish_ambush_score = (rally_pressure_score * upper_shadow_score * mf_distribution_evidence).pow(1/3).fillna(0.0)
        total_energy = (bullish_exhaustion_score + bearish_ambush_score) / 2
        balance_factor = 1 - (bullish_exhaustion_score - bearish_ambush_score).abs()
        micro_stalemate_score = (total_energy * balance_factor).fillna(0.0)
        # --- 3. 维度二：宏观信念动摇 (Macro-Conviction Erosion) ---
        profit_pressure_score = get_adaptive_mtf_normalized_score(winner_rate_raw, df_index, ascending=True, tf_weights=default_weights)
        conviction_decay_score = get_adaptive_mtf_normalized_score(conviction_slope_raw.clip(upper=0).abs(), df_index, ascending=True, tf_weights=default_weights)
        instability_score = get_adaptive_mtf_normalized_score(winner_stability_slope_raw.clip(upper=0).abs(), df_index, ascending=True, tf_weights=default_weights)
        macro_erosion_score = (profit_pressure_score * conviction_decay_score * instability_score).pow(1/3).fillna(0.0)
        # --- 4. 最终合成 ---
        stagnation_evidence = (micro_stalemate_score * 0.6 + macro_erosion_score * 0.4)
        is_rising_or_flat = (pct_change >= -0.005).astype(float)
        final_stagnation_evidence = (stagnation_evidence * is_rising_or_flat).clip(0, 1)
        # [修改的代码行] 移除探针代码，恢复生产版本
        return final_stagnation_evidence.astype(np.float32)

    def _diagnose_lower_shadow_quality(self, df: pd.DataFrame, stagnation_evidence: pd.Series) -> pd.Series:
        """
        【V12.0 · 伏击战役版】诊断下影线承接品质。
        - 核心重构: 废弃了基于“法医式”K线形态拼凑的 V11.0 模型。引入基于“伏击战役”
                      思想的全新三位一体诊断模型，旨在审判整个“诱空-反击”的战役全过程。
        - 伏击战役三要素:
          1. 战役背景 (The Lure): 审判是否存在真实的恐慌抛盘，为伏击提供前提。采用 `panic_selling_cascade_D`。
          2. 核心行动 (The Ambush): 审判主力在下跌中是否发动了主动、持续的反击。采用 `active_buying_support_D`。
          3. 战役结果 (The Victory): 审判反击是否成功收复失地，巩固战果。采用 `closing_strength_index_D`。
        - 数学模型: 品质分 = (背景分^0.3 * 行动分^0.4 * 结果分^0.3) * 滞涨压制器
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 1. 维度一：获取伏击战役三要素的原始数据 ---
        context_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        action_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        outcome_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_diagnose_lower_shadow_quality")
        # --- 2. 维度一：计算伏击战役基础品质分 ---
        context_score = get_adaptive_mtf_normalized_score(context_raw, df.index, ascending=True, tf_weights=default_weights)
        action_score = get_adaptive_mtf_normalized_score(action_raw, df.index, ascending=True, tf_weights=default_weights)
        outcome_score = normalize_score(outcome_raw, df.index, 55)
        # [修改的代码行] 采用加权几何平均，融合“伏击战役”三要素
        base_quality_score = (
            (context_score + 1e-9).pow(0.3) *
            (action_score + 1e-9).pow(0.4) *
            (outcome_score + 1e-9).pow(0.3)
        ).fillna(0.0)
        # --- 3. 维度二：构建战略环境压制器 ---
        stagnation_suppressor = (1 - stagnation_evidence).clip(0, 1)
        # --- 4. 最终合成 ---
        final_lower_shadow_quality = (base_quality_score * stagnation_suppressor).clip(0, 1)
        # --- 深度战术探针 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _diagnose_lower_shadow_quality @ {probe_date_str}")
                print(f"        --- [维度一: 伏击战役三要素] ---")
                print(f"          - 原始值: 战役背景(恐慌级联)={context_raw.loc[probe_ts]:.2f}, 核心行动(主动买盘)={action_raw.loc[probe_ts]:.2f}, 战役结果(收盘强度)={outcome_raw.loc[probe_ts]:.2f}")
                print(f"          - 要素得分: 背景分={context_score.loc[probe_ts]:.4f}, 行动分={action_score.loc[probe_ts]:.4f}, 结果分={outcome_score.loc[probe_ts]:.4f}")
                print(f"          - 基础品质分(几何平均): {base_quality_score.loc[probe_ts]:.4f}")
                print(f"        --- [维度二: 战略环境压制器] ---")
                print(f"          - 原始滞涨证据分: {stagnation_evidence.loc[probe_ts]:.4f}")
                print(f"          - 滞涨压制器乘数: {stagnation_suppressor.loc[probe_ts]:.4f}")
                print(f"        --- [最终合成] ---")
                print(f"        - 最终下影线品质分 (压制后): {final_lower_shadow_quality.loc[probe_ts]:.4f}")
        return final_lower_shadow_quality.astype(np.float32)

    def _diagnose_distribution_intent(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V5.0 · 派发罪证链版】诊断派发意图。
        - 核心重构: 废弃了基于单一卖压结果的 V4.0 模型。引入基于“动机-凶器-指纹”
                      三位一体的“罪证链”诊断模型，旨在审判一次完整的派发行为。
        - 派发罪证链三要素:
          1. 动机 (Motive): 审判市场是否存在强烈的获利了结动机。采用 `profit_taking_flow_ratio_D`。
          2. 凶器 (Weapon): 审判是否存在反弹受阻或冲高回落的卖压行为。融合 `rally_distribution_pressure_D` 和 `upper_shadow_selling_pressure_D`。
          3. 指纹 (Fingerprint): 审判主力是否留下了“言行不一”的隐蔽派发痕迹。采用 `main_force_execution_alpha_D` 的负值部分。
        - 数学模型: 派发分 = (动机分^0.2 * 凶器分^0.4 * 指纹分^0.4)
        """
        # --- 1. 获取“罪证链”三要素的原始数据 ---
        motive_raw = self._get_safe_series(df, 'profit_taking_flow_ratio_D', 0.0, method_name="_diagnose_distribution_intent")
        rally_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_distribution_intent")
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_distribution_intent")
        fingerprint_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_distribution_intent")
        # --- 2. 计算各要素得分 ---
        motive_score = get_adaptive_mtf_normalized_score(motive_raw, df.index, ascending=True, tf_weights=tf_weights)
        # [修改的代码行] 将两种卖压行为融合成“凶器”得分
        rally_pressure_score = get_adaptive_mtf_normalized_score(rally_pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        upper_shadow_score = get_adaptive_mtf_normalized_score(upper_shadow_pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        weapon_score = (rally_pressure_score * 0.5 + upper_shadow_score * 0.5)
        # [修改的代码行] 提取主力隐蔽派发的“指纹”得分
        fingerprint_score = get_adaptive_mtf_normalized_score(fingerprint_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=tf_weights)
        # --- 3. “罪证链”三要素合成 ---
        distribution_intent_score = (
            (motive_score + 1e-9).pow(0.2) *
            (weapon_score + 1e-9).pow(0.4) *
            (fingerprint_score + 1e-9).pow(0.4)
        ).fillna(0.0)
        # --- 深度战术探针 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _diagnose_distribution_intent @ {probe_date_str}")
                print(f"        --- [派发罪证链三要素] ---")
                print(f"          - 原始值: 动机(获利盘流比)={motive_raw.loc[probe_ts]:.2f}, 凶器(反弹派压)={rally_pressure_raw.loc[probe_ts]:.2f}, 凶器(上影线)={upper_shadow_pressure_raw.loc[probe_ts]:.2f}, 指纹(主力Alpha)={fingerprint_raw.loc[probe_ts]:.4f}")
                print(f"          - 要素得分: 动机分={motive_score.loc[probe_ts]:.4f}, 凶器分={weapon_score.loc[probe_ts]:.4f}, 指纹分={fingerprint_score.loc[probe_ts]:.4f}")
                print(f"          - 最终派发意图分 (罪证链): {distribution_intent_score.loc[probe_ts]:.4f}")
        return distribution_intent_score.clip(0, 1).astype(np.float32)

    def _diagnose_ambush_counterattack(self, df: pd.DataFrame, offensive_absorption_intent: pd.Series) -> pd.Series:
        """
        【V2.1 · 生产版】诊断伏击式反攻信号
        - 核心重构: 废弃旧的、基于底层特征拼凑的 V1.0 模型。引入基于“因果传导”哲学的
                      分层门控模型，从根本上解决了旧模型的“战术意图错配”和“因果倒置”缺陷。
        - 战术三要素:
          1. 伏击过程 (核心基石): 直接采用高阶战术信号 SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT，
                                  代表对整个伏击行为的最终诊断。
          2. 反击结果 (品质验证): 采用 closing_strength_index_D，衡量反攻的最终战果。
          3. 战场环境 (价值放大): 采用 panic_selling_cascade_D，量化恐慌环境，放大在危急时刻
                                  完成的伏击反攻的战术价值。
        - 数学模型: 伏击反攻分 = 伏击过程分 * (反击结果分 * 战场环境分) ^ 0.5
        """
        required_signals = ['closing_strength_index_D', 'panic_selling_cascade_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_ambush_counterattack"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 1. 获取三大战术要素 ---
        # 核心基石：伏击过程分 (直接使用传入的高阶意图信号)
        ambush_process_score = offensive_absorption_intent
        # 品质调节器元素1：反击结果分
        counterattack_result_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_diagnose_ambush_counterattack")
        counterattack_result_score = normalize_score(counterattack_result_raw, df.index, 55)
        # 品质调节器元素2：战场环境分
        battlefield_environment_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_ambush_counterattack")
        battlefield_environment_score = get_adaptive_mtf_normalized_score(battlefield_environment_raw, df.index, ascending=True, tf_weights=default_weights)
        # --- 2. 构建品质调节器 ---
        quality_modulator = (counterattack_result_score * battlefield_environment_score).pow(0.5).fillna(0.0)
        # --- 3. 分层门控合成 ---
        ambush_counterattack_score = (ambush_process_score * quality_modulator).clip(0, 1)
        return ambush_counterattack_score.astype(np.float32)

    def _diagnose_breakout_failure_risk(self, df: pd.DataFrame, distribution_intent: pd.Series) -> pd.Series:
        """
        【V2.1 · 生产版】诊断突破失败级联风险
        - 核心重构: 废弃了基于简单价格比较的“机械式突破谬误”模型。引入基于“诱多-伏击-套牢”
                      诡道剧本的全新三维诊断模型，旨在精确识别高迷惑性的“牛市陷阱”。
        - 战术三要素:
          1. 诱饵 (The Lure): 使用更高阶的 `breakout_quality_score_D` 替代简单的价格突破判断，
                               量化突破行为的“迷惑性”。一次看似完美的突破，其陷阱价值才最大。
          2. 伏击 (The Ambush): 继续使用强大的 `distribution_intent` 信号，量化主力在诱多
                                过程中的真实派发意图，即“收割”的坚决性。
          3. 套牢盘 (The Trapped Force): 继续使用 `volume_ratio_D`，量化在陷阱高位被套牢的
                                         资金规模，即未来级联崩塌的“燃料”。
        - 数学模型: 风险分 = 诱饵分 * (伏击分 ^ 0.6 * 套牢盘分 ^ 0.4)
        """
        required_signals = ['breakout_quality_score_D', 'volume_ratio_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_breakout_failure_risk"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 1. 获取三大战术要素 ---
        # 战术要素一：诱饵 (突破迷惑性)
        breakout_quality_raw = self._get_safe_series(df, 'breakout_quality_score_D', 0.0, method_name="_diagnose_breakout_failure_risk")
        lure_score = get_adaptive_mtf_normalized_score(breakout_quality_raw, df.index, ascending=True, tf_weights=default_weights)
        # 战术要素二：伏击 (派发坚决性)
        ambush_score = distribution_intent # 直接使用传入的、强大的派发意图分数
        # 战术要素三：套牢盘 (潜在抛压规模)
        volume_raw = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_diagnose_breakout_failure_risk")
        trapped_force_score = get_adaptive_mtf_normalized_score(volume_raw, df.index, ascending=True, tf_weights=default_weights)
        # --- 2. 风险合成 ---
        internal_risk_factor = (ambush_score.pow(0.6) * trapped_force_score.pow(0.4)).fillna(0.0)
        breakout_failure_risk = (lure_score * internal_risk_factor).clip(0, 1)
        return breakout_failure_risk.astype(np.float32)

    def _diagnose_divergence_quality(self, df: pd.DataFrame, absorption_strength: pd.Series, distribution_intent: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        【V3.2 · 生产版】诊断高品质价量/价资背离
        - 核心修复: 废弃了隐式的状态依赖，重构为显式的参数注入。方法现在直接接收
                      absorption_strength 和 distribution_intent 作为参数，彻底解决了
                      因调用时序不当导致的依赖信号获取失败问题。
        - 诊断三要素:
          1. 背离幅度 (Magnitude): 价格创出新高/低，但 `main_force_conviction_index_D` 逆势而行。
          2. 战场位置 (Location): 牛市背离发生在套牢盘极度痛苦 (`loser_pain_index_D`) 之时；
                                  熊市背离发生在获利盘极不稳固 (`winner_stability_index_D`) 之际。
          3. 确认信号 (Confirmation): 牛市背离需由 `absorption_strength` 确认；熊市背离需由
                                      `distribution_intent` 确认。
        - 数学模型: 品质分 = (幅度分^0.5 * 位置分^0.3 * 确认分^0.2)
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        divergence_window = 5
        # --- 1. 获取三维度原始数据 ---
        price = self._get_safe_series(df, 'close_D', 0.0, method_name="_diagnose_divergence_quality")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_divergence_quality")
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_diagnose_divergence_quality")
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_diagnose_divergence_quality")
        # --- 2. 计算牛市背离 (价格新低 vs 信念走高) ---
        price_new_low = (price == price.rolling(divergence_window, min_periods=1).min()).astype(float)
        conviction_trend_up = (conviction_raw > conviction_raw.rolling(divergence_window, min_periods=1).mean()).astype(float)
        bullish_magnitude_score = (price_new_low * conviction_trend_up).fillna(0.0)
        bullish_location_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df.index, ascending=True, tf_weights=default_weights)
        bullish_confirmation_score = absorption_strength
        bullish_divergence_quality = (
            (bullish_magnitude_score + 1e-9).pow(0.5) *
            (bullish_location_score + 1e-9).pow(0.3) *
            (bullish_confirmation_score + 1e-9).pow(0.2)
        ).fillna(0.0)
        # --- 3. 计算熊市背离 (价格新高 vs 信念走低) ---
        price_new_high = (price == price.rolling(divergence_window, min_periods=1).max()).astype(float)
        conviction_trend_down = (conviction_raw < conviction_raw.rolling(divergence_window, min_periods=1).mean()).astype(float)
        bearish_magnitude_score = (price_new_high * conviction_trend_down).fillna(0.0)
        winner_instability_raw = 1 - winner_stability_raw # 获利盘不稳定性
        bearish_location_score = get_adaptive_mtf_normalized_score(winner_instability_raw, df.index, ascending=True, tf_weights=default_weights)
        bearish_confirmation_score = distribution_intent
        bearish_divergence_quality = (
            (bearish_magnitude_score + 1e-9).pow(0.5) *
            (bearish_location_score + 1e-9).pow(0.3) *
            (bearish_confirmation_score + 1e-9).pow(0.2)
        ).fillna(0.0)
        return bullish_divergence_quality.clip(0, 1).astype(np.float32), bearish_divergence_quality.clip(0, 1).astype(np.float32)

    def _calculate_volume_burst_quality(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】计算高品质看涨量能爆发信号。
        - 核心重构: 废弃了基于“净值结果谬误”的 V1.4 模型。引入基于“信念-效率-战果”
                      军事突击思想的全新四维诊断模型，旨在穿透“冲高派发”等诡道迷雾。
        - 战术四要素:
          1. 幅度 (Magnitude): 保留 `volume_ratio_D`，衡量兵力投入规模。
          2. 信念 (Conviction): 废弃简单的 flow_ratio，采用 `main_force_conviction_index_D`，
                                衡量主力真实的、不可动摇的进攻决心。
          3. 效率 (Efficiency): 升级为 `impulse_quality_ratio_D`，衡量战术执行的凌厉程度。
          4. 战果 (Result): 新增 `closing_strength_index_D`，作为最终审判官，衡量多头是否
                             成功巩固了胜利果实，严惩“冲高回落”式的失败进攻。
        - 数学模型: 品质分 = (幅度分 * 信念分 * 效率分 * 战果分) ^ (1/4)
        """
        # --- 1. 获取四维度原始数据 ---
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_calculate_volume_burst_quality")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_volume_burst_quality")
        efficiency_raw = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name="_calculate_volume_burst_quality")
        result_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_calculate_volume_burst_quality")
        # --- 2. 计算各维度得分 ---
        # 维度一：幅度分
        magnitude_score = get_adaptive_mtf_normalized_score(volume_ratio, df.index, ascending=True, tf_weights=tf_weights)
        # 维度二：信念分
        conviction_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        # 维度三：效率分
        efficiency_score = get_adaptive_mtf_normalized_score(efficiency_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 维度四：战果分
        result_score = normalize_score(result_raw, df.index, 55)
        # --- 3. 四维战术合成 ---
        volume_burst_quality = (
            (magnitude_score + 1e-9) *
            (conviction_score + 1e-9) *
            (efficiency_score + 1e-9) *
            (result_score + 1e-9)
        ).pow(1/4).fillna(0.0)
        return volume_burst_quality.clip(0, 1).astype(np.float32)

    def _calculate_volume_atrophy(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】计算高品质成交量萎缩信号。
        - 核心重构: 废弃了基于“静态快照谬误”的 V1.2 模型。引入基于“环境决定论”的
                      全新二元结构模型，旨在精确区分“良性惜售”与“恶性阴跌”。
        - 核心二元结构:
          1. 萎缩程度 (Atrophy Degree): 继续使用 `volume_ratio_D` 作为基础，衡量缩量状态。
          2. 高质量环境调节器 (Quality Modulator): 引入三位一体的环境审判机制，对基础分进行调制。
             - 获利盘锁定度: 使用 `winner_stability_index_D` 衡量赢家惜售意愿。
             - 套牢盘枯竭度: 使用 `loser_pain_index_D` 衡量套牢盘是否被洗净。
             - 浮筹清洗度: 使用 `floating_chip_cleansing_efficiency_D` 评估洗盘效率。
        - 数学模型: 品质分 = 萎缩程度分 * (锁定分 * 枯竭分 * 清洗分) ^ (1/3)
        """
        # --- 1. 获取原始数据 ---
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_calculate_volume_atrophy")
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_calculate_volume_atrophy")
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_calculate_volume_atrophy")
        cleansing_efficiency_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_calculate_volume_atrophy")
        # --- 2. 计算各维度得分 ---
        # 维度一：萎缩程度分 (与量比负相关)
        atrophy_degree_score = 1 - get_adaptive_mtf_normalized_score(volume_ratio, df.index, ascending=True, tf_weights=tf_weights)
        # 构建高质量环境调节器
        lockup_score = get_adaptive_mtf_normalized_score(winner_stability_raw, df.index, ascending=True, tf_weights=tf_weights)
        exhaustion_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df.index, ascending=True, tf_weights=tf_weights)
        cleansing_score = get_adaptive_mtf_normalized_score(cleansing_efficiency_raw, df.index, ascending=True, tf_weights=tf_weights)
        quality_modulator = (
            (lockup_score + 1e-9) *
            (exhaustion_score + 1e-9) *
            (cleansing_score + 1e-9)
        ).pow(1/3).fillna(0.0)
        # --- 3. 最终品质合成 ---
        volume_atrophy_quality = (atrophy_degree_score * quality_modulator).clip(0, 1)
        return volume_atrophy_quality.astype(np.float32)

    def _calculate_absorption_strength(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 生产版】计算高品质承接强度信号。
        - 核心重构: 废弃了基于“下影线幻觉”的 V1.3 模型。引入基于“压力-意图-结果”
                      战役分析框架的全新三维诊断模型，旨在精确区分“强力承接”与“诱空回补”。
        - 战术三要素:
          1. 战场环境 (Battlefield Context): 引入 `panic_selling_cascade_D`，衡量承接发生时
                                             的抛压严重性。压力越大，承接价值越高。
          2. 承接意图 (Absorption Intent): 采用 `main_force_conviction_index_D`，审判承接
                                           行为是否由信念坚定的主力发起。
          3. 承接结果 (Absorption Result): 使用下影线长度作为战果体现，只有在环境和意图
                                           被证实后，形态才有意义。
        - 数学模型: 强度分 = (环境分 * 意图分 * 结果分) ^ (1/3)
        """
        # --- 1. 获取三维度原始数据 ---
        panic_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_calculate_absorption_strength")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_absorption_strength")
        # 计算承接形态的原始值 (下影线占比)
        absorption_form_raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        absorption_form_raw = absorption_form_raw.clip(0, 1)
        # --- 2. 计算各维度得分 ---
        # 维度一：战场环境分
        battlefield_context_score = get_adaptive_mtf_normalized_score(panic_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 维度二：承接意图分
        absorption_intent_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        # 维度三：承接结果分
        absorption_result_score = normalize_score(absorption_form_raw, df.index, 55)
        # --- 3. 三维战术合成 ---
        absorption_strength = (
            (battlefield_context_score + 1e-9) *
            (absorption_intent_score + 1e-9) *
            (absorption_result_score + 1e-9)
        ).pow(1/3).fillna(0.0)
        return absorption_strength.clip(0, 1).astype(np.float32)

    def _diagnose_shakeout_confirmation(self, df: pd.DataFrame, absorption_strength: pd.Series, distribution_intent: pd.Series) -> pd.Series:
        """
        【V2.1 · 生产版】诊断震荡洗盘确认信号。
        - 核心重构: 废弃了基于“过程融合谬误”的 V1.3 模型。引入基于“政变三部曲”
                      （前提-行动-成果）的全新门控模型，旨在精确识别主力主动控盘的洗盘行为。
        - 政变三部曲:
          1. 前提门控 (Precondition): 审查动机与环境。必须满足“无派发意图”和“非失控恐慌”
                                      两大前提，否则“政变”无从谈起。
          2. 核心引擎 (Core Action): 废弃 `downward_resistance`，直接采用更高阶的
                                     `absorption_strength` 作为唯一核心。一场没有强力反攻
                                     的洗盘，就是一次失败的政变。
          3. 战术成果 (Result): 采用 `floating_chip_cleansing_efficiency_D` 作为最终战果
                                  的验证，确认“政变”是否成功清洗了浮筹。
        - 数学模型: 确认分 = 前提门控分 * (核心引擎分 * 战术成果分) ^ 0.5
        """
        # --- 1. 获取原始数据 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        panic_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_shakeout_confirmation")
        cleansing_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_shakeout_confirmation")
        # --- 2. 计算三部曲的各个组件得分 ---
        # 组件一：前提门控分
        no_distribution_intent_score = (1 - distribution_intent).clip(0, 1)
        panic_score = get_adaptive_mtf_normalized_score(panic_raw, df.index, ascending=True, tf_weights=default_weights)
        controllable_environment_score = (1 - panic_score).clip(0, 1)
        precondition_gate_score = (no_distribution_intent_score * controllable_environment_score).pow(0.5)
        # 组件二：核心引擎分 (决定性反击)
        core_action_score = absorption_strength
        # 组件三：战术成果分 (清算效率)
        tactical_result_score = get_adaptive_mtf_normalized_score(cleansing_raw, df.index, ascending=True, tf_weights=default_weights)
        # --- 3. “政变”三部曲合成 ---
        internal_confirmation = (core_action_score * tactical_result_score).pow(0.5).fillna(0.0)
        shakeout_confirmation_score = (precondition_gate_score * internal_confirmation).clip(0, 1)
        return shakeout_confirmation_score.astype(np.float32)

    def _diagnose_offensive_absorption_intent(self, df: pd.DataFrame, lower_shadow_quality: pd.Series, distribution_intent: pd.Series) -> pd.Series:
        """
        【V3.0 · 战略反击许可版】诊断进攻性承接意图。
        - 核心重构: 废弃了孤立看待下影线的 V2.0 模型。引入“战术胜利+战略许可+司令部意志”
                      三层诊断框架，确保战术行为的战略有效性。
        - 三层诊断框架:
          1. 战术胜利 (Tactical Victory): 必须存在一次高质量的下影线承接。采用 `lower_shadow_quality`。
          2. 战略许可 (Strategic Clearance): 必须在主力未进行战略派发的背景下。使用 `distribution_intent` 作为一票否决压制器。
          3. 司令部意志 (Commander's Will): 必须得到主力真实信念的支持。采用 `main_force_conviction_index_D` 作为进攻意图放大器。
        - 数学模型: 意图分 = (战术胜利分 * 战略许可压制器) * 司令部意志放大器
        """
        # --- 1. 获取“司令部意志”原始数据 ---
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_offensive_absorption_intent")
        # --- 2. 构建三层诊断框架 ---
        # 基础层：战术胜利分
        tactical_victory_score = lower_shadow_quality
        # 过滤层：战略许可压制器
        strategic_clearance_suppressor = (1 - distribution_intent).clip(0, 1)
        # 增强层：司令部意志放大器
        # [修改的代码行] 修复TypeError：使用 normalize_to_bipolar 替换 normalize_score，以实现正确的双极性归一化
        conviction_amplifier = (normalize_to_bipolar(conviction_raw.clip(-5, 5), df.index, 55) + 1.0).clip(0.5, 1.5)
        # --- 3. 最终合成 ---
        base_intent_score = tactical_victory_score * strategic_clearance_suppressor
        final_offensive_absorption_intent = (base_intent_score * conviction_amplifier).clip(0, 1)
        # --- 深度战术探针 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [行为探针] _diagnose_offensive_absorption_intent @ {probe_date_str}")
                print(f"        --- [三层诊断框架] ---")
                print(f"          - 输入信号: 战术胜利(下影线品质)={tactical_victory_score.loc[probe_ts]:.4f}, 战略风险(派发意图)={distribution_intent.loc[probe_ts]:.4f}, 司令部意志(主力信念)={conviction_raw.loc[probe_ts]:.2f}")
                print(f"          - 中间计算: 战略许可压制器={strategic_clearance_suppressor.loc[probe_ts]:.4f}, 司令部意志放大器={conviction_amplifier.loc[probe_ts]:.4f}")
                print(f"          - 基础意图分 (压制后): {base_intent_score.loc[probe_ts]:.4f}")
                print(f"          - 最终进攻承接意图分 (放大后): {final_offensive_absorption_intent.loc[probe_ts]:.4f}")
        return final_offensive_absorption_intent.astype(np.float32)

    def _diagnose_deception_index(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.2 · 语义净化版】诊断博弈欺骗指数
        - 核心修复: 对 `deception_index_D` 应用 `.clip(lower=0)` 进行语义净化，
                      确保只有正值的“欺骗指数”才能被视为有效证据，彻底解决了因负值
                      导致融合计算不合逻辑的“语义错配谬误”。
        """
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 1. 获取原始数据
        # [修改的代码行] 对原始信号进行语义净化
        base_deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_diagnose_deception_index").clip(lower=0)
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_deception_index")
        closing_strength_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_diagnose_deception_index")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_deception_index")
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_diagnose_deception_index").replace(0, 1e-9)
        closing_ambush_raw = self._get_safe_series(df, 'closing_auction_ambush_D', 0.0, method_name="_diagnose_deception_index")
        # 2. 归一化和计算组件
        deception_evidence_score = get_adaptive_mtf_normalized_score((base_deception_raw + wash_trade_raw).pow(0.5), df.index, ascending=True, tf_weights=default_weights)
        flow_ratio = main_force_flow / amount
        flow_direction_score = get_adaptive_mtf_normalized_bipolar_score(flow_ratio, df.index, default_weights)
        normalized_strength = normalize_score(closing_strength_raw, df.index, 55)
        # 3. 计算看涨/看跌欺骗
        bullish_deception_score = (1 - normalized_strength) * deception_evidence_score * flow_direction_score.clip(lower=0)
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
                # [修改的代码行] 更新探针以反映净化后的值
                print(f"        - 欺骗证据分: {deception_evidence_score.loc[probe_ts]:.4f} (基础欺骗(净化后)={base_deception_raw.loc[probe_ts]:.2f}, 对倒={wash_trade_raw.loc[probe_ts]:.2f})")
                print(f"        - 主力流向分: {flow_direction_score.loc[probe_ts]:.4f} (原始流比率={flow_ratio.loc[probe_ts]:.4f})")
                print(f"        - 看涨欺骗分: {bullish_deception_score.loc[probe_ts]:.4f} (收盘弱势={1 - normalized_strength.loc[probe_ts]:.2f}, 原始强弱={closing_strength_raw.loc[probe_ts]:.2f})")
                print(f"        - 看跌欺骗分: {bearish_deception_score.loc[probe_ts]:.4f} (收盘强势={normalized_strength.loc[probe_ts]:.2f})")
                print(f"        - 基础欺骗指数: {base_deception_index.loc[probe_ts]:.4f}")
                print(f"        - 尾盘偷袭分: {closing_ambush_score.loc[probe_ts]:.4f} (原始值={closing_ambush_raw.loc[probe_ts]:.2f})")
                print(f"        - 最终博弈欺骗指数: {final_deception_index.loc[probe_ts]:.4f}")
        return final_deception_index.astype(np.float32)

    def _apply_neutral_zone_filter(self, series: pd.Series, threshold: float) -> pd.Series:
        """
        【V1.0 · 新增】应用中性“死区”过滤器。
        - 核心职责: 将信号中绝对值小于阈值的“噪声”强制归零，以符合业务逻辑。
        """
        if threshold > 0:
            return series.where(series.abs() > threshold, 0.0)
        return series

    def _probe_raw_material_diagnostics(self, df: pd.DataFrame, probe_ts: pd.Timestamp):
        """
        【V1.0 · 新增】原料数据深度探针。
        - 核心职责: 打印出导致关键信号归零的最底层、最原始的输入数据，
                      用于终极的根源验证。
        """
        probe_date_str = probe_ts.strftime('%Y-%m-%d')
        print(f"      [原料探针] 关键输入数据 @ {probe_date_str}")
        # --- 追溯“意图分”和“驱动分”的根源 ---
        raw_mf_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        raw_amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        flow_ratio = (raw_mf_flow / raw_amount) if raw_amount > 0 else 0
        print(f"        - 主力资金流 (根源): raw_mf_flow={raw_mf_flow:.2f}, amount={raw_amount:.2f} -> flow_ratio={flow_ratio:.6f}")
        # --- 追溯“恐慌承接度”的根源 ---
        raw_panic = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        raw_capitulation = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        print(f"        - 恐慌承接度 (根源): panic_cascade={raw_panic:.2f}, capitulation_absorption={raw_capitulation:.2f}")
        # --- 追溯“滞涨证据”中“宏观风险”的根源 ---
        raw_winner_rate = self._get_safe_series(df, 'total_winner_rate_D', 50.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        raw_trend_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name="_probe_raw_material_diagnostics")
        vitality_diff = raw_trend_vitality.diff(3).clip(upper=0).abs().loc[probe_ts]
        print(f"        - 宏观风险 (根源): winner_rate={raw_winner_rate:.2f}, trend_vitality_decay={vitality_diff:.2f}")










