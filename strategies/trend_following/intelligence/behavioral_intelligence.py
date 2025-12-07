# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional, List, Any
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score, normalize_to_bipolar
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
        【V30.1 · 依赖感知型背离品质重构版】行为情报模块总指挥
        - 核心重构: 修复了因信号生成顺序导致的依赖缺失问题。
                    现在，所有依赖信号（特别是 SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT）
                    都会在被使用之前生成并合并到主数据帧中，确保情报流的完整性。
        """
        all_behavioral_states = {}

        # [代码修改] 调整执行顺序：首先生成 SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT
        # 因为它是 _diagnose_divergence_quality 的依赖项，而 _diagnose_divergence_quality
        # 在 _diagnose_behavioral_axioms 内部被调用。
        micro_intent_signals = self._diagnose_microstructure_intent(df)
        self.strategy.atomic_states.update(micro_intent_signals)
        all_behavioral_states.update(micro_intent_signals)
        # [代码修改] 立即将微观意图信号合并到df中，供后续方法使用 _get_safe_series 获取
        for k, v in micro_intent_signals.items():
            if k not in df.columns:
                df[k] = v

        # [代码修改] 接着生成核心公理信号，此时 _diagnose_divergence_quality 可以访问到微观意图信号
        atomic_signals = self._diagnose_behavioral_axioms(df)
        # 如果核心公理诊断失败，则提前返回，防止后续错误
        if not atomic_signals:
            print("    -> [行为情报引擎] 核心公理诊断失败，行为分析中止。")
            return {}
        self.strategy.atomic_states.update(atomic_signals)
        all_behavioral_states.update(atomic_signals)
        # [代码修改] 将原子信号合并到df中，供后续的 _calculate_signal_dynamics 方法使用
        for k, v in atomic_signals.items():
            if k not in df.columns:
                df[k] = v

        # 生成上下文新高强度信号
        context_new_high_strength = self._diagnose_context_new_high_strength(df)
        self.strategy.atomic_states.update(context_new_high_strength)
        all_behavioral_states.update(context_new_high_strength)
        # [代码修改] 将上下文信号合并到df中
        for k, v in context_new_high_strength.items():
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
        downward_momentum_score = self._diagnose_downward_momentum(df)
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
        lower_shadow_quality = self._diagnose_lower_shadow_quality(df)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_quality
        # [代码修改] 调整调用顺序，并将 final_overextension_score 作为参数直接传递
        distribution_intent = self._diagnose_distribution_intent(df, default_weights, final_overextension_score)
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
        - 核心重构: 废弃了基于“表观强度幻觉”的 V2.0 模型。引入基于“闪电战三要素”
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

    def _diagnose_downward_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · Production Ready版】诊断价格下跌动能。
        - 核心重构: 废弃V2.0“单点否决”逻辑，引入基于“多头防御体系系统性崩溃”的全新诊断模型。
        - 诊断框架 (三要素):
          1. 前沿阵地失守 (The Frontline Breach): 审判下跌的破坏力 (跌幅 * 下跌效率)。
          2. 防御工事崩塌 (The Fortress Crumbling): 审判抵抗的缺席 (1 - 防御力量分)。
          3. 指挥系统溃败 (The Command Collapse): 审判主力信心的崩塌 (1 - 主力正面信念分)。
        - 数学模型: 动能分 = (破坏力 * 防御真空 * 信念真空) ^ (1/3)
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('scorched_earth_params'), {})
        weights = get_param_value(params.get('fusion_weights'), {'breach_force': 0.4, 'defense_vacuum': 0.3, 'command_vacuum': 0.3})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {})
        # --- 2. 获取原始数据 ---
        pct_change_raw = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_downward_momentum")
        efficiency_raw = self._get_safe_series(df, 'vacuum_traversal_efficiency_D', 0.0, method_name="_diagnose_downward_momentum")
        dip_absorption_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_downward_momentum")
        active_buying_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_diagnose_downward_momentum")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_downward_momentum")
        # --- 3. 计算核心组件 ---
        # 组件一：前沿阵地失守 (破坏力)
        raw_drop = pct_change_raw.clip(upper=0).abs()
        drop_score = get_adaptive_mtf_normalized_score(raw_drop, df.index, ascending=True, tf_weights=default_weights)
        efficiency_score = get_adaptive_mtf_normalized_score(efficiency_raw, df.index, ascending=True, tf_weights=default_weights)
        breach_force_score = (drop_score * efficiency_score).pow(0.5)
        # 组件二：防御工事崩塌 (防御真空)
        dip_absorption_score = get_adaptive_mtf_normalized_score(dip_absorption_raw, df.index, ascending=True, tf_weights=default_weights)
        active_buying_score = get_adaptive_mtf_normalized_score(active_buying_raw, df.index, ascending=True, tf_weights=default_weights)
        defense_power_score = (dip_absorption_score * 0.5 + active_buying_score * 0.5)
        defense_vacuum_score = (1 - defense_power_score).clip(0, 1)
        # 组件三：指挥系统溃败 (信念真空)
        positive_conviction_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        command_vacuum_score = (1 - positive_conviction_score).clip(0, 1)
        # --- 4. 最终合成 ---
        downward_momentum_score = (
            (breach_force_score + 1e-9).pow(weights.get('breach_force', 0.4)) *
            (defense_vacuum_score + 1e-9).pow(weights.get('defense_vacuum', 0.3)) *
            (command_vacuum_score + 1e-9).pow(weights.get('command_vacuum', 0.3))
        ).fillna(0.0)
        return downward_momentum_score.clip(0, 1).astype(np.float32)

    def _diagnose_offensive_absorption_intent(self, df: pd.DataFrame, lower_shadow_quality: pd.Series, distribution_intent: pd.Series) -> pd.Series:
        """
        【V4.0 · Production Ready版】诊断进攻性承接意图。
        - 核心重构: 废弃了对“下影线”形态的路径依赖，引入基于“战役过程”的全新诊断模型。
        - 诊断框架:
          1. 战略前提 (Strategic Prerequisite): 主力无派发意图 (`distribution_intent`)。
          2. 战役背景 (The Crisis): 审判战场抛压的烈度 (`panic_selling_cascade_D`)。
          3. 核心行动 (The Response): 审判多头的反攻力量 (融合 `dip_absorption_power_D` 和 `active_buying_support_D`)。
          4. 司令部意志 (Commander's Will): 审判主力真实信念 (`main_force_conviction_index_D`)。
        - 数学模型: 意图分 = 战略前提 * (背景分 * 行动分 * 意志分) ^ (1/3)
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('offensive_absorption_params'), {})
        weights = get_param_value(params.get('fusion_weights'), {'crisis_context': 0.3, 'counter_offensive_force': 0.4, 'commanders_will': 0.3})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {})
        # --- 2. 获取原始数据 ---
        crisis_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_offensive_absorption_intent")
        dip_absorption_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_offensive_absorption_intent")
        active_buying_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_diagnose_offensive_absorption_intent")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_offensive_absorption_intent")
        # --- 3. 计算各组件得分 ---
        # 组件一：战略前提 (一票否决)
        strategic_prerequisite_score = (1 - distribution_intent).clip(0, 1)
        # 组件二：战役背景分
        crisis_context_score = get_adaptive_mtf_normalized_score(crisis_raw, df.index, ascending=True, tf_weights=default_weights)
        # 组件三：核心行动分
        dip_absorption_score = get_adaptive_mtf_normalized_score(dip_absorption_raw, df.index, ascending=True, tf_weights=default_weights)
        active_buying_score = get_adaptive_mtf_normalized_score(active_buying_raw, df.index, ascending=True, tf_weights=default_weights)
        counter_offensive_force_score = (dip_absorption_score * 0.5 + active_buying_score * 0.5)
        # 组件四：司令部意志分
        commanders_will_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        # --- 4. 最终合成 ---
        base_quality_score = (
            (crisis_context_score + 1e-9).pow(weights.get('crisis_context', 0.3)) *
            (counter_offensive_force_score + 1e-9).pow(weights.get('counter_offensive_force', 0.4)) *
            (commanders_will_score + 1e-9).pow(weights.get('commanders_will', 0.3))
        ).fillna(0.0)
        final_offensive_absorption_intent = (base_quality_score * strategic_prerequisite_score).clip(0, 1)
        return final_offensive_absorption_intent.astype(np.float32)

    def _diagnose_intraday_bull_control(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V6.0 · Production Ready版】诊断“日内多头控制力”
        - 核心重构: 废弃V5.1“最后一分钟谎言谬误”模型，引入“战果×过程×叙事”的全新三维诊断框架。
        - 诊断三维度:
          1. 战略位置 (Strategic Position): 评估最终战果，即收盘价相对VWAP的位置。
          2. 过程品质 (Process Quality): 评估全天攻防动作的综合效率与信念。
          3. 叙事诚信度 (Narrative Integrity): 审判结局与过程是否一致，惩罚“尾盘偷袭”等欺骗行为。
        - 数学模型: 最终控制力 = 战略位置分 * (过程品质分 * 叙事诚信度分) ^ 0.5
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('chronos_protocol_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {'process_quality': 0.5, 'narrative_integrity': 0.5})
        # --- 2. 获取三维度原始数据 ---
        # 维度一：战略位置
        position_raw = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, method_name="_diagnose_intraday_bull_control")
        # 维度二：过程品质
        purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_intraday_bull_control")
        resistance_raw = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name="_diagnose_intraday_bull_control")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_intraday_bull_control")
        # 维度三：叙事诚信度
        posture_raw = self._get_safe_series(df, 'intraday_posture_score_D', 0.0, method_name="_diagnose_intraday_bull_control")
        ambush_raw = self._get_safe_series(df, 'closing_auction_ambush_D', 0.0, method_name="_diagnose_intraday_bull_control")
        # --- 3. 计算各维度得分 ---
        # 维度一：战略位置分 (作为基础分)
        strategic_position_score = position_raw.clip(-1, 1)
        # 维度二：过程品质分 (作为调节器)
        purity_score = get_adaptive_mtf_normalized_score(purity_raw, df.index, ascending=True, tf_weights=tf_weights)
        resistance_score = get_adaptive_mtf_normalized_score(resistance_raw, df.index, ascending=True, tf_weights=tf_weights)
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(conviction_raw, df.index, tf_weights)
        process_quality_score = ((purity_score + resistance_score) / 2 * (conviction_score.clip(0,1) + 1) / 2).clip(0, 1)
        # 维度三：叙事诚信度分 (作为调节器)
        posture_score = posture_raw.clip(-1, 1)
        ambush_score = get_adaptive_mtf_normalized_score(ambush_raw, df.index, ascending=True, tf_weights=tf_weights)
        narrative_deception_score = (ambush_score * (1 - posture_score.clip(lower=0))).clip(0, 1)
        narrative_integrity_score = (1 - narrative_deception_score)
        # --- 4. “时序裁决”三维合成 ---
        quality_modulator = (
            process_quality_score * fusion_weights.get('process_quality', 0.5) +
            narrative_integrity_score * fusion_weights.get('narrative_integrity', 0.5)
        )
        final_control_score = (strategic_position_score * quality_modulator).clip(-1, 1)
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        return final_control_score.astype(np.float32)

    def _diagnose_deception_index(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · Production Ready版】诊断博弈欺骗指数
        - 核心重构: 废弃V1.2“结果导向”模型，引入基于“认知失调”的全新诊断模型。
        - 核心逻辑: 欺骗分 = (主力真实意图向量 - K线表象剧本向量) * 证据放大器
                      直接量化“意图”与“表象”的背离程度，能识别更高明的欺骗形态。
        - 诊断三要素:
          1. 舞台剧本 (The Apparent Narrative): K线收盘位置讲述的故事 (`closing_strength_index_D`)。
          2. 幕后黑手 (The Hidden Intent): 主力真实的订单流意图 (`main_force_ofi_D`)。
          3. 作案工具 (The Deceptive Tools): 对倒、欺骗等行为 (`deception_index_D`, `wash_trade_intensity_D`)，作为放大器。
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('puppeteers_gambit_params'), {})
        k_amplifier = params.get('evidence_amplifier_k', 0.5)
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {})
        # --- 2. 获取原始数据 ---
        narrative_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_diagnose_deception_index")
        intent_raw = self._get_safe_series(df, 'main_force_ofi_D', 0.0, method_name="_diagnose_deception_index")
        base_deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_diagnose_deception_index").clip(lower=0)
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_deception_index")
        # --- 3. 计算核心组件 ---
        # 组件一：舞台剧本向量
        narrative_vector = normalize_to_bipolar(narrative_raw, df.index, 55)
        # 组件二：幕后意图向量
        intent_vector = get_adaptive_mtf_normalized_bipolar_score(intent_raw, df.index, default_weights)
        # 组件三：证据放大器
        deception_evidence_score = get_adaptive_mtf_normalized_score((base_deception_raw + wash_trade_raw).pow(0.5), df.index, ascending=True, tf_weights=default_weights)
        evidence_amplifier = 1 + k_amplifier * deception_evidence_score
        # --- 4. 计算认知失调并施加放大器 ---
        cognitive_dissonance_vector = (intent_vector - narrative_vector) / 2
        final_deception_index = (cognitive_dissonance_vector * evidence_amplifier).clip(-1, 1)
        return final_deception_index.astype(np.float32)

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
        【V3.0 · Production Ready版】诊断高品质上涨效率。
        - 核心重构: 废弃V2.1“皮洛士胜利谬误”模型，引入“战术品质 × 战略地形”的全新双维诊断框架。
        - 诊断双维度:
          1. 战术强攻品质 (The Spearhead's Edge): 沿用V2.1逻辑，评估“矛头”的锋利度。
          2. 战略环境地形 (The Battlefield Terrain): 新增战略评估，审判前方“地形”的阻力。
        - 数学模型: 最终效率分 = 战术品质分 * (1 - 战略阻力分)
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('pathfinder_protocol_params'), {})
        resistance_weights = get_param_value(params.get('resistance_weights'), {'chip_fatigue': 0.6, 'loser_pain': 0.4})
        # --- 2. 获取两大维度原始数据 ---
        # 维度一：战术品质
        purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_upward_efficiency")
        offensive_efficiency_raw = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name="_diagnose_upward_efficiency")
        suppression_raw = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name="_diagnose_upward_efficiency")
        # 维度二：战略地形
        chip_fatigue_raw = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name="_diagnose_upward_efficiency")
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_diagnose_upward_efficiency")
        # --- 3. 计算各维度得分 ---
        # 维度一：战术强攻品质分
        purity_score = get_adaptive_mtf_normalized_score(purity_raw, df.index, ascending=True, tf_weights=tf_weights)
        offensive_efficiency_score = get_adaptive_mtf_normalized_score(offensive_efficiency_raw, df.index, ascending=True, tf_weights=tf_weights)
        suppression_score = get_adaptive_mtf_normalized_score(suppression_raw, df.index, ascending=True, tf_weights=tf_weights)
        tactical_assault_score = (
            (purity_score + 1e-9).pow(0.4) *
            (offensive_efficiency_score + 1e-9).pow(0.3) *
            (suppression_score + 1e-9).pow(0.3)
        ).fillna(0.0)
        # 维度二：战略环境地形分
        chip_fatigue_score = get_adaptive_mtf_normalized_score(chip_fatigue_raw, df.index, ascending=True, tf_weights=tf_weights)
        loser_pain_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df.index, ascending=True, tf_weights=tf_weights)
        strategic_resistance_score = (
            chip_fatigue_score * resistance_weights.get('chip_fatigue', 0.6) +
            loser_pain_score * resistance_weights.get('loser_pain', 0.4)
        ).clip(0, 1)
        strategic_environment_score = (1 - strategic_resistance_score)
        # --- 4. 最终合成：战术品质 × 战略环境 ---
        final_upward_efficiency = (tactical_assault_score * strategic_environment_score).clip(0, 1)
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        return final_upward_efficiency.astype(np.float32)

    def _diagnose_downward_resistance(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V3.0 · Production Ready版】诊断高品质下跌抵抗。
        - 核心重构: 废弃V2.1“空城计谬误”模型，引入“战术应对 × 战略意图”的全新双维诊断框架。
        - 诊断双维度:
          1. 战术应对能力 (The Tactical Response): 沿用V2.1逻辑，评估防线的坚固度。
          2. 战略欺诈意图 (The Strategic Feint): 新增战略评估，审判抵抗的真实目的。
        - 数学模型: 最终抵抗分 = 战术应对分 * 战略意图分
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('elastic_defense_params'), {})
        intent_weights = get_param_value(params.get('intent_weights'), {'conviction': 0.6, 'cleansing': 0.4})
        # --- 2. 获取两大维度原始数据 ---
        # 维度一：战术应对
        passive_absorption_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_downward_resistance")
        active_defense_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name="_diagnose_downward_resistance")
        counter_attack_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_diagnose_downward_resistance")
        # 维度二：战略意图
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_downward_resistance")
        cleansing_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_downward_resistance")
        # --- 3. 计算各维度得分 ---
        # 维度一：战术应对能力分
        passive_absorption_score = get_adaptive_mtf_normalized_score(passive_absorption_raw, df.index, ascending=True, tf_weights=tf_weights)
        active_defense_score = get_adaptive_mtf_normalized_score(active_defense_raw, df.index, ascending=True, tf_weights=tf_weights)
        counter_attack_score = get_adaptive_mtf_normalized_score(counter_attack_raw, df.index, ascending=True, tf_weights=tf_weights)
        tactical_response_score = (
            (passive_absorption_score + 1e-9).pow(0.2) *
            (active_defense_score + 1e-9).pow(0.4) *
            (counter_attack_score + 1e-9).pow(0.4)
        ).fillna(0.0)
        # 维度二：战略意图分
        conviction_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        cleansing_score = get_adaptive_mtf_normalized_score(cleansing_raw, df.index, ascending=True, tf_weights=tf_weights)
        strategic_intent_score = (
            conviction_score * intent_weights.get('conviction', 0.6) +
            cleansing_score * intent_weights.get('cleansing', 0.4)
        ).clip(0, 1)
        # --- 4. 最终合成：战术应对 × 战略意图 ---
        final_downward_resistance = (tactical_response_score * strategic_intent_score).clip(0, 1)
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        return final_downward_resistance.astype(np.float32)

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
        【V2.0 · Production Ready版】微观结构意图诊断引擎
        - 核心重构: 废弃V1.3“静态快照”模型，引入“核心意图强度 × 环境适应性 × 行为一致性”的全新三维诊断框架。
        - 诊断三维度:
          1. 核心意图强度 (Core Intent Magnitude): 诊断主力订单流和挂单枯竭的原始意图。
          2. 环境适应性 (Environmental Adaptability): 根据市场波动率和流动性校准意图的显著性。
          3. 行为一致性 (Behavioral Coherence): 通过价格脉冲纯度、欺诈指数印证意图的真实性。
        - 数学模型: 最终微观意图 = 核心意图强度 × 环境适应性因子 × 行为一致性因子
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('fog_of_war_protocol_params'), {})
        core_intent_weights = get_param_value(params.get('core_intent_weights'), {"ofi": 0.6, "quote_exhaustion": 0.4})
        env_adapt_weights = get_param_value(params.get('environmental_adaptability_weights'), {"volatility_sensitivity": 0.5, "liquidity_sensitivity": 0.5})
        behavior_coherence_weights = get_param_value(params.get('behavioral_coherence_weights'), {"impulse_purity": 0.7, "deception_penalty": 0.3})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 2. 维度一：核心意图强度 (Core Intent Magnitude) ---
        required_signals_core = ['main_force_ofi_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D']
        if not self._validate_required_signals(df, required_signals_core, "_diagnose_microstructure_intent"):
            return {'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT': pd.Series(0.0, index=df.index)}
        ofi_raw = self._get_safe_series(df, 'main_force_ofi_D', 0.0, method_name="_diagnose_microstructure_intent")
        buy_sweep_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_microstructure_intent")
        sell_sweep_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_microstructure_intent")
        ofi_score = get_adaptive_mtf_normalized_bipolar_score(ofi_raw, df.index, default_weights)
        buy_sweep_score = get_adaptive_mtf_normalized_score(buy_sweep_raw, df.index, ascending=True, tf_weights=default_weights)
        sell_sweep_score = get_adaptive_mtf_normalized_score(sell_sweep_raw, df.index, ascending=True, tf_weights=default_weights)
        bullish_core_intent = (ofi_score.clip(lower=0) * core_intent_weights.get('ofi', 0.6) + buy_sweep_score * core_intent_weights.get('quote_exhaustion', 0.4))
        bearish_core_intent = (ofi_score.clip(upper=0).abs() * core_intent_weights.get('ofi', 0.6) + sell_sweep_score * core_intent_weights.get('quote_exhaustion', 0.4))
        core_intent_magnitude = (bullish_core_intent - bearish_core_intent).clip(-1, 1)
        # --- 3. 维度二：环境适应性 (Environmental Adaptability) ---
        required_signals_env = ['ATR_14_D', 'volume_ratio_D']
        environmental_adaptability_factor = pd.Series(1.0, index=df.index)
        if self._validate_required_signals(df, required_signals_env, "_diagnose_microstructure_intent"):
            atr_raw = self._get_safe_series(df, 'ATR_14_D', 0.0, method_name="_diagnose_microstructure_intent")
            volume_ratio_raw = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_diagnose_microstructure_intent")
            volatility_score = get_adaptive_mtf_normalized_score(atr_raw, df.index, ascending=True, tf_weights=default_weights)
            liquidity_score = (1 - get_adaptive_mtf_normalized_score(volume_ratio_raw, df.index, ascending=True, tf_weights=default_weights))
            environmental_adaptability_factor_raw = (
                volatility_score * env_adapt_weights.get('volatility_sensitivity', 0.5) +
                liquidity_score * env_adapt_weights.get('liquidity_sensitivity', 0.5)
            ).clip(0, 1)
            environmental_adaptability_factor = 0.5 + environmental_adaptability_factor_raw * 0.5 # 映射到 [0.5, 1.0]
        # --- 4. 维度三：行为一致性 (Behavioral Coherence) ---
        required_signals_behavior = ['upward_impulse_purity_D', 'vacuum_traversal_efficiency_D', 'deception_index_D']
        behavioral_coherence_factor = pd.Series(1.0, index=df.index)
        if self._validate_required_signals(df, required_signals_behavior, "_diagnose_microstructure_intent"):
            upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_diagnose_microstructure_intent")
            downward_purity_raw = self._get_safe_series(df, 'vacuum_traversal_efficiency_D', 0.0, method_name="_diagnose_microstructure_intent")
            deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_diagnose_microstructure_intent")
            upward_purity_score = get_adaptive_mtf_normalized_score(upward_purity_raw, df.index, ascending=True, tf_weights=default_weights)
            downward_purity_score = get_adaptive_mtf_normalized_score(downward_purity_raw, df.index, ascending=True, tf_weights=default_weights)
            deception_score = get_adaptive_mtf_normalized_score(deception_raw, df.index, ascending=True, tf_weights=default_weights)
            purity_coherence = pd.Series(0.0, index=df.index)
            purity_coherence = purity_coherence.mask(core_intent_magnitude > 0, upward_purity_score * (1 - downward_purity_score))
            purity_coherence = purity_coherence.mask(core_intent_magnitude < 0, downward_purity_score * (1 - upward_purity_score))
            purity_coherence = purity_coherence.mask(core_intent_magnitude == 0, 0.5)
            deception_penalty_factor = (1 - deception_score * behavior_coherence_weights.get('deception_penalty', 0.3)).clip(0, 1)
            behavioral_coherence_factor_raw = (
                purity_coherence * behavior_coherence_weights.get('impulse_purity', 0.7) +
                deception_penalty_factor * (1 - behavior_coherence_weights.get('impulse_purity', 0.7))
            ).clip(0, 1)
            behavioral_coherence_factor = 0.5 + behavioral_coherence_factor_raw * 0.5 # 映射到 [0.5, 1.0]
        # --- 5. 最终合成：三维融合 ---
        final_micro_intent = (core_intent_magnitude * environmental_adaptability_factor * behavioral_coherence_factor).clip(-1, 1)
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        states = {'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT': final_micro_intent.astype(np.float32)}
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

    def _diagnose_lower_shadow_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        【V13.0 · Production Ready版】诊断下影线承接品质。
        - 核心重构: 废弃V12.1“战地记者”模型，引入“剧本×表演×意图”的全新三幕式诊断框架。
        - 诊断三幕剧:
          1. 剧本 (The Script): 审判“危机”的真实性与烈度 (`panic_selling_cascade_D`)。
          2. 表演 (The Performance): 审判“主角”救场的完成度 (融合 `active_buying_support_D` 等)。
          3. 意图 (The Intent): 审判“导演”的真实内心独白 (融合 `main_force_conviction_index_D` 等)。
        - 数学模型: 品质分 = (剧本品质 * 表演品质) ^ 0.5 * 导演意图分
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('directors_cut_params'), {})
        intent_weights = get_param_value(params.get('intent_weights'), {'conviction': 0.7, 'covert_ops': 0.3})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {})
        # --- 2. 获取三幕剧的原料数据 ---
        # Act I: 剧本
        script_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        # Act II: 表演
        performance_active_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        performance_dip_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        # Act III: 意图
        intent_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        intent_covert_ops_raw = self._get_safe_series(df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_lower_shadow_quality")
        # --- 3. 计算各幕得分 ---
        # 第一幕：剧本品质分
        script_quality_score = get_adaptive_mtf_normalized_score(script_raw, df.index, ascending=True, tf_weights=default_weights)
        # 第二幕：表演品质分
        performance_active_score = get_adaptive_mtf_normalized_score(performance_active_raw, df.index, ascending=True, tf_weights=default_weights)
        performance_dip_score = get_adaptive_mtf_normalized_score(performance_dip_raw, df.index, ascending=True, tf_weights=default_weights)
        performance_quality_score = (performance_active_score * 0.6 + performance_dip_score * 0.4)
        # 第三幕：导演意图分
        intent_conviction_score = get_adaptive_mtf_normalized_score(intent_conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        intent_covert_ops_score = get_adaptive_mtf_normalized_score(intent_covert_ops_raw, df.index, ascending=True, tf_weights=default_weights)
        directors_intent_score = (
            intent_conviction_score * intent_weights.get('conviction', 0.7) +
            intent_covert_ops_score * intent_weights.get('covert_ops', 0.3)
        ).clip(0, 1)
        # --- 4. 最终合成 ---
        base_drama_quality = (script_quality_score * performance_quality_score).pow(0.5).fillna(0.0)
        final_lower_shadow_quality = (base_drama_quality * directors_intent_score).clip(0, 1)
        return final_lower_shadow_quality.astype(np.float32)

    def _diagnose_distribution_intent(self, df: pd.DataFrame, tf_weights: Dict, overextension_raw: pd.Series) -> pd.Series:
        """
        【V7.0 · Production Ready版】诊断派发意图。
        - 核心重构: 废弃V6.0“战术绝对主义”乘法模型，引入“双轨独立审判”框架。
        - 诊断双轨:
          1. 战术风险 (Tactical Risk): 评估主动派发动作，即“风暴”强度。
          2. 战略风险 (Strategic Risk): 评估战场环境恶化，即“大气压”读数。
        - 数学模型: 最终风险 = max(战术风险, 战略风险) * (1 + 协同奖励)
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('atmospheric_pressure_params'), {})
        synergy_bonus = get_param_value(params.get('synergy_bonus_factor'), 0.2)
        env_params = get_param_value(p_conf.get('judgment_day_protocol_params'), {})
        env_weights = get_param_value(env_params.get('environment_weights'), {'fatigue': 0.4, 'decay': 0.3, 'betrayal': 0.3})
        # --- 2. 轨道一：战术风险评估 (风暴强度) ---
        motive_raw = self._get_safe_series(df, 'profit_taking_flow_ratio_D', 0.0, method_name="_diagnose_distribution_intent")
        rally_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_distribution_intent")
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_distribution_intent")
        fingerprint_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_distribution_intent")
        motive_score = get_adaptive_mtf_normalized_score(motive_raw, df.index, ascending=True, tf_weights=tf_weights)
        rally_pressure_score = get_adaptive_mtf_normalized_score(rally_pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        upper_shadow_score = get_adaptive_mtf_normalized_score(upper_shadow_pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        weapon_score = (rally_pressure_score * 0.5 + upper_shadow_score * 0.5)
        fingerprint_score = get_adaptive_mtf_normalized_score(fingerprint_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=tf_weights)
        tactical_risk_score = (
            (motive_score + 1e-9).pow(0.2) *
            (weapon_score + 1e-9).pow(0.4) *
            (fingerprint_score + 1e-9).pow(0.4)
        ).fillna(0.0)
        # --- 3. 轨道二：战略风险评估 (大气压读数) ---
        vitality_raw = self._get_safe_series(df, 'trend_vitality_index_D', 0.5, method_name="_diagnose_distribution_intent")
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_diagnose_distribution_intent")
        control_solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.5, method_name="_diagnose_distribution_intent")
        conviction_slope_raw = self._get_safe_series(df, 'SLOPE_5_main_force_conviction_index_D', 0.0, method_name="_diagnose_distribution_intent")
        vitality_score = get_adaptive_mtf_normalized_score(vitality_raw, df.index, ascending=True, tf_weights=tf_weights)
        bullish_fatigue_score = ((1 - vitality_score) * overextension_raw).pow(0.5)
        stability_score = get_adaptive_mtf_normalized_score(winner_stability_raw, df.index, ascending=True, tf_weights=tf_weights)
        solidity_score = get_adaptive_mtf_normalized_score(control_solidity_raw, df.index, ascending=True, tf_weights=tf_weights)
        fortress_decay_score = ((1 - stability_score) * (1 - solidity_score)).pow(0.5)
        commanders_betrayal_score = get_adaptive_mtf_normalized_score(conviction_slope_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=tf_weights)
        strategic_risk_score = (
            bullish_fatigue_score * env_weights.get('fatigue', 0.4) +
            fortress_decay_score * env_weights.get('decay', 0.3) +
            commanders_betrayal_score * env_weights.get('betrayal', 0.3)
        ).clip(0, 1)
        # --- 4. 最终合成：双轨独立审判 ---
        base_risk = pd.concat([tactical_risk_score, strategic_risk_score], axis=1).max(axis=1)
        synergy_amplifier = 1 + (tactical_risk_score * strategic_risk_score).pow(0.5) * synergy_bonus
        final_distribution_intent = (base_risk * synergy_amplifier).clip(0, 1)
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        return final_distribution_intent.astype(np.float32)

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
        【V5.0 · 诡道背离协议 (探针激活版)】诊断高品质价量/价资背离
        - 核心重构: 废弃V4.0“宏观趋势分析”模型，引入“背离深度与广度 × 战略位置 × 双重确认 × 欺骗叙事确认”的全新四维诊断框架。
        - 诊断四维度:
          1. 背离深度与广度 (Divergence Depth & Breadth): 使用斜率更鲁棒地检测价格趋势和主力信念趋势。
          2. 战略位置 (Strategic Location): 评估背离是否发生在绝望区或获利盘不稳定区。
          3. 双重确认 (Dual Confirmation): 由“主力承接/派发”和“微观意图”进行双重印证。
          4. 欺骗叙事确认 (Deceptive Narrative Confirmation): 引入欺骗指数的负向部分，捕捉诱多本质。
        - 数学模型: 品质分 = (背离深度与广度分^0.4 * 战略位置分^0.3 * 主力承接/派发确认分^0.2 * 微观意图确认分^0.1 * 欺骗叙事确认分^0.1)
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('deceptive_divergence_protocol_params'), {})
        bullish_magnitude_params = get_param_value(params.get('bullish_magnitude_params'), {"price_downtrend_slope_window": 5, "conviction_uptrend_slope_window": 5})
        bearish_magnitude_params = get_param_value(params.get('bearish_magnitude_params'), {"price_slope_window": 5, "conviction_downtrend_slope_window": 5})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"magnitude": 0.4, "location": 0.3, "bullish_absorption_confirmation": 0.2, "bearish_distribution_confirmation": 0.2, "micro_intent_confirmation": 0.1, "deceptive_narrative_confirmation": 0.1})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 2. 获取所有原始数据 ---
        required_signals = [
            'close_D', 'main_force_conviction_index_D', 'loser_pain_index_D', 'winner_stability_index_D',
            f'SLOPE_{bullish_magnitude_params.get("price_downtrend_slope_window", 5)}_close_D',
            f'SLOPE_{bullish_magnitude_params.get("conviction_uptrend_slope_window", 5)}_main_force_conviction_index_D',
            f'SLOPE_{bearish_magnitude_params.get("price_slope_window", 5)}_close_D', # [代码修改] 确保熊市价格斜率窗口参数被使用
            f'SLOPE_{bearish_magnitude_params.get("conviction_downtrend_slope_window", 5)}_main_force_conviction_index_D', # [代码修改] 确保熊市信念斜率窗口参数被使用
            'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT',
            'deception_index_D' # [代码修改] 新增欺骗指数信号
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_divergence_quality"):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        price = self._get_safe_series(df, 'close_D', 0.0, method_name="_diagnose_divergence_quality")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_divergence_quality")
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_diagnose_divergence_quality")
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_diagnose_divergence_quality")
        # [代码修改] 使用各自的斜率窗口参数
        bullish_price_slope_raw = self._get_safe_series(df, f'SLOPE_{bullish_magnitude_params.get("price_downtrend_slope_window", 5)}_close_D', 0.0, method_name="_diagnose_divergence_quality")
        bullish_conviction_slope_raw = self._get_safe_series(df, f'SLOPE_{bullish_magnitude_params.get("conviction_uptrend_slope_window", 5)}_main_force_conviction_index_D', 0.0, method_name="_diagnose_divergence_quality")
        bearish_price_slope_raw = self._get_safe_series(df, f'SLOPE_{bearish_magnitude_params.get("price_slope_window", 5)}_close_D', 0.0, method_name="_diagnose_divergence_quality")
        bearish_conviction_slope_raw = self._get_safe_series(df, f'SLOPE_{bearish_magnitude_params.get("conviction_downtrend_slope_window", 5)}_main_force_conviction_index_D', 0.0, method_name="_diagnose_divergence_quality")
        micro_intent_raw = self._get_safe_series(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0, method_name="_diagnose_divergence_quality")
        deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_diagnose_divergence_quality") # [代码修改] 获取欺骗指数原始数据
        # --- 3. 计算牛市背离 (价格下跌趋势 vs 信念上升趋势) ---
        # 维度一：背离深度与广度 (Magnitude)
        # 价格下跌趋势：斜率为负且显著
        price_downtrend_score = get_adaptive_mtf_normalized_score(bullish_price_slope_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        # 信念上升趋势：斜率为正且显著
        conviction_uptrend_score = get_adaptive_mtf_normalized_score(bullish_conviction_slope_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        # 融合：价格下跌趋势与信念上升趋势同时存在
        bullish_magnitude_score = (price_downtrend_score * conviction_uptrend_score).pow(0.5).fillna(0.0)
        # 维度二：战略位置 (Location)
        bullish_location_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df.index, ascending=True, tf_weights=default_weights)
        # 维度三：双重确认 (Dual Confirmation)
        bullish_absorption_confirmation_score = absorption_strength
        bullish_micro_intent_confirmation_score = micro_intent_raw.clip(lower=0) # 只取看涨部分
        # --- 4. 牛市背离品质合成 ---
        bullish_divergence_quality = (
            (bullish_magnitude_score + 1e-9).pow(fusion_weights.get('magnitude', 0.4)) *
            (bullish_location_score + 1e-9).pow(fusion_weights.get('location', 0.3)) *
            (bullish_absorption_confirmation_score + 1e-9).pow(fusion_weights.get('bullish_absorption_confirmation', 0.2)) *
            (bullish_micro_intent_confirmation_score + 1e-9).pow(fusion_weights.get('micro_intent_confirmation', 0.1))
        ).fillna(0.0)
        # --- 5. 计算熊市背离 (价格上升趋势 vs 信念下降趋势) ---
        # 维度一：背离深度与广度 (Magnitude)
        price_uptrend_score = get_adaptive_mtf_normalized_score(bearish_price_slope_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        conviction_downtrend_score = get_adaptive_mtf_normalized_score(bearish_conviction_slope_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        bearish_magnitude_score = (price_uptrend_score * conviction_downtrend_score).pow(0.5).fillna(0.0)
        # 维度二：战略位置 (Location)
        winner_instability_raw = 1 - winner_stability_raw # 获利盘不稳定性
        bearish_location_score = get_adaptive_mtf_normalized_score(winner_instability_raw, df.index, ascending=True, tf_weights=default_weights)
        # 维度三：双重确认 (Dual Confirmation)
        bearish_distribution_confirmation_score = distribution_intent
        bearish_micro_intent_confirmation_score = micro_intent_raw.clip(upper=0).abs() # 只取看跌部分
        # [代码修改] 新增欺骗叙事确认因子
        # 欺骗指数为负（即价格表现强于主力真实意图）时，该因子得分高，作为熊市背离的额外确认
        deceptive_narrative_confirmation_score = get_adaptive_mtf_normalized_score(
            deception_raw.clip(upper=0).abs(), # 取负向欺骗的绝对值
            df.index,
            ascending=True,
            tf_weights=default_weights
        )
        # --- 6. 熊市背离品质合成 ---
        bearish_divergence_quality = (
            (bearish_magnitude_score + 1e-9).pow(fusion_weights.get('magnitude', 0.4)) *
            (bearish_location_score + 1e-9).pow(fusion_weights.get('location', 0.3)) *
            (bearish_distribution_confirmation_score + 1e-9).pow(fusion_weights.get('bearish_distribution_confirmation', 0.2)) * # [代码修改] 使用bearish_distribution_confirmation权重
            (bearish_micro_intent_confirmation_score + 1e-9).pow(fusion_weights.get('micro_intent_confirmation', 0.1)) *
            (deceptive_narrative_confirmation_score + 1e-9).pow(fusion_weights.get('deceptive_narrative_confirmation', 0.1)) # [代码修改] 新增欺骗叙事确认因子
        ).fillna(0.0)
        # --- [探针逻辑] 暴露所有计算节点 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates and not df.empty:
            for probe_date_str in probe_dates:
                try:
                    probe_date = pd.to_datetime(probe_date_str).tz_localize(df.index.tz)
                    if probe_date in df.index:
                        print(f"      [行为探针 V5.0] _diagnose_divergence_quality @ {probe_date_str}")
                        # --- 打印原料数据 ---
                        print(f"        --- [原料数据] ---")
                        print(f"          - 价格 (close_D): {price.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 主力信念指数 (main_force_conviction_index_D): {conviction_raw.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 亏损盘痛苦指数 (loser_pain_index_D): {loser_pain_raw.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 获利盘稳定性指数 (winner_stability_index_D): {winner_stability_raw.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 牛市价格斜率 (SLOPE_{bullish_magnitude_params.get('price_downtrend_slope_window', 5)}_close_D): {bullish_price_slope_raw.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 牛市信念斜率 (SLOPE_{bullish_magnitude_params.get('conviction_uptrend_slope_window', 5)}_main_force_conviction_index_D): {bullish_conviction_slope_raw.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 熊市价格斜率 (SLOPE_{bearish_magnitude_params.get('price_slope_window', 5)}_close_D): {bearish_price_slope_raw.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 熊市信念斜率 (SLOPE_{bearish_magnitude_params.get('conviction_downtrend_slope_window', 5)}_main_force_conviction_index_D): {bearish_conviction_slope_raw.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 承接强度 (absorption_strength): {absorption_strength.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 派发意图 (distribution_intent): {distribution_intent.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 微观结构意图 (SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT): {micro_intent_raw.get(probe_date, 'N/A'):.4f}")
                        print(f"          - 欺骗指数 (deception_index_D): {deception_raw.get(probe_date, 'N/A'):.4f}") # [代码修改] 新增欺骗指数原始数据探针
                        # --- 打印关键计算节点 - 牛市背离 ---
                        print(f"        --- [关键计算节点 - 牛市背离品质] ---")
                        print(f"          - [维度一] 背离深度与广度:")
                        print(f"              - 价格下跌趋势分: {price_downtrend_score.get(probe_date, 'N/A'):.4f}")
                        print(f"              - 信念上升趋势分: {conviction_uptrend_score.get(probe_date, 'N/A'):.4f}")
                        print(f"              - [融合] 背离深度与广度分 (bullish_magnitude_score): {bullish_magnitude_score.get(probe_date, 'N/A'):.4f}")
                        print(f"          - [维度二] 战略位置分 (bullish_location_score): {bullish_location_score.get(probe_date, 'N/A'):.4f}")
                        print(f"          - [维度三] 双重确认:")
                        print(f"              - 主力承接确认分 (bullish_absorption_confirmation_score): {bullish_absorption_confirmation_score.get(probe_date, 'N/A'):.4f}")
                        print(f"              - 微观意图确认分 (bullish_micro_intent_confirmation_score): {bullish_micro_intent_confirmation_score.get(probe_date, 'N/A'):.4f}")
                        # --- 最终结果 - 牛市背离 ---
                        print(f"        --- [最终结果 - 牛市背离品质] ---")
                        print(f"        - 最终牛市背离品质分: {bullish_divergence_quality.get(probe_date, 0.0):.4f}")
                        # --- 打印关键计算节点 - 熊市背离 ---
                        print(f"        --- [关键计算节点 - 熊市背离品质] ---")
                        print(f"          - [维度一] 背离深度与广度:")
                        print(f"              - 价格上升趋势分: {price_uptrend_score.get(probe_date, 'N/A'):.4f}")
                        print(f"              - 信念下降趋势分: {conviction_downtrend_score.get(probe_date, 'N/A'):.4f}")
                        print(f"              - [融合] 背离深度与广度分 (bearish_magnitude_score): {bearish_magnitude_score.get(probe_date, 'N/A'):.4f}")
                        print(f"          - [维度二] 战略位置分 (bearish_location_score): {bearish_location_score.get(probe_date, 'N/A'):.4f}")
                        print(f"          - [维度三] 三重确认:") # [代码修改] 更改为三重确认
                        print(f"              - 主力派发确认分 (bearish_distribution_confirmation_score): {bearish_distribution_confirmation_score.get(probe_date, 'N/A'):.4f}")
                        print(f"              - 微观意图确认分 (bearish_micro_intent_confirmation_score): {bearish_micro_intent_confirmation_score.get(probe_date, 'N/A'):.4f}")
                        print(f"              - 欺骗叙事确认分 (deceptive_narrative_confirmation_score): {deceptive_narrative_confirmation_score.get(probe_date, 'N/A'):.4f}") # [代码修改] 新增欺骗叙事确认因子探针
                        # --- 最终结果 - 熊市背离 ---
                        print(f"        --- [最终结果 - 熊市背离品质] ---")
                        print(f"        - 最终熊市背离品质分: {bearish_divergence_quality.get(probe_date, 0.0):.4f}")
                except Exception as e:
                    print(f"    -> [行为探针错误] _diagnose_divergence_quality 处理日期 {probe_date_str} 失败: {e}")
        return bullish_divergence_quality.clip(0, 1).astype(np.float32), bearish_divergence_quality.clip(0, 1).astype(np.float32)

    def _calculate_volume_burst_quality(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V3.0 · Production Ready版】计算高品质看涨量能爆发信号。
        - 核心重构: 废弃V2.1“战术近视眼”模型，引入“战术品质 × 战略环境”的全新双维诊断框架。
        - 诊断双维度:
          1. 战术强攻品质 (Tactical Assault Quality): 保留V2.1四维模型(幅度、信念、效率、战果)，评估登陆部队战斗力。
          2. 战略环境评估 (Strategic Environment Assessment): 新增“滩头阵地阻力指数”，评估登陆点上方的套牢盘压力。
        - 数学模型: 品质分 = 战术品质分 * (1 - 战略阻力分)
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('beachhead_protocol_params'), {})
        strategic_weights = get_param_value(params.get('strategic_weights'), {'chip_fatigue': 0.6, 'loser_pain': 0.4})
        # --- 2. 维度一：战术强攻品质评估 (沿用V2.1逻辑) ---
        # 2.1 获取战术原料数据
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_calculate_volume_burst_quality")
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_volume_burst_quality")
        efficiency_raw = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name="_calculate_volume_burst_quality")
        result_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_calculate_volume_burst_quality")
        # 2.2 计算战术各要素得分
        magnitude_score = get_adaptive_mtf_normalized_score(volume_ratio, df.index, ascending=True, tf_weights=tf_weights)
        conviction_score = get_adaptive_mtf_normalized_score(conviction_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        efficiency_score = get_adaptive_mtf_normalized_score(efficiency_raw, df.index, ascending=True, tf_weights=tf_weights)
        result_score = normalize_score(result_raw, df.index, 55)
        # 2.3 合成战术强攻品质分
        tactical_assault_quality_score = (
            (magnitude_score + 1e-9) * (conviction_score + 1e-9) *
            (efficiency_score + 1e-9) * (result_score + 1e-9)
        ).pow(1/4).fillna(0.0)
        # --- 3. 维度二：战略环境评估 ---
        # 3.1 获取战略原料数据
        chip_fatigue_raw = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name="_calculate_volume_burst_quality")
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_calculate_volume_burst_quality")
        # 3.2 计算战略阻力各要素得分
        chip_fatigue_score = get_adaptive_mtf_normalized_score(chip_fatigue_raw, df.index, ascending=True, tf_weights=tf_weights)
        loser_pain_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 3.3 合成滩头阵地阻力指数
        beachhead_resistance_score = (
            chip_fatigue_score * strategic_weights.get('chip_fatigue', 0.6) +
            loser_pain_score * strategic_weights.get('loser_pain', 0.4)
        ).clip(0, 1)
        strategic_environment_score = (1 - beachhead_resistance_score)
        # --- 4. 最终合成：战术品质 × 战略环境 ---
        final_burst_quality = (tactical_assault_quality_score * strategic_environment_score).clip(0, 1)
        return final_burst_quality.astype(np.float32)

    def _calculate_volume_atrophy(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V3.0 · Production Ready版】计算高品质成交量萎缩信号。
        - 核心重构: 废弃V2.1“静态快照谬误”模型，引入“战略环境×静态筹码×动态过程”的全新三维诊断框架。
        - 诊断三维度:
          1. 战略环境门控 (The Furnace Check): 审判多头是否掌控日内主导权，作为点火前提。
          2. 筹码纯度诊断 (The Purity Test): 沿用V2.1逻辑，评估“炉料”品质（获利盘、套牢盘、浮筹）。
          3. 过程稳定性封印 (The Stability Seal): 新增动态诊断，审判“淬炼”过程是否平稳（低波动率）。
        - 数学模型: 品质分 = 战略门控 * 基础萎缩分 * (纯度分 * 稳定分) ^ 0.5
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('crucible_protocol_params'), {})
        stability_window = get_param_value(params.get('stability_window'), 5)
        quality_weights = get_param_value(params.get('quality_weights'), {'purity_score': 0.6, 'stability_score': 0.4})
        # --- 2. 获取原料数据 ---
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_calculate_volume_atrophy")
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name="_calculate_volume_atrophy")
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_calculate_volume_atrophy")
        cleansing_efficiency_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_calculate_volume_atrophy")
        vwap_control_raw = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, method_name="_calculate_volume_atrophy")
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name="_calculate_volume_atrophy")
        # --- 3. 计算核心组件 ---
        # 组件一：战略环境门控 (The Furnace Check)
        # 只有当多头至少取得平局或优势时，门控才开启
        strategic_context_gate = normalize_score(vwap_control_raw, df.index, 55)
        # 组件二：基础萎缩分
        base_atrophy_score = 1 - get_adaptive_mtf_normalized_score(volume_ratio, df.index, ascending=True, tf_weights=tf_weights)
        # 组件三：筹码纯度诊断 (The Purity Test)
        lockup_score = get_adaptive_mtf_normalized_score(winner_stability_raw, df.index, ascending=True, tf_weights=tf_weights)
        exhaustion_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df.index, ascending=True, tf_weights=tf_weights)
        cleansing_score = get_adaptive_mtf_normalized_score(cleansing_efficiency_raw, df.index, ascending=True, tf_weights=tf_weights)
        purity_score = ((lockup_score + 1e-9) * (exhaustion_score + 1e-9) * (cleansing_score + 1e-9)).pow(1/3).fillna(0.0)
        # 组件四：过程稳定性封印 (The Stability Seal)
        price_volatility = close_price.pct_change().rolling(window=stability_window).std().fillna(0)
        normalized_volatility = get_adaptive_mtf_normalized_score(price_volatility, df.index, ascending=True, tf_weights=tf_weights)
        stability_score = (1 - normalized_volatility).clip(0, 1)
        # --- 4. 最终品质合成 ---
        quality_modulator = (
            (purity_score).pow(quality_weights.get('purity_score', 0.6)) *
            (stability_score).pow(quality_weights.get('stability_score', 0.4))
        ).fillna(0.0)
        final_atrophy_quality = (strategic_context_gate * base_atrophy_score * quality_modulator).clip(0, 1)
        return final_atrophy_quality.astype(np.float32)

    def _calculate_absorption_strength(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V3.0 · Production Ready版】计算高品质承接强度信号。
        - 核心重构: 废弃V2.1“孤城谬误”模型，引入“地基×行动×意图”的全新三维诊断框架。
        - 诊断三维度:
          1. 地基勘探 (The Foundation Survey): 审判承接是否发生在经过验证的坚固支撑位上。
          2. 构筑行动 (The Construction Action): 审判承接过程本身的强度与主动性。
          3. 总督意志 (The Governor's Will): 审判承接行为背后的主力真实信念。
        - 数学模型: 强度分 = (地基品质分 * 构筑行动分 * 总督意志分) ^ (1/3)
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('citadel_protocol_params'), {})
        action_weights = get_param_value(params.get('action_weights'), {'dip_absorption': 0.6, 'active_buying': 0.4})
        # --- 2. 获取三维度原始数据 ---
        # 维度一：地基
        foundation_raw = self._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name="_calculate_absorption_strength")
        # 维度二：行动
        action_dip_raw = self._get_safe_series(df, 'dip_absorption_power_D', 0.0, method_name="_calculate_absorption_strength")
        action_active_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_absorption_strength")
        # 维度三：意图
        intent_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_absorption_strength")
        # --- 3. 计算各维度得分 ---
        # 维度一：地基品质分
        foundation_score = get_adaptive_mtf_normalized_score(foundation_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 维度二：构筑行动分
        action_dip_score = get_adaptive_mtf_normalized_score(action_dip_raw, df.index, ascending=True, tf_weights=tf_weights)
        action_active_score = get_adaptive_mtf_normalized_score(action_active_raw, df.index, ascending=True, tf_weights=tf_weights)
        construction_action_score = (
            action_dip_score * action_weights.get('dip_absorption', 0.6) +
            action_active_score * action_weights.get('active_buying', 0.4)
        )
        # 维度三：总督意志分
        governors_will_score = get_adaptive_mtf_normalized_score(intent_raw.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
        # --- 4. “堡垒协议”三维合成 ---
        absorption_strength = (
            (foundation_score + 1e-9) *
            (construction_action_score + 1e-9) *
            (governors_will_score + 1e-9)
        ).pow(1/3).fillna(0.0)
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        return absorption_strength.clip(0, 1).astype(np.float32)

    def _diagnose_shakeout_confirmation(self, df: pd.DataFrame, absorption_strength: pd.Series, distribution_intent: pd.Series) -> pd.Series:
        """
        【V3.0 · Production Ready版】诊断震荡洗盘确认信号。
        - 核心重构: 废弃V2.1“事件审计员”模型，引入“意图×行动×品质”的全新三维诊断框架。
        - 诊断三维度:
          1. 战略意图 (Strategic Intent): 审判动机，必须满足“无派发意图”。
          2. 战术行动 (Tactical Action): 评估核心承接力量 (`absorption_strength`)。
          3. 执行品质 (Execution Quality): 评估洗盘技艺（效率、控制力、决断力）。
        - 数学模型: 确认分 = 战略意图 * (战术行动 * 执行品质) ^ 0.5
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('grandmasters_protocol_params'), {})
        quality_weights = get_param_value(params.get('quality_weights'), {'efficiency': 0.4, 'control': 0.4, 'decisiveness': 0.2})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_conf.get('default_weights'), {})
        # --- 2. 获取三维度原始数据 ---
        # 维度一：战略意图 (已作为参数传入)
        # 维度二：战术行动 (已作为参数传入)
        # 维度三：执行品质
        efficiency_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_diagnose_shakeout_confirmation")
        control_raw = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, method_name="_diagnose_shakeout_confirmation")
        high = self._get_safe_series(df, 'high_D', 0.0, method_name="_diagnose_shakeout_confirmation")
        low = self._get_safe_series(df, 'low_D', 0.0, method_name="_diagnose_shakeout_confirmation")
        close = self._get_safe_series(df, 'close_D', 0.0, method_name="_diagnose_shakeout_confirmation")
        # --- 3. 计算各维度得分 ---
        # 维度一：战略意图分
        strategic_intent_score = (1 - distribution_intent).clip(0, 1)
        # 维度二：战术行动分
        tactical_action_score = absorption_strength
        # 维度三：执行品质分 (工匠指数)
        efficiency_score = get_adaptive_mtf_normalized_score(efficiency_raw, df.index, ascending=True, tf_weights=default_weights)
        control_score = normalize_score(control_raw, df.index, 55) # VWAP控制力本身就是[-1,1]附近，用简单归一化即可
        decisiveness_score = ((close - low) / (high - low + 1e-9)).fillna(0.5).clip(0, 1)
        execution_quality_score = (
            efficiency_score * quality_weights.get('efficiency', 0.4) +
            control_score * quality_weights.get('control', 0.4) +
            decisiveness_score * quality_weights.get('decisiveness', 0.2)
        ).clip(0, 1)
        # --- 4. “大国工匠协议”三维合成 ---
        base_confirmation = (tactical_action_score * execution_quality_score).pow(0.5).fillna(0.0)
        shakeout_confirmation_score = (strategic_intent_score * base_confirmation).clip(0, 1)
        # [代码修改] 移除整个探针逻辑块，恢复生产状态
        return shakeout_confirmation_score.astype(np.float32)

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










