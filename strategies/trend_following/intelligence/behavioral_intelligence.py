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
        【V30.2 · 依赖感知型背离品质重构版】行为情报模块总指挥
        - 核心重构: 修复了因信号生成顺序导致的依赖缺失问题。
                    现在，所有依赖信号（特别是 SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT）
                    都会在被使用之前生成并合并到主数据帧中，确保情报流的完整性。
        - 【修正】确保df在整个调用链中正确更新。
        """
        all_behavioral_states = {}
        # 调整执行顺序：首先生成 SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT
        # 因为它是 _diagnose_divergence_quality 的依赖项，而 _diagnose_divergence_quality
        # 在 _diagnose_behavioral_axioms 内部被调用。
        micro_intent_signals = self._diagnose_microstructure_intent(df)
        self.strategy.atomic_states.update(micro_intent_signals)
        all_behavioral_states.update(micro_intent_signals)
        # 立即将微观意图信号合并到df中，供后续方法使用 _get_safe_series 获取
        for k, v in micro_intent_signals.items():
            if k not in df.columns:
                df[k] = v
        # 接着生成核心公理信号，此时 _diagnose_divergence_quality 可以访问到微观意图信号
        atomic_signals = self._diagnose_behavioral_axioms(df)
        # 如果核心公理诊断失败，则提前返回，防止后续错误
        if not atomic_signals:
            print("    -> [行为情报引擎] 核心公理诊断失败，行为分析中止。")
            return {}
        self.strategy.atomic_states.update(atomic_signals)
        all_behavioral_states.update(atomic_signals)
        # 将原子信号合并到df中，供后续的 _calculate_signal_dynamics 方法使用
        for k, v in atomic_signals.items():
            if k not in df.columns:
                df[k] = v
        # 生成上下文新高强度信号
        context_new_high_strength = self._diagnose_context_new_high_strength(df)
        self.strategy.atomic_states.update(context_new_high_strength)
        # 修正NameError: context_new_high_high_strength -> context_new_high_strength
        all_behavioral_states.update(context_new_high_strength) 
        # 将上下文信号合并到df中
        for k, v in context_new_high_strength.items():
            if k not in df.columns:
                df[k] = v
        # 将_calculate_signal_dynamics返回的新的DataFrame重新赋值给df
        df = self._calculate_signal_dynamics(df) 
        dynamic_cols = [c for c in df.columns if c.startswith(('MOMENTUM_', 'POTENTIAL_', 'THRUST_', 'RESONANCE_'))] # 从更新后的df中获取列
        self.strategy.atomic_states.update(df[dynamic_cols])
        all_behavioral_states.update(df[dynamic_cols])
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        - 废弃了基础的K线形态指标，全面转向使用更能反映主力意图和过程质量的微观结构信号。
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
        
        # 获取 default_weights 并传递给 get_adaptive_mtf_normalized_score
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        
        # 过程质量评估：融合“微观结构效率”和“脉冲质量”，评估日内走势的含金量
        micro_efficiency = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'microstructure_efficiency_index_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index, tf_weights=default_weights)
        impulse_quality = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'impulse_quality_ratio_D', pd.Series(0.0, index=df.index), method_name="_calculate_behavioral_day_quality"), df.index, tf_weights=default_weights)
        
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
        【V34.11 · 依赖编排与调试增强版】原子信号中心
        - 核心升级: 适配了 V5.0 "派发罪证链" 和 V3.0 "战略反击许可" 模型，
                      并调整了内部调用顺序以确保逻辑依赖的正确性。
        - 【修正】确保所有派生信号在被需要时已添加到df中。
        - 【新增】将鲁棒斜率计算提升到方法开头，解决循环依赖问题。
        - 【调试】增加详细打印，追踪df列的添加情况，包括df的内存地址和完整列列表。
        - 【重要修正】确保所有新生成的信号在添加到states字典的同时，也立即添加到传入的df中，以解决内部依赖问题。
        - 【命名修正】统一long_term_斜率的命名，使其与robust_斜率的命名风格一致。
        """
        method_name = "_diagnose_behavioral_axioms"
        # 获取所有参数配置，用于动态构建required_signals
        p_behavioral_div_conf = get_params_block(self.strategy, 'behavioral_divergence_params', {})
        
        # 健壮地获取 mtf_slopes_params，确保 'weights' 键存在
        mtf_slopes_params_from_config = p_behavioral_div_conf.get('multi_timeframe_slopes')
        # 定义一个完整的默认 mtf_slopes_params 结构
        default_mtf_slopes_config = {"enabled": True, "periods": [5, 13], "weights": {"5": 0.7, "13": 0.3}}

        if mtf_slopes_params_from_config is None:
            # 如果配置中没有 multi_timeframe_slopes，则使用完整的默认值
            mtf_slopes_params = default_mtf_slopes_config
        else:
            # 如果配置中有，则将其与默认值合并，确保所有键都存在
            mtf_slopes_params = {**default_mtf_slopes_config, **mtf_slopes_params_from_config}
            # 特别处理 'weights' 子字典，进行深度合并
            if 'weights' in mtf_slopes_params_from_config and isinstance(mtf_slopes_params_from_config['weights'], dict):
                mtf_slopes_params['weights'] = {**default_mtf_slopes_config['weights'], **mtf_slopes_params_from_config['weights']}
            elif 'weights' not in mtf_slopes_params_from_config:
                # 如果配置中没有 'weights' 键，则使用默认的 'weights'
                mtf_slopes_params['weights'] = default_mtf_slopes_config['weights']

        mtf_periods = mtf_slopes_params.get('periods', [5])
        multi_level_resonance_params = get_param_value(p_behavioral_div_conf.get('multi_level_resonance_params'), {"enabled": True, "long_term_period": 21, "resonance_bonus": 0.2})
        long_term_period = multi_level_resonance_params.get('long_term_period', 21)
        pattern_sequence_params = get_param_value(p_behavioral_div_conf.get('pattern_sequence_params'), {"enabled": True, "lookback_window": 3, "volume_drying_up_ratio": 0.8, "volume_climax_ratio": 1.5, "reversal_pct_change_threshold": 0.01, "sequence_bonus": 0.2})
        pattern_lookback_window = pattern_sequence_params.get('lookback_window', 3)
        accel_period = mtf_periods[0] # 使用最短MTF周期作为加速度周期
        # 重新构建required_signals，确保包含所有原始输入信号
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
            'sell_quote_exhaustion_rate_D', 'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D', 
            'microstructure_efficiency_index_D', 'upward_impulse_purity_D', 'vacuum_traversal_efficiency_D',
            'support_validation_strength_D', 'impulse_quality_ratio_D', 'floating_chip_cleansing_efficiency_D',
            'panic_selling_cascade_D', 'capitulation_absorption_index_D', 'covert_accumulation_signal_D',
            'VOL_MA_5_D', 'VOL_MA_13_D', 'VOL_MA_21_D', 'loser_pain_index_D',
            'wash_trade_intensity_D', 'closing_auction_ambush_D', 'mf_retail_battle_intensity_D',
            'main_force_conviction_index_D', 'SLOPE_5_loser_pain_index_D',
            'pressure_rejection_strength_D', 'active_buying_support_D', 'vwap_control_strength_D',
            'SLOPE_5_winner_stability_index_D',
            'winner_stability_index_D',
            'chip_fatigue_index_D',
            'main_force_net_flow_calibrated_D',
            'retail_fomo_premium_index_D',
            'BBP_21_2.0_D', 'BIAS_5_D',
            'ATR_14_D', 'BBW_21_2.0_D', 'ADX_14_D'
        ]
        # 动态添加MTF斜率信号到required_signals
        for period in mtf_periods:
            for indicator in ['close', 'RSI_13', 'MACDh_13_34_8', 'volume', 'BBW_21_2.0', 'pct_change']:
                required_signals.append(f'SLOPE_{period}_{indicator}_D')
        # 添加长期斜率信号
        for indicator in ['close', 'RSI_13', 'MACDh_13_34_8', 'volume', 'ADX_14']:
            required_signals.append(f'SLOPE_{long_term_period}_{indicator}_D')
        # 添加模式序列所需的斜率
        for indicator in ['close', 'volume']:
            required_signals.append(f'SLOPE_{pattern_lookback_window}_{indicator}_D')
        # 添加加速度信号
        for indicator in ['close', 'RSI_13', 'MACDh_13_34_8', 'volume']:
            required_signals.append(f'ACCEL_{accel_period}_{indicator}_D')
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [行为情报引擎] {method_name}: 核心公理诊断失败，缺少必要原始信号，行为分析中止。")
            return {}
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        # 修正键名 'default_weights' 为 'default'
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 修正键名 'long_term_weights' 为 'long_term_stability'
        long_term_weights = get_param_value(p_mtf.get('long_term_stability'), {'21': 0.5, '55': 0.3, '89': 0.2})
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        probe_ts = None
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            if valid_probe_dates:
                probe_ts = valid_probe_dates[0]
        # --- 基础信号计算 ---
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        price_accel = self._get_safe_series(df, 'ACCEL_5_pct_change_D', pct_change.diff(5).fillna(0.0), method_name=method_name)
        # 1. 计算鲁棒斜率 (Robust Slopes)
        robust_slopes = {}
        for indicator in ['close', 'RSI_13', 'MACDh_13_34_8', 'volume', 'BBW_21_2.0', 'pct_change']:
            weighted_slope = pd.Series(0.0, index=df.index)
            total_weight = 0.0
            for period in mtf_periods:
                col_name = f'SLOPE_{period}_{indicator}_D'
                weight = mtf_slopes_params['weights'].get(str(period), 0.0)
                if col_name in df.columns:
                    weighted_slope += self._get_safe_series(df, col_name, 0.0, method_name=method_name) * weight
                    total_weight += weight
                else:
                    if is_debug_enabled and probe_ts and probe_ts in df.index:
                        print(f"    -> [行为情报警告] {method_name}: 缺少MTF斜率数据 '{col_name}'，跳过该周期。")
            if total_weight > 0:
                robust_slopes[indicator] = weighted_slope / total_weight
            else:
                first_period_col = f'SLOPE_{mtf_periods[0]}_{indicator}_D' if mtf_periods else None
                robust_slopes[indicator] = self._get_safe_series(df, first_period_col, 0.0, method_name=method_name)
            # 将鲁棒斜率添加到df中，并添加到states中
            df[f'robust_{indicator}_slope'] = robust_slopes[indicator]
            states[f'robust_{indicator}_slope'] = robust_slopes[indicator]
        # 计算长期斜率 - 提升到这里
        long_term_period = multi_level_resonance_params.get('long_term_period', 21) # 确保long_term_period已定义
        long_term_close_slope = self._get_safe_series(df, f'SLOPE_{long_term_period}_close_D', 0.0, method_name=method_name)
        long_term_rsi_slope = self._get_safe_series(df, f'SLOPE_{long_term_period}_RSI_13_D', 0.0, method_name=method_name)
        long_term_macd_slope = self._get_safe_series(df, f'SLOPE_{long_term_period}_MACDh_13_34_8_D', 0.0, method_name=method_name)
        long_term_volume_slope = self._get_safe_series(df, f'SLOPE_{long_term_period}_volume_D', 0.0, method_name=method_name)
        long_term_adx_slope = self._get_safe_series(df, f'SLOPE_{long_term_period}_ADX_14_D', 0.0, method_name=method_name)
        # 将长期斜率添加到df中，并添加到states中，修正命名
        df['long_term_close_slope'] = long_term_close_slope
        states['long_term_close_slope'] = long_term_close_slope
        df['long_term_RSI_13_slope'] = long_term_rsi_slope # 修正列名
        states['long_term_RSI_13_slope'] = long_term_rsi_slope # 修正列名
        df['long_term_MACDh_13_34_8_slope'] = long_term_macd_slope # 修正列名
        states['long_term_MACDh_13_34_8_slope'] = long_term_macd_slope # 修正列名
        df['long_term_volume_slope'] = long_term_volume_slope
        states['long_term_volume_slope'] = long_term_volume_slope
        df['long_term_adx_slope'] = long_term_adx_slope
        states['long_term_adx_slope'] = long_term_adx_slope
        # 模式序列所需的斜率 - 提升到这里
        pattern_lookback_window = pattern_sequence_params.get('lookback_window', 3)
        pattern_close_slope = self._get_safe_series(df, f'SLOPE_{pattern_lookback_window}_close_D', 0.0, method_name=method_name)
        pattern_volume_slope = self._get_safe_series(df, f'SLOPE_{pattern_lookback_window}_volume_D', 0.0, method_name=method_name)
        # 将模式序列斜率添加到df中，并添加到states中
        df['pattern_close_slope'] = pattern_close_slope
        states['pattern_close_slope'] = pattern_close_slope
        df['pattern_volume_slope'] = pattern_volume_slope
        states['pattern_volume_slope'] = pattern_volume_slope
        # --- 动能信号 ---
        upward_momentum_score = self._diagnose_upward_momentum(df, default_weights)
        states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = upward_momentum_score.astype(np.float32)
        print(f"    -> [行为情校验] 计算“价格上涨动量(SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM)” 分数：{upward_momentum_score.mean():.4f}")
        df['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = upward_momentum_score.astype(np.float32)
        downward_momentum_score = self._diagnose_downward_momentum(df)
        states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = downward_momentum_score.astype(np.float32)
        df['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = downward_momentum_score.astype(np.float32)
        # --- 行为铁三角 (依赖于df中的信号，所以先计算并添加到df) ---
        upward_efficiency_score = self._diagnose_upward_efficiency(df, default_weights)
        states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = upward_efficiency_score.astype(np.float32)
        df['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = upward_efficiency_score.astype(np.float32)
        downward_resistance_score = self._diagnose_downward_resistance(df, default_weights)
        states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = downward_resistance_score.astype(np.float32)
        df['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = downward_resistance_score.astype(np.float32)
        intraday_bull_control_score = self._diagnose_intraday_bull_control(df, default_weights)
        states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = intraday_bull_control_score.astype(np.float32)
        df['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = intraday_bull_control_score.astype(np.float32)
        # --- 超买信号 (依赖于df中的信号，所以先计算并添加到df) ---
        # 调用重构后的纯行为超买信号
        final_overextension_score = self._calculate_behavioral_price_overextension(df, default_weights, is_debug_enabled, probe_ts)
        states['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = final_overextension_score.astype(np.float32)
        df['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = final_overextension_score.astype(np.float32)
        # --- 滞涨信号 (依赖于df中的信号，所以先计算并添加到df) ---
        # 调用重构后的纯行为滞涨信号
        stagnation_evidence = self._calculate_behavioral_stagnation_evidence(df, default_weights, is_debug_enabled, probe_ts)
        states['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'] = stagnation_evidence.astype(np.float32)
        df['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'] = stagnation_evidence.astype(np.float32)
        # --- 其他信号 ---
        lower_shadow_quality = self._diagnose_lower_shadow_quality(df)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_quality
        df['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_quality
        deception_index = self._diagnose_deception_index(df)
        states['SCORE_BEHAVIOR_DECEPTION_INDEX'] = deception_index
        df['SCORE_BEHAVIOR_DECEPTION_INDEX'] = deception_index
        distribution_intent = self._diagnose_distribution_intent(df, default_weights, final_overextension_score)
        states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'] = distribution_intent
        df['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'] = distribution_intent
        offensive_absorption_intent = self._diagnose_offensive_absorption_intent(df, lower_shadow_quality, distribution_intent)
        states['SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT'] = offensive_absorption_intent
        df['SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT'] = offensive_absorption_intent
        states['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK'] = self._diagnose_ambush_counterattack(df, offensive_absorption_intent)
        df['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK'] = states['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK']
        states['SCORE_RISK_BREAKOUT_FAILURE_CASCADE'] = self._diagnose_breakout_failure_risk(
            df,
            distribution_intent,
            final_overextension_score,
            deception_index,
            is_debug_enabled,
            probe_ts
        )
        df['SCORE_RISK_BREAKOUT_FAILURE_CASCADE'] = states['SCORE_RISK_BREAKOUT_FAILURE_CASCADE']
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = self._calculate_volume_burst_quality(df, default_weights)
        print(f"    -> [微观行为情报校验] 计算“成交量爆发(SCORE_BEHAVIOR_VOLUME_BURST)” 分数：{states['SCORE_BEHAVIOR_VOLUME_BURST'].mean():.4f}")
        df['SCORE_BEHAVIOR_VOLUME_BURST'] = states['SCORE_BEHAVIOR_VOLUME_BURST']
        states['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = self._calculate_volume_atrophy(df, default_weights)
        df['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = states['SCORE_BEHAVIOR_VOLUME_ATROPHY']
        states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'] = self._calculate_absorption_strength(df, default_weights)
        df['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'] = states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH']
        states['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION'] = self._diagnose_shakeout_confirmation(
            df,
            states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'],
            states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT']
        )
        df['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION'] = states['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION']
        # 调用新的纯行为背离诊断方法，严格遵循V1.0定义
        bullish_pure_div, bearish_pure_div = self._diagnose_pure_behavioral_divergence(
            df,
            default_weights,
            is_debug_enabled,
            probe_ts
        )
        states['SCORE_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_pure_div # 新增 V1.0 看涨背离信号
        df['SCORE_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_pure_div
        states['SCORE_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_pure_div # 新增 V1.0 看跌背离信号
        df['SCORE_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_pure_div
        # 保持对 _QUALITY 版本的调用，因为它们是不同的、更高级的信号
        bullish_divergence_quality, bearish_divergence_quality = self._diagnose_divergence_quality(
            df,
            states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'],
            states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT']
        )
        states['SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY'] = bullish_divergence_quality
        df['SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY'] = bullish_divergence_quality
        states['SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY'] = bearish_divergence_quality
        df['SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY'] = bearish_divergence_quality
        # --- 机会与风险信号 ---
        is_rising = (pct_change > 0).astype(float)
        is_falling = (pct_change < 0).astype(float)
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = (is_rising * states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] * states['SCORE_BEHAVIOR_VOLUME_ATROPHY']).pow(1/3).astype(np.float32)
        df['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = states['SCORE_OPPORTUNITY_LOCKUP_RALLY']
        capitulation_raw = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name=method_name)
        selling_deceleration_score = (1 - get_adaptive_mtf_normalized_score(price_accel.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)).clip(0, 1)
        capitulation_confirm_score = get_adaptive_mtf_normalized_score(capitulation_raw, df.index, ascending=True, tf_weights=default_weights)
        selling_exhaustion_score = (
            states['SCORE_BEHAVIOR_VOLUME_ATROPHY'].pow(0.3) *
            states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'].pow(0.3) *
            selling_deceleration_score.pow(0.2) *
            capitulation_confirm_score.pow(0.2)
        ).fillna(0.0)
        states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = (is_falling * selling_exhaustion_score).astype(np.float32)
        df['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION']
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (is_falling * states['SCORE_BEHAVIOR_VOLUME_BURST'] * states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM']).pow(1/2).astype(np.float32)
        df['SCORE_RISK_LIQUIDITY_DRAIN'] = states['SCORE_RISK_LIQUIDITY_DRAIN']
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        # 移除整个探针逻辑块，恢复生产状态
        return final_control_score.astype(np.float32)

    def _diagnose_deception_index(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.1 · Production Ready版】诊断博弈欺骗指数
        - 核心重构: 废弃V1.2“结果导向”模型，引入基于“认知失调”的全新诊断模型。
        - 核心逻辑: 欺骗分 = (主力真实意图向量 - K线表象剧本向量) * 证据放大器
                      直接量化“意图”与“表象”的背离程度，能识别更高明的欺骗形态。
        - 诊断三要素:
          1. 舞台剧本 (The Apparent Narrative): K线收盘位置讲述的故事 (`closing_strength_index_D`)。
          2. 幕后黑手 (The Hidden Intent): 主力真实的订单流意图 (`main_force_ofi_D`)。
          3. 作案工具 (The Deceptive Tools): 对倒、欺骗等行为 (`deception_lure_long_intensity_D`, `deception_lure_short_intensity_D`, `wash_trade_intensity_D`)，作为放大器。
        - 【调优】原始指标deception_index被拆分为deception_lure_long_intensity、deception_lure_short_intensity，本方法已更新以利用这两个更精细的指标。
        """
        method_name = "_diagnose_deception_index"
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('puppeteers_gambit_params'), {})
        k_amplifier = params.get('evidence_amplifier_k', 0.5)
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # --- 2. 获取原始数据 ---
        required_signals = [
            'closing_strength_index_D', 'main_force_ofi_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'wash_trade_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {required_signals}，返回默认值。")
            return pd.Series(0.0, index=df.index)
        narrative_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name=method_name)
        intent_raw = self._get_safe_series(df, 'main_force_ofi_D', 0.0, method_name=method_name)
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name)
        deception_lure_short_raw = self._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name)
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        # --- 3. 计算核心组件 ---
        # 组件一：舞台剧本向量
        # 修正 normalize_to_bipolar 的调用参数
        narrative_vector = normalize_to_bipolar(narrative_raw, 55)
        # 组件二：幕后意图向量
        intent_vector = get_adaptive_mtf_normalized_bipolar_score(intent_raw, df.index, default_weights)
        # 计算认知失调向量 (先计算，以便后续选择欺骗证据)
        cognitive_dissonance_vector = (intent_vector - narrative_vector) / 2
        # 组件三：证据放大器 (根据认知失调方向选择对应的欺骗强度)
        relevant_deception_intensity = pd.Series(0.0, index=df.index)
        # 如果认知失调为正 (主力意图看涨，K线表象偏弱 -> 看涨诱惑)
        relevant_deception_intensity = relevant_deception_intensity.mask(cognitive_dissonance_vector > 0, deception_lure_long_raw)
        # 如果认知失调为负 (主力意图看跌，K线表象偏强 -> 看跌诱惑)
        relevant_deception_intensity = relevant_deception_intensity.mask(cognitive_dissonance_vector < 0, deception_lure_short_raw)
        deception_evidence_score = get_adaptive_mtf_normalized_score((relevant_deception_intensity + wash_trade_raw).pow(0.5), df.index, ascending=True, tf_weights=default_weights)
        evidence_amplifier = 1 + k_amplifier * deception_evidence_score
        # --- 4. 计算认知失调并施加放大器 ---
        final_deception_index = (cognitive_dissonance_vector * evidence_amplifier).clip(-1, 1)
        print(f"    -> [行为情报调试] {method_name} 计算完成。")
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
        # 移除整个探针逻辑块，恢复生产状态
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
        # 移除整个探针逻辑块，恢复生产状态
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 修正键名 'long_term_weights' 为 'long_term_stability'
        long_term_weights = get_param_value(p_mtf.get('long_term_stability'), {'21': 0.5, '55': 0.3, '89': 0.2})
        price_breakthrough_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_context_new_high_strength").clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        ma_slope_score = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'SLOPE_5_EMA_55_D', pd.Series(0.0, index=df.index), method_name="_diagnose_context_new_high_strength"), df.index, ascending=True, tf_weights=default_weights)
        bias_health_score = 1 - get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'BIAS_55_D', pd.Series(0.0, index=df.index), method_name="_diagnose_context_new_high_strength").clip(lower=0), df.index, ascending=True, tf_weights=long_term_weights)
        new_high_strength = (price_breakthrough_score * ma_slope_score * bias_health_score).pow(1/3).fillna(0.0)
        return {'CONTEXT_NEW_HIGH_STRENGTH': new_high_strength.astype(np.float32)}

    def _resolve_pressure_absorption_dynamics(self, provisional_pressure: pd.Series, intent_diagnosis: pd.Series) -> Dict[str, pd.Series]:
        """
        【V3.3 · 情报校验加固版】压力-承接能量转化模型
        - 调用从 utils.py 导入的公共归一化工具。
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        【V2.1 · Production Ready版】微观结构意图诊断引擎
        - 核心重构: 废弃V1.3“静态快照”模型，引入“核心意图强度 × 环境适应性 × 行为一致性”的全新三维诊断框架。
        - 诊断三维度:
          1. 核心意图强度 (Core Intent Magnitude): 诊断主力订单流和挂单枯竭的原始意图。
          2. 环境适应性 (Environmental Adaptability): 根据市场波动率和流动性校准意图的显著性。
          3. 行为一致性 (Behavioral Coherence): 通过价格脉冲纯度、欺诈指数印证意图的真实性。
        - 数学模型: 最终微观意图 = 核心意图强度 × 环境适应性因子 × 行为一致性因子
        - 【调优】原始指标deception_index被拆分为deception_lure_long_intensity、deception_lure_short_intensity，本方法已更新以利用这两个更精细的指标。
        """
        method_name = "_diagnose_microstructure_intent"
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('fog_of_war_protocol_params'), {})
        core_intent_weights = get_param_value(params.get('core_intent_weights'), {"ofi": 0.6, "quote_exhaustion": 0.4})
        env_adapt_weights = get_param_value(params.get('environmental_adaptability_weights'), {"volatility_sensitivity": 0.5, "liquidity_sensitivity": 0.5})
        behavior_coherence_weights = get_param_value(params.get('behavioral_coherence_weights'), {"impulse_purity": 0.7, "deception_penalty": 0.3})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        # 修正键名 'default_weights' 为 'default'，并从 p_mtf 获取
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # --- 2. 维度一：核心意图强度 (Core Intent Magnitude) ---
        required_signals_core = ['main_force_ofi_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D']
        if not self._validate_required_signals(df, required_signals_core, method_name):
            print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {required_signals_core}，返回默认值。")
            return {'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT': pd.Series(0.0, index=df.index)}
        ofi_raw = self._get_safe_series(df, 'main_force_ofi_D', 0.0, method_name=method_name)
        buy_sweep_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name=method_name)
        sell_sweep_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', 0.0, method_name=method_name)
        ofi_score = get_adaptive_mtf_normalized_bipolar_score(ofi_raw, df.index, default_weights)
        buy_sweep_score = get_adaptive_mtf_normalized_score(buy_sweep_raw, df.index, ascending=True, tf_weights=default_weights)
        sell_sweep_score = get_adaptive_mtf_normalized_score(sell_sweep_raw, df.index, ascending=True, tf_weights=default_weights)
        bullish_core_intent = (ofi_score.clip(lower=0) * core_intent_weights.get('ofi', 0.6) + buy_sweep_score * core_intent_weights.get('quote_exhaustion', 0.4))
        bearish_core_intent = (ofi_score.clip(upper=0).abs() * core_intent_weights.get('ofi', 0.6) + sell_sweep_score * core_intent_weights.get('quote_exhaustion', 0.4))
        core_intent_magnitude = (bullish_core_intent - bearish_core_intent).clip(-1, 1)
        # --- 3. 维度二：环境适应性 (Environmental Adaptability) ---
        required_signals_env = ['ATR_14_D', 'volume_ratio_D']
        environmental_adaptability_factor = pd.Series(1.0, index=df.index)
        if self._validate_required_signals(df, required_signals_env, method_name):
            atr_raw = self._get_safe_series(df, 'ATR_14_D', 0.0, method_name=method_name)
            volume_ratio_raw = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name=method_name)
            volatility_score = get_adaptive_mtf_normalized_score(atr_raw, df.index, ascending=True, tf_weights=default_weights)
            liquidity_score = (1 - get_adaptive_mtf_normalized_score(volume_ratio_raw, df.index, ascending=True, tf_weights=default_weights))
            environmental_adaptability_factor_raw = (
                volatility_score * env_adapt_weights.get('volatility_sensitivity', 0.5) +
                liquidity_score * env_adapt_weights.get('liquidity_sensitivity', 0.5)
            ).clip(0, 1)
            environmental_adaptability_factor = 0.5 + environmental_adaptability_factor_raw * 0.5 # 映射到 [0.5, 1.0]
        # --- 4. 维度三：行为一致性 (Behavioral Coherence) ---
        required_signals_behavior = ['upward_impulse_purity_D', 'vacuum_traversal_efficiency_D', 'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D'] # 修改行
        behavioral_coherence_factor = pd.Series(1.0, index=df.index)
        if self._validate_required_signals(df, required_signals_behavior, method_name):
            upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
            downward_purity_raw = self._get_safe_series(df, 'vacuum_traversal_efficiency_D', 0.0, method_name=method_name)
            deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name) # 新增行
            deception_lure_short_raw = self._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name) # 新增行
            # 综合欺骗强度，取两者最大值作为惩罚因子 # 新增行
            deception_raw = pd.concat([deception_lure_long_raw, deception_lure_short_raw], axis=1).max(axis=1) # 新增行
            upward_purity_score = get_adaptive_mtf_normalized_score(upward_purity_raw, df.index, ascending=True, tf_weights=default_weights)
            downward_purity_score = get_adaptive_mtf_normalized_score(downward_purity_raw, df.index, ascending=True, tf_weights=default_weights)
            deception_score = get_adaptive_mtf_normalized_score(deception_raw, df.index, ascending=True, tf_weights=default_weights) # 修改行
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
        print(f"    -> [行为情报调试] {method_name} 计算完成。")
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        # 移除探针代码，恢复生产版本
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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        - 升级说明: 增加了详细探针，用于调试和检查每一步计算。
        """
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('atmospheric_pressure_params'), {})
        synergy_bonus = get_param_value(params.get('synergy_bonus_factor'), 0.2)
        env_params = get_param_value(p_conf.get('judgment_day_protocol_params'), {})
        env_weights = get_param_value(env_params.get('environment_weights'), {'fatigue': 0.4, 'decay': 0.3, 'betrayal': 0.3})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # --- 2. 轨道一：战术风险评估 (风暴强度) ---
        motive_raw = self._get_safe_series(df, 'profit_taking_flow_ratio_D', 0.0, method_name="_diagnose_distribution_intent")
        rally_pressure_raw = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name="_diagnose_distribution_intent")
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_diagnose_distribution_intent")
        fingerprint_raw = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_distribution_intent")
        motive_score = get_adaptive_mtf_normalized_score(motive_raw, df.index, ascending=True, tf_weights=default_weights)
        rally_pressure_score = get_adaptive_mtf_normalized_score(rally_pressure_raw, df.index, ascending=True, tf_weights=default_weights)
        upper_shadow_score = get_adaptive_mtf_normalized_score(upper_shadow_pressure_raw, df.index, ascending=True, tf_weights=default_weights)
        weapon_score = (rally_pressure_score * 0.5 + upper_shadow_score * 0.5)
        fingerprint_score = get_adaptive_mtf_normalized_score(fingerprint_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
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
        vitality_score = get_adaptive_mtf_normalized_score(vitality_raw, df.index, ascending=True, tf_weights=default_weights)
        bullish_fatigue_score = ((1 - vitality_score) * overextension_raw).pow(0.5)
        stability_score = get_adaptive_mtf_normalized_score(winner_stability_raw, df.index, ascending=True, tf_weights=default_weights)
        solidity_score = get_adaptive_mtf_normalized_score(control_solidity_raw, df.index, ascending=True, tf_weights=default_weights)
        fortress_decay_score = ((1 - stability_score) * (1 - solidity_score)).pow(0.5)
        commanders_betrayal_score = get_adaptive_mtf_normalized_score(conviction_slope_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        strategic_risk_score = (
            bullish_fatigue_score * env_weights.get('fatigue', 0.4) +
            fortress_decay_score * env_weights.get('decay', 0.3) +
            commanders_betrayal_score * env_weights.get('betrayal', 0.3)
        ).clip(0, 1)
        # --- 4. 最终合成：双轨独立审判 ---
        base_risk = pd.concat([tactical_risk_score, strategic_risk_score], axis=1).max(axis=1)
        synergy_amplifier = 1 + (tactical_risk_score * strategic_risk_score).pow(0.5) * synergy_bonus
        final_distribution_intent = (base_risk * synergy_amplifier).clip(0, 1)
        return final_distribution_intent.astype(np.float32)

    def _diagnose_ambush_counterattack(self, df: pd.DataFrame, offensive_absorption_intent: pd.Series) -> pd.Series:
        """
        【V5.1 · 诡道反击协议】诊断伏击式反攻信号。
        - 核心重构: 引入“脆弱战场 × 幽灵诡计 × 突袭品质”的全新三维诊断框架。
        - 诊断三维度:
          1. 脆弱战场 (Vulnerable Battlefield): 评估市场先前的脆弱性（恐慌、短期下跌趋势/价格停滞、亏损盘痛苦）。
          2. 幽灵诡计 (Phantom Trick): 评估主力承接的强度及其看涨欺骗性（制造弱势假象）。
          3. 突袭品质 (Strike Quality): 评估反攻的有效性与纯粹性（收盘强度、上涨脉冲纯度）。
        - 数学模型: 伏击反攻分 = (脆弱战场分^W1 * 幽灵诡计分^W2 * 突袭品质分^W3)
        - 【调优】原始指标deception_index被拆分为deception_lure_long_intensity、deception_lure_short_intensity，本方法已更新以利用这两个更精细的指标。
        """
        method_name = "_diagnose_ambush_counterattack"
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('ambush_counterattack_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"context": 0.3, "action": 0.4, "quality": 0.3})
        context_weights = get_param_value(params.get('context_weights'), {"panic": 0.3, "prior_weakness_slope": 0.4, "loser_pain": 0.2, "price_stagnation": 0.1})
        prior_weakness_slope_window = get_param_value(params.get('prior_weakness_slope_window'), 5)
        price_stagnation_params = get_param_value(params.get('price_stagnation_params'), {"slope_window": 5, "max_abs_slope_threshold": 0.005, "max_bbw_score_threshold": 0.3})
        action_weights = get_param_value(params.get('action_weights'), {"absorption": 0.6, "deception_positive": 0.4})
        quality_weights = get_param_value(params.get('quality_weights'), {"closing_strength": 0.6, "upward_purity": 0.4})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # --- 2. 获取所有原始数据 ---
        required_signals = [
            'panic_selling_cascade_D', f'SLOPE_{prior_weakness_slope_window}_close_D', 'loser_pain_index_D',
            'closing_strength_index_D', 'upward_impulse_purity_D',
            f'SLOPE_{price_stagnation_params.get("slope_window", 5)}_close_D', 'BBW_21_2.0_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {required_signals}，返回默认值。")
            return pd.Series(0.0, index=df.index)
        panic_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name=method_name)
        prior_weakness_slope_raw = self._get_safe_series(df, f'SLOPE_{prior_weakness_slope_window}_close_D', 0.0, method_name=method_name)
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name=method_name)
        deception_raw = self._get_safe_series(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0, method_name=method_name) # 修改行: 使用 _diagnose_deception_index 的输出
        closing_strength_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name=method_name)
        upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        stagnation_slope_raw = self._get_safe_series(df, f'SLOPE_{price_stagnation_params.get("slope_window", 5)}_close_D', 0.0, method_name=method_name)
        bbw_raw = self._get_safe_series(df, 'BBW_21_2.0_D', 0.0, method_name=method_name)
        # --- 3. 维度一：脆弱战场 (Vulnerable Battlefield) ---
        panic_score = get_adaptive_mtf_normalized_score(panic_raw, df.index, ascending=True, tf_weights=default_weights)
        prior_weakness_score = get_adaptive_mtf_normalized_score(prior_weakness_slope_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        loser_pain_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df.index, ascending=True, tf_weights=default_weights)
        bbw_score = get_adaptive_mtf_normalized_score(bbw_raw, df.index, ascending=False, tf_weights=default_weights)
        price_stagnation_score = pd.Series(0.0, index=df.index)
        price_stagnation_condition = (stagnation_slope_raw.abs() < price_stagnation_params.get("max_abs_slope_threshold", 0.005)) & \
                                     (bbw_score > (1 - price_stagnation_params.get("max_bbw_score_threshold", 0.3)))
        price_stagnation_score[price_stagnation_condition] = (bbw_score[price_stagnation_condition] * (1 - stagnation_slope_raw.abs()[price_stagnation_condition] / price_stagnation_params.get("max_abs_slope_threshold", 0.005))).clip(0,1)
        ambush_context_score = (
            (panic_score + 1e-9).pow(context_weights.get('panic', 0.3)) *
            (prior_weakness_score + 1e-9).pow(context_weights.get('prior_weakness_slope', 0.4)) *
            (loser_pain_score + 1e-9).pow(context_weights.get('loser_pain', 0.2)) *
            (price_stagnation_score + 1e-9).pow(context_weights.get('price_stagnation', 0.1))
        ).pow(1/(context_weights.get('panic', 0.3) + context_weights.get('prior_weakness_slope', 0.4) + context_weights.get('loser_pain', 0.2) + context_weights.get('price_stagnation', 0.1))).fillna(0.0)
        # --- 4. 维度二：幽灵诡计 (Phantom Trick) ---
        absorption_score = offensive_absorption_intent
        # 对于看涨的伏击反攻，我们关注的是“诱空”或“制造弱势假象”的欺骗，这对应于deception_index的负向部分 # 修改行
        deceptive_narrative_score = get_adaptive_mtf_normalized_score(deception_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights) # 修改行
        deceptive_action_score = (
            (absorption_score + 1e-9).pow(action_weights.get('absorption', 0.6)) *
            (deceptive_narrative_score + 1e-9).pow(action_weights.get('deception_positive', 0.4))
        ).pow(1/(action_weights.get('absorption', 0.6) + action_weights.get('deception_positive', 0.4))).fillna(0.0)
        # --- 5. 维度三：突袭品质 (Strike Quality) ---
        # 修正 normalize_score 的调用参数
        closing_strength_score = normalize_score(closing_strength_raw, 55)
        upward_purity_score = get_adaptive_mtf_normalized_score(upward_purity_raw, df.index, ascending=True, tf_weights=default_weights)
        counterattack_quality_score = (
            (closing_strength_score + 1e-9).pow(quality_weights.get('closing_strength', 0.6)) *
            (upward_purity_score + 1e-9).pow(quality_weights.get('upward_purity', 0.4))
        ).pow(1/(quality_weights.get('closing_strength', 0.6) + quality_weights.get('upward_purity', 0.4))).fillna(0.0)
        # --- 6. 最终合成：三维融合 ---
        ambush_counterattack_score = (
            (ambush_context_score + 1e-9).pow(fusion_weights.get('context', 0.3)) *
            (deceptive_action_score + 1e-9).pow(fusion_weights.get('action', 0.4)) *
            (counterattack_quality_score + 1e-9).pow(fusion_weights.get('quality', 0.3))
        ).pow(1/(fusion_weights.get('context', 0.3) + fusion_weights.get('action', 0.4) + fusion_weights.get('quality', 0.3))).fillna(0.0)
        print(f"    -> [行为情报调试] {method_name} 计算完成。")
        return ambush_counterattack_score.clip(0, 1).astype(np.float32)

    def _diagnose_breakout_failure_risk(self, df: pd.DataFrame, distribution_intent: pd.Series, overextension_score_series: pd.Series, deception_index_series: pd.Series, debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        【V5.3 · 行为模式精微化版】诊断突破失败级联风险
        - 核心重构: 废弃了基于简单价格比较的“机械式突破谬误”模型。引入基于“诱多-伏击-情境”
                      诡道剧本的全新三维诊断模型，旨在精确识别高迷惑性的“牛市陷阱”。
        - 行为情报核心聚焦: 严格遵循“只分析行为类原始数据”的原则。本模块不再分析原始筹码/资金流信号，
                              也不依赖其他情报层的高阶融合信号。原有的“套牢盘痛苦度”和“主力背弃度”维度
                              因其本质依赖筹码/资金流数据，已从本方法中移除，以确保行为情报的纯粹性。
                              本信号现在专注于纯粹的市场行为模式。
        """
        method_name = "_diagnose_breakout_failure_risk"
        required_signals = [
            'breakout_quality_score_D', 'retail_fomo_premium_index_D', 'trend_vitality_index_D',
            'active_buying_support_D', 'upward_impulse_purity_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'RSI_13_D', 'SLOPE_5_RSI_13_D', 'BIAS_55_D', 'SLOPE_5_close_D' # 用于计算行为动量背离
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        breakout_params = get_param_value(p_conf.get('breakout_failure_risk_params'), {})
        core_risk_weights = get_param_value(breakout_params.get('core_risk_weights'), {"ambush": 1.0})
        context_amplifier_weights = get_param_value(breakout_params.get('context_amplifier_weights'), {"overextension": 0.4, "positive_deception": 0.3, "retail_fomo": 0.3})
        max_amplification_factor = get_param_value(breakout_params.get('max_amplification_factor'), 0.5)
        lure_weakness_multiplier = get_param_value(breakout_params.get('lure_weakness_multiplier'), 0.5)
        ambush_fusion_weights = get_param_value(breakout_params.get('ambush_fusion_weights'), {"distribution_intent": 0.7, "covert_ambush_intent": 0.3})
        covert_ambush_intent_weights = get_param_value(breakout_params.get('covert_ambush_intent_weights'), {"weak_buying_support": 0.6, "declining_impulse_purity": 0.4})
        deceptive_calm_weights = get_param_value(breakout_params.get('deceptive_calm_weights'), {"overextension_inverse": 0.3, "positive_deception_inverse": 0.3, "retail_fomo_inverse": 0.4})
        deceptive_calm_multiplier = get_param_value(breakout_params.get('deceptive_calm_multiplier'), 0.2)
        deceptive_calm_threshold = get_param_value(breakout_params.get('deceptive_calm_threshold'), 0.5)
        base_dynamic_risk_weight_exponent = get_param_value(breakout_params.get('base_dynamic_risk_weight_exponent'), 1.5)
        volatility_exponent_multiplier = get_param_value(breakout_params.get('volatility_exponent_multiplier'), 0.5)
        trend_vitality_exponent_multiplier = get_param_value(breakout_params.get('trend_vitality_exponent_multiplier'), 0.5)
        core_risk_synergy_exponent = get_param_value(breakout_params.get('core_risk_synergy_exponent'), 2.0)
        core_risk_high_end_stretch_power = get_param_value(breakout_params.get('core_risk_high_end_stretch_power'), 2.0)
        risk_ema_span = get_param_value(breakout_params.get('risk_ema_span'), 5)
        risk_trend_slope_window = get_param_value(breakout_params.get('risk_trend_slope_window'), 5)
        risk_momentum_diff_window = get_param_value(breakout_params.get('risk_momentum_diff_window'), 1)
        risk_trend_mod_multiplier = get_param_value(breakout_params.get('risk_trend_mod_multiplier'), 0.2)
        risk_momentum_mod_multiplier = get_param_value(breakout_params.get('risk_momentum_mod_multiplier'), 0.1)
        # 新增行为动量背离和行为情绪极端信号的权重
        behavioral_momentum_divergence_weights = get_param_value(breakout_params.get('behavioral_momentum_divergence_weights'), {"price_slope_weight": 0.5, "rsi_slope_weight": 0.5})
        behavioral_sentiment_extreme_weights = get_param_value(breakout_params.get('behavioral_sentiment_extreme_weights'), {"retail_fomo": 0.4, "rsi_extreme": 0.3, "bias_extreme": 0.3})
        # --- 1. 获取核心战术要素的原始数据 ---
        # 诱饵 (The Lure)
        breakout_quality_raw = self._get_safe_series(df, 'breakout_quality_score_D', 0.0, method_name=method_name)
        # 伏击 (The Ambush) - 直接使用传入的 distribution_intent
        # 情境放大器 (Contextual Amplifier) - 从参数获取
        overextension_score = overextension_score_series
        deception_raw = deception_index_series
        retail_fomo_raw = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name=method_name)
        trend_vitality_raw = self._get_safe_series(df, 'trend_vitality_index_D', 0.5, method_name=method_name)
        active_buying_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        # 获取计算新信号所需的原始行为数据
        close_slope_raw = self._get_safe_series(df, 'SLOPE_5_close_D', 0.0, method_name=method_name)
        rsi_raw = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name=method_name)
        rsi_slope_raw = self._get_safe_series(df, 'SLOPE_5_RSI_13_D', 0.0, method_name=method_name)
        bias_raw = self._get_safe_series(df, 'BIAS_55_D', 0.0, method_name=method_name)
        # --- 2. 计算各要素得分 ---
        lure_score = get_adaptive_mtf_normalized_score(breakout_quality_raw, df.index, ascending=True, tf_weights=default_weights)
        market_weakness_score = get_adaptive_mtf_normalized_score(trend_vitality_raw, df.index, ascending=False, tf_weights=default_weights)
        lure_score_modulated = (lure_score * (1 + market_weakness_score * lure_weakness_multiplier)).clip(0, 1)
        weak_buying_support_score = (1 - get_adaptive_mtf_normalized_score(active_buying_raw, df.index, ascending=True, tf_weights=default_weights)).clip(0,1)
        declining_impulse_purity_score = (1 - get_adaptive_mtf_normalized_score(upward_purity_raw, df.index, ascending=True, tf_weights=default_weights)).clip(0,1)
        covert_ambush_intent_score = (
            weak_buying_support_score * covert_ambush_intent_weights.get('weak_buying_support', 0.6) +
            declining_impulse_purity_score * covert_ambush_intent_weights.get('declining_impulse_purity', 0.4)
        ).clip(0,1)
        # 计算行为动量背离信号
        behavioral_momentum_divergence_raw = pd.Series(0.0, index=df.index)
        # 价格上涨 (斜率 > 0) 且 RSI 动量下降 (斜率 < 0)
        bullish_divergence_mask = (close_slope_raw > 0) & (rsi_slope_raw < 0)
        # 背离强度 = 价格上涨斜率的绝对值 + RSI下降斜率的绝对值
        behavioral_momentum_divergence_raw.loc[bullish_divergence_mask] = \
            close_slope_raw.loc[bullish_divergence_mask].abs() * behavioral_momentum_divergence_weights.get('price_slope_weight', 0.5) + \
            rsi_slope_raw.loc[bullish_divergence_mask].abs() * behavioral_momentum_divergence_weights.get('rsi_slope_weight', 0.5)
        behavioral_momentum_divergence_score = get_adaptive_mtf_normalized_score(behavioral_momentum_divergence_raw, df.index, ascending=True, tf_weights=default_weights)
        # 将行为动量背离融入 ambush_score
        ambush_fusion_weights['behavioral_momentum_divergence'] = ambush_fusion_weights.get('behavioral_momentum_divergence', 0.1) # 确保键存在
        ambush_score = (
            distribution_intent * ambush_fusion_weights.get('distribution_intent', 0.7) +
            covert_ambush_intent_score * ambush_fusion_weights.get('covert_ambush_intent', 0.2) +
            behavioral_momentum_divergence_score * ambush_fusion_weights.get('behavioral_momentum_divergence', 0.1)
        ).clip(0,1)
        trapped_force_score = pd.Series(0.0, index=df.index) # 设为0，不再参与计算
        mf_abandonment_score = pd.Series(0.0, index=df.index) # 设为0，不再参与计算
        positive_deception_score = deception_raw.clip(lower=0)
        retail_fomo_raw = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name=method_name) # 确保 retail_fomo_raw 已定义
        retail_fomo_score = get_adaptive_mtf_normalized_score(retail_fomo_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        # 计算行为情绪极端信号
        norm_retail_fomo_extreme = get_adaptive_mtf_normalized_score(retail_fomo_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        norm_rsi_extreme = get_adaptive_mtf_normalized_score(rsi_raw.clip(70, 100), df.index, ascending=True, tf_weights=default_weights) # RSI > 70 视为极端
        norm_bias_extreme = get_adaptive_mtf_normalized_score(bias_raw.clip(0.1, 1.0), df.index, ascending=True, tf_weights=default_weights) # BIAS > 0.1 视为极端
        behavioral_sentiment_extreme_score = (
            norm_retail_fomo_extreme * behavioral_sentiment_extreme_weights.get('retail_fomo', 0.4) +
            norm_rsi_extreme * behavioral_sentiment_extreme_weights.get('rsi_extreme', 0.3) +
            norm_bias_extreme * behavioral_sentiment_extreme_weights.get('bias_extreme', 0.3)
        ).clip(0,1)
        # 将行为情绪极端融入 context_amplifier_factor
        context_amplifier_weights['behavioral_sentiment_extreme'] = context_amplifier_weights.get('behavioral_sentiment_extreme', 0.3) # 确保键存在
        context_amplifier_factor = (
            overextension_score * context_amplifier_weights.get('overextension', 0.3) +
            positive_deception_score * context_amplifier_weights.get('positive_deception', 0.2) +
            retail_fomo_score * context_amplifier_weights.get('retail_fomo', 0.2) +
            behavioral_sentiment_extreme_score * context_amplifier_weights.get('behavioral_sentiment_extreme', 0.3)
        ).clip(0, 1)
        # --- 3. 核心风险基准分合成 ---
        normalized_volatility = get_adaptive_mtf_normalized_score(volatility_instability_raw, df.index, ascending=True, tf_weights=default_weights)
        normalized_inverse_trend_vitality = (1 - get_adaptive_mtf_normalized_score(trend_vitality_raw, df.index, ascending=True, tf_weights=default_weights)).clip(0,1)
        adaptive_dynamic_risk_weight_exponent = (
            base_dynamic_risk_weight_exponent +
            normalized_volatility * volatility_exponent_multiplier +
            normalized_inverse_trend_vitality * trend_vitality_exponent_multiplier
        ).clip(1.0, 3.0)
        dynamic_ambush_contribution = (ambush_score.pow(adaptive_dynamic_risk_weight_exponent)) * core_risk_weights.get('ambush', 1.0)
        total_dynamic_contribution = dynamic_ambush_contribution
        dynamic_ambush_weight = pd.Series(1.0, index=df.index)
        ambush_score_pow = (ambush_score + 1e-9).pow(core_risk_synergy_exponent)
        weighted_avg_risk = (dynamic_ambush_weight * ambush_score_pow).pow(1 / core_risk_synergy_exponent).fillna(0.0)
        stretched_weighted_avg_risk = (1 - (1 - weighted_avg_risk).pow(core_risk_high_end_stretch_power)).clip(0,1)
        core_risk_base_initial = (lure_score_modulated * stretched_weighted_avg_risk).clip(0,1).fillna(0.0)
        deceptive_calm_weights['overextension_inverse'] = deceptive_calm_weights.get('overextension_inverse', 0.3) # 确保键存在
        deceptive_calm_weights['positive_deception_inverse'] = deceptive_calm_weights.get('positive_deception_inverse', 0.3) # 确保键存在
        deceptive_calm_weights['retail_fomo_inverse'] = deceptive_calm_weights.get('retail_fomo_inverse', 0.4) # 确保键存在
        deceptive_calm_score = (
            (1 - overextension_score) * deceptive_calm_weights.get('overextension_inverse', 0.3) +
            (1 - positive_deception_score) * deceptive_calm_weights.get('positive_deception_inverse', 0.3) +
            (1 - retail_fomo_score) * deceptive_calm_weights.get('retail_fomo_inverse', 0.4)
        ).clip(0,1)
        deceptive_calm_effect = deceptive_calm_score * deceptive_calm_multiplier * (core_risk_base_initial > deceptive_calm_threshold).astype(float)
        final_amplifier = 1 + (context_amplifier_factor * max_amplification_factor) + deceptive_calm_effect
        # 计算风险动态调制因子
        if len(core_risk_base_initial) < max(risk_ema_span, risk_trend_slope_window, risk_momentum_diff_window) + 1:
            risk_dynamic_modulator = pd.Series(1.0, index=df.index)
            risk_ema = pd.Series(0.0, index=df.index)
            risk_trend = pd.Series(0.0, index=df.index)
            risk_momentum = pd.Series(0.0, index=df.index)
            risk_trend_score = pd.Series(0.0, index=df.index)
            risk_momentum_score = pd.Series(0.0, index=df.index)
        else:
            risk_ema = core_risk_base_initial.ewm(span=risk_ema_span, adjust=False).mean()
            risk_trend = risk_ema.diff(risk_trend_slope_window).fillna(0.0)
            risk_momentum = risk_trend.diff(risk_momentum_diff_window).fillna(0.0)
            risk_trend_score = get_adaptive_mtf_normalized_bipolar_score(risk_trend, df.index, default_weights)
            risk_momentum_score = get_adaptive_mtf_normalized_bipolar_score(risk_momentum, df.index, default_weights)
            risk_dynamic_modulator = (
                1 +
                (risk_trend_score * risk_trend_mod_multiplier) +
                (risk_momentum_score * risk_momentum_mod_multiplier)
            ).clip(0.5, 1.5)
        # --- 4. 最终风险合成 ---
        breakout_failure_risk = (core_risk_base_initial * final_amplifier * risk_dynamic_modulator).clip(0, 1)
        return breakout_failure_risk.astype(np.float32)

    def _diagnose_divergence_quality(self, df: pd.DataFrame, absorption_strength: pd.Series, distribution_intent: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        【V5.1 · Production Ready版】诊断高品质价量/价资背离
        - 核心重构: 废弃V4.0“宏观趋势分析”模型，引入“背离深度与广度 × 战略位置 × 双重确认 × 欺骗叙事确认”的全新四维诊断框架。
        - 诊断四维度:
          1. 背离深度与广度 (Divergence Depth & Breadth): 使用斜率更鲁棒地检测价格趋势和主力信念趋势。
          2. 战略位置 (Strategic Location): 评估背离是否发生在绝望区或获利盘不稳定区。
          3. 双重确认 (Dual Confirmation): 由“主力承接/派发”和“微观意图”进行双重印证。
          4. 欺骗叙事确认 (Deceptive Narrative Confirmation): 引入欺骗指数的负向部分，捕捉诱多本质。
        - 数学模型: 品质分 = (背离深度与广度分^0.4 * 战略位置分^0.3 * 主力承接/派发确认分^0.2 * 微观意图确认分^0.1 * 欺骗叙事确认分^0.1)
        - 【调优】原始指标deception_index被拆分为deception_lure_long_intensity、deception_lure_short_intensity，本方法已更新以利用这两个更精细的指标。
        """
        method_name = "_diagnose_divergence_quality"
        # --- 1. 获取参数 ---
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        params = get_param_value(p_conf.get('deceptive_divergence_protocol_params'), {})
        bullish_magnitude_params = get_param_value(params.get('bullish_magnitude_params'), {"price_downtrend_slope_window": 5, "conviction_uptrend_slope_window": 5})
        bearish_magnitude_params = get_param_value(params.get('bearish_magnitude_params'), {"price_slope_window": 5, "conviction_downtrend_slope_window": 5})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"magnitude": 0.4, "location": 0.3, "bullish_absorption_confirmation": 0.2, "bearish_distribution_confirmation": 0.2, "micro_intent_confirmation": 0.1, "deceptive_narrative_confirmation": 0.1})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # --- 2. 获取所有原始数据 ---
        required_signals = [
            'close_D', 'main_force_conviction_index_D', 'loser_pain_index_D', 'winner_stability_index_D',
            f'SLOPE_{bullish_magnitude_params.get("price_downtrend_slope_window", 5)}_close_D',
            f'SLOPE_{bullish_magnitude_params.get("conviction_uptrend_slope_window", 5)}_main_force_conviction_index_D',
            f'SLOPE_{bearish_magnitude_params.get("price_slope_window", 5)}_close_D',
            f'SLOPE_{bearish_magnitude_params.get("conviction_downtrend_slope_window", 5)}_main_force_conviction_index_D',
            'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {required_signals}，返回默认值。")
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        price = self._get_safe_series(df, 'close_D', 0.0, method_name=method_name)
        conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name=method_name)
        winner_stability_raw = self._get_safe_series(df, 'winner_stability_index_D', 0.5, method_name=method_name)
        bullish_price_slope_raw = self._get_safe_series(df, f'SLOPE_{bullish_magnitude_params.get("price_downtrend_slope_window", 5)}_close_D', 0.0, method_name=method_name)
        bullish_conviction_slope_raw = self._get_safe_series(df, f'SLOPE_{bullish_magnitude_params.get("conviction_uptrend_slope_window", 5)}_main_force_conviction_index_D', 0.0, method_name=method_name)
        bearish_price_slope_raw = self._get_safe_series(df, f'SLOPE_{bearish_magnitude_params.get("price_slope_window", 5)}_close_D', 0.0, method_name=method_name)
        bearish_conviction_slope_raw = self._get_safe_series(df, f'SLOPE_{bearish_magnitude_params.get("conviction_downtrend_slope_window", 5)}_main_force_conviction_index_D', 0.0, method_name=method_name)
        micro_intent_raw = self._get_safe_series(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0, method_name=method_name)
        deception_raw = self._get_safe_series(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0, method_name=method_name) # 修改行: 使用 _diagnose_deception_index 的输出
        # --- 3. 计算牛市背离 (价格下跌趋势 vs 信念上升趋势) ---
        # 维度一：背离深度与广度 (Magnitude)
        price_downtrend_score = get_adaptive_mtf_normalized_score(bullish_price_slope_raw.clip(upper=0).abs(), df.index, ascending=True, tf_weights=default_weights)
        conviction_uptrend_score = get_adaptive_mtf_normalized_score(bullish_conviction_slope_raw.clip(lower=0), df.index, ascending=True, tf_weights=default_weights)
        bullish_magnitude_score = (price_downtrend_score * conviction_uptrend_score).pow(0.5).fillna(0.0)
        # 维度二：战略位置 (Location)
        bullish_location_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df.index, ascending=True, tf_weights=default_weights)
        # 维度三：双重确认 (Dual Confirmation)
        bullish_absorption_confirmation_score = absorption_strength
        bullish_micro_intent_confirmation_score = micro_intent_raw.clip(lower=0)
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
        winner_instability_raw = 1 - winner_stability_raw
        bearish_location_score = get_adaptive_mtf_normalized_score(winner_instability_raw, df.index, ascending=True, tf_weights=default_weights)
        # 维度三：三重确认 (Triple Confirmation)
        bearish_distribution_confirmation_score = distribution_intent
        bearish_micro_intent_confirmation_score = micro_intent_raw.clip(upper=0).abs()
        # 对于看跌背离，我们关注的是“诱多”或“制造强势假象”的欺骗，这对应于deception_index的负向部分 # 修改行
        deceptive_narrative_confirmation_score = get_adaptive_mtf_normalized_score(
            deception_raw.clip(upper=0).abs(), # 修改行
            df.index,
            ascending=True,
            tf_weights=default_weights
        )
        # --- 6. 熊市背离品质合成 ---
        bearish_divergence_quality = (
            (bearish_magnitude_score + 1e-9).pow(fusion_weights.get('magnitude', 0.4)) *
            (bearish_location_score + 1e-9).pow(fusion_weights.get('location', 0.3)) *
            (bearish_distribution_confirmation_score + 1e-9).pow(fusion_weights.get('bearish_distribution_confirmation', 0.2)) *
            (bearish_micro_intent_confirmation_score + 1e-9).pow(fusion_weights.get('micro_intent_confirmation', 0.1)) *
            (deceptive_narrative_confirmation_score + 1e-9).pow(fusion_weights.get('deceptive_narrative_confirmation', 0.1))
        ).fillna(0.0)
        print(f"    -> [行为情报调试] {method_name} 计算完成。")
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
        # 修正 normalize_score 的调用参数
        result_score = normalize_score(result_raw, 55)
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
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        # 修正 normalize_score 的调用参数
        strategic_context_gate = normalize_score(vwap_control_raw, 55)
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
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        # 移除整个探针逻辑块，恢复生产状态
        return absorption_strength.clip(0, 1).astype(np.float32)

    def _calculate_behavioral_price_overextension(self, df: pd.DataFrame, tf_weights: Dict, debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        【V4.0 · 行为纯化版】计算纯粹基于行为类原始数据的价格超买亢奋原始分。
        - 核心升级: 引入“行为惯性”和“动态阈值”概念，使亢奋诊断更具情境感知和前瞻性。
        - 优化点:
          1. 行为惯性: 考虑价格和成交量斜率的加速度，作为动量过热的补充证据。
          2. 动态阈值: 根据市场波动率（ATR）动态调整BIAS和BBP的超买阈值。
          3. 成交量极端细化: 区分放量滞涨和放量加速上涨，避免误判。
          4. 融合函数优化: 调整融合权重，并引入一个“亢奋加速度”因子。
        """
        method_name = "_calculate_behavioral_price_overextension"
        p_conf = get_params_block(self.strategy, 'behavioral_divergence_params', {})
        overextension_params = get_param_value(p_conf.get('price_overextension_params'), {
            "enabled": True, "rsi_overbought_threshold": 70, "bias_overbought_threshold": 0.05,
            "bbp_overbought_threshold": 0.95, "volume_climax_multiplier": 1.8,
            "upward_efficiency_decay_penalty": 0.1, "intraday_control_decay_penalty": 0.1,
            "dynamic_bias_bbp_atr_multiplier": 0.005, # [新增] 动态阈值ATR乘数
            "momentum_accel_bonus": 0.1 # [新增] 动量加速度奖励
        })
        if not overextension_params.get('enabled', False):
            return pd.Series(0.0, index=df.index)
        # 获取所需纯行为数据和派生信号
        required_signals = [
            'close_D', 'RSI_13_D', 'MACDh_13_34_8_D', 'volume_D', 'VOL_MA_21_D',
            'BIAS_5_D', 'BBP_21_2.0_D', 'ATR_14_D', # [新增] ATR用于动态阈值
            'ACCEL_5_close_D', 'ACCEL_5_RSI_13_D',
            'ACCEL_5_MACDh_13_34_8_D', 'ACCEL_5_volume_D',
            'robust_pct_change_slope', 'robust_volume_slope', # [新增] 鲁棒斜率用于行为惯性
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            # print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {required_signals}，返回默认值。") # [清理探针]
            return pd.Series(0.0, index=df.index)
        close_price = self._get_safe_series(df, 'close_D', df['close_D'], method_name=method_name)
        rsi_val = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name=method_name)
        macd_val = self._get_safe_series(df, 'MACDh_13_34_8_D', 0.0, method_name=method_name)
        current_volume = self._get_safe_series(df, 'volume_D', 0.0, method_name=method_name)
        volume_avg = self._get_safe_series(df, 'VOL_MA_21_D', 0.0, method_name=method_name)
        bias_val = self._get_safe_series(df, 'BIAS_5_D', 0.0, method_name=method_name)
        bbp_val = self._get_safe_series(df, 'BBP_21_2.0_D', 0.5, method_name=method_name)
        atr_val = self._get_safe_series(df, 'ATR_14_D', 0.0, method_name=method_name) # [新增]
        accel_close = self._get_safe_series(df, 'ACCEL_5_close_D', 0.0, method_name=method_name)
        accel_rsi = self._get_safe_series(df, 'ACCEL_5_RSI_13_D', 0.0, method_name=method_name)
        accel_macd = self._get_safe_series(df, 'ACCEL_5_MACDh_13_34_8_D', 0.0, method_name=method_name)
        accel_volume = self._get_safe_series(df, 'ACCEL_5_volume_D', 0.0, method_name=method_name)
        robust_pct_change_slope = self._get_safe_series(df, 'robust_pct_change_slope', 0.0, method_name=method_name) # [新增]
        robust_volume_slope = self._get_safe_series(df, 'robust_volume_slope', 0.0, method_name=method_name) # [新增]
        upward_efficiency = self._get_safe_series(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5, method_name=method_name)
        intraday_bull_control = self._get_safe_series(df, 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', 0.5, method_name=method_name)
        # 1. 价格偏离度 (Price Deviation)
        rsi_overbought_threshold = overextension_params.get('rsi_overbought_threshold', 70)
        bias_overbought_threshold = overextension_params.get('bias_overbought_threshold', 0.05)
        bbp_overbought_threshold = overextension_params.get('bbp_overbought_threshold', 0.95)
        dynamic_bias_bbp_atr_multiplier = overextension_params.get('dynamic_bias_bbp_atr_multiplier', 0.005) # [新增]
        # 动态调整BIAS和BBP阈值
        dynamic_bias_threshold = bias_overbought_threshold + atr_val * dynamic_bias_bbp_atr_multiplier
        dynamic_bbp_threshold = bbp_overbought_threshold + atr_val * dynamic_bias_bbp_atr_multiplier * 5 # BBP通常变化范围小，乘数大一些
        # RSI超买归一化 (0-1)
        norm_rsi_overbought = (rsi_val - rsi_overbought_threshold).clip(lower=0) / (100 - rsi_overbought_threshold)
        norm_rsi_overbought = norm_rsi_overbought.fillna(0).clip(0, 1)
        # BIAS超买归一化 (0-1) - 使用动态阈值
        norm_bias_overbought = (bias_val - dynamic_bias_threshold).clip(lower=0) / (bias_val.max() - dynamic_bias_threshold)
        norm_bias_overbought = norm_bias_overbought.fillna(0).clip(0, 1)
        # BBP超买归一化 (0-1) - 使用动态阈值
        norm_bbp_overbought = (bbp_val - dynamic_bbp_threshold).clip(lower=0) / (1 - dynamic_bbp_threshold)
        norm_bbp_overbought = norm_bbp_overbought.fillna(0).clip(0, 1)
        # 2. 动量过热 (Momentum Overheating)
        # RSI和MACD加速向上，进一步增强亢奋
        momentum_accel_factor = pd.Series(0.0, index=df.index)
        momentum_accel_factor = momentum_accel_factor.mask((accel_rsi > 0) & (accel_macd > 0), overextension_params.get('momentum_accel_bonus', 0.1)) # 双重加速奖励
        momentum_accel_factor = momentum_accel_factor.mask(((accel_rsi > 0) | (accel_macd > 0)) & (momentum_accel_factor == 0), overextension_params.get('momentum_accel_bonus', 0.1) / 2) # 单一加速奖励
        # [新增] 行为惯性：价格和成交量加速上涨，进一步增强亢奋
        behavioral_inertia_bonus = pd.Series(0.0, index=df.index)
        is_price_accelerating = (robust_pct_change_slope > 0) & (accel_close > 0)
        is_volume_accelerating = (robust_volume_slope > 0) & (accel_volume > 0)
        behavioral_inertia_bonus = behavioral_inertia_bonus.mask(is_price_accelerating & is_volume_accelerating, overextension_params.get('momentum_accel_bonus', 0.1))
        behavioral_inertia_bonus = behavioral_inertia_bonus.mask((is_price_accelerating | is_volume_accelerating) & (behavioral_inertia_bonus == 0), overextension_params.get('momentum_accel_bonus', 0.1) / 2)
        momentum_overheat_score = (norm_rsi_overbought + norm_bias_overbought + momentum_accel_factor + behavioral_inertia_bonus).clip(0, 1)
        # 3. 成交量极端 (Volume Extremity)
        volume_climax_multiplier = overextension_params.get('volume_climax_multiplier', 1.8)
        # [细化] 只有在价格上涨时，放量才被视为亢奋证据
        is_volume_climax = (current_volume > volume_avg * volume_climax_multiplier) & (close_price > close_price.shift(1))
        volume_extremity_score = is_volume_climax.astype(float) * (current_volume / volume_avg).clip(1, 2) # 量比越大，分数越高
        # 4. 日内行为极端 (Intraday Behavioral Extremity)
        upward_efficiency_decay_penalty = overextension_params.get('upward_efficiency_decay_penalty', 0.1)
        intraday_control_decay_penalty = overextension_params.get('intraday_control_decay_penalty', 0.1)
        # 上涨效率衰减 (效率越低，亢奋风险越高)
        norm_upward_efficiency_decay = (1 - upward_efficiency).clip(0, 1) * upward_efficiency_decay_penalty
        # 日内多头控制力减弱 (控制力越弱，亢奋风险越高)
        norm_intraday_control_decay = (1 - intraday_bull_control).clip(0, 1) * intraday_control_decay_penalty
        intraday_extremity_score = (norm_upward_efficiency_decay + norm_intraday_control_decay).clip(0, 1)
        # 非线性融合所有亢奋证据
        # 采用几何平均，确保所有因子都贡献，且因子为0时整体为0
        # 权重分配：价格偏离 (0.3), 动量过热 (0.3), 成交量极端 (0.2), 日内行为极端 (0.2)
        overextension_score = (
            (norm_bbp_overbought + 1e-9).pow(0.3) *
            (momentum_overheat_score + 1e-9).pow(0.3) *
            (volume_extremity_score + 1e-9).pow(0.2) *
            (intraday_extremity_score + 1e-9).pow(0.2)
        ).pow(1/1.0).fillna(0.0).clip(0, 1) # 归一化到0-1
        return overextension_score.astype(np.float32)

    def _calculate_behavioral_stagnation_evidence(self, df: pd.DataFrame, tf_weights: Dict, debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        【V4.0 · 行为纯化版】计算纯粹基于行为类原始数据的滞涨证据原始分。
        - 核心升级: 引入“行为惯性”和“动态阈值”概念，使滞涨诊断更具情境感知和前瞻性。
        - 优化点:
          1. 行为惯性: 考虑价格和成交量斜率的减速或负向加速度，作为滞涨的补充证据。
          2. 动态阈值: 根据市场波动率（ATR）动态调整K线形态（长上影线、小实体）的阈值。
          3. 动量背离细化: 价格上涨但动量指标下降，且下降速度加快，则滞涨证据更强。
          4. 成交量异常细化: 区分放量滞涨和缩量上涨，后者在某些情境下也可能是滞涨证据。
          5. 融合函数优化: 调整融合权重，并引入一个“滞涨加速度”因子。
        """
        method_name = "_calculate_behavioral_stagnation_evidence"
        p_conf = get_params_block(self.strategy, 'behavioral_divergence_params', {})
        stagnation_params = get_param_value(p_conf.get('stagnation_evidence_params'), {
            "enabled": True, "upper_shadow_ratio_threshold": 0.4, "body_ratio_threshold": 0.3,
            "volume_stagnation_multiplier": 1.2, "momentum_divergence_penalty": 0.15,
            "upward_efficiency_decay_bonus": 0.1, "intraday_control_decay_bonus": 0.1,
            "dynamic_kline_atr_multiplier": 0.005, # [新增] 动态K线形态阈值ATR乘数
            "momentum_deceleration_bonus": 0.1, # [新增] 动量减速奖励
            "volume_drying_up_multiplier": 0.8 # [新增] 缩量上涨乘数
        })
        if not stagnation_params.get('enabled', False):
            return pd.Series(0.0, index=df.index)
        # 获取所需纯行为数据和派生信号
        required_signals = [
            'close_D', 'open_D', 'high_D', 'low_D', 'volume_D', 'VOL_MA_21_D', 'ATR_14_D', # [新增] ATR用于动态阈值
            'robust_close_slope', 'robust_RSI_13_slope', 'robust_MACDh_13_34_8_slope', 'robust_volume_slope',
            'ACCEL_5_close_D', 'ACCEL_5_RSI_13_D', 'ACCEL_5_MACDh_13_34_8_D', 'ACCEL_5_volume_D', # [新增] 加速度用于行为惯性
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'pct_change_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            # print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {required_signals}，返回默认值。") # [清理探针]
            return pd.Series(0.0, index=df.index)
        open_price = self._get_safe_series(df, 'open_D', df['close_D'], method_name=method_name)
        high_price = self._get_safe_series(df, 'high_D', df['close_D'], method_name=method_name)
        low_price = self._get_safe_series(df, 'low_D', df['close_D'], method_name=method_name)
        close_price = self._get_safe_series(df, 'close_D', df['close_D'], method_name=method_name)
        current_volume = self._get_safe_series(df, 'volume_D', 0.0, method_name=method_name)
        volume_avg = self._get_safe_series(df, 'VOL_MA_21_D', 0.0, method_name=method_name)
        pct_change_val = self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        atr_val = self._get_safe_series(df, 'ATR_14_D', 0.0, method_name=method_name) # [新增]
        robust_close_slope = self._get_safe_series(df, 'robust_close_slope', 0.0, method_name=method_name)
        robust_rsi_slope = self._get_safe_series(df, 'robust_RSI_13_slope', 0.0, method_name=method_name)
        robust_macd_slope = self._get_safe_series(df, 'robust_MACDh_13_34_8_slope', 0.0, method_name=method_name)
        robust_volume_slope = self._get_safe_series(df, 'robust_volume_slope', 0.0, method_name=method_name)
        accel_close = self._get_safe_series(df, 'ACCEL_5_close_D', 0.0, method_name=method_name) # [新增]
        accel_rsi = self._get_safe_series(df, 'ACCEL_5_RSI_13_D', 0.0, method_name=method_name) # [新增]
        accel_macd = self._get_safe_series(df, 'ACCEL_5_MACDh_13_34_8_D', 0.0, method_name=method_name) # [新增]
        accel_volume = self._get_safe_series(df, 'ACCEL_5_volume_D', 0.0, method_name=method_name) # [新增]
        upward_efficiency = self._get_safe_series(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5, method_name=method_name)
        intraday_bull_control = self._get_safe_series(df, 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', 0.5, method_name=method_name)
        # K线形态分析
        total_range = high_price - low_price
        total_range_safe = total_range.replace(0, 1e-9)
        body_range = (close_price - open_price).abs()
        upper_shadow = high_price - high_price.mask(close_price > open_price, close_price)
        upper_shadow_ratio = (upper_shadow / total_range_safe).clip(0, 1)
        body_ratio = (body_range / total_range_safe).clip(0, 1)
        upper_shadow_ratio_threshold = stagnation_params.get('upper_shadow_ratio_threshold', 0.4)
        body_ratio_threshold = stagnation_params.get('body_ratio_threshold', 0.3)
        volume_stagnation_multiplier = stagnation_params.get('volume_stagnation_multiplier', 1.2)
        momentum_divergence_penalty = stagnation_params.get('momentum_divergence_penalty', 0.15)
        upward_efficiency_decay_bonus = stagnation_params.get('upward_efficiency_decay_bonus', 0.1) # [修正] 命名为bonus，但实际是penalty
        intraday_control_decay_bonus = stagnation_params.get('intraday_control_decay_bonus', 0.1) # [修正] 命名为bonus，但实际是penalty
        dynamic_kline_atr_multiplier = stagnation_params.get('dynamic_kline_atr_multiplier', 0.005) # [新增]
        momentum_deceleration_bonus = stagnation_params.get('momentum_deceleration_bonus', 0.1) # [新增]
        volume_drying_up_multiplier = stagnation_params.get('volume_drying_up_multiplier', 0.8) # [新增]
        # 动态调整K线形态阈值
        dynamic_upper_shadow_threshold = upper_shadow_ratio_threshold + atr_val * dynamic_kline_atr_multiplier
        dynamic_body_ratio_threshold = body_ratio_threshold - atr_val * dynamic_kline_atr_multiplier # 波动率高时，小实体可能更大
        # 1. 价格行为疲软 (Price Action Weakness)
        is_long_upper_shadow = (upper_shadow_ratio > dynamic_upper_shadow_threshold) # [使用动态阈值]
        is_small_body = (body_ratio < dynamic_body_ratio_threshold) # [使用动态阈值]
        is_high_open_low_close = (open_price > close_price) & (pct_change_val < 0) # 高开低走
        price_weakness_score = pd.Series(0.0, index=df.index)
        price_weakness_score = price_weakness_score.mask(is_long_upper_shadow & is_small_body, 0.3)
        price_weakness_score = price_weakness_score.mask(is_high_open_low_close, price_weakness_score + 0.4) # 高开低走更严重
        # 2. 动量背离 (Momentum Divergence)
        # 价格上涨，但RSI或MACD斜率减弱或转负
        is_price_rising = (robust_close_slope > 0)
        is_rsi_momentum_decay = (robust_rsi_slope < 0)
        is_macd_momentum_decay = (robust_macd_slope < 0)
        momentum_divergence_score = pd.Series(0.0, index=df.index)
        # [细化] 动量下降且加速度为负（下降速度加快），则奖励更高
        rsi_deceleration_bonus = (is_rsi_momentum_decay & (accel_rsi < 0)).astype(int) * momentum_deceleration_bonus
        macd_deceleration_bonus = (is_macd_momentum_decay & (accel_macd < 0)).astype(int) * momentum_deceleration_bonus
        momentum_divergence_score = momentum_divergence_score.mask(
            is_price_rising & (is_rsi_momentum_decay | is_macd_momentum_decay),
            momentum_divergence_penalty * (is_rsi_momentum_decay.astype(int) + is_macd_momentum_decay.astype(int)) + \
            rsi_deceleration_bonus + macd_deceleration_bonus # [新增] 加速度奖励
        )
        momentum_divergence_score = momentum_divergence_score.clip(0, 0.3)
        # 3. 成交量异常 (Volume Anomaly)
        # 放量滞涨 (价格上涨幅度小，但成交量大)
        is_volume_stagnation = (pct_change_val.abs() < 0.01) & (current_volume > volume_avg * volume_stagnation_multiplier) & (robust_close_slope > 0)
        volume_extremity_score = is_volume_stagnation.astype(float) * (current_volume / volume_avg).clip(1, 2) # 量比越大，分数越高
        # [新增] 缩量上涨 (价格上涨，但成交量萎缩)
        is_volume_drying_up = (pct_change_val > 0) & (current_volume < volume_avg * volume_drying_up_multiplier) & (robust_close_slope > 0)
        volume_drying_up_score = is_volume_drying_up.astype(float) * (1 - (current_volume / volume_avg)).clip(0, 1) # 萎缩越多，分数越高
        volume_anomaly_score = (volume_extremity_score + volume_drying_up_score).clip(0, 1) # 两种情况叠加
        # 4. 日内控制力减弱 (Intraday Control Weakness)
        upward_efficiency_decay_penalty = stagnation_params.get('upward_efficiency_decay_bonus', 0.1) # [修正] 命名为bonus，但实际是penalty
        intraday_control_decay_penalty = stagnation_params.get('intraday_control_decay_bonus', 0.1) # [修正] 命名为bonus，但实际是penalty
        # 上涨效率衰减 (效率越低，滞涨风险越高)
        norm_upward_efficiency_decay = (1 - upward_efficiency).clip(0, 1) * upward_efficiency_decay_penalty
        # 日内多头控制力减弱 (控制力越弱，滞涨风险越高)
        norm_intraday_control_decay = (1 - intraday_bull_control).clip(0, 1) * intraday_control_decay_penalty
        intraday_control_weakness_score = (norm_upward_efficiency_decay + norm_intraday_control_decay).clip(0, 1)
        # 非线性融合所有滞涨证据
        # 采用几何平均，确保所有因子都贡献，且因子为0时整体为0
        # 权重分配：价格行为疲软 (0.3), 动量背离 (0.3), 成交量异常 (0.2), 日内控制力减弱 (0.2)
        stagnation_score = (
            (price_weakness_score + 1e-9).pow(0.3) *
            (momentum_divergence_score + 1e-9).pow(0.3) *
            (volume_anomaly_score + 1e-9).pow(0.2) *
            (intraday_control_weakness_score + 1e-9).pow(0.2)
        ).pow(1/1.0).fillna(0.0).clip(0, 1) # 归一化到0-1
        return stagnation_score.astype(np.float32)

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
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
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
        # 修正 normalize_score 的调用参数
        control_score = normalize_score(control_raw, 55) # VWAP控制力本身就是[-1,1]附近，用简单归一化即可
        decisiveness_score = ((close - low) / (high - low + 1e-9)).fillna(0.5).clip(0, 1)
        execution_quality_score = (
            efficiency_score * quality_weights.get('efficiency', 0.4) +
            control_score * quality_weights.get('control', 0.4) +
            decisiveness_score * quality_weights.get('decisiveness', 0.2)
        ).clip(0, 1)
        # --- 4. “大国工匠协议”三维合成 ---
        base_confirmation = (tactical_action_score * execution_quality_score).pow(0.5).fillna(0.0)
        shakeout_confirmation_score = (strategic_intent_score * base_confirmation).clip(0, 1)
        # 移除整个探针逻辑块，恢复生产状态
        return shakeout_confirmation_score.astype(np.float32)

    def _diagnose_pure_behavioral_divergence(self, df: pd.DataFrame, tf_weights: Dict, debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> Tuple[pd.Series, pd.Series]:
        """
        【V8.1 · 行为背离强度惯性与自适应引擎版】诊断纯粹基于行为类原始数据的看涨/看跌背离信号。
        - 核心重构: 提取看涨/看跌背离的公共计算逻辑到辅助方法，减少代码冗余，提高可读性和维护性。
        - 优化效率: 集中获取参数和信号，避免重复计算。
        - 清理探针: 移除所有调试打印，使代码更简洁。
        """
        method_name = "_diagnose_pure_behavioral_divergence"
        # 1. 获取所有配置参数
        p_conf = get_params_block(self.strategy, 'behavioral_divergence_params', {})
        # 修正键名 'default_weights' 为 'default'，并从 p_mtf 获取
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights_from_config = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        params = { # 将所有参数打包成一个字典
            'mtf_slopes_params': get_param_value(p_conf.get('multi_timeframe_slopes'), {"enabled": True, "periods": [5, 13], "weights": {"5": 0.7, "13": 0.3}}),
            'multi_level_resonance_params': get_param_value(p_conf.get('multi_level_resonance_params'), {"enabled": True, "long_term_period": 21, "resonance_bonus": 0.2}),
            'persistence_params': get_param_value(p_conf.get('persistence_params'), {"enabled": True, "min_duration": 2, "max_duration_window": 5, "quality_decay_factor": 0.05}),
            'adaptive_thresholds_params': get_param_value(p_conf.get('adaptive_thresholds'), {"enabled": True, "min_slope_diff_atr_multiplier": 0.005, "min_slope_diff_base": 0.005, "rsi_oversold_trend_adjust_factor": 5, "rsi_overbought_trend_adjust_factor": 5}),
            'synergy_weights_params': get_param_value(p_conf.get('synergy_weights'), {"enabled": True, "two_indicators_synergy_bonus": 0.1, "three_indicators_synergy_bonus": 0.2}),
            'structural_context_weights_params': get_param_value(p_conf.get('structural_context_weights'), {"enabled": True, "bbw_slope_penalty_factor": 0.1}),
            'price_action_confirmation_params': get_param_value(p_conf.get('price_action_confirmation_params'), {"enabled": True, "lower_shadow_ratio_threshold": 0.4, "upper_shadow_ratio_threshold": 0.4, "body_ratio_threshold": 0.3, "confirmation_bonus": 0.15, "engulfing_pattern_bonus": 0.1, "hammer_shooting_star_bonus": 0.1}),
            'volume_price_structure_params': get_param_value(p_conf.get('volume_price_structure_params'), {"enabled": True, "volume_climax_threshold_multiplier": 2.0, "volume_drying_up_threshold_multiplier": 0.5, "narrow_range_body_ratio": 0.2, "wide_range_body_ratio": 0.7, "structure_bonus": 0.15}),
            'purity_assessment_params': get_param_value(p_conf.get('purity_assessment_params'), {"enabled": True, "slope_std_dev_threshold": 0.5, "whipsaw_penalty_factor": 0.1}),
            'market_regime_params': get_param_value(p_conf.get('market_regime_params'), {"enabled": True, "adx_trend_threshold": 25, "adx_ranging_threshold": 20, "adx_div_weight_max_adjust": 0.3, "adx_conf_weight_max_adjust": 0.3, "atr_conf_weight_max_adjust": 0.2}),
            'market_context_params': get_param_value(p_conf.get('market_context_params'), {"enabled": True, "price_momentum_window": 5, "favorable_sentiment_slope_threshold": 0.001, "unfavorable_sentiment_slope_threshold": -0.001, "slope_stability_threshold": 0.7, "adx_strength_threshold": 25, "favorable_context_bonus": 0.1, "unfavorable_context_penalty": 0.1}),
            'signal_freshness_params': get_param_value(p_conf.get('signal_freshness_params'), {"enabled": True, "freshness_bonus": 0.1}),
            'behavioral_consistency_params': get_param_value(p_conf.get('behavioral_consistency_params'), {"enabled": True, "min_consistent_indicators_for_bonus": 2, "conflict_penalty_threshold": 1, "consistency_bonus": 0.1, "conflict_penalty": 0.15}),
            'pattern_sequence_params': get_param_value(p_conf.get('pattern_sequence_params'), {"enabled": True, "lookback_window": 3, "volume_drying_up_ratio": 0.8, "volume_climax_ratio": 1.5, "reversal_pct_change_threshold": 0.01, "sequence_bonus": 0.2}),
            'behavioral_strength_params': get_param_value(p_conf.get('behavioral_strength_params'), {"enabled": True, "acceleration_bonus": 0.05}),
            'behavioral_inertia_params': get_param_value(p_conf.get('behavioral_inertia_params'), {"enabled": True, "long_term_adx_threshold": 30, "long_term_slope_stability_threshold": 0.8, "high_inertia_penalty": 0.1, "low_inertia_bonus": 0.05}),
            'adaptive_fusion_weights_params': get_param_value(p_conf.get('adaptive_fusion_weights_params'), {"enabled": True, "trend_strong_penalty_factor": 0.1, "ranging_bonus_factor": 0.1, "volatility_high_penalty_factor": 0.05}),
            'bullish_div_weights': get_param_value(p_conf.get('bullish_divergence_weights'), {"price_rsi": 0.4, "price_macd": 0.3, "price_volume": 0.3}),
            'bearish_div_weights': get_param_value(p_conf.get('bearish_divergence_weights'), {"price_rsi": 0.4, "price_macd": 0.3, "price_volume": 0.3}),
            'bullish_conf_weights': get_param_value(p_conf.get('bullish_confirmation_weights'), {"rsi_oversold": 0.3, "volume_increase": 0.3, "buying_support": 0.2, "volatility_high": 0.2}),
            'bearish_conf_weights': get_param_value(p_conf.get('bearish_confirmation_weights'), {"rsi_overbought": 0.3, "volume_decrease": 0.3, "selling_pressure": 0.2, "volatility_high": 0.2}),
            'rsi_oversold_threshold_base': get_param_value(p_conf.get('rsi_oversold_threshold'), 30),
            'rsi_overbought_threshold_base': get_param_value(p_conf.get('rsi_overbought_threshold'), 70),
            'min_divergence_slope_diff_base': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('min_slope_diff_base'), 0.005),
            'min_slope_diff_atr_multiplier': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('min_slope_diff_atr_multiplier'), 0.005),
            'rsi_oversold_trend_adjust_factor': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('rsi_oversold_trend_adjust_factor'), 5),
            'rsi_overbought_trend_adjust_factor': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('rsi_overbought_trend_adjust_factor'), 5),
            'long_term_period': get_param_value(get_param_value(p_conf.get('multi_level_resonance_params'), {}).get('long_term_period'), 21)
        }
        # 2. 获取所需原始数据和派生信号
        mtf_periods = params['mtf_slopes_params'].get('periods', [5])
        accel_period = mtf_periods[0]
        required_signals = [
            'close_D', 'RSI_13_D', 'MACDh_13_34_8_D', 'volume_D', 'ATR_14_D', 'BBW_21_2.0_D',
            'active_buying_support_D', 'active_selling_pressure_D', 'trend_vitality_index_D',
            'open_D', 'high_D', 'low_D', 'ADX_14_D', 'VOL_MA_21_D', 'pct_change_D',
            'BIAS_5_D', 'BBP_21_2.0_D',
            'ACCEL_5_pct_change_D',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW',
            'robust_close_slope', 'robust_RSI_13_slope', 'robust_MACDh_13_34_8_slope', 'robust_volume_slope',
            'robust_BBW_21_2.0_slope', 'robust_pct_change_slope',
            'long_term_close_slope', 'long_term_RSI_13_slope', 'long_term_MACDh_13_34_8_slope', 'long_term_volume_slope',
            'long_term_adx_slope',
            'pattern_close_slope', 'pattern_volume_slope',
            f'ACCEL_{accel_period}_close_D',
            f'ACCEL_{accel_period}_RSI_13_D',
            f'ACCEL_{accel_period}_MACDh_13_34_8_D',
            f'ACCEL_{accel_period}_volume_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {required_signals}，返回默认值。")
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        # 集中获取所有信号
        robust_close_slope = self._get_safe_series(df, 'robust_close_slope', 0.0, method_name=method_name)
        robust_rsi_slope = self._get_safe_series(df, 'robust_RSI_13_slope', 0.0, method_name=method_name)
        robust_macd_slope = self._get_safe_series(df, 'robust_MACDh_13_34_8_slope', 0.0, method_name=method_name)
        robust_volume_slope = self._get_safe_series(df, 'robust_volume_slope', 0.0, method_name=method_name)
        robust_bbw_slope = self._get_safe_series(df, 'robust_BBW_21_2.0_slope', 0.0, method_name=method_name)
        robust_pct_change_slope = self._get_safe_series(df, 'robust_pct_change_slope', 0.0, method_name=method_name)
        long_term_close_slope = self._get_safe_series(df, 'long_term_close_slope', 0.0, method_name=method_name)
        long_term_rsi_slope = self._get_safe_series(df, 'long_term_RSI_13_slope', 0.0, method_name=method_name)
        long_term_macd_slope = self._get_safe_series(df, 'long_term_MACDh_13_34_8_slope', 0.0, method_name=method_name)
        long_term_volume_slope = self._get_safe_series(df, 'long_term_volume_slope', 0.0, method_name=method_name)
        long_term_adx_slope = self._get_safe_series(df, 'long_term_adx_slope', 0.0, method_name=method_name)
        pattern_close_slope = self._get_safe_series(df, 'pattern_close_slope', 0.0, method_name=method_name)
        pattern_volume_slope = self._get_safe_series(df, 'pattern_volume_slope', 0.0, method_name=method_name)
        accel_close = self._get_safe_series(df, f'ACCEL_{accel_period}_close_D', 0.0, method_name=method_name)
        accel_rsi = self._get_safe_series(df, f'ACCEL_{accel_period}_RSI_13_D', 0.0, method_name=method_name)
        accel_macd = self._get_safe_series(df, f'ACCEL_{accel_period}_MACDh_13_34_8_D', 0.0, method_name=method_name)
        accel_volume = self._get_safe_series(df, f'ACCEL_{accel_period}_volume_D', 0.0, method_name=method_name)
        rsi_val = self._get_safe_series(df, 'RSI_13_D', 50.0, method_name=method_name)
        atr_val = self._get_safe_series(df, 'ATR_14_D', 0.0, method_name=method_name)
        active_buying = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        active_selling = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name=method_name)
        raw_trend_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.5, method_name=method_name)
        # 修正 normalize_score 的调用参数
        trend_vitality = normalize_score(raw_trend_vitality, 55)
        open_price = self._get_safe_series(df, 'open_D', df['close_D'], method_name=method_name)
        high_price = self._get_safe_series(df, 'high_D', df['close_D'], method_name=method_name)
        low_price = self._get_safe_series(df, 'low_D', df['close_D'], method_name=method_name)
        close_price = self._get_safe_series(df, 'close_D', df['close_D'], method_name=method_name)
        adx_val = self._get_safe_series(df, 'ADX_14_D', 0.0, method_name=method_name)
        current_volume = self._get_safe_series(df, 'volume_D', 0.0, method_name=method_name)
        volume_avg = self._get_safe_series(df, 'VOL_MA_21_D', 0.0, method_name=method_name)
        pct_change_val = self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        upward_efficiency_val = self._get_safe_series(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5, method_name=method_name)
        intraday_bull_control_val = self._get_safe_series(df, 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', 0.5, method_name=method_name)
        internal_price_overextension_raw = self._get_safe_series(df, 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.0, method_name=method_name)
        internal_stagnation_evidence_raw = self._get_safe_series(df, 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0, method_name=method_name)
        # 归一化确认因子 (0到1)
        norm_active_buying = get_adaptive_mtf_normalized_score(active_buying, df.index, ascending=True, tf_weights=tf_weights)
        norm_active_selling = get_adaptive_mtf_normalized_score(active_selling, df.index, ascending=True, tf_weights=tf_weights)
        norm_atr = get_adaptive_mtf_normalized_score(atr_val, df.index, ascending=True, tf_weights=tf_weights)
        # 动态阈值计算
        dynamic_min_divergence_slope_diff = params['min_divergence_slope_diff_base'] + atr_val * params['min_slope_diff_atr_multiplier']
        rsi_oversold_threshold_dynamic = params['rsi_oversold_threshold_base'] - (trend_vitality - 0.5) * params['rsi_oversold_trend_adjust_factor']
        rsi_overbought_threshold_dynamic = params['rsi_overbought_threshold_base'] + (trend_vitality - 0.5) * params['rsi_overbought_trend_adjust_factor']
        # 3. 调用辅助方法计算看涨和看跌背离
        bullish_divergence_score = self._calculate_single_divergence_type(
            df, True, params, default_weights_from_config, # 修改：传递 default_weights_from_config
            # 移除 debug_enabled, probe_ts
            robust_close_slope, robust_rsi_slope, robust_macd_slope, robust_volume_slope, robust_bbw_slope, robust_pct_change_slope,
            long_term_close_slope, long_term_rsi_slope, long_term_macd_slope, long_term_volume_slope, long_term_adx_slope,
            pattern_close_slope, pattern_volume_slope,
            accel_close, accel_rsi, accel_macd, accel_volume,
            rsi_val, atr_val, active_buying, active_selling, trend_vitality,
            open_price, high_price, low_price, close_price, adx_val, current_volume, volume_avg, pct_change_val,
            upward_efficiency_val, intraday_bull_control_val,
            internal_price_overextension_raw, internal_stagnation_evidence_raw,
            norm_active_buying, norm_active_selling, norm_atr,
            dynamic_min_divergence_slope_diff, rsi_oversold_threshold_dynamic, rsi_overbought_threshold_dynamic
        )
        bearish_divergence_score = self._calculate_single_divergence_type(
            df, False, params, default_weights_from_config, # 修改：传递 default_weights_from_config
            # 移除 debug_enabled, probe_ts
            robust_close_slope, robust_rsi_slope, robust_macd_slope, robust_volume_slope, robust_bbw_slope, robust_pct_change_slope,
            long_term_close_slope, long_term_rsi_slope, long_term_macd_slope, long_term_volume_slope, long_term_adx_slope,
            pattern_close_slope, pattern_volume_slope,
            accel_close, accel_rsi, accel_macd, accel_volume,
            rsi_val, atr_val, active_buying, active_selling, trend_vitality,
            open_price, high_price, low_price, close_price, adx_val, current_volume, volume_avg, pct_change_val,
            upward_efficiency_val, intraday_bull_control_val,
            internal_price_overextension_raw, internal_stagnation_evidence_raw,
            norm_active_buying, norm_active_selling, norm_atr,
            dynamic_min_divergence_slope_diff, rsi_oversold_threshold_dynamic, rsi_overbought_threshold_dynamic
        )
        return bullish_divergence_score, bearish_divergence_score

    def _calculate_single_divergence_type(self, df: pd.DataFrame, is_bullish: bool, params: Dict, tf_weights: Dict,
                                          # Pre-computed common signals
                                          robust_close_slope, robust_rsi_slope, robust_macd_slope, robust_volume_slope, robust_bbw_slope, robust_pct_change_slope,
                                          long_term_close_slope, long_term_rsi_slope, long_term_macd_slope, long_term_volume_slope, long_term_adx_slope,
                                          pattern_close_slope, pattern_volume_slope,
                                          accel_close, accel_rsi, accel_macd, accel_volume,
                                          rsi_val, atr_val, active_buying, active_selling, trend_vitality,
                                          open_price, high_price, low_price, close_price, adx_val, current_volume, volume_avg, pct_change_val,
                                          upward_efficiency_val, intraday_bull_control_val,
                                          internal_price_overextension_raw, internal_stagnation_evidence_raw,
                                          norm_active_buying, norm_active_selling, norm_atr,
                                          dynamic_min_divergence_slope_diff, rsi_oversold_threshold_dynamic, rsi_overbought_threshold_dynamic) -> pd.Series:
        """
        【V8.1 · 行为背离强度惯性与自适应引擎版】辅助方法：计算单一类型的行为背离（看涨或看跌）。
        """
        method_name = "_calculate_single_divergence_type"
        # 从params字典中获取所有子参数
        mtf_slopes_params = params['mtf_slopes_params']
        multi_level_resonance_params = params['multi_level_resonance_params']
        persistence_params = params['persistence_params']
        adaptive_thresholds_params = params['adaptive_thresholds_params']
        synergy_weights_params = params['synergy_weights_params']
        structural_context_weights_params = params['structural_context_weights_params']
        price_action_confirmation_params = params['price_action_confirmation_params']
        volume_price_structure_params = params['volume_price_structure_params']
        purity_assessment_params = params['purity_assessment_params']
        market_regime_params = params['market_regime_params']
        market_context_params = params['market_context_params']
        signal_freshness_params = params['signal_freshness_params']
        behavioral_consistency_params = params['behavioral_consistency_params']
        pattern_sequence_params = params['pattern_sequence_params']
        behavioral_strength_params = params['behavioral_strength_params']
        behavioral_inertia_params = params['behavioral_inertia_params']
        adaptive_fusion_weights_params = params['adaptive_fusion_weights_params']
        bullish_div_weights = params['bullish_div_weights']
        bearish_div_weights = params['bearish_div_weights']
        bullish_conf_weights = params['bullish_conf_weights']
        bearish_conf_weights = params['bearish_conf_weights']
        long_term_period = params['long_term_period']
        # 根据is_bullish设置方向性变量
        price_trend_condition = (robust_close_slope < 0) if is_bullish else (robust_close_slope > 0)
        rsi_indicator_trend = (robust_rsi_slope > 0) if is_bullish else (robust_rsi_slope < 0)
        macd_indicator_trend = (robust_macd_slope > 0) if is_bullish else (robust_macd_slope < 0)
        volume_indicator_trend = (robust_volume_slope > 0) if is_bullish else (robust_volume_slope < 0)
        div_condition_raw = price_trend_condition & (rsi_indicator_trend | macd_indicator_trend | volume_indicator_trend)
        # 行为强度与持续性 - 强度 (Accelerated Strength)
        accelerated_strength = pd.Series(0.0, index=df.index)
        if behavioral_strength_params.get('enabled'):
            acceleration_bonus = behavioral_strength_params.get('acceleration_bonus', 0.05)
            accel_period = mtf_slopes_params.get("periods", [5])[0]
            accel_rsi_condition = (accel_rsi > 0) if is_bullish else (accel_rsi < 0)
            accel_macd_condition = (accel_macd > 0) if is_bullish else (accel_macd < 0)
            accel_volume_condition = (accel_volume > 0) if is_bullish else (accel_volume < 0)
            accel_strength_rsi = (rsi_indicator_trend & accel_rsi_condition).astype(int) * acceleration_bonus
            accel_strength_macd = (macd_indicator_trend & accel_macd_condition).astype(int) * acceleration_bonus
            accel_strength_volume = (volume_indicator_trend & accel_volume_condition).astype(int) * acceleration_bonus
            accelerated_strength = (accel_strength_rsi + accel_strength_macd + accel_strength_volume).clip(0, 0.15)
        # 计算背离强度 (融合加速度奖励)
        div_strength_rsi = (rsi_indicator_trend * (robust_rsi_slope - robust_close_slope * (-1 if is_bullish else 1))).clip(lower=0)
        div_strength_macd = (macd_indicator_trend * (robust_macd_slope - robust_close_slope * (-1 if is_bullish else 1))).clip(lower=0)
        div_strength_volume = (volume_indicator_trend * (robust_volume_slope - robust_close_slope * (-1 if is_bullish else 1))).clip(lower=0)
        div_weights = bullish_div_weights if is_bullish else bearish_div_weights
        total_div_strength = (
            div_strength_rsi * div_weights.get('price_rsi', 0.4) +
            div_strength_macd * div_weights.get('price_macd', 0.3) +
            div_strength_volume * div_weights.get('price_volume', 0.3)
        )
        total_div_strength = total_div_strength.where(total_div_strength > dynamic_min_divergence_slope_diff, 0.0)
        norm_total_div_strength = get_adaptive_mtf_normalized_score(total_div_strength, df.index, ascending=True, tf_weights=tf_weights)
        final_strength_factor = norm_total_div_strength * (1 + accelerated_strength)
        final_strength_factor = final_strength_factor.clip(0, 1.5)
        # 确认因子 (精细化为连续值)
        conf_weights = bullish_conf_weights if is_bullish else bearish_conf_weights
        if is_bullish:
            rsi_conf = get_adaptive_mtf_normalized_score((rsi_oversold_threshold_dynamic - rsi_val).clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
            volume_change_conf = get_adaptive_mtf_normalized_score(robust_volume_slope.clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
            active_flow_conf = norm_active_buying
        else: # Bearish
            rsi_conf = get_adaptive_mtf_normalized_score((rsi_val - rsi_overbought_threshold_dynamic).clip(lower=0), df.index, ascending=True, tf_weights=tf_weights)
            volume_change_conf = get_adaptive_mtf_normalized_score(robust_volume_slope.clip(upper=0).abs(), df.index, ascending=True, tf_weights=tf_weights)
            active_flow_conf = norm_active_selling
        total_conf_factor = (
            rsi_conf * conf_weights.get('rsi_oversold' if is_bullish else 'rsi_overbought', 0.3) +
            volume_change_conf * conf_weights.get('volume_increase' if is_bullish else 'volume_decrease', 0.3) +
            active_flow_conf * conf_weights.get('buying_support' if is_bullish else 'selling_pressure', 0.2) +
            norm_atr * conf_weights.get('volatility_high', 0.2)
        )
        # 修正 normalize_score 的调用参数
        norm_total_conf_factor = normalize_score(total_conf_factor, 55)
        # 行为强度与持续性 - 持续性 (Persistence Factor with Quality)
        persistence_factor = pd.Series(0.0, index=df.index)
        if persistence_params.get('enabled'):
            min_persistence_duration = persistence_params.get('min_duration', 2)
            max_persistence_window = persistence_params.get('max_duration_window', 5)
            quality_decay_factor = persistence_params.get('quality_decay_factor', 0.05)
            accel_close_condition = (accel_close > 0) if is_bullish else (accel_close < 0)
            accel_volume_condition = (accel_volume < 0) if is_bullish else (accel_volume < 0) # Volume drying up for bullish, volume decreasing for bearish
            persistence_quality = (accel_close_condition.astype(int) + accel_volume_condition.astype(int)).clip(0, 2)
            persistence_count = div_condition_raw.astype(int).rolling(window=max_persistence_window).apply(lambda x: (x == 1).sum(), raw=True).fillna(0)
            persistence_factor = (persistence_count / max_persistence_window) * (1 + persistence_quality * quality_decay_factor * persistence_count)
            persistence_factor = persistence_factor.where(persistence_count >= min_persistence_duration, 0.0)
            persistence_factor = persistence_factor.clip(0, 1.5)
        # 多指标协同奖励 (Synergy Bonus)
        num_indicators_diverging = rsi_indicator_trend.astype(int) + macd_indicator_trend.astype(int) + volume_indicator_trend.astype(int)
        synergy_bonus = pd.Series(0.0, index=df.index)
        if synergy_weights_params.get('enabled'):
            synergy_bonus = synergy_bonus.mask(num_indicators_diverging == 2, synergy_weights_params.get('two_indicators_synergy_bonus', 0.1))
            synergy_bonus = synergy_bonus.mask(num_indicators_diverging >= 3, synergy_weights_params.get('three_indicators_synergy_bonus', 0.2))
        synergy_factor = (1 + synergy_bonus).clip(1, 1.5)
        # 结构上下文惩罚 (Structural Context Penalty)
        bbw_slope_penalty = pd.Series(0.0, index=df.index)
        if structural_context_weights_params.get('enabled'):
            bbw_slope_penalty = robust_bbw_slope.clip(lower=0) * structural_context_weights_params.get('bbw_slope_penalty_factor', 0.1)
        structural_context_factor = (1 - get_adaptive_mtf_normalized_score(bbw_slope_penalty, df.index, ascending=True, tf_weights=tf_weights)).clip(0.5, 1)
        # 多级别背离共振 (Multi-Level Divergence Resonance)
        resonance_factor = pd.Series(1.0, index=df.index)
        if multi_level_resonance_params.get('enabled'):
            long_term_price_trend = (long_term_close_slope < 0) if is_bullish else (long_term_close_slope > 0)
            long_term_rsi_trend = (long_term_rsi_slope > 0) if is_bullish else (long_term_rsi_slope < 0)
            long_term_macd_trend = (long_term_macd_slope > 0) if is_bullish else (long_term_macd_slope < 0)
            long_term_volume_trend = (long_term_volume_slope > 0) if is_bullish else (long_term_volume_slope < 0)
            long_term_div_condition = long_term_price_trend & (long_term_rsi_trend | long_term_macd_trend | long_term_volume_trend)
            resonance_factor = resonance_factor.mask(div_condition_raw & long_term_div_condition, 1 + multi_level_resonance_params.get('resonance_bonus', 0.2))
        # 价格行为结构确认 (Price Action Structural Confirmation)
        price_action_conf = pd.Series(0.0, index=df.index)
        if price_action_confirmation_params.get('enabled'):
            body_range = (close_price - open_price).abs()
            total_range = high_price - low_price
            total_range_safe = total_range.replace(0, 1e-9)
            lower_shadow = low_price.mask(close_price > open_price, open_price) - low_price.mask(close_price > open_price, close_price)
            upper_shadow = high_price - high_price.mask(close_price > open_price, close_price)
            lower_shadow_ratio = (lower_shadow / total_range_safe).clip(0, 1)
            upper_shadow_ratio = (upper_shadow / total_range_safe).clip(0, 1)
            body_ratio = (body_range / total_range_safe).clip(0, 1)
            is_long_lower_shadow = (lower_shadow_ratio > price_action_confirmation_params.get('lower_shadow_ratio_threshold', 0.4))
            is_long_upper_shadow = (upper_shadow_ratio > price_action_confirmation_params.get('upper_shadow_ratio_threshold', 0.4))
            is_small_body = (body_ratio < price_action_confirmation_params.get('body_ratio_threshold', 0.3))
            if is_bullish:
                is_engulfing = (
                    (close_price > open_price) & # 当前是阳线
                    (df['close_D'].shift(1) < df['open_D'].shift(1)) & # 前一日是阴线
                    (close_price > df['open_D'].shift(1)) & # 当前收盘价高于前一日开盘价
                    (open_price < df['close_D'].shift(1)) # 当前开盘价低于前一日收盘价
                )
                is_reversal_candle = (
                    (close_price > open_price) & # 阳线
                    (is_long_lower_shadow) & # 长下影线
                    (is_small_body) & # 小实体
                    (upper_shadow_ratio < 0.1) # 短上影线 (锤头)
                )
            else: # Bearish
                is_engulfing = (
                    (close_price < open_price) & # 当前是阴线
                    (df['close_D'].shift(1) > df['open_D'].shift(1)) & # 前一日是阳线
                    (close_price < df['open_D'].shift(1)) & # 当前收盘价低于前一日开盘价
                    (open_price > df['close_D'].shift(1)) # 当前开盘价高于前一日开盘价
                )
                is_reversal_candle = (
                    (close_price < open_price) & # 阴线
                    (is_long_upper_shadow) & # 长上影线
                    (is_small_body) & # 小实体
                    (lower_shadow_ratio < 0.1) # 短下影线 (射击之星)
                )
            price_action_conf = price_action_conf.mask((is_long_lower_shadow if is_bullish else is_long_upper_shadow) & is_small_body, price_action_confirmation_params.get('confirmation_bonus', 0.15))
            price_action_conf = price_action_conf.mask(is_engulfing, price_action_confirmation_params.get('engulfing_pattern_bonus', 0.1))
            price_action_conf = price_action_conf.mask(is_reversal_candle, price_action_confirmation_params.get('hammer_shooting_star_bonus', 0.1))
        price_action_factor = (1 + price_action_conf).clip(1, 1.5)
        # 多维度量价结构确认 (Multi-Dimensional Volume-Price Structure Confirmation)
        volume_price_structure_factor = pd.Series(1.0, index=df.index)
        if volume_price_structure_params.get('enabled'):
            vol_climax_mult = volume_price_structure_params.get('volume_climax_threshold_multiplier', 2.0)
            vol_drying_mult = volume_price_structure_params.get('volume_drying_up_threshold_multiplier', 0.5)
            narrow_body_ratio = volume_price_structure_params.get('narrow_range_body_ratio', 0.2)
            wide_body_ratio = volume_price_structure_params.get('wide_range_body_ratio', 0.7)
            structure_bonus = volume_price_structure_params.get('structure_bonus', 0.15)
            is_volume_drying_up = (current_volume < volume_avg * vol_drying_mult)
            is_volume_climax = (current_volume > volume_avg * vol_climax_mult)
            total_range_safe = (high_price - low_price).replace(0, 1e-9)
            body_ratio = ((close_price - open_price).abs() / total_range_safe).clip(0, 1)
            is_narrow_range_bar = (body_ratio < narrow_body_ratio)
            is_wide_range_bar = (body_ratio > wide_body_ratio)
            volume_price_structure_conf = pd.Series(0.0, index=df.index)
            if is_bullish:
                volume_price_structure_conf = volume_price_structure_conf.mask(
                    is_volume_drying_up & is_narrow_range_bar, structure_bonus
                )
                volume_price_structure_conf = volume_price_structure_conf.mask(
                    price_trend_condition & is_volume_climax & (close_price > open_price), structure_bonus
                )
            else: # Bearish
                volume_price_structure_conf = volume_price_structure_conf.mask(
                    is_volume_drying_up & is_wide_range_bar, structure_bonus
                )
                volume_price_structure_conf = volume_price_structure_conf.mask(
                    price_trend_condition & is_volume_climax & (close_price < open_price), structure_bonus
                )
            volume_price_structure_factor = (1 + volume_price_structure_conf).clip(1, 1.5)
        # 背离的“纯度”与“质量”评估 (Divergence Purity and Quality Assessment)
        purity_factor = pd.Series(1.0, index=df.index)
        if purity_assessment_params.get('enabled'):
            short_term_close_slopes = self._get_safe_series(df, f'SLOPE_{mtf_slopes_params.get("periods", [5])[0]}_close_D', 0.0, method_name=method_name)
            slope_std_dev = short_term_close_slopes.rolling(window=mtf_slopes_params.get("periods", [5])[0]).std().fillna(0)
            # 修正 normalize_score 的调用参数
            norm_slope_std_dev = normalize_score(slope_std_dev, 55, ascending=False)
            # 修改结束
            purity_penalty = pd.Series(0.0, index=df.index)
            purity_penalty = purity_penalty.mask(
                price_trend_condition & (norm_slope_std_dev > purity_assessment_params.get('slope_std_dev_threshold', 0.5)),
                norm_slope_std_dev * purity_assessment_params.get('whipsaw_penalty_factor', 0.1)
            )
            purity_factor = (1 - purity_penalty).clip(0.5, 1)
        # 市场情境动态权重 (Market Regime Dynamic Weighting)
        dynamic_div_weight_multiplier = pd.Series(1.0, index=df.index)
        dynamic_conf_weight_multiplier = pd.Series(1.0, index=df.index)
        if market_regime_params.get('enabled'):
            adx_trend_threshold = market_regime_params.get('adx_trend_threshold', 25)
            adx_ranging_threshold = market_regime_params.get('adx_ranging_threshold', 20)
            adx_div_max_adjust = market_regime_params.get('adx_div_weight_max_adjust', 0.3)
            adx_conf_max_adjust = market_regime_params.get('adx_conf_weight_max_adjust', 0.3)
            # 修正 normalize_score 的调用参数
            norm_adx = normalize_score(adx_val, 55)
            # 修改结束
            dynamic_div_weight_multiplier = 1 + norm_adx * adx_div_max_adjust
            dynamic_conf_weight_multiplier = dynamic_conf_weight_multiplier.mask(
                adx_val < adx_ranging_threshold, 1 + (1 - norm_adx) * adx_conf_max_adjust
            )
            dynamic_conf_weight_multiplier = dynamic_conf_weight_multiplier * (1 - norm_atr * market_regime_params.get('atr_conf_weight_max_adjust', 0.2))
            dynamic_conf_weight_multiplier = dynamic_conf_weight_multiplier.clip(0.5, 1.5)
        # 多维情境感知 (Multi-Dimensional Contextual Awareness)
        market_context_factor = pd.Series(1.0, index=df.index)
        if market_context_params.get('enabled'):
            favorable_sentiment_slope_threshold = market_context_params.get('favorable_sentiment_slope_threshold', 0.001)
            unfavorable_sentiment_slope_threshold = market_context_params.get('unfavorable_sentiment_slope_threshold', -0.001)
            slope_stability_threshold = market_context_params.get('slope_stability_threshold', 0.7)
            adx_strength_threshold = market_context_params.get('adx_strength_threshold', 25)
            favorable_context_bonus = market_context_params.get('favorable_context_bonus', 0.1)
            unfavorable_context_penalty = market_context_params.get('unfavorable_context_penalty', 0.1)
            is_favorable_sentiment = (robust_pct_change_slope > favorable_sentiment_slope_threshold)
            is_unfavorable_sentiment = (robust_pct_change_slope < unfavorable_sentiment_slope_threshold)
            is_healthy_trend = (adx_val > adx_strength_threshold) & ((1 - norm_slope_std_dev) > slope_stability_threshold)
            market_context_bonus_penalty = pd.Series(0.0, index=df.index)
            if is_bullish:
                market_context_bonus_penalty = market_context_bonus_penalty.mask(
                    is_favorable_sentiment & is_healthy_trend, favorable_context_bonus
                )
                market_context_bonus_penalty = market_context_bonus_penalty.mask(
                    is_unfavorable_sentiment & ~is_healthy_trend, -unfavorable_context_penalty
                )
            else: # Bearish
                market_context_bonus_penalty = market_context_bonus_penalty.mask(
                    is_unfavorable_sentiment & is_healthy_trend, favorable_context_bonus
                )
                market_context_bonus_penalty = market_context_bonus_penalty.mask(
                    is_favorable_sentiment & ~is_healthy_trend, -unfavorable_context_penalty
                )
            market_context_factor = (1 + market_context_bonus_penalty).clip(0.5, 1.5)
        # 信号新鲜度 (Signal Freshness)
        freshness_factor = pd.Series(1.0, index=df.index)
        if signal_freshness_params.get('enabled'):
            freshness_bonus = signal_freshness_params.get('freshness_bonus', 0.1)
            persistence_count = div_condition_raw.astype(int).rolling(window=max_persistence_window).apply(lambda x: (x == 1).sum(), raw=True).fillna(0) # [修改的代码行] 重新计算persistence_count
            freshness_factor = freshness_factor.mask(
                persistence_count == 1, 1 + freshness_bonus
            )
        freshness_factor = freshness_factor.clip(1, 1.5)
        # 行为一致性与冲突评估 (Behavioral Consistency and Conflict Assessment)
        consistency_factor = pd.Series(1.0, index=df.index)
        if behavioral_consistency_params.get('enabled'):
            min_consistent_indicators = behavioral_consistency_params.get('min_consistent_indicators_for_bonus', 2)
            conflict_penalty_threshold = behavioral_consistency_params.get('conflict_penalty_threshold', 1)
            consistency_bonus = behavioral_consistency_params.get('consistency_bonus', 0.1)
            conflict_penalty = behavioral_consistency_params.get('conflict_penalty', 0.15)
            diverging_indicators_count = rsi_indicator_trend.astype(int) + macd_indicator_trend.astype(int) + volume_indicator_trend.astype(int)
            conflicting_indicators_count = pd.Series(0, index=df.index)
            if is_bullish:
                conflicting_indicators_count += (~rsi_indicator_trend & (robust_rsi_slope < 0)).astype(int)
                conflicting_indicators_count += (~macd_indicator_trend & (robust_macd_slope < 0)).astype(int)
                conflicting_indicators_count += (~volume_indicator_trend & (robust_volume_slope < 0)).astype(int)
            else: # Bearish
                conflicting_indicators_count += (~rsi_indicator_trend & (robust_rsi_slope > 0)).astype(int)
                conflicting_indicators_count += (~macd_indicator_trend & (robust_macd_slope > 0)).astype(int)
                conflicting_indicators_count += (~volume_indicator_trend & (robust_volume_slope > 0)).astype(int)
            consistency_factor = consistency_factor.mask(
                diverging_indicators_count >= min_consistent_indicators, 1 + consistency_bonus
            )
            consistency_factor = consistency_factor.mask(
                (diverging_indicators_count < min_consistent_indicators) & (conflicting_indicators_count >= conflict_penalty_threshold),
                1 - conflict_penalty
            )
        consistency_factor = consistency_factor.clip(0.5, 1.5)
        # 行为模式序列识别 (Behavioral Pattern Sequence Recognition)
        pattern_sequence_factor = pd.Series(1.0, index=df.index)
        if pattern_sequence_params.get('enabled'):
            lookback_window = pattern_sequence_params.get('lookback_window', 3)
            volume_drying_up_ratio = pattern_sequence_params.get('volume_drying_up_ratio', 0.8)
            volume_climax_ratio = pattern_sequence_params.get('volume_climax_ratio', 1.5)
            reversal_pct_change_threshold = pattern_sequence_params.get('reversal_pct_change_threshold', 0.01)
            sequence_bonus = pattern_sequence_params.get('sequence_bonus', 0.2)
            price_trend_in_window = (pattern_close_slope < 0) if is_bullish else (pattern_close_slope > 0)
            volume_drying_up_in_window = (pattern_volume_slope < 0) & (current_volume < volume_avg * volume_drying_up_ratio)
            if is_bullish:
                current_bar_reversal = (pct_change_val > reversal_pct_change_threshold) & (current_volume > volume_avg * volume_climax_ratio) & (close_price > open_price)
            else: # Bearish
                current_bar_reversal = (pct_change_val < -reversal_pct_change_threshold) & (current_volume > volume_avg * volume_climax_ratio) & (close_price < open_price)
            is_pattern_sequence = price_trend_in_window & volume_drying_up_in_window & current_bar_reversal
            pattern_sequence_factor = pattern_sequence_factor.mask(is_pattern_sequence, 1 + sequence_bonus)
        pattern_sequence_factor = pattern_sequence_factor.clip(1, 1.5)
        # “行为惯性”评估 (Behavioral Inertia Assessment)
        inertia_factor = pd.Series(1.0, index=df.index)
        if behavioral_inertia_params.get('enabled'):
            long_term_adx_threshold = behavioral_inertia_params.get('long_term_adx_threshold', 30)
            long_term_slope_stability_threshold = behavioral_inertia_params.get('long_term_slope_stability_threshold', 0.8)
            high_inertia_penalty = behavioral_inertia_params.get('high_inertia_penalty', 0.1)
            low_inertia_bonus = behavioral_inertia_params.get('low_inertia_bonus', 0.05)
            long_term_adx_mean = adx_val.rolling(long_term_period).mean()
            is_strong_long_term_trend = (long_term_adx_mean > long_term_adx_threshold)
            long_term_close_slopes_series = self._get_safe_series(df, f'SLOPE_{long_term_period}_close_D', 0.0, method_name=method_name)
            long_term_slope_std_dev = long_term_close_slopes_series.rolling(window=long_term_period).std().fillna(0)
            # 修正 normalize_score 的调用参数
            norm_long_term_slope_std_dev = normalize_score(long_term_slope_std_dev, 55, ascending=False)
            # 修改结束
            is_stable_long_term_slope = (norm_long_term_slope_std_dev > long_term_slope_stability_threshold)
            is_high_inertia_market = is_strong_long_term_trend & is_stable_long_term_slope
            is_low_inertia_market = ~is_strong_long_term_trend | ~is_stable_long_term_slope
            inertia_adjustment = pd.Series(0.0, index=df.index)
            inertia_adjustment = inertia_adjustment.mask(is_high_inertia_market, -high_inertia_penalty)
            inertia_adjustment = inertia_adjustment.mask(is_low_inertia_market, low_inertia_bonus)
            inertia_factor = (1 + inertia_adjustment).clip(0.5, 1.5)
        # 自适应参数调整的规则引擎 (Adaptive Fusion Weights)
        adaptive_fusion_weight_multiplier = pd.Series(1.0, index=df.index)
        if behavioral_inertia_params.get('enabled'):
            trend_strong_penalty_factor = adaptive_fusion_weights_params.get('trend_strong_penalty_factor', 0.1)
            ranging_bonus_factor = adaptive_fusion_weights_params.get('ranging_bonus_factor', 0.1)
            volatility_high_penalty_factor = adaptive_fusion_weights_params.get('volatility_high_penalty_factor', 0.05)
            is_strong_trend = (adx_val > market_regime_params.get('adx_trend_threshold', 25))
            adaptive_fusion_weight_multiplier = adaptive_fusion_weight_multiplier.mask(
                is_strong_trend, adaptive_fusion_weight_multiplier * (1 - trend_strong_penalty_factor)
            )
            is_ranging_market = (adx_val < market_regime_params.get('adx_ranging_threshold', 20))
            adaptive_fusion_weight_multiplier = adaptive_fusion_weight_multiplier.mask(
                is_ranging_market, adaptive_fusion_weight_multiplier * (1 + ranging_bonus_factor)
            )
            # 修正 normalize_score 的调用参数
            is_high_volatility = (normalize_score(atr_val, 55) > 0.8)
            # 修改结束
            adaptive_fusion_weight_multiplier = adaptive_fusion_weight_multiplier.mask(
                is_high_volatility, adaptive_fusion_weight_multiplier * (1 - volatility_high_penalty_factor)
            )
        adaptive_fusion_weight_multiplier = adaptive_fusion_weight_multiplier.clip(0.5, 1.5)
        # 最终背离分数 (非线性融合)
        divergence_score = (
            (final_strength_factor + 1e-9).pow(0.4 * dynamic_div_weight_multiplier) *
            (persistence_factor + 1e-9).pow(0.2) *
            (norm_total_conf_factor + 1e-9).pow(0.3 * dynamic_conf_weight_multiplier) *
            (synergy_factor + 1e-9).pow(0.1) *
            (structural_context_factor + 1e-9).pow(0.1) *
            (resonance_factor + 1e-9).pow(0.1) *
            (price_action_factor + 1e-9).pow(0.1) *
            (volume_price_structure_factor + 1e-9).pow(0.1) *
            (purity_factor + 1e-9).pow(0.1) *
            (market_context_factor + 1e-9).pow(0.1) *
            (freshness_factor + 1e-9).pow(0.1) *
            (consistency_factor + 1e-9).pow(0.1) *
            (pattern_sequence_factor + 1e-9).pow(0.1) *
            (inertia_factor + 1e-9).pow(0.1)
        ).pow(1 / (2.2 * adaptive_fusion_weight_multiplier)).fillna(0.0).clip(0, 1)
        divergence_score = divergence_score.where(div_condition_raw, 0.0)
        return divergence_score.astype(np.float32)

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










