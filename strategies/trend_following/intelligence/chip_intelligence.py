# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V325.0 配置驱动最终版】筹码情报最高司令部
        - 核心重构:
          1. 【配置即代码】: 重构了 'bucket_upper' 信号的配置方式。现在，分级信号的每个等级都使用其【最终的、完整的信号名】作为配置字典的键，彻底消除了代码中的任何字符串拼接或特殊命名逻辑。
          2. 【逻辑纯化】: 信号生成循环的逻辑变得极其纯粹，仅负责从配置中读取信号名和阈值并执行计算，不再关心信号名的构造方式。
          3. 【极致可维护性】: 任何信号（包括复杂的分级信号）的命名、修改或增删，现在都100%在 MASTER_SIGNAL_CONFIG 中完成，代码逻辑保持绝对稳定。
        """
        print("        -> [筹码情报最高司令部 V325.0 配置驱动最终版] 启动...")
        states = {}
        triggers = {}
        # --- 步骤 1: 调用评分中心 ---
        df = self.diagnose_quantitative_chip_scores(df)
        p = get_params_block(self.strategy, 'chip_feature_params')
        if not get_param_value(p.get('enabled'), False):
            return states, triggers
        # --- 步骤 2: 定义主信号配置字典 ---
        # [修改] 'bucket_upper' 类型的配置方式被重构，以信号全称为键
        MASTER_SIGNAL_CONFIG = {
            # --- 司令部顶层信号 ---
            'CONTEXT_CHIP_STRATEGIC_GATHERING': ('CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING', 0.60, 120, 'state'),
            'CONTEXT_CHIP_STRATEGIC_DISTRIBUTION': ('CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING', 0.40, 120, 'state_lt'),
            'CONTEXT_EUPHORIC_RALLY_WARNING': ('CHIP_SCORE_CONTEXT_EUPHORIC_RALLY', 0.90, 120, 'state'),
            'TRIGGER_CHIP_IGNITION': ('CHIP_SCORE_TRIGGER_IGNITION', 0.98, 120, 'trigger'),
            'RISK_CONTEXT_LONG_TERM_DISTRIBUTION': ('CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION', 0.90, 120, 'state'),
            'RISK_CHIP_CONC_ACCEL_WORSENING': ('CHIP_SCORE_RISK_WORSENING_TURN', 0.80, 60, 'state_gt_zero'),
            # ... (其他信号配置保持不变，此处省略) ...
            'OPP_STATIC_DYN_BREAKTHROUGH_S': ('CHIP_SCORE_OPP_BREAKTHROUGH', 0.95, 120, 'state'),
            'RISK_STATIC_DYN_COLLAPSE_S': ('CHIP_SCORE_RISK_COLLAPSE', 0.95, 120, 'state'),
            'OPP_STATIC_DYN_INFLECTION_A': ('CHIP_SCORE_OPP_INFLECTION', 0.90, 120, 'state'),
            # --- 分级信号 (Bucket Signals) ---
            'BUCKET_CHIP_CONC_GATHERING': ('CHIP_SCORE_GATHERING_INTENSITY', {
                'CHIP_CONC_STEADY_GATHERING_C': 0.70,
                'CHIP_CONC_ACCELERATED_GATHERING_B': 0.85,
                'CHIP_CONC_INTENSIFYING_B_PLUS': 0.95
            }, 120, 'bucket_upper'),
            'BUCKET_RISK_BEHAVIOR_WINNERS_FLEEING': ('CHIP_SCORE_RISK_FLEEING_IN_HIGH_ZONE', {
                'RISK_BEHAVIOR_WINNERS_FLEEING_C': 0.60,
                'RISK_BEHAVIOR_WINNERS_FLEEING_B': 0.75,
                'RISK_BEHAVIOR_WINNERS_FLEEING_A': 0.90
            }, 120, 'bucket_upper'),
        }
        available_cols = set(df.columns)
        all_generated_states = {}
        # --- 步骤 3: 按 (评分列, 窗口, 信号大类) 对信号配置进行分组 ---
        from collections import defaultdict
        grouped_signals = defaultdict(list)
        for signal_name, (score_col, quantile_or_dict, window, signal_type) in MASTER_SIGNAL_CONFIG.items():
            if score_col in available_cols:
                group_key = 'gt_zero' if 'gt_zero' in signal_type else 'bucket_upper' if signal_type == 'bucket_upper' else 'standard'
                grouped_signals[(score_col, window, group_key)].append((signal_name, quantile_or_dict, signal_type))
        # --- 步骤 4: 批处理所有信号 ---
        for (score_col, window, group_key), tasks in grouped_signals.items():
            score = df[score_col]
            if group_key == 'bucket_upper':
                quantiles_needed = sorted(list(set(tasks[0][1].values())))
            else:
                quantiles_needed = sorted(list(set(q for _, q, _ in tasks)))
            thresholds_df = None
            if group_key in ['standard', 'bucket_upper']:
                if not score.isnull().all():
                    thresholds_df = score.rolling(window).quantile(quantiles_needed)
            elif group_key == 'gt_zero':
                positive_scores = score[score > 0]
                if not positive_scores.empty:
                    thresholds_df = positive_scores.rolling(window).quantile(quantiles_needed).reindex(score.index).ffill()
            if thresholds_df is None: continue
            if isinstance(thresholds_df, pd.Series):
                thresholds_df = thresholds_df.to_frame(name=quantiles_needed[0])
            # [修改] 逻辑分发，bucket_upper 的处理逻辑更简洁
            if group_key == 'bucket_upper':
                _, quantile_dict, _ = tasks[0] # 一个分组只有一个 bucket_upper 任务
                sorted_levels = sorted(quantile_dict.items(), key=lambda item: item[1])
                for i, (final_signal_name, q_lower) in enumerate(sorted_levels):
                    lower_thresh = thresholds_df[q_lower]
                    if i == len(sorted_levels) - 1:
                        signal = score > lower_thresh
                    else:
                        _, q_upper = sorted_levels[i+1]
                        upper_thresh = thresholds_df[q_upper]
                        signal = (score > lower_thresh) & (score <= upper_thresh)
                    all_generated_states[final_signal_name] = signal
            else: # 处理 standard 和 gt_zero
                for signal_name, quantile, signal_type in tasks:
                    threshold = thresholds_df[quantile]
                    signal = pd.Series(False, index=df.index)
                    if threshold.notna().any():
                        if signal_type in ['state', 'trigger', 'state_gt_zero']:
                            signal = score > threshold
                        elif signal_type == 'state_le':
                            signal = score <= threshold
                        elif signal_type == 'state_lt':
                            signal = score < threshold
                        elif signal_type == 'state_gt_zero_event':
                            signal = (score > threshold) & (score > 0)
                    if signal_type == 'trigger':
                        triggers[signal_name] = signal
                    else:
                        all_generated_states[signal_name] = signal
        # --- 步骤 5: 最终状态更新 ---
        states.update(all_generated_states)
        self.strategy.atomic_states.update(all_generated_states)
        return states, triggers

    def diagnose_quantitative_chip_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.4 最终性能版】筹码信号量化评分诊断模块
        - 核心优化:
          1. 【原子得分缓存】: 在函数开始时，将所有用到的原子得分（来自其他模块）一次性计算并缓存，避免在后续逻辑中反复调用和查找，减少开销。
          2. 【NumPy加速聚合】: 引入新的辅助方法 _max_of_series，使用 np.maximum.reduce 来高效计算多个Series的元素级最大值，取代较慢的 pd.concat().max()。
          3. 【继承优化】: 完整保留了V3.3版本的所有优点，包括批处理归一化、内存友好的source_series_map以及一次性合并结果的df.assign()。
        """
        print("        -> [筹码信号量化评分模块 V3.4 最终性能版] 启动...")
        # --- 阶段 1: 预计算衍生指标 ---
        derivative_calcs = {
            'SLOPE_5_concentration_90pct_D': ('concentration_90pct_D', 5), 'SLOPE_21_concentration_90pct_D': ('concentration_90pct_D', 21), 'SLOPE_55_concentration_90pct_D': ('concentration_90pct_D', 55),
            'ACCEL_5_concentration_90pct_D': ('SLOPE_5_concentration_90pct_D', 1), 'ACCEL_21_concentration_90pct_D': ('SLOPE_21_concentration_90pct_D', 1),
            'SLOPE_5_peak_cost_D': ('peak_cost_D', 5), 'ACCEL_5_peak_cost_D': ('SLOPE_5_peak_cost_D', 1),
            'SLOPE_5_turnover_from_winners_ratio_D': ('turnover_from_winners_ratio_D', 5), 'SLOPE_21_turnover_from_winners_ratio_D': ('turnover_from_winners_ratio_D', 21), 'SLOPE_55_turnover_from_winners_ratio_D': ('turnover_from_winners_ratio_D', 55),
            'ACCEL_5_turnover_from_winners_ratio_D': ('SLOPE_5_turnover_from_winners_ratio_D', 1),
            'SLOPE_5_turnover_from_losers_ratio_D': ('turnover_from_losers_ratio_D', 5), 'ACCEL_5_turnover_from_losers_ratio_D': ('SLOPE_5_turnover_from_losers_ratio_D', 1),
            'SLOPE_5_winner_profit_margin_D': ('winner_profit_margin_D', 5),
            'SLOPE_5_close_D': ('close_D', 5), 'SLOPE_21_close_D': ('close_D', 21),
            'ACCEL_5_close_D': ('SLOPE_5_close_D', 1), 'ACCEL_21_close_D': ('SLOPE_21_close_D', 1),
            'SLOPE_5_total_winner_rate_D': ('total_winner_rate_D', 5), 'ACCEL_5_total_winner_rate_D': ('SLOPE_5_total_winner_rate_D', 1),
            'SLOPE_5_chip_health_score_D': ('chip_health_score_D', 5), 'SLOPE_21_chip_health_score_D': ('chip_health_score_D', 21),
            'SLOPE_55_EMA_55_D': ('EMA_55_D', 55),
        }
        available_cols = set(df.columns)
        for new_col, (base_col, period) in derivative_calcs.items():
            if base_col in available_cols and new_col not in available_cols:
                source_series = df.get(base_col)
                if source_series is not None:
                    df[new_col] = source_series.diff(period)
        def _get_atomic_score(name: str, default: float = 0.5) -> pd.Series:
            return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))
        # --- 阶段 1.1: 缓存原子得分 ---
        atomic_score_keys = [
            'COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', 'BEHAVIOR_SCORE_OPP_WASHOUT_ABSORPTION',
            'SCORE_MA_STATE_BOTTOM_PASSIVATION', 'SCORE_STRUCTURE_EARLY_REVERSAL',
            'COGNITIVE_SCORE_TREND_STAGE_EARLY'
        ]
        cached_atomic_scores = {key: _get_atomic_score(key) for key in atomic_score_keys}
        # --- 阶段 1.5: 构建源数据序列字典 (source_series_map) ---
        source_series_map = {col: df[col] for col in df.columns}
        if 'peak_strength_ratio_D' in source_series_map:
            source_series_map['temp_peak_strength_ratio_D_filled'] = source_series_map['peak_strength_ratio_D'].fillna(0)
        if 'SLOPE_5_peak_cost_D' in source_series_map:
            source_series_map['temp_abs_SLOPE_5_peak_cost_D'] = source_series_map['SLOPE_5_peak_cost_D'].abs()
        if 'concentration_90pct_D' in source_series_map:
            source_series_map['temp_conc_worsened_ratio'] = (source_series_map['concentration_90pct_D'] / source_series_map['concentration_90pct_D'].shift(21).replace(0, np.nan)).fillna(1)
        if 'volume_D' in source_series_map and 'VOL_MA_21_D' in source_series_map:
            source_series_map['temp_vol_ratio'] = source_series_map['volume_D'] / source_series_map['VOL_MA_21_D']
        if 'pct_change_D' in source_series_map:
            source_series_map['temp_abs_pct_change_D'] = source_series_map['pct_change_D'].abs()
        if 'peak_cost_D' in source_series_map:
            source_series_map['temp_peak_change_pct'] = source_series_map['peak_cost_D'].pct_change().abs().fillna(0)
            source_series_map['temp_peak_cost_volatility'] = (source_series_map['peak_cost_D'].rolling(5).std() / source_series_map['peak_cost_D'].rolling(5).mean()).fillna(1)
            source_series_map['temp_avg_instability'] = source_series_map['temp_peak_change_pct'].rolling(10, min_periods=3).mean().fillna(1)
        if 'SLOPE_55_EMA_55_D' in source_series_map:
            source_series_map['temp_prior_trend_slope'] = source_series_map['SLOPE_55_EMA_55_D'].shift(1).fillna(0)
        if 'ACCEL_21_concentration_90pct_D' in source_series_map:
            source_series_map['temp_worsening_intensity'] = source_series_map['ACCEL_21_concentration_90pct_D'].clip(lower=0)
        if 'chip_fault_strength_D' in source_series_map:
            source_series_map['temp_fault_strength'] = source_series_map['chip_fault_strength_D'].fillna(0)
        if 'chip_fault_vacuum_percent_D' in source_series_map:
            source_series_map['temp_vacuum_clearance'] = source_series_map['chip_fault_vacuum_percent_D'].fillna(100)
        if 'pct_change_D' in source_series_map:
            source_series_map['temp_price_resilience'] = source_series_map['pct_change_D'].clip(lower=-0.02)
        if 'close_D' in source_series_map and 'EMA_55_D' in source_series_map:
            source_series_map['temp_close_ema_ratio'] = source_series_map['close_D'] / source_series_map['EMA_55_D']
        if 'SLOPE_21_close_D' in source_series_map:
            source_series_map['SLOPE_21_close_D_shifted'] = source_series_map['SLOPE_21_close_D'].shift(1).fillna(0)
        if 'concentration_90pct_D' in source_series_map and 'concentration_70pct_D' in source_series_map:
            source_series_map['concentration_gap'] = source_series_map['concentration_90pct_D'] - source_series_map['concentration_70pct_D']
        # --- 阶段 1.6: 定义归一化任务总配置 ---
        NORMALIZATION_CONFIG = {
            'score_5d': ('SLOPE_5_concentration_90pct_D', 120, False), 'score_21d': ('SLOPE_21_concentration_90pct_D', 120, False), 'score_55d': ('SLOPE_55_concentration_90pct_D', 120, False),
            'cost_support_momentum': ('SLOPE_5_peak_cost_D', 120, True), 'trigger_ignition': ('ACCEL_5_peak_cost_D', 120, True),
            'control_score': ('peak_control_ratio_D', 120, True), 'strength_score': ('peak_strength_ratio_D', 120, True), 'single_peak_purity_score': ('temp_peak_strength_ratio_D_filled', 120, False),
            'divergence_risk': ('SLOPE_5_concentration_90pct_D', 120, True), 'profit_taking_risk': ('SLOPE_5_turnover_from_winners_ratio_D', 120, True),
            'pressure_5d': ('SLOPE_5_turnover_from_winners_ratio_D', 120, True), 'pressure_21d': ('SLOPE_21_turnover_from_winners_ratio_D', 120, True), 'pressure_55d': ('SLOPE_55_turnover_from_winners_ratio_D', 120, True),
            'selling_exhaustion': ('ACCEL_5_turnover_from_winners_ratio_D', 120, False), 'sharp_drop': ('pct_change_D', 60, False), 'loser_turnover': ('turnover_from_losers_ratio_D', 120, True),
            'profit_cushion': ('winner_profit_margin_D', 120, True), 'absorption_intensity': ('peak_absorption_intensity_D', 120, True),
            'price_momentum': ('price_to_peak_ratio_D', 120, True), 'price_below_peak': ('price_to_peak_ratio_D', 120, False), 'compactness': ('concentration_gap', 120, False),
            'profit_margin_rising': ('SLOPE_5_winner_profit_margin_D', 120, True), 'profit_margin_shrinking': ('SLOPE_5_winner_profit_margin_D', 120, False), 'cost_falling': ('SLOPE_5_peak_cost_D', 120, False),
            'concentration_norm': ('concentration_90pct_D', 120, False), 'cost_stability': ('temp_abs_SLOPE_5_peak_cost_D', 120, False),
            'profit_taking_intensity': ('turnover_from_winners_ratio_D', 120, True), 'loser_capitulation_intensity': ('turnover_from_losers_ratio_D', 120, True),
            'despair_zone': ('price_to_peak_ratio_D', 120, False), 'bleeding_stopping': ('ACCEL_5_turnover_from_losers_ratio_D', 120, False),
            'stealth_buying': ('SLOPE_5_concentration_90pct_D', 120, False), 'price_turning': ('SLOPE_5_close_D', 120, True),
            'absorption_accel': ('ACCEL_5_concentration_90pct_D', 120, False), 'losers_giving_up': ('SLOPE_5_turnover_from_losers_ratio_D', 120, False),
            'euphoria': ('total_winner_rate_D', 120, True), 'winner_selling_accel': ('ACCEL_5_turnover_from_winners_ratio_D', 120, True),
            'chip_diverging': ('SLOPE_5_concentration_90pct_D', 120, True), 'price_momentum_fading': ('ACCEL_5_close_D', 120, False),
            'price_accel': ('ACCEL_5_close_D', 120, True), 'mid_term_price_trend': ('SLOPE_21_close_D', 120, True),
            'short_term_concentration': ('SLOPE_5_concentration_90pct_D', 120, False), 'short_term_price_trend': ('SLOPE_5_close_D', 120, True),
            'mid_term_divergence': ('SLOPE_21_concentration_90pct_D', 120, True), 'short_term_divergence': ('SLOPE_5_concentration_90pct_D', 120, True),
            'mid_term_price_accel': ('ACCEL_21_close_D', 120, True), 'prior_downtrend_strength': ('SLOPE_21_close_D_shifted', 60, False),
            'new_high': ('close_D', 60, True), 'cost_accelerating': ('ACCEL_5_peak_cost_D', 120, True),
            'battle_intensity': ('turnover_at_peak_ratio_D', 120, True), 'high_volume': ('temp_vol_ratio', 120, True),
            'price_rising_meaningfully': ('pct_change_D', 120, True), 'price_stagnant': ('temp_abs_pct_change_D', 120, False),
            'concentration_accel': ('ACCEL_5_concentration_90pct_D', 120, False), 'conc_worsened': ('temp_conc_worsened_ratio', 120, True),
            'slope_21d_diverging': ('SLOPE_21_concentration_90pct_D', 120, True), 'bailout_failure_sub_score': ('SLOPE_5_concentration_90pct_D', 120, True),
            'cost_constructive': ('SLOPE_5_peak_cost_D', 60, True), 'winner_rate_rising': ('SLOPE_5_total_winner_rate_D', 120, True),
            'winner_rate_collapse': ('SLOPE_5_total_winner_rate_D', 120, False), 'winner_rate_accel_collapse': ('ACCEL_5_total_winner_rate_D', 120, False),
            'health_improving': ('SLOPE_5_chip_health_score_D', 120, True), 'health_deteriorating': ('SLOPE_5_chip_health_score_D', 120, False),
            'uptrend_context': ('temp_close_ema_ratio', 120, True), 'peak_change_intensity': ('temp_peak_change_pct', 60, True),
            'formation_stability': ('temp_peak_cost_volatility', 60, False), 'formation_duration': ('temp_avg_instability', 60, False),
            'high_volume_60d': ('temp_vol_ratio', 60, True), 'prior_downtrend': ('temp_prior_trend_slope', 60, False), 'prior_uptrend': ('temp_prior_trend_slope', 60, True),
            'diverging_short': ('SLOPE_5_concentration_90pct_D', 120, True), 'worsening_turn': ('temp_worsening_intensity', 120, True),
            'fleeing_b': ('CHIP_SCORE_BEHAVIOR_PROFIT_TAKING', 120, True), 'fault_strength': ('temp_fault_strength', 120, True),
            'vacuum_clearance': ('temp_vacuum_clearance', 120, False), 'pressure': ('turnover_from_winners_ratio_D', 120, True),
            'resilience': ('temp_price_resilience', 60, True), 'new_high_120d': ('close_D', 120, True),
            'conc_worsening_21d': ('SLOPE_21_concentration_90pct_D', 120, False), 'health_worsening_21d': ('SLOPE_21_chip_health_score_D', 120, False),
            'high_winner_rate': ('total_winner_rate_D', 120, True), 'high_profit_margin': ('winner_profit_margin_D', 120, True),
            'risk_sub_score_1': ('SLOPE_5_concentration_90pct_D', 120, True), 'risk_sub_score_2': ('SLOPE_5_peak_cost_D', 120, False),
            'risk_sub_score_3': ('SLOPE_5_total_winner_rate_D', 120, False), 'risk_sub_score_4': ('SLOPE_5_chip_health_score_D', 120, False),
        }
        # --- 阶段 1.7: 批处理执行所有归一化计算 ---
        from collections import defaultdict
        grouped_calcs = defaultdict(list)
        for score_name, (source_col, window, ascending) in NORMALIZATION_CONFIG.items():
            if source_col in source_series_map:
                grouped_calcs[(source_col, window)].append((score_name, ascending))
        precomputed_scores = {}
        for (source_col, window), tasks in grouped_calcs.items():
            source_series = source_series_map[source_col]
            rolling_obj = source_series.rolling(window=window, min_periods=max(1, window // 5))
            for score_name, ascending in tasks:
                precomputed_scores[score_name] = rolling_obj.rank(pct=True, ascending=ascending).fillna(0.5)
        # --- 阶段 2-8: 业务逻辑计算，结果存入 new_scores 字典 ---
        new_scores = {}
        default_series = pd.Series(0.5, index=df.index)
        new_scores['CHIP_SCORE_CONCENTRATION_RESONANCE'] = (precomputed_scores.get('score_5d', default_series) * 0.5 + precomputed_scores.get('score_21d', default_series) * 0.3 + precomputed_scores.get('score_55d', default_series) * 0.2)
        new_scores['CHIP_SCORE_CONCENTRATION_DIVERGENCE'] = precomputed_scores.get('score_5d', default_series) - precomputed_scores.get('score_55d', default_series)
        new_scores['CHIP_SCORE_CONCENTRATION_MOMENTUM'] = precomputed_scores.get('score_5d', default_series)
        new_scores['CHIP_SCORE_CONTEXT_STRATEGIC_GATHERING'] = precomputed_scores.get('score_55d', default_series)
        new_scores['CHIP_SCORE_COST_SUPPORT_MOMENTUM'] = precomputed_scores.get('cost_support_momentum', default_series)
        new_scores['CHIP_SCORE_TRIGGER_IGNITION'] = precomputed_scores.get('trigger_ignition', default_series)
        new_scores['CHIP_SCORE_STRUCTURE_STABILITY'] = 0.3 * precomputed_scores.get('control_score', default_series) + 0.3 * precomputed_scores.get('strength_score', default_series) + 0.4 * precomputed_scores.get('single_peak_purity_score', default_series)
        new_scores['CHIP_SCORE_STRUCTURE_FORTRESS'] = precomputed_scores.get('control_score', default_series) * precomputed_scores.get('strength_score', default_series) * precomputed_scores.get('single_peak_purity_score', default_series)
        new_scores['CHIP_SCORE_DISTRIBUTION_RISK'] = (precomputed_scores.get('divergence_risk', default_series) + precomputed_scores.get('profit_taking_risk', default_series)) / 2
        new_scores['CHIP_SCORE_BEHAVIOR_PROFIT_TAKING'] = (precomputed_scores.get('pressure_5d', default_series) * 0.5 + precomputed_scores.get('pressure_21d', default_series) * 0.3 + precomputed_scores.get('pressure_55d', default_series) * 0.2)
        new_scores['CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION'] = precomputed_scores.get('selling_exhaustion', default_series)
        new_scores['CHIP_SCORE_BEHAVIOR_PANIC_CAPITULATION'] = precomputed_scores.get('loser_turnover', default_series) * precomputed_scores.get('sharp_drop', default_series)
        new_scores['CHIP_SCORE_OPP_PROFIT_CUSHION'] = precomputed_scores.get('profit_cushion', default_series)
        new_scores['CHIP_SCORE_OPP_ABSORPTION_INTENSITY'] = precomputed_scores.get('absorption_intensity', default_series)
        new_scores['CHIP_SCORE_STATIC_PRICE_MOMENTUM'] = precomputed_scores.get('price_momentum', default_series)
        new_scores['CHIP_SCORE_STATIC_PRICE_BELOW_PEAK'] = precomputed_scores.get('price_below_peak', default_series)
        new_scores['CHIP_SCORE_STATIC_COMPACTNESS'] = precomputed_scores.get('compactness', default_series)
        if 'price_to_peak_ratio_D' in source_series_map:
            proximity = (1 - (source_series_map['price_to_peak_ratio_D'] - 1).abs()).clip(lower=0)
            new_scores['CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY'] = proximity
            new_scores['CHIP_SCORE_SCENARIO_MAIN_WAVE_RESONANCE'] = new_scores.get('CHIP_SCORE_STRUCTURE_STABILITY', default_series) * new_scores.get('CHIP_SCORE_CONCENTRATION_RESONANCE', default_series) * new_scores.get('CHIP_SCORE_COST_SUPPORT_MOMENTUM', default_series) * precomputed_scores.get('profit_margin_rising', default_series)
            new_scores['CHIP_SCORE_SCENARIO_BATTLEZONE_TURNING_POINT'] = proximity * new_scores.get('CHIP_SCORE_CONCENTRATION_DIVERGENCE', 0).clip(lower=0) * new_scores.get('CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION', default_series)
            new_scores['CHIP_SCORE_SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP'] = new_scores.get('CHIP_SCORE_STATIC_PRICE_MOMENTUM', default_series) * (1 - new_scores.get('CHIP_SCORE_CONCENTRATION_RESONANCE', default_series)) * precomputed_scores.get('profit_margin_shrinking', default_series)
            new_scores['CHIP_SCORE_SCENARIO_FORTRESS_INTERNAL_COLLAPSE'] = new_scores.get('CHIP_SCORE_STRUCTURE_STABILITY', default_series) * (-new_scores.get('CHIP_SCORE_CONCENTRATION_DIVERGENCE', 0)).clip(lower=0) * precomputed_scores.get('cost_falling', default_series)
        new_scores['CHIP_SCORE_STRUCTURE_LOCKED_STABLE'] = precomputed_scores.get('concentration_norm', default_series) * precomputed_scores.get('cost_stability', default_series)
        new_scores['CHIP_SCORE_BEHAVIOR_PROFIT_TAKING_INTENSITY'] = precomputed_scores.get('profit_taking_intensity', default_series)
        new_scores['CHIP_SCORE_BEHAVIOR_LOSER_CAPITULATION_INTENSITY'] = precomputed_scores.get('loser_capitulation_intensity', default_series)
        new_scores['CHIP_SCORE_OPP_DEEP_WATER_REVERSAL'] = precomputed_scores.get('despair_zone', default_series) * precomputed_scores.get('bleeding_stopping', default_series) * precomputed_scores.get('stealth_buying', default_series) * precomputed_scores.get('price_turning', default_series)
        new_scores['CHIP_SCORE_OPP_ACCUMULATION_CONFIRMED'] = new_scores.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * precomputed_scores.get('absorption_accel', default_series) * precomputed_scores.get('losers_giving_up', default_series)
        new_scores['CHIP_SCORE_RISK_EUPHORIA_TRAP'] = precomputed_scores.get('euphoria', default_series) * precomputed_scores.get('winner_selling_accel', default_series) * precomputed_scores.get('chip_diverging', default_series) * precomputed_scores.get('price_momentum_fading', default_series)
        new_scores['CHIP_SCORE_OPP_BREAKTHROUGH'] = new_scores.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * precomputed_scores.get('price_accel', default_series) * precomputed_scores.get('mid_term_price_trend', default_series) * precomputed_scores.get('short_term_concentration', default_series)
        new_scores['CHIP_SCORE_RISK_COLLAPSE'] = new_scores.get('CHIP_SCORE_RISK_EUPHORIA_TRAP', 0) * precomputed_scores.get('short_term_price_trend', default_series) * precomputed_scores.get('mid_term_divergence', default_series) * precomputed_scores.get('short_term_divergence', default_series)
        new_scores['CHIP_SCORE_OPP_INFLECTION'] = new_scores.get('CHIP_SCORE_OPP_DEEP_WATER_REVERSAL', 0) * precomputed_scores.get('mid_term_price_accel', default_series) * precomputed_scores.get('short_term_price_trend', default_series) * precomputed_scores.get('prior_downtrend_strength', default_series)
        new_scores['CHIP_SCORE_PRIME_OPPORTUNITY'] = new_scores.get('CHIP_SCORE_STRUCTURE_LOCKED_STABLE', default_series) * cached_atomic_scores['COGNITIVE_SCORE_TREND_STAGE_EARLY']
        new_scores['CHIP_SCORE_RISK_PRICE_DIVERGENCE'] = precomputed_scores.get('new_high', default_series) * precomputed_scores.get('short_term_divergence', default_series)
        new_scores['CHIP_SCORE_OPP_LOCKED_IGNITION'] = new_scores.get('CHIP_SCORE_STRUCTURE_LOCKED_STABLE', default_series) * precomputed_scores.get('cost_accelerating', default_series)
        battle_setup_score = new_scores.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * precomputed_scores.get('battle_intensity', default_series) * precomputed_scores.get('high_volume', default_series)
        had_recent_setup_score = battle_setup_score.shift(1).rolling(window=3, min_periods=1).max().fillna(0)
        new_scores['CHIP_SCORE_OPP_PEAK_BATTLE_BREAKOUT'] = had_recent_setup_score * precomputed_scores.get('price_rising_meaningfully', default_series)
        new_scores['CHIP_SCORE_RISK_PEAK_BATTLE_DISTRIBUTION'] = battle_setup_score * precomputed_scores.get('price_stagnant', default_series)
        new_scores['CHIP_SCORE_GATHERING_INTENSITY'] = new_scores.get('CHIP_SCORE_CONCENTRATION_MOMENTUM', default_series) * precomputed_scores.get('concentration_accel', default_series)
        new_scores['CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION'] = ((precomputed_scores.get('conc_worsened', default_series) + precomputed_scores.get('slope_21d_diverging', default_series)) / 2) * cached_atomic_scores['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE']
        new_scores['CHIP_SCORE_OPP_BREAKOUT_PRECURSOR'] = new_scores.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * new_scores.get('CHIP_SCORE_CONCENTRATION_RESONANCE', default_series)
        new_scores['CHIP_SCORE_RISK_DISTRIBUTION_CONFIRMED'] = new_scores.get('CHIP_SCORE_STATIC_PRICE_MOMENTUM', default_series) * (1 - new_scores.get('CHIP_SCORE_CONCENTRATION_RESONANCE', default_series))
        new_scores['CHIP_SCORE_RISK_BAILOUT_FAILURE'] = new_scores.get('CHIP_SCORE_SCENARIO_BATTLEZONE_PROXIMITY', 0) * precomputed_scores.get('bailout_failure_sub_score', default_series)
        breakout_candle_score = precomputed_scores.get('price_rising_meaningfully', default_series) * precomputed_scores.get('high_volume', default_series)
        new_scores['CHIP_SCORE_OPP_LOCKED_BREAKOUT'] = new_scores.get('CHIP_SCORE_STRUCTURE_LOCKED_STABLE', default_series) * breakout_candle_score
        concentrating_score_1 = new_scores.get('CHIP_SCORE_CONCENTRATION_MOMENTUM', default_series) * precomputed_scores.get('cost_constructive', default_series)
        concentrating_score_2 = cached_atomic_scores['BEHAVIOR_SCORE_OPP_WASHOUT_ABSORPTION']
        concentrating_score_3 = new_scores.get('CHIP_SCORE_CONCENTRATION_MOMENTUM', default_series) * cached_atomic_scores['SCORE_MA_STATE_BOTTOM_PASSIVATION']
        new_scores['CHIP_SCORE_DYN_CONCENTRATING'] = self._max_of_series(concentrating_score_1, concentrating_score_2, concentrating_score_3)
        new_scores['CHIP_SCORE_DYN_DIVERGING'] = (1 - new_scores.get('CHIP_SCORE_CONCENTRATION_MOMENTUM', default_series)) * cached_atomic_scores['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE']
        new_scores['CHIP_SCORE_DYN_S_ACCEL_CONCENTRATING'] = new_scores.get('CHIP_SCORE_DYN_CONCENTRATING', default_series) * precomputed_scores.get('concentration_accel', default_series)
        new_scores['CHIP_SCORE_DYN_ACCEL_DIVERGING'] = precomputed_scores.get('absorption_accel', default_series) * cached_atomic_scores['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE']
        new_scores['CHIP_SCORE_DYN_COST_RISING'] = new_scores.get('CHIP_SCORE_COST_SUPPORT_MOMENTUM', default_series)
        new_scores['CHIP_SCORE_DYN_COST_ACCELERATING'] = new_scores.get('CHIP_SCORE_TRIGGER_IGNITION', default_series)
        new_scores['CHIP_SCORE_DYN_COST_FALLING'] = precomputed_scores.get('cost_falling', default_series)
        new_scores['CHIP_SCORE_DYN_WINNER_RATE_RISING'] = precomputed_scores.get('winner_rate_rising', default_series)
        new_scores['CHIP_SCORE_DYN_WINNER_RATE_COLLAPSING'] = precomputed_scores.get('winner_rate_collapse', default_series)
        new_scores['CHIP_SCORE_DYN_WINNER_RATE_ACCEL_COLLAPSING'] = precomputed_scores.get('winner_rate_accel_collapse', default_series)
        new_scores['CHIP_SCORE_DYN_HEALTH_IMPROVING'] = precomputed_scores.get('health_improving', default_series) * new_scores.get('CHIP_SCORE_DYN_CONCENTRATING', default_series)
        new_scores['CHIP_SCORE_DYN_HEALTH_DETERIORATING'] = precomputed_scores.get('health_deteriorating', default_series) * new_scores.get('CHIP_SCORE_DYN_DIVERGING', default_series)
        new_scores['CHIP_SCORE_CONTEXT_UPTREND'] = precomputed_scores.get('uptrend_context', default_series).clip(lower=0)
        new_scores['CHIP_SCORE_CONTEXT_SAFE'] = self._max_of_series(cached_atomic_scores['SCORE_MA_STATE_BOTTOM_PASSIVATION'], cached_atomic_scores['SCORE_STRUCTURE_EARLY_REVERSAL'])
        formation_confidence_score = precomputed_scores.get('peak_change_intensity', default_series) * precomputed_scores.get('formation_stability', default_series) * precomputed_scores.get('formation_duration', default_series)
        new_scores['PEAK_SCORE_OPP_FORTRESS_SUPPORT'] = formation_confidence_score * precomputed_scores.get('high_volume_60d', default_series) * precomputed_scores.get('prior_downtrend', default_series)
        new_scores['PEAK_SCORE_RISK_EXHAUSTION_TOP'] = formation_confidence_score * precomputed_scores.get('high_volume_60d', default_series) * precomputed_scores.get('prior_uptrend', default_series)
        new_scores['PEAK_SCORE_OPP_STEALTH_ACCUMULATION'] = formation_confidence_score * (1 - precomputed_scores.get('high_volume_60d', default_series)) * precomputed_scores.get('prior_downtrend', default_series) * new_scores.get('CHIP_SCORE_CONTEXT_SAFE', default_series)
        new_scores['CHIP_SCORE_RISK_PEAK_BATTLE_DISTRIBUTION_IN_ZONE'] = new_scores.get('CHIP_SCORE_RISK_PEAK_BATTLE_DISTRIBUTION', default_series) * cached_atomic_scores['COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE']
        new_scores['CHIP_SCORE_SCENARIO_WASHOUT_BELOW_FORTRESS'] = new_scores.get('CHIP_SCORE_STATIC_COMPACTNESS', default_series) * precomputed_scores.get('diverging_short', default_series) * new_scores.get('CHIP_SCORE_CONCENTRATION_RESONANCE', default_series)
        new_scores['CHIP_SCORE_RISK_WORSENING_TURN'] = precomputed_scores.get('worsening_turn', default_series)
        new_scores['CHIP_SCORE_RISK_PANIC_FLEEING'] = precomputed_scores.get('fleeing_b', default_series) * (1 - new_scores.get('CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION', default_series))
        new_scores['CHIP_SCORE_OPP_FAULT_REBIRTH'] = precomputed_scores.get('fault_strength', default_series) * precomputed_scores.get('vacuum_clearance', default_series) * new_scores.get('CHIP_SCORE_CONTEXT_UPTREND', default_series)
        new_scores['CHIP_SCORE_BEHAVIOR_ABSORBING_PRESSURE'] = precomputed_scores.get('pressure', default_series) * precomputed_scores.get('resilience', default_series)
        new_scores['CHIP_SCORE_RISK_MULTI_DIVERGENCE_WEAKNESS'] = precomputed_scores.get('new_high_120d', default_series) * precomputed_scores.get('conc_worsening_21d', default_series) * precomputed_scores.get('health_worsening_21d', default_series)
        new_scores['CHIP_SCORE_CONTEXT_EUPHORIC_RALLY'] = precomputed_scores.get('high_winner_rate', default_series) * precomputed_scores.get('high_profit_margin', default_series)
        new_scores['CHIP_SCORE_OPP_INTENSE_ABSORPTION'] = new_scores.get('CHIP_SCORE_OPP_ABSORPTION_INTENSITY', default_series) * new_scores.get('CHIP_SCORE_DYN_CONCENTRATING', default_series)
        high_zone_score = cached_atomic_scores.get('COGNITIVE_SCORE_RISK_HIGH_LEVEL_ZONE', pd.Series(0.0, index=df.index)).astype(float)
        new_scores['CHIP_SCORE_RISK_FLEEING_IN_HIGH_ZONE'] = new_scores.get('CHIP_SCORE_BEHAVIOR_PROFIT_TAKING', default_series) * high_zone_score
        new_scores['CHIP_SCORE_OPP_SELLING_EXHAUSTION_SAFE'] = new_scores.get('CHIP_SCORE_BEHAVIOR_SELLING_EXHAUSTION', default_series) * new_scores.get('CHIP_SCORE_CONTEXT_SAFE', default_series)
        new_scores['CHIP_SCORE_OPP_PANIC_CAPITULATION_SAFE'] = new_scores.get('CHIP_SCORE_BEHAVIOR_PANIC_CAPITULATION', default_series) * new_scores.get('CHIP_SCORE_CONTEXT_SAFE', default_series)
        new_scores['CHIP_SCORE_EUPHORIA'] = precomputed_scores.get('euphoria', default_series)
        # --- 阶段 9: 终极风险裁定 ---
        risk_series_list = [
            precomputed_scores.get('risk_sub_score_1'), precomputed_scores.get('risk_sub_score_2'),
            precomputed_scores.get('risk_sub_score_3'), precomputed_scores.get('risk_sub_score_4')
        ]
        risk_score_cols = [
            'CHIP_SCORE_RISK_LONG_TERM_DISTRIBUTION', 'CHIP_SCORE_RISK_WORSENING_TURN', 'CHIP_SCORE_RISK_PANIC_FLEEING',
            'CHIP_SCORE_RISK_PRICE_DIVERGENCE', 'CHIP_SCORE_RISK_DISTRIBUTION_CONFIRMED', 'CHIP_SCORE_RISK_BAILOUT_FAILURE',
            'CHIP_SCORE_SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP', 'CHIP_SCORE_SCENARIO_FORTRESS_INTERNAL_COLLAPSE'
        ]
        for col in risk_score_cols:
            if col in new_scores:
                risk_series_list.append(new_scores[col])
        risk_series_list = [s for s in risk_series_list if s is not None]
        if risk_series_list:
            risk_matrix = pd.concat(risk_series_list, axis=1)
            new_scores['CHIP_SCORE_RISK_CRITICAL_FAILURE'] = risk_matrix.max(axis=1)
        # --- 阶段 10: 一次性将所有新得分合并到DataFrame ---
        df = df.assign(**new_scores)
        print("        -> [筹码信号量化评分模块 V3.4] 计算完毕。")
        return df

    def _calculate_normalized_score(self, series: pd.Series, window: int, ascending: bool = True) -> pd.Series:
        """
        【V2.0 性能优化版】计算滚动归一化得分的辅助函数。
        - 核心: 使用滚动分位数排名，将一个指标转换为0-1之间的得分。
        - 优化: 直接利用rank函数的`ascending`参数，移除if/else分支和额外的减法运算，代码更简洁高效。
        :param series: 原始数据Series。
        :param window: 滚动窗口大小。
        :param ascending: 排序方向。True表示值越大得分越高，False反之。
        :return: 归一化后的得分Series。
        """
        # min_periods 设为窗口的20%，与项目中其他模块保持一致。
        return series.rolling(
            window=window, 
            min_periods=max(1, window // 5) # 确保 min_periods 至少为1
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5)

    def _max_of_series(self, *series: pd.Series) -> pd.Series:
        """
        【V1.0 高性能版】计算多个pandas Series的元素级最大值。
        - 核心优化: 使用 np.maximum.reduce，它在NumPy数组上直接操作，
          比 pd.concat([...]).max(axis=1) 更快，因为它避免了创建中间DataFrame的开销。
        :param series: 一个或多个pandas Series。
        :return: 一个新的Series，其每个元素是输入Series对应位置元素的最大值。
        """
        # 过滤掉所有为None的Series
        valid_series = [s for s in series if s is not None]
        if not valid_series:
            # 如果没有有效的Series，返回一个空的Series
            return pd.Series(dtype=np.float64)
        # 将所有Series的值提取为NumPy数组列表
        arrays = [s.values for s in valid_series]
        # 使用np.maximum.reduce高效计算元素级最大值
        max_values = np.maximum.reduce(arrays)
        # 将结果包装回一个带有原始索引的pandas Series
        return pd.Series(max_values, index=valid_series[0].index)


