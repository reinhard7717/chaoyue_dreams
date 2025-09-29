# 文件: strategies/trend_following/intelligence/chip_intelligence.py
# 筹码情报模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, calculate_context_scores

class ChipIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化筹码情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]: # 修正返回类型注解
        """
        【V500.1 · 协议统一版】筹码情报最高司令部
        - 核心修复: 修正了方法签名，确保返回一个单一的字典，而不是元组，以修复与IntelligenceLayer的数据流中断问题。
        """
        # print("        -> [筹码情报最高司令部 V500.1 · 协议统一版] 启动...") # 更新版本号
        
        all_chip_states = {}
        
        # 步骤 1: 执行唯一的、统一的终极信号引擎
        unified_states = self.diagnose_unified_chip_signals(df)
        all_chip_states.update(unified_states)

        # 步骤 2: 执行具有特殊战术意义的“剧本”诊断模块 (作为补充)
        accumulation_states = self.diagnose_accumulation_playbooks(df)
        all_chip_states.update(accumulation_states)

        # 步骤 3: 执行独立的“恐慌投降”原子状态与剧本诊断 (作为补充)
        setup_states = self._diagnose_setup_capitulation_ready(df)
        all_chip_states.update(setup_states)
        
        # 为了让下游的 _synthesize_playbook_capitulation_reversal 能消费到最新的信号，
        # 我们需要临时将当前状态合并到 df 中。这是一个可以未来优化的点。
        temp_df_for_playbook = df.assign(**all_chip_states)
        
        trigger_states = self._diagnose_trigger_capitulation_fire(temp_df_for_playbook)
        all_chip_states.update(trigger_states)
        
        temp_df_for_playbook = temp_df_for_playbook.assign(**trigger_states)
        
        playbook_states = self._synthesize_playbook_capitulation_reversal(temp_df_for_playbook)
        all_chip_states.update(playbook_states)

        # print(f"        -> [筹码情报最高司令部 V500.1] 分析完毕，共生成 {len(all_chip_states)} 个筹码信号。")
        
        # 只返回包含所有状态的单一字典
        return all_chip_states

    def diagnose_unified_chip_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V13.0 · 动态分统一版】统一筹码信号诊断引擎
        - 核心重构: 彻底统一动态分哲学。废除 d_bull 和 d_bear，统一使用中性的“动态强度分” d_intensity。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states

        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        bottom_context_bonus_factor = get_param_value(p_conf.get('bottom_context_bonus_factor'), 0.5)
        top_context_bonus_factor = get_param_value(p_conf.get('top_context_bonus_factor'), 0.8)

        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)

        # 更新健康度数据结构，统一使用 d_intensity
        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = {
            'quantitative': self._calculate_quantitative_health,
            'advanced': self._calculate_advanced_dynamics_health,
            'internal': self._calculate_internal_structure_health,
            'holder': self._calculate_holder_behavior_health,
            'fault': self._calculate_fault_health,
        }
        for name, calculator in calculators.items():
            # 更新调用签名，接收三元组
            s_bull, s_bear, d_intensity = calculator(df, norm_window, dynamic_weights, periods)
            health_data['s_bull'].append(s_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_intensity'].append(d_intensity)

        overall_health = {}
        pillar_names = list(pillar_weights.keys())
        weights_array = np.array([pillar_weights.get(name, 0) for name in calculators.keys()])
        use_equal_weights = not pillar_weights or weights_array.sum() == 0

        # 更新健康度融合逻辑，使用 d_intensity
        for health_type, health_sources in [
            ('s_bull', health_data['s_bull']),
            ('s_bear', health_data['s_bear']),
            ('d_intensity', health_data['d_intensity'])
        ]:
            overall_health[health_type] = {}
            for p in periods:
                if not health_sources: continue
                valid_pillars = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if not valid_pillars: continue
                stacked_values = np.stack(valid_pillars, axis=0)
                if use_equal_weights:
                    fused_values = np.prod(stacked_values, axis=0) ** (1.0 / stacked_values.shape[0])
                else:
                    fused_values = np.prod(stacked_values ** weights_array[:, np.newaxis], axis=0)
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)

        self.strategy.atomic_states['__CHIP_overall_health'] = overall_health
        default_series = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 所有动态分均使用 d_intensity
        bullish_resonance_health = {p: overall_health['s_bull'][p] * overall_health['d_intensity'][p] for p in periods if p in overall_health.get('s_bull', {}) and p in overall_health.get('d_intensity', {})}
        bullish_short_force_res = (bullish_resonance_health.get(1, default_series) * bullish_resonance_health.get(5, default_series))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, default_series) * bullish_resonance_health.get(21, default_series))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, default_series)
        overall_bullish_resonance = ((bullish_short_force_res ** resonance_tf_weights['short']) * (bullish_medium_trend_res ** resonance_tf_weights['medium']) * (bullish_long_inertia_res ** resonance_tf_weights['long']))
        
        bullish_reversal_health = {p: overall_health['s_bear'][p] * overall_health['d_intensity'][p] for p in periods if p in overall_health.get('s_bear', {}) and p in overall_health.get('d_intensity', {})}
        bullish_short_force_rev = (bullish_reversal_health.get(1, default_series) * bullish_reversal_health.get(5, default_series))**0.5
        bullish_medium_trend_rev = (bullish_reversal_health.get(13, default_series) * bullish_reversal_health.get(21, default_series))**0.5
        bullish_long_inertia_rev = bullish_reversal_health.get(55, default_series)
        overall_bullish_reversal_trigger = ((bullish_short_force_rev ** reversal_tf_weights['short']) * (bullish_medium_trend_rev ** reversal_tf_weights['medium']) * (bullish_long_inertia_rev ** reversal_tf_weights['long']))
        final_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + bottom_context_bonus_factor * bottom_context_score)).clip(0, 1)

        bearish_resonance_health = {p: overall_health['s_bear'][p] * overall_health['d_intensity'][p] for p in periods if p in overall_health.get('s_bear', {}) and p in overall_health.get('d_intensity', {})}
        bearish_short_force_res = (bearish_resonance_health.get(1, default_series) * bearish_resonance_health.get(5, default_series))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, default_series) * bearish_resonance_health.get(21, default_series))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, default_series)
        overall_bearish_resonance = ((bearish_short_force_res ** resonance_tf_weights['short']) * (bearish_medium_trend_res ** resonance_tf_weights['medium']) * (bearish_long_inertia_res ** resonance_tf_weights['long']))
        
        bearish_reversal_health = {p: overall_health['s_bull'][p] * overall_health['d_intensity'][p] for p in periods if p in overall_health.get('s_bull', {}) and p in overall_health.get('d_intensity', {})}
        bearish_short_force_rev = (bearish_reversal_health.get(1, default_series) * bearish_reversal_health.get(5, default_series))**0.5
        bearish_medium_trend_rev = (bearish_reversal_health.get(13, default_series) * bearish_reversal_health.get(21, default_series))**0.5
        bearish_long_inertia_rev = bearish_reversal_health.get(55, default_series)
        overall_bearish_reversal_trigger = ((bearish_short_force_rev ** reversal_tf_weights['short']) * (bearish_medium_trend_rev ** reversal_tf_weights['medium']) * (bearish_long_inertia_rev ** reversal_tf_weights['long']))
        final_top_reversal_score = (overall_bearish_reversal_trigger * (1 + top_context_bonus_factor * top_context_score)).clip(0, 1)

        final_signal_map = {
            'SCORE_CHIP_BULLISH_RESONANCE': overall_bullish_resonance,
            'SCORE_CHIP_BOTTOM_REVERSAL': final_bottom_reversal_score,
            'SCORE_CHIP_BEARISH_RESONANCE': overall_bearish_resonance,
            'SCORE_CHIP_TOP_REVERSAL': final_top_reversal_score
        }
        for signal_name, score in final_signal_map.items():
            states[signal_name] = score.astype(np.float32)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_quantitative_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V3.4 · 动态分统一版】计算基础量化维度的三维健康度"""
        # 更新方法签名和初始化
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        static_bull_conc = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=False)
        static_bull_health = normalize_score(df.get('chip_health_score_D'), df.index, norm_window, ascending=True)
        overall_static_bull = (static_bull_conc * static_bull_health)**0.5
        
        static_bear_conc = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=True)
        static_bear_health = normalize_score(df.get('chip_health_score_D'), df.index, norm_window, ascending=False)
        overall_static_bear = (static_bear_conc * static_bear_health)**0.5

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 计算统一的、中性的动态强度分 d_intensity
            conc_mom_strength = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D').abs(), df.index, norm_window, ascending=True)
            conc_accel_strength = normalize_score(df.get(f'ACCEL_{p}_concentration_90pct_D').abs(), df.index, norm_window, ascending=True)
            dynamic_conc = (conc_mom_strength * conc_accel_strength)**0.5

            cost_mom_strength = normalize_score(df.get(f'SLOPE_{p}_peak_cost_D').abs(), df.index, norm_window, ascending=True)
            cost_accel_strength = normalize_score(df.get(f'ACCEL_{p}_peak_cost_D').abs(), df.index, norm_window, ascending=True)
            dynamic_cost = (cost_mom_strength * cost_accel_strength)**0.5

            health_mom_strength = normalize_score(df.get(f'SLOPE_{p}_chip_health_score_D').abs(), df.index, norm_window, ascending=True)
            health_accel_strength = normalize_score(df.get(f'ACCEL_{p}_chip_health_score_D').abs(), df.index, norm_window, ascending=True)
            dynamic_health = (health_mom_strength * health_accel_strength)**0.5
            
            d_intensity[p] = (dynamic_conc * dynamic_cost * dynamic_health)**(1/3)
        
        # 返回三元组
        return s_bull, s_bear, d_intensity

    def _calculate_advanced_dynamics_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V3.5 · 动态分统一版】计算高级动态维度的三维健康度"""
        # 更新方法签名和初始化
        s_bull, s_bear, d_intensity = {}, {}, {}
        required_cols = ['peak_control_ratio_D', 'peak_strength_ratio_D', 'peak_stability_D', 'is_multi_peak_D']
        if any(col not in df.columns for col in required_cols):
            # 确保在跳过时返回符合新签名的空字典
            return {}, {}, {}

        is_multi_peak_series = df.get('is_multi_peak_D', pd.Series(0.0, index=df.index)).astype(float)
        overall_static_bull = (normalize_score(df.get('peak_control_ratio_D'), df.index, norm_window) * normalize_score(df.get('peak_strength_ratio_D'), df.index, norm_window) * normalize_score(df.get('peak_stability_D'), df.index, norm_window))**(1/3)
        overall_static_bear = (normalize_score(df.get('peak_control_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('peak_strength_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('peak_stability_D'), df.index, norm_window, ascending=False) * is_multi_peak_series)**(1/4)

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 计算统一的、中性的动态强度分 d_intensity
            control_mom_strength = normalize_score(df.get(f'SLOPE_{p}_peak_control_ratio_D').abs(), df.index, norm_window)
            control_accel_strength = normalize_score(df.get(f'ACCEL_{p}_peak_control_ratio_D').abs(), df.index, norm_window)
            dynamic_control = (control_mom_strength * control_accel_strength)**0.5

            stability_mom_strength = normalize_score(df.get(f'SLOPE_{p}_peak_stability_D').abs(), df.index, norm_window)
            stability_accel_strength = normalize_score(df.get(f'ACCEL_{p}_peak_stability_D').abs(), df.index, norm_window)
            dynamic_stability = (stability_mom_strength * stability_accel_strength)**0.5
            
            d_intensity[p] = (dynamic_control * dynamic_stability)**0.5
            
        # 返回三元组
        return s_bull, s_bear, d_intensity

    def _calculate_internal_structure_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V3.3 · 动态分统一版】计算内部结构维度的三维健康度"""
        # 更新方法签名和初始化
        s_bull, s_bear, d_intensity = {}, {}, {}

        overall_static_bull = (normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=False) * (normalize_score(df.get('support_below_D'), df.index, norm_window) * normalize_score(df.get('pressure_above_D'), df.index, norm_window, ascending=False))**0.5)**0.5
        overall_static_bear = (normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=True) * (normalize_score(df.get('support_below_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('pressure_above_D'), df.index, norm_window, ascending=True))**0.5)**0.5

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 计算统一的、中性的动态强度分 d_intensity
            core_conc_mom_strength = normalize_score(df.get(f'SLOPE_{p}_concentration_70pct_D').abs(), df.index, norm_window)
            core_conc_accel_strength = normalize_score(df.get(f'ACCEL_{p}_concentration_70pct_D').abs(), df.index, norm_window)
            dynamic_core_conc = (core_conc_mom_strength * core_conc_accel_strength)**0.5

            support_mom_strength = normalize_score(df.get(f'SLOPE_{p}_support_below_D').abs(), df.index, norm_window)
            pressure_mom_strength = normalize_score(df.get(f'SLOPE_{p}_pressure_above_D').abs(), df.index, norm_window)
            dynamic_net_support = (support_mom_strength * pressure_mom_strength)**0.5
            
            d_intensity[p] = (dynamic_core_conc * dynamic_net_support)**0.5
            
        # 返回三元组
        return s_bull, s_bear, d_intensity

    def _calculate_holder_behavior_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V3.3 · 动态分统一版】计算持仓者行为维度的三维健康度"""
        # 更新方法签名和初始化
        s_bull, s_bear, d_intensity = {}, {}, {}

        overall_static_bull = (normalize_score(df.get('cost_divergence_D'), df.index, norm_window) * normalize_score(df.get('winner_profit_margin_D'), df.index, norm_window) * normalize_score(df.get('total_winner_rate_D'), df.index, norm_window) * normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False))**(1/4)
        overall_static_bear = (normalize_score(df.get('cost_divergence_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('winner_profit_margin_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('total_winner_rate_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=True))**(1/4)

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 计算统一的、中性的动态强度分 d_intensity
            cost_div_mom_strength = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D').abs(), df.index, norm_window)
            cost_div_accel_strength = normalize_score(df.get(f'ACCEL_{p}_cost_divergence_D').abs(), df.index, norm_window)
            dynamic_cost_div = (cost_div_mom_strength * cost_div_accel_strength)**0.5

            margin_mom_strength = normalize_score(df.get(f'SLOPE_{p}_winner_profit_margin_D').abs(), df.index, norm_window)
            margin_accel_strength = normalize_score(df.get(f'ACCEL_{p}_winner_profit_margin_D').abs(), df.index, norm_window)
            dynamic_margin = (margin_mom_strength * margin_accel_strength)**0.5

            turnover_mom_strength = normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D').abs(), df.index, norm_window)
            turnover_accel_strength = normalize_score(df.get(f'ACCEL_{p}_turnover_from_winners_ratio_D').abs(), df.index, norm_window)
            dynamic_turnover = (turnover_mom_strength * turnover_accel_strength)**0.5
            
            d_intensity[p] = (dynamic_cost_div * dynamic_margin * dynamic_turnover)**(1/3)
            
        # 返回三元组
        return s_bull, s_bear, d_intensity

    def _calculate_fault_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V2.3 · 动态分统一版】计算筹码断层维度的三维健康度"""
        # 更新方法签名和初始化
        s_bull, s_bear, d_intensity = {}, {}, {}

        overall_static_bull = (normalize_score(df.get('chip_fault_strength_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=False))**0.5
        overall_static_bear = (normalize_score(df.get('chip_fault_strength_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=True))**0.5

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 计算统一的、中性的动态强度分 d_intensity
            strength_mom_strength = normalize_score(df.get(f'SLOPE_{p}_chip_fault_strength_D').abs(), df.index, norm_window)
            strength_accel_strength = normalize_score(df.get(f'ACCEL_{p}_chip_fault_strength_D').abs(), df.index, norm_window)
            dynamic_strength = (strength_mom_strength * strength_accel_strength)**0.5

            vacuum_mom_strength = normalize_score(df.get(f'SLOPE_{p}_chip_fault_vacuum_percent_D').abs(), df.index, norm_window)
            vacuum_accel_strength = normalize_score(df.get(f'ACCEL_{p}_chip_fault_vacuum_percent_D').abs(), df.index, norm_window)
            dynamic_vacuum = (vacuum_mom_strength * vacuum_accel_strength)**0.5
            
            d_intensity[p] = (dynamic_strength * dynamic_vacuum)**0.5
            
        # 返回三元组
        return s_bull, s_bear, d_intensity

    # ==============================================================================
    # 以下为保留的、具有特殊战术意义的“剧本”诊断模块
    # ==============================================================================

    def diagnose_accumulation_playbooks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.5 · 重构修复版】主力吸筹模式与风险诊断引擎 (战术模块，予以保留)
        """
        states = {}
        norm_window = 120
        # 修正对 normalize_score 的调用
        conc_slope_score = normalize_score(df.get('SLOPE_5_concentration_90pct_D'), df.index, norm_window, ascending=False)
        conc_accel_score = normalize_score(df.get('ACCEL_5_concentration_90pct_D'), df.index, norm_window, ascending=False)
        concentration_improving_score = (conc_slope_score * conc_accel_score)
        cost_rising_score = normalize_score(df.get('SLOPE_5_peak_cost_D'), df.index, norm_window, ascending=True)
        cost_falling_score = normalize_score(df.get('SLOPE_5_peak_cost_D'), df.index, norm_window, ascending=False)
        winner_holding_score = normalize_score(df.get('SLOPE_5_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False)
        loser_capitulating_score = normalize_score(df.get('turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True)
        
        rally_accumulation_score = (cost_rising_score * winner_holding_score).astype(np.float32)
        states['SCORE_CHIP_PLAYBOOK_RALLY_ACCUMULATION'] = rally_accumulation_score
        suppress_accumulation_score = (concentration_improving_score * cost_falling_score * loser_capitulating_score).astype(np.float32)
        states['SCORE_CHIP_PLAYBOOK_SUPPRESS_ACCUMULATION'] = suppress_accumulation_score
        true_accumulation_score = np.maximum(rally_accumulation_score, suppress_accumulation_score)
        states['SCORE_CHIP_TRUE_ACCUMULATION'] = true_accumulation_score.astype(np.float32)
        
        return states

    def _diagnose_setup_capitulation_ready(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.2 · 重构修复版】诊断“恐慌已弥漫”的战备(Setup)状态 (战术模块，予以保留)"""
        states = {}
        required_col = 'total_loser_rate_D'
        if required_col not in df.columns:
            print(f"        -> [筹码情报-恐慌战备诊断] 警告: 缺少关键数据列 '{required_col}'，模块已跳过！")
            return states
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 修正对 normalize_score 的调用，这是导致错误的根源
        deep_capitulation_score = normalize_score(df['total_loser_rate_D'], df.index, norm_window, ascending=True)
        
        long_term_window = 250
        min_periods_long = long_term_window // 4
        
        # 修正对 normalize_score 的调用
        rank_score = normalize_score(df['close_D'], df.index, window=long_term_window, ascending=False)
        
        rolling_low = df['low_D'].rolling(window=long_term_window, min_periods=min_periods_long).min()
        rolling_high = df['high_D'].rolling(window=long_term_window, min_periods=min_periods_long).max()
        price_range = (rolling_high - rolling_low).replace(0, 1e-9)
        position_in_range = (df['close_D'] - rolling_low) / price_range
        range_score = 1.0 - position_in_range.clip(0, 1)
        price_pos_score = np.maximum(rank_score, range_score.fillna(0.5))
        setup_score = (deep_capitulation_score * price_pos_score).astype(np.float32)
        states['SCORE_SETUP_CAPITULATION_READY'] = setup_score
        return states

    def _diagnose_trigger_capitulation_fire(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.1 · 重构修复版】诊断“卖压出清”的点火(Trigger)行为 (战术模块，予以保留)"""
        states = {}
        required_cols = ['turnover_from_losers_ratio_D', 'ACCEL_5_turnover_from_losers_ratio_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"        -> [筹码情报-卖压出清诊断] 警告: 缺少关键数据列 {missing_cols}，模块已跳过！")
            return states
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 修正对 normalize_score 的调用
        relative_turnover_score = normalize_score(df['turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        
        k = get_param_value(p.get('logistic_k', 0.1))
        x0 = get_param_value(p.get('logistic_x0', 50.0))
        absolute_turnover_score = 1 / (1 + np.exp(-k * (df['turnover_from_losers_ratio_D'] - x0)))
        loser_turnover_score = np.maximum(relative_turnover_score, absolute_turnover_score)
        
        # 修正对 normalize_score 的调用
        loser_turnover_accel_score = normalize_score(df['ACCEL_5_turnover_from_losers_ratio_D'], df.index, norm_window, ascending=True)
        
        trigger_score = (loser_turnover_score * loser_turnover_accel_score).astype(np.float32)
        states['SCORE_TRIGGER_CAPITULATION_FIRE'] = trigger_score
        return states

    def _synthesize_playbook_capitulation_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.1】合成“恐慌盘投降反转”剧本 (战术模块，予以保留)"""
        states = {}
        required_cols = ['SCORE_SETUP_CAPITULATION_READY', 'SCORE_TRIGGER_CAPITULATION_FIRE']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"        -> [筹码情报-投降反转剧本] 警告: 缺少前置信号 {missing_cols}，剧本合成已跳过！")
            return states
        p = get_params_block(self.strategy, 'capitulation_reversal_params', {})
        setup_score = df['SCORE_SETUP_CAPITULATION_READY']
        trigger_score = df['SCORE_TRIGGER_CAPITULATION_FIRE']
        was_setup_yesterday = setup_score.shift(1).fillna(0.0)
        raw_score = (was_setup_yesterday * trigger_score)
        exponent = get_param_value(p.get('final_score_exponent'), 1.0)
        final_score = (raw_score ** exponent).astype(np.float32)
        states['SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL'] = final_score
        return states


