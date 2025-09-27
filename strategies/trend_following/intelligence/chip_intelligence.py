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

    def run_chip_intelligence_command(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V500.0 · 统一范式版】筹码情报最高司令部
        - 核心重构 (本次修改):
          - [架构统一] 废除所有旧的、各自为战的诊断引擎，统一调用唯一的终极信号引擎 `diagnose_unified_chip_signals`。
          - [哲学统一] 所有信号生成逻辑均遵循“上下文门控”范式，实现了整个模块内部思想的完全统一。
          - [保留特例] 保留了 `diagnose_accumulation_playbooks` 和 `_synthesize_playbook_capitulation_reversal` 等
                        具有特殊战术意义的“剧本”诊断模块，它们作为终极信号的补充而存在。
        - 收益: 实现了前所未有的架构清晰度、逻辑一致性和哲学完备性。
        """
        # print("        -> [筹码情报最高司令部 V500.0 · 统一范式版] 启动...") # 更新版本号
        
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
        trigger_states = self._diagnose_trigger_capitulation_fire(df)
        all_chip_states.update(trigger_states)
        df = df.assign(**all_chip_states) # 确保原子状态可被下游消费
        playbook_states = self._synthesize_playbook_capitulation_reversal(df)
        all_chip_states.update(playbook_states)

        # print(f"        -> [筹码情报最高司令部 V500.0] 分析完毕，共生成 {len(all_chip_states)} 个筹码信号。") # 更新版本号
        return all_chip_states, {}

    def diagnose_unified_chip_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V11.0 · 终极哲学统一版】统一筹码信号诊断引擎
        - 核心修复: 1. 将最终信号合成逻辑从“加法模型”彻底修改为“加权几何平均”。
                      2. 将所有动态健康度(d_bull/d_bear)的计算从“加法模型”修改为“几何平均”。
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

        health_data = { 's_bull': [], 'd_bull': [], 's_bear': [], 'd_bear': [] } 
        calculators = {
            'quantitative': self._calculate_quantitative_health,
            'advanced': self._calculate_advanced_dynamics_health,
            'internal': self._calculate_internal_structure_health,
            'holder': self._calculate_holder_behavior_health,
            'fault': self._calculate_fault_health,
        }
        for name, calculator in calculators.items():
            s_bull, d_bull, s_bear, d_bear = calculator(df, norm_window, dynamic_weights, periods)
            health_data['s_bull'].append(s_bull) 
            health_data['d_bull'].append(d_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_bear'].append(d_bear) 

        overall_health = {}
        pillar_names = list(pillar_weights.keys())
        weights_array = np.array([pillar_weights.get(name, 0) for name in calculators.keys()])

        use_equal_weights = False
        if not pillar_weights or weights_array.sum() == 0:
            print(f"        -> [筹码情报引擎] 警告: 'pillar_weights' 在配置文件中缺失或总和为0。将临时采用等权重融合。")
            use_equal_weights = True

        
        for health_type, health_sources in [
            ('s_bull', health_data['s_bull']),
            ('d_bull', health_data['d_bull']),
            ('s_bear', health_data['s_bear']),
            ('d_bear', health_data['d_bear'])
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
        
        # 将所有最终信号合成逻辑从加法改为乘法（加权几何平均）
        bullish_resonance_health = {p: overall_health['s_bull'][p] * overall_health['d_bull'][p] for p in periods if p in overall_health.get('s_bull', {}) and p in overall_health.get('d_bull', {})}
        bullish_short_force_res = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, 0.5)
        overall_bullish_resonance = (
            (bullish_short_force_res ** resonance_tf_weights['short']) *
            (bullish_medium_trend_res ** resonance_tf_weights['medium']) *
            (bullish_long_inertia_res ** resonance_tf_weights['long'])
        )
        
        bullish_reversal_health = {p: overall_health['s_bear'][p] * overall_health['d_bull'][p] for p in periods if p in overall_health.get('s_bear', {}) and p in overall_health.get('d_bull', {})}
        bullish_short_force_rev = (bullish_reversal_health.get(1, 0.5) * bullish_reversal_health.get(5, 0.5))**0.5
        bullish_medium_trend_rev = (bullish_reversal_health.get(13, 0.5) * bullish_reversal_health.get(21, 0.5))**0.5
        bullish_long_inertia_rev = bullish_reversal_health.get(55, 0.5)
        overall_bullish_reversal_trigger = (
            (bullish_short_force_rev ** reversal_tf_weights['short']) *
            (bullish_medium_trend_rev ** reversal_tf_weights['medium']) *
            (bullish_long_inertia_rev ** reversal_tf_weights['long'])
        )
        final_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + bottom_context_bonus_factor * bottom_context_score)).clip(0, 1)

        bearish_resonance_health = {p: overall_health['s_bear'][p] * overall_health['d_bear'][p] for p in periods if p in overall_health.get('s_bear', {}) and p in overall_health.get('d_bear', {})}
        bearish_short_force_res = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, 0.5)
        overall_bearish_resonance = (
            (bearish_short_force_res ** resonance_tf_weights['short']) *
            (bearish_medium_trend_res ** resonance_tf_weights['medium']) *
            (bearish_long_inertia_res ** resonance_tf_weights['long'])
        )
        
        bearish_reversal_health = {p: overall_health['s_bull'][p] * overall_health['d_bear'][p] for p in periods if p in overall_health.get('s_bull', {}) and p in overall_health.get('d_bear', {})}
        bearish_short_force_rev = (bearish_reversal_health.get(1, 0.5) * bearish_reversal_health.get(5, 0.5))**0.5
        bearish_medium_trend_rev = (bearish_reversal_health.get(13, 0.5) * bearish_reversal_health.get(21, 0.5))**0.5
        bearish_long_inertia_rev = bearish_reversal_health.get(55, 0.5)
        overall_bearish_reversal_trigger = (
            (bearish_short_force_rev ** reversal_tf_weights['short']) *
            (bearish_medium_trend_rev ** reversal_tf_weights['medium']) *
            (bearish_long_inertia_rev ** reversal_tf_weights['long'])
        )
        final_top_reversal_score = (overall_bearish_reversal_trigger * (1 + top_context_bonus_factor * top_context_score)).clip(0, 1)
        

        for prefix, score in [('SCORE_CHIP_BULLISH_RESONANCE', overall_bullish_resonance), ('SCORE_CHIP_BOTTOM_REVERSAL', final_bottom_reversal_score),
                              ('SCORE_CHIP_BEARISH_RESONANCE', overall_bearish_resonance), ('SCORE_CHIP_TOP_REVERSAL', final_top_reversal_score)]:
            states[f'{prefix}_S_PLUS'] = score.astype(np.float32)
            states[f'{prefix}_S'] = (score * 0.8).astype(np.float32)
            states[f'{prefix}_A'] = (score * 0.6).astype(np.float32)
            states[f'{prefix}_B'] = (score * 0.4).astype(np.float32)
        
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_quantitative_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V3.2 · 终极哲学统一版】计算基础量化健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        static_bull_conc = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=False)
        static_bull_health = normalize_score(df.get('chip_health_score_D'), df.index, norm_window, ascending=True)
        overall_static_bull = (static_bull_conc * static_bull_health)**0.5
        
        static_bear_conc = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=True)
        static_bear_health = normalize_score(df.get('chip_health_score_D'), df.index, norm_window, ascending=False)
        overall_static_bear = (static_bear_conc * static_bear_health)**0.5

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 根除所有动态分计算中的加法
            d_bull_conc_slope = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, norm_window, ascending=False)
            d_bull_conc_accel = normalize_score(df.get(f'ACCEL_{p}_concentration_90pct_D'), df.index, norm_window, ascending=False)
            d_bull_conc = (d_bull_conc_slope * d_bull_conc_accel)**0.5

            d_bull_cost_slope = normalize_score(df.get(f'SLOPE_{p}_peak_cost_D'), df.index, norm_window, ascending=True)
            d_bull_cost_accel = normalize_score(df.get(f'ACCEL_{p}_peak_cost_D'), df.index, norm_window, ascending=True)
            d_bull_cost = (d_bull_cost_slope * d_bull_cost_accel)**0.5

            d_bull_health_slope = normalize_score(df.get(f'SLOPE_{p}_chip_health_score_D'), df.index, norm_window, ascending=True)
            d_bull_health_accel = normalize_score(df.get(f'ACCEL_{p}_chip_health_score_D'), df.index, norm_window, ascending=True)
            d_bull_health = (d_bull_health_slope * d_bull_health_accel)**0.5
            d_bull[p] = (d_bull_conc * d_bull_cost * d_bull_health)**(1/3)

            d_bear_conc_slope = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, norm_window, ascending=True)
            d_bear_conc_accel = normalize_score(df.get(f'ACCEL_{p}_concentration_90pct_D'), df.index, norm_window, ascending=True)
            d_bear_conc = (d_bear_conc_slope * d_bear_conc_accel)**0.5

            d_bear_cost_slope = normalize_score(df.get(f'SLOPE_{p}_peak_cost_D'), df.index, norm_window, ascending=False)
            d_bear_cost_accel = normalize_score(df.get(f'ACCEL_{p}_peak_cost_D'), df.index, norm_window, ascending=False)
            d_bear_cost = (d_bear_cost_slope * d_bear_cost_accel)**0.5

            d_bear_health_slope = normalize_score(df.get(f'SLOPE_{p}_chip_health_score_D'), df.index, norm_window, ascending=False)
            d_bear_health_accel = normalize_score(df.get(f'ACCEL_{p}_chip_health_score_D'), df.index, norm_window, ascending=False)
            d_bear_health = (d_bear_health_slope * d_bear_health_accel)**0.5
            d_bear[p] = (d_bear_conc * d_bear_cost * d_bear_health)**(1/3)
            
        
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_advanced_dynamics_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V3.2 · 终极哲学统一版】计算高级动态健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}

        required_cols = ['peak_control_ratio_D', 'peak_strength_ratio_D', 'peak_stability_D', 'is_multi_peak_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"        -> [筹码情报-高级动态健康度] 警告: 缺少关键数据列 {missing_cols}，模块已跳过！")
            return s_bull, d_bull, s_bear, d_bear

        is_multi_peak_series = df.get('is_multi_peak_D', pd.Series(0.0, index=df.index)).astype(float)

        overall_static_bull = (normalize_score(df.get('peak_control_ratio_D'), df.index, norm_window) * normalize_score(df.get('peak_strength_ratio_D'), df.index, norm_window) * normalize_score(df.get('peak_stability_D'), df.index, norm_window) * (1.0 - is_multi_peak_series))**(1/4)
        overall_static_bear = (normalize_score(df.get('peak_control_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('peak_strength_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('peak_stability_D'), df.index, norm_window, ascending=False) * is_multi_peak_series)**(1/4)

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 根除所有动态分计算中的加法
            d_bull_control_slope = normalize_score(df.get(f'SLOPE_{p}_peak_control_ratio_D'), df.index, norm_window)
            d_bull_control_accel = normalize_score(df.get(f'ACCEL_{p}_peak_control_ratio_D'), df.index, norm_window)
            d_bull_control = (d_bull_control_slope * d_bull_control_accel)**0.5

            d_bull_stability_slope = normalize_score(df.get(f'SLOPE_{p}_peak_stability_D'), df.index, norm_window)
            d_bull_stability_accel = normalize_score(df.get(f'ACCEL_{p}_peak_stability_D'), df.index, norm_window)
            d_bull_stability = (d_bull_stability_slope * d_bull_stability_accel)**0.5
            d_bull[p] = (d_bull_control * d_bull_stability)**0.5

            d_bear_control_slope = normalize_score(df.get(f'SLOPE_{p}_peak_control_ratio_D'), df.index, norm_window, ascending=False)
            d_bear_control_accel = normalize_score(df.get(f'ACCEL_{p}_peak_control_ratio_D'), df.index, norm_window, ascending=False)
            d_bear_control = (d_bear_control_slope * d_bear_control_accel)**0.5

            d_bear_stability_slope = normalize_score(df.get(f'SLOPE_{p}_peak_stability_D'), df.index, norm_window, ascending=False)
            d_bear_stability_accel = normalize_score(df.get(f'ACCEL_{p}_peak_stability_D'), df.index, norm_window, ascending=False)
            d_bear_stability = (d_bear_stability_slope * d_bear_stability_accel)**0.5
            d_bear[p] = (d_bear_control * d_bear_stability)**0.5
            

        return s_bull, d_bull, s_bear, d_bear

    def _calculate_internal_structure_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V3.1 · 终极哲学统一版】计算内部结构健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}

        overall_static_bull = (normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=False) * (normalize_score(df.get('support_below_D'), df.index, norm_window) * normalize_score(df.get('pressure_above_D'), df.index, norm_window, ascending=False))**0.5)**0.5
        overall_static_bear = (normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=True) * (normalize_score(df.get('support_below_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('pressure_above_D'), df.index, norm_window, ascending=True))**0.5)**0.5

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 根除所有动态分计算中的加法
            d_bull_core_conc_slope = normalize_score(df.get(f'SLOPE_{p}_concentration_70pct_D'), df.index, norm_window, ascending=False)
            d_bull_core_conc_accel = normalize_score(df.get(f'ACCEL_{p}_concentration_70pct_D'), df.index, norm_window, ascending=False)
            d_bull_core_conc = (d_bull_core_conc_slope * d_bull_core_conc_accel)**0.5

            d_bull_net_support = (normalize_score(df.get(f'SLOPE_{p}_support_below_D'), df.index, norm_window) * normalize_score(df.get(f'SLOPE_{p}_pressure_above_D'), df.index, norm_window, ascending=False))**0.5
            d_bull[p] = (d_bull_core_conc * d_bull_net_support)**0.5

            d_bear_core_conc_slope = normalize_score(df.get(f'SLOPE_{p}_concentration_70pct_D'), df.index, norm_window, ascending=True)
            d_bear_core_conc_accel = normalize_score(df.get(f'ACCEL_{p}_concentration_70pct_D'), df.index, norm_window, ascending=True)
            d_bear_core_conc = (d_bear_core_conc_slope * d_bear_core_conc_accel)**0.5

            d_bear_net_support = (normalize_score(df.get(f'SLOPE_{p}_support_below_D'), df.index, norm_window, ascending=False) * normalize_score(df.get(f'SLOPE_{p}_pressure_above_D'), df.index, norm_window, ascending=True))**0.5
            d_bear[p] = (d_bear_core_conc * d_bear_net_support)**0.5
            

        return s_bull, d_bull, s_bear, d_bear

    def _calculate_holder_behavior_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V3.1 · 终极哲学统一版】计算持仓者行为与情绪健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}

        overall_static_bull = (normalize_score(df.get('cost_divergence_D'), df.index, norm_window) * normalize_score(df.get('winner_profit_margin_D'), df.index, norm_window) * normalize_score(df.get('total_winner_rate_D'), df.index, norm_window) * normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False))**(1/4)
        overall_static_bear = (normalize_score(df.get('cost_divergence_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('winner_profit_margin_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('total_winner_rate_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('turnover_from_winners_ratio_D'), df.index, norm_window, ascending=True))**(1/4)

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 根除所有动态分计算中的加法
            d_bull_cost_div_slope = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window)
            d_bull_cost_div_accel = normalize_score(df.get(f'ACCEL_{p}_cost_divergence_D'), df.index, norm_window)
            d_bull_cost_div = (d_bull_cost_div_slope * d_bull_cost_div_accel)**0.5

            d_bull_margin_slope = normalize_score(df.get(f'SLOPE_{p}_winner_profit_margin_D'), df.index, norm_window)
            d_bull_margin_accel = normalize_score(df.get(f'ACCEL_{p}_winner_profit_margin_D'), df.index, norm_window)
            d_bull_margin = (d_bull_margin_slope * d_bull_margin_accel)**0.5

            d_bull_turnover_slope = normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False)
            d_bull_turnover_accel = normalize_score(df.get(f'ACCEL_{p}_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False)
            d_bull_turnover = (d_bull_turnover_slope * d_bull_turnover_accel)**0.5
            d_bull[p] = (d_bull_cost_div * d_bull_margin * d_bull_turnover)**(1/3)

            d_bear_cost_div_slope = normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False)
            d_bear_cost_div_accel = normalize_score(df.get(f'ACCEL_{p}_cost_divergence_D'), df.index, norm_window, ascending=False)
            d_bear_cost_div = (d_bear_cost_div_slope * d_bear_cost_div_accel)**0.5

            d_bear_margin_slope = normalize_score(df.get(f'SLOPE_{p}_winner_profit_margin_D'), df.index, norm_window, ascending=False)
            d_bear_margin_accel = normalize_score(df.get(f'ACCEL_{p}_winner_profit_margin_D'), df.index, norm_window, ascending=False)
            d_bear_margin = (d_bear_margin_slope * d_bear_margin_accel)**0.5

            d_bear_turnover_slope = normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=True)
            d_bear_turnover_accel = normalize_score(df.get(f'ACCEL_{p}_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=True)
            d_bear_turnover = (d_bear_turnover_slope * d_bear_turnover_accel)**0.5
            d_bear[p] = (d_bear_cost_div * d_bear_margin * d_bear_turnover)**(1/3)
            

        return s_bull, d_bull, s_bear, d_bear

    def _calculate_fault_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """【V2.1 · 终极哲学统一版】计算筹码断层健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}

        overall_static_bull = (normalize_score(df.get('chip_fault_strength_D'), df.index, norm_window, ascending=False) * normalize_score(df.get('chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=False))**0.5
        overall_static_bear = (normalize_score(df.get('chip_fault_strength_D'), df.index, norm_window, ascending=True) * normalize_score(df.get('chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=True))**0.5

        for p in periods:
            s_bull[p] = overall_static_bull
            s_bear[p] = overall_static_bear

            # 根除所有动态分计算中的加法
            d_bull_strength_slope = normalize_score(df.get(f'SLOPE_{p}_chip_fault_strength_D'), df.index, norm_window, ascending=False)
            d_bull_strength_accel = normalize_score(df.get(f'ACCEL_{p}_chip_fault_strength_D'), df.index, norm_window, ascending=False)
            d_bull_strength = (d_bull_strength_slope * d_bull_strength_accel)**0.5

            d_bull_vacuum_slope = normalize_score(df.get(f'SLOPE_{p}_chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=False)
            d_bull_vacuum_accel = normalize_score(df.get(f'ACCEL_{p}_chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=False)
            d_bull_vacuum = (d_bull_vacuum_slope * d_bull_vacuum_accel)**0.5
            d_bull[p] = (d_bull_strength * d_bull_vacuum)**0.5

            d_bear_strength_slope = normalize_score(df.get(f'SLOPE_{p}_chip_fault_strength_D'), df.index, norm_window, ascending=True)
            d_bear_strength_accel = normalize_score(df.get(f'ACCEL_{p}_chip_fault_strength_D'), df.index, norm_window, ascending=True)
            d_bear_strength = (d_bear_strength_slope * d_bear_strength_accel)**0.5

            d_bear_vacuum_slope = normalize_score(df.get(f'SLOPE_{p}_chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=True)
            d_bear_vacuum_accel = normalize_score(df.get(f'ACCEL_{p}_chip_fault_vacuum_percent_D'), df.index, norm_window, ascending=True)
            d_bear_vacuum = (d_bear_vacuum_slope * d_bear_vacuum_accel)**0.5
            d_bear[p] = (d_bear_strength * d_bear_vacuum)**0.5
            
            
        return s_bull, d_bull, s_bear, d_bear

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


