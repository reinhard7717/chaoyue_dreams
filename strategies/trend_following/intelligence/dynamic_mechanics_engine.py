# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, calculate_context_scores, calculate_holographic_dynamics

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def run_dynamic_analysis_command(self) -> Dict[str, pd.Series]: # 修正返回类型注解，并移除 -> None
        """
        【V4.1 · 协议统一版】动态力学引擎总指挥
        - 核心重构: 不再返回None，而是返回一个包含所有生成信号的字典，遵循标准汇报协议。
        """
        # print("    -> [动态力学引擎总指挥 V4.1 · 协议统一版] 启动...") # 更新版本号
        ultimate_dynamic_states = self.diagnose_ultimate_dynamic_mechanics_signals(self.strategy.df_indicators)
        if ultimate_dynamic_states:
            # self.strategy.atomic_states.update(ultimate_dynamic_states) # IntelligenceLayer会做这个
            # print(f"    -> [动态力学引擎总指挥 V4.1] 分析完毕，共生成 {len(ultimate_dynamic_states)} 个终极动态力学信号。")
            # 返回包含所有状态的单一字典
            return ultimate_dynamic_states
        return {} # 如果没有生成信号，返回一个空字典

    def diagnose_ultimate_dynamic_mechanics_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 关系动力赋能版】
        - 核心升级: 引入权威的“关系动力分”，为本模块的看涨信号进行最终赋能。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states

        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        bottom_context_bonus_factor = get_param_value(p_conf.get('bottom_context_bonus_factor'), 0.5)

        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        universal_bottom_pattern_score = self.strategy.atomic_states.get('SCORE_UNIVERSAL_BOTTOM_PATTERN', pd.Series(0.0, index=df.index))
        
        # 获取权威的“关系动力分”
        relational_dynamics_power = self.strategy.atomic_states.get('SCORE_ATOMIC_RELATIONAL_DYNAMICS', pd.Series(0.5, index=df.index))

        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = {
            'volatility': self._calculate_volatility_health,
            'efficiency': self._calculate_efficiency_health,
            'momentum': self._calculate_kinetic_energy_health,
            'inertia': self._calculate_inertia_health,
        }
        for name, calculator in calculators.items():
            s_bull, s_bear, d_intensity = calculator(df, norm_window, dynamic_weights, periods)
            health_data['s_bull'].append(s_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_intensity'].append(d_intensity)

        overall_health = {}
        weights_array = np.array([pillar_weights.get(name, 0.25) for name in calculators.keys()])

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
                fused_values = np.prod(stacked_values ** weights_array[:, np.newaxis], axis=0)
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)

        self.strategy.atomic_states['__DYN_overall_health'] = overall_health
        default_series = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 使用“关系动力分”对看涨共振进行赋能
        bullish_resonance_health = {p: overall_health['s_bull'][p] * overall_health['d_intensity'][p] * relational_dynamics_power for p in periods if p in overall_health.get('s_bull', {}) and p in overall_health.get('d_intensity', {})}
        bullish_short_force_res = (bullish_resonance_health.get(1, default_series) * bullish_resonance_health.get(5, default_series))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, default_series) * bullish_resonance_health.get(21, default_series))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, default_series)
        overall_bullish_resonance = ((bullish_short_force_res ** resonance_tf_weights['short']) * (bullish_medium_trend_res ** resonance_tf_weights['medium']) * (bullish_long_inertia_res ** resonance_tf_weights['long']))
        
        # 使用“关系动力分”对底部反转进行赋能
        bullish_reversal_health = {
            p: np.maximum(universal_bottom_pattern_score * overall_health['s_bull'][p], overall_health['s_bear'][p]) * overall_health['d_intensity'][p] * relational_dynamics_power
            for p in periods if p in overall_health.get('s_bull', {}) and p in overall_health.get('d_intensity', {})
        }
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
        
        bearish_reversal_health = {p: overall_health['s_bear'][p] * overall_health['d_intensity'][p] for p in periods if p in overall_health.get('s_bear', {}) and p in overall_health.get('d_intensity', {})}
        bearish_short_force_rev = (bearish_reversal_health.get(1, default_series) * bearish_reversal_health.get(5, default_series))**0.5
        bearish_medium_trend_rev = (bearish_reversal_health.get(13, default_series) * bearish_reversal_health.get(21, default_series))**0.5
        bearish_long_inertia_rev = bearish_reversal_health.get(55, default_series)
        overall_bearish_reversal_trigger = ((bearish_short_force_rev ** reversal_tf_weights['short']) * (bearish_medium_trend_rev ** reversal_tf_weights['medium']) * (bearish_long_inertia_rev ** reversal_tf_weights['long']))
        final_top_reversal_score = (overall_bearish_reversal_trigger * top_context_score).clip(0, 1)
        
        final_signal_map = {
            'SCORE_DYN_BULLISH_RESONANCE': overall_bullish_resonance,
            'SCORE_DYN_BOTTOM_REVERSAL': final_bottom_reversal_score,
            'SCORE_DYN_BEARISH_RESONANCE': overall_bearish_resonance,
            'SCORE_DYN_TOP_REVERSAL': final_top_reversal_score
        }
        for signal_name, score in final_signal_map.items():
            states[signal_name] = score.astype(np.float32)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器
    # ==============================================================================

    def _calculate_volatility_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V2.1 · 调用适配版】计算波动率(BBW)维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        static_bull = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False)
        static_bear = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=True)

        # 调用中央引擎获取元组，然后在调用处进行融合
        bull_holo, bear_holo = calculate_holographic_dynamics(df, 'BBW_21_2.0', norm_window)
        unified_d_intensity = (bull_holo + bear_holo) / 2.0

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            if p in [1, 5, 13]:
                d_intensity[p] = unified_d_intensity
            else:
                d_intensity[p] = pd.Series(0.5, index=df.index, dtype=np.float32)

        return s_bull, s_bear, d_intensity

    def _calculate_efficiency_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V2.1 · 调用适配版】计算效率(VPA)维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        static_bull = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window)
        static_bear = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False)

        # 调用中央引擎获取元组，然后在调用处进行融合
        bull_holo, bear_holo = calculate_holographic_dynamics(df, 'VPA_EFFICIENCY', norm_window)
        unified_d_intensity = (bull_holo + bear_holo) / 2.0

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_kinetic_energy_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V2.1 · 调用适配版】计算动能(ATR)维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        static_bull = normalize_score(df.get('ATR_14_D'), df.index, norm_window)
        static_bear = normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=False)

        # 调用中央引擎获取元组，然后在调用处进行融合
        bull_holo, bear_holo = calculate_holographic_dynamics(df, 'ATR_14', norm_window)
        unified_d_intensity = (bull_holo + bear_holo) / 2.0

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            d_intensity[p] = unified_d_intensity

        return s_bull, s_bear, d_intensity

    def _calculate_inertia_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict]:
        """【V2.1 · 调用适配版】计算惯性(ADX)维度的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        static_bull = normalize_score(df.get('ADX_14_D'), df.index, norm_window)
        static_bear = normalize_score(df.get('ADX_14_D'), df.index, norm_window, ascending=False)

        # 调用中央引擎获取元组，然后在调用处进行融合
        bull_holo, bear_holo = calculate_holographic_dynamics(df, 'ADX_14', norm_window)
        unified_d_intensity = (bull_holo + bear_holo) / 2.0

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity
















