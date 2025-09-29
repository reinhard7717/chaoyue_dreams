# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, calculate_holographic_dynamics, normalize_score, calculate_context_scores

class StructuralIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化结构情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def diagnose_structural_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 终极信号版】结构情报分析总指挥
        - 核心重构: 遵循终极信号范式，本模块不再返回一堆零散的原子信号。
                      现在只调用唯一的终极信号引擎 `diagnose_ultimate_structural_signals`，
                      并将其产出的16个S+/S/A/B级信号作为本模块的最终输出。
        - 收益: 架构与其他情报模块完全统一，极大提升了信号质量和架构清晰度。
        """
        # print("      -> [结构情报分析总指挥 V2.0 终极信号版] 启动...")
        # 直接调用终极信号引擎，并将其结果作为本模块的唯一输出
        ultimate_structural_states = self.diagnose_ultimate_structural_signals(df)
        # print(f"      -> [结构情报分析总指挥 V2.0] 分析完毕，共生成 {len(ultimate_structural_states)} 个终极结构信号。")
        return ultimate_structural_states

    def diagnose_ultimate_structural_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V14.0 · 回声版】
        - 核心升级: 使用“反转回声”信号 (SCORE_CONTEXT_RECENT_REVERSAL) 替代瞬时的底部形态分，为“底部反转”信号提供持续赋能。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        bottom_context_bonus_factor = get_param_value(p_conf.get('bottom_context_bonus_factor'), 0.5)

        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        # 消费“反转回声”信号
        recent_reversal_context = self.strategy.atomic_states.get('SCORE_CONTEXT_RECENT_REVERSAL', pd.Series(0.0, index=df.index))
        relational_dynamics_power = self.strategy.atomic_states.get('SCORE_ATOMIC_RELATIONAL_DYNAMICS', pd.Series(0.5, index=df.index))

        health_data = { 's_bull': [], 's_bear': [], 'd_intensity': [] } 
        calculators = { 'ma': self._calculate_ma_health, 'mechanics': self._calculate_mechanics_health, 'mtf': self._calculate_mtf_health, 'pattern': self._calculate_pattern_health }
        for name, calculator in calculators.items():
            s_bull, s_bear, d_intensity = calculator(df, periods, norm_window, dynamic_weights)
            health_data['s_bull'].append(s_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_intensity'].append(d_intensity)
        
        overall_health = {}
        
        for health_type, health_sources in [ ('s_bull', health_data['s_bull']), ('s_bear', health_data['s_bear']), ('d_intensity', health_data['d_intensity']) ]:
            overall_health[health_type] = {}
            for p in periods:
                components_for_period = [pillar_dict[p].values for pillar_dict in health_sources if p in pillar_dict]
                if components_for_period:
                    stacked_values = np.stack(components_for_period, axis=0)
                    fused_values = np.prod(stacked_values, axis=0) ** (1.0 / stacked_values.shape[0])
                    overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index, dtype=np.float32)

        self.strategy.atomic_states['__STRUCTURE_overall_health'] = overall_health
        default_series = pd.Series(0.5, index=df.index, dtype=np.float32)

        bullish_resonance_health = {p: np.maximum(overall_health['s_bull'][p], relational_dynamics_power) * overall_health['d_intensity'][p] for p in periods}
        bullish_short_force_res = (bullish_resonance_health.get(1, default_series) * bullish_resonance_health.get(5, default_series))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, default_series) * bullish_resonance_health.get(21, default_series))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, default_series)
        overall_bullish_resonance = ((bullish_short_force_res ** resonance_tf_weights['short']) * (bullish_medium_trend_res ** resonance_tf_weights['medium']) * (bullish_long_inertia_res ** resonance_tf_weights['long']))
        
        # 使用“反转回声”替代瞬时的“底部形态分”
        bullish_reversal_health = {p: recent_reversal_context * relational_dynamics_power * overall_health['d_intensity'][p] for p in periods}
        bullish_short_force_rev = (bullish_reversal_health.get(1, default_series) * bullish_reversal_health.get(5, default_series))**0.5
        bullish_medium_trend_rev = (bullish_reversal_health.get(13, default_series) * bullish_reversal_health.get(21, default_series))**0.5
        bullish_long_inertia_rev = bullish_reversal_health.get(55, default_series)
        overall_bullish_reversal_trigger = ((bullish_short_force_rev ** reversal_tf_weights['short']) * (bullish_medium_trend_rev ** reversal_tf_weights['medium']) * (bullish_long_inertia_rev ** reversal_tf_weights['long']))
        final_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + bottom_context_score * bottom_context_bonus_factor)).clip(0, 1)

        bearish_resonance_health = {p: overall_health['s_bear'][p] * overall_health['d_intensity'][p] for p in periods}
        bearish_short_force_res = (bearish_resonance_health.get(1, default_series) * bearish_resonance_health.get(5, default_series))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, default_series) * bearish_resonance_health.get(21, default_series))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, default_series)
        overall_bearish_resonance = ((bearish_short_force_res ** resonance_tf_weights['short']) * (bearish_medium_trend_res ** resonance_tf_weights['medium']) * (bearish_long_inertia_res ** resonance_tf_weights['long']))

        bearish_reversal_health = {p: overall_health['s_bear'][p] * overall_health['d_intensity'][p] for p in periods}
        bearish_short_force_rev = (bearish_reversal_health.get(1, default_series) * bearish_reversal_health.get(5, default_series))**0.5
        bearish_medium_trend_rev = (bearish_reversal_health.get(13, default_series) * bearish_reversal_health.get(21, default_series))**0.5
        bearish_long_inertia_rev = bearish_reversal_health.get(55, default_series)
        overall_bearish_reversal_trigger = ((bearish_short_force_rev ** reversal_tf_weights['short']) * (bearish_medium_trend_rev ** reversal_tf_weights['medium']) * (bearish_long_inertia_rev ** reversal_tf_weights['long']))
        final_top_reversal_score = (overall_bearish_reversal_trigger * top_context_score).clip(0, 1)
        
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)
        final_signal_map = {
            'SCORE_STRUCTURE_BULLISH_RESONANCE': (overall_bullish_resonance ** exponent),
            'SCORE_STRUCTURE_BOTTOM_REVERSAL': (final_bottom_reversal_score ** exponent),
            'SCORE_STRUCTURE_BEARISH_RESONANCE': (overall_bearish_resonance ** exponent),
            'SCORE_STRUCTURE_TOP_REVERSAL': (final_top_reversal_score ** exponent)
        }
        for signal_name, score in final_signal_map.items():
            states[signal_name] = score.astype(np.float32)
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器，现在返回四维健康度
    # ==============================================================================

    def _calculate_ma_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict]:
        """【V3.1 · 调用适配版】计算MA支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}

        ma_periods = [5, 10, 20, 60, 120]
        bull_alignment_scores = []
        bear_alignment_scores = []
        for i in range(len(ma_periods) - 1):
            short_col = f'EMA_{ma_periods[i]}_D'
            long_col = f'EMA_{ma_periods[i+1]}_D'
            if short_col in df and long_col in df:
                bull_alignment_scores.append((df[short_col] > df[long_col]).astype(float))
                bear_alignment_scores.append((df[short_col] < df[long_col]).astype(float))
        
        if bull_alignment_scores:
            static_bull_score = pd.DataFrame(bull_alignment_scores).mean().fillna(0.5)
            static_bear_score = pd.DataFrame(bear_alignment_scores).mean().fillna(0.5)
        else:
            static_bull_score = pd.Series(0.5, index=df.index)
            static_bear_score = pd.Series(0.5, index=df.index)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            static_col = f'EMA_{p}' if p > 1 else 'close'
            # 调用中央引擎获取元组，然后在调用处进行融合
            bull_holo, bear_holo = calculate_holographic_dynamics(df, static_col, norm_window)
            d_intensity[p] = (bull_holo + bear_holo) / 2.0

        return s_bull, s_bear, d_intensity

    def _calculate_mechanics_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict]:
        """【V3.1 · 调用适配版】计算力学支柱的三维健康度"""
        s_bull, s_bear, d_intensity = {}, {}, {}
        static_bull_energy = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=True)
        static_bear_energy = normalize_score(df.get('energy_ratio_D'), df.index, norm_window, ascending=False)
        
        # 调用中央引擎获取元组，然后在调用处进行融合
        bull_cost, bear_cost = calculate_holographic_dynamics(df, 'peak_cost', norm_window)
        cost_mom_strength = (bull_cost + bear_cost) / 2.0
        
        bull_conc, bear_conc = calculate_holographic_dynamics(df, 'concentration_90pct', norm_window)
        conc_mom_strength = (bull_conc + bear_conc) / 2.0
        
        unified_d_intensity = (cost_mom_strength * conc_mom_strength)**0.5

        for p in periods:
            s_bull[p] = static_bull_energy
            s_bear[p] = static_bear_energy
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_mtf_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.2 · 动态分统一版】计算MTF(多时间框架)支柱的三维健康度
        - 核心重构: 废除 d_bull 和 d_bear，统一返回中性的“动态强度分” d_intensity。
        """
        # 更新方法签名和初始化，统一返回 d_intensity
        s_bull, s_bear, d_intensity = {}, {}, {}

        weekly_cols = [col for col in df.columns if 'EMA' in col and col.endswith('_W')]
        if len(weekly_cols) > 1:
            bull_align_matrix = np.stack([(df[weekly_cols[i]] > df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1)], axis=0)
            bear_align_matrix = np.stack([(df[weekly_cols[i]] < df[weekly_cols[i+1]]).values for i in range(len(weekly_cols)-1)], axis=0)
            static_bull_score = pd.Series(np.mean(bull_align_matrix, axis=0), index=df.index, dtype=np.float32)
            static_bear_score = pd.Series(np.mean(bear_align_matrix, axis=0), index=df.index, dtype=np.float32)
        else:
            static_bull_score = static_bear_score = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 辅助函数现在计算中性的强度分
        def get_avg_strength_score(cols: list[str]) -> pd.Series:
            if not cols: return pd.Series(0.5, index=df.index, dtype=np.float32)
            # 使用 .abs() 获取强度
            scores_matrix = np.stack([normalize_score(df.get(c).abs(), df.index, norm_window, ascending=True).values for c in cols], axis=0)
            return pd.Series(np.mean(scores_matrix, axis=0), index=df.index, dtype=np.float32)

        weekly_slope_cols = [c for c in df.columns if 'SLOPE' in c and c.endswith('_W')]
        weekly_accel_cols = [c for c in df.columns if 'ACCEL' in c and c.endswith('_W')]
        
        # 计算统一的、中性的动态强度分 d_intensity
        dynamic_intensity_score = (get_avg_strength_score(weekly_slope_cols) * get_avg_strength_score(weekly_accel_cols))**0.5

        s_bull = {p: static_bull_score for p in periods}
        s_bear = {p: static_bear_score for p in periods}
        d_intensity = {p: dynamic_intensity_score for p in periods}
        
        # 返回符合新协议的三元组
        return s_bull, s_bear, d_intensity

    def _calculate_pattern_health(self, df: pd.DataFrame, periods: list, norm_window: int, dynamic_weights: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.2 · 动态分统一版】计算形态支柱的三维健康度
        - 核心重构: 废除 d_bull 和 d_bear，统一返回中性的“动态强度分” d_intensity。
        """
        # 更新方法签名和初始化，统一返回 d_intensity
        s_bull, s_bear, d_intensity = {}, {}, {}

        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        is_consolidation = df.get('is_consolidation_D', 0).astype(float)
        is_distribution = df.get('is_distribution_D', 0).astype(float)
        static_bull_score = pd.Series(np.maximum(is_accumulation, is_consolidation), index=df.index).replace(0, 0.5)
        static_bear_score = pd.Series(is_distribution, index=df.index).replace(0, 0.5)

        # 动态强度分现在衡量“事件发生的强度”，而不是方向
        is_breakthrough = df.get('is_breakthrough_D', 0).astype(float)
        is_breakdown = df.get('is_breakdown_D', 0).astype(float)
        # 任何一种突破或破位事件都代表了动态强度的增加
        dynamic_intensity_score = pd.Series(np.maximum(is_breakthrough, is_breakdown), index=df.index).replace(0, 0.5)

        s_bull = {p: static_bull_score for p in periods}
        s_bear = {p: static_bear_score for p in periods}
        d_intensity = {p: dynamic_intensity_score for p in periods}
        
        # 返回符合新协议的三元组
        return s_bull, s_bear, d_intensity






















