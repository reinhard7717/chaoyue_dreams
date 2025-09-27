# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, calculate_context_scores

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def run_dynamic_analysis_command(self) -> None:
        """
        【V4.0 · 终极信号版】动态力学引擎总指挥
        - 核心重构: 遵循终极信号范式，调用唯一的终极信号引擎并更新状态。
        """
        # print("    -> [动态力学引擎总指挥 V4.0 终极信号版] 启动...")
        ultimate_dynamic_states = self.diagnose_ultimate_dynamic_mechanics_signals(self.strategy.df_indicators)
        if ultimate_dynamic_states:
            self.strategy.atomic_states.update(ultimate_dynamic_states)
            # print(f"    -> [动态力学引擎总指挥 V4.0] 分析完毕，共生成 {len(ultimate_dynamic_states)} 个终极动态力学信号。")

    def diagnose_ultimate_dynamic_mechanics_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.7 · 键名统一版】终极动态力学信号诊断模块
        - 核心修复: 统一了 overall_health 缓存的键名，使用 s_bull, d_bull, s_bear, d_bear。
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
        top_context_bonus_factor = get_param_value(p_conf.get('top_context_bonus_factor'), 0.8)

        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)

        health_data = { 's_bull': [], 'd_bull': [], 's_bear': [], 'd_bear': [] } 
        calculators = {
            'volatility': self._calculate_volatility_health,
            'efficiency': self._calculate_efficiency_health,
            'momentum': self._calculate_kinetic_energy_health,
            'inertia': self._calculate_inertia_health,
        }
        for name, calculator in calculators.items():
            s_bull, d_bull, s_bear, d_bear = calculator(df, norm_window, dynamic_weights, periods)
            health_data['s_bull'].append(s_bull) 
            health_data['d_bull'].append(d_bull) 
            health_data['s_bear'].append(s_bear) 
            health_data['d_bear'].append(d_bear) 

        overall_health = {}
        pillar_names = list(pillar_weights.keys())
        
        if not pillar_weights:
            print(f"        -> [动态力学引擎] 警告: 'pillar_weights' 在配置文件中缺失。将临时采用等权重融合。")
            weights_array = np.full(len(calculators), 1.0 / len(calculators))
        else:
            weights_array = np.array([pillar_weights.get(name, 0) for name in calculators.keys()])

        
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
                # 使用加权几何平均 (乘法) 替换加权求和 (加法)
                fused_values = np.prod(stacked_values ** weights_array[:, np.newaxis], axis=0)
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)

        self.strategy.atomic_states['__DYN_overall_health'] = overall_health
        
        
        bullish_resonance_health = {p: overall_health['s_bull'][p] * overall_health['d_bull'][p] for p in periods if p in overall_health.get('s_bull', {}) and p in overall_health.get('d_bull', {})}
        bullish_short_force_res = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, 0.5)
        overall_bullish_resonance = (bullish_short_force_res * resonance_tf_weights['short'] + bullish_medium_trend_res * resonance_tf_weights['medium'] + bullish_long_inertia_res * resonance_tf_weights['long'])
        
        bullish_reversal_health = {p: overall_health['s_bear'][p] * overall_health['d_bull'][p] for p in periods if p in overall_health.get('s_bear', {}) and p in overall_health.get('d_bull', {})}
        bullish_short_force_rev = (bullish_reversal_health.get(1, 0.5) * bullish_reversal_health.get(5, 0.5))**0.5
        bullish_medium_trend_rev = (bullish_reversal_health.get(13, 0.5) * bullish_reversal_health.get(21, 0.5))**0.5
        bullish_long_inertia_rev = bullish_reversal_health.get(55, 0.5)
        overall_bullish_reversal_trigger = (bullish_short_force_rev * reversal_tf_weights['short'] + bullish_medium_trend_rev * reversal_tf_weights['medium'] + bullish_long_inertia_rev * reversal_tf_weights['long'])
        final_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + bottom_context_bonus_factor * bottom_context_score)).clip(0, 1)

        bearish_resonance_health = {p: overall_health['s_bear'][p] * overall_health['d_bear'][p] for p in periods if p in overall_health.get('s_bear', {}) and p in overall_health.get('d_bear', {})}
        bearish_short_force_res = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, 0.5)
        overall_bearish_resonance = (bearish_short_force_res * resonance_tf_weights['short'] + bearish_medium_trend_res * resonance_tf_weights['medium'] + bearish_long_inertia_res * resonance_tf_weights['long'])
        
        bearish_reversal_health = {p: overall_health['s_bull'][p] * overall_health['d_bear'][p] for p in periods if p in overall_health.get('s_bull', {}) and p in overall_health.get('d_bear', {})}
        bearish_short_force_rev = (bearish_reversal_health.get(1, 0.5) * bearish_reversal_health.get(5, 0.5))**0.5
        bearish_medium_trend_rev = (bearish_reversal_health.get(13, 0.5) * bearish_reversal_health.get(21, 0.5))**0.5
        bearish_long_inertia_rev = bearish_reversal_health.get(55, 0.5)
        overall_bearish_reversal_trigger = (bearish_short_force_rev * reversal_tf_weights['short'] + bearish_medium_trend_rev * reversal_tf_weights['medium'] + bearish_long_inertia_rev * reversal_tf_weights['long'])
        final_top_reversal_score = (overall_bearish_reversal_trigger * (1 + top_context_bonus_factor * top_context_score)).clip(0, 1)
        

        for prefix, score in [('SCORE_DYN_BULLISH_RESONANCE', overall_bullish_resonance), ('SCORE_DYN_BOTTOM_REVERSAL', final_bottom_reversal_score),
                              ('SCORE_DYN_BEARISH_RESONANCE', overall_bearish_resonance), ('SCORE_DYN_TOP_REVERSAL', final_top_reversal_score)]:
            states[f'{prefix}_S_PLUS'] = score.astype(np.float32)
            states[f'{prefix}_S'] = (score * 0.8).astype(np.float32)
            states[f'{prefix}_A'] = (score * 0.6).astype(np.float32)
            states[f'{prefix}_B'] = (score * 0.4).astype(np.float32)
        
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器
    # ==============================================================================

    def _calculate_volatility_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V1.2 · 签名修复版】计算波动率(BBW)维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        static_bull = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False) # 压缩为好
        static_bear = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=True)  # 扩张为坏

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            if p in [1, 5, 13]:
                d_bull[p] = normalize_score(df.get(f'SLOPE_{p}_BBW_21_2.0_D'), df.index, norm_window, ascending=False) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_BBW_21_2.0_D'), df.index, norm_window, ascending=False) * dynamic_weights['accel']
                d_bear[p] = normalize_score(df.get(f'SLOPE_{p}_BBW_21_2.0_D'), df.index, norm_window, ascending=True) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_BBW_21_2.0_D'), df.index, norm_window, ascending=True) * dynamic_weights['accel']
            else:
                d_bull[p] = pd.Series(0.5, index=df.index, dtype=np.float32)
                d_bear[p] = pd.Series(0.5, index=df.index, dtype=np.float32)

        return s_bull, d_bull, s_bear, d_bear

    def _calculate_efficiency_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V1.2 · 签名修复版】计算效率(VPA)维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        static_bull = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window)
        static_bear = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False)

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            d_bull[p] = normalize_score(df.get(f'SLOPE_{p}_VPA_EFFICIENCY_D'), df.index, norm_window) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_VPA_EFFICIENCY_D'), df.index, norm_window) * dynamic_weights['accel']
            d_bear[p] = normalize_score(df.get(f'SLOPE_{p}_VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False) * dynamic_weights['accel']

        return s_bull, d_bull, s_bear, d_bear

    def _calculate_kinetic_energy_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V1.4 · 哲学统一版】计算动能(ATR)维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        static_bull = normalize_score(df.get('ATR_14_D'), df.index, norm_window) # 动能放大为好
        static_bear = normalize_score(df.get('ATR_14_D'), df.index, norm_window, ascending=False) # 动能萎缩为坏

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            
            d_bull_slope = normalize_score(df.get(f'SLOPE_{p}_ATR_14_D'), df.index, norm_window)
            d_bull_accel = normalize_score(df.get(f'ACCEL_{p}_ATR_14_D'), df.index, norm_window)
            # 根除加法，使用乘法（几何平均）
            d_bull[p] = (d_bull_slope * d_bull_accel)**0.5

            # 看跌动态：动能斜率、加速度减少为坏
            d_bear_slope = normalize_score(df.get(f'SLOPE_{p}_ATR_14_D'), df.index, norm_window, ascending=False)
            d_bear_accel = normalize_score(df.get(f'ACCEL_{p}_ATR_14_D'), df.index, norm_window, ascending=False)
            # 根除加法，使用乘法（几何平均）
            d_bear[p] = (d_bear_slope * d_bear_accel)**0.5
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_inertia_health(self, df: pd.DataFrame, norm_window: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V1.2 · 签名修复版】计算惯性(ADX)维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        static_bull = normalize_score(df.get('ADX_14_D'), df.index, norm_window) # 惯性强为好
        static_bear = normalize_score(df.get('ADX_14_D'), df.index, norm_window, ascending=False) # 惯性弱为坏

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            d_bull[p] = normalize_score(df.get(f'SLOPE_{p}_ADX_14_D'), df.index, norm_window) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_ADX_14_D'), df.index, norm_window) * dynamic_weights['accel']
            d_bear[p] = normalize_score(df.get(f'SLOPE_{p}_ADX_14_D'), df.index, norm_window, ascending=False) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_ADX_14_D'), df.index, norm_window, ascending=False) * dynamic_weights['accel']

        return s_bull, d_bull, s_bear, d_bear















