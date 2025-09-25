# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

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
        【V8.0 · 职责重构 & 对称逻辑版】终极动态力学信号诊断模块
        - 核心重构 (本次修改):
          - [职责净化] 移除价格、成交量、主力强度等维度，使其职责聚焦于纯粹的“力学”属性。
          - [代码重构] 将臃肿的计算逻辑拆分为多个职责单一的 `_calculate_*_health` 辅助函数。
          - [哲学统一] 全面贯彻“四维健康度”和对称逻辑，与其他所有情报引擎在哲学上完全统一。
        - 收益: 模块职责清晰，代码结构优雅，信号逻辑严谨，实现了系统哲学的最终完备。
        """
        print("        -> [终极动态力学信号诊断模块 V8.0 · 职责重构 & 对称逻辑版] 启动...") # 更新版本号和说明
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
        
        # --- 1. 定义权重与参数 ---
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)

        # --- 2. 计算“外部宏观位置”门控 (用于反转) ---
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range
        
        # --- 3. 调用所有力学健康度组件计算器 ---
        health_data = {
            'bullish_static': [], 'bullish_dynamic': [],
            'bearish_static': [], 'bearish_dynamic': []
        }
        
        calculators = {
            'volatility': self._calculate_volatility_health,
            'efficiency': self._calculate_efficiency_health,
            'kinetic_energy': self._calculate_kinetic_energy_health,
            'inertia': self._calculate_inertia_health,
        }

        for name, calculator in calculators.items():
            s_bull, d_bull, s_bear, d_bear = calculator(df, norm_window, min_periods, dynamic_weights, periods)
            health_data['bullish_static'].append(s_bull)
            health_data['bullish_dynamic'].append(d_bull)
            health_data['bearish_static'].append(s_bear)
            health_data['bearish_dynamic'].append(d_bear)

        # --- 4. 独立融合，生成四个全局健康度 ---
        overall_health = {}
        for health_type in health_data:
            overall_health[health_type] = {}
            for p in periods:
                components_for_period = [pillar_dict[p].values for pillar_dict in health_data[health_type] if p in pillar_dict]
                if components_for_period:
                    overall_health[health_type][p] = pd.Series(np.mean(np.stack(components_for_period, axis=0), axis=0), index=df.index)
                else:
                    overall_health[health_type][p] = pd.Series(0.5, index=df.index)

        # --- 5. 终极信号合成 (采用对称逻辑) ---
        # 5.1 看涨信号合成
        bullish_resonance_health = {p: overall_health['bullish_static'][p] * overall_health['bullish_dynamic'][p] for p in periods}
        bullish_short_force_res = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, 0.5)
        overall_bullish_resonance = (bullish_short_force_res * resonance_tf_weights['short'] + bullish_medium_trend_res * resonance_tf_weights['medium'] + bullish_long_inertia_res * resonance_tf_weights['long'])
        
        bullish_dynamic_health = overall_health['bullish_dynamic']
        bullish_short_force_rev = (bullish_dynamic_health.get(1, 0.5) * bullish_dynamic_health.get(5, 0.5))**0.5
        bullish_medium_trend_rev = (bullish_dynamic_health.get(13, 0.5) * bullish_dynamic_health.get(21, 0.5))**0.5
        bullish_long_inertia_rev = bullish_dynamic_health.get(55, 0.5)
        overall_bullish_reversal_trigger = (bullish_short_force_rev * reversal_tf_weights['short'] + bullish_medium_trend_rev * reversal_tf_weights['medium'] + bullish_long_inertia_rev * reversal_tf_weights['long'])
        final_bottom_reversal_score = bottom_context_score * overall_bullish_reversal_trigger

        # 5.2 看跌信号合成
        bearish_resonance_health = {p: overall_health['bearish_static'][p] * overall_health['bearish_dynamic'][p] for p in periods}
        bearish_short_force_res = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, 0.5)
        overall_bearish_resonance = (bearish_short_force_res * resonance_tf_weights['short'] + bearish_medium_trend_res * resonance_tf_weights['medium'] + bearish_long_inertia_res * resonance_tf_weights['long'])

        bearish_dynamic_health = overall_health['bearish_dynamic']
        bearish_short_force_rev = (bearish_dynamic_health.get(1, 0.5) * bearish_dynamic_health.get(5, 0.5))**0.5
        bearish_medium_trend_rev = (bearish_dynamic_health.get(13, 0.5) * bearish_dynamic_health.get(21, 0.5))**0.5
        bearish_long_inertia_rev = bearish_dynamic_health.get(55, 0.5)
        overall_bearish_reversal_trigger = (bearish_short_force_rev * reversal_tf_weights['short'] + bearish_medium_trend_rev * reversal_tf_weights['medium'] + bearish_long_inertia_rev * reversal_tf_weights['long'])
        final_top_reversal_score = top_context_score * overall_bearish_reversal_trigger

        # 5.3 赋值
        for prefix, score in [('SCORE_DYN_BULLISH_RESONANCE', overall_bullish_resonance), ('SCORE_DYN_BOTTOM_REVERSAL', final_bottom_reversal_score),
                              ('SCORE_DYN_BEARISH_RESONANCE', overall_bearish_resonance), ('SCORE_DYN_TOP_REVERSAL', final_top_reversal_score)]:
            states[f'{prefix}_S_PLUS'] = (score ** exponent).astype(np.float32)
            states[f'{prefix}_S'] = (states[f'{prefix}_S_PLUS'] * 0.8).astype(np.float32)
            states[f'{prefix}_A'] = (states[f'{prefix}_S_PLUS'] * 0.6).astype(np.float32)
            states[f'{prefix}_B'] = (states[f'{prefix}_S_PLUS'] * 0.4).astype(np.float32)
        
        return states

    # ==============================================================================
    # 以下为重构后的健康度组件计算器
    # ==============================================================================

    def _calculate_volatility_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V1.0 · 新增】计算波动率(BBW)维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        static_bull = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, min_periods, ascending=False) # 压缩为好
        static_bear = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, min_periods, ascending=True)  # 扩张为坏

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            
            # 动态分只在中短期有意义
            if p in [1, 5, 13]:
                d_bull[p] = normalize_score(df.get(f'SLOPE_{p}_BBW_21_2.0_D'), df.index, norm_window, min_periods, ascending=False) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_BBW_21_2.0_D'), df.index, norm_window, min_periods) * dynamic_weights['accel']
                d_bear[p] = normalize_score(df.get(f'SLOPE_{p}_BBW_21_2.0_D'), df.index, norm_window, min_periods, ascending=True) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_BBW_21_2.0_D'), df.index, norm_window, min_periods, ascending=True) * dynamic_weights['accel']
            else: # 长周期波动率动态意义不大，给中性分
                d_bull[p] = pd.Series(0.5, index=df.index, dtype=np.float32)
                d_bear[p] = pd.Series(0.5, index=df.index, dtype=np.float32)

        return s_bull, d_bull, s_bear, d_bear

    def _calculate_efficiency_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V1.0 · 新增】计算效率(VPA)维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        static_bull = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, min_periods)
        static_bear = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, min_periods, ascending=False)

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            d_bull[p] = normalize_score(df.get(f'SLOPE_{p}_VPA_EFFICIENCY_D'), df.index, norm_window, min_periods) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_VPA_EFFICIENCY_D'), df.index, norm_window, min_periods) * dynamic_weights['accel']
            d_bear[p] = normalize_score(df.get(f'SLOPE_{p}_VPA_EFFICIENCY_D'), df.index, norm_window, min_periods, ascending=False) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_VPA_EFFICIENCY_D'), df.index, norm_window, min_periods, ascending=False) * dynamic_weights['accel']

        return s_bull, d_bull, s_bear, d_bear

    def _calculate_kinetic_energy_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V1.0 · 新增】计算动能(ATR)维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        static_bull = normalize_score(df.get('ATR_14_D'), df.index, norm_window, min_periods) # 动能放大为好
        static_bear = normalize_score(df.get('ATR_14_D'), df.index, norm_window, min_periods, ascending=False) # 动能萎缩为坏

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            d_bull[p] = normalize_score(df.get(f'SLOPE_{p}_ATR_14_D'), df.index, norm_window, min_periods) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_ATR_14_D'), df.index, norm_window, min_periods) * dynamic_weights['accel']
            d_bear[p] = normalize_score(df.get(f'SLOPE_{p}_ATR_14_D'), df.index, norm_window, min_periods, ascending=True) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_ATR_14_D'), df.index, norm_window, min_periods, ascending=True) * dynamic_weights['accel'] # 看跌时，动能放大也是风险

        return s_bull, d_bull, s_bear, d_bear

    def _calculate_inertia_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, dynamic_weights: Dict, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """【V1.0 · 新增】计算惯性(ADX)维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        static_bull = normalize_score(df.get('ADX_14_D'), df.index, norm_window, min_periods) # 惯性强为好
        static_bear = normalize_score(df.get('ADX_14_D'), df.index, norm_window, min_periods, ascending=False) # 惯性弱为坏

        for p in periods:
            s_bull[p] = static_bull
            s_bear[p] = static_bear
            d_bull[p] = normalize_score(df.get(f'SLOPE_{p}_ADX_14_D'), df.index, norm_window, min_periods) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_ADX_14_D'), df.index, norm_window, min_periods) * dynamic_weights['accel']
            d_bear[p] = normalize_score(df.get(f'SLOPE_{p}_ADX_14_D'), df.index, norm_window, min_periods, ascending=False) * dynamic_weights['slope'] + normalize_score(df.get(f'ACCEL_{p}_ADX_14_D'), df.index, norm_window, min_periods, ascending=False) * dynamic_weights['accel']

        return s_bull, d_bull, s_bear, d_bear















