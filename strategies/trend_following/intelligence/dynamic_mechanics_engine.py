# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def _normalize_series(self, series: pd.Series, norm_window: int, min_periods: int, ascending: bool = True) -> pd.Series:
        """
        辅助函数：将Pandas Series进行滚动窗口排名归一化。
        """
        # 使用滚动窗口计算百分比排名，空值用0.5填充，代表中位数水平
        rank = series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        # 根据ascending参数决定排名方向
        result = rank if ascending else 1 - rank
        return result.astype(np.float32)

    def run_dynamic_analysis_command(self) -> None:
        """
        【V3.0 终极信号版】动态力学引擎总指挥
        - 核心重构: 遵循 `BehavioralIntelligence` 的终极信号范式。
                      本模块不再返回一堆零散的原子信号，而是调用唯一的终极信号引擎
                      `diagnose_ultimate_dynamic_mechanics_signals`，并将其产出的
                      16个S+/S/A/B级信号作为本模块的最终输出。
        - 收益: 架构与行为、筹码等其他情报模块完全统一，极大提升了信号质量和架构清晰度。
                上层模块只需消费这16个经过深度交叉验证的终极动态力学信号。
        """
        # print("    -> [动态力学引擎总指挥 V3.0 终极信号版] 启动...")
        # 直接调用终极信号引擎
        ultimate_dynamic_states = self.diagnose_ultimate_dynamic_mechanics_signals(self.strategy.df_indicators)
        # 将其结果作为本模块的唯一输出，更新到原子状态库
        if ultimate_dynamic_states:
            self.strategy.atomic_states.update(ultimate_dynamic_states)
            # print(f"    -> [动态力学引擎总指挥 V3.0] 分析完毕，共生成 {len(ultimate_dynamic_states)} 个终极动态力学信号。")

    def diagnose_ultimate_dynamic_mechanics_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 · 意图驱动加权版】终极动态力学信号诊断模块
        - 核心重构 (本次修改):
          - [哲学升级] 引入“意图驱动加权”范式，为“共振”和“反转”信号定义不同的时间框架权重。
          - [动态优先] 所有七个子维度的健康度计算都彻底移除静态值。
          - [新范式] 共振信号重用中长周期，反转信号重用短中周期。
        - 收益: 实现了信号的“专业化”，力学共振信号更稳定，力学反转信号更灵敏。
        """
        print("        -> [终极动态力学信号诊断模块 V5.0 · 意图驱动加权版] 启动...") # 更新版本号和说明
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)

        # 定义意图驱动的时间框架权重
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        # 定义纯动态指标的内部权重
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}

        # 上下文分数计算保持不变
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range
        
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        
        bullish_health = {}
        
        # 预计算所有动态分数
        price_mom = {p: self._normalize_series(df[f'SLOPE_{p}_close_D'], norm_window, min_periods) for p in periods}
        price_accel = {p: self._normalize_series(df[f'ACCEL_{p}_close_D'], norm_window, min_periods) for p in periods}
        volume_mom = {p: self._normalize_series(df[f'SLOPE_{p}_volume_D'], norm_window, min_periods) for p in periods}
        volume_accel = {p: self._normalize_series(df[f'ACCEL_{p}_volume_D'], norm_window, min_periods) for p in periods}
        volatility_mom = {p: self._normalize_series(df[f'SLOPE_{p}_BBW_21_2.0_D'], norm_window, min_periods) for p in [5]}
        volatility_accel = {p: self._normalize_series(df[f'ACCEL_{p}_BBW_21_2.0_D'], norm_window, min_periods) for p in [5]}
        efficiency_mom = {p: self._normalize_series(df[f'SLOPE_{p}_VPA_EFFICIENCY_D'], norm_window, min_periods) for p in periods}
        efficiency_accel = {p: self._normalize_series(df[f'ACCEL_{p}_VPA_EFFICIENCY_D'], norm_window, min_periods) for p in periods}
        force_quality_mom = {p: self._normalize_series(df[f'SLOPE_{p}_main_force_flow_intensity_ratio_D'], norm_window, min_periods) for p in periods}
        force_quality_accel = {p: self._normalize_series(df[f'ACCEL_{p}_main_force_flow_intensity_ratio_D'], norm_window, min_periods) for p in periods}
        kinetic_energy_mom = {p: self._normalize_series(df[f'SLOPE_{p}_ATR_14_D'], norm_window, min_periods) for p in periods}
        kinetic_energy_accel = {p: self._normalize_series(df[f'ACCEL_{p}_ATR_14_D'], norm_window, min_periods) for p in periods}
        inertia_mom = {p: self._normalize_series(df[f'SLOPE_{p}_ADX_14_D'], norm_window, min_periods) for p in periods}
        inertia_accel = {p: self._normalize_series(df[f'ACCEL_{p}_ADX_14_D'], norm_window, min_periods) for p in periods}

        for p in periods:
            # 所有子维度的健康度计算都彻底移除静态值
            price_health = price_mom[p] * dynamic_weights['slope'] + price_accel[p] * dynamic_weights['accel']
            volume_health = volume_mom[p] * dynamic_weights['slope'] + volume_accel[p] * dynamic_weights['accel']
            if p in volatility_mom:
                volatility_health = (1 - volatility_mom[p]) * dynamic_weights['slope'] + volatility_accel[p] * dynamic_weights['accel']
            else:
                volatility_health = 0.5
            efficiency_health = efficiency_mom[p] * dynamic_weights['slope'] + efficiency_accel[p] * dynamic_weights['accel']
            force_quality_health = force_quality_mom[p] * dynamic_weights['slope'] + force_quality_accel[p] * dynamic_weights['accel']
            kinetic_energy_health = kinetic_energy_mom[p] * dynamic_weights['slope'] + kinetic_energy_accel[p] * dynamic_weights['accel']
            inertia_health = inertia_mom[p] * dynamic_weights['slope'] + inertia_accel[p] * dynamic_weights['accel']
            
            health_components = [price_health, volume_health, volatility_health, efficiency_health, force_quality_health, kinetic_energy_health, inertia_health]
            bullish_health[p] = pd.Series(np.mean([s.values for s in health_components], axis=0), index=df.index, dtype=np.float32)
            
        # --- 使用新的加权范式重构信号合成 ---
        bullish_short_force = (bullish_health[1] * bullish_health[5])**0.5
        bullish_medium_trend = (bullish_health[13] * bullish_health[21])**0.5
        bullish_long_inertia = bullish_health[55]
        
        # 1. 看涨共振信号合成
        overall_bullish_resonance = (bullish_short_force * resonance_tf_weights['short'] +
                                     bullish_medium_trend * resonance_tf_weights['medium'] +
                                     bullish_long_inertia * resonance_tf_weights['long'])
        states['SCORE_DYN_BULLISH_RESONANCE_S_PLUS'] = (overall_bullish_resonance ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S'] = (states['SCORE_DYN_BULLISH_RESONANCE_S_PLUS'] * 0.8).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_A'] = (states['SCORE_DYN_BULLISH_RESONANCE_S_PLUS'] * 0.6).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_B'] = (states['SCORE_DYN_BULLISH_RESONANCE_S_PLUS'] * 0.4).astype(np.float32)

        # 2. 底部反转信号合成
        overall_bullish_reversal_trigger = (bullish_short_force * reversal_tf_weights['short'] +
                                            bullish_medium_trend * reversal_tf_weights['medium'] +
                                            bullish_long_inertia * reversal_tf_weights['long'])
        final_bottom_reversal_score = bottom_context_score * overall_bullish_reversal_trigger
        states['SCORE_DYN_BOTTOM_REVERSAL_S_PLUS'] = (final_bottom_reversal_score ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S'] = (states['SCORE_DYN_BOTTOM_REVERSAL_S_PLUS'] * 0.8).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_A'] = (states['SCORE_DYN_BOTTOM_REVERSAL_S_PLUS'] * 0.6).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_B'] = (states['SCORE_DYN_BOTTOM_REVERSAL_S_PLUS'] * 0.4).astype(np.float32)

        # 3. 对称实现下跌共振和顶部反转信号
        bearish_short_force = ((1 - bullish_health[1]) * (1 - bullish_health[5]))**0.5
        bearish_medium_trend = ((1 - bullish_health[13]) * (1 - bullish_health[21]))**0.5
        bearish_long_inertia = (1 - bullish_health[55])
        overall_bearish_resonance = (bearish_short_force * resonance_tf_weights['short'] +
                                     bearish_medium_trend * resonance_tf_weights['medium'] +
                                     bearish_long_inertia * resonance_tf_weights['long'])
        states['SCORE_DYN_BEARISH_RESONANCE_S_PLUS'] = (overall_bearish_resonance ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S'] = (states['SCORE_DYN_BEARISH_RESONANCE_S_PLUS'] * 0.8).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_A'] = (states['SCORE_DYN_BEARISH_RESONANCE_S_PLUS'] * 0.6).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_B'] = (states['SCORE_DYN_BEARISH_RESONANCE_S_PLUS'] * 0.4).astype(np.float32)
        
        overall_bearish_reversal_trigger = (bearish_short_force * reversal_tf_weights['short'] +
                                            bearish_medium_trend * reversal_tf_weights['medium'] +
                                            bearish_long_inertia * reversal_tf_weights['long'])
        final_top_reversal_score = top_context_score * overall_bearish_reversal_trigger
        states['SCORE_DYN_TOP_REVERSAL_S_PLUS'] = (final_top_reversal_score ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S'] = (states['SCORE_DYN_TOP_REVERSAL_S_PLUS'] * 0.8).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_A'] = (states['SCORE_DYN_TOP_REVERSAL_S_PLUS'] * 0.6).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_B'] = (states['SCORE_DYN_TOP_REVERSAL_S_PLUS'] * 0.4).astype(np.float32)
        
        return states











