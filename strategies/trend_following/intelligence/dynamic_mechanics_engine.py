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
        【V3.6 · 信号融合重构版】终极动态力学信号诊断模块
        - 核心重构 (本次修改):
          - [信号哲学重构] 与 structural_intelligence V2.0 同步，废除了旧的基于“乘法融合”的健康度计算，全面转向基于“加权平均”的新范式。
        - 收益: 彻底解决了因“几何平均暴政”导致底层反转信号过弱的问题，将显著提升在关键反转日的信号强度。
        """
        print("        -> [终极动态力学信号诊断模块 V3.6 · 信号融合重构版] 启动...") # 更新版本号和说明
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # 定义新的加权平均权重
        health_weights = {'static': 0.2, 'slope': 0.5, 'accel': 0.3}
        # 使用加权平均重构健康度计算
        bullish_health = {}
        bearish_health = {}
        # 预计算所有归一化分数
        price_static = self._normalize_series(df['close_D'], norm_window, min_periods)
        price_mom = {p: self._normalize_series(df[f'SLOPE_{p}_close_D'], norm_window, min_periods) for p in periods}
        price_accel = {p: self._normalize_series(df[f'ACCEL_{p}_close_D'], norm_window, min_periods) for p in periods}
        # ... (此处省略其他指标的预计算，逻辑与原版一致) ...
        volume_static = self._normalize_series(df['volume_D'], norm_window, min_periods)
        volume_mom = {p: self._normalize_series(df[f'SLOPE_{p}_volume_D'], norm_window, min_periods) for p in periods}
        volume_accel = {p: self._normalize_series(df[f'ACCEL_{p}_volume_D'], norm_window, min_periods) for p in periods}
        volatility_static = self._normalize_series(df['BBW_21_2.0_D'], norm_window, min_periods, ascending=False)
        volatility_mom = {p: self._normalize_series(df[f'SLOPE_{p}_BBW_21_2.0_D'], norm_window, min_periods) for p in [5]}
        volatility_accel = {p: self._normalize_series(df[f'ACCEL_{p}_BBW_21_2.0_D'], norm_window, min_periods) for p in [5]}
        efficiency_static = self._normalize_series(df['VPA_EFFICIENCY_D'], norm_window, min_periods)
        efficiency_mom = {p: self._normalize_series(df[f'SLOPE_{p}_VPA_EFFICIENCY_D'], norm_window, min_periods) for p in periods}
        efficiency_accel = {p: self._normalize_series(df[f'ACCEL_{p}_VPA_EFFICIENCY_D'], norm_window, min_periods) for p in periods}
        force_quality_static = self._normalize_series(df['main_force_flow_intensity_ratio_D'], norm_window, min_periods)
        force_quality_mom = {p: self._normalize_series(df[f'SLOPE_{p}_main_force_flow_intensity_ratio_D'], norm_window, min_periods) for p in periods}
        force_quality_accel = {p: self._normalize_series(df[f'ACCEL_{p}_main_force_flow_intensity_ratio_D'], norm_window, min_periods) for p in periods}
        kinetic_energy_static = self._normalize_series(df['ATR_14_D'], norm_window, min_periods)
        kinetic_energy_mom = {p: self._normalize_series(df[f'SLOPE_{p}_ATR_14_D'], norm_window, min_periods) for p in periods}
        kinetic_energy_accel = {p: self._normalize_series(df[f'ACCEL_{p}_ATR_14_D'], norm_window, min_periods) for p in periods}
        inertia_static = self._normalize_series(df['ADX_14_D'], norm_window, min_periods)
        inertia_mom = {p: self._normalize_series(df[f'SLOPE_{p}_ADX_14_D'], norm_window, min_periods) for p in periods}
        inertia_accel = {p: self._normalize_series(df[f'ACCEL_{p}_ADX_14_D'], norm_window, min_periods) for p in periods}
        for p in periods:
            price_health = price_static * health_weights['static'] + price_mom[p] * health_weights['slope'] + price_accel[p] * health_weights['accel']
            volume_health = volume_static * health_weights['static'] + volume_mom[p] * health_weights['slope'] + volume_accel[p] * health_weights['accel']
            if p in volatility_mom:
                volatility_health = volatility_static * health_weights['static'] + volatility_mom[p] * health_weights['slope'] + volatility_accel[p] * health_weights['accel']
            else:
                volatility_health = volatility_static # 如果没有动态数据，则只使用静态分
            efficiency_health = efficiency_static * health_weights['static'] + efficiency_mom[p] * health_weights['slope'] + efficiency_accel[p] * health_weights['accel']
            force_quality_health = force_quality_static * health_weights['static'] + force_quality_mom[p] * health_weights['slope'] + force_quality_accel[p] * health_weights['accel']
            kinetic_energy_health = kinetic_energy_static * health_weights['static'] + kinetic_energy_mom[p] * health_weights['slope'] + kinetic_energy_accel[p] * health_weights['accel']
            inertia_health = inertia_static * health_weights['static'] + inertia_mom[p] * health_weights['slope'] + inertia_accel[p] * health_weights['accel']
            
            health_components = [price_health, volume_health, volatility_health, efficiency_health, force_quality_health, kinetic_energy_health, inertia_health]
            bullish_health[p] = pd.Series(np.mean([s.values for s in health_components], axis=0), index=df.index, dtype=np.float32)
            bearish_health[p] = 1 - bullish_health[p]
        # 后续的信号合成逻辑保持不变
        bullish_short_force = (bullish_health[1] * bullish_health[5])**0.5
        bullish_medium_trend = (bullish_health[13] * bullish_health[21])**0.5
        bullish_long_inertia = bullish_health[55]
        bearish_short_force = (bearish_health[1] * bearish_health[5])**0.5
        bearish_medium_trend = (bearish_health[13] * bearish_health[21])**0.5
        bearish_long_inertia = bearish_health[55]
        raw_bullish_s = (bullish_short_force * bullish_medium_trend)
        raw_bullish_s_plus = (raw_bullish_s * bullish_long_inertia)
        raw_bearish_s = (bearish_short_force * bearish_medium_trend)
        raw_bearish_s_plus = (raw_bearish_s * bearish_long_inertia)
        states['SCORE_DYN_BULLISH_RESONANCE_B'] = (bullish_health[5] ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_A'] = ((bullish_health[5] * bullish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S'] = (raw_bullish_s ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S_PLUS'] = (raw_bullish_s_plus ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_B'] = (bearish_health[5] ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_A'] = ((bearish_health[5] * bearish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S'] = (raw_bearish_s ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S_PLUS'] = (raw_bearish_s_plus ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_B'] = ((bottom_context_score * bullish_health[1] * (1 - bullish_health[21])) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_A'] = ((bottom_context_score * bullish_health[5] * (1 - bullish_health[21])) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S'] = ((bottom_context_score * bullish_short_force * (1 - bullish_long_inertia)) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S_PLUS'] = ((bottom_context_score * bullish_short_force * bullish_medium_trend * (1 - bullish_long_inertia)) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_B'] = ((top_context_score * bearish_health[1] * (1 - bearish_health[21])) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_A'] = ((top_context_score * bearish_health[5] * (1 - bearish_health[21])) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S'] = ((top_context_score * bearish_short_force * (1 - bearish_long_inertia)) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S_PLUS'] = ((top_context_score * bearish_short_force * bearish_medium_trend * (1 - bearish_long_inertia)) ** exponent).astype(np.float32)
        return states











