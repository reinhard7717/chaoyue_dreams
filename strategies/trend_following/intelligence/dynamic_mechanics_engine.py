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
        【V3.4 · 终极修复版】终极动态力学信号诊断模块
        - 核心修复 (本次修改):
          - [BUG修复-1] 彻底检查并统一了 `bullish_health` 和 `bearish_health` 的变量名使用，解决了 `NameError`。
          - [BUG修复-2] 修正了S+级信号的计算公式，通过先计算原始分再统一缩放，避免了“双重缩放”逻辑错误。
        - 收益: 确保所有终极动态力学信号都能正确、稳定地生成，且数值量纲合理。
        """
        print("        -> [终极动态力学信号诊断模块 V3.4 · 终极修复版] 启动...") # [代码修改] 更新版本号
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
            
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)
        
        # --- 定义“位置上下文”分数 ---
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9) # 使用一个极小值代替np.nan
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        metrics = [
            'close_D', 'volume_D', 'BBW_21_2.0_D', 'VPA_EFFICIENCY_D',
            'main_force_flow_intensity_ratio_D', 'ATR_14_D', 'ADX_14_D'
        ]
        required_cols = set(metrics)
        for p in periods:
            for metric in metrics:
                if metric in ['BBW_21_2.0_D'] and p not in [5]:
                     continue
                # 动态力学引擎需要的数据列名不包含指标本身，直接是SLOPE_{p}_{metric}
                # 修正：确保所有需要的列都被检查
                if metric == 'main_force_flow_intensity_ratio_D':
                    # 这个指标可能没有斜率和加速度，需要健壮性处理
                    # 假设它有，如果数据工程层没有生成，下面会报错
                    pass
                required_cols.add(f'SLOPE_{p}_{metric}')
                required_cols.add(f'ACCEL_{p}_{metric}')
        missing_cols = list(required_cols - set(df.columns))
        if missing_cols:
            print(f"          -> [严重警告] 终极动态力学引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        price_static = self._normalize_series(df['close_D'], norm_window, min_periods)
        price_mom = {p: self._normalize_series(df[f'SLOPE_{p}_close_D'], norm_window, min_periods) for p in periods}
        price_accel = {p: self._normalize_series(df[f'ACCEL_{p}_close_D'], norm_window, min_periods) for p in periods}
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
        # --- 3. 计算每个周期的“完美动态力学健康度” (Intra-Timeframe Validation) ---
        bullish_health = {}
        bearish_health = {}
        price_static_arr = price_static.values
        volume_static_arr = volume_static.values
        volatility_static_arr = volatility_static.values
        efficiency_static_arr = efficiency_static.values
        force_quality_static_arr = force_quality_static.values
        kinetic_energy_static_arr = kinetic_energy_static.values
        inertia_static_arr = inertia_static.values
        for p in periods:
            price_health_arr = (price_static_arr * price_mom[p].values * price_accel[p].values)**(1/3)
            volume_health_arr = (volume_static_arr * volume_mom[p].values * volume_accel[p].values)**(1/3)
            if p in volatility_mom:
                 volatility_health_arr = (volatility_static_arr * volatility_mom[p].values * volatility_accel[p].values)**(1/3)
            else:
                 volatility_health_arr = volatility_static_arr
            efficiency_health_arr = (efficiency_static_arr * efficiency_mom[p].values * efficiency_accel[p].values)**(1/3)
            force_quality_health_arr = (force_quality_static_arr * force_quality_mom[p].values * force_quality_accel[p].values)**(1/3)
            kinetic_energy_health_arr = (kinetic_energy_static_arr * kinetic_energy_mom[p].values * kinetic_energy_accel[p].values)**(1/3)
            inertia_health_arr = (inertia_static_arr * inertia_mom[p].values * inertia_accel[p].values)**(1/3)
            health_components_arr = np.stack([
                price_health_arr, volume_health_arr, volatility_health_arr, efficiency_health_arr,
                force_quality_health_arr, kinetic_energy_health_arr, inertia_health_arr
            ], axis=0)
            final_health_arr = np.prod(health_components_arr, axis=0)**(1/7)
            bullish_health[p] = pd.Series(final_health_arr, index=df.index, dtype=np.float32)
            bearish_health_components_arr = np.stack([
                1 - price_health_arr, 1 - volume_health_arr, 1 - volatility_health_arr, 1 - efficiency_health_arr,
                1 - force_quality_health_arr, 1 - kinetic_energy_health_arr, 1 - inertia_health_arr
            ], axis=0)
            final_bearish_health_arr = np.prod(bearish_health_components_arr, axis=0)**(1/7)
            bearish_health[p] = pd.Series(final_bearish_health_arr, index=df.index, dtype=np.float32)
        
        # [代码修改] 统一使用 bullish_health 和 bearish_health 字典，并分离原始信号计算
        # --- 4. 定义原始信号组件 (Raw Components) ---
        bullish_short_force = (bullish_health[1] * bullish_health[5])**0.5
        bullish_medium_trend = (bullish_health[13] * bullish_health[21])**0.5
        bullish_long_inertia = bullish_health[55]
        bearish_short_force = (bearish_health[1] * bearish_health[5])**0.5
        bearish_medium_trend = (bearish_health[13] * bearish_health[21])**0.5
        bearish_long_inertia = bearish_health[55]

        # --- 5. 定义原始信号 (Raw Signals) ---
        raw_bullish_s = (bullish_short_force * bullish_medium_trend)
        raw_bullish_s_plus = (raw_bullish_s * bullish_long_inertia)
        raw_bearish_s = (bearish_short_force * bearish_medium_trend)
        raw_bearish_s_plus = (raw_bearish_s * bearish_long_inertia)
        
        # --- 6. 对所有终极信号应用指数缩放 ---
        states['SCORE_DYN_BULLISH_RESONANCE_B'] = (bullish_health[5] ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_A'] = ((bullish_health[5] * bullish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S'] = (raw_bullish_s ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S_PLUS'] = (raw_bullish_s_plus ** exponent).astype(np.float32)
        
        states['SCORE_DYN_BEARISH_RESONANCE_B'] = (bearish_health[5] ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_A'] = ((bearish_health[5] * bearish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S'] = (raw_bearish_s ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S_PLUS'] = (raw_bearish_s_plus ** exponent).astype(np.float32)
        
        states['SCORE_DYN_BOTTOM_REVERSAL_B'] = ((bottom_context_score * bullish_health[1] * bearish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_A'] = ((bottom_context_score * bullish_health[5] * bearish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S'] = ((bottom_context_score * bullish_short_force * bearish_long_inertia) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S_PLUS'] = ((bottom_context_score * bullish_short_force * bullish_medium_trend * bearish_long_inertia) ** exponent).astype(np.float32)
        
        states['SCORE_DYN_TOP_REVERSAL_B'] = ((top_context_score * bearish_health[1] * bullish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_A'] = ((top_context_score * bearish_health[5] * bearish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S'] = ((top_context_score * bearish_short_force * bullish_long_inertia) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S_PLUS'] = ((top_context_score * bearish_short_force * bearish_medium_trend * bullish_long_inertia) ** exponent).astype(np.float32)
        
        return states











