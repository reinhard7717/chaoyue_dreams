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
        return rank if ascending else 1 - rank

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
        print("    -> [动态力学引擎总指挥 V3.0 终极信号版] 启动...")
        # 直接调用终极信号引擎
        ultimate_dynamic_states = self.diagnose_ultimate_dynamic_mechanics_signals(self.strategy.df_indicators)
        # 将其结果作为本模块的唯一输出，更新到原子状态库
        if ultimate_dynamic_states:
            self.strategy.atomic_states.update(ultimate_dynamic_states)
            print(f"    -> [动态力学引擎总指挥 V3.0] 分析完毕，共生成 {len(ultimate_dynamic_states)} 个终极动态力学信号。")

    def diagnose_ultimate_dynamic_mechanics_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 七维力学版】终极动态力学信号诊断模块
        - 核心重构 (本次修改):
          - 1. 维度扩展: 在V2.0纯粹力学引擎的四维（价、量、波、效）基础上，新增三大核心力学维度：
              - 力的性质 (主力资金强度): `main_force_flow_intensity_ratio_D`
              - 动能 (真实波幅): `ATR_14_D`
              - 惯性 (趋势强度): `ADX_14_D`
          - 2. 模型升级: 构建了一个七维超完备力学模型，对每个周期的“完美动态力学健康度”进行更深度的交叉验证。
        - 收益: 引擎对市场动态的刻画更全面、更立体。例如，它现在不仅知道趋势在加速（价格），还知道这种加速是由“主力”推动（力的性质），且趋势本身的“惯性”正在增强，极大提升了信号的置信度和可靠性。
        """
        print("        -> [终极动态力学信号诊断模块 V3.0 七维力学版] 启动...")
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        metrics = [
            'close_D', 'volume_D', 'BBW_21_2.0_D', 'VPA_EFFICIENCY_D',
            'main_force_flow_intensity_ratio_D', 'ATR_14_D', 'ADX_14_D'
        ]
        required_cols = set(metrics) # 静态值
        for p in periods:
            for metric in metrics:
                # 动态构建所需列名，并处理特殊情况
                if metric in ['BBW_21_2.0_D'] and p not in [5]: # BBW的斜率/加速度通常只计算短期
                     continue
                required_cols.add(f'SLOPE_{p}_{metric}')
                required_cols.add(f'ACCEL_{p}_{metric}')
        missing_cols = list(required_cols - set(df.columns))
        if missing_cols:
            print(f"          -> [严重警告] 终极动态力学引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        # 1. 价格健康度 (越高越好)
        price_static = self._normalize_series(df['close_D'], norm_window, min_periods)
        price_mom = {p: self._normalize_series(df[f'SLOPE_{p}_close_D'], norm_window, min_periods) for p in periods}
        price_accel = {p: self._normalize_series(df[f'ACCEL_{p}_close_D'], norm_window, min_periods) for p in periods}
        # 2. 成交量健康度 (放量/增量为优)
        volume_static = self._normalize_series(df['volume_D'], norm_window, min_periods)
        volume_mom = {p: self._normalize_series(df[f'SLOPE_{p}_volume_D'], norm_window, min_periods) for p in periods}
        volume_accel = {p: self._normalize_series(df[f'ACCEL_{p}_volume_D'], norm_window, min_periods) for p in periods}
        # 3. 波动率健康度 (压缩蓄势为优静态, 扩张爆发为优动态)
        volatility_static = self._normalize_series(df['BBW_21_2.0_D'], norm_window, min_periods, ascending=False)
        volatility_mom = {p: self._normalize_series(df[f'SLOPE_{p}_BBW_21_2.0_D'], norm_window, min_periods) for p in [5]}
        volatility_accel = {p: self._normalize_series(df[f'ACCEL_{p}_BBW_21_2.0_D'], norm_window, min_periods) for p in [5]}
        # 4. 效率健康度 (越高越好)
        efficiency_static = self._normalize_series(df['VPA_EFFICIENCY_D'], norm_window, min_periods)
        efficiency_mom = {p: self._normalize_series(df[f'SLOPE_{p}_VPA_EFFICIENCY_D'], norm_window, min_periods) for p in periods}
        efficiency_accel = {p: self._normalize_series(df[f'ACCEL_{p}_VPA_EFFICIENCY_D'], norm_window, min_periods) for p in periods}
        # 5. 力的性质健康度 (主力强度越高越好)
        force_quality_static = self._normalize_series(df['main_force_flow_intensity_ratio_D'], norm_window, min_periods)
        force_quality_mom = {p: self._normalize_series(df[f'SLOPE_{p}_main_force_flow_intensity_ratio_D'], norm_window, min_periods) for p in periods}
        force_quality_accel = {p: self._normalize_series(df[f'ACCEL_{p}_main_force_flow_intensity_ratio_D'], norm_window, min_periods) for p in periods}
        # 6. 动能健康度 (波动越大, 动能越强)
        kinetic_energy_static = self._normalize_series(df['ATR_14_D'], norm_window, min_periods)
        kinetic_energy_mom = {p: self._normalize_series(df[f'SLOPE_{p}_ATR_14_D'], norm_window, min_periods) for p in periods}
        kinetic_energy_accel = {p: self._normalize_series(df[f'ACCEL_{p}_ATR_14_D'], norm_window, min_periods) for p in periods}
        # 7. 惯性健康度 (趋势强度越大越好)
        inertia_static = self._normalize_series(df['ADX_14_D'], norm_window, min_periods)
        inertia_mom = {p: self._normalize_series(df[f'SLOPE_{p}_ADX_14_D'], norm_window, min_periods) for p in periods}
        inertia_accel = {p: self._normalize_series(df[f'ACCEL_{p}_ADX_14_D'], norm_window, min_periods) for p in periods}
        # --- 3. 计算每个周期的“完美动态力学健康度” (Intra-Timeframe Validation) ---
        bullish_health = {}
        for p in periods:
            price_health = (price_static * price_mom[p] * price_accel[p])**(1/3)
            volume_health = (volume_static * volume_mom[p] * volume_accel[p])**(1/3)
            if p in volatility_mom:
                 volatility_health = (volatility_static * volatility_mom[p] * volatility_accel[p])**(1/3)
            else:
                 volatility_health = volatility_static
            efficiency_health = (efficiency_static * efficiency_mom[p] * efficiency_accel[p])**(1/3)
            force_quality_health = (force_quality_static * force_quality_mom[p] * force_quality_accel[p])**(1/3)
            kinetic_energy_health = (kinetic_energy_static * kinetic_energy_mom[p] * kinetic_energy_accel[p])**(1/3)
            inertia_health = (inertia_static * inertia_mom[p] * inertia_accel[p])**(1/3)
            # 最终的周期健康度是七大模块健康度的几何平均
            bullish_health[p] = (price_health * volume_health * volatility_health * efficiency_health * force_quality_health * kinetic_energy_health * inertia_health)**(1/7)
        bearish_health = {p: 1.0 - bullish_health[p] for p in periods}
        # --- 4. 定义信号组件 (此部分逻辑不变) ---
        bullish_short_force = (bullish_health[1] * bullish_health[5])**0.5
        bullish_medium_trend = (bullish_health[13] * bullish_health[21])**0.5
        bullish_long_inertia = bullish_health[55]
        bearish_short_force = (bearish_health[1] * bearish_health[5])**0.5
        bearish_medium_trend = (bearish_health[13] * bearish_health[21])**0.5
        bearish_long_inertia = bearish_health[55]
        # --- 5. 共振信号合成 (此部分逻辑不变) ---
        states['SCORE_DYN_BULLISH_RESONANCE_B'] = bullish_health[5].astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_A'] = (bullish_health[5] * bullish_health[21]).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S'] = (bullish_short_force * bullish_medium_trend).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S_PLUS'] = (states['SCORE_DYN_BULLISH_RESONANCE_S'] * bullish_long_inertia).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_B'] = bearish_health[5].astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_A'] = (bearish_health[5] * bearish_health[21]).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S'] = (bearish_short_force * bearish_medium_trend).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S_PLUS'] = (states['SCORE_DYN_BEARISH_RESONANCE_S'] * bearish_long_inertia).astype(np.float32)
        # --- 6. 反转信号合成 (此部分逻辑不变) ---
        states['SCORE_DYN_BOTTOM_REVERSAL_B'] = (bullish_health[1] * bearish_health[21]).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_A'] = (bullish_health[5] * bearish_health[21]).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S'] = (bullish_short_force * bearish_long_inertia).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S_PLUS'] = (bullish_short_force * bearish_medium_trend * bearish_long_inertia).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_B'] = (bearish_health[1] * bullish_health[21]).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_A'] = (bearish_health[5] * bullish_health[21]).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S'] = (bearish_short_force * bullish_long_inertia).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S_PLUS'] = (bearish_short_force * bearish_medium_trend * bullish_long_inertia).astype(np.float32)
        print(f"        -> [终极动态力学信号诊断模块 V3.0] 分析完毕，生成 {len(states)} 个终极信号。")
        return states












