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
        【V3.3 · 信号缩放版】终极动态力学信号诊断模块
        - 核心升级 (本次修改):
          - [信号缩放] 对所有16个终极信号的计算结果，统一应用指数缩放（开根号），
                        从根本上解决因多因子乘法导致的“量纲压缩”问题。
        - 收益: 确保所有终极信号的量级都在一个合理范围内，使其在计分系统中能发挥应有的作用。
        """
        print("        -> [终极动态力学信号诊断模块 V3.3 · 信号缩放版] 启动...") # [代码修改] 更新版本号
        states = {}
        p_conf = get_params_block(self.strategy, 'dynamic_mechanics_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
            
        # [代码修改] 获取缩放指数
        exponent = get_param_value(p_conf.get('final_score_exponent'), 1.0)

        # ... (此部分代码保持不变) ...
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range
        # ... (所有健康度计算的代码保持不变) ...
        # ...
        bullish_short_force = (bullish_health[1] * bullish_health[5])**0.5
        bullish_medium_trend = (bullish_health[13] * bullish_health[21])**0.5
        bullish_long_inertia = bullish_health[55]
        bearish_short_force = (bearish_health[1] * bearish_health[5])**0.5
        bearish_medium_trend = (bearish_health[13] * bearish_health[21])**0.5
        bearish_long_inertia = bearish_health[55]

        # --- [代码修改] 对所有终极信号应用指数缩放 ---
        states['SCORE_DYN_BULLISH_RESONANCE_B'] = (bullish_health[5] ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_A'] = ((bullish_health[5] * bullish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S'] = ((bullish_short_force * bullish_medium_trend) ** exponent).astype(np.float32)
        states['SCORE_DYN_BULLISH_RESONANCE_S_PLUS'] = ((states['SCORE_DYN_BULLISH_RESONANCE_S'] * bullish_long_inertia) ** exponent).astype(np.float32)
        
        states['SCORE_DYN_BEARISH_RESONANCE_B'] = (bearish_health[5] ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_A'] = ((bearish_health[5] * bearish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S'] = ((bearish_short_force * bearish_medium_trend) ** exponent).astype(np.float32)
        states['SCORE_DYN_BEARISH_RESONANCE_S_PLUS'] = ((states['SCORE_DYN_BEARISH_RESONANCE_S'] * bearish_long_inertia) ** exponent).astype(np.float32)
        
        states['SCORE_DYN_BOTTOM_REVERSAL_B'] = ((bottom_context_score * bullish_health[1] * bearish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_A'] = ((bottom_context_score * bullish_health[5] * bearish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S'] = ((bottom_context_score * bullish_short_force * bearish_long_inertia) ** exponent).astype(np.float32)
        states['SCORE_DYN_BOTTOM_REVERSAL_S_PLUS'] = ((bottom_context_score * bullish_short_force * bullish_medium_trend * bearish_long_inertia) ** exponent).astype(np.float32)
        
        states['SCORE_DYN_TOP_REVERSAL_B'] = ((top_context_score * bearish_health[1] * bullish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_A'] = ((top_context_score * bearish_health[5] * bullish_health[21]) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S'] = ((top_context_score * bearish_short_force * bullish_long_inertia) ** exponent).astype(np.float32)
        states['SCORE_DYN_TOP_REVERSAL_S_PLUS'] = ((top_context_score * bearish_short_force * bearish_medium_trend * bullish_long_inertia) ** exponent).astype(np.float32)
        
        return states











