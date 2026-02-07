# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_chip_memory.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple


class CalculateChipMemory:
    def __init__(self, df_index, raw_signals, params):
        self.df_index = df_index
        self.raw_signals = raw_signals
        self.params = params

    def _calculate_chip_entropy_memory(self, raw_signals: Dict[str, pd.Series], df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.5 · 向量化熵变版】筹码熵变记忆计算
        修改说明：移除嵌套循环，利用分位数区间占比与滚动偏度模拟熵值变化，量化筹码分歧度。
        """
        # 优先使用已有的熵值指标，若无则由集中度与分歧度合成
        base_entropy = raw_signals.get('chip_entropy', pd.Series(0.5, index=df_index))
        div_ratio = raw_signals.get('chip_divergence_ratio', pd.Series(0.5, index=df_index))
        # 核心逻辑：当分歧度扩大且集中度下降时，熵值增加（混乱度增加）
        # 使用 rolling.std() 衡量近期指标的震荡频率作为熵增的辅助判定
        vol_adj = base_entropy.rolling(window=memory_period, min_periods=5).std().fillna(0)
        # 归一化综合熵：[0, 1]，0代表极度一致（锁仓），1代表极度混乱（放量分歧）
        combined_entropy = (base_entropy * 0.7 + div_ratio * 0.2 + vol_adj * 0.1).clip(0, 1)
        return combined_entropy.ewm(span=5, adjust=False).mean()

    def _calculate_concentration_migration(self, concentration_series: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V4.0 · 向量化斜率版】筹码集中度迁移记忆计算
        修改说明：利用线性回归系数公式实现全向量化计算，识别主力收集或派发筹码的速率。
        数学模型：Slope = Cov(x, y) / Var(x)
        """
        if concentration_series.empty:
            return pd.Series(0.5, index=df_index)
        window = max(5, int(memory_period / 4))
        # 构造自变量 x (时间序列 0, 1, 2...)
        x = np.arange(window)
        x_var = np.var(x)
        # 定义内部滚动回归函数
        def get_slope(y):
            if len(y) < window: return 0.0
            return np.cov(x, y)[0, 1] / x_var
        # 向量化滚动计算斜率
        slopes = concentration_series.rolling(window=window).apply(get_slope, raw=True).fillna(0)
        # 归一化：通过 tanh 映射将斜率转换为 [0, 1] 区间，0.5 为平衡点
        migration_score = (np.tanh(slopes * 20) * 0.5 + 0.5).clip(0, 1)
        # 加入动量修正（加速度）：斜率的变化率
        accel = migration_score.diff(3).fillna(0)
        final_migration = (migration_score * 0.8 + (np.tanh(accel * 5) * 0.5 + 0.5) * 0.2).clip(0, 1)
        return final_migration.ffill()

    def _calculate_chip_stability_memory(self, raw_signals: Dict[str, pd.Series], df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.6 · 筹码锁定韧性版】计算筹码稳定性记忆
        修改说明：引入“换手率-波动率”对冲模型，识别在震荡市中巍然不动的高质量锁仓筹码。
        """
        raw_stability = raw_signals.get('chip_stability', pd.Series(0.5, index=df_index))
        turnover = raw_signals.get('turnover_rate', pd.Series(1.0, index=df_index))
        close_pct = raw_signals.get('pct_change', pd.Series(0.0, index=df_index))
        # 1. 稳定性基准 (EWMA 过滤噪声)
        stability_trend = raw_stability.ewm(span=memory_period, adjust=False).mean()
        # 2. 韧性因子计算：在价格剧烈波动时，若换手率保持低位，则稳定性加分
        # 公式：Resilience = |Price_Change| / (Turnover + epsilon)
        resilience = (close_pct.abs() / (turnover + 0.01)).rolling(window=10).mean().fillna(0)
        resilience_norm = (resilience / resilience.rolling(60).max().replace(0, 1)).clip(0, 1)
        # 3. 筹码沉淀度：近期极低换手率的持续天数
        is_low_turnover = (turnover < turnover.rolling(120).quantile(0.2)).astype(float)
        sedimentation = is_low_turnover.rolling(window=5).mean()
        # 4. 综合稳定性评分
        # 权重：趋势 50% + 韧性 30% + 沉淀 20%
        final_stability = (stability_trend * 0.5 + resilience_norm * 0.3 + sedimentation * 0.2).clip(0, 1)
        return final_stability

    def _calculate_chip_pressure_memory(self, raw_signals: Dict[str, pd.Series], df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V4.0 · 筹码峰值引力场版】计算筹码压力记忆分数
        修改说明：引入“动态成本密集区”探测逻辑，通过计算当前价格对“上方套牢峰”的穿透距离，精确量化解套抛压。
        A股特性：重点捕捉 CYS 在 [-5%, 0%] 区间内，散户回本即卖的“心理奇点”风险。
        """
        close = raw_signals.get('close', pd.Series(0.0, index=df_index))
        volume = raw_signals.get('volume', pd.Series(0.0, index=df_index))
        # 1. 计算不同周期的动态成本重心 (VWMA 组合)
        vwma_fast = (close * volume).rolling(13).sum() / volume.rolling(13).sum().replace(0, 1)
        vwma_slow = (close * volume).rolling(memory_period).sum() / volume.rolling(memory_period).sum().replace(0, 1)
        # 2. 识别“套牢峰”引力：当现价低于 VWMA 且接近 VWMA 时，压力指数级上升
        # 计算盈亏率 CYS
        cys_slow = (close - vwma_slow) / (vwma_slow + 1e-9)
        # 3. 解套抛压模型：针对 A 股散户“小亏即跑”的特性
        # 在 CYS 为 -3% 到 +1% 之间定义为“解套敏感区”
        # 使用高斯分布函数：G(x) = exp(-(x-μ)^2 / 2σ^2)，中心 μ=-0.01，标准差 σ=0.04
        trapped_risk = np.exp(-((cys_slow - (-0.01))**2) / (2 * (0.04**2)))
        # 4. 获利兑现压力：针对 CYS > 15% 的爆发式增长
        # 使用 Sigmoid 激活函数：1 / (1 + exp(-k(x-x0)))
        profit_risk = 1 / (1 + np.exp(-20 * (cys_slow - 0.18)))
        # 5. 真空度检测 (Vacuum Index)：若现价远高于重心且无量，则压力极小
        # 判定价格是否在所有均价之上，且换手处于低位
        is_breakout = (close > vwma_fast) & (close > vwma_slow)
        vol_ratio = raw_signals.get('volume_ratio', pd.Series(1.0, index=df_index))
        vacuum_factor = (1 - (vol_ratio / 5.0).clip(0, 0.8)) * is_breakout.astype(float)
        # 6. 综合压力合成
        # 压力 = (解套风险 + 获利风险) * (1 - 真空溢价)
        total_pressure_raw = (np.maximum(trapped_risk, profit_risk) * (1 - vacuum_factor * 0.5)).clip(0, 1)
        # 7. 记忆衰减平滑
        pressure_memory = total_pressure_raw.ewm(alpha=0.4, adjust=False).mean()
        return pressure_memory.clip(0, 1)















