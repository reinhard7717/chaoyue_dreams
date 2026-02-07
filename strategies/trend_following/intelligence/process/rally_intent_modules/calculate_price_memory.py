# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_price_memory.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

class CalculatePriceMemory:
    def __init__(self, df_index: pd.Index, price: pd.Series, config: Dict) -> None:
        self.df_index = df_index
        self.price = price
        self.config = config

    def calculate_adaptive_momentum_memory(self, close_series: pd.Series, df_index: pd.Index, base_period: int) -> pd.Series:
        """
        【V3.1 · 紧急修复版】自适应动量记忆计算
        修复说明：解决Pandas EWM不支持动态span导致的ValueError crash问题。
        改为使用固定周期计算基础RSI，并在结果层应用波动率调节。
        """
        # 1. 计算波动率因子 (保留原逻辑用于后处理)
        returns = close_series.pct_change().fillna(0)
        # 使用固定窗口计算波动率
        volatility = returns.rolling(window=20).std().fillna(0.01)
        # 2. 计算RSI (使用固定base_period，避免Series ambiguous错误)
        # 确保base_period为有效整数
        safe_period = max(2, int(base_period))
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        # 使用固定周期的指数加权移动平均
        avg_gain = gains.ewm(span=safe_period, adjust=False).mean()
        avg_loss = losses.ewm(span=safe_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        # 归一化到[0, 1]
        rsi_norm = (rsi / 100).clip(0, 1)
        # 3. 动量加速度（二阶差分）
        momentum_accel = rsi_norm.diff().diff().fillna(0)
        # 4. 应用自适应调节 (替代原有的动态周期)
        # 逻辑：高波动率时(volatility高)，信号置信度略降；低波动率时置信度高
        # 构建调节因子：波动率越低，因子越接近1.1；波动率越高，因子越接近0.9
        adaptive_modulator = 1.0 + (0.02 - volatility).clip(-0.05, 0.05)
        # 综合动量记忆
        momentum_memory = (rsi_norm * 0.7 + (0.5 + momentum_accel * 0.5) * 0.3)
        momentum_memory = (momentum_memory * adaptive_modulator).clip(0, 1)
        return momentum_memory

    def calculate_volatility_memory(self, returns_series: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】波动率记忆计算（GARCH简化版）
        数学模型：EWMA波动率 + 波动率聚集效应
        """
        # EWMA波动率（RiskMetrics方法）
        lambda_factor = 0.94
        squared_returns = returns_series ** 2
        # 初始化
        vol_memory = pd.Series(0.0, index=df_index)
        if len(vol_memory) > 0:
            vol_memory.iloc[0] = squared_returns.iloc[:min(30, len(squared_returns))].mean()
        # 递归计算
        for i in range(1, len(vol_memory)):
            vol_memory.iloc[i] = (lambda_factor * vol_memory.iloc[i-1] + 
                                 (1 - lambda_factor) * squared_returns.iloc[i-1])
        # 年化波动率并归一化（假设年化波动率在10%-50%之间）
        annualized_vol = np.sqrt(vol_memory * 252)
        vol_norm = ((annualized_vol - 0.1) / 0.4).clip(0, 1)
        return vol_norm

    def calculate_support_resistance_memory(self, raw_signals: Dict[str, pd.Series], df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.2 · 筹码结构记忆版】计算支撑阻力记忆分数
        核心理念：结合'市场平均持仓成本(Rolling VWAP)'与'箱体结构位置'，量化价格的支撑/阻力状态。
        A股特性：价格位于筹码密集区上方为强支撑(Score>0.5)，下方为强套牢阻力(Score<0.5)。
        """
        # 1. 获取基础数据
        close = raw_signals.get('close', pd.Series(0.0, index=df_index))
        high = raw_signals.get('high', pd.Series(0.0, index=df_index))
        low = raw_signals.get('low', pd.Series(0.0, index=df_index))
        volume = raw_signals.get('volume', pd.Series(0.0, index=df_index))
        # 2. 计算周期内的滚动成本均价 (Rolling VWAP)
        # 公式: sum(price * vol) / sum(vol) over memory_period
        pv = close * volume
        rolling_pv = pv.rolling(window=memory_period, min_periods=int(memory_period/2)).sum()
        rolling_vol = volume.rolling(window=memory_period, min_periods=int(memory_period/2)).sum()
        # 避免除零，使用ffill填充空值
        rolling_vwap = (rolling_pv / rolling_vol.replace(0, np.nan)).ffill()
        # 3. 计算成本偏离度得分 (Cost Deviation Score)
        # 逻辑：价格 > VWAP -> 获利盘主导 -> 支撑强 -> 分数高
        # 使用tanh将偏离率映射到 [0, 1] 区间
        deviation = (close - rolling_vwap) / (rolling_vwap + 1e-9)
        # 系数10用于放大微小的偏离，使信号更敏感
        cost_score = np.tanh(deviation * 10) * 0.5 + 0.5
        # 4. 计算结构位置得分 (Structural Position Score)
        # 逻辑：接近周期高点为强势(接近阻力突破)，接近低点为弱势
        rolling_high = high.rolling(window=memory_period, min_periods=int(memory_period/2)).max()
        rolling_low = low.rolling(window=memory_period, min_periods=int(memory_period/2)).min()
        range_span = rolling_high - rolling_low
        # 避免除零
        position_score = (close - rolling_low) / range_span.replace(0, 1.0)
        # 5. 综合支撑阻力记忆
        # 权重分配：成本记忆(筹码)占60%，结构记忆(形态)占40%
        # 结果说明：接近1表示上方无阻力且获利盘多(强支撑记忆)，接近0表示深套且处于低位(强阻力记忆)
        support_resistance_memory = (cost_score * 0.6 + position_score.clip(0, 1) * 0.4)
        return support_resistance_memory.ffill().fillna(0.5)










































