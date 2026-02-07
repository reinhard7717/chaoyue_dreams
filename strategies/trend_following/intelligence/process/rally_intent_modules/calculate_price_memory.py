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
        【V3.5 · 筹码力矩博弈版】支撑阻力记忆计算
        逻辑：通过成交量加权的密度分布（VAP）识别博弈平台，量化“解套抛压”与“获利支撑”的动态对冲关系。
        A股特性：深度整合了整数关口心理博弈与记忆衰减效应。
        """
        close = raw_signals.get('close', pd.Series(0.0, index=df_index))
        high = raw_signals.get('high', pd.Series(0.0, index=df_index))
        low = raw_signals.get('low', pd.Series(0.0, index=df_index))
        volume = raw_signals.get('volume', pd.Series(0.0, index=df_index))
        sr_score = pd.Series(0.5, index=df_index)
        rolling_max = high.rolling(window=memory_period).max()
        rolling_min = low.rolling(window=memory_period).min()
        for i in range(memory_period, len(df_index)):
            curr_close = close.iloc[i]
            w_high = high.iloc[i-memory_period:i].values
            w_low = low.iloc[i-memory_period:i].values
            w_vol = volume.iloc[i-memory_period:i].values
            # 1. 构建记忆衰减权重 (线性模拟记忆遗忘，近期权重设为1.0)
            decay = np.linspace(0.5, 1.0, memory_period)
            # 2. 定位价格中轴
            price_pivots = (w_high + w_low) / 2
            # 3. 计算双向博弈力矩
            # 获利盘支撑力矩 (当前价位下方)
            support_mask = price_pivots < curr_close
            s_torque = np.sum(w_vol[support_mask] * decay[support_mask])
            # 套牢盘压力力矩 (当前价位上方)
            resistance_mask = ~support_mask
            r_torque = np.sum(w_vol[resistance_mask] * decay[resistance_mask])
            # 4. 计算力矩比率分值
            total_torque = s_torque + r_torque + 1e-9
            density_score = s_torque / total_torque
            # 5. 相对区间位置修正
            p_range = rolling_max.iloc[i] - rolling_min.iloc[i] + 1e-9
            relative_pos = (curr_close - rolling_min.iloc[i]) / p_range
            # 6. A股整数心理关口建模
            psych_mod = 1.0
            # 针对关键整数位进行敏感度探测
            for level in [10, 20, 50, 100, 200, 500]:
                if abs(curr_close - level) / level < 0.008:
                    # 进攻状态(阳线)遇关口产生阻力，回踩状态(阴线)遇关口产生支撑
                    if curr_close > close.iloc[i-1]:
                        psych_mod = 0.94 # 阻力抑制
                    else:
                        psych_mod = 1.06 # 支撑增强
                    break
            # 7. 综合多维度博弈得分
            # 权重分配：65%筹码密度，35%区间位置
            combined_val = (density_score * 0.65 + relative_pos * 0.35) * psych_mod
            sr_score.iloc[i] = combined_val
        return sr_score.ffill().fillna(0.5).clip(0, 1)









































