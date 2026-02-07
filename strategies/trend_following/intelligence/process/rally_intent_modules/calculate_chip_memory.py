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
        【V3.0】筹码熵变记忆计算
        核心理念：使用信息熵衡量筹码分布的混乱程度
        数学模型：信息熵计算 + 熵变趋势
        """
        entropy_scores = pd.Series(0.5, index=df_index)
        # 获取筹码相关信号
        chip_signals = [
            ('chip_concentration_ratio', 0.4),
            ('chip_convergence_ratio', 0.3),
            ('chip_divergence_ratio', 0.3)
        ]
        for i in range(memory_period, len(df_index)):
            if i < memory_period:
                entropy_scores.iloc[i] = 0.5
                continue
            entropy_components = []
            for signal_name, weight in chip_signals:
                if signal_name not in raw_signals:
                    continue
                signal_series = raw_signals[signal_name]
                # 获取窗口数据
                window_data = signal_series.iloc[i-memory_period:i]
                if len(window_data) < 5:
                    continue
                # 离散化：将数据分成5个区间
                try:
                    # 使用分位数进行离散化
                    bins = np.quantile(window_data, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    # 确保边界值唯一
                    bins = np.unique(bins)
                    if len(bins) < 2:
                        continue
                    # 计算当前值所在区间的概率分布
                    hist, _ = np.histogram(window_data, bins=bins)
                    # 转换为概率
                    prob = hist / len(window_data)
                    prob = prob[prob > 0]  # 只保留正概率
                    # 计算信息熵：H = -sum(p * log2(p))
                    if len(prob) > 0:
                        entropy = -np.sum(prob * np.log2(prob))
                        # 归一化：最大熵为log2(n)
                        max_entropy = np.log2(len(prob))
                        if max_entropy > 0:
                            normalized_entropy = entropy / max_entropy
                            entropy_components.append(normalized_entropy * weight)
                except:
                    continue
            if entropy_components:
                # 加权平均熵
                total_weight = sum(weight for _, weight in chip_signals if _ in raw_signals)
                if total_weight > 0:
                    entropy_score = sum(entropy_components) / total_weight
                    entropy_scores.iloc[i] = entropy_score
                else:
                    entropy_scores.iloc[i] = 0.5
        entropy_scores = entropy_scores.ffill().fillna(0.5)
        return entropy_scores

    def _calculate_concentration_migration(self, concentration_series: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】筹码集中度迁移记忆计算
        核心理念：跟踪筹码集中度的变化方向和速度
        数学模型：趋势斜率 + 动量分析
        """
        if concentration_series.empty:
            return pd.Series(0.5, index=df_index)
        migration_scores = pd.Series(0.5, index=df_index)
        # 使用滚动窗口计算趋势
        window = min(13, memory_period // 2)
        for i in range(window, len(df_index)):
            if i < window:
                migration_scores.iloc[i] = 0.5
                continue
            # 获取窗口数据
            window_data = concentration_series.iloc[i-window:i]
            if len(window_data) < 3:
                migration_scores.iloc[i] = 0.5
                continue
            # 1. 计算线性趋势斜率
            x = np.arange(len(window_data))
            y = window_data.values
            # 线性回归
            A = np.vstack([x, np.ones(len(x))]).T
            try:
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            except:
                slope = 0
            # 2. 计算动量（近期变化）
            recent_change = window_data.iloc[-1] - window_data.iloc[0]
            momentum = recent_change / (window_data.max() - window_data.min() + 1e-9)
            # 3. 计算加速度（斜率变化）
            if i >= window * 2:
                prev_window = concentration_series.iloc[i-window*2:i-window]
                if len(prev_window) >= 3:
                    x_prev = np.arange(len(prev_window))
                    y_prev = prev_window.values
                    A_prev = np.vstack([x_prev, np.ones(len(x_prev))]).T
                    try:
                        slope_prev, _ = np.linalg.lstsq(A_prev, y_prev, rcond=None)[0]
                        acceleration = slope - slope_prev
                    except:
                        acceleration = 0
                else:
                    acceleration = 0
            else:
                acceleration = 0
            # 4. 综合迁移评分（集中度上升为正，下降为负）
            # 归一化处理
            slope_score = np.tanh(slope * 10) * 0.5 + 0.5  # 映射到[0, 1]
            momentum_score = momentum * 0.5 + 0.5  # 映射到[0, 1]
            accel_score = np.tanh(acceleration * 5) * 0.5 + 0.5  # 映射到[0, 1]
            migration_score = (
                slope_score * 0.5 +
                momentum_score * 0.3 +
                accel_score * 0.2
            )
            migration_scores.iloc[i] = migration_score.clip(0, 1)
        migration_scores = migration_scores.ffill().fillna(0.5)
        return migration_scores

    def _calculate_chip_stability_memory(self, raw_signals: Dict[str, pd.Series], df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.3 · 锁仓惯性记忆版】计算筹码稳定性记忆
        核心理念：通过'低波动持续性'与'缩量惜售特征'，识别主力锁仓行为的真实度。
        A股特性：真正的筹码稳定表现为指标本身的低方差(不乱动)以及低换手(没人卖)。
        """
        # 1. 获取基础信号
        # chip_stability 原始值通常在 [0, 1] 之间，越高越稳定
        raw_stability = raw_signals.get('chip_stability', pd.Series(0.5, index=df_index))
        turnover = raw_signals.get('turnover_rate', pd.Series(0.0, index=df_index))
        # 2. 计算稳定性的趋势惯性 (Trend Inertia)
        # 使用EWMA提取长期记忆，衰减因子根据周期调整
        span_val = max(5, int(memory_period / 2))
        stability_trend = raw_stability.ewm(span=span_val, adjust=False).mean()
        # 3. 计算稳定性的波动惩罚 (Variance Penalty)
        # 逻辑：筹码稳定性指标本身的波动率越低，说明结构越稳固（主力控盘）
        # 如果稳定性指标上蹿下跳，说明筹码在剧烈交换，记忆不可靠
        stability_vol = raw_stability.rolling(window=span_val).std().fillna(0)
        # 归一化波动率，假设 0.2 以上为高波动
        vol_penalty = (stability_vol / 0.2).clip(0, 1)
        consistency_score = 1.0 - vol_penalty
        # 4. 计算缩量确认因子 (Volume Confirmation)
        # 逻辑：低换手率是筹码稳定的最佳背书（惜售）
        # 对换手率进行动态归一化 (Rolling Rank)
        rolling_to_max = turnover.rolling(window=memory_period*2, min_periods=5).max()
        turnover_ratio = (turnover / rolling_to_max.replace(0, 1)).fillna(1)
        # 换手越低，确认度越高；换手极高时，稳定性记忆打折
        volume_confirmation = 1.0 - turnover_ratio.clip(0, 0.8) # 保留0.2的基础分
        # 5. 综合筹码稳定性记忆
        # 权重分配：趋势惯性50% + 自身稳定性30% + 缩量确认20%
        final_memory = (
            stability_trend * 0.5 +
            consistency_score * 0.3 +
            volume_confirmation * 0.2
        )
        return final_memory.clip(0, 1)

    def _calculate_chip_pressure_memory(self, raw_signals: Dict[str, pd.Series], df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.4 · 双向筹码压力场版】计算筹码压力记忆分数
        核心理念：量化"解套抛压"（Trapped Pressure）与"获利兑现"（Profit Taking）的双重风险。
        A股特性：
        1. 价格从下方接近成本区时，解套盘抛压最大（即将回本时最想卖）。
        2. 价格远超成本区时，获利盘兑现欲望增强（恐高）。
        返回：压力分数 [0, 1]，1表示压力极大（顶部或强阻力位），0表示压力极小（真空区或底部）。
        """
        # 1. 准备数据
        close = raw_signals.get('close', pd.Series(0.0, index=df_index))
        volume = raw_signals.get('volume', pd.Series(0.0, index=df_index))
        # 2. 计算市场平均成本 (VWMA - Volume Weighted Moving Average)
        # 使用 memory_period 作为周期（如55日）
        pv = close * volume
        # 动态计算滚动VWMA
        rolling_pv = pv.rolling(window=memory_period, min_periods=int(memory_period/2)).sum()
        rolling_vol = volume.rolling(window=memory_period, min_periods=int(memory_period/2)).sum()
        vwma = (rolling_pv / rolling_vol.replace(0, np.nan)).ffill()
        # 3. 计算市场盈亏率 CYS (Cost Yield Simple)
        # CYS > 0 代表获利，CYS < 0 代表套牢
        cys = (close - vwma) / (vwma + 1e-9)
        # 4. 计算解套抛压 (Trapped Pressure) - 针对 CYS < 0 的部分
        # 逻辑：当 CYS 在 -0.15 到 0 之间时（亏损15%以内），抛压随价格上涨指数级增加
        # 使用高斯函数模拟：峰值设在 -0.02 (亏损2%时抛压最大，人性使然)，标准差设为 0.08
        # 当深套（如 -30%）时，抛压反而减小（装死不动）
        trapped_pressure = np.exp(-((cys - (-0.02)) ** 2) / (2 * (0.08 ** 2)))
        # 仅保留 CYS < 0.05 的部分作为解套压力（允许小幅获利出逃）
        trapped_pressure = trapped_pressure.where(cys < 0.05, 0.0)
        # 5. 计算获利兑现压力 (Profit Pressure) - 针对 CYS > 0 的部分
        # 逻辑：当获利超过 20% 后，抛压显著增加；超过 40% 极度危险
        # 使用 Sigmoid 函数：中心点设在 0.25 (25%获利)，斜率 k=15
        profit_pressure = 1 / (1 + np.exp(-15 * (cys - 0.25)))
        # 仅保留 CYS >= 0 的部分
        profit_pressure = profit_pressure.where(cys >= 0, 0.0)
        # 6. 综合压力记忆
        # 取两者的最大值，因为同一时间通常只有一种主导压力
        total_pressure = np.maximum(trapped_pressure, profit_pressure)
        # 7. 平滑处理
        # 压力具有记忆性，不会瞬间消失
        pressure_memory = total_pressure.ewm(span=5, adjust=False).mean()
        return pressure_memory.clip(0, 1)
















