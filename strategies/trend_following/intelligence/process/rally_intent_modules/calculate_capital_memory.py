# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_capital_memory.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

class CalculateCapitalMemory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _normalize_capital_vector(self, capital_series: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V3.0】资金向量归一化方法
        核心理念：将原始资金流数据归一化到[-1, 1]范围，保留方向信息
        数学模型：Robust Z-score归一化，使用滚动窗口统计
        """
        if capital_series.empty:
            return pd.Series(0.0, index=df_index)
        # 使用滚动窗口统计（21日窗口）
        window = 21
        min_periods = int(window * 0.7)  # 70%的最小观测值
        # 计算滚动均值和标准差
        rolling_mean = capital_series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = capital_series.rolling(window=window, min_periods=min_periods).std()
        # 避免除零，设置最小标准差
        min_std = rolling_std.abs().quantile(0.1)  # 取10%分位数作为最小标准差
        if min_std == 0:
            min_std = 1e-9
        rolling_std = rolling_std.clip(lower=min_std)
        # Robust Z-score归一化：z = (x - mean) / std
        z_scores = (capital_series - rolling_mean) / rolling_std
        # 限制极端值（使用tanh函数将Z-score映射到[-1, 1]）
        normalized = np.tanh(z_scores * 0.5)  # 0.5为缩放因子，控制映射敏感度
        # 前向填充NaN值
        normalized = normalized.ffill().fillna(0.0)
        return normalized

    def _calculate_capital_persistence(self, capital_flow: pd.Series, df_index: pd.Index, period: int) -> pd.Series:
        """
        【V3.0】资金持续性检测（Hurst指数简化版）
        数学模型：重标极差分析（R/S）简化版本
        """
        persistence_scores = pd.Series(0.5, index=df_index)
        for i in range(period, len(df_index)):
            if i < period:
                continue
            # 取最近period日数据
            segment = capital_flow.iloc[i-period:i]
            if len(segment) < 10:
                persistence_scores.iloc[i] = 0.5
                continue
            # 计算均值
            mean_val = segment.mean()
            # 计算累积离差
            deviations = segment - mean_val
            cumulative_dev = deviations.cumsum()
            # 计算极差
            R = cumulative_dev.max() - cumulative_dev.min()
            # 计算标准差
            S = segment.std()
            # 避免除零
            if S == 0:
                persistence_scores.iloc[i] = 0.5
                continue
            # R/S比率
            rs_ratio = R / S
            # 简化Hurst指数估计
            # log(R/S) ≈ H * log(n)
            n = len(segment)
            if rs_ratio > 0 and n > 1:
                H = np.log(rs_ratio) / np.log(n)
            else:
                H = 0.5
            # H值映射到[0, 1]
            # H>0.5: 持续性，H<0.5: 反持续性，H=0.5: 随机
            persistence_score = (H - 0.3) / 0.4  # 映射到[0,1]，0.3-0.7范围
            persistence_scores.iloc[i] = persistence_score.clip(0, 1)
        return persistence_scores

    def _detect_capital_anomaly(self, capital_flow: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】资金异常检测方法
        核心理念：使用统计方法检测资金流的异常波动
        数学模型：IQR异常检测 + 历史分位数比较
        """
        anomaly_scores = pd.Series(0.0, index=df_index)
        # 使用滚动窗口检测异常
        window = min(60, memory_period * 3)  # 足够长的窗口以获得稳定统计
        for i in range(window, len(df_index)):
            if i < window:
                anomaly_scores.iloc[i] = 0.0
                continue
            # 获取历史窗口数据
            historical_data = capital_flow.iloc[i-window:i]
            # 1. IQR方法检测异常（适用于非正态分布）
            Q1 = historical_data.quantile(0.25)
            Q3 = historical_data.quantile(0.75)
            IQR = Q3 - Q1
            # 避免IQR过小
            if IQR < 1e-9:
                IQR = historical_data.std()
                if IQR < 1e-9:
                    IQR = 1e-9
            # 计算当前值与历史分布的偏差
            current_value = capital_flow.iloc[i]
            # 下界和上界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # 2. 计算Z-score异常
            historical_mean = historical_data.mean()
            historical_std = historical_data.std()
            if historical_std < 1e-9:
                historical_std = 1e-9
            z_score = abs((current_value - historical_mean) / historical_std)
            # 3. 计算历史分位数异常
            percentile_rank = (historical_data <= current_value).sum() / len(historical_data)
            # 距离中位数的偏差（0.5为中间）
            quantile_deviation = abs(percentile_rank - 0.5) * 2  # 映射到[0, 1]
            # 4. 综合异常评分
            # 如果超出IQR边界，则异常程度较高
            is_outlier = (current_value < lower_bound) or (current_value > upper_bound)
            # 异常评分组合
            anomaly_score = (
                (1.0 if is_outlier else 0.0) * 0.4 +  # IQR边界异常权重
                min(z_score / 3.0, 1.0) * 0.4 +       # Z-score异常（3σ对应1.0）
                quantile_deviation * 0.2              # 分位数异常
            ).clip(0, 1)
            anomaly_scores.iloc[i] = anomaly_score
        # 前向填充初始值
        anomaly_scores = anomaly_scores.ffill().fillna(0.0)
        return anomaly_scores

    def _calculate_capital_efficiency(self, capital_flow: pd.Series, price_change: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.0】资金效率计算方法
        核心理念：评估单位资金推动价格上涨的效率
        数学模型：资金-价格弹性系数 + 领先滞后效率
        """
        efficiency_scores = pd.Series(0.5, index=df_index)  # 默认中等效率
        # 确保数据对齐
        if len(capital_flow) != len(price_change):
            min_len = min(len(capital_flow), len(price_change))
            capital_flow = capital_flow.iloc[:min_len]
            price_change = price_change.iloc[:min_len]
        # 窗口大小
        window = min(20, memory_period)
        for i in range(window, len(df_index)):
            if i < window:
                efficiency_scores.iloc[i] = 0.5
                continue
            # 1. 计算资金-价格弹性（简单线性回归斜率）
            capital_window = capital_flow.iloc[i-window:i].values
            price_window = price_change.iloc[i-window:i].values
            # 避免零变化
            if np.std(capital_window) < 1e-9 or np.std(price_window) < 1e-9:
                efficiency_scores.iloc[i] = 0.5
                continue
            # 计算相关系数
            correlation = np.corrcoef(capital_window, price_window)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            # 2. 计算单位资金推动的价格变化（弹性系数）
            # 标准化后计算斜率
            capital_normalized = (capital_window - np.mean(capital_window)) / (np.std(capital_window) + 1e-9)
            price_normalized = (price_window - np.mean(price_window)) / (np.std(price_window) + 1e-9)
            # 简单线性回归：斜率 = cov(x,y) / var(x)
            covariance = np.cov(capital_normalized, price_normalized)[0, 1]
            variance = np.var(capital_normalized)
            if variance < 1e-9:
                slope = 0
            else:
                slope = covariance / variance
            # 3. 计算领先滞后效率（资金是否领先于价格）
            # 使用交叉相关找到最大相关性的滞后
            max_corr = 0
            best_lag = 0
            for lag in range(-3, 4):  # ±3天滞后
                if lag <= 0:
                    # 资金领先（负滞后）
                    capital_shifted = capital_window[:len(capital_window)+lag] if lag < 0 else capital_window
                    price_shifted = price_window[-lag:] if lag < 0 else price_window
                else:
                    # 价格领先（正滞后）
                    capital_shifted = capital_window[lag:]
                    price_shifted = price_window[:len(price_window)-lag]
                # 确保长度一致
                min_len = min(len(capital_shifted), len(price_shifted))
                if min_len < 5:
                    continue
                capital_shifted = capital_shifted[:min_len]
                price_shifted = price_shifted[:min_len]
                corr = np.corrcoef(capital_shifted, price_shifted)[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag
            # 资金领先（负滞后）为高效，价格领先（正滞后）为低效
            lag_efficiency = max(0, -best_lag) / 3.0  # 映射到[0, 1]，资金领先最多3天
            # 4. 综合效率评分
            # 相关系数（方向一致性）
            correlation_score = (correlation + 1) / 2  # 映射到[0, 1]
            # 斜率（单位资金推动力）
            slope_score = np.tanh(slope * 0.5) * 0.5 + 0.5  # 映射到[0, 1]
            # 综合效率
            efficiency_score = (
                correlation_score * 0.4 +      # 方向一致性
                slope_score * 0.3 +            # 推动力度
                lag_efficiency * 0.3           # 领先滞后
            ).clip(0, 1)
            efficiency_scores.iloc[i] = efficiency_score
        # 前向填充
        efficiency_scores = efficiency_scores.ffill().fillna(0.5)
        return efficiency_scores


























