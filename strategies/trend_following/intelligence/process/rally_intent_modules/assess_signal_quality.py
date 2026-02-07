# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\assess_signal_quality.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

class SignalQualityAssessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _calculate_individual_signal_quality(self, signal_series: pd.Series, signal_name: str) -> pd.Series:
        """
        【V4.0】计算单个信号的个体质量指标
        评估维度：
        1. 稳定性（波动率倒数）
        2. 信噪比（趋势强度/噪声强度）
        3. 极端值频率（异常值比例）
        4. 数据完整性（缺失值比例）
        """
        if signal_series.empty:
            return pd.Series(0.5, index=signal_series.index)
        quality_scores = pd.Series(0.5, index=signal_series.index)
        for i in range(len(signal_series)):
            if i < 20:  # 需要足够的数据来计算质量
                quality_scores.iloc[i] = 0.5
                continue
            # 1. 稳定性评分（基于滚动窗口波动率）
            window_data = signal_series.iloc[max(0, i-20):i]
            if len(window_data) >= 10:
                volatility = window_data.std()
                if volatility > 0:
                    # 波动率越低，稳定性越高（倒U型，适中最好）
                    # 对于大多数代理信号，0.1-0.3的波动率是理想的
                    optimal_volatility = 0.2
                    stability_score = 1.0 - min(abs(volatility - optimal_volatility) / optimal_volatility, 1.0)
                else:
                    stability_score = 0.5
            else:
                stability_score = 0.5
            # 2. 信噪比评分（趋势强度 vs 噪声强度）
            if len(window_data) >= 15:
                # 计算趋势（线性回归斜率）
                x = np.arange(len(window_data))
                y = window_data.values
                try:
                    slope, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
                    # 趋势强度的绝对值
                    trend_strength = abs(slope) * 100  # 放大到合理范围
                    # 噪声强度（去趋势后的残差标准差）
                    residuals = y - (slope * x + _)
                    noise_strength = np.std(residuals)
                    if noise_strength > 0:
                        snr = trend_strength / noise_strength
                        # SNR在1-3之间为理想范围
                        snr_score = min(snr / 3.0, 1.0)
                    else:
                        snr_score = 1.0 if trend_strength > 0 else 0.5
                except:
                    snr_score = 0.5
            else:
                snr_score = 0.5
            # 3. 极端值频率评分
            if len(window_data) >= 20:
                # 使用IQR方法检测极端值
                Q1 = window_data.quantile(0.25)
                Q3 = window_data.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = window_data[(window_data < lower_bound) | (window_data > upper_bound)]
                    outlier_ratio = len(outliers) / len(window_data)
                    # 极端值越少越好
                    extreme_score = 1.0 - outlier_ratio
                else:
                    extreme_score = 1.0
            else:
                extreme_score = 0.5
            # 4. 数据完整性评分
            # 计算窗口内缺失值比例
            if len(window_data) > 0:
                missing_ratio = window_data.isna().sum() / len(window_data)
                completeness_score = 1.0 - missing_ratio
            else:
                completeness_score = 1.0
            # 综合个体质量评分
            # 权重分配：稳定性30%，信噪比30%，极端值20%，完整性20%
            individual_quality = (
                stability_score * 0.3 +
                snr_score * 0.3 +
                extreme_score * 0.2 +
                completeness_score * 0.2
            )
            quality_scores.iloc[i] = individual_quality
        # 前向填充并平滑
        quality_scores = quality_scores.ffill().fillna(0.5)
        quality_scores = quality_scores.rolling(window=5, min_periods=3).mean().fillna(0.5)
        return quality_scores.clip(0, 1)

    def _calculate_signal_consistency_quality(self, all_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V4.0】计算信号间的一致性质量
        评估维度：
        1. 方向一致性（各信号方向相同的比例）
        2. 幅度一致性（变化幅度相关性）
        3. 时序一致性（领先滞后关系的稳定性）
        """
        # 获取第一个信号的索引作为基准
        first_signal = next(iter(all_signals.values()))
        if first_signal.empty:
            return pd.Series(0.5, index=first_signal.index)
        consistency_scores = pd.Series(0.5, index=first_signal.index)
        for i in range(len(consistency_scores)):
            if i < 30:  # 需要足够的数据
                consistency_scores.iloc[i] = 0.5
                continue
            # 获取最近30天的数据窗口
            window_start = max(0, i-30)
            window_end = i
            # 收集各信号在窗口内的数据
            window_signals = {}
            for signal_name, signal_series in all_signals.items():
                if len(signal_series) > window_end:
                    window_data = signal_series.iloc[window_start:window_end]
                    if len(window_data) >= 10:
                        window_signals[signal_name] = window_data
            if len(window_signals) < 2:
                consistency_scores.iloc[i] = 0.5
                continue
            # 1. 方向一致性评分
            direction_scores = []
            for day in range(max(0, window_end-5), window_end):  # 最近5天
                if day >= len(first_signal):
                    continue
                directions = []
                for signal_name, signal_series in window_signals.items():
                    if day < len(signal_series):
                        # 计算方向（与前一天比较）
                        if day > 0 and (day-1) < len(signal_series):
                            change = signal_series.iloc[day] - signal_series.iloc[day-1]
                            direction = 1 if change > 0 else (-1 if change < 0 else 0)
                            directions.append(direction)
                if len(directions) >= 2:
                    # 计算方向一致比例
                    positive_count = sum(1 for d in directions if d > 0)
                    negative_count = sum(1 for d in directions if d < 0)
                    max_uniform = max(positive_count, negative_count)
                    direction_consistency = max_uniform / len(directions) if len(directions) > 0 else 0
                    direction_scores.append(direction_consistency)
            direction_avg = np.mean(direction_scores) if direction_scores else 0.5
            # 2. 幅度一致性评分（相关系数）
            correlation_scores = []
            signal_names = list(window_signals.keys())
            for j in range(len(signal_names)):
                for k in range(j+1, len(signal_names)):
                    signal1 = window_signals[signal_names[j]]
                    signal2 = window_signals[signal_names[k]]
                    if len(signal1) >= 10 and len(signal2) >= 10:
                        # 确保长度一致
                        min_len = min(len(signal1), len(signal2))
                        signal1_trimmed = signal1.iloc[:min_len]
                        signal2_trimmed = signal2.iloc[:min_len]
                        # 计算相关系数
                        corr = signal1_trimmed.corr(signal2_trimmed)
                        if not np.isnan(corr):
                            correlation_scores.append(abs(corr))  # 取绝对值
            correlation_avg = np.mean(correlation_scores) if correlation_scores else 0.5
            # 3. 时序一致性评分（领先滞后关系的稳定性）
            timing_scores = []
            # 简化实现：计算信号变化的同步性
            for j in range(len(signal_names)):
                for k in range(j+1, len(signal_names)):
                    signal1 = window_signals[signal_names[j]]
                    signal2 = window_signals[signal_names[k]]
                    if len(signal1) >= 10 and len(signal2) >= 10:
                        # 计算一阶差分的相关性（同步性）
                        diff1 = signal1.diff().dropna()
                        diff2 = signal2.diff().dropna()
                        if len(diff1) >= 5 and len(diff2) >= 5:
                            min_len = min(len(diff1), len(diff2))
                            diff1_trimmed = diff1.iloc[:min_len]
                            diff2_trimmed = diff2.iloc[:min_len]
                            
                            sync_corr = diff1_trimmed.corr(diff2_trimmed)
                            if not np.isnan(sync_corr):
                                timing_scores.append(abs(sync_corr))
            timing_avg = np.mean(timing_scores) if timing_scores else 0.5
            # 综合一致性评分
            consistency_score = (
                direction_avg * 0.4 +
                correlation_avg * 0.4 +
                timing_avg * 0.2
            )
            consistency_scores.iloc[i] = consistency_score
        # 平滑处理
        consistency_scores = consistency_scores.ffill().fillna(0.5)
        consistency_scores = consistency_scores.rolling(window=5, min_periods=3).mean().fillna(0.5)
        return consistency_scores.clip(0, 1)

    def _calculate_predictive_quality(self, all_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V4.0】计算信号的预测有效性质量
        评估维度：信号对未来价格变动的预测能力
        数学模型：滞后相关性分析 + 信息系数（IC）
        """
        # 注意：这里需要价格数据，假设我们有一个全局的价格序列
        # 由于在上下文中没有价格数据，我们将使用相对强度信号作为代理
        # 简化实现：使用信号自身的变化来评估预测能力
        # 在实际应用中，应该使用信号与未来收益的相关性
        predictive_scores = pd.Series(0.5, index=next(iter(all_signals.values())).index)
        for i in range(len(predictive_scores)):
            if i < 40:  # 需要足够的历史数据
                predictive_scores.iloc[i] = 0.5
                continue
            # 使用最近20天的数据评估预测能力
            window_start = max(0, i-20)
            window_end = i
            # 收集各信号的预测能力评分
            signal_predictive_scores = []
            for signal_name, signal_series in all_signals.items():
                if len(signal_series) <= window_end:
                    continue
                window_data = signal_series.iloc[window_start:window_end]
                if len(window_data) < 10:
                    continue
                # 计算信号变化的预测能力
                # 简化：假设信号的趋势变化具有一定的持续性
                # 实际中应该计算信号与未来n日收益的相关性
                # 计算信号的自相关性（滞后1期）
                if len(window_data) >= 10:
                    autocorr = window_data.autocorr(lag=1)
                    if not np.isnan(autocorr):
                        # 适度的正自相关表示有一定的预测能力
                        # 但过高的自相关可能意味着信号过于平滑，反应迟钝
                        if autocorr > 0:
                            # 0.1-0.4的自相关为理想范围
                            if autocorr < 0.1:
                                pred_score = autocorr / 0.1
                            elif autocorr > 0.4:
                                pred_score = 1.0 - (autocorr - 0.4) / 0.6
                            else:
                                pred_score = 1.0
                        else:
                            pred_score = 0.0
                        signal_predictive_scores.append(pred_score)
            # 计算平均预测能力
            if signal_predictive_scores:
                predictive_score = np.mean(signal_predictive_scores)
            else:
                predictive_score = 0.5
            predictive_scores.iloc[i] = predictive_score
        # 平滑处理
        predictive_scores = predictive_scores.ffill().fillna(0.5)
        predictive_scores = predictive_scores.rolling(window=10, min_periods=5).mean().fillna(0.5)
        return predictive_scores.clip(0, 1)

    def _calculate_dynamic_quality_weights(self, individual_qualities: Dict[str, pd.Series], consistency_quality: pd.Series, predictive_quality: pd.Series) -> Dict[str, pd.Series]:
        """
        【V4.0】计算动态质量权重
        核心理念：根据各信号的历史表现动态调整其在质量评估中的权重
        表现好的信号获得更高权重，表现差的信号权重降低
        """
        # 初始化权重
        dynamic_weights = {}
        # 获取时间索引
        if individual_qualities:
            index = next(iter(individual_qualities.values())).index
        else:
            return {}
        # 为每个信号创建权重序列
        for signal_name in individual_qualities:
            dynamic_weights[signal_name] = pd.Series(1.0 / len(individual_qualities), index=index)
        # 如果只有1-2个信号，直接返回等权重
        if len(individual_qualities) <= 2:
            return dynamic_weights
        # 动态调整权重（基于最近的表现）
        for i in range(len(index)):
            if i < 30:  # 前30天使用等权重
                continue
            # 计算各信号最近20天的平均质量
            recent_qualities = {}
            total_quality = 0
            for signal_name, quality_series in individual_qualities.items():
                if i < len(quality_series):
                    recent_window = quality_series.iloc[max(0, i-20):i]
                    recent_avg = recent_window.mean() if len(recent_window) > 0 else 0.5
                    recent_qualities[signal_name] = recent_avg
                    total_quality += recent_avg
                else:
                    recent_qualities[signal_name] = 0.5
                    total_quality += 0.5
            # 计算归一化权重
            if total_quality > 0:
                for signal_name in individual_qualities:
                    if signal_name in recent_qualities:
                        # 权重与质量成正比
                        normalized_weight = recent_qualities[signal_name] / total_quality
                        # 限制权重范围：0.05 - 0.4
                        normalized_weight = max(0.05, min(normalized_weight, 0.4))
                        dynamic_weights[signal_name].iloc[i] = normalized_weight
                    else:
                        dynamic_weights[signal_name].iloc[i] = 1.0 / len(individual_qualities)
        return dynamic_weights
