# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\assess_signal_quality.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

class SignalQualityAssessor:
    def __init__(self):
        pass

    def _calculate_individual_signal_quality(self, signal_series: pd.Series, signal_name: str) -> pd.Series:
        """
        【V5.0 · 工业级高频哨兵版】计算单个信号的个体质量指标
        修改思路：摒弃循环计算，采用全向量化滚动算子，集成卡夫曼效率比与非线性缺失惩罚。
        """
        if signal_series.empty:
            return pd.Series(0.5, index=signal_series.index)
        # 1. 采样有效性检查 (Sampling Validity)
        # 统计窗口内的非有限值（NaN/Inf）比例，实施阶梯式惩罚
        validity = (~np.isfinite(signal_series)).astype(float).rolling(window=20).mean().fillna(0)
        completeness_score = (1.0 - validity * 1.5).clip(0, 1)
        # 2. 动态信噪比 (卡夫曼效率比 K-ER)
        # 逻辑：Net_Change / Sum_of_Absolute_Changes
        net_change = signal_series.diff(10).abs()
        sum_abs_change = signal_series.diff().abs().rolling(window=10).sum()
        er = (net_change / (sum_abs_change + 1e-9)).clip(0, 1)
        # A股特性：ER在0.3-0.6之间代表信号质量最佳，过高（1.0）可能是断层，过低（0.1）是杂波
        snr_score = (1.0 - (er - 0.45).abs() * 1.5).clip(0, 1)
        # 3. 稳定性评分 (Volatility Clustering Detection)
        # 衡量信号本身的波动率波动，识别“平稳期”
        signal_vol = signal_series.diff().rolling(window=20).std()
        vol_stability = 1.0 - (signal_vol / (signal_vol.rolling(window=60).max() + 1e-9)).fillna(0.5)
        # 4. 极端值分布 (Fat-Tail Defense)
        # 计算偏度与峰度对质量的影响，过滤由于数据毛刺产生的伪信号
        is_outlier = (signal_series - signal_series.rolling(20).mean()).abs() > (3 * signal_series.rolling(20).std())
        extreme_penalty = 1.0 - is_outlier.astype(float).rolling(window=20).mean().fillna(0)
        # 综合个体质量
        # 权重：信噪比 40%，采样有效性 25%，稳定性 20%，极端值防御 15%
        individual_quality = (
            snr_score * 0.4 +
            completeness_score * 0.25 +
            vol_stability * 0.2 +
            extreme_penalty * 0.15
        )
        return individual_quality.ewm(span=5, adjust=False).mean().clip(0, 1)

    def _calculate_signal_consistency_quality(self, all_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.0 · 向量化一致性版】计算多维信号间的共振质量
        修改思路：通过向量化相关性矩阵迹（Trace）计算，衡量各代理信号是否处于“协同进攻”状态。
        """
        # 收集有效信号
        valid_series = [s for s in all_signals.values() if not s.empty]
        if len(valid_series) < 2:
            return pd.Series(0.5, index=next(iter(all_signals.values())).index)
        signal_matrix = pd.concat(valid_series, axis=1)
        # 计算滚动相关性均值
        # 逻辑：在 A 股反转初期，相关性较低是正常的；但在拉升中期，高相关性代表共振合力强
        rolling_corr = signal_matrix.rolling(window=20).corr().groupby(level=0).mean().mean(axis=1)
        # 映射逻辑：一致性得分 = |Avg_Correlation|
        consistency_score = rolling_corr.abs().fillna(0.5).clip(0, 1)
        return consistency_score.ewm(span=5, adjust=False).mean()

    def _calculate_predictive_quality(self, all_signals: Dict[str, pd.Series]) -> pd.Series:
        """
        【V5.0 · 解析度衰减版】评估信号的预测潜质
        修改思路：通过分析信号的“自相关衰减速度”识别虚假平滑。
        """
        first_s = next(iter(all_signals.values()))
        # 计算信号的 1 阶与 3 阶自相关差值
        # 如果 1 阶极高且 3 阶骤降，说明信号仅有瞬时噪声，无持续预测力
        # 此处使用代理信号的动量一致性作为预测能力的模拟
        combined_predictive = pd.Series(0.0, index=first_s.index)
        for s in all_signals.values():
            r1 = s.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=True).fillna(0)
            r3 = s.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=3), raw=True).fillna(0)
            # 预测力得分：当 r1 高且 r1/r3 比例适中时，信号最具解析度
            s_pred = (r1 * (r3 / (r1 + 1e-9))).clip(0, 1)
            combined_predictive += s_pred
        return (combined_predictive / len(all_signals)).fillna(0.5).clip(0, 1)

    def _calculate_dynamic_quality_weights(self, individual_qualities: Dict[str, pd.Series], consistency_quality: pd.Series, predictive_quality: pd.Series) -> Dict[str, pd.Series]:
        """
        【V5.0 · Softmax温度版】动态分配信号置信度权重
        修改思路：引入质量温度参数 T。当信号质量出现断层时，自动将权重聚焦于高质量维度。
        """
        index = consistency_quality.index
        dynamic_weights = {name: pd.Series(0.0, index=index) for name in individual_qualities.keys()}
        # 预先堆叠质量矩阵以提升性能
        quality_matrix = pd.concat(list(individual_qualities.values()), axis=1)
        # 逐日执行 Softmax 权重分配
        for i in range(len(index)):
            raw_q = quality_matrix.iloc[i].values
            # 计算动态温度：如果整体一致性差，提高温度（趋向均分权重）；一致性好，降低温度（聚焦优势维度）
            temp = 1.5 - consistency_quality.iloc[i]  # T ∈ [0.5, 1.5]
            # 执行带有预测增益修正的 Softmax
            exp_q = np.exp((raw_q + predictive_quality.iloc[i] * 0.2) / temp)
            norm_w = exp_q / (np.sum(exp_q) + 1e-9)
            for idx, name in enumerate(individual_qualities.keys()):
                dynamic_weights[name].iloc[i] = norm_w[idx]
        return dynamic_weights
