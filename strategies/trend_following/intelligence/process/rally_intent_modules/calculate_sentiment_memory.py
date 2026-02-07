# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_sentiment_memory.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple


class CalculateSentimentMemory:
    def __init__(self, df_index, raw_signals, params):
        self.df_index = df_index
        self.raw_signals = raw_signals
        self.params = params


    def _calculate_sentiment_divergence(self, sentiment_series: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.5 · 量价背离版】计算情绪背离度
        核心理念：识别"价格创新高但情绪未跟随"的顶部背离，或"价格新低但情绪回暖"的底部背离。
        注意：此方法返回的是"背离风险度"，值越大代表背离越严重(风险/机会越大)，需结合趋势方向判断。
        """
        # 需要获取价格序列进行对比（这里假设从self.strategy获取或作为参数传入较复杂，
        # 简化为使用sentiment自身的趋势背离，即实际值与线性回归值的偏离）
        # 更优解：计算价格动量与情绪动量的相关性
        # 这里我们模拟一个"价格动量"的代理变量，假设 raw_signals 已在外部准备好，
        # 若无法获取，则计算情绪自身的RSI背离特征
        # 方案：计算情绪指标的短期趋势(5日)与长期趋势(21日)的乖离率
        # 逻辑：短期情绪过快偏离长期均值，视为不可持续的"情绪超涨/超跌"
        short_trend = sentiment_series.rolling(window=5).mean()
        long_trend = sentiment_series.rolling(window=memory_period).mean()
        # 计算乖离率 (Bias)
        bias = (short_trend - long_trend) / (long_trend + 1e-9)
        # 计算背离风险
        # 逻辑：乖离率过大(>0.2)或过小(<-0.2)都是背离
        # 使用高斯函数反向映射：乖离率越接近0，分歧越小(0)；乖离率越大，分歧越大(1)
        divergence_score = 1.0 - np.exp(-((bias) ** 2) / (2 * (0.1 ** 2)))
        return divergence_score.fillna(0).clip(0, 1)

    def _detect_sentiment_extreme(self, sentiment_series: pd.Series, df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.6 · 修复版】检测情绪极端值
        修复说明：解决 min_periods > window 导致的 ValueError。动态设置 min_periods。
        核心理念：利用Z-Score识别情绪的"冰点"(Panic)与"沸点"(Euphoria)。
        """
        # 动态计算安全的 min_periods，确保不超过 memory_period
        # 逻辑：至少需要 2 个数据点，且不超过窗口的 80% 或 20 中的较小值（如果窗口很大，至少要20个样本才稳健；如果窗口小，则适配窗口）
        # 但为了彻底避免错误，最安全的做法是直接取 min(20, memory_period) 且保证 <= memory_period
        # 这里使用宽松策略：取窗口的 2/3 作为最小样本数，既保证统计意义又不报错
        safe_min_periods = max(2, int(memory_period * 0.6))
        # 1. 计算滚动均值与标准差 (Bollinger Band logic)
        rolling_mean = sentiment_series.rolling(window=memory_period, min_periods=safe_min_periods).mean()
        rolling_std = sentiment_series.rolling(window=memory_period, min_periods=safe_min_periods).std()
        # 2. 计算 Z-Score
        # (当前值 - 历史均值) / 历史波动率
        z_score = (sentiment_series - rolling_mean) / (rolling_std + 1e-9)
        # 3. 极端性映射
        # 我们关注的是"极端程度"，不分方向。方向由外部逻辑判断。
        # Z > 2 或 Z < -2 视为极端
        # 映射到 [0, 1]，2倍标准差对应 0.8 分，3倍对应 1.0 分
        extreme_score = (z_score.abs() / 3.0).clip(0, 1)
        # 4. 非线性增强
        # 只有当Z-Score超过1.5时，极端分才开始显著增加
        extreme_score = extreme_score.where(z_score.abs() > 1.5, extreme_score * 0.5)
        return extreme_score.fillna(0)

    def _calculate_sentiment_consistency(self, raw_signals: Dict[str, pd.Series], df_index: pd.Index, memory_period: int) -> pd.Series:
        """
        【V3.5 · 多维共振版】计算情绪一致性
        核心理念：当"大盘"、"板块"、"龙头"三者情绪共振时，趋势最强。
        A股特性：只有龙头涨而板块不跟（一致性低），通常是诱多；全线普涨（一致性高）才是真反转。
        """
        # 1. 获取三个维度的情绪指标
        # 市场整体情绪
        mkt_sentiment = raw_signals.get('market_sentiment', pd.Series(0.5, index=df_index))
        # 板块广度 (涨跌家数比等)
        breadth = raw_signals.get('industry_breadth', pd.Series(0.5, index=df_index))
        # 龙头强度 (最高板高度、龙头涨幅)
        leader = raw_signals.get('industry_leader', pd.Series(0.5, index=df_index))
        # 2. 归一化处理 (确保都在0-1之间，方便比较)
        # 假设原始信号已经是归一化过的，若没有，建议在此处再次Robust Scaling
        # 这里直接使用
        # 3. 计算离散度 (Standard Deviation)
        # 构建DataFrame以便按行计算
        components = pd.DataFrame({
            'mkt': mkt_sentiment,
            'breadth': breadth,
            'leader': leader
        })
        # 计算横截面标准差
        std_dev = components.std(axis=1)
        # 4. 转换为一致性得分
        # 标准差越小，一致性越高
        # 假设最大标准差约为 0.5 (全部分散)，我们希望 Std=0 -> Score=1
        consistency_score = 1.0 - (std_dev / 0.3).clip(0, 1)
        # 5. 趋势方向确认 (可选增强)
        # 只有在三者都 > 0.5 (强势共振) 或都 < 0.5 (弱势共振) 时，才给予高分
        # 若均值在 0.5 附近震荡，即使标准差小，也是无效的一致性
        mean_val = components.mean(axis=1)
        intensity_factor = (mean_val - 0.5).abs() * 2  # 0.5->0, 0/1->1
        final_consistency = consistency_score * (0.5 + 0.5 * intensity_factor)
        return final_consistency.fillna(0).clip(0, 1)













