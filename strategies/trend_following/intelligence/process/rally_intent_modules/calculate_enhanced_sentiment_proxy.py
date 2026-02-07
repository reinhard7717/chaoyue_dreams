# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_enhanced_sentiment_proxy.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple


class EnhancedSentimentProxyCalculator:
    def __init__(self):
        pass

    def _calculate_sentiment_divergence_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.3 · 量价情绪背离版】计算情绪背离指数
        核心理念：价格创新高而情绪未跟随（顶背离），或价格创新低而情绪回暖（底背离）。
        A股特性：指数往往在情绪退潮期通过权重股掩护进行最后的诱多。
        """
        # 1. 获取价格趋势 (使用RSI或pct_change的均值作为代理)
        price_trend = normalized_signals.get('RSI_norm', pd.Series(0.5, index=df_index))
        # 2. 获取情绪趋势
        sentiment_trend = normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index))
        # 3. 计算趋势斜率 (5日)
        p_slope = price_trend.diff(5).fillna(0)
        s_slope = sentiment_trend.diff(5).fillna(0)
        # 4. 识别背离
        # 顶背离：价格向上(>0) 但 情绪向下(<0)
        top_div = (p_slope > 0.05) & (s_slope < -0.05)
        # 底背离：价格向下(<0) 但 情绪向上(>0) -> 这通常是好事，但在"背离风险"语境下，我们主要关注顶背离风险
        # 这里我们计算"背离程度"，不分方向，作为不确定性指标
        divergence = pd.Series(0.0, index=df_index)
        # 计算背离角：二者运动方向相反的程度
        # 乘积为负代表方向相反
        product = p_slope * s_slope
        # 归一化背离强度
        div_intensity = (p_slope.abs() + s_slope.abs()) * 0.5
        divergence = divergence.mask(product < -0.0025, div_intensity) # 阈值过滤微小波动
        # 映射到 [0, 1]
        return np.tanh(divergence * 5).clip(0, 1)

    def _calculate_sentiment_extremity_index(self, base_sentiment: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.3 · 冰点沸点版】计算情绪极端指数
        核心理念：均值回归。情绪偏离均值越远，反转概率越大。
        """
        # 1. 动态基准线 (21日均线)
        rolling_mean = base_sentiment.rolling(window=21).mean()
        rolling_std = base_sentiment.rolling(window=21).std()
        # 2. 偏离度 (Z-Score)
        z_score = (base_sentiment - rolling_mean) / (rolling_std + 1e-9)
        # 3. 极端性映射 (双尾)
        # 我们关注 Z > 2 (沸点) 或 Z < -2 (冰点)
        extremity = (z_score.abs() - 1.5).clip(lower=0) # 从1.5倍标准差开始计分
        # 4. 归一化
        # 假设 3倍标准差为满分
        return (extremity / 1.5).clip(0, 1)

    def _calculate_sentiment_contagion_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.3 · 龙头板块共振版】计算情绪传染指数
        核心理念：龙头股(点)能否带动板块(面)。
        """
        # 1. 龙头强度
        leader = normalized_signals.get('industry_leader_norm', pd.Series(0.5, index=df_index))
        # 2. 板块广度 (赚钱效应)
        breadth = normalized_signals.get('industry_breadth_norm', pd.Series(0.5, index=df_index))
        # 3. 计算协动性 (Co-movement)
        # 简单乘积：两者都强 = 传染强；一强一弱 = 割裂
        # 调整：如果 Leader 强 (0.8) 但 Breadth 弱 (0.3)，传染性 = 0.3 (被拖累)
        # 如果 Leader 弱 Breadth 强，说明是补涨，传染性一般
        # 使用几何平均强调"短板效应"
        contagion = np.sqrt(leader * breadth)
        # 4. 动态修正
        # 如果两者差值过大，说明断层，扣分
        diff_penalty = (leader - breadth).abs()
        contagion = contagion * (1.0 - diff_penalty * 0.5)
        return contagion.clip(0, 1)

    def _calculate_sentiment_stability_index(self, base_sentiment: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.3 · 情绪稳态版】计算情绪稳定性指数
        核心理念：稳步推升的情绪最有利于趋势延续。
        """
        # 1. 计算短期波动率 (5日)
        volatility = base_sentiment.rolling(window=5).std()
        # 2. 归一化反转
        # 假设波动率 0.2 为高波动
        stability = 1.0 - (volatility / 0.2).clip(0, 1)
        # 3. 平滑处理
        return stability.rolling(window=3).mean().fillna(0.5)

    def _fuzzy_logic_synthesis(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V4.3 · 模糊门控版】模糊逻辑合成
        核心理念：非线性决策。
        规则示例：
        - IF Extremity IS High THEN Weight(Momentum) reduced.
        - IF Contagion IS Low THEN Sentiment IS Fake.
        """
        # 提取组件
        base = components.get('base', pd.Series(0.5, index=df_index))
        extremity = components.get('extremity', pd.Series(0.0, index=df_index))
        contagion = components.get('contagion', pd.Series(0.5, index=df_index))
        divergence = components.get('divergence', pd.Series(0.0, index=df_index))
        # 基础线性加权
        weighted_sum = pd.Series(0.0, index=df_index)
        total_weight = sum(weights.values())
        for key, val in components.items():
            weighted_sum += val * weights.get(key, 0)
        linear_score = weighted_sum / total_weight
        # 应用模糊规则 (Gate Mechanisms)
        # 规则1: 极端性惩罚 (均值回归压力)
        # 当极端性很高时，情绪得分倾向于中性(0.5)或反转，这里简单做压制
        # Extremity 0 -> 1.0, Extremity 1 -> 0.5
        extremity_gate = 1.0 - extremity * 0.5
        # 规则2: 传染性增强 (真实性确认)
        # 传染性低，情绪得分打折
        # Contagion 0 -> 0.6, Contagion 1 -> 1.1
        contagion_gate = 0.6 + contagion * 0.5
        # 规则3: 背离否决
        # 背离严重时，看多情绪失效
        # Divergence 1 -> 0.2
        divergence_gate = 1.0 - divergence * 0.8
        # 合成
        final_score = linear_score * extremity_gate * contagion_gate * divergence_gate
        return final_score.clip(0, 1)

    def _calculate_sentiment_modulator(self, market_phase: pd.Series, config: Dict) -> pd.Series:
        """
        【V4.3 · 周期共振版】计算情绪调节器
        核心理念：在情绪周期的不同阶段，情绪指标的有效性不同。
        """
        modulator = pd.Series(1.0, index=market_phase.index)
        # 映射逻辑
        # 启动期/主升期：情绪与价格正反馈，放大信号 (x 1.2)
        # 调整期：情绪波动大，适度抑制 (x 0.9)
        # 反转风险/衰退期：情绪容易骗线，显著抑制 (x 0.7)
        # 崩溃/冰点期：情绪具有反指意义，需特殊处理 (这里简单做抑制)
        mask_bull = (market_phase == "启动") | (market_phase == "主升")
        mask_bear = (market_phase == "反转风险") | (market_phase == "趋势下跌")
        mask_correction = (market_phase == "调整")
        modulator = modulator.mask(mask_bull, 1.2)
        modulator = modulator.mask(mask_bear, 0.7)
        modulator = modulator.mask(mask_correction, 0.9)
        return modulator.fillna(1.0)









