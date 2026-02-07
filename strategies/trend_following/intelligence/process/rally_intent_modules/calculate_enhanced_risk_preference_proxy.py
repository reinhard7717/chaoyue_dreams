# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_enhanced_risk_preference_proxy.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

class EnhancedRiskPreferenceProxyCalculator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _calculate_risky_asset_performance_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.6 · 龙头溢价版】计算风险资产表现指数
        核心理念：行业龙头强度和市场整体情绪是风险偏好的最佳代理。
        逻辑：龙头强且情绪高 = 市场愿意承担风险追逐阿尔法。
        """
        # 1. 行业龙头强度 (代表进攻性资产表现)
        leader_strength = normalized_signals.get('industry_leader_norm', pd.Series(0.5, index=df_index))
        # 2. 市场整体情绪 (代表系统性风险偏好)
        market_sentiment = normalized_signals.get('market_sentiment_norm', pd.Series(0.5, index=df_index))
        # 3. 行业广度 (作为辅助验证)
        breadth = normalized_signals.get('industry_breadth_norm', pd.Series(0.5, index=df_index))
        # 4. 合成
        # 龙头强度最重要 (0.5)，情绪次之 (0.3)，广度 (0.2)
        performance_index = leader_strength * 0.5 + market_sentiment * 0.3 + breadth * 0.2
        return performance_index.clip(0, 1)

    def _calculate_risk_aversion_degree_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.6 · 派发逃逸版】计算风险规避程度指数
        核心理念：主力派发和空头行为反映了资金的避险意愿。
        逻辑：派发分数高 + 行为偏空 = 极度厌恶风险。
        """
        # 1. 派发分数 (Distribution Score)
        dist_score = normalized_signals.get('distribution_score_norm', pd.Series(0.5, index=df_index))
        # 2. 行为派发 (Behavior Distribution)
        behav_dist = normalized_signals.get('behavior_distribution_norm', pd.Series(0.5, index=df_index))
        # 3. 下行趋势强度 (Downtrend Strength)
        downtrend = normalized_signals.get('downtrend_strength_norm', pd.Series(0.5, index=df_index))
        # 4. 合成
        # 这是一个负向指标，值越高代表规避程度越高（风险偏好越低）
        aversion_index = dist_score * 0.4 + behav_dist * 0.3 + downtrend * 0.3
        return aversion_index.clip(0, 1)

    def _calculate_risk_transfer_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.6 · 资金跃迁版】计算风险转移指数
        核心理念：资金流的加速度代表了配置意愿的剧烈变化（从观望转为进场）。
        逻辑：资金加速流入 = 风险承担意愿激增。
        """
        # 1. 资金流加速度
        flow_accel = normalized_signals.get('flow_acceleration_norm', pd.Series(0.5, index=df_index))
        # 2. 净流入比率
        net_ratio = normalized_signals.get('net_amount_ratio_norm', pd.Series(0.5, index=df_index))
        # 3. 计算转移强度
        # 我们关注的是变化的速率(加速度)和方向(净流入)
        # 加速度 > 0.5 且 流入 > 0.5 -> 正向风险转移 (Risk On)
        transfer_index = flow_accel * 0.6 + net_ratio * 0.4
        return transfer_index.clip(0, 1)

    def _calculate_risk_pricing_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.6 · 波动定价版】计算风险定价指数
        核心理念：高波动率意味着市场要求更高的风险溢价。
        逻辑：波动率高 = 风险定价高 (Risk Pricing High)。通常这会抑制风险偏好，除非在极强趋势中。
        """
        # 1. 真实波幅 (ATR)
        atr = normalized_signals.get('ATR_norm', pd.Series(0.5, index=df_index))
        # 2. 布林带宽 (BBW)
        bbw = normalized_signals.get('BBW_norm', pd.Series(0.5, index=df_index))
        # 3. 合成
        pricing_index = atr * 0.5 + bbw * 0.5
        return pricing_index.clip(0, 1)

    def _calculate_risk_sentiment_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.6 · 突破信仰版】计算风险情绪指数
        核心理念：敢于在关键位突破和吸筹，体现了最强的风险承担情绪。
        逻辑：突破信心高 + 吸筹活跃 = 风险情绪高昂。
        """
        # 1. 突破信心 (Breakout Confidence)
        breakout_conf = normalized_signals.get('breakout_confidence_norm', pd.Series(0.5, index=df_index))
        # 2. 吸筹分数 (Accumulation Score)
        accum_score = normalized_signals.get('accumulation_score_norm', pd.Series(0.5, index=df_index))
        # 3. 趋势确认 (Trend Confirmation)
        trend_conf = normalized_signals.get('trend_confirmation_norm', pd.Series(0.5, index=df_index))
        # 4. 合成
        sentiment_index = breakout_conf * 0.4 + accum_score * 0.3 + trend_conf * 0.3
        return sentiment_index.clip(0, 1)

    def _calculate_risk_contagion_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.6 · 恐慌蔓延版】计算风险传导指数
        核心理念：预警信号的密集出现预示着风险的连锁反应。
        逻辑：反转预警 + 假突破风险 = 传染风险高。
        """
        # 1. 反转预警分数
        reversal_warn = normalized_signals.get('reversal_warning_score_norm', pd.Series(0.5, index=df_index))
        # 2. 突破风险预警
        breakout_risk = normalized_signals.get('breakout_risk_warning_norm', pd.Series(0.5, index=df_index))
        # 3. 合成
        # 这是一个负向指标，值越高风险越大
        contagion_index = reversal_warn * 0.6 + breakout_risk * 0.4
        return contagion_index.clip(0, 1)

    def _neural_inspired_synthesis(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V4.6 · 神经元激活版】神经元启发式合成
        核心理念：模拟单个神经元的加权求和与非线性激活，引入偏置项处理基础风险偏好。
        公式：output = sigmoid(sum(w * x) + bias)
        """
        weighted_sum = pd.Series(0.0, index=df_index)
        # 1. 加权求和 (线性部分)
        for key, val in components.items():
            w = weights.get(key, 0)
            # 对于负向指标（规避、传导），组件值已经预处理为 (1-x)，所以这里直接加权即可
            # 假设传入的components已经是正向化处理过的（即值越大越利于风险偏好）
            weighted_sum += val * w
        # 2. 引入偏置 (Bias)
        # A股由于散户多，基础风险偏好往往略高于理性值（容易跟风），设 bias = 0.1
        bias = 0.1
        # 3. 缩放因子 (Scale)
        # 为了让 Sigmoid 在 0.5 附近有较好的区分度
        scale = 6.0
        # 4. 激活函数 (Sigmoid)
        # Input range approx [0, 1], centered at 0.5
        # (x - 0.5) * scale + bias
        z = (weighted_sum - 0.5) * scale + bias
        output = 1 / (1 + np.exp(-z))
        return output.clip(0, 1)

    def _identify_economic_cycle(self, components: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.6 · 周期象限版】识别微观市场周期
        核心理念：利用趋势(增长代理)和动量(热度代理)划分四个象限。
        1. 复苏 (Recovery): 趋势弱但动量起 (Risk On)
        2. 过热 (Expansion): 趋势强且动量强 (Risk On++)
        3. 滞胀 (Stagflation): 趋势强但动量落 (Risk Off)
        4. 衰退 (Contraction): 趋势弱且动量弱 (Risk Off++)
        """
        # 使用 Risk Performance 代表趋势/增长潜力
        trend_proxy = components.get('risky_performance', pd.Series(0.5, index=df_index))
        # 使用 Risk Pricing (波动率) 反向 或 Risk Transfer (资金动量) 作为动量/热度代理
        # 这里选用 Risk Transfer (资金加速)
        momentum_proxy = components.get('risk_transfer', pd.Series(0.5, index=df_index))
        cycles = pd.Series("Unknown", index=df_index)
        threshold = 0.5
        for i in range(len(df_index)):
            t_val = trend_proxy.iloc[i]
            m_val = momentum_proxy.iloc[i]
            if t_val > threshold and m_val > threshold:
                cycles.iloc[i] = "Expansion" # 过热/繁荣
            elif t_val < threshold and m_val < threshold:
                cycles.iloc[i] = "Contraction" # 衰退/冰点
            elif t_val > threshold and m_val <= threshold:
                cycles.iloc[i] = "Slowdown" # 滞胀/筑顶
            elif t_val <= threshold and m_val > threshold:
                cycles.iloc[i] = "Recovery" # 复苏/启动
                
        return cycles

    def _calculate_risk_preference_modulator(self, economic_cycle: pd.Series, config: Dict) -> pd.Series:
        """
        【V4.6 · 顺势调节版】计算风险偏好调节器
        核心理念：在复苏和过热期放大信号，在滞胀和衰退期抑制信号。
        """
        modulator = pd.Series(1.0, index=economic_cycle.index)
        # 映射逻辑
        modulator = modulator.mask(economic_cycle == "Expansion", 1.2) # 激进
        modulator = modulator.mask(economic_cycle == "Recovery", 1.1)  # 积极
        modulator = modulator.mask(economic_cycle == "Slowdown", 0.8)  # 谨慎
        modulator = modulator.mask(economic_cycle == "Contraction", 0.6) # 防御
        # 平滑处理
        return modulator.rolling(window=5, min_periods=1).mean().fillna(1.0)