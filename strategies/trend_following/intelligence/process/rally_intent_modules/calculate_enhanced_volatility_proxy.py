# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_enhanced_volatility_proxy.py

import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

class EnhancedVolatilityProxyCalculator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config


    def _calculate_historical_volatility_index(self, pct_change_norm: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.5 · 多周期融合版】计算历史波动率指数
        核心理念：综合短中长周期波动率，构建全景波动画像。
        """
        # 还原近似收益率 (假设 norm 是归一化后的，这里需要还原波动特性，直接计算 norm 的 std 也可以代表相对波动)
        # 更好的是使用原始 pct_change，但接口只传了 norm。我们假设 norm 保留了分布特征。
        # 1. 计算不同周期的滚动标准差 (代表波动率)
        vol_5 = pct_change_norm.rolling(window=5).std()
        vol_20 = pct_change_norm.rolling(window=20).std()
        vol_60 = pct_change_norm.rolling(window=60).std()
        # 2. 加权融合 (近期权重更高)
        # 归一化处理：假设波动率 norm 后在 0.1-0.5 之间
        composite_vol = vol_5 * 0.5 + vol_20 * 0.3 + vol_60 * 0.2
        # 3. 映射到 [0, 1] (0代表极度平静，1代表极度动荡)
        # 假设最大波动率为 0.5 (归一化数据的标准差)
        return (composite_vol / 0.5).clip(0, 1).fillna(0)

    def _calculate_volatility_skew_index(self, pct_change_norm: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.5 · 偏度博弈版】计算波动率偏度
        核心理念：区分"良性波动"(上涨)与"恶性波动"(下跌)。
        Skew > 0.5: 上行波动主导 (进攻)
        Skew < 0.5: 下行波动主导 (恐慌)
        """
        # 0.5 是 norm 的中性点
        returns = pct_change_norm - 0.5
        # 1. 上行波动 (Semivariance Upside)
        up_returns = returns.where(returns > 0, 0)
        up_vol = up_returns.rolling(window=20).std()
        # 2. 下行波动 (Semivariance Downside)
        down_returns = returns.where(returns < 0, 0)
        down_vol = down_returns.rolling(window=20).std()
        # 3. 计算偏度比率
        # 避免除零
        skew_ratio = up_vol / (down_vol + 1e-9)
        # 4. 映射到 [0, 1] (1 表示极端上行波动，0 表示极端下行波动，0.5 平衡)
        # 使用 sigmoid 压缩
        return (1 / (1 + np.exp(-5 * (skew_ratio - 1)))).fillna(0.5)

    def _calculate_volatility_clustering_index(self, pct_change_norm: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.5 · GARCH效应版】计算波动率聚集指数
        核心理念：波动率具有记忆性。聚集度高意味着当前处于"变盘窗口"。
        """
        # 1. 取绝对收益 (代表波动幅度)
        abs_returns = (pct_change_norm - 0.5).abs()
        # 2. 计算自相关性 (Autocorrelation)
        # 滚动计算 lag-1 的自相关系数
        clustering = abs_returns.rolling(window=20).apply(lambda x: x.autocorr(lag=1), raw=False)
        # 3. 映射 (自相关性越高，聚集效应越强 -> 风险/机会越大)
        # corr 通常在 [-0.5, 0.5] 之间，负相关代表均值回归，正相关代表趋势延续
        # 我们关注正相关 (聚集)
        return clustering.clip(0, 1).fillna(0)

    def _calculate_volatility_smile_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.5 · 肥尾代理版】计算波动率微笑代理
        核心理念：极端行情出现的频率。频率越高，隐含的"黑天鹅"溢价越高。
        """
        # 使用 pct_change_norm
        returns = normalized_signals.get('pct_change_norm', pd.Series(0.5, index=df_index)) - 0.5
        # 1. 计算峰度 (Kurtosis) - 滚动窗口
        kurt = returns.rolling(window=20).kurt()
        # 2. 映射 (峰度越高，肥尾越严重，微笑曲线越陡)
        # 正态分布峰度为0 (Pandas Fisher定义)。A股通常尖峰肥尾，Kurt > 0
        # 映射到 [0, 1]
        smile_index = np.tanh(kurt * 0.2).clip(0, 1)
        return smile_index.fillna(0)

    def _calculate_volatility_term_structure_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.5 · 期限结构版】计算波动率期限结构
        核心理念：短期波动 vs 长期波动。
        倒挂 (Short > Long) -> 恐慌/高潮 -> 1.0
        正挂 (Short < Long) -> 酝酿/平稳 -> 0.0
        """
        returns = normalized_signals.get('pct_change_norm', pd.Series(0.5, index=df_index))
        # 1. 短期波动 (5日)
        vol_short = returns.rolling(window=5).std()
        # 2. 长期波动 (60日)
        vol_long = returns.rolling(window=60).std()
        # 3. 比率
        ratio = vol_short / (vol_long + 1e-9)
        # 4. 映射 (Ratio > 1 为倒挂)
        # 使用 sigmoid 中心化在 1.0
        term_structure = 1 / (1 + np.exp(-5 * (ratio - 1)))
        return term_structure.fillna(0.5)

    def _calculate_volatility_risk_premium_index(self, historical_volatility: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.5 · 帕金森差值版】计算波动率风险溢价代理
        核心理念：盘中震幅 (High-Low) 与 收盘变动 (Close-Close) 的差值。
        差值大 -> 盘中分歧大但方向未明 -> 风险溢价高。
        注：这里缺少 High/Low 数据，使用 absolute_change_strength (作为震幅代理) 和 pct_change (作为收盘变动代理)
        """
        # 假设 absolute_change_strength_norm 代表震幅强度
        # historical_volatility 代表收盘波动率
        # 模拟 Parkinson Volatility (震幅波动)
        # 震幅通常大于收盘变动
        parkinson_proxy = historical_volatility * 1.5 # 简化假设
        # 溢价 = 震幅波动 - 收盘波动
        premium = (parkinson_proxy - historical_volatility).clip(lower=0)
        return np.tanh(premium * 5).fillna(0)

    def _inverse_u_transform(self, series: pd.Series, optimal_low: float, optimal_high: float) -> pd.Series:
        """
        【V4.5 · 倒U效用版】倒U型变换
        核心理念：波动不是越低越好，也不是越高越好。适度的波动最利于趋势跟踪。
        """
        # 中心点
        center = (optimal_low + optimal_high) / 2
        width = (optimal_high - optimal_low) / 2
        # 高斯函数模拟倒U型
        # exp(- (x - center)^2 / (2 * width^2))
        score = np.exp(-((series - center) ** 2) / (2 * (width ** 2)))
        return score.fillna(0)

    def _bayesian_synthesis(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V4.5 · 信念更新版】贝叶斯合成
        核心理念：基于各子信号的"信度"(如波动率聚集度)动态调整权重。
        这里简化为加权平均，但引入了非线性增强。
        """
        weighted_sum = pd.Series(0.0, index=df_index)
        total_weight = 0.0
        for key, val in components.items():
            w = weights.get(key, 0)
            weighted_sum += val * w
            total_weight += w
            
        if total_weight == 0:
            return pd.Series(0.5, index=df_index)
            
        # 基础评分
        base_score = weighted_sum / total_weight
        # 贝叶斯后验修正 (模拟)
        # 如果"聚集度"(Clustering)很高，说明信号可信度高，强化当前评分偏离中性的程度
        clustering = components.get('clustering', pd.Series(0.0, index=df_index))
        # 强化因子：Clustering 越大，Score 越远离 0.5
        # Score' = 0.5 + (Score - 0.5) * (1 + Clustering)
        final_score = 0.5 + (base_score - 0.5) * (1 + clustering * 0.5)
        return final_score.clip(0, 1)

    def _identify_volatility_regime(self, components: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.5 · 波动体制版】识别波动率体制
        返回：
        0: Low Vol (死水/筑底)
        1: Expanding (突破/升波)
        2: High Vol (高潮/剧震)
        3: Contracting (衰退/降波)
        """
        hist_vol = components.get('historical', pd.Series(0.0, index=df_index))
        term_structure = components.get('term_structure', pd.Series(0.5, index=df_index)) # >0.5 倒挂(短>长)
        regimes = pd.Series(0, index=df_index)
        # 1. High Vol: 历史波动高 & 期限结构倒挂(极度恐慌/高潮)
        mask_high = (hist_vol > 0.6) & (term_structure > 0.6)
        # 2. Expanding: 历史波动中低 & 期限结构开始倒挂(短期波动起)
        mask_expand = (hist_vol <= 0.6) & (term_structure > 0.5)
        # 3. Contracting: 历史波动高 & 期限结构正挂(短期平复)
        mask_contract = (hist_vol > 0.6) & (term_structure <= 0.5)
        # 4. Low Vol: 双低
        mask_low = (hist_vol <= 0.3) & (term_structure <= 0.5)
        regimes[mask_expand] = 1
        regimes[mask_high] = 2
        regimes[mask_contract] = 3
        regimes[mask_low] = 0
        return regimes

    def _calculate_volatility_modulator(self, market_regime: pd.Series, config: Dict) -> pd.Series:
        """
        【V4.5 · 波动风控版】计算波动率调节器
        逻辑：
        - Expanding (1): 机会最大，放大 (1.2)
        - Low Vol (0): 容易假突破，适中 (1.0)
        - Contracting (3): 鱼尾行情，减仓 (0.8)
        - High Vol (2): 风险极大，显著抑制 (0.6)
        """
        modulator = pd.Series(1.0, index=market_regime.index)
        modulator = modulator.mask(market_regime == 1, 1.2) # 升波期追涨
        modulator = modulator.mask(market_regime == 0, 1.0) # 低波期观望
        modulator = modulator.mask(market_regime == 3, 0.8) # 降波期止盈
        modulator = modulator.mask(market_regime == 2, 0.6) # 高波期风控
        return modulator.rolling(window=3).mean().fillna(1.0)

