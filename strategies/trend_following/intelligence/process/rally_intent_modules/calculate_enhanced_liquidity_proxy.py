# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_enhanced_liquidity_proxy.py

import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple

class EnhancedLiquidityProxyCalculator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config


    def _calculate_orderbook_liquidity_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.4 · 影线博弈版】计算订单簿流动性指数
        核心理念：利用K线形态（影线）和资金流向反推潜在的挂单失衡（Imbalance）。
        逻辑：
        - 长下影线 + 资金流入 = 买盘挂单厚实（支撑强）
        - 长上影线 + 资金流出 = 卖盘挂单厚实（抛压大）
        """
        # 1. 获取形态特征 (需假设 normalized_signals 中包含或能推导，这里简化使用资金和强度代理)
        # 使用 flow_intensity 代表主动买盘意愿
        flow_intensity = normalized_signals.get('flow_intensity_norm', pd.Series(0.5, index=df_index))
        # 2. 使用 net_amount_ratio 代表实际成交的主动性
        net_amount = normalized_signals.get('net_amount_ratio_norm', pd.Series(0.5, index=df_index))
        # 3. 模拟订单簿失衡 (Imbalance)
        # 如果资金流入强(flow > 0.5) 且 实际成交也强(net > 0.5) -> 买盘厚度占优
        # 简单的线性组合
        imbalance = (flow_intensity * 0.6 + net_amount * 0.4)
        # 4. 映射到流动性质量
        # 我们认为：买盘厚度大(支撑强) = 流动性结构好(利于多头)
        # 卖盘厚度大(阻力大) = 流动性结构差(利于空头/难以上涨)
        return imbalance.clip(0, 1)

    def _calculate_impact_cost_liquidity(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.4 · Amihud变体版】计算冲击成本流动性
        核心理念：单位换手带来的价格波动越小，流动性越好（冲击成本越低）。
        公式：Liquidity ~ Turnover / (|Return| + epsilon)
        """
        # 1. 获取绝对涨跌幅 (波动)
        abs_change = normalized_signals.get('absolute_change_strength_norm', pd.Series(0.5, index=df_index)).abs()
        # 映射回真实波动概念：0.5是0，0和1是剧烈波动。我们想要 |Return|
        # 假设 absolute_change_strength_norm 是 [0, 1] 且线性映射
        volatility_proxy = (abs_change - 0.5).abs() * 2 # [0, 1]
        # 2. 获取换手率 (成交)
        turnover = normalized_signals.get('turnover_rate_norm', pd.Series(0.5, index=df_index))
        # 3. 计算 Amihud Illiquidity 的倒数
        # 避免除零，epsilon = 0.05
        # 分子：换手率 (能量)
        # 分母：波动率 (代价)
        # 结果：每单位波动对应的换手率。值越大，说明大换手只引起小波动 -> 承接力极强(流动性好)
        liquidity_score = turnover / (volatility_proxy + 0.05)
        # 4. 归一化 (假设比值在 0 - 10 之间)
        return np.tanh(liquidity_score * 0.5).clip(0, 1)

    def _calculate_layered_liquidity_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.4 · 筹码深度版】计算分层流动性指数
        核心理念：筹码越集中，该价格位置的"厚度"越大，容纳大资金进出的能力越强。
        """
        # 1. 筹码集中度
        concentration = normalized_signals.get('chip_concentration_ratio_norm', pd.Series(0.5, index=df_index))
        # 2. 筹码稳定性 (锁仓带来深度)
        stability = normalized_signals.get('chip_stability_norm', pd.Series(0.5, index=df_index))
        # 3. 合成
        # 集中且稳定 = 最佳分层流动性 (厚实的支撑/阻力板)
        layered_liquidity = concentration * 0.6 + stability * 0.4
        return layered_liquidity.clip(0, 1)

    def _calculate_liquidity_risk_premium(self, volume_liquidity: pd.Series, turnover_liquidity: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.4 · 枯竭惩罚版】计算流动性风险溢价
        核心理念：量能枯竭时，风险溢价飙升。
        返回：溢价程度 [0, 1]，1表示极度危险（枯竭），0表示安全（充裕）。
        """
        # 1. 综合量能
        combined_vol = (volume_liquidity + turnover_liquidity) / 2
        # 2. 风险模型
        # 流动性越低，溢价越高 (反比)
        # 使用指数衰减模拟：流动性低于 0.3 时风险急剧上升
        risk_premium = np.exp(-5 * combined_vol) # vol=0.2 -> exp(-1)=0.36; vol=0 -> 1.0
        return risk_premium.clip(0, 1)

    def _optimized_synthesis(self, components: Dict[str, pd.Series], weights: Dict[str, float], df_index: pd.Index) -> pd.Series:
        """
        【V4.4 · 短板效应版】最优化合成
        核心理念：流动性取决于最差的一环。使用惩罚型加权。
        """
        # 1. 基础加权平均
        weighted_sum = pd.Series(0.0, index=df_index)
        total_weight = sum(weights.values())
        min_val = pd.Series(1.0, index=df_index)
        for key, val in components.items():
            w = weights.get(key, 0)
            weighted_sum += val * w
            
            # 记录短板 (排除反向指标，如风险溢价)
            if key not in ['risk_premium', 'impact_cost']: # 这些指标已在外部处理为"越好分越高"
                 min_val = np.minimum(min_val, val)
        avg_score = weighted_sum / total_weight
        # 2. 短板惩罚
        # 如果任一核心指标过低 (<0.3)，则大幅拉低总分
        penalty_factor = min_val.map(lambda x: 1.0 if x > 0.3 else 0.5 + x/0.6)
        final_score = avg_score * penalty_factor
        return final_score.clip(0, 1)

    def _calculate_liquidity_modulator(self, market_volatility: pd.Series, config: Dict) -> pd.Series:
        """
        【V4.4 · 潮汐调节版】计算流动性调节器
        核心理念：
        - 流动性适中(稳健)：不调节 (1.0)
        - 流动性枯竭(危险)：抑制开仓 (0.7)
        - 流动性泛滥(疯狂)：适度抑制防骗线 (0.9)
        """
        # 注意：这里的输入参数名为 market_volatility，但实际业务逻辑依赖的是综合流动性代理值。
        # 在主流程中，enhanced_liquidity_proxy 计算完成后会作为信号输出。
        # 此处我们假设输入的 volatility 实际上是作为流动性环境的一个参考，
        # 或者我们需要基于内部计算的 liquidity_proxy 来生成 modulator。
        # 由于此方法是在构建 proxy 内部调用的，我们还没有 final proxy。
        # 我们可以使用 components 的均值作为临时参考。
        # 简化逻辑：利用输入的 market_volatility (波动率) 作为参考
        # 波动率过大通常意味着流动性消耗过快 -> 降低权重
        # 波动率过小意味着死水一潭 -> 降低权重
        modulator = pd.Series(1.0, index=market_volatility.index)
        # 低波动 (死水)
        modulator = modulator.mask(market_volatility < 0.2, 0.8)
        # 高波动 (失控)
        modulator = modulator.mask(market_volatility > 0.8, 0.85)
        # 适中波动 (健康)
        mask_healthy = (market_volatility >= 0.2) & (market_volatility <= 0.8)
        modulator = modulator.mask(mask_healthy, 1.1) # 适度奖励
        return modulator.rolling(window=5).mean().fillna(1.0)
