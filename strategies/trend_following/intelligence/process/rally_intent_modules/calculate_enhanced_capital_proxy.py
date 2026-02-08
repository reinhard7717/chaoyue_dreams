# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_enhanced_capital_proxy.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple



class EnhancedCapitalProxyCalculator:
    def __init__(self, config):
        self.config = config

    def _calculate_multi_level_capital_flow(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index, config: Dict) -> pd.Series:
        """
        【V4.2 · 资金分层合成版】计算多级资金综合流向
        核心理念：给予"聪明钱"(特大单)更高权重，"情绪钱"(小单)负权重。
        假设 normalized_signals 中已包含各分层资金的归一化数据(如 buy_elg_amount_norm)。
        若无直接数据，则降级使用 net_amount_ratio_norm。
        """
        # 尝试获取分层数据
        elg = normalized_signals.get('buy_elg_amount_norm', pd.Series(0.0, index=df_index)) - \
              normalized_signals.get('sell_elg_amount_norm', pd.Series(0.0, index=df_index))
        lg = normalized_signals.get('buy_lg_amount_norm', pd.Series(0.0, index=df_index)) - \
             normalized_signals.get('sell_lg_amount_norm', pd.Series(0.0, index=df_index))
        # 如果没有分层数据，使用整体净流入替代
        if elg.sum() == 0 and lg.sum() == 0:
             return normalized_signals.get('net_amount_ratio_norm', pd.Series(0.5, index=df_index)).clip(0, 1)
        # 权重配置：特大单最重要，大单次之，小单反向
        # 这里的 norm 数据通常在 [0, 1] 之间，0.5 为中性
        # 将其映射回 [-0.5, 0.5] 进行加权
        elg_flow = (elg) * 1.5  # 放大特大单影响
        lg_flow = (lg) * 1.0
        # 合成资金流 (-1 到 1)
        composite_flow = (elg_flow * 0.6 + lg_flow * 0.4)
        # 映射回 [0, 1]
        return np.tanh(composite_flow).clip(-1, 1) * 0.5 + 0.5

    def _calculate_capital_efficiency_factor(self, capital_flow: pd.Series, price_trend: pd.Series, volume_ratio: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.2 · 资金效能版】计算资本效率因子 (CEF)
        核心理念：单位资金推动的价格涨幅。
        高效 = 资金流入少但涨幅大 (锁筹)
        低效 = 资金流入大但涨幅小 (滞涨/对倒)
        """
        # 将 [0,1] 映射回 [-1, 1] 以处理方向
        flow_signed = (capital_flow - 0.5) * 2
        trend_signed = (price_trend - 0.5) * 2
        # 1. 方向一致性 (Directional Consistency)
        # 同向为正，反向为负
        consistency = np.sign(flow_signed) * np.sign(trend_signed)
        # 2. 推动效率 (Thrust Efficiency)
        # 效率 = 价格变动 / (资金投入 + 摩擦成本)
        # 加上 volume_ratio 作为分母惩罚：放量滞涨是低效的典型
        efficiency = trend_signed.abs() / (flow_signed.abs() + volume_ratio * 0.5 + 1e-9)
        # 3. 综合评分
        # 只有在方向一致(资金做多价格涨)时，高效率才有意义
        cef_score = 0.5 + (consistency * 0.3) + (consistency * np.tanh(efficiency) * 0.2)
        return cef_score.clip(0, 1)

    def _calculate_fund_structure_index(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.2 · 主力控盘度版】计算资金结构指数
        核心理念：主力资金占比越高，结构越紧凑。
        """
        # 使用流向强度作为结构的代理变量
        # 强流向通常意味着主力在主导
        intensity = normalized_signals.get('flow_intensity_norm', pd.Series(0.5, index=df_index))
        # 结合筹码集中度 (如果资金在流入且筹码在集中，说明结构极好)
        concentration = normalized_signals.get('chip_concentration_ratio_norm', pd.Series(0.5, index=df_index))
        # 结构指数
        structure_index = (intensity * 0.6 + concentration * 0.4)
        return structure_index.clip(0, 1)

    def _calculate_fund_persistence_index(self, capital_flow: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.2 · 持续性检测版】计算资金持续性指数
        核心理念：检测资金流的自相关性。
        """
        persistence = pd.Series(0.5, index=df_index)
        window = 10
        # 向量化计算自相关性较为困难，使用滚动窗口简化逻辑
        # 计算近期资金流方向的连贯性
        flow_dir = np.sign(capital_flow - 0.5)
        # 滚动求和：如果是持续流入，和会很大
        rolling_sum = flow_dir.rolling(window=window).sum()
        # 归一化：最大值为 window，最小值为 -window
        consistency = rolling_sum / window # [-1, 1]
        # 映射到 [0, 1]，0.5 为中性
        persistence = consistency * 0.5 + 0.5
        return persistence.fillna(0.5).clip(0, 1)

    def _calculate_fund_lead_lag_ratio(self, capital_flow: pd.Series, price_trend: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.2 · 时空错位版】计算资金领先滞后比
        核心理念：资金是否先于价格启动？
        逻辑：计算 Flow(t-1) 与 Price(t) 的相关性。
        """
        # 简单实现：比较资金趋势与价格趋势的时间差
        # 资金趋势
        flow_ma = capital_flow.rolling(window=5).mean()
        price_ma = price_trend.rolling(window=5).mean()
        # 资金斜率
        flow_slope = flow_ma.diff(3)
        price_slope = price_ma.diff(3)
        # 如果资金斜率 > 0 且 价格斜率 <= 0 (资金涨价格未涨) -> 领先 (潜伏)
        # 如果资金斜率 > 0 且 价格斜率 > 0 (同步) -> 同步
        # 如果资金斜率 < 0 且 价格斜率 > 0 (背离) -> 滞后/出货
        score = pd.Series(0.5, index=df_index)
        # 领先情况
        lead_cond = (flow_slope > 0.05) & (price_slope <= 0.02)
        score[lead_cond] = 0.8
        # 同步情况
        sync_cond = (flow_slope > 0.05) & (price_slope > 0.02)
        score[sync_cond] = 0.6
        # 滞后/背离情况
        lag_cond = (flow_slope < -0.05) & (price_slope > 0.02)
        score[lag_cond] = 0.3
        return score

    def _detect_fund_anomalies(self, capital_flow: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.2 · 老鼠仓检测版】检测资金异常
        核心理念：Z-Score 异常检测。
        返回：异常程度 [0, 1]，值越大越异常。
        """
        # 计算 Z-Score
        rolling_mean = capital_flow.rolling(window=21).mean()
        rolling_std = capital_flow.rolling(window=21).std()
        z_score = ((capital_flow - rolling_mean) / (rolling_std + 1e-9)).abs()
        # 映射：Z > 2.5 开始视为异常
        anomaly_score = (z_score - 2.0).clip(lower=0)
        anomaly_score = np.tanh(anomaly_score) # 映射到 [0, 1]
        return anomaly_score.fillna(0)

    def _assess_market_liquidity_state(self, normalized_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.2 · 水位评估版】评估市场流动性状态
        返回：0(枯竭) -> 1(亢奋)
        """
        turnover = normalized_signals.get('turnover_rate_norm', pd.Series(0.5, index=df_index))
        vol_ratio = normalized_signals.get('volume_ratio_norm', pd.Series(0.5, index=df_index))
        # 综合流动性
        liquidity = turnover * 0.6 + vol_ratio * 0.4
        return liquidity.clip(0, 1)

    def _calculate_capital_modulator(self, liquidity_state: pd.Series, config: Dict) -> pd.Series:
        """
        【V4.2 · 动态信度版】计算资本调节器
        逻辑：
        - 流动性适中 (0.4-0.7): 资金信号最可信 -> 系数 1.2
        - 流动性枯竭 (<0.2): 资金容易骗线(少量即可拉升) -> 系数 0.7
        - 流动性亢奋 (>0.8): 资金分歧大，噪音多 -> 系数 0.9
        """
        modulator = pd.Series(1.0, index=liquidity_state.index)
        # 枯竭区
        modulator = modulator.mask(liquidity_state < 0.2, 0.7)
        # 适中区 (主力最喜欢的环境)
        mask_optimal = (liquidity_state >= 0.4) & (liquidity_state <= 0.7)
        modulator = modulator.mask(mask_optimal, 1.2)
        # 亢奋区
        modulator = modulator.mask(liquidity_state > 0.8, 0.9)
        return modulator.rolling(window=3).mean().fillna(1.0)













