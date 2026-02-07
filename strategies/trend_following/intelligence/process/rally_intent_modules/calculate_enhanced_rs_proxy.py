# strategies\trend_following\intelligence\process\calculate_main_force_rally_intent\calculate_enhanced_rs_proxy.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any, Tuple


class EnhancedRSProxyCalculator:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def _calculate_adaptive_rsi_strength(self, base_rsi: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.1 · 波动率加权版】自适应RSI强度
        核心理念：避免Pandas EWM动态span错误，改用"双轨RSI动态加权"。
        逻辑：
        1. 计算短期(Fast=6)和长期(Slow=24)两组RSI。
        2. 根据近期波动率(Volatility)计算权重因子。
        3. 波动大(变盘期) -> 偏向Fast RSI；波动小(盘整期) -> 偏向Slow RSI。
        """
        # 假设 df 中有 close 数据，若没有则需从外部传入，这里简化假设可以直接获取或使用 base_rsi 近似推导
        # 为了严谨，建议在 calculate 方法中传入 close，此处若无 close，仅对 base_rsi 做非线性变换
        # 这里演示标准逻辑：假设 base_rsi 是标准的 RSI_14
        # 模拟波动率调节（利用RSI自身的稳定性）
        # RSI 变化率大 = 波动大
        rsi_vol = base_rsi.diff().abs().rolling(window=10).mean()
        # 归一化波动因子 [0, 1]
        # 假设 RSI 日变动平均 5-10 为高波
        alpha = (rsi_vol / 10.0).clip(0, 1)
        # 模拟 Fast RSI (通过放大 base_rsi 的波动模拟)
        fast_proxy = (base_rsi - 50) * 1.5 + 50
        # 模拟 Slow RSI (通过平滑 base_rsi 模拟)
        slow_proxy = base_rsi.rolling(window=5).mean()
        # 动态合成
        # 波动大时(alpha大)，信赖 fast_proxy；波动小时，信赖 slow_proxy
        adaptive_rsi = fast_proxy * alpha + slow_proxy * (1 - alpha)
        return (adaptive_rsi / 100.0).clip(0, 1)

    def _calculate_price_acceleration_strength(self, price_trend_mtf: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.1 · 动力学版】价格加速度强度
        逻辑：利用MTF趋势信号的一阶差分(速度)和二阶差分(加速度)合成。
        """
        # 1. 速度 (Velocity)
        velocity = price_trend_strength = price_trend_mtf # MTF趋势本身即包含了方向和力度
        # 2. 加速度 (Acceleration)
        # 计算趋势的变化率
        acceleration = price_trend_mtf.diff(3).fillna(0)
        # 3. 归一化与合成
        # 我们希望捕捉：趋势向上(Vel>0) 且 加速向上(Acc>0)
        # 或者：趋势虽弱但强力加速(拐点)
        # 归一化 Acc 到 [-1, 1]
        acc_norm = np.tanh(acceleration * 5)
        # 综合评分：基础趋势占60%，加速度占40%
        # 只有当两者同向为正时，得分最高
        strength = (price_trend_mtf * 0.6 + acc_norm * 0.4).clip(0, 1)
        return strength

    def _calculate_structural_strength(self, normalized_signals: Dict[str, pd.Series], mtf_signals: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.1 · 结构位阶版】计算结构强度
        逻辑：结合'突破质量'和'当前价格在周期内的相对位置'。
        """
        # 1. 突破质量 (Breakout Quality)
        bq_score = normalized_signals.get('breakout_quality_norm', pd.Series(0.5, index=df_index))
        # 2. 趋势确认度 (Trend Confirmation)
        tc_score = normalized_signals.get('trend_confirmation_norm', pd.Series(0.5, index=df_index))
        # 3. 价格相对位置 (利用 MTF 趋势信号推断)
        # MTF趋势强代表价格处于高位
        pos_score = mtf_signals.get('mtf_price_trend', pd.Series(0.5, index=df_index))
        # 4. 结构强度合成
        # 逻辑：在高位(pos高)且突破质量好(bq高) = 强结构
        # 如果在高位但突破质量差 = 假突破(弱结构)
        structural_strength = (pos_score * 0.4 + bq_score * 0.4 + tc_score * 0.2)
        return structural_strength.clip(0, 1)

    def _calculate_volume_relative_strength(self, volume_ratio_norm: pd.Series, price_trend: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.1 · 量价配合版】量能相对强度
        逻辑：价涨量增(正相关) = 强；价涨量缩(背离) = 弱；价跌量增(恐慌) = 弱。
        """
        # 1. 价格趋势方向 (-1 到 1)
        # 假设 price_trend 输入是 [0, 1]，先映射回 [-1, 1] 以判断方向
        price_dir = (price_trend - 0.5) * 2
        # 2. 量能强度 (0 到 1)
        vol_strength = volume_ratio_norm
        # 3. 量价配合度
        # 理想情况：价格向上(>0) 且 量能充足(>0.5)
        # 这是一个"多头量能"评分，所以下跌放量给低分
        # 基础分：量能 * 价格方向
        # Price > 0, Vol High -> High Score (量价齐升)
        # Price > 0, Vol Low  -> Medium Score (缩量上涨，可能是控盘也可能是背离)
        # Price < 0, Vol High -> Low Score (放量下跌)
        base_score = 0.5 + (price_dir * 0.3) + (price_dir * (vol_strength - 0.5) * 0.2)
        return base_score.clip(0, 1)

    def _calculate_capital_flow_relative_strength(self, net_amount_ratio_norm: pd.Series, price_trend: pd.Series, df_index: pd.Index) -> pd.Series:
        """
        【V4.1 · 资金效率版】资金流相对强度
        逻辑：资金流入强度 + 资金推动效率。
        """
        # 1. 资金净流入强度
        flow_strength = net_amount_ratio_norm
        # 2. 资金-价格一致性
        # 资金流入(>0.5) 且 价格上涨(>0.5) -> 共振
        consistency = 1.0 - abs(flow_strength - price_trend) # 差值越小一致性越高
        # 3. 综合强度
        # 资金强度优先，一致性辅助
        strength = flow_strength * 0.7 + consistency * 0.3
        return strength.clip(0, 1)

    def _detect_market_state_for_rs(self, rs_components: Dict[str, pd.Series], df_index: pd.Index) -> pd.Series:
        """
        【V4.1 · 状态机版】检测RS市场状态
        返回状态：
        0: Decline (下跌)
        1: Accumulation (震荡/吸筹)
        2: Acceleration (加速/主升)
        3: Distribution (高位/派发)
        """
        states = pd.Series(1, index=df_index) # 默认震荡
        price = rs_components.get('price_trend', pd.Series(0.5, index=df_index))
        accel = rs_components.get('acceleration', pd.Series(0.5, index=df_index))
        vol = rs_components.get('volume', pd.Series(0.5, index=df_index))
        # 向量化条件判断
        # Acceleration: 价格强 (>0.6) & 加速强 (>0.55) & 量能配合 (>0.5)
        cond_accel = (price > 0.6) & (accel > 0.55) & (vol > 0.5)
        # Decline: 价格弱 (<0.4)
        cond_decline = (price < 0.4)
        # Distribution: 价格高 (>0.7) 但 加速减弱 (<0.4) (滞涨)
        cond_dist = (price > 0.7) & (accel < 0.4)
        # 应用状态 (顺序很重要，优先级：Decline > Dist > Accel > Accumulation)
        states[cond_accel] = 2
        states[cond_dist] = 3
        states[cond_decline] = 0
        return states

    def _calculate_rs_modulator(self, market_state: pd.Series, config: Dict) -> pd.Series:
        """
        【V4.1 · 动态增益版】计算RS调节系数
        逻辑：
        - 加速期 (State 2): 放大RS信号 (x 1.2) -> 追涨
        - 震荡期 (State 1): 保持原状 (x 1.0)
        - 派发期 (State 3): 抑制RS信号 (x 0.8) -> 防坑
        - 下跌期 (State 0): 严厉抑制 (x 0.5) -> 抄底需谨慎
        """
        modulator = pd.Series(1.0, index=market_state.index)
        # 映射状态到系数
        # State 0 (Decline)
        modulator = modulator.mask(market_state == 0, 0.5)
        # State 2 (Acceleration)
        modulator = modulator.mask(market_state == 2, 1.2)
        # State 3 (Distribution)
        modulator = modulator.mask(market_state == 3, 0.8)
        # 平滑处理，避免系数跳变过大
        return modulator.rolling(window=3).mean().fillna(1.0)












