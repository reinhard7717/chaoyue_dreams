# 文件: strategies/trend_following/intelligence/pattern_intelligence.py

import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class PatternIntelligence:
    """
    【V5.0 · 四象限重构版】形态智能引擎
    - 核心重构: 废弃旧的“四维聚变”模型，全面升级为与其他情报模块统一的“四象限动态分析法”。
    - 收益: 实现了信号逻辑的清晰、统一和可解释性，彻底解决了信号命名与含义不符的问题。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 · 四象限重构版】形态分析总指挥
        - 核心重构: 废弃旧的“四维聚变”逻辑，全面升级为基于“双极性快照+四象限动态分析”的统一模型。
        """

        states = {}
        p = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p.get('enabled'), True):
            return {}
        weights = get_param_value(p.get('fusion_weights'), {})
        norm_window = 60

        # 步骤一：计算双极性“形态健康度”快照分
        price_slope = df.get('SLOPE_5_close_D', pd.Series(0, index=df.index))
        rsi_slope = df.get('SLOPE_5_RSI_13_D', pd.Series(0, index=df.index))
        macd_slope = df.get('SLOPE_5_MACDh_13_34_8_D', pd.Series(0, index=df.index))
        rsi_accel = df.get('ACCEL_5_RSI_13_D', pd.Series(0, index=df.index))
        macd_accel = df.get('ACCEL_5_MACDh_13_34_8_D', pd.Series(0, index=df.index))
        bbw_slope = df.get('SLOPE_5_BBW_21_2.0_D', pd.Series(0, index=df.index))
        
        # 看涨证据
        bullish_rsi_div = (price_slope < 0) & (rsi_slope > 0)
        bullish_macd_div = (price_slope < 0) & (macd_slope > 0)
        bullish_divergence = np.maximum(bullish_rsi_div, bullish_macd_div).astype(float)
        bullish_reversal_accel = (normalize_score(rsi_accel.clip(lower=0), df.index, norm_window) * normalize_score(macd_accel.clip(lower=0), df.index, norm_window))**0.5
        bullish_breakout = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        
        # 看跌证据
        bearish_rsi_div = (price_slope > 0) & (rsi_slope < 0)
        bearish_macd_div = (price_slope > 0) & (macd_slope < 0)
        bearish_divergence = np.maximum(bearish_rsi_div, bearish_macd_div).astype(float)
        bearish_reversal_accel = (normalize_score(rsi_accel.clip(upper=0).abs(), df.index, norm_window) * normalize_score(macd_accel.clip(upper=0).abs(), df.index, norm_window))**0.5
        bearish_breakdown = (df['close_D'] < df.get('dynamic_consolidation_low_D', -np.inf)).astype(float)
        
        # 能量催化剂 (中性)
        energy_catalyst = normalize_score(bbw_slope.clip(lower=0), df.index, norm_window)
        
        # 融合看涨/看跌快照分
        bullish_snapshot = (
            bullish_divergence * weights.get('divergence', 0.3) +
            bullish_reversal_accel * weights.get('momentum_reversal', 0.4) +
            bullish_breakout * weights.get('structural_breakout', 0.3)
        ) * energy_catalyst
        
        bearish_snapshot = (
            bearish_divergence * weights.get('divergence', 0.3) +
            bearish_reversal_accel * weights.get('momentum_reversal', 0.4) +
            bearish_breakdown * weights.get('structural_breakout', 0.3)
        ) * energy_catalyst
        
        bipolar_snapshot = (bullish_snapshot - bearish_snapshot).clip(-1, 1)
        
        # 步骤二：分离为纯粹的看涨/看跌健康分，并计算静态共振信号
        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_snapshot)
        states['SCORE_PATTERN_BULLISH_RESONANCE'] = bullish_resonance.astype(np.float32)
        states['SCORE_PATTERN_BEARISH_RESONANCE'] = bearish_resonance.astype(np.float32)
        
        # 步骤三：计算四象限动态信号
        bull_divergence = self._calculate_holographic_divergence_pattern(bullish_resonance, 5, 21, norm_window)
        bullish_acceleration = bull_divergence.clip(0, 1)
        top_reversal = (bull_divergence.clip(-1, 0) * -1)
        
        bear_divergence = self._calculate_holographic_divergence_pattern(bearish_resonance, 5, 21, norm_window)
        bearish_acceleration = bear_divergence.clip(0, 1)
        bottom_reversal = (bear_divergence.clip(-1, 0) * -1)
        
        # 步骤四：赋值给命名准确的终极信号
        states['SCORE_PATTERN_BULLISH_ACCELERATION'] = bullish_acceleration.astype(np.float32)
        states['SCORE_PATTERN_TOP_REVERSAL'] = top_reversal.astype(np.float32)
        states['SCORE_PATTERN_BEARISH_ACCELERATION'] = bearish_acceleration.astype(np.float32)
        states['SCORE_PATTERN_BOTTOM_REVERSAL'] = bottom_reversal.astype(np.float32)
        
        # 步骤五：重铸战术反转信号
        states['SCORE_PATTERN_TACTICAL_REVERSAL'] = (bullish_resonance * top_reversal).clip(0, 1).astype(np.float32)
        
        return states
        

    def _calculate_holographic_divergence_pattern(self, series: pd.Series, short_p: int, long_p: int, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】形态层专用的全息背离计算引擎
        - 战略意义: 洞察多时间维度的“结构性背离”，输出一个[-1, 1]的双极性背离分数。
        """
        # [代码新增开始]
        # 维度一：速度背离 (短期斜率 vs 长期斜率)
        slope_short = series.diff(short_p).fillna(0)
        slope_long = series.diff(long_p).fillna(0)
        velocity_divergence = slope_short - slope_long
        velocity_divergence_score = normalize_to_bipolar(velocity_divergence, series.index, norm_window)
        
        # 维度二：加速度背离 (短期加速度 vs 长期加速度)
        accel_short = slope_short.diff(short_p).fillna(0)
        accel_long = slope_long.diff(long_p).fillna(0)
        acceleration_divergence = accel_short - accel_long
        acceleration_divergence_score = normalize_to_bipolar(acceleration_divergence, series.index, norm_window)
        
        # 融合：速度背离和加速度背离的加权平均
        final_divergence_score = (velocity_divergence_score * 0.6 + acceleration_divergence_score * 0.4).clip(-1, 1)
        return final_divergence_score.astype(np.float32)
        # [代码新增结束]


