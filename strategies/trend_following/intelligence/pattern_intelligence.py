# 文件: strategies/trend_following/intelligence/pattern_intelligence.py

import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class PatternIntelligence:
    """
    【V3.1 · 语境修正版】形态智能引擎
    - 核心升级: 引入了“下跌动能衰竭”作为第四个识别维度，赋予引擎在明确反转信号出现前，
                  就能识别潜在底部形态的“预测”能力。
    - 本次修改: 为“上涨动能衰竭”信号引入语境判断，修复其在趋势起点被错误触发的致命BUG。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 · 四维聚变版】形态分析总指挥
        - 核心升级: 废弃旧的“四模式或逻辑”，全面升级为基于“背离、反转、突破、跃迁”的四维动态融合模型。
                      利用 SLOPE 和 ACCEL 指标，实现对形态更深层次的动态评估。
        """
        # 整个方法被重写以实现四维聚变逻辑
        p = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p.get('enabled'), True):
            return {}
        weights = get_param_value(p.get('fusion_weights'), {})
        norm_window = 60
        # --- 准备基础数据 ---
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        macd_hist = df.get('MACDh_13_34_8_D', pd.Series(0, index=df.index))
        price_slope = df.get('SLOPE_5_close_D', pd.Series(0, index=df.index))
        rsi_slope = df.get('SLOPE_5_RSI_13_D', pd.Series(0, index=df.index))
        macd_slope = df.get('SLOPE_5_MACDh_13_34_8_D', pd.Series(0, index=df.index))
        rsi_accel = df.get('ACCEL_5_RSI_13_D', pd.Series(0, index=df.index))
        macd_accel = df.get('ACCEL_5_MACDh_13_34_8_D', pd.Series(0, index=df.index))
        # --- 维度一: 经典背离 (Classic Divergence) ---
        bullish_rsi_div = (price_slope < 0) & (rsi_slope > 0)
        bullish_macd_div = (price_slope < 0) & (macd_slope > 0)
        score_bullish_divergence = np.maximum(bullish_rsi_div, bullish_macd_div).astype(float)
        bearish_rsi_div = (price_slope > 0) & (rsi_slope < 0)
        bearish_macd_div = (price_slope > 0) & (macd_slope < 0)
        score_bearish_divergence = np.maximum(bearish_rsi_div, bearish_macd_div).astype(float)
        # --- 维度二: 动能反转 (Momentum Reversal) ---
        bullish_reversal_accel = (normalize_score(rsi_accel.clip(lower=0), df.index, norm_window) * normalize_score(macd_accel.clip(lower=0), df.index, norm_window))**0.5
        bearish_reversal_accel = (normalize_score(rsi_accel.clip(upper=0).abs(), df.index, norm_window) * normalize_score(macd_accel.clip(upper=0).abs(), df.index, norm_window))**0.5
        # --- 维度三: 结构突破 (Structural Breakout) ---
        score_consolidation_breakout = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        score_consolidation_breakdown = (df['close_D'] < df.get('dynamic_consolidation_low_D', -np.inf)).astype(float)
        # --- 维度四: 能量跃迁 (Energy Transition) ---
        bbw_slope = df.get('SLOPE_5_BBW_21_2.0_D', pd.Series(0, index=df.index))
        atr_slope = df.get('SLOPE_5_ATR_14_D', pd.Series(0, index=df.index))
        energy_expansion_score = (normalize_score(bbw_slope.clip(lower=0), df.index, norm_window) * normalize_score(atr_slope.clip(lower=0), df.index, norm_window))**0.5
        # --- 融合看涨形态 ---
        bullish_pillars = {
            'divergence': score_bullish_divergence,
            'momentum_reversal': bullish_reversal_accel,
            'structural_breakout': score_consolidation_breakout,
            'energy_transition': energy_expansion_score
        }
        valid_bull_scores = [s.values for name, s in bullish_pillars.items() if weights.get(name, 0) > 0]
        valid_bull_weights = [weights.get(name) for name in bullish_pillars if weights.get(name, 0) > 0]
        if valid_bull_scores:
            bull_weights_array = np.array(valid_bull_weights) / sum(valid_bull_weights)
            bottom_pattern_score = np.prod(np.stack(valid_bull_scores, axis=0) ** bull_weights_array[:, np.newaxis], axis=0)
        else:
            bottom_pattern_score = np.zeros(len(df.index))
        # --- 融合看跌形态 ---
        bearish_pillars = {
            'divergence': score_bearish_divergence,
            'momentum_reversal': bearish_reversal_accel,
            'structural_breakout': score_consolidation_breakdown,
            'energy_transition': energy_expansion_score
        }
        valid_bear_scores = [s.values for name, s in bearish_pillars.items() if weights.get(name, 0) > 0]
        valid_bear_weights = [weights.get(name) for name in bearish_pillars if weights.get(name, 0) > 0]
        if valid_bear_scores:
            bear_weights_array = np.array(valid_bear_weights) / sum(valid_bear_weights)
            top_pattern_score = np.prod(np.stack(valid_bear_scores, axis=0) ** bear_weights_array[:, np.newaxis], axis=0)
        else:
            top_pattern_score = np.zeros(len(df.index))
        # --- 延续形态 (逻辑优化) ---
        p_cont = get_params_block(self.strategy, 'pattern_params.continuation_params', {})
        rsi_cont_thresh = get_param_value(p_cont.get('rsi_threshold'), 55)
        adx_cont_thresh = get_param_value(p_cont.get('adx_threshold'), 20)
        adx = df.get('ADX_14_D', pd.Series(20, index=df.index))
        is_trending = adx > adx_cont_thresh
        bullish_pattern_score = (rsi > rsi_cont_thresh) & is_trending
        bearish_pattern_score = (rsi < (100 - rsi_cont_thresh)) & is_trending
        states = {
            'SCORE_PATTERN_BOTTOM_REVERSAL': pd.Series(bottom_pattern_score, index=df.index).astype(np.float32),
            'SCORE_PATTERN_BULLISH_RESONANCE': bullish_pattern_score.astype(np.float32),
            'SCORE_PATTERN_TOP_REVERSAL': pd.Series(top_pattern_score, index=df.index).astype(np.float32),
            'SCORE_PATTERN_BEARISH_RESONANCE': bearish_pattern_score.astype(np.float32),
        }
        return states











