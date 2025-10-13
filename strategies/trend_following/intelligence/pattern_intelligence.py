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
        【V3.1 · 语境修正版】形态分析总指挥
        - 核心升级: 补完缺失的“四位一体”看跌形态识别逻辑，使引擎具备完整的风险诊断能力。
        - 本次修改: 为“上涨动能衰竭”信号引入RSI高位语境，防止在底部反转初期误报风险。
        """
        p = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p.get('enabled'), True):
            return {}
        
        # --- 看涨形态识别 (四位一体) ---
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        macd_hist = df.get('MACDh_13_34_8_D', pd.Series(0, index=df.index))
        
        # 模式一: RSI从超卖区反转 (经典V反)
        was_oversold = (rsi.rolling(window=5, min_periods=1).min() < 35)
        is_recovering = (df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)) > 0)
        score_rsi_reversal = (was_oversold & is_recovering).astype(float)
        
        # 模式二: 突破动态盘整平台 (箱体突破)
        is_breaking_consolidation = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        score_consolidation_breakout = is_breaking_consolidation * 0.8
        
        # 模式三: MACD柱状线金叉 (趋势扭转)
        is_macd_bull_cross = ((macd_hist > 0) & (macd_hist.shift(1) <= 0)).astype(float)
        score_macd_bullish_cross = is_macd_bull_cross

        # 模式四: 下跌动能衰竭 (预测性指标)
        rsi_slope_abs = df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)).abs()
        macd_hist_slope_abs = df.get('SLOPE_1_MACDh_13_34_8_D', pd.Series(0, index=df.index)).abs()
        rsi_exhaustion_score = normalize_score(rsi_slope_abs, df.index, window=60, ascending=False)
        macd_exhaustion_score = normalize_score(macd_hist_slope_abs, df.index, window=60, ascending=False)
        score_momentum_exhaustion = (rsi_exhaustion_score * macd_exhaustion_score)**0.5
        
        # 融合四种看涨模式
        bottom_pattern_score = np.maximum.reduce([
            score_rsi_reversal.values, 
            score_consolidation_breakout.values, 
            score_macd_bullish_cross.values,
            score_momentum_exhaustion.values
        ])
        bottom_pattern_score = pd.Series(bottom_pattern_score, index=df.index)

        # 看涨共振形态
        bullish_pattern_score = (rsi > 50).astype(float) * normalize_score(df.get('ADX_14_D', pd.Series(20, index=df.index)), df.index, 120)

        # --- 看跌形态识别 (四位一体) ---
        # 模式一: RSI从超买区回落 (顶部反转)
        was_overbought = (rsi.rolling(window=5, min_periods=1).max() > 70)
        is_falling = (df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)) < 0)
        score_rsi_top_reversal = (was_overbought & is_falling).astype(float)

        # 模式二: 跌破动态盘整平台 (箱体破位)
        is_breaking_down = (df['close_D'] < df.get('dynamic_consolidation_low_D', -np.inf)).astype(float)
        score_consolidation_breakdown = is_breaking_down * 0.8

        # 模式三: MACD柱状线死叉 (趋势扭转)
        is_macd_bear_cross = ((macd_hist < 0) & (macd_hist.shift(1) >= 0)).astype(float)
        score_macd_bearish_cross = is_macd_bear_cross

        # 新增开始: 为“上涨动能衰竭”引入语境判断
        # 只有在RSI已经处于高位（例如大于60）时，动能的衰竭才被视为一种风险
        is_uptrend_context = (rsi > 60).astype(float)
        # 修改行: 引入语境调节器，防止在底部误判
        score_up_momentum_exhaustion = score_momentum_exhaustion * is_uptrend_context
        # 新增结束

        # 融合四种看跌模式
        top_pattern_score = np.maximum.reduce([
            score_rsi_top_reversal.values,
            score_consolidation_breakdown.values,
            score_macd_bearish_cross.values,
            score_up_momentum_exhaustion.values
        ])
        top_pattern_score = pd.Series(top_pattern_score, index=df.index)

        # 看跌共振形态
        bearish_pattern_score = (rsi < 50).astype(float) * normalize_score(df.get('ADX_14_D', pd.Series(20, index=df.index)), df.index, 120)

        states = {
            'SCORE_PATTERN_BOTTOM_REVERSAL': bottom_pattern_score.astype(np.float32),
            'SCORE_PATTERN_BULLISH_RESONANCE': bullish_pattern_score.astype(np.float32),
            'SCORE_PATTERN_TOP_REVERSAL': top_pattern_score.astype(np.float32),
            'SCORE_PATTERN_BEARISH_RESONANCE': bearish_pattern_score.astype(np.float32),
        }
        return states
