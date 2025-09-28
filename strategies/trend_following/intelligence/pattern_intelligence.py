# 文件: strategies/trend_following/intelligence/pattern_intelligence.py

import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class PatternIntelligence:
    """
    【V2.1 · 预测能力版】形态智能引擎
    - 核心升级: 引入了“下跌动能衰竭”作为第四个识别维度，赋予引擎在明确反转信号出现前，
                  就能识别潜在底部形态的“预测”能力。
    - 收益: 极大提升了在阴跌末期行情中的信号灵敏度。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.2 · 信号净化版】形态分析总指挥
        - 核心修复: 净化了所有输出信号的名称，移除了 '_S' 后缀，以完全对齐信号字典。
        """
        # print("      -> 正在运行 [形态智能引擎 V2.2 · 信号净化版]...") # 更新版本号
        p = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p.get('enabled'), True):
            return {}
        
        # --- 底部反转形态识别 (四位一体，增加预测能力) ---
        
        # 模式一: RSI从超卖区反转 (经典V反)
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        was_oversold = (rsi.rolling(window=5, min_periods=1).min() < 35)
        is_recovering = (df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)) > 0)
        score_rsi_reversal = (was_oversold & is_recovering).astype(float)
        
        # 模式二: 突破动态盘整平台 (箱体突破)
        is_breaking_consolidation = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        score_consolidation_breakout = is_breaking_consolidation * 0.8
        
        # 模式三: MACD柱状线金叉 (趋势扭转)
        macd_hist = df.get('MACDh_13_34_8_D', pd.Series(0, index=df.index))
        is_macd_bull_cross = ((macd_hist > 0) & (macd_hist.shift(1) <= 0)).astype(float)
        score_macd_bullish_cross = is_macd_bull_cross

        # 模式四: 下跌动能衰竭 (预测性指标)
        rsi_slope_abs = df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)).abs()
        macd_hist_slope_abs = df.get('SLOPE_1_MACDh_13_34_8_D', pd.Series(0, index=df.index)).abs()
        # 归一化，值越小分数越高 (ascending=False)
        rsi_exhaustion_score = normalize_score(rsi_slope_abs, df.index, window=60, ascending=False)
        macd_exhaustion_score = normalize_score(macd_hist_slope_abs, df.index, window=60, ascending=False)
        score_momentum_exhaustion = (rsi_exhaustion_score * macd_exhaustion_score)**0.5
        
        # 融合四种模式: 只要有一种模式触发，就认为形态成立
        bottom_pattern_score = np.maximum.reduce([
            score_rsi_reversal.values, 
            score_consolidation_breakout.values, 
            score_macd_bullish_cross.values,
            score_momentum_exhaustion.values # 加入新的预测性分数
        ])
        bottom_pattern_score = pd.Series(bottom_pattern_score, index=df.index)

        # 看涨共振形态逻辑保持不变
        bullish_pattern_score = (rsi > 50).astype(float) * normalize_score(df.get('ADX_14_D', pd.Series(20, index=df.index)), df.index, 120)

        states = {
            
            'SCORE_PATTERN_BOTTOM_REVERSAL': bottom_pattern_score.astype(np.float32),
            
            'SCORE_PATTERN_BULLISH_RESONANCE': bullish_pattern_score.astype(np.float32),
        }
        
        
        states['SCORE_PATTERN_TOP_REVERSAL'] = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        states['SCORE_PATTERN_BEARISH_RESONANCE'] = pd.Series(0.0, index=df.index, dtype=np.float32)

        return states


















