# 文件: strategies/trend_following/intelligence/pattern_intelligence.py

import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class PatternIntelligence:
    """
    【V2.0 · 多模态识别版】形态智能引擎
    - 核心升级: 彻底重构底部形态识别逻辑，从单一的RSI规则升级为“RSI反转、平台突破、MACD金叉”三位一体的多模态识别框架。
    - 收益: 大幅提升了底部形态识别的覆盖率和准确性。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        形态分析总指挥。
        """
        print("      -> 正在运行 [形态智能引擎 V2.0 · 多模态识别版]...") # 更新版本号和说明
        p = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p.get('enabled'), True):
            return {}
        
        # --- 底部反转形态识别 (三位一体) ---
        
        # 模式一: RSI从超卖区反转 (经典V反)
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        was_oversold = (rsi.rolling(window=5, min_periods=1).min() < 35) # 放宽超卖阈值到35
        is_recovering = (df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)) > 0)
        score_rsi_reversal = (was_oversold & is_recovering).astype(float)
        
        # 模式二: 突破动态盘整平台 (箱体突破)
        # 假设 indicator_service 已经计算并添加了 'dynamic_consolidation_high_D'
        is_breaking_consolidation = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        score_consolidation_breakout = is_breaking_consolidation * 0.8 # 给予0.8的基础分
        
        # 模式三: MACD柱状线金叉 (趋势扭转)
        macd_hist = df.get('MACDh_13_34_8_D', pd.Series(0, index=df.index))
        is_macd_bull_cross = ((macd_hist > 0) & (macd_hist.shift(1) <= 0)).astype(float)
        score_macd_bullish_cross = is_macd_bull_cross
        
        # 融合三种模式: 只要有一种模式触发，就认为形态成立
        bottom_pattern_score = np.maximum.reduce([
            score_rsi_reversal.values, 
            score_consolidation_breakout.values, 
            score_macd_bullish_cross.values
        ])
        bottom_pattern_score = pd.Series(bottom_pattern_score, index=df.index)

        # 看涨共振形态逻辑保持不变
        bullish_pattern_score = (rsi > 50).astype(float) * normalize_score(df.get('ADX_14_D', pd.Series(20, index=df.index)), df.index, 120)

        states = {
            'SCORE_PATTERN_BOTTOM_REVERSAL_S': bottom_pattern_score.astype(np.float32),
            'SCORE_PATTERN_BULLISH_RESONANCE_S': bullish_pattern_score.astype(np.float32),
        }
        
        states['SCORE_PATTERN_TOP_REVERSAL_S'] = pd.Series(0.0, index=df.index, dtype=np.float32)
        states['SCORE_PATTERN_BEARISH_RESONANCE_S'] = pd.Series(0.0, index=df.index, dtype=np.float32)

        return states

















