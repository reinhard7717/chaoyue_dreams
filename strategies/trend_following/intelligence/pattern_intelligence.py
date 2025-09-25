# 文件: strategies/trend_following/intelligence/pattern_intelligence.py

import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class PatternIntelligence:
    """
    【V1.0 · 初始框架版】形态智能引擎
    - 核心职责: 识别经典技术分析形态（如W底、头肩底、杯柄等），并生成相应的'PATTERN_...'信号。
    - 当前实现: 作为一个初始框架，暂时使用代理逻辑（如RSI超卖反弹）来生成分数，为后续实现复杂形态识别算法奠定基础。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        形态分析总指挥。
        """
        print("      -> 正在运行 [形态智能引擎 V1.0]...")
        p = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p.get('enabled'), True):
            return {}
        
        # 在未来，这里将调用各种复杂的形态识别函数
        # 例如: self._diagnose_w_bottom(), self._diagnose_cup_and_handle()
        
        # 【V1.0 代理逻辑】
        # 作为一个初始版本，我们先用一个简单的代理逻辑来生成分数，以打通整个信号链路。
        # 逻辑：当RSI从超卖区（<30）回升时，我们认为可能存在一个底部反转形态。
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        
        # 条件1: RSI曾经进入超卖区
        was_oversold = (rsi.rolling(window=5, min_periods=1).min() < 30)
        # 条件2: 当前RSI正在回升
        is_recovering = (df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)) > 0)
        
        # 综合得分
        bottom_pattern_score = (was_oversold & is_recovering).astype(float)
        
        # 看涨共振形态暂时用一个简单逻辑代理
        bullish_pattern_score = (rsi > 50).astype(float) * normalize_score(df.get('ADX_14_D', pd.Series(20, index=df.index)), df.index, 120)

        states = {
            'SCORE_PATTERN_BOTTOM_REVERSAL_S': bottom_pattern_score.astype(np.float32),
            'SCORE_PATTERN_BULLISH_RESONANCE_S': bullish_pattern_score.astype(np.float32),
        }
        
        # 为未来看跌形态留出接口
        states['SCORE_PATTERN_TOP_REVERSAL_S'] = pd.Series(0.0, index=df.index, dtype=np.float32)
        states['SCORE_PATTERN_BEARISH_RESONANCE_S'] = pd.Series(0.0, index=df.index, dtype=np.float32)

        return states
