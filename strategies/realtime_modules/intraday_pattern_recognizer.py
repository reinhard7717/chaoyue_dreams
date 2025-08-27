# 文件: strategies/realtime_modules/intraday_pattern_recognizer.py

import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional

class IntradayPatternRecognizer:
    """
    盘中K线形态识别器。
    使用 pandas_ta 识别常见的K线形态。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.patterns_to_check = config.get('patterns', [])
        print("IntradayPatternRecognizer initialized.")

    def recognize_patterns(self, df: pd.DataFrame, timeframe: str) -> Dict[str, bool]:
        """
        识别给定DataFrame中最新K线的K线形态。
        Args:
            df (pd.DataFrame): 包含OHLCV数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min')。
        Returns:
            Dict[str, bool]: 识别到的形态及其布尔值。
        """
        if not self.enabled or timeframe not in self.apply_on or df.empty:
            return {}

        patterns_found = {}
        # pandas_ta的K线形态函数通常返回一个DataFrame，最后一列是形态识别结果
        # 结果为100表示看涨，-100表示看跌，0表示无形态
        
        # 确保df有足够的历史数据来计算形态
        if len(df) < 2: # 大部分形态至少需要2根K线
            return {}

        # 锤头线 (Hammer)
        if "CDL_HAMMER" in self.patterns_to_check:
            hammer = ta.cdl_hammer(df['open'], df['high'], df['low'], df['close'])
            if hammer is not None and not hammer.empty and hammer.iloc[-1] == 100:
                patterns_found["CDL_HAMMER"] = True
            else:
                patterns_found["CDL_HAMMER"] = False

        # 吞没形态 (Engulfing)
        if "CDL_ENGULFING" in self.patterns_to_check:
            engulfing = ta.cdl_engulfing(df['open'], df['high'], df['low'], df['close'])
            if engulfing is not None and not engulfing.empty and engulfing.iloc[-1] != 0:
                patterns_found["CDL_ENGULFING"] = True
                patterns_found["CDL_ENGULFING_BULLISH"] = (engulfing.iloc[-1] == 100)
                patterns_found["CDL_ENGULFING_BEARISH"] = (engulfing.iloc[-1] == -100)
            else:
                patterns_found["CDL_ENGULFING"] = False
                patterns_found["CDL_ENGULFING_BULLISH"] = False
                patterns_found["CDL_ENGULFING_BEARISH"] = False

        # 十字星 (Doji)
        if "CDL_DOJI" in self.patterns_to_check:
            doji = ta.cdl_doji(df['open'], df['high'], df['low'], df['close'])
            if doji is not None and not doji.empty and doji.iloc[-1] != 0:
                patterns_found["CDL_DOJI"] = True
            else:
                patterns_found["CDL_DOJI"] = False

        # 启明星/黄昏星 (Star) - pandas_ta没有直接的CDL_STAR，通常是启明星/黄昏星
        # 这里我们检查启明星 (Morning Star) 和黄昏星 (Evening Star)
        if "CDL_STAR" in self.patterns_to_check:
            morning_star = ta.cdl_morningstar(df['open'], df['high'], df['low'], df['close'])
            evening_star = ta.cdl_leaveningstar(df['open'], df['high'], df['low'], df['close']) # 注意：pandas_ta中是leaveningstar
            if morning_star is not None and not morning_star.empty and morning_star.iloc[-1] == 100:
                patterns_found["CDL_MORNINGSTAR"] = True
            else:
                patterns_found["CDL_MORNINGSTAR"] = False
            if evening_star is not None and not evening_star.empty and evening_star.iloc[-1] == -100:
                patterns_found["CDL_EVENINGSTAR"] = True
            else:
                patterns_found["CDL_EVENINGSTAR"] = False
            patterns_found["CDL_STAR"] = patterns_found["CDL_MORNINGSTAR"] or patterns_found["CDL_EVENINGSTAR"]
        
        return patterns_found

