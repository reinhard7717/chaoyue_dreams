# 文件: strategies/realtime_modules/intraday_pattern_recognizer.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Optional

class IntradayPatternRecognizer:
    """
    盘中K线形态识别器。
    使用 pandas_ta 识别常见的K线形态，并量化其强度。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.patterns_to_check = config.get('patterns', [])
        self.hammer_strength_params = config.get('hammer_strength_params', {}) # 加载锤头线强度参数
        print("IntradayPatternRecognizer initialized.")
    def recognize_patterns(self, df: pd.DataFrame, timeframe: str) -> Dict[str, float]: # 修改返回类型为 float
        """
        识别给定DataFrame中最新K线的K线形态，并返回量化强度。
        Args:
            df (pd.DataFrame): 包含OHLCV数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min')。
        Returns:
            Dict[str, float]: 识别到的形态及其量化强度 (0-1之间，或布尔值)。
        """
        if not self.enabled or timeframe not in self.apply_on or df.empty:
            return {}
        patterns_found = {}
        if len(df) < 2:
            return {}
        current_kline = df.iloc[-1]
        open_price = current_kline['open']
        high_price = current_kline['high']
        low_price = current_kline['low']
        close_price = current_kline['close']
        # 确保价格数据有效
        if any(pd.isna([open_price, high_price, low_price, close_price])):
            return {}
        # 锤头线 (Hammer)
        if "CDL_HAMMER" in self.patterns_to_check:
            hammer = ta.cdl_hammer(df['open'], df['high'], df['low'], df['close'])
            if hammer is not None and not hammer.empty and hammer.iloc[-1] == 100:
                patterns_found["CDL_HAMMER"] = 1.0 # 标记为True
                # 计算锤头线强度
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                lower_shadow = min(open_price, close_price) - low_price
                upper_shadow = high_price - max(open_price, close_price)
                if total_range > 0 and body_size > 0:
                    min_body_ratio = self.hammer_strength_params.get('min_body_ratio', 0.1)
                    max_upper_shadow_ratio = self.hammer_strength_params.get('max_upper_shadow_ratio', 0.2)
                    min_lower_shadow_ratio = self.hammer_strength_params.get('min_lower_shadow_ratio', 0.6)
                    # 锤头线强度评分逻辑 (0-1之间)
                    # 实体越小越好 (反向分数)
                    body_score = max(0, 1 - (body_size / total_range) / min_body_ratio) if (body_size / total_range) > min_body_ratio else 1
                    # 上影线越短越好 (反向分数)
                    upper_shadow_score = max(0, 1 - (upper_shadow / total_range) / max_upper_shadow_ratio) if (upper_shadow / total_range) > max_upper_shadow_ratio else 1
                    # 下影线越长越好 (正向分数)
                    lower_shadow_score = min(1, (lower_shadow / total_range) / min_lower_shadow_ratio)
                    # 综合强度，可以加权平均
                    hammer_strength = (body_score * 0.3 + upper_shadow_score * 0.3 + lower_shadow_score * 0.4)
                    patterns_found["HAMMER_STRENGTH_SCORE"] = hammer_strength
                else:
                    patterns_found["HAMMER_STRENGTH_SCORE"] = 0.0
            else:
                patterns_found["CDL_HAMMER"] = 0.0
                patterns_found["HAMMER_STRENGTH_SCORE"] = 0.0
        # 吞没形态 (Engulfing)
        if "CDL_ENGULFING" in self.patterns_to_check:
            engulfing = ta.cdl_engulfing(df['open'], df['high'], df['low'], df['close'])
            if engulfing is not None and not engulfing.empty and engulfing.iloc[-1] != 0:
                patterns_found["CDL_ENGULFING"] = 1.0
                patterns_found["CDL_ENGULFING_BULLISH"] = 1.0 if (engulfing.iloc[-1] == 100) else 0.0
                patterns_found["CDL_ENGULFING_BEARISH"] = 1.0 if (engulfing.iloc[-1] == -100) else 0.0
            else:
                patterns_found["CDL_ENGULFING"] = 0.0
                patterns_found["CDL_ENGULFING_BULLISH"] = 0.0
                patterns_found["CDL_ENGULFING_BEARISH"] = 0.0
        # 十字星 (Doji)
        if "CDL_DOJI" in self.patterns_to_check:
            doji = ta.cdl_doji(df['open'], df['high'], df['low'], df['close'])
            if doji is not None and not doji.empty and doji.iloc[-1] != 0:
                patterns_found["CDL_DOJI"] = 1.0
            else:
                patterns_found["CDL_DOJI"] = 0.0
        # 启明星/黄昏星 (Star)
        if "CDL_STAR" in self.patterns_to_check:
            morning_star = ta.cdl_morningstar(df['open'], df['high'], df['low'], df['close'])
            evening_star = ta.cdl_leaveningstar(df['open'], df['high'], df['low'], df['close'])
            patterns_found["CDL_MORNINGSTAR"] = 1.0 if (morning_star is not None and not morning_star.empty and morning_star.iloc[-1] == 100) else 0.0
            patterns_found["CDL_EVENINGSTAR"] = 1.0 if (evening_star is not None and not evening_star.empty and evening_star.iloc[-1] == -100) else 0.0
            patterns_found["CDL_STAR"] = max(patterns_found["CDL_MORNINGSTAR"], patterns_found["CDL_EVENINGSTAR"])
        return patterns_found

