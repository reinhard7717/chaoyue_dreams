# 文件: strategies/realtime_modules/intraday_micro_price_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict

class IntradayMicroPriceAnalyzer:
    """
    价格行为微观结构分析器。
    分析影线长度、实体大小、价格拒绝和K线收盘价位置，并返回量化值。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.min_shadow_body_ratio = config.get('min_shadow_body_ratio', 0.8)
        self.rejection_threshold_pct = config.get('rejection_threshold_pct', 0.003)
        print("IntradayMicroPriceAnalyzer initialized.")
    def analyze_micro_price_action(self, kline: pd.Series, prev_kline: pd.Series) -> Dict[str, float]: # 修改返回类型为 float
        """
        分析当前K线的微观价格行为，并返回量化值。
        Args:
            kline (pd.Series): 当前K线数据。
            prev_kline (pd.Series): 前一根K线数据。
        Returns:
            Dict[str, float]: 识别到的微观价格行为特征及其量化值。
        """
        if not self.enabled or kline.empty:
            return {}
        micro_features = {}
        open_price = kline['open']
        high_price = kline['high']
        low_price = kline['low']
        close_price = kline['close']
        if any(pd.isna([open_price, high_price, low_price, close_price])):
            return {}
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        if total_range == 0:
            micro_features["SMALL_BODY_CANDLE"] = 1.0
            micro_features["LONG_UPPER_SHADOW"] = 0.0
            micro_features["LONG_LOWER_SHADOW"] = 0.0
            micro_features["PRICE_REJECTION_UPPER"] = 0.0
            micro_features["PRICE_REJECTION_LOWER"] = 0.0
            micro_features["CLOSE_NEAR_HIGH"] = 0.0
            micro_features["CLOSE_NEAR_LOW"] = 0.0
            micro_features["UPPER_SHADOW_RATIO"] = 0.0
            micro_features["LOWER_SHADOW_RATIO"] = 0.0
            micro_features["CLOSE_POSITION_IN_CANDLE_RATIO"] = 0.5
            return micro_features
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        # 影线长度与总K线范围的比率 (量化值)
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        micro_features["UPPER_SHADOW_RATIO"] = upper_shadow_ratio
        micro_features["LOWER_SHADOW_RATIO"] = lower_shadow_ratio
        micro_features["SMALL_BODY_CANDLE"] = 1.0 if body_size / total_range < (1 - self.min_shadow_body_ratio) else 0.0
        micro_features["LONG_UPPER_SHADOW"] = 1.0 if upper_shadow_ratio > self.min_shadow_body_ratio else 0.0
        micro_features["LONG_LOWER_SHADOW"] = 1.0 if lower_shadow_ratio > self.min_shadow_body_ratio else 0.0
        # 价格拒绝 (Price Rejection)
        micro_features["PRICE_REJECTION_UPPER"] = 1.0 if (upper_shadow_ratio > self.min_shadow_body_ratio) and \
                                                  (close_price < high_price * (1 - self.rejection_threshold_pct)) else 0.0
        micro_features["PRICE_REJECTION_LOWER"] = 1.0 if (lower_shadow_ratio > self.min_shadow_body_ratio) and \
                                                  (close_price > low_price * (1 + self.rejection_threshold_pct)) else 0.0
        # K线收盘价在K线实体中的位置 (0-1之间，0最低，1最高)
        # 对于阳线：(close - open) / body_size
        # 对于阴线：(open - close) / body_size
        # 更通用的：(close - low) / total_range
        close_position_in_candle_ratio = (close_price - low_price) / total_range
        micro_features["CLOSE_POSITION_IN_CANDLE_RATIO"] = close_position_in_candle_ratio
        micro_features["CLOSE_NEAR_HIGH"] = 1.0 if close_position_in_candle_ratio > 0.9 else 0.0
        micro_features["CLOSE_NEAR_LOW"] = 1.0 if close_position_in_candle_ratio < 0.1 else 0.0
        return micro_features

