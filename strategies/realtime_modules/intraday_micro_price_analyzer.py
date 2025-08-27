# 文件: strategies/realtime_modules/intraday_micro_price_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict

class IntradayMicroPriceAnalyzer:
    """
    价格行为微观结构分析器。
    分析影线长度、实体大小、价格拒绝和K线收盘价位置。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.min_shadow_body_ratio = config.get('min_shadow_body_ratio', 0.8)
        self.rejection_threshold_pct = config.get('rejection_threshold_pct', 0.003)
        print("IntradayMicroPriceAnalyzer initialized.")

    def analyze_micro_price_action(self, kline: pd.Series, prev_kline: pd.Series) -> Dict[str, bool]:
        """
        分析当前K线的微观价格行为。
        Args:
            kline (pd.Series): 当前K线数据。
            prev_kline (pd.Series): 前一根K线数据。
        Returns:
            Dict[str, bool]: 识别到的微观价格行为特征。
        """
        if not self.enabled or kline.empty:
            return {}

        micro_features = {}
        
        open_price = kline['open']
        high_price = kline['high']
        low_price = kline['low']
        close_price = kline['close']

        if pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
            return {}

        # K线实体大小
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price

        if total_range == 0: # 避免除以零
            micro_features["SMALL_BODY_CANDLE"] = True
            micro_features["LONG_UPPER_SHADOW"] = False
            micro_features["LONG_LOWER_SHADOW"] = False
            micro_features["PRICE_REJECTION_UPPER"] = False
            micro_features["PRICE_REJECTION_LOWER"] = False
            micro_features["CLOSE_NEAR_HIGH"] = False
            micro_features["CLOSE_NEAR_LOW"] = False
            return micro_features

        # 影线长度与实体大小
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price

        micro_features["SMALL_BODY_CANDLE"] = body_size / total_range < (1 - self.min_shadow_body_ratio)
        micro_features["LONG_UPPER_SHADOW"] = upper_shadow / total_range > self.min_shadow_body_ratio
        micro_features["LONG_LOWER_SHADOW"] = lower_shadow / total_range > self.min_shadow_body_ratio

        # 价格拒绝 (Price Rejection)
        # 价格触及某个高点/低点后迅速反向移动，留下长影线
        # 向上拒绝 (长上影线，收盘价远离高点)
        micro_features["PRICE_REJECTION_UPPER"] = (upper_shadow / total_range > self.min_shadow_body_ratio) and \
                                                  (close_price < high_price * (1 - self.rejection_threshold_pct))
        # 向下拒绝 (长下影线，收盘价远离低点)
        micro_features["PRICE_REJECTION_LOWER"] = (lower_shadow / total_range > self.min_shadow_body_ratio) and \
                                                  (close_price > low_price * (1 + self.rejection_threshold_pct))

        # K线收盘价在K线实体中的位置
        # 收盘价接近最高价 (强势)
        micro_features["CLOSE_NEAR_HIGH"] = (high_price - close_price) / total_range < 0.1 if total_range > 0 else False
        # 收盘价接近最低价 (弱势)
        micro_features["CLOSE_NEAR_LOW"] = (close_price - low_price) / total_range < 0.1 if total_range > 0 else False

        return micro_features

