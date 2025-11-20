# 文件: strategies/realtime_modules/intraday_volume_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict

class IntradayVolumeAnalyzer:
    """
    盘中成交量异动分析器。
    识别巨量、缩量和量价背离，并返回量化值。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.giant_volume_multiplier = config.get('giant_volume_multiplier', 2.0)
        self.shrinking_volume_ratio = config.get('shrinking_volume_ratio', 0.5)
        self.volume_ma_period = config.get('volume_ma_period', 21)
        print("IntradayVolumeAnalyzer initialized.")
    def analyze_volume(self, df: pd.DataFrame, timeframe: str) -> Dict[str, float]: # 修改返回类型为 float
        """
        分析给定DataFrame中最新K线的成交量异动，并返回量化值。
        Args:
            df (pd.DataFrame): 包含OHLCV和VOL_MA数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min')。
        Returns:
            Dict[str, float]: 识别到的成交量异动及其量化值。
        """
        if not self.enabled or timeframe not in self.apply_on or df.empty:
            return {}
        volume_anomalies = {}
        if len(df) < self.volume_ma_period + 1:
            return {}
        current_kline = df.iloc[-1]
        prev_kline = df.iloc[-2] if len(df) >= 2 else None
        vol_ma_col = f"VOL_MA_{self.volume_ma_period}_{timeframe.replace('min','')}"
        if vol_ma_col not in df.columns or pd.isna(current_kline[vol_ma_col]) or current_kline[vol_ma_col] == 0:
            return {}
        # 成交量与均量的比率
        volume_ratio_to_ma = current_kline['volume'] / current_kline[vol_ma_col]
        volume_anomalies["VOLUME_RATIO_TO_MA"] = volume_ratio_to_ma # 返回量化值
        # 巨量 (布尔值，但评分会使用VOLUME_RATIO_TO_MA)
        volume_anomalies["GIANT_VOLUME"] = 1.0 if volume_ratio_to_ma > self.giant_volume_multiplier else 0.0
        # 缩量 (布尔值)
        volume_anomalies["SHRINKING_VOLUME"] = 1.0 if volume_ratio_to_ma < self.shrinking_volume_ratio else 0.0
        # 量价背离 (布尔值)
        volume_anomalies["VOLUME_PRICE_DIVERGENCE_BULLISH"] = 0.0
        volume_anomalies["VOLUME_PRICE_DIVERGENCE_BEARISH"] = 0.0
        if prev_kline is not None:
            # 价格上涨，成交量萎缩 (潜在顶部背离)
            if current_kline['close'] > prev_kline['close'] and volume_ratio_to_ma < self.shrinking_volume_ratio:
                volume_anomalies["VOLUME_PRICE_DIVERGENCE_BULLISH"] = 1.0
            # 价格下跌，成交量放大 (潜在底部背离或加速下跌)
            if current_kline['close'] < prev_kline['close'] and volume_ratio_to_ma > self.giant_volume_multiplier:
                volume_anomalies["VOLUME_PRICE_DIVERGENCE_BEARISH"] = 1.0
        return volume_anomalies

