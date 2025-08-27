# 文件: strategies/realtime_modules/intraday_volume_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict

class IntradayVolumeAnalyzer:
    """
    盘中成交量异动分析器。
    识别巨量、缩量和量价背离。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.giant_volume_multiplier = config.get('giant_volume_multiplier', 2.0)
        self.shrinking_volume_ratio = config.get('shrinking_volume_ratio', 0.5)
        self.volume_ma_period = config.get('volume_ma_period', 21)
        print("IntradayVolumeAnalyzer initialized.")

    def analyze_volume(self, df: pd.DataFrame, timeframe: str) -> Dict[str, bool]:
        """
        分析给定DataFrame中最新K线的成交量异动。
        Args:
            df (pd.DataFrame): 包含OHLCV和VOL_MA数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min')。
        Returns:
            Dict[str, bool]: 识别到的成交量异动及其布尔值。
        """
        if not self.enabled or timeframe not in self.apply_on or df.empty:
            return {}

        volume_anomalies = {}
        if len(df) < self.volume_ma_period + 1: # 需要足够的历史数据来计算均量
            return {}

        current_kline = df.iloc[-1]
        prev_kline = df.iloc[-2] if len(df) >= 2 else None
        
        vol_ma_col = f"VOL_MA_{self.volume_ma_period}_{timeframe.replace('min','')}"
        if vol_ma_col not in df.columns or pd.isna(current_kline[vol_ma_col]):
            # print(f"  [成交量分析] 警告: 缺少 {vol_ma_col} 列或其值为NaN，无法分析成交量异动。")
            return {}

        # 巨量
        volume_anomalies["GIANT_VOLUME"] = current_kline['volume'] > current_kline[vol_ma_col] * self.giant_volume_multiplier

        # 缩量
        volume_anomalies["SHRINKING_VOLUME"] = current_kline['volume'] < current_kline[vol_ma_col] * self.shrinking_volume_ratio

        # 量价背离 (简单实现：价格创新高/低但成交量萎缩/放大)
        volume_anomalies["VOLUME_PRICE_DIVERGENCE_BULLISH"] = False # 价格上涨，成交量萎缩
        volume_anomalies["VOLUME_PRICE_DIVERGENCE_BEARISH"] = False # 价格下跌，成交量放大

        if prev_kline is not None:
            # 价格上涨，成交量萎缩 (潜在顶部背离)
            if current_kline['close'] > prev_kline['close'] and current_kline['volume'] < prev_kline['volume'] * self.shrinking_volume_ratio:
                volume_anomalies["VOLUME_PRICE_DIVERGENCE_BULLISH"] = True
            # 价格下跌，成交量放大 (潜在底部背离或加速下跌)
            if current_kline['close'] < prev_kline['close'] and current_kline['volume'] > prev_kline['volume'] * self.giant_volume_multiplier:
                volume_anomalies["VOLUME_PRICE_DIVERGENCE_BEARISH"] = True

        return volume_anomalies

