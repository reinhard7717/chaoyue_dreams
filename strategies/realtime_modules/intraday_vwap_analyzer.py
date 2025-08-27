# 文件: strategies/realtime_modules/intraday_vwap_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict

class IntradayVWAPAnalyzer:
    """
    盘中VWAP相关高级指标分析器。
    计算VWAP乖离率、VWAP斜率和VWAP通道。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.vwap_ma_period = config.get('vwap_ma_period', 5)
        self.vwap_channel_std_dev = config.get('vwap_channel_std_dev', 1.0)
        self.vwap_deviation_threshold_pct = config.get('vwap_deviation_threshold_pct', 0.005)
        print("IntradayVWAPAnalyzer initialized.")

    def analyze_vwap(self, df: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """
        分析给定DataFrame中最新K线的VWAP相关指标。
        Args:
            df (pd.DataFrame): 包含OHLCV和VWAP数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min')。
        Returns:
            Dict[str, float]: 识别到的VWAP特征。
        """
        if not self.enabled or timeframe not in self.apply_on or df.empty:
            return {}

        vwap_features = {}
        vwap_col = f"vwap_{timeframe.replace('min','')}"
        if vwap_col not in df.columns or len(df) < self.vwap_ma_period + 1:
            # print(f"  [VWAP分析] 警告: 缺少 {vwap_col} 列或数据不足，无法分析VWAP。")
            return {}

        current_kline = df.iloc[-1]
        current_vwap = current_kline[vwap_col]

        if pd.isna(current_vwap) or current_vwap == 0:
            return {}

        # VWAP乖离率
        vwap_deviation = (current_kline['close'] - current_vwap) / current_vwap
        vwap_features["VWAP_DEVIATION_PCT"] = vwap_deviation
        vwap_features["PRICE_ABOVE_VWAP_SIGNIFICANTLY"] = vwap_deviation > self.vwap_deviation_threshold_pct
        vwap_features["PRICE_BELOW_VWAP_SIGNIFICANTLY"] = vwap_deviation < -self.vwap_deviation_threshold_pct

        # VWAP斜率 (使用短期MA的斜率)
        vwap_series = df[vwap_col].dropna()
        if len(vwap_series) >= self.vwap_ma_period:
            vwap_ma = ta.sma(vwap_series, length=self.vwap_ma_period)
            if vwap_ma is not None and not vwap_ma.empty and len(vwap_ma) >= 2:
                vwap_slope = (vwap_ma.iloc[-1] - vwap_ma.iloc[-2]) / vwap_ma.iloc[-2] if vwap_ma.iloc[-2] != 0 else 0
                vwap_features["VWAP_SLOPE"] = vwap_slope
                vwap_features["VWAP_SLOPE_UP"] = vwap_slope > 0
                vwap_features["VWAP_SLOPE_DOWN"] = vwap_slope < 0
            else:
                vwap_features["VWAP_SLOPE"] = np.nan
                vwap_features["VWAP_SLOPE_UP"] = False
                vwap_features["VWAP_SLOPE_DOWN"] = False
        else:
            vwap_features["VWAP_SLOPE"] = np.nan
            vwap_features["VWAP_SLOPE_UP"] = False
            vwap_features["VWAP_SLOPE_DOWN"] = False

        # VWAP通道 (简单实现：VWAP +/- N倍标准差)
        # 需要计算VWAP的波动性，这里简化为价格与VWAP的偏差标准差
        if len(df) > self.vwap_ma_period:
            price_vwap_diff = df['close'] - df[vwap_col]
            std_dev_diff = price_vwap_diff.rolling(self.vwap_ma_period).std().iloc[-1]
            if not pd.isna(std_dev_diff):
                vwap_upper_channel = current_vwap + self.vwap_channel_std_dev * std_dev_diff
                vwap_lower_channel = current_vwap - self.vwap_channel_std_dev * std_dev_diff
                vwap_features["VWAP_UPPER_CHANNEL"] = vwap_upper_channel
                vwap_features["VWAP_LOWER_CHANNEL"] = vwap_lower_channel
                vwap_features["PRICE_TOUCHING_VWAP_LOWER_CHANNEL"] = current_kline['low'] <= vwap_lower_channel and current_kline['close'] > vwap_lower_channel
                vwap_features["PRICE_TOUCHING_VWAP_UPPER_CHANNEL"] = current_kline['high'] >= vwap_upper_channel and current_kline['close'] < vwap_upper_channel
            else:
                vwap_features["VWAP_UPPER_CHANNEL"] = np.nan
                vwap_features["VWAP_LOWER_CHANNEL"] = np.nan
                vwap_features["PRICE_TOUCHING_VWAP_LOWER_CHANNEL"] = False
                vwap_features["PRICE_TOUCHING_VWAP_UPPER_CHANNEL"] = False
        else:
            vwap_features["VWAP_UPPER_CHANNEL"] = np.nan
            vwap_features["VWAP_LOWER_CHANNEL"] = np.nan
            vwap_features["PRICE_TOUCHING_VWAP_LOWER_CHANNEL"] = False
            vwap_features["PRICE_TOUCHING_VWAP_UPPER_CHANNEL"] = False

        return vwap_features

