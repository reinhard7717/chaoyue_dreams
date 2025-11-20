# 文件: strategies/realtime_modules/intraday_vwap_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict

class IntradayVWAPAnalyzer:
    """
    盘中VWAP相关高级指标分析器。
    计算VWAP乖离率、VWAP斜率和VWAP通道，并返回量化值。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.vwap_ma_period = config.get('vwap_ma_period', 5)
        self.vwap_channel_std_dev = config.get('vwap_channel_std_dev', 1.0)
        self.vwap_deviation_threshold_pct = config.get('vwap_deviation_threshold_pct', 0.005)
        print("IntradayVWAPAnalyzer initialized.")
    def analyze_vwap(self, df: pd.DataFrame, timeframe: str) -> Dict[str, float]: # 修改返回类型为 float
        """
        分析给定DataFrame中最新K线的VWAP相关指标，并返回量化值。
        Args:
            df (pd.DataFrame): 包含OHLCV和VWAP数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min')。
        Returns:
            Dict[str, float]: 识别到的VWAP特征及其量化值。
        """
        if not self.enabled or timeframe not in self.apply_on or df.empty:
            return {}
        vwap_features = {}
        vwap_col = f"vwap_{timeframe.replace('min','')}"
        if vwap_col not in df.columns or len(df) < self.vwap_ma_period + 1:
            return {}
        current_kline = df.iloc[-1]
        current_vwap = current_kline[vwap_col]
        if pd.isna(current_vwap) or current_vwap == 0:
            return {}
        # VWAP乖离率
        vwap_deviation = (current_kline['close'] - current_vwap) / current_vwap
        vwap_features["VWAP_DEVIATION_PCT"] = vwap_deviation # 返回量化值
        vwap_features["PRICE_ABOVE_VWAP_SIGNIFICANTLY"] = 1.0 if vwap_deviation > self.vwap_deviation_threshold_pct else 0.0
        vwap_features["PRICE_BELOW_VWAP_SIGNIFICANTLY"] = 1.0 if vwap_deviation < -self.vwap_deviation_threshold_pct else 0.0
        # VWAP斜率 (使用短期MA的斜率)
        vwap_series = df[vwap_col].dropna()
        if len(vwap_series) >= self.vwap_ma_period:
            vwap_ma = ta.sma(vwap_series, length=self.vwap_ma_period)
            if vwap_ma is not None and not vwap_ma.empty and len(vwap_ma) >= 2:
                # 计算斜率的百分比变化
                vwap_slope = (vwap_ma.iloc[-1] - vwap_ma.iloc[-2]) / vwap_ma.iloc[-2] if vwap_ma.iloc[-2] != 0 else 0
                vwap_features["VWAP_SLOPE"] = vwap_slope # 返回量化值
                vwap_features["VWAP_SLOPE_UP"] = 1.0 if vwap_slope > 0 else 0.0
                vwap_features["VWAP_SLOPE_DOWN"] = 1.0 if vwap_slope < 0 else 0.0
            else:
                vwap_features["VWAP_SLOPE"] = np.nan
                vwap_features["VWAP_SLOPE_UP"] = 0.0
                vwap_features["VWAP_SLOPE_DOWN"] = 0.0
        else:
            vwap_features["VWAP_SLOPE"] = np.nan
            vwap_features["VWAP_SLOPE_UP"] = 0.0
            vwap_features["VWAP_SLOPE_DOWN"] = 0.0
        # VWAP通道
        if len(df) > self.vwap_ma_period:
            price_vwap_diff = df['close'] - df[vwap_col]
            std_dev_diff = price_vwap_diff.rolling(self.vwap_ma_period).std().iloc[-1]
            if not pd.isna(std_dev_diff) and std_dev_diff != 0: # 避免除以零
                vwap_upper_channel = current_vwap + self.vwap_channel_std_dev * std_dev_diff
                vwap_lower_channel = current_vwap - self.vwap_channel_std_dev * std_dev_diff
                vwap_features["VWAP_UPPER_CHANNEL"] = vwap_upper_channel
                vwap_features["VWAP_LOWER_CHANNEL"] = vwap_lower_channel
                # 价格触及通道的程度 (例如，触及下轨后反弹的幅度)
                # 这里可以更复杂，暂时只返回布尔值
                vwap_features["PRICE_TOUCHING_VWAP_LOWER_CHANNEL"] = 1.0 if (current_kline['low'] <= vwap_lower_channel and current_kline['close'] > vwap_lower_channel) else 0.0
                vwap_features["PRICE_TOUCHING_VWAP_UPPER_CHANNEL"] = 1.0 if (current_kline['high'] >= vwap_upper_channel and current_kline['close'] < vwap_upper_channel) else 0.0
            else:
                vwap_features["VWAP_UPPER_CHANNEL"] = np.nan
                vwap_features["VWAP_LOWER_CHANNEL"] = np.nan
                vwap_features["PRICE_TOUCHING_VWAP_LOWER_CHANNEL"] = 0.0
                vwap_features["PRICE_TOUCHING_VWAP_UPPER_CHANNEL"] = 0.0
        else:
            vwap_features["VWAP_UPPER_CHANNEL"] = np.nan
            vwap_features["VWAP_LOWER_CHANNEL"] = np.nan
            vwap_features["PRICE_TOUCHING_VWAP_LOWER_CHANNEL"] = 0.0
            vwap_features["PRICE_TOUCHING_VWAP_UPPER_CHANNEL"] = 0.0
        return vwap_features

