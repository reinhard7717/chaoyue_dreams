# 文件: strategies/realtime_modules/intraday_volatility_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict

class IntradayVolatilityAnalyzer:
    """
    盘中波动率分析器。
    计算ATR，检查布林带宽度变化率，日内振幅，并返回量化值。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.bbw_slope_period = config.get('bbw_slope_period', 5)
        self.intraday_range_threshold_pct = config.get('intraday_range_threshold_pct', 0.01)
        self.bbw_lookback_window = config.get('bbw_lookback_window', 120) # BBW分位数回溯窗口
        self.bbw_squeeze_percentile_threshold = config.get('bbw_squeeze_percentile_threshold', 0.3) # BBW收缩分位数阈值
        self.boll_period = config.get('indicators', {}).get('boll_bands_and_width', {}).get('configs', [{}])[0].get('periods', [20])[0]
        self.boll_std_dev = config.get('indicators', {}).get('boll_bands_and_width', {}).get('configs', [{}])[0].get('std_dev', 2.0)
        print("IntradayVolatilityAnalyzer initialized.")
    def analyze_volatility(self, df: pd.DataFrame, timeframe: str) -> Dict[str, float]: # 修改返回类型为 float
        """
        分析给定DataFrame中最新K线的波动率，并返回量化值。
        Args:
            df (pd.DataFrame): 包含OHLCV和BBW数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min')。
        Returns:
            Dict[str, float]: 识别到的波动率特征及其量化值。
        """
        if not self.enabled or timeframe not in self.apply_on or df.empty:
            return {}
        volatility_features = {}
        if len(df) < max(self.boll_period, self.bbw_slope_period, self.bbw_lookback_window) + 1: # 确保足够数据计算BBW分位数
            return {}
        current_kline = df.iloc[-1]
        bbw_col = f"BBW_{self.boll_period}_{self.boll_std_dev}_{timeframe.replace('min','')}"
        if bbw_col in df.columns and not df[bbw_col].isnull().all(): # 确保BBW列存在且有数据
            bbw_series = df[bbw_col].dropna()
            # BBW斜率
            if len(bbw_series) >= self.bbw_slope_period:
                bbw_slope = np.polyfit(range(len(bbw_series[-self.bbw_slope_period:])), bbw_series[-self.bbw_slope_period:], 1)[0]
                volatility_features["BBW_SLOPE"] = bbw_slope
                volatility_features["BBW_SQUEEZING"] = 1.0 if bbw_slope < 0 else 0.0
                volatility_features["BBW_EXPANDING"] = 1.0 if bbw_slope > 0 else 0.0
            else:
                volatility_features["BBW_SLOPE"] = np.nan
                volatility_features["BBW_SQUEEZING"] = 0.0
                volatility_features["BBW_EXPANDING"] = 0.0
            # BBW分位数 (量化收缩程度)
            if len(bbw_series) >= self.bbw_lookback_window:
                current_bbw = current_kline[bbw_col]
                if not pd.isna(current_bbw):
                    # 计算当前BBW在过去N根K线中的分位数 (越小越收缩，分位数越低)
                    bbw_percentile = (bbw_series.iloc[-self.bbw_lookback_window:].rank(pct=True).iloc[-1])
                    volatility_features["BBW_PERCENTILE"] = bbw_percentile # 返回BBW分位数
                    volatility_features["BBW_IS_SQUEEZED"] = 1.0 if bbw_percentile < self.bbw_squeeze_percentile_threshold else 0.0
                else:
                    volatility_features["BBW_PERCENTILE"] = np.nan
                    volatility_features["BBW_IS_SQUEEZED"] = 0.0
            else:
                volatility_features["BBW_PERCENTILE"] = np.nan
                volatility_features["BBW_IS_SQUEEZED"] = 0.0
        else:
            volatility_features["BBW_SLOPE"] = np.nan
            volatility_features["BBW_SQUEEZING"] = 0.0
            volatility_features["BBW_EXPANDING"] = 0.0
            volatility_features["BBW_PERCENTILE"] = np.nan
            volatility_features["BBW_IS_SQUEEZED"] = 0.0
        # 日内振幅
        if 'open' in current_kline and 'high' in current_kline and 'low' in current_kline and current_kline['open'] != 0:
            intraday_range_pct = (current_kline['high'] - current_kline['low']) / current_kline['open']
            volatility_features["INTRADAY_RANGE_PCT"] = intraday_range_pct
            volatility_features["HIGH_VOLATILITY_CANDLE"] = 1.0 if intraday_range_pct > self.intraday_range_threshold_pct else 0.0
        else:
            volatility_features["INTRADAY_RANGE_PCT"] = np.nan
            volatility_features["HIGH_VOLATILITY_CANDLE"] = 0.0
        return volatility_features

