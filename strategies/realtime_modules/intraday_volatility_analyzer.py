# 文件: strategies/realtime_modules/intraday_volatility_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict

class IntradayVolatilityAnalyzer:
    """
    盘中波动率分析器。
    计算ATR，检查布林带宽度变化率，日内振幅。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.apply_on = config.get('apply_on', ['5min'])
        self.bbw_slope_period = config.get('bbw_slope_period', 5)
        self.intraday_range_threshold_pct = config.get('intraday_range_threshold_pct', 0.01)
        # 布林带参数从indicators配置中获取
        self.boll_period = config.get('indicators', {}).get('boll_bands_and_width', {}).get('configs', [{}])[0].get('periods', [20])[0]
        self.boll_std_dev = config.get('indicators', {}).get('boll_bands_and_width', {}).get('configs', [{}])[0].get('std_dev', 2.0)
        print("IntradayVolatilityAnalyzer initialized.")

    def analyze_volatility(self, df: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """
        分析给定DataFrame中最新K线的波动率。
        Args:
            df (pd.DataFrame): 包含OHLCV和BBW数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min')。
        Returns:
            Dict[str, float]: 识别到的波动率特征。
        """
        if not self.enabled or timeframe not in self.apply_on or df.empty:
            return {}

        volatility_features = {}
        if len(df) < max(self.boll_period, self.bbw_slope_period) + 1:
            return {}

        current_kline = df.iloc[-1]
        
        # 布林带宽度 (BBW) 变化率
        bbw_col = f"BBW_{self.boll_period}_{self.boll_std_dev}_{timeframe.replace('min','')}"
        if bbw_col in df.columns and len(df) >= self.bbw_slope_period:
            bbw_series = df[bbw_col].dropna()
            if len(bbw_series) >= self.bbw_slope_period:
                # 计算BBW的斜率
                bbw_slope = np.polyfit(range(len(bbw_series[-self.bbw_slope_period:])), bbw_series[-self.bbw_slope_period:], 1)[0]
                volatility_features["BBW_SLOPE"] = bbw_slope
                volatility_features["BBW_SQUEEZING"] = bbw_slope < 0 # 斜率为负表示收缩
                volatility_features["BBW_EXPANDING"] = bbw_slope > 0 # 斜率为正表示扩张
            else:
                volatility_features["BBW_SLOPE"] = np.nan
                volatility_features["BBW_SQUEEZING"] = False
                volatility_features["BBW_EXPANDING"] = False
        else:
            volatility_features["BBW_SLOPE"] = np.nan
            volatility_features["BBW_SQUEEZING"] = False
            volatility_features["BBW_EXPANDING"] = False

        # 日内振幅 (当前K线的高低点与开盘价的相对距离)
        if 'open' in current_kline and 'high' in current_kline and 'low' in current_kline:
            intraday_range_pct = (current_kline['high'] - current_kline['low']) / current_kline['open'] if current_kline['open'] != 0 else 0
            volatility_features["INTRADAY_RANGE_PCT"] = intraday_range_pct
            volatility_features["HIGH_VOLATILITY_CANDLE"] = intraday_range_pct > self.intraday_range_threshold_pct
        else:
            volatility_features["INTRADAY_RANGE_PCT"] = np.nan
            volatility_features["HIGH_VOLATILITY_CANDLE"] = False

        return volatility_features

