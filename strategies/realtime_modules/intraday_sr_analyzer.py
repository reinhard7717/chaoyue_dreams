# 文件: strategies/realtime_modules/intraday_sr_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, Optional

class IntradaySRAnalyzer:
    """
    盘中支撑与阻力位分析器。
    计算枢轴点，并检查价格与前日收盘价/高低点的关系。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.pivot_points_enabled = config.get('pivot_points_enabled', True)
        self.prev_day_close_enabled = config.get('prev_day_close_enabled', True)
        self.prev_day_data: Optional[Dict] = None # 存储前一日的OHLC数据
        self.pivot_points: Dict = {} # 存储当日计算的枢轴点
        print("IntradaySRAnalyzer initialized.")

    def set_prev_day_data(self, prev_day_data: Dict):
        """
        设置前一日的OHLC数据，用于计算枢轴点。
        Args:
            prev_day_data (Dict): 包含 'open', 'high', 'low', 'close' 的字典。
        """
        self.prev_day_data = prev_day_data
        if self.pivot_points_enabled and self.prev_day_data:
            self._calculate_pivot_points()
        else:
            self.pivot_points = {}

    def _calculate_pivot_points(self):
        """
        根据前一日数据计算经典枢轴点。
        Pivot Point (PP) = (High + Low + Close) / 3
        Resistance 1 (R1) = (2 * PP) - Low
        Support 1 (S1) = (2 * PP) - High
        Resistance 2 (R2) = PP + (High - Low)
        Support 2 (S2) = PP - (High - Low)
        Resistance 3 (R3) = High + 2 * (PP - Low)
        Support 3 (S3) = Low - 2 * (High - PP)
        """
        if not self.prev_day_data:
            self.pivot_points = {}
            return

        prev_high = self.prev_day_data.get('high')
        prev_low = self.prev_day_data.get('low')
        prev_close = self.prev_day_data.get('close')

        if any(pd.isna([prev_high, prev_low, prev_close])):
            self.pivot_points = {}
            return

        pp = (prev_high + prev_low + prev_close) / 3
        r1 = (2 * pp) - prev_low
        s1 = (2 * pp) - prev_high
        r2 = pp + (prev_high - prev_low)
        s2 = pp - (prev_high - prev_low)
        r3 = prev_high + 2 * (pp - prev_low)
        s3 = prev_low - 2 * (prev_high - pp)

        self.pivot_points = {
            'PP': pp, 'R1': r1, 'S1': s1,
            'R2': r2, 'S2': s2, 'R3': r3, 'S3': s3
        }
        # print(f"  [SR分析] 枢轴点计算完成: {self.pivot_points}")

    def analyze_sr_levels(self, current_kline: pd.Series, timeframe: str, tolerance_pct: float = 0.001) -> Dict[str, bool]:
        """
        分析当前K线与支撑阻力位的关系。
        Args:
            current_kline (pd.Series): 当前K线数据。
            timeframe (str): 当前时间周期 (e.g., '5min')。
            tolerance_pct (float): 价格触及支撑阻力位允许的误差百分比。
        Returns:
            Dict[str, bool]: 识别到的支撑阻力关系。
        """
        if not self.enabled or current_kline.empty:
            return {}

        sr_features = {}
        current_price = current_kline['close']
        current_high = current_kline['high']
        current_low = current_kline['low']

        # 枢轴点分析
        if self.pivot_points_enabled and self.pivot_points:
            for level_name, level_value in self.pivot_points.items():
                if pd.isna(level_value): continue
                
                # 价格在支撑位附近反弹
                if level_name.startswith('S'):
                    is_touching = (current_low <= level_value * (1 + tolerance_pct)) and \
                                  (current_high >= level_value * (1 - tolerance_pct))
                    is_rebounding = is_touching and (current_price > level_value) and (current_kline['open'] < current_price)
                    sr_features[f"PRICE_REBOUNDING_FROM_{level_name}"] = is_rebounding
                
                # 价格在阻力位附近受阻
                elif level_name.startswith('R'):
                    is_touching = (current_high >= level_value * (1 - tolerance_pct)) and \
                                  (current_low <= level_value * (1 + tolerance_pct))
                    is_rejected = is_touching and (current_price < level_value) and (current_kline['open'] > current_price)
                    sr_features[f"PRICE_REJECTED_AT_{level_name}"] = is_rejected
                
                # 价格在PP附近
                elif level_name == 'PP':
                    is_around_pp = (current_low <= level_value * (1 + tolerance_pct)) and \
                                   (current_high >= level_value * (1 - tolerance_pct))
                    sr_features["PRICE_AROUND_PP"] = is_around_pp

        # 前日收盘价/高低点分析
        if self.prev_day_close_enabled and self.prev_day_data:
            prev_day_close = self.prev_day_data.get('close')
            prev_day_high = self.prev_day_data.get('high')
            prev_day_low = self.prev_day_data.get('low')

            if not pd.isna(prev_day_close):
                sr_features["PRICE_ABOVE_PREV_CLOSE"] = current_price > prev_day_close * (1 + tolerance_pct)
                sr_features["PRICE_BELOW_PREV_CLOSE"] = current_price < prev_day_close * (1 - tolerance_pct)
                sr_features["PRICE_AROUND_PREV_CLOSE"] = (current_low <= prev_day_close * (1 + tolerance_pct)) and \
                                                         (current_high >= prev_day_close * (1 - tolerance_pct))
            
            if not pd.isna(prev_day_high):
                sr_features["PRICE_BREAKING_PREV_HIGH"] = current_price > prev_day_high * (1 + tolerance_pct)
                sr_features["PRICE_REJECTED_AT_PREV_HIGH"] = (current_high >= prev_day_high * (1 - tolerance_pct)) and \
                                                             (current_price < prev_day_high * (1 - tolerance_pct))
            
            if not pd.isna(prev_day_low):
                sr_features["PRICE_BREAKING_PREV_LOW"] = current_price < prev_day_low * (1 - tolerance_pct)
                sr_features["PRICE_REBOUNDING_FROM_PREV_LOW"] = (current_low <= prev_day_low * (1 + tolerance_pct)) and \
                                                                (current_price > prev_day_low * (1 + tolerance_pct))

        return sr_features

