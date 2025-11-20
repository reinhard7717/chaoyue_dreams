# 文件: strategies/realtime_modules/intraday_sr_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, Optional

class IntradaySRAnalyzer:
    """
    盘中支撑与阻力位分析器。
    计算枢轴点，并检查价格与前日收盘价/高低点的关系，返回量化值。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.pivot_points_enabled = config.get('pivot_points_enabled', True)
        self.prev_day_close_enabled = config.get('prev_day_close_enabled', True)
        self.rebound_strength_min_pct = config.get('rebound_strength_min_pct', 0.005)
        self.rejection_strength_min_pct = config.get('rejection_strength_min_pct', 0.005)
        self.prev_day_data: Optional[Dict] = None
        self.pivot_points: Dict = {}
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
    def analyze_sr_levels(self, current_kline: pd.Series, timeframe: str, tolerance_pct: float = 0.001) -> Dict[str, float]: # 修改返回类型为 float
        """
        分析当前K线与支撑阻力位的关系，并返回量化值。
        Args:
            current_kline (pd.Series): 当前K线数据。
            timeframe (str): 当前时间周期 (e.g., '5min')。
            tolerance_pct (float): 价格触及支撑阻力位允许的误差百分比。
        Returns:
            Dict[str, float]: 识别到的支撑阻力关系及其量化值。
        """
        if not self.enabled or current_kline.empty:
            return {}
        sr_features = {}
        current_price = current_kline['close']
        current_high = current_kline['high']
        current_low = current_kline['low']
        current_open = current_kline['open']
        if any(pd.isna([current_price, current_high, current_low, current_open])):
            return {}
        # 枢轴点分析
        if self.pivot_points_enabled and self.pivot_points:
            for level_name, level_value in self.pivot_points.items():
                if pd.isna(level_value) or level_value == 0: continue
                # 价格在支撑位附近反弹
                if level_name.startswith('S'):
                    # 价格触及支撑位
                    is_touching = (current_low <= level_value * (1 + tolerance_pct)) and \
                                  (current_high >= level_value * (1 - tolerance_pct))
                    if is_touching and current_price > current_open: # 阳线反弹
                        rebound_strength = (current_price - max(current_open, level_value)) / level_value # 反弹幅度
                        if rebound_strength > self.rebound_strength_min_pct: # 达到最小反弹强度
                            sr_features[f"PRICE_REBOUNDING_FROM_SUPPORT"] = rebound_strength # 返回反弹强度
                            sr_features[f"PRICE_REBOUNDING_FROM_{level_name}"] = 1.0 # 布尔值
                    else:
                        sr_features[f"PRICE_REBOUNDING_FROM_{level_name}"] = 0.0
                # 价格在阻力位附近受阻
                elif level_name.startswith('R'):
                    # 价格触及阻力位
                    is_touching = (current_high >= level_value * (1 - tolerance_pct)) and \
                                  (current_low <= level_value * (1 + tolerance_pct))
                    if is_touching and current_price < current_open: # 阴线拒绝
                        rejection_strength = (min(current_open, level_value) - current_price) / level_value # 拒绝幅度
                        if rejection_strength > self.rejection_strength_min_pct: # 达到最小拒绝强度
                            sr_features[f"PRICE_REJECTION_AT_RESISTANCE"] = rejection_strength # 返回拒绝强度
                            sr_features[f"PRICE_REJECTED_AT_{level_name}"] = 1.0 # 布尔值
                    else:
                        sr_features[f"PRICE_REJECTED_AT_{level_name}"] = 0.0
                # 价格在PP附近
                elif level_name == 'PP':
                    is_around_pp = (current_low <= level_value * (1 + tolerance_pct)) and \
                                   (current_high >= level_value * (1 - tolerance_pct))
                    sr_features["PRICE_AROUND_PP"] = 1.0 if is_around_pp else 0.0
        # 前日收盘价/高低点分析
        if self.prev_day_close_enabled and self.prev_day_data:
            prev_day_close = self.prev_day_data.get('close')
            prev_day_high = self.prev_day_data.get('high')
            prev_day_low = self.prev_day_data.get('low')
            if not pd.isna(prev_day_close) and prev_day_close != 0:
                sr_features["PRICE_ABOVE_PREV_CLOSE"] = 1.0 if current_price > prev_day_close * (1 + tolerance_pct) else 0.0
                sr_features["PRICE_BELOW_PREV_CLOSE"] = 1.0 if current_price < prev_day_close * (1 - tolerance_pct) else 0.0
                sr_features["PRICE_AROUND_PREV_CLOSE"] = 1.0 if ((current_low <= prev_day_close * (1 + tolerance_pct)) and \
                                                         (current_high >= prev_day_close * (1 - tolerance_pct))) else 0.0
            if not pd.isna(prev_day_high) and prev_day_high != 0:
                sr_features["PRICE_BREAKING_PREV_HIGH"] = 1.0 if current_price > prev_day_high * (1 + tolerance_pct) else 0.0
                sr_features["PRICE_REJECTED_AT_PREV_HIGH"] = 1.0 if ((current_high >= prev_day_high * (1 - tolerance_pct)) and \
                                                             (current_price < prev_day_high * (1 - tolerance_pct))) else 0.0
            if not pd.isna(prev_day_low) and prev_day_low != 0:
                sr_features["PRICE_BREAKING_PREV_LOW"] = 1.0 if current_price < prev_day_low * (1 - tolerance_pct) else 0.0
                sr_features["PRICE_REBOUNDING_FROM_PREV_LOW"] = 1.0 if ((current_low <= prev_day_low * (1 + tolerance_pct)) and \
                                                                (current_price > prev_day_low * (1 + tolerance_pct))) else 0.0
        return sr_features

