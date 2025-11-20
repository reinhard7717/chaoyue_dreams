# 文件: strategies/realtime_modules/intraday_multi_timeframe_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List

class IntradayMultiTimeframeAnalyzer:
    """
    多周期共振分析器。
    检查不同时间周期（如5min, 30min, 60min）之间的指标共振。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.min_30min_ema_slope = config.get('playbooks', {}).get('multi_timeframe_ema_confluence', {}).get('min_30min_ema_slope', 0.0001)
        self.min_60min_ema_slope = config.get('playbooks', {}).get('multi_timeframe_ema_confluence', {}).get('min_60min_ema_slope', 0.00005)
        print("IntradayMultiTimeframeAnalyzer initialized.")
    def analyze_confluence(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """
        分析不同时间周期之间的共振信号。
        Args:
            all_data (Dict[str, pd.DataFrame]): 包含所有时间周期数据的字典。
        Returns:
            Dict[str, bool]: 识别到的多周期共振特征。
        """
        if not self.enabled:
            return {}
        confluence_features = {}
        # 检查5min, 30min, 60min EMA多头排列共振
        confluence_features["EMA_BULLISH_CONFLUENCE"] = self._check_ema_alignment_confluence(all_data)
        # 可以添加更多共振检查，例如：
        # - 5min K线形态在30min支撑位附近
        # - 5min VWAP突破，同时30min VWAP也向上
        # ...
        return confluence_features
    def _check_ema_alignment_confluence(self, all_data: Dict[str, pd.DataFrame]) -> bool:
        """
        检查5min, 30min, 60min EMA是否形成多头排列共振。
        """
        # 确保所有时间周期数据都存在且足够长
        if '5min' not in all_data or all_data['5min'].empty or \
           '30min' not in all_data or all_data['30min'].empty or \
           '60min' not in all_data or all_data['60min'].empty:
            return False
        df_5min = all_data['5min']
        df_30min = all_data['30min']
        df_60min = all_data['60min']
        # 检查5分钟EMA多头排列
        # 假设EMA周期为5, 13, 21
        ema_5_5min_col = "EMA_5_5"
        ema_13_5min_col = "EMA_13_5"
        ema_21_5min_col = "EMA_21_5"
        if not all(col in df_5min.columns for col in [ema_5_5min_col, ema_13_5min_col, ema_21_5min_col]):
            # print(f"  [多周期分析] 警告: 5分钟EMA列缺失。")
            return False
        current_5min = df_5min.iloc[-1]
        cond_5min_ema_bullish = (current_5min[ema_5_5min_col] > current_5min[ema_13_5min_col] and
                                 current_5min[ema_13_5min_col] > current_5min[ema_21_5min_col])
        # 检查30分钟EMA多头排列及斜率
        ema_5_30min_col = "EMA_5_30"
        ema_13_30min_col = "EMA_13_30"
        ema_21_30min_col = "EMA_21_30"
        if not all(col in df_30min.columns for col in [ema_5_30min_col, ema_13_30min_col, ema_21_30min_col]):
            # print(f"  [多周期分析] 警告: 30分钟EMA列缺失。")
            return False
        current_30min = df_30min.iloc[-1]
        prev_30min = df_30min.iloc[-2] if len(df_30min) >= 2 else None
        cond_30min_ema_bullish = (current_30min[ema_5_30min_col] > current_30min[ema_13_30min_col] and
                                  current_30min[ema_13_30min_col] > current_30min[ema_21_30min_col])
        cond_30min_ema_slope_up = False
        if prev_30min is not None:
            # 检查EMA5的斜率
            if prev_30min[ema_5_30min_col] != 0:
                slope_30min = (current_30min[ema_5_30min_col] - prev_30min[ema_5_30min_col]) / prev_30min[ema_5_30min_col]
                cond_30min_ema_slope_up = slope_30min > self.min_30min_ema_slope
        # 检查60分钟EMA多头排列及斜率
        ema_5_60min_col = "EMA_5_60"
        ema_13_60min_col = "EMA_13_60"
        ema_21_60min_col = "EMA_21_60"
        if not all(col in df_60min.columns for col in [ema_5_60min_col, ema_13_60min_col, ema_21_60min_col]):
            # print(f"  [多周期分析] 警告: 60分钟EMA列缺失。")
            return False
        current_60min = df_60min.iloc[-1]
        prev_60min = df_60min.iloc[-2] if len(df_60min) >= 2 else None
        cond_60min_ema_bullish = (current_60min[ema_5_60min_col] > current_60min[ema_13_60min_col] and
                                  current_60min[ema_13_60min_col] > current_60min[ema_21_60min_col])
        cond_60min_ema_slope_up = False
        if prev_60min is not None:
            # 检查EMA5的斜率
            if prev_60min[ema_5_60min_col] != 0:
                slope_60min = (current_60min[ema_5_60min_col] - prev_60min[ema_5_60min_col]) / prev_60min[ema_5_60min_col]
                cond_60min_ema_slope_up = slope_60min > self.min_60min_ema_slope
        # 只有当所有周期都满足多头排列且大周期EMA斜率向上时才算共振
        return all([cond_5min_ema_bullish, cond_30min_ema_bullish, cond_30min_ema_slope_up,
                    cond_60min_ema_bullish, cond_60min_ema_slope_up])

