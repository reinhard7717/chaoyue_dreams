# 文件: strategies/realtime_modules/intraday_data_aggregator.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple

class IntradayDataAggregator:
    """
    负责将1分钟K线数据聚合到指定的时间周期 (如5min, 30min, 60min)，
    并计算所有配置的技术指标。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.base_timeframe = config.get('base_timeframe', '1min')
        self.processed_timeframes = [tf for tf in config.get('processed_timeframes', ['5min', '30min', '60min']) if tf != self.base_timeframe] # 确保不重复处理base_timeframe
        self.min_data_points = {
            '5min': config.get('min_data_points_5min', 21),
            '30min': config.get('min_data_points_30min', 34),
            '60min': config.get('min_data_points_60min', 55),
        }
        self.indicators_config = config.get('indicators', {})
        self.data_buffer = {tf: pd.DataFrame() for tf in self.processed_timeframes + [self.base_timeframe]}
        self.last_processed_time = {tf: None for tf in self.processed_timeframes}
        print("IntradayDataAggregator initialized.")
    def update_1min_data(self, new_1min_kline: pd.Series):
        """
        接收新的1分钟K线数据，并更新内部数据缓冲区。
        Args:
            new_1min_kline (pd.Series): 包含 'open', 'high', 'low', 'close', 'volume' 的1分钟K线数据。
                                        索引应为 datetime 对象。
        """
        if not isinstance(new_1min_kline.name, datetime):
            raise ValueError("new_1min_kline 索引必须是 datetime 对象。")
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in new_1min_kline:
                new_1min_kline[col] = np.nan # 确保列存在，即使为空
        new_1min_kline['volume'] = new_1min_kline['volume'].astype(float)
        # 将新的1分钟K线添加到缓冲区
        current_date = new_1min_kline.name.date()
        if self.data_buffer[self.base_timeframe].empty or self.data_buffer[self.base_timeframe].index[-1].date() != current_date:
            # 新的一天，清空缓冲区
            self.data_buffer[self.base_timeframe] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            for tf in self.processed_timeframes:
                self.data_buffer[tf] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                self.last_processed_time[tf] = None
            print(f"  [数据聚合器] 新的一天 {current_date}，清空数据缓冲区。")
        # 避免重复添加
        if not self.data_buffer[self.base_timeframe].empty and new_1min_kline.name <= self.data_buffer[self.base_timeframe].index[-1]:
            # print(f"  [数据聚合器] 警告: 收到重复或旧的1分钟K线数据 {new_1min_kline.name}，跳过。")
            return
        self.data_buffer[self.base_timeframe] = pd.concat([self.data_buffer[self.base_timeframe], pd.DataFrame([new_1min_kline])])
        self.data_buffer[self.base_timeframe].index.name = 'datetime'
        # print(f"  [数据聚合器] 添加1分钟K线: {new_1min_kline.name}")
    def aggregate_and_calculate_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        聚合1分钟数据到所有配置的时间周期，并计算指标。
        Returns:
            Dict[str, pd.DataFrame]: 包含所有时间周期数据的字典。
        """
        if self.data_buffer[self.base_timeframe].empty:
            return {tf: pd.DataFrame() for tf in self.processed_timeframes + [self.base_timeframe]}
        current_data = self.data_buffer[self.base_timeframe].copy()
        all_processed_data = {self.base_timeframe: current_data}
        for tf_str in self.processed_timeframes:
            freq = self._convert_timeframe_to_freq(tf_str)
            # 确保当前时间周期的数据至少有一行，并且时间戳是递增的
            if not self.data_buffer[tf_str].empty and current_data.index[-1] <= self.data_buffer[tf_str].index[-1]:
                # 如果最新的1分钟数据时间戳小于或等于已聚合的最高时间戳，则无需重新聚合
                # print(f"  [数据聚合器] {tf_str} 数据已是最新，跳过聚合。")
                all_processed_data[tf_str] = self.data_buffer[tf_str]
                continue
            # 聚合数据
            ohlcv_agg = current_data['volume'].resample(freq).apply(self._ohlcv_agg_func)
            ohlcv_agg = ohlcv_agg.dropna() # 移除空K线
            if ohlcv_agg.empty:
                all_processed_data[tf_str] = self.data_buffer[tf_str]
                continue
            # 确保聚合后的数据与之前的历史数据拼接正确
            if not self.data_buffer[tf_str].empty:
                # 找到ohlcv_agg中比data_buffer[tf_str]最新时间戳更新的数据
                last_tf_time = self.data_buffer[tf_str].index[-1]
                new_ohlcv_agg = ohlcv_agg[ohlcv_agg.index > last_tf_time]
                if not new_ohlcv_agg.empty:
                    self.data_buffer[tf_str] = pd.concat([self.data_buffer[tf_str], new_ohlcv_agg])
                    self.data_buffer[tf_str] = self.data_buffer[tf_str].drop_duplicates(subset=['open', 'high', 'low', 'close', 'volume'], keep='last') # 避免重复
                    self.data_buffer[tf_str] = self.data_buffer[tf_str].sort_index()
            else:
                self.data_buffer[tf_str] = ohlcv_agg
            # 计算指标
            if len(self.data_buffer[tf_str]) >= self.min_data_points.get(tf_str, 1):
                self._calculate_technical_indicators(self.data_buffer[tf_str], tf_str)
            all_processed_data[tf_str] = self.data_buffer[tf_str]
            # print(f"  [数据聚合器] {tf_str} 聚合完成，当前数据量: {len(self.data_buffer[tf_str])}")
        return all_processed_data
    def _ohlcv_agg_func(self, group: pd.Series) -> pd.Series:
        """自定义OHLCV聚合函数"""
        if group.empty:
            return pd.Series({'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan, 'volume': np.nan})
        # 获取对应时间段的原始K线数据
        group_df = self.data_buffer[self.base_timeframe].loc[group.index]
        return pd.Series({
            'open': group_df['open'].iloc[0],
            'high': group_df['high'].max(),
            'low': group_df['low'].min(),
            'close': group_df['close'].iloc[-1],
            'volume': group_df['volume'].sum()
        })
    def _convert_timeframe_to_freq(self, tf_str: str) -> str:
        """将时间周期字符串转换为pandas resample频率字符串"""
        if tf_str.endswith('min'):
            return tf_str.replace('min', 'T')
        elif tf_str.endswith('h'):
            return tf_str.replace('h', 'H')
        return tf_str # 默认返回原字符串
    def _calculate_technical_indicators(self, df: pd.DataFrame, timeframe: str):
        """
        为给定的DataFrame计算技术指标。
        Args:
            df (pd.DataFrame): 包含OHLCV数据的DataFrame。
            timeframe (str): 当前时间周期 (e.g., '5min', '30min', '60min')。
        """
        # EMA
        ema_config = self.indicators_config.get('ema', {})
        if ema_config.get('enabled'):
            for cfg in ema_config.get('configs', []):
                if timeframe in cfg.get('apply_on', []):
                    for period in cfg.get('periods', []):
                        col_name = f"EMA_{period}_{timeframe.replace('min','')}"
                        df[col_name] = ta.ema(df['close'], length=period)
        # VOL_MA
        vol_ma_config = self.indicators_config.get('vol_ma', {})
        if vol_ma_config.get('enabled'):
            for cfg in vol_ma_config.get('configs', []):
                if timeframe in cfg.get('apply_on', []):
                    for period in cfg.get('periods', []):
                        col_name = f"VOL_MA_{period}_{timeframe.replace('min','')}"
                        df[col_name] = ta.sma(df['volume'], length=period)
        # Bollinger Bands and Width
        boll_config = self.indicators_config.get('boll_bands_and_width', {})
        if boll_config.get('enabled'):
            for cfg in boll_config.get('configs', []):
                if timeframe in cfg.get('apply_on', []):
                    period = cfg.get('periods', [20])[0]
                    std_dev = cfg.get('std_dev', 2.0)
                    bbands = ta.bbands(df['close'], length=period, std=std_dev)
                    if bbands is not None and not bbands.empty:
                        df[f"BBL_{period}_{std_dev}_{timeframe.replace('min','')}"] = bbands[f"BBL_{period}_{std_dev}"]
                        df[f"BBM_{period}_{std_dev}_{timeframe.replace('min','')}"] = bbands[f"BBM_{period}_{std_dev}"]
                        df[f"BBU_{period}_{std_dev}_{timeframe.replace('min','')}"] = bbands[f"BBU_{period}_{std_dev}"]
                        df[f"BBB_{period}_{std_dev}_{timeframe.replace('min','')}"] = bbands[f"BBB_{period}_{std_dev}"] # Bollinger Bandwidth
                        df[f"BBP_{period}_{std_dev}_{timeframe.replace('min','')}"] = bbands[f"BBP_{period}_{std_dev}"] # Bollinger Band Percent
                        # BBW (Bollinger Band Width)
                        df[f"BBW_{period}_{std_dev}_{timeframe.replace('min','')}"] = (df[f"BBU_{period}_{std_dev}_{timeframe.replace('min','')}"] - df[f"BBL_{period}_{std_dev}_{timeframe.replace('min','')}"]) / df[f"BBM_{period}_{std_dev}_{timeframe.replace('min','')}"]
        # VWAP
        vwap_config = self.indicators_config.get('vwap', {})
        if vwap_config.get('enabled') and timeframe in vwap_config.get('apply_on', []):
            # VWAP需要每日重新计算，这里假设df是当日数据
            # ta.vwap 默认从数据开始计算，对于盘中实时数据，需要确保是当日的累积VWAP
            # 简单实现：假设df是当日数据，直接计算
            df[f"vwap_{timeframe.replace('min','')}"] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        # ATR
        atr_config = self.indicators_config.get('atr', {})
        if atr_config.get('enabled'):
            for cfg in atr_config.get('configs', []):
                if timeframe in cfg.get('apply_on', []):
                    for period in cfg.get('periods', []):
                        col_name = f"ATR_{period}_{timeframe.replace('min','')}"
                        df[col_name] = ta.atr(df['high'], df['low'], df['close'], length=period)
        # 确保所有指标列都是数值类型，非数值设为NaN
        for col in df.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume'] and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

