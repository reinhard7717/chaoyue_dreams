# 文件: strategies/realtime_strategy.py

import logging
import pandas as pd
from typing import Dict, Optional, List
from datetime import time
import traceback

logger = logging.getLogger("strategy")

class RealtimeStrategy:
    """
    盘中决策引擎。
    基于对Tushare数据源的正确理解，使用可靠的量价指标和五档盘口压力指标，
    执行多个交易剧本，寻找高置信度的盘中交易机会。
    """
    def __init__(self, params: Dict):
        """
        初始化策略，所有阈值都应通过参数传入，便于调优。
        
        Args:
            params (Dict): 包含所有策略参数的字典。
        """
        self.params = params
        # 默认交易时间窗口
        self.trade_start_time = params.get('trade_start_time', time(9, 45))
        self.trade_end_time = params.get('trade_end_time', time(14, 50))
        self.min_data_points_5min = params.get('min_data_points_5min', 21)
        self.min_data_points_30min = params.get('min_data_points_30min', 34) # EMA_34_30 需要34周期
        self.min_data_points_60min = params.get('min_data_points_60min', 55) # EMA_55_60 需要55周期

    def run_strategy(self, df_intraday: pd.DataFrame, daily_signal_info: Dict) -> Optional[Dict]:
        """
        对单个股票执行所有交易剧本，寻找第一个满足条件的入场信号。
        
        Args:
            df_intraday (pd.DataFrame): 包含分钟级别特征的战术情报矩阵。
            daily_signal_info (Dict): 盘后信号的关键信息。

        Returns:
            Optional[Dict]: 如果找到入场信号，返回包含详细信息的字典；否则返回None。
        """
        stock_code = daily_signal_info.get('stock_code', 'Unknown')
        print(f"    -> [盘中策略引擎] 开始对 {stock_code} 执行多剧本分析...")
        
        # 忽略数据不足的情况，检查最细粒度的5分钟数据点是否足够
        if len(df_intraday) < self.min_data_points_5min:
            print(f"      - 提示: {stock_code} 5分钟数据点不足 ({len(df_intraday)} < {self.min_data_points_5min})，跳过分析。")
            return None

        # 确保所有必要的列都存在，增强健壮性
        required_cols_5min = ['vwap_5', 'close_5', 'volume_5', 'VOL_MA_21_5', 'BBU_21_2.0_5', 'open_5', 'low_5', 'high_5', 'BBW_21_2.0_5']
        required_cols_30min = ['close_30', 'EMA_5_30', 'EMA_13_30', 'EMA_21_30', 'vwap_30', 'open_30', 'low_30', 'high_30']
        required_cols_60min = ['close_60', 'EMA_5_60', 'EMA_13_60', 'EMA_21_60', 'open_60', 'low_60', 'high_60']

        if not all(col in df_intraday.columns for col in required_cols_5min):
            print(f"      - 提示: {stock_code} 缺少5分钟关键指标列，跳过分析。缺失: {[col for col in required_cols_5min if col not in df_intraday.columns]}")
            return None
        if not all(col in df_intraday.columns for col in required_cols_30min):
            print(f"      - 提示: {stock_code} 缺少30分钟关键指标列，跳过分析。缺失: {[col for col in required_cols_30min if col not in df_intraday.columns]}")
            return None
        if not all(col in df_intraday.columns for col in required_cols_60min):
            print(f"      - 提示: {stock_code} 缺少60分钟关键指标列，跳过分析。缺失: {[col for col in required_cols_60min if col not in df_intraday.columns]}")
            return None

        # 遍历每一根分钟K线，寻找交易机会，从足够计算指标的索引开始
        start_idx = max(self.min_data_points_5min, self.min_data_points_30min, self.min_data_points_60min) - 1
        if len(df_intraday) <= start_idx:
            print(f"      - 提示: {stock_code} 数据点不足以计算所有指标 ({len(df_intraday)} <= {start_idx})，跳过分析。")
            return None

        for i in range(start_idx, len(df_intraday)):
            try:
                current_kline = df_intraday.iloc[i]
                prev_kline = df_intraday.iloc[i-1]
                
                # 检查是否在有效交易时间内
                if not (self.trade_start_time <= current_kline.name.time() <= self.trade_end_time):
                    continue

                # --- 按优先级执行剧本 ---
                # 剧本1: 5分钟VWAP突破动能
                vwap_breakout_signal = self._check_5min_vwap_breakout(stock_code, current_kline, prev_kline)
                if vwap_breakout_signal:
                    print(f"      - [信号触发!] {vwap_breakout_signal['reason']}")
                    return vwap_breakout_signal

                # 剧本2: 5分钟布林带突破
                bollinger_breakout_signal = self._check_5min_bollinger_breakout(stock_code, current_kline, prev_kline)
                if bollinger_breakout_signal:
                    print(f"      - [信号触发!] {bollinger_breakout_signal['reason']}")
                    return bollinger_breakout_signal

                # 剧本3: 30分钟EMA多头排列突破
                ema_bullish_signal = self._check_30min_ema_bullish_breakout(stock_code, current_kline, prev_kline)
                if ema_bullish_signal:
                    print(f"      - [信号触发!] {ema_bullish_signal['reason']}")
                    return ema_bullish_signal

                # 剧本4: 盘中回调支撑反弹 (5分钟VWAP)
                pullback_rebound_signal = self._check_5min_pullback_rebound(stock_code, current_kline, prev_kline)
                if pullback_rebound_signal:
                    print(f"      - [信号触发!] {pullback_rebound_signal['reason']}")
                    return pullback_rebound_signal
            
            # 增加更具体的KeyError处理
            except KeyError as ke:
                print(f"      - 错误: 在处理 {stock_code} 第 {i} 行K线时缺少键: {ke}。请检查数据列是否完整。")
                traceback.print_exc()
                return None
            except Exception as e:
                print(f"      - 错误: 在处理 {stock_code} 第 {i} 行K线时发生异常: {e}")
                traceback.print_exc()
                # 发生一次错误后，为安全起见，终止对该股票的分析
                return None
        
        print(f"    -> [盘中策略引擎] {stock_code} 分析完成，未触发任何信号。")
        return None

    # 新增剧本1: 5分钟VWAP突破动能
    def _check_5min_vwap_breakout(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> Optional[Dict]:
        """
        剧本1: 5分钟VWAP突破动能。
        寻找价格在放量推动下，从下方突破5分钟VWAP的信号。
        """
        try:
            # 条件1: 价格刚刚从下向上突破5分钟VWAP
            cond_price_break = kline['close_5'] > kline['vwap_5'] and prev_kline['close_5'] <= prev_kline['vwap_5']
            
            # 条件2: 成交量显著放大 (高于21周期成交量均线)
            # 从嵌套字典获取参数
            volume_multiplier = self.params.get('playbooks', {}).get('vwap_breakout', {}).get('vwap_breakout_volume_multiplier', 1.5)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            
            # 条件3: 突破时的涨幅不能太小 (避免微小波动)
            # 从嵌套字典获取参数
            min_pct_change = self.params.get('playbooks', {}).get('vwap_breakout', {}).get('vwap_breakout_min_pct_change', 0.005)
            cond_pct_change = (kline['close_5'] / prev_kline['close_5'] - 1) > min_pct_change

            if all([cond_price_break, cond_volume, cond_pct_change]):
                return {
                    "stock_code": stock_code, # 使用传入的 stock_code
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "5min VWAP Breakout",
                    "reason": f"5分钟VWAP突破 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_5min_vwap_breakout): 检查条件时缺少键: {ke}")
            return None

    # 新增剧本2: 5分钟布林带突破
    def _check_5min_bollinger_breakout(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> Optional[Dict]:
        """
        剧本2: 5分钟布林带突破。
        寻找价格突破布林带上轨，伴随放量的信号。
        """
        try:
            # 条件1: 价格突破布林带上轨
            cond_price_break = kline['close_5'] > kline['BBU_21_2.0_5']
            
            # 条件2: 成交量显著放大
            # 从嵌套字典获取参数
            volume_multiplier = self.params.get('playbooks', {}).get('bollinger_breakout', {}).get('bollinger_breakout_volume_multiplier', 1.8)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            
            # 条件3: 突破前布林带宽度不能过大 (避免在宽幅震荡中追高)
            # BBW_21_2.0_5 是布林带宽度指标
            # 从嵌套字典获取参数
            max_bbw_threshold = self.params.get('playbooks', {}).get('bollinger_breakout', {}).get('bollinger_breakout_max_bbw', 0.02)
            cond_bbw = prev_kline['BBW_21_2.0_5'] < max_bbw_threshold

            if all([cond_price_break, cond_volume, cond_bbw]):
                return {
                    "stock_code": stock_code, # 使用传入的 stock_code
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "5min Bollinger Breakout",
                    "reason": f"5分钟布林带突破 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_5min_bollinger_breakout): 检查条件时缺少键: {ke}")
            return None

    # 剧本3: 30分钟EMA多头排列突破
    def _check_30min_ema_bullish_breakout(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> Optional[Dict]:
        """
        剧本3: 30分钟EMA多头排列突破。
        寻找30分钟级别EMA形成多头排列，且价格突破关键EMA的信号。
        """
        try:
            # 条件1: 30分钟EMA多头排列 (EMA5 > EMA13 > EMA21)
            cond_ema_alignment = (kline['EMA_5_30'] > kline['EMA_13_30'] and
                                  kline['EMA_13_30'] > kline['EMA_21_30'])
            
            # 条件2: 价格刚刚突破30分钟EMA5 (或EMA13)
            cond_price_break = (kline['close_30'] > kline['EMA_5_30'] and
                                prev_kline['close_30'] <= prev_kline['EMA_5_30'])
            
            # 条件3: 30分钟VWAP也向上 (确认趋势)
            cond_vwap_up = kline['close_30'] > kline['vwap_30']

            if all([cond_ema_alignment, cond_price_break, cond_vwap_up]):
                return {
                    "stock_code": stock_code, # 使用传入的 stock_code
                    "entry_time": kline.name,
                    "entry_price": kline['close_30'],
                    "signal_type": "BUY",
                    "playbook": "30min EMA Bullish Breakout",
                    "reason": f"30分钟EMA多头突破 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_30min_ema_bullish_breakout): 检查条件时缺少键: {ke}")
            return None

    # 剧本4: 盘中回调支撑反弹 (5分钟VWAP)
    def _check_5min_pullback_rebound(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> Optional[Dict]:
        """
        剧本4: 盘中回调支撑反弹 (5分钟VWAP)。
        寻找价格回调至5分钟VWAP附近获得支撑并反弹的信号。
        """
        try:
            # 条件1: 价格在VWAP下方或触及VWAP (回调)
            # 允许价格略微跌破VWAP，但不能太远
            # 从嵌套字典获取参数
            vwap_tolerance_pct = self.params.get('playbooks', {}).get('pullback_rebound', {}).get('pullback_vwap_tolerance_pct', 0.002)
            cond_pullback = (kline['low_5'] <= kline['vwap_5'] * (1 + vwap_tolerance_pct) and
                             kline['high_5'] >= kline['vwap_5'] * (1 - vwap_tolerance_pct))
            
            # 条件2: 当前K线收盘价高于开盘价 (反弹迹象)
            cond_rebound_candle = kline['close_5'] > kline['open_5']
            
            # 条件3: 价格最终收在VWAP上方 (确认支撑有效)
            cond_close_above_vwap = kline['close_5'] > kline['vwap_5']
            
            # 条件4: 成交量不能过大，表明是吸筹或自然反弹，而非恐慌性抛售后的反弹
            # 从嵌套字典获取参数
            max_volume_multiplier = self.params.get('playbooks', {}).get('pullback_rebound', {}).get('pullback_max_volume_multiplier', 1.2)
            cond_volume_moderate = kline['volume_5'] < kline['VOL_MA_21_5'] * max_volume_multiplier

            if all([cond_pullback, cond_rebound_candle, cond_close_above_vwap, cond_volume_moderate]):
                return {
                    "stock_code": stock_code, # 使用传入的 stock_code
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "5min VWAP Pullback Rebound",
                    "reason": f"5分钟VWAP回调反弹 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_5min_pullback_rebound): 检查条件时缺少键: {ke}")
            return None










