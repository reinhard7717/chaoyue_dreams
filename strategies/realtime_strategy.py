# 文件: strategies/realtime_strategy.py

import logging
import pandas as pd
import numpy as np #导入 numpy
from typing import Dict, Optional, List
from datetime import time, datetime #导入 datetime
import traceback

#导入自定义模块
from strategies.realtime_modules.intraday_data_aggregator import IntradayDataAggregator #导入
from strategies.realtime_modules.intraday_pattern_recognizer import IntradayPatternRecognizer #导入
from strategies.realtime_modules.intraday_volume_analyzer import IntradayVolumeAnalyzer #导入
from strategies.realtime_modules.intraday_volatility_analyzer import IntradayVolatilityAnalyzer #导入
from strategies.realtime_modules.intraday_vwap_analyzer import IntradayVWAPAnalyzer #导入
from strategies.realtime_modules.intraday_sr_analyzer import IntradaySRAnalyzer #导入
from strategies.realtime_modules.intraday_micro_price_analyzer import IntradayMicroPriceAnalyzer #导入
from strategies.realtime_modules.intraday_multi_timeframe_analyzer import IntradayMultiTimeframeAnalyzer #导入

logger = logging.getLogger("strategy")

class RealtimeStrategy:
    """
    盘中决策引擎。
    基于对Tushare数据源的正确理解，使用可靠的量价指标和五档盘口压力指标，
    执行多个交易剧本，寻找高置信度的盘中交易机会。
    """
    # 修改 __init__ 方法签名，增加 config 和 prev_day_data 参数
    def __init__(self, params: Dict, config: Dict, prev_day_data: Optional[Dict] = None): # 修改行
        """
        初始化策略，所有阈值都应通过参数传入，便于调优。
        Args:
            params (Dict): 包含所有策略参数的字典。
            config (Dict): 完整的策略配置字典。 #参数说明
            prev_day_data (Optional[Dict]): 前一日的OHLC数据，用于计算枢轴点等。 #参数说明
        """
        self.config = config # 修改行
        self.realtime_config = self.config['strategy_params']['trend_follow'].get('realtime_strategy_params', {})
        self.enabled = self.realtime_config.get('enabled', False)
        if not self.enabled:
            print("RealtimeStrategy is disabled in config.")
            return

        self.base_timeframe = self.realtime_config.get('base_timeframe', '1min')
        self.processed_timeframes = self.realtime_config.get('processed_timeframes', ['5min', '30min', '60min'])
        self.data: Dict[str, pd.DataFrame] = {tf: pd.DataFrame() for tf in self.processed_timeframes + [self.base_timeframe]}

        self.daily_high = -np.inf
        self.daily_low = np.inf
        self.prev_day_close = prev_day_data.get('close') if prev_day_data else None
        self.prev_day_high = prev_day_data.get('high') if prev_day_data else None
        self.prev_day_low = prev_day_data.get('low') if prev_day_data else None
        self.pivot_points = {}
        self.current_date = None # 用于跟踪当前交易日

        self._load_realtime_config()
        
        # 初始化数据聚合器和分析模块
        self.data_aggregator = IntradayDataAggregator(self.realtime_config)
        self.pattern_recognizer = IntradayPatternRecognizer(self.realtime_config.get('kline_patterns', {}))
        self.volume_analyzer = IntradayVolumeAnalyzer(self.realtime_config.get('volume_anomalies', {}))
        self.volatility_analyzer = IntradayVolatilityAnalyzer(self.realtime_config.get('volatility_params', {}))
        self.vwap_analyzer = IntradayVWAPAnalyzer(self.realtime_config.get('advanced_vwap_params', {}))
        self.sr_analyzer = IntradaySRAnalyzer(self.realtime_config.get('intraday_sr_params', {}))
        self.sr_analyzer.set_prev_day_data(prev_day_data) # 设置前一日数据
        self.micro_price_analyzer = IntradayMicroPriceAnalyzer(self.realtime_config.get('micro_price_action_params', {}))
        self.multi_timeframe_analyzer = IntradayMultiTimeframeAnalyzer(self.realtime_config)

        print("RealtimeStrategy initialized with config and modules.")

    #方法：加载实时策略配置
    def _load_realtime_config(self): #方法
        self.trade_start_time = datetime.strptime(self.realtime_config.get('trade_start_time', '09:45'), '%H:%M').time()
        self.trade_end_time = datetime.strptime(self.realtime_config.get('trade_end_time', '14:50'), '%H:%M').time()
        self.min_data_points_5min = self.realtime_config.get('min_data_points_5min', 21)
        self.min_data_points_30min = self.realtime_config.get('min_data_points_30min', 34)
        self.min_data_points_60min = self.realtime_config.get('min_data_points_60min', 55)
        self.playbook_params = self.realtime_config.get('playbooks', {}) # 加载所有剧本参数

    #方法：更新盘中数据
    def update_data(self, new_1min_kline: pd.Series): #方法
        """
        接收新的1分钟K线数据，更新内部数据缓冲区，并聚合计算所有时间周期的指标。
        Args:
            new_1min_kline (pd.Series): 包含 'open', 'high', 'low', 'close', 'volume' 的1分钟K线数据。
                                        索引应为 datetime 对象。
        """
        if not self.enabled:
            return
        
        # 更新当日高低点
        if self.current_date is None or new_1min_kline.name.date() != self.current_date:
            self.current_date = new_1min_kline.name.date()
            self.daily_high = -np.inf
            self.daily_low = np.inf
            print(f"  [盘中策略] 新交易日 {self.current_date}，重置当日高低点。")

        self.daily_high = max(self.daily_high, new_1min_kline['high'])
        self.daily_low = min(self.daily_low, new_1min_kline['low'])

        self.data_aggregator.update_1min_data(new_1min_kline)
        self.data = self.data_aggregator.aggregate_and_calculate_indicators()
        # print(f"  [盘中策略] 数据更新至 {new_1min_kline.name}，5min数据量: {len(self.data.get('5min', []))}")

    # 修改 run_strategy 方法签名，不再接收 df_intraday
    def run_strategy(self, stock_code: str, daily_signal_info: Dict) -> Optional[Dict]: # 修改行
        """
        对单个股票执行所有交易剧本，寻找第一个满足条件的入场信号。
        
        Args:
            stock_code (str): 股票代码。 #参数说明
            daily_signal_info (Dict): 盘后信号的关键信息。

        Returns:
            Optional[Dict]: 如果找到入场信号，返回包含详细信息的字典；否则返回None。
        """
        # stock_code = daily_signal_info.get('stock_code', 'Unknown') # 删除行
        print(f"    -> [盘中策略引擎] 开始对 {stock_code} 执行多剧本分析...")
        
        # 检查最细粒度的5分钟数据点是否足够
        df_5min = self.data.get('5min', pd.DataFrame()) # 修改行
        if len(df_5min) < self.min_data_points_5min: # 修改行
            print(f"      - 提示: {stock_code} 5分钟数据点不足 ({len(df_5min)} < {self.min_data_points_5min})，跳过分析。") # 修改行
            return None

        # 确保所有必要的列都存在，增强健壮性
        # 这里的检查可以简化，因为数据聚合器已经负责计算这些指标
        # 只需要检查最新的K线是否存在
        if df_5min.empty or df_5min.iloc[-1].isnull().all():
            print(f"      - 提示: {stock_code} 5分钟最新K线数据为空，跳过分析。")
            return None

        # 获取最新的K线数据
        current_kline_5min = df_5min.iloc[-1]
        prev_kline_5min = df_5min.iloc[-2] if len(df_5min) >= 2 else None
        
        df_30min = self.data.get('30min', pd.DataFrame())
        current_kline_30min = df_30min.iloc[-1] if not df_30min.empty else None
        prev_kline_30min = df_30min.iloc[-2] if len(df_30min) >= 2 else None

        df_60min = self.data.get('60min', pd.DataFrame())
        current_kline_60min = df_60min.iloc[-1] if not df_60min.empty else None
        prev_kline_60min = df_60min.iloc[-2] if len(df_60min) >= 2 else None

        # 检查是否在有效交易时间内
        if not (self.trade_start_time <= current_kline_5min.name.time() <= self.trade_end_time): # 修改行
            # print(f"      - 提示: {stock_code} 当前时间 {current_kline_5min.name.time()} 不在交易时段 {self.trade_start_time}-{self.trade_end_time} 内。")
            return None

        # --- 提取更多对策略有支撑作用的信息 --- #注释
        # 1. 盘中K线形态识别
        intraday_patterns_5min = self.pattern_recognizer.recognize_patterns(df_5min, '5min')
        intraday_patterns_30min = self.pattern_recognizer.recognize_patterns(df_30min, '30min') if current_kline_30min is not None else {}
        
        # 2. 盘中成交量异动分析
        volume_anomalies_5min = self.volume_analyzer.analyze_volume(df_5min, '5min')

        # 3. 盘中波动率分析
        volatility_features_5min = self.volatility_analyzer.analyze_volatility(df_5min, '5min')

        # 4. VWAP相关高级指标
        advanced_vwap_features_5min = self.vwap_analyzer.analyze_vwap(df_5min, '5min')

        # 5. 盘中支撑与阻力位
        sr_features_5min = self.sr_analyzer.analyze_sr_levels(current_kline_5min, '5min', self.playbook_params.get('pivot_point_reversal', {}).get('pivot_tolerance_pct', 0.0015))

        # 6. 价格行为微观结构
        micro_price_features_5min = {}
        if prev_kline_5min is not None:
            micro_price_features_5min = self.micro_price_analyzer.analyze_micro_price_action(current_kline_5min, prev_kline_5min)
        
        # 7. 多周期共振分析
        multi_timeframe_confluence = self.multi_timeframe_analyzer.analyze_confluence(self.data)

        # 将所有特征合并到 current_kline_5min (方便剧本调用)
        current_kline_5min = pd.concat([current_kline_5min, 
                                        pd.Series(intraday_patterns_5min, name=current_kline_5min.name),
                                        pd.Series(volume_anomalies_5min, name=current_kline_5min.name),
                                        pd.Series(volatility_features_5min, name=current_kline_5min.name),
                                        pd.Series(advanced_vwap_features_5min, name=current_kline_5min.name),
                                        pd.Series(sr_features_5min, name=current_kline_5min.name),
                                        pd.Series(micro_price_features_5min, name=current_kline_5min.name),
                                        pd.Series(multi_timeframe_confluence, name=current_kline_5min.name)])

        # --- 按优先级执行剧本 ---
        # 剧本1: 5分钟VWAP突破动能
        vwap_breakout_signal = self._check_5min_vwap_breakout(stock_code, current_kline_5min, prev_kline_5min) # 修改行
        if vwap_breakout_signal:
            print(f"      - [信号触发!] {vwap_breakout_signal['reason']}")
            return vwap_breakout_signal

        # 剧本2: 5分钟布林带突破
        bollinger_breakout_signal = self._check_5min_bollinger_breakout(stock_code, current_kline_5min, prev_kline_5min) # 修改行
        if bollinger_breakout_signal:
            print(f"      - [信号触发!] {bollinger_breakout_signal['reason']}")
            return bollinger_breakout_signal

        # 剧本3: 30分钟EMA多头排列突破
        ema_bullish_signal = self._check_30min_ema_bullish_breakout(stock_code, current_kline_30min, prev_kline_30min) # 修改行
        if ema_bullish_signal:
            print(f"      - [信号触发!] {ema_bullish_signal['reason']}")
            return ema_bullish_signal

        # 剧本4: 盘中回调支撑反弹 (5分钟VWAP)
        pullback_rebound_signal = self._check_5min_pullback_rebound(stock_code, current_kline_5min, prev_kline_5min) # 修改行
        if pullback_rebound_signal:
            print(f"      - [信号触发!] {pullback_rebound_signal['reason']}")
            return pullback_rebound_signal
        
        # --- 新增剧本 --- #注释
        # 剧本5: 盘中K线反转形态
        candlestick_reversal_signal = self._check_intraday_candlestick_reversal(stock_code, current_kline_5min)
        if candlestick_reversal_signal:
            print(f"      - [信号触发!] {candlestick_reversal_signal['reason']}")
            return candlestick_reversal_signal

        # 剧本6: 布林带压缩后放量突破
        bbw_squeeze_breakout_signal = self._check_volume_breakout_with_bbw_squeeze(stock_code, current_kline_5min, df_5min)
        if bbw_squeeze_breakout_signal:
            print(f"      - [信号触发!] {bbw_squeeze_breakout_signal['reason']}")
            return bbw_squeeze_breakout_signal

        # 剧本7: VWAP通道支撑反弹
        vwap_channel_rebound_signal = self._check_vwap_channel_rebound(stock_code, current_kline_5min)
        if vwap_channel_rebound_signal:
            print(f"      - [信号触发!] {vwap_channel_rebound_signal['reason']}")
            return vwap_channel_rebound_signal

        # 剧本8: 多周期EMA共振突破
        multi_ema_confluence_signal = self._check_multi_timeframe_ema_confluence(stock_code, current_kline_5min)
        if multi_ema_confluence_signal:
            print(f"      - [信号触发!] {multi_ema_confluence_signal['reason']}")
            return multi_ema_confluence_signal

        # 剧本9: 枢轴点支撑反转
        pivot_reversal_signal = self._check_pivot_point_reversal(stock_code, current_kline_5min)
        if pivot_reversal_signal:
            print(f"      - [信号触发!] {pivot_reversal_signal['reason']}")
            return pivot_reversal_signal

        # 剧本10: 微观价格行为拒绝反弹
        micro_price_rejection_signal = self._check_micro_price_rejection_rebound(stock_code, current_kline_5min)
        if micro_price_rejection_signal:
            print(f"      - [信号触发!] {micro_price_rejection_signal['reason']}")
            return micro_price_rejection_signal
        
        print(f"    -> [盘中策略引擎] {stock_code} 分析完成，未触发任何信号。")
        return None

    #剧本1: 5分钟VWAP突破动能
    def _check_5min_vwap_breakout(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> Optional[Dict]:
        """
        剧本1: 5分钟VWAP突破动能。
        寻找价格在放量推动下，从下方突破5分钟VWAP的信号。
        """
        if not self.playbook_params.get('vwap_breakout', {}).get('enabled', False): return None
        if prev_kline is None: return None
        try:
            # 条件1: 价格刚刚从下向上突破5分钟VWAP
            cond_price_break = kline['close_5'] > kline['vwap_5'] and prev_kline['close_5'] <= prev_kline['vwap_5']
            
            # 条件2: 成交量显著放大 (高于21周期成交量均线)
            # 从嵌套字典获取参数
            volume_multiplier = self.playbook_params.get('vwap_breakout', {}).get('vwap_breakout_volume_multiplier', 1.5) # 修改行
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            
            # 条件3: 突破时的涨幅不能太小 (避免微小波动)
            # 从嵌套字典获取参数
            min_pct_change = self.playbook_params.get('vwap_breakout', {}).get('vwap_breakout_min_pct_change', 0.005) # 修改行
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

    #剧本2: 5分钟布林带突破
    def _check_5min_bollinger_breakout(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> Optional[Dict]:
        """
        剧本2: 5分钟布林带突破。
        寻找价格突破布林带上轨，伴随放量的信号。
        """
        if not self.playbook_params.get('bollinger_breakout', {}).get('enabled', False): return None
        if prev_kline is None: return None
        try:
            # 条件1: 价格突破布林带上轨
            cond_price_break = kline['close_5'] > kline['BBU_21_2.0_5']
            
            # 条件2: 成交量显著放大
            # 从嵌套字典获取参数
            volume_multiplier = self.playbook_params.get('bollinger_breakout', {}).get('bollinger_breakout_volume_multiplier', 1.8) # 修改行
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            
            # 条件3: 突破前布林带宽度不能过大 (避免在宽幅震荡中追高)
            # BBW_21_2.0_5 是布林带宽度指标
            # 从嵌套字典获取参数
            max_bbw_threshold = self.playbook_params.get('bollinger_breakout', {}).get('bollinger_breakout_max_bbw', 0.02) # 修改行
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
        if not self.playbook_params.get('ema_bullish_breakout_30min', {}).get('enabled', False): return None
        if kline is None or prev_kline is None: return None
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
        if not self.playbook_params.get('pullback_rebound', {}).get('enabled', False): return None
        if prev_kline is None: return None
        try:
            # 条件1: 价格在VWAP下方或触及VWAP (回调)
            # 允许价格略微跌破VWAP，但不能太远
            # 从嵌套字典获取参数
            vwap_tolerance_pct = self.playbook_params.get('pullback_rebound', {}).get('pullback_vwap_tolerance_pct', 0.002) # 修改行
            cond_pullback = (kline['low_5'] <= kline['vwap_5'] * (1 + vwap_tolerance_pct) and
                             kline['high_5'] >= kline['vwap_5'] * (1 - vwap_tolerance_pct))
            
            # 条件2: 当前K线收盘价高于开盘价 (反弹迹象)
            cond_rebound_candle = kline['close_5'] > kline['open_5']
            
            # 条件3: 价格最终收在VWAP上方 (确认支撑有效)
            cond_close_above_vwap = kline['close_5'] > kline['vwap_5']
            
            # 条件4: 成交量不能过大，表明是吸筹或自然反弹，而非恐慌性抛售后的反弹
            # 从嵌套字典获取参数
            max_volume_multiplier = self.playbook_params.get('pullback_rebound', {}).get('pullback_max_volume_multiplier', 1.2) # 修改行
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

    #剧本5: 盘中K线反转形态
    def _check_intraday_candlestick_reversal(self, stock_code: str, kline: pd.Series) -> Optional[Dict]: #方法
        """
        剧本5: 盘中K线反转形态。
        寻找5分钟K线出现看涨反转形态，并伴随一定放量和涨幅。
        """
        if not self.playbook_params.get('intraday_candlestick_reversal', {}).get('enabled', False): return None
        
        try:
            # 条件1: 识别到看涨K线反转形态 (例如锤头线、看涨吞没、启明星)
            cond_bullish_pattern = kline.get("CDL_HAMMER", False) or \
                                   kline.get("CDL_ENGULFING_BULLISH", False) or \
                                   kline.get("CDL_MORNINGSTAR", False)
            
            if not cond_bullish_pattern: return None

            # 条件2: 成交量放大 (高于均量)
            volume_multiplier = self.playbook_params.get('intraday_candlestick_reversal', {}).get('min_reversal_volume_multiplier', 1.2)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier

            # 条件3: K线本身有一定涨幅 (确认反转力量)
            min_pct_change = self.playbook_params.get('intraday_candlestick_reversal', {}).get('min_reversal_pct_change', 0.008)
            cond_pct_change = (kline['close_5'] / kline['open_5'] - 1) > min_pct_change

            if all([cond_bullish_pattern, cond_volume, cond_pct_change]):
                pattern_name = "未知反转形态"
                if kline.get("CDL_HAMMER"): pattern_name = "锤头线"
                elif kline.get("CDL_ENGULFING_BULLISH"): pattern_name = "看涨吞没"
                elif kline.get("CDL_MORNINGSTAR"): pattern_name = "启明星"

                return {
                    "stock_code": stock_code,
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "Intraday Candlestick Reversal",
                    "reason": f"5分钟K线看涨反转 ({pattern_name}) @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_intraday_candlestick_reversal): 检查条件时缺少键: {ke}")
            return None

    #剧本6: 布林带压缩后放量突破
    def _check_volume_breakout_with_bbw_squeeze(self, stock_code: str, kline: pd.Series, df_5min: pd.DataFrame) -> Optional[Dict]: #方法
        """
        剧本6: 布林带压缩后放量突破。
        寻找布林带宽度长时间收缩后，价格伴随巨量突破布林带上轨的信号。
        """
        if not self.playbook_params.get('volume_breakout_with_bbw_squeeze', {}).get('enabled', False): return None
        
        try:
            # 条件1: 布林带宽度处于低位 (压缩状态)
            bbw_col = f"BBW_21_2.0_5" # 假设布林带参数为21, 2.0
            if bbw_col not in df_5min.columns or df_5min[bbw_col].isnull().all(): return None
            
            bbw_squeeze_quantile = self.playbook_params.get('volume_breakout_with_bbw_squeeze', {}).get('bbw_squeeze_quantile', 0.2)
            # 检查当前BBW是否低于历史某个分位数 (例如过去N根K线的20%分位数)
            if len(df_5min) < 60: return None # 至少需要60根K线来计算分位数
            cond_bbw_squeezed = kline[bbw_col] < df_5min[bbw_col].iloc[-60:].quantile(bbw_squeeze_quantile)

            # 条件2: 价格突破布林带上轨
            cond_price_break = kline['close_5'] > kline['BBU_21_2.0_5']

            # 条件3: 伴随巨量
            volume_multiplier = self.playbook_params.get('volume_breakout_with_bbw_squeeze', {}).get('volume_multiplier', 2.0)
            cond_giant_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier

            if all([cond_bbw_squeezed, cond_price_break, cond_giant_volume]):
                return {
                    "stock_code": stock_code,
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "BBW Squeeze Volume Breakout",
                    "reason": f"5分钟布林带压缩后放量突破 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_volume_breakout_with_bbw_squeeze): 检查条件时缺少键: {ke}")
            return None

    #剧本7: VWAP通道支撑反弹
    def _check_vwap_channel_rebound(self, stock_code: str, kline: pd.Series) -> Optional[Dict]: #方法
        """
        剧本7: VWAP通道支撑反弹。
        寻找价格回调至VWAP下通道线附近获得支撑并反弹的信号。
        """
        if not self.playbook_params.get('vwap_channel_rebound', {}).get('enabled', False): return None
        
        try:
            # 条件1: 价格触及VWAP下通道线并反弹
            cond_touch_rebound = kline.get("PRICE_TOUCHING_VWAP_LOWER_CHANNEL", False)
            
            # 条件2: 当前K线收阳线
            cond_rebound_candle = kline['close_5'] > kline['open_5']

            # 条件3: 反弹幅度不能太小
            rebound_min_pct_change = self.playbook_params.get('vwap_channel_rebound', {}).get('rebound_min_pct_change', 0.003)
            cond_pct_change = (kline['close_5'] / kline['open_5'] - 1) > rebound_min_pct_change

            if all([cond_touch_rebound, cond_rebound_candle, cond_pct_change]):
                return {
                    "stock_code": stock_code,
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "VWAP Channel Rebound",
                    "reason": f"5分钟VWAP通道支撑反弹 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_vwap_channel_rebound): 检查条件时缺少键: {ke}")
            return None

    #剧本8: 多周期EMA共振突破
    def _check_multi_timeframe_ema_confluence(self, stock_code: str, kline: pd.Series) -> Optional[Dict]: #方法
        """
        剧本8: 多周期EMA共振突破。
        寻找5分钟、30分钟、60分钟EMA均形成多头排列，且5分钟价格突破短期EMA的信号。
        """
        if not self.playbook_params.get('multi_timeframe_ema_confluence', {}).get('enabled', False): return None
        
        try:
            # 条件1: 多周期EMA多头排列共振 (由 IntradayMultiTimeframeAnalyzer 提供)
            cond_ema_confluence = kline.get("EMA_BULLISH_CONFLUENCE", False)
            
            # 条件2: 5分钟价格突破短期EMA (例如EMA5)
            cond_price_break_5min = kline['close_5'] > kline['EMA_5_5'] and kline['open_5'] <= kline['EMA_5_5'] # 假设open_5是前一根K线的close_5

            if all([cond_ema_confluence, cond_price_break_5min]):
                return {
                    "stock_code": stock_code,
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "Multi-timeframe EMA Confluence",
                    "reason": f"多周期EMA共振突破 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_multi_timeframe_ema_confluence): 检查条件时缺少键: {ke}")
            return None

    #剧本9: 枢轴点支撑反转
    def _check_pivot_point_reversal(self, stock_code: str, kline: pd.Series) -> Optional[Dict]: #方法
        """
        剧本9: 枢轴点支撑反转。
        寻找价格在枢轴点S1/S2/S3附近获得支撑并反弹的信号。
        """
        if not self.playbook_params.get('pivot_point_reversal', {}).get('enabled', False): return None
        
        try:
            # 条件1: 价格在S1/S2/S3附近反弹 (由 IntradaySRAnalyzer 提供)
            cond_rebound_from_pivot = kline.get("PRICE_REBOUNDING_FROM_S1", False) or \
                                      kline.get("PRICE_REBOUNDING_FROM_S2", False) or \
                                      kline.get("PRICE_REBOUNDING_FROM_S3", False)
            
            if not cond_rebound_from_pivot: return None

            # 条件2: 当前K线收阳线
            cond_rebound_candle = kline['close_5'] > kline['open_5']

            # 条件3: 伴随一定成交量 (确认反弹强度)
            volume_multiplier = self.playbook_params.get('pivot_point_reversal', {}).get('rebound_volume_multiplier', 1.2)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier

            if all([cond_rebound_from_pivot, cond_rebound_candle, cond_volume]):
                pivot_level = "S1" if kline.get("PRICE_REBOUNDING_FROM_S1") else \
                              ("S2" if kline.get("PRICE_REBOUNDING_FROM_S2") else "S3")
                return {
                    "stock_code": stock_code,
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "Pivot Point Reversal",
                    "reason": f"枢轴点 {pivot_level} 支撑反弹 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_pivot_point_reversal): 检查条件时缺少键: {ke}")
            return None

    #剧本10: 微观价格行为拒绝反弹
    def _check_micro_price_rejection_rebound(self, stock_code: str, kline: pd.Series) -> Optional[Dict]: #方法
        """
        剧本10: 微观价格行为拒绝反弹。
        寻找价格向下拒绝（长下影线）后，收盘价反弹的信号。
        """
        if not self.playbook_params.get('micro_price_rejection_rebound', {}).get('enabled', False): return None
        
        try:
            # 条件1: 价格向下拒绝 (长下影线，收盘价远离低点)
            cond_price_rejection_lower = kline.get("PRICE_REJECTION_LOWER", False)
            
            if not cond_price_rejection_lower: return None

            # 条件2: 当前K线收阳线
            cond_rebound_candle = kline['close_5'] > kline['open_5']

            # 条件3: 伴随一定成交量 (确认反弹强度)
            volume_multiplier = self.playbook_params.get('micro_price_rejection_rebound', {}).get('rejection_volume_multiplier', 1.1)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier

            if all([cond_price_rejection_lower, cond_rebound_candle, cond_volume]):
                return {
                    "stock_code": stock_code,
                    "entry_time": kline.name,
                    "entry_price": kline['close_5'],
                    "signal_type": "BUY",
                    "playbook": "Micro Price Rejection Rebound",
                    "reason": f"微观价格行为向下拒绝反弹 @ {kline.name.time()}"
                }
            return None
        except KeyError as ke:
            print(f"      - 错误 (_check_micro_price_rejection_rebound): 检查条件时缺少键: {ke}")
            return None




















