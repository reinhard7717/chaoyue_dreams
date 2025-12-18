# 文件: strategies/realtime_strategy.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple # 导入 Tuple
from datetime import time, datetime
import traceback

# 导入自定义模块
from strategies.realtime_modules.intraday_data_aggregator import IntradayDataAggregator
from strategies.realtime_modules.intraday_pattern_recognizer import IntradayPatternRecognizer
from strategies.realtime_modules.intraday_volume_analyzer import IntradayVolumeAnalyzer
from strategies.realtime_modules.intraday_volatility_analyzer import IntradayVolatilityAnalyzer
from strategies.realtime_modules.intraday_vwap_analyzer import IntradayVWAPAnalyzer
from strategies.realtime_modules.intraday_sr_analyzer import IntradaySRAnalyzer
from strategies.realtime_modules.intraday_micro_price_analyzer import IntradayMicroPriceAnalyzer
from strategies.realtime_modules.intraday_multi_timeframe_analyzer import IntradayMultiTimeframeAnalyzer

logger = logging.getLogger("strategy")

class RealtimeStrategy:
    """
    盘中决策引擎。
    基于对Tushare数据源的正确理解，使用可靠的量价指标和五档盘口压力指标，
    执行多个交易剧本，寻找高置信度的盘中交易机会。
    """
    def __init__(self, params: Dict, config: Dict, prev_day_data: Optional[Dict] = None):
        """
        初始化策略，所有阈值都应通过参数传入，便于调优。
        Args:
            params (Dict): 包含所有策略参数的字典。
            config (Dict): 完整的策略配置字典。
            prev_day_data (Optional[Dict]): 前一日的OHLC数据，用于计算枢轴点等。
        """
        self.config = config
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
        self.current_date = None
        self._load_realtime_config()
        # 初始化数据聚合器和分析模块
        self.data_aggregator = IntradayDataAggregator(self.realtime_config)
        self.pattern_recognizer = IntradayPatternRecognizer(self.realtime_config.get('kline_patterns', {}))
        self.volume_analyzer = IntradayVolumeAnalyzer(self.realtime_config.get('volume_anomalies', {}))
        self.volatility_analyzer = IntradayVolatilityAnalyzer(self.realtime_config.get('volatility_params', {}))
        self.vwap_analyzer = IntradayVWAPAnalyzer(self.realtime_config.get('advanced_vwap_params', {}))
        self.sr_analyzer = IntradaySRAnalyzer(self.realtime_config.get('intraday_sr_params', {}))
        self.sr_analyzer.set_prev_day_data(prev_day_data)
        self.micro_price_analyzer = IntradayMicroPriceAnalyzer(self.realtime_config.get('micro_price_action_params', {}))
        self.multi_timeframe_analyzer = IntradayMultiTimeframeAnalyzer(self.realtime_config)
        print("RealtimeStrategy initialized with config and modules.")
    def _load_realtime_config(self):
        self.trade_start_time = datetime.strptime(self.realtime_config.get('trade_start_time', '09:45'), '%H:%M').time()
        self.trade_end_time = datetime.strptime(self.realtime_config.get('trade_end_time', '14:50'), '%H:%M').time()
        self.min_data_points_5min = self.realtime_config.get('min_data_points_5min', 21)
        self.min_data_points_30min = self.realtime_config.get('min_data_points_30min', 34)
        self.min_data_points_60min = self.realtime_config.get('min_data_points_60min', 55)
        self.playbook_params = self.realtime_config.get('playbooks', {})
        # 加载盘中评分参数
        self.intraday_scoring_params = self.realtime_config.get('intraday_scoring_params', {})
        self.base_score_per_playbook = self.intraday_scoring_params.get('base_score_per_playbook', {})
        self.tiered_feature_scoring = self.intraday_scoring_params.get('tiered_feature_scoring', {}) # 加载分层特征评分配置
        self.fixed_feature_scoring = self.intraday_scoring_params.get('fixed_feature_scoring', {}) # 加载固定特征评分配置
        self.rating_thresholds = self.intraday_scoring_params.get('rating_thresholds', {})
        self.daily_score_influence_multiplier = self.intraday_scoring_params.get('daily_score_influence_multiplier', {}).get('value', 0.5)
        self.daily_risk_penalty_multiplier = self.intraday_scoring_params.get('daily_risk_penalty_multiplier', {}).get('value', 0.8)
    def update_data(self, new_1min_kline: pd.Series):
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
    def run_strategy(self, stock_code: str, daily_signal_info: Dict) -> Optional[Dict]:
        """
        对单个股票执行所有交易剧本，寻找第一个满足条件的入场信号。
        Args:
            stock_code (str): 股票代码。
            daily_signal_info (Dict): 盘后信号的关键信息。
        Returns:
            Optional[Dict]: 如果找到入场信号，返回包含详细信息的字典；否则返回None。
        """
        print(f"    -> [盘中策略引擎] 开始对 {stock_code} 执行多剧本分析...")
        df_5min = self.data.get('5min', pd.DataFrame())
        if len(df_5min) < self.min_data_points_5min:
            print(f"      - 提示: {stock_code} 5分钟数据点不足 ({len(df_5min)} < {self.min_data_points_5min})，跳过分析。")
            return None
        if df_5min.empty or df_5min.iloc[-1].isnull().all():
            print(f"      - 提示: {stock_code} 5分钟最新K线数据为空，跳过分析。")
            return None
        current_kline_5min = df_5min.iloc[-1]
        prev_kline_5min = df_5min.iloc[-2] if len(df_5min) >= 2 else None
        df_30min = self.data.get('30min', pd.DataFrame())
        current_kline_30min = df_30min.iloc[-1] if not df_30min.empty else None
        prev_kline_30min = df_30min.iloc[-2] if len(df_30min) >= 2 else None
        df_60min = self.data.get('60min', pd.DataFrame())
        current_kline_60min = df_60min.iloc[-1] if not df_60min.empty else None
        prev_kline_60min = df_60min.iloc[-2] if len(df_60min) >= 2 else None
        if not (self.trade_start_time <= current_kline_5min.name.time() <= self.trade_end_time):
            return None
        try:
            # --- 提取更多对策略有支撑作用的信息 ---
            # 各分析器现在返回 Dict[str, float]，包含量化值
            intraday_patterns_5min = self.pattern_recognizer.recognize_patterns(df_5min, '5min')
            intraday_patterns_30min = self.pattern_recognizer.recognize_patterns(df_30min, '30min') if current_kline_30min is not None else {}
            volume_anomalies_5min = self.volume_analyzer.analyze_volume(df_5min, '5min')
            volatility_features_5min = self.volatility_analyzer.analyze_volatility(df_5min, '5min')
            advanced_vwap_features_5min = self.vwap_analyzer.analyze_vwap(df_5min, '5min')
            sr_features_5min = self.sr_analyzer.analyze_sr_levels(current_kline_5min, '5min', self.playbook_params.get('pivot_point_reversal', {}).get('pivot_tolerance_pct', 0.0015))
            micro_price_features_5min = {}
            if prev_kline_5min is not None:
                micro_price_features_5min = self.micro_price_analyzer.analyze_micro_price_action(current_kline_5min, prev_kline_5min)
            multi_timeframe_confluence = self.multi_timeframe_analyzer.analyze_confluence(self.data)
            # 将所有特征合并到一个字典，方便传递给评分函数
            # all_intraday_features 现在包含量化值
            all_intraday_features = {
                **intraday_patterns_5min,
                **intraday_patterns_30min,
                **volume_anomalies_5min,
                **volatility_features_5min,
                **advanced_vwap_features_5min,
                **sr_features_5min,
                **micro_price_features_5min,
                **multi_timeframe_confluence
            }
            # --- 运行所有剧本并收集触发的剧本名称 ---
            triggered_playbooks = []
            # 剧本1: 5分钟VWAP突破动能
            if self._check_5min_vwap_breakout(stock_code, current_kline_5min, prev_kline_5min):
                triggered_playbooks.append("5min VWAP Breakout")
            # 剧本2: 5分钟布林带突破
            if self._check_5min_bollinger_breakout(stock_code, current_kline_5min, prev_kline_5min):
                triggered_playbooks.append("5min Bollinger Breakout")
            # 剧本3: 30分钟EMA多头排列突破
            if self._check_30min_ema_bullish_breakout(stock_code, current_kline_30min, prev_kline_30min):
                triggered_playbooks.append("30min EMA Bullish Breakout")
            # 剧本4: 盘中回调支撑反弹 (5分钟VWAP)
            if self._check_5min_pullback_rebound(stock_code, current_kline_5min, prev_kline_5min):
                triggered_playbooks.append("5min VWAP Pullback Rebound")
            # 剧本5: 盘中K线反转形态
            if self._check_intraday_candlestick_reversal(stock_code, current_kline_5min):
                triggered_playbooks.append("Intraday Candlestick Reversal")
            # 剧本6: 布林带压缩后放量突破
            if self._check_volume_breakout_with_bbw_squeeze(stock_code, current_kline_5min, df_5min):
                triggered_playbooks.append("BBW Squeeze Volume Breakout")
            # 剧本7: VWAP通道支撑反弹
            if self._check_vwap_channel_rebound(stock_code, current_kline_5min):
                triggered_playbooks.append("VWAP Channel Rebound")
            # 剧本8: 多周期EMA共振突破
            if self._check_multi_timeframe_ema_confluence(stock_code, current_kline_5min):
                triggered_playbooks.append("Multi-timeframe EMA Confluence")
            # 剧本9: 枢轴点支撑反转
            if self._check_pivot_point_reversal(stock_code, current_kline_5min):
                triggered_playbooks.append("Pivot Point Reversal")
            # 剧本10: 微观价格行为拒绝反弹
            if self._check_micro_price_rejection_rebound(stock_code, current_kline_5min):
                triggered_playbooks.append("Micro Price Rejection Rebound")
            # --- 综合评分和评级 ---
            intraday_score, intraday_rating, reason = self._calculate_intraday_rating(
                stock_code, current_kline_5min, all_intraday_features, triggered_playbooks, daily_signal_info
            )
            if intraday_rating in ["STRONG_BUY", "BUY"]:
                print(f"      - [盘中评级] {stock_code} 评级: {intraday_rating} (分数: {intraday_score}) - {reason}")
                return {
                    "stock_code": stock_code,
                    "entry_time": current_kline_5min.name,
                    "entry_price": current_kline_5min['close_5'],
                    "signal_type": "BUY",
                    "playbook": "Intraday Composite Signal",
                    "reason": reason,
                    "intraday_score": intraday_score,
                    "intraday_rating": intraday_rating
                }
            elif intraday_rating in ["SELL", "STRONG_SELL"]:
                print(f"      - [盘中评级] {stock_code} 评级: {intraday_rating} (分数: {intraday_score}) - {reason}")
                return {
                    "stock_code": stock_code,
                    "exit_time": current_kline_5min.name,
                    "exit_price": current_kline_5min['close_5'],
                    "signal_type": "SELL",
                    "playbook": "Intraday Composite Exit Signal",
                    "reason": reason,
                    "intraday_score": intraday_score,
                    "intraday_rating": intraday_rating
                }
            else:
                # print(f"      - [盘中评级] {stock_code} 评级: {intraday_rating} (分数: {intraday_score})")
                return None
        except KeyError as ke:
            print(f"      - 错误: 在处理 {stock_code} K线时缺少键: {ke}。请检查数据列是否完整。")
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"      - 错误: 在处理 {stock_code} K线时发生异常: {e}")
            traceback.print_exc()
            return None
    #剧本1: 5分钟VWAP突破动能
    def _check_5min_vwap_breakout(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> bool:
        """
        剧本1: 5分钟VWAP突破动能。
        寻找价格在放量推动下，从下方突破5分钟VWAP的信号。
        """
        if not self.playbook_params.get('vwap_breakout', {}).get('enabled', False): return False
        if prev_kline is None: return False
        try:
            cond_price_break = kline['close_5'] > kline['vwap_5'] and prev_kline['close_5'] <= prev_kline['vwap_5']
            volume_multiplier = self.playbook_params.get('vwap_breakout', {}).get('vwap_breakout_volume_multiplier', 1.5)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            min_pct_change = self.playbook_params.get('vwap_breakout', {}).get('vwap_breakout_min_pct_change', 0.005)
            cond_pct_change = (kline['close_5'] / prev_kline['close_5'] - 1) > min_pct_change
            return all([cond_price_break, cond_volume, cond_pct_change])
        except KeyError as ke:
            return False
    # 剧本2: 5分钟布林带突破
    def _check_5min_bollinger_breakout(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> bool:
        """
        剧本2: 5分钟布林带突破。
        寻找价格突破布林带上轨，伴随放量的信号。
        """
        if not self.playbook_params.get('bollinger_breakout', {}).get('enabled', False): return False
        if prev_kline is None: return False
        try:
            cond_price_break = kline['close_5'] > kline['BBU_21_2.0_5']
            volume_multiplier = self.playbook_params.get('bollinger_breakout', {}).get('bollinger_breakout_volume_multiplier', 1.8)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            max_bbw_threshold = self.playbook_params.get('bollinger_breakout', {}).get('bollinger_breakout_max_bbw', 0.02)
            cond_bbw = prev_kline['BBW_21_2.0_5'] < max_bbw_threshold
            return all([cond_price_break, cond_volume, cond_bbw])
        except KeyError as ke:
            return False
    # 剧本3: 30分钟EMA多头排列突破
    def _check_30min_ema_bullish_breakout(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> bool:
        """
        剧本3: 30分钟EMA多头排列突破。
        寻找30分钟级别EMA形成多头排列，且价格突破关键EMA的信号。
        """
        if not self.playbook_params.get('ema_bullish_breakout_30min', {}).get('enabled', False): return False
        if kline is None or prev_kline is None: return False
        try:
            cond_ema_alignment = (kline['EMA_5_30'] > kline['EMA_13_30'] and
                                  kline['EMA_13_30'] > kline['EMA_21_30'])
            cond_price_break = (kline['close_30'] > kline['EMA_5_30'] and
                                prev_kline['close_30'] <= prev_kline['EMA_5_30'])
            cond_vwap_up = kline['close_30'] > kline['vwap_30']
            return all([cond_ema_alignment, cond_price_break, cond_vwap_up])
        except KeyError as ke:
            return False
    # 剧本4: 盘中回调支撑反弹 (5分钟VWAP)
    def _check_5min_pullback_rebound(self, stock_code: str, kline: pd.Series, prev_kline: pd.Series) -> bool:
        """
        剧本4: 盘中回调支撑反弹 (5分钟VWAP)。
        寻找价格回调至5分钟VWAP附近获得支撑并反弹的信号。
        """
        if not self.playbook_params.get('pullback_rebound', {}).get('enabled', False): return False
        if prev_kline is None: return False 
        try:
            vwap_tolerance_pct = self.playbook_params.get('pullback_rebound', {}).get('pullback_vwap_tolerance_pct', 0.002)
            cond_pullback = (kline['low_5'] <= kline['vwap_5'] * (1 + vwap_tolerance_pct) and
                             kline['high_5'] >= kline['vwap_5'] * (1 - vwap_tolerance_pct))
            cond_rebound_candle = kline['close_5'] > kline['open_5']
            cond_close_above_vwap = kline['close_5'] > kline['vwap_5']
            max_volume_multiplier = self.playbook_params.get('pullback_rebound', {}).get('pullback_max_volume_multiplier', 1.2)
            cond_volume_moderate = kline['volume_5'] < kline['VOL_MA_21_5'] * max_volume_multiplier
            return all([cond_pullback, cond_rebound_candle, cond_close_above_vwap, cond_volume_moderate])
        except KeyError as ke:
            return False
    # 新增剧本5: 盘中K线反转形态
    def _check_intraday_candlestick_reversal(self, stock_code: str, kline: pd.Series) -> bool:
        """
        剧本5: 盘中K线反转形态。
        寻找5分钟K线出现看涨反转形态，并伴随一定放量和涨幅。
        """
        if not self.playbook_params.get('intraday_candlestick_reversal', {}).get('enabled', False): return False
        try:
            cond_bullish_pattern = kline.get("CDL_HAMMER", 0.0) > 0.5 or \
                                   kline.get("CDL_ENGULFING_BULLISH", 0.0) > 0.5 or \
                                   kline.get("CDL_MORNINGSTAR", 0.0) > 0.5 # 使用get获取浮点值并判断
            if not cond_bullish_pattern: return False
            volume_multiplier = self.playbook_params.get('intraday_candlestick_reversal', {}).get('min_reversal_volume_multiplier', 1.2)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            min_pct_change = self.playbook_params.get('intraday_candlestick_reversal', {}).get('min_reversal_pct_change', 0.008)
            cond_pct_change = (kline['close_5'] / kline['open_5'] - 1) > min_pct_change
            return all([cond_bullish_pattern, cond_volume, cond_pct_change])
        except KeyError as ke:
            return False
    # 新增剧本6: 布林带压缩后放量突破
    def _check_volume_breakout_with_bbw_squeeze(self, stock_code: str, kline: pd.Series, df_5min: pd.DataFrame) -> bool:
        """
        剧本6: 布林带压缩后放量突破。
        寻找布林带宽度长时间收缩后，价格伴随巨量突破布林带上轨的信号。
        """
        if not self.playbook_params.get('volume_breakout_with_bbw_squeeze', {}).get('enabled', False): return False
        try:
            bbw_col = f"BBW_21_2.0_5"
            if bbw_col not in df_5min.columns or df_5min[bbw_col].isnull().all(): return False
            bbw_squeeze_quantile = self.playbook_params.get('volume_breakout_with_bbw_squeeze', {}).get('bbw_squeeze_quantile', 0.2)
            if len(df_5min) < 60: return False
            cond_bbw_squeezed = kline[bbw_col] < df_5min[bbw_col].iloc[-60:].quantile(bbw_squeeze_quantile)
            cond_price_break = kline['close_5'] > kline['BBU_21_2.0_5']
            volume_multiplier = self.playbook_params.get('volume_breakout_with_bbw_squeeze', {}).get('volume_multiplier', 2.0)
            cond_giant_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            return all([cond_bbw_squeezed, cond_price_break, cond_giant_volume])
        except KeyError as ke:
            return False
    # 新增剧本7: VWAP通道支撑反弹
    def _check_vwap_channel_rebound(self, stock_code: str, kline: pd.Series) -> bool:
        """
        剧本7: VWAP通道支撑反弹。
        寻找价格回调至VWAP下通道线附近获得支撑并反弹的信号。
        """
        if not self.playbook_params.get('vwap_channel_rebound', {}).get('enabled', False): return False
        try:
            cond_touch_rebound = kline.get("PRICE_TOUCHING_VWAP_LOWER_CHANNEL", 0.0) > 0.5 # 使用get获取浮点值并判断
            cond_rebound_candle = kline['close_5'] > kline['open_5']
            rebound_min_pct_change = self.playbook_params.get('vwap_channel_rebound', {}).get('rebound_min_pct_change', 0.003)
            cond_pct_change = (kline['close_5'] / kline['open_5'] - 1) > rebound_min_pct_change
            return all([cond_touch_rebound, cond_rebound_candle, cond_pct_change])
        except KeyError as ke:
            return False
    # 新增剧本8: 多周期EMA共振突破
    def _check_multi_timeframe_ema_confluence(self, stock_code: str, kline: pd.Series) -> bool:
        """
        剧本8: 多周期EMA共振突破。
        寻找5分钟、30分钟、60分钟EMA均形成多头排列，且5分钟价格突破短期EMA的信号。
        """
        if not self.playbook_params.get('multi_timeframe_ema_confluence', {}).get('enabled', False): return False
        try:
            cond_ema_confluence = kline.get("EMA_BULLISH_CONFLUENCE", 0.0) > 0.5 # 使用get获取浮点值并判断
            cond_price_break_5min = kline['close_5'] > kline['EMA_5_5'] and kline['open_5'] <= kline['EMA_5_5']
            return all([cond_ema_confluence, cond_price_break_5min])
        except KeyError as ke:
            return False
    # 新增剧本9: 枢轴点支撑反转
    def _check_pivot_point_reversal(self, stock_code: str, kline: pd.Series) -> bool:
        """
        剧本9: 枢轴点支撑反转。
        寻找价格在枢轴点S1/S2/S3附近获得支撑并反弹的信号。
        """
        if not self.playbook_params.get('pivot_point_reversal', {}).get('enabled', False): return False
        try:
            cond_rebound_from_pivot = kline.get("PRICE_REBOUNDING_FROM_S1", 0.0) > 0.5 or \
                                      kline.get("PRICE_REBOUNDING_FROM_S2", 0.0) > 0.5 or \
                                      kline.get("PRICE_REBOUNDING_FROM_S3", 0.0) > 0.5 # 使用get获取浮点值并判断
            if not cond_rebound_from_pivot: return False
            cond_rebound_candle = kline['close_5'] > kline['open_5']
            volume_multiplier = self.playbook_params.get('pivot_point_reversal', {}).get('rebound_volume_multiplier', 1.2)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            return all([cond_rebound_from_pivot, cond_rebound_candle, cond_volume])
        except KeyError as ke:
            return False
    # 新增剧本10: 微观价格行为拒绝反弹
    def _check_micro_price_rejection_rebound(self, stock_code: str, kline: pd.Series) -> bool:
        """
        剧本10: 微观价格行为拒绝反弹。
        寻找价格向下拒绝（长下影线）后，收盘价反弹的信号。
        """
        if not self.playbook_params.get('micro_price_rejection_rebound', {}).get('enabled', False): return False
        try:
            cond_price_rejection_lower = kline.get("PRICE_REJECTION_LOWER", 0.0) > 0.5 # 使用get获取浮点值并判断
            if not cond_price_rejection_lower: return False
            cond_rebound_candle = kline['close_5'] > kline['open_5']
            volume_multiplier = self.playbook_params.get('micro_price_rejection_rebound', {}).get('rejection_volume_multiplier', 1.1)
            cond_volume = kline['volume_5'] > kline['VOL_MA_21_5'] * volume_multiplier
            return all([cond_price_rejection_lower, cond_rebound_candle, cond_volume])
        except KeyError as ke:
            return False
    # 修改方法：计算盘中综合评分和评级
    def _calculate_intraday_rating(self, stock_code: str, current_kline_5min: pd.Series,
                                   all_intraday_features: Dict[str, float], # 特征值可以是浮点数
                                   triggered_playbooks: List[str],
                                   daily_signal_info: Dict) -> Tuple[int, str, str]:
        """
        根据所有盘中特征和触发的剧本，计算综合盘中评分并给出评级。
        Args:
            stock_code (str): 股票代码。
            current_kline_5min (pd.Series): 当前5分钟K线数据。
            all_intraday_features (Dict[str, float]): 所有盘中特征的字典，包含量化值。
            triggered_playbooks (List[str]): 触发的剧本名称列表。
            daily_signal_info (Dict): 盘后信号的关键信息。
        Returns:
            Tuple[int, str, str]: (盘中分数, 盘中评级, 评级理由)。
        """
        if not self.intraday_scoring_params.get('enabled', False):
            return 0, "NEUTRAL", "盘中评分系统未启用。"
        intraday_score = 0
        reasons = []
        # 1. 剧本加分
        for playbook_name in triggered_playbooks:
            score = self.base_score_per_playbook.get(playbook_name, 0)
            intraday_score += score
            reasons.append(f"剧本[{playbook_name}]: +{score}")
        # 2. 特征加减分 (分层判定)
        for feature_config_name, config in self.tiered_feature_scoring.items():
            metric_name = config['metric']
            direction = config['direction']
            tiers = config['tiers']
            metric_value = all_intraday_features.get(metric_name)
            if metric_value is None or pd.isna(metric_value):
                continue
            score_added = 0
            reason_detail = ""
            # 根据方向对层级进行排序，以便正确匹配最高阈值
            if direction == "positive": # 值越大越好，从高阈值开始检查
                sorted_tiers = sorted(tiers, key=lambda x: x['threshold'], reverse=True)
                for tier in sorted_tiers:
                    if metric_value >= tier['threshold']:
                        score_added = tier['score']
                        reason_detail = tier['description']
                        break
            elif direction == "negative": # 值越大越差（例如长上影线），从高阈值开始检查
                sorted_tiers = sorted(tiers, key=lambda x: x['threshold'], reverse=True)
                for tier in sorted_tiers:
                    if metric_value >= tier['threshold']:
                        score_added = tier['score']
                        reason_detail = tier['description']
                        break
            elif direction == "positive_inverse": # 值越小越好（例如BBW分位数），从低阈值开始检查
                sorted_tiers = sorted(tiers, key=lambda x: x['threshold'], reverse=False)
                for tier in sorted_tiers:
                    if metric_value <= tier['threshold']:
                        score_added = tier['score']
                        reason_detail = tier['description']
                        break
            elif direction == "dynamic": # 根据K线涨跌动态判断
                is_bullish_candle = current_kline_5min['close_5'] > current_kline_5min['open_5']
                sorted_tiers = sorted(tiers, key=lambda x: x['threshold'], reverse=True)
                for tier in sorted_tiers:
                    if metric_value >= tier['threshold']:
                        score_added = tier['score_up'] if is_bullish_candle else tier['score_down']
                        reason_detail = tier['description']
                        break
            if score_added != 0:
                intraday_score += score_added
                reasons.append(f"特征[{metric_name} - {reason_detail}]: {'+' if score_added > 0 else ''}{score_added}")
        # 3. 特征加减分 (固定判定)
        for feature_name, score in self.fixed_feature_scoring.items():
            # 对于固定特征，我们检查其值是否为1.0 (表示True)
            if all_intraday_features.get(feature_name, 0.0) > 0.5:
                intraday_score += score
                reasons.append(f"特征[{feature_name}]: {'+' if score > 0 else ''}{score}")
        # 4. 与日线策略结合
        daily_entry_score = daily_signal_info.get('entry_score', 0)
        daily_risk_score = daily_signal_info.get('risk_score', 0)
        # 日线买入信号的加成
        if daily_entry_score > 0:
            bonus = daily_entry_score * self.daily_score_influence_multiplier
            intraday_score += bonus
            reasons.append(f"日线买入信号加成: +{bonus:.0f}")
        # 日线风险信号的惩罚
        if daily_risk_score > 0:
            penalty = daily_risk_score * self.daily_risk_penalty_multiplier
            intraday_score -= penalty
            reasons.append(f"日线风险信号惩罚: -{penalty:.0f}")
        # 5. 确定最终评级
        final_rating = "NEUTRAL" # 默认中性
        strong_buy_thresh = self.rating_thresholds.get("STRONG_BUY", 150)
        buy_thresh = self.rating_thresholds.get("BUY", 80)
        sell_thresh = self.rating_thresholds.get("SELL", -100)
        strong_sell_thresh = self.rating_thresholds.get("STRONG_SELL", -150)
        if intraday_score >= strong_buy_thresh:
            final_rating = "STRONG_BUY"
        elif intraday_score >= buy_thresh:
            final_rating = "BUY"
        elif intraday_score <= strong_sell_thresh:
            final_rating = "STRONG_SELL"
        elif intraday_score <= sell_thresh:
            final_rating = "SELL"
        # 如果分数在 BUY 和 SELL 之间，则为 NEUTRAL
        elif buy_thresh > intraday_score > sell_thresh:
            final_rating = "NEUTRAL"
        # 组合理由
        combined_reason = f"盘中综合评分: {intraday_score:.0f}。详情: " + "; ".join(reasons)
        if len(combined_reason) > 500: # 限制理由长度
            combined_reason = combined_reason[:497] + "..."
        return int(intraday_score), final_rating, combined_reason


