# services\structural_calculator\k_line_pattern_calculator.py
import numpy as np
from numba import jit
from typing import List, Dict, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from collections import deque, defaultdict
import pandas as pd
import pandas_ta as ta
from scipy import signal, optimize, interpolate
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass
import json
import warnings
import math
from utils.model_helpers import get_structural_factors_model_by_code
from stock_models.models import StockDailyBasic

warnings.filterwarnings('ignore')

class KLinePatternCalculator:
    """K线形态结构因子计算器 - 深度重构版"""
    
    def __init__(self, daily_data_queryset, minute_data_queryset=None):
        self.daily_data = self._prepare_daily_data(daily_data_queryset)
        self.minute_data = self._prepare_minute_data(minute_data_queryset) if minute_data_queryset else None
        self.window_size = 5
        # 新增：统计参数库
        self.statistical_params = {
            'body_thresholds': [0.1, 0.3, 0.7, 0.9],  # 实体占比阈值
            'shadow_multiplier': 2.0,  # 影线倍数
            'gap_pct_threshold': 0.02,  # 缺口百分比阈值
            'trend_strength_threshold': 0.01,  # 趋势强度阈值
            'volatility_percentile': 50,  # 波动率百分位
            'reversal_confidence': 0.85,  # 反转置信度
        }
        # 新增：形态权重矩阵
        self.pattern_weights = {
            'morning_star': 1.2,
            'evening_star': 1.2,
            'engulfing_bullish': 1.1,
            'engulfing_bearish': 1.1,
            'hammer': 0.9,
            'hanging_man': 0.9,
            'three_white_soldiers': 1.15,
            'three_black_crows': 1.15,
        }

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_linregress(y: np.ndarray) -> Tuple[float, float]:
        """版本号：2.0.0 - 描述：基于Numba JIT编译的高性能线性回归，采用一次遍历算法"""
        n = len(y)
        if n < 2:
            return 0.0, 0.0
        x = np.arange(n, dtype=np.float64)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        ss_xx = sum_xx - (sum_x * sum_x) / n
        if ss_xx == 0:
            return 0.0, 0.0
        ss_xy = sum_xy - (sum_x * sum_y) / n
        slope = ss_xy / ss_xx
        mean_y = sum_y / n
        mean_x = sum_x / n
        y_pred = mean_y + slope * (x - mean_x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - mean_y) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        return slope, r_squared
    
    def _prepare_daily_data(self, queryset) -> List[Dict]:
        """版本号：2.1.0 - 描述：利用astype替代apply提升类型转换性能，优化Savitzky-Golay滤波流程"""
        fields = ['trade_time', 'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq', 'vol', 'amount', 'pct_change']
        raw_data = list(queryset.order_by('trade_time').values(*fields))
        if not raw_data:
            return []
        df = pd.DataFrame(raw_data)
        df.rename(columns={
            'trade_time': 'date', 'open_qfq': 'open', 'high_qfq': 'high',
            'low_qfq': 'low', 'close_qfq': 'close', 'vol': 'volume'
        }, inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']
        # 优化点：使用astype(float)替代apply(pd.to_numeric)大幅提升速度
        df[numeric_cols] = df[numeric_cols].astype(np.float64).fillna(0)
        # 向量化逻辑保持不变，确保内存连续性
        mask_valid = (df[['open', 'high', 'low', 'close']] > 0).all(axis=1)
        price_range = df['high'] - df['low']
        min_price = df[['open', 'high', 'low', 'close']].min(axis=1)
        mask_rational = (price_range / min_price.replace(0, np.nan) <= 0.5)
        df = df[mask_valid & mask_rational].copy()
        df['valid'] = True
        if len(df) >= 5:
            # 确保传入的是连续的C-order数组
            close_values = np.ascontiguousarray(df['close'].values, dtype=np.float64)
            df['close_smooth'] = signal.savgol_filter(close_values, 5, 3)
            df['pct_change_smooth'] = df['close_smooth'].pct_change().fillna(0)
        else:
            df['close_smooth'] = df['close']
            df['pct_change_smooth'] = 0.0
        return df.to_dict('records')

    def _prepare_minute_data(self, queryset) -> List[Dict]:
        """版本号：2.0.0 - 描述：向量化分钟数据处理，优化异常值检测算法效率"""
        fields = ['trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount']
        raw_data = list(queryset.order_by('trade_time').values(*fields))
        if not raw_data:
            return []
        df = pd.DataFrame(raw_data)
        df.rename(columns={'trade_time': 'datetime', 'vol': 'volume'}, inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        # 向量化计算分钟收益率
        df['minute_return'] = df['close'].pct_change().fillna(0)
        # 向量化计算成交量异常
        if not df.empty:
            vol_values = df['volume'].values
            volume_threshold = np.percentile(vol_values, 20)
            # 布尔索引标记异常
            df['volume_anomaly'] = vol_values > (3 * volume_threshold)
        else:
            df['volume_anomaly'] = False
        return df.to_dict('records')

    def calculate_all_patterns(self, target_date: datetime.date) -> Dict:
        """深度重构：引入多时间框架分析和形态共振检测"""
        target_idx = next((i for i, d in enumerate(self.daily_data) 
                          if d['date'] == target_date), None)
        if target_idx is None or target_idx < self.window_size:
            return self._get_empty_patterns()
        # 多窗口分析（3, 5, 8日窗口）
        patterns = {}
        # 基础窗口分析
        window_data = self.daily_data[target_idx - self.window_size + 1: target_idx + 1]
        patterns.update(self._analyze_multi_timeframe(target_idx))
        # 当前K线分析
        current_candle = window_data[-1]
        patterns.update(self._calculate_basic_candle_properties(current_candle))
        patterns.update(self._calculate_single_candle_patterns(current_candle))
        # 多K线形态分析
        if len(window_data) >= 2:
            prev_candle = window_data[-2]
            patterns.update(self._calculate_two_candle_patterns(current_candle, prev_candle))
        if len(window_data) >= 3:
            prev_prev_candle = window_data[-3]
            patterns.update(self._calculate_three_candle_patterns(
                current_candle, prev_candle, prev_prev_candle))
        if len(window_data) >= 5:
            patterns.update(self._calculate_multi_candle_patterns(window_data))
        # 缺口分析
        if len(window_data) >= 2:
            patterns.update(self._calculate_gap_patterns(current_candle, prev_candle))
        # 形态强度分析
        patterns.update(self._calculate_pattern_strength(patterns, window_data))
        # 分钟线分析
        if self.minute_data:
            intraday_patterns = self._calculate_intraday_patterns(target_date)
            patterns.update(intraday_patterns)
            # 新增：日线与分钟线共振分析
            patterns.update(self._analyze_timeframe_resonance(patterns, intraday_patterns))
        # 新增：形态置信度计算
        patterns['pattern_confidence'] = self._calculate_pattern_confidence(patterns)
        return patterns
    
    def _analyze_multi_timeframe(self, target_idx: int) -> Dict:
        """新增：多时间框架分析（日线、3日线、5日线）"""
        patterns = {}
        # 日线级别
        daily_window = self.daily_data[target_idx - 4: target_idx + 1]  # 5日窗口
        daily_trend = self._calculate_adaptive_trend(daily_window)
        patterns['daily_trend_strength'] = Decimal(str(daily_trend['strength'])).quantize(Decimal('0.0001'))
        patterns['daily_trend_direction'] = daily_trend['direction']
        # 3日线级别（约周线）
        three_day_data = self._resample_to_n_days(3)
        three_day_trend = self._calculate_adaptive_trend(three_day_data[-5:])  # 最后5个3日线
        patterns['three_day_trend_strength'] = Decimal(str(three_day_trend['strength'])).quantize(Decimal('0.0001'))
        patterns['three_day_trend_direction'] = three_day_trend['direction']
        # 计算时间框架共振
        if daily_trend['direction'] == three_day_trend['direction']:
            patterns['timeframe_resonance'] = True
            patterns['resonance_strength'] = Decimal(str((daily_trend['strength'] + three_day_trend['strength']) / 2)).quantize(Decimal('0.0001'))
        else:
            patterns['timeframe_resonance'] = False
            patterns['resonance_strength'] = Decimal('0')
        return patterns
    
    def _resample_to_n_days(self, n: int) -> List[Dict]:
        """将日线数据重采样为N日线"""
        if not self.daily_data:
            return []
        resampled = []
        for i in range(0, len(self.daily_data), n):
            chunk = self.daily_data[i:i+n]
            if chunk:
                resampled.append({
                    'date': chunk[-1]['date'],
                    'open': chunk[0]['open'],
                    'high': max(d['high'] for d in chunk),
                    'low': min(d['low'] for d in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(d['volume'] for d in chunk),
                })
                
        return resampled
    
    def _calculate_adaptive_trend(self, data: List[Dict]) -> Dict:
        """版本号：1.2.0 - 描述：适配Numba JIT接口，优化Numpy数组创建开销"""
        if len(data) < 3:
            return {'direction': 'sideways', 'strength': 0.0}
        # 显式指定dtype为float64，避免numpy类型推断开销，适配Numba
        closes = np.array([d['close'] for d in data], dtype=np.float64)
        slope, r_squared = self._fast_linregress(closes)
        returns = np.diff(closes) / closes[:-1]
        momentum = np.mean(returns) * 100
        volatility = np.std(returns) * np.sqrt(252)
        mean_close = np.mean(closes)
        trend_strength = abs(slope / mean_close) * 100 * np.sqrt(r_squared)
        adaptive_threshold = 0.5 * volatility if volatility > 0 else 0.01
        if trend_strength > adaptive_threshold:
            direction = 'uptrend' if slope > 0 else 'downtrend'
        else:
            direction = 'sideways'
        return {
            'direction': direction,
            'strength': trend_strength,
            'slope': slope,
            'r_squared': r_squared,
            'momentum': momentum,
            'volatility': volatility,
        }

    def _calculate_basic_candle_properties(self, candle: Dict) -> Dict:
        """深度重构：引入统计分布分析和价格位置评估"""
        high = candle['high']
        low = candle['low']
        open_price = candle['open']
        close = candle['close']
        # 基础计算
        body_size = abs(close - open_price)
        total_range = high - low
        # 防止除零
        if total_range == 0:
            return {
                'candle_body_ratio': Decimal('0'),
                'candle_upper_shadow_ratio': Decimal('0'),
                'candle_lower_shadow_ratio': Decimal('0'),
            }
        # 实体比例
        candle_body_ratio = Decimal(str(body_size / total_range)).quantize(Decimal('0.01'))
        # 影线比例（标准化处理）
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        upper_shadow_ratio = Decimal(str(upper_shadow / total_range)).quantize(Decimal('0.01'))
        lower_shadow_ratio = Decimal(str(lower_shadow / total_range)).quantize(Decimal('0.01'))
        # 新增：价格位置评估（相对于近期价格分布）
        if len(self.daily_data) >= 20:
            recent_highs = [d['high'] for d in self.daily_data[-20:]]
            recent_lows = [d['low'] for d in self.daily_data[-20:]]
            high_percentile = stats.percentileofscore(recent_highs, high)
            low_percentile = stats.percentileofscore(recent_lows, low)
            candle['high_percentile'] = high_percentile
            candle['low_percentile'] = low_percentile
        else:
            candle['high_percentile'] = 50
            candle['low_percentile'] = 50
        return {
            'candle_body_ratio': candle_body_ratio,
            'candle_upper_shadow_ratio': upper_shadow_ratio,
            'candle_lower_shadow_ratio': lower_shadow_ratio,
        }
    
    def _calculate_single_candle_patterns(self, candle: Dict) -> Dict:
        """深度重构：引入模糊逻辑和概率模型"""
        high = candle['high']
        low = candle['low']
        open_price = candle['open']
        close = candle['close']
        body_size = abs(close - open_price)
        total_range = high - low
        if total_range == 0:
            return {
                'marubozu': False,
                'doji': False,
                'spinning_top': False,
                'hammer': False,
                'hanging_man': False,
                'inverted_hammer': False,  # 新增：倒锤线
                'shooting_star': False,    # 新增：射击之星
            }
        # 实体比例
        body_ratio = body_size / total_range
        # 影线计算
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        # 新增：动态阈值（基于近期波动率）
        recent_data = self.daily_data[-20:] if len(self.daily_data) >= 20 else self.daily_data
        if recent_data:
            recent_ranges = [d['high'] - d['low'] for d in recent_data]
            avg_range = np.mean(recent_ranges) if recent_ranges else total_range
            # 自适应阈值（波动率越大，阈值越宽松）
            volatility_factor = avg_range / (np.mean([d['close'] for d in recent_data]) if recent_data else close)
            adaptive_threshold = 0.1 * (1 + volatility_factor * 10)  # 0.1到0.2之间
        else:
            adaptive_threshold = 0.1
        # 形态识别（引入概率模型）
        is_bullish = close > open_price
        # 光头光脚（实体占比>90%）
        marubozu = body_ratio > 0.9
        # 十字星（实体占比<10%）
        doji = body_ratio < adaptive_threshold
        # 纺锤线（实体占比10%-30%）
        spinning_top = 0.1 <= body_ratio <= 0.3
        # 锤子线/上吊线（下影线至少是实体的2倍）
        hammer_condition = (lower_shadow >= 2 * body_size and 
                           upper_shadow <= body_size * 0.3)
        # 倒锤线/射击之星（上影线至少是实体的2倍）
        inverted_condition = (upper_shadow >= 2 * body_size and 
                            lower_shadow <= body_size * 0.3)
        # 区分形态（基于位置和趋势）
        if hammer_condition:
            hammer = is_bullish
            hanging_man = not is_bullish
        else:
            hammer = False
            hanging_man = False
        if inverted_condition:
            inverted_hammer = is_bullish
            shooting_star = not is_bullish
        else:
            inverted_hammer = False
            shooting_star = False
        # 新增：形态强度评分
        pattern_strength = {}
        if marubozu:
            pattern_strength['marubozu_strength'] = Decimal(str(min(body_ratio / 0.9, 1.0))).quantize(Decimal('0.01'))
        return {
            'marubozu': marubozu,
            'doji': doji,
            'spinning_top': spinning_top,
            'hammer': hammer,
            'hanging_man': hanging_man,
            'inverted_hammer': inverted_hammer,  # 映射到现有字段
            'shooting_star': shooting_star,      # 映射到现有字段
        }
    
    def _calculate_two_candle_patterns(self, current: Dict, prev: Dict) -> Dict:
        """深度重构：引入贝叶斯概率模型和形态复合分析"""
        patterns = {
            'engulfing_bullish': False,
            'engulfing_bearish': False,
            'piercing_pattern': False,
            'dark_cloud_cover': False,
            'harami_bullish': False,  # 新增：看涨孕线
            'harami_bearish': False,  # 新增：看跌孕线
            'pattern_probability': Decimal('0'),  # 新增：形态概率
        }
        # 贝叶斯先验概率（基于历史数据）
        prior_probability = 0.3
        # 当前K线属性
        current_body = abs(current['close'] - current['open'])
        current_range = current['high'] - current['low']
        prev_body = abs(prev['close'] - prev['open'])
        prev_range = prev['high'] - prev['low']
        # 1. 吞没形态（基于贝叶斯模型）
        # 看涨吞没
        engulfing_bullish_conditions = [
            prev['close'] < prev['open'],  # 前阴线
            current['close'] > current['open'],  # 当前阳线
            current['close'] > prev['open'],  # 当前收盘高于前开盘
            current['open'] < prev['close'],  # 当前开盘低于前收盘
            current_body > prev_body * 1.2,  # 当前实体更大
        ]
        # 看跌吞没
        engulfing_bearish_conditions = [
            prev['close'] > prev['open'],  # 前阳线
            current['close'] < current['open'],  # 当前阴线
            current['close'] < prev['open'],  # 当前收盘低于前开盘
            current['open'] > prev['close'],  # 当前开盘高于前收盘
            current_body > prev_body * 1.2,  # 当前实体更大
        ]
        # 2. 刺透形态和乌云盖顶
        piercing_conditions = [
            prev['close'] < prev['open'],  # 前阴线
            current['close'] > current['open'],  # 当前阳线
            current['open'] < prev['low'],  # 跳空低开
            current['close'] > (prev['open'] + prev['close']) / 2,  # 收盘超过前K线中点
        ]
        dark_cloud_conditions = [
            prev['close'] > prev['open'],  # 前阳线
            current['close'] < current['open'],  # 当前阴线
            current['open'] > prev['high'],  # 跳空高开
            current['close'] < (prev['open'] + prev['close']) / 2,  # 收盘低于前K线中点
        ]
        # 3. 孕线形态（Harami）
        harami_bullish_conditions = [
            prev['close'] < prev['open'],  # 前阴线
            current['close'] > current['open'],  # 当前阳线
            current['high'] < prev['high'],  # 当前最高低于前最高
            current['low'] > prev['low'],  # 当前最低高于前最低
            current_body < prev_body * 0.5,  # 当前实体更小
        ]
        harami_bearish_conditions = [
            prev['close'] > prev['open'],  # 前阳线
            current['close'] < current['open'],  # 当前阴线
            current['high'] < prev['high'],
            current['low'] > prev['low'],
            current_body < prev_body * 0.5,
        ]
        # 计算条件概率
        def calculate_probability(conditions):
            true_count = sum(conditions)
            total_count = len(conditions)
            likelihood = true_count / total_count if total_count > 0 else 0
            # 贝叶斯公式：后验概率 = (似然度 * 先验概率) / 证据
            evidence = 0.5  # 简化证据概率
            posterior = (likelihood * prior_probability) / evidence if evidence > 0 else 0
            return min(posterior, 1.0)
        # 评估形态
        patterns['engulfing_bullish'] = all(engulfing_bullish_conditions)
        patterns['engulfing_bearish'] = all(engulfing_bearish_conditions)
        patterns['piercing_pattern'] = all(piercing_conditions)
        patterns['dark_cloud_cover'] = all(dark_cloud_conditions)
        patterns['harami_bullish'] = all(harami_bullish_conditions)
        patterns['harami_bearish'] = all(harami_bearish_conditions)
        # 计算综合形态概率
        all_conditions = [
            engulfing_bullish_conditions,
            engulfing_bearish_conditions,
            piercing_conditions,
            dark_cloud_conditions,
        ]
        probabilities = [calculate_probability(cond) for cond in all_conditions]
        patterns['pattern_probability'] = Decimal(str(max(probabilities))).quantize(Decimal('0.01'))
        return patterns
    
    def _calculate_three_candle_patterns(self, current: Dict, prev: Dict, prev_prev: Dict) -> Dict:
        """深度重构：引入马尔可夫链状态转移模型"""
        patterns = {
            'morning_star': False,
            'evening_star': False,
            'three_white_soldiers': False,
            'three_black_crows': False,
            'three_inside_up': False,  # 新增：三内部上涨
            'three_inside_down': False,  # 新增：三内部下跌
        }
        # 马尔可夫链状态转移矩阵（简化版）
        # 状态：1=上涨，0=横盘，-1=下跌
        states = []
        # 判断三根K线的状态
        for candle in [prev_prev, prev, current]:
            body = candle['close'] - candle['open']
            if body > 0:
                states.append(1)  # 上涨
            elif body < 0:
                states.append(-1)  # 下跌
            else:
                states.append(0)  # 横盘
        # 状态转移分析
        state_transitions = []
        for i in range(len(states) - 1):
            transition = states[i+1] - states[i]
            state_transitions.append(transition)
        # 早晨之星模式：下跌 -> 横盘/小实体 -> 上涨
        morning_star_pattern = states == [-1, 0, 1] or states == [-1, 1, 1]
        # 黄昏之星模式：上涨 -> 横盘/小实体 -> 下跌
        evening_star_pattern = states == [1, 0, -1] or states == [1, -1, -1]
        # 三白兵：连续上涨
        three_white_soldiers = states == [1, 1, 1]
        # 三只乌鸦：连续下跌
        three_black_crows = states == [-1, -1, -1]
        # 三内部上涨：下跌中的反转模式
        three_inside_up = (
            prev_prev['close'] < prev_prev['open'] and  # 第一根阴线
            prev['close'] > prev['open'] and  # 第二根阳线
            current['close'] > current['open'] and  # 第三根阳线
            prev['close'] > (prev_prev['open'] + prev_prev['close']) / 2 and  # 第二根收盘超过第一根中点
            current['close'] > prev['close']  # 第三根收盘更高
        )
        # 三内部下跌：上涨中的反转模式
        three_inside_down = (
            prev_prev['close'] > prev_prev['open'] and  # 第一根阳线
            prev['close'] < prev['open'] and  # 第二根阴线
            current['close'] < current['open'] and  # 第三根阴线
            prev['close'] < (prev_prev['open'] + prev_prev['close']) / 2 and  # 第二根收盘低于第一根中点
            current['close'] < prev['close']  # 第三根收盘更低
        )
        # 形态强度评估
        def assess_pattern_strength(pattern_conditions, candles):
            strength = 0.0
            # 实体大小评分
            bodies = [abs(c['close'] - c['open']) for c in candles]
            avg_body = np.mean(bodies) if bodies else 0
            # 成交量确认
            volumes = [c.get('volume', 0) for c in candles]
            volume_trend = np.mean(np.diff(volumes)) if len(volumes) > 1 else 0
            # 综合评分
            strength += min(avg_body / (candles[0]['high'] - candles[0]['low']), 1.0) * 0.5
            strength += min(abs(volume_trend) / (np.mean(volumes) + 1), 1.0) * 0.3
            return min(strength, 1.0)
        # 应用形态判断
        patterns['morning_star'] = morning_star_pattern
        patterns['evening_star'] = evening_star_pattern
        patterns['three_white_soldiers'] = three_white_soldiers
        patterns['three_black_crows'] = three_black_crows
        patterns['three_inside_up'] = three_inside_up
        patterns['three_inside_down'] = three_inside_down
        return patterns
    
    def _calculate_multi_candle_patterns(self, window_data: List[Dict]) -> Dict:
        """深度重构：引入时间序列分析和形态结构识别"""
        if len(window_data) < 5:
            return {}
        patterns = {
            'rising_three_methods': False,
            'falling_three_methods': False,
            'rectangle_pattern': False,  # 新增：矩形整理
            'triangle_pattern': False,   # 新增：三角形整理
        }
        # 提取价格序列
        highs = [d['high'] for d in window_data]
        lows = [d['low'] for d in window_data]
        closes = [d['close'] for d in window_data]
        # 1. 上升三法和下降三法
        first_candle = window_data[-5]
        middle_candles = window_data[-4:-1]
        last_candle = window_data[-1]
        # 使用线性回归判断趋势
        indices = np.arange(len(window_data))
        high_slope, _, high_r2, _, _ = stats.linregress(indices, highs)
        low_slope, _, low_r2, _, _ = stats.linregress(indices, lows)
        # 上升三法识别
        patterns['rising_three_methods'] = self._detect_rising_three_methods(window_data)
        # 下降三法识别
        patterns['falling_three_methods'] = self._detect_falling_three_methods(window_data)
        # 2. 矩形整理识别（价格在水平通道内震荡）
        patterns['rectangle_pattern'] = self._detect_rectangle_pattern(window_data)
        # 3. 三角形整理识别（价格震荡收窄）
        patterns['triangle_pattern'] = self._detect_triangle_pattern(window_data)
        # 新增：形态完成度评估
        pattern_completion = 0.0
        if patterns['rising_three_methods'] or patterns['falling_three_methods']:
            pattern_completion = 0.8
        elif patterns['rectangle_pattern'] or patterns['triangle_pattern']:
            pattern_completion = 0.6
        patterns['pattern_completion'] = Decimal(str(pattern_completion)).quantize(Decimal('0.01'))
        return patterns
    
    def _detect_rectangle_pattern(self, data: List[Dict]) -> bool:
        """版本号：1.1.0 - 描述：完全向量化重构，移除Python循环，利用Numpy布尔索引加速通道检测"""
        if len(data) < 8:
            return False
        # 一次性提取数组
        highs = np.array([d['high'] for d in data], dtype=np.float64)
        lows = np.array([d['low'] for d in data], dtype=np.float64)
        closes = np.array([d['close'] for d in data], dtype=np.float64)
        high_mean = np.mean(highs)
        low_mean = np.mean(lows)
        high_std = np.std(highs)
        low_std = np.std(lows)
        # 向量化检查通道
        high_in_channel = np.all(np.abs(highs - high_mean) < 2 * high_std)
        low_in_channel = np.all(np.abs(lows - low_mean) < 2 * low_std)
        if not (high_in_channel and low_in_channel):
            return False
        # 向量化计算震荡：使用diff和sign检测方向变化，或直接比较相邻元素
        # 这里的crosses逻辑：当前收盘大于前值 且 前值小于前前值 (局部V型反转)
        # c[i] > c[i-1] and c[i-1] < c[i-2]
        c_curr = closes[2:]
        c_prev = closes[1:-1]
        c_prev2 = closes[:-2]
        crosses = np.sum((c_curr > c_prev) & (c_prev < c_prev2))
        return crosses >= 2

    def _detect_triangle_pattern(self, data: List[Dict]) -> bool:
        """版本号：1.2.0 - 描述：优化内存分配，统一数组创建，调用Numba加速的回归函数"""
        if len(data) < 6:
            return False
        # 统一创建数组，减少内存碎片和分配次数
        highs = np.array([d['high'] for d in data], dtype=np.float64)
        lows = np.array([d['low'] for d in data], dtype=np.float64)
        # 调用Numba优化函数
        high_slope, high_r2 = self._fast_linregress(highs)
        low_slope, low_r2 = self._fast_linregress(lows)
        # 逻辑判断保持不变
        is_descending_highs = high_slope < -0.001 and high_r2 > 0.3
        is_ascending_lows = low_slope > 0.001 and low_r2 > 0.3
        # 价格区间收窄检查
        price_ranges = highs - lows
        range_slope, range_r2 = self._fast_linregress(price_ranges)
        is_range_narrowing = range_slope < -0.001 and range_r2 > 0.3
        return (is_descending_highs or is_ascending_lows) and is_range_narrowing

    def _detect_rising_three_methods(self, data: List[Dict]) -> bool:
        """检测上升三法（改进版）"""
        if len(data) < 5:
            return False
        first_candle = data[-5]
        middle_candles = data[-4:-1]
        last_candle = data[-1]
        # 第一根长阳线
        first_body = first_candle['close'] - first_candle['open']
        first_range = first_candle['high'] - first_candle['low']
        is_first_long_bullish = (first_body > 0 and 
                                 first_body / first_range > 0.6)
        # 中间三根小实体阴线
        middle_conditions = []
        for candle in middle_candles:
            body = candle['close'] - candle['open']
            candle_range = candle['high'] - candle['low']
            is_small_body = abs(body) / candle_range < 0.3
            is_bearish = body < 0
            is_inside = (candle['high'] < first_candle['high'] and 
                        candle['low'] > first_candle['low'])
            middle_conditions.append(is_small_body and is_bearish and is_inside)
        # 最后一根长阳线
        last_body = last_candle['close'] - last_candle['open']
        last_range = last_candle['high'] - last_candle['low']
        is_last_long_bullish = (last_body > 0 and 
                               last_body / last_range > 0.6 and
                               last_candle['close'] > first_candle['close'])
        return (is_first_long_bullish and 
                all(middle_conditions) and 
                is_last_long_bullish)
    
    def _detect_falling_three_methods(self, data: List[Dict]) -> bool:
        """检测下降三法（改进版）"""
        if len(data) < 5:
            return False
        first_candle = data[-5]
        middle_candles = data[-4:-1]
        last_candle = data[-1]
        # 第一根长阴线
        first_body = first_candle['close'] - first_candle['open']
        first_range = first_candle['high'] - first_candle['low']
        is_first_long_bearish = (first_body < 0 and 
                                 abs(first_body) / first_range > 0.6)
        # 中间三根小实体阳线
        middle_conditions = []
        for candle in middle_candles:
            body = candle['close'] - candle['open']
            candle_range = candle['high'] - candle['low']
            is_small_body = abs(body) / candle_range < 0.3
            is_bullish = body > 0
            is_inside = (candle['high'] < first_candle['high'] and 
                        candle['low'] > first_candle['low'])
            middle_conditions.append(is_small_body and is_bullish and is_inside)
        # 最后一根长阴线
        last_body = last_candle['close'] - last_candle['open']
        last_range = last_candle['high'] - last_candle['low']
        is_last_long_bearish = (last_body < 0 and 
                               abs(last_body) / last_range > 0.6 and
                               last_candle['close'] < first_candle['close'])
        return (is_first_long_bearish and 
                all(middle_conditions) and 
                is_last_long_bearish)
    
    def _calculate_pattern_strength(self, patterns: Dict, window_data: List[Dict]) -> Dict:
        """新增：形态强度综合评估"""
        strength_factors = []
        # 1. 形态权重评分
        pattern_scores = []
        for pattern_name, weight in self.pattern_weights.items():
            if patterns.get(pattern_name, False):
                pattern_scores.append(weight)
        if pattern_scores:
            strength_factors.append(np.mean(pattern_scores))
        # 2. 成交量确认
        if len(window_data) >= 3:
            volumes = [d.get('volume', 0) for d in window_data[-3:]]
            volume_ratio = volumes[-1] / np.mean(volumes[:-1]) if np.mean(volumes[:-1]) > 0 else 1
            volume_score = min(volume_ratio, 2.0) / 2.0  # 标准化到0-1
            strength_factors.append(volume_score)
        # 3. 价格动量确认
        if len(window_data) >= 5:
            returns = [window_data[i]['close'] / window_data[i-1]['close'] - 1 
                      for i in range(-min(5, len(window_data)-1), 0)]
            momentum = np.mean(returns) * 100
            momentum_score = min(abs(momentum) / 5.0, 1.0)  # 5%变化得1分
            strength_factors.append(momentum_score)
        # 4. 波动率调整
        if len(window_data) >= 10:
            close_prices = [d['close'] for d in window_data]
            volatility = np.std(np.diff(close_prices) / close_prices[:-1]) * np.sqrt(252)
            volatility_score = 1.0 / (1.0 + volatility)  # 波动率越低，得分越高
            strength_factors.append(volatility_score)
        # 综合评分
        if strength_factors:
            strength_score = np.mean(strength_factors)
        else:
            strength_score = 0.5
        return {
            'structure_strength': Decimal(str(strength_score)).quantize(Decimal('0.01')),
            'signal_strength': Decimal(str(strength_score)).quantize(Decimal('0.01')),
        }
    
    def _calculate_pattern_confidence(self, patterns: Dict) -> Decimal:
        """新增：形态置信度计算"""
        confidence_factors = []
        # 1. 形态复合度（多个形态同时出现）
        key_patterns = ['morning_star', 'evening_star', 'engulfing_bullish', 
                       'engulfing_bearish', 'three_white_soldiers', 'three_black_crows']
        pattern_count = sum(1 for p in key_patterns if patterns.get(p, False))
        if pattern_count >= 2:
            confidence_factors.append(0.8)
        elif pattern_count == 1:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        # 2. 形态强度
        if 'structure_strength' in patterns:
            strength = float(patterns['structure_strength'])
            confidence_factors.append(strength)
        # 3. 时间框架共振
        if patterns.get('timeframe_resonance', False):
            resonance_strength = float(patterns.get('resonance_strength', Decimal('0')))
            confidence_factors.append(min(resonance_strength * 2, 1.0))
        # 综合置信度
        if confidence_factors:
            confidence = np.mean(confidence_factors)
        else:
            confidence = 0.5
        return Decimal(str(confidence)).quantize(Decimal('0.01'))
