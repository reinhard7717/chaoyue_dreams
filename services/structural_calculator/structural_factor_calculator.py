# services\structural_calculator\structural_factor_calculator.py
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

class StructuralFactorCalculator:
    """
    综合结构因子计算器
    """
    
    def __init__(self, stock_code: str, market_type: str):
        self.stock_code = stock_code
        self.market_type = market_type
        self.fib_calculator = FibonacciPriceCalculator(stock_code, market_type)
        
    def calculate_all_factors(self,
                             trade_date: datetime,
                             daily_data: List[Dict],
                             minute_data: Optional[List[Dict]] = None) -> Dict:
        """
        计算所有斐波那契结构因子
        Args:
            trade_date: 交易日期
            daily_data: 日线数据
            minute_data: 分钟线数据
        Returns:
            所有结构因子
        """
        factors = {}
        # 当前价格
        current_price = daily_data[-1]['close'] if daily_data else 0
        # 1. 计算斐波那契价格因子
        price_factors = self.fib_calculator.calculate_fibonacci_price_levels(
            daily_data, current_price
        )
        factors.update(price_factors)
        # 2. 提取重要日期
        significant_dates = self.fib_calculator.extract_significant_dates(daily_data)
        # 3. 计算斐波那契时间因子
        time_factors = self.fib_calculator.calculate_fibonacci_time_windows(
            trade_date, significant_dates
        )
        factors.update(time_factors)
        # 4. 计算时间-价格共振因子
        resonance_factors = self.fib_calculator.calculate_time_price_resonance(
            price_factors, time_factors, daily_data
        )
        factors.update(resonance_factors)
        # 5. 计算目标位
        target_factors = self._calculate_price_targets(
            daily_data, current_price, significant_dates
        )
        factors.update(target_factors)
        # 6. 计算综合评分
        composite_factors = self._calculate_composite_scores(
            price_factors, time_factors, resonance_factors, daily_data
        )
        factors.update(composite_factors)
        return factors
    
    def _calculate_price_targets(self,
                                daily_data: List[Dict],
                                current_price: float,
                                significant_dates: List[datetime]) -> Dict:
        """
        计算价格目标位
        Args:
            daily_data: 日线数据
            current_price: 当前价格
            significant_dates: 重要日期
        Returns:
            目标位因子
        """
        targets = {}
        if len(daily_data) < 20:
            return targets
        # 斐波那契扩展目标
        fib_targets = self._calculate_fibonacci_targets(daily_data, current_price)
        # 形态目标（基于近期高低点）
        pattern_targets = self._calculate_pattern_targets(daily_data, current_price)
        # 支撑阻力位
        support_levels, resistance_levels = self._calculate_support_resistance(
            daily_data, current_price
        )
        targets['price_targets'] = fib_targets
        targets['support_levels'] = support_levels
        targets['resistance_levels'] = resistance_levels
        targets['fib_targets'] = fib_targets
        targets['pattern_targets'] = pattern_targets
        return targets
    
    def _calculate_fibonacci_targets(self,
                                    daily_data: List[Dict],
                                    current_price: float) -> List[Dict]:
        """
        计算斐波那契扩展目标位
        """
        if len(daily_data) < 60:
            return []
        highs = [d['high'] for d in daily_data[-60:]]
        lows = [d['low'] for d in daily_data[-60:]]
        swing_high = max(highs)
        swing_low = min(lows)
        price_range = swing_high - swing_low
        targets = []
        # 斐波那契扩展位
        for ratio_name, ratio in FibonacciPriceCalculator.FIB_RATIOS.items():
            if ratio_name in ['1272', '1618', '2618']:
                target_price = swing_low + price_range * ratio
                targets.append({
                    'type': f'fib_extension_{ratio_name}',
                    'price': round(target_price, 2),
                    'distance_percent': round((target_price - current_price) / current_price * 100, 2)
                })
        return targets
    
    def _calculate_pattern_targets(self,
                                  daily_data: List[Dict],
                                  current_price: float) -> List[Dict]:
        """
        计算形态目标位
        """
        targets = []
        if len(daily_data) < 20:
            return targets
        # 基于近期高低点计算目标
        recent_highs = [d['high'] for d in daily_data[-20:]]
        recent_lows = [d['low'] for d in daily_data[-20:]]
        recent_high = max(recent_highs)
        recent_low = min(recent_lows)
        recent_range = recent_high - recent_low
        # 突破目标
        if current_price > recent_high:
            # 向上突破，目标 = 突破点 + 幅度
            target_price = current_price + recent_range
            targets.append({
                'type': 'breakout_up',
                'price': round(target_price, 2),
                'distance_percent': round((target_price - current_price) / current_price * 100, 2)
            })
        elif current_price < recent_low:
            # 向下突破，目标 = 突破点 - 幅度
            target_price = current_price - recent_range
            targets.append({
                'type': 'breakout_down',
                'price': round(target_price, 2),
                'distance_percent': round((target_price - current_price) / current_price * 100, 2)
            })
        return targets
    
    def _calculate_support_resistance(self,
                                     daily_data: List[Dict],
                                     current_price: float) -> Tuple[List, List]:
        """
        计算支撑阻力位
        """
        if len(daily_data) < 60:
            return [], []
        closes = [d['close'] for d in daily_data[-60:]]
        highs = [d['high'] for d in daily_data[-60:]]
        lows = [d['low'] for d in daily_data[-60:]]
        # 价格密集区识别
        price_levels = []
        # 添加近期高低点
        price_levels.extend(highs[-10:])
        price_levels.extend(lows[-10:])
        # 添加移动平均线
        if len(closes) >= 20:
            ma20 = sum(closes[-20:]) / 20
            price_levels.append(ma20)
        if len(closes) >= 60:
            ma60 = sum(closes[-60:]) / 60
            price_levels.append(ma60)
        # 分类支撑阻力
        support_levels = [round(level, 2) for level in price_levels if level < current_price]
        resistance_levels = [round(level, 2) for level in price_levels if level > current_price]
        # 去重排序
        support_levels = sorted(set(support_levels), reverse=True)[:5]  # 取最强的5个支撑
        resistance_levels = sorted(set(resistance_levels))[:5]  # 取最强的5个阻力
        return support_levels, resistance_levels
    
    def _calculate_composite_scores(self,
                                   price_factors: Dict,
                                   time_factors: Dict,
                                   resonance_factors: Dict,
                                   daily_data: List[Dict]) -> Dict:
        """
        计算综合评分
        """
        scores = {}
        # 结构强度得分
        structure_strength = self._calculate_structure_strength(
            price_factors, time_factors, resonance_factors
        )
        scores['structure_strength'] = Decimal(str(structure_strength))
        # 结构得分
        structure_score = self._calculate_structure_score(
            price_factors, time_factors, resonance_factors, daily_data
        )
        scores['structure_score'] = Decimal(str(structure_score))
        # 信号置信度
        signal_confidence = self._calculate_signal_confidence(
            structure_strength, structure_score
        )
        scores['signal_confidence'] = Decimal(str(signal_confidence))
        return scores
    
    def _calculate_structure_strength(self,
                                     price_factors: Dict,
                                     time_factors: Dict,
                                     resonance_factors: Dict) -> float:
        """
        计算结构强度
        """
        strength = 0
        # 价格因子权重
        price_signals = sum(1 for k, v in price_factors.items() 
                          if ('fib_price_level' in k or 'fib_price_extension' in k) and v)
        strength += price_signals * 15
        # 时间因子权重
        time_signals = sum(1 for k, v in time_factors.items() 
                          if 'fib_time_window' in k and v)
        strength += time_signals * 10
        # 共振因子权重
        if resonance_factors.get('fib_time_price_resonance'):
            strength += 30
            resonance_level = resonance_factors.get('resonance_level', 0)
            strength += resonance_level * 10
        return min(strength, 100)
    
    def _calculate_structure_score(self,
                                  price_factors: Dict,
                                  time_factors: Dict,
                                  resonance_factors: Dict,
                                  daily_data: List[Dict]) -> float:
        """
        计算结构得分
        """
        base_score = self._calculate_structure_strength(
            price_factors, time_factors, resonance_factors
        )
        # 波动率调整
        volatility = self.fib_calculator._calculate_price_volatility(daily_data)
        volatility_factor = 1.0 + min(volatility, 0.5)  # 波动率越高，得分越高
        # 成交量确认
        volume_confirmation = self.fib_calculator._check_volume_confirmation(daily_data)
        volume_factor = 1.2 if volume_confirmation else 0.9
        final_score = base_score * volatility_factor * volume_factor
        return min(final_score, 100)
    
    def _calculate_signal_confidence(self,
                                    structure_strength: float,
                                    structure_score: float) -> float:
        """
        计算信号置信度
        """
        # 综合强度和得分
        confidence = (structure_strength * 0.6 + structure_score * 0.4)
        # 归一化
        return min(confidence, 100)
