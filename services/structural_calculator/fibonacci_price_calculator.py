# services\structural_calculator\fibonacci_price_calculator.py
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

class FibonacciPriceCalculator:
    """
    斐波那契价格结构因子计算器 - 深度重构版
    融入混沌理论、分形几何、非线性动力学等高级数学模型
    """
    # 扩展的斐波那契比率（包含反黄金分割和衍生比率）
    FIB_RATIOS = {
        '236': 0.236,
        '382': 0.382,
        '500': 0.500,
        '618': 0.618,
        '786': 0.786,
        '886': 0.886,  # 反黄金分割的平方根
        '1272': 1.272,  # √φ
        '1414': 1.414,  # √2
        '1618': 1.618,  # φ
        '2618': 2.618,  # φ²
        '4236': 4.236,  # φ³
    }
    # 扩展的时间窗口（包含卢卡斯数列和其他重要数列）
    FIB_TIME_WINDOWS = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    # 卢卡斯数列（Lucas numbers）用于时间分析
    LUCAS_WINDOWS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199]
    
    def __init__(self, stock_code: str, market_type: str = 'SH'):
        self.stock_code = stock_code
        self.market_type = market_type
        
    def calculate_fibonacci_price_levels(self, daily_data: List[Dict],current_price: float,lookback_period: int = 120) -> Dict:
        """
        深度重构版：融入多重分形分析、混沌吸引子检测、非线性回归分析
        """
        if len(daily_data) < 30:
            return {}
        # 使用自适应窗口大小（基于波动率）
        optimal_lookback = self._calculate_adaptive_lookback(daily_data, lookback_period)
        if optimal_lookback < 30:
            optimal_lookback = 30
        # 提取价格数据
        closes = np.array([d['close'] for d in daily_data[-optimal_lookback:]])
        highs = np.array([d['high'] for d in daily_data[-optimal_lookback:]])
        lows = np.array([d['low'] for d in daily_data[-optimal_lookback:]])
        # === 核心改进1：多尺度分形分析 ===
        fractal_levels = self._calculate_fractal_support_resistance(highs, lows)
        # === 核心改进2：混沌理论分析 ===
        chaotic_attractors = self._detect_chaotic_attractors(closes)
        # === 核心改进3：非线性回归分析 ===
        regression_levels = self._calculate_nonlinear_regression_levels(closes)
        # === 核心改进4：傅里叶频谱分析 ===
        spectral_levels = self._calculate_spectral_price_levels(closes)
        # 传统斐波那契计算（优化版）
        fib_factors = self._calculate_enhanced_fibonacci_levels(
            highs, lows, closes, current_price, fractal_levels
        )
        # === 核心改进5：多维度融合 ===
        merged_levels = self._merge_multidimensional_levels(
            fib_factors, fractal_levels, chaotic_attractors, 
            regression_levels, spectral_levels, current_price
        )
        # 更新因子
        fib_factors.update(merged_levels)
        # 计算支撑阻力位（改进版）
        support_resistance = self._calculate_advanced_support_resistance(
            highs, lows, closes, current_price, 
            fractal_levels, chaotic_attractors
        )
        fib_factors.update(support_resistance)
        return fib_factors
    
    def calculate_fibonacci_time_windows(self, trade_date: datetime,significant_dates: List[datetime]) -> Dict:
        """
        深度重构版：融入周期分析、谐波共振、时间分形理论
        """
        fib_factors = {}
        if not significant_dates:
            return fib_factors
        # === 核心改进1：多重时间尺度分析 ===
        time_windows = self._analyze_multiple_time_scales(significant_dates, trade_date)
        # === 核心改进2：谐波共振检测 ===
        harmonic_resonance = self._detect_harmonic_resonance(significant_dates, trade_date)
        # === 核心改进3：时间分形分析 ===
        time_fractals = self._analyze_time_fractals(significant_dates, trade_date)
        # 基础斐波那契时间窗口
        for window in self.FIB_TIME_WINDOWS:
            factor_name = f'fib_time_window_{window}'
            fib_factors[factor_name] = self._is_time_in_window(trade_date, significant_dates, window)
        # 添加卢卡斯数列时间窗口
        for window in self.LUCAS_WINDOWS:
            factor_name = f'lucas_time_window_{window}'
            fib_factors[factor_name] = self._is_time_in_window(trade_date, significant_dates, window)
        # === 核心改进4：复合时间得分 ===
        fib_factors['fib_time_score'] = self._calculate_composite_time_score(
            trade_date, significant_dates, time_windows, harmonic_resonance, time_fractals
        )
        # 时间结构强度
        fib_factors['time_structure_strength'] = self._calculate_time_structure_strength(
            time_windows, harmonic_resonance, time_fractals
        )
        return fib_factors
    
    def calculate_time_price_resonance(self,price_factors: Dict,time_factors: Dict,daily_data: List[Dict]) -> Dict:
        """
        深度重构版：融入时空共振、相位同步、非线性耦合分析
        """
        resonance_factors = {}
        if not daily_data:
            return resonance_factors
        closes = np.array([d['close'] for d in daily_data[-60:]])
        # === 核心改进1：时空相位分析 ===
        phase_sync = self._analyze_time_price_phase_sync(
            price_factors, time_factors, closes
        )
        # === 核心改进2：非线性耦合强度 ===
        coupling_strength = self._calculate_nonlinear_coupling(
            price_factors, time_factors, closes
        )
        # === 核心改进3：多重共振检测 ===
        multi_resonance = self._detect_multi_resonance(
            price_factors, time_factors, daily_data
        )
        # 基础共振检测
        time_windows_active = [
            window for window, active in time_factors.items() 
            if ('fib_time_window' in window or 'lucas_time_window' in window) and active
        ]
        price_levels_active = [
            level for level, active in price_factors.items()
            if ('fib_price_level' in level or 'fib_price_extension' in level) and active
        ]
        if time_windows_active and price_levels_active:
            resonance_factors['fib_time_price_resonance'] = True
            # === 核心改进4：动态共振得分 ===
            resonance_score = self._calculate_dynamic_resonance_score(
                len(time_windows_active),
                len(price_levels_active),
                phase_sync,
                coupling_strength,
                multi_resonance,
                daily_data
            )
            resonance_factors['resonance_score'] = Decimal(str(resonance_score))
            # 共振级别（改进版）
            resonance_level = self._determine_enhanced_resonance_level(
                resonance_score,
                phase_sync,
                coupling_strength,
                multi_resonance
            )
            resonance_factors['resonance_level'] = resonance_level
            # === 核心改进5：共振稳定性 ===
            resonance_stability = self._calculate_resonance_stability(
                price_factors, time_factors, daily_data
            )
            resonance_factors['resonance_stability'] = Decimal(str(resonance_stability))
        else:
            resonance_factors['fib_time_price_resonance'] = False
            resonance_factors['resonance_score'] = None
            resonance_factors['resonance_level'] = None
        return resonance_factors
    
    def _calculate_adaptive_lookback(self, daily_data: List[Dict], base_lookback: int) -> int:
        """
        基于波动率和自相关性的自适应窗口计算
        """
        if len(daily_data) < 60:
            return base_lookback
        closes = np.array([d['close'] for d in daily_data[-60:]])
        # 计算波动率
        returns = np.diff(np.log(closes))
        volatility = np.std(returns) * np.sqrt(252)
        # 计算自相关性
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        # 自适应调整
        if volatility > 0.4:  # 高波动率
            adjusted = int(base_lookback * (1 - 0.3 * (volatility - 0.4)))
        elif volatility < 0.15:  # 低波动率
            adjusted = int(base_lookback * (1 + 0.5 * (0.15 - volatility)))
        else:
            adjusted = base_lookback
        # 自相关性调整
        if abs(autocorr) > 0.2:
            adjusted = int(adjusted * (1 + 0.3 * abs(autocorr)))
        return max(30, min(adjusted, 250))
    
    def _calculate_fractal_support_resistance(self, highs: np.ndarray, lows: np.ndarray) -> Dict:
        """
        基于多重分形理论计算支撑阻力位
        """
        # 计算赫斯特指数（Hurst Exponent）
        def hurst_exponent(time_series):
            lags = range(2, 100)
            tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        # 价格序列的分形分析
        price_range = highs - lows
        hurst_h = hurst_exponent(highs) if len(highs) > 100 else 0.5
        hurst_l = hurst_exponent(lows) if len(lows) > 100 else 0.5
        # 分形维度
        fractal_dim = 2 - hurst_h
        # 识别分形支撑阻力
        levels = {}
        # 使用箱计数法（Box-counting method）识别分形边界
        n_boxes = 20
        box_size = (np.max(highs) - np.min(lows)) / n_boxes
        price_counts = {}
        for h, l in zip(highs, lows):
            box_idx_h = int((h - np.min(lows)) / box_size)
            box_idx_l = int((l - np.min(lows)) / box_size)
            for idx in range(box_idx_l, box_idx_h + 1):
                price_counts[idx] = price_counts.get(idx, 0) + 1
        # 找出密度最大的区域
        if price_counts:
            sorted_boxes = sorted(price_counts.items(), key=lambda x: x[1], reverse=True)
            top_boxes = sorted_boxes[:5]
            for i, (box_idx, count) in enumerate(top_boxes):
                level_price = np.min(lows) + (box_idx + 0.5) * box_size
                levels[f'fractal_level_{i+1}'] = {
                    'price': float(level_price),
                    'density': float(count / len(highs)),
                    'strength': float(count / max(price_counts.values()))
                }
        levels['hurst_exponent'] = float((hurst_h + hurst_l) / 2)
        levels['fractal_dimension'] = float(fractal_dim)
        return levels
    
    def _detect_chaotic_attractors(self, closes: np.ndarray) -> Dict:
        """
        基于混沌理论检测价格吸引子
        """
        if len(closes) < 100:
            return {}
        returns = np.diff(np.log(closes))
        # 计算Lyapunov指数（简化的Wolf算法）
        def lyapunov_exponent(series, emb_dim=3, tau=1):
            n = len(series)
            if n < emb_dim:
                return 0
            # 相空间重构
            emb_series = []
            for i in range(n - (emb_dim - 1) * tau):
                emb_series.append(series[i:i + emb_dim * tau:tau])
            if len(emb_series) < 10:
                return 0
            emb_series = np.array(emb_series)
            # 计算最近邻距离
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(emb_series)
            distances, indices = nbrs.kneighbors(emb_series)
            # 平均分离率
            sep_rates = []
            for i in range(len(emb_series) - 1):
                if i < len(distances) - 1:
                    d1 = distances[i, 1]
                    if i + 1 < len(distances):
                        j = indices[i, 1]
                        if j < len(emb_series) - 1:
                            d2 = np.linalg.norm(emb_series[i+1] - emb_series[j+1])
                            if d1 > 0 and d2 > 0:
                                sep_rates.append(np.log(d2 / d1))
            if sep_rates:
                return np.mean(sep_rates)
            return 0
        lyap_exp = lyapunov_exponent(returns[:100])
        # 检测吸引子区域
        hist, bin_edges = np.histogram(closes, bins=20)
        max_bin_idx = np.argmax(hist)
        attractor_price = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
        # 计算吸引子强度
        price_std = np.std(closes)
        attractor_strength = hist[max_bin_idx] / len(closes)
        return {
            'lyapunov_exponent': float(lyap_exp),
            'attractor_price': float(attractor_price),
            'attractor_strength': float(attractor_strength),
            'is_chaotic': abs(lyap_exp) > 0.1
        }
    
    def _calculate_nonlinear_regression_levels(self, closes: np.ndarray) -> Dict:
        """
        使用非线性回归模型识别关键价格水平
        """
        n = len(closes)
        if n < 50:
            return {}
        # 多项式回归识别趋势
        x = np.arange(n)
        y = closes
        # 尝试不同阶数的多项式
        levels = {}
        for degree in [2, 3, 4]:
            try:
                coeffs = np.polyfit(x, y, degree)
                poly = np.poly1d(coeffs)
                # 计算极值点
                derivative = np.polyder(poly)
                roots = np.roots(derivative)
                real_roots = [root.real for root in roots if abs(root.imag) < 1e-10]
                valid_roots = [root for root in real_roots if 0 <= root < n]
                for i, root in enumerate(valid_roots):
                    price_level = poly(root)
                    levels[f'poly_{degree}_level_{i}'] = float(price_level)
                    
            except:
                continue
        # 指数平滑识别水平
        alpha = 0.3
        exp_smooth = np.zeros_like(closes)
        exp_smooth[0] = closes[0]
        for i in range(1, n):
            exp_smooth[i] = alpha * closes[i] + (1 - alpha) * exp_smooth[i-1]
        # 识别平滑后的平台区域
        smooth_diff = np.diff(exp_smooth)
        plateaus = np.where(np.abs(smooth_diff) < np.std(smooth_diff) * 0.5)[0]
        if len(plateaus) > 0:
            plateau_prices = []
            for idx in plateaus:
                if 0 <= idx < len(closes):
                    plateau_prices.append(closes[idx])
            if plateau_prices:
                levels['plateau_level'] = float(np.mean(plateau_prices))
        return levels
    
    def _calculate_spectral_price_levels(self, closes: np.ndarray) -> Dict:
        """
        使用傅里叶变换进行频谱分析，识别周期价格水平
        """
        n = len(closes)
        if n < 64:
            return {}
        # 去趋势
        x = np.arange(n)
        coeffs = np.polyfit(x, closes, 1)
        trend = np.poly1d(coeffs)(x)
        detrended = closes - trend
        # 傅里叶变换
        yf = fft(detrended)
        xf = fftfreq(n, 1)
        # 找出主要频率成分
        amplitudes = np.abs(yf[:n//2])
        freqs = xf[:n//2]
        # 找出前5个主要频率
        if len(amplitudes) > 5:
            top_indices = np.argsort(amplitudes)[-5:][::-1]
            top_freqs = freqs[top_indices]
            top_amplitudes = amplitudes[top_indices]
            levels = {}
            for i, (freq, amp) in enumerate(zip(top_freqs, top_amplitudes)):
                if freq > 0:
                    period = 1 / freq
                    levels[f'spectral_period_{i}'] = float(period)
                    # 计算对应价格水平
                    phase = np.angle(yf[top_indices[i]])
                    spectral_component = amp * np.cos(2 * np.pi * freq * x + phase)
                    # 极值点
                    if len(spectral_component) > 10:
                        spectral_diff = np.diff(spectral_component)
                        zero_crossings = np.where(np.diff(np.sign(spectral_diff)))[0]
                        if len(zero_crossings) > 0:
                            for j, crossing in enumerate(zero_crossings[:3]):
                                if 0 <= crossing < len(closes):
                                    price_level = trend[crossing] + spectral_component[crossing]
                                    levels[f'spectral_level_{i}_{j}'] = float(price_level)
            return levels
        return {}
    
    def _calculate_enhanced_fibonacci_levels(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,current_price: float,fractal_levels: Dict) -> Dict:
        """
        增强版斐波那契水平计算，融入动态权重和自适应调整
        """
        fib_factors = {}
        # 识别多个波动区间（多重分形）
        price_ranges = self._identify_multiple_price_ranges(highs, lows)
        for range_name, (range_high, range_low) in price_ranges.items():
            range_size = range_high - range_low
            for ratio_name, ratio in self.FIB_RATIOS.items():
                factor_key = f'fib_{range_name}_level_{ratio_name}'
                # 根据波动方向调整计算
                if range_high > range_low:  # 上升趋势
                    fib_level = range_high - range_size * ratio
                else:  # 下降趋势
                    fib_level = range_low + range_size * ratio
                # 动态容差（基于波动率和分形维度）
                tolerance = self._calculate_dynamic_tolerance(
                    closes, fractal_levels.get('fractal_dimension', 1.5)
                )
                fib_factors[factor_key] = self._is_price_near_level(
                    current_price, fib_level, tolerance
                )
                # 添加水平强度
                strength_key = f'fib_{range_name}_strength_{ratio_name}'
                fib_factors[strength_key] = self._calculate_level_strength(
                    fib_level, highs, lows, closes
                )
        return fib_factors
    
    def _identify_multiple_price_ranges(self, highs: np.ndarray, lows: np.ndarray) -> Dict:
        """
        识别多个价格波动区间（用于多重斐波那契分析）
        """
        ranges = {}
        # 整体区间
        ranges['overall'] = (np.max(highs), np.min(lows))
        # 近期区间（最近1/3数据）
        n = len(highs)
        recent_idx = n // 3
        ranges['recent'] = (np.max(highs[-recent_idx:]), np.min(lows[-recent_idx:]))
        # 中期区间（中间1/3数据）
        if n >= 9:
            mid_start = n // 3
            mid_end = 2 * n // 3
            ranges['mid'] = (np.max(highs[mid_start:mid_end]), np.min(lows[mid_start:mid_end]))
        # 使用Z-score识别异常波动区间
        from scipy.stats import zscore
        price_ranges = highs - lows
        z_scores = zscore(price_ranges)
        # 高波动区间
        high_vol_idx = np.where(z_scores > 1.5)[0]
        if len(high_vol_idx) > 0:
            high_vol_highs = highs[high_vol_idx]
            high_vol_lows = lows[high_vol_idx]
            ranges['high_vol'] = (np.max(high_vol_highs), np.min(high_vol_lows))
        # 低波动区间
        low_vol_idx = np.where(z_scores < -1.5)[0]
        if len(low_vol_idx) > 0:
            low_vol_highs = highs[low_vol_idx]
            low_vol_lows = lows[low_vol_idx]
            ranges['low_vol'] = (np.max(low_vol_highs), np.min(low_vol_lows))
        return ranges
    
    def _calculate_dynamic_tolerance(self, closes: np.ndarray, fractal_dim: float) -> float:
        """
        基于波动率和分形维度的动态容差计算
        """
        returns = np.diff(np.log(closes))
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.2
        # 分形维度影响容差（分形维度越高，市场越复杂，容差越大）
        fract_factor = 0.01 * (fractal_dim - 1.5) if fractal_dim > 1.5 else 0
        # 基础容差 + 波动率调整 + 分形调整
        base_tolerance = 0.02
        vol_adjustment = min(volatility * 0.1, 0.05)
        return base_tolerance + vol_adjustment + fract_factor
    
    def _calculate_level_strength(self, level: float, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """
        计算价格水平的强度（基于历史触碰次数和反弹力度）
        """
        # 计算触碰次数
        touches = 0
        rebound_strength = 0
        for i in range(len(closes)):
            # 检查K线是否触及该水平
            if lows[i] <= level <= highs[i]:
                touches += 1
                # 计算反弹力度
                if i > 0 and i < len(closes) - 1:
                    # 检查是否是反转点
                    if (lows[i-1] > level and closes[i] > level) or \
                       (highs[i-1] < level and closes[i] < level):
                        rebound = abs(closes[i+1] - level) / level
                        rebound_strength += rebound
        # 归一化强度
        max_touches = len(closes)
        strength = (touches / max_touches) * 0.6 + (rebound_strength / max(1, touches)) * 0.4
        return float(strength)
    
    def _merge_multidimensional_levels(self, fib_factors: Dict,fractal_levels: Dict,chaotic_attractors: Dict,regression_levels: Dict,spectral_levels: Dict,current_price: float) -> Dict:
        """
        融合多维度价格水平
        """
        merged = {}
        # 收集所有价格水平
        all_levels = []
        # 斐波那契水平
        for key, value in fib_factors.items():
            if isinstance(value, bool) and value:
                level_str = key.split('_')[-1]
                if level_str.replace('.', '').isdigit():
                    try:
                        level = float(level_str)
                        all_levels.append(('fibonacci', level, 1.0))
                    except:
                        pass
        # 分形水平
        for key, value in fractal_levels.items():
            if isinstance(value, dict) and 'price' in value:
                all_levels.append(('fractal', value['price'], value.get('strength', 0.5)))
        # 混沌吸引子
        attractor_price = chaotic_attractors.get('attractor_price')
        if attractor_price:
            strength = chaotic_attractors.get('attractor_strength', 0.5)
            all_levels.append(('chaotic', attractor_price, strength))
        # 回归水平
        for key, value in regression_levels.items():
            if isinstance(value, (int, float)):
                all_levels.append(('regression', value, 0.6))
        # 频谱水平
        for key, value in spectral_levels.items():
            if isinstance(value, (int, float)):
                all_levels.append(('spectral', value, 0.7))
        # 聚类分析
        if all_levels:
            from sklearn.cluster import DBSCAN
            prices = np.array([[level[1]] for level in all_levels])
            if len(prices) >= 3:
                # 使用DBSCAN聚类
                clustering = DBSCAN(eps=0.01*np.mean(prices), min_samples=2).fit(prices)
                cluster_levels = {}
                for i, (level_type, price, strength) in enumerate(all_levels):
                    cluster_id = clustering.labels_[i]
                    if cluster_id != -1:  # 不是噪声点
                        cluster_key = f'cluster_{cluster_id}'
                        if cluster_key not in cluster_levels:
                            cluster_levels[cluster_key] = []
                        cluster_levels[cluster_key].append((price, strength, level_type))
                # 计算聚类中心
                for cluster_key, points in cluster_levels.items():
                    if len(points) >= 2:
                        weights = [s for _, s, _ in points]
                        prices_cluster = [p for p, _, _ in points]
                        weighted_price = np.average(prices_cluster, weights=weights)
                        cluster_strength = np.mean(weights)
                        # 计算与当前价格的距离
                        distance = abs(weighted_price - current_price) / current_price
                        merged[f'cluster_price_{cluster_key}'] = float(weighted_price)
                        merged[f'cluster_strength_{cluster_key}'] = float(cluster_strength)
                        merged[f'cluster_distance_{cluster_key}'] = float(distance)
                        # 标记是否接近当前价格
                        if distance < 0.02:
                            merged[f'price_near_cluster_{cluster_key}'] = True
        # 计算水平密度
        if len(all_levels) > 0:
            price_band = 0.02 * current_price
            density_scores = []
            for _, price, _ in all_levels:
                nearby_count = sum(1 for _, p, _ in all_levels if abs(p - price) < price_band)
                density_scores.append(nearby_count)
            merged['price_level_density'] = float(np.mean(density_scores))
            merged['price_level_concentration'] = float(np.max(density_scores) if density_scores else 0)
        return merged
    
    def _calculate_advanced_support_resistance(self,highs: np.ndarray,lows: np.ndarray,closes: np.ndarray,current_price: float,fractal_levels: Dict,chaotic_attractors: Dict) -> Dict:
        """
        高级支撑阻力计算，融入机器学习思想但不训练
        """
        result = {}
        # 使用核密度估计（KDE）识别价格密度区域
        from scipy.stats import gaussian_kde
        try:
            # 价格核密度估计
            prices = np.concatenate([highs, lows])
            if len(prices) > 10:
                kde = gaussian_kde(prices)
                # 生成测试点
                x_test = np.linspace(np.min(prices), np.max(prices), 100)
                density = kde(x_test)
                # 寻找局部极值点
                from scipy.signal import argrelextrema
                max_indices = argrelextrema(density, np.greater)[0]
                min_indices = argrelextrema(density, np.less)[0]
                # 阻力位（价格密度峰值）
                resistance_levels = []
                for idx in max_indices:
                    price = x_test[idx]
                    if price > current_price:
                        resistance_levels.append(float(price))
                # 支撑位（价格密度谷值或上升中的密度峰值）
                support_levels = []
                for idx in max_indices:
                    price = x_test[idx]
                    if price < current_price:
                        support_levels.append(float(price))
                # 添加密度谷值作为潜在支撑
                for idx in min_indices:
                    price = x_test[idx]
                    if price < current_price:
                        # 检查是否是有效的支撑（有足够的密度支撑）
                        if kde(price) > np.mean(density) * 0.7:
                            support_levels.append(float(price))
                result['kde_support_levels'] = sorted(support_levels)[-3:]  # 最近3个
                result['kde_resistance_levels'] = sorted(resistance_levels)[:3]  # 最近3个
        except:
            pass
        # 基于分形维度的支撑阻力
        fractal_dim = fractal_levels.get('fractal_dimension', 1.5)
        # 分形维度影响支撑阻力强度
        if fractal_dim > 1.7:
            # 高分形维度，市场复杂，支撑阻力较弱
            result['fractal_support_strength'] = 0.4
            result['fractal_resistance_strength'] = 0.4
        elif fractal_dim < 1.3:
            # 低分形维度，市场简单，支撑阻力较强
            result['fractal_support_strength'] = 0.8
            result['fractal_resistance_strength'] = 0.8
        else:
            result['fractal_support_strength'] = 0.6
            result['fractal_resistance_strength'] = 0.6
        # 混沌吸引子作为动态支撑阻力
        attractor_price = chaotic_attractors.get('attractor_price')
        if attractor_price:
            if attractor_price < current_price:
                result['attractor_support'] = float(attractor_price)
                result['attractor_support_strength'] = chaotic_attractors.get('attractor_strength', 0.5)
            else:
                result['attractor_resistance'] = float(attractor_price)
                result['attractor_resistance_strength'] = chaotic_attractors.get('attractor_strength', 0.5)
        return result
    
    def _analyze_multiple_time_scales(self, significant_dates: List[datetime], trade_date: datetime) -> Dict:
        """
        多时间尺度分析
        """
        if not significant_dates:
            return {}
        time_windows = {}
        # 计算不同时间尺度的斐波那契窗口命中率
        for days in [5, 10, 20, 40, 60, 120, 240]:
            scale_key = f'scale_{days}d'
            count = 0
            for sig_date in significant_dates:
                days_diff = (trade_date - sig_date).days
                # 检查是否在斐波那契窗口内
                for fib_window in self.FIB_TIME_WINDOWS:
                    if abs(days_diff - fib_window) <= max(1, fib_window // 10):
                        count += 1
                        break
            time_windows[scale_key] = count / len(significant_dates) if significant_dates else 0
        # 计算时间序列的自相似性（分形时间）
        if len(significant_dates) > 10:
            intervals = []
            for i in range(1, len(significant_dates)):
                days_diff = (significant_dates[i] - significant_dates[i-1]).days
                intervals.append(days_diff)
            # 计算间隔的斐波那契比例
            fib_ratios = []
            for i in range(1, len(intervals)):
                if intervals[i-1] > 0:
                    ratio = intervals[i] / intervals[i-1]
                    fib_ratios.append(ratio)
            # 计算接近斐波那契比例的比率
            fib_hits = 0
            for ratio in fib_ratios:
                for fib_ratio in [0.382, 0.5, 0.618, 1.0, 1.618, 2.618]:
                    if abs(ratio - fib_ratio) < 0.1:
                        fib_hits += 1
                        break
            time_windows['time_self_similarity'] = fib_hits / len(fib_ratios) if fib_ratios else 0
        return time_windows
    
    def _detect_harmonic_resonance(self, significant_dates: List[datetime], trade_date: datetime) -> Dict:
        """
        检测谐波共振
        """
        if len(significant_dates) < 5:
            return {}
        resonance = {}
        # 计算时间间隔的谐波关系
        intervals = []
        for i in range(1, len(significant_dates)):
            days_diff = (significant_dates[i] - significant_dates[i-1]).days
            intervals.append(days_diff)
        # 检查谐波序列
        harmonic_patterns = []
        # 斐波那契谐波
        fib_harmonics = 0
        for interval in intervals:
            for fib_num in self.FIB_TIME_WINDOWS:
                if fib_num > 0 and abs(interval - fib_num) <= max(1, fib_num // 10):
                    fib_harmonics += 1
                    break
        resonance['fibonacci_harmonics'] = fib_harmonics / len(intervals) if intervals else 0
        # 卢卡斯谐波
        lucas_harmonics = 0
        for interval in intervals:
            for lucas_num in self.LUCAS_WINDOWS:
                if lucas_num > 0 and abs(interval - lucas_num) <= max(1, lucas_num // 10):
                    lucas_harmonics += 1
                    break
        resonance['lucas_harmonics'] = lucas_harmonics / len(intervals) if intervals else 0
        # 黄金分割谐波
        golden_harmonics = 0
        for i in range(1, len(intervals)):
            if intervals[i-1] > 0:
                ratio = intervals[i] / intervals[i-1]
                if abs(ratio - 1.618) < 0.1 or abs(ratio - 0.618) < 0.1:
                    golden_harmonics += 1
        resonance['golden_harmonics'] = golden_harmonics / max(1, len(intervals) - 1)
        # 共振强度
        total_harmonics = (resonance['fibonacci_harmonics'] + 
                          resonance['lucas_harmonics'] + 
                          resonance['golden_harmonics'])
        resonance['harmonic_resonance_strength'] = total_harmonics / 3
        return resonance
    
    def _analyze_time_fractals(self, significant_dates: List[datetime], trade_date: datetime) -> Dict:
        """
        时间分形分析
        """
        if len(significant_dates) < 8:
            return {}
        fractals = {}
        # 计算时间间隔序列
        intervals = []
        for i in range(1, len(significant_dates)):
            days_diff = (significant_dates[i] - significant_dates[i-1]).days
            intervals.append(days_diff)
        # 计算赫斯特指数（时间分形）
        def hurst(ts):
            lags = range(2, min(20, len(ts)))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        if len(intervals) >= 10:
            hurst_exp = hurst(np.array(intervals))
            fractals['time_hurst_exponent'] = float(hurst_exp)
            # 判断时间序列的性质
            if hurst_exp > 0.6:
                fractals['time_persistence'] = 'persistent'  # 趋势持续
            elif hurst_exp < 0.4:
                fractals['time_persistence'] = 'anti-persistent'  # 均值回归
            else:
                fractals['time_persistence'] = 'random'
        # 计算时间分形维度
        if len(intervals) >= 5:
            # 简化的盒计数法
            max_interval = max(intervals)
            min_interval = min(intervals)
            if max_interval > min_interval:
                n_boxes = 10
                box_size = (max_interval - min_interval) / n_boxes
                boxes_needed = 0
                for interval in intervals:
                    box_idx = int((interval - min_interval) / box_size)
                    boxes_needed = max(boxes_needed, box_idx + 1)
                if boxes_needed > 0 and n_boxes > 0:
                    fractal_dim = np.log(boxes_needed) / np.log(n_boxes)
                    fractals['time_fractal_dimension'] = float(fractal_dim)
        return fractals
    
    def _is_time_in_window(self, trade_date: datetime, significant_dates: List[datetime], window: int) -> bool:
        """
        改进的时间窗口检测（包含相位偏移检测）
        """
        for sig_date in significant_dates:
            days_diff = (trade_date - sig_date).days
            # 动态容差（窗口越大，容差越大）
            tolerance = max(1, window // 10)
            # 检查是否在窗口内
            if abs(days_diff - window) <= tolerance:
                return True
            # 检查谐波窗口（1/2, 2倍等）
            for harmonic in [0.5, 2, 3]:
                harmonic_window = window * harmonic
                if abs(days_diff - harmonic_window) <= tolerance * harmonic:
                    return True
        return False
    
    def _calculate_composite_time_score(self,trade_date: datetime,significant_dates: List[datetime],time_windows: Dict,harmonic_resonance: Dict,time_fractals: Dict) -> Decimal:
        """
        复合时间得分计算
        """
        if not significant_dates:
            return Decimal('0')
        score = 0
        max_score = 100
        # 1. 基础斐波那契窗口命中（40%）
        fib_score = 0
        for window in self.FIB_TIME_WINDOWS:
            if self._is_time_in_window(trade_date, significant_dates, window):
                fib_score += self._get_fib_window_weight(window)
        # 归一化
        fib_max = sum(self._get_fib_window_weight(w) for w in self.FIB_TIME_WINDOWS[:5])
        fib_normalized = min(fib_score / fib_max * 40, 40) if fib_max > 0 else 0
        score += fib_normalized
        # 2. 谐波共振得分（30%）
        harmonic_strength = harmonic_resonance.get('harmonic_resonance_strength', 0)
        score += harmonic_strength * 30
        # 3. 时间分形得分（20%）
        time_hurst = time_fractals.get('time_hurst_exponent', 0.5)
        # Hurst指数接近0.5得分最高（布朗运动），远离0.5得分递减
        hurst_score = 1 - 2 * abs(time_hurst - 0.5)
        score += max(hurst_score, 0) * 20
        # 4. 多时间尺度得分（10%）
        scale_score = 0
        for key, value in time_windows.items():
            if key.startswith('scale_'):
                scale_score += value
        scale_count = sum(1 for key in time_windows.keys() if key.startswith('scale_'))
        scale_normalized = (scale_score / scale_count * 10) if scale_count > 0 else 0
        score += scale_normalized
        return Decimal(str(round(min(score, max_score), 2)))
    
    def _calculate_time_structure_strength(self,time_windows: Dict,harmonic_resonance: Dict,time_fractals: Dict) -> Decimal:
        """
        计算时间结构强度
        """
        strength = 0
        # 时间窗口密度
        window_density = 0
        for key, value in time_windows.items():
            if key.startswith('scale_'):
                window_density += value
        scale_count = sum(1 for key in time_windows.keys() if key.startswith('scale_'))
        if scale_count > 0:
            strength += (window_density / scale_count) * 40
        # 谐波共振强度
        harmonic_strength = harmonic_resonance.get('harmonic_resonance_strength', 0)
        strength += harmonic_strength * 40
        # 时间分形稳定性
        time_hurst = time_fractals.get('time_hurst_exponent', 0.5)
        # Hurst指数越接近0.5越稳定
        stability_score = 1 - 2 * abs(time_hurst - 0.5)
        strength += max(stability_score, 0) * 20
        return Decimal(str(round(min(strength, 100), 2)))
    
    def _analyze_time_price_phase_sync(self,price_factors: Dict,time_factors: Dict,closes: np.ndarray) -> float:
        """
        分析时空相位同步
        """
        if len(closes) < 20:
            return 0.0
        # 价格相位（基于希尔伯特变换）
        from scipy.signal import hilbert
        # 去趋势
        x = np.arange(len(closes))
        coeffs = np.polyfit(x, closes, 1)
        trend = np.poly1d(coeffs)(x)
        detrended = closes - trend
        # 希尔伯特变换获取瞬时相位
        analytic_signal = hilbert(detrended)
        price_phase = np.angle(analytic_signal)
        # 时间相位（基于重要日期的周期）
        if 'time_fractal_dimension' in time_factors:
            time_dim = float(time_factors['time_fractal_dimension'])
        else:
            time_dim = 1.5
        # 计算相位同步指标
        phase_sync = 0
        # 简化的相位差分析
        if len(price_phase) > 10:
            phase_diff = np.diff(price_phase)
            phase_std = np.std(phase_diff)
            # 低相位变化表示高同步
            phase_sync = 1.0 / (1.0 + phase_std)
        # 时间分形维度影响
        phase_sync *= (1.0 - 0.2 * abs(time_dim - 1.5))
        return float(min(max(phase_sync, 0), 1))
    
    def _calculate_nonlinear_coupling(self,price_factors: Dict,time_factors: Dict, closes: np.ndarray) -> float:
        """
        计算非线性耦合强度
        """
        if len(closes) < 30:
            return 0.0
        # 价格序列的复杂度
        price_complexity = self._calculate_sample_entropy(closes)
        # 时间结构的复杂度
        time_complexity = 0.5  # 默认值
        if 'time_fractal_dimension' in time_factors:
            time_dim = float(time_factors['time_fractal_dimension'])
            time_complexity = abs(time_dim - 1.5) * 2  # 偏离1.5越多越复杂
        # 耦合强度（复杂度匹配度）
        complexity_match = 1.0 - abs(price_complexity - time_complexity)
        # 价格水平密度影响
        price_density = price_factors.get('price_level_density', 1.0)
        density_factor = min(price_density / 5.0, 1.0)  # 归一化
        # 时间窗口密度
        time_density = 0
        for key, value in time_factors.items():
            if 'fib_time_window_' in key and value:
                time_density += 1
        time_density_factor = min(time_density / 5.0, 1.0)
        # 综合耦合强度
        coupling = (complexity_match * 0.4 + 
                   density_factor * 0.3 + 
                   time_density_factor * 0.3)
        return float(min(max(coupling, 0), 1))
    
    def _calculate_sample_entropy(self, time_series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        计算样本熵（复杂度度量）
        """
        n = len(time_series)
        if n < m + 1:
            return 0
        # 标准化
        std = np.std(time_series)
        if std == 0:
            return 0
        normalized = (time_series - np.mean(time_series)) / std
        def _maxdist(x_i, x_j):
            return max([abs(x_i[k] - x_j[k]) for k in range(m)])
        def _phi(m):
            x = [[normalized[j] for j in range(i, i + m)] for i in range(n - m + 1)]
            C = 0
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j and _maxdist(x[i], x[j]) <= r:
                        C += 1
            return C / ((n - m + 1) * (n - m))
        if n - m + 1 <= 1:
            return 0
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m == 0 or phi_m1 == 0:
            return 0
        return -np.log(phi_m1 / phi_m)
    
    def _detect_multi_resonance(self,price_factors: Dict,time_factors: Dict,daily_data: List[Dict]) -> Dict:
        """
        检测多重共振
        """
        resonance = {}
        if not daily_data or len(daily_data) < 20:
            return resonance
        closes = np.array([d['close'] for d in daily_data[-20:]])
        volumes = np.array([d['vol'] for d in daily_data[-20:]]) if 'vol' in daily_data[0] else None
        # 1. 价格-时间共振
        price_time_resonance = 0
        active_price_levels = sum(1 for k, v in price_factors.items() 
                                 if ('fib_price_level' in k or 'fib_price_extension' in k) and v)
        active_time_windows = sum(1 for k, v in time_factors.items() 
                                 if ('fib_time_window' in k or 'lucas_time_window' in k) and v)
        if active_price_levels > 0 and active_time_windows > 0:
            price_time_resonance = min(active_price_levels * active_time_windows / 25.0, 1.0)
        resonance['price_time'] = price_time_resonance
        # 2. 价格-成交量共振
        price_volume_resonance = 0
        if volumes is not None and len(volumes) > 5:
            # 计算价格和成交量的相关性
            price_changes = np.diff(closes) / closes[:-1]
            volume_changes = np.diff(volumes) / volumes[:-1]
            if len(price_changes) == len(volume_changes) and len(price_changes) > 3:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                price_volume_resonance = abs(correlation)
        resonance['price_volume'] = price_volume_resonance
        # 3. 多时间框架共振
        multi_timeframe_resonance = 0
        # 检查不同时间尺度的斐波那契窗口
        fib_windows_hit = sum(1 for k, v in time_factors.items() 
                             if 'fib_time_window_' in k and v)
        lucas_windows_hit = sum(1 for k, v in time_factors.items() 
                               if 'lucas_time_window_' in k and v)
        total_windows = len([k for k in time_factors.keys() 
                           if 'fib_time_window_' in k or 'lucas_time_window_' in k])
        if total_windows > 0:
            multi_timeframe_resonance = (fib_windows_hit + lucas_windows_hit) / total_windows
        resonance['multi_timeframe'] = multi_timeframe_resonance
        # 综合共振强度
        resonance['composite_strength'] = (
            price_time_resonance * 0.4 +
            price_volume_resonance * 0.3 +
            multi_timeframe_resonance * 0.3
        )
        return resonance
    
    def _calculate_dynamic_resonance_score(self,time_signals: int,price_signals: int,phase_sync: float,coupling_strength: float,multi_resonance: Dict,daily_data: List[Dict]) -> float:
        """
        动态共振得分计算
        """
        # 基础共振得分
        base_score = min(time_signals * price_signals * 8, 40)
        # 相位同步得分
        phase_score = phase_sync * 20
        # 非线性耦合得分
        coupling_score = coupling_strength * 15
        # 多重共振得分
        multi_res_score = multi_resonance.get('composite_strength', 0) * 15
        # 波动率调整
        volatility = self._calculate_price_volatility_advanced(daily_data)
        vol_factor = 1.0 + min(volatility, 0.5)
        # 成交量确认
        volume_confirmation = self._check_volume_confirmation_advanced(daily_data)
        volume_factor = 1.2 if volume_confirmation else 0.9
        # 计算最终得分
        final_score = (base_score + phase_score + coupling_score + multi_res_score) * vol_factor * volume_factor
        return min(final_score, 100)
    
    def _calculate_price_volatility_advanced(self, daily_data: List[Dict]) -> float:
        """
        高级波动率计算（包含极值和跳跃）
        """
        if len(daily_data) < 20:
            return 0.0
        closes = [d['close'] for d in daily_data[-20:]]
        highs = [d['high'] for d in daily_data[-20:]]
        lows = [d['low'] for d in daily_data[-20:]]
        # 基础波动率（对数收益率标准差）
        returns = np.diff(np.log(closes))
        base_vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        # 极值波动率（日内范围）
        daily_ranges = [(h - l) / l for h, l in zip(highs, lows)]
        range_vol = np.std(daily_ranges) * np.sqrt(252) if daily_ranges else 0
        # 跳跃检测（大幅涨跌）
        jumps = 0
        for i in range(1, len(closes)):
            ret = abs(closes[i] - closes[i-1]) / closes[i-1]
            if ret > 0.03:  # 3%以上视为跳跃
                jumps += 1
        jump_ratio = jumps / len(closes)
        # 综合波动率
        composite_vol = base_vol * 0.5 + range_vol * 0.3 + jump_ratio * 0.2
        return float(composite_vol)
    
    def _check_volume_confirmation_advanced(self, daily_data: List[Dict]) -> bool:
        """
        高级成交量确认（包含分布和异常检测）
        """
        if len(daily_data) < 10:
            return False
        volumes = [d['vol'] for d in daily_data[-10:]]
        if len(volumes) < 5:
            return False
        # 检查成交量分布
        from scipy.stats import skew, kurtosis
        vol_skew = skew(volumes) if len(volumes) > 2 else 0
        vol_kurt = kurtosis(volumes) if len(volumes) > 3 else 0
        # 右偏分布（大部分交易日在低成交量，近期放量）
        is_right_skewed = vol_skew > 0.5
        # 高峰态（成交量集中在某些日子）
        is_peaked = vol_kurt > 1.0
        # 近期成交量 vs 平均水平
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        is_above_avg = recent_volume > avg_volume * 1.2
        # 成交量突破（超过过去N日最高）
        max_historical = max(volumes[:-3]) if len(volumes) > 3 else max(volumes)
        is_breakout = recent_volume > max_historical * 1.1
        # 综合确认
        return is_above_avg or is_breakout or (is_right_skewed and is_peaked)
    
    def _determine_enhanced_resonance_level(self,resonance_score: float,phase_sync: float,coupling_strength: float,multi_resonance: Dict) -> int:
        """
        确定增强的共振级别
        """
        if resonance_score >= 80 and phase_sync >= 0.7 and coupling_strength >= 0.7:
            return 4  # 极强共振
        elif resonance_score >= 70 and phase_sync >= 0.6 and coupling_strength >= 0.6:
            return 3  # 强共振
        elif resonance_score >= 60:
            return 2  # 中等共振
        elif resonance_score >= 50:
            return 1  # 弱共振
        else:
            return 0  # 无共振
    
    def _calculate_resonance_stability(self,price_factors: Dict,time_factors: Dict,daily_data: List[Dict]) -> float:
        """
        计算共振稳定性
        """
        if len(daily_data) < 10:
            return 0.0
        # 价格稳定性
        closes = [d['close'] for d in daily_data[-10:]]
        price_returns = np.diff(np.log(closes)) if len(closes) > 1 else [0]
        price_stability = 1.0 / (1.0 + np.std(price_returns) * 10)
        # 时间结构稳定性
        time_stability = 0.5  # 默认值
        if 'time_fractal_dimension' in time_factors:
            time_dim = float(time_factors['time_fractal_dimension'])
            # 分形维度越接近1.5越稳定（布朗运动）
            time_stability = 1.0 - 2.0 * abs(time_dim - 1.5)
        # 价格水平稳定性
        price_level_stability = 0
        level_prices = []
        for key, value in price_factors.items():
            if isinstance(value, bool) and value:
                # 提取价格水平
                parts = key.split('_')
                for part in parts:
                    if part.replace('.', '').isdigit():
                        try:
                            level = float(part)
                            level_prices.append(level)
                        except:
                            pass
        if len(level_prices) >= 2:
            # 计算价格水平的聚集程度
            level_std = np.std(level_prices) if level_prices else 0
            avg_level = np.mean(level_prices) if level_prices else 0
            if avg_level > 0:
                price_level_stability = 1.0 / (1.0 + level_std / avg_level * 10)
        # 综合稳定性
        composite_stability = (
            price_stability * 0.4 +
            time_stability * 0.3 +
            price_level_stability * 0.3
        )
        return float(min(max(composite_stability, 0), 1))
    
    def _get_fib_window_weight(self, window: int) -> int:
        """
        获取斐波那契时间窗口权重（优化版）
        """
        # 基于市场经验的重要窗口权重
        if window in [3, 5, 8]:
            return 8    # 短期重要窗口
        elif window in [13, 21]:
            return 12   # 中期重要窗口
        elif window in [34, 55]:
            return 15   # 长期重要窗口
        elif window in [89, 144]:
            return 10   # 超长期窗口
        elif window in [233, 377]:
            return 8    # 极长期窗口
        else:
            return 5
    
    def _is_price_near_level(self, current_price: float, fib_level: float, tolerance: float = 0.02) -> bool:
        """
        保持原方法签名，内部逻辑已在前面的方法中重构
        """
        if fib_level == 0:
            return False
        price_diff = abs(current_price - fib_level)
        diff_percent = price_diff / fib_level
        return diff_percent <= tolerance
