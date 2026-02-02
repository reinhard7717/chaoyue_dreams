# services\structural_calculator\fibonacci_time_calculator.py
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

@dataclass
class FibonacciConfig:
    """斐波那契时间窗口配置 - 升级版"""
    # 扩展斐波那契数列（包含卢卡斯序列、佩尔数列等广义斐波那契数列）
    FIB_SEQUENCES = {
        'standard': [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],  # 标准序列
        'lucas': [4, 7, 11, 18, 29, 47, 76, 123, 199, 322],  # 卢卡斯序列
        'pell': [2, 5, 12, 29, 70, 169, 408],  # 佩尔数列
        'golden': [1.618, 2.618, 4.236, 6.854, 11.09, 17.944, 29.034, 46.978, 76.013]  # 黄金分割倍数
    }
    
    # 动态时间窗口容差 - 基于波动率自适应
    VOLATILITY_ADAPTIVE_TOLERANCE = True
    MIN_TOLERANCE = 0.5  # 最小容差天数
    MAX_TOLERANCE = 2.0  # 最大容差天数
    
    # 多周期共振权重矩阵
    RESONANCE_WEIGHTS = {
        'intraday': {'weight': 0.2, 'decay': 0.95},
        'daily': {'weight': 0.35, 'decay': 0.85},
        'weekly': {'weight': 0.25, 'decay': 0.75},
        'monthly': {'weight': 0.2, 'decay': 0.65}
    }
    
    # 价格结构匹配模型参数
    STRUCTURE_MATCHING = {
        'fractal_dimension_threshold': 1.5,  # 分形维度阈值
        'hurst_exponent_window': 20,  # 赫斯特指数计算窗口
        'cycle_detection_sensitivity': 0.7,  # 周期检测灵敏度
    }

class FibonacciTimeCalculator:
    """斐波那契时间结构因子计算器 - 深度重构版"""
    
    def __init__(self, stock_code: str, market_type: str = 'SH'):
        """
        初始化计算器
        Args:
            stock_code: 股票代码
            market_type: 市场类型 SH/SZ/CY/KC/BJ
        """
        self.stock_code = stock_code
        self.market_type = market_type
        self.config = FibonacciConfig()
        
    def calculate_fibonacci_time_windows(self, trade_date: datetime.date,lookback_days: int = 500) -> Dict:
        """
        重构：多维斐波那契时间窗口计算
        核心改进：
        1. 引入广义斐波那契数列
        2. 波动率自适应窗口匹配
        3. 多时间周期同步分析
        4. 分形市场理论应用
        5. 混沌时间序列识别
        """
        try:
            # 1. 获取多周期价格数据
            price_data = self._get_multi_frequency_data(trade_date, lookback_days)
            if price_data.empty:
                return self._get_default_result()
            # 2. 计算市场结构特征
            market_structure = self._analyze_market_structure(price_data)
            # 3. 识别关键时间节点（多维度）
            turning_points = self._identify_multi_dimensional_points(price_data, market_structure)
            # 4. 动态时间窗口匹配（广义斐波那契）
            fib_windows = self._match_generalized_fibonacci_windows(trade_date, turning_points, market_structure)
            # 5. 计算结构强度得分（非线性加权）
            fib_score = self._calculate_structure_score(fib_windows, price_data, market_structure)
            # 6. 识别多周期共振（混沌同步）
            resonance_result = self._identify_multi_scale_resonance(
                trade_date, price_data, fib_windows, market_structure
            )
            # 7. 构建时间结构场（时空耦合）
            time_structure_field = self._build_time_structure_field(
                trade_date, price_data, fib_windows
            )
            # 8. 组合高级结果
            result = self._compile_advanced_results(
                fib_windows, fib_score, resonance_result, time_structure_field
            )
            return result
        except Exception as e:
            print(f"计算斐波那契时间窗口时出错: {e}")
            return self._get_default_result()
    
    def _get_multi_frequency_data(self, trade_date: datetime.date, lookback_days: int) -> pd.DataFrame:
        """获取多周期频率数据（日线、周线、月线特征）"""
        try:
            from_date = trade_date - timedelta(days=lookback_days)
            # 获取日线数据
            daily_queryset = StockDailyBasic.objects.filter(
                stock__stock_code=self.stock_code,
                trade_time__gte=from_date,
                trade_time__lte=trade_date
            ).order_by('trade_time')
            data = []
            for record in daily_queryset:
                data.append({
                    'date': record.trade_time,
                    'close': float(record.close) if record.close else 0,
                    'high': float(record.high) if hasattr(record, 'high') else 0,
                    'low': float(record.low) if hasattr(record, 'low') else 0,
                    'open': float(record.open) if hasattr(record, 'open') else 0,
                    'volume': float(record.vol) if hasattr(record, 'vol') else 0,
                    'amount': float(record.amount) if hasattr(record, 'amount') else 0,
                    'turnover': float(record.turnover_rate) if record.turnover_rate else 0,
                })
            df = pd.DataFrame(data)
            if df.empty:
                return df
            # 计算周线、月线聚合特征
            df.set_index('date', inplace=True)
            # 周线特征（5日滚动窗口）
            df['weekly_close'] = df['close'].rolling(5).mean()
            df['weekly_high'] = df['high'].rolling(5).max()
            df['weekly_low'] = df['low'].rolling(5).min()
            df['weekly_volume'] = df['volume'].rolling(5).mean()
            # 月线特征（20日滚动窗口）
            df['monthly_close'] = df['close'].rolling(20).mean()
            df['monthly_high'] = df['high'].rolling(20).max()
            df['monthly_low'] = df['low'].rolling(20).min()
            # 计算价格变化率
            df['daily_return'] = df['close'].pct_change()
            df['weekly_return'] = df['weekly_close'].pct_change()
            df['monthly_return'] = df['monthly_close'].pct_change()
            # 计算波动率
            df['daily_volatility'] = df['daily_return'].rolling(20).std()
            df['weekly_volatility'] = df['weekly_return'].rolling(20).std()
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            print(f"获取多周期数据失败: {e}")
            return pd.DataFrame()
    
    def _analyze_market_structure(self, price_data: pd.DataFrame) -> Dict:
        """
        深度重构：市场结构多维度分析
        核心算法：
        1. 赫斯特指数（长期记忆性）
        2. 分形维度（市场复杂性）
        3. 波动率聚类效应
        4. 价格-成交量相关性
        5. 市场状态识别（趋势/震荡）
        """
        if price_data.empty:
            return {}
        structure = {}
        # 1. 计算赫斯特指数（重标极差法）
        if len(price_data) >= 100:
            structure['hurst_exponent'] = self._calculate_hurst_exponent(
                price_data['close'].values
            )
        else:
            structure['hurst_exponent'] = 0.5
        # 2. 计算分形维度（盒计数法）
        structure['fractal_dimension'] = self._calculate_fractal_dimension(
            price_data[['close', 'volume']].values
        )
        # 3. 计算波动率结构
        structure['volatility_clustering'] = self._detect_volatility_clustering(
            price_data['daily_return'].values
        )
        # 4. 量价相关性分析
        structure['price_volume_correlation'] = self._analyze_price_volume_relation(
            price_data['close'].values, 
            price_data['volume'].values
        )
        # 5. 市场状态识别
        structure['market_regime'] = self._identify_market_regime(
            price_data['close'].values,
            price_data['daily_volatility'].values
        )
        # 6. 计算自适应时间窗口容差
        structure['dynamic_tolerance'] = self._calculate_dynamic_tolerance(
            price_data['daily_volatility'].values
        )
        return structure
    
    def _identify_multi_dimensional_points(self, price_data: pd.DataFrame, market_structure: Dict) -> List[Dict]:
        """
        重构：多维度关键节点识别
        识别策略：
        1. 价格极值点（局部高低点）
        2. 成交量异常点（量价背离）
        3. 波动率突变点
        4. 技术指标背离点
        5. 市场状态转换点
        """
        if price_data.empty:
            return []
        turning_points = []
        price_series = price_data['close'].values
        volume_series = price_data['volume'].values
        dates = price_data['date'].values
        # 1. 价格转折点检测（使用多尺度算法）
        price_points = self._detect_multi_scale_turning_points(price_series, market_structure)
        # 2. 成交量异常点检测
        volume_points = self._detect_volume_anomalies(volume_series, price_series)
        # 3. 波动率结构突变点
        volatility_points = self._detect_volatility_breaks(price_data['daily_volatility'].values)
        # 4. 多维度融合
        for i, point_type in enumerate(price_points):
            if point_type != 'none':
                # 计算综合强度
                strength = self._calculate_comprehensive_strength(
                    i, price_series, volume_series, market_structure
                )
                # 检查成交量确认
                volume_confirmation = self._check_advanced_volume_confirmation(
                    i, volume_series, price_series
                )
                # 检查技术指标确认
                indicator_confirmation = self._check_indicator_confirmation(
                    i, price_data
                )
                turning_points.append({
                    'date': dates[i],
                    'price': price_series[i],
                    'type': point_type,
                    'strength': strength,
                    'volume_confirmation': volume_confirmation,
                    'indicator_confirmation': indicator_confirmation,
                    'dimensional_score': self._calculate_dimensional_score(
                        strength, volume_confirmation, indicator_confirmation
                    )
                })
        # 5. 按维度评分排序，取前N个
        turning_points.sort(key=lambda x: x['dimensional_score'], reverse=True)
        return turning_points[:30]  # 保留最重要的30个转折点
    
    def _match_generalized_fibonacci_windows(self, current_date: datetime.date,turning_points: List[Dict], market_structure: Dict) -> Dict:
        """
        重构：广义斐波那契窗口匹配
        匹配策略：
        1. 多斐波那契序列同时匹配
        2. 自适应容差机制
        3. 窗口质量分级
        4. 序列共振检测
        """
        # 计算动态容差
        tolerance = market_structure.get('dynamic_tolerance', self.config.MIN_TOLERANCE)
        # 初始化匹配结果
        fib_matches = {}
        for seq_name, seq in self.config.FIB_SEQUENCES.items():
            fib_matches[seq_name] = {fib: [] for fib in seq}
        # 计算时间间隔并进行匹配
        for point in turning_points:
            point_date = point['date']
            days_diff = (current_date - point_date).days
            # 对每个斐波那契序列进行匹配
            for seq_name, seq in self.config.FIB_SEQUENCES.items():
                for fib_num in seq:
                    if abs(days_diff - fib_num) <= tolerance:
                        # 计算匹配质量（考虑多维度）
                        match_quality = self._calculate_advanced_match_quality(
                            days_diff, fib_num, point, seq_name, market_structure
                        )
                        fib_matches[seq_name][fib_num].append({
                            'days_diff': days_diff,
                            'quality': match_quality,
                            'point_info': point,
                            'sequence': seq_name
                        })
        return fib_matches
    
    def _calculate_structure_score(self, fib_windows: Dict, price_data: pd.DataFrame,market_structure: Dict) -> Decimal:
        """
        重构：结构强度得分计算
        计算策略：
        1. 多序列加权融合
        2. 时间衰减效应
        3. 波动率调整
        4. 市场状态权重
        """
        if not fib_windows:
            return Decimal('0.0')
        sequence_scores = []
        sequence_weights = []
        # 对每个斐波那契序列计算得分
        for seq_name, matches in fib_windows.items():
            if any(matches.values()):  # 序列中有匹配
                seq_score = self._calculate_sequence_score(
                    matches, seq_name, market_structure
                )
                # 序列权重（标准序列权重最高）
                weight = self._get_sequence_weight(seq_name)
                sequence_scores.append(seq_score)
                sequence_weights.append(weight)
        if sequence_scores:
            # 加权平均
            weighted_score = np.average(sequence_scores, weights=sequence_weights)
            # 市场状态调整
            market_adjustment = self._adjust_by_market_regime(
                weighted_score, market_structure
            )
            # 波动率调整
            volatility_adjustment = self._adjust_by_volatility(
                market_adjustment, price_data['daily_volatility'].iloc[-1] 
                if not price_data.empty else 0
            )
            # 归一化
            final_score = min(max(volatility_adjustment, 0), 1)
            return Decimal(str(round(final_score, 2)))
        return Decimal('0.0')
    
    def _identify_multi_scale_resonance(self, trade_date: datetime.date,price_data: pd.DataFrame,fib_windows: Dict,market_structure: Dict) -> Dict:
        """
        重构：多尺度时间-价格共振识别
        共振检测策略：
        1. 多时间尺度同步
        2. 价格结构匹配
        3. 成交量验证
        4. 技术指标确认
        """
        resonance_result = {
            'resonance': False,
            'score': 0.0,
            'level': 0,
            'type': 'none',
            'multi_scale_details': []
        }
        if price_data.empty:
            return resonance_result
        # 1. 多时间尺度分析
        time_scales = ['intraday', 'daily', 'weekly', 'monthly']
        for scale in time_scales:
            scale_result = self._analyze_time_scale_resonance(
                scale, trade_date, price_data, fib_windows, market_structure
            )
            if scale_result['resonance']:
                resonance_result['multi_scale_details'].append(scale_result)
        # 2. 综合共振评估
        if resonance_result['multi_scale_details']:
            resonance_result['resonance'] = True
            # 计算综合共振得分
            scale_scores = []
            scale_weights = []
            for detail in resonance_result['multi_scale_details']:
                weight = self.config.RESONANCE_WEIGHTS.get(
                    detail['scale'], {'weight': 0.2}
                )['weight']
                scale_scores.append(detail['score'])
                scale_weights.append(weight)
            resonance_result['score'] = round(
                np.average(scale_scores, weights=scale_weights), 2
            )
            # 确定共振类型和级别
            resonance_type = self._determine_resonance_type(
                resonance_result['multi_scale_details']
            )
            resonance_result['type'] = resonance_type
            resonance_result['level'] = self._determine_resonance_level(
                resonance_result['score'], resonance_type
            )
        return resonance_result
    
    def _build_time_structure_field(self, trade_date: datetime.date,price_data: pd.DataFrame,fib_windows: Dict) -> Dict:
        """
        重构：构建时间结构场
        构建时间序列的能量场，用于预测未来关键时间点
        """
        time_field = {
            'energy_distribution': {},
            'future_nodes': [],
            'critical_periods': []
        }
        if price_data.empty:
            return time_field
        # 1. 计算时间能量分布
        time_field['energy_distribution'] = self._calculate_time_energy_distribution(
            fib_windows, price_data
        )
        # 2. 预测未来关键时间节点
        time_field['future_nodes'] = self._predict_future_time_nodes(
            fib_windows, trade_date
        )
        # 3. 识别关键时间周期
        time_field['critical_periods'] = self._identify_critical_periods(
            fib_windows, time_field['energy_distribution']
        )
        return time_field
    
    def _calculate_hurst_exponent(self, time_series: np.ndarray) -> float:
        """计算赫斯特指数（重标极差法）"""
        if len(time_series) < 100:
            return 0.5
        lags = range(2, 100)
        tau = []
        for lag in lags:
            # 计算子序列
            n = len(time_series)
            k = n // lag
            rs_values = []
            for i in range(k):
                sub_series = time_series[i*lag:(i+1)*lag]
                if len(sub_series) < 2:
                    continue
                # 计算重标极差
                mean = np.mean(sub_series)
                deviations = sub_series - mean
                z = np.cumsum(deviations)
                r = np.max(z) - np.min(z)
                s = np.std(sub_series)
                if s > 0:
                    rs_values.append(r / s)
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        if len(tau) > 1:
            # 线性回归拟合斜率
            x = np.log(lags[:len(tau)])
            y = np.array(tau)
            if len(x) >= 2 and len(y) >= 2:
                try:
                    hurst, _ = np.polyfit(x, y, 1)
                    return hurst
                except:
                    return 0.5
        return 0.5
    
    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """计算分形维度（盒计数法简化版）"""
        if len(data) < 10:
            return 1.0
        # 简化版盒计数
        scales = [2, 4, 8, 16]
        counts = []
        for scale in scales:
            if scale >= len(data):
                continue
            # 计算在该尺度下覆盖数据所需的盒子数
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            ranges = max_vals - min_vals
            if np.any(ranges == 0):
                continue
            # 将数据空间分割成scale*scale的网格
            normalized_data = (data - min_vals) / ranges
            grid_indices = (normalized_data * scale).astype(int)
            # 统计非空网格数
            unique_grids = set()
            for indices in grid_indices:
                unique_grids.add(tuple(indices))
            counts.append(len(unique_grids))
        if len(counts) >= 2:
            # 对数线性拟合
            x = np.log(scales[:len(counts)])
            y = np.log(counts)
            try:
                slope, _ = np.polyfit(x, y, 1)
                return -slope
            except:
                return 1.0
        return 1.0
    
    def _detect_volatility_clustering(self, returns: np.ndarray) -> float:
        """检测波动率聚类效应"""
        if len(returns) < 20:
            return 0.0
        # 计算自相关系数
        abs_returns = np.abs(returns)
        # 计算滞后1期的自相关系数
        if len(abs_returns) > 1:
            correlation = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
            return max(0, correlation)  # 只关心正相关
        else:
            return 0.0
    
    def _analyze_price_volume_relation(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """分析量价关系"""
        if len(prices) < 10 or len(volumes) < 10:
            return 0.0
        # 计算价格变化与成交量变化的相关系数
        price_changes = np.diff(prices)
        volume_changes = np.diff(volumes)
        if len(price_changes) > 0 and len(volume_changes) > 0:
            min_len = min(len(price_changes), len(volume_changes))
            correlation = np.corrcoef(
                price_changes[:min_len], 
                volume_changes[:min_len]
            )[0, 1]
            return correlation
        else:
            return 0.0
    
    def _identify_market_regime(self, prices: np.ndarray, volatilities: np.ndarray) -> str:
        """识别市场状态"""
        if len(prices) < 30:
            return 'neutral'
        # 计算趋势强度
        returns = np.diff(prices)
        if len(returns) == 0:
            return 'neutral'
        # 平均绝对收益
        mean_abs_return = np.mean(np.abs(returns[-20:])) if len(returns) >= 20 else 0
        # 趋势指标
        if len(prices) >= 20:
            ma_short = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
            ma_long = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            trend_strength = (ma_short - ma_long) / (np.std(prices[-20:]) + 1e-10)
        else:
            trend_strength = 0
        # 波动率水平
        current_vol = volatilities[-1] if len(volatilities) > 0 else 0
        # 状态判断
        if abs(trend_strength) > 0.5:
            if trend_strength > 0:
                return 'strong_trend_up'
            else:
                return 'strong_trend_down'
        elif current_vol > np.percentile(volatilities, 75) if len(volatilities) > 10 else False:
            return 'high_volatility'
        elif mean_abs_return < 0.01:
            return 'low_volatility'
        else:
            return 'neutral'
    
    def _calculate_dynamic_tolerance(self, volatilities: np.ndarray) -> float:
        """计算动态容差（基于波动率）"""
        if not self.config.VOLATILITY_ADAPTIVE_TOLERANCE:
            return self.config.MIN_TOLERANCE
        if len(volatilities) < 10:
            return self.config.MIN_TOLERANCE
        current_vol = volatilities[-1]
        median_vol = np.median(volatilities[-20:]) if len(volatilities) >= 20 else current_vol
        # 波动率越高，容差越大
        if median_vol > 0:
            vol_ratio = current_vol / median_vol
            tolerance = self.config.MIN_TOLERANCE * min(vol_ratio, 2.0)
            return min(tolerance, self.config.MAX_TOLERANCE)
        return self.config.MIN_TOLERANCE
    
    def _detect_multi_scale_turning_points(self, price_series: np.ndarray, market_structure: Dict) -> List[str]:
        """多尺度转折点检测"""
        n = len(price_series)
        point_types = ['none'] * n
        if n < 10:
            return point_types
        # 多尺度窗口大小
        scales = [3, 5, 8, 13]
        for scale in scales:
            for i in range(scale, n - scale):
                local_window = price_series[i-scale:i+scale+1]
                # 局部高点检测
                if price_series[i] == np.max(local_window):
                    # 检查是否形成有效高点
                    if (price_series[i] > price_series[i-1] and 
                        price_series[i] > price_series[i+1]):
                        # 根据尺度调整点类型
                        if scale >= 8:
                            point_types[i] = 'major_high'
                        elif point_types[i] == 'none':
                            point_types[i] = 'minor_high'
                # 局部低点检测
                elif price_series[i] == np.min(local_window):
                    if (price_series[i] < price_series[i-1] and 
                        price_series[i] < price_series[i+1]):
                        if scale >= 8:
                            point_types[i] = 'major_low'
                        elif point_types[i] == 'none':
                            point_types[i] = 'minor_low'
        return point_types
    
    def _detect_volume_anomalies(self, volume_series: np.ndarray, price_series: np.ndarray) -> List[bool]:
        """成交量异常点检测"""
        n = len(volume_series)
        anomalies = [False] * n
        if n < 20:
            return anomalies
        # 计算成交量均值和标准差
        volume_mean = np.mean(volume_series)
        volume_std = np.std(volume_series)
        for i in range(n):
            # 成交量异常放大
            if volume_series[i] > volume_mean + 2 * volume_std:
                anomalies[i] = True
            # 量价背离检测（简化版）
            if i > 0 and i < n-1:
                price_up = price_series[i] > price_series[i-1]
                volume_down = volume_series[i] < volume_series[i-1]
                if price_up and volume_down:
                    anomalies[i] = True
        return anomalies
    
    def _detect_volatility_breaks(self, volatilities: np.ndarray) -> List[bool]:
        """波动率结构突变点检测"""
        n = len(volatilities)
        breaks = [False] * n
        if n < 10:
            return breaks
        # 计算波动率变化率
        vol_changes = np.diff(volatilities)
        for i in range(1, n-1):
            # 波动率大幅变化
            if i < len(vol_changes) and abs(vol_changes[i]) > np.std(vol_changes) * 2:
                breaks[i] = True
        return breaks
    
    def _calculate_comprehensive_strength(self, index: int, price_series: np.ndarray,volume_series: np.ndarray,market_structure: Dict) -> float:
        """计算综合强度"""
        n = len(price_series)
        if index < 10 or index >= n - 10:
            return 0.0
        # 1. 价格变化强度
        lookback = min(10, index, n - index - 1)
        forward_changes = []
        backward_changes = []
        for i in range(1, lookback + 1):
            if index + i < n:
                forward_change = abs(price_series[index] - price_series[index + i]) / price_series[index]
                forward_changes.append(forward_change)
            if index - i >= 0:
                backward_change = abs(price_series[index] - price_series[index - i]) / price_series[index]
                backward_changes.append(backward_change)
        price_strength = np.mean(forward_changes + backward_changes) if (forward_changes or backward_changes) else 0
        # 2. 成交量确认强度
        if index < len(volume_series):
            avg_volume = np.mean(volume_series[max(0, index-5):min(n, index+5)])
            current_volume = volume_series[index]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_strength = min(volume_ratio / 2, 1.0)  # 归一化
        else:
            volume_strength = 0.5
        # 3. 市场状态权重
        market_regime = market_structure.get('market_regime', 'neutral')
        if market_regime in ['strong_trend_up', 'strong_trend_down']:
            regime_weight = 1.2
        elif market_regime == 'high_volatility':
            regime_weight = 1.1
        else:
            regime_weight = 1.0
        # 综合强度
        strength = (price_strength * 0.6 + volume_strength * 0.4) * regime_weight
        return min(strength, 1.0)
    
    def _check_advanced_volume_confirmation(self, index: int, volume_series: np.ndarray,price_series: np.ndarray) -> bool:
        """高级成交量确认检查"""
        if index >= len(volume_series) or index >= len(price_series):
            return False
        n = len(volume_series)
        # 1. 绝对成交量检查
        if index > 0:
            avg_volume = np.mean(volume_series[max(0, index-20):index])
            if volume_series[index] > avg_volume * 1.5:
                return True
        # 2. 量价关系检查
        if index > 0 and index < n-1:
            price_change = price_series[index] - price_series[index-1]
            volume_change = volume_series[index] - volume_series[index-1]
            # 量价齐升或量价齐跌
            if price_change * volume_change > 0:
                return True
        return False
    
    def _check_indicator_confirmation(self, index: int, price_data: pd.DataFrame) -> bool:
        """技术指标确认检查"""
        # 简化版技术指标确认
        if len(price_data) < 20:
            return False
        # 检查是否在移动平均线附近
        if 'weekly_close' in price_data.columns and 'monthly_close' in price_data.columns:
            if index < len(price_data):
                price = price_data.iloc[index]['close']
                ma_short = price_data.iloc[index]['weekly_close']
                ma_long = price_data.iloc[index]['monthly_close']
                # 价格接近重要均线
                if abs(price - ma_short) / price < 0.02 or abs(price - ma_long) / price < 0.02:
                    return True
        return False
    
    def _calculate_dimensional_score(self, strength: float, volume_confirmation: bool,indicator_confirmation: bool) -> float:
        """计算维度评分"""
        score = strength
        if volume_confirmation:
            score *= 1.2
        if indicator_confirmation:
            score *= 1.1
        return score
    
    def _calculate_advanced_match_quality(self, days_diff: int, fib_num: int,point_info: Dict,sequence_name: str,market_structure: Dict) -> float:
        """计算高级匹配质量"""
        # 1. 时间匹配质量
        tolerance = market_structure.get('dynamic_tolerance', self.config.MIN_TOLERANCE)
        time_diff = abs(days_diff - fib_num)
        time_quality = 1 - (time_diff / tolerance)
        # 2. 转折点质量
        point_quality = point_info.get('dimensional_score', 0.5)
        # 3. 序列权重
        sequence_weight = 1.0
        if sequence_name == 'standard':
            sequence_weight = 1.2
        elif sequence_name == 'lucas':
            sequence_weight = 1.1
        # 4. 市场状态调整
        market_regime = market_structure.get('market_regime', 'neutral')
        if market_regime in ['strong_trend_up', 'strong_trend_down']:
            market_weight = 1.1
        else:
            market_weight = 1.0
        # 综合质量
        quality = time_quality * 0.4 + point_quality * 0.6
        quality *= sequence_weight * market_weight
        return min(quality, 1.0)
    
    def _calculate_sequence_score(self, matches: Dict, sequence_name: str,market_structure: Dict) -> float:
        """计算序列得分"""
        if not any(matches.values()):
            return 0.0
        # 收集所有匹配的质量
        all_qualities = []
        for fib_num, match_list in matches.items():
            if match_list:
                # 取该窗口的最高质量
                max_quality = max([m['quality'] for m in match_list])
                all_qualities.append(max_quality)
        if not all_qualities:
            return 0.0
        # 考虑匹配数量和集中度
        match_count = sum(len(m) for m in matches.values())
        count_factor = min(match_count / 5, 1.0)
        # 质量分布
        mean_quality = np.mean(all_qualities)
        # 序列得分
        sequence_score = mean_quality * (0.6 + 0.4 * count_factor)
        # 序列权重调整
        if sequence_name == 'standard':
            sequence_score *= 1.1
        return min(sequence_score, 1.0)
    
    def _get_sequence_weight(self, sequence_name: str) -> float:
        """获取序列权重"""
        weights = {
            'standard': 1.0,
            'lucas': 0.8,
            'pell': 0.7,
            'golden': 0.9
        }
        return weights.get(sequence_name, 0.5)
    
    def _adjust_by_market_regime(self, score: float, market_structure: Dict) -> float:
        """根据市场状态调整得分"""
        market_regime = market_structure.get('market_regime', 'neutral')
        # 不同市场状态下的调整因子
        adjustments = {
            'strong_trend_up': 1.1,
            'strong_trend_down': 1.1,
            'high_volatility': 0.9,
            'low_volatility': 1.0,
            'neutral': 1.0
        }
        adjustment = adjustments.get(market_regime, 1.0)
        return score * adjustment
    
    def _adjust_by_volatility(self, score: float, current_volatility: float) -> float:
        """根据波动率调整得分"""
        if current_volatility > 0.03:  # 高波动率
            return score * 0.9
        elif current_volatility < 0.01:  # 低波动率
            return score * 1.05
        else:
            return score
    
    def _analyze_time_scale_resonance(self, scale: str, trade_date: datetime.date,price_data: pd.DataFrame,fib_windows: Dict,market_structure: Dict) -> Dict:
        """分析特定时间尺度的共振"""
        scale_result = {
            'scale': scale,
            'resonance': False,
            'score': 0.0,
            'details': []
        }
        # 不同尺度的分析参数
        scale_params = {
            'intraday': {'lookback': 10, 'weight': 0.2},
            'daily': {'lookback': 60, 'weight': 0.35},
            'weekly': {'lookback': 120, 'weight': 0.25},
            'monthly': {'lookback': 240, 'weight': 0.2}
        }
        params = scale_params.get(scale, scale_params['daily'])
        # 简化的共振分析
        if not price_data.empty:
            # 检查是否有匹配的时间窗口
            has_matches = False
            for seq_name, matches in fib_windows.items():
                if any(matches.values()):
                    has_matches = True
                    break
            if has_matches:
                # 检查价格位置
                current_price = price_data['close'].iloc[-1]
                price_range = price_data['close'].max() - price_data['close'].min()
                if price_range > 0:
                    # 计算价格在斐波那契位附近的可能性
                    price_position = (current_price - price_data['close'].min()) / price_range
                    # 简化的斐波那契价格位
                    fib_price_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                    for fib_level in fib_price_levels:
                        if abs(price_position - fib_level) < 0.05:  # 5%容差
                            scale_result['resonance'] = True
                            scale_result['score'] = 1 - abs(price_position - fib_level) * 10
                            scale_result['details'].append({
                                'price_level': fib_level,
                                'position': price_position
                            })
        return scale_result
    
    def _determine_resonance_type(self, multi_scale_details: List[Dict]) -> str:
        """确定共振类型"""
        if not multi_scale_details:
            return 'none'
        # 检查共振的尺度分布
        scales_with_resonance = [d for d in multi_scale_details if d['resonance']]
        if len(scales_with_resonance) >= 3:
            return 'multi_scale_strong'
        elif len(scales_with_resonance) == 2:
            return 'dual_scale_moderate'
        elif len(scales_with_resonance) == 1:
            return 'single_scale_weak'
        else:
            return 'none'
    
    def _determine_resonance_level(self, score: float, resonance_type: str) -> int:
        """确定共振级别"""
        # 根据得分和类型确定级别
        base_thresholds = {
            'multi_scale_strong': 0.7,
            'dual_scale_moderate': 0.75,
            'single_scale_weak': 0.8,
            'none': 1.0
        }
        threshold = base_thresholds.get(resonance_type, 0.8)
        if score >= threshold * 1.2:
            return 4  # 顶级共振
        elif score >= threshold * 1.1:
            return 3  # 高级共振
        elif score >= threshold:
            return 2  # 中级共振
        elif score >= threshold * 0.9:
            return 1  # 初级共振
        else:
            return 0
    
    def _calculate_time_energy_distribution(self, fib_windows: Dict,price_data: pd.DataFrame) -> Dict:
        """计算时间能量分布"""
        energy_dist = {}
        # 简化的能量分布计算
        for seq_name, matches in fib_windows.items():
            for fib_num, match_list in matches.items():
                if match_list:
                    # 计算该窗口的总能量
                    total_energy = sum([m['quality'] ** 2 for m in match_list])
                    energy_dist[f"{seq_name}_{fib_num}"] = total_energy
        return energy_dist
    
    def _predict_future_time_nodes(self, fib_windows: Dict,current_date: datetime.date) -> List[Dict]:
        """预测未来关键时间节点"""
        future_nodes = []
        # 基于当前匹配窗口预测未来
        for seq_name, matches in fib_windows.items():
            for fib_num, match_list in matches.items():
                if match_list and len(match_list) > 0:
                    # 计算平均质量
                    avg_quality = np.mean([m['quality'] for m in match_list])
                    if avg_quality > 0.6:
                        # 预测下一个斐波那契窗口
                        next_date = current_date + timedelta(days=fib_num)
                        future_nodes.append({
                            'date': next_date,
                            'fib_number': fib_num,
                            'sequence': seq_name,
                            'expected_quality': avg_quality * 0.8,  # 衰减因子
                            'source_windows': len(match_list)
                        })
        # 按预期质量排序
        future_nodes.sort(key=lambda x: x['expected_quality'], reverse=True)
        return future_nodes[:10]  # 返回前10个预测节点
    
    def _identify_critical_periods(self, fib_windows: Dict,energy_distribution: Dict) -> List[int]:
        """识别关键时间周期"""
        critical_periods = []
        # 寻找能量集中的周期
        periods = []
        energies = []
        for key, energy in energy_distribution.items():
            # 解析斐波那契数字
            parts = key.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                period = int(parts[-1])
                periods.append(period)
                energies.append(energy)
        if periods:
            # 找出能量前3的周期
            sorted_indices = np.argsort(energies)[::-1]
            for idx in sorted_indices[:3]:
                critical_periods.append(periods[idx])
        return critical_periods
    
    def _compile_advanced_results(self, fib_windows: Dict, fib_score: Decimal,resonance_result: Dict, time_structure_field: Dict) -> Dict:
        """编译高级结果"""
        # 提取各序列的窗口匹配结果
        standard_windows = {}
        lucas_windows = {}
        pell_windows = {}
        golden_windows = {}
        for seq_name, matches in fib_windows.items():
            for fib_num, match_list in matches.items():
                if match_list:
                    if seq_name == 'standard':
                        standard_windows[fib_num] = True
                    elif seq_name == 'lucas':
                        lucas_windows[fib_num] = True
                    elif seq_name == 'pell':
                        pell_windows[fib_num] = True
                    elif seq_name == 'golden':
                        golden_windows[fib_num] = True
        # 构建结果字典
        result = {
            # 标准斐波那契窗口
            'fib_time_window_3': standard_windows.get(3, False),
            'fib_time_window_5': standard_windows.get(5, False),
            'fib_time_window_8': standard_windows.get(8, False),
            'fib_time_window_13': standard_windows.get(13, False),
            'fib_time_window_21': standard_windows.get(21, False),
            'fib_time_window_34': standard_windows.get(34, False),
            'fib_time_window_55': standard_windows.get(55, False),
            'fib_time_window_89': standard_windows.get(89, False),
            'fib_time_window_144': standard_windows.get(144, False),
            'fib_time_window_233': standard_windows.get(233, False),
            # 广义斐波那契窗口
            'fib_lucas_window_4': lucas_windows.get(4, False),
            'fib_lucas_window_7': lucas_windows.get(7, False),
            'fib_lucas_window_11': lucas_windows.get(11, False),
            'fib_lucas_window_18': lucas_windows.get(18, False),
            'fib_lucas_window_29': lucas_windows.get(29, False),
            'fib_pell_window_2': pell_windows.get(2, False),
            'fib_pell_window_5': pell_windows.get(5, False),
            'fib_pell_window_12': pell_windows.get(12, False),
            'fib_pell_window_29': pell_windows.get(29, False),
            'fib_golden_window_1618': golden_windows.get(1.618, False),
            'fib_golden_window_2618': golden_windows.get(2.618, False),
            'fib_golden_window_4236': golden_windows.get(4.236, False),
            # 斐波那契时间得分
            'fib_time_score': fib_score,
            # 时间-价格共振
            'fib_time_price_resonance': resonance_result['resonance'],
            'resonance_score': Decimal(str(resonance_result['score'])) if resonance_result['resonance'] else None,
            'resonance_level': resonance_result['level'] if resonance_result['resonance'] else None,
            'resonance_type': resonance_result['type'] if resonance_result['resonance'] else 'none',
            # 时间结构场信息（存储在remarks中）
            'remarks': f"时间结构场: 关键周期{time_structure_field['critical_periods']}"
        }
        return result
    
    def _get_default_result(self) -> Dict:
        """获取默认结果"""
        return {
            'fib_time_window_3': False,
            'fib_time_window_5': False,
            'fib_time_window_8': False,
            'fib_time_window_13': False,
            'fib_time_window_21': False,
            'fib_time_window_34': False,
            'fib_time_window_55': False,
            'fib_time_window_89': False,
            'fib_time_window_144': False,
            'fib_time_window_233': False,
            'fib_lucas_window_4': False,
            'fib_lucas_window_7': False,
            'fib_lucas_window_11': False,
            'fib_lucas_window_18': False,
            'fib_lucas_window_29': False,
            'fib_pell_window_2': False,
            'fib_pell_window_5': False,
            'fib_pell_window_12': False,
            'fib_pell_window_29': False,
            'fib_golden_window_1618': False,
            'fib_golden_window_2618': False,
            'fib_golden_window_4236': False,
            'fib_time_score': Decimal('0.0'),
            'fib_time_price_resonance': False,
            'resonance_score': None,
            'resonance_level': None,
            'resonance_type': 'none',
            'remarks': '默认结果：无有效匹配'
        }
