# services\structural_calculator\volatility_structure_calculator.py
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

class VolatilityStructureCalculator:
    """
    波动率结构因子计算器 - 基于幻方量化A股经验
    结合日线和分钟线数据计算波动率结构因子
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化计算器
        Args:
            config: 计算配置参数
        """
        self.config = config or self._get_default_config()
        # 幻方量化经验参数
        self.high_volatility_threshold = 0.15  # 高波动率阈值
        self.low_volatility_threshold = 0.05   # 低波动率阈值
        self.compression_period = 20           # 波动率压缩观察期
        self.expansion_threshold = 1.5         # 波动率扩张倍数阈值
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'atr_period': 14,           # ATR计算周期
            'bb_period': 20,            # 布林带周期
            'bb_std': 2.0,              # 布林带标准差倍数
            'volatility_window': 20,    # 波动率计算窗口
            'minute_volatility_window': 60,  # 分钟线波动率窗口
            'compression_lookback': 50, # 压缩状态回看期
            'expansion_lookback': 20,   # 扩张状态回看期
            'regime_classification': {  # 波动率状态分类阈值
                'very_low': 0.03,
                'low': 0.06,
                'medium': 0.12,
                'high': 0.20,
                'very_high': 0.30
            }
        }
    
    def calculate_daily_volatility_factors(self, daily_data: pd.DataFrame,minute_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        计算日线级别的波动率结构因子
        Args:
            daily_data: 日线数据DataFrame，包含OHLCV
            minute_data: 分钟线数据DataFrame（可选）
        Returns:
            波动率结构因子字典
        """
        if daily_data.empty or len(daily_data) < 30:
            return self._get_empty_volatility_factors()
        # 计算基本波动率指标
        volatility_factors = {}
        # 1. 计算ATR相关指标
        atr_factors = self._calculate_atr_factors(daily_data)
        volatility_factors.update(atr_factors)
        # 2. 计算历史波动率
        hist_vol_factors = self._calculate_historical_volatility(daily_data)
        volatility_factors.update(hist_vol_factors)
        # 3. 计算波动率状态
        regime_factors = self._calculate_volatility_regime(daily_data)
        volatility_factors.update(regime_factors)
        # 4. 计算波动率压缩/扩张
        compression_factors = self._calculate_volatility_compression_expansion(daily_data)
        volatility_factors.update(compression_factors)
        # 5. 如果提供分钟线数据，计算日内波动率因子
        if minute_data is not None:
            intraday_factors = self._calculate_intraday_volatility_factors(minute_data)
            volatility_factors.update(intraday_factors)
            # 结合日内波动率更新波动率状态
            combined_regime = self._combine_intraday_daily_regime(
                volatility_factors.get('volatility_regime'),
                intraday_factors.get('intraday_volatility_regime', 'medium')
            )
            volatility_factors['volatility_regime'] = combined_regime
        # 6. 计算波动率结构强度
        strength_factors = self._calculate_volatility_strength(volatility_factors)
        volatility_factors.update(strength_factors)
        return volatility_factors
    
    def _calculate_atr_factors(self, data: pd.DataFrame) -> Dict:
        """
        计算ATR相关因子
        """
        # 计算ATR
        atr = ta.atr(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            length=self.config['atr_period']
        )
        # 计算ATR相对位置（当前ATR在最近N日中的百分位）
        recent_atr = atr.tail(self.config['volatility_window']).dropna()
        if len(recent_atr) > 0:
            current_atr = recent_atr.iloc[-1]
            atr_percentile = np.percentile(recent_atr, current_atr) if len(recent_atr) > 1 else 0.5
        else:
            current_atr = 0
            atr_percentile = 0.5
        # 计算ATR变化率
        atr_returns = atr.pct_change().dropna()
        atr_momentum = atr_returns.tail(5).mean() if len(atr_returns) >= 5 else 0
        # ATR与价格比值（标准化波动率）
        close_prices = data['close'].tail(self.config['atr_period']).dropna()
        if len(close_prices) > 0:
            avg_close = close_prices.mean()
            atr_ratio = current_atr / avg_close if avg_close > 0 else 0
        else:
            atr_ratio = 0
        return {
            'atr': float(current_atr),
            'atr_position': float(atr_percentile),
            'atr_momentum': float(atr_momentum),
            'atr_ratio': float(atr_ratio),
            'atr_trend': 'rising' if atr_momentum > 0.01 else 'falling' if atr_momentum < -0.01 else 'stable'
        }
    
    def _calculate_historical_volatility(self, data: pd.DataFrame) -> Dict:
        """
        计算历史波动率因子
        """
        returns = data['close'].pct_change().dropna()
        if len(returns) < 20:
            return {
                'historical_volatility_20d': 0,
                'historical_volatility_60d': 0,
                'volatility_ratio_short_long': 1.0,
                'volatility_clustering': False
            }
        # 计算不同周期的历史波动率
        vol_20d = returns.tail(20).std() * np.sqrt(252)  # 年化
        vol_60d = returns.tail(60).std() * np.sqrt(252)
        # 短期/长期波动率比率
        vol_ratio = vol_20d / vol_60d if vol_60d > 0 else 1.0
        # 波动率聚集性（GARCH效应）
        # 计算自相关性判断是否存在波动率聚集
        volatility_clustering = self._check_volatility_clustering(returns.tail(60))
        # 波动率微笑/偏斜（需要更多数据，这里简化）
        volatility_smile = self._assess_volatility_smile(returns)
        return {
            'historical_volatility_20d': float(vol_20d),
            'historical_volatility_60d': float(vol_60d),
            'volatility_ratio_short_long': float(vol_ratio),
            'volatility_clustering': volatility_clustering,
            'volatility_smile_strength': float(volatility_smile)
        }
    
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> Dict:
        """
        识别波动率状态
        """
        # 计算多个波动率指标
        returns = data['close'].pct_change().dropna()
        if len(returns) < 30:
            return {'volatility_regime': 'unknown'}
        # 1. 基于历史波动率分类
        hist_vol = returns.tail(60).std() * np.sqrt(252)
        # 2. 基于ATR比率分类
        atr_ratio = self._calculate_atr_factors(data)['atr_ratio']
        # 3. 基于价格波动范围分类
        price_range = (data['high'].tail(20) - data['low'].tail(20)).mean()
        avg_price = data['close'].tail(20).mean()
        range_ratio = price_range / avg_price if avg_price > 0 else 0
        # 4. 基于波动率变化率分类
        vol_changes = returns.rolling(20).std().pct_change().dropna().tail(10)
        vol_momentum = vol_changes.mean() if len(vol_changes) > 0 else 0
        # 综合评分
        regime_score = self._calculate_regime_score(hist_vol, atr_ratio, range_ratio, vol_momentum)
        # 确定波动率状态
        regime = self._classify_volatility_regime(regime_score, hist_vol, vol_momentum)
        return {
            'volatility_regime': regime,
            'regime_score': float(regime_score),
            'regime_stability': self._assess_regime_stability(data)
        }
    
    def _calculate_volatility_compression_expansion(self, data: pd.DataFrame) -> Dict:
        """
        计算波动率压缩和扩张状态
        """
        # 计算布林带宽度作为波动率代理
        bb = ta.bbands(
            close=data['close'],
            length=self.config['bb_period'],
            std=self.config['bb_std']
        )
        if bb is None or bb.empty:
            return {
                'volatility_compression': False,
                'volatility_expansion': False,
                'boll_squeeze': False,
                'boll_breakout': False
            }
        bb_width = (bb[f'BBU_{self.config["bb_period"]}_{self.config["bb_std"]}'] - 
                   bb[f'BBL_{self.config["bb_period"]}_{self.config["bb_std"]}']) / bb[f'BBM_{self.config["bb_period"]}_{self.config["bb_std"]}']
        if bb_width.empty or len(bb_width) < 30:
            return {
                'volatility_compression': False,
                'volatility_expansion': False,
                'boll_squeeze': False,
                'boll_breakout': False
            }
        current_width = bb_width.iloc[-1]
        recent_widths = bb_width.tail(self.config['compression_lookback'])
        # 波动率压缩：当前宽度低于历史低位
        compression_threshold = recent_widths.quantile(0.2)
        is_compressed = current_width < compression_threshold if len(recent_widths) > 5 else False
        # 波动率扩张：当前宽度显著高于近期均值
        expansion_threshold = recent_widths.mean() * self.config['expansion_threshold']
        is_expanded = current_width > expansion_threshold if len(recent_widths) > 5 else False
        # 布林带挤压（特殊形式的压缩）
        squeeze_period = 10
        if len(bb_width) >= squeeze_period:
            recent_squeeze = bb_width.tail(squeeze_period)
            squeeze_threshold = recent_squeeze.mean() * 0.7
            is_squeeze = all(recent_squeeze < squeeze_threshold)
        else:
            is_squeeze = False
        # 布林带突破检测
        is_breakout = self._detect_boll_breakout(data, bb)
        return {
            'volatility_compression': bool(is_compressed),
            'volatility_expansion': bool(is_expanded),
            'boll_squeeze': bool(is_squeeze),
            'boll_breakout': bool(is_breakout),
            'boll_bandwidth': float(current_width) if not pd.isna(current_width) else 0,
            'compression_degree': float(1 - (current_width / compression_threshold)) if compression_threshold > 0 else 0
        }
    
    def _calculate_intraday_volatility_factors(self, minute_data: pd.DataFrame) -> Dict:
        """
        基于分钟线计算日内波动率因子
        """
        if minute_data.empty or len(minute_data) < 100:
            return {}
        # 确保数据按时间排序
        minute_data = minute_data.sort_index()
        # 1. 计算日内收益率波动
        intraday_returns = minute_data['close'].pct_change().dropna()
        # 2. 计算不同时间段的波动率
        morning_vol = self._calculate_session_volatility(minute_data, '09:30', '11:30')
        afternoon_vol = self._calculate_session_volatility(minute_data, '13:00', '15:00')
        # 3. 计算日内波动率模式
        volatility_pattern = self._analyze_intraday_volatility_pattern(minute_data)
        # 4. 计算开盘/收盘波动率
        opening_vol = self._calculate_opening_volatility(minute_data)
        closing_vol = self._calculate_closing_volatility(minute_data)
        # 5. 日内波动率状态
        intraday_regime = self._classify_intraday_volatility_regime(
            intraday_returns.std() * np.sqrt(240)  # 日内年化波动率
        )
        # 6. 计算分钟线波动率结构
        intraday_structure = self._analyze_intraday_volatility_structure(minute_data)
        return {
            'intraday_volatility': float(intraday_returns.std() * np.sqrt(240)),
            'intraday_volatility_regime': intraday_regime,
            'morning_volatility': float(morning_vol),
            'afternoon_volatility': float(afternoon_vol),
            'volatility_ratio_am_pm': float(morning_vol / afternoon_vol) if afternoon_vol > 0 else 1.0,
            'opening_volatility': float(opening_vol),
            'closing_volatility': float(closing_vol),
            'intraday_volatility_pattern': volatility_pattern,
            'intraday_volatility_structure': intraday_structure
        }
    
    def _calculate_volatility_strength(self, factors: Dict) -> Dict:
        """
        计算波动率结构强度
        """
        # 综合多个指标计算强度得分
        strength_score = 0
        confidence_score = 0
        # 1. 波动率状态强度
        regime_strength = {
            'very_low': 0.2,
            'low': 0.4,
            'medium': 0.6,
            'high': 0.8,
            'very_high': 1.0,
            'unknown': 0.5
        }.get(factors.get('volatility_regime', 'unknown'), 0.5)
        # 2. 压缩/扩张状态强度
        compression_strength = 0.8 if factors.get('volatility_compression', False) else 0.2
        expansion_strength = 0.8 if factors.get('volatility_expansion', False) else 0.2
        # 3. ATR位置强度
        atr_position = factors.get('atr_position', 0.5)
        atr_strength = abs(atr_position - 0.5) * 2  # 离0.5越远强度越高
        # 4. 波动率聚集强度
        clustering_strength = 0.8 if factors.get('volatility_clustering', False) else 0.3
        # 综合得分
        weights = {
            'regime': 0.3,
            'compression': 0.25,
            'expansion': 0.25,
            'atr': 0.1,
            'clustering': 0.1
        }
        strength_score = (
            regime_strength * weights['regime'] +
            compression_strength * weights['compression'] +
            expansion_strength * weights['expansion'] +
            atr_strength * weights['atr'] +
            clustering_strength * weights['clustering']
        )
        # 置信度得分（基于数据质量和指标一致性）
        confidence_factors = [
            factors.get('regime_stability', 0.5),
            1.0 - abs(compression_strength - expansion_strength),  # 压缩扩张互斥性
            min(1.0, len(factors) / 10)  # 数据完整性
        ]
        confidence_score = np.mean(confidence_factors)
        return {
            'volatility_structure_strength': float(strength_score),
            'volatility_signal_confidence': float(confidence_score)
        }
    
    # ====================== 辅助方法 ======================
    
    def _check_volatility_clustering(self, returns: pd.Series) -> bool:
        """检查波动率聚集现象"""
        if len(returns) < 30:
            return False
        # 计算绝对收益率的自相关性
        abs_returns = returns.abs()
        autocorr = abs_returns.autocorr(lag=1)
        # 如果有显著的正自相关性，则存在波动率聚集
        return autocorr > 0.1
    
    def _assess_volatility_smile(self, returns: pd.Series) -> float:
        """评估波动率微笑/偏斜（简化版）"""
        if len(returns) < 100:
            return 0.5
        # 计算不同幅度的收益波动率
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(positive_returns) > 10 and len(negative_returns) > 10:
            vol_pos = positive_returns.std()
            vol_neg = negative_returns.std()
            smile_ratio = vol_neg / vol_pos if vol_pos > 0 else 1.0
            return float(min(2.0, max(0.5, smile_ratio)))
        return 1.0
    
    def _calculate_regime_score(self, hist_vol: float, atr_ratio: float, range_ratio: float, vol_momentum: float) -> float:
        """计算波动率状态综合得分"""
        # 标准化各项指标
        hist_vol_norm = min(1.0, hist_vol / 0.5)  # 假设0.5为极高波动率
        atr_ratio_norm = min(1.0, atr_ratio / 0.1)  # 假设0.1为极高ATR比率
        range_ratio_norm = min(1.0, range_ratio / 0.15)  # 假设0.15为极高价格范围
        # 动量方向调整
        momentum_factor = 1.0 + vol_momentum * 5  # 放大动量影响
        # 加权得分
        weights = [0.4, 0.3, 0.2, 0.1]
        scores = [hist_vol_norm, atr_ratio_norm, range_ratio_norm, 
                  max(0, min(1, (momentum_factor - 0.5) * 2))]
        regime_score = np.average(scores, weights=weights)
        return float(regime_score)
    
    def _classify_volatility_regime(self, regime_score: float, hist_vol: float, vol_momentum: float) -> str:
        """分类波动率状态"""
        thresholds = self.config['regime_classification']
        # 基于历史波动率
        if hist_vol < thresholds['very_low']:
            base_regime = 'very_low'
        elif hist_vol < thresholds['low']:
            base_regime = 'low'
        elif hist_vol < thresholds['medium']:
            base_regime = 'medium'
        elif hist_vol < thresholds['high']:
            base_regime = 'high'
        else:
            base_regime = 'very_high'
        # 基于动量调整
        if vol_momentum > 0.05:
            regime = f'{base_regime}_rising'
        elif vol_momentum < -0.05:
            regime = f'{base_regime}_falling'
        else:
            regime = f'{base_regime}_stable'
        return regime
    
    def _assess_regime_stability(self, data: pd.DataFrame) -> float:
        """评估波动率状态的稳定性"""
        returns = data['close'].pct_change().dropna()
        if len(returns) < 60:
            return 0.5
        # 计算滚动波动率
        rolling_vol = returns.rolling(20).std().dropna()
        if len(rolling_vol) < 20:
            return 0.5
        # 计算波动率的变化率
        vol_changes = rolling_vol.pct_change().abs().dropna()
        # 稳定性得分：变化率越低，稳定性越高
        stability = 1.0 - min(1.0, vol_changes.tail(20).mean() * 10)
        return float(stability)
    
    def _detect_boll_breakout(self, data: pd.DataFrame, bb: pd.DataFrame) -> bool:
        """检测布林带突破"""
        if bb is None or bb.empty or len(data) < 5:
            return False
        # 获取最近的价格和布林带
        recent_close = data['close'].tail(5).values
        recent_upper = bb[f'BBU_{self.config["bb_period"]}_{self.config["bb_std"]}'].tail(5).values
        recent_lower = bb[f'BBL_{self.config["bb_period"]}_{self.config["bb_std"]}'].tail(5).values
        # 检查是否有连续突破
        upper_break = all(recent_close > recent_upper)
        lower_break = all(recent_close < recent_lower)
        return bool(upper_break or lower_break)
    
    def _calculate_session_volatility(self, minute_data: pd.DataFrame, start_time: str, end_time: str) -> float:
        """计算特定交易时段的波动率"""
        # 过滤指定时段的数据
        session_mask = (minute_data.index.time >= pd.to_datetime(start_time).time()) & \
                      (minute_data.index.time <= pd.to_datetime(end_time).time())
        session_data = minute_data[session_mask]
        if len(session_data) < 10:
            return 0
        # 计算收益率波动率
        session_returns = session_data['close'].pct_change().dropna()
        return float(session_returns.std() * np.sqrt(240))  # 年化
    
    def _analyze_intraday_volatility_pattern(self, minute_data: pd.DataFrame) -> str:
        """分析日内波动率模式"""
        if len(minute_data) < 100:
            return 'unknown'
        # 计算每小时波动率
        hourly_vol = []
        for hour in range(9, 16):
            if hour == 12:  # 午休
                continue
            hour_mask = minute_data.index.hour == hour
            hour_data = minute_data[hour_mask]
            if len(hour_data) > 5:
                returns = hour_data['close'].pct_change().dropna()
                hour_vol = returns.std() if len(returns) > 1 else 0
                hourly_vol.append(hour_vol)
            else:
                hourly_vol.append(0)
        # 分析模式
        if len(hourly_vol) >= 5:
            # U型（开盘和收盘波动率高）
            if hourly_vol[0] > np.mean(hourly_vol) * 1.2 and hourly_vol[-1] > np.mean(hourly_vol) * 1.2:
                return 'u_shaped'
            # L型（开盘波动率高，随后下降）
            elif hourly_vol[0] > np.mean(hourly_vol) * 1.5 and hourly_vol[-1] < np.mean(hourly_vol) * 0.8:
                return 'l_shaped'
            # J型（收盘波动率高）
            elif hourly_vol[-1] > np.mean(hourly_vol) * 1.5:
                return 'j_shaped'
            # 平坦型
            elif max(hourly_vol) / min([v for v in hourly_vol if v > 0] or [1]) < 1.3:
                return 'flat'
        return 'mixed'
    
    def _calculate_opening_volatility(self, minute_data: pd.DataFrame) -> float:
        """计算开盘波动率（前30分钟）"""
        opening_mask = (minute_data.index.hour == 9) & (minute_data.index.minute >= 30) & \
                      (minute_data.index.minute <= 59)
        opening_data = minute_data[opening_mask]
        if len(opening_data) < 5:
            return 0
        returns = opening_data['close'].pct_change().dropna()
        return float(returns.std() * np.sqrt(240)) if len(returns) > 1 else 0
    
    def _calculate_closing_volatility(self, minute_data: pd.DataFrame) -> float:
        """计算收盘波动率（最后30分钟）"""
        closing_mask = (minute_data.index.hour == 14) & (minute_data.index.minute >= 30) | \
                      (minute_data.index.hour == 15) & (minute_data.index.minute == 0)
        closing_data = minute_data[closing_mask]
        if len(closing_data) < 5:
            return 0
        returns = closing_data['close'].pct_change().dropna()
        return float(returns.std() * np.sqrt(240)) if len(returns) > 1 else 0
    
    def _classify_intraday_volatility_regime(self, intraday_vol: float) -> str:
        """分类日内波动率状态"""
        if intraday_vol < 0.1:
            return 'very_low'
        elif intraday_vol < 0.2:
            return 'low'
        elif intraday_vol < 0.4:
            return 'medium'
        elif intraday_vol < 0.6:
            return 'high'
        else:
            return 'very_high'
    
    def _analyze_intraday_volatility_structure(self, minute_data: pd.DataFrame) -> Dict:
        """分析日内波动率结构"""
        if len(minute_data) < 200:
            return {}
        # 计算滚动日内波动率
        returns = minute_data['close'].pct_change().dropna()
        rolling_vol = returns.rolling(30).std().dropna() * np.sqrt(240)
        # 分析结构特征
        if len(rolling_vol) > 10:
            # 波动率趋势
            vol_trend = 'rising' if rolling_vol.iloc[-1] > rolling_vol.iloc[-10] else 'falling'
            # 波动率聚类
            vol_clusters = self._detect_volatility_clusters(rolling_vol)
            # 波动率跳跃
            vol_jumps = self._detect_volatility_jumps(rolling_vol)
            return {
                'intraday_vol_trend': vol_trend,
                'volatility_clusters': vol_clusters,
                'volatility_jumps': vol_jumps,
                'intraday_volatility_persistence': self._calculate_volatility_persistence(rolling_vol)
            }
        return {}
    
    def _detect_volatility_clusters(self, rolling_vol: pd.Series) -> List[Dict]:
        """检测波动率聚类"""
        clusters = []
        if len(rolling_vol) < 50:
            return clusters
        # 简单聚类检测：寻找连续的高/低波动率区域
        threshold_high = rolling_vol.mean() + rolling_vol.std()
        threshold_low = rolling_vol.mean() - rolling_vol.std()
        current_cluster = None
        cluster_start = None
        for i, vol in enumerate(rolling_vol.values):
            if vol > threshold_high:
                if current_cluster != 'high':
                    if current_cluster is not None:
                        clusters.append({
                            'type': current_cluster,
                            'start': cluster_start,
                            'end': i-1,
                            'duration': i - cluster_start
                        })
                    current_cluster = 'high'
                    cluster_start = i
            elif vol < threshold_low:
                if current_cluster != 'low':
                    if current_cluster is not None:
                        clusters.append({
                            'type': current_cluster,
                            'start': cluster_start,
                            'end': i-1,
                            'duration': i - cluster_start
                        })
                    current_cluster = 'low'
                    cluster_start = i
            else:
                if current_cluster is not None:
                    clusters.append({
                        'type': current_cluster,
                        'start': cluster_start,
                        'end': i-1,
                        'duration': i - cluster_start
                    })
                    current_cluster = None
                    cluster_start = None
        return clusters[-3:]  # 返回最近3个聚类
    
    def _detect_volatility_jumps(self, rolling_vol: pd.Series) -> List[Dict]:
        """检测波动率跳跃"""
        jumps = []
        if len(rolling_vol) < 10:
            return jumps
        # 检测显著的波动率变化
        vol_changes = rolling_vol.pct_change().abs().dropna()
        for i in range(1, min(10, len(vol_changes))):
            if vol_changes.iloc[-i] > 0.5:  # 50%以上的变化
                jumps.append({
                    'index': len(vol_changes) - i,
                    'magnitude': float(vol_changes.iloc[-i]),
                    'direction': 'up' if rolling_vol.iloc[-i] > rolling_vol.iloc[-i-1] else 'down'
                })
        return jumps[-3:]  # 返回最近3个跳跃
    
    def _calculate_volatility_persistence(self, rolling_vol: pd.Series) -> float:
        """计算波动率持续性（Hurst指数简化版）"""
        if len(rolling_vol) < 30:
            return 0.5
        # 简化版的持续性估计
        diff = rolling_vol.diff().dropna()
        if len(diff) < 10:
            return 0.5
        # 计算自相关性
        autocorr = diff.autocorr(lag=1)
        if pd.isna(autocorr):
            return 0.5
        # 转换为持续性得分
        persistence = 0.5 + autocorr * 0.5
        return float(max(0, min(1, persistence)))
    
    def _combine_intraday_daily_regime(self, daily_regime: str, intraday_regime: str) -> str:
        """结合日线和日内波动率状态"""
        if daily_regime is None or intraday_regime is None:
            return daily_regime or intraday_regime or 'unknown'
        # 映射到数值
        regime_values = {
            'very_low': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'very_high': 4
        }
        # 提取基础状态（去掉趋势后缀）
        daily_base = daily_regime.split('_')[0]
        intraday_base = intraday_regime.split('_')[0]
        # 加权平均
        daily_val = regime_values.get(daily_base, 2)
        intraday_val = regime_values.get(intraday_base, 2)
        combined_val = (daily_val * 0.6 + intraday_val * 0.4)
        # 映射回状态
        if combined_val < 0.5:
            combined_regime = 'very_low'
        elif combined_val < 1.5:
            combined_regime = 'low'
        elif combined_val < 2.5:
            combined_regime = 'medium'
        elif combined_val < 3.5:
            combined_regime = 'high'
        else:
            combined_regime = 'very_high'
        # 添加趋势信息
        if '_rising' in daily_regime:
            combined_regime += '_rising'
        elif '_falling' in daily_regime:
            combined_regime += '_falling'
        else:
            combined_regime += '_stable'
        return combined_regime
    
    def _get_empty_volatility_factors(self) -> Dict:
        """返回空的波动率因子字典"""
        return {
            'atr': 0,
            'atr_position': 0.5,
            'atr_momentum': 0,
            'atr_ratio': 0,
            'atr_trend': 'stable',
            'historical_volatility_20d': 0,
            'historical_volatility_60d': 0,
            'volatility_ratio_short_long': 1.0,
            'volatility_clustering': False,
            'volatility_smile_strength': 0.5,
            'volatility_regime': 'unknown',
            'regime_score': 0.5,
            'regime_stability': 0.5,
            'volatility_compression': False,
            'volatility_expansion': False,
            'boll_squeeze': False,
            'boll_breakout': False,
            'boll_bandwidth': 0,
            'compression_degree': 0,
            'volatility_structure_strength': 0,
            'volatility_signal_confidence': 0
        }











