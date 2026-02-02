# services\structural_calculator\volume_structure_calculator.py
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

class VolumeStructureCalculator:
    """成交量结构因子计算器 - 基于幻方量化的A股经验"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化计算器
        Args:
            config: 配置参数，可包含：
                - volume_cluster_threshold: 成交量密集区阈值（默认0.8）
                - spike_multiplier: 成交量峰值倍数（默认2.5）
                - dry_up_threshold: 成交量枯竭阈值（默认0.3）
                - lookback_window: 回看窗口（默认20）
                - divergence_window: 背离检测窗口（默认14）
        """
        self.config = config or {}
        self.volume_cluster_threshold = self.config.get('volume_cluster_threshold', 0.8)
        self.spike_multiplier = self.config.get('spike_multiplier', 2.5)
        self.dry_up_threshold = self.config.get('dry_up_threshold', 0.3)
        self.lookback_window = self.config.get('lookback_window', 20)
        self.divergence_window = self.config.get('divergence_window', 14)
        
    def calculate_volume_structure(self,daily_data: pd.DataFrame,minute_data: Optional[pd.DataFrame] = None,stock_code: str = None) -> Dict:
        """
        计算完整的成交量结构因子
        Args:
            daily_data: 日线数据，包含['trade_time', 'vol', 'amount', 'close', 'high', 'low']
            minute_data: 分钟线数据，可选
            stock_code: 股票代码，用于标识
        Returns:
            包含所有成交量结构因子的字典
        """
        if daily_data.empty:
            return self._get_empty_structure()
        # 预处理数据
        df = daily_data.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.sort_values('trade_time').reset_index(drop=True)
        # 计算基础成交量指标
        df = self._calculate_basic_volume_metrics(df)
        # 计算成交量结构类型
        volume_structure_type = self._determine_volume_structure_type(df)
        # 计算成交量密集区
        volume_cluster = self._detect_volume_clusters(df)
        # 检测成交量峰值
        volume_spike = self._detect_volume_spikes(df)
        # 检测成交量枯竭
        volume_dry_up = self._detect_volume_dry_up(df)
        # 检测量价背离
        price_volume_divergence = self._detect_price_volume_divergence(df)
        # 计算分钟线相关因子（如果提供分钟数据）
        intraday_volume_features = {}
        if minute_data is not None and not minute_data.empty:
            intraday_volume_features = self._calculate_intraday_volume_features(minute_data)
        # 整合所有因子
        latest_data = df.iloc[-1] if len(df) > 0 else None
        result = {
            'volume_structure_type': volume_structure_type,
            'volume_cluster': volume_cluster,
            'volume_spike': volume_spike,
            'volume_dry_up': volume_dry_up,
            'price_volume_divergence': price_volume_divergence,
            'volume_trend_strength': float(latest_data['volume_trend_strength']) if latest_data is not None else None,
            'volume_momentum': float(latest_data['volume_momentum']) if latest_data is not None else None,
            'volume_vwap_deviation': float(latest_data['volume_vwap_deviation']) if latest_data is not None else None,
            **intraday_volume_features,
            'volume_quality_score': self._calculate_volume_quality_score(df),
            'calculated_at': datetime.now(),
            'stock_code': stock_code
        }
        return result
    
    def _calculate_basic_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础成交量指标"""
        # 成交量移动平均
        df['volume_ma5'] = df['vol'].rolling(window=5).mean()
        df['volume_ma10'] = df['vol'].rolling(window=10).mean()
        df['volume_ma20'] = df['vol'].rolling(window=20).mean()
        df['volume_ma60'] = df['vol'].rolling(window=60).mean()
        # 成交量标准差
        df['volume_std20'] = df['vol'].rolling(window=20).std()
        # 成交量相对位置
        df['volume_position'] = (df['vol'] - df['volume_ma20']) / df['volume_ma20']
        # 成交量变异系数（离散程度）
        df['volume_cv20'] = df['volume_std20'] / df['volume_ma20']
        # 成交量趋势强度
        df['volume_trend_strength'] = self._calculate_volume_trend_strength(df)
        # 成交量动量
        df['volume_momentum'] = self._calculate_volume_momentum(df)
        # 成交量与VWAP的偏差
        if 'amount' in df.columns and 'vol' in df.columns:
            df['vwap'] = df['amount'] / (df['vol'] * 100)  # 假设vol是手数，amount是千元
            df['volume_vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        # 成交量比率指标
        df['volume_ratio'] = df['vol'] / df['volume_ma20']
        df['volume_oscillator'] = (df['volume_ma5'] - df['volume_ma20']) / df['volume_ma20']
        return df
    
    def _calculate_volume_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量趋势强度"""
        if len(df) < 20:
            return pd.Series([0] * len(df))
        # 使用线性回归计算成交量趋势
        from scipy import stats
        strengths = []
        for i in range(len(df)):
            if i < 20:
                strengths.append(0)
                continue
                
            window_data = df.iloc[i-19:i+1]
            x = np.arange(20)
            y = window_data['vol'].values
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            # 趋势强度 = 斜率 * R平方（考虑方向和拟合度）
            strength = slope * (r_value ** 2) * 100
            strengths.append(strength)
        return pd.Series(strengths, index=df.index)
    
    def _calculate_volume_momentum(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量动量"""
        if len(df) < 10:
            return pd.Series([0] * len(df))
        # 短期成交量与长期成交量的比率变化
        momentum = []
        for i in range(len(df)):
            if i < 10:
                momentum.append(0)
                continue
                
            # 计算短期（5日）和中期（10日）成交量变化率
            short_term = df.iloc[i-4:i+1]['vol'].mean() if i >= 4 else df.iloc[:i+1]['vol'].mean()
            medium_term = df.iloc[i-9:i+1]['vol'].mean() if i >= 9 else df.iloc[:i+1]['vol'].mean()
            # 动量 = 短期/中期的变化率
            mom = (short_term - medium_term) / medium_term if medium_term > 0 else 0
            momentum.append(mom)
        return pd.Series(momentum, index=df.index)
    
    def _determine_volume_structure_type(self, df: pd.DataFrame) -> str:
        """判断成交量结构类型"""
        if len(df) < 5:
            return "数据不足"
        latest = df.iloc[-1]
        # 判断基础类型
        volume_ratio = latest.get('volume_ratio', 1)
        price_change = latest.get('pct_change', 0) if 'pct_change' in latest else 0
        volume_trend = latest.get('volume_trend_strength', 0)
        types = []
        # 成交量水平判断
        if volume_ratio > 2.0:
            types.append("巨量")
        elif volume_ratio > 1.5:
            types.append("放量")
        elif volume_ratio > 0.8:
            types.append("平量")
        elif volume_ratio > 0.3:
            types.append("缩量")
        else:
            types.append("地量")
        # 量价关系判断
        if abs(price_change) > 0.03:  # 价格波动大于3%
            if price_change > 0 and volume_ratio > 1.2:
                types.append("价升量增")
            elif price_change > 0 and volume_ratio < 0.8:
                types.append("价升量缩")
            elif price_change < 0 and volume_ratio > 1.2:
                types.append("价跌量增")
            elif price_change < 0 and volume_ratio < 0.8:
                types.append("价跌量缩")
        # 成交量趋势判断
        if volume_trend > 0.1:
            types.append("量能增强")
        elif volume_trend < -0.1:
            types.append("量能减弱")
        else:
            types.append("量能平稳")
        # 成交量分布判断
        volume_cv = latest.get('volume_cv20', 0)
        if volume_cv > 0.5:
            types.append("量能波动大")
        elif volume_cv < 0.2:
            types.append("量能稳定")
        # 成交量形态判断（近期模式）
        if len(df) >= 10:
            recent_volumes = df.iloc[-10:]['vol'].values
            if self._is_volume_increasing(recent_volumes):
                types.append("量能递增")
            elif self._is_volume_decreasing(recent_volumes):
                types.append("量能递减")
            elif self._is_volume_accumulating(recent_volumes):
                types.append("蓄量整理")
        return "|".join(types)
    
    def _detect_volume_clusters(self, df: pd.DataFrame) -> bool:
        """检测成交量密集区"""
        if len(df) < 20:
            return False
        # 计算成交量在特定区间内的集中程度
        recent_volumes = df.iloc[-10:]['vol'].values
        volume_ma20 = df.iloc[-1]['volume_ma20']
        # 判断成交量是否在均线附近密集
        within_band = sum(1 for v in recent_volumes 
                         if abs(v - volume_ma20) / volume_ma20 < 0.2)
        # 密集区定义为近期70%的成交量在均线±20%范围内
        return within_band / len(recent_volumes) >= 0.7
    
    def _detect_volume_spikes(self, df: pd.DataFrame) -> bool:
        """检测成交量峰值"""
        if len(df) < 20:
            return False
        latest_volume = df.iloc[-1]['vol']
        volume_ma20 = df.iloc[-1]['volume_ma20']
        volume_std20 = df.iloc[-1]['volume_std20']
        # 峰值判断：超过均线+2倍标准差
        spike_threshold = volume_ma20 + self.spike_multiplier * volume_std20
        is_spike = latest_volume > spike_threshold
        # 同时检查是否为异常值（超过近期最大值的80%）
        if len(df) >= 10:
            recent_max = df.iloc[-10:-1]['vol'].max()
            is_spike = is_spike or (latest_volume > recent_max * 0.8)
        return bool(is_spike)
    
    def _detect_volume_dry_up(self, df: pd.DataFrame) -> bool:
        """检测成交量枯竭"""
        if len(df) < 20:
            return False
        latest_volume = df.iloc[-1]['vol']
        volume_ma20 = df.iloc[-1]['volume_ma20']
        # 枯竭判断：低于均线的特定比例
        is_dry_up = latest_volume < volume_ma20 * self.dry_up_threshold
        # 同时检查是否连续多日低于均线
        if len(df) >= 5:
            recent_volumes = df.iloc[-5:]['vol'].values
            below_ma_count = sum(1 for v in recent_volumes if v < volume_ma20)
            is_dry_up = is_dry_up and (below_ma_count >= 4)
        return bool(is_dry_up)
    
    def _detect_price_volume_divergence(self, df: pd.DataFrame) -> str:
        """检测量价背离"""
        if len(df) < self.divergence_window * 2:
            return "数据不足"
        # 获取价格和成交量数据
        prices = df['close'].values
        volumes = df['vol'].values
        # 计算价格和成交量的趋势（使用移动平均）
        price_ma = pd.Series(prices).rolling(window=self.divergence_window).mean().values
        volume_ma = pd.Series(volumes).rolling(window=self.divergence_window).mean().values
        # 检测顶背离：价格创新高，成交量未创新高
        latest_price = prices[-1]
        latest_volume = volumes[-1]
        # 获取近期价格和成交量的极值点
        price_extremes = self._find_price_extremes(prices, window=self.divergence_window)
        volume_extremes = self._find_volume_extremes(volumes, window=self.divergence_window)
        divergences = []
        # 顶背离检测
        if (price_extremes[-1] == 'high' and 
            volume_extremes[-1] != 'high' and
            prices[-1] > np.max(prices[-self.divergence_window*2:-self.divergence_window]) and
            volumes[-1] < np.max(volumes[-self.divergence_window*2:-self.divergence_window])):
            divergences.append("顶背离")
        # 底背离检测
        if (price_extremes[-1] == 'low' and 
            volume_extremes[-1] != 'low' and
            prices[-1] < np.min(prices[-self.divergence_window*2:-self.divergence_window]) and
            volumes[-1] > np.min(volumes[-self.divergence_window*2:-self.divergence_window])):
            divergences.append("底背离")
        # 隐藏背离检测（价格未创新高/新低，但成交量有显著变化）
        if len(price_extremes) >= 3 and len(volume_extremes) >= 3:
            if (price_extremes[-1] == price_extremes[-2] and
                volume_extremes[-1] != volume_extremes[-2] and
                abs(volumes[-1] - volumes[-2]) / volumes[-2] > 0.3):
                if price_extremes[-1] == 'high':
                    divergences.append("隐藏顶背离")
                else:
                    divergences.append("隐藏底背离")
        return "|".join(divergences) if divergences else "无量价背离"
    
    def _calculate_intraday_volume_features(self, minute_data: pd.DataFrame) -> Dict:
        """计算日内成交量特征"""
        if minute_data.empty:
            return {}
        df_min = minute_data.copy()
        df_min['trade_time'] = pd.to_datetime(df_min['trade_time'])
        df_min = df_min.sort_values('trade_time')
        # 按时间段划分
        df_min['hour'] = df_min['trade_time'].dt.hour
        df_min['minute'] = df_min['trade_time'].dt.minute
        # 早盘（9:30-11:30）
        morning_mask = ((df_min['hour'] == 9) & (df_min['minute'] >= 30)) | \
                      ((df_min['hour'] == 10)) | \
                      ((df_min['hour'] == 11) & (df_min['minute'] <= 30))
        # 午盘（13:00-14:30）
        afternoon_mask = ((df_min['hour'] == 13)) | \
                        ((df_min['hour'] == 14) & (df_min['minute'] <= 30))
        # 尾盘（14:30-15:00）
        closing_mask = ((df_min['hour'] == 14) & (df_min['minute'] > 30)) | \
                      ((df_min['hour'] == 15) & (df_min['minute'] == 0))
        # 计算各时段成交量
        morning_volume = df_min[morning_mask]['vol'].sum() if morning_mask.any() else 0
        afternoon_volume = df_min[afternoon_mask]['vol'].sum() if afternoon_mask.any() else 0
        closing_volume = df_min[closing_mask]['vol'].sum() if closing_mask.any() else 0
        total_volume = df_min['vol'].sum()
        # 计算比例
        morning_ratio = morning_volume / total_volume if total_volume > 0 else 0
        afternoon_ratio = afternoon_volume / total_volume if total_volume > 0 else 0
        closing_ratio = closing_volume / total_volume if total_volume > 0 else 0
        # 检测日内成交量峰值
        intraday_spikes = self._detect_intraday_volume_spikes(df_min)
        # 计算成交量分布特征
        volume_distribution = self._analyze_volume_distribution(df_min)
        return {
            'morning_volume_ratio': float(morning_ratio),
            'afternoon_volume_ratio': float(afternoon_ratio),
            'closing_volume_ratio': float(closing_ratio),
            'intraday_volume_spikes': intraday_spikes,
            'volume_distribution_skew': volume_distribution.get('skewness', 0),
            'volume_distribution_kurtosis': volume_distribution.get('kurtosis', 0),
            'intraday_volume_volatility': volume_distribution.get('volatility', 0),
        }
    
    def _detect_intraday_volume_spikes(self, df_min: pd.DataFrame) -> List[Dict]:
        """检测日内成交量峰值"""
        spikes = []
        if len(df_min) < 10:
            return spikes
        # 计算分钟成交量移动平均和标准差
        df_min['volume_ma'] = df_min['vol'].rolling(window=10, min_periods=3).mean()
        df_min['volume_std'] = df_min['vol'].rolling(window=10, min_periods=3).std()
        # 检测峰值（超过均值+2倍标准差）
        spike_mask = (df_min['vol'] > df_min['volume_ma'] + 2 * df_min['volume_std'])
        spike_times = df_min[spike_mask]['trade_time']
        for time in spike_times:
            spikes.append({
                'time': time.strftime('%H:%M'),
                'volume': float(df_min.loc[df_min['trade_time'] == time, 'vol'].iloc[0]),
                'intensity': float((df_min.loc[df_min['trade_time'] == time, 'vol'].iloc[0] - 
                                   df_min.loc[df_min['trade_time'] == time, 'volume_ma'].iloc[0]) / 
                                   df_min.loc[df_min['trade_time'] == time, 'volume_ma'].iloc[0])
            })
        return spikes
    
    def _analyze_volume_distribution(self, df_min: pd.DataFrame) -> Dict:
        """分析成交量分布特征"""
        if df_min.empty:
            return {}
        volumes = df_min['vol'].values
        from scipy import stats
        return {
            'skewness': float(stats.skew(volumes)) if len(volumes) > 0 else 0,
            'kurtosis': float(stats.kurtosis(volumes)) if len(volumes) > 0 else 0,
            'volatility': float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0,
            'median_ratio': float(np.median(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0,
        }
    
    def _calculate_volume_quality_score(self, df: pd.DataFrame) -> float:
        """计算成交量质量得分（0-1）"""
        if len(df) < 20:
            return 0.5
        scores = []
        # 1. 成交量稳定性得分
        volume_cv = df.iloc[-1].get('volume_cv20', 0)
        stability_score = max(0, 1 - volume_cv)
        scores.append(stability_score * 0.3)
        # 2. 成交量趋势明确性得分
        trend_strength = abs(df.iloc[-1].get('volume_trend_strength', 0))
        trend_score = min(1, trend_strength * 10)  # 归一化
        scores.append(trend_score * 0.3)
        # 3. 量价配合得分
        volume_ratio = df.iloc[-1].get('volume_ratio', 1)
        price_change = df.iloc[-1].get('pct_change', 0) if 'pct_change' in df.columns else 0
        if abs(price_change) > 0.01:  # 价格有变化
            # 量价同向为正，反向为负
            if (price_change > 0 and volume_ratio > 1) or (price_change < 0 and volume_ratio < 1):
                price_volume_score = 0.8
            else:
                price_volume_score = 0.3
        else:
            price_volume_score = 0.5  # 价格无变化，中性
        scores.append(price_volume_score * 0.4)
        return float(np.mean(scores))
    
    def _find_price_extremes(self, prices: np.ndarray, window: int = 14) -> List[str]:
        """寻找价格极值点"""
        extremes = []
        for i in range(len(prices)):
            if i < window or i >= len(prices) - window:
                extremes.append('neutral')
                continue
                
            window_prices = prices[i-window:i+window+1]
            if prices[i] == np.max(window_prices):
                extremes.append('high')
            elif prices[i] == np.min(window_prices):
                extremes.append('low')
            else:
                extremes.append('neutral')
        return extremes
    
    def _find_volume_extremes(self, volumes: np.ndarray, window: int = 14) -> List[str]:
        """寻找成交量极值点"""
        extremes = []
        for i in range(len(volumes)):
            if i < window or i >= len(volumes) - window:
                extremes.append('neutral')
                continue
                
            window_volumes = volumes[i-window:i+window+1]
            if volumes[i] == np.max(window_volumes):
                extremes.append('high')
            elif volumes[i] == np.min(window_volumes):
                extremes.append('low')
            else:
                extremes.append('neutral')
        return extremes
    
    def _is_volume_increasing(self, volumes: np.ndarray) -> bool:
        """判断成交量是否递增"""
        if len(volumes) < 3:
            return False
        # 使用线性回归判断趋势
        from scipy import stats
        x = np.arange(len(volumes))
        slope, _, _, _, _ = stats.linregress(x, volumes)
        return slope > 0 and (volumes[-1] > volumes[0] * 1.1)
    
    def _is_volume_decreasing(self, volumes: np.ndarray) -> bool:
        """判断成交量是否递减"""
        if len(volumes) < 3:
            return False
        # 使用线性回归判断趋势
        from scipy import stats
        x = np.arange(len(volumes))
        slope, _, _, _, _ = stats.linregress(x, volumes)
        return slope < 0 and (volumes[-1] < volumes[0] * 0.9)
    
    def _is_volume_accumulating(self, volumes: np.ndarray) -> bool:
        """判断成交量是否蓄量整理"""
        if len(volumes) < 5:
            return False
        # 计算变异系数
        cv = np.std(volumes) / np.mean(volumes)
        # 蓄量特征：低波动，但整体水平较高
        return cv < 0.3 and np.mean(volumes) > np.median(volumes) * 0.8
    
    def _get_empty_structure(self) -> Dict:
        """获取空的结构因子字典"""
        return {
            'volume_structure_type': "数据不足",
            'volume_cluster': False,
            'volume_spike': False,
            'volume_dry_up': False,
            'price_volume_divergence': "数据不足",
            'volume_trend_strength': None,
            'volume_momentum': None,
            'volume_vwap_deviation': None,
            'morning_volume_ratio': None,
            'afternoon_volume_ratio': None,
            'closing_volume_ratio': None,
            'intraday_volume_spikes': [],
            'volume_distribution_skew': None,
            'volume_distribution_kurtosis': None,
            'intraday_volume_volatility': None,
            'volume_quality_score': 0.0,
            'calculated_at': datetime.now(),
            'stock_code': None
        }
