# services\structural_calculator\structural_calculator.py

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

class StructuralCalculator:
    """结构因子计算器 - 幻方量化深度通道结构分析模型"""
    
    def __init__(self):
        # 自适应参数系统
        self.bb_period = 20  # 布林带基础周期
        self.bb_std = 2.0    # 布林带基础标准差倍数
        self.donchian_period = 20  # 唐奇安通道周期
        self.volatility_lookback = 10
        # 幻方量化特色参数
        self.fractal_dimension_window = 10  # 分形维度计算窗口
        self.hurst_exponent_window = 30    # 赫斯特指数计算窗口
        self.adaptive_lookback = 60        # 自适应参数回看期
        self.resonance_threshold = 0.618   # 黄金分割共振阈值
        
    def calculate_price_channel_factors(self, daily_data: pd.DataFrame,minute_data: Optional[pd.DataFrame] = None,lookback_period: int = 60) -> Dict:
        """
        幻方量化深度通道结构因子计算
        融合分形理论、自适应参数、共振原理等数学模型
        """
        factors = {}
        # === 1. 自适应参数计算 ===
        adaptive_params = self._calculate_adaptive_parameters(daily_data)
        # === 2. 布林带结构深度分析（加入分形特征） ===
        bb_factors = self._calculate_bollinger_bands_structure(daily_data)
        # 加入分形特征
        fractal_features = self._calculate_fractal_features(daily_data['close'])
        bb_factors.update(fractal_features)
        # 计算赫斯特指数判断市场记忆性
        hurst_exp = self._calculate_hurst_exponent(daily_data['close'])
        bb_factors['hurst_exponent'] = hurst_exp
        bb_factors['market_memory'] = '强记忆' if hurst_exp > 0.7 else '弱记忆' if hurst_exp < 0.3 else '中性'
        factors.update(bb_factors)
        # === 3. 量子唐奇安通道 ===
        donchian_factors = self._calculate_donchian_channels(daily_data)
        # 加入量子隧穿效应检测
        quantum_tunnel = self._detect_quantum_tunneling(daily_data, donchian_factors)
        donchian_factors.update(quantum_tunnel)
        factors.update(donchian_factors)
        # === 4. 多尺度共振突破信号 ===
        breakout_factors = self._calculate_channel_breakouts(daily_data, bb_factors, donchian_factors)
        # 加入混沌理论突破确认
        chaos_confirmation = self._chaos_breakout_confirmation(daily_data, breakout_factors)
        breakout_factors.update(chaos_confirmation)
        factors.update(breakout_factors)
        # === 5. 日内分形通道结构 ===
        if minute_data is not None and not minute_data.empty:
            intraday_factors = self._calculate_intraday_channel_factors(minute_data, daily_data)
            # 计算日内分形维度
            intraday_fractal = self._calculate_intraday_fractal_dimension(minute_data)
            intraday_factors.update(intraday_fractal)
            factors.update(intraday_factors)
        # === 6. 多维通道收敛分析 ===
        convergence_factors = self._calculate_channel_convergence(daily_data, bb_factors, donchian_factors)
        # 加入相空间重构分析
        phase_space = self._analyze_phase_space_convergence(daily_data)
        convergence_factors.update(phase_space)
        factors.update(convergence_factors)
        # === 7. 非线性通道位置计算 ===
        position_factors = self._calculate_channel_position(daily_data, bb_factors, donchian_factors)
        # 加入拓扑结构分析
        topology_analysis = self._analyze_price_topology(daily_data)
        position_factors.update(topology_analysis)
        factors.update(position_factors)
        # === 8. 波动率分形分析 ===
        atr_factors = self._calculate_atr_position(daily_data)
        # 加入多重分形谱分析
        multifractal = self._calculate_multifractal_spectrum(daily_data)
        atr_factors.update(multifractal)
        factors.update(atr_factors)
        # === 9. 混沌边缘检测 ===
        chaos_edge = self._detect_chaos_edge(daily_data)
        factors.update(chaos_edge)
        # === 10. 自适应共振网络评分 ===
        resonance_score = self._calculate_adaptive_resonance_score(factors)
        factors['adaptive_resonance_score'] = resonance_score
        return factors
    
    def _calculate_bollinger_bands_structure(self, data: pd.DataFrame) -> Dict:
        """重构：自适应布林带 + 分形维度 + 量子共振"""
        factors = {}
        if len(data) < max(self.bb_period, self.fractal_dimension_window):
            return factors
        close_prices = data['close']
        # === 1. 自适应布林带参数 ===
        volatility = ta.atr(data['high'], data['low'], close_prices, length=14)
        recent_volatility = volatility.iloc[-self.adaptive_lookback:].mean() if len(volatility) >= self.adaptive_lookback else volatility.mean()
        # 动态调整布林带参数
        adaptive_std = self.bb_std * (1 + np.tanh(recent_volatility / close_prices.mean() * 10))
        adaptive_period = max(10, min(30, int(self.bb_period * (1 - recent_volatility / close_prices.mean() * 5))))
        # === 2. 计算多重布林带 ===
        bb_multiple = {}
        # 基础布林带
        bb_base = ta.bbands(close_prices, length=adaptive_period, std=adaptive_std)
        # 短期布林带（捕捉细节）
        bb_short = ta.bbands(close_prices, length=int(adaptive_period/2), std=adaptive_std*0.8)
        # 长期布林带（趋势过滤）
        bb_long = ta.bbands(close_prices, length=adaptive_period*2, std=adaptive_std*1.2)
        # === 3. 布林带拓扑结构分析 ===
        if not bb_base.empty:
            bb_upper = bb_base[f'BBU_{adaptive_period}_{adaptive_std:.1f}']
            bb_middle = bb_base[f'BBM_{adaptive_period}_{adaptive_std:.1f}']
            bb_lower = bb_base[f'BBL_{adaptive_period}_{adaptive_std:.1f}']
            # 计算布林带曲率（二阶导数近似）
            bb_width = bb_upper - bb_lower
            bb_curvature = bb_width.diff().diff()
            # 布林带分形维度
            fractal_dim = self._calculate_series_fractal_dimension(bb_width)
            # 自适应位置计算（非线性映射）
            current_close = close_prices.iloc[-1]
            bb_position_raw = (current_close - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            # 使用sigmoid函数进行非线性转换
            bb_position_nonlinear = 1 / (1 + np.exp(-10 * (bb_position_raw - 0.5)))
            bb_position = self._get_bollinger_position(bb_position_nonlinear * 100)
            # 量子隧穿概率计算
            distance_to_upper = bb_upper.iloc[-1] - current_close
            distance_to_lower = current_close - bb_lower.iloc[-1]
            quantum_tunnel_prob = np.exp(-min(distance_to_upper, distance_to_lower) / (bb_width.iloc[-1] * 0.1))
            # 布林带谐波共振检测
            harmonic_resonance = self._detect_harmonic_resonance(bb_upper, bb_middle, bb_lower, close_prices)
            factors.update({
                'boll_position': bb_position,
                'boll_bandwidth': float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else None,
                'boll_curvature': float(bb_curvature.iloc[-1]) if pd.notna(bb_curvature.iloc[-1]) else None,
                'boll_fractal_dim': float(fractal_dim) if not np.isnan(fractal_dim) else None,
                'quantum_tunnel_probability': float(quantum_tunnel_prob),
                'boll_harmonic_resonance': harmonic_resonance,
                'adaptive_std': float(adaptive_std),
                'adaptive_period': adaptive_period
            })
            # 挤压检测（使用多重时间尺度）
            squeeze_multiscale = self._multiscale_squeeze_detection(bb_base, bb_short, bb_long)
            factors['boll_squeeze'] = bool(squeeze_multiscale['squeeze'])
            factors['squeeze_strength'] = squeeze_multiscale['strength']
            # 突破检测（量子跃迁模型）
            quantum_breakout = self._quantum_breakout_detection(close_prices, bb_upper, bb_lower, bb_middle)
            factors['boll_breakout'] = bool(quantum_breakout['breakout'])
            factors['breakout_energy'] = quantum_breakout['energy']
            factors['_bollinger_data'] = {
                'upper': bb_upper.iloc[-20:].tolist(),
                'middle': bb_middle.iloc[-20:].tolist(),
                'lower': bb_lower.iloc[-20:].tolist(),
                'width': bb_width.iloc[-20:].tolist(),
                'curvature': bb_curvature.iloc[-20:].tolist(),
                'position_nonlinear': float(bb_position_nonlinear * 100),
                'position_raw': float(bb_position_raw * 100)
            }
        return factors
    
    def _calculate_donchian_channels(self, data: pd.DataFrame) -> Dict:
        """重构：自适应唐奇安通道 + 拓扑不变性分析"""
        factors = {}
        if len(data) < max(self.donchian_period, self.hurst_exponent_window):
            return factors
        high = data['high']
        low = data['low']
        close = data['close']
        # === 1. 自适应唐奇安通道 ===
        # 基于赫斯特指数调整周期
        hurst = self._calculate_hurst_exponent(close)
        adaptive_donchian_period = max(
            10, 
            min(50, int(self.donchian_period * (1 + 0.5 * (hurst - 0.5))))
        )
        # === 2. 多重时间尺度唐奇安通道 ===
        donchian_multiscale = {}
        # 基础通道
        donchian_base = ta.donchian(high, low, length=adaptive_donchian_period)
        # 快速通道（捕捉短期波动）
        donchian_fast = ta.donchian(high, low, length=max(5, adaptive_donchian_period // 2))
        # 慢速通道（识别主要趋势）
        donchian_slow = ta.donchian(high, low, length=min(100, adaptive_donchian_period * 2))
        # === 3. 唐奇安通道拓扑分析 ===
        if donchian_base is not None and not donchian_base.empty:
            # 基础通道数据
            donchian_high = donchian_base[f'DCU_{adaptive_donchian_period}_{adaptive_donchian_period}']
            donchian_low = donchian_base[f'DCL_{adaptive_donchian_period}_{adaptive_donchian_period}']
            donchian_width = donchian_high - donchian_low
            # 通道宽度分形分析
            width_fractal = self._calculate_series_fractal_dimension(donchian_width)
            # 通道持久性分析（基于自相关函数）
            autocorr = donchian_width.autocorr(lag=1)
            # 计算通道曲率（趋势加速度）
            width_derivative = donchian_width.diff()
            width_curvature = width_derivative.diff()
            # 通道共振强度计算
            current_price = close.iloc[-1]
            high_distance = donchian_high.iloc[-1] - current_price
            low_distance = current_price - donchian_low.iloc[-1]
            # 使用双曲正切函数计算相对距离
            resonance_to_high = np.tanh(high_distance / donchian_width.iloc[-1] * 5) if donchian_width.iloc[-1] > 0 else 0
            resonance_to_low = np.tanh(low_distance / donchian_width.iloc[-1] * 5) if donchian_width.iloc[-1] > 0 else 0
            # 拓扑不变性检测（通道形态保持性）
            topology_invariance = self._calculate_topology_invariance(donchian_high, donchian_low)
            factors.update({
                'donchian_high': float(donchian_high.iloc[-1]) if pd.notna(donchian_high.iloc[-1]) else None,
                'donchian_low': float(donchian_low.iloc[-1]) if pd.notna(donchian_low.iloc[-1]) else None,
                'donchian_width': float(donchian_width.iloc[-1]) if pd.notna(donchian_width.iloc[-1]) else None,
                'donchian_autocorr': float(autocorr) if not np.isnan(autocorr) else None,
                'donchian_fractal_dim': float(width_fractal) if not np.isnan(width_fractal) else None,
                'donchian_curvature': float(width_curvature.iloc[-1]) if pd.notna(width_curvature.iloc[-1]) else None,
                'resonance_to_high': float(resonance_to_high),
                'resonance_to_low': float(resonance_to_low),
                'topology_invariance': float(topology_invariance),
                'adaptive_donchian_period': adaptive_donchian_period
            })
            factors['_donchian_data'] = {
                'high': donchian_high.iloc[-20:].tolist(),
                'low': donchian_low.iloc[-20:].tolist(),
                'width': donchian_width.iloc[-20:].tolist(),
                'width_derivative': width_derivative.iloc[-20:].tolist(),
                'topology_score': [float(topology_invariance)] * 20
            }
        return factors
    
    def _calculate_channel_breakouts(self, data: pd.DataFrame,bb_factors: Dict,donchian_factors: Dict) -> Dict:
        """重构：多尺度共振突破 + 量子跃迁模型"""
        factors = {}
        if len(data) < 5:
            factors['channel_breakout'] = None
            return factors
        close = data['close']
        high = data['high']
        low = data['low']
        # 获取布林带和唐奇安数据
        bb_data = bb_factors.get('_bollinger_data', {})
        donchian_data = donchian_factors.get('_donchian_data', {})
        # === 1. 多尺度突破信号生成 ===
        breakout_signals = []
        signal_energies = []
        # 布林带突破检测
        if bb_data:
            bb_upper = bb_data.get('upper', [])
            bb_lower = bb_data.get('lower', [])
            if len(bb_upper) >= 3 and len(bb_lower) >= 3:
                current_close = close.iloc[-1]
                prev_close = close.iloc[-2] if len(close) >= 2 else current_close
                # 计算突破能量（基于相对距离和速度）
                upper_break_energy = 0
                lower_break_energy = 0
                bb_upper_last = bb_upper[-1] if bb_upper else np.nan
                bb_lower_last = bb_lower[-1] if bb_lower else np.nan
                if not np.isnan(bb_upper_last):
                    distance_to_upper = current_close - bb_upper_last
                    velocity_to_upper = distance_to_upper / bb_upper_last * 100
                    upper_break_energy = np.exp(velocity_to_upper * 2)
                    if current_close > bb_upper_last and prev_close <= bb_upper_last:
                        breakout_signals.append('布林带上突破')
                        signal_energies.append(upper_break_energy)
                if not np.isnan(bb_lower_last):
                    distance_to_lower = bb_lower_last - current_close
                    velocity_to_lower = distance_to_lower / bb_lower_last * 100
                    lower_break_energy = np.exp(velocity_to_lower * 2)
                    if current_close < bb_lower_last and prev_close >= bb_lower_last:
                        breakout_signals.append('布林带下突破')
                        signal_energies.append(lower_break_energy)
        # 唐奇安通道突破检测
        if donchian_data:
            donchian_high = donchian_data.get('high', [])
            donchian_low = donchian_data.get('low', [])
            if len(donchian_high) >= 1 and len(donchian_low) >= 1:
                current_high = high.iloc[-1]
                current_low = low.iloc[-1]
                donchian_high_last = donchian_high[-1] if donchian_high else np.nan
                donchian_low_last = donchian_low[-1] if donchian_low else np.nan
                if not np.isnan(donchian_high_last) and current_high >= donchian_high_last:
                    breakout_signals.append('唐奇安上突破')
                    signal_energies.append(1.5)  # 唐奇安突破权重更高
                if not np.isnan(donchian_low_last) and current_low <= donchian_low_last:
                    breakout_signals.append('唐奇安下突破')
                    signal_energies.append(1.5)
        # === 2. 突破信号融合与能量计算 ===
        if breakout_signals:
            # 计算总突破能量
            total_energy = sum(signal_energies) if signal_energies else 0
            # 多重确认（成交量+波动率+动量）
            volume_confirmation = self._check_breakout_volume_confirmation(data)
            volatility_confirmation = self._check_breakout_volatility_confirmation(data)
            momentum_confirmation = self._check_breakout_momentum_confirmation(data)
            confirmation_score = sum([volume_confirmation, volatility_confirmation, momentum_confirmation])
            # 量子跃迁概率计算
            quantum_transition_prob = min(1.0, total_energy / 10) * (confirmation_score / 3)
            if confirmation_score >= 2 and quantum_transition_prob > 0.3:
                # 判断突破方向
                if any('上突破' in signal for signal in breakout_signals):
                    factors['channel_breakout'] = '量子向上跃迁'
                    factors['breakout_type'] = 'quantum_up'
                elif any('下突破' in signal for signal in breakout_signals):
                    factors['channel_breakout'] = '量子向下跃迁'
                    factors['breakout_type'] = 'quantum_down'
                else:
                    factors['channel_breakout'] = '共振突破'
                    factors['breakout_type'] = 'resonance'
                factors['breakout_energy'] = float(total_energy)
                factors['quantum_probability'] = float(quantum_transition_prob)
                factors['confirmation_score'] = confirmation_score
            else:
                factors['channel_breakout'] = '假突破风险'
                factors['breakout_type'] = 'false_breakout'
                factors['false_breakout_prob'] = float(1 - quantum_transition_prob)
        else:
            factors['channel_breakout'] = '无突破'
            factors['breakout_type'] = 'no_breakout'
            factors['breakout_energy'] = 0.0
        # === 3. 突破后行为预测 ===
        if factors.get('breakout_type') in ['quantum_up', 'quantum_down']:
            post_breakout_analysis = self._analyze_post_breakout_behavior(data, factors['breakout_type'])
            factors.update(post_breakout_analysis)
        # 存储详细突破信息
        factors['_breakout_details'] = {
            'signals': breakout_signals,
            'energies': signal_energies,
            'volume_confirmation': volume_confirmation,
            'volatility_confirmation': volatility_confirmation,
            'momentum_confirmation': momentum_confirmation,
            'quantum_transition_prob': factors.get('quantum_probability', 0)
        }
        return factors
    
    def _calculate_intraday_channel_factors(self, minute_data: pd.DataFrame, daily_data: pd.DataFrame) -> Dict:
        """重构：分形日内通道 + 量子隧穿效应"""
        factors = {}
        if minute_data.empty:
            return factors
        # 多重时间尺度分析
        intraday_scales = ['5min', '15min', '30min', '60min']
        scale_factors = {}
        for scale in intraday_scales:
            # 重采样到不同时间尺度
            resampled = minute_data.resample(scale).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            if len(resampled) >= 20:
                # 计算该尺度的布林带
                bb_result = ta.bbands(resampled['close'], length=20, std=2.0)
                if bb_result is not None and not bb_result.empty:
                    # 提取布林带数据
                    bb_upper = bb_result['BBU_20_2.0']
                    bb_lower = bb_result['BBL_20_2.0']
                    bb_width = bb_upper - bb_lower
                    # 计算分形维度
                    fractal_dim = self._calculate_series_fractal_dimension(resampled['close'])
                    # 计算通道位置
                    current_price = resampled['close'].iloc[-1]
                    position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                    scale_factors[scale] = {
                        'position': float(position),
                        'width': float(bb_width.iloc[-1]),
                        'fractal_dim': float(fractal_dim),
                        'volume_ratio': resampled['volume'].iloc[-1] / resampled['volume'].mean()
                    }
        # 多尺度共振分析
        if scale_factors:
            # 计算多尺度位置一致性
            positions = [f['position'] for f in scale_factors.values()]
            position_std = np.std(positions)
            # 计算分形维度谱
            fractal_dims = [f['fractal_dim'] for f in scale_factors.values()]
            fractal_spectrum_std = np.std(fractal_dims)
            # 量子隧穿效应检测
            quantum_tunneling = self._detect_intraday_quantum_tunneling(minute_data, scale_factors)
            factors.update({
                '_intraday_multiscale': scale_factors,
                'intraday_position_std': float(position_std),
                'intraday_fractal_spectrum_std': float(fractal_spectrum_std),
                'intraday_quantum_tunneling': quantum_tunneling.get('detected', False),
                'tunneling_intensity': quantum_tunneling.get('intensity', 0)
            })
            # 主要尺度分析（使用15min）
            if '15min' in scale_factors:
                main_scale = scale_factors['15min']
                factors.update({
                    'intraday_channel_position': main_scale['position'],
                    'intraday_channel_width': main_scale['width'],
                    'intraday_fractal_dim': main_scale['fractal_dim']
                })
        return factors
    
    def _calculate_channel_convergence(self, data: pd.DataFrame,bb_factors: Dict,donchian_factors: Dict) -> Dict:
        """重构：多维通道收敛 + 相空间分析"""
        factors = {}
        # 获取通道数据
        bb_width_data = bb_factors.get('_bollinger_data', {}).get('width', [])
        donchian_width_data = donchian_factors.get('_donchian_data', {}).get('width', [])
        if not bb_width_data or not donchian_width_data:
            return factors
        # === 1. 多重收敛指标计算 ===
        bb_width_series = pd.Series(bb_width_data)
        donchian_width_series = pd.Series(donchian_width_data)
        # 分形收敛度（基于Hurst指数）
        bb_hurst = self._calculate_hurst_exponent(bb_width_series)
        donchian_hurst = self._calculate_hurst_exponent(donchian_width_series)
        # 收敛能量计算（负熵）
        bb_entropy = self._calculate_shannon_entropy(bb_width_series)
        donchian_entropy = self._calculate_shannon_entropy(donchian_width_series)
        # 通道宽度相关性
        if len(bb_width_series) == len(donchian_width_series):
            width_correlation = bb_width_series.corr(donchian_width_series)
        else:
            min_len = min(len(bb_width_series), len(donchian_width_series))
            width_correlation = bb_width_series.iloc[-min_len:].corr(donchian_width_series.iloc[-min_len:])
        # === 2. 收敛状态判定 ===
        convergence_states = []
        # 基于赫斯特指数的收敛判断
        if bb_hurst < 0.4 and donchian_hurst < 0.4:
            convergence_states.append('强收敛')
        elif bb_hurst < 0.5 and donchian_hurst < 0.5:
            convergence_states.append('弱收敛')
        # 基于熵的收敛判断
        if bb_entropy < 1.5 and donchian_entropy < 1.5:
            convergence_states.append('低熵收敛')
        # 基于相关性的收敛判断
        if width_correlation > 0.7:
            convergence_states.append('高相关收敛')
        # === 3. 综合收敛评分 ===
        convergence_score = 0
        if convergence_states:
            convergence_score = len(convergence_states) * 25  # 每个状态25分
        # 判断最终收敛状态
        if convergence_score >= 75:
            convergence_state = '强烈收敛'
        elif convergence_score >= 50:
            convergence_state = '中度收敛'
        elif convergence_score >= 25:
            convergence_state = '轻度收敛'
        else:
            convergence_state = '中性'
        factors.update({
            'channel_convergence': convergence_state,
            'convergence_score': convergence_score,
            'bb_hurst_convergence': float(bb_hurst),
            'donchian_hurst_convergence': float(donchian_hurst),
            'bb_entropy': float(bb_entropy),
            'donchian_entropy': float(donchian_entropy),
            'width_correlation': float(width_correlation) if not np.isnan(width_correlation) else 0,
            'convergence_states': convergence_states
        })
        # === 4. 收敛动力学分析 ===
        dynamics = self._analyze_convergence_dynamics(bb_width_series, donchian_width_series)
        factors.update(dynamics)
        return factors
    
    def _calculate_channel_position(self, data: pd.DataFrame,bb_factors: Dict,donchian_factors: Dict) -> Dict:
        """重构：拓扑通道位置 + 量子概率分布"""
        factors = {}
        if len(data) == 0:
            return factors
        current_price = data['close'].iloc[-1]
        # 获取通道边界数据
        bb_data = bb_factors.get('_bollinger_data', {})
        donchian_data = donchian_factors.get('_donchian_data', {})
        if not bb_data or not donchian_data:
            return factors
        # === 1. 多重位置计算 ===
        positions = []
        position_weights = []
        # 布林带位置
        bb_upper = bb_data.get('upper', [])
        bb_lower = bb_data.get('lower', [])
        if bb_upper and bb_lower:
            bb_upper_last = bb_upper[-1]
            bb_lower_last = bb_lower[-1]
            if bb_upper_last != bb_lower_last and not np.isnan(bb_upper_last) and not np.isnan(bb_lower_last):
                bb_position = (current_price - bb_lower_last) / (bb_upper_last - bb_lower_last)
                positions.append(bb_position)
                position_weights.append(0.6)  # 布林带权重60%
        # 唐奇安位置
        donchian_high = donchian_data.get('high', [])
        donchian_low = donchian_data.get('low', [])
        if donchian_high and donchian_low:
            donchian_high_last = donchian_high[-1]
            donchian_low_last = donchian_low[-1]
            if donchian_high_last != donchian_low_last and not np.isnan(donchian_high_last) and not np.isnan(donchian_low_last):
                donchian_position = (current_price - donchian_low_last) / (donchian_high_last - donchian_low_last)
                positions.append(donchian_position)
                position_weights.append(0.4)  # 唐奇安权重40%
        if positions:
            # 加权平均位置
            weighted_position = np.average(positions, weights=position_weights[:len(positions)])
            # === 2. 拓扑位置分类 ===
            position_category = self._get_topological_position(weighted_position, positions)
            # === 3. 量子概率分布 ===
            quantum_distribution = self._calculate_quantum_position_distribution(weighted_position, positions)
            # === 4. 位置稳定性分析 ===
            if len(data) >= 10:
                recent_positions = self._calculate_recent_positions(data, bb_data, donchian_data)
                position_stability = np.std(recent_positions) if recent_positions else 0
            else:
                position_stability = 0
            factors.update({
                'channel_position': position_category,
                'channel_position_score': float(weighted_position * 100),
                'position_stability': float(position_stability * 100),
                'quantum_position_mean': quantum_distribution['mean'],
                'quantum_position_std': quantum_distribution['std'],
                'position_uncertainty': quantum_distribution['uncertainty'],
                'position_weights': position_weights,
                'raw_positions': positions
            })
        return factors
    
    def _calculate_atr_position(self, data: pd.DataFrame) -> Dict:
        """重构：多重分形波动率 + 混沌波动率"""
        factors = {}
        if len(data) < 14:
            return factors
        # === 1. 多重时间尺度ATR ===
        atr_scales = [7, 14, 21, 28]
        atr_values = {}
        for scale in atr_scales:
            atr = ta.atr(data['high'], data['low'], data['close'], length=scale)
            if atr is not None and not atr.empty:
                atr_values[f'atr_{scale}'] = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0
        # === 2. 波动率分形分析 ===
        if atr_values:
            # 波动率比（多重尺度）
            atr_ratio_7_14 = atr_values.get('atr_7', 1) / atr_values.get('atr_14', 1) if atr_values.get('atr_14', 0) > 0 else 1
            atr_ratio_14_28 = atr_values.get('atr_14', 1) / atr_values.get('atr_28', 1) if atr_values.get('atr_28', 0) > 0 else 1
            # 波动率曲率（二阶变化）
            atr_14 = ta.atr(data['high'], data['low'], data['close'], length=14)
            if atr_14 is not None and len(atr_14) >= 3:
                atr_curvature = atr_14.iloc[-1] - 2*atr_14.iloc[-2] + atr_14.iloc[-3]
            else:
                atr_curvature = 0
            # 波动率记忆性（自相关）
            if atr_14 is not None and len(atr_14) >= 10:
                atr_autocorr = atr_14.autocorr(lag=1)
            else:
                atr_autocorr = 0
            # === 3. 波动率状态判定 ===
            current_atr = atr_values.get('atr_14', 0)
            if len(data) >= 20:
                atr_history = ta.atr(data['high'], data['low'], data['close'], length=14).iloc[-20:]
                atr_percentile = stats.percentileofscore(atr_history, current_atr) / 100
                if atr_percentile > 0.8:
                    volatility_state = '极端高波动'
                elif atr_percentile > 0.6:
                    volatility_state = '高波动'
                elif atr_percentile < 0.2:
                    volatility_state = '极端低波动'
                elif atr_percentile < 0.4:
                    volatility_state = '低波动'
                else:
                    volatility_state = '中波动'
            else:
                atr_percentile = 0.5
                volatility_state = '中波动'
            # === 4. 混沌波动率检测 ===
            chaos_volatility = self._detect_chaos_in_volatility(atr_14 if atr_14 is not None else pd.Series())
            factors.update({
                'atr_position': float(atr_percentile * 100),
                'volatility_regime': volatility_state,
                'atr_7_14_ratio': float(atr_ratio_7_14),
                'atr_14_28_ratio': float(atr_ratio_14_28),
                'atr_curvature': float(atr_curvature),
                'atr_autocorrelation': float(atr_autocorr) if not np.isnan(atr_autocorr) else 0,
                'volatility_compression': atr_curvature < -0.1,
                'volatility_expansion': atr_curvature > 0.1,
                'chaos_volatility_detected': chaos_volatility.get('detected', False),
                'volatility_lyapunov': chaos_volatility.get('lyapunov', 0)
            })
        return factors
    
    # ====================== 幻方量化核心数学方法 ======================
    
    def _calculate_fractal_features(self, price_series: pd.Series) -> Dict:
        """计算价格序列的分形特征"""
        features = {}
        if len(price_series) < self.fractal_dimension_window:
            return features
        # 1. 分形维度（盒计数法）
        fractal_dim = self._calculate_series_fractal_dimension(price_series)
        # 2. 赫斯特指数（R/S分析）
        hurst_exp = self._calculate_hurst_exponent(price_series)
        # 3. 多重分形谱宽度
        mf_spectrum = self._calculate_multifractal_spectrum_simple(price_series)
        features.update({
            'fractal_dimension': float(fractal_dim) if not np.isnan(fractal_dim) else None,
            'hurst_exponent': float(hurst_exp) if not np.isnan(hurst_exp) else None,
            'multifractal_width': float(mf_spectrum.get('width', 0)),
            'fractal_complexity': float(mf_spectrum.get('complexity', 0))
        })
        return features
    
    def _calculate_series_fractal_dimension(self, series: pd.Series) -> float:
        """
        计算序列的分形维度（盒计数法） - v2.0 Numba优化版
        优化思路：使用Numba JIT编译替代Numpy的reshape和split操作，避免中间数组内存分配，
        直接在底层循环中计算极差和，显著减少内存开销和CPU耗时。
        """
        if len(series) < 10:
            return np.nan
        values = series.values.astype(np.float64)
        # 调用Numba静态内核
        return self._numba_calc_fractal_dim(values)

    @staticmethod
    @jit(nopython=True, nogil=True, cache=True)
    def _numba_calc_fractal_dim(values: np.ndarray) -> float:
        """分形维度计算Numba内核"""
        n = len(values)
        if n < 10:
            return np.nan
        scales = np.array([2, 4, 8, 16, 32, 64])
        counts = np.empty(len(scales), dtype=np.float64)
        valid_idx = 0
        valid_scales = np.empty(len(scales), dtype=np.float64)
        for i in range(len(scales)):
            scale = scales[i]
            if scale > n // 2:
                break
            n_groups = n // scale
            if n_groups < 1:
                continue
            sum_ranges = 0.0
            for g in range(n_groups):
                start = g * scale
                end = start + scale
                # 手动计算切片极差，避免切片拷贝
                min_val = values[start]
                max_val = values[start]
                for k in range(start + 1, end):
                    val = values[k]
                    if val < min_val: min_val = val
                    if val > max_val: max_val = val
                sum_ranges += (max_val - min_val)
            if sum_ranges > 0:
                counts[valid_idx] = sum_ranges / n_groups  # 归一化
                valid_scales[valid_idx] = scale
                valid_idx += 1
        if valid_idx >= 3:
            # 简单的线性回归计算斜率
            log_scales = np.log(valid_scales[:valid_idx])
            log_counts = np.log(counts[:valid_idx])
            # 手写线性回归以保持nopython模式
            x_mean = np.mean(log_scales)
            y_mean = np.mean(log_counts)
            numerator = np.sum((log_scales - x_mean) * (log_counts - y_mean))
            denominator = np.sum((log_scales - x_mean) ** 2)
            if denominator != 0:
                slope = numerator / denominator
                return 2.0 - slope
        return np.nan

    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """
        计算赫斯特指数（重标极差法） - v2.0 Numba优化版
        优化思路：原逻辑涉及多次数组切片和聚合，Numba版本通过一次性扫描计算特定Lag下的RS值，
        大幅提升在滑动窗口中重复计算Hurst指数的效率。
        """
        if len(series) < self.hurst_exponent_window:
            return 0.5
        values = series.values.astype(np.float64)
        # 仅取最近窗口数据
        window_values = values[-self.hurst_exponent_window:]
        return self._numba_calc_hurst(window_values)

    @staticmethod
    @jit(nopython=True, nogil=True, cache=True)
    def _numba_calc_hurst(values: np.ndarray) -> float:
        """赫斯特指数计算Numba内核"""
        n = len(values)
        min_lag = 2
        max_lag = n // 2
        lags = np.arange(min_lag, max_lag)
        if len(lags) < 2:
            return 0.5
        tau = np.empty(len(lags), dtype=np.float64)
        valid_count = 0
        valid_lags = np.empty(len(lags), dtype=np.float64)
        for i in range(len(lags)):
            lag = lags[i]
            n_segments = n // lag
            rs_sum = 0.0
            count_seg = 0
            for g in range(n_segments):
                start = g * lag
                end = start + lag
                # 计算片段均值
                seg_sum = 0.0
                for k in range(start, end):
                    seg_sum += values[k]
                mean = seg_sum / lag
                # 计算累积离差范围和标准差
                current_sum = 0.0
                max_cum = -1e20
                min_cum = 1e20
                sum_sq_diff = 0.0
                for k in range(start, end):
                    diff = values[k] - mean
                    current_sum += diff
                    if current_sum > max_cum: max_cum = current_sum
                    if current_sum < min_cum: min_cum = current_sum
                    sum_sq_diff += diff * diff
                std = np.sqrt(sum_sq_diff / lag)
                if std > 1e-8:
                    rs = (max_cum - min_cum) / std
                    rs_sum += rs
                    count_seg += 1
            if count_seg > 0:
                avg_rs = rs_sum / count_seg
                if avg_rs > 0:
                    tau[valid_count] = np.log(avg_rs)
                    valid_lags[valid_count] = np.log(lag)
                    valid_count += 1
        if valid_count >= 2:
            # 线性回归
            x = valid_lags[:valid_count]
            y = tau[:valid_count]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            num = np.sum((x - x_mean) * (y - y_mean))
            den = np.sum((x - x_mean) ** 2)
            if den != 0:
                return num / den
        return 0.5

    def _detect_quantum_tunneling(self, data: pd.DataFrame, donchian_factors: Dict) -> Dict:
        """检测量子隧穿效应（价格穿过理论屏障）"""
        tunneling = {}
        if len(data) < 5:
            return tunneling
        close = data['close']
        donchian_high = donchian_factors.get('donchian_high')
        donchian_low = donchian_factors.get('donchian_low')
        if donchian_high is None or donchian_low is None:
            return tunneling
        # 计算隧穿概率
        current_price = close.iloc[-1]
        barrier_height = min(
            abs(current_price - donchian_high),
            abs(current_price - donchian_low)
        )
        # 使用量子隧穿公式简化版
        if donchian_high != donchian_low:
            barrier_width = donchian_high - donchian_low
            tunneling_prob = np.exp(-2 * barrier_height / (barrier_width * 0.1)) if barrier_width > 0 else 0
            # 检测实际隧穿
            recent_prices = close.iloc[-5:]
            crossed_high = any(p > donchian_high for p in recent_prices)
            crossed_low = any(p < donchian_low for p in recent_prices)
            tunneling.update({
                'quantum_tunneling_prob': float(tunneling_prob),
                'tunneling_occurred': crossed_high or crossed_low,
                'tunneling_direction': '向上' if crossed_high else '向下' if crossed_low else '无'
            })
        return tunneling
    
    def _chaos_breakout_confirmation(self, data: pd.DataFrame, breakout_factors: Dict) -> Dict:
        """基于混沌理论的突破确认"""
        confirmation = {}
        if len(data) < 20:
            return confirmation
        # 计算Lyapunov指数（混沌程度）
        close_series = data['close']
        lyapunov = self._estimate_lyapunov_exponent(close_series)
        # 计算关联维度
        correlation_dim = self._estimate_correlation_dimension(close_series)
        # 混沌确认规则
        if lyapunov > 0.1:  # 系统处于混沌状态
            confirmation['chaos_confirmation'] = '混沌驱动突破'
            confirmation['breakout_chaos_level'] = '高'
        elif correlation_dim > 2.0:  # 高维混沌
            confirmation['chaos_confirmation'] = '高维混沌突破'
            confirmation['breakout_chaos_level'] = '中'
        else:
            confirmation['chaos_confirmation'] = '确定性突破'
            confirmation['breakout_chaos_level'] = '低'
        confirmation.update({
            'lyapunov_exponent': float(lyapunov),
            'correlation_dimension': float(correlation_dim)
        })
        return confirmation
    
    def _calculate_intraday_fractal_dimension(self, minute_data: pd.DataFrame) -> Dict:
        """计算日内分形维度"""
        features = {}
        if minute_data.empty or len(minute_data) < 30:
            return features
        # 使用不同时间尺度计算分形维度
        scales = [5, 15, 30, 60]  # 分钟
        fractal_dims = []
        for scale in scales:
            resampled = minute_data.resample(f'{scale}min').agg({
                'close': 'last'
            }).dropna()
            if len(resampled) >= 10:
                dim = self._calculate_series_fractal_dimension(resampled['close'])
                if not np.isnan(dim):
                    fractal_dims.append(dim)
        if fractal_dims:
            features.update({
                'intraday_fractal_mean': float(np.mean(fractal_dims)),
                'intraday_fractal_std': float(np.std(fractal_dims)),
                'intraday_fractal_range': float(max(fractal_dims) - min(fractal_dims))
            })
        return features
    
    def _analyze_phase_space_convergence(self, data: pd.DataFrame) -> Dict:
        """
        相空间收敛分析 - v2.0 向量化优化版
        优化思路：使用numpy.lib.stride_tricks.sliding_window_view替代循环构建，
        实现零内存拷贝的相空间视图构建，极大提升大数据量下的处理速度。
        """
        analysis = {}
        if len(data) < 20:
            return analysis
        close_series = data['close'].values
        tau = 1  # 延迟
        m = 3    # 嵌入维度
        # 使用stride_tricks进行零拷贝滑动窗口视图构建
        # sliding_window_view(arr, window_shape) -> [N-w+1, w]
        try:
            # 确保输入是numpy数组
            windows = sliding_window_view(close_series, window_shape=m)
            # 如果有延迟tau > 1，需要切片：windows[::tau] 但这里我们用stride trick生成的已经是连续的
            # 对于tau=1，直接使用；对于tau>1，需要更复杂的切片，此处保持tau=1的高效实现
            phase_array = windows
            if len(phase_array) >= 10:
                # 向量化计算极差
                ranges = np.ptp(phase_array, axis=0) # [m]
                phase_volume = np.prod(ranges)
                # 计算收敛趋势
                if len(phase_array) >= 20:
                    recent_points = phase_array[-10:]
                    recent_volume = np.prod(np.ptp(recent_points, axis=0))
                    convergence_ratio = recent_volume / phase_volume if phase_volume > 0 else 1.0
                else:
                    convergence_ratio = 1.0
                analysis.update({
                    'phase_space_volume': float(phase_volume),
                    'phase_convergence_ratio': float(convergence_ratio),
                    'phase_space_dimension': m
                })
        except Exception:
            # 降级处理或记录错误
            pass
        return analysis

    def _analyze_price_topology(self, data: pd.DataFrame) -> Dict:
        """价格拓扑结构分析"""
        topology = {}
        if len(data) < 10:
            return topology
        close_series = data['close']
        # 计算Betti数（拓扑不变量）的近似
        # 使用持久同调简化计算
        persistence = self._calculate_persistence_homology(close_series)
        # 计算拓扑复杂性
        complexity = len(persistence.get('features', []))
        topology.update({
            'topological_complexity': complexity,
            'betti_0_approx': persistence.get('betti_0', 1),
            'topological_features': persistence.get('features', [])
        })
        return topology
    
    def _calculate_multifractal_spectrum(self, data: pd.DataFrame) -> Dict:
        """
        计算多重分形谱 - v2.0 Numba并行计算版
        优化思路：移除了原有的Pandas切片循环，使用Numba内核在一次遍历中计算所有Q值和尺度的配分函数。
        """
        spectrum = {}
        if len(data) < 30:
            return spectrum
        # 提取Numpy数组，确保类型为float64
        close_values = data['close'].values.astype(np.float64)
        # 定义参数
        q_values = np.array([-5, -2, -1, 0, 1, 2, 5], dtype=np.float64)
        scales = np.array([2, 4, 8, 16, 32], dtype=np.float64) # 尺度必须小于序列长度
        # 过滤有效尺度
        valid_scales = scales[scales <= len(close_values) // 2]
        if len(valid_scales) < 2:
            return spectrum
        # 调用Numba内核计算tau(q)
        tau_q = self._numba_calc_mf_partition(close_values, q_values, valid_scales)
        # 计算多重分形谱宽度 (Python层向量化计算)
        # alpha = d(tau)/dq, f(alpha) = q*alpha - tau
        if not np.any(np.isnan(tau_q)):
            # 使用梯度近似导数
            alpha = np.gradient(tau_q, q_values)
            f_alpha = q_values * alpha - tau_q
            spectrum_width = np.max(f_alpha) - np.min(f_alpha)
            spectrum.update({
                'multifractal_spectrum_width': float(spectrum_width),
                'multifractal_tau_q': tau_q.tolist(),
                'multifractality_present': spectrum_width > 0.1
            })
        return spectrum

    @staticmethod
    @jit(nopython=True, nogil=True, cache=True)
    def _numba_calc_mf_partition(values: np.ndarray, q_values: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """多重分形配分函数计算内核"""
        n = len(values)
        n_qs = len(q_values)
        n_scales = len(scales)
        tau_q = np.empty(n_qs, dtype=np.float64)
        # 预分配对数尺度，用于回归
        log_scales = np.log(scales)
        # 对每个Q值计算Tau
        for i in range(n_qs):
            q = q_values[i]
            z_q = np.empty(n_scales, dtype=np.float64)
            valid_z_count = 0
            for j in range(n_scales):
                scale = int(scales[j])
                n_segments = n // scale
                sum_segment_measure = 0.0
                if n_segments > 0:
                    for k in range(n_segments):
                        start = k * scale
                        end = start + scale
                        # 计算片段内的方差作为测度
                        # 手动计算方差以避免数组切片
                        seg_sum = 0.0
                        seg_sq_sum = 0.0
                        for m in range(start, end):
                            val = values[m]
                            seg_sum += val
                            seg_sq_sum += val * val
                        mean = seg_sum / scale
                        variance = (seg_sq_sum / scale) - (mean * mean)
                        # 避免精度问题导致的负方差
                        if variance < 1e-10: variance = 1e-10
                        if q == 0:
                            sum_segment_measure += np.log(variance)
                        else:
                            sum_segment_measure += variance ** (q / 2.0)
                    # 归一化并取对数
                    if q == 0:
                        z_val = sum_segment_measure / n_segments # q=0时的特殊处理
                    else:
                        z_val = np.log(sum_segment_measure / n_segments) if sum_segment_measure > 0 else -1e10
                    z_q[j] = z_val
                    valid_z_count += 1
            # 线性回归计算斜率 tau(q)
            # fit log(Z_q) ~ tau * log(scale)
            # 注意：标准定义中 Z ~ scale^tau，所以是双对数回归
            if valid_z_count >= 2:
                sx = 0.0
                sy = 0.0
                sxx = 0.0
                sxy = 0.0
                count = 0
                for j in range(n_scales):
                    if z_q[j] > -1e9: # 简单过滤无效值
                        x = log_scales[j]
                        y = z_q[j]
                        sx += x
                        sy += y
                        sxx += x * x
                        sxy += x * y
                        count += 1
                if count >= 2:
                    denom = (count * sxx - sx * sx)
                    if denom != 0:
                        slope = (count * sxy - sx * sy) / denom
                        tau_q[i] = slope
                    else:
                        tau_q[i] = 0.0
                else:
                    tau_q[i] = 0.0
            else:
                tau_q[i] = 0.0
        return tau_q

    def _detect_chaos_edge(self, data: pd.DataFrame) -> Dict:
        """混沌边缘检测"""
        chaos = {}
        if len(data) < 50:
            return chaos
        close_series = data['close']
        # 计算多个混沌指标
        lyapunov = self._estimate_lyapunov_exponent(close_series)
        correlation_dim = self._estimate_correlation_dimension(close_series)
        entropy = self._calculate_approximate_entropy(close_series)
        # 混沌边缘判定
        edge_score = 0
        if 0.05 < lyapunov < 0.3:
            edge_score += 1
        if 1.5 < correlation_dim < 3.0:
            edge_score += 1
        if 0.1 < entropy < 0.5:
            edge_score += 1
        chaos.update({
            'chaos_edge_detected': edge_score >= 2,
            'chaos_edge_score': edge_score,
            'chaos_lyapunov': float(lyapunov),
            'chaos_correlation_dim': float(correlation_dim),
            'chaos_entropy': float(entropy)
        })
        return chaos
    
    def _calculate_adaptive_resonance_score(self, factors: Dict) -> float:
        """计算自适应共振评分"""
        score_components = []
        # 1. 通道共振评分
        if 'boll_harmonic_resonance' in factors:
            resonance = factors['boll_harmonic_resonance']
            if resonance:
                score_components.append(25)
        # 2. 突破能量评分
        if 'breakout_energy' in factors:
            energy = factors['breakout_energy']
            score_components.append(min(25, energy * 5))
        # 3. 收敛状态评分
        if 'convergence_score' in factors:
            conv_score = factors['convergence_score']
            score_components.append(conv_score * 0.25)  # 转换为0-25分
        # 4. 位置稳定性评分
        if 'position_stability' in factors:
            stability = factors['position_stability']
            score_components.append(25 * (1 - min(1, stability / 50)))  # 稳定性越高，评分越高
        # 5. 波动率状态评分
        if 'volatility_regime' in factors:
            regime = factors['volatility_regime']
            if regime in ['中波动', '低波动']:
                score_components.append(25)
            elif regime == '高波动':
                score_components.append(15)
            else:
                score_components.append(5)
        # 计算平均分
        if score_components:
            return float(np.mean(score_components))
        else:
            return 50.0
    
    def _calculate_adaptive_parameters(self, data: pd.DataFrame) -> Dict:
        """计算自适应参数"""
        params = {}
        if len(data) < 20:
            return params
        close = data['close']
        # 基于波动率调整参数
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        # 自适应布林带参数
        if volatility > 0:
            adaptive_std = self.bb_std * (1 + np.tanh(volatility * 10))
            adaptive_period = max(10, min(30, int(self.bb_period * (1 - volatility * 5))))
        else:
            adaptive_std = self.bb_std
            adaptive_period = self.bb_period
        # 自适应唐奇安周期
        hurst = self._calculate_hurst_exponent(close)
        adaptive_donchian = max(10, min(50, int(self.donchian_period * (1 + 0.5 * (hurst - 0.5)))))
        params.update({
            'adaptive_bb_std': adaptive_std,
            'adaptive_bb_period': adaptive_period,
            'adaptive_donchian_period': adaptive_donchian,
            'underlying_volatility': volatility,
            'underlying_hurst': hurst
        })
        return params
    
    def _multiscale_squeeze_detection(self, bb_base: pd.DataFrame, bb_short: pd.DataFrame, bb_long: pd.DataFrame) -> Dict:
        """多尺度挤压检测"""
        squeeze = {'squeeze': False, 'strength': 0}
        try:
            # 提取不同尺度的布林带宽度
            width_base = bb_base.iloc[:, 0] - bb_base.iloc[:, 2]  # 假设第一列是上轨，第三列是下轨
            width_short = bb_short.iloc[:, 0] - bb_short.iloc[:, 2]
            width_long = bb_long.iloc[:, 0] - bb_long.iloc[:, 2]
            # 计算相对宽度
            if len(width_base) >= 20 and len(width_short) >= 20 and len(width_long) >= 20:
                rel_width_base = width_base.iloc[-1] / width_base.mean()
                rel_width_short = width_short.iloc[-1] / width_short.mean()
                rel_width_long = width_long.iloc[-1] / width_long.mean()
                # 多尺度挤压条件
                squeeze_condition = (rel_width_base < 0.7 and 
                                   rel_width_short < 0.8 and 
                                   rel_width_long < 0.9)
                squeeze_strength = (1 - rel_width_base) * 0.4 + \
                                  (1 - rel_width_short) * 0.3 + \
                                  (1 - rel_width_long) * 0.3
                squeeze.update({
                    'squeeze': bool(squeeze_condition),
                    'strength': float(squeeze_strength)
                })
        except:
            pass
        return squeeze
    
    def _quantum_breakout_detection(self, close: pd.Series, bb_upper: pd.Series, bb_lower: pd.Series, bb_middle: pd.Series) -> Dict:
        """量子突破检测"""
        breakout = {'breakout': False, 'energy': 0}
        if len(close) < 5:
            return breakout
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        # 计算突破能量
        distance_to_upper = bb_upper.iloc[-1] - current_close if len(bb_upper) > 0 else 0
        distance_to_lower = current_close - bb_lower.iloc[-1] if len(bb_lower) > 0 else 0
        # 量子隧穿效应
        tunnel_energy = 0
        if distance_to_upper < 0 or distance_to_lower < 0:
            # 价格在通道之外，计算隧穿能量
            tunnel_distance = min(abs(distance_to_upper), abs(distance_to_lower))
            bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1] if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 1
            tunnel_energy = np.exp(-tunnel_distance / (bb_width * 0.1))
        # 动量能量
        momentum = (current_close - prev_close) / prev_close * 100
        momentum_energy = np.exp(abs(momentum) * 0.1)
        # 综合能量
        total_energy = tunnel_energy + momentum_energy
        # 突破判定
        is_breakout = (distance_to_upper < 0 and prev_close <= bb_upper.iloc[-1]) or \
                     (distance_to_lower < 0 and prev_close >= bb_lower.iloc[-1])
        breakout.update({
            'breakout': bool(is_breakout),
            'energy': float(total_energy),
            'tunnel_energy': float(tunnel_energy),
            'momentum_energy': float(momentum_energy)
        })
        return breakout
    
    def _detect_harmonic_resonance(self, bb_upper: pd.Series, bb_middle: pd.Series, bb_lower: pd.Series, close: pd.Series) -> bool:
        """谐波共振检测"""
        if len(bb_upper) < 10 or len(close) < 10:
            return False
        # 计算通道宽度序列的傅里叶变换
        bb_width = bb_upper - bb_lower
        width_values = bb_width.iloc[-20:].values
        if len(width_values) < 10:
            return False
        # 快速傅里叶变换
        fft_result = np.fft.fft(width_values)
        frequencies = np.fft.fftfreq(len(width_values))
        # 寻找主频率
        magnitude = np.abs(fft_result)
        main_freq_idx = np.argmax(magnitude[1:]) + 1  # 忽略直流分量
        main_freq = frequencies[main_freq_idx]
        # 检查是否存在谐波共振（主频率为基频的整数倍）
        fundamental_freq = 1.0 / len(width_values)  # 基频
        if main_freq > 0:
            harmonic_ratio = main_freq / fundamental_freq
            # 检查是否接近整数倍（公差10%）
            return abs(harmonic_ratio - round(harmonic_ratio)) < 0.1
        return False
    
    def _calculate_topology_invariance(self, high_series: pd.Series, low_series: pd.Series) -> float:
        """计算拓扑不变性（通道形态保持性）"""
        if len(high_series) < 10 or len(low_series) < 10:
            return 0.5
        # 计算通道宽度序列的自相似性
        width_series = high_series - low_series
        if len(width_series) >= 20:
            # 分割为两个子序列
            half_len = len(width_series) // 2
            first_half = width_series.iloc[:half_len]
            second_half = width_series.iloc[half_len:]
            # 计算统计相似性
            mean_ratio = second_half.mean() / first_half.mean() if first_half.mean() != 0 else 1
            std_ratio = second_half.std() / first_half.std() if first_half.std() != 0 else 1
            # 计算拓扑不变性得分（0-1）
            invariance = 1.0 - 0.5 * abs(mean_ratio - 1) - 0.5 * abs(std_ratio - 1)
            return max(0, min(1, invariance))
        return 0.5
    
    def _check_breakout_momentum_confirmation(self, data: pd.DataFrame) -> bool:
        """检查突破动量确认"""
        if len(data) < 5:
            return False
        # 计算RSI动量
        rsi = ta.rsi(data['close'], length=14)
        if rsi is not None and len(rsi) >= 2:
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            # 突破时的RSI动量条件
            price_change = data['close'].iloc[-1] / data['close'].iloc[-2] - 1
            if price_change > 0:  # 向上突破
                return current_rsi > 50 and current_rsi > prev_rsi
            elif price_change < 0:  # 向下突破
                return current_rsi < 50 and current_rsi < prev_rsi
        return False
    
    def _analyze_post_breakout_behavior(self, data: pd.DataFrame, breakout_type: str) -> Dict:
        """分析突破后行为"""
        analysis = {}
        if len(data) < 10:
            return analysis
        # 计算突破后的价格行为
        returns = data['close'].pct_change()
        if breakout_type == 'quantum_up':
            # 向上突破后的行为
            post_breakout_returns = returns.iloc[-3:].mean() if len(returns) >= 3 else 0
            analysis['post_breakout_momentum'] = float(post_breakout_returns * 100)
            analysis['breakout_sustainability'] = '强' if post_breakout_returns > 0 else '弱'
        elif breakout_type == 'quantum_down':
            # 向下突破后的行为
            post_breakout_returns = returns.iloc[-3:].mean() if len(returns) >= 3 else 0
            analysis['post_breakout_momentum'] = float(post_breakout_returns * 100)
            analysis['breakout_sustainability'] = '强' if post_breakout_returns < 0 else '弱'
        return analysis
    
    def _detect_intraday_quantum_tunneling(self, minute_data: pd.DataFrame, scale_factors: Dict) -> Dict:
        """检测日内量子隧穿效应"""
        tunneling = {'detected': False, 'intensity': 0}
        if minute_data.empty or not scale_factors:
            return tunneling
        # 检查不同时间尺度的通道突破一致性
        tunneling_signals = []
        for scale, factors in scale_factors.items():
            position = factors.get('position', 0.5)
            # 极端位置可能发生隧穿
            if position > 0.9 or position < 0.1:
                tunneling_signals.append(True)
        # 计算隧穿强度
        if tunneling_signals:
            tunneling_intensity = sum(tunneling_signals) / len(tunneling_signals)
            tunneling.update({
                'detected': tunneling_intensity > 0.5,
                'intensity': tunneling_intensity
            })
        return tunneling
    
    def _calculate_shannon_entropy(self, series: pd.Series) -> float:
        """计算香农熵"""
        if len(series) < 10:
            return 0
        # 离散化序列
        hist, _ = np.histogram(series, bins=10)
        prob = hist / hist.sum()
        # 计算熵
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        return float(entropy)
    
    def _analyze_convergence_dynamics(self, bb_width: pd.Series, donchian_width: pd.Series) -> Dict:
        """分析收敛动力学"""
        dynamics = {}
        if len(bb_width) < 10 or len(donchian_width) < 10:
            return dynamics
        # 计算收敛速度（宽度变化率）
        bb_speed = bb_width.diff().abs().mean()
        donchian_speed = donchian_width.diff().abs().mean()
        # 计算收敛加速度
        bb_acceleration = bb_width.diff().diff().abs().mean()
        donchian_acceleration = donchian_width.diff().diff().abs().mean()
        dynamics.update({
            'convergence_speed_bb': float(bb_speed),
            'convergence_speed_donchian': float(donchian_speed),
            'convergence_acceleration_bb': float(bb_acceleration),
            'convergence_acceleration_donchian': float(donchian_acceleration)
        })
        return dynamics
    
    def _get_topological_position(self, weighted_position: float, raw_positions: List[float]) -> str:
        """获取拓扑位置分类"""
        # 基于分形理论的非线性分类
        if weighted_position > 0.85:
            return '拓扑上界'
        elif weighted_position > 0.7:
            return '拓扑上区'
        elif weighted_position > 0.6:
            return '拓扑中上'
        elif weighted_position > 0.4:
            return '拓扑中心'
        elif weighted_position > 0.3:
            return '拓扑中下'
        elif weighted_position > 0.15:
            return '拓扑下区'
        else:
            return '拓扑下界'
    
    def _calculate_quantum_position_distribution(self, weighted_position: float, raw_positions: List[float]) -> Dict:
        """计算量子位置分布"""
        distribution = {
            'mean': weighted_position,
            'std': np.std(raw_positions) if raw_positions else 0,
            'uncertainty': 0
        }
        # 量子不确定性（海森堡原理启发）
        if raw_positions:
            position_range = max(raw_positions) - min(raw_positions)
            momentum_uncertainty = np.std([p - weighted_position for p in raw_positions])
            distribution['uncertainty'] = float(position_range * momentum_uncertainty)
        return distribution
    
    def _calculate_recent_positions(self, data: pd.DataFrame, bb_data: Dict, donchian_data: Dict) -> List[float]:
        """计算近期位置序列"""
        positions = []
        if len(data) < 5:
            return positions
        # 获取最近5天的位置
        for i in range(min(5, len(data))):
            idx = -1 - i
            price = data['close'].iloc[idx]
            # 布林带位置
            bb_pos = None
            if 'upper' in bb_data and 'lower' in bb_data:
                bb_upper = bb_data['upper'][idx] if len(bb_data['upper']) > abs(idx) else None
                bb_lower = bb_data['lower'][idx] if len(bb_data['lower']) > abs(idx) else None
                if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
                    bb_pos = (price - bb_lower) / (bb_upper - bb_lower)
            # 唐奇安位置
            donchian_pos = None
            if 'high' in donchian_data and 'low' in donchian_data:
                donchian_high = donchian_data['high'][idx] if len(donchian_data['high']) > abs(idx) else None
                donchian_low = donchian_data['low'][idx] if len(donchian_data['low']) > abs(idx) else None
                if donchian_high is not None and donchian_low is not None and donchian_high != donchian_low:
                    donchian_pos = (price - donchian_low) / (donchian_high - donchian_low)
            # 计算综合位置
            if bb_pos is not None and donchian_pos is not None:
                positions.append(0.6 * bb_pos + 0.4 * donchian_pos)
            elif bb_pos is not None:
                positions.append(bb_pos)
            elif donchian_pos is not None:
                positions.append(donchian_pos)
        return positions[::-1]  # 按时间顺序返回
    
    def _detect_chaos_in_volatility(self, atr_series: pd.Series) -> Dict:
        """检测波动率中的混沌 - v1.2 增强健壮性"""
        chaos = {'detected': False, 'lyapunov': 0}
        if len(atr_series) < 30: # 增加最小样本要求
            return chaos
        try:
            # 使用优化后的Lyapunov估算
            lyapunov = self._estimate_lyapunov_exponent(atr_series)
            chaos.update({
                'detected': lyapunov > 0.05,
                'lyapunov': float(lyapunov)
            })
        except Exception:
            pass
        return chaos

    def _estimate_lyapunov_exponent(self, series: pd.Series) -> float:
        """
        估计Lyapunov指数 - v2.0 向量化版
        优化思路：利用Numpy的高效差分和过滤操作，去除了Python层面的显式循环。
        """
        if len(series) < 30:
            return 0.0
        values = series.values
        n = min(50, len(values))
        sample = values[-n:]
        # 向量化计算：一步即可完成差分和绝对值
        divergences = np.abs(np.diff(sample))
        # 向量化过滤：布尔索引比循环快得多
        valid_divergences = divergences[divergences > 1e-10] # 避免log(0)和极小噪点
        if len(valid_divergences) > 0:
            # 向量化对数均值
            lyapunov = np.mean(np.log(valid_divergences))
            return float(lyapunov)
        return 0.0

    def _estimate_correlation_dimension(self, series: pd.Series) -> float:
        """
        估计关联维度 - v2.0 Numba优化版
        优化思路：原O(N^2)广播矩阵计算消耗大量内存。Numba版本使用双重循环直接计数，
        完全避免了距离矩阵的内存分配，对于金融时间序列分析至关重要。
        """
        if len(series) < 20:
            return 1.0
        x = series.values.astype(np.float64)
        n = min(50, len(x)) # 限制样本量以保证实时性，Numba下可适当放宽
        x_sample = x[-n:]
        return self._numba_calc_correlation_dim(x_sample)

    @staticmethod
    @jit(nopython=True, nogil=True, cache=True)
    def _numba_calc_correlation_dim(x: np.ndarray) -> float:
        """关联维度计算Numba内核"""
        n = len(x)
        # 计算所有点对距离并构建直方图
        # 这里为了效率，不存储所有距离，直接在对数空间进行分箱统计
        epsilons = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
        counts = np.zeros(len(epsilons), dtype=np.float64)
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(x[i] - x[j])
                if dist > 0:
                    for k in range(len(epsilons)):
                        if dist < epsilons[k]:
                            counts[k] += 1
                    total_pairs += 1
        if total_pairs == 0:
            return 1.0
        # 过滤有效点并回归
        valid_x = []
        valid_y = []
        for k in range(len(epsilons)):
            if counts[k] > 0 and counts[k] < total_pairs:
                valid_x.append(np.log(epsilons[k]))
                valid_y.append(np.log(counts[k] / total_pairs))
        if len(valid_x) >= 3:
            vx = np.array(valid_x)
            vy = np.array(valid_y)
            # 简单回归
            slope = (np.mean(vx * vy) - np.mean(vx) * np.mean(vy)) / (np.mean(vx**2) - np.mean(vx)**2)
            return slope
        return 1.0

    def _calculate_approximate_entropy(self, series: pd.Series) -> float:
        """
        计算近似熵 - v2.0 Numba优化版
        优化思路：近似熵计算核心是模式匹配，复杂度为O(N^2)。Numba实现避免了构建三维特征矩阵，
        通过直接索引比较，大幅提高缓存命中率。
        """
        if len(series) < 20:
            return 0.0
        u = series.values.astype(np.float64)
        return self._numba_calc_apen(u)

    @staticmethod
    @jit(nopython=True, nogil=True, cache=True)
    def _numba_calc_apen(u: np.ndarray) -> float:
        """近似熵计算Numba内核"""
        n = len(u)
        m = 2
        r = 0.2 * np.std(u)
        if r == 0: return 0.0
        def _phi(m_val, seq, N, R):
            count_sum = 0.0
            possible_vectors = N - m_val + 1
            for i in range(possible_vectors):
                match_count = 0
                for j in range(possible_vectors):
                    # 比较向量 i 和 j
                    dist = 0.0
                    for k in range(m_val):
                        d = abs(seq[i+k] - seq[j+k])
                        if d > dist:
                            dist = d
                    if dist <= R:
                        match_count += 1
                if match_count > 0:
                    count_sum += np.log(match_count / possible_vectors)
            return count_sum / possible_vectors
        phi_m = _phi(m, u, n, r)
        phi_m1 = _phi(m + 1, u, n, r)
        return max(0.0, phi_m - phi_m1)

    def _calculate_persistence_homology(self, series: pd.Series) -> Dict:
        """
        计算持久同调/拓扑极值 - v2.0 Numba单遍扫描版
        优化思路：移除了scipy依赖，使用Numba一次遍历同时找出极大值和极小值，
        大幅减少内存分配和函数调用开销。
        """
        persistence = {'betti_0': 1, 'features': []}
        if len(series) < 10:
            return persistence
        values = series.values.astype(np.float64)
        # 调用Numba内核获取极值点索引
        max_indices, min_indices = self._numba_find_extrema(values)
        features = []
        # 收集极大值特征
        for idx in max_indices:
            features.append({
                'type': 'maximum', 
                'value': float(values[idx]), 
                'position': int(idx)
            })
        # 收集极小值特征
        for idx in min_indices:
            features.append({
                'type': 'minimum', 
                'value': float(values[idx]), 
                'position': int(idx)
            })
        # 按位置排序特征
        features.sort(key=lambda x: x['position'])
        persistence.update({
            'betti_0': len(features) // 2 + 1 if features else 1,
            'features': features[:10]  # 只保留前10个显著特征
        })
        return persistence

    @staticmethod
    @jit(nopython=True, nogil=True, cache=True)
    def _numba_find_extrema(values: np.ndarray) -> tuple:
        """极值查找内核：一次遍历同时寻找极大值和极小值"""
        n = len(values)
        if n < 3:
            return np.array([0], dtype=np.int64)[:0], np.array([0], dtype=np.int64)[:0]
        # 预估最大可能的极值点数量，通常远小于n
        max_size = n // 2 + 1
        max_idxs = np.empty(max_size, dtype=np.int64)
        min_idxs = np.empty(max_size, dtype=np.int64)
        max_count = 0
        min_count = 0
        # 单遍扫描，2阶比较 (比左右都大/小)
        for i in range(1, n - 1):
            curr = values[i]
            prev = values[i-1]
            next_val = values[i+1]
            # 检测极大值
            if curr > prev and curr > next_val:
                max_idxs[max_count] = i
                max_count += 1
            # 检测极小值
            elif curr < prev and curr < next_val:
                min_idxs[min_count] = i
                min_count += 1
        return max_idxs[:max_count], min_idxs[:min_count]

    def _calculate_multifractal_spectrum_simple(self, series: pd.Series) -> Dict:
        """简化多重分形谱计算（向量化优化版）- v1.1"""
        spectrum = {'width': 0, 'complexity': 0}
        if len(series) < 30:
            return spectrum
        values = series.values
        try:
            # CWT部分（保持scipy调用，因其底层已优化）
            widths = np.arange(1, 11)
            cwtmatr = signal.cwt(values, signal.ricker, widths)
            max_coeffs = np.max(np.abs(cwtmatr), axis=1)
            if len(max_coeffs) >= 2:
                spectrum_width = np.std(max_coeffs) / np.mean(max_coeffs) if np.mean(max_coeffs) != 0 else 0
                spectrum['width'] = float(spectrum_width)
                complexity = np.mean(np.abs(cwtmatr))
                spectrum['complexity'] = float(complexity)
        except:
            pass
        return spectrum

    # ====================== 原始方法保持兼容性 ======================
    
    def _get_bollinger_position(self, position_pct: float) -> str:
        """根据百分比位置确定布林带位置（保持原接口）"""
        if np.isnan(position_pct):
            return None
        if position_pct > 80:
            return '上轨上方'
        elif position_pct > 60:
            return '上轨附近'
        elif position_pct > 40:
            return '中轨附近'
        elif position_pct > 20:
            return '下轨附近'
        else:
            return '下轨下方'
    
    def _check_breakout_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """检查突破的成交量确认（保持原接口）"""
        if 'volume' not in data.columns or len(data) < 20:
            return False
        volumes = data['volume'].iloc[-20:]
        current_volume = volumes.iloc[-1] if len(volumes) > 0 else 0
        avg_volume = volumes.iloc[:-1].mean() if len(volumes) > 1 else current_volume
        # 突破时成交量应放大（超过均量的1.5倍）
        return current_volume > avg_volume * 1.5 if avg_volume > 0 else False
    
    def _check_breakout_volatility_confirmation(self, data: pd.DataFrame) -> bool:
        """检查突破的波动率确认（保持原接口）"""
        if len(data) < 20:
            return False
        # 使用pandas-ta计算ATR
        atr_result = ta.atr(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            length=14
        )
        if atr_result is None or atr_result.empty:
            return False
        current_atr = atr_result.iloc[-1] if pd.notna(atr_result.iloc[-1]) else 0
        avg_atr = atr_result.iloc[-20:-1].mean() if len(atr_result) > 20 else current_atr
        # 突破时波幅应扩大（超过平均波幅的1.2倍）
        return current_atr > avg_atr * 1.2 if avg_atr > 0 else True
    
    # ====================== 原有高级方法保留 ======================
    
    def calculate_multi_timeframe_channels(self,daily_data: pd.DataFrame,weekly_data: Optional[pd.DataFrame] = None,monthly_data: Optional[pd.DataFrame] = None) -> Dict:
        """计算多时间框架通道结构（增强版）"""
        factors = {}
        # 计算日线通道（使用深度方法）
        daily_factors = self.calculate_price_channel_factors(daily_data)
        factors['daily'] = daily_factors
        # 计算周线通道
        if weekly_data is not None and not weekly_data.empty:
            weekly_factors = self.calculate_price_channel_factors(weekly_data)
            factors['weekly'] = weekly_factors
            # 多尺度分形共振分析
            resonance = self._check_multi_timeframe_resonance(daily_factors, weekly_factors)
            factors['weekly_resonance'] = resonance
            # 分形维度一致性
            daily_fractal = daily_factors.get('fractal_dimension', 1.5)
            weekly_fractal = weekly_factors.get('fractal_dimension', 1.5)
            factors['fractal_consistency'] = abs(daily_fractal - weekly_fractal) < 0.2
        # 计算月线通道
        if monthly_data is not None and not monthly_data.empty:
            monthly_factors = self.calculate_price_channel_factors(monthly_data)
            factors['monthly'] = monthly_factors
            # 多时间框架混沌同步
            if 'weekly' in factors:
                multi_resonance = self._check_multi_timeframe_resonance(
                    daily_factors, weekly_factors, monthly_factors
                )
                factors['multi_timeframe_resonance'] = multi_resonance
        # 计算多时间框架拓扑对齐度
        alignment_score = self._calculate_multi_timeframe_alignment(factors)
        factors['timeframe_alignment_score'] = alignment_score
        # 计算时间分形
        factors['time_fractal_analysis'] = self._analyze_time_fractal(
            daily_factors, 
            factors.get('weekly', {}), 
            factors.get('monthly', {})
        )
        return factors
    
    def _check_multi_timeframe_resonance(self, *timeframe_factors: Dict) -> Dict:
        """检查多时间框架共振（增强版）"""
        if not timeframe_factors:
            return {}
        resonance_signals = []
        resonance_strength = 0
        # 检查突破共振
        breakout_signals = []
        for factors in timeframe_factors:
            breakout = factors.get('channel_breakout')
            if breakout and breakout not in ['无突破', '突破待确认', '假突破风险']:
                breakout_signals.append(breakout)
        if breakout_signals:
            # 计算突破共振强度
            if len(set(breakout_signals)) == 1:
                resonance_signals.append(f"多时间框架{breakout_signals[0]}共振")
                resonance_strength += 1
        # 检查通道位置共振
        position_categories = []
        for factors in timeframe_factors:
            position = factors.get('channel_position')
            if position:
                position_categories.append(position)
        if position_categories:
            # 使用拓扑位置分类
            simplified_positions = []
            for pos in position_categories:
                if '上' in pos:
                    simplified_positions.append('高位')
                elif '下' in pos:
                    simplified_positions.append('低位')
                else:
                    simplified_positions.append('中位')
            if len(set(simplified_positions)) == 1:
                resonance_signals.append(f"多时间框架{simplified_positions[0]}共振")
                resonance_strength += 1
        # 检查混沌状态共振
        chaos_states = []
        for factors in timeframe_factors:
            chaos = factors.get('chaos_edge_detected', False)
            chaos_states.append(chaos)
        if len(set(chaos_states)) == 1 and chaos_states[0]:
            resonance_signals.append("多时间框架混沌共振")
            resonance_strength += 1
        return {
            'signals': resonance_signals,
            'strength': resonance_strength,
            'has_resonance': len(resonance_signals) > 0,
            'resonance_score': resonance_strength / 3.0 * 100  # 归一化到0-100
        }
    
    def _calculate_multi_timeframe_alignment(self, factors: Dict) -> float:
        """计算多时间框架对齐度（增强版）"""
        alignment_scores = []
        # 检查不同时间框架的通道方向是否一致
        if 'daily' in factors and 'weekly' in factors:
            daily_breakout = factors['daily'].get('channel_breakout')
            weekly_breakout = factors['weekly'].get('channel_breakout')
            if daily_breakout and weekly_breakout:
                if daily_breakout == weekly_breakout:
                    alignment_scores.append(100)
                elif '突破' in daily_breakout and '突破' in weekly_breakout:
                    # 不同方向的突破
                    alignment_scores.append(0)
                else:
                    alignment_scores.append(50)
        if 'weekly' in factors and 'monthly' in factors:
            weekly_breakout = factors['weekly'].get('channel_breakout')
            monthly_breakout = factors['monthly'].get('channel_breakout')
            if weekly_breakout and monthly_breakout:
                if weekly_breakout == monthly_breakout:
                    alignment_scores.append(100)
                elif '突破' in weekly_breakout and '突破' in monthly_breakout:
                    alignment_scores.append(0)
                else:
                    alignment_scores.append(50)
        # 计算分形维度对齐度
        fractal_dims = []
        for timeframe in ['daily', 'weekly', 'monthly']:
            if timeframe in factors:
                fractal_dim = factors[timeframe].get('fractal_dimension')
                if fractal_dim is not None:
                    fractal_dims.append(fractal_dim)
        if len(fractal_dims) >= 2:
            fractal_std = np.std(fractal_dims)
            fractal_alignment = 100 * np.exp(-fractal_std * 2)  # 标准差越小，对齐度越高
            alignment_scores.append(fractal_alignment)
        if alignment_scores:
            return float(np.mean(alignment_scores))
        else:
            return 50.0
    
    def _analyze_time_fractal(self, daily_factors: Dict, weekly_factors: Dict, monthly_factors: Dict) -> Dict:
        """分析时间分形结构"""
        time_fractal = {}
        # 收集各时间框架的分形特征
        fractals = []
        hurst_exponents = []
        for factors in [daily_factors, weekly_factors, monthly_factors]:
            if factors:
                fractal_dim = factors.get('fractal_dimension')
                hurst = factors.get('hurst_exponent')
                if fractal_dim is not None:
                    fractals.append(fractal_dim)
                if hurst is not None:
                    hurst_exponents.append(hurst)
        if fractals:
            time_fractal.update({
                'fractal_dimension_mean': float(np.mean(fractals)),
                'fractal_dimension_std': float(np.std(fractals)),
                'fractal_consistency': '高' if np.std(fractals) < 0.2 else '低'
            })
        if hurst_exponents:
            time_fractal.update({
                'hurst_mean': float(np.mean(hurst_exponents)),
                'hurst_std': float(np.std(hurst_exponents)),
                'market_memory_consistency': '高' if np.std(hurst_exponents) < 0.1 else '低'
            })
        return time_fractal
