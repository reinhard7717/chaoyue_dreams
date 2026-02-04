# services\structural_calculator\support_resistance_calculator.py
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

class SupportResistanceCalculator:
    """支撑阻力结构因子计算器（数学深度重构版）"""
    
    def __init__(self, stock_code: str):
        """
        初始化计算器
        Args:
            stock_code: 股票代码
        """
        self.stock_code = stock_code
        # 自适应窗口策略：根据波动率动态调整
        self.volatility_regimes = {
            'high_vol': {'short': 15, 'medium': 45, 'long': 90},
            'normal_vol': {'short': 20, 'medium': 60, 'long': 120},
            'low_vol': {'short': 30, 'medium': 90, 'long': 180}
        }
        self.current_regime = 'normal_vol'
        # 混沌系统参数（基于A股市场特性）
        self.lyapunov_exponent = 0.3  # 李雅普诺夫指数，衡量系统混沌程度
        self.fractal_dimension = 1.6  # 分形维数
        
    def calculate_price_clusters(self, daily_data: pd.DataFrame, price_column: str = 'close') -> Dict[str, Any]:
        """
        计算价格密集区 - 基于高阶统计分析与拓扑学方法
        重构算法：
        1. 使用拓扑数据分析(TDA)识别价格流形的关键特征
        2. 基于黎曼几何度量价格密度分布的曲率
        3. 应用调和分析提取多尺度价格聚集模式
        Args:
            daily_data: 日线数据，需包含['close', 'high', 'low', 'vol', 'amount']
            price_column: 价格列名
        Returns:
            价格密集区分析结果
        """
        if len(daily_data) < 60:  # 最小样本要求
            return {}
        # 提取数据
        prices = daily_data[price_column].values
        volumes = daily_data['vol'].values
        highs = daily_data['high'].values
        lows = daily_data['low'].values
        # ========== 算法1：基于拓扑持久同调的密度模式识别 ==========
        def topological_density_analysis(price_series, volume_weights, n_clusters=5):
            """
            使用拓扑数据分析识别价格密度模式
            基于持续同调理论，提取价格分布的拓扑特征
            """
            # 1. 构造价格-时间的嵌入空间
            embedded_points = []
            for i in range(len(price_series)):
                # 价格-时间-成交量的三维嵌入
                embedded_points.append([
                    price_series[i] / np.mean(price_series),
                    i / len(price_series),
                    volume_weights[i] / np.max(volume_weights) if np.max(volume_weights) > 0 else 0
                ])
            embedded_points = np.array(embedded_points)
            # 2. 计算持续同调（简化实现）
            # 实际应用中可以使用gudhi等库，这里用距离矩阵近似
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(embedded_points))
            # 3. 基于距离层次聚类识别密度核心
            from scipy.cluster.hierarchy import fcluster, linkage
            Z = linkage(distances, method='ward')
            clusters = fcluster(Z, t=0.3 * np.max(Z[:, 2]), criterion='distance')
            # 4. 提取每个簇的统计特征
            cluster_stats = []
            unique_clusters = np.unique(clusters)
            for cluster_id in unique_clusters[:n_clusters]:
                mask = clusters == cluster_id
                if np.sum(mask) < 3:  # 太小的簇忽略
                    continue
                cluster_prices = price_series[mask]
                cluster_volumes = volume_weights[mask]
                # 计算簇的持久性（寿命）
                cluster_min = np.min(cluster_prices)
                cluster_max = np.max(cluster_prices)
                persistence = cluster_max - cluster_min
                # 计算簇的密度（带成交量权重）
                density = np.sum(cluster_volumes) / persistence if persistence > 0 else 0
                # 计算簇的质心（加权平均）
                centroid = np.average(cluster_prices, weights=cluster_volumes)
                cluster_stats.append({
                    'price_range': [float(cluster_min), float(cluster_max)],
                    'centroid': float(centroid),
                    'density': float(density),
                    'persistence': float(persistence),
                    'volume': float(np.sum(cluster_volumes))
                })
            return sorted(cluster_stats, key=lambda x: x['density'], reverse=True)
        # ========== 算法2：基于黎曼流形的曲率分析 ==========
        def riemannian_curvature_analysis(prices, volumes):
            """
            计算价格分布的黎曼曲率，识别结构突变点
            """
            # 使用滑动窗口计算局部曲率
            window = min(20, len(prices) // 4)
            curvatures = []
            for i in range(window, len(prices) - window):
                window_prices = prices[i-window:i+window]
                window_volumes = volumes[i-window:i+window] if len(volumes) == len(prices) else np.ones_like(window_prices)
                # 拟合局部多项式（三次样条）
                x = np.arange(len(window_prices))
                coeffs = np.polyfit(x, window_prices, 3)
                # 计算曲率：k = |f''(x)| / (1 + f'(x)^2)^(3/2)
                f_prime = np.polyval(np.polyder(coeffs, 1), window // 2)
                f_double_prime = np.polyval(np.polyder(coeffs, 2), window // 2)
                curvature = abs(f_double_prime) / (1 + f_prime**2)**1.5
                curvatures.append(curvature)
            # 标准化曲率
            if curvatures:
                curvatures = np.array(curvatures)
                curvatures = (curvatures - np.mean(curvatures)) / (np.std(curvatures) + 1e-10)
            return curvatures
        # ========== 算法3：多尺度小波变换密度估计 ==========
        def wavelet_density_estimation(prices, scales=[1, 2, 4, 8, 16]):
            """
            使用小波变换分析不同尺度下的价格密度
            """
            from scipy import signal
            wavelet_densities = {}
            for scale in scales:
                if scale > len(prices) // 4:
                    continue
                # 使用墨西哥帽小波
                wavelet = signal.ricker(min(200, len(prices)), scale)
                # 卷积得到小波系数
                coeffs = np.convolve(prices, wavelet, mode='same')
                # 寻找局部极值点（零交叉）
                zero_crossings = np.where(np.diff(np.sign(coeffs)))[0]
                # 计算该尺度下的密度特征
                if len(zero_crossings) > 1:
                    avg_spacing = np.mean(np.diff(zero_crossings))
                    density_measure = 1.0 / avg_spacing if avg_spacing > 0 else 0
                else:
                    density_measure = 0
                wavelet_densities[f'scale_{scale}'] = {
                    'density': float(density_measure),
                    'zero_crossings': int(len(zero_crossings))
                }
            return wavelet_densities
        # ========== 执行计算 ==========
        # 1. 拓扑密度分析
        topological_clusters = topological_density_analysis(prices[-120:], volumes[-120:] if len(volumes) >= 120 else np.ones_like(prices[-120:]))
        # 2. 曲率分析
        curvatures = riemannian_curvature_analysis(prices[-120:], volumes[-120:] if len(volumes) >= 120 else np.ones_like(prices[-120:]))
        curvature_strength = np.mean(np.abs(curvatures)) if len(curvatures) > 0 else 0
        # 3. 小波密度估计
        wavelet_densities = wavelet_density_estimation(prices[-120:])
        # 4. 区分支撑/阻力聚类
        current_price = prices[-1]
        support_clusters = []
        resistance_clusters = []
        for cluster in topological_clusters:
            centroid = cluster['centroid']
            cluster_info = {
                'price': centroid,
                'density': cluster['density'],
                'volume': cluster['volume'],
                'persistence': cluster['persistence'],
                'price_range': cluster['price_range'],
                'type': 'topological'
            }
            # 添加曲率特征
            if len(curvatures) > 0:
                # 找到最近的价格点
                price_idx = np.argmin(np.abs(prices[-120:] - centroid))
                if price_idx < len(curvatures):
                    cluster_info['curvature'] = float(curvatures[price_idx])
            # 添加小波特征
            for scale, wavelet_data in wavelet_densities.items():
                cluster_info[f'wavelet_{scale}'] = wavelet_data['density']
            if centroid < current_price:
                cluster_info['distance_to_price'] = (current_price - centroid) / current_price
                support_clusters.append(cluster_info)
            else:
                cluster_info['distance_to_price'] = (centroid - current_price) / current_price
                resistance_clusters.append(cluster_info)
        # 5. 计算整体密集区强度（使用信息熵和拓扑复杂度）
        def calculate_cluster_entropy(clusters):
            if not clusters:
                return 0
            densities = [c['density'] for c in clusters]
            total_density = np.sum(densities)
            if total_density <= 0:
                return 0
            # 归一化
            probs = np.array(densities) / total_density
            # 计算香农熵
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            # 计算拓扑复杂度（基于聚类数量和分布）
            complexity = len(clusters) * (1.0 - np.exp(-entropy))
            return complexity
        support_entropy = calculate_cluster_entropy(support_clusters)
        resistance_entropy = calculate_cluster_entropy(resistance_clusters)
        price_cluster_strength = (support_entropy + resistance_entropy) / 2
        # 6. 添加分形特征
        fractal_dimension = self._calculate_fractal_dimension(prices[-60:])
        return {
            'price_cluster_strength': float(price_cluster_strength),
            'support_cluster': support_clusters[:5],
            'resistance_cluster': resistance_clusters[:5],
            'curvature_strength': float(curvature_strength),
            'fractal_dimension': float(fractal_dimension),
            'wavelet_densities': wavelet_densities
        }
    
    def calculate_swing_structure(self, daily_data: pd.DataFrame,window: int = 5) -> Dict[str, bool]:
        """
        计算摆动结构 - 基于动力学系统与突变理论
        重构算法：
        1. 应用哈密顿力学构造价格相空间
        2. 使用突变理论识别价格状态的分岔点
        3. 基于李雅普诺夫指数量化趋势稳定性
        Args:
            daily_data: 日线数据
            window: 摆动点识别窗口
        Returns:
            摆动结构标记
        """
        if len(daily_data) < window * 3:
            return {}
        highs = daily_data['high'].values
        lows = daily_data['low'].values
        closes = daily_data['close'].values
        volumes = daily_data['vol'].values
        results = {
            'swing_high': False,
            'swing_low': False,
            'higher_high': False,
            'higher_low': False,
            'lower_high': False,
            'lower_low': False,
        }
        # ========== 算法1：相空间重构与庞加莱截面 ==========
        def poincare_section_analysis(prices, dimension=3, delay=1):
            """
            构造价格相空间并计算庞加莱截面
            识别价格运动的周期性结构
            """
            n = len(prices)
            if n < dimension * delay:
                return None
            # 重构相空间
            embedded = []
            for i in range(n - (dimension-1)*delay):
                point = []
                for j in range(dimension):
                    point.append(prices[i + j*delay])
                embedded.append(point)
            embedded = np.array(embedded)
            # 计算庞加莱截面（简化：价格返回截面）
            returns = np.diff(prices) / prices[:-1]
            # 识别穿越零点的时刻（价格方向变化）
            zero_crossings = np.where(np.diff(np.sign(returns)))[0]
            return {
                'embedded_dimension': dimension,
                'zero_crossings': zero_crossings.tolist() if len(zero_crossings) > 0 else [],
                'return_variance': float(np.var(returns))
            }
        # ========== 算法2：突变理论分析 ==========
        def catastrophe_theory_analysis(prices, volumes):
            """
            应用突变理论识别价格状态的突变点
            主要检测尖点突变(cusp catastrophe)
            """
            if len(prices) < 20:
                return {}
            # 计算价格的二阶差分（加速度）
            price_acceleration = np.diff(prices, 2)
            # 识别加速度的符号变化（突变点）
            acceleration_sign_changes = np.where(np.diff(np.sign(price_acceleration)))[0]
            # 计算突变强度（基于价格变化率和成交量）
            catastrophe_points = []
            for idx in acceleration_sign_changes:
                if idx + 2 < len(prices):
                    price_change = abs(prices[idx+2] - prices[idx])
                    volume_spike = volumes[idx+2] / np.mean(volumes[max(0, idx-5):idx+1]) if idx > 5 else 1.0
                    catastrophe_strength = price_change * volume_spike
                    catastrophe_points.append({
                        'index': int(idx),
                        'strength': float(catastrophe_strength),
                        'type': 'bullish' if price_acceleration[idx] > 0 else 'bearish'
                    })
            return catastrophe_points
        # ========== 算法3：李雅普诺夫指数计算 ==========
        def lyapunov_exponent_estimation(prices, embedded_dim=3, delay=1):
            """
            估计价格序列的李雅普诺夫指数
            衡量系统对初始条件的敏感度（混沌程度）
            """
            n = len(prices)
            if n < embedded_dim * delay * 10:
                return 0.0
            # 重构相空间
            m = n - (embedded_dim-1)*delay
            embedded = np.zeros((m, embedded_dim))
            for i in range(m):
                for j in range(embedded_dim):
                    embedded[i, j] = prices[i + j*delay]
            # 简化计算：使用最近邻方法估计李雅普诺夫指数
            # 实际应用中可以使用更精确的算法
            distances = []
            for i in range(m-1):
                # 找到最近邻点（排除太近的点）
                if i > 0:
                    distances.append(np.abs(embedded[i] - embedded[i-1]).mean())
            if distances:
                avg_distance = np.mean(distances)
                lyapunov_est = np.log(avg_distance + 1e-10) / m if m > 0 else 0
            else:
                lyapunov_est = 0
            return lyapunov_est
        # ========== 执行计算 ==========
        current_idx = -1
        current_high = highs[current_idx]
        current_low = lows[current_idx]
        current_close = closes[current_idx]
        # 1. 庞加莱截面分析
        poincare_data = poincare_section_analysis(closes[-60:])
        # 2. 突变理论分析
        catastrophe_points = catastrophe_theory_analysis(closes[-60:], volumes[-60:] if len(volumes) >= 60 else np.ones(60))
        # 3. 李雅普诺夫指数
        lyapunov_exp = lyapunov_exponent_estimation(closes[-60:])
        # 4. 基于动力学的摆动点识别
        # 传统方法升级：考虑相空间结构
        is_swing_high = False
        is_swing_low = False
        # 检查最近是否有突变点
        recent_catastrophe = False
        if catastrophe_points:
            latest_catastrophe = max(catastrophe_points, key=lambda x: x['index'])
            if latest_catastrophe['index'] >= len(closes) - 10:  # 最近10天内
                recent_catastrophe = True
                if latest_catastrophe['type'] == 'bullish':
                    is_swing_high = True
                else:
                    is_swing_low = True
        # 如果没有突变点，使用传统方法但加入相空间约束
        if not recent_catastrophe:
            # 传统摆动点检测，但加入庞加莱截面验证
            is_swing_high = all(current_high > highs[current_idx-i] for i in range(1, min(window+1, len(highs)+current_idx)))
            is_swing_low = all(current_low < lows[current_idx-i] for i in range(1, min(window+1, len(lows)+current_idx)))
            # 庞加莱截面验证：检查是否在截面附近
            if poincare_data and poincare_data['zero_crossings']:
                last_zero_crossing = poincare_data['zero_crossings'][-1] if poincare_data['zero_crossings'] else 0
                time_since_crossing = len(closes) - last_zero_crossing
                if time_since_crossing < 5:  # 近期有方向变化
                    # 降低摆动点置信度
                    is_swing_high = is_swing_high and (volumes[current_idx] > 1.5 * np.mean(volumes[current_idx-5:current_idx]))
                    is_swing_low = is_swing_low and (volumes[current_idx] > 1.5 * np.mean(volumes[current_idx-5:current_idx]))
        results['swing_high'] = is_swing_high
        results['swing_low'] = is_swing_low
        # 5. 趋势结构识别（加入李雅普诺夫指数调整）
        if len(closes) >= 20:
            # 计算趋势稳定性
            trend_stability = 1.0 / (1.0 + abs(lyapunov_exp))
            # 寻找显著高低点（考虑稳定性）
            significant_highs = []
            significant_lows = []
            for i in range(max(-30, -len(closes)), 0):
                if i == -1:
                    continue
                # 高点检测（加入稳定性权重）
                is_local_high = True
                lookback = max(3, int(5 * trend_stability))
                for j in range(1, lookback):
                    if i - j >= -len(closes) and i + j < 0:
                        if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                            is_local_high = False
                            break
                if is_local_high:
                    significance = trend_stability * (volumes[i] / np.mean(volumes[i-5:i]) if i-5 >= -len(volumes) else 1.0)
                    significant_highs.append((i, highs[i], significance))
                # 低点检测
                is_local_low = True
                for j in range(1, lookback):
                    if i - j >= -len(closes) and i + j < 0:
                        if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                            is_local_low = False
                            break
                if is_local_low:
                    significance = trend_stability * (volumes[i] / np.mean(volumes[i-5:i]) if i-5 >= -len(volumes) else 1.0)
                    significant_lows.append((i, lows[i], significance))
            # 判断趋势结构
            if len(significant_highs) >= 2:
                significant_highs.sort(key=lambda x: x[0])
                _, high1, sig1 = significant_highs[-2]
                _, high2, sig2 = significant_highs[-1]
                # 加权比较
                weighted_high1 = high1 * sig1
                weighted_high2 = high2 * sig2
                if weighted_high2 > weighted_high1 * 1.01:  # 考虑稳定性阈值
                    results['higher_high'] = True
                elif weighted_high2 < weighted_high1 * 0.99:
                    results['lower_high'] = True
            if len(significant_lows) >= 2:
                significant_lows.sort(key=lambda x: x[0])
                _, low1, sig1 = significant_lows[-2]
                _, low2, sig2 = significant_lows[-1]
                weighted_low1 = low1 * sig1
                weighted_low2 = low2 * sig2
                if weighted_low2 > weighted_low1 * 1.01:
                    results['higher_low'] = True
                elif weighted_low2 < weighted_low1 * 0.99:
                    results['lower_low'] = True
        # 6. 添加动力学特征
        results['lyapunov_exponent'] = float(lyapunov_exp)
        results['catastrophe_points'] = catastrophe_points
        results['poincare_crossings'] = poincare_data['zero_crossings'] if poincare_data else []
        return results
    
    def calculate_trendline_analysis(self, daily_data: pd.DataFrame,pivot_window: int = 10) -> Dict[str, Any]:
        """
        计算趋势线分析 - 基于微分几何与变分原理
        重构算法：
        1. 使用测地线方程拟合价格流形的最短路径
        2. 应用变分法求解能量最小化的趋势线
        3. 基于高斯曲率评估趋势线的稳定性
        Args:
            daily_data: 日线数据
            pivot_window: 关键点识别窗口
        Returns:
            趋势线分析结果
        """
        if len(daily_data) < pivot_window * 3:
            return {}
        closes = daily_data['close'].values
        highs = daily_data['high'].values
        lows = daily_data['low'].values
        volumes = daily_data['vol'].values
        # ========== 算法1：测地线趋势线拟合 ==========
        def geodesic_trendline_fitting(points, weights=None):
            """
            使用测地线方程拟合价格流形上的最短路径
            考虑价格空间的黎曼度量
            """
            n = len(points)
            if n < 3:
                return None
            # 构造时间-价格空间
            t = np.arange(n)
            p = np.array(points)
            # 定义黎曼度量：ds² = dt² + α*dp² + β*dv²（如果提供成交量权重）
            # 这里简化：只考虑价格和时间
            # 使用测地线方程：d²p/dt² + Γ*(dp/dt)² = 0
            # 其中Γ是克里斯托费尔符号
            # 简化：使用变分法求解能量最小化曲线
            # 能量泛函：E[p(t)] = ∫(p'(t)² + λ*p''(t)²)dt
            # 离散化求解
            def energy_function(coeffs):
                # 拟合多项式
                poly = np.poly1d(coeffs)
                p_fit = poly(t)
                # 计算一阶导数和二阶导数
                p_prime = np.gradient(p_fit, t)
                p_double_prime = np.gradient(p_prime, t)
                # 能量：平滑性+拟合误差
                smoothness = np.sum(p_double_prime**2)
                fit_error = np.sum((p_fit - p)**2)
                return 0.7 * smoothness + 0.3 * fit_error
            # 优化求解（三次多项式）
            from scipy.optimize import minimize
            initial_coeffs = np.polyfit(t, p, 3)
            result = minimize(energy_function, initial_coeffs, method='BFGS')
            if result.success:
                optimal_coeffs = result.x
                # 计算趋势线的几何特征
                poly = np.poly1d(optimal_coeffs)
                slope = np.polyder(poly, 1)(n-1)  # 末端斜率
                # 计算曲率
                curvature = abs(np.polyder(poly, 2)(n-1)) / (1 + slope**2)**1.5
                return {
                    'coefficients': optimal_coeffs.tolist(),
                    'slope': float(slope),
                    'curvature': float(curvature),
                    'energy': float(result.fun)
                }
            return None
        # ========== 算法2：基于高斯过程的趋势不确定性 ==========
        def gaussian_process_trend_uncertainty(points, length_scale=10.0):
            """
            使用高斯过程回归估计趋势线的不确定性
            """
            n = len(points)
            t = np.arange(n).reshape(-1, 1)
            p = np.array(points)
            # 定义径向基函数核
            def rbf_kernel(t1, t2, length_scale=length_scale):
                dist_sq = np.sum((t1 - t2.T)**2, axis=1)
                return np.exp(-dist_sq / (2 * length_scale**2))
            # 计算核矩阵
            K = rbf_kernel(t, t)
            # 添加噪声
            noise_level = 0.1 * np.var(p)
            K_noise = K + noise_level * np.eye(n)
            # 预测（简化版，完整高斯过程需要更多计算）
            try:
                K_inv = np.linalg.inv(K_noise)
                # 预测函数
                def predict(t_new):
                    t_new = np.array(t_new).reshape(-1, 1)
                    k_star = rbf_kernel(t_new, t)
                    k_star_star = rbf_kernel(t_new, t_new)
                    mean = k_star @ K_inv @ p
                    cov = k_star_star - k_star @ K_inv @ k_star.T
                    return mean.flatten(), np.diag(cov)
                # 计算不确定性
                _, variances = predict([n-1])
                uncertainty = np.sqrt(variances[0]) / np.std(p) if np.std(p) > 0 else 0
                return float(uncertainty)
            except:
                return 0.5  # 默认中等不确定性
        # ========== 算法3：基于曲率流的趋势演化 ==========
        def curvature_flow_analysis(points, time_steps=5):
            """
            应用曲率流分析趋势的演化
            模拟趋势线在曲率驱动下的变形
            """
            n = len(points)
            if n < 10:
                return {'curvature_flow': 0, 'mean_curvature': 0}
            # 计算离散曲率
            curvatures = []
            for i in range(1, n-1):
                p_prev = points[i-1]
                p_curr = points[i]
                p_next = points[i+1]
                # 离散曲率估计
                curvature = abs(p_next - 2*p_curr + p_prev) / (1 + ((p_next - p_prev)/2)**2)**1.5
                curvatures.append(curvature)
            if curvatures:
                mean_curvature = np.mean(curvatures)
                # 曲率流：曲率沿趋势线的变化
                curvature_gradient = np.gradient(curvatures) if len(curvatures) > 1 else [0]
                curvature_flow = np.mean(np.abs(curvature_gradient))
            else:
                mean_curvature = 0
                curvature_flow = 0
            return {
                'curvature_flow': float(curvature_flow),
                'mean_curvature': float(mean_curvature)
            }
        # ========== 执行计算 ==========
        # 1. 识别关键点（使用曲率极值点）
        def find_curvature_extrema(series, window=5):
            """基于曲率寻找关键点"""
            n = len(series)
            if n < window * 2:
                return []
            # 计算曲率
            curvatures = []
            for i in range(window, n-window):
                segment = series[i-window:i+window+1]
                if len(segment) < 3:
                    curvatures.append(0)
                    continue
                # 拟合二次曲线计算曲率
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 2)
                curvature = abs(2 * coeffs[0])  # 二次项系数*2
                curvatures.append(curvature)
            # 寻找曲率极值点
            extrema = []
            for i in range(1, len(curvatures)-1):
                if curvatures[i] > curvatures[i-1] and curvatures[i] > curvatures[i+1]:
                    # 局部极大值
                    actual_idx = i + window
                    if actual_idx < len(series):
                        extrema.append((actual_idx, series[actual_idx], curvatures[i]))
            return sorted(extrema, key=lambda x: x[2], reverse=True)  # 按曲率排序
        high_curvature_points = find_curvature_extrema(highs[-60:], window=pivot_window)
        low_curvature_points = find_curvature_extrema(lows[-60:], window=pivot_window)
        # 2. 拟合测地线趋势线
        uptrend_line = None
        downtrend_line = None
        if len(low_curvature_points) >= 2:
            low_points = sorted(low_curvature_points[:2], key=lambda x: x[0])  # 取曲率最大的两个低点
            low_indices = [p[0] for p in low_points]
            low_prices = [p[1] for p in low_points]
            # 提取两个低点之间的数据
            start_idx, end_idx = min(low_indices), max(low_indices)
            segment_prices = lows[start_idx:end_idx+1]
            if len(segment_prices) >= 3:
                uptrend_line = geodesic_trendline_fitting(segment_prices)
                if uptrend_line:
                    uptrend_line['start_idx'] = start_idx
                    uptrend_line['end_idx'] = end_idx
        if len(high_curvature_points) >= 2:
            high_points = sorted(high_curvature_points[:2], key=lambda x: x[0])
            high_indices = [p[0] for p in high_points]
            high_prices = [p[1] for p in high_points]
            start_idx, end_idx = min(high_indices), max(high_indices)
            segment_prices = highs[start_idx:end_idx+1]
            if len(segment_prices) >= 3:
                downtrend_line = geodesic_trendline_fitting(segment_prices)
                if downtrend_line:
                    downtrend_line['start_idx'] = start_idx
                    downtrend_line['end_idx'] = end_idx
        # 3. 趋势线突破分析
        current_price = closes[-1]
        last_price = closes[-2] if len(closes) >= 2 else current_price
        trendline_break = False
        trendline_type = None
        break_confidence = 0.0
        if uptrend_line is not None:
            # 重建趋势线函数
            coeffs = uptrend_line['coefficients']
            poly = np.poly1d(coeffs)
            # 计算当前时间点（相对于趋势线起点）的趋势线值
            t_relative = len(closes) - 1 - uptrend_line['start_idx']
            if t_relative >= 0:
                trendline_value = poly(t_relative)
                # 高斯过程不确定性
                uncertainty = gaussian_process_trend_uncertainty(
                    lows[uptrend_line['start_idx']:min(uptrend_line['end_idx']+1, len(lows))]
                )
                # 突破判断（考虑不确定性）
                uncertainty_band = uncertainty * np.std(lows[uptrend_line['start_idx']:uptrend_line['end_idx']+1])
                if current_price < trendline_value - uncertainty_band and last_price >= trendline_value:
                    trendline_break = True
                    trendline_type = 'uptrend_break'
                    break_confidence = min(1.0, abs(current_price - trendline_value) / uncertainty_band)
        if downtrend_line is not None and not trendline_break:
            coeffs = downtrend_line['coefficients']
            poly = np.poly1d(coeffs)
            t_relative = len(closes) - 1 - downtrend_line['start_idx']
            if t_relative >= 0:
                trendline_value = poly(t_relative)
                uncertainty = gaussian_process_trend_uncertainty(
                    highs[downtrend_line['start_idx']:min(downtrend_line['end_idx']+1, len(highs))]
                )
                uncertainty_band = uncertainty * np.std(highs[downtrend_line['start_idx']:downtrend_line['end_idx']+1])
                if current_price > trendline_value + uncertainty_band and last_price <= trendline_value:
                    trendline_break = True
                    trendline_type = 'downtrend_break'
                    break_confidence = min(1.0, abs(current_price - trendline_value) / uncertainty_band)
        # 4. 曲率流分析
        curvature_flow_result = curvature_flow_analysis(closes[-30:])
        return {
            'trendline_break': trendline_break,
            'trendline_type': trendline_type,
            'break_confidence': float(break_confidence),
            'uptrend_line': uptrend_line,
            'downtrend_line': downtrend_line,
            'curvature_flow': curvature_flow_result['curvature_flow'],
            'mean_curvature': curvature_flow_result['mean_curvature']
        }
    
    def calculate_fibonacci_price_levels(self,daily_data: pd.DataFrame,lookback_period: int = 120) -> Dict[str, Any]:
        """
        计算斐波那契价格水平 - 基于分形几何与自相似性理论
        重构算法：
        1. 使用多重分形谱分析价格序列的自相似结构
        2. 基于黄金分割的量子化能级模型
        3. 应用重整化群理论识别关键价格尺度
        Args:
            daily_data: 日线数据
            lookback_period: 回溯周期
        Returns:
            斐波那契价格水平
        """
        if len(daily_data) < lookback_period:
            return {}
        closes = daily_data['close'].values
        highs = daily_data['high'].values
        lows = daily_data['low'].values
        volumes = daily_data['vol'].values
        # ========== 算法1：多重分形谱分析 ==========
        def multifractal_spectrum_analysis(prices, q_min=-5, q_max=5, q_step=1):
            """
            计算价格序列的多重分形谱
            量化价格波动的尺度不变性
            """
            n = len(prices)
            if n < 64:  # 需要足够数据点
                return {'hurst_exponent': 0.5, 'multifractal_width': 0}
            # 使用去趋势波动分析(DFA)
            def dfa_analysis(series):
                # 简化DFA实现
                n = len(series)
                scales = [16, 32, 64, 128, 256]
                scales = [s for s in scales if s <= n//4]
                if not scales:
                    return 0.5
                fluctuations = []
                for scale in scales:
                    # 分割序列
                    n_segments = n // scale
                    if n_segments < 2:
                        continue
                    segment_fluctuations = []
                    for i in range(n_segments):
                        segment = series[i*scale:(i+1)*scale]
                        if len(segment) < 2:
                            continue
                        # 去趋势（线性去趋势）
                        x = np.arange(len(segment))
                        coeffs = np.polyfit(x, segment, 1)
                        trend = np.polyval(coeffs, x)
                        detrended = segment - trend
                        # 计算波动
                        fluctuation = np.sqrt(np.mean(detrended**2))
                        segment_fluctuations.append(fluctuation)
                    if segment_fluctuations:
                        avg_fluctuation = np.mean(segment_fluctuations)
                        fluctuations.append((scale, avg_fluctuation))
                if len(fluctuations) < 2:
                    return 0.5
                # 拟合幂律：F(s) ~ s^H
                log_scales = np.log([f[0] for f in fluctuations])
                log_fluctuations = np.log([f[1] for f in fluctuations])
                coeffs = np.polyfit(log_scales, log_fluctuations, 1)
                hurst_exponent = coeffs[0]  # H值
                return hurst_exponent
            # 计算Hurst指数
            hurst = dfa_analysis(prices)
            # 多重分形宽度（简化估计）
            # 使用不同q阶矩的广义Hurst指数
            q_values = np.arange(q_min, q_max + q_step, q_step)
            hq_values = []
            for q in q_values:
                if q == 0:
                    # q=0时的特殊处理
                    hq = hurst
                else:
                    # 使用波动函数的q阶矩
                    series_q = np.sign(prices - np.mean(prices)) * np.abs(prices - np.mean(prices))**(q/2)
                    hq = dfa_analysis(series_q)
                hq_values.append(hq)
            # 多重分形谱宽度
            if len(hq_values) > 1:
                multifractal_width = max(hq_values) - min(hq_values)
            else:
                multifractal_width = 0
            return {
                'hurst_exponent': float(hurst),
                'multifractal_width': float(multifractal_width),
                'hq_spectrum': {f'q_{q}': float(hq) for q, hq in zip(q_values, hq_values)}
            }
        # ========== 算法2：量子化能级模型 ==========
        def quantum_level_model(prices, base_energy=None):
            """
            将价格运动建模为量子化能级跃迁
            基于黄金分割比例构造能级结构
            """
            if base_energy is None:
                base_energy = np.mean(prices)
            # 黄金分割比例
            phi = (1 + np.sqrt(5)) / 2  # 1.618...
            phi_inv = 1 / phi  # 0.618...
            # 扩展的斐波那契比例序列
            fib_ratios = [
                0.236,  # φ^-3
                0.382,  # φ^-2
                0.500,  # 1/2
                0.618,  # φ^-1
                0.786,  # √φ^-1
                1.000,  # 基态
                1.272,  # √φ
                1.618,  # φ
                2.618,  # φ^2
                4.236,  # φ^3
            ]
            # 计算能级
            price_range = np.max(prices) - np.min(prices)
            energy_levels = {}
            for ratio in fib_ratios:
                if ratio < 1:
                    # 支撑能级
                    energy = np.min(prices) + price_range * ratio
                    energy_type = 'support'
                elif ratio == 1:
                    # 基态能级
                    energy = base_energy
                    energy_type = 'ground'
                else:
                    # 阻力能级
                    energy = np.min(prices) + price_range * min(ratio, 4.236)
                    energy_type = 'resistance'
                energy_levels[f'level_{ratio}'] = {
                    'energy': float(energy),
                    'type': energy_type,
                    'ratio': float(ratio)
                }
            return energy_levels
        # ========== 算法3：重整化群尺度分析 ==========
        def renormalization_group_analysis(prices, min_scale=5, max_scale=50):
            """
            应用重整化群理论识别关键价格尺度
            分析价格在不同尺度下的不变性
            """
            n = len(prices)
            scales = []
            scale_factors = []
            for scale in range(min_scale, min(max_scale, n//2)):
                # 粗粒化：将序列分块并取平均
                n_blocks = n // scale
                if n_blocks < 2:
                    continue
                coarse_grained = []
                for i in range(n_blocks):
                    block = prices[i*scale:(i+1)*scale]
                    coarse_grained.append(np.mean(block))
                # 计算粗粒化前后的统计相似性
                orig_stats = {
                    'mean': np.mean(prices),
                    'std': np.std(prices),
                    'skew': stats.skew(prices),
                    'kurt': stats.kurtosis(prices)
                }
                coarse_stats = {
                    'mean': np.mean(coarse_grained),
                    'std': np.std(coarse_grained),
                    'skew': stats.skew(coarse_grained),
                    'kurt': stats.kurtosis(coarse_grained)
                }
                # 计算相似度
                similarity = 0
                for key in orig_stats:
                    if key == 'std' and orig_stats[key] > 0 and coarse_stats[key] > 0:
                        similarity += 1 - abs(orig_stats[key] - coarse_stats[key]) / max(orig_stats[key], coarse_stats[key])
                    elif key != 'std':
                        similarity += 1 - min(1.0, abs(orig_stats[key] - coarse_stats[key]))
                similarity /= len(orig_stats)
                scales.append(scale)
                scale_factors.append(similarity)
            # 寻找最优尺度（相似度最高的）
            if scales and scale_factors:
                best_idx = np.argmax(scale_factors)
                optimal_scale = scales[best_idx]
                max_similarity = scale_factors[best_idx]
            else:
                optimal_scale = 20
                max_similarity = 0.5
            return {
                'optimal_scale': int(optimal_scale),
                'scale_similarity': float(max_similarity),
                'scale_spectrum': {f'scale_{s}': float(f) for s, f in zip(scales, scale_factors)}
            }
        # ========== 执行计算 ==========
        recent_data = daily_data.iloc[-lookback_period:]
        significant_high = recent_data['high'].max()
        significant_low = recent_data['low'].min()
        price_range = significant_high - significant_low
        # 1. 多重分形谱分析
        multifractal_result = multifractal_spectrum_analysis(closes[-lookback_period:])
        hurst_exponent = multifractal_result['hurst_exponent']
        # 2. 量子化能级模型
        # 使用成交量加权平均作为基态能量
        if len(volumes) >= lookback_period:
            base_energy = np.average(closes[-lookback_period:], weights=volumes[-lookback_period:])
        else:
            base_energy = np.mean(closes[-lookback_period:])
        quantum_levels = quantum_level_model(closes[-lookback_period:], base_energy)
        # 3. 重整化群分析
        renormalization_result = renormalization_group_analysis(closes[-lookback_period:])
        # 4. 斐波那契价格水平（传统+增强）
        # 传统斐波那契回撤位
        traditional_fib_levels = {
            0.236: significant_high - price_range * 0.236,
            0.382: significant_high - price_range * 0.382,
            0.500: significant_high - price_range * 0.500,
            0.618: significant_high - price_range * 0.618,
            0.786: significant_high - price_range * 0.786,
        }
        # 斐波那契扩展位
        fib_extensions = {
            1.272: significant_high + price_range * 0.272,
            1.618: significant_high + price_range * 0.618,
            2.618: significant_high + price_range * 1.618,
        }
        # 5. 结合分形特征的动态阈值
        current_price = closes[-1]
        tolerance = 0.01 + (1 - hurst_exponent) * 0.02  # 根据Hurst指数调整容差
        # 判断是否在关键斐波那契价格位
        fib_price_flags = {}
        # 检查传统斐波那契位
        for level_name, level_value in {
            'fib_price_level_236': traditional_fib_levels[0.236],
            'fib_price_level_382': traditional_fib_levels[0.382],
            'fib_price_level_500': traditional_fib_levels[0.500],
            'fib_price_level_618': traditional_fib_levels[0.618],
            'fib_price_level_786': traditional_fib_levels[0.786],
            'fib_price_extension_1272': fib_extensions[1.272],
            'fib_price_extension_1618': fib_extensions[1.618],
            'fib_price_extension_2618': fib_extensions[2.618],
        }.items():
            price_diff = abs(current_price - level_value)
            relative_diff = price_diff / current_price if current_price > 0 else 1
            # 动态阈值：分形特征越强，阈值越小
            if relative_diff <= tolerance:
                fib_price_flags[level_name] = True
                # 计算共振强度
                resonance_strength = 1.0 - relative_diff / tolerance
                fib_price_flags[f'{level_name}_strength'] = float(resonance_strength)
            else:
                fib_price_flags[level_name] = False
                fib_price_flags[f'{level_name}_strength'] = 0.0
        # 6. 量子能级分析
        fib_price_resistance = None
        fib_price_support = None
        for ratio, level_info in quantum_levels.items():
            if level_info['type'] == 'support' and level_info['energy'] < current_price:
                if fib_price_support is None or level_info['energy'] > fib_price_support:
                    fib_price_support = level_info['energy']
            elif level_info['type'] == 'resistance' and level_info['energy'] > current_price:
                if fib_price_resistance is None or level_info['energy'] < fib_price_resistance:
                    fib_price_resistance = level_info['energy']
        return {
            **fib_price_flags,
            'fib_price_resistance': float(fib_price_resistance) if fib_price_resistance else None,
            'fib_price_support': float(fib_price_support) if fib_price_support else None,
            'significant_high': float(significant_high),
            'significant_low': float(significant_low),
            'traditional_fib_levels': {k: float(v) for k, v in traditional_fib_levels.items()},
            'fib_extensions': {k: float(v) for k, v in fib_extensions.items()},
            'quantum_levels': quantum_levels,
            'hurst_exponent': hurst_exponent,
            'multifractal_width': multifractal_result['multifractal_width'],
            'renormalization_scale': renormalization_result['optimal_scale']
        }
    
    def _calculate_fractal_dimension(self, series):
        """计算序列的分形维数（盒计数法）"""
        if len(series) < 10:
            return 1.0
        # 归一化序列
        series = (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-10)
        scales = []
        counts = []
        # 选择尺度
        n = len(series)
        max_scale = n // 4
        for scale in range(2, min(20, max_scale)):
            # 分割序列
            n_boxes = n // scale
            if n_boxes < 2:
                continue
            # 计算每个盒子的极差
            box_ranges = []
            for i in range(n_boxes):
                segment = series[i*scale:(i+1)*scale]
                if len(segment) > 0:
                    box_range = np.max(segment) - np.min(segment)
                    box_ranges.append(box_range)
            if box_ranges:
                # 统计非空盒子数
                threshold = 0.01  # 阈值，避免噪声
                non_empty_boxes = sum(1 for r in box_ranges if r > threshold)
                if non_empty_boxes > 0:
                    scales.append(scale)
                    counts.append(non_empty_boxes)
        if len(scales) >= 2:
            # 拟合log(count) ~ -D * log(scale)
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            coeffs = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = -coeffs[0]  # 分形维数
            return max(1.0, min(2.0, fractal_dim))  # 限制在1-2之间
        return 1.5  # 默认值
    
    def calculate_support_resistance_levels(self,daily_data: pd.DataFrame,minute_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        综合计算支撑阻力位 - 基于信息几何与最优传输理论
        重构算法：
        1. 使用Wasserstein距离度量价格分布间的相似性
        2. 基于信息几何构造支撑阻力的统计流形
        3. 应用最优传输理论识别价格流动的关键障碍
        Args:
            daily_data: 日线数据
            minute_data: 分钟线数据（可选）
        Returns:
            综合支撑阻力位
        """
        # 调用已有的高级计算方法
        fib_results = self.calculate_fibonacci_price_levels(daily_data)
        cluster_results = self.calculate_price_clusters(daily_data)
        swing_results = self.calculate_swing_structure(daily_data)
        # ========== 算法1：基于Wasserstein距离的分布对齐 ==========
        def wasserstein_distance_analysis(prices, reference_distribution='normal'):
            """
            计算价格分布与参考分布之间的Wasserstein距离
            识别分布的关键特征点
            """
            from scipy.stats import wasserstein_distance
            # 经验分布
            hist, bin_edges = np.histogram(prices, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            if reference_distribution == 'normal':
                # 拟合正态分布
                mu, sigma = np.mean(prices), np.std(prices)
                reference_pdf = stats.norm.pdf(bin_centers, mu, sigma)
                # 计算Wasserstein距离
                w_distance = wasserstein_distance(hist, reference_pdf)
                # 计算分布的关键分位数
                key_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
                key_levels = np.percentile(prices, [q*100 for q in key_quantiles])
                return {
                    'wasserstein_distance': float(w_distance),
                    'key_levels': {f'q_{q}': float(lvl) for q, lvl in zip(key_quantiles, key_levels)},
                    'distribution_type': 'normal' if w_distance < 0.1 else 'non_normal'
                }
            return {'wasserstein_distance': 0, 'key_levels': {}, 'distribution_type': 'unknown'}
        # ========== 算法2：信息几何与统计流形 ==========
        def information_geometry_analysis(prices, volumes=None):
            """
            使用信息几何方法分析价格分布
            计算Fisher信息度量关键参数的变化敏感度
            """
            n = len(prices)
            if n < 20:
                return {'fisher_information': 0, 'geodesic_length': 0}
            # 计算对数收益
            returns = np.diff(np.log(prices + 1e-10))
            # Fisher信息：度量模型参数变化的敏感度
            # 简化：使用收益率的二阶矩
            fisher_info = np.var(returns) if len(returns) > 1 else 0
            # 计算统计流形上的测地线长度
            # 使用KL散度序列
            if len(returns) >= 10:
                # 滑动窗口计算分布变化
                window_size = min(10, len(returns)//2)
                kl_divergences = []
                for i in range(len(returns) - window_size):
                    dist1 = returns[i:i+window_size//2]
                    dist2 = returns[i+window_size//2:i+window_size]
                    if len(dist1) > 1 and len(dist2) > 1:
                        # 计算KL散度（高斯近似）
                        mu1, sigma1 = np.mean(dist1), np.std(dist1)
                        mu2, sigma2 = np.mean(dist2), np.std(dist2)
                        # 高斯分布的KL散度
                        kl = np.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5
                        kl_divergences.append(max(0, kl))
                if kl_divergences:
                    geodesic_length = np.sum(kl_divergences)
                else:
                    geodesic_length = 0
            else:
                geodesic_length = 0
            return {
                'fisher_information': float(fisher_info),
                'geodesic_length': float(geodesic_length),
                'avg_return': float(np.mean(returns)) if len(returns) > 0 else 0
            }
        # ========== 算法3：最优传输理论 ==========
        def optimal_transport_analysis(price_histories, num_levels=5):
            """
            应用最优传输理论识别价格流动的最小成本路径
            计算支撑阻力的传输障碍
            """
            if len(price_histories) < 2:
                return {'transport_cost': 0, 'bottleneck_levels': []}
            # 简化实现：计算价格分布的质心移动成本
            centroids = []
            for i in range(len(price_histories)-1):
                dist1 = price_histories[i]
                dist2 = price_histories[i+1]
                if len(dist1) > 0 and len(dist2) > 0:
                    centroid1 = np.mean(dist1)
                    centroid2 = np.mean(dist2)
                    # 传输成本：质心距离
                    transport_cost = abs(centroid2 - centroid1)
                    centroids.append({
                        'from': float(centroid1),
                        'to': float(centroid2),
                        'cost': float(transport_cost)
                    })
            # 识别传输瓶颈（成本最高的转移）
            if centroids:
                sorted_centroids = sorted(centroids, key=lambda x: x['cost'], reverse=True)
                bottleneck_levels = []
                for centroid in sorted_centroids[:num_levels]:
                    bottleneck_levels.append({
                        'price': (centroid['from'] + centroid['to']) / 2,
                        'transport_cost': centroid['cost'],
                        'type': 'bottleneck'
                    })
                avg_transport_cost = np.mean([c['cost'] for c in centroids])
            else:
                bottleneck_levels = []
                avg_transport_cost = 0
            return {
                'transport_cost': float(avg_transport_cost),
                'bottleneck_levels': bottleneck_levels
            }
        # ========== 执行计算 ==========
        closes = daily_data['close'].values
        current_price = closes[-1] if len(closes) > 0 else 0
        # 1. Wasserstein距离分析
        wasserstein_result = wasserstein_distance_analysis(closes[-60:])
        # 2. 信息几何分析
        info_geometry_result = information_geometry_analysis(closes[-60:])
        # 3. 最优传输分析
        # 分割价格历史为多个分布
        price_histories = []
        segment_size = max(10, len(closes) // 6)
        for i in range(0, len(closes) - segment_size, segment_size // 2):
            segment = closes[i:i+segment_size]
            price_histories.append(segment)
        optimal_transport_result = optimal_transport_analysis(price_histories)
        # 4. 综合所有支撑阻力位（使用信息几何加权）
        all_support_levels = []
        all_resistance_levels = []
        # 从斐波那契结果添加
        fib_support = fib_results.get('fib_price_support')
        if fib_support and fib_support < current_price:
            # 加权：根据分形特征
            fib_strength = 0.7 + 0.3 * fib_results.get('hurst_exponent', 0.5)
            all_support_levels.append({
                'price': fib_support,
                'type': 'fibonacci',
                'strength': float(fib_strength),
                'timeframe': 'daily',
                'method': 'quantum_level'
            })
        # 从价格聚类结果添加
        for cluster in cluster_results.get('support_cluster', []):
            cluster_strength = cluster.get('density', 0)
            # 调整强度：考虑曲率特征
            curvature_adjustment = 1.0 + 0.5 * abs(cluster.get('curvature', 0))
            adjusted_strength = min(1.0, cluster_strength * curvature_adjustment)
            all_support_levels.append({
                'price': cluster['price'],
                'type': 'price_cluster',
                'strength': float(adjusted_strength),
                'timeframe': 'daily',
                'method': 'topological_density'
            })
        # 从Wasserstein关键位添加
        for q_name, q_level in wasserstein_result.get('key_levels', {}).items():
            if q_level < current_price:
                # 分位数越低，支撑强度越高
                q_value = float(q_name.split('_')[1])
                q_strength = (1.0 - q_value) * 0.8  # 0.05分位有0.76强度
                all_support_levels.append({
                    'price': q_level,
                    'type': 'distribution_quantile',
                    'strength': float(q_strength),
                    'timeframe': 'daily',
                    'method': 'wasserstein'
                })
        # 从最优传输瓶颈添加
        for bottleneck in optimal_transport_result.get('bottleneck_levels', []):
            bottleneck_price = bottleneck['price']
            if bottleneck_price < current_price:
                # 传输成本越高，支撑/阻力越强
                cost_strength = min(1.0, bottleneck['transport_cost'] / np.std(closes[-60:]) if np.std(closes[-60:]) > 0 else 0.5)
                all_support_levels.append({
                    'price': bottleneck_price,
                    'type': 'transport_bottleneck',
                    'strength': float(cost_strength),
                    'timeframe': 'daily',
                    'method': 'optimal_transport'
                })
        # 阻力位类似处理（价格高于当前价）
        fib_resistance = fib_results.get('fib_price_resistance')
        if fib_resistance and fib_resistance > current_price:
            fib_strength = 0.7 + 0.3 * fib_results.get('hurst_exponent', 0.5)
            all_resistance_levels.append({
                'price': fib_resistance,
                'type': 'fibonacci',
                'strength': float(fib_strength),
                'timeframe': 'daily',
                'method': 'quantum_level'
            })
        for cluster in cluster_results.get('resistance_cluster', []):
            cluster_strength = cluster.get('density', 0)
            curvature_adjustment = 1.0 + 0.5 * abs(cluster.get('curvature', 0))
            adjusted_strength = min(1.0, cluster_strength * curvature_adjustment)
            all_resistance_levels.append({
                'price': cluster['price'],
                'type': 'price_cluster',
                'strength': float(adjusted_strength),
                'timeframe': 'daily',
                'method': 'topological_density'
            })
        for q_name, q_level in wasserstein_result.get('key_levels', {}).items():
            if q_level > current_price:
                q_value = float(q_name.split('_')[1])
                q_strength = q_value * 0.8  # 0.95分位有0.76强度
                all_resistance_levels.append({
                    'price': q_level,
                    'type': 'distribution_quantile',
                    'strength': float(q_strength),
                    'timeframe': 'daily',
                    'method': 'wasserstein'
                })
        for bottleneck in optimal_transport_result.get('bottleneck_levels', []):
            bottleneck_price = bottleneck['price']
            if bottleneck_price > current_price:
                cost_strength = min(1.0, bottleneck['transport_cost'] / np.std(closes[-60:]) if np.std(closes[-60:]) > 0 else 0.5)
                all_resistance_levels.append({
                    'price': bottleneck_price,
                    'type': 'transport_bottleneck',
                    'strength': float(cost_strength),
                    'timeframe': 'daily',
                    'method': 'optimal_transport'
                })
        # 5. 去重和排序（使用更智能的合并策略）
        all_support_levels = self._geometric_deduplicate_levels(all_support_levels, closes)
        all_resistance_levels = self._geometric_deduplicate_levels(all_resistance_levels, closes)
        # 6. 计算价格目标（基于信息几何）
        price_targets = self._calculate_information_geometry_targets(
            daily_data, 
            all_support_levels, 
            all_resistance_levels,
            info_geometry_result
        )
        return {
            'support_levels': all_support_levels[:10],  # 最多10个最强的支撑位
            'resistance_levels': all_resistance_levels[:10],  # 最多10个最强的阻力位
            'price_targets': price_targets,
            'wasserstein_analysis': wasserstein_result,
            'information_geometry': info_geometry_result,
            'optimal_transport': optimal_transport_result
        }
    
    def _geometric_deduplicate_levels(self, levels, price_series):
        """基于几何度量的智能去重"""
        if not levels:
            return []
        # 按价格排序
        levels.sort(key=lambda x: x['price'])
        merged_levels = []
        current_group = []
        for level in levels:
            if not current_group:
                current_group.append(level)
            else:
                # 检查是否可以合并到当前组
                last_price = current_group[-1]['price']
                current_price = level['price']
                # 动态合并阈值：基于价格波动率
                price_std = np.std(price_series[-20:]) if len(price_series) >= 20 else abs(price_series[-1] * 0.01)
                merge_threshold = price_std * 0.5
                if abs(current_price - last_price) <= merge_threshold:
                    current_group.append(level)
                else:
                    # 合并当前组
                    if current_group:
                        merged_level = self._merge_level_group(current_group, price_std)
                        merged_levels.append(merged_level)
                    current_group = [level]
        # 合并最后一组
        if current_group:
            merged_level = self._merge_level_group(current_group, price_std)
            merged_levels.append(merged_level)
        # 按强度排序
        merged_levels.sort(key=lambda x: x['strength'], reverse=True)
        return merged_levels
    
    def _merge_level_group(self, group, price_std):
        """合并一组相近的价格水平"""
        if not group:
            return None
        # 加权平均价格（按强度加权）
        total_strength = sum(level['strength'] for level in group)
        if total_strength > 0:
            weighted_price = sum(level['price'] * level['strength'] for level in group) / total_strength
            avg_strength = total_strength / len(group)
            # 合并类型和方法
            merged_types = set()
            merged_methods = set()
            for level in group:
                merged_types.add(level['type'])
                merged_methods.add(level.get('method', 'unknown'))
        else:
            weighted_price = np.mean([level['price'] for level in group])
            avg_strength = np.mean([level['strength'] for level in group])
            merged_types = {'mixed'}
            merged_methods = {'mixed'}
        # 计算置信度：价格分散度越小，置信度越高
        price_std_in_group = np.std([level['price'] for level in group])
        confidence = 1.0 - min(1.0, price_std_in_group / price_std) if price_std > 0 else 0.8
        return {
            'price': float(weighted_price),
            'strength': float(avg_strength * confidence),
            'type': ','.join(sorted(merged_types)),
            'method': ','.join(sorted(merged_methods)),
            'timeframe': group[0].get('timeframe', 'daily'),
            'confidence': float(confidence),
            'group_size': len(group)
        }
    
    def _calculate_information_geometry_targets(self,daily_data,support_levels,resistance_levels,info_geometry):
        """基于信息几何计算价格目标"""
        closes = daily_data['close'].values
        current_price = closes[-1] if len(closes) > 0 else 0
        # Fisher信息衡量市场的信息含量
        fisher_info = info_geometry.get('fisher_information', 0)
        # 计算目标位（考虑信息几何）
        bullish_targets = []
        bearish_targets = []
        if resistance_levels:
            nearest_resistance = min(level['price'] for level in resistance_levels)
            # 信息几何调整：Fisher信息越高，突破后的动量越大
            info_adjustment = 1.0 + fisher_info * 10  # 放大效应
            for multiplier in [1.0, 1.618, 2.618]:
                base_target = nearest_resistance + (nearest_resistance - current_price) * multiplier
                adjusted_target = current_price + (base_target - current_price) * info_adjustment
                if adjusted_target > current_price:
                    bullish_targets.append({
                        'price': float(adjusted_target),
                        'multiplier': float(multiplier),
                        'info_adjustment': float(info_adjustment),
                        'type': 'resistance_breakout'
                    })
        if support_levels:
            nearest_support = max(level['price'] for level in support_levels)
            info_adjustment = 1.0 + fisher_info * 10
            for multiplier in [1.0, 1.618, 2.618]:
                base_target = nearest_support - (current_price - nearest_support) * multiplier
                adjusted_target = current_price - (current_price - base_target) * info_adjustment
                if adjusted_target < current_price:
                    bearish_targets.append({
                        'price': float(adjusted_target),
                        'multiplier': float(multiplier),
                        'info_adjustment': float(info_adjustment),
                        'type': 'support_breakdown'
                    })
        # 基于测地线长度的趋势延续目标
        geodesic_length = info_geometry.get('geodesic_length', 0)
        if geodesic_length > 0:
            # 测地线长度越大，趋势延续性越强
            trend_strength = min(1.0, geodesic_length * 10)
            avg_return = info_geometry.get('avg_return', 0)
            if avg_return > 0:  # 上升趋势
                continuation_target = current_price * np.exp(avg_return * (1 + trend_strength) * 10)
                bullish_targets.append({
                    'price': float(continuation_target),
                    'trend_strength': float(trend_strength),
                    'type': 'geodesic_continuation'
                })
            elif avg_return < 0:  # 下降趋势
                continuation_target = current_price * np.exp(avg_return * (1 + trend_strength) * 10)
                bearish_targets.append({
                    'price': float(continuation_target),
                    'trend_strength': float(trend_strength),
                    'type': 'geodesic_continuation'
                })
        # 排序和去重
        def deduplicate_targets(targets):
            if not targets:
                return []
            # 按价格分组
            targets.sort(key=lambda x: x['price'])
            deduplicated = []
            last_price = None
            price_tolerance = current_price * 0.01
            for target in targets:
                if last_price is None or abs(target['price'] - last_price) > price_tolerance:
                    deduplicated.append(target)
                    last_price = target['price']
            return deduplicated
        bullish_targets = deduplicate_targets(bullish_targets)
        bearish_targets = deduplicate_targets(bearish_targets)
        # 按价格排序
        bullish_targets.sort(key=lambda x: x['price'])
        bearish_targets.sort(key=lambda x: x['price'], reverse=True)
        return {
            'bullish': bullish_targets[:5],  # 最多5个看涨目标
            'bearish': bearish_targets[:5],  # 最多5个看跌目标
            'neutral': [{'price': float(current_price), 'type': 'current'}]
        }
    
    def calculate_all_support_resistance_factors(self,daily_data: pd.DataFrame,minute_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        计算所有支撑阻力结构因子
        主入口函数，整合所有高级数学方法
        Args:
            daily_data: 日线数据
            minute_data: 分钟线数据（可选）
        Returns:
            完整的支撑阻力结构因子字典
        """
        results = {}
        # 1. 价格密集区分析（拓扑学+微分几何）
        cluster_results = self.calculate_price_clusters(daily_data)
        results.update(cluster_results)
        # 2. 摆动结构分析（动力学系统+突变理论）
        swing_results = self.calculate_swing_structure(daily_data)
        results.update(swing_results)
        # 3. 趋势线分析（微分几何+高斯过程）
        trendline_results = self.calculate_trendline_analysis(daily_data)
        results.update(trendline_results)
        # 4. 斐波那契价格水平（分形几何+量子模型）
        fib_results = self.calculate_fibonacci_price_levels(daily_data)
        results.update(fib_results)
        # 5. 综合支撑阻力位（信息几何+最优传输）
        sr_levels = self.calculate_support_resistance_levels(daily_data, minute_data)
        results.update({
            'support_levels': sr_levels.get('support_levels', []),
            'resistance_levels': sr_levels.get('resistance_levels', []),
            'price_targets': sr_levels.get('price_targets', {}),
        })
        # 6. 计算综合结构强度（基于多方法融合）
        results['structure_strength'] = self._calculate_advanced_structure_strength(results)
        results['structure_score'] = results['structure_strength']  # 兼容字段
        return results
    
    def _calculate_advanced_structure_strength(self, factors: Dict[str, Any]) -> float:
        """基于多方法融合计算结构强度综合得分"""
        score_components = []
        weights = []
        # 1. 拓扑密度强度（权重25%）
        cluster_strength = factors.get('price_cluster_strength', 0)
        curvature_strength = factors.get('curvature_strength', 0)
        topological_score = cluster_strength * 0.7 + curvature_strength * 0.3
        score_components.append(topological_score)
        weights.append(0.25)
        # 2. 动力学稳定性（权重20%）
        lyapunov_exp = factors.get('lyapunov_exponent', 0.3)
        # 李雅普诺夫指数越小，系统越稳定
        stability_score = 100 * np.exp(-lyapunov_exp * 2)
        # 摆动结构确认
        swing_score = 0
        if factors.get('higher_high') and factors.get('higher_low'):
            swing_score = 80  # 上升趋势
        elif factors.get('lower_high') and factors.get('lower_low'):
            swing_score = 20  # 下降趋势
        else:
            swing_score = 50  # 震荡
        dynamics_score = stability_score * 0.4 + swing_score * 0.6
        score_components.append(dynamics_score)
        weights.append(0.20)
        # 3. 趋势线几何特征（权重15%）
        trendline_score = 50  # 基准分
        if factors.get('trendline_break'):
            break_confidence = factors.get('break_confidence', 0.5)
            trendline_score = 30 + break_confidence * 40  # 30-70分
        curvature_flow = factors.get('curvature_flow', 0)
        # 适中的曲率流最好（既不过于平直也不过于弯曲）
        curvature_score = 100 * np.exp(-abs(curvature_flow - 0.1) * 10)
        geometric_score = trendline_score * 0.6 + curvature_score * 0.4
        score_components.append(geometric_score)
        weights.append(0.15)
        # 4. 分形与量子特征（权重15%）
        hurst_exponent = factors.get('hurst_exponent', 0.5)
        # Hurst指数在0.5-0.7之间最理想（有趋势但不完全随机）
        if 0.5 <= hurst_exponent <= 0.7:
            hurst_score = 80 + (hurst_exponent - 0.5) * 100  # 80-100分
        else:
            hurst_score = max(0, 100 - abs(hurst_exponent - 0.6) * 200)
        # 斐波那契共振强度
        fib_score = 0
        fib_flags = [
            'fib_price_level_236', 'fib_price_level_382', 'fib_price_level_500',
            'fib_price_level_618', 'fib_price_level_786'
        ]
        for flag in fib_flags:
            if factors.get(flag):
                strength_key = f'{flag}_strength'
                strength = factors.get(strength_key, 0.5)
                fib_score += 20 * strength  # 加权强度
        fractal_score = hurst_score * 0.6 + min(100, fib_score) * 0.4
        score_components.append(fractal_score)
        weights.append(0.15)
        # 5. 信息几何特征（权重25%）
        # 支撑阻力位清晰度
        support_count = len(factors.get('support_levels', []))
        resistance_count = len(factors.get('resistance_levels', []))
        clarity_score = min(100, (support_count + resistance_count) * 15)
        # 支撑阻力位强度
        support_strength = np.mean([s.get('strength', 0) for s in factors.get('support_levels', [])]) * 100 if support_count > 0 else 0
        resistance_strength = np.mean([r.get('strength', 0) for r in factors.get('resistance_levels', [])]) * 100 if resistance_count > 0 else 0
        strength_score = (support_strength + resistance_strength) / 2
        information_score = clarity_score * 0.4 + strength_score * 0.6
        score_components.append(information_score)
        weights.append(0.25)
        # 加权平均（确保权重和为1）
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            structure_strength = sum(s * w for s, w in zip(score_components, normalized_weights))
        else:
            structure_strength = 50
        return float(structure_strength)
