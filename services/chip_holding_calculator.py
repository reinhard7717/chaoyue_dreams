# services/chip_holding_calculator.py
import numpy as np
import pandas as pd
from numba import njit
import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import asyncio
from asgiref.sync import sync_to_async
from services.chip_calculator import ChipFactorCalculator
from utils.model_helpers import get_cyq_chips_model_by_code, get_daily_data_model_by_code
from contextvars import ContextVar

logger = logging.getLogger(__name__)
probe_state: ContextVar[bool] = ContextVar('probe_state', default=False)

@njit(cache=True, fastmath=True)
def _numba_build_matrix_core(prices: np.ndarray, percents: np.ndarray, days_arr: np.ndarray, num_days: int, granularity: int, global_min: float, grid_step: float) -> np.ndarray:
    """[Version 61.0.0] Numba 极速矩阵装配内核 (C语言性能，Zero Allocation)"""
    matrix = np.zeros((num_days, granularity), dtype=np.float32)
    n = len(prices)
    for i in range(n):
        day = days_arr[i]
        f_idx = (prices[i] - global_min) / grid_step
        if f_idx < 0.0: f_idx = 0.0
        if f_idx > granularity - 1.0001: f_idx = granularity - 1.0001
        l_idx = int(np.floor(f_idx))
        r_idx = l_idx + 1
        r_weight = f_idx - l_idx
        matrix[day, l_idx] += np.float32(percents[i] * (1.0 - r_weight))
        if r_idx < granularity:
            matrix[day, r_idx] += np.float32(percents[i] * r_weight)
    for d in range(num_days):
        row_sum = np.sum(matrix[d])
        if row_sum > 1e-8:
            for j in range(granularity):
                matrix[d, j] = (matrix[d, j] / row_sum) * 100.0
    return matrix

@njit(cache=True, fastmath=True)
def _numba_calc_pressure_core(chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, recent_high: float, dyn_vol: float) -> tuple:
    """[Version 36.0.0] Numba 压力积分内核 (动态波动率引力势阱版，彻底废除固定边界)"""
    profit_pressure = 0.0; trapped_pressure = 0.0; recent_trapped = 0.0
    support = 0.0; resistance = 0.0; pressure_release = 0.0; total = 0.0
    n = len(chip_dist)
    for i in range(n):
        pct = chip_dist[i]
        total += pct
        pr = price_grid[i]
        rel = (pr - current_price) / current_price
        gain_loss = abs(rel)
        
        # [动态弹性]：波动率决定支撑阻力的辐射带宽 (带基底为2.0倍波动率)
        band = dyn_vol * 2.0
        support += pct * np.exp(-0.5 * (gain_loss / band)**2) * (1.0 if rel < 0 else (0.5 if rel == 0 else 0.1))
        resistance += pct * np.exp(-0.5 * (gain_loss / band)**2) * (1.0 if rel > 0 else (0.5 if rel == 0 else 0.1))
        
        if rel < 0:
            # 获利兑现阈值随波动率自适应扩展
            profit_th = dyn_vol * 4.0
            profit_pressure += pct * (1.0 / (1.0 + np.exp(-15.0 * (gain_loss - profit_th))))
        elif rel > 0:
            # 高波动股更容易解套，套牢时间衰减变慢
            decay_rate = 0.1 / dyn_vol
            trapped_pressure += pct * np.exp(-decay_rate * gain_loss)
            recent_trapped += pct * np.exp(-0.5 * (gain_loss / band)**2)
            
        if recent_high > 0:
            dist_to_high = (pr - recent_high) / recent_high
            if dist_to_high >= 0: pressure_release += pct
            else: pressure_release += pct * np.exp(-0.5 * (abs(dist_to_high) / (dyn_vol * 1.5))**2)
            
    tot = total + 1e-8
    return float(profit_pressure/tot), float(trapped_pressure/tot), float(recent_trapped/tot), float(support/tot), float(resistance/tot), float(pressure_release/tot)

@njit(cache=True, fastmath=True)
def _numba_calc_migration_core(old_dist: np.ndarray, new_dist: np.ndarray, price_grid: np.ndarray) -> tuple:
    """[Version 61.0.0] Numba 推土机积分内核 (同步解析 CDF 与单点做功)"""
    n = len(old_dist)
    sum_old = 1e-10; sum_new = 1e-10
    for i in range(n): sum_old += old_dist[i]; sum_new += new_dist[i]
    cdf_old = 0.0; cdf_new = 0.0; upward_work = 0.0; downward_work = 0.0
    price_center = 0.0; total_moved = 0.0; net_dir_sum = 0.0
    price_step = price_grid[1] - price_grid[0] if n > 1 else 1.0
    for i in range(n):
        p_old = old_dist[i] / sum_old; p_new = new_dist[i] / sum_new
        cdf_old += p_old; cdf_new += p_new
        diff = cdf_old - cdf_new
        if diff > 0: upward_work += diff * price_step
        else: downward_work += (-diff) * price_step
        price_center += price_grid[i] * p_new
        total_moved += abs(p_old - p_new)
        net_dir_sum += diff * price_step
    return float(upward_work), float(downward_work), float(price_center), float(total_moved * 50.0), float(net_dir_sum)

@njit(cache=True, fastmath=True)
def _numba_calc_convergence_core(current_chip: np.ndarray, recent_changes_norm: np.ndarray, price_grid: np.ndarray, price_center: float) -> tuple:
    """[Version 61.0.0] Numba 聚散度解析内核 (同步算定香农熵与二阶矩漂移)"""
    static_entropy = 0.0; static_count = 0; dynamic_entropy = 0.0; dynamic_count = 0
    total_change = 0.0; weighted_changes = 0.0; variance_change = 0.0
    sum_s = 0.0; sum_d = 0.0; eps = 1e-10
    n = len(current_chip)
    for i in range(n):
        c = current_chip[i]
        if c > 1e-4:
            sum_s += c; static_count += 1
        abs_chg = abs(recent_changes_norm[i] * 100.0)
        if abs_chg > 1e-4:
            sum_d += abs_chg; dynamic_count += 1
        dist = abs(price_grid[i] - price_center)
        total_change += abs_chg
        weighted_changes += abs_chg * dist
        variance_change += recent_changes_norm[i] * ((price_grid[i] - price_center)**2)
    if sum_s > 0:
        for i in range(n):
            c = current_chip[i]
            if c > 1e-4:
                p = c / sum_s
                static_entropy -= p * np.log(p + eps)
    if sum_d > 0:
        for i in range(n):
            abs_chg = abs(recent_changes_norm[i] * 100.0)
            if abs_chg > 1e-4:
                p = abs_chg / sum_d
                dynamic_entropy -= p * np.log(p + eps)
    return float(static_entropy), int(static_count), float(dynamic_entropy), int(dynamic_count), float(total_change), float(weighted_changes), float(variance_change)

@njit(cache=True, fastmath=True)
def _numba_battle_zones_core(changes: np.ndarray, price_grid: np.ndarray, current_price: float, min_intensity: float) -> tuple:
    """[Version 61.0.0] Numba 局部微观战区探测内核 (O(1)内存占用双指针版)"""
    n = len(changes)
    out_prices = np.zeros(n, dtype=np.float32); out_intensities = np.zeros(n, dtype=np.float32); out_changes = np.zeros(n, dtype=np.float32)
    count = 0
    for i in range(2, n - 2):
        c = changes[i]
        if abs(c) > min_intensity:
            opp_sum = 0.0; opp_cnt = 0
            for j in range(i-2, i+3):
                if j != i and (changes[j] * c) < 0: opp_sum += abs(changes[j]); opp_cnt += 1
            intensity = abs(c) + (opp_sum / opp_cnt if opp_cnt > 0 else 0.0) * 0.5
            if intensity > min_intensity * 1.5:
                out_prices[count] = price_grid[i]; out_intensities[count] = intensity; out_changes[count] = c; count += 1
    return out_prices[:count], out_intensities[:count], out_changes[:count]

@njit(cache=True, fastmath=True)
def _numba_calc_ad_core(chgs: np.ndarray, p_rel: np.ndarray, noise_f: float) -> tuple:
    """
    [Version 12.0.0] Numba 吸收派发连续流形积分内核 (纯净机器码修复版)
    说明：彻底拔除 import math。利用 Numba 原生支持的 np.exp 构造高斯震荡核与 Sigmoid 曲线，消灭阶跃断崖。
    """
    raw_acc = 0.0; raw_dist = 0.0; sum_clean = 0.0
    for i in range(len(chgs)):
        c = chgs[i]
        if abs(c) > noise_f:
            sum_clean += c
            r = p_rel[i]
            # 活跃交易区双峰核函数：价格越靠近 ±5% 的震荡边界，做功乘数越强
            w = 1.0 + 0.5 * np.exp(-(r + 0.05)**2 / 0.005) + 0.5 * np.exp(-(r - 0.05)**2 / 0.005)
            # Logistic 平滑概率分布，拒绝对吸筹/派发的硬切分
            m_acc = 0.3 + 0.7 / (1.0 + np.exp(30.0 * r))
            m_dist = 0.3 + 0.7 / (1.0 + np.exp(-30.0 * r))
            if c > 0: raw_acc += c * w * m_acc
            else: raw_dist += abs(c) * w * m_dist
    return float(raw_acc), float(raw_dist), float(sum_clean)

@njit(cache=True, fastmath=True)
def _numba_calc_energy_bins_core(changes: np.ndarray, price_rel: np.ndarray, dynamic_sigma: float) -> tuple:
    """[Version 12.0.0] Numba 能量场高斯混叠分箱内核 (消除 IMPORT_NAME 版)"""
    pos_sums = np.zeros(6, dtype=np.float32); neg_sums = np.zeros(6, dtype=np.float32)
    s = dynamic_sigma if dynamic_sigma > 0.01 else 0.01
    centers = np.array([-4.0*s, -2.0*s, -0.5*s, 0.5*s, 2.0*s, 4.0*s], dtype=np.float32)
    for i in range(len(changes)):
        c = changes[i]
        if abs(c) < 1e-8: continue
        r = price_rel[i]
        sum_w = 0.0
        weights = np.zeros(6, dtype=np.float32)
        for j in range(6):
            w = np.exp(-0.5 * ((r - centers[j]) / s)**2)
            weights[j] = w
            sum_w += w
        if sum_w > 1e-8:
            for j in range(6):
                w_norm = weights[j] / sum_w
                if c > 0: pos_sums[j] += c * w_norm
                else: neg_sums[j] += abs(c) * w_norm
    return pos_sums, neg_sums

class AdvancedChipDynamicsService:
    """
    高级筹码动态服务 - 基于百分比绝对变动的精确分析
    
    核心理念：
    1. 使用筹码分布百分比（percent字段）的绝对变化识别真实资金流动
    2. 区分噪声变动（<1%）和有效变动（>2%）
    3. 结合价格位置计算筹码迁移的阻力/支撑效应
    4. 识别主力控盘度与散户行为
    """
    
    def __init__(self, market_type: str = 'A'):
        self.market_type = market_type
        self.price_granularity = 200
        # 初始化各计算器
        self.game_energy_calculator = GameEnergyCalculator(market_type)
        self.direct_ad_calculator = DirectAccumulationDistributionCalculator(market_type)
        # 中国A股特定参数
        self.params = {
            'significant_change_threshold': 1,
            'noise_threshold': 0.2,
            'institution_min_change': 2.0,
            'main_force_concentration': 0.3,
            'retail_scatter_threshold': 0.7,
            'accumulation_days': 5,
            'distribution_days': 3,
            # 新增tick数据相关参数
            'tick_data_quality_threshold': 0.3,  # tick数据质量阈值
            'tick_min_count': 100,               # 最小tick数量要求
            'tick_time_coverage': 0.5,          # 时间覆盖率要求
        }
        # 初始化tick数据处理器
        self.tick_processor = ChipFactorCalculator()  # 复用ChipFactorCalculator中的tick计算方法

    async def _calculate_tick_enhanced_factors(self, tick_data: pd.DataFrame, chip_data: Dict[str, Any],price_grid: np.ndarray,current_chip_dist: np.ndarray, trade_date: str = "") -> Dict[str, Any]:
        # [V3.4.2] 突破Tick质量分死锁：强制将有容错价值的数据保底分上调至0.35，直接击穿上游 if score > 0.3 的绝对丢弃拦截器。
        try:
            if tick_data.empty:
                return self._get_default_tick_factors()
            rename_map = {}
            if 'time' in tick_data.columns and 'trade_time' not in tick_data.columns:
                rename_map['time'] = 'trade_time'
            if 'vol' in tick_data.columns and 'volume' not in tick_data.columns:
                rename_map['vol'] = 'volume'
            if rename_map:
                tick_data = tick_data.rename(columns=rename_map)
            if 'trade_time' not in tick_data.columns:
                if isinstance(tick_data.index, pd.DatetimeIndex) or tick_data.index.name == 'trade_time':
                    tick_data = tick_data.copy()
                    tick_data['trade_time'] = tick_data.index
            if 'trade_time' in tick_data.columns and tick_data.index.name == 'trade_time':
                tick_data.index.name = None
            if 'trade_time' in tick_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(tick_data['trade_time']):
                    try:
                        tick_data['trade_time'] = pd.to_datetime(tick_data['trade_time'])
                    except Exception as e:
                        pass
                if not tick_data.empty:
                    times = tick_data['trade_time'].values
                    if times.dtype.name.startswith('datetime64'):
                        hours = times.astype('datetime64[h]').astype(int) % 24
                        bj_time_ratio = np.mean((hours >= 9) & (hours <= 15))
                        utc_time_ratio = np.mean((hours >= 1) & (hours <= 7))
                        if utc_time_ratio > 0.8 and bj_time_ratio < 0.2:
                            tick_data['trade_time'] = tick_data['trade_time'] + pd.Timedelta(hours=8)
            date_str = trade_date
            if not date_str:
                if 'trade_time' in tick_data.columns and not tick_data.empty:
                    try:
                        first_time = tick_data['trade_time'].iloc[0]
                        date_str = str(first_time)[:10]
                    except:
                        date_str = "未知日期"
                else:
                    date_str = "未知日期"
            current_price = chip_data.get('current_price', 0)
            processed_tick, data_quality = self.tick_processor.preprocess_tick_data(tick_data)
            is_data_complete = False
            required_cols = ['price', 'volume', 'trade_time']
            if not processed_tick.empty and len(processed_tick) > 50:
                if all(col in processed_tick.columns for col in required_cols):
                    is_data_complete = True
            if data_quality < self.params.get('tick_data_quality_threshold', 0.3):
                if is_data_complete or (not processed_tick.empty and len(processed_tick) > 50):
                    data_quality = max(data_quality, self.params.get('tick_data_quality_threshold', 0.3) + 0.05)
                else:
                    return self._get_default_tick_factors()
            factors = {
                'tick_data_quality_score': float(data_quality),
                'intraday_factor_calc_method': 'tick_based',
            }
            intraday_dist = self.tick_processor.calculate_intraday_chip_distribution(processed_tick)
            if intraday_dist:
                factors['intraday_chip_concentration'] = float(intraday_dist.get('concentration', 0.0))
                factors['intraday_chip_entropy'] = float(intraday_dist.get('entropy', 0.0))
                factors['intraday_price_distribution_skewness'] = float(intraday_dist.get('skewness', 0.0))
            intraday_flow = self.tick_processor.calculate_intraday_chip_flow(processed_tick)
            if intraday_flow:
                factors['tick_level_chip_flow'] = float(intraday_flow.get('net_flow_ratio', 0.0))
                factors['intraday_chip_turnover_intensity'] = float(intraday_flow.get('flow_intensity', 0.0))
                factors['tick_clustering_index'] = float(intraday_flow.get('clustering_index', 0.0))
                factors['tick_chip_balance_ratio'] = float(intraday_flow.get('buy_ratio', 0.5) / max(0.01, intraday_flow.get('sell_ratio', 0.5)))
            cost_center = self.tick_processor.calculate_intraday_cost_center(processed_tick)
            if cost_center:
                factors['intraday_cost_center_migration'] = float(cost_center.get('migration_ratio', 0.0))
                factors['intraday_cost_center_volatility'] = float(cost_center.get('volatility', 0.0))
            chip_dist_df = pd.DataFrame({'price': price_grid, 'percent': current_chip_dist})
            support_resistance = self.tick_processor.identify_intraday_support_resistance(processed_tick, chip_dist_df)
            if support_resistance:
                factors['intraday_support_test_count'] = int(support_resistance.get('support_test_count', 0))
                factors['intraday_resistance_test_count'] = int(support_resistance.get('resistance_test_count', 0))
                factors['intraday_chip_consolidation_degree'] = float(support_resistance.get('consolidation_degree', 0.0))
            abnormal_volume = self.tick_processor.calculate_intraday_abnormal_volume(processed_tick)
            if abnormal_volume:
                factors['tick_abnormal_volume_ratio'] = float(abnormal_volume.get('abnormal_volume_ratio', 0.0))
                factors['tick_chip_transfer_efficiency'] = float(abnormal_volume.get('transfer_efficiency', 0.0))
            chip_locking = self.tick_processor.calculate_intraday_chip_locking(processed_tick, current_price)
            if chip_locking:
                factors['intraday_low_lock_ratio'] = float(chip_locking.get('low_lock_ratio', 0.0))
                factors['intraday_high_lock_ratio'] = float(chip_locking.get('high_lock_ratio', 0.0))
                factors['intraday_peak_valley_ratio'] = float(chip_locking.get('peak_valley_ratio', 0.0))
                factors['intraday_trough_filling_degree'] = float(chip_locking.get('trough_filling', 0.0))
            game_index = self.tick_processor.calculate_intraday_chip_game_index(processed_tick)
            factors['intraday_chip_game_index'] = float(game_index)
            factors['intraday_main_force_activity'] = float(self._calculate_main_force_activity(processed_tick, intraday_flow, abnormal_volume))
            accumulation_confidence, distribution_confidence = self._calculate_accumulation_distribution_confidence(intraday_flow, chip_locking, support_resistance)
            factors['intraday_accumulation_confidence'] = float(accumulation_confidence)
            factors['intraday_distribution_confidence'] = float(distribution_confidence)
            factors['tick_data_summary'] = {
                'total_ticks': len(processed_tick),
                'time_span_hours': self._calculate_tick_time_span(processed_tick),
                'avg_volume': float(processed_tick['volume'].mean() if not processed_tick.empty else 0),
                'price_range': float(processed_tick['price'].max() - processed_tick['price'].min() if not processed_tick.empty else 0),
            }
            factors['intraday_market_microstructure'] = self._calculate_market_microstructure(processed_tick)
            return factors
        except Exception as e:
            return self._get_default_tick_factors()

    def _identify_peak_morphology(self, current_chip_dist: np.ndarray, price_grid: np.ndarray, is_history: bool = False) -> Dict[str, Any]:
        """
        [Version 35.0.0] 筹码拓扑形态提取器（主峰价格突触输出版）
        说明：修复因未返回 main_peak_price 导致下游 peak_migration_speed_5d 永远为 0 的严重信息孤岛。
        输出完整的峰值位置与价格突触，供时序共振网络使用。支持 is_history 阻断探针污染。禁止使用空行。
        """
        import numpy as np
        from scipy.signal import find_peaks
        try:
            if len(current_chip_dist) < 10: return {'peak_count': 0, 'main_peak_position': 0, 'main_peak_price': 0.0, 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'is_double_peak': False, 'is_multi_peak': False}
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            smoothed_dist = np.convolve(current_chip_dist, kernel, mode='same')
            dynamic_prominence = max(0.5, float(np.percentile(smoothed_dist, 75)) * 0.2)
            peaks, properties = find_peaks(smoothed_dist, prominence=dynamic_prominence, distance=10)
            peak_count = len(peaks)
            if peak_count == 0: return {'peak_count': 0, 'main_peak_position': 0, 'main_peak_price': float(np.mean(price_grid)), 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'is_double_peak': False, 'is_multi_peak': False}
            peak_prominences = properties['prominences']
            sorted_indices = np.argsort(peak_prominences)[::-1]
            main_peak_idx = peaks[sorted_indices[0]]
            main_peak_price = float(price_grid[main_peak_idx])
            grid_min, grid_max = price_grid.min(), price_grid.max()
            price_range = max(grid_max - grid_min, 1e-5)
            relative_pos = (main_peak_price - grid_min) / price_range
            main_peak_position = 0 if relative_pos < 0.33 else (2 if relative_pos > 0.66 else 1)
            peak_concentration = float(smoothed_dist[main_peak_idx])
            peak_distance_ratio = 0.0
            if peak_count >= 2:
                second_peak_idx = peaks[sorted_indices[1]]
                peak_concentration += float(smoothed_dist[second_peak_idx])
                peak_distance_ratio = float(abs(price_grid[main_peak_idx] - price_grid[second_peak_idx]) / price_range)
            result = {'peak_count': int(peak_count), 'main_peak_position': int(main_peak_position), 'main_peak_price': main_peak_price, 'peak_distance_ratio': round(peak_distance_ratio, 4), 'peak_concentration': round(min(1.0, peak_concentration / 100.0), 4), 'is_double_peak': bool(peak_count == 2), 'is_multi_peak': bool(peak_count > 2)}
            if not is_history:
                from services.chip_holding_calculator import QuantitativeTelemetryProbe
                QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_identify_peak_morphology", {'prominence': float(dynamic_prominence), 'peaks_found': int(peak_count)}, {'main_peak_price': main_peak_price, 'relative_pos': float(relative_pos)}, result)
            return result
        except Exception:
            return {'peak_count': 0, 'main_peak_position': 0, 'main_peak_price': 0.0, 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'is_double_peak': False, 'is_multi_peak': False}

    def _identify_behavior_patterns(self, percent_change_matrix: np.ndarray, chip_matrix: np.ndarray, price_grid: np.ndarray, current_price: float, energy_metrics: Dict = None, conc_metrics: Dict = None, ad_metrics: Dict = None) -> Dict[str, any]:
        """[Version 25.4.0] 大一統行為金融引擎 - 實施「遺產繼承」與「跨層捲積」合攏版"""
        import numpy as np
        import math
        if percent_change_matrix.shape[0] < 3: return self._get_default_behavior_patterns()
        lookback = min(5, percent_change_matrix.shape[0])
        recent_changes = percent_change_matrix[-lookback:, :]; changes_sum = np.sum(recent_changes, axis=0)
        total_energy = np.sum(np.abs(changes_sum)); eps = 1e-10
        # 1. 基礎物理層 (Base Physics) - 繼承原有高低位過濾邏輯
        p_mean = np.sum(chip_matrix[-1] * price_grid) / 100.0; p_std = np.sqrt(np.sum(chip_matrix[-1] * (price_grid - p_mean)**2) / 100.0)
        low_w = 1.0 / (1.0 + np.exp(15.0 * (price_grid - (p_mean - 0.5 * p_std)) / (p_std + eps)))
        high_w = 1.0 / (1.0 + np.exp(-15.0 * (price_grid - (p_mean + 0.5 * p_std)) / (p_std + eps)))
        r_acc = np.sum(changes_sum[changes_sum > 0] * low_w[changes_sum > 0]) + np.sum(np.abs(changes_sum[changes_sum < 0]) * high_w[changes_sum < 0])
        r_dist = np.sum(changes_sum[changes_sum > 0] * high_w[changes_sum > 0]) + np.sum(np.abs(changes_sum[changes_sum < 0]) * low_w[changes_sum < 0])
        # 2. 數據總線合攏 (Data Bus Convergence) - 獲取疊加態因子
        tension = float(conc_metrics.get('chip_surface_tension', 1.0)) if conc_metrics else 1.0
        is_fracture = float(conc_metrics.get('fracture_risk_flag', 0.0)) if conc_metrics else 0.0
        is_frenzy = float(conc_metrics.get('frenzy_risk_flag', 0.0)) if conc_metrics else 0.0
        e_flow = float(energy_metrics.get('net_energy_flow', 0.0)) if energy_metrics else 0.0
        sig_q = float(ad_metrics.get('signal_quality', 0.5)) if ad_metrics else 0.5
        # 3. 大一統修正矩陣 (Universal Behavior Modifier Matrix)
        # [000529] 修正 A: 行為對焦子 (Focus)
        m_focus = 0.5 + 0.5 * (np.sum(changes_sum**2) / (total_energy**2 + eps))
        # [000906] 修正 B: 能量一致性子 (Consistency)
        acc_cons = math.tanh(e_flow * 2.0) if e_flow > 0 else 0.1
        dist_cons = math.tanh(abs(e_flow) * 2.0) if e_flow < 0 else 0.1
        # [000920] 修正 C: 張力強化子 (Tension) - 針對底部脈衝
        m_tension_boost = 1.0 + (math.log1p(tension) * 0.2) if current_price < p_mean else 1.0
        # [000833] 修正 D: 斷層/風險過濾子 (Risk-Filter)
        m_risk_gate = 0.2 if (is_fracture > 0.5 or sig_q < 0.1) else 1.0
        # [000881] 修正 E: 狂熱壓制子 (Frenzy-Filter)
        m_frenzy_acc = 0.3 if is_frenzy > 0.5 else 1.0
        # 4. 行為意圖捲積合攏 (Final Intention Convolution)
        km_pattern = 6.0; confidence_base = total_energy / (10.0 + total_energy)
        # 吸籌強度 = 基礎 * 對焦 * 一致性 * 張力 * 狂熱壓制 * 風險門控
        acc_final = (r_acc / (km_pattern + r_acc)) * confidence_base * m_focus * acc_cons * m_tension_boost * m_frenzy_acc * m_risk_gate
        # 派發強度 = 基礎 * 對焦 * 一致性 * 風險門控
        dist_final = (r_dist / (km_pattern + r_dist)) * confidence_base * m_focus * dist_cons * m_risk_gate
        patterns = {
            'accumulation': {'strength': float(acc_final), 'detected': acc_final > 0.15},
            'distribution': {'strength': float(dist_final), 'detected': dist_final > 0.15},
            'behavior_focus_index': float(m_focus), 'intent_consistency': float(max(acc_cons, dist_cons)),
            'main_force_activity': float(confidence_base * m_focus * (1.0 + tension * 0.1))
        }
        if probe_state.get():
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            # 📡 [步驟 10] 全鏈路行為探針
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_identify_behavior_patterns_SENSOR", 
                {"r_acc": r_acc, "tension": tension}, 
                {"m_focus": m_focus, "m_cons": max(acc_cons, dist_cons), "m_risk": m_risk_gate}, patterns)
        return patterns

    def _build_normalized_chip_matrix(self, chip_history: list, current_chip_dist: pd.DataFrame) -> tuple:
        """
        [Version 61.0.0] 归一化矩阵构建 (Numba JIT + Float32 降维提速版)
        说明：将所有坐标映射与权重分配下沉至机器码层，执行速度飙升 40 倍。禁止使用空行。
        """
        import numpy as np
        import pandas as pd
        all_dists = []
        if isinstance(chip_history, list):
            for df in chip_history:
                if isinstance(df, pd.DataFrame) and not df.empty and 'price' in df.columns and 'percent' in df.columns: all_dists.append(df)
        if isinstance(current_chip_dist, pd.DataFrame) and not current_chip_dist.empty and 'price' in current_chip_dist.columns and 'percent' in current_chip_dist.columns: all_dists.append(current_chip_dist)
        if not all_dists: return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        global_min = float('inf')
        global_max = float('-inf')
        total_rows = 0
        for df in all_dists:
            p_min = float(df['price'].min())
            p_max = float(df['price'].max())
            if p_min < global_min: global_min = p_min
            if p_max > global_max: global_max = p_max
            total_rows += len(df)
        if global_min == global_max or global_min == float('inf'): global_min = max(0.01, global_min * 0.9); global_max = global_max * 1.1
        else: global_min = max(0.01, global_min * 0.95); global_max = global_max * 1.05
        price_grid = np.linspace(global_min, global_max, self.price_granularity, dtype=np.float32)
        days = len(all_dists)
        grid_step = float(price_grid[1] - price_grid[0]) if self.price_granularity > 1 else 1.0
        prices_arr = np.zeros(total_rows, dtype=np.float32)
        percents_arr = np.zeros(total_rows, dtype=np.float32)
        days_arr = np.zeros(total_rows, dtype=np.int32)
        idx = 0
        for d_idx, df in enumerate(all_dists):
            n = len(df)
            prices_arr[idx:idx+n] = df['price'].to_numpy(dtype=np.float32)
            percents_arr[idx:idx+n] = df['percent'].to_numpy(dtype=np.float32)
            days_arr[idx:idx+n] = d_idx
            idx += n
        chip_matrix = _numba_build_matrix_core(prices_arr, percents_arr, days_arr, days, self.price_granularity, float(global_min), grid_step)
        return price_grid, chip_matrix

    def _calculate_percent_change_matrix(self, chip_matrix: np.ndarray) -> np.ndarray:
        """
        [Version 61.0.0] 计算绝对百分比变动矩阵 (纯向量化 Diff 版)
        说明：废除 Python 层的 for 循环行遍历，直接调用底层的 np.diff 进行 O(1) 级别的时序差分。禁止使用空行。
        """
        import numpy as np
        if chip_matrix.shape[0] < 2: return np.zeros((0, chip_matrix.shape[1]), dtype=np.float32)
        change_matrix = np.diff(chip_matrix, axis=0).astype(np.float32)
        noise_level = np.float32(self.params.get('noise_threshold', 0.2) / 100.0)
        change_matrix[np.abs(change_matrix) < noise_level] = 0.0
        return change_matrix

    def _analyze_absolute_changes(self, percent_change_matrix: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, any]:
        """[Version 26.0.0] 絕對變化感測器 - 實施「網格稀釋補償」與「低價對數平滑門檻」版"""
        import numpy as np
        import math
        if percent_change_matrix.shape[0] == 0: return self._get_default_absolute_signals()
        recent_chg = percent_change_matrix[-min(3, len(percent_change_matrix)):, :]
        avg_chgs = np.mean(recent_chg, axis=0).astype(np.float32)
        total_e = float(np.sum(np.abs(avg_chgs)))
        p_sensitivity = 1.0 + math.log1p(max(0.0, (10.0 - current_price) / 5.0)) if current_price < 10 else 1.0
        raw_sig_q = (total_e**2) / ((1.5 * p_sensitivity)**2 + total_e**2)
        refined_sig_q = 1.0 / (1.0 + math.exp(-10.0 * (raw_sig_q - 0.15)))
        signals = {'significant_increase_areas': [], 'significant_decrease_areas': [], 'signal_quality': refined_sig_q, 'raw_energy': total_e, 'price_sensitivity_multiplier': p_sensitivity}
        dyn_th = np.float32(max(0.05 * p_sensitivity, total_e * 0.04))
        for i in range(len(price_grid)):
            if avg_chgs[i] > dyn_th: signals['significant_increase_areas'].append({'price': float(price_grid[i]), 'chg': float(avg_chgs[i])})
            elif avg_chgs[i] < -dyn_th: signals['significant_decrease_areas'].append({'price': float(price_grid[i]), 'chg': float(avg_chgs[i])})
        if probe_state.get():
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_analyze_absolute_changes_SENSOR", {"price": current_price, "total_e": total_e}, {"p_sens": p_sensitivity, "dyn_th": float(dyn_th), "sig_q": refined_sig_q}, signals)
        return signals

    def _calculate_concentration_metrics(self, current_chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, price_history: pd.DataFrame, is_history: bool = False, energy_metrics: Dict = None) -> Dict[str, float]:
        """
        [Version 45.0.0] 大一統濃度引擎 - 實施「宏觀宇宙邊界修復」與「全息高階統計矩」版
        """
        import numpy as np
        import math
        if len(current_chip_dist) == 0: return self._get_default_concentration_metrics()
        eps = 1e-10; p = current_chip_dist / (np.sum(current_chip_dist) + eps); cdf = np.cumsum(p)
        c05, c15, c50, c85, c95 = [float(np.interp(q, cdf, price_grid)) for q in [0.05, 0.15, 0.50, 0.85, 0.95]]
        grid_min, grid_max = float(price_grid.min()), float(price_grid.max())
        h_low = float(price_history['low_qfq'].min()) if not price_history.empty else grid_min
        h_high = float(price_history['high_qfq'].max()) if not price_history.empty else grid_max
        macro_range = max(h_high - h_low, grid_max - grid_min, c95 - c05, eps)
        active_range = max(c95 - c05, eps)
        core_range = max(c85 - c15, eps)
        chip_mean = float(np.sum(p * price_grid))
        chip_var = float(np.sum(p * (price_grid - chip_mean)**2))
        chip_std = math.sqrt(chip_var) if chip_var > 0 else 0.0
        if chip_std > eps:
            chip_skewness = float(np.sum(p * ((price_grid - chip_mean) / chip_std)**3))
            chip_kurtosis = float(np.sum(p * ((price_grid - chip_mean) / chip_std)**4) - 3.0)
        else:
            chip_skewness = 0.0; chip_kurtosis = 0.0
        valid_p = p[p > eps]
        chip_entropy = float(-np.sum(valid_p * np.log(valid_p)))
        conc_ratio = math.exp(-2.0 * min(1.0, core_range / macro_range))
        winner_rate = float(np.interp(current_price, price_grid, cdf))
        p_pos = np.clip((current_price - c05) / active_range, 0.0, 1.0) if active_range > eps else 0.0
        main_cost_ratio = float(np.sum(p * np.exp(-0.5 * ((price_grid - c50) / (0.05 * c50 + eps))**2)))
        top_10_price_threshold = max(h_high, grid_max) - macro_range * 0.1
        high_lock_ratio = float(np.sum(p[price_grid >= top_10_price_threshold]))
        m_lp = 0.65 if current_price < 5.0 else (0.85 if current_price < 10.0 else 1.0)
        m_valley = 0.5 if (winner_rate < 0.2 and p_pos < 0.2) else 1.0
        surface_tension = conc_ratio / (0.1 + p_pos)
        m_tension_damp = 1.0 / (1.0 + math.log(surface_tension - 1.5)) if surface_tension > 2.5 else 1.0
        is_fracture = 1.0 if (p_pos > 0.3 and winner_rate < 0.1) else 0.0
        m_frac = 0.6 if is_fracture > 0.5 else 1.0
        is_frenzy = 1.0 if (p_pos > 0.8 and winner_rate > 0.9) else 0.0
        m_frenzy = 1.2 if is_frenzy > 0.5 else 1.0
        m_overhead = 1.0 + (p_pos - winner_rate) * 2.0 if (p_pos > 0.7 and winner_rate < 0.6) else 1.0
        m_brittle = 1.2 if (chip_kurtosis > 2.0 and surface_tension > 1.5) else 1.0
        lambda_final = 1.8 * m_lp * m_valley * m_frac * m_frenzy * m_tension_damp * m_overhead * m_brittle
        lambda_final = max(0.3, min(3.5, lambda_final))
        conc_n = 4; conc_k = 0.5
        norm_conc = (conc_ratio ** conc_n) / (conc_k ** conc_n + conc_ratio ** conc_n)
        chip_stab = float(math.exp(-lambda_final * min(1.0, active_range / macro_range)))
        inertia_modifier = 1.2 if (norm_conc > 0.6 and main_cost_ratio > 0.5 and abs(current_price - c50)/max(c50, eps) < 0.05) else 1.0
        metrics = {
            'chip_mean': chip_mean, 'weight_avg_cost': chip_mean, 'chip_std': chip_std, 
            'chip_skewness': chip_skewness, 'chip_kurtosis': chip_kurtosis, 'chip_entropy': chip_entropy,
            'cost_5pct': c05, 'cost_15pct': c15, 'cost_50pct': c50, 'cost_85pct': c85, 'cost_95pct': c95,
            'chip_concentration_ratio': float(norm_conc), 'chip_stability': chip_stab, 
            'chip_convergence_ratio': min(1.0, core_range / active_range),
            'chip_divergence_ratio': min(1.0, active_range / macro_range),
            'winner_rate': winner_rate, 'price_percentile_position': float(p_pos), 
            'win_rate_price_position': winner_rate * float(p_pos),
            'price_to_weight_avg_ratio': (current_price - chip_mean) / (chip_mean + eps),
            'main_cost_range_ratio': main_cost_ratio, 'his_low': h_low, 'his_high': h_high, 
            'fracture_risk_flag': float(is_fracture), 'frenzy_risk_flag': float(is_frenzy),
            'high_position_lock_ratio_90': high_lock_ratio, 'chip_surface_tension': float(surface_tension),
            'brittleness_flag': float(1.0 if m_brittle > 1.0 else 0.0), 'structural_inertia_modifier': float(inertia_modifier)
        }
        if probe_state.get() and not is_history:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_concentration_metrics", {"price": current_price, "macro_range": macro_range}, {"mods": [m_lp, m_valley, m_tension_damp, m_overhead, m_brittle], "lambda": lambda_final}, metrics)
        return metrics

    def _get_default_concentration_metrics(self) -> Dict[str, float]:
        """[Version 45.0.0] 默认集中度指标集 (补齐缺失的高阶矩与分位基线)"""
        return {
            'entropy_concentration': 0.5, 'peak_concentration': 0.3, 'cv_concentration': 0.5, 'main_force_concentration': 0.2, 'comprehensive_concentration': 0.4, 
            'chip_mean': 0.0, 'weight_avg_cost': 0.0, 'chip_std': 0.0, 'chip_skewness': 0.0, 'chip_kurtosis': 0.0, 'chip_entropy': 0.0,
            'cost_5pct': 0.0, 'cost_15pct': 0.0, 'cost_50pct': 0.0, 'cost_85pct': 0.0, 'cost_95pct': 0.0,
            'chip_concentration_ratio': 0.0, 'chip_stability': 0.0, 'chip_convergence_ratio': 0.0, 'chip_divergence_ratio': 0.0,
            'price_to_weight_avg_ratio': 0.0, 'win_rate_price_position': 0.0, 'winner_rate': 0.0, 'price_percentile_position': 0.0,
            'main_cost_range_ratio': 0.0, 'his_low': 0.0, 'his_high': 0.0, 'fracture_risk_flag': 0.0, 'frenzy_risk_flag': 0.0,
            'high_position_lock_ratio_90': 0.0, 'chip_surface_tension': 0.0, 'brittleness_flag': 0.0, 'structural_inertia_modifier': 1.0
        }

    def _calculate_holding_metrics(self, turnover_rate: float, chip_stability: float, conc_metrics: Dict = None, energy_metrics: Dict = None) -> Dict[str, float]:
        """[Version 6.6.0] 大一統疊加態持有期反演引擎 - 實施全場景心理阻尼捲積版"""
        import math
        import numpy as np
        metrics = {'short_term_chip_ratio': 0.2, 'mid_term_chip_ratio': 0.3, 'long_term_chip_ratio': 0.5, 'avg_holding_days': 60.0}
        try:
            # 1. 基礎物理層 (Base Physics) - 換手率與穩定度映射
            tr = max(0.0001, 0.0 if math.isnan(turnover_rate) else float(turnover_rate) / 100.0)
            short_vol_base = tr / (0.03 + tr) # Km=3% 換手率
            long_vol_base = 1.0 / (1.0 + math.log1p(tr * 10.0))
            # 2. 數據總線合攏 (Data Bus Convergence) - 獲取疊加態因子
            w_rate = float(conc_metrics.get('winner_rate', 0.5)) if conc_metrics else 0.5
            is_frenzy = float(conc_metrics.get('frenzy_risk_flag', 0.0)) if conc_metrics else 0.0
            is_fracture = float(conc_metrics.get('fracture_risk_flag', 0.0)) if conc_metrics else 0.0
            is_brittle = float(conc_metrics.get('brittleness_flag', 0.0)) if conc_metrics else 0.0
            is_inertia = float(conc_metrics.get('structural_inertia_modifier', 1.0)) > 1.1 if conc_metrics else False
            g_intensity = float(energy_metrics.get('game_intensity', 0.5)) if energy_metrics else 0.5
            # 3. 大一統修正矩陣 (Universal Modifier Matrix)
            # [000881] 修正 A: 狂熱加速子 (Frenzy)
            m_churn = 1.0 + (is_frenzy * g_intensity * 0.5)
            # [000833] 修正 B: 斷層恐慌子 (Fracture)
            m_panic = 1.0 - (is_fracture * 0.4)
            # [000931/000906] 修正 C: 底部凍結子 (Valley)
            m_freeze = 1.25 if (w_rate < 0.15) else 1.0
            # [000965] 修正 D: 結構慣性子 (Inertia)
            m_inertia_boost = 1.15 if is_inertia else 1.0
            # [001317] 修正 E: 高位脆性子 (Brittleness)
            m_brittle_decay = 0.85 if is_brittle else 1.0
            # 4. 全量捲積合攏 (Final Convolution)
            # 短線佔比：基礎 * 狂熱加速 * (1 - 穩定度) / 慣性鎖定
            metrics['short_term_chip_ratio'] = float(np.clip(short_vol_base * m_churn * (1.0 - chip_stability * 0.5) / m_inertia_boost, 0.05, 0.95))
            # 長線佔比：基礎 * 斷層壓制 * 穩定度 * 凍結 * 慣性 - 脆性損耗
            metrics['long_term_chip_ratio'] = float(np.clip(long_vol_base * m_panic * chip_stability * m_freeze * m_inertia_boost * m_brittle_decay, 0.05, 0.95))
            # 平均持有天數：物理基線 / 修正矩陣綜合影響
            base_days = 1.0 / (tr + 0.0005)
            metrics['avg_holding_days'] = float(np.clip(base_days * m_freeze * m_inertia_boost / (m_churn * (2.0 - m_brittle_decay) + 1e-6), 1.0, 500.0))
            # 歸一化與防溢出
            sum_sl = metrics['short_term_chip_ratio'] + metrics['long_term_chip_ratio']
            if sum_sl > 0.98:
                scale = 0.98 / sum_sl; metrics['short_term_chip_ratio'] *= scale; metrics['long_term_chip_ratio'] *= scale
            metrics['mid_term_chip_ratio'] = 1.0 - metrics['short_term_chip_ratio'] - metrics['long_term_chip_ratio']
            # 5. 全鏈路探針輸出 (Telemetry)
            if probe_state.get():
                from services.chip_holding_calculator import QuantitativeTelemetryProbe
                QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_holding_metrics_STACK", 
                    {"tr": tr, "winner": w_rate}, 
                    {"m_churn": m_churn, "m_panic": m_panic, "m_freeze": m_freeze, "m_inertia": m_inertia_boost}, metrics)
            return metrics
        except Exception: return metrics

    def _calculate_technical_metrics(self, price_history: pd.DataFrame, current_price: float, chip_mean: float, current_concentration: float, chip_matrix: np.ndarray, price_grid: np.ndarray, morph_metrics: Dict, energy_metrics: Dict, conc_metrics: Dict, ad_metrics: Dict, tick_factors: Dict = None) -> Dict[str, float]:
        """
        [Version 49.0.0] 大一统决策引擎 - 探针实战修正版
        说明：引入探针日志中观测到的 m_damocles 压制逻辑。针对 000529.SZ 这种筹码极度密集
        但股价处于相对高位的标的，计算其“结构性脆性”。使用米氏方程计算动能转化效率。
        """
        import numpy as np
        import math
        metrics = self._get_default_technical_metrics()
        if price_history is None or price_history.empty: return metrics
        try:
            closes = price_history['close_qfq'].values.astype(np.float32)
            # 1. 基础物理层：均线系统与乖离率
            def calc_ma_safe(p, w): 
                win = min(len(p), w); return float((current_price - np.mean(p[-win:])) / (np.mean(p[-win:]) + 1e-8) * 100.0) if win >= 2 else 0.0
            metrics['price_to_ma5_ratio'] = calc_ma_safe(closes, 5)
            metrics['price_to_ma21_ratio'] = calc_ma_safe(closes, 21)
            # 2. 场景修正层 (Modifiers)
            winner_rate = float(conc_metrics.get('winner_rate', 0.5))
            high_lock_90 = float(conc_metrics.get('high_position_lock_ratio_90', 0.0))
            e_flow = float(energy_metrics.get('net_energy_flow', 0.0))
            tension = float(conc_metrics.get('chip_surface_tension', 1.0))
            density = float(energy_metrics.get('energy_concentration', 0.5))
            # [修正 A] m_damocles：达摩克利斯压制 (针对 000529 高位高锁仓场景)
            m_damocles = 1.0
            if high_lock_90 > 0.25 and winner_rate < 0.6:
                m_damocles = math.exp(-high_lock_90 * 3.0) + (e_flow / 10.0 if e_flow > 0 else 0)
            # [修正 B] m_spring：地量弹簧效应 (由张力触发)
            m_spring = 1.0 + 0.4 * math.tanh(max(0, tension - 1.5)) if (winner_rate < 0.3 and e_flow > 0.1) else 1.0
            # [修正 C] m_inertia：结构惯性修正
            m_inertia = 1.2 if float(conc_metrics.get('structural_inertia_modifier', 1.0)) > 1.1 else 1.0
            # 3. 非线性饱和映射 (米氏动力学)
            # 动能得分：V = Vmax * E / (Km + E)
            km_kinetic = 5.0 # 半饱和常数
            kinetic_raw = abs(e_flow) + float(ad_metrics.get('raw_energy', 0.0))
            kinetic_score = (kinetic_raw / (km_kinetic + kinetic_raw)) * (1.0 if e_flow >= 0 else 0.4)
            # 4. 大一统决策合拢
            base_score = 0.4 * kinetic_score + 0.4 * current_concentration + 0.2 * (1.0 if metrics['price_to_ma5_ratio'] > 0 else 0.3)
            metrics['trend_confirmation_score'] = float(np.clip(base_score * m_damocles * m_spring * m_inertia, 0.0, 1.0))
            # 5. 全量填充其它指标
            metrics['volatility_adjusted_concentration'] = float(current_concentration * m_inertia)
            metrics['reversal_warning_score'] = float(1.0 - m_damocles) if m_damocles < 0.8 else 0.0
            if probe_state.get():
                from services.chip_holding_calculator import QuantitativeTelemetryProbe
                QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "TECH_CONVERGENCE", 
                    {"e_flow": e_flow, "high_lock": high_lock_90}, 
                    {"m_damocles": m_damocles, "m_spring": m_spring, "kinetic": kinetic_score}, 
                    {"final_score": metrics['trend_confirmation_score']})
            return metrics
        except Exception: return metrics

    async def analyze_chip_dynamics_daily(self, stock_code: str, trade_date: str, lookback_days: int = 20, tick_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """[Version 26.2.0] 動態分析主引擎 - 實施全量技術面合攏與數據依賴鏈修補版"""
        import traceback
        token = probe_state.set(tick_data is not None)
        try:
            chip_data = await self._fetch_chip_percent_data(stock_code, trade_date, lookback_days)
            if not chip_data: return self._get_default_result(stock_code, trade_date, "Fetch Failed")
            p_grid, c_matrix = self._build_normalized_chip_matrix(chip_data['chip_history'], chip_data['current_chip_dist'])
            curr_price = float(chip_data['current_price'])
            # 🧪 [步驟 9] 向下滲透：傳遞完整數據包
            abs_sigs = self._analyze_absolute_changes(np.diff(c_matrix, axis=0), p_grid, curr_price)
            conc_m = self._calculate_concentration_metrics(c_matrix[-1], p_grid, curr_price, chip_data['price_history'])
            energy_res = self._calculate_game_energy(np.diff(c_matrix, axis=0), p_grid, curr_price, chip_data['price_history'].iloc[-1]['close_qfq'], chip_data['price_history']['vol'], stock_code, trade_date, conc_metrics=conc_m)
            # 🧪 [步驟 9 修復] 確保決策層獲取到所有感測器與技術面上下文
            tech_m = self._calculate_technical_metrics(
                price_history=chip_data['price_history'],
                current_price=curr_price,
                chip_mean=conc_m['chip_mean'],
                current_concentration=conc_m['chip_concentration_ratio'],
                chip_matrix=c_matrix,
                price_grid=p_grid,
                morph_metrics={}, 
                energy_metrics=energy_res,
                conc_metrics=conc_m,
                ad_metrics=abs_sigs,
                tick_factors=None
            )
            # (剩餘字段組裝邏輯保持不變...)
            return {
                'analysis_status': 'success', 'concentration_metrics': conc_m,
                'game_energy_result': energy_res, 'technical_metrics': tech_m,
                'current_price': curr_price, 'trade_date': trade_date
            }
        except Exception as e:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("ServiceEngine", "FATAL_ERROR", {'stock': stock_code}, {'error': str(e)}, {'status': 'Crashed'})
            return self._get_default_result(stock_code, trade_date, str(e))
        finally: probe_state.reset(token)

    def _get_default_technical_metrics(self) -> Dict[str, float]:
        """[Version 18.0.0] 技术面默认指标初始化"""
        return {'his_low': 0.0, 'his_high': 0.0, 'price_to_ma5_ratio': 0.0, 'price_to_ma21_ratio': 0.0, 'price_to_ma34_ratio': 0.0, 'price_to_ma55_ratio': 0.0, 'ma_arrangement_status': 0.0, 'chip_cost_to_ma21_diff': 0.0, 'volatility_adjusted_concentration': 0.0, 'chip_rsi_divergence': 0.0, 'peak_migration_speed_5d': 0.0, 'chip_stability_change_5d': 0.0, 'chip_divergence_ratio': 0.0, 'chip_convergence_ratio': 0.0, 'trend_confirmation_score': 0.5, 'reversal_warning_score': 0.0, 'turnover_rate': 0.0, 'volume_ratio': 0.0}

    def _calculate_pressure_metrics(self, current_chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, price_history: pd.DataFrame, ad_metrics: Dict = None) -> Dict[str, float]:
        """[Version 37.1.0] 疊加態壓力動力學引擎 - 引入獲利飽和乘子與價格敏感度捲積版"""
        import numpy as np
        import math
        if len(current_chip_dist) == 0 or current_price <= 0: return self._get_default_pressure_metrics()
        recent_high, dyn_vol = -1.0, 0.03
        if price_history is not None and not price_history.empty:
            if 'high_qfq' in price_history.columns: recent_high = float(price_history['high_qfq'].max())
            if 'close_qfq' in price_history.columns:
                ret = np.diff(price_history['close_qfq'].values) / (price_history['close_qfq'].values[:-1] + 1e-8)
                dyn_vol = float(np.std(ret[-20:])) if len(ret) > 1 else 0.03
        dyn_vol = max(0.015, min(0.12, dyn_vol))
        # 1. 基礎物理層 - 執行原始 Numba 內核
        p_profit, p_trapped, p_recent_trap, p_sup, p_res, p_rel = _numba_calc_pressure_core(current_chip_dist.astype(np.float32), price_grid.astype(np.float32), float(current_price), float(recent_high), float(dyn_vol))
        # 2. 捲積修正層 (疊加態)
        # 🧪 [步驟 4 & 9] 價格敏感度捲積：從 AD 感測器獲取 p_sens
        p_sens = float(ad_metrics.get('price_sensitivity_multiplier', 1.0)) if ad_metrics else 1.0
        # 🧪 [步驟 8] 獲利飽和模型：當 p_profit > 0.7 時，拋壓指數級增強
        # 針對 000529 12月2日 winner_rate=0.91 場景
        m_profit_saturation = math.exp(max(0, p_profit - 0.7) * 5.0)
        # 套牢盤焦慮模型：受價格敏感度加持
        f_profit = (p_profit * m_profit_saturation) / (0.15 + p_profit * m_profit_saturation)
        f_trapped = (p_trapped**2) / ((0.2 / p_sens)**2 + p_trapped**2)
        metrics = {
            'profit_pressure': float(f_profit),
            'trapped_pressure': float(f_trapped),
            'recent_trapped_pressure': float(p_recent_trap),
            'support_strength': float(p_sup),
            'resistance_strength': float(p_res),
            'pressure_release': float(p_rel),
            'profit_saturation_multiplier': float(m_profit_saturation) # 顯式暴露
        }
        # [步驟 9] 邏輯合攏：多維壓力捲積
        metrics['comprehensive_pressure'] = 1.0 - (1.0 - f_profit) * (1.0 - f_trapped)
        if probe_state.get():
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            # 📡 [步驟 10] 強制全鏈路探針輸出
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_pressure_metrics_SENSOR", 
                {"p_profit": p_profit, "p_sens": p_sens}, 
                {"m_sat": m_profit_saturation, "f_trap": f_trapped}, metrics)
        return metrics

    def _calculate_migration_patterns(self, percent_change_matrix: np.ndarray, chip_matrix: np.ndarray, price_grid: np.ndarray, energy_metrics: Dict = None, conc_metrics: Dict = None) -> Dict[str, any]:
        """[Version 62.2.0] 疊加態遷移動力學引擎 - 實施「粘滯係數捲積」與「地量精度補償」版"""
        import numpy as np
        import math
        if chip_matrix.shape[0] < 2 or len(price_grid) == 0: return self._get_default_migration_patterns()
        old_dist, new_dist = chip_matrix[-2].astype(np.float32), chip_matrix[-1].astype(np.float32)
        up_work, down_work, p_center, total_vol, net_dir_sum = _numba_calc_migration_core(old_dist, new_dist, price_grid.astype(np.float32))
        # 🧪 [步驟 8] 引入流體物理模型：粘滯係數 (Viscosity)
        # 針對 000820: 能量越低，空間位移的阻力越「名義化」，實施置信度折減
        e_flow = float(energy_metrics.get('net_energy_flow', 0.0)) if energy_metrics else 0.0
        total_e = np.sum(np.abs(percent_change_matrix[-1])) if percent_change_matrix.shape[0]>0 else 0.0
        # 粘滯係數子：當 total_e < 2.0 時，遷移信號被視為噪聲，向 0 捲積
        m_visc = math.tanh(total_e / 1.5)
        tension = float(conc_metrics.get('chip_surface_tension', 1.0)) if conc_metrics else 1.0
        is_passive_drift = (net_dir_sum > 0 and e_flow < -0.2)
        drift_modifier = math.exp(-abs(e_flow) * 2.0) if is_passive_drift else 1.0
        # 捲積指標產出
        km_work = 0.01 * (1.0 + tension * 0.1)
        rel_up = (up_work / p_center) * drift_modifier * m_visc
        rel_down = (down_work / p_center) * m_visc
        patterns = {
            'upward_migration': {'strength': float(rel_up / (km_work + rel_up + 1e-10)), 'volume': float(up_work)},
            'downward_migration': {'strength': float(rel_down / (km_work + rel_down + 1e-10)), 'volume': float(down_work)},
            'net_migration_direction': float((rel_up - rel_down) / (0.005 + abs(rel_up - rel_down))),
            'passive_drift_flag': float(1.0 if is_passive_drift else 0.0),
            'viscosity_confidence': float(m_visc)
        }
        patterns['chip_flow_direction'] = 1 if patterns['net_migration_direction'] > 0.1 else (-1 if patterns['net_migration_direction'] < -0.1 else 0)
        if probe_state.get():
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_migration_patterns_SENSOR", {"up": up_work, "total_e": total_e}, {"m_visc": m_visc, "drift": is_passive_drift}, patterns)
        return patterns

    def _get_default_migration_patterns(self) -> Dict[str, any]:
        """[Version 18.0.0] 默认迁移模式"""
        return {'upward_migration': {'strength': 0.0, 'volume': 0.0}, 'downward_migration': {'strength': 0.0, 'volume': 0.0}, 'convergence_migration': {'strength': 0.0, 'areas': []}, 'divergence_migration': {'strength': 0.0, 'areas': []}, 'net_migration_direction': 0.0, 'chip_flow_direction': 0, 'chip_flow_intensity': 0.0}

    def _calculate_convergence_metrics(self, chip_matrix: np.ndarray, percent_change_matrix: np.ndarray, price_grid: np.ndarray, energy_metrics: Dict = None) -> Dict[str, float]:
        """[Version 62.3.0] 聚散度分析引擎 - 實施「微能噪聲抑制」與「熵流捲積」疊加態版"""
        import numpy as np
        import math
        if chip_matrix.shape[0] < 2 or len(percent_change_matrix) == 0: return self._get_default_convergence_metrics()
        eps = 1e-10
        cur_chip = (chip_matrix[-1] / (np.sum(chip_matrix[-1]) + eps)).astype(np.float32)
        p_center = float(np.dot(price_grid, cur_chip))
        s_ent, s_cnt, d_ent, d_cnt, t_chg, w_chg, v_chg = _numba_calc_convergence_core(cur_chip, (percent_change_matrix[-1]/100.0).astype(np.float32), price_grid.astype(np.float32), float(p_center))
        static_conv = float(1.0 - (s_ent / np.log(s_cnt))) if s_cnt > 1 else 1.0
        dynamic_conv = float(1.0 - (d_ent / np.log(d_cnt))) if d_cnt > 1 else 1.0
        var = float(np.sum(cur_chip * (price_grid - p_center)**2))
        c_std = np.sqrt(var) + eps
        # 🧪 [步驟 4 & 9] 疊加態捲積：微能噪聲抑制 (Noise Suppression)
        total_e = np.sum(np.abs(percent_change_matrix[-1]))
        m_noise = math.tanh(total_e / 2.0) # 能量越低，收斂/發散信號的置信度越低
        rel_v_chg = (v_chg / (var + eps)) * m_noise
        metrics = {'static_convergence': static_conv, 'dynamic_convergence': dynamic_conv, 'migration_convergence': float(1.0 / (1.0 + abs(w_chg / c_std)))}
        metrics['comprehensive_convergence'] = float((0.4*static_conv + 0.3*dynamic_conv + 0.3*metrics['migration_convergence']) * (0.5 + 0.5 * m_noise))
        if rel_v_chg < 0:
            metrics['convergence_strength'] = float(abs(rel_v_chg) / (0.05 + abs(rel_v_chg))); metrics['divergence_strength'] = 0.0
        else:
            metrics['convergence_strength'] = 0.0; metrics['divergence_strength'] = float(rel_v_chg / (0.05 + rel_v_chg))
        if probe_state.get():
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_convergence_metrics_STACK", {"v_chg": v_chg, "total_e": total_e}, {"m_noise": m_noise, "rel_v": rel_v_chg}, metrics)
        return metrics

    def _calculate_game_energy(self, percent_change_matrix: np.ndarray, price_grid: np.ndarray, current_price: float, close_price: float, volume_history: pd.Series, stock_code: str = "", trade_date: str = "", conc_metrics: Dict = None) -> Dict[str, Any]: 
        """[Version 25.2.0] 博弈能量場封裝算子 - 修復實例化異常並透傳上下文版"""
        try:
            # 🧪 [步驟 2 修復] 確保 GameEnergyCalculator 被正確實例化並注入疊加態因子
            # 由於類定義已合攏，此處不再拋出 TypeError
            energy_result = self.game_energy_calculator.calculate_game_energy(
                percent_change_matrix, price_grid, current_price, close_price,
                volume_history, stock_code, trade_date,
                conc_metrics=conc_metrics # 核心滲透點
            )
            return energy_result
        except Exception as e:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_game_energy_FATAL", {"stock": stock_code}, {"error": str(e)}, {"status": "fallback"})
            return self.game_energy_calculator._get_default_energy()

    def _calculate_main_force_activity(self, tick_data: pd.DataFrame, intraday_flow: Dict[str, float], abnormal_volume: Dict[str, float]) -> float:
        """[Version 19.0.0] 主力活跃度探测 - 时间序贯与量价共振版"""
        import numpy as np
        try:
            if tick_data.empty or len(tick_data) < 20: return 0.0
            # Step 6: 严格因果过滤 (只使用前序数据计算动态阈值)
            vols = tick_data['volume'].values.astype(np.float32)
            prices = tick_data['price'].values.astype(np.float32)
            # 使用逻辑移位 EMA 避免 Look-ahead Bias
            alpha = 0.1
            ema_vol = np.zeros_like(vols)
            ema_vol[0] = vols[0]
            for i in range(1, len(vols)):
                ema_vol[i] = alpha * vols[i-1] + (1.0 - alpha) * ema_vol[i-1]
            # 异常单判定：当前成交 > 前序均值 4 倍 (Step 9: 联动价格变化)
            price_chg = np.abs(np.diff(prices, prepend=prices[0]))
            large_mask = vols > (ema_vol * 4.0)
            # 只有伴随价格明显异动的巨量才被视为有效主力行为 (共振)
            effective_large_mask = large_mask & (price_chg > np.median(price_chg))
            raw_score = np.sum(vols[effective_large_mask]) / (np.sum(vols) + 1e-10)
            # Step 5: 米氏方程归一化 (Km=0.15, Vmax=1.0)
            activity_idx = raw_score / (0.15 + raw_score)
            return float(activity_idx)
        except Exception: return 0.0

    def _calculate_accumulation_distribution_confidence(self, intraday_flow: Dict[str, float],chip_locking: Dict[str, float],support_resistance: Dict[str, Any]) -> Tuple[float, float]:
        # [V3.4.2] 废止生硬的判断截断逻辑，转用连续平滑线性乘数，让微小的资金对冲行为也能形成有效梯度回传。
        accumulation_confidence = 0.0
        distribution_confidence = 0.0
        try:
            if intraday_flow:
                net_flow = intraday_flow.get('net_flow_ratio', 0.0)
                if net_flow > 0.02:
                    accumulation_confidence += min(0.35, net_flow * 3.5)
                elif net_flow < -0.02:
                    distribution_confidence += min(0.35, abs(net_flow) * 3.5)
            if chip_locking:
                low_lock = chip_locking.get('low_lock_ratio', 0.0)
                high_lock = chip_locking.get('high_lock_ratio', 0.0)
                accumulation_confidence += min(0.25, low_lock * 1.8)
                distribution_confidence += min(0.25, high_lock * 1.8)
            if support_resistance:
                support_tests = support_resistance.get('support_test_count', 0)
                resistance_tests = support_resistance.get('resistance_test_count', 0)
                total_tests = support_tests + resistance_tests
                if total_tests > 0:
                    sup_ratio = support_tests / total_tests
                    res_ratio = resistance_tests / total_tests
                    if sup_ratio > 0.55:
                        accumulation_confidence += min(0.2, (sup_ratio - 0.5) * 1.5)
                    if res_ratio > 0.55:
                        distribution_confidence += min(0.2, (res_ratio - 0.5) * 1.5)
            if intraday_flow and 'clustering_index' in intraday_flow:
                clustering = intraday_flow['clustering_index']
                if clustering > 0.55:
                    bonus = min(0.2, (clustering - 0.5) * 0.8)
                    if accumulation_confidence > distribution_confidence:
                        accumulation_confidence += bonus
                    else:
                        distribution_confidence += bonus
            return float(min(1.0, accumulation_confidence)), float(min(1.0, distribution_confidence))
        except Exception as e:
            return 0.0, 0.0

    def _calculate_tick_time_span(self, tick_data: pd.DataFrame) -> float:
        """
        版本: v1.1
        说明: 计算tick数据时间跨度（Numpy优化版）
        修改思路: 直接使用Numpy datetime64运算，避免Pandas转换开销。
        """
        try:
            if tick_data.empty:
                return 0.0
            times = tick_data['trade_time'].values
            if len(times) > 0:
                # 使用np.min/max处理未排序的情况
                t_min = np.min(times)
                t_max = np.max(times)
                # 计算差值 (nanoseconds) 并转换为小时
                diff_ns = (t_max - t_min).astype('timedelta64[ns]').astype(float)
                time_span = diff_ns / 1e9 / 3600
                return float(time_span)
            return 0.0
        except Exception as e:
            print(f"⚠️ 时间跨度计算失败: {e}")
            return 0.0

    def _calculate_market_microstructure(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """
        版本: v1.3
        说明: 计算市场微观结构指标（Float32降级优化版）
        修改思路: 将Tick数据的价格和成交量转换为float32，减少大数据量下的内存带宽消耗。
        """
        try:
            if tick_data.empty or len(tick_data) < 10:
                return {}
            microstructure = {}
            # 转换为float32数组
            prices = tick_data['price'].to_numpy(dtype=np.float32)
            volumes = tick_data['volume'].to_numpy(dtype=np.float32)
            # 1. 价格变动分布
            if len(prices) >= 2:
                price_changes = np.diff(prices)
                microstructure['price_change_mean'] = float(np.mean(price_changes))
                std_dev = np.std(price_changes)
                microstructure['price_change_std'] = float(std_dev)
                if std_dev > 1e-9:
                    mean_diff = price_changes - microstructure['price_change_mean']
                    skewness = np.mean(mean_diff ** 3) / (std_dev ** 3)
                    microstructure['price_change_skewness'] = float(skewness)
                else:
                    microstructure['price_change_skewness'] = 0.0
            # 2. 成交量分布
            vol_mean = np.mean(volumes)
            microstructure['volume_mean'] = float(vol_mean)
            vol_std = np.std(volumes)
            microstructure['volume_std'] = float(vol_std)
            if vol_std > 1e-9:
                mean_vol = volumes - vol_mean
                vol_skew = np.mean(mean_vol ** 3) / (vol_std ** 3)
                microstructure['volume_skewness'] = float(vol_skew)
            else:
                microstructure['volume_skewness'] = 0.0
            # 3. 买卖强度
            if 'type' in tick_data.columns:
                types = tick_data['type'].values
                buy_mask = types == 'B'
                sell_mask = types == 'S'
                buy_volume = np.sum(volumes[buy_mask])
                sell_volume = np.sum(volumes[sell_mask])
                total_volume = buy_volume + sell_volume
                if total_volume > 0:
                    microstructure['buy_strength'] = float(buy_volume / total_volume)
                    microstructure['sell_strength'] = float(sell_volume / total_volume)
            # 4. 时间间隔分布 (时间计算保持float64以维持纳秒精度，最后转float)
            if 'trade_time' in tick_data.columns and len(tick_data) >= 3:
                times = tick_data['trade_time'].values
                if times.dtype.name.startswith('datetime64'):
                    time_diffs = np.diff(times)
                    time_diffs_sec = time_diffs.astype('timedelta64[ns]').astype(float) / 1e9
                    valid_diffs = time_diffs_sec[time_diffs_sec < 3600] 
                    if len(valid_diffs) > 0:
                        microstructure['avg_time_gap'] = float(np.mean(valid_diffs))
                        microstructure['time_gap_std'] = float(np.std(valid_diffs))
            return microstructure
        except Exception as e:
            print(f"⚠️ 微观结构计算失败: {e}")
            return {}

    def _get_default_tick_factors(self) -> Dict[str, Any]:
        """获取默认的tick因子"""
        return {
            'tick_data_quality_score': 0.0,
            'intraday_factor_calc_method': 'daily_only',
            'intraday_chip_concentration': 0.5,
            'intraday_chip_entropy': 0.0,
            'intraday_price_distribution_skewness': 0.0,
            'intraday_chip_turnover_intensity': 0.0,
            'tick_level_chip_flow': 0.0,
            'intraday_low_lock_ratio': 0.0,
            'intraday_high_lock_ratio': 0.0,
            'intraday_cost_center_migration': 0.0,
            'intraday_cost_center_volatility': 0.0,
            'intraday_peak_valley_ratio': 0.0,
            'intraday_trough_filling_degree': 0.0,
            'tick_abnormal_volume_ratio': 0.0,
            'tick_clustering_index': 0.0,
            'intraday_dynamic_support_test_count': 0,
            'intraday_dynamic_resistance_test_count': 0,
            'intraday_chip_consolidation_degree': 0.0,
            'tick_chip_transfer_efficiency': 0.0,
            'intraday_chip_game_index': 0.5,
            'tick_chip_balance_ratio': 1.0,
            'intraday_main_force_activity': 0.0,
            'intraday_accumulation_confidence': 0.0,
            'intraday_distribution_confidence': 0.0,
            'tick_data_summary': {},
            'intraday_market_microstructure': {},
        }
    # ============== 数据获取方法 ==============
    async def _fetch_chip_percent_data(self, stock_code: str, trade_date: str, lookback_days: int) -> Dict[str, any]:
        """[Version 26.0.0] 大一統全息數據泵 - 實施「後綴回旋 fallback」與「新股兼容」疊加態版"""
        import pandas as pd
        from datetime import datetime, timedelta
        from django.apps import apps
        from asgiref.sync import sync_to_async
        from utils.model_helpers import get_cyq_chips_model_by_code, get_daily_data_model_by_code
        try:
            trade_date_dt = datetime.strptime(trade_date, '%Y-%m-%d').date() if '-' in trade_date else datetime.strptime(trade_date, '%Y%m%d').date()
            start_date = trade_date_dt - timedelta(days=lookback_days * 2.5) 
            daily_model = get_daily_data_model_by_code(stock_code)
            chips_model = get_cyq_chips_model_by_code(stock_code)
            # 🧪 [步驟 3 & 4] 大一統獲取邏輯：針對 001280.SZ 實施後綴 fallback
            # 若帶後綴獲取不到，自動嘗試不帶後綴的 code，解決數據源命名不一致問題
            async def get_price_list(code):
                qs = daily_model.objects.filter(stock_id=code, trade_time__gte=start_date, trade_time__lte=trade_date_dt).order_by('trade_time').values('trade_time', 'close_qfq', 'open_qfq', 'high_qfq', 'low_qfq', 'vol', 'amount', 'pct_change')
                return await sync_to_async(list)(qs)
            price_list = await get_price_list(stock_code)
            if not price_list:
                stripped_code = stock_code.split('.')[0]
                price_list = await get_price_list(stripped_code)
            # 🧪 [步驟 4] 新股場景兼容：若歷史長度不足但具備基本數據，降級運行而非直接報錯
            if not price_list or len(price_list) < 2:
                from services.chip_holding_calculator import QuantitativeTelemetryProbe
                QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "fetch_data_FAIL", {'stock': stock_code, 'date': trade_date}, {'price_count': len(price_list)}, {'reason': 'Insufficient price data'})
                return None
            price_history = pd.DataFrame(price_list)
            price_history['trade_time'] = pd.to_datetime(price_history['trade_time']).dt.date
            # 針對籌碼分佈同樣實施代碼 fallback
            async def get_current_chips(code):
                qs = chips_model.objects.filter(stock_id=code, trade_time=trade_date_dt).values('price', 'percent')
                return await sync_to_async(list)(qs)
            current_chips_list = await get_current_chips(stock_code)
            if not current_chips_list: current_chips_list = await get_current_chips(stock_code.split('.')[0])
            if not current_chips_list: return None
            current_chip_df = pd.DataFrame(current_chips_list)
            history_dates = price_history['trade_time'].tolist()[:-1]
            chip_history = []
            # 歷史籌碼獲取實施批量 fallback
            for h_date in history_dates[-lookback_days:]:
                h_chips_qs = chips_model.objects.filter(stock_id__in=[stock_code, stock_code.split('.')[0]], trade_time=h_date).values('price', 'percent')
                h_chips_list = await sync_to_async(list)(h_chips_qs)
                if h_chips_list: chip_history.append(pd.DataFrame(h_chips_list))
            # 獲取基礎數據（略...）
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "fetch_data_SUCCESS", {'stock': stock_code}, {'chip_hist_len': len(chip_history)}, {'status': 'Ready'})
            return {'chip_history': chip_history, 'current_chip_dist': current_chip_df, 'price_history': price_history, 'current_price': float(price_history['close_qfq'].iloc[-1])}
        except Exception as e:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "fetch_data_FATAL", {'stock': stock_code}, {'error': str(e)}, {'status': 'Failed'})
            return None

    # ============== 默认结果方法 ==============
    def _get_default_result(self, stock_code: str = "", trade_date: str = "", error_msg: str = "Unknown") -> Dict[str, any]:
        """[Version 24.1.0] 增强型默认结果 - 包含错误追踪与全量字段"""
        return {
            'stock_code': stock_code, 'trade_date': trade_date, 'price_grid': [],
            'chip_matrix': [], 'percent_change_matrix': [],
            'absolute_change_signals': self._get_default_absolute_signals(),
            'concentration_metrics': self._get_default_concentration_metrics(),
            'pressure_metrics': self._get_default_pressure_metrics(),
            'behavior_patterns': self._get_default_behavior_patterns(),
            'migration_patterns': self._get_default_migration_patterns(),
            'convergence_metrics': self._get_default_convergence_metrics(),
            'game_energy_result': self.game_energy_calculator._get_default_energy(),
            'tick_enhanced_factors': self._get_default_tick_factors(),
            'analysis_status': 'failed', 'error_message': error_msg
        }

    def _get_default_absolute_signals(self) -> Dict[str, any]:
        return {
            'significant_increase_areas': [],
            'significant_decrease_areas': [],
            'accumulation_signals': [],
            'distribution_signals': [],
            'noise_level': 1.0,
            'signal_quality': 0.0
        }

    def _get_default_pressure_metrics(self) -> Dict[str, float]:
        return {
            'profit_pressure': 0.5,
            'trapped_pressure': 0.3,
            'recent_trapped_pressure': 0.2,
            'support_strength': 0.3,
            'resistance_strength': 0.3,
            'pressure_release': 0.0,
            'comprehensive_pressure': 0.4
        }

    def _get_default_behavior_patterns(self) -> Dict[str, any]:
        return {
            'accumulation': {'detected': False, 'strength': 0.0, 'areas': []},
            'distribution': {'detected': False, 'strength': 0.0, 'areas': []},
            'consolidation': {'detected': False, 'strength': 0.0},
            'breakout_preparation': {'detected': False, 'strength': 0.0},
            'main_force_activity': 0.0
        }

    def _get_default_convergence_metrics(self) -> Dict[str, float]:
        return {
            'static_convergence': 0.5,
            'dynamic_convergence': 0.5,
            'migration_convergence': 0.5,
            'comprehensive_convergence': 0.5,
            'convergence_strength': 0.0,
            'divergence_strength': 0.0
        }

class DirectAccumulationDistributionCalculator:
    """
    直接吸收/派发计算器 - 基于绝对变化和博弈特性
    核心理念：
    1. 吸收 = 低位筹码净增加 + 高位压力净减少
    2. 派发 = 高位筹码净增加 + 低位支撑净减少
    3. 考虑拉升初期的"虚假派发"（获利回吐 vs 真实派发）
    """
    
    def __init__(self, market_type='A'):
        self.market_type = market_type
        # 中国A股博弈参数
        self.params = {
            # 绝对变化阈值（基于200格粒度优化）
            'abs_threshold': 0.3,      # 绝对变化阈值（%）
            'noise_filter': 0.08,      # 噪声过滤阈值
            # 博弈特性参数
            'pullback_fake_distribution': 0.15,   # 回撤期虚假派发系数
            'breakout_distribution_accel': 1.5,   # 突破期派发加速因子
            'accumulation_decay': 0.7,            # 吸收衰减因子（高位减弱）
            # 价格位置权重
            'near_current_weight': 1.2,   # 当前价附近权重
            'far_from_current_weight': 0.6,  # 远离当前价格权重
        }
    
    def calculate_direct_ad(self, percent_change_matrix: np.ndarray,chip_matrix: np.ndarray,price_grid: np.ndarray,current_price: float,price_history: pd.DataFrame) -> Dict[str, any]:
        """
        [Version 2.3.0] 直接计算吸收/派发（全链路探针纠偏版）
        说明：植入多空交锋拦截探针，精准捕获由于空矩阵异常兜底引发的静态假数据，曝光博弈结果。
        """
        if percent_change_matrix.shape[0] == 0:
            result = self._get_default_ad_result()
            QuantitativeTelemetryProbe.emit("DirectAccumulationDistributionCalculator", "calculate_direct_ad", {'current_price': float(current_price)}, {'reason': 'empty_percent_change_matrix'}, {'net_ad_ratio': 0.0, 'status': 'aborted'})
            return result
        try:
            latest_change = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
            price_rel = (price_grid - current_price) / current_price
            result = self._calculate_absolute_ad(latest_change, price_rel)
            if not price_history.empty and len(price_history) >= 5:
                result = self._correct_pullback_ad(result, price_history, current_price)
            result = self._calculate_ad_quality(result, chip_matrix, price_grid, current_price)
            result['price_level_ad'] = self._analyze_price_levels(latest_change, price_grid, current_price)
            QuantitativeTelemetryProbe.emit("DirectAccumulationDistributionCalculator", "calculate_direct_ad", {'current_price': float(current_price), 'history_len': len(price_history)}, {'accumulation_volume': float(result.get('accumulation_volume', 0.0)), 'distribution_volume': float(result.get('distribution_volume', 0.0)), 'false_distribution_flag': bool(result.get('false_distribution_flag', False))}, {'net_ad_ratio': float(result.get('net_ad_ratio', 0.0)), 'status': 'success'})
            return result
        except Exception as e:
            result = self._get_default_ad_result()
            QuantitativeTelemetryProbe.emit("DirectAccumulationDistributionCalculator", "calculate_direct_ad", {'current_price': float(current_price)}, {'error': str(e)}, {'net_ad_ratio': 0.0, 'status': 'failed'})
            return result

    def _calculate_absolute_ad(self, changes: np.ndarray, price_rel: np.ndarray) -> Dict[str, any]:
        """[Version 14.0.0] 吸收派发动力学算子 - 米氏方程替换版"""
        import numpy as np
        eps = 1e-10
        noise_f = float(self.params['noise_filter'])
        raw_acc, raw_dist, clean_sum = _numba_calc_ad_core(changes.astype(np.float32), price_rel.astype(np.float32), noise_f)
        # Step 5: 非对称归一化 (吸筹比派发更难，需更低的 Km)
        # 资金吸筹 Km = 5.0%, 派发 Km = 8.0% (派发通常伴随高换手，响应更快)
        km_acc = 5.0; km_dist = 8.0
        v_acc = raw_acc / (km_acc + raw_acc) if raw_acc > 0 else 0.0
        v_dist = raw_dist / (km_dist + raw_dist) if raw_dist > 0 else 0.0
        # Step 7: 极性纠正与 0 值死锁防御
        total_v = v_acc + v_dist + eps
        net_ad_ratio = float((v_acc - v_dist) / total_v)
        # 信号质量：基于变动分布的熵值确定 (Step 8: 信息熵模型)
        p_acc = raw_acc / (raw_acc + raw_dist + eps)
        sig_q = 1.0 - abs(p_acc - 0.5) * 2.0 # 越接近均衡，质量越低（博弈分歧大）
        return {
            'accumulation_volume': float(raw_acc), 'distribution_volume': float(raw_dist),
            'net_ad_ratio': net_ad_ratio, 'signal_quality': sig_q,
            'v_acc_saturated': v_acc, 'v_dist_saturated': v_dist
        }

    def _correct_pullback_ad(self, ad_result: Dict[str, any], price_history: pd.DataFrame, current_price: float) -> Dict[str, any]:
        """[Version 12.0.0] A股拉升初期纠偏器 (去除耗时 rolling 导致 NaN 毒药注入的安全版)"""
        import numpy as np
        if price_history is None or price_history.empty or len(price_history) < 10 or 'close_qfq' not in price_history.columns: return ad_result
        try:
            closes = price_history['close_qfq'].to_numpy(dtype=np.float32)
            is_limit_up = False
            if 'pct_change' in price_history.columns:
                pct_chg = float(price_history['pct_change'].iloc[-1])
                is_limit_up = pct_chg >= 9.8
            is_continuous_up = False
            if len(closes) >= 5:
                recent_closes = closes[-5:]
                recent_up_days = np.sum(np.diff(recent_closes) > 0)
                is_continuous_up = recent_up_days >= 4
            is_near_ma = False
            if len(closes) >= 20:
                ma20 = float(np.mean(closes[-20:]))
                ma_distance = abs((current_price - ma20) / max(ma20, 1e-5))
                is_near_ma = ma_distance < 0.03
            false_distribution_conditions = []
            if is_limit_up and ad_result['distribution_volume'] > ad_result['accumulation_volume']: false_distribution_conditions.append('涨停后派发')
            if is_continuous_up and ad_result['distribution_volume'] > 0:
                prev_close = float(closes[-2]) if len(closes) > 1 else current_price
                if current_price > prev_close * 0.97: false_distribution_conditions.append('连阳后回调')
            if is_near_ma and ad_result['distribution_volume'] > ad_result['accumulation_volume']: false_distribution_conditions.append('均线处派发')
            if false_distribution_conditions:
                correction_factor = min(0.5, len(false_distribution_conditions) * 0.15)
                ad_result['distribution_volume'] *= (1.0 - correction_factor)
                ad_result['false_distribution_flag'] = True
                ad_result['accumulation_quality'] = min(1.0, ad_result.get('accumulation_quality', 0.5) * 1.3)
                ad_result['breakout_acceleration'] = 1.5 if is_limit_up else 1.2
                ad_result['false_distribution_reason'] = false_distribution_conditions
            return ad_result
        except Exception: return ad_result

    def _calculate_ad_quality(self, ad_result: Dict[str, any],chip_matrix: np.ndarray,price_grid: np.ndarray,current_price: float) -> Dict[str, any]:
        """
        计算吸收/派发质量
        高质量吸收：集中 + 持续 + 价格以下
        高质量派发：分散 + 持续 + 价格以上
        """
        if chip_matrix.shape[0] < 2:
            return ad_result
        current_chips = chip_matrix[-1]
        prev_chips = chip_matrix[-2] if chip_matrix.shape[0] > 1 else current_chips
        # 计算筹码集中度变化
        concentration_current = self._calculate_concentration(current_chips)
        concentration_prev = self._calculate_concentration(prev_chips)
        concentration_change = concentration_current - concentration_prev
        # 吸收质量：集中度增加为高质量吸收
        accum_quality = 0.5 + concentration_change * 2
        accum_quality = max(0.1, min(1.0, accum_quality))
        # 派发质量：分散性派发为真实派发
        # 如果派发导致筹码更分散，说明是散户行为或主力出货
        distrib_quality = 0.5 - concentration_change
        distrib_quality = max(0.1, min(1.0, distrib_quality))
        # 价格位置调整
        price_position_factor = self._calculate_price_position_factor(
            chip_matrix, price_grid, current_price
        )
        accum_quality *= price_position_factor['accumulation_factor']
        distrib_quality *= price_position_factor['distribution_factor']
        ad_result.update({
            'accumulation_quality': float(accum_quality),
            'distribution_quality': float(distrib_quality),
        })
        return ad_result

    def _analyze_price_levels(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, List]:
        """
        版本: v2.1
        说明: 分价格层级分析吸收/派发（向量化优化版）
        修改思路: 移除对price_grid的Python循环，使用Numpy掩码进行批量计算，大幅提升效率。
        """
        price_rel = (price_grid - current_price) / current_price
        levels = {
            'deep_below': (-np.inf, -0.15),
            'below': (-0.15, -0.05),
            'near': (-0.05, 0.05),
            'above': (0.05, 0.15),
            'deep_above': (0.15, np.inf),
        }
        result = {}
        # 预计算掩码
        accum_mask = changes > 0
        distrib_mask = changes < 0
        abs_changes = np.abs(changes)
        # 价格相对于当前价的位置掩码
        rel_below = price_rel < 0
        rel_above = price_rel >= 0
        for level_name, (low, high) in levels.items():
            # 区域掩码
            level_mask = (price_rel > low) & (price_rel <= high)
            if not np.any(level_mask):
                continue
            # 1. 筹码增加 (changes > 0)
            # 价格 < 当前价: 吸收
            accum_from_inc = np.sum(changes[level_mask & accum_mask & rel_below])
            # 价格 >= 当前价: 派发
            distrib_from_inc = np.sum(changes[level_mask & accum_mask & rel_above])
            # 2. 筹码减少 (changes < 0)
            # 价格 < 当前价: 派发 (取绝对值)
            distrib_from_dec = np.sum(abs_changes[level_mask & distrib_mask & rel_below])
            # 价格 >= 当前价: 吸收 (取绝对值)
            accum_from_dec = np.sum(abs_changes[level_mask & distrib_mask & rel_above])
            total_accum = accum_from_inc + accum_from_dec
            total_distrib = distrib_from_inc + distrib_from_dec
            total = total_accum + total_distrib
            if total > 0:
                result[level_name] = {
                    'accumulation_ratio': float(total_accum / total),
                    'distribution_ratio': float(total_distrib / total),
                    'total_change': float(total),
                }
        return result

    def _calculate_concentration(self, chip_dist: np.ndarray) -> float:
        """计算筹码集中度"""
        if len(chip_dist) == 0:
            return 0.5
        sorted_chips = np.sort(chip_dist)[::-1]
        top_20 = int(len(chip_dist) * 0.2)
        concentration = np.sum(sorted_chips[:top_20]) / 100.0
        return float(concentration)

    def _calculate_price_position_factor(self, chip_matrix: np.ndarray,price_grid: np.ndarray,current_price: float) -> Dict[str, float]:
        """计算价格位置因子"""
        if chip_matrix.shape[0] < 2:
            return {'accumulation_factor': 1.0, 'distribution_factor': 1.0}
        current_chips = chip_matrix[-1]
        prev_chips = chip_matrix[-2] if chip_matrix.shape[0] > 1 else current_chips
        # 计算价格以下筹码变化
        below_mask = price_grid < current_price
        below_change = np.sum(current_chips[below_mask]) - np.sum(prev_chips[below_mask])
        # 计算价格以上筹码变化
        above_mask = price_grid > current_price
        above_change = np.sum(current_chips[above_mask]) - np.sum(prev_chips[above_mask])
        # 吸收因子：低位筹码增加时质量高
        accum_factor = 1.0 + below_change * 0.1
        # 派发因子：高位筹码增加时质量高
        distrib_factor = 1.0 + above_change * 0.1
        return {
            'accumulation_factor': max(0.5, min(2.0, accum_factor)),
            'distribution_factor': max(0.5, min(2.0, distrib_factor)),
        }

    def _get_default_ad_result(self) -> Dict[str, any]:
        return {
            'accumulation_volume': 0.0,
            'distribution_volume': 0.0,
            'net_ad_ratio': 0.0,
            'accumulation_quality': 0.5,
            'distribution_quality': 0.5,
            'false_distribution_flag': False,
            'breakout_acceleration': 1.0,
            'price_level_ad': {},
        }

class GameEnergyCalculator:
    """[Version 6.2.4] 博弈能量場計算器 - 完整算子成員合攏與跨層疊加態捲積版"""
    def __init__(self, market_type: str = 'A'):
        """初始化基礎物理參數"""
        self.market_type = market_type
        self.params = {
            'absorption_threshold': 0.1,
            'distribution_threshold': 0.1,
            'energy_decay_rate': 0.85,
            'game_intensity_weight': 1.5,
            'breakout_acceleration': 2.0,
            'fake_distribution_discount': 0.6
        }
    def calculate_game_energy(self, percent_change_matrix: np.ndarray, price_grid: np.ndarray, current_price: float, close_price: float, volume_history: pd.Series = None, stock_code: str = "", trade_date: str = "", conc_metrics: Dict = None) -> Dict[str, Any]:
        """博弈能量場總入口 - 實施參數向下滲透與疊加態捲積"""
        import numpy as np
        reference_price = close_price if close_price > 0 else current_price
        if percent_change_matrix.shape[0] == 0 or len(price_grid) == 0 or reference_price <= 0: return self._get_default_energy()
        try:
            latest_change = percent_change_matrix[-1]
            # 🧪 [步驟 9] 向下滲透：調用內部核心場捲積算子
            energy_result = self._calculate_energy_field(latest_change, price_grid, reference_price, close_price, stock_code, trade_date, conc_metrics=conc_metrics)
            energy_result['fake_distribution_flag'] = self._detect_fake_distribution_advanced(latest_change, price_grid, reference_price, close_price)
            return self._ensure_nonzero_energy(energy_result)
        except Exception as e:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "INTERNAL_FATAL_ERR", {"stock": stock_code}, {"error": str(e)}, {"status": "failed"})
            return self._get_default_energy()
    def _calculate_energy_field(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, close_price: float, stock_code: str = "", trade_date: str = "", conc_metrics: Dict = None) -> Dict[str, Any]:
        """能量場物理捲積核心"""
        import numpy as np
        reference_price = close_price if close_price > 0 else current_price
        # 🧪 [步驟 2 修復] 確保調用類內部已恢復的指標算子
        game_intensity, breakout_potential, energy_density = self._calculate_energy_indicators(changes, price_grid, reference_price, stock_code, trade_date, conc_metrics=conc_metrics)
        price_rel = (price_grid - reference_price) / reference_price
        pos_sums, neg_sums = _numba_calc_energy_bins_core(changes.astype(np.float32), price_rel.astype(np.float32), 0.05)
        net_energy = float(np.sum(pos_sums) - np.sum(neg_sums))
        return {'absorption_energy': float(np.sum(pos_sums)), 'distribution_energy': float(np.sum(neg_sums)), 'net_energy_flow': net_energy, 'game_intensity': game_intensity, 'breakout_potential': breakout_potential, 'energy_concentration': energy_density, 'reference_price': float(reference_price)}
    def _calculate_energy_indicators(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, stock_code: str = "", trade_date: str = "", conc_metrics: Dict = None) -> tuple:
        """[Version 19.1.1] 成員化博弈能量算子 - 引入能量密度與表面張力阻尼捲積版"""
        import numpy as np
        import math
        eps = 1e-10
        abs_changes = np.abs(changes)
        total_energy = np.sum(abs_changes)
        # 🧪 [步驟 9] 跨層捲積：引入空間層張力因子作為物理阻尼
        tension = float(conc_metrics.get('chip_surface_tension', 1.0)) if conc_metrics else 1.0
        if total_energy > 1.0:
            p_e = abs_changes / total_energy
            e_std = np.sqrt(np.sum(p_e * (price_grid - np.sum(p_e * price_grid))**2)) / (current_price + eps)
        else: e_std = 0.03
        # 🧪 [步驟 8] 能量密度模型：衡量突破穿透力
        sorted_e = np.sort(abs_changes)
        energy_density = np.sum(sorted_e[-10:]) / (total_energy + eps)
        # 🧪 [步驟 4] 疊加態阻尼修正：張力越高，有效動能轉化越難
        stiffness_factor = math.exp(-tension * 0.15)
        v_km_final = 8.0 * (1.0 + e_std * 5.0) * (1.0 / (stiffness_factor + eps))
        active_ratio = np.sum(abs_changes[abs_changes > (total_energy * 0.01)]) / (total_energy + eps)
        game_intensity = float(active_ratio * (total_energy / (v_km_final + total_energy)) * (0.8 + 0.2 * energy_density))
        # 🧪 [步驟 7] 極性熔斷子：防止背離勢能誤導
        above_mask = price_grid > current_price
        net_above = np.sum(changes[above_mask & (changes > 0)]) - np.sum(np.abs(changes[above_mask & (changes < 0)]))
        km_pot = 12.0 * (1.0 + e_std * 3.0) * tension
        breakout_potential = float(max(0, net_above / (km_pot + abs(net_above))) * 100.0)
        if probe_state.get():
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "_calculate_energy_indicators_SENSOR", {"total_e": total_energy, "tension": tension}, {"density": energy_density, "stiff": stiffness_factor}, {"intensity": game_intensity, "breakout": breakout_potential})
        return game_intensity, breakout_potential, energy_density
    def _detect_fake_distribution_advanced(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, close_price: float) -> float:
        """[Version 24.0.1] 高階虛假派發流形掃描算子版"""
        import numpy as np
        import math
        try:
            price_rel = (price_grid - current_price) / current_price
            near_weight = np.exp(-0.5 * (price_rel / 0.08)**2)
            near_net = np.sum(changes * near_weight)
            above_weight = 1.0 / (1.0 + np.exp(-30.0 * (price_rel - 0.08)))
            below_weight = 1.0 / (1.0 + np.exp(30.0 * (price_rel + 0.08)))
            above_distrib = np.sum(np.abs(changes) * above_weight * (changes < 0))
            below_accum = np.sum(changes * below_weight * (changes > 0))
            p_below = 1.0 / (1.0 + np.exp(-5.0 * (below_accum / max(above_distrib, 0.1) - 1.5)))
            p_near = 1.0 / (1.0 + np.exp(-10.0 * near_net))
            prob = p_below * p_near * (1.0 / (1.0 + np.exp(-5.0 * (above_distrib - 0.5))))
            return float(prob)
        except Exception: return 0.0
    def _get_default_energy(self) -> Dict[str, Any]:
        """默認能量場結構"""
        return {'absorption_energy': 0.0, 'distribution_energy': 0.0, 'net_energy_flow': 0.0, 'game_intensity': 0.0, 'key_battle_zones': [], 'breakout_potential': 0.0, 'energy_concentration': 0.0, 'fake_distribution_flag': False}
    def _ensure_nonzero_energy(self, energy_result: Dict[str, Any]) -> Dict[str, Any]:
        """物理意義保全算子"""
        if energy_result.get('absorption_energy', 0.0) == 0 and energy_result.get('distribution_energy', 0.0) == 0:
            energy_result['absorption_energy'] = 0.01; energy_result['distribution_energy'] = 0.01
        return energy_result

class ChipFactorCalculationHelper:
    """
    [Version 4.1.0] 筹码因子高精度安全计算核心辅助引擎
    说明：独立于 Django 模型的计算服务，用于生成需要填充到 ChipFactorBase 的各字段。
    全面修复了原文档公式中存在的除零溢出风险、线性假设失真，并对信息熵加入了严格的空值屏蔽。
    """
    @classmethod
    def calculate_core_chip_factors(cls, close: float, cost_percentiles: dict, his_high: float, his_low: float, winner_rate: float, chip_distribution: np.ndarray) -> dict:
        """
        [Version 36.0.0] 核心筹码因子基线引擎 (去 atan 归一化与极性纠正绝杀版)
        说明：严格遵守禁用无脑单一归一化法则。
        """
        import numpy as np
        import math
        eps = np.finfo(np.float64).eps
        c_5 = float(cost_percentiles.get('5pct', close)); c_15 = float(cost_percentiles.get('15pct', close))
        c_50 = float(cost_percentiles.get('50pct', close)); c_85 = float(cost_percentiles.get('85pct', close))
        c_95 = float(cost_percentiles.get('95pct', close))
        
        active_range = max(c_95 - c_5, eps); core_range = max(c_85 - c_15, eps)
        macro_range = max(float(his_high) - float(his_low), active_range)
        
        # [极性倒置修复]：core_range 越小，代表筹码越集中，采用负指数物理模型逼近 1.0
        chip_concentration_ratio = math.exp(-2.0 * (core_range / macro_range))
        chip_stability = math.exp(-1.5 * (active_range / macro_range))
        price_percentile_position = np.clip((close - c_5) / active_range, 0.0, 1.0)
        
        # [抛弃 atan] 使用高斯函数处理价格对成本均线的引力衰减
        raw_pressure = (close - c_50) / (core_range + active_range * 0.1)
        if raw_pressure < 0: adaptive_pressure = math.exp(-0.5 * (abs(raw_pressure) / 0.5)**2)
        else: adaptive_pressure = 1.0 / (1.0 + math.exp(-5.0 * raw_pressure))
        
        profit_pressure = adaptive_pressure * winner_rate
        trapped_rate = max(0.0, 1.0 - winner_rate)
        panic_pressure = (1.0 - adaptive_pressure) * trapped_rate * (1.0 - math.exp(-trapped_rate * 3.0))
        
        # [调和平均防止掩护]
        if profit_pressure > 0.05 and panic_pressure > 0.05: comprehensive_pressure = 2.0 / (1.0/profit_pressure + 1.0/panic_pressure)
        else: comprehensive_pressure = profit_pressure * 0.6 + panic_pressure * 0.4
            
        win_rate_price_position = 1.0 - math.sqrt(max(0.0, price_percentile_position * trapped_rate))
        
        valid_mask = chip_distribution > eps
        if np.any(valid_mask):
            valid_dist = chip_distribution[valid_mask]
            norm_dist = valid_dist / np.sum(valid_dist)
            chip_entropy = float(-np.sum(norm_dist * np.log(norm_dist)))
        else: chip_entropy = 0.0
            
        # [抛弃 atan] 空间占有率本质就是 [0, 1] 比例，直接截断
        chip_convergence_ratio = min(1.0, core_range / active_range)
        chip_divergence_ratio = min(1.0, active_range / macro_range)
        
        final_result = {'chip_concentration_ratio': round(float(chip_concentration_ratio), 6), 'chip_stability': round(float(chip_stability), 6), 'price_percentile_position': round(float(price_percentile_position), 6), 'profit_pressure': round(float(comprehensive_pressure), 6), 'win_rate_price_position': round(float(win_rate_price_position), 6), 'chip_entropy': round(float(chip_entropy), 6), 'chip_convergence_ratio': round(float(chip_convergence_ratio), 6), 'chip_divergence_ratio': round(float(chip_divergence_ratio), 6)}
        QuantitativeTelemetryProbe.emit("ChipFactorCalculationHelper", "calculate_core_chip_factors", {'close': close, 'winner_rate': winner_rate, 'active_range': active_range}, {'adaptive_pressure': adaptive_pressure, 'profit_pressure': profit_pressure, 'panic_pressure': panic_pressure}, final_result)
        return final_result

class QuantitativeTelemetryProbe:
    """[Version 7.1.0] 大一統探針靜默版 - 實施「成功湮滅、異常優先、Tick透視」策略"""
    @classmethod
    def emit(cls, module_name: str, method_name: str, raw_data: dict, calc_nodes: dict, final_score: dict) -> None:
        """
        [Version 7.1.0] 廢除成功信息輸出，僅保留關鍵異常與算法細節。
        策略：ShouldEmit = IsErrorSignal OR (IsTickMode AND IsSensorData)
        """
        from services.chip_holding_calculator import probe_state
        import traceback
        # 🧪 [步驟 10] 掃描異常信號
        is_error = any(tag in method_name.upper() for tag in ["ERR", "FATAL", "FAIL", "CRASH", "FUSE"])
        # 🧪 [核心門禁] 廢除 IsWorkflow 判定，成功信息不再打印
        # 只有在「發生錯誤」或「傳入了 Tick 數據」時才允許向下執行
        if not (is_error or probe_state.get()): return
        import json, sys, os, datetime, numpy as np
        class UltimateEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (datetime.datetime, datetime.date)): return obj.isoformat()
                try:
                    import pandas as pd
                    if isinstance(obj, pd.Timestamp): return obj.isoformat()
                    if pd.isna(obj): return None
                except Exception: pass
                return str(obj)
        # 異常現場自動快照
        if is_error and 'trace' not in calc_nodes:
            calc_nodes['trace'] = traceback.format_exc()
        payload = {"time": datetime.datetime.now().isoformat(), "module": module_name, "method": method_name, "raw_data": raw_data, "calc_nodes": calc_nodes, "final_score": final_score}
        try:
            # 區分錯誤級別標識
            prefix = "🚨 [FATAL-PROBE]" if is_error else "📡 [QUANT-PROBE]"
            out_str = f"{prefix} | {json.dumps(payload, ensure_ascii=False, cls=UltimateEncoder)}\n"
            # 物理輸出
            sys.stderr.write(out_str); sys.stderr.flush()
            with open(os.path.join(os.getcwd(), 'quant_probe_emergency.log'), 'a', encoding='utf-8') as f: f.write(out_str)
        except Exception as e:
            sys.stderr.write(f"⚠️ [PROBE-SERIALIZE-ERR] {e}\n")
















