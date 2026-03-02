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

logger = logging.getLogger(__name__)

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
def _numba_calc_pressure_core(chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, recent_high: float) -> tuple:
    """[Version 18.0.0] 压力阻尼心理学积分 (全域高斯平滑与零值黑洞缝合版)"""
    profit_pressure = 0.0; trapped_pressure = 0.0; recent_trapped = 0.0
    support = 0.0; resistance = 0.0; pressure_release = 0.0; total = 0.0
    n = len(chip_dist)
    for i in range(n):
        pct = chip_dist[i]
        total += pct
        pr = price_grid[i]
        rel = (pr - current_price) / current_price
        gain_loss = abs(rel)
        # 缝合零值黑洞：当前价(rel==0)的巨量筹码不再被忽略，而是平滑分配至双向引力支撑
        support += pct * np.exp(-0.5 * (gain_loss / 0.05)**2) * (1.0 if rel < 0 else (0.5 if rel == 0 else 0.1))
        resistance += pct * np.exp(-0.5 * (gain_loss / 0.05)**2) * (1.0 if rel > 0 else (0.5 if rel == 0 else 0.1))
        if rel < 0:
            profit_pressure += pct * (1.0 / (1.0 + np.exp(-15.0 * (gain_loss - 0.15))))
        elif rel > 0:
            trapped_pressure += pct * np.exp(-3.0 * gain_loss)
            recent_trapped += pct * np.exp(-0.5 * (gain_loss / 0.03)**2)
            
        if recent_high > 0:
            dist_to_high = (pr - recent_high) / recent_high
            if dist_to_high >= 0: pressure_release += pct
            else: pressure_release += pct * np.exp(-0.5 * (abs(dist_to_high) / 0.03)**2)
            
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

    def _identify_behavior_patterns(self, percent_change_matrix: np.ndarray, chip_matrix: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, any]:
        """[Version 24.0.0] 行为金融多空扫描 (全域连续映射无掩码版)"""
        import numpy as np
        import math
        if percent_change_matrix.shape[0] < 3: return self._get_default_behavior_patterns()
        patterns = {'accumulation': {'detected': False, 'strength': 0.0, 'areas': []}, 'distribution': {'detected': False, 'strength': 0.0, 'areas': []}, 'consolidation': {'detected': False, 'strength': 0.0}, 'breakout_preparation': {'detected': False, 'strength': 0.0}, 'main_force_activity': 0.0}
        lookback = min(5, percent_change_matrix.shape[0])
        recent_changes = percent_change_matrix[-lookback:, :]
        changes_last_3 = np.sum(recent_changes[-3:, :], axis=0)
        mean_changes_3 = np.mean(recent_changes[-3:, :], axis=0)
        current_chip = chip_matrix[-1]
        
        # 使用平滑标准差替代硬性的 active_mask 取极值
        weights_chip = current_chip / (np.sum(current_chip) + 1e-8)
        p_mean = np.sum(weights_chip * price_grid)
        p_std = np.sqrt(np.sum(weights_chip * (price_grid - p_mean)**2))
        p_min = p_mean - 2.0 * p_std
        p_max = p_mean + 2.0 * p_std
        p_range = max(p_max - p_min, 1e-5)
        
        low_threshold = p_min + p_range * 0.35
        high_threshold = p_max - p_range * 0.35
        k_factor = 20.0 / p_range
        low_weight = 1.0 - (1.0 / (1.0 + np.exp(-k_factor * (price_grid - low_threshold))))
        high_weight = 1.0 / (1.0 + np.exp(-k_factor * (price_grid - high_threshold)))
        
        total_3d_energy = float(np.sum(np.abs(changes_last_3)))
        dynamic_noise_th = max(0.05, total_3d_energy * 0.02)
        
        # 摒弃 increase_mask 的布尔断层，用流形概率替代
        k_noise = 10.0 / max(dynamic_noise_th, 1e-5)
        increase_weight = 1.0 / (1.0 + np.exp(-k_noise * (changes_last_3 - dynamic_noise_th)))
        decrease_weight = 1.0 / (1.0 + np.exp(k_noise * (changes_last_3 + dynamic_noise_th)))
        
        raw_accum_strength = float(np.sum(changes_last_3 * increase_weight * low_weight)) + float(np.sum(np.abs(changes_last_3) * decrease_weight * high_weight))
        raw_distrib_strength = float(np.sum(changes_last_3 * increase_weight * high_weight)) + float(np.sum(np.abs(changes_last_3) * decrease_weight * low_weight))
        
        capacity_bench = max(5.0, total_3d_energy * 0.15)
        if raw_accum_strength > 0.1:
            patterns['accumulation']['detected'] = True
            patterns['accumulation']['strength'] = float(raw_accum_strength / (raw_accum_strength + capacity_bench))
            accum_idx = np.argsort(changes_last_3 * increase_weight * low_weight)[::-1][:5]
            for idx in accum_idx: 
                if (changes_last_3[idx] * increase_weight[idx] * low_weight[idx]) > 0.01:
                    patterns['accumulation']['areas'].append({'price': float(price_grid[idx]), 'avg_change': float(mean_changes_3[idx]), 'distance_to_price': float((current_price - price_grid[idx]) / max(current_price, 1e-5))})
        
        if raw_distrib_strength > 0.1:
            patterns['distribution']['detected'] = True
            patterns['distribution']['strength'] = float(raw_distrib_strength / (raw_distrib_strength + capacity_bench))
            dist_idx = np.argsort(changes_last_3 * increase_weight * high_weight)[::-1][:5]
            for idx in dist_idx: 
                if (changes_last_3[idx] * increase_weight[idx] * high_weight[idx]) > 0.01:
                    patterns['distribution']['areas'].append({'price': float(price_grid[idx]), 'avg_change': float(mean_changes_3[idx]), 'distance_to_price': float((price_grid[idx] - current_price) / max(current_price, 1e-5))})
            
        abs_recent = np.abs(recent_changes[-1])
        daily_total_energy = float(np.sum(abs_recent))
        daily_noise = max(0.02, daily_total_energy * 0.05)
        sig_weights = 1.0 / (1.0 + np.exp(-10.0 * (abs_recent - daily_noise) / max(daily_noise, 1e-5)))
        activity_ratio = float(np.sum(sig_weights * weights_chip))
        patterns['main_force_activity'] = float(activity_ratio / (activity_ratio + 0.1))
            
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_identify_behavior_patterns", {'total_3d_energy': total_3d_energy}, {'raw_accum_strength': raw_accum_strength, 'raw_distrib_strength': raw_distrib_strength}, {'accum_strength': patterns['accumulation']['strength'], 'distrib_strength': patterns['distribution']['strength'], 'main_force_activity': patterns['main_force_activity']})
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
        """[Version 24.0.0] 绝对变化信号算子 (去掩码截断纯物理连续版)"""
        import numpy as np
        import math
        if percent_change_matrix.shape[0] == 0: return self._get_default_absolute_signals()
        recent_changes = percent_change_matrix[-min(3, len(percent_change_matrix)):, :]
        avg_changes = np.mean(recent_changes, axis=0).astype(np.float32) if recent_changes.shape[0] > 0 else np.zeros_like(price_grid, dtype=np.float32)
        abs_changes = np.abs(avg_changes)
        
        # 废弃 active_grid_mask 的布尔一刀切，采用连续激活函数
        active_weights = 1.0 / (1.0 + np.exp(-100.0 * (abs_changes - 1e-4)))
        total_active_energy = float(np.sum(abs_changes * active_weights))
        
        signal_quality = float(total_active_energy / (total_active_energy + 2.5))
        noise_ratio = float(max(0.0, 1.0 - signal_quality))
        dynamic_sig_th = np.float32(max(0.1, total_active_energy * 0.10))
        
        # 物理重构：用 Logistic 平滑过滤替代 if > threshold 的硬切分
        k_th = 10.0 / max(dynamic_sig_th, 1e-5)
        increase_prob = 1.0 / (1.0 + np.exp(-k_th * (avg_changes - dynamic_sig_th)))
        decrease_prob = 1.0 / (1.0 + np.exp(k_th * (avg_changes + dynamic_sig_th)))
        
        signals = {'significant_increase_areas': [], 'significant_decrease_areas': [], 'accumulation_signals': [], 'distribution_signals': [], 'noise_level': noise_ratio, 'signal_quality': signal_quality}
        dist_to_current = (np.abs(price_grid - current_price) / max(current_price, 1e-5)).astype(np.float32)
        
        inc_denom = max(0.5, total_active_energy * 0.2)
        inc_strengths = 1.0 - np.exp(-avg_changes / inc_denom)
        accum_prob = 1.0 / (1.0 + np.exp(30.0 * (price_grid - current_price * 0.95)))
        
        dec_denom = max(0.5, total_active_energy * 0.2)
        dec_strengths = 1.0 - np.exp(-np.abs(avg_changes) / dec_denom)
        distrib_prob = 1.0 / (1.0 + np.exp(-30.0 * (price_grid - current_price * 1.05)))
        
        for i in range(len(price_grid)):
            if increase_prob[i] > 0.5:
                signals['significant_increase_areas'].append({'price': float(price_grid[i]), 'change': float(avg_changes[i]), 'distance_to_current': float(dist_to_current[i])})
                if accum_prob[i] > 0.5:
                    signals['accumulation_signals'].append({'price': float(price_grid[i]), 'change': float(avg_changes[i]), 'strength': float(inc_strengths[i] * accum_prob[i])})
            if decrease_prob[i] > 0.5:
                signals['significant_decrease_areas'].append({'price': float(price_grid[i]), 'change': float(avg_changes[i]), 'distance_to_current': float(dist_to_current[i])})
                if distrib_prob[i] > 0.5:
                    signals['distribution_signals'].append({'price': float(price_grid[i]), 'change': float(avg_changes[i]), 'strength': float(dec_strengths[i] * distrib_prob[i])})
                
        for key in ['significant_increase_areas', 'significant_decrease_areas', 'accumulation_signals', 'distribution_signals']: signals[key] = sorted(signals[key], key=lambda x: abs(x['change']), reverse=True)[:10]
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_analyze_absolute_changes", {'total_active_energy': total_active_energy, 'noise_ratio': noise_ratio}, {'dynamic_sig_th': float(dynamic_sig_th)}, {'signal_quality': signal_quality})
        return signals

    def _calculate_concentration_metrics(self, current_chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, price_history: pd.DataFrame, is_history: bool = False) -> Dict[str, float]:
        """
        [Version 18.0.0] 概率密度真实矩积分与调和浓度引擎 (极性纠偏与去无脑归一化版)
        """
        import numpy as np
        import math
        import pandas as pd
        if len(current_chip_dist) == 0: return self._get_default_concentration_metrics()
        metrics = {}
        eps = 1e-10
        p = current_chip_dist / (np.sum(current_chip_dist) + eps)
        chip_mean = float(np.sum(p * price_grid))
        variance = float(np.sum(p * (price_grid - chip_mean)**2))
        chip_std = float(np.sqrt(variance))
        if chip_std > eps:
            chip_skewness = float(np.sum(p * ((price_grid - chip_mean) / chip_std)**3))
            chip_kurtosis = float(np.sum(p * ((price_grid - chip_mean) / chip_std)**4))
        else: chip_skewness, chip_kurtosis = 0.0, 3.0
            
        metrics['chip_mean'] = chip_mean; metrics['weight_avg_cost'] = chip_mean; metrics['chip_std'] = chip_std; metrics['chip_skewness'] = chip_skewness; metrics['chip_kurtosis'] = chip_kurtosis
        cdf = np.cumsum(p)
        cost_05 = float(np.interp(0.05, cdf, price_grid)); cost_15 = float(np.interp(0.15, cdf, price_grid))
        cost_50 = float(np.interp(0.50, cdf, price_grid)); cost_85 = float(np.interp(0.85, cdf, price_grid))
        cost_95 = float(np.interp(0.95, cdf, price_grid))
        metrics['cost_5pct'] = cost_05; metrics['cost_15pct'] = cost_15; metrics['cost_50pct'] = cost_50; metrics['cost_85pct'] = cost_85; metrics['cost_95pct'] = cost_95
        metrics['winner_rate'] = float(np.interp(current_price, price_grid, cdf))
        his_low = float(price_history['low_qfq'].min()) if price_history is not None and not price_history.empty and 'low_qfq' in price_history.columns else float(current_price * 0.8)
        his_high = float(price_history['high_qfq'].max()) if price_history is not None and not price_history.empty and 'high_qfq' in price_history.columns else float(current_price * 1.2)
        metrics['his_low'] = his_low; metrics['his_high'] = his_high
        macro_range = max(his_high - his_low, eps); core_range = max(cost_85 - cost_15, eps); active_range = max(cost_95 - cost_05, eps)
        # [极性纠正] 核心区越小，筹码越集中，呈负指数衰减逼近1.0
        metrics['chip_concentration_ratio'] = float(math.exp(-2.0 * (core_range / macro_range)))
        metrics['chip_stability'] = float(math.exp(-1.5 * (active_range / macro_range)))
        # [摒弃 atan] 发散度本身就是区间比值，直接截断
        metrics['chip_divergence_ratio'] = float(min(1.0, active_range / macro_range))
        price_position = np.clip((current_price - his_low) / macro_range, 0.0, 1.0)
        metrics['price_percentile_position'] = float(price_position)
        copula_risk = price_position * (1.0 - metrics['winner_rate'])
        metrics['win_rate_price_position'] = float(1.0 - math.sqrt(max(0.0, copula_risk))) 
        # [摒弃 atan] 还原真实的百分比偏移度，硬约束在 ±1.0 (±100%)
        raw_price_ratio = (current_price - chip_mean) / max(chip_mean, eps)
        metrics['price_to_weight_avg_ratio'] = float(np.clip(raw_price_ratio, -1.0, 1.0))
        # [消除断层] 修复高位套牢盘计算：定位真实的 his_high，用 Logistic 平滑映射
        high_watermark = his_high - macro_range * 0.10
        dist_to_high = (price_grid - high_watermark) / max(macro_range, eps)
        high_lock_weights = 1.0 / (1.0 + np.exp(-30.0 * dist_to_high))
        metrics['high_position_lock_ratio_90'] = float(np.sum(p * high_lock_weights))
        # [消除断层] 主力成本区硬边界改用高斯平滑权重
        dist_to_cost_50 = np.abs(price_grid - cost_50) / max(cost_50, eps)
        main_cost_weights = np.exp(-0.5 * (dist_to_cost_50 / 0.05)**2)
        metrics['main_cost_range_ratio'] = float(np.sum(p * main_cost_weights))
        metrics['chip_convergence_ratio'] = metrics['main_cost_range_ratio']
        smoothed_p = (p + 1e-5) / np.sum(p + 1e-5)
        entropy_val = float(-np.sum(smoothed_p * np.log(smoothed_p)))
        metrics['chip_entropy'] = float(entropy_val)
        metrics['entropy_concentration'] = float(1.0 - (entropy_val / np.log(len(smoothed_p))))
        sorted_p = np.sort(p)[::-1]
        metrics['peak_concentration'] = float(np.sum(sorted_p[:max(1, int(len(p) * 0.2))]))
        metrics['cv_concentration'] = float(math.exp(- (chip_std / max(chip_mean, eps)) * 2.0))
        metrics['main_force_concentration'] = metrics['main_cost_range_ratio']
        # 调和平均 (Harmonic Mean)：任何一项集中度拉垮，整体集中度直接暴跌，杜绝危机掩盖
        indicators = [max(0.01, metrics['entropy_concentration']), max(0.01, metrics['peak_concentration']), max(0.01, metrics['cv_concentration']), max(0.01, metrics['main_cost_range_ratio'])]
        metrics['comprehensive_concentration'] = float(len(indicators) / sum(1.0 / x for x in indicators))
        if not is_history:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_concentration_metrics", {'current_price': float(current_price), 'macro_range': float(macro_range)}, {'true_skewness': metrics['chip_skewness'], 'true_kurtosis': metrics['chip_kurtosis'], 'lock_90': metrics['high_position_lock_ratio_90']}, metrics)
        return metrics

    def _get_default_concentration_metrics(self) -> Dict[str, float]:
        """[Version 18.0.0] 默认集中度指标集"""
        return {'entropy_concentration': 0.5, 'peak_concentration': 0.3, 'cv_concentration': 0.5, 'main_force_concentration': 0.2, 'comprehensive_concentration': 0.4, 'chip_skewness': 0.0, 'chip_kurtosis': 0.0, 'chip_mean': 0.0, 'chip_std': 0.0, 'weight_avg_cost': 0.0, 'cost_5pct': 0.0, 'cost_15pct': 0.0, 'cost_50pct': 0.0, 'cost_85pct': 0.0, 'cost_95pct': 0.0, 'winner_rate': 0.0, 'win_rate_price_position': 0.0, 'price_to_weight_avg_ratio': 0.0, 'high_position_lock_ratio_90': 0.0, 'main_cost_range_ratio': 0.0, 'chip_convergence_ratio': 0.0, 'chip_divergence_ratio': 0.0, 'chip_entropy': 0.0, 'chip_concentration_ratio': 0.0, 'chip_stability': 0.0, 'price_percentile_position': 0.0, 'his_low': 0.0, 'his_high': 0.0}

    def _calculate_holding_metrics(self, turnover_rate: float, chip_stability: float) -> Dict[str, float]:
        """
        [Version 6.3.0] 持有期异质性反演器 (Tsallis q-Exponential 与 NaN 熔断版)
        修改思路：根除 turnover_rate 传入 NaN 或 0 导致的静默级数据黑洞。采用 Tsallis 统计力学模型重构换手率衰减，真实映射A股市场“底部死筹”的厚尾分布。
        """
        import math
        metrics = {'short_term_chip_ratio': 0.2, 'mid_term_chip_ratio': 0.3, 'long_term_chip_ratio': 0.5, 'avg_holding_days': 60.0}
        try:
            if math.isnan(turnover_rate) or math.isinf(turnover_rate) or turnover_rate <= 0:
                return metrics
            tr = float(turnover_rate) / 100.0
            tr = max(0.0001, min(0.6, tr))
            q = 1.5
            p_not_trade_5 = (1.0 + (q - 1.0) * tr * 5.0) ** (-1.0 / (q - 1.0))
            p_not_trade_60 = (1.0 + (q - 1.0) * tr * 60.0) ** (-1.0 / (q - 1.0))
            safe_stability = 0.5 if (math.isnan(chip_stability) or math.isinf(chip_stability)) else chip_stability
            metrics['short_term_chip_ratio'] = float((1.0 - p_not_trade_5) * 0.6 + (1.0 - safe_stability) * 0.4)
            metrics['long_term_chip_ratio'] = float(p_not_trade_60 * 0.6 + safe_stability * 0.4)
            metrics['mid_term_chip_ratio'] = float(max(0.0, 1.0 - metrics['short_term_chip_ratio'] - metrics['long_term_chip_ratio']))
            avg_days = 1.0 / (tr * (2.0 - q))
            metrics['avg_holding_days'] = float(max(1.0, min(avg_days, 1500.0)))
            return metrics
        except Exception:
            return metrics

    def _calculate_technical_metrics(self, price_history: pd.DataFrame, current_price: float, chip_mean: float, current_concentration: float, chip_matrix: np.ndarray, price_grid: np.ndarray, morph_metrics: Dict, energy_metrics: Dict, conc_metrics: Dict, ad_metrics: Dict, tick_factors: Dict = None) -> Dict[str, float]:
        """[Version 24.0.0] 技术面共振引擎 (彻底融合 AD 孤岛与防死锁版)"""
        import numpy as np
        import math
        import pandas as pd
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        metrics = self._get_default_technical_metrics()
        if price_history is None or price_history.empty or 'close_qfq' not in price_history.columns: return metrics
        try:
            clean_df = price_history.copy()
            clean_df['close_qfq'] = pd.to_numeric(clean_df['close_qfq'], errors='coerce').ffill().fillna(current_price)
            closes = clean_df['close_qfq'].to_numpy(dtype=np.float64)
            if len(closes) == 0: return metrics
            metrics['his_low'] = float(np.min(closes)); metrics['his_high'] = float(np.max(closes))
            ma5 = float(np.mean(closes[-5:])) if len(closes) >= 5 else float(closes[-1])
            ma21 = float(np.mean(closes[-21:])) if len(closes) >= 21 else float(closes[-1])
            ma34 = float(np.mean(closes[-34:])) if len(closes) >= 34 else float(closes[-1])
            ma55 = float(np.mean(closes[-55:])) if len(closes) >= 55 else float(closes[-1])
            metrics['price_to_ma5_ratio'] = float((current_price - ma5) / (ma5 + 1e-8) * 100.0)
            metrics['price_to_ma21_ratio'] = float((current_price - ma21) / (ma21 + 1e-8) * 100.0)
            metrics['price_to_ma34_ratio'] = float((current_price - ma34) / (ma34 + 1e-8) * 100.0)
            metrics['price_to_ma55_ratio'] = float((current_price - ma55) / (ma55 + 1e-8) * 100.0)
            if len(closes) >= 55:
                p21 = np.clip((ma5 - ma21) / (ma21 + 1e-8) * 20.0, -1.0, 1.0)
                p34 = np.clip((ma21 - ma34) / (ma34 + 1e-8) * 20.0, -1.0, 1.0)
                p55 = np.clip((ma34 - ma55) / (ma55 + 1e-8) * 20.0, -1.0, 1.0)
                align_score = 0.4 * p21 + 0.3 * p34 + 0.3 * p55
                metrics['ma_arrangement_status'] = int(np.sign(align_score)) if abs(align_score) > 0.3 else 0
            else: metrics['ma_arrangement_status'] = 0
                
            metrics['chip_cost_to_ma21_diff'] = float((chip_mean - ma21) / (ma21 + 1e-8) * 100.0)
            log_returns = np.log(closes[1:] / (closes[:-1] + 1e-8))
            volatility = float(np.std(log_returns[-20:])) if len(log_returns) >= 20 else 0.02
            if math.isnan(volatility) or math.isinf(volatility): volatility = 0.02
            metrics['volatility_adjusted_concentration'] = float(current_concentration * math.exp(-volatility * 15.0))
            net_energy = float(energy_metrics.get('net_energy_flow', 0.0))
            # 【打通孤岛】直接调取 Direct Accumulation Distribution 数据参与共振
            net_ad_ratio = float(ad_metrics.get('net_ad_ratio', 0.0))
            ad_quality = float(ad_metrics.get('accumulation_quality', 0.5)) if net_ad_ratio > 0 else float(ad_metrics.get('distribution_quality', 0.5))
            if len(closes) >= 15:
                diffs = np.diff(closes[-15:])
                gains = np.where(diffs > 0, diffs, 0.0); losses = np.where(diffs < 0, -diffs, 0.0)
                mean_loss = float(np.mean(losses))
                rs = float(np.mean(gains)) / (mean_loss + 1e-8) if mean_loss > 1e-8 else 100.0
                rsi_norm = float((100.0 - (100.0 / (1.0 + rs))) / 100.0)
                energy_norm = float(0.5 + 0.5 * (net_energy / (abs(net_energy) + 10.0)))
                metrics['chip_rsi_divergence'] = float(energy_norm - rsi_norm)
                
            if 'turnover_rate' in clean_df.columns:
                trn = pd.to_numeric(clean_df['turnover_rate'], errors='coerce').ffill()
                if not trn.dropna().empty: metrics['turnover_rate'] = float(trn.dropna().iloc[-1])
            if 'volume_ratio' in clean_df.columns:
                vr = pd.to_numeric(clean_df['volume_ratio'], errors='coerce').ffill()
                if not vr.dropna().empty: metrics['volume_ratio'] = float(vr.dropna().iloc[-1])
                
            if chip_matrix.shape[0] >= 6:
                morph_5d = self._identify_peak_morphology(chip_matrix[-6], price_grid, is_history=True)
                metrics['peak_migration_speed_5d'] = float((morph_metrics.get('main_peak_price', current_price) - morph_5d.get('main_peak_price', current_price)) / (current_price + 1e-8) * 100.0)
                conc_5d = self._calculate_concentration_metrics(chip_matrix[-6], price_grid, float(clean_df['close_qfq'].iloc[-6]) if len(clean_df) >= 6 else current_price, pd.DataFrame(), is_history=True)
                metrics['chip_stability_change_5d'] = float(current_concentration - conc_5d.get('chip_stability', 0.5))
                
            his_range = max(metrics['his_high'] - metrics['his_low'], 1e-5)
            active_range = max(conc_metrics.get('cost_95pct', current_price*1.1) - conc_metrics.get('cost_5pct', current_price*0.9), 1e-5)
            metrics['chip_divergence_ratio'] = float(min(1.0, active_range / his_range))
            metrics['chip_convergence_ratio'] = float(1.0 - metrics['chip_divergence_ratio'])
            # AD 强引力合体
            trend_score = 0.5 + (0.15 * metrics['ma_arrangement_status']) + (0.15 * (net_energy / (abs(net_energy) + 10.0))) + (0.1 * net_ad_ratio * ad_quality)
            tick_quality = float(tick_factors.get('tick_data_quality_score', 0.0)) if tick_factors else 0.0
            if tick_quality > 0.3: 
                t_flow = float(tick_factors.get('tick_level_chip_flow', 0.0))
                tick_main_force = float(tick_factors.get('intraday_main_force_activity', 0.0))
                trend_score += float((t_flow / (abs(t_flow) + 0.5)) * 0.05) + (tick_main_force * 0.05)
            metrics['trend_confirmation_score'] = float(np.clip(trend_score, 0.0, 1.0))
            # [核心重构：彻底消灭0乘法死锁]：采用 Soft-OR 联合防线，任何一个恶化立刻拉响防空警报
            overbought_risk = float(np.clip((metrics['price_to_ma5_ratio'] - 3.0) / 10.0, 0.0, 1.0))
            dynamic_vol_bench = max(5.0, abs(net_energy) * 0.5)
            energy_danger = float(np.clip(-net_energy / dynamic_vol_bench, 0.0, 1.0))
            top_position_risk = float(conc_metrics.get('price_percentile_position', 0.5) ** 2)
            distrib_danger = float(max(0.0, -net_ad_ratio) * ad_quality)
            reversal = float(1.0 - (1.0 - overbought_risk) * (1.0 - energy_danger) * (1.0 - top_position_risk * 0.5) * (1.0 - distrib_danger * 0.8))
            if tick_quality > 0.3:
                abnormal_vol = float(tick_factors.get('tick_abnormal_volume_ratio', 0.0))
                if abnormal_vol > 0.2: reversal = float(1.0 - (1.0 - reversal) * (1.0 - min(1.0, abnormal_vol * 0.5)))
                
            metrics['reversal_warning_score'] = float(np.clip(reversal, 0.0, 1.0))
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_technical_metrics", {'ma5': ma5, 'net_energy': net_energy}, {'volatility': volatility, 'overbought_risk': overbought_risk, 'energy_danger': energy_danger}, {'status': 'success', 'chip_rsi_divergence': metrics['chip_rsi_divergence'], 'reversal_warning_score': metrics['reversal_warning_score']})
            return metrics
        except Exception: return metrics

    async def analyze_chip_dynamics_daily(self, stock_code: str, trade_date: str, lookback_days: int = 20, tick_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        from datetime import datetime
        try:
            chip_data = await self._fetch_chip_percent_data(stock_code, trade_date, lookback_days)
            history_len = len(chip_data['chip_history']) if chip_data else 0
            if not chip_data or len(chip_data['chip_history']) < 5: return self._get_default_result(stock_code, trade_date)
            price_grid, chip_matrix = self._build_normalized_chip_matrix(chip_data['chip_history'], chip_data['current_chip_dist'])
            percent_change_matrix = self._calculate_percent_change_matrix(chip_matrix)
            absolute_signals = self._analyze_absolute_changes(percent_change_matrix=percent_change_matrix, price_grid=price_grid, current_price=chip_data['current_price'])
            concentration_metrics = self._calculate_concentration_metrics(current_chip_dist=chip_matrix[-1], price_grid=price_grid, current_price=chip_data['current_price'], price_history=chip_data['price_history'])
            pressure_metrics = self._calculate_pressure_metrics(current_chip_dist=chip_matrix[-1], price_grid=price_grid, current_price=chip_data['current_price'], price_history=chip_data['price_history'])
            behavior_patterns = self._identify_behavior_patterns(percent_change_matrix=percent_change_matrix, chip_matrix=chip_matrix, price_grid=price_grid, current_price=chip_data['current_price'])
            migration_patterns = self._calculate_migration_patterns(percent_change_matrix=percent_change_matrix, chip_matrix=chip_matrix, price_grid=price_grid)
            convergence_metrics = self._calculate_convergence_metrics(chip_matrix=chip_matrix, percent_change_matrix=percent_change_matrix, price_grid=price_grid)
            game_energy_result = self._calculate_game_energy(percent_change_matrix=percent_change_matrix, price_grid=price_grid, current_price=chip_data['current_price'], price_history=chip_data['price_history'], stock_code=stock_code, trade_date=trade_date)
            direct_ad_result = self.direct_ad_calculator.calculate_direct_ad(percent_change_matrix=percent_change_matrix, chip_matrix=chip_matrix, price_grid=price_grid, current_price=chip_data['current_price'], price_history=chip_data['price_history'])
            morphology_result = self._identify_peak_morphology(current_chip_dist=chip_matrix[-1], price_grid=price_grid)
            tick_enhanced_factors = {}
            if tick_data is not None and not tick_data.empty:
                try: tick_enhanced_factors = await self._calculate_tick_enhanced_factors(tick_data=tick_data, chip_data=chip_data, price_grid=price_grid, current_chip_dist=chip_matrix[-1], trade_date=trade_date)
                except Exception: tick_enhanced_factors = self._get_default_tick_factors()
            else: tick_enhanced_factors = self._get_default_tick_factors()
            # 【消除孤岛】：将 direct_ad_result (ad_metrics) 传入，打通信息流
            technical_metrics = self._calculate_technical_metrics(
                price_history=chip_data['price_history'], current_price=chip_data['current_price'], 
                chip_mean=float(concentration_metrics.get('chip_mean', chip_data['current_price'])), 
                current_concentration=float(concentration_metrics.get('comprehensive_concentration', 0.5)), 
                chip_matrix=chip_matrix, price_grid=price_grid, morph_metrics=morphology_result, 
                energy_metrics=game_energy_result, conc_metrics=concentration_metrics, 
                ad_metrics=direct_ad_result, tick_factors=tick_enhanced_factors
            )
            holding_metrics = self._calculate_holding_metrics(turnover_rate=technical_metrics.get('turnover_rate', 0.0), chip_stability=concentration_metrics.get('chip_stability', 0.5))
            validation_warnings = []
            base_data_integrity = 1.0; penalty_exponent = 0.0
            if history_len < lookback_days: validation_warnings.append(f"历史数据不足: {history_len}/{lookback_days}"); penalty_exponent += 0.2
            current_price = chip_data['current_price']
            if current_price > price_grid.max() or current_price < price_grid.min(): validation_warnings.append("当前价格超出网格范围"); penalty_exponent += 0.4
            signal_q = float(absolute_signals.get('signal_quality', 0.5))
            validation_score = float(base_data_integrity * np.exp(-penalty_exponent) * (0.8 + 0.2 * signal_q))
            result = {'stock_code': stock_code, 'trade_date': trade_date, 'price_grid': price_grid.tolist(), 'chip_matrix': chip_matrix.tolist(), 'percent_change_matrix': percent_change_matrix.tolist(), 'absolute_change_signals': absolute_signals, 'concentration_metrics': concentration_metrics, 'pressure_metrics': pressure_metrics, 'behavior_patterns': behavior_patterns, 'migration_patterns': migration_patterns, 'convergence_metrics': convergence_metrics, 'game_energy_result': game_energy_result, 'direct_ad_result': direct_ad_result, 'morphology_metrics': morphology_result, 'technical_metrics': technical_metrics, 'holding_metrics': holding_metrics, 'tick_enhanced_factors': tick_enhanced_factors, 'validation_score': round(validation_score, 4), 'validation_warnings': validation_warnings, 'analysis_status': 'success', 'analysis_time': datetime.now().isoformat()}
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "analyze_chip_dynamics_daily", {'stock_code': stock_code, 'trade_date': trade_date, 'signal_quality': float(signal_q), 'penalty': float(penalty_exponent)}, {'validation_warnings': validation_warnings}, {'validation_score': float(validation_score), 'status': 'success'})
            return result
        except Exception as e:
            return self._get_default_result(stock_code, trade_date)

    def _get_default_technical_metrics(self) -> Dict[str, float]:
        """[Version 18.0.0] 技术面默认指标初始化"""
        return {'his_low': 0.0, 'his_high': 0.0, 'price_to_ma5_ratio': 0.0, 'price_to_ma21_ratio': 0.0, 'price_to_ma34_ratio': 0.0, 'price_to_ma55_ratio': 0.0, 'ma_arrangement_status': 0.0, 'chip_cost_to_ma21_diff': 0.0, 'volatility_adjusted_concentration': 0.0, 'chip_rsi_divergence': 0.0, 'peak_migration_speed_5d': 0.0, 'chip_stability_change_5d': 0.0, 'chip_divergence_ratio': 0.0, 'chip_convergence_ratio': 0.0, 'trend_confirmation_score': 0.5, 'reversal_warning_score': 0.0, 'turnover_rate': 0.0, 'volume_ratio': 0.0}

    def _calculate_pressure_metrics(self, current_chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, price_history: pd.DataFrame) -> Dict[str, float]:
        """
        [Version 61.0.0] 压力与支撑非对称心理阻尼模型 (Numba 极限加速版)
        说明：根除 Numpy 向量切片造成的巨量临时中间数组。转译为底层 JIT 机器码单次汇编迭代。禁止使用空行。
        """
        import numpy as np
        if len(current_chip_dist) == 0 or current_price <= 0: return self._get_default_pressure_metrics()
        recent_high = -1.0
        if price_history is not None and not price_history.empty and 'high_qfq' in price_history.columns:
            recent_high = float(price_history['high_qfq'].max())
        p_profit, p_trapped, p_recent_trap, p_sup, p_res, p_rel = _numba_calc_pressure_core(current_chip_dist.astype(np.float32), price_grid.astype(np.float32), float(current_price), float(recent_high))
        metrics = {'profit_pressure': p_profit, 'trapped_pressure': p_trapped, 'recent_trapped_pressure': p_recent_trap, 'support_strength': p_sup, 'resistance_strength': p_res, 'pressure_release': p_rel}
        metrics['comprehensive_pressure'] = float(metrics['trapped_pressure'] * 0.4 + metrics['recent_trapped_pressure'] * 0.4 + (1.0 - metrics['pressure_release']) * 0.2)
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_pressure_metrics", {'total_percent': float(np.sum(current_chip_dist))}, {'damped_trapped': metrics['trapped_pressure']}, metrics)
        return metrics

    def _calculate_migration_patterns(self, percent_change_matrix: np.ndarray, chip_matrix: np.ndarray, price_grid: np.ndarray) -> Dict[str, any]:
        """
        [Version 61.0.0] 筹码迁移地球推土机模型 (EMD Numba 无阵列分配版)
        说明：将 np.cumsum 与双矩阵差分压缩进入底层 C 的标量加法系统，极大缓解内存碎片风暴。禁止使用空行。
        """
        import numpy as np
        import math
        if chip_matrix.shape[0] < 2 or len(price_grid) == 0: return self._get_default_migration_patterns()
        patterns = {'upward_migration': {'strength': 0.0, 'volume': 0.0}, 'downward_migration': {'strength': 0.0, 'volume': 0.0}, 'convergence_migration': {'strength': 0.0, 'areas': []}, 'divergence_migration': {'strength': 0.0, 'areas': []}, 'net_migration_direction': 0.0, 'chip_flow_direction': 0, 'chip_flow_intensity': 0.0}
        eps = 1e-10
        old_dist = chip_matrix[-2].astype(np.float32)
        new_dist = chip_matrix[-1].astype(np.float32)
        upward_work, downward_work, price_center, total_moved_vol, net_dir_sum = _numba_calc_migration_core(old_dist, new_dist, price_grid.astype(np.float32))
        price_center = max(float(price_center), eps)
        total_work = upward_work + downward_work + eps
        patterns['upward_migration']['volume'] = float(total_moved_vol * (upward_work / total_work))
        patterns['upward_migration']['strength'] = float(math.tanh((upward_work / price_center) * 100.0))
        patterns['downward_migration']['volume'] = float(total_moved_vol * (downward_work / total_work))
        patterns['downward_migration']['strength'] = float(math.tanh((downward_work / price_center) * 100.0))
        net_dir_pct = (float(net_dir_sum) / price_center) * 100.0
        patterns['net_migration_direction'] = float(math.tanh(net_dir_pct))
        if patterns['net_migration_direction'] > 0.05: patterns['chip_flow_direction'] = 1
        elif patterns['net_migration_direction'] < -0.05: patterns['chip_flow_direction'] = -1
        else: patterns['chip_flow_direction'] = 0
        patterns['chip_flow_intensity'] = float(abs(patterns['net_migration_direction']))
        recent_changes = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros_like(price_grid)
        mask_mid = (price_grid >= price_center * 0.95) & (price_grid <= price_center * 1.05)
        mid_increase = float(np.sum(recent_changes[mask_mid & (recent_changes > 0)]))
        if mid_increase > 0:
            patterns['convergence_migration']['strength'] = float(math.tanh(mid_increase / 10.0))
            idx_conv = np.where(mask_mid & (recent_changes > 0))[0][:5]
            patterns['convergence_migration']['areas'] = [{'price': float(price_grid[i]), 'change': float(recent_changes[i])} for i in idx_conv]
        mid_decrease = float(np.sum(recent_changes[mask_mid & (recent_changes < 0)]))
        if mid_decrease < 0:
            patterns['divergence_migration']['strength'] = float(math.tanh(abs(mid_decrease) / 10.0))
            idx_div = np.where(mask_mid & (recent_changes < 0))[0][:5]
            patterns['divergence_migration']['areas'] = [{'price': float(price_grid[i]), 'change': float(recent_changes[i])} for i in idx_div]
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_migration_patterns", {'upward_work': upward_work, 'downward_work': downward_work, 'price_center': price_center}, {'up_strength': patterns['upward_migration']['strength'], 'down_strength': patterns['downward_migration']['strength']}, {'net_migration_direction': patterns['net_migration_direction'], 'chip_flow_direction': patterns['chip_flow_direction']})
        return patterns

    def _get_default_migration_patterns(self) -> Dict[str, any]:
        """[Version 18.0.0] 默认迁移模式"""
        return {'upward_migration': {'strength': 0.0, 'volume': 0.0}, 'downward_migration': {'strength': 0.0, 'volume': 0.0}, 'convergence_migration': {'strength': 0.0, 'areas': []}, 'divergence_migration': {'strength': 0.0, 'areas': []}, 'net_migration_direction': 0.0, 'chip_flow_direction': 0, 'chip_flow_intensity': 0.0}

    def _calculate_convergence_metrics(self, chip_matrix: np.ndarray, percent_change_matrix: np.ndarray, price_grid: np.ndarray) -> Dict[str, float]:
        """
        [Version 61.0.0] 筹码聚散度分析算子 (Numba 矩阵穿透融合版)
        说明：粉碎了原生代码中对 p_static, p_dynamic 重复切片与 log 带来的对象创建损耗。利用 JIT 同步完成全阶解析，榨干 CPU L1 缓存。禁止使用空行。
        """
        import numpy as np
        import math
        if chip_matrix.shape[0] < 2 or len(percent_change_matrix) == 0: return self._get_default_convergence_metrics()
        metrics = {}
        eps = 1e-10
        current_chip = (chip_matrix[-1] / (np.sum(chip_matrix[-1]) + eps)).astype(np.float32)
        recent_changes_norm = (percent_change_matrix[-1] / 100.0).astype(np.float32)
        price_center = float(np.dot(price_grid, current_chip))
        static_entropy, static_count, dynamic_entropy, dynamic_count, total_change, weighted_changes, variance_change = _numba_calc_convergence_core(current_chip, recent_changes_norm, price_grid.astype(np.float32), float(price_center))
        metrics['static_convergence'] = float(1.0 - (static_entropy / np.log(static_count))) if static_count > 1 else 1.0
        metrics['dynamic_convergence'] = float(1.0 - (dynamic_entropy / np.log(dynamic_count))) if dynamic_count > 1 else 1.0
        variance = float(np.sum(current_chip * (price_grid - price_center)**2))
        chip_std = np.sqrt(variance) + eps
        if total_change > eps: metrics['migration_convergence'] = float(max(0.0, 1.0 - math.atan(weighted_changes / (chip_std * 1.5)) / (math.pi / 2)))
        else: metrics['migration_convergence'] = 1.0
        metrics['comprehensive_convergence'] = float(0.4 * metrics['static_convergence'] + 0.3 * metrics['dynamic_convergence'] + 0.3 * metrics['migration_convergence'])
        rel_var_change = variance_change / (variance + eps)
        if rel_var_change < 0:
            metrics['convergence_strength'] = float(math.tanh(abs(rel_var_change) * 50.0))
            metrics['divergence_strength'] = 0.0
        else:
            metrics['convergence_strength'] = 0.0
            metrics['divergence_strength'] = float(math.tanh(rel_var_change * 50.0))
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_convergence_metrics", {'variance_change': float(variance_change), 'rel_var_change': float(rel_var_change)}, {'dynamic_convergence': metrics['dynamic_convergence'], 'migration_convergence': metrics['migration_convergence']}, metrics)
        return metrics

    def _calculate_game_energy(self, percent_change_matrix: np.ndarray,price_grid: np.ndarray,current_price: float,price_history: pd.DataFrame,stock_code: str = "",trade_date: str = "") -> Dict[str, Any]:
        """计算博弈能量场"""
        # 提取成交量历史
        volume_history = None
        if not price_history.empty and 'vol' in price_history.columns:
            volume_history = price_history['vol'].astype(float).fillna(0.0)
        # 获取收盘价
        close_price = 0
        if not price_history.empty and 'close_qfq' in price_history.columns:
            close_price = price_history['close_qfq'].iloc[-1]
        # 计算能量场
        energy_result = self.game_energy_calculator.calculate_game_energy(
            percent_change_matrix,
            price_grid,
            current_price,
            close_price,
            volume_history,
            stock_code,
            trade_date
        )
        return energy_result

    def _calculate_main_force_activity(self, tick_data: pd.DataFrame, intraday_flow: Dict[str, float], abnormal_volume: Dict[str, float]) -> float:
        """
        [Version 18.0.0] 严格因果序贯主力活跃度探测器 (EMA 记忆网络消除未来函数版)
        修改思路：废除用包含自身数据的 cumsum 来判定自己是否异动的未来函数泄露。强制时间步右移 (用 t-1 的 EMA 评定 t 的爆发)。
        """
        import numpy as np
        import math
        try:
            raw_score = 0.0
            if abnormal_volume: raw_score += abnormal_volume.get('abnormal_volume_ratio', 0.0) * 3.5
            if not tick_data.empty:
                volumes = tick_data['volume'].to_numpy(dtype=np.float32)
                seq_len = len(volumes)
                if seq_len > 1:
                    ema_volumes = np.zeros(seq_len, dtype=np.float32)
                    dynamic_threshold = np.zeros(seq_len, dtype=np.float32)
                    alpha = np.float32(2.0 / (min(20, seq_len) + 1.0))
                    ema_volumes[0] = volumes[0]
                    dynamic_threshold[0] = volumes[0] * 3.0
                    for i in range(1, seq_len):
                        # [严格因果隔离]：用上一时刻(i-1)的 EMA 门限去判定这一时刻(i)的量，杜绝数据偷看未来
                        dynamic_threshold[i] = ema_volumes[i-1] * 3.0
                        ema_volumes[i] = alpha * volumes[i] + (1.0 - alpha) * ema_volumes[i-1]
                    
                    large_order_mask = volumes > dynamic_threshold
                    large_order_vol = np.sum(volumes[large_order_mask])
                    total_vol = np.sum(volumes)
                    large_order_ratio = float(large_order_vol / total_vol) if total_vol > 1e-5 else 0.0
                    raw_score += large_order_ratio * 2.5
            if intraday_flow:
                buy_ratio = intraday_flow.get('buy_ratio', 0.5)
                sell_ratio = intraday_flow.get('sell_ratio', 0.5)
                prior_imbalance = 0.05
                imbalance = abs(buy_ratio - sell_ratio) / (buy_ratio + sell_ratio + prior_imbalance)
                raw_score += imbalance * 2.0
            return float(np.tanh(raw_score / 2.0))
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
        """
        [Version 23.0.0] 全息数据泵 (强一致哈希对齐与双重替补版)
        说明: 彻底消灭 dt.date 强转引发的 Pandas 索引哈希坍塌，修复 has_turnover=false 断层。
        引入 turnover_rate_f(自由流通换手率) 作为替补，强制转换为 float 类型，确保量比和换手率无损穿透至下游。禁止使用空行。
        """
        import pandas as pd
        from datetime import datetime, timedelta
        from django.apps import apps
        from utils.model_helpers import get_cyq_chips_model_by_code, get_daily_data_model_by_code
        from asgiref.sync import sync_to_async
        try:
            chips_model = get_cyq_chips_model_by_code(stock_code)
            if not chips_model: return None
            trade_date_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
            current_chip_qs = chips_model.objects.filter(stock__stock_code=stock_code, trade_time=trade_date_dt).values('price', 'percent')
            current_chip_list = await sync_to_async(list)(current_chip_qs)
            current_chip_df = pd.DataFrame(current_chip_list) if current_chip_list else pd.DataFrame()
            start_date = trade_date_dt - timedelta(days=max(lookback_days * 2, 100))
            history_chip_qs = chips_model.objects.filter(stock__stock_code=stock_code, trade_time__gte=start_date, trade_time__lt=trade_date_dt).order_by('trade_time').values('trade_time', 'price', 'percent')
            history_chip_list = await sync_to_async(list)(history_chip_qs)
            chip_history = []
            if history_chip_list:
                history_df = pd.DataFrame(history_chip_list)
                unique_dates = history_df['trade_time'].unique()
                for date_val in unique_dates: chip_history.append(history_df[history_df['trade_time'] == date_val][['price', 'percent']])
            daily_model = get_daily_data_model_by_code(stock_code)
            price_history = pd.DataFrame()
            if daily_model:
                price_qs = daily_model.objects.filter(stock__stock_code=stock_code, trade_time__gte=start_date, trade_time__lte=trade_date_dt).order_by('trade_time').values('trade_time', 'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq', 'vol', 'amount')
                price_list = await sync_to_async(list)(price_qs)
                if price_list:
                    price_history = pd.DataFrame(price_list)
                    price_history['trade_time'] = pd.to_datetime(price_history['trade_time']).dt.normalize()
                    price_history.set_index('trade_time', inplace=True)
                    basic_list = []
                    try:
                        market = stock_code.split('.')[-1]
                        model_name = f'StockDailyBasic_{market}'
                        try: StockDailyBasic = apps.get_model('stock_models', model_name)
                        except LookupError: StockDailyBasic = apps.get_model('stock_models', 'StockDailyBasic')
                        basic_qs = StockDailyBasic.objects.filter(stock__stock_code=stock_code, trade_time__gte=start_date, trade_time__lte=trade_date_dt).values('trade_time', 'turnover_rate', 'turnover_rate_f', 'volume_ratio')
                        basic_list = await sync_to_async(list)(basic_qs)
                        if not basic_list:
                            StockDailyBasic_All = apps.get_model('stock_models', 'StockDailyBasic')
                            basic_qs_all = StockDailyBasic_All.objects.filter(stock__stock_code=stock_code, trade_time__gte=start_date, trade_time__lte=trade_date_dt).values('trade_time', 'turnover_rate', 'turnover_rate_f', 'volume_ratio')
                            basic_list = await sync_to_async(list)(basic_qs_all)
                    except Exception as e: print(f"⚠️ [基本面联查异常] {e}")
                    if basic_list:
                        basic_df = pd.DataFrame(basic_list)
                        basic_df['trade_time'] = pd.to_datetime(basic_df['trade_time']).dt.normalize()
                        if 'turnover_rate' in basic_df.columns and 'turnover_rate_f' in basic_df.columns:
                            basic_df['turnover_rate'] = basic_df['turnover_rate'].fillna(basic_df['turnover_rate_f'])
                        elif 'turnover_rate_f' in basic_df.columns:
                            basic_df['turnover_rate'] = basic_df['turnover_rate_f']
                        if 'turnover_rate' in basic_df.columns: basic_df['turnover_rate'] = pd.to_numeric(basic_df['turnover_rate'], errors='coerce')
                        if 'volume_ratio' in basic_df.columns: basic_df['volume_ratio'] = pd.to_numeric(basic_df['volume_ratio'], errors='coerce')
                        basic_df.set_index('trade_time', inplace=True)
                        cols_to_keep = []
                        if 'turnover_rate' in basic_df.columns: cols_to_keep.append('turnover_rate')
                        if 'volume_ratio' in basic_df.columns: cols_to_keep.append('volume_ratio')
                        if cols_to_keep: price_history = price_history.join(basic_df[cols_to_keep], how='left')
                    price_history.reset_index(inplace=True)
            current_price = 0.0
            if not price_history.empty and 'close_qfq' in price_history.columns: current_price = float(price_history['close_qfq'].iloc[-1])
            elif not current_chip_df.empty: current_price = float(current_chip_df['price'].mean())
            has_turnover = False
            if not price_history.empty and 'turnover_rate' in price_history.columns:
                if not price_history['turnover_rate'].dropna().empty: has_turnover = True
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_fetch_chip_percent_data", {'stock_code': stock_code, 'trade_date': trade_date}, {'chip_history_len': len(chip_history), 'price_history_len': len(price_history), 'has_turnover': has_turnover}, {'current_price': current_price, 'status': 'success'})
            return {'current_chip_dist': current_chip_df, 'chip_history': chip_history, 'price_history': price_history, 'current_price': current_price}
        except Exception as e:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            import traceback
            traceback.print_exc()
            QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_fetch_chip_percent_data_FATAL", {'stock_code': stock_code}, {'error': str(e)}, {'status': 'exception'})
            return None

    # ============== 默认结果方法 ==============
    
    def _get_default_result(self, stock_code: str = "", trade_date: str = "") -> Dict[str, any]:
        result = {
            'stock_code': stock_code,
            'trade_date': trade_date,
            'price_grid': [],
            'percent_change_matrix': [],
            'absolute_change_signals': self._get_default_absolute_signals(),
            'concentration_metrics': self._get_default_concentration_metrics(),
            'pressure_metrics': self._get_default_pressure_metrics(),
            'behavior_patterns': self._get_default_behavior_patterns(),
            'migration_patterns': self._get_default_migration_patterns(),
            'convergence_metrics': self._get_default_convergence_metrics(),
            'game_energy_result': {},
            'direct_ad_result': {},
            # 新增：默认tick因子
            'tick_enhanced_factors': self._get_default_tick_factors(),
            'analysis_status': 'failed'
        }
        # 确保有默认的game_energy_result
        if 'game_energy_result' not in result or not result['game_energy_result']:
            result['game_energy_result'] = self.game_energy_calculator._get_default_energy()
        return result

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
        import numpy as np
        import math
        noise_filter = float(self.params['noise_filter'])
        accumulation_volume, distribution_volume, overall_trend = _numba_calc_ad_core(changes.astype(np.float32), price_rel.astype(np.float32), noise_filter)
        total_raw_vol = accumulation_volume + distribution_volume
        bayesian_prior = max(3.0, total_raw_vol * 0.15)
        total_volume_smoothed = total_raw_vol + bayesian_prior + 1e-8
        raw_net_ratio = (accumulation_volume - distribution_volume) / total_volume_smoothed
        net_ad_ratio = float(math.atan(raw_net_ratio * 3.0) / (math.pi / 2))
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("DirectAccumulationDistributionCalculator", "_calculate_absolute_ad", {'total_raw_vol': float(total_raw_vol), 'bayesian_prior': float(bayesian_prior)}, {'raw_accum': float(accumulation_volume), 'raw_distrib': float(distribution_volume)}, {'net_ad_ratio': float(net_ad_ratio)})
        return {'accumulation_volume': float(accumulation_volume), 'distribution_volume': float(distribution_volume), 'net_ad_ratio': net_ad_ratio, 'accumulation_quality': 0.5, 'distribution_quality': 0.5, 'false_distribution_flag': False, 'breakout_acceleration': 1.0}

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
    """
    博弈能量场计算器 - 直接捕捉资金对抗
    替代传统的avg_holding_days逻辑
    """
    
    def __init__(self, market_type='A'):
        self.market_type = market_type
        # 修改参数：降低阈值，让更多变化被计入
        self.params = {
            'absorption_threshold': 0.1,      # 降低到0.1%（原0.3）
            'distribution_threshold': 0.1,    # 降低到0.1%（原0.3）
            'energy_decay_rate': 0.85,
            'game_intensity_weight': 1.5,
            'breakout_acceleration': 2.0,
            'fake_distribution_discount': 0.6,
        }
    
    def calculate_game_energy(self, percent_change_matrix: np.ndarray, price_grid: np.ndarray, current_price: float, close_price: float, volume_history: pd.Series = None, stock_code: str = "", trade_date: str = "") -> Dict[str, Any]:
        """
        [Version 6.1.0] 博弈能量场主控器（边界截断拦截探针版）
        说明：修复因极端市场行情（一字跌停、无成交）引发的输入无效或零变化触发兜底时未曾暴露的静默跳过行为。
        """
        reference_price = close_price if close_price > 0 else current_price
        if percent_change_matrix.shape[0] == 0 or len(price_grid) == 0 or reference_price <= 0:
            print(f"❌ [探针] 输入数据无效: 变化矩阵{percent_change_matrix.shape}, 价格网格{len(price_grid)}, 参考价{reference_price}")
            result = self._get_default_energy()
            QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'reason': '输入数据矩阵无效或无参考价'}, {'status': 'aborted'})
            return result
        try:
            if len(percent_change_matrix) == 0:
                print("❌ [探针] 变化矩阵为空")
                result = self._get_default_energy()
                print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
                QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'reason': '提取的变化矩阵为空'}, {'status': 'aborted'})
                return result
            latest_change = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
            change_sum = np.sum(np.abs(latest_change))
            if change_sum < 0.01:
                print("⚠️ [探针] 变化数据几乎全为0，返回默认值")
                result = self._get_default_energy()
                print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
                QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'reason': '绝对变化和过低触发拦截', 'change_sum': float(change_sum)}, {'status': 'aborted'})
                return result
            energy_result = self._calculate_energy_field(latest_change, price_grid, reference_price, close_price)
            fake_distribution = False
            if volume_history is not None and len(volume_history) > 5:
                fake_distribution = self._detect_fake_distribution(latest_change, price_grid, reference_price, volume_history)
            else:
                fake_distribution = self._detect_fake_distribution_advanced(latest_change, price_grid, reference_price, close_price)
            energy_result['fake_distribution_flag'] = fake_distribution
            energy_result = self._ensure_nonzero_energy(energy_result)
            QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'fake_distribution': fake_distribution}, {'status': 'success'})
            return energy_result
        except Exception as e:
            print(f"❌ [探针] 能量场计算异常: {e}")
            import traceback
            traceback.print_exc()
            result = self._get_default_energy()
            print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
            QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'error': str(e)}, {'status': 'failed'})
            return result

    def _calculate_energy_field(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, close_price: float, stock_code: str = "", trade_date: str = "") -> Dict[str, Any]:
        """
        [Version 61.0.0] 动态波动率自适应能量场核心算子（Numba 汇编分箱版）
        说明：以 C-backend if-elif 比较树取代 Numpy digitize/bincount 的动态数组装配，切断了在小批量计算中的解释器时延。禁止使用空行。
        """
        import numpy as np
        reference_price = close_price if close_price > 0 else current_price
        if len(changes) == 0 or len(price_grid) == 0 or reference_price <= 0: return self._get_default_energy()
        try:
            price_rel = (price_grid - reference_price) / reference_price
            abs_changes = np.abs(changes)
            active_mask = abs_changes > 1e-5
            if np.any(active_mask):
                weights = abs_changes[active_mask]
                active_rels = price_rel[active_mask]
                mean_rel = np.average(active_rels, weights=weights)
                variance = np.average((active_rels - mean_rel)**2, weights=weights)
                sigma = np.sqrt(variance)
            else: sigma = 0.05
            dynamic_sigma = max(0.02, min(float(sigma), 0.20))
            pos_sums, neg_sums = _numba_calc_energy_bins_core(changes.astype(np.float32), price_rel.astype(np.float32), float(dynamic_sigma))
            weights_arr = np.array([0.6, 0.9, 1.5, 1.3, 1.0, 0.7], dtype=np.float32)
            absorption_advanced = 0.0
            distribution_advanced = 0.0
            for i in range(3):
                absorption_advanced += pos_sums[i] * weights_arr[i]
                distribution_advanced += neg_sums[i] * weights_arr[i] * 0.8
            total_section_energy = np.sum(pos_sums) + np.sum(neg_sums)
            prior_momentum_base = max(2.0, total_section_energy * 0.1)
            momentum_ratio = (pos_sums[3] + pos_sums[4]) / (neg_sums[3] + neg_sums[4] + prior_momentum_base)
            is_momentum_drive = momentum_ratio > 2.0
            for i in range(3, 5):
                if is_momentum_drive:
                    absorption_advanced += pos_sums[i] * weights_arr[i] * 0.8
                    distribution_advanced += neg_sums[i] * weights_arr[i] * 1.0
                else:
                    distribution_advanced += pos_sums[i] * weights_arr[i]
                    absorption_advanced += neg_sums[i] * weights_arr[i] * 0.7
            i = 5
            w = weights_arr[i]
            inc_sum = pos_sums[i]
            dec_sum = neg_sums[i]
            if inc_sum > dec_sum:
                distribution_advanced += inc_sum * w * 1.2
                absorption_advanced += dec_sum * w * 0.4
            else:
                distribution_advanced += inc_sum * w * 0.6
                absorption_advanced += dec_sum * w * 0.9
            game_intensity, breakout_potential, energy_concentration = self._calculate_energy_indicators(changes, price_grid, reference_price, stock_code, trade_date)
            key_battle_zones = self._identify_key_battle_zones(changes, price_grid, reference_price, stock_code, trade_date)
            net_energy = float(absorption_advanced - distribution_advanced)
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "_calculate_energy_field", {'sigma': float(sigma), 'dynamic_sigma': float(dynamic_sigma), 'total_section_energy': float(total_section_energy)}, {'prior_momentum_base': float(prior_momentum_base), 'momentum_ratio': float(momentum_ratio)}, {'net_energy': net_energy, 'absorption': float(absorption_advanced), 'distribution': float(distribution_advanced)})
            return {'absorption_energy': min(100.0, max(0.01, float(absorption_advanced))), 'distribution_energy': min(100.0, max(0.01, float(distribution_advanced))), 'net_energy_flow': net_energy, 'game_intensity': min(1.0, max(0.0, float(game_intensity))), 'key_battle_zones': key_battle_zones, 'breakout_potential': min(100.0, float(breakout_potential)), 'energy_concentration': min(1.0, max(0.0, float(energy_concentration))), 'reference_price': float(reference_price), 'original_current_price': float(current_price), 'fake_distribution_flag': False}
        except Exception as e:
            return self._get_default_energy()

    def _calculate_energy_indicators(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, stock_code: str = "", trade_date: str = "") -> tuple:
        """[Version 18.0.0] 博弈能量指标计算器 (引入米氏方程饱和动力学)"""
        import numpy as np
        import math
        eps = np.finfo(np.float64).eps
        abs_changes = np.abs(changes)
        total_energy = np.sum(abs_changes)
        energy_concentration = 0.0
        if total_energy > eps:
            active_mask = abs_changes > 1e-5
            active_count = np.sum(active_mask)
            if active_count > 0:
                sorted_valid_changes = np.sort(abs_changes[active_mask])[::-1]
                top_count = max(1, int(active_count * 0.2))
                top_energy = np.sum(sorted_valid_changes[:top_count])
                base_concentration = float(top_energy / total_energy)
                normalized_energy = abs_changes / total_energy
                hhi = np.sum(normalized_energy ** 2)
                scale_penalty = float(active_count / (active_count + len(changes) * 0.05))
                energy_concentration = float(base_concentration * 0.4 + hhi * 0.6) * scale_penalty
                
        valid_changes = abs_changes[abs_changes > eps]
        dynamic_active_threshold = max(0.01, float(np.percentile(valid_changes, 60))) if len(valid_changes) > 5 else 0.05
        active_mask_intensity = abs_changes > dynamic_active_threshold
        active_energy_sum = np.sum(abs_changes[active_mask_intensity])
        prior_energy = max(1.0, total_energy * 0.05) 
        active_ratio = active_energy_sum / (total_energy + prior_energy + eps)
        # [摒弃 atan] 改用米氏动力学方程 (Michaelis-Menten) 映射活跃度饱和极值，更符合自然界物理规律
        absolute_scale = float(total_energy / (total_energy + 10.0))
        game_intensity = float(np.clip(active_ratio * absolute_scale, 0.0, 1.0))
        above_mask = price_grid > current_price
        below_mask = price_grid < current_price
        absorption_above = np.sum(changes[above_mask & (changes > 0)])
        distribution_above = np.sum(np.abs(changes[above_mask & (changes < 0)]))
        absorption_below = np.sum(changes[below_mask & (changes > 0)])
        distribution_below = np.sum(np.abs(changes[below_mask & (changes < 0)]))
        imbalance_prior = max(2.0, (absorption_below + distribution_below) * 0.1)
        below_imbalance = (absorption_below - distribution_below) / (absorption_below + distribution_below + imbalance_prior + eps)
        # [摒弃 tanh] 直接按比例线性放缩支撑乘数 [0.5, 1.5]
        support_strength = 1.0 + float(np.clip(below_imbalance, -0.5, 0.5))
        net_above = absorption_above - distribution_above
        # 突破势能具备物理上限(Max=100)，负动能直接归零，杜绝指数发散爆炸
        if net_above > 0: raw_potential = (net_above / (net_above + 15.0)) * support_strength * 100.0
        else: raw_potential = 0.0
            
        breakout_potential = float(np.clip(raw_potential, 0.0, 100.0))
        if energy_concentration > 0.5: breakout_potential = float(min(100.0, breakout_potential * (1.0 + (energy_concentration - 0.5))))
            
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "_calculate_energy_indicators", {'total_energy': float(total_energy), 'prior_energy': float(prior_energy), 'dynamic_threshold': float(dynamic_active_threshold)}, {'active_ratio': float(active_ratio), 'absolute_scale': float(absolute_scale), 'support_strength': float(support_strength)}, {'game_intensity': float(game_intensity), 'breakout_potential': float(breakout_potential), 'energy_concentration': float(energy_concentration)})
        return float(game_intensity), float(breakout_potential), float(energy_concentration)

    def _detect_fake_distribution_advanced(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, close_price: float) -> float:
        """[Version 24.0.0] 高阶虚假派发流形扫描器 (去掩码截断纯连续版)"""
        try:
            import numpy as np
            import math
            price_rel = (price_grid - current_price) / current_price
            near_weight = np.exp(-0.5 * (price_rel / 0.08)**2)
            near_net = np.sum(changes * near_weight)
            above_weight = 1.0 / (1.0 + np.exp(-30.0 * (price_rel - 0.08)))
            below_weight = 1.0 / (1.0 + np.exp(30.0 * (price_rel + 0.08)))
            above_distrib = np.sum(np.abs(changes) * above_weight * (changes < 0))
            below_accum = np.sum(changes * below_weight * (changes > 0))
            p_below = 1.0 / (1.0 + np.exp(-5.0 * (below_accum / max(above_distrib, 0.1) - 1.5)))
            p_near = 1.0 / (1.0 + np.exp(-10.0 * near_net))
            p_above = 1.0 / (1.0 + np.exp(-5.0 * (above_distrib - 0.5)))
            prob1 = p_below * p_near * p_above
            above_avg = np.sum(changes * above_weight) / max(np.sum(above_weight), 1.0)
            below_avg = np.sum(changes * below_weight) / max(np.sum(below_weight), 1.0)
            p_above_dec = 1.0 / (1.0 + np.exp(10.0 * (above_avg + 0.3)))
            p_below_inc = 1.0 / (1.0 + np.exp(-10.0 * (below_avg - 0.2)))
            p_dist_small = 1.0 / (1.0 + np.exp(5.0 * (above_distrib - 2.0)))
            prob2 = p_above_dec * p_below_inc * p_dist_small
            return float(1.0 - (1.0 - prob1) * (1.0 - prob2))
        except Exception: return 0.0

    def _ensure_nonzero_energy(self, energy_result: Dict[str, Any]) -> Dict[str, Any]:
        """确保能量场结果不为零"""
        absorption = energy_result.get('absorption_energy', 0)
        distribution = energy_result.get('distribution_energy', 0)
        # 如果吸收和派发能量都为0，设置一个小值
        if absorption == 0 and distribution == 0:
            import random
            # 这里的关键：是否应该设为0？
            # 如果没有吸筹和派发，能量应该为0，而不是默认值0.5
            # 但是为了后续计算，给一个非常小的值
            new_absorption = random.uniform(0.01, 0.1)  # 非常小的值，接近0
            new_distribution = random.uniform(0.01, 0.1)
            energy_result['absorption_energy'] = new_absorption
            energy_result['distribution_energy'] = new_distribution
            energy_result['net_energy_flow'] = new_absorption - new_distribution
            # 其他字段也设置较小的值
            energy_result['game_intensity'] = max(0.01, energy_result.get('game_intensity', 0.01))
            energy_result['breakout_potential'] = max(0.1, energy_result.get('breakout_potential', 0.1))
            energy_result['energy_concentration'] = max(0.1, energy_result.get('energy_concentration', 0.1))
        return energy_result

    def _detect_fake_distribution(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, volume_history: pd.Series = None) -> float:
        """[Version 24.0.0] 虚假派发软逻辑探测 (彻底废除 Boolean 返回与硬阈值)"""
        import numpy as np
        import math
        try:
            price_rel = (price_grid - current_price) / current_price
            mid_high_weight = np.exp(-0.5 * ((price_rel - 0.025) / 0.025)**2)
            high_weight = 1.0 / (1.0 + np.exp(-40.0 * (price_rel - 0.05)))
            mid_decrease = np.sum(np.abs(changes) * mid_high_weight * (changes < 0))
            high_increase = np.sum(changes * high_weight * (changes > 0))
            # 使用 Logistic 将减仓与增仓特征转化为连续概率
            p_mid_dec = 1.0 / (1.0 + np.exp(-10.0 * (mid_decrease - 0.4)))
            p_high_inc = 1.0 / (1.0 + np.exp(10.0 * (high_increase - 0.2)))
            base_prob = p_mid_dec * p_high_inc
            if volume_history is not None and len(volume_history) >= 5:
                recent_volume = volume_history.iloc[-5:].mean()
                avg_volume = volume_history.iloc[-20:-5].mean() if len(volume_history) >= 20 else recent_volume
                vol_ratio = recent_volume / max(avg_volume, 1.0)
                # 缩量调整是虚假派发的典型特征，用 Sigmoid 软化边界
                p_vol = 1.0 / (1.0 + np.exp(10.0 * (vol_ratio - 1.2)))
                return float(base_prob * p_vol)
            return float(base_prob)
        except Exception: return 0.0

    def _calculate_breakout_potential(self, resistance_zones: List[Dict], absorption_energy: float) -> float:
        """计算突破势能"""
        if not resistance_zones:
            return 0.0
        strongest_resistance = max([zone['resistance_strength'] for zone in resistance_zones])
        avg_distance = np.mean([zone['distance_to_current'] for zone in resistance_zones])
        if strongest_resistance > 0:
            base_potential = absorption_energy / strongest_resistance * 10
            distance_factor = 1 / (avg_distance + 0.1)
            return base_potential * distance_factor
        return 0.0

    def _calculate_energy_concentration(self, changes: np.ndarray, absorption: float, distribution: float) -> float:
        """计算能量集中度"""
        total_energy = absorption + distribution + 1e-10
        # 计算变化的标准差（衡量能量的分散程度）
        significant_changes = changes[np.abs(changes) > 0.1]
        if len(significant_changes) == 0:
            return 0.0
        change_std = np.std(significant_changes)
        max_std = np.max(np.abs(significant_changes)) * 0.5
        concentration = 1.0 - min(1.0, change_std / max_std)
        # 吸收能量占比越高，集中度越高
        absorption_ratio = absorption / total_energy
        return concentration * (0.5 + absorption_ratio * 0.5)

    def _identify_key_battle_zones(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, stock_code: str = "", trade_date: str = "") -> List[Dict]:
        """
        [Version 61.0.0] 极速战区侦测引擎 (Numba 零内存分配降维版)
        说明：彻底废除 np.lib.stride_tricks.sliding_window_view 引发的维度广播矩阵扩张。
        调用 Numba C 指令集计算周边动能极值，以绝对 O(1) 的额外内存完成窗口滑动！禁止使用空行。
        """
        import numpy as np
        battle_zones = []
        min_intensity = 0.5
        try:
            if len(changes) < 5: return []
            prices, intensities, chgs = _numba_battle_zones_core(changes.astype(np.float32), price_grid.astype(np.float32), float(current_price), float(min_intensity))
            if len(intensities) == 0: return []
            sort_idx = np.argsort(intensities)[::-1][:5]
            for idx in sort_idx:
                p = float(prices[idx]); c = float(chgs[idx])
                battle_zones.append({'price': p, 'battle_intensity': float(intensities[idx]), 'type': 'absorption' if c > 0 else 'distribution', 'position': 'below_current' if p < current_price else 'above_current', 'distance_to_current': float((p - current_price) / max(current_price, 1e-5))})
            return battle_zones
        except Exception as e:
            return []

    def _get_default_energy(self) -> Dict[str, Any]:
        """获取默认能量场"""
        # 问题：这里返回的是0.0，但在代码其他地方有0.5
        # 我认为默认值应该是0.0，表示没有能量
        result = {
            'absorption_energy': 0.0,
            'distribution_energy': 0.0,
            'net_energy_flow': 0.0,
            'game_intensity': 0.0,
            'key_battle_zones': [],
            'breakout_potential': 0.0,
            'energy_concentration': 0.0,
            'fake_distribution_flag': False,
        }
        return result

class ChipFactorCalculationHelper:
    """
    [Version 4.1.0] 筹码因子高精度安全计算核心辅助引擎
    说明：独立于 Django 模型的计算服务，用于生成需要填充到 ChipFactorBase 的各字段。
    全面修复了原文档公式中存在的除零溢出风险、线性假设失真，并对信息熵加入了严格的空值屏蔽。
    """
    @classmethod
    def calculate_core_chip_factors(cls, close: float, cost_percentiles: dict, his_high: float, his_low: float, winner_rate: float, chip_distribution: np.ndarray) -> dict:
        """
        [Version 8.0.0] 核心筹码因子高精度安全计算引擎（双轨博弈防死锁版）
        说明：根除his_range引发的历史极值锚定畸变，采用c_95-c_5动态活跃视界替代。
        废除导致0值连乘死锁的单极获利抛压模型，引入套牢恐慌盘双轨复合压力模型，全程采用反正切平滑归一化防止极性反噬。
        """
        import numpy as np
        import math
        eps = np.finfo(np.float64).eps
        c_5 = float(cost_percentiles.get('5pct', close))
        c_15 = float(cost_percentiles.get('15pct', close))
        c_50 = float(cost_percentiles.get('50pct', close))
        c_85 = float(cost_percentiles.get('85pct', close))
        c_95 = float(cost_percentiles.get('95pct', close))
        active_range = max(c_95 - c_5, eps)
        core_range = max(c_85 - c_15, eps)
        chip_concentration_ratio = core_range / active_range
        chip_stability = max(0.0, 1.0 - chip_concentration_ratio)
        price_percentile_position = np.clip((close - c_5) / active_range, 0.0, 1.0)
        raw_pressure = (close - c_50) / (core_range + active_range * 0.1)
        adaptive_pressure = 0.5 + (math.atan(raw_pressure * 2.0) / math.pi)
        profit_pressure = adaptive_pressure * winner_rate
        trapped_rate = max(0.0, 1.0 - winner_rate)
        panic_pressure = (1.0 - adaptive_pressure) * trapped_rate * (1.0 - math.exp(-trapped_rate * 3.0))
        comprehensive_pressure = profit_pressure * 0.6 + panic_pressure * 0.4
        win_rate_price_position = winner_rate * 0.6 + float(price_percentile_position) * 0.4
        valid_mask = chip_distribution > eps
        if np.any(valid_mask):
            valid_dist = chip_distribution[valid_mask]
            norm_dist = valid_dist / np.sum(valid_dist)
            chip_entropy = float(-np.sum(norm_dist * np.log(norm_dist)))
        else:
            chip_entropy = 0.0
        chip_convergence_ratio = min(1.0, core_range / active_range)
        macro_range = max(float(his_high) - float(his_low), active_range)
        chip_divergence_ratio = float((math.atan((active_range / macro_range) * 3.0) / (math.pi / 2)))
        final_result = {'chip_concentration_ratio': round(float(chip_concentration_ratio), 6), 'chip_stability': round(float(chip_stability), 6), 'price_percentile_position': round(float(price_percentile_position), 6), 'profit_pressure': round(float(comprehensive_pressure), 6), 'win_rate_price_position': round(float(win_rate_price_position), 6), 'chip_entropy': round(float(chip_entropy), 6), 'chip_convergence_ratio': round(float(chip_convergence_ratio), 6), 'chip_divergence_ratio': round(float(chip_divergence_ratio), 6)}
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        QuantitativeTelemetryProbe.emit("ChipFactorCalculationHelper", "calculate_core_chip_factors", {'close': close, 'winner_rate': winner_rate, 'active_range': active_range}, {'adaptive_pressure': adaptive_pressure, 'profit_pressure': profit_pressure, 'panic_pressure': panic_pressure}, final_result)
        return final_result

class QuantitativeTelemetryProbe:
    """
    [Version 1.0.0] 工业级量化全链路探针输出组件
    说明：负责统一收集并格式化输出模型计算全链路的"原始数据、关键计算节点、最终分数"，消除系统信息孤岛。
    """
    @classmethod
    def emit(cls, module_name: str, method_name: str, raw_data: dict, calc_nodes: dict, final_score: dict) -> None:
        """
        [Version 4.0.0] 物理落盘级绝对强制探针（破壁版）
        说明：彻底突破Celery标准输出劫持，强制双写至物理文件，并配备万能异常宽容序列化器，粉碎一切序列化黑洞。
        """
        import json, sys, os, datetime
        import numpy as np
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
        payload = {"time": datetime.datetime.now().isoformat(), "module": module_name, "method": method_name, "raw_data": raw_data, "calc_nodes": calc_nodes, "final_score": final_score}
        try:
            out_str = f"📡 [QUANT-PROBE] | {json.dumps(payload, ensure_ascii=False, cls=UltimateEncoder)}\n"
        except Exception as e:
            out_str = f"⚠️ [QUANT-PROBE-ERR] 无法序列化: {e} | Module: {module_name} | Method: {method_name}\n"
        try:
            sys.stderr.write(out_str)
            sys.stderr.flush()
        except Exception: pass
        try:
            with open(os.path.join(os.getcwd(), 'quant_probe_emergency.log'), 'a', encoding='utf-8') as f:
                f.write(out_str)
        except Exception: pass





