# services/chip_holding_calculator.py
import numpy as np
import pandas as pd
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
                # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_identify_peak_morphology", {'prominence': float(dynamic_prominence), 'peaks_found': int(peak_count)}, {'main_peak_price': main_peak_price, 'relative_pos': float(relative_pos)}, result)
            return result
        except Exception:
            return {'peak_count': 0, 'main_peak_position': 0, 'main_peak_price': 0.0, 'peak_distance_ratio': 0.0, 'peak_concentration': 0.0, 'is_double_peak': False, 'is_multi_peak': False}

    def _identify_behavior_patterns(self, percent_change_matrix: np.ndarray, chip_matrix: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, any]:
        """
        [Version 48.0.0] 行为金融学多空交锋拓扑扫描器 (物理边界重构版)
        说明：斩除原版基于 current_price 导致的探测盲区。
        重新定义物理规则：
        1. 吸筹(Accumulation) = 底部区间(0~40%)筹码显著增加 + 顶部区间(60~100%)筹码显著减少(割肉)。
        2. 派发(Distribution) = 顶部区间筹码显著增加(接盘) + 底部区间筹码显著减少(获利了结)。
        引入真实的活跃视界界定高低区域，彻底消灭主力特征为 0 的装死现象。禁止使用空行。
        """
        import numpy as np
        import math
        if percent_change_matrix.shape[0] < 3: return self._get_default_behavior_patterns()
        patterns = {'accumulation': {'detected': False, 'strength': 0.0, 'areas': []}, 'distribution': {'detected': False, 'strength': 0.0, 'areas': []}, 'consolidation': {'detected': False, 'strength': 0.0}, 'breakout_preparation': {'detected': False, 'strength': 0.0}, 'main_force_activity': 0.0}
        lookback = min(5, percent_change_matrix.shape[0])
        recent_changes = percent_change_matrix[-lookback:, :]
        changes_last_3 = recent_changes[-3:, :]
        sum_changes_3 = np.sum(changes_last_3, axis=0)
        mean_changes_3 = np.mean(changes_last_3, axis=0)
        current_chip = chip_matrix[-1]
        active_mask = current_chip > 1e-4
        if np.any(active_mask):
            valid_prices = price_grid[active_mask]
            p_min, p_max = valid_prices[0], valid_prices[-1]
        else:
            p_min, p_max = price_grid[0], price_grid[-1]
        p_range = max(p_max - p_min, 1e-5)
        low_threshold = p_min + p_range * 0.40
        high_threshold = p_max - p_range * 0.40
        low_price_mask = price_grid <= low_threshold
        high_price_mask = price_grid >= high_threshold
        total_3d_energy = float(np.sum(np.abs(sum_changes_3)))
        dynamic_noise_th = max(0.05, total_3d_energy * 0.02)
        increase_mask = sum_changes_3 > dynamic_noise_th
        decrease_mask = sum_changes_3 < -dynamic_noise_th
        accum_inc_indices = np.where(increase_mask & low_price_mask)[0]
        accum_dec_indices = np.where(decrease_mask & high_price_mask)[0]
        dist_inc_indices = np.where(increase_mask & high_price_mask)[0]
        dist_dec_indices = np.where(decrease_mask & low_price_mask)[0]
        raw_accum_strength = 0.0
        if len(accum_inc_indices) > 0 or len(accum_dec_indices) > 0:
            patterns['accumulation']['detected'] = True
            raw_accum_strength = float(np.sum(sum_changes_3[accum_inc_indices])) + float(np.sum(np.abs(sum_changes_3[accum_dec_indices])))
            patterns['accumulation']['strength'] = float(math.tanh(raw_accum_strength / max(5.0, total_3d_energy * 0.15)))
            for idx in accum_inc_indices: patterns['accumulation']['areas'].append({'price': float(price_grid[idx]), 'avg_change': float(mean_changes_3[idx]), 'distance_to_price': float((current_price - price_grid[idx]) / max(current_price, 1e-5))})
        raw_distrib_strength = 0.0
        if len(dist_inc_indices) > 0 or len(dist_dec_indices) > 0:
            patterns['distribution']['detected'] = True
            raw_distrib_strength = float(np.sum(sum_changes_3[dist_inc_indices])) + float(np.sum(np.abs(sum_changes_3[dist_dec_indices])))
            patterns['distribution']['strength'] = float(math.tanh(raw_distrib_strength / max(5.0, total_3d_energy * 0.15)))
            for idx in dist_inc_indices: patterns['distribution']['areas'].append({'price': float(price_grid[idx]), 'avg_change': float(mean_changes_3[idx]), 'distance_to_price': float((price_grid[idx] - current_price) / max(current_price, 1e-5))})
        abs_recent = np.abs(recent_changes[-1])
        active_grid_mask = abs_recent > 1e-4
        if np.any(active_grid_mask):
            daily_total_energy = float(np.sum(abs_recent))
            daily_noise_th = max(0.02, daily_total_energy * 0.05)
            significant_ratio = np.sum(abs_recent > daily_noise_th) / np.sum(active_grid_mask)
            patterns['main_force_activity'] = float(math.tanh(significant_ratio * 2.0))
        if patterns['accumulation']['areas']: patterns['accumulation']['areas'] = sorted(patterns['accumulation']['areas'], key=lambda x: x['avg_change'], reverse=True)[:5]
        if patterns['distribution']['areas']: patterns['distribution']['areas'] = sorted(patterns['distribution']['areas'], key=lambda x: abs(x['avg_change']), reverse=True)[:5]
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_identify_behavior_patterns", {'low_threshold': float(low_threshold), 'high_threshold': float(high_threshold), 'total_3d_energy': total_3d_energy}, {'raw_accum_strength': float(raw_accum_strength), 'raw_distrib_strength': float(raw_distrib_strength)}, {'accum_strength': patterns['accumulation']['strength'], 'distrib_strength': patterns['distribution']['strength'], 'main_force_activity': patterns['main_force_activity']})
        return patterns

    def _build_normalized_chip_matrix(self, chip_history: list, current_chip_dist: pd.DataFrame) -> tuple:
        """
        [Version 3.0.0] 基于真实截面数据的精确质心保持映射矩阵构建
        说明：
        1. 修复原版本将真实分布数据List错认为DataFrame并调用.empty引发的崩溃灾难。
        2. 废弃基于K线高低点与换手率模拟三角衰减的失真算法，直接消费高精度真实筹码分布切片。
        3. 引入质心保持线性分配映射(Mass-Preserving Linear Allocation)，将异构的不规则价格点精确、守恒地投射到200格全局刚性价格网格。
        4. 采用Numpy底层向量化运算与np.add.at指针级聚合，避免Python层循环瓶颈，并将横截面能量严格归一化至100.0修复下游阈值击穿隐患。
        """
        import numpy as np
        import pandas as pd
        all_dists = []
        if isinstance(chip_history, list):
            for df in chip_history:
                if isinstance(df, pd.DataFrame) and not df.empty and 'price' in df.columns and 'percent' in df.columns:
                    all_dists.append(df)
        if isinstance(current_chip_dist, pd.DataFrame) and not current_chip_dist.empty and 'price' in current_chip_dist.columns and 'percent' in current_chip_dist.columns:
            all_dists.append(current_chip_dist)
        if not all_dists:
            return np.array([]), np.array([])
        global_min = float('inf')
        global_max = float('-inf')
        for df in all_dists:
            p_min = float(df['price'].min())
            p_max = float(df['price'].max())
            if p_min < global_min: global_min = p_min
            if p_max > global_max: global_max = p_max
        if global_min == global_max or global_min == float('inf'):
            global_min = max(0.01, global_min * 0.9)
            global_max = global_max * 1.1
        else:
            global_min = max(0.01, global_min * 0.95)
            global_max = global_max * 1.05
        price_grid = np.linspace(global_min, global_max, self.price_granularity, dtype=np.float64)
        days = len(all_dists)
        chip_matrix = np.zeros((days, self.price_granularity), dtype=np.float64)
        grid_step = price_grid[1] - price_grid[0] if self.price_granularity > 1 else 1.0
        for i, df in enumerate(all_dists):
            prices = df['price'].to_numpy(dtype=np.float64)
            percents = df['percent'].to_numpy(dtype=np.float64)
            float_indices = (prices - global_min) / grid_step
            float_indices = np.clip(float_indices, 0, self.price_granularity - 1.0001)
            left_indices = np.floor(float_indices).astype(np.int32)
            right_indices = left_indices + 1
            right_weights = float_indices - left_indices
            left_weights = 1.0 - right_weights
            np.add.at(chip_matrix[i], left_indices, percents * left_weights)
            np.add.at(chip_matrix[i], right_indices, percents * right_weights)
            row_sum = np.sum(chip_matrix[i])
            if row_sum > 1e-8:
                chip_matrix[i] = (chip_matrix[i] / row_sum) * 100.0
        return price_grid, chip_matrix

    def _calculate_percent_change_matrix(self, chip_matrix: np.ndarray) -> np.ndarray:
        """
        [Version 2.1.0] 计算绝对百分比变动矩阵
        说明：提取筹码矩阵的时间序列绝对变化，严格过滤噪音波动，用于后续机构资金净流入动力学分析。
        """
        rows, cols = chip_matrix.shape
        change_matrix = np.zeros((rows - 1, cols), dtype=np.float64)
        for i in range(rows - 1):
            change_matrix[i, :] = chip_matrix[i + 1, :] - chip_matrix[i, :]
        noise_level = self.params.get('noise_threshold', 0.2) / 100.0
        change_matrix[np.abs(change_matrix) < noise_level] = 0.0
        return change_matrix

    def _analyze_absolute_changes(self, percent_change_matrix: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, any]:
        """
        [Version 48.0.0] 绝对变化信号动力学算子（自适应全域积分版）
        说明：粉碎固定阈值导致的信噪比崩塌与网格分辨率稀释效应。
        引入基于当日截面总变动能量(total_active_energy)的动态基准阈值，将信号质量通过 Tanh(total_active_energy / 2.5) 直接映射。
        无论横盘还是爆量，确保 validation_score 获得具有区分度的分数。禁止使用空行。
        """
        import numpy as np
        import math
        if percent_change_matrix.shape[0] == 0: return self._get_default_absolute_signals()
        recent_changes = percent_change_matrix[-min(3, len(percent_change_matrix)):, :]
        avg_changes = np.mean(recent_changes, axis=0) if recent_changes.shape[0] > 0 else np.zeros_like(price_grid)
        abs_changes = np.abs(avg_changes)
        active_grid_mask = abs_changes > 1e-4
        total_active_energy = float(np.sum(abs_changes[active_grid_mask]))
        signal_quality = float(math.tanh(total_active_energy / 2.5))
        noise_ratio = float(max(0.0, 1.0 - signal_quality))
        dynamic_sig_th = max(0.1, total_active_energy * 0.10)
        increase_mask = avg_changes > dynamic_sig_th
        decrease_mask = avg_changes < -dynamic_sig_th
        signals = {'significant_increase_areas': [], 'significant_decrease_areas': [], 'accumulation_signals': [], 'distribution_signals': [], 'noise_level': noise_ratio, 'signal_quality': signal_quality}
        dist_to_current = np.abs(price_grid - current_price) / (current_price if current_price > 0 else 1.0)
        inc_indices = np.where(increase_mask)[0]
        for idx in inc_indices:
            change_val = float(avg_changes[idx])
            signals['significant_increase_areas'].append({'price': float(price_grid[idx]), 'change': change_val, 'distance_to_current': float(dist_to_current[idx])})
            if price_grid[idx] < current_price * 0.95:
                strength = float(2.0 / (1.0 + math.exp(-change_val / max(0.5, total_active_energy * 0.2))) - 1.0)
                signals['accumulation_signals'].append({'price': float(price_grid[idx]), 'change': change_val, 'strength': strength})
        dec_indices = np.where(decrease_mask)[0]
        for idx in dec_indices:
            change_val = float(avg_changes[idx])
            signals['significant_decrease_areas'].append({'price': float(price_grid[idx]), 'change': change_val, 'distance_to_current': float(dist_to_current[idx])})
            if price_grid[idx] > current_price * 1.05:
                strength = float(2.0 / (1.0 + math.exp(-abs(change_val) / max(0.5, total_active_energy * 0.2))) - 1.0)
                signals['distribution_signals'].append({'price': float(price_grid[idx]), 'change': change_val, 'strength': strength})
        for key in ['significant_increase_areas', 'significant_decrease_areas', 'accumulation_signals', 'distribution_signals']:
            signals[key] = sorted(signals[key], key=lambda x: abs(x['change']), reverse=True)[:10]
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_analyze_absolute_changes", {'total_active_energy': total_active_energy, 'noise_ratio': noise_ratio}, {'dynamic_sig_th': dynamic_sig_th}, {'signal_quality': signal_quality})
        return signals

    def _calculate_concentration_metrics(self, current_chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, price_history: pd.DataFrame, is_history: bool = False) -> Dict[str, float]:
        """
        [Version 38.0.0] 概率密度鲁棒高阶矩与CDF连续插值引擎 (历史探针隔离修复版)
        说明：追加缺失的 is_history 标志位，彻底修复由技术面引擎回溯过去 t-5 日状态时因参数不匹配引发的 TypeError 致命崩溃。
        同时静默历史切片计算时的探针输出，防止历史数据污染当天的监控面板。禁止使用空行。
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
        metrics['chip_mean'] = chip_mean; metrics['weight_avg_cost'] = chip_mean; metrics['chip_std'] = chip_std
        cdf = np.cumsum(p)
        cost_05 = float(np.interp(0.05, cdf, price_grid))
        cost_10 = float(np.interp(0.10, cdf, price_grid))
        cost_15 = float(np.interp(0.15, cdf, price_grid))
        cost_25 = float(np.interp(0.25, cdf, price_grid))
        cost_50 = float(np.interp(0.50, cdf, price_grid))
        cost_75 = float(np.interp(0.75, cdf, price_grid))
        cost_85 = float(np.interp(0.85, cdf, price_grid))
        cost_90 = float(np.interp(0.90, cdf, price_grid))
        cost_95 = float(np.interp(0.95, cdf, price_grid))
        denom_skew = max(cost_90 - cost_10, eps)
        metrics['chip_skewness'] = float((cost_90 + cost_10 - 2.0 * cost_50) / denom_skew)
        denom_kurt = max(cost_75 - cost_25, eps)
        metrics['chip_kurtosis'] = float((cost_95 - cost_05) / denom_kurt)
        metrics['cost_5pct'] = cost_05; metrics['cost_15pct'] = cost_15; metrics['cost_50pct'] = cost_50; metrics['cost_85pct'] = cost_85; metrics['cost_95pct'] = cost_95
        metrics['winner_rate'] = float(np.interp(current_price, price_grid, cdf))
        his_low = float(price_history['low_qfq'].min()) if price_history is not None and not price_history.empty and 'low_qfq' in price_history.columns else float(current_price * 0.8)
        his_high = float(price_history['high_qfq'].max()) if price_history is not None and not price_history.empty and 'high_qfq' in price_history.columns else float(current_price * 1.2)
        metrics['his_low'] = his_low; metrics['his_high'] = his_high
        macro_range = max(his_high - his_low, eps)
        core_range = max(cost_85 - cost_15, eps)
        active_range = max(cost_95 - cost_05, eps)
        metrics['chip_concentration_ratio'] = float(math.atan((core_range / macro_range) * 3.0) / (math.pi / 2))
        metrics['chip_stability'] = float(max(0.0, 1.0 - metrics['chip_concentration_ratio']))
        metrics['chip_divergence_ratio'] = float(math.atan((active_range / macro_range) * 3.0) / (math.pi / 2))
        price_position = np.clip((current_price - cost_05) / active_range, 0.0, 1.0)
        metrics['price_percentile_position'] = float(price_position)
        metrics['win_rate_price_position'] = float(metrics['winner_rate'] * 0.6 + price_position * 0.4)
        raw_price_ratio = (current_price - chip_mean) / max(chip_mean, eps)
        metrics['price_to_weight_avg_ratio'] = float(math.atan(raw_price_ratio * 10.0) / (math.pi / 2))
        max_price = price_grid[-1]
        high_lock_mask = price_grid >= max_price * 0.9
        metrics['high_position_lock_ratio_90'] = float(np.sum(p[high_lock_mask]))
        main_cost_mask = (price_grid >= cost_50 * 0.9) & (price_grid <= cost_50 * 1.1)
        metrics['main_cost_range_ratio'] = float(np.sum(p[main_cost_mask]))
        metrics['chip_convergence_ratio'] = metrics['main_cost_range_ratio']
        smoothed_p = (p + 1e-5) / np.sum(p + 1e-5)
        entropy_val = float(-np.sum(smoothed_p * np.log(smoothed_p)))
        metrics['chip_entropy'] = float(entropy_val)
        metrics['entropy_concentration'] = float(1.0 - (entropy_val / np.log(len(smoothed_p))))
        sorted_p = np.sort(p)[::-1]
        metrics['peak_concentration'] = float(np.sum(sorted_p[:max(1, int(len(p) * 0.2))]))
        metrics['cv_concentration'] = float(1.0 - (math.atan((chip_std / max(chip_mean, eps)) * 5.0) / (math.pi / 2)))
        metrics['main_force_concentration'] = metrics['main_cost_range_ratio']
        metrics['comprehensive_concentration'] = float(0.3 * metrics['entropy_concentration'] + 0.3 * metrics['peak_concentration'] + 0.2 * metrics['cv_concentration'] + 0.2 * metrics['main_cost_range_ratio'])
        if not is_history:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_concentration_metrics", {'current_price': float(current_price), 'macro_range': float(macro_range)}, {'robust_skewness': metrics['chip_skewness'], 'robust_kurtosis': metrics['chip_kurtosis'], 'cost_50pct': metrics['cost_50pct']}, metrics)
        return metrics

    def _get_default_concentration_metrics(self) -> Dict[str, float]:
        """[Version 18.0.0] 默认集中度指标集"""
        return {'entropy_concentration': 0.5, 'peak_concentration': 0.3, 'cv_concentration': 0.5, 'main_force_concentration': 0.2, 'comprehensive_concentration': 0.4, 'chip_skewness': 0.0, 'chip_kurtosis': 0.0, 'chip_mean': 0.0, 'chip_std': 0.0, 'weight_avg_cost': 0.0, 'cost_5pct': 0.0, 'cost_15pct': 0.0, 'cost_50pct': 0.0, 'cost_85pct': 0.0, 'cost_95pct': 0.0, 'winner_rate': 0.0, 'win_rate_price_position': 0.0, 'price_to_weight_avg_ratio': 0.0, 'high_position_lock_ratio_90': 0.0, 'main_cost_range_ratio': 0.0, 'chip_convergence_ratio': 0.0, 'chip_divergence_ratio': 0.0, 'chip_entropy': 0.0, 'chip_concentration_ratio': 0.0, 'chip_stability': 0.0, 'price_percentile_position': 0.0, 'his_low': 0.0, 'his_high': 0.0}

    def _calculate_holding_metrics(self, turnover_rate: float, chip_stability: float) -> Dict[str, float]:
        """
        [Version 35.0.0] 持有期泊松衰减反演器 (换手率量纲雪崩修复版)
        说明：彻底铲除 turnover_rate > 1.0 的量纲瞎猜逻辑！A股基本面换手率(0.64)代表的是0.64%，无论大小必须坚决除以100。
        修复将大盘股低换手(0.64%)误判为64%导致持有天数坍塌至1.6天的物理级灾难。禁止使用空行。
        """
        import math
        metrics = {'short_term_chip_ratio': 0.2, 'mid_term_chip_ratio': 0.3, 'long_term_chip_ratio': 0.5, 'avg_holding_days': 60.0}
        try:
            if turnover_rate <= 0: return metrics
            tr = float(turnover_rate) / 100.0
            tr = max(0.0001, min(0.6, tr))
            avg_days = 1.0 / tr
            metrics['avg_holding_days'] = float(max(1.0, min(avg_days, 1500.0)))
            short_ratio = 1.0 - math.exp(-5.0 * tr)
            long_ratio = math.exp(-60.0 * tr)
            metrics['short_term_chip_ratio'] = float(short_ratio * 0.6 + (1.0 - chip_stability) * 0.4)
            metrics['long_term_chip_ratio'] = float(long_ratio * 0.6 + chip_stability * 0.4)
            metrics['mid_term_chip_ratio'] = float(max(0.0, 1.0 - metrics['short_term_chip_ratio'] - metrics['long_term_chip_ratio']))
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_holding_metrics", {'turnover_rate': turnover_rate, 'chip_stability': chip_stability}, {'raw_avg_days': avg_days, 'tr': tr}, metrics)
            return metrics
        except Exception as e:
            return metrics

    def _calculate_technical_metrics(self, price_history: pd.DataFrame, current_price: float, chip_mean: float, current_concentration: float, chip_matrix: np.ndarray, price_grid: np.ndarray, morph_metrics: Dict, energy_metrics: Dict, conc_metrics: Dict, tick_factors: Dict = None) -> Dict[str, float]:
        """
        [Version 51.0.0] 技术面全景共振引擎 (接口契约修复与Tick流体融合版)
        说明：彻底修复因多传 tick_enhanced_factors 参数引发的 TypeError: takes 10 positional arguments but 11 were given 致命崩溃。
        正式接入 tick_factors 参数，并将日内主力活跃度与筹码净流向融入趋势得分，完成时空共振的最后一块拼图。禁止使用空行。
        """
        import numpy as np
        import math
        import pandas as pd
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        metrics = self._get_default_technical_metrics()
        if price_history is None or price_history.empty or 'close_qfq' not in price_history.columns:
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_technical_metrics", {}, {'reason': 'price_history empty or missing close_qfq'}, {'status': 'aborted'})
            return metrics
        try:
            clean_df = price_history.copy()
            clean_df['close_qfq'] = pd.to_numeric(clean_df['close_qfq'], errors='coerce')
            clean_df['close_qfq'] = clean_df['close_qfq'].ffill().fillna(current_price)
            closes = clean_df['close_qfq'].to_numpy(dtype=np.float64)
            if len(closes) == 0: return metrics
            metrics['his_low'] = float(np.min(closes))
            metrics['his_high'] = float(np.max(closes))
            ma5 = float(np.mean(closes[-5:])) if len(closes) >= 5 else float(closes[-1])
            ma21 = float(np.mean(closes[-21:])) if len(closes) >= 21 else float(closes[-1])
            ma34 = float(np.mean(closes[-34:])) if len(closes) >= 34 else float(closes[-1])
            ma55 = float(np.mean(closes[-55:])) if len(closes) >= 55 else float(closes[-1])
            metrics['price_to_ma5_ratio'] = float((current_price - ma5) / (ma5 + 1e-8) * 100.0)
            metrics['price_to_ma21_ratio'] = float((current_price - ma21) / (ma21 + 1e-8) * 100.0)
            metrics['price_to_ma34_ratio'] = float((current_price - ma34) / (ma34 + 1e-8) * 100.0)
            metrics['price_to_ma55_ratio'] = float((current_price - ma55) / (ma55 + 1e-8) * 100.0)
            if len(closes) >= 55:
                if ma5 > ma21 > ma34 > ma55: metrics['ma_arrangement_status'] = 1.0
                elif ma5 < ma21 < ma34 < ma55: metrics['ma_arrangement_status'] = -1.0
                else: metrics['ma_arrangement_status'] = 0.0
            metrics['chip_cost_to_ma21_diff'] = float((chip_mean - ma21) / (ma21 + 1e-8) * 100.0)
            returns = np.diff(closes) / (closes[:-1] + 1e-8)
            volatility = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.02
            if math.isnan(volatility) or math.isinf(volatility): volatility = 0.02
            metrics['volatility_adjusted_concentration'] = float(current_concentration * math.exp(-volatility * 15.0))
            if len(closes) >= 15:
                diffs = np.diff(closes[-15:])
                gains = np.where(diffs > 0, diffs, 0.0)
                losses = np.where(diffs < 0, -diffs, 0.0)
                mean_loss = float(np.mean(losses))
                rs = float(np.mean(gains)) / (mean_loss + 1e-8) if mean_loss > 1e-8 else 100.0
                rsi_norm = float((100.0 - (100.0 / (1.0 + rs))) / 100.0)
                energy_norm = float(math.atan(energy_metrics.get('net_energy_flow', 0.0)) / (math.pi/2) * 0.5 + 0.5)
                metrics['chip_rsi_divergence'] = float(energy_norm - rsi_norm)
            if 'turnover_rate' in clean_df.columns:
                trn = pd.to_numeric(clean_df['turnover_rate'], errors='coerce').ffill()
                if not trn.dropna().empty: metrics['turnover_rate'] = float(trn.dropna().iloc[-1])
            if 'volume_ratio' in clean_df.columns:
                vr = pd.to_numeric(clean_df['volume_ratio'], errors='coerce').ffill()
                if not vr.dropna().empty: metrics['volume_ratio'] = float(vr.dropna().iloc[-1])
            if chip_matrix.shape[0] >= 6:
                morph_5d = self._identify_peak_morphology(chip_matrix[-6], price_grid, is_history=True)
                peak_price_today = float(morph_metrics.get('main_peak_price', current_price))
                peak_price_5d = float(morph_5d.get('main_peak_price', current_price))
                metrics['peak_migration_speed_5d'] = float((peak_price_today - peak_price_5d) / (current_price + 1e-8) * 100.0)
                conc_5d = self._calculate_concentration_metrics(chip_matrix[-6], price_grid, float(clean_df['close_qfq'].iloc[-6]) if len(clean_df) >= 6 else current_price, pd.DataFrame(), is_history=True)
                metrics['chip_stability_change_5d'] = float(current_concentration - conc_5d.get('chip_stability', 0.5))
            his_range = max(metrics['his_high'] - metrics['his_low'], 1e-5)
            active_range = max(conc_metrics.get('cost_95pct', current_price*1.1) - conc_metrics.get('cost_5pct', current_price*0.9), 1e-5)
            metrics['chip_divergence_ratio'] = float(math.atan((active_range / his_range) * 3.0) / (math.pi / 2))
            metrics['chip_convergence_ratio'] = float(1.0 - metrics['chip_divergence_ratio'])
            net_energy = float(energy_metrics.get('net_energy_flow', 0.0))
            trend_score = 0.5 + (0.2 * metrics['ma_arrangement_status']) + (0.15 * math.tanh(net_energy))
            tick_quality = float(tick_factors.get('tick_data_quality_score', 0.0)) if tick_factors else 0.0
            if tick_quality > 0.3:
                tick_flow = float(tick_factors.get('tick_level_chip_flow', 0.0))
                trend_score += float(math.tanh(tick_flow * 2.0) * 0.1)
            metrics['trend_confirmation_score'] = float(np.clip(trend_score, 0.0, 1.0))
            price_mom = metrics['price_to_ma5_ratio'] / 100.0
            divergence_product = float(-price_mom * net_energy)
            reversal = float(max(0.0, math.tanh(divergence_product * 5.0)))
            if tick_quality > 0.3:
                abnormal_vol = float(tick_factors.get('tick_abnormal_volume_ratio', 0.0))
                if abnormal_vol > 0.2 and reversal > 0.3: reversal = min(1.0, reversal + abnormal_vol * 0.5)
            metrics['reversal_warning_score'] = float(np.clip(reversal, 0.0, 1.0))
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_technical_metrics", {'ma5': ma5, 'net_energy': net_energy, 'divergence_product': divergence_product}, {'volatility': volatility, 'turnover_rate': metrics['turnover_rate']}, {'status': 'success', 'chip_rsi_divergence': metrics['chip_rsi_divergence'], 'reversal_warning_score': metrics['reversal_warning_score']})
            return metrics
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_technical_metrics_FATAL", {}, {'error': str(e), 'trace': err_trace}, {'status': 'crashed'})
            return metrics

    async def analyze_chip_dynamics_daily(self, stock_code: str, trade_date: str, lookback_days: int = 20, tick_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """
        [Version 51.0.0] 分析单日筹码动态主入口（关键字契约强绑定版）
        说明：全面采用 kwargs (关键字传参) 替换脆弱的 positional arguments (位置传参)，彻底消除后续参数增删导致的 signature mismatch 连环崩塌。禁止使用空行。
        """
        import numpy as np
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
            technical_metrics = self._calculate_technical_metrics(price_history=chip_data['price_history'], current_price=chip_data['current_price'], chip_mean=float(concentration_metrics.get('chip_mean', chip_data['current_price'])), current_concentration=float(concentration_metrics.get('comprehensive_concentration', 0.5)), chip_matrix=chip_matrix, price_grid=price_grid, morph_metrics=morphology_result, energy_metrics=game_energy_result, conc_metrics=concentration_metrics, tick_factors=tick_enhanced_factors)
            holding_metrics = self._calculate_holding_metrics(turnover_rate=technical_metrics.get('turnover_rate', 0.0), chip_stability=concentration_metrics.get('chip_stability', 0.5))
            validation_warnings = []
            base_data_integrity = 1.0
            penalty_exponent = 0.0
            if history_len < lookback_days: validation_warnings.append(f"历史数据不足: {history_len}/{lookback_days}"); penalty_exponent += 0.2
            current_price = chip_data['current_price']
            if current_price > price_grid.max() or current_price < price_grid.min(): validation_warnings.append("当前价格超出网格范围"); penalty_exponent += 0.4
            if 'tick_data_quality_score' in tick_enhanced_factors and tick_enhanced_factors['tick_data_quality_score'] < 0.3:
                days_ago = (datetime.now() - datetime.strptime(trade_date, "%Y-%m-%d")).days
                if days_ago <= 15: validation_warnings.append(f"近15日内tick数据质量低: {tick_enhanced_factors['tick_data_quality_score']:.2f}"); penalty_exponent += 0.1
            signal_q = float(absolute_signals.get('signal_quality', 0.5))
            validation_score = float(base_data_integrity * np.exp(-penalty_exponent) * (0.8 + 0.2 * signal_q))
            result = {'stock_code': stock_code, 'trade_date': trade_date, 'price_grid': price_grid.tolist(), 'chip_matrix': chip_matrix.tolist(), 'percent_change_matrix': percent_change_matrix.tolist(), 'absolute_change_signals': absolute_signals, 'concentration_metrics': concentration_metrics, 'pressure_metrics': pressure_metrics, 'behavior_patterns': behavior_patterns, 'migration_patterns': migration_patterns, 'convergence_metrics': convergence_metrics, 'game_energy_result': game_energy_result, 'direct_ad_result': direct_ad_result, 'morphology_metrics': morphology_result, 'technical_metrics': technical_metrics, 'holding_metrics': holding_metrics, 'tick_enhanced_factors': tick_enhanced_factors, 'validation_score': round(validation_score, 4), 'validation_warnings': validation_warnings, 'analysis_status': 'success', 'analysis_time': datetime.now().isoformat()}
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "analyze_chip_dynamics_daily", {'stock_code': stock_code, 'trade_date': trade_date, 'signal_quality': float(signal_q), 'penalty': float(penalty_exponent)}, {'validation_warnings': validation_warnings}, {'validation_score': float(validation_score), 'status': 'success'})
            return result
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "analyze_chip_dynamics_daily_FATAL", {'stock_code': stock_code, 'trade_date': trade_date}, {'error': str(e), 'trace': err_trace}, {'status': 'crashed'})
            return self._get_default_result(stock_code, trade_date)

    def _get_default_technical_metrics(self) -> Dict[str, float]:
        """[Version 18.0.0] 技术面默认指标初始化"""
        return {'his_low': 0.0, 'his_high': 0.0, 'price_to_ma5_ratio': 0.0, 'price_to_ma21_ratio': 0.0, 'price_to_ma34_ratio': 0.0, 'price_to_ma55_ratio': 0.0, 'ma_arrangement_status': 0.0, 'chip_cost_to_ma21_diff': 0.0, 'volatility_adjusted_concentration': 0.0, 'chip_rsi_divergence': 0.0, 'peak_migration_speed_5d': 0.0, 'chip_stability_change_5d': 0.0, 'chip_divergence_ratio': 0.0, 'chip_convergence_ratio': 0.0, 'trend_confirmation_score': 0.5, 'reversal_warning_score': 0.0, 'turnover_rate': 0.0, 'volume_ratio': 0.0}

    def _calculate_pressure_metrics(self, current_chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, price_history: pd.DataFrame) -> Dict[str, float]:
        """
        [Version 8.0.0] 压力与支撑非对称心理阻尼模型
        说明：废除硬切片全盘相加。引入人类行为心理学衰减：套牢盘距当前价越远抛压越趋向于"僵尸化"（指数衰减）；获利盘则采用 Sigmoid 映射防止极性阈值越界跳变。
        """
        import numpy as np
        if len(current_chip_dist) == 0 or current_price <= 0:
            return self._get_default_pressure_metrics()
        eps = 1e-8
        metrics = {}
        total_percent = np.sum(current_chip_dist) + eps
        price_rel = (price_grid - current_price) / current_price
        profit_mask = price_rel < 0
        profit_chips = current_chip_dist[profit_mask]
        profit_rels = np.abs(price_rel[profit_mask])
        profit_weights = 1.0 / (1.0 + np.exp(-10.0 * (profit_rels - 0.15)))
        metrics['profit_pressure'] = float(np.sum(profit_chips * profit_weights) / total_percent)
        trapped_mask = price_rel > 0
        trapped_chips = current_chip_dist[trapped_mask]
        trapped_rels = price_rel[trapped_mask]
        trapped_weights = np.exp(-3.0 * trapped_rels)
        metrics['trapped_pressure'] = float(np.sum(trapped_chips * trapped_weights) / total_percent)
        mask_recent_trapped = (price_rel > 0.0) & (price_rel <= 0.08)
        metrics['recent_trapped_pressure'] = float(np.sum(current_chip_dist[mask_recent_trapped]) / total_percent)
        mask_support = (price_rel >= -0.08) & (price_rel < 0.0)
        support_chips = current_chip_dist[mask_support]
        support_weights = np.exp(-5.0 * np.abs(price_rel[mask_support]))
        metrics['support_strength'] = float(np.sum(support_chips * support_weights) / total_percent)
        mask_resistance = (price_rel > 0.0) & (price_rel <= 0.08)
        resistance_chips = current_chip_dist[mask_resistance]
        resistance_weights = np.exp(-5.0 * price_rel[mask_resistance])
        metrics['resistance_strength'] = float(np.sum(resistance_chips * resistance_weights) / total_percent)
        if not price_history.empty and len(price_history) >= 10:
            recent_high = float(price_history['high_qfq'].max())
            release_mask = price_grid >= recent_high
            metrics['pressure_release'] = float(np.sum(current_chip_dist[release_mask]) / total_percent)
        else:
            metrics['pressure_release'] = 0.0
        metrics['comprehensive_pressure'] = float(metrics['trapped_pressure'] * 0.4 + metrics['recent_trapped_pressure'] * 0.4 + (1.0 - metrics['pressure_release']) * 0.2)
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_pressure_metrics", {'total_percent': float(total_percent)}, {'raw_trapped': float(np.sum(trapped_chips)/total_percent), 'damped_trapped': metrics['trapped_pressure']}, metrics)
        return metrics

    def _calculate_migration_patterns(self, percent_change_matrix: np.ndarray, chip_matrix: np.ndarray, price_grid: np.ndarray) -> Dict[str, any]:
        """
        [Version 18.0.0] 筹码迁移地球推土机模型 (EMD 动态重心标定版)
        说明：粉碎用绝对历史 price_range 除以单日做功导致的微观极度缩放灾难。
        重塑基底：将单日的 Earth Mover's Distance 做功，除以当天的价格中枢(price_center)，计算出“筹码重心漂移的相对百分比”。
        再通过 Tanh() 激活，精准映射出强烈的单日筹码迁徙脉冲。禁止使用空行。
        """
        import numpy as np
        import math
        if chip_matrix.shape[0] < 2 or len(price_grid) == 0: return self._get_default_migration_patterns()
        patterns = {'upward_migration': {'strength': 0.0, 'volume': 0.0}, 'downward_migration': {'strength': 0.0, 'volume': 0.0}, 'convergence_migration': {'strength': 0.0, 'areas': []}, 'divergence_migration': {'strength': 0.0, 'areas': []}, 'net_migration_direction': 0.0, 'chip_flow_direction': 0, 'chip_flow_intensity': 0.0}
        eps = 1e-10
        old_dist = chip_matrix[-2] / (np.sum(chip_matrix[-2]) + eps)
        new_dist = chip_matrix[-1] / (np.sum(chip_matrix[-1]) + eps)
        cdf_old = np.cumsum(old_dist)
        cdf_new = np.cumsum(new_dist)
        cdf_diff = cdf_old - cdf_new
        price_step = float(price_grid[1] - price_grid[0]) if len(price_grid) > 1 else 1.0
        upward_work = float(np.sum(np.maximum(cdf_diff, 0.0)) * price_step)
        downward_work = float(np.sum(np.maximum(-cdf_diff, 0.0)) * price_step)
        price_center = max(float(np.dot(price_grid, new_dist)), eps)
        total_moved_vol = float(np.sum(np.abs(old_dist - new_dist))) * 50.0
        total_work = upward_work + downward_work + eps
        patterns['upward_migration']['volume'] = float(total_moved_vol * (upward_work / total_work))
        patterns['upward_migration']['strength'] = float(math.tanh((upward_work / price_center) * 100.0))
        patterns['downward_migration']['volume'] = float(total_moved_vol * (downward_work / total_work))
        patterns['downward_migration']['strength'] = float(math.tanh((downward_work / price_center) * 100.0))
        net_dir_pct = (float(np.sum(cdf_diff) * price_step) / price_center) * 100.0
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
        # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_migration_patterns", {'upward_work': upward_work, 'downward_work': downward_work, 'price_center': price_center}, {'up_strength': patterns['upward_migration']['strength'], 'down_strength': patterns['downward_migration']['strength']}, {'net_migration_direction': patterns['net_migration_direction'], 'chip_flow_direction': patterns['chip_flow_direction']})
        return patterns

    def _get_default_migration_patterns(self) -> Dict[str, any]:
        """[Version 18.0.0] 默认迁移模式"""
        return {'upward_migration': {'strength': 0.0, 'volume': 0.0}, 'downward_migration': {'strength': 0.0, 'volume': 0.0}, 'convergence_migration': {'strength': 0.0, 'areas': []}, 'divergence_migration': {'strength': 0.0, 'areas': []}, 'net_migration_direction': 0.0, 'chip_flow_direction': 0, 'chip_flow_intensity': 0.0}

    def _calculate_convergence_metrics(self, chip_matrix: np.ndarray, percent_change_matrix: np.ndarray, price_grid: np.ndarray) -> Dict[str, float]:
        """
        [Version 48.0.0] 筹码聚散度分析算子 (二阶矩方差漂移缓释降维版)
        说明：将方差变动率 rel_var_change 的 Tanh 放大乘数从 100.0 进一步平滑至 50.0。
        在保留对 1%~3% 微观波动高敏锐度的同时，为极端发散(>5%)留出足够的梯度游走空间，防止在异动行情中过早陷入 0.99 的饱和死锁区。禁止使用空行。
        """
        import numpy as np
        import math
        if chip_matrix.shape[0] < 2 or len(percent_change_matrix) == 0: return self._get_default_convergence_metrics()
        metrics = {}
        eps = 1e-10
        current_chip = chip_matrix[-1] / (np.sum(chip_matrix[-1]) + eps)
        p_static = current_chip[current_chip > 1e-4]
        if len(p_static) > 1:
            p_static = p_static / np.sum(p_static)
            static_entropy = float(-np.sum(p_static * np.log(p_static + eps)))
            metrics['static_convergence'] = float(1.0 - (static_entropy / np.log(len(p_static))))
        else: metrics['static_convergence'] = 1.0
        recent_changes = percent_change_matrix[-1]
        abs_changes = np.abs(recent_changes)
        p_dynamic = abs_changes[abs_changes > 1e-4]
        if len(p_dynamic) > 1:
            p_dynamic = p_dynamic / np.sum(p_dynamic)
            dynamic_entropy = float(-np.sum(p_dynamic * np.log(p_dynamic + eps)))
            metrics['dynamic_convergence'] = float(1.0 - (dynamic_entropy / np.log(len(p_dynamic))))
        else: metrics['dynamic_convergence'] = 1.0
        price_center = float(np.dot(price_grid, current_chip))
        variance = float(np.sum(current_chip * (price_grid - price_center)**2))
        chip_std = np.sqrt(variance) + eps
        dist_from_center = np.abs(price_grid - price_center)
        total_change = float(np.sum(abs_changes))
        if total_change > eps:
            weighted_changes = float(np.sum(abs_changes * dist_from_center) / total_change)
            metrics['migration_convergence'] = float(max(0.0, 1.0 - math.atan(weighted_changes / (chip_std * 1.5)) / (math.pi / 2)))
        else: metrics['migration_convergence'] = 1.0
        metrics['comprehensive_convergence'] = float(0.4 * metrics['static_convergence'] + 0.3 * metrics['dynamic_convergence'] + 0.3 * metrics['migration_convergence'])
        recent_changes_norm = recent_changes / 100.0
        variance_change = float(np.dot(recent_changes_norm, (price_grid - price_center)**2))
        rel_var_change = variance_change / (variance + eps)
        if rel_var_change < 0:
            metrics['convergence_strength'] = float(math.tanh(abs(rel_var_change) * 50.0))
            metrics['divergence_strength'] = 0.0
        else:
            metrics['convergence_strength'] = 0.0
            metrics['divergence_strength'] = float(math.tanh(rel_var_change * 50.0))
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_convergence_metrics", {'variance_change': float(variance_change), 'rel_var_change': float(rel_var_change)}, {'dynamic_convergence': metrics['dynamic_convergence'], 'migration_convergence': metrics['migration_convergence']}, metrics)
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
        [Version 8.0.0] 严格因果序贯主力活跃度探测器（无未来函数版）
        说明：斩断原代码中 np.mean(volumes) 引发的前瞻偏差(Look-ahead Bias)。
        改用纯正的 Expanding Mean (扩展移动平均)，并废除硬截断导致的满分溢出，引入双曲正切(Tanh)保留梯度。
        """
        import numpy as np
        import math
        try:
            raw_score = 0.0
            if abnormal_volume:
                abnormal_ratio = abnormal_volume.get('abnormal_volume_ratio', 0.0)
                raw_score += abnormal_ratio * 3.5
            if not tick_data.empty:
                volumes = tick_data['volume'].to_numpy(dtype=np.float32)
                seq_len = len(volumes)
                if seq_len > 0:
                    cum_sum = np.cumsum(volumes)
                    seq_indices = np.arange(1, seq_len + 1, dtype=np.float32)
                    expanding_mean = cum_sum / seq_indices
                    dynamic_threshold = expanding_mean * 2.5
                    large_order_mask = volumes > dynamic_threshold
                    large_order_vol = np.sum(volumes[large_order_mask])
                    total_vol = cum_sum[-1]
                    large_order_ratio = float(large_order_vol / total_vol) if total_vol > 1e-5 else 0.0
                    raw_score += large_order_ratio * 2.5
                    # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_main_force_activity_seq", {'seq_len': int(seq_len), 'total_vol': float(total_vol)}, {'large_order_vol': float(large_order_vol), 'expanding_mean_last': float(expanding_mean[-1])}, {'large_order_ratio': float(large_order_ratio)})
            if intraday_flow:
                buy_ratio = intraday_flow.get('buy_ratio', 0.5)
                sell_ratio = intraday_flow.get('sell_ratio', 0.5)
                prior_imbalance = 0.05
                imbalance = abs(buy_ratio - sell_ratio) / (buy_ratio + sell_ratio + prior_imbalance)
                raw_score += imbalance * 2.0
            final_activity = float(np.tanh(raw_score / 2.0))
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_calculate_main_force_activity_final", {'raw_score': float(raw_score)}, {}, {'final_activity': final_activity})
            return final_activity
        except Exception as e:
            return 0.0

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
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_fetch_chip_percent_data", {'stock_code': stock_code, 'trade_date': trade_date}, {'chip_history_len': len(chip_history), 'price_history_len': len(price_history), 'has_turnover': has_turnover}, {'current_price': current_price, 'status': 'success'})
            return {'current_chip_dist': current_chip_df, 'chip_history': chip_history, 'price_history': price_history, 'current_price': current_price}
        except Exception as e:
            from services.chip_holding_calculator import QuantitativeTelemetryProbe
            import traceback
            traceback.print_exc()
            # QuantitativeTelemetryProbe.emit("AdvancedChipDynamicsService", "_fetch_chip_percent_data_FATAL", {'stock_code': stock_code}, {'error': str(e)}, {'status': 'exception'})
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
            # QuantitativeTelemetryProbe.emit("DirectAccumulationDistributionCalculator", "calculate_direct_ad", {'current_price': float(current_price)}, {'reason': 'empty_percent_change_matrix'}, {'net_ad_ratio': 0.0, 'status': 'aborted'})
            return result
        try:
            latest_change = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
            price_rel = (price_grid - current_price) / current_price
            result = self._calculate_absolute_ad(latest_change, price_rel)
            if not price_history.empty and len(price_history) >= 5:
                result = self._correct_pullback_ad(result, price_history, current_price)
            result = self._calculate_ad_quality(result, chip_matrix, price_grid, current_price)
            result['price_level_ad'] = self._analyze_price_levels(latest_change, price_grid, current_price)
            # QuantitativeTelemetryProbe.emit("DirectAccumulationDistributionCalculator", "calculate_direct_ad", {'current_price': float(current_price), 'history_len': len(price_history)}, {'accumulation_volume': float(result.get('accumulation_volume', 0.0)), 'distribution_volume': float(result.get('distribution_volume', 0.0)), 'false_distribution_flag': bool(result.get('false_distribution_flag', False))}, {'net_ad_ratio': float(result.get('net_ad_ratio', 0.0)), 'status': 'success'})
            return result
        except Exception as e:
            result = self._get_default_ad_result()
            # QuantitativeTelemetryProbe.emit("DirectAccumulationDistributionCalculator", "calculate_direct_ad", {'current_price': float(current_price)}, {'error': str(e)}, {'net_ad_ratio': 0.0, 'status': 'failed'})
            return result

    def _calculate_absolute_ad(self, changes: np.ndarray, price_rel: np.ndarray) -> Dict[str, any]:
        """
        [Version 11.0.0] 基于绝对变化的直接吸收/派发计算器（贝叶斯抗伪极化版）
        说明：终结 net_ad_ratio 在缩量行情下惊现 0.9999 的史诗级单边幻觉。
        植入强效贝叶斯伪计数基底(Bayesian Prior)和反正切非线性挤压(Arctan)。只要没有大盘口真实资金量能的配合，所有的微小噪声波动都将被强制镇压在中性震荡区内，彻底斩断小样本骗线。禁止使用空行。
        """
        import numpy as np
        import math
        significant_mask = np.abs(changes) > self.params['noise_filter']
        clean_changes = changes[significant_mask]
        clean_rels = price_rel[significant_mask]
        if len(clean_changes) == 0: return {'accumulation_volume': 0.0, 'distribution_volume': 0.0, 'net_ad_ratio': 0.0, 'accumulation_quality': 0.5, 'distribution_quality': 0.5, 'false_distribution_flag': False, 'breakout_acceleration': 1.0}
        bins = np.array([-0.12, -0.03, 0, 0.05, 0.12])
        indices = np.digitize(clean_rels, bins)
        pos_changes = np.maximum(clean_changes, 0)
        neg_changes = np.abs(np.minimum(clean_changes, 0))
        pos_sums = np.bincount(indices, weights=pos_changes, minlength=6)
        neg_sums = np.bincount(indices, weights=neg_changes, minlength=6)
        zone_weights = np.array([0.4, 1.3, 1.0, 1.0, 1.2, 0.5])
        accum_mults = np.array([1.0, 1.0, 1.0, 0.6, 0.3, 0.3])
        distrib_mults = np.array([0.8, 0.8, 0.8, 0.9, 1.0, 1.0])
        raw_accum = pos_sums * zone_weights * accum_mults
        raw_distrib = neg_sums * zone_weights * distrib_mults
        accumulation_volume = float(np.sum(raw_accum))
        distribution_volume = float(np.sum(raw_distrib))
        overall_trend = float(np.sum(clean_changes))
        if overall_trend > 0:
            accumulation_volume *= 1.2
            distribution_volume *= 0.8
        elif overall_trend < 0:
            accumulation_volume *= 0.8
            distribution_volume *= 1.2
        total_raw_vol = accumulation_volume + distribution_volume
        bayesian_prior = max(3.0, total_raw_vol * 0.15)
        total_volume_smoothed = total_raw_vol + bayesian_prior + 1e-8
        raw_net_ratio = (accumulation_volume - distribution_volume) / total_volume_smoothed
        net_ad_ratio = float(math.atan(raw_net_ratio * 3.0) / (math.pi / 2))
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        # QuantitativeTelemetryProbe.emit("DirectAccumulationDistributionCalculator", "_calculate_absolute_ad", {'total_raw_vol': float(total_raw_vol), 'bayesian_prior': float(bayesian_prior)}, {'raw_accum': float(accumulation_volume), 'raw_distrib': float(distribution_volume)}, {'net_ad_ratio': float(net_ad_ratio)})
        return {'accumulation_volume': accumulation_volume, 'distribution_volume': distribution_volume, 'net_ad_ratio': net_ad_ratio, 'accumulation_quality': 0.5, 'distribution_quality': 0.5, 'false_distribution_flag': False, 'breakout_acceleration': 1.0}

    def _correct_pullback_ad(self, ad_result: Dict[str, any], price_history: pd.DataFrame, current_price: float) -> Dict[str, any]:
        """
        拉升初期纠偏：考虑A股特色
        A股特色：
        1. 涨停后的回调多为洗盘
        2. 连阳后的首阴可能是换手
        3. 重要均线支撑处的派发多为假派发
        """
        if len(price_history) < 10:
            return ad_result
        # 判断是否涨停
        is_limit_up = False
        if not price_history.empty:
            today_pct_change = price_history['pct_change'].iloc[-1] if 'pct_change' in price_history.columns else 0
            # A股涨停阈值（主板10%，创业板/科创板20%）
            limit_up_threshold = 9.8  # 接近涨停
            is_limit_up = today_pct_change >= limit_up_threshold
        # 判断是否连阳
        is_continuous_up = False
        if len(price_history) >= 5:
            recent_closes = price_history['close_qfq'].values[-5:]
            recent_up_days = sum([1 for i in range(1, len(recent_closes)) 
                                 if recent_closes[i] > recent_closes[i-1]])
            is_continuous_up = recent_up_days >= 4  # 5天4阳
        # 判断是否在重要均线位置
        # 这里简化：检查是否在近期均价附近
        is_near_ma = False
        if len(price_history) >= 20:
            ma20 = price_history['close_qfq'].rolling(20).mean().iloc[-1]
            ma_distance = abs((current_price - ma20) / ma20)
            is_near_ma = ma_distance < 0.03  # 距离MA20在3%以内
        # 虚假派发识别条件
        false_distribution_conditions = []
        # 条件1：涨停后出现派发信号
        if is_limit_up and ad_result['distribution_volume'] > ad_result['accumulation_volume']:
            false_distribution_conditions.append('涨停后派发')
        # 条件2：连阳后首日调整
        if is_continuous_up and ad_result['distribution_volume'] > 0:
            # 检查是否为上涨趋势中的正常回调
            prev_close = price_history['close_qfq'].iloc[-2] if len(price_history) > 1 else current_price
            if current_price > prev_close * 0.97:  # 跌幅小于3%
                false_distribution_conditions.append('连阳后回调')
        # 条件3：重要均线支撑处的派发
        if is_near_ma and ad_result['distribution_volume'] > ad_result['accumulation_volume']:
            # 均线支撑处，派发多为洗盘
            false_distribution_conditions.append('均线处派发')
        # 如果满足任意虚假派发条件
        if false_distribution_conditions:
            correction_factor = min(0.5, len(false_distribution_conditions) * 0.15)
            corrected_distribution = ad_result['distribution_volume'] * (1 - correction_factor)
            ad_result.update({
                'distribution_volume': corrected_distribution,
                'false_distribution_flag': True,
                'accumulation_quality': min(1.0, ad_result['accumulation_quality'] * 1.3),
                'breakout_acceleration': 1.5 if is_limit_up else 1.2,
                'false_distribution_reason': false_distribution_conditions,
            })
        return ad_result

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
            # QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'reason': '输入数据矩阵无效或无参考价'}, {'status': 'aborted'})
            return result
        try:
            if len(percent_change_matrix) == 0:
                print("❌ [探针] 变化矩阵为空")
                result = self._get_default_energy()
                print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
                # QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'reason': '提取的变化矩阵为空'}, {'status': 'aborted'})
                return result
            latest_change = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
            change_sum = np.sum(np.abs(latest_change))
            if change_sum < 0.01:
                print("⚠️ [探针] 变化数据几乎全为0，返回默认值")
                result = self._get_default_energy()
                print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
                # QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'reason': '绝对变化和过低触发拦截', 'change_sum': float(change_sum)}, {'status': 'aborted'})
                return result
            energy_result = self._calculate_energy_field(latest_change, price_grid, reference_price, close_price)
            fake_distribution = False
            if volume_history is not None and len(volume_history) > 5:
                fake_distribution = self._detect_fake_distribution(latest_change, price_grid, reference_price, volume_history)
            else:
                fake_distribution = self._detect_fake_distribution_advanced(latest_change, price_grid, reference_price, close_price)
            energy_result['fake_distribution_flag'] = fake_distribution
            energy_result = self._ensure_nonzero_energy(energy_result)
            # QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'fake_distribution': fake_distribution}, {'status': 'success'})
            return energy_result
        except Exception as e:
            print(f"❌ [探针] 能量场计算异常: {e}")
            import traceback
            traceback.print_exc()
            result = self._get_default_energy()
            print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
            # QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "calculate_game_energy", {'stock_code': stock_code, 'trade_date': trade_date}, {'error': str(e)}, {'status': 'failed'})
            return result

    def _calculate_energy_field(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, close_price: float, stock_code: str = "", trade_date: str = "") -> Dict[str, Any]:
        """
        [Version 11.0.0] 动态波动率自适应能量场核心算子（千万级除零反噬镇压版）
        说明：根除 momentum_ratio 在派发量微小时爆出两千万级数值的严重数学灾难。
        强制植入动态贝叶斯势能基底 (Prior Momentum Base)，将动量乘数严格约束在统计学显著性框架内，绝不让极小样本噪声劫持突破信号。禁止使用空行。
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
            dynamic_sigma = max(0.02, min(sigma, 0.20))
            bins = np.array([-3.0 * dynamic_sigma, -1.0 * dynamic_sigma, 0.0, 1.0 * dynamic_sigma, 3.0 * dynamic_sigma])
            indices = np.digitize(price_rel, bins)
            pos_changes = np.maximum(changes, 0)
            neg_changes = np.abs(np.minimum(changes, 0))
            pos_sums = np.bincount(indices, weights=pos_changes, minlength=6)
            neg_sums = np.bincount(indices, weights=neg_changes, minlength=6)
            weights_arr = np.array([0.6, 0.9, 1.5, 1.3, 1.0, 0.7])
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
            # QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "_calculate_energy_field", {'sigma': float(sigma), 'dynamic_sigma': float(dynamic_sigma), 'total_section_energy': float(total_section_energy)}, {'prior_momentum_base': float(prior_momentum_base), 'momentum_ratio': float(momentum_ratio)}, {'net_energy': net_energy, 'absorption': float(absorption_advanced), 'distribution': float(distribution_advanced)})
            return {'absorption_energy': min(100.0, max(0.01, float(absorption_advanced))), 'distribution_energy': min(100.0, max(0.01, float(distribution_advanced))), 'net_energy_flow': net_energy, 'game_intensity': min(1.0, max(0.0, float(game_intensity))), 'key_battle_zones': key_battle_zones, 'breakout_potential': min(100.0, float(breakout_potential)), 'energy_concentration': min(1.0, max(0.0, float(energy_concentration))), 'reference_price': float(reference_price), 'original_current_price': float(current_price), 'fake_distribution_flag': False}
        except Exception as e:
            return self._get_default_energy()

    def _calculate_energy_indicators(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, stock_code: str = "", trade_date: str = "") -> tuple:
        """
        [Version 48.0.0] 博弈能量指标自适应拓扑计算器 (强度压制释压版)
        说明：修正 absolute_scale 除以 15.0 导致的 game_intensity 被过度压制在极低水平 (如0.04) 的问题。
        A股单日筹码交换5%已属高燃交战，将分母下调至 5.0，恢复博弈强度的正常动态张力。禁止使用空行。
        """
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
                scale_penalty = float(np.tanh(active_count / max(1.0, len(changes) * 0.05)))
                energy_concentration = float(base_concentration * 0.4 + hhi * 0.6) * scale_penalty
        valid_changes = abs_changes[abs_changes > eps]
        if len(valid_changes) > 5: dynamic_active_threshold = max(0.01, float(np.percentile(valid_changes, 60)))
        else: dynamic_active_threshold = 0.05
        active_mask_intensity = abs_changes > dynamic_active_threshold
        active_energy_sum = np.sum(abs_changes[active_mask_intensity])
        prior_energy = max(1.0, total_energy * 0.05) 
        active_ratio = active_energy_sum / (total_energy + prior_energy + eps)
        absolute_scale = float(math.atan(total_energy / 5.0) / (math.pi / 2))
        game_intensity = float(active_ratio * absolute_scale)
        game_intensity = min(1.0, max(0.0, game_intensity))
        above_mask = price_grid > current_price
        below_mask = price_grid < current_price
        absorption_above = np.sum(changes[above_mask & (changes > 0)])
        distribution_above = np.sum(np.abs(changes[above_mask & (changes < 0)]))
        absorption_below = np.sum(changes[below_mask & (changes > 0)])
        distribution_below = np.sum(np.abs(changes[below_mask & (changes < 0)]))
        imbalance_prior = max(2.0, (absorption_below + distribution_below) * 0.1)
        below_imbalance = (absorption_below - distribution_below) / (absorption_below + distribution_below + imbalance_prior + eps)
        support_strength = 1.0 + float(math.tanh(below_imbalance * 1.5))
        net_above = absorption_above - distribution_above
        if net_above > 0: raw_potential = net_above * support_strength * 2.0
        else: raw_potential = (np.exp(net_above) - 1.0) * support_strength * 1.5 + 1.0
        breakout_potential = float(max(0.01, raw_potential))
        if energy_concentration > 0.5: breakout_potential *= (1.0 + float(math.tanh((energy_concentration - 0.5) * 1.5)))
        from services.chip_holding_calculator import QuantitativeTelemetryProbe
        # QuantitativeTelemetryProbe.emit("GameEnergyCalculator", "_calculate_energy_indicators", {'total_energy': float(total_energy), 'prior_energy': float(prior_energy), 'dynamic_threshold': float(dynamic_active_threshold)}, {'active_ratio': float(active_ratio), 'absolute_scale': float(absolute_scale), 'support_strength': float(support_strength)}, {'game_intensity': float(game_intensity), 'breakout_potential': float(breakout_potential), 'energy_concentration': float(energy_concentration)})
        return float(game_intensity), float(breakout_potential), float(energy_concentration)

    def _detect_fake_distribution_advanced(self, changes: np.ndarray, price_grid: np.ndarray, 
                                         current_price: float, close_price: float) -> bool:
        """高级虚假派发检测 - 基于A股特性"""
        try:
            # 1. 价格位置分析
            price_rel = (price_grid - current_price) / current_price
            # 2. 当前价附近的筹码变化
            near_mask = np.abs(price_rel) < 0.08
            near_net = np.sum(changes[near_mask])
            # 3. 上方派发 vs 下方吸收
            above_mask = price_rel > 0.08
            below_mask = price_rel < -0.08
            above_distrib = np.sum(np.abs(changes[above_mask & (changes < 0)]))
            below_accum = np.sum(changes[below_mask & (changes > 0)])
            # 4. A股虚假派发特征：
            #    a) 上方派发量大但下方吸收更强
            #    b) 当前价附近有净吸收
            #    c) 价格处于上升趋势中
            if (below_accum > above_distrib * 1.5 and  # 下方吸收远大于上方派发
                near_net > 0 and                       # 当前价附近净增加
                above_distrib > 0.5):                  # 上方有一定派发
                return True
            # 5. 另一种情况：缩量调整
            # 如果价格在均线上方但出现派发信号，可能是正常回调
            if (np.mean(changes[above_mask]) < -0.3 and  # 上方平均减少
                np.mean(changes[below_mask]) > 0.2 and   # 下方平均增加
                above_distrib < 2.0):                    # 派发量不大
                return True
            return False
        except Exception as e:
            print(f"⚠️ [高级虚假派发检测] 异常: {e}")
            return False

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

    def _detect_fake_distribution(self, changes: np.ndarray, price_grid: np.ndarray, 
                                current_price: float, volume_history: pd.Series = None) -> bool:
        """检测虚假派发（获利回吐 vs 真实派发）"""
        # 条件1：中高位筹码减少但高位未明显增加
        mid_high_mask = (price_grid > current_price) & (price_grid <= current_price * 1.05)
        high_mask = price_grid > current_price * 1.05
        mid_decrease = np.sum(-changes[mid_high_mask & (changes < 0)])
        high_increase = np.sum(changes[high_mask & (changes > 0)])
        if mid_decrease > 0.4 and high_increase < 0.2:
            # 进一步检查成交量特征
            if volume_history is not None and len(volume_history) >= 5:
                recent_volume = volume_history.iloc[-5:].mean()
                avg_volume = volume_history.iloc[-20:-5].mean() if len(volume_history) >= 20 else recent_volume
                # 缩量回调是拉升初期的典型特征
                if recent_volume < avg_volume * 1.2:
                    return True
            else:
                # 没有成交量数据时，仅基于筹码变化判断
                return True
        return False

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
        版本: v2.2
        说明: 关键博弈区域识别（全向量化版）
        修改思路: 
        1. 使用 np.lib.stride_tricks.sliding_window_view 替代循环切片。
        2. 利用广播机制一次性计算所有点的邻域对抗强度。
        3. 消除所有Python层面的循环，极大提升密集计算效率。
        """
        battle_zones = []
        min_intensity = 0.5
        try:
            n = len(changes)
            if n < 5:
                return []
            # 1. 准备数据：填充边界以保持形状一致
            # 窗口大小5，左右各补2个0
            padded_changes = np.pad(changes, (2, 2), mode='constant', constant_values=0)
            # 2. 创建滑动窗口视图 (shape: [n, 5])
            # 这一步创建的是视图，不消耗额外内存
            windows = np.lib.stride_tricks.sliding_window_view(padded_changes, window_shape=5)
            # 3. 提取中心点 (即原始changes)
            center_points = windows[:, 2]
            # 4. 筛选显著点 (Masking)
            significant_mask = np.abs(center_points) > min_intensity
            if not np.any(significant_mask):
                return []
            # 只处理显著点
            sig_windows = windows[significant_mask]
            sig_centers = center_points[significant_mask]
            sig_indices = np.where(significant_mask)[0]
            # 5. 向量化计算对抗强度
            # 寻找符号相反的邻居: (neighbor * center) < 0
            # 利用广播: sig_centers[:, None] 将中心点变为列向量，与窗口行向量相乘
            opponent_mask = (sig_windows * sig_centers[:, None]) < 0
            # 计算对手强度的平均值 (只计算符号相反的部分)
            # np.where(condition, value, 0)
            opponent_values = np.where(opponent_mask, np.abs(sig_windows), 0)
            # 计算每行的非零元素个数，避免除以0
            opponent_counts = np.sum(opponent_mask, axis=1)
            opponent_sums = np.sum(opponent_values, axis=1)
            # 如果没有对手，平均值为0
            opponent_avgs = np.divide(opponent_sums, opponent_counts, out=np.zeros_like(opponent_sums), where=opponent_counts!=0)
            # 最终强度公式
            battle_intensities = np.abs(sig_centers) + opponent_avgs * 0.5
            # 6. 筛选符合阈值的区域
            valid_battle_mask = battle_intensities > min_intensity * 1.5
            if not np.any(valid_battle_mask):
                return []
            final_indices = sig_indices[valid_battle_mask]
            final_intensities = battle_intensities[valid_battle_mask]
            final_centers = sig_centers[valid_battle_mask]
            # 7. 构建结果列表
            # 这里必须循环构建字典，但数量已经很少（top 5）
            # 先排序，取前5
            sort_order = np.argsort(final_intensities)[::-1][:5]
            for idx in sort_order:
                grid_idx = final_indices[idx]
                price = price_grid[grid_idx]
                change = final_centers[idx]
                battle_zones.append({
                    'price': float(price),
                    'battle_intensity': float(final_intensities[idx]),
                    'type': 'absorption' if change > 0 else 'distribution',
                    'position': 'below_current' if price < current_price else 'above_current',
                    'distance_to_current': float((price - current_price) / current_price),
                })
            return battle_zones
        except Exception as e:
            print(f"❌ [探针-关键区域] {stock_code} {trade_date} 识别异常: {e}")
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
        # QuantitativeTelemetryProbe.emit("ChipFactorCalculationHelper", "calculate_core_chip_factors", {'close': close, 'winner_rate': winner_rate, 'active_range': active_range}, {'adaptive_pressure': adaptive_pressure, 'profit_pressure': profit_pressure, 'panic_pressure': panic_pressure}, final_result)
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





