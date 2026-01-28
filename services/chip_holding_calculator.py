# services/chip_holding_calculator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import asyncio
from asgiref.sync import sync_to_async
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
        self.price_granularity = 200  # 价格粒度
        # 中国A股特定参数
        # 修正：降低阈值以适应 200 格的价格粒度 (平均每格仅 0.5%)
        self.params = {
            'significant_change_threshold': 0.5,  # 显著变化阈值（%）原 2.0
            'noise_threshold': 0.1,              # 噪声阈值（%）原 1.0
            'institution_min_change': 1.0,       # 机构行为最小变化（%）原 5.0
            'main_force_concentration': 0.3,     # 主力控盘集中度阈值
            'retail_scatter_threshold': 0.7,     # 散户分散阈值
            'accumulation_days': 5,              # 吸筹天数判定
            'distribution_days': 3,              # 派发天数判定
        }
   
    async def analyze_chip_dynamics_daily(self, stock_code: str, trade_date: str, lookback_days: int = 20) -> Dict[str, any]:
        """
        分析单日筹码动态 - 主入口函数
        """
        try:
            # 1. 获取筹码分布历史数据（包含百分比）
            chip_data = await self._fetch_chip_percent_data(
                stock_code, trade_date, lookback_days
            )
            # PROBE: 检查数据获取结果
            history_len = len(chip_data['chip_history']) if chip_data else 0
            if not chip_data or len(chip_data['chip_history']) < 5:
                print(f"⚠️ [PROBE-WARN] 数据不足 (历史天数 {history_len} < 5)，返回默认结果")
                return self._get_default_result(stock_code, trade_date)
            # 2. 构建价格网格和归一化筹码矩阵
            price_grid, chip_matrix = self._build_normalized_chip_matrix(
                chip_data['chip_history'],
                chip_data['current_chip_dist']
            )
            # 3. 计算百分比变化矩阵（核心）
            percent_change_matrix = self._calculate_percent_change_matrix(chip_matrix)
            # 4. 基于绝对变化的行为分析
            absolute_signals = self._analyze_absolute_changes(
                percent_change_matrix,
                price_grid,
                chip_data['current_price']
            )
            # 5. 计算筹码集中度指标
            concentration_metrics = self._calculate_concentration_metrics(
                chip_matrix[-1],  # 当前筹码分布
                price_grid
            )
            # 6. 计算压力与支撑指标
            pressure_metrics = self._calculate_pressure_metrics(
                chip_matrix[-1],
                price_grid,
                chip_data['current_price'],
                chip_data['price_history']
            )
            # 7. 识别主力行为模式
            behavior_patterns = self._identify_behavior_patterns(
                percent_change_matrix,
                chip_matrix,
                price_grid,
                chip_data['current_price']
            )
            # 8. 计算筹码迁移模式
            migration_patterns = self._calculate_migration_patterns(
                percent_change_matrix,
                chip_matrix,
                price_grid
            )
            # 9. 计算综合聚散度
            convergence_metrics = self._calculate_convergence_metrics(
                chip_matrix,
                percent_change_matrix,
                price_grid
            )
            # =======================================================
            # 新增：构建验证信息
            # =======================================================
            validation_warnings = []
            # 基于信号质量作为基础分
            validation_score = absolute_signals.get('signal_quality', 0.5)
            # 检查数据长度
            if history_len < lookback_days:
                validation_warnings.append(f"历史数据不足: {history_len}/{lookback_days}")
                validation_score *= 0.8
            # 检查价格覆盖
            current_price = chip_data['current_price']
            if current_price > price_grid.max() or current_price < price_grid.min():
                validation_warnings.append("当前价格超出网格范围")
                validation_score *= 0.9

            result = {
                'stock_code': stock_code,
                'trade_date': trade_date,
                'price_grid': price_grid.tolist(),
                # 新增：返回原始筹码矩阵（用于 matrix_data）
                'chip_matrix': chip_matrix.tolist(),
                'percent_change_matrix': percent_change_matrix.tolist(),
                'absolute_change_signals': absolute_signals,
                'concentration_metrics': concentration_metrics,
                'pressure_metrics': pressure_metrics,
                'behavior_patterns': behavior_patterns,
                'migration_patterns': migration_patterns,
                'convergence_metrics': convergence_metrics,
                # 新增：验证字段
                'validation_score': round(validation_score, 2),
                'validation_warnings': validation_warnings,
                'analysis_status': 'success',
                'analysis_time': datetime.now().isoformat()
            }
            return result
        except Exception as e:
            logger.error(f"筹码动态分析失败 {stock_code} {trade_date}: {e}")
            print(f"❌ [筹码动态分析异常] {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_result(stock_code, trade_date)

    def _build_normalized_chip_matrix(self, chip_history: List[pd.DataFrame], current_chip: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        版本: v2.0
        说明: 构建归一化筹码矩阵（向量化优化版）
        优化: 使用np.interp替代scipy.interpolate.interp1d，移除对象构建开销
        """
        all_chips = chip_history + [current_chip]
        all_prices = []
        for chip_df in all_chips:
            if not chip_df.empty:
                all_prices.extend(chip_df['price'].values)
        if not all_prices:
            min_price, max_price = 1.0, 100.0
        else:
            min_price, max_price = np.min(all_prices), np.max(all_prices)
            price_range = max_price - min_price
            min_price = max(0.01, min_price - price_range * 0.15)
            max_price = max_price + price_range * 0.15
        price_grid = np.linspace(min_price, max_price, self.price_granularity)
        n_days = len(all_chips)
        n_prices = len(price_grid)
        chip_matrix = np.zeros((n_days, n_prices))
        uniform_dist = np.full(n_prices, 100.0 / n_prices)
        for day_idx, chip_df in enumerate(all_chips):
            if chip_df.empty or len(chip_df) < 3:
                chip_matrix[day_idx, :] = uniform_dist
                continue
            try:
                valid_mask = (~chip_df['price'].isna()) & (~chip_df['percent'].isna())
                df_valid = chip_df[valid_mask].sort_values('price').drop_duplicates('price')
                if len(df_valid) < 2:
                    chip_matrix[day_idx, :] = uniform_dist
                    continue
                x = df_valid['price'].values
                y = df_valid['percent'].values
                total_p = np.sum(y)
                if total_p == 0:
                    chip_matrix[day_idx, :] = uniform_dist
                    continue
                y_normalized = y * (100.0 / total_p)
                interpolated = np.interp(price_grid, x, y_normalized, left=0.0, right=0.0)
                sum_interp = np.sum(interpolated)
                if sum_interp > 0:
                    chip_matrix[day_idx, :] = interpolated * (100.0 / sum_interp)
                else:
                    chip_matrix[day_idx, :] = uniform_dist
            except Exception as e:
                print(f"⚠️ 第{day_idx}天插值失败: {e}")
                chip_matrix[day_idx, :] = uniform_dist
        return price_grid, chip_matrix

    def _calculate_percent_change_matrix(self, chip_matrix: np.ndarray) -> np.ndarray:
        """
        版本: v2.0
        说明: 计算百分比变化矩阵（完全向量化版）
        优化: 移除Python循环，使用Numpy切片相减
        """
        if chip_matrix.shape[0] < 2:
            return np.zeros((1, chip_matrix.shape[1]))
        return chip_matrix[1:] - chip_matrix[:-1]

    def _analyze_absolute_changes(self, percent_change_matrix: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, any]:
        """
        版本: v2.0
        说明: 基于绝对变化的信号分析（向量化筛选版）
        优化: 移除对价格网格的循环，使用Numpy掩码和where进行批量筛选
        """
        if percent_change_matrix.shape[0] == 0:
            return self._get_default_absolute_signals()
        recent_changes = percent_change_matrix[-min(3, len(percent_change_matrix)):, :]
        avg_changes = np.mean(recent_changes, axis=0) if recent_changes.shape[0] > 0 else np.zeros_like(price_grid)
        increase_mask = avg_changes > self.params['significant_change_threshold']
        decrease_mask = avg_changes < -self.params['significant_change_threshold']
        noise_mask = np.abs(avg_changes) < self.params['noise_threshold']
        signals = {
            'significant_increase_areas': [],
            'significant_decrease_areas': [],
            'accumulation_signals': [],
            'distribution_signals': [],
            'noise_level': float(np.mean(noise_mask)),
            'signal_quality': 1.0 - float(np.mean(noise_mask))
        }
        dist_to_current = np.abs(price_grid - current_price) / (current_price if current_price > 0 else 1.0)
        inc_indices = np.where(increase_mask)[0]
        for idx in inc_indices:
            signals['significant_increase_areas'].append({
                'price': float(price_grid[idx]),
                'change': float(avg_changes[idx]),
                'distance_to_current': float(dist_to_current[idx])
            })
            if price_grid[idx] < current_price * 0.95:
                signals['accumulation_signals'].append({
                    'price': float(price_grid[idx]),
                    'change': float(avg_changes[idx]),
                    'strength': min(1.0, float(avg_changes[idx]) / 10.0)
                })
        dec_indices = np.where(decrease_mask)[0]
        for idx in dec_indices:
            signals['significant_decrease_areas'].append({
                'price': float(price_grid[idx]),
                'change': float(avg_changes[idx]),
                'distance_to_current': float(dist_to_current[idx])
            })
            if price_grid[idx] > current_price * 1.05:
                signals['distribution_signals'].append({
                    'price': float(price_grid[idx]),
                    'change': float(avg_changes[idx]),
                    'strength': min(1.0, abs(float(avg_changes[idx])) / 10.0)
                })
        for key in ['significant_increase_areas', 'significant_decrease_areas', 'accumulation_signals', 'distribution_signals']:
            signals[key] = sorted(signals[key], key=lambda x: abs(x['change']), reverse=True)[:10]
        return signals

    def _calculate_concentration_metrics(self, current_chip_dist: np.ndarray, price_grid: np.ndarray) -> Dict[str, float]:
        """
        版本: v2.0
        说明: 计算集中度指标（精简计算版）
        优化: 优化数学计算流程，减少临时列表创建
        """
        if len(current_chip_dist) == 0:
            return self._get_default_concentration_metrics()
        metrics = {}
        chip_prob = current_chip_dist + 1e-10
        metrics['entropy_concentration'] = 1.0 - (entropy(chip_prob) / np.log(len(current_chip_dist)))
        sorted_chip = np.sort(current_chip_dist)[::-1]
        top_20_idx = int(len(current_chip_dist) * 0.2)
        metrics['peak_concentration'] = np.sum(sorted_chip[:top_20_idx]) / 100.0
        price_mean = np.dot(price_grid, current_chip_dist) / 100.0
        if price_mean > 0:
            price_std = np.sqrt(np.dot(current_chip_dist, (price_grid - price_mean) ** 2) / 100.0)
            metrics['cv_concentration'] = 1.0 - min(1.0, price_std / price_mean)
            mask_main = (price_grid >= price_mean * 0.9) & (price_grid <= price_mean * 1.1)
            metrics['main_force_concentration'] = np.sum(current_chip_dist[mask_main]) / 100.0
        else:
            metrics['cv_concentration'] = 0.5
            metrics['main_force_concentration'] = 0.0
        metrics['comprehensive_concentration'] = (
            0.3 * metrics['entropy_concentration'] +
            0.3 * metrics['peak_concentration'] +
            0.2 * metrics['cv_concentration'] +
            0.2 * metrics['main_force_concentration']
        )
        metrics['chip_skewness'] = float(skew(current_chip_dist))
        metrics['chip_kurtosis'] = float(kurtosis(current_chip_dist))
        return metrics

    def _calculate_pressure_metrics(self, current_chip_dist: np.ndarray, price_grid: np.ndarray, current_price: float, price_history: pd.DataFrame) -> Dict[str, float]:
        """
        版本: v2.0
        说明: 计算压力指标（向量化掩码版）
        优化: 统一掩码计算逻辑，确保历史数据计算高效
        """
        if len(current_chip_dist) == 0 or current_price <= 0:
            return self._get_default_pressure_metrics()
        metrics = {}
        total_percent = 100.0
        metrics['profit_pressure'] = np.sum(current_chip_dist[price_grid < current_price]) / total_percent
        metrics['trapped_pressure'] = np.sum(current_chip_dist[price_grid > current_price * 1.10]) / total_percent
        mask_recent_trapped = (price_grid > current_price * 1.05) & (price_grid <= current_price * 1.10)
        metrics['recent_trapped_pressure'] = np.sum(current_chip_dist[mask_recent_trapped]) / total_percent
        mask_support = (price_grid >= current_price * 0.95) & (price_grid < current_price)
        metrics['support_strength'] = np.sum(current_chip_dist[mask_support]) / total_percent
        mask_resistance = (price_grid > current_price) & (price_grid <= current_price * 1.05)
        metrics['resistance_strength'] = np.sum(current_chip_dist[mask_resistance]) / total_percent
        if not price_history.empty and len(price_history) >= 10:
            recent_low = float(price_history['low'].min())
            recent_high = float(price_history['high'].max())
            mask_released = (price_grid >= recent_high * 1.05) | (price_grid <= recent_low * 0.95)
            metrics['pressure_release'] = np.sum(current_chip_dist[mask_released]) / total_percent
        else:
            metrics['pressure_release'] = 0.0
        metrics['comprehensive_pressure'] = (
            metrics['trapped_pressure'] * 0.5 +
            metrics['recent_trapped_pressure'] * 0.3 +
            (1 - metrics['pressure_release']) * 0.2
        )
        return metrics
    
    def _identify_behavior_patterns(self, percent_change_matrix: np.ndarray, chip_matrix: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, any]:
        """
        版本: v2.0
        说明: 识别主力行为模式（向量化逻辑版）
        优化: 移除外层价格循环，使用矩阵轴向操作(axis=0)批量判断连续天数条件
        """
        if percent_change_matrix.shape[0] < 3:
            return self._get_default_behavior_patterns()
        patterns = {
            'accumulation': {'detected': False, 'strength': 0.0, 'areas': []},
            'distribution': {'detected': False, 'strength': 0.0, 'areas': []},
            'consolidation': {'detected': False, 'strength': 0.0},
            'breakout_preparation': {'detected': False, 'strength': 0.0},
            'main_force_activity': 0.0
        }
        lookback = min(5, percent_change_matrix.shape[0])
        recent_changes = percent_change_matrix[-lookback:, :]
        changes_last_3 = recent_changes[-3:, :]
        mean_changes_3 = np.mean(changes_last_3, axis=0)
        is_accumulating = np.all(changes_last_3 > self.params['noise_threshold'], axis=0)
        is_distributing = np.all(changes_last_3 < -self.params['noise_threshold'], axis=0)
        low_price_mask = price_grid < current_price * 0.95
        high_price_mask = price_grid > current_price * 1.05
        accum_indices = np.where(is_accumulating & low_price_mask)[0]
        if len(accum_indices) > 0:
            patterns['accumulation']['detected'] = True
            patterns['accumulation']['strength'] = min(1.0, np.sum(mean_changes_3[accum_indices]) / 10.0)
            for idx in accum_indices:
                patterns['accumulation']['areas'].append({
                    'price': float(price_grid[idx]),
                    'avg_change': float(mean_changes_3[idx]),
                    'distance_to_price': (current_price - price_grid[idx]) / current_price
                })
        dist_indices = np.where(is_distributing & high_price_mask)[0]
        if len(dist_indices) > 0:
            patterns['distribution']['detected'] = True
            patterns['distribution']['strength'] = min(1.0, np.sum(np.abs(mean_changes_3[dist_indices])) / 10.0)
            for idx in dist_indices:
                patterns['distribution']['areas'].append({
                    'price': float(price_grid[idx]),
                    'avg_change': float(mean_changes_3[idx]),
                    'distance_to_price': (price_grid[idx] - current_price) / current_price
                })
        significant_changes = np.abs(recent_changes) > self.params['significant_change_threshold']
        patterns['main_force_activity'] = np.sum(significant_changes) / significant_changes.size
        if patterns['accumulation']['areas']:
            patterns['accumulation']['areas'] = sorted(patterns['accumulation']['areas'], key=lambda x: x['avg_change'], reverse=True)[:5]
        if patterns['distribution']['areas']:
            patterns['distribution']['areas'] = sorted(patterns['distribution']['areas'], key=lambda x: abs(x['avg_change']), reverse=True)[:5]
        current_concentration = self._calculate_concentration_metrics(chip_matrix[-1], price_grid)['comprehensive_concentration']
        if 0.4 <= current_concentration <= 0.6:
            patterns['consolidation']['detected'] = True
            patterns['consolidation']['strength'] = 1.0 - abs(current_concentration - 0.5) * 2
        resistance_mask = np.abs(price_grid - current_price * 1.05)
        resistance_idx = np.argmin(resistance_mask)
        if resistance_idx > 0:
            support_area = np.sum(chip_matrix[-1, :resistance_idx]) / 100.0
            if support_area > 0.6:
                patterns['breakout_preparation']['detected'] = True
                patterns['breakout_preparation']['strength'] = min(1.0, support_area)
        return patterns

    def _calculate_migration_patterns(self, percent_change_matrix: np.ndarray, chip_matrix: np.ndarray, price_grid: np.ndarray) -> Dict[str, any]:
        """
        版本: v2.0
        说明: 计算筹码迁移模式（合并掩码优化版）
        优化: 合并布尔掩码操作，减少中间数组内存分配
        """
        if percent_change_matrix.shape[0] < 2:
            return self._get_default_migration_patterns()
        patterns = {
            'upward_migration': {'strength': 0.0, 'volume': 0.0},
            'downward_migration': {'strength': 0.0, 'volume': 0.0},
            'convergence_migration': {'strength': 0.0, 'areas': []},
            'divergence_migration': {'strength': 0.0, 'areas': []},
            'net_migration_direction': 0.0
        }
        recent_changes = percent_change_matrix[-1, :] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
        price_center = np.dot(price_grid, chip_matrix[-1]) / 100.0
        abs_changes = np.abs(recent_changes)
        sum_abs_changes = np.sum(abs_changes) + 1e-10
        weighted_changes = np.dot(recent_changes, price_grid) / sum_abs_changes
        patterns['net_migration_direction'] = weighted_changes - price_center
        mask_low = price_grid < price_center * 0.9
        mask_high = price_grid > price_center * 1.1
        low_decrease = np.sum(recent_changes[mask_low & (recent_changes < 0)])
        high_increase = np.sum(recent_changes[mask_high & (recent_changes > 0)])
        diff_up = abs(high_increase - low_decrease)
        patterns['upward_migration']['strength'] = min(1.0, diff_up / 50.0)
        patterns['upward_migration']['volume'] = float(diff_up)
        high_decrease = np.sum(recent_changes[mask_high & (recent_changes < 0)])
        low_increase = np.sum(recent_changes[mask_low & (recent_changes > 0)])
        diff_down = abs(high_decrease - low_increase)
        patterns['downward_migration']['strength'] = min(1.0, diff_down / 50.0)
        patterns['downward_migration']['volume'] = float(diff_down)
        mask_mid = (price_grid >= price_center * 0.95) & (price_grid <= price_center * 1.05)
        mid_increase = np.sum(recent_changes[mask_mid & (recent_changes > 0)])
        if mid_increase > 0:
            patterns['convergence_migration']['strength'] = min(1.0, mid_increase / 30.0)
            idx_conv = np.where(mask_mid & (recent_changes > 0))[0][:5]
            patterns['convergence_migration']['areas'] = [
                {'price': float(price_grid[i]), 'change': float(recent_changes[i])}
                for i in idx_conv
            ]
        mid_decrease = np.sum(recent_changes[mask_mid & (recent_changes < 0)])
        if mid_decrease < 0:
            patterns['divergence_migration']['strength'] = min(1.0, abs(mid_decrease) / 30.0)
            idx_div = np.where(mask_mid & (recent_changes < 0))[0][:5]
            patterns['divergence_migration']['areas'] = [
                {'price': float(price_grid[i]), 'change': float(recent_changes[i])}
                for i in idx_div
            ]
        return patterns
    
    def _calculate_convergence_metrics(self, chip_matrix: np.ndarray, percent_change_matrix: np.ndarray, price_grid: np.ndarray) -> Dict[str, float]:
        """
        版本: v2.0
        说明: 计算聚散度指标（计数优化版）
        优化: 使用np.count_nonzero替代数组筛选后的len()，避免拷贝
        """
        if chip_matrix.shape[0] < 2 or len(percent_change_matrix) == 0:
            return self._get_default_convergence_metrics()
        metrics = {}
        current_chip = chip_matrix[-1]
        chip_entropy = entropy(current_chip + 1e-10)
        metrics['static_convergence'] = 1.0 - (chip_entropy / np.log(len(current_chip)))
        recent_changes = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
        n_pos = np.count_nonzero(recent_changes > 0)
        n_neg = np.count_nonzero(recent_changes < 0)
        total_changes = len(recent_changes)
        if n_pos > 0 and n_neg > 0:
            divergence_ratio = min(n_pos, n_neg) / total_changes
            metrics['dynamic_convergence'] = 1.0 - divergence_ratio
        else:
            metrics['dynamic_convergence'] = 1.0
        price_center = np.dot(price_grid, current_chip) / 100.0
        abs_changes = np.abs(recent_changes)
        dist_from_center = np.abs(price_grid - price_center)
        weighted_changes = np.dot(abs_changes, dist_from_center) / (np.sum(abs_changes) + 1e-10)
        max_distance = np.max(dist_from_center)
        metrics['migration_convergence'] = 1.0 - (weighted_changes / max_distance) if max_distance > 0 else 0.5
        metrics['comprehensive_convergence'] = (
            0.4 * metrics['static_convergence'] +
            0.3 * metrics['dynamic_convergence'] +
            0.3 * metrics['migration_convergence']
        )
        net_change_direction = np.dot(recent_changes, price_grid - price_center)
        if net_change_direction > 0:
            metrics['convergence_strength'] = min(1.0, net_change_direction / 100.0)
            metrics['divergence_strength'] = 0.0
        else:
            metrics['convergence_strength'] = 0.0
            metrics['divergence_strength'] = min(1.0, abs(net_change_direction) / 100.0)
        return metrics

    # ============== 数据获取方法 ==============
    
    async def _fetch_chip_percent_data(self, stock_code: str, trade_date: str, lookback_days: int) -> Dict[str, any]:
        """
        获取筹码百分比数据
        """
        try:
            chips_model = get_cyq_chips_model_by_code(stock_code)
            if not chips_model:
                print(f"🕵️ [PROBE-FETCH] 无法获取模型 {stock_code}")
                return None
            trade_date_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
            # 获取当前日期的筹码分布
            current_chip_qs = chips_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time=trade_date_dt
            ).values('price', 'percent')
            current_chip_list = await sync_to_async(list)(current_chip_qs)
            current_chip_df = pd.DataFrame(current_chip_list) if current_chip_list else pd.DataFrame()
            # 获取历史筹码分布
            start_date = trade_date_dt - timedelta(days=lookback_days * 2)
            history_chip_qs = chips_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_date,
                trade_time__lt=trade_date_dt
            ).order_by('trade_time').values('trade_time', 'price', 'percent')
            history_chip_list = await sync_to_async(list)(history_chip_qs)
            # 按日期分组
            chip_history = []
            if history_chip_list:
                history_df = pd.DataFrame(history_chip_list)
                unique_dates = history_df['trade_time'].unique()
                for date in unique_dates:
                    day_df = history_df[history_df['trade_time'] == date][['price', 'percent']]
                    chip_history.append(day_df)
            # 获取价格历史
            daily_model = get_daily_data_model_by_code(stock_code)
            price_history = pd.DataFrame()
            if daily_model:
                from stock_models.stock_basic import StockInfo
                stock = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
                price_qs = daily_model.objects.filter(
                    stock=stock,
                    trade_time__gte=start_date,
                    trade_time__lte=trade_date_dt
                ).order_by('trade_time').values('trade_time', 'open', 'high', 'low', 'close')
                price_list = await sync_to_async(list)(price_qs)
                price_history = pd.DataFrame(price_list) if price_list else pd.DataFrame()
            current_price = 0
            if not current_chip_df.empty:
                current_price = current_chip_df['price'].mean()
            elif not price_history.empty:
                current_price = price_history['close'].iloc[-1]
            return {
                'current_chip_dist': current_chip_df,
                'chip_history': chip_history,
                'price_history': price_history,
                'current_price': current_price
            }
        except Exception as e:
            logger.error(f"获取筹码数据失败 {stock_code}: {e}")
            print(f"❌ [PROBE-FETCH-ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    # ============== 默认结果方法 ==============
    
    def _get_default_result(self, stock_code: str = "", trade_date: str = "") -> Dict[str, any]:
        return {
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
            'analysis_status': 'failed'
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
    
    def _get_default_concentration_metrics(self) -> Dict[str, float]:
        return {
            'entropy_concentration': 0.5,
            'peak_concentration': 0.3,
            'cv_concentration': 0.5,
            'main_force_concentration': 0.2,
            'comprehensive_concentration': 0.4,
            'chip_skewness': 0.0,
            'chip_kurtosis': 0.0
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
    
    def _get_default_migration_patterns(self) -> Dict[str, any]:
        return {
            'upward_migration': {'strength': 0.0, 'volume': 0.0},
            'downward_migration': {'strength': 0.0, 'volume': 0.0},
            'convergence_migration': {'strength': 0.0, 'areas': []},
            'divergence_migration': {'strength': 0.0, 'areas': []},
            'net_migration_direction': 0.0
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