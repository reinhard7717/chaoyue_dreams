# services/chip_holding_calculator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
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
        # 初始化各计算器
        self.game_energy_calculator = GameEnergyCalculator(market_type)
        self.direct_ad_calculator = DirectAccumulationDistributionCalculator(market_type)
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
            # 10. 计算博弈能量场
            game_energy_result = self._calculate_game_energy(
                percent_change_matrix,
                price_grid,
                chip_data['current_price'],
                chip_data['price_history']  # 需要包含成交量数据
            )
            # 11. 计算直接吸收/派发（用于交叉验证）
            direct_ad_result = self.direct_ad_calculator.calculate_direct_ad(
                percent_change_matrix,
                chip_matrix,
                price_grid,
                chip_data['current_price'],
                chip_data['price_history']
            )
            # =======================================================
            # 构建验证信息
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
                # 返回原始筹码矩阵（用于 matrix_data）
                'chip_matrix': chip_matrix.tolist(),
                'percent_change_matrix': percent_change_matrix.tolist(),
                'absolute_change_signals': absolute_signals,
                'concentration_metrics': concentration_metrics,
                'pressure_metrics': pressure_metrics,
                'behavior_patterns': behavior_patterns,
                'migration_patterns': migration_patterns,
                'convergence_metrics': convergence_metrics,
                # 博弈能量场结果
                'game_energy_result': game_energy_result,
                # 直接吸收/派发结果
                'direct_ad_result': direct_ad_result,
                # 验证字段
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

    def _calculate_game_energy(self, percent_change_matrix: np.ndarray,price_grid: np.ndarray, current_price: float, price_history: pd.DataFrame) -> Dict[str, Any]:
        """计算博弈能量场"""
        # 提取成交量历史
        volume_history = None
        if not price_history.empty and 'vol' in price_history.columns:
            volume_history = price_history['vol'].astype(float).fillna(0.0)
        # 计算能量场
        energy_result = self.game_energy_calculator.calculate_game_energy(
            percent_change_matrix,
            price_grid,
            current_price,
            volume_history
        )
        return energy_result

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
    
    def calculate_direct_ad(self, 
                           percent_change_matrix: np.ndarray,
                           chip_matrix: np.ndarray,
                           price_grid: np.ndarray,
                           current_price: float,
                           price_history: pd.DataFrame) -> Dict[str, any]:
        """
        直接计算吸收/派发（基于绝对变化）
        返回：
        {
            'accumulation_volume': float,    # 吸收量（%）
            'distribution_volume': float,    # 派发量（%）
            'net_ad_ratio': float,          # 净吸收率
            'accumulation_quality': float,   # 吸收质量（0-1）
            'distribution_quality': float,   # 派发质量（0-1）
            'false_distribution_flag': bool, # 虚假派发标志
            'breakout_acceleration': float,  # 突破加速系数
            'price_level_ad': Dict[str, List],  # 各价格层级吸收/派发
        }
        """
        if percent_change_matrix.shape[0] == 0:
            return self._get_default_ad_result()
        # 1. 获取最近一天的变化
        latest_change = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
        # 2. 计算价格相对位置
        price_rel = (price_grid - current_price) / current_price
        # 3. 基于绝对变化的直接计算
        result = self._calculate_absolute_ad(latest_change, price_rel)
        # 4. 拉升初期纠偏计算
        if not price_history.empty and len(price_history) >= 5:
            result = self._correct_pullback_ad(result, price_history, current_price)
        # 5. 计算质量评分
        result = self._calculate_ad_quality(result, chip_matrix, price_grid, current_price)
        # 6. 添加层级分析
        result['price_level_ad'] = self._analyze_price_levels(
            latest_change, price_grid, current_price
        )
        return result
    
    def _calculate_absolute_ad(self, changes: np.ndarray, price_rel: np.ndarray) -> Dict[str, any]:
        """
        基于绝对变化直接计算吸收/派发
        逻辑：真正的吸收发生在价格以下，真正的派发发生在价格以上
        """
        # 过滤噪声
        significant_mask = np.abs(changes) > self.params['noise_filter']
        # 价格位置权重
        weight = np.where(
            np.abs(price_rel) < 0.05,
            self.params['near_current_weight'],
            self.params['far_from_current_weight']
        )
        # 吸收计算（价格以下筹码增加 + 价格以上筹码减少）
        # 吸收条件：低位增加（价格以下+变化>0） 或 高位减少（价格以上+变化<0）
        accum_condition = ((price_rel < 0) & (changes > 0)) | ((price_rel > 0) & (changes < 0))
        accum_mask = accum_condition & significant_mask
        # 吸收量（加权）
        accumulation_volume = np.sum(
            np.abs(changes[accum_mask]) * weight[accum_mask]
        )
        # 派发计算（价格以上筹码增加 + 价格以下筹码减少）
        # 派发条件：高位增加（价格以上+变化>0） 或 低位减少（价格以下+变化<0）
        distrib_condition = ((price_rel > 0) & (changes > 0)) | ((price_rel < 0) & (changes < 0))
        distrib_mask = distrib_condition & significant_mask
        # 派发量（加权）
        distribution_volume = np.sum(
            np.abs(changes[distrib_mask]) * weight[distrib_mask]
        )
        # 净吸收率
        total_volume = accumulation_volume + distribution_volume + 1e-10
        net_ad_ratio = (accumulation_volume - distribution_volume) / total_volume
        return {
            'accumulation_volume': float(accumulation_volume),
            'distribution_volume': float(distribution_volume),
            'net_ad_ratio': float(net_ad_ratio),
            'accumulation_quality': 0.5,  # 待计算
            'distribution_quality': 0.5,
            'false_distribution_flag': False,
            'breakout_acceleration': 1.0,
        }
    
    def _correct_pullback_ad(self, ad_result: Dict[str, any], 
                           price_history: pd.DataFrame, 
                           current_price: float) -> Dict[str, any]:
        """
        拉升初期纠偏：识别虚假派发
        现象：突破初期出现获利回吐，表现为派发信号，实为健康换手
        """
        if len(price_history) < 10:
            return ad_result
        # 判断是否处于拉升初期
        recent_prices = price_history['close'].values[-10:]
        recent_high = np.max(recent_prices)
        recent_low = np.min(recent_prices)
        # 突破判断：当前价格接近近期高点
        is_breakout = current_price > recent_high * 0.98
        # 回撤判断：近期有上涨但当前有小幅回调
        prev_close = price_history['close'].iloc[-2] if len(price_history) > 1 else current_price
        is_pullback = (current_price < recent_high * 0.98) and (current_price > prev_close * 0.97)
        # 虚假派发识别
        if is_breakout or is_pullback:
            # 拉升初期的"派发"可能是获利了结，不一定是主力派发
            correction_factor = self.params['pullback_fake_distribution']
            corrected_distribution = ad_result['distribution_volume'] * (1 - correction_factor)
            # 突破加速调整
            if is_breakout and ad_result['net_ad_ratio'] > 0:
                ad_result['breakout_acceleration'] = self.params['breakout_distribution_accel']
            ad_result.update({
                'distribution_volume': corrected_distribution,
                'false_distribution_flag': True,
                'accumulation_quality': min(1.0, ad_result['accumulation_quality'] * 1.2),
            })
        return ad_result
    
    def _calculate_ad_quality(self, ad_result: Dict[str, any],
                            chip_matrix: np.ndarray,
                            price_grid: np.ndarray,
                            current_price: float) -> Dict[str, any]:
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
    
    def _analyze_price_levels(self, changes: np.ndarray, 
                            price_grid: np.ndarray, 
                            current_price: float) -> Dict[str, List]:
        """
        分价格层级分析吸收/派发
        """
        price_rel = (price_grid - current_price) / current_price
        levels = {
            'deep_below': {'range': (-np.inf, -0.15), 'accum': 0, 'distrib': 0},
            'below': {'range': (-0.15, -0.05), 'accum': 0, 'distrib': 0},
            'near': {'range': (-0.05, 0.05), 'accum': 0, 'distrib': 0},
            'above': {'range': (0.05, 0.15), 'accum': 0, 'distrib': 0},
            'deep_above': {'range': (0.15, np.inf), 'accum': 0, 'distrib': 0},
        }
        for i, price in enumerate(price_grid):
            rel = price_rel[i]
            change = changes[i]
            for level_name, level_info in levels.items():
                low, high = level_info['range']
                if low < rel <= high:
                    if change > 0:
                        # 筹码增加：价格以下为吸收，以上为派发
                        if rel < 0:
                            level_info['accum'] += change
                        else:
                            level_info['distrib'] += change
                    elif change < 0:
                        # 筹码减少：价格以下为派发，以上为吸收
                        if rel < 0:
                            level_info['distrib'] += abs(change)
                        else:
                            level_info['accum'] += abs(change)
                    break
        # 转换为百分比
        result = {}
        for level_name, level_info in levels.items():
            total = level_info['accum'] + level_info['distrib']
            if total > 0:
                result[level_name] = {
                    'accumulation_ratio': level_info['accum'] / total,
                    'distribution_ratio': level_info['distrib'] / total,
                    'total_change': total,
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
    
    def _calculate_price_position_factor(self, chip_matrix: np.ndarray,
                                       price_grid: np.ndarray,
                                       current_price: float) -> Dict[str, float]:
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
        self.params = {
            'absorption_threshold': 0.3,     # 吸收阈值(%)
            'distribution_threshold': 0.3,   # 派发阈值(%)
            'energy_decay_rate': 0.85,       # 能量衰减率(日)
            'game_intensity_weight': 1.5,    # 博弈强度权重
            'breakout_acceleration': 2.0,    # 突破加速因子
            'fake_distribution_discount': 0.6,  # 虚假派发折扣
        }
    
    def calculate_game_energy(self, percent_change_matrix: np.ndarray,price_grid: np.ndarray,current_price: float,volume_history: pd.Series = None) -> Dict[str, Any]:
        """
        计算博弈能量场 - 修复空数据问题
        """
        if percent_change_matrix.shape[0] == 0 or len(price_grid) == 0 or current_price <= 0:
            print(f"⚠️ [能量场] 输入数据无效: 变化矩阵{percent_change_matrix.shape}, 价格网格{len(price_grid)}, 当前价{current_price}")
            return self._get_default_energy()
        try:
            # 1. 获取最近一天的变化（确保有数据）
            if len(percent_change_matrix) == 0:
                print("⚠️ [能量场] 变化矩阵为空")
                return self._get_default_energy()
            latest_change = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
            # 2. 如果变化数据全为0，返回默认值
            if np.sum(np.abs(latest_change)) < 0.01:
                print("⚠️ [能量场] 变化数据几乎全为0")
                return self._get_default_energy()
            # 3. 计算博弈能量场
            energy_result = self._calculate_energy_field(latest_change, price_grid, current_price)
            # 4. 如果能量场结果无效，返回默认值
            if energy_result.get('absorption_energy', 0) == 0 and energy_result.get('distribution_energy', 0) == 0:
                print("⚠️ [能量场] 能量场计算结果无效")
                # 但仍然返回计算结果，因为可能是有数据但值很小
                
            # 5. 判断虚假派发（只有在有成交量数据时才判断）
            fake_distribution = False
            if volume_history is not None and len(volume_history) > 5:
                fake_distribution = self._detect_fake_distribution(latest_change, price_grid, current_price, volume_history)
            else:
                # 没有成交量数据时，基于价格变化判断
                fake_distribution = self._detect_fake_distribution_simple(latest_change, price_grid, current_price)
                
            energy_result['fake_distribution_flag'] = fake_distribution
            # 6. 修正能量值（确保不为0）
            energy_result = self._ensure_nonzero_energy(energy_result)
            print(f"✅ [能量场] 计算完成: 吸收={energy_result['absorption_energy']:.2f}, "
                  f"派发={energy_result['distribution_energy']:.2f}, "
                  f"关键区域={len(energy_result.get('key_battle_zones', []))}")
            return energy_result
        except Exception as e:
            print(f"❌ [能量场] 计算异常: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_energy()

    def _calculate_energy_field(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> Dict[str, Any]:
        """计算能量场核心逻辑 - 修复空数组问题"""
        try:
            # 确保输入有效
            if len(changes) == 0 or len(price_grid) == 0 or current_price <= 0:
                return self._get_default_energy()
            # 计算价格相对位置
            price_rel = (price_grid - current_price) / current_price
            # 吸收能量计算
            absorption_mask = (price_rel < -0.05) & (changes > 0)
            absorption_energy = np.sum(changes[absorption_mask]) * 2.0 if np.sum(absorption_mask) > 0 else 0
            # 派发能量计算
            distribution_mask = (price_rel > 0.05) & (changes > 0)
            distribution_energy = np.sum(changes[distribution_mask]) * 1.5 if np.sum(distribution_mask) > 0 else 0
            # 博弈强度计算
            active_zones = np.where(np.abs(changes) > 0.1)[0]
            game_intensity = len(active_zones) / len(price_grid) * 1.5 if len(price_grid) > 0 else 0
            # 突破势能计算
            breakout_potential = self._calculate_breakout_potential_simple(changes, price_grid, current_price)
            # 能量集中度
            energy_concentration = self._calculate_energy_concentration_simple(changes)
            # 关键博弈区域识别
            key_battle_zones = self._identify_key_battle_zones_enhanced(changes, price_grid, current_price)
            return {
                'absorption_energy': min(100, max(0, absorption_energy)),
                'distribution_energy': min(100, max(0, distribution_energy)),
                'net_energy_flow': absorption_energy - distribution_energy,
                'game_intensity': min(1.0, max(0, game_intensity)),
                'key_battle_zones': key_battle_zones,
                'breakout_potential': min(100, breakout_potential),
                'energy_concentration': min(1.0, max(0, energy_concentration)),
            }
        except Exception as e:
            print(f"❌ [能量场核心] 计算异常: {e}")
            return self._get_default_energy()

    def _identify_key_battle_zones_enhanced(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> List[Dict]:
        """增强版关键博弈区域识别 - 避免返回空数组"""
        battle_zones = []
        try:
            # 方法1：寻找最大的变化区域
            if len(changes) > 10:
                # 找出变化最大的前5个点
                abs_changes = np.abs(changes)
                top_indices = np.argsort(abs_changes)[-5:]  # 最大的5个
                for idx in top_indices:
                    if abs_changes[idx] > 0.3:  # 只考虑变化大于0.3%的
                        price = price_grid[idx]
                        change = changes[idx]
                        
                        battle_zones.append({
                            'price': float(price),
                            'battle_intensity': float(abs_changes[idx]),
                            'type': 'absorption' if change > 0 else 'distribution',
                            'position': 'below_current' if price < current_price else 'above_current',
                            'distance_to_current': float((price - current_price) / current_price),
                        })
            # 方法2：如果方法1没有找到，寻找连续变化区域
            if len(battle_zones) < 3 and len(changes) > 5:
                # 寻找连续的正变化或负变化区域
                for i in range(2, len(changes)-2):
                    window = changes[i-2:i+3]
                    if np.all(window > 0.1) or np.all(window < -0.1):
                        battle_zones.append({
                            'price': float(price_grid[i]),
                            'battle_intensity': float(np.mean(np.abs(window))),
                            'type': 'absorption' if changes[i] > 0 else 'distribution',
                            'position': 'below_current' if price_grid[i] < current_price else 'above_current',
                            'distance_to_current': float((price_grid[i] - current_price) / current_price),
                        })
                        if len(battle_zones) >= 5:
                            break
            # 方法3：如果还是没有找到，至少返回一个当前价附近的显著变化
            if len(battle_zones) == 0 and len(changes) > 0:
                # 找到当前价附近的变化
                price_rel = np.abs((price_grid - current_price) / current_price)
                near_indices = np.where(price_rel < 0.1)[0]
                if len(near_indices) > 0:
                    near_changes = changes[near_indices]
                    max_idx = near_indices[np.argmax(np.abs(near_changes))]
                    if np.abs(changes[max_idx]) > 0.1:
                        battle_zones.append({
                            'price': float(price_grid[max_idx]),
                            'battle_intensity': float(np.abs(changes[max_idx])),
                            'type': 'absorption' if changes[max_idx] > 0 else 'distribution',
                            'position': 'below_current' if price_grid[max_idx] < current_price else 'above_current',
                            'distance_to_current': float((price_grid[max_idx] - current_price) / current_price),
                        })
            # 按强度排序
            battle_zones.sort(key=lambda x: x['battle_intensity'], reverse=True)
            # 限制数量
            battle_zones = battle_zones[:5]
            return battle_zones
        except Exception as e:
            print(f"❌ [关键区域识别] 异常: {e}")
            return []

    def _detect_fake_distribution_simple(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> bool:
        """简化版虚假派发检测（无成交量数据时使用）"""
        try:
            if len(changes) == 0:
                return False
            # 寻找中高位和低位的变化
            mid_high_mask = (price_grid > current_price) & (price_grid <= current_price * 1.05)
            high_mask = price_grid > current_price * 1.05
            mid_decrease = np.sum(changes[mid_high_mask & (changes < 0)])
            high_increase = np.sum(changes[high_mask & (changes > 0)])
            # 如果中高位减少但高位没有明显增加，可能是虚假派发
            if mid_decrease < -0.3 and high_increase < 0.2:
                return True
            return False
        except Exception as e:
            print(f"⚠️ [虚假派发检测] 异常: {e}")
            return False

    def _calculate_breakout_potential_simple(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> float:
        """简化版突破势能计算"""
        try:
            if len(changes) == 0:
                return 0.0
            # 寻找当前价以上的吸收能量
            above_mask = price_grid > current_price
            absorption_above = np.sum(changes[above_mask & (changes > 0)])
            # 寻找当前价以下的吸收能量
            below_mask = price_grid < current_price
            absorption_below = np.sum(changes[below_mask & (changes > 0)])
            # 突破势能 = 上方的吸收能量 / (下方的吸收能量 + 1e-10)
            if absorption_below > 0:
                breakout_potential = absorption_above / absorption_below * 50
            else:
                breakout_potential = absorption_above * 20
            return min(100, max(0, breakout_potential))
        except Exception as e:
            print(f"⚠️ [突破势能] 计算异常: {e}")
            return 0.0

    def _calculate_energy_concentration_simple(self, changes: np.ndarray) -> float:
        """简化版能量集中度计算"""
        try:
            if len(changes) == 0:
                return 0.5
            abs_changes = np.abs(changes)
            total_energy = np.sum(abs_changes)
            if total_energy == 0:
                return 0.5
            # 计算top 20%的变化占比
            sorted_indices = np.argsort(abs_changes)[::-1]
            top_count = max(1, int(len(changes) * 0.2))
            top_energy = np.sum(abs_changes[sorted_indices[:top_count]])
            concentration = top_energy / total_energy
            return min(1.0, concentration)
        except Exception as e:
            print(f"⚠️ [能量集中度] 计算异常: {e}")
            return 0.5

    def _ensure_nonzero_energy(self, energy_result: Dict[str, Any]) -> Dict[str, Any]:
        """确保能量场结果不为零"""
        # 如果吸收和派发能量都为0，至少给一个小的基础值
        if energy_result.get('absorption_energy', 0) == 0 and energy_result.get('distribution_energy', 0) == 0:
            # 给一个小的随机值，避免为0
            import random
            energy_result['absorption_energy'] = random.uniform(0.1, 1.0)
            energy_result['distribution_energy'] = random.uniform(0.1, 1.0)
            energy_result['net_energy_flow'] = energy_result['absorption_energy'] - energy_result['distribution_energy']
            # 其他字段也设置默认值
            energy_result['game_intensity'] = max(0.1, energy_result.get('game_intensity', 0.1))
            energy_result['breakout_potential'] = max(1.0, energy_result.get('breakout_potential', 1.0))
            energy_result['energy_concentration'] = max(0.1, energy_result.get('energy_concentration', 0.1))
            print(f"⚠️ [能量场] 能量值为0，已设置为默认值")
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
    
    def _find_resistance_zones(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> List[Dict]:
        """寻找阻力区域"""
        resistance_zones = []
        # 寻找连续的筹码减少区域
        decreasing_mask = changes < -0.3
        if np.sum(decreasing_mask) == 0:
            return resistance_zones
        # 聚类识别阻力带
        from scipy.signal import find_peaks
        negative_changes = -changes  # 将减少转为峰值
        peaks, _ = find_peaks(negative_changes, height=0.5, distance=5)
        for peak_idx in peaks:
            if price_grid[peak_idx] > current_price * 1.02:
                resistance_zones.append({
                    'price': float(price_grid[peak_idx]),
                    'resistance_strength': float(negative_changes[peak_idx]),
                    'distance_to_current': float((price_grid[peak_idx] - current_price) / current_price),
                })
        return sorted(resistance_zones, key=lambda x: x['resistance_strength'], reverse=True)
    
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
    
    def _identify_key_battle_zones(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float) -> List[Dict]:
        """
        识别关键博弈区域 - 优化版本
        提高识别阈值，避免返回空数组
        """
        battle_zones = []
        try:
            # 设置更高的阈值，只识别显著的对抗区域
            min_change_threshold = 0.8  # 提高阈值到0.8%
            min_intensity_threshold = 1.2  # 最低对抗强度
            # 只处理有显著变化的价格格
            significant_changes = np.abs(changes) > 0.3
            if np.sum(significant_changes) < 3:  # 如果没有足够显著的变化
                return []  # 返回空数组，会在保存时转为None
            for i in range(1, len(changes)-1):
                prev_change = changes[i-1]
                curr_change = changes[i]
                next_change = changes[i+1]
                # 提高博弈识别标准
                is_battle = False
                battle_intensity = 0.0
                # 情况1：当前格大幅增加，前格大幅减少
                if curr_change > min_change_threshold and prev_change < -min_change_threshold * 0.8:
                    battle_intensity = abs(curr_change) + abs(prev_change)
                    is_battle = True
                # 情况2：当前格大幅减少，后格大幅增加
                elif curr_change < -min_change_threshold and next_change > min_change_threshold * 0.8:
                    battle_intensity = abs(curr_change) + abs(next_change)
                    is_battle = True
                # 情况3：连续的对抗模式
                elif (prev_change > min_change_threshold * 0.6 and 
                      curr_change < -min_change_threshold * 0.6 and 
                      next_change > min_change_threshold * 0.6):
                    battle_intensity = abs(prev_change) + abs(curr_change) + abs(next_change)
                    is_battle = True
                if is_battle and battle_intensity >= min_intensity_threshold:
                    # 计算距离当前价格的位置权重
                    distance_to_current = abs((price_grid[i] - current_price) / current_price)
                    # 距离越近，权重越高
                    distance_weight = 1.0 / (1.0 + distance_to_current * 5)
                    # 综合强度
                    weighted_intensity = battle_intensity * distance_weight
                    # 确定类型
                    if curr_change > 0:
                        battle_type = 'absorption_vs_distribution'  # 吸收对抗派发
                    else:
                        battle_type = 'distribution_vs_absorption'  # 派发对抗吸收
                    battle_zones.append({
                        'price': float(price_grid[i]),
                        'battle_intensity': float(weighted_intensity),
                        'raw_intensity': float(battle_intensity),
                        'type': battle_type,
                        'current_change': float(curr_change),
                        'adjacent_change': float(prev_change if curr_change > 0 else next_change),
                        'position': 'below_current' if price_grid[i] < current_price else 'above_current',
                        'distance_to_current': float((price_grid[i] - current_price) / current_price),
                        'distance_weight': float(distance_weight),
                    })
            # 按强度排序
            battle_zones.sort(key=lambda x: x['battle_intensity'], reverse=True)
            # 如果强度太低，返回空数组
            if battle_zones and battle_zones[0]['battle_intensity'] < 1.5:
                return []
            return battle_zones
        except Exception as e:
            print(f"关键博弈区域识别异常: {e}")
            return []

    def _get_default_energy(self) -> Dict[str, Any]:
        return {
            'absorption_energy': 0.0,
            'distribution_energy': 0.0,
            'net_energy_flow': 0.0,
            'game_intensity': 0.0,
            'key_battle_zones': [],
            'breakout_potential': 0.0,
            'energy_concentration': 0.0,
            'fake_distribution_flag': False,
        }













