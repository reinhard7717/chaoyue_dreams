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
from utils.model_helpers import get_cyq_chips_model_by_code

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
            print(f"🔍 [筹码动态分析] 开始分析 {stock_code} {trade_date}")
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
            print(f"✅ [筹码动态分析] 完成分析 {stock_code} {trade_date}")
            return result
        except Exception as e:
            logger.error(f"筹码动态分析失败 {stock_code} {trade_date}: {e}")
            print(f"❌ [筹码动态分析异常] {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_result(stock_code, trade_date)

    def _build_normalized_chip_matrix(self,chip_history: List[pd.DataFrame],current_chip: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建归一化的筹码矩阵
        关键：确保每个分布都归一化为总和100%
        """
        # 合并历史和当前数据
        all_chips = chip_history + [current_chip]
        # 提取所有价格点
        all_prices = []
        for chip_df in all_chips:
            if not chip_df.empty:
                all_prices.extend(chip_df['price'].values)
        if not all_prices:
            # 默认价格范围
            min_price, max_price = 1.0, 100.0
        else:
            min_price = np.min(all_prices)
            max_price = np.max(all_prices)
            # 扩展范围
            price_range = max_price - min_price
            min_price = max(0.01, min_price - price_range * 0.15)
            max_price = max_price + price_range * 0.15
        # 创建价格网格
        price_grid = np.linspace(min_price, max_price, self.price_granularity)
        # 构建筹码矩阵
        n_days = len(all_chips)
        n_prices = len(price_grid)
        chip_matrix = np.zeros((n_days, n_prices))
        for day_idx, chip_df in enumerate(all_chips):
            if chip_df.empty or len(chip_df) < 3:
                # 均匀分布
                chip_matrix[day_idx, :] = 100.0 / n_prices
                continue
            # 线性插值到价格网格
            from scipy.interpolate import interp1d
            try:
                # 确保数据有效
                valid_mask = (~chip_df['price'].isna()) & (~chip_df['percent'].isna())
                chip_df_valid = chip_df[valid_mask].copy()
                if len(chip_df_valid) < 2:
                    chip_matrix[day_idx, :] = 100.0 / n_prices
                    continue
                # 排序并去重
                chip_df_valid = chip_df_valid.sort_values('price')
                chip_df_valid = chip_df_valid.drop_duplicates('price')
                # 归一化确保总和为100%
                total_percent = chip_df_valid['percent'].sum()
                if total_percent == 0:
                    chip_matrix[day_idx, :] = 100.0 / n_prices
                    continue
                chip_df_valid['percent_normalized'] = chip_df_valid['percent'] * (100.0 / total_percent)
                # 线性插值
                f = interp1d(
                    chip_df_valid['price'],
                    chip_df_valid['percent_normalized'],
                    kind='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                interpolated = f(price_grid)
                # 再次归一化
                if interpolated.sum() > 0:
                    interpolated = interpolated * (100.0 / interpolated.sum())
                chip_matrix[day_idx, :] = interpolated
            except Exception as e:
                print(f"⚠️ 第{day_idx}天插值失败: {e}")
                chip_matrix[day_idx, :] = 100.0 / n_prices
        return price_grid, chip_matrix
    
    def _calculate_percent_change_matrix(self,chip_matrix: np.ndarray) -> np.ndarray:
        """
        计算百分比变化矩阵
        返回: n_days-1 × n_prices 矩阵
        每个值 = 当天百分比 - 前一天百分比
        """
        if chip_matrix.shape[0] < 2:
            return np.zeros((1, chip_matrix.shape[1]))
        # 计算每日变化
        n_days = chip_matrix.shape[0]
        change_matrix = np.zeros((n_days - 1, chip_matrix.shape[1]))
        for i in range(1, n_days):
            change_matrix[i-1, :] = chip_matrix[i, :] - chip_matrix[i-1, :]
        return change_matrix
    
    def _analyze_absolute_changes(self,percent_change_matrix: np.ndarray,price_grid: np.ndarray,current_price: float) -> Dict[str, any]:
        """
        基于绝对变化的信号分析
        关键：区分有效变动（>2%）和噪声（<1%）
        """
        if percent_change_matrix.shape[0] == 0:
            return self._get_default_absolute_signals()
        # 获取最近3天的变化
        recent_changes = percent_change_matrix[-min(3, len(percent_change_matrix)):, :]
        signals = {
            'significant_increase_areas': [],  # 显著增加区域
            'significant_decrease_areas': [],  # 显著减少区域
            'accumulation_signals': [],        # 吸筹信号
            'distribution_signals': [],        # 派发信号
            'noise_level': 0.0,                # 噪声水平
            'signal_quality': 0.0              # 信号质量
        }
        # 阈值
        increase_threshold = self.params['significant_change_threshold']
        decrease_threshold = -self.params['significant_change_threshold']
        noise_threshold = self.params['noise_threshold']
        # 分析每个价格区间
        for price_idx, price in enumerate(price_grid):
            # 最近3天的平均变化
            avg_change = np.mean(recent_changes[:, price_idx]) if recent_changes.shape[0] > 0 else 0
            # 判断信号类型
            if avg_change > increase_threshold:
                signal_type = 'significant_increase'
                signals['significant_increase_areas'].append({
                    'price': float(price),
                    'change': float(avg_change),
                    'distance_to_current': abs(price - current_price) / current_price if current_price > 0 else 1.0
                })
                # 吸筹判断：价格低于当前价 + 显著增加
                if price < current_price * 0.95:
                    signals['accumulation_signals'].append({
                        'price': float(price),
                        'change': float(avg_change),
                        'strength': min(1.0, avg_change / 10.0)
                    })
                    
            elif avg_change < decrease_threshold:
                signal_type = 'significant_decrease'
                signals['significant_decrease_areas'].append({
                    'price': float(price),
                    'change': float(avg_change),
                    'distance_to_current': abs(price - current_price) / current_price if current_price > 0 else 1.0
                })
                # 派发判断：价格高于当前价 + 显著减少
                if price > current_price * 1.05:
                    signals['distribution_signals'].append({
                        'price': float(price),
                        'change': float(avg_change),
                        'strength': min(1.0, abs(avg_change) / 10.0)
                    })
            # 噪声水平计算
            if abs(avg_change) < noise_threshold:
                signals['noise_level'] += 1
        # 计算噪声水平百分比
        signals['noise_level'] = signals['noise_level'] / len(price_grid) if len(price_grid) > 0 else 1.0
        # 信号质量 = 1 - 噪声水平
        signals['signal_quality'] = 1.0 - signals['noise_level']
        # 排序并限制数量
        for key in ['significant_increase_areas', 'significant_decrease_areas', 
                   'accumulation_signals', 'distribution_signals']:
            signals[key] = sorted(signals[key], key=lambda x: abs(x['change']), reverse=True)[:10]
        return signals
    
    def _calculate_concentration_metrics(self,current_chip_dist: np.ndarray,price_grid: np.ndarray) -> Dict[str, float]:
        """
        计算集中度指标 - 基于筹码分布百分比
        """
        if len(current_chip_dist) == 0:
            return self._get_default_concentration_metrics()
        metrics = {}
        # 1. 熵值集中度（越低越集中）
        chip_entropy = entropy(current_chip_dist + 1e-10)
        max_entropy = np.log(len(current_chip_dist))
        metrics['entropy_concentration'] = 1.0 - (chip_entropy / max_entropy)
        # 2. 峰值集中度（前20%峰值的占比）
        sorted_chip = np.sort(current_chip_dist)[::-1]
        top_20_percent = int(len(current_chip_dist) * 0.2)
        metrics['peak_concentration'] = sorted_chip[:top_20_percent].sum() / 100.0
        # 3. 价格变异系数（CV越低越集中）
        price_mean = np.sum(price_grid * current_chip_dist) / 100.0
        if price_mean > 0:
            variance = np.sum(current_chip_dist * (price_grid - price_mean) ** 2) / 100.0
            price_std = np.sqrt(variance)
            metrics['cv_concentration'] = 1.0 - min(1.0, price_std / price_mean)
        else:
            metrics['cv_concentration'] = 0.5
        # 4. 主力集中度（假设主力集中在窄区间）
        # 计算筹码在±10%区间内的集中度
        if price_mean > 0:
            mask = (price_grid >= price_mean * 0.9) & (price_grid <= price_mean * 1.1)
            metrics['main_force_concentration'] = current_chip_dist[mask].sum() / 100.0
        else:
            metrics['main_force_concentration'] = 0.0
        # 5. 综合集中度
        weights = [0.3, 0.3, 0.2, 0.2]
        values = [
            metrics['entropy_concentration'],
            metrics['peak_concentration'],
            metrics['cv_concentration'],
            metrics['main_force_concentration']
        ]
        metrics['comprehensive_concentration'] = np.sum(np.array(weights) * np.array(values))
        # 6. 峰度与偏度
        metrics['chip_skewness'] = float(skew(current_chip_dist))
        metrics['chip_kurtosis'] = float(kurtosis(current_chip_dist))
        return metrics
    
    def _calculate_pressure_metrics(self,current_chip_dist: np.ndarray,price_grid: np.ndarray,current_price: float,price_history: pd.DataFrame) -> Dict[str, float]:
        """
        计算压力指标 - 基于绝对百分比
        """
        if len(current_chip_dist) == 0 or current_price <= 0:
            return self._get_default_pressure_metrics()
        metrics = {}
        # 1. 获利盘压力（成本低于当前价）
        profit_mask = price_grid < current_price
        metrics['profit_pressure'] = current_chip_dist[profit_mask].sum() / 100.0
        # 2. 套牢盘压力（成本高于当前价10%以上）
        trapped_mask = price_grid > current_price * 1.10
        metrics['trapped_pressure'] = current_chip_dist[trapped_mask].sum() / 100.0
        # 3. 近期套牢盘（成本在当前价5-10%之间）
        recent_trapped_mask = (price_grid > current_price * 1.05) & (price_grid <= current_price * 1.10)
        metrics['recent_trapped_pressure'] = current_chip_dist[recent_trapped_mask].sum() / 100.0
        # 4. 支撑强度（当前价下方5%内的筹码）
        support_mask = (price_grid >= current_price * 0.95) & (price_grid < current_price)
        metrics['support_strength'] = current_chip_dist[support_mask].sum() / 100.0
        # 5. 阻力强度（当前价上方5%内的筹码）
        resistance_mask = (price_grid > current_price) & (price_grid <= current_price * 1.05)
        metrics['resistance_strength'] = current_chip_dist[resistance_mask].sum() / 100.0
        # 6. 压力释放度（需要历史数据）
        if not price_history.empty and len(price_history) >= 10:
            # 计算最近10天价格区间
            recent_low = price_history['low'].min()
            recent_high = price_history['high'].max()
            # 释放的套牢盘
            released_mask = (price_grid >= recent_high * 1.05) | (price_grid <= recent_low * 0.95)
            metrics['pressure_release'] = current_chip_dist[released_mask].sum() / 100.0
        else:
            metrics['pressure_release'] = 0.0
        # 7. 综合压力分数
        metrics['comprehensive_pressure'] = (
            metrics['trapped_pressure'] * 0.5 +
            metrics['recent_trapped_pressure'] * 0.3 +
            (1 - metrics['pressure_release']) * 0.2
        )
        return metrics
    
    def _identify_behavior_patterns(self,percent_change_matrix: np.ndarray,chip_matrix: np.ndarray,price_grid: np.ndarray,current_price: float) -> Dict[str, any]:
        """
        识别主力行为模式 - 基于绝对百分比变化
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
        # 分析最近3-5天的变化模式
        lookback = min(5, percent_change_matrix.shape[0])
        recent_changes = percent_change_matrix[-lookback:, :]
        # 1. 寻找显著的连续变化区域
        for price_idx, price in enumerate(price_grid):
            price_changes = recent_changes[:, price_idx]
            # 连续3天增加（吸筹迹象）
            if len(price_changes) >= 3:
                if np.all(price_changes[-3:] > self.params['noise_threshold']):
                    if price < current_price * 0.95:  # 低位吸筹
                        patterns['accumulation']['detected'] = True
                        patterns['accumulation']['strength'] += np.mean(price_changes[-3:]) / 10.0
                        patterns['accumulation']['areas'].append({
                            'price': float(price),
                            'avg_change': float(np.mean(price_changes[-3:])),
                            'distance_to_price': (current_price - price) / current_price
                        })
            # 连续3天减少（派发迹象）
            if len(price_changes) >= 3:
                if np.all(price_changes[-3:] < -self.params['noise_threshold']):
                    if price > current_price * 1.05:  # 高位派发
                        patterns['distribution']['detected'] = True
                        patterns['distribution']['strength'] += abs(np.mean(price_changes[-3:])) / 10.0
                        patterns['distribution']['areas'].append({
                            'price': float(price),
                            'avg_change': float(np.mean(price_changes[-3:])),
                            'distance_to_price': (price - current_price) / current_price
                        })
        # 2. 计算主力活跃度
        significant_changes = np.abs(recent_changes) > self.params['significant_change_threshold']
        patterns['main_force_activity'] = np.sum(significant_changes) / significant_changes.size
        # 3. 整理信号（吸筹/派发）
        if patterns['accumulation']['detected']:
            patterns['accumulation']['strength'] = min(1.0, patterns['accumulation']['strength'])
            patterns['accumulation']['areas'] = sorted(
                patterns['accumulation']['areas'],
                key=lambda x: x['avg_change'],
                reverse=True
            )[:5]
        if patterns['distribution']['detected']:
            patterns['distribution']['strength'] = min(1.0, patterns['distribution']['strength'])
            patterns['distribution']['areas'] = sorted(
                patterns['distribution']['areas'],
                key=lambda x: abs(x['avg_change']),
                reverse=True
            )[:5]
        # 4. 整理信号（震荡/突破准备）
        current_concentration = self._calculate_concentration_metrics(
            chip_matrix[-1], price_grid
        )['comprehensive_concentration']
        if 0.4 <= current_concentration <= 0.6:
            patterns['consolidation']['detected'] = True
            patterns['consolidation']['strength'] = 1.0 - abs(current_concentration - 0.5) * 2
        # 5. 突破准备：筹码在阻力位下方聚集
        resistance_idx = np.argmin(np.abs(price_grid - current_price * 1.05))
        if resistance_idx > 0:
            support_area = chip_matrix[-1, :resistance_idx].sum() / 100.0
            if support_area > 0.6:  # 60%以上筹码在阻力位下方
                patterns['breakout_preparation']['detected'] = True
                patterns['breakout_preparation']['strength'] = min(1.0, support_area)
        return patterns
    
    def _calculate_migration_patterns(self,percent_change_matrix: np.ndarray,chip_matrix: np.ndarray,price_grid: np.ndarray) -> Dict[str, any]:
        """
        计算筹码迁移模式
        """
        if percent_change_matrix.shape[0] < 2:
            return self._get_default_migration_patterns()
        patterns = {
            'upward_migration': {'strength': 0.0, 'volume': 0.0},
            'downward_migration': {'strength': 0.0, 'volume': 0.0},
            'convergence_migration': {'strength': 0.0, 'areas': []},
            'divergence_migration': {'strength': 0.0, 'areas': []},
            'net_migration_direction': 0.0  # 正值向上，负值向下
        }
        # 使用最近的变化矩阵
        recent_changes = percent_change_matrix[-1, :] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
        # 1. 计算净迁移方向
        price_center = np.sum(price_grid * chip_matrix[-1]) / 100.0
        weighted_changes = np.sum(recent_changes * price_grid) / np.sum(np.abs(recent_changes) + 1e-10)
        patterns['net_migration_direction'] = weighted_changes - price_center
        # 2. 向上迁移：低价区减少，高价区增加
        low_price_mask = price_grid < price_center * 0.9
        high_price_mask = price_grid > price_center * 1.1
        low_decrease = np.sum(recent_changes[low_price_mask][recent_changes[low_price_mask] < 0])
        high_increase = np.sum(recent_changes[high_price_mask][recent_changes[high_price_mask] > 0])
        patterns['upward_migration']['strength'] = min(1.0, abs(high_increase - low_decrease) / 50.0)
        patterns['upward_migration']['volume'] = abs(high_increase - low_decrease)
        # 3. 向下迁移：高价区减少，低价区增加
        high_decrease = np.sum(recent_changes[high_price_mask][recent_changes[high_price_mask] < 0])
        low_increase = np.sum(recent_changes[low_price_mask][recent_changes[low_price_mask] > 0])
        patterns['downward_migration']['strength'] = min(1.0, abs(high_decrease - low_increase) / 50.0)
        patterns['downward_migration']['volume'] = abs(high_decrease - low_increase)
        # 4. 收敛迁移：向中间价格聚集
        mid_price_mask = (price_grid >= price_center * 0.95) & (price_grid <= price_center * 1.05)
        mid_increase = np.sum(recent_changes[mid_price_mask][recent_changes[mid_price_mask] > 0])
        if mid_increase > 0:
            patterns['convergence_migration']['strength'] = min(1.0, mid_increase / 30.0)
            patterns['convergence_migration']['areas'] = [
                {'price': float(price_grid[i]), 'change': float(recent_changes[i])}
                for i in np.where(mid_price_mask & (recent_changes > 0))[0][:5]
            ]
        # 5. 发散迁移：从中间价格向两侧分散
        mid_decrease = np.sum(recent_changes[mid_price_mask][recent_changes[mid_price_mask] < 0])
        if mid_decrease < 0:
            patterns['divergence_migration']['strength'] = min(1.0, abs(mid_decrease) / 30.0)
            patterns['divergence_migration']['areas'] = [
                {'price': float(price_grid[i]), 'change': float(recent_changes[i])}
                for i in np.where(mid_price_mask & (recent_changes < 0))[0][:5]
            ]
        return patterns
    
    def _calculate_convergence_metrics(self,chip_matrix: np.ndarray,percent_change_matrix: np.ndarray,price_grid: np.ndarray) -> Dict[str, float]:
        """
        计算聚散度指标 - 基于百分比绝对变化
        """
        if chip_matrix.shape[0] < 2 or len(percent_change_matrix) == 0:
            return self._get_default_convergence_metrics()
        metrics = {}
        # 1. 静态聚散度（基于当前分布）
        current_chip = chip_matrix[-1]
        # 筹码熵（越低越聚集）
        chip_entropy = entropy(current_chip + 1e-10)
        max_entropy = np.log(len(current_chip))
        metrics['static_convergence'] = 1.0 - (chip_entropy / max_entropy)
        # 2. 动态聚散度（基于变化）
        recent_changes = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
        # 计算变化的方向一致性
        positive_changes = recent_changes[recent_changes > 0]
        negative_changes = recent_changes[recent_changes < 0]
        if len(positive_changes) > 0 and len(negative_changes) > 0:
            # 双向变化 = 发散
            divergence_ratio = min(len(positive_changes), len(negative_changes)) / len(recent_changes)
            metrics['dynamic_convergence'] = 1.0 - divergence_ratio
        else:
            # 单向变化 = 聚集
            metrics['dynamic_convergence'] = 1.0
        # 3. 迁移聚散度（净迁移方向强度）
        price_center = np.sum(price_grid * current_chip) / 100.0
        weighted_changes = np.sum(np.abs(recent_changes) * np.abs(price_grid - price_center)) / np.sum(np.abs(recent_changes) + 1e-10)
        max_distance = np.max(np.abs(price_grid - price_center))
        metrics['migration_convergence'] = 1.0 - (weighted_changes / max_distance)
        # 4. 综合聚散度
        weights = [0.4, 0.3, 0.3]
        values = [
            metrics['static_convergence'],
            metrics['dynamic_convergence'],
            metrics['migration_convergence']
        ]
        metrics['comprehensive_convergence'] = np.sum(np.array(weights) * np.array(values))
        # 5. 收敛强度（正值收敛，负值发散）
        net_change_direction = np.sum(recent_changes * (price_grid - price_center))
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
            print(f"🕵️ [PROBE-FETCH] {stock_code} {trade_date} (lookback={lookback_days})")
            print(f"   > Start Date: {start_date}")
            print(f"   > Current Rows: {len(current_chip_list)}")
            print(f"   > History Raw Rows: {len(history_chip_list)}")
            # 按日期分组
            chip_history = []
            if history_chip_list:
                history_df = pd.DataFrame(history_chip_list)
                unique_dates = history_df['trade_time'].unique()
                print(f"   > History Unique Dates: {len(unique_dates)}")
                for date in unique_dates:
                    day_df = history_df[history_df['trade_time'] == date][['price', 'percent']]
                    chip_history.append(day_df)
            # 获取价格历史
            from utils.model_helpers import get_daily_data_model_by_code
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