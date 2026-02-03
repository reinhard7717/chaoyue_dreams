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

    async def analyze_chip_dynamics_daily(self, stock_code: str, trade_date: str, lookback_days: int = 20, tick_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """
        分析单日筹码动态 - 主入口函数（增强版：支持tick数据）
        Args:
            tick_data: 可选的tick数据DataFrame，包含['trade_time', 'price', 'volume', 'type']
        修改思路:
        1. 调用_calculate_tick_enhanced_factors时传入trade_date，确保底层能获取准确日期。
        """
        try:
            # 1. 获取筹码分布历史数据
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
                chip_data['price_history'],
                stock_code,
                trade_date
            )
            # 11. 计算直接吸收/派发
            direct_ad_result = self.direct_ad_calculator.calculate_direct_ad(
                percent_change_matrix,
                chip_matrix,
                price_grid,
                chip_data['current_price'],
                chip_data['price_history']
            )
            # =======================================================
            # 新增：计算tick数据增强因子
            # =======================================================
            tick_enhanced_factors = {}
            if tick_data is not None and not tick_data.empty:
                try:
                    # 修改：传入trade_date参数
                    tick_enhanced_factors = await self._calculate_tick_enhanced_factors(
                        tick_data, chip_data, price_grid, chip_matrix[-1], trade_date
                    )
                except Exception as e:
                    print(f"⚠️ [tick因子] 计算失败: {e}")
                    tick_enhanced_factors = self._get_default_tick_factors()
            else:
                tick_enhanced_factors = self._get_default_tick_factors()
            # =======================================================
            # 构建验证信息
            # =======================================================
            validation_warnings = []
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
            # tick数据质量检查
            if 'tick_data_quality_score' in tick_enhanced_factors:
                tick_quality = tick_enhanced_factors['tick_data_quality_score']
                if tick_quality < 0.3:
                    validation_warnings.append(f"tick数据质量低: {tick_quality:.2f}")
                    validation_score *= 0.9
            result = {
                'stock_code': stock_code,
                'trade_date': trade_date,
                'price_grid': price_grid.tolist(),
                'chip_matrix': chip_matrix.tolist(),
                'percent_change_matrix': percent_change_matrix.tolist(),
                'absolute_change_signals': absolute_signals,
                'concentration_metrics': concentration_metrics,
                'pressure_metrics': pressure_metrics,
                'behavior_patterns': behavior_patterns,
                'migration_patterns': migration_patterns,
                'convergence_metrics': convergence_metrics,
                'game_energy_result': game_energy_result,
                'direct_ad_result': direct_ad_result,
                # 新增tick增强因子
                'tick_enhanced_factors': tick_enhanced_factors,
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

    async def _calculate_tick_enhanced_factors(self, tick_data: pd.DataFrame, chip_data: Dict[str, Any],price_grid: np.ndarray,current_chip_dist: np.ndarray, trade_date: str = "") -> Dict[str, Any]:
        """
        计算tick数据增强因子 (v1.4)
        修改说明:
        1. 增加数据完整性保底检查：只要包含必要列且行数>50，即使评分低也继续计算。
        2. 解决因成交量小导致评分低而被过滤的问题，优先保证数据完整性。
        3. 保留索引/列名处理和时区修正逻辑。
        """
        try:
            if tick_data.empty:
                return self._get_default_tick_factors()
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
                        print(f"⚠️ [tick因子] trade_time 转换失败: {e}")
                if not tick_data.empty:
                    hours = tick_data['trade_time'].dt.hour
                    bj_time_ratio = ((hours >= 9) & (hours <= 15)).mean()
                    utc_time_ratio = ((hours >= 1) & (hours <= 7)).mean()
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
            close_price = current_price
            processed_tick, data_quality = ChipFactorCalculator.preprocess_tick_data(tick_data)
            is_data_complete = False
            required_cols = ['price', 'volume', 'trade_time']
            if not processed_tick.empty and len(processed_tick) > 50:
                if all(col in processed_tick.columns for col in required_cols):
                    is_data_complete = True
            if data_quality < self.params['tick_data_quality_threshold']:
                if is_data_complete:
                    print(f"ℹ️ [tick因子] {date_str} 质量评分较低 ({data_quality:.2f}) 但数据完整 (行数: {len(processed_tick)})，继续计算")
                    data_quality = max(data_quality, self.params['tick_data_quality_threshold'])
                else:
                    print(f"⚠️ [tick因子-探针] {date_str} 数据质量低 ({data_quality:.2f} < {self.params['tick_data_quality_threshold']})，原因分析:")
                    print(f"   - 原始行数: {len(tick_data)}")
                    print(f"   - 处理后行数: {len(processed_tick)}")
                    print(f"   - 包含列名: {list(tick_data.columns)}")
                    if 'volume' in tick_data.columns:
                        vol_sum = tick_data['volume'].sum()
                        vol_mean = tick_data['volume'].mean()
                        print(f"   - 总成交量: {vol_sum:.0f}, 平均成交量: {vol_mean:.2f}")
                    else:
                        print(f"   - 缺失 'volume' 列")
                    if 'trade_time' in tick_data.columns and not tick_data.empty:
                        try:
                            t_min = tick_data['trade_time'].min()
                            t_max = tick_data['trade_time'].max()
                            if hasattr(t_min, 'strftime'):
                                print(f"   - 时间范围: {t_min.strftime('%H:%M:%S')} -> {t_max.strftime('%H:%M:%S')}")
                            else:
                                print(f"   - 时间范围: {t_min} -> {t_max}")
                        except Exception as e:
                            print(f"   - 时间解析失败: {e}")
                    else:
                        print(f"   - 缺失 'trade_time' 列或数据为空")
                    return self._get_default_tick_factors()
            factors = {
                'tick_data_quality_score': data_quality,
                'intraday_factor_calc_method': 'tick_based',
            }
            intraday_dist = ChipFactorCalculator.calculate_intraday_chip_distribution(processed_tick)
            if intraday_dist:
                factors['intraday_chip_concentration'] = intraday_dist.get('concentration', 0.0)
                factors['intraday_chip_entropy'] = intraday_dist.get('entropy', 0.0)
                factors['intraday_price_distribution_skewness'] = intraday_dist.get('skewness', 0.0)
            intraday_flow = ChipFactorCalculator.calculate_intraday_chip_flow(processed_tick)
            if intraday_flow:
                factors['tick_level_chip_flow'] = intraday_flow.get('net_flow_ratio', 0.0)
                factors['intraday_chip_turnover_intensity'] = intraday_flow.get('flow_intensity', 0.0)
                factors['tick_clustering_index'] = intraday_flow.get('clustering_index', 0.0)
                factors['tick_chip_balance_ratio'] = intraday_flow.get('buy_ratio', 0.5) / max(0.01, intraday_flow.get('sell_ratio', 0.5))
            cost_center = ChipFactorCalculator.calculate_intraday_cost_center(processed_tick)
            if cost_center:
                factors['intraday_cost_center_migration'] = cost_center.get('migration_ratio', 0.0)
                factors['intraday_cost_center_volatility'] = cost_center.get('volatility', 0.0)
            chip_dist_df = pd.DataFrame({
                'price': price_grid,
                'percent': current_chip_dist
            })
            support_resistance = ChipFactorCalculator.identify_intraday_support_resistance(
                processed_tick, chip_dist_df
            )
            if support_resistance:
                factors['intraday_dynamic_support_test_count'] = support_resistance.get('support_test_count', 0)
                factors['intraday_dynamic_resistance_test_count'] = support_resistance.get('resistance_test_count', 0)
                factors['intraday_chip_consolidation_degree'] = support_resistance.get('consolidation_degree', 0.0)
            abnormal_volume = ChipFactorCalculator.calculate_intraday_abnormal_volume(processed_tick)
            if abnormal_volume:
                factors['tick_abnormal_volume_ratio'] = abnormal_volume.get('abnormal_volume_ratio', 0.0)
                factors['tick_chip_transfer_efficiency'] = abnormal_volume.get('transfer_efficiency', 0.0)
            chip_locking = ChipFactorCalculator.calculate_intraday_chip_locking(processed_tick, current_price)
            if chip_locking:
                factors['intraday_low_lock_ratio'] = chip_locking.get('low_lock_ratio', 0.0)
                factors['intraday_high_lock_ratio'] = chip_locking.get('high_lock_ratio', 0.0)
                factors['intraday_peak_valley_ratio'] = chip_locking.get('peak_valley_ratio', 0.0)
                factors['intraday_trough_filling_degree'] = chip_locking.get('trough_filling', 0.0)
            game_index = ChipFactorCalculator.calculate_intraday_chip_game_index(processed_tick)
            factors['intraday_chip_game_index'] = game_index
            factors['intraday_main_force_activity'] = self._calculate_main_force_activity(
                processed_tick, intraday_flow, abnormal_volume
            )
            accumulation_confidence, distribution_confidence = self._calculate_accumulation_distribution_confidence(
                intraday_flow, chip_locking, support_resistance
            )
            factors['intraday_accumulation_confidence'] = accumulation_confidence
            factors['intraday_distribution_confidence'] = distribution_confidence
            factors['tick_data_summary'] = {
                'total_ticks': len(processed_tick),
                'time_span_hours': self._calculate_tick_time_span(processed_tick),
                'avg_volume': float(processed_tick['volume'].mean() if not processed_tick.empty else 0),
                'price_range': float(processed_tick['price'].max() - processed_tick['price'].min() if not processed_tick.empty else 0),
            }
            factors['intraday_market_microstructure'] = self._calculate_market_microstructure(processed_tick)
            return factors
        except Exception as e:
            print(f"❌ [tick因子] 计算异常: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_tick_factors()

    def _build_normalized_chip_matrix(self, chip_history: List[pd.DataFrame], current_chip: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        版本: v2.1
        说明: 构建归一化筹码矩阵（Pandas操作优化版）
        修改思路: 
        1. 优化价格范围计算，使用np.concatenate一次性处理。
        2. 优化DataFrame清洗，使用dropna替代复杂的布尔索引。
        3. 保持np.interp的高效插值。
        """
        all_chips = chip_history + [current_chip]
        # 收集所有价格以确定网格范围
        all_prices = []
        for chip_df in all_chips:
            if not chip_df.empty:
                # 直接获取values，避免Series开销
                all_prices.append(chip_df['price'].values)
        if not all_prices:
            min_price, max_price = 1.0, 100.0
        else:
            # 使用concatenate一次性合并，比extend循环更高效
            flat_prices = np.concatenate(all_prices)
            if len(flat_prices) == 0:
                min_price, max_price = 1.0, 100.0
            else:
                min_price, max_price = np.min(flat_prices), np.max(flat_prices)
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
                # 优化：直接dropna，比布尔索引更快
                df_valid = chip_df.dropna(subset=['price', 'percent'])
                if df_valid.empty:
                    chip_matrix[day_idx, :] = uniform_dist
                    continue
                # 排序和去重
                df_valid = df_valid.sort_values('price').drop_duplicates('price')
                if len(df_valid) < 2:
                    chip_matrix[day_idx, :] = uniform_dist
                    continue
                x = df_valid['price'].values
                y = df_valid['percent'].values
                total_p = np.sum(y)
                if total_p == 0:
                    chip_matrix[day_idx, :] = uniform_dist
                    continue
                # 归一化并插值
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
            recent_low = float(price_history['low_qfq'].min())
            recent_high = float(price_history['high_qfq'].max())
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

    def _calculate_main_force_activity(self, tick_data: pd.DataFrame, intraday_flow: Dict[str, float],abnormal_volume: Dict[str, float]) -> float:
        """计算主力活跃度"""
        try:
            activity_score = 0.0
            # 1. 异常成交量权重
            if abnormal_volume:
                abnormal_ratio = abnormal_volume.get('abnormal_volume_ratio', 0.0)
                activity_score += min(0.4, abnormal_ratio * 2)
            # 2. 大单占比（假设成交量>平均3倍为大单）
            if not tick_data.empty:
                avg_volume = tick_data['volume'].mean()
                large_order_mask = tick_data['volume'] > avg_volume * 3
                large_order_ratio = large_order_mask.sum() / len(tick_data)
                activity_score += min(0.3, large_order_ratio * 3)
            # 3. 买卖不平衡度
            if intraday_flow:
                buy_ratio = intraday_flow.get('buy_ratio', 0.5)
                sell_ratio = intraday_flow.get('sell_ratio', 0.5)
                imbalance = abs(buy_ratio - sell_ratio)
                activity_score += min(0.3, imbalance * 2)
            return min(1.0, activity_score)
        except Exception as e:
            print(f"⚠️ 主力活跃度计算失败: {e}")
            return 0.0

    def _calculate_accumulation_distribution_confidence(self, intraday_flow: Dict[str, float],chip_locking: Dict[str, float],support_resistance: Dict[str, Any]) -> Tuple[float, float]:
        """计算吸筹/派发置信度"""
        accumulation_confidence = 0.0
        distribution_confidence = 0.0
        try:
            # 1. 基于筹码流动判断
            if intraday_flow:
                net_flow = intraday_flow.get('net_flow_ratio', 0.0)
                if net_flow > 0.1:  # 净流入
                    accumulation_confidence += 0.3
                elif net_flow < -0.1:  # 净流出
                    distribution_confidence += 0.3
            # 2. 基于筹码锁定判断
            if chip_locking:
                low_lock = chip_locking.get('low_lock_ratio', 0.0)
                high_lock = chip_locking.get('high_lock_ratio', 0.0)
                if low_lock > 0.1:
                    accumulation_confidence += 0.2  # 低位锁定，可能是吸筹
                if high_lock > 0.15:
                    distribution_confidence += 0.2  # 高位锁定，可能是派发或套牢
            # 3. 基于支撑阻力测试
            if support_resistance:
                support_tests = support_resistance.get('support_test_count', 0)
                resistance_tests = support_resistance.get('resistance_test_count', 0)
                if support_tests > resistance_tests * 2:
                    accumulation_confidence += 0.2  # 支撑测试多，可能是吸筹
                elif resistance_tests > support_tests * 2:
                    distribution_confidence += 0.2  # 阻力测试多，可能是派发
            # 4. 博弈指数影响
            if intraday_flow and 'clustering_index' in intraday_flow:
                clustering = intraday_flow['clustering_index']
                if clustering > 0.7:  # 高度聚类，可能是主力行为
                    if accumulation_confidence > distribution_confidence:
                        accumulation_confidence += 0.1
                    else:
                        distribution_confidence += 0.1
            return min(1.0, accumulation_confidence), min(1.0, distribution_confidence)
        except Exception as e:
            print(f"⚠️ 置信度计算失败: {e}")
            return 0.0, 0.0

    def _calculate_tick_time_span(self, tick_data: pd.DataFrame) -> float:
        """计算tick数据时间跨度（小时）"""
        try:
            if tick_data.empty:
                return 0.0
            time_min = tick_data['trade_time'].min()
            time_max = tick_data['trade_time'].max()
            time_span = (time_max - time_min).total_seconds() / 3600
            return float(time_span)
        except Exception as e:
            print(f"⚠️ 时间跨度计算失败: {e}")
            return 0.0

    def _calculate_market_microstructure(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """计算市场微观结构指标"""
        try:
            if tick_data.empty or len(tick_data) < 10:
                return {}
            microstructure = {}
            # 1. 价格变动分布
            if len(tick_data) >= 2:
                price_changes = tick_data['price'].diff().dropna()
                microstructure['price_change_mean'] = float(price_changes.mean())
                microstructure['price_change_std'] = float(price_changes.std())
                microstructure['price_change_skewness'] = float(price_changes.skew())
            # 2. 成交量分布
            volume_series = tick_data['volume']
            microstructure['volume_mean'] = float(volume_series.mean())
            microstructure['volume_std'] = float(volume_series.std())
            microstructure['volume_skewness'] = float(volume_series.skew())
            # 3. 买卖强度
            if 'type' in tick_data.columns:
                buy_mask = tick_data['type'] == 'B'
                sell_mask = tick_data['type'] == 'S'
                buy_volume = tick_data.loc[buy_mask, 'volume'].sum() if buy_mask.any() else 0
                sell_volume = tick_data.loc[sell_mask, 'volume'].sum() if sell_mask.any() else 0
                total_volume = buy_volume + sell_volume
                if total_volume > 0:
                    microstructure['buy_strength'] = float(buy_volume / total_volume)
                    microstructure['sell_strength'] = float(sell_volume / total_volume)
            # 4. 时间间隔分布
            if len(tick_data) >= 3:
                time_diffs = tick_data['trade_time'].diff().dt.total_seconds().dropna()
                microstructure['avg_time_gap'] = float(time_diffs.mean())
                microstructure['time_gap_std'] = float(time_diffs.std())
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
                ).order_by('trade_time').values('trade_time', 'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq')
                price_list = await sync_to_async(list)(price_qs)
                price_history = pd.DataFrame(price_list) if price_list else pd.DataFrame()
            # 修改价格获取逻辑
            current_price = 0
            # 优先使用日线收盘价
            if not price_history.empty and 'close_qfq' in price_history.columns:
                current_price = price_history['close_qfq'].iloc[-1]
            # 如果没有日线数据，使用筹码平均价作为后备
            elif not current_chip_df.empty:
                current_price = current_chip_df['price'].mean()
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
    
    def calculate_direct_ad(self, percent_change_matrix: np.ndarray,chip_matrix: np.ndarray,price_grid: np.ndarray,current_price: float,price_history: pd.DataFrame) -> Dict[str, any]:
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
        版本: v2.1
        说明: 基于绝对变化的直接计算吸收/派发（Numpy分箱优化版）
        修改思路: 
        1. 使用np.digitize替代多次布尔掩码，将O(K*N)复杂度降为O(N)。
        2. 使用np.bincount快速聚合各区间的正负变化量。
        3. 向量化计算各区间的吸收/派发贡献。
        """
        # 过滤噪声
        significant_mask = np.abs(changes) > self.params['noise_filter']
        clean_changes = changes[significant_mask]
        clean_rels = price_rel[significant_mask]
        if len(clean_changes) == 0:
            return {
                'accumulation_volume': 0.0,
                'distribution_volume': 0.0,
                'net_ad_ratio': 0.0,
                'accumulation_quality': 0.5,
                'distribution_quality': 0.5,
                'false_distribution_flag': False,
                'breakout_acceleration': 1.0,
            }

        # 定义区间边界 [-0.12, -0.03, 0, 0.05, 0.12]
        # 0: deep_below (< -0.12)
        # 1: below (-0.12 ~ -0.03)
        # 2: near_below (-0.03 ~ 0)
        # 3: near_above (0 ~ 0.05)
        # 4: above (0.05 ~ 0.12)
        # 5: deep_above (>= 0.12)
        bins = np.array([-0.12, -0.03, 0, 0.05, 0.12])
        indices = np.digitize(clean_rels, bins)
        # 分离正负变化
        pos_changes = np.maximum(clean_changes, 0)
        neg_changes = np.abs(np.minimum(clean_changes, 0))
        # 聚合各区间的总量 (minlength=6 确保所有区间都有值)
        pos_sums = np.bincount(indices, weights=pos_changes, minlength=6)
        neg_sums = np.bincount(indices, weights=neg_changes, minlength=6)
        # 定义各区间的权重
        # 索引: 0, 1, 2, 3, 4, 5
        zone_weights = np.array([0.4, 1.3, 1.0, 1.0, 1.2, 0.5])
        # 吸收系数 (Accumulation Multipliers)
        # deep_below(1.0), below(1.0), near_below(1.0), near_above(0.6), above(0.3), deep_above(0.3)
        accum_mults = np.array([1.0, 1.0, 1.0, 0.6, 0.3, 0.3])
        # 派发系数 (Distribution Multipliers)
        # deep_below(0.8), below(0.8), near_below(0.8), near_above(0.9), above(1.0), deep_above(1.0)
        distrib_mults = np.array([0.8, 0.8, 0.8, 0.9, 1.0, 1.0])
        # 向量化计算基础吸收/派发量
        # 逻辑保持一致：
        # Below区域: Pos -> Accum, Neg -> Distrib
        # Above区域: Pos -> Accum*0.3, Neg -> Distrib
        raw_accum = pos_sums * zone_weights * accum_mults
        raw_distrib = neg_sums * zone_weights * distrib_mults
        accumulation_volume = np.sum(raw_accum)
        distribution_volume = np.sum(raw_distrib)
        # 趋势修正
        overall_trend = np.sum(clean_changes)
        if overall_trend > 0:
            accumulation_volume *= 1.2
            distribution_volume *= 0.8
        elif overall_trend < 0:
            accumulation_volume *= 0.8
            distribution_volume *= 1.2
        total_volume = accumulation_volume + distribution_volume + 1e-10
        net_ad_ratio = (accumulation_volume - distribution_volume) / total_volume
        return {
            'accumulation_volume': float(accumulation_volume),
            'distribution_volume': float(distribution_volume),
            'net_ad_ratio': float(net_ad_ratio),
            'accumulation_quality': 0.5,
            'distribution_quality': 0.5,
            'false_distribution_flag': False,
            'breakout_acceleration': 1.0,
        }

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
    
    def calculate_game_energy(self, percent_change_matrix: np.ndarray,
                            price_grid: np.ndarray,
                            current_price: float, close_price: float,
                            volume_history: pd.Series = None,
                            stock_code: str = "",
                            trade_date: str = "") -> Dict[str, Any]:
        """
        计算博弈能量场 - 修复空数据问题
        """
        # 优先使用收盘价作为参考价格
        reference_price = close_price if close_price > 0 else current_price
        if percent_change_matrix.shape[0] == 0 or len(price_grid) == 0 or reference_price <= 0:
            print(f"❌ [探针] 输入数据无效: 变化矩阵{percent_change_matrix.shape}, 价格网格{len(price_grid)}, 参考价{reference_price}")
            result = self._get_default_energy()
            return result
        try:
            # 1. 获取最近一天的变化（确保有数据）
            if len(percent_change_matrix) == 0:
                print("❌ [探针] 变化矩阵为空")
                result = self._get_default_energy()
                print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
                return result
            latest_change = percent_change_matrix[-1] if len(percent_change_matrix) > 0 else np.zeros(len(price_grid))
            # 2. 如果变化数据全为0，返回默认值
            change_sum = np.sum(np.abs(latest_change))
            if change_sum < 0.01:
                print("⚠️ [探针] 变化数据几乎全为0，返回默认值")
                result = self._get_default_energy()
                print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
                return result
            # 3. 计算博弈能量场
            energy_result = self._calculate_energy_field(latest_change, price_grid, reference_price, close_price)
            # 5. 判断虚假派发（只有在有成交量数据时才判断）
            fake_distribution = False
            if volume_history is not None and len(volume_history) > 5:
                fake_distribution = self._detect_fake_distribution(latest_change, price_grid, reference_price, volume_history)
            else:
                # 没有成交量数据时，基于价格变化判断
                fake_distribution = self._detect_fake_distribution_advanced(latest_change, price_grid, reference_price, close_price)
            energy_result['fake_distribution_flag'] = fake_distribution
            # 6. 修正能量值（确保不为0）
            energy_result = self._ensure_nonzero_energy(energy_result)
            return energy_result
        except Exception as e:
            print(f"❌ [探针] 能量场计算异常: {e}")
            import traceback
            traceback.print_exc()
            result = self._get_default_energy()
            print(f"   返回默认值: absorption={result['absorption_energy']}, distribution={result['distribution_energy']}")
            return result

    def _calculate_energy_field(self, changes: np.ndarray, price_grid: np.ndarray, 
                                     current_price: float, close_price: float,
                                     stock_code: str = "", trade_date: str = "") -> Dict[str, Any]:
        """
        版本: v2.1
        说明: 能量场计算（Numpy分箱优化版）
        修改思路: 
        1. 使用np.digitize和np.bincount替代循环掩码，提高计算效率。
        2. 保持原有的A股博弈权重逻辑。
        """
        reference_price = close_price if close_price > 0 else current_price
        if len(changes) == 0 or len(price_grid) == 0 or reference_price <= 0:
            return self._get_default_energy()
        try:
            price_rel = (price_grid - reference_price) / reference_price
            # 定义区间边界 [-0.15, -0.05, 0, 0.05, 0.15]
            # 0: deep_below (< -0.15)
            # 1: below (-0.15 ~ -0.05)
            # 2: near_below (-0.05 ~ 0)
            # 3: near_above (0 ~ 0.05)
            # 4: above (0.05 ~ 0.15)
            # 5: deep_above (>= 0.15)
            bins = np.array([-0.15, -0.05, 0, 0.05, 0.15])
            indices = np.digitize(price_rel, bins)
            # 分离正负变化
            pos_changes = np.maximum(changes, 0)
            neg_changes = np.abs(np.minimum(changes, 0))
            # 聚合各区间的总量
            pos_sums = np.bincount(indices, weights=pos_changes, minlength=6)
            neg_sums = np.bincount(indices, weights=neg_changes, minlength=6)
            # 区域权重
            # deep_below(0.6), below(0.9), near_below(1.5), near_above(1.3), above(1.0), deep_above(0.7)
            weights = np.array([0.6, 0.9, 1.5, 1.3, 1.0, 0.7])
            absorption_advanced = 0.0
            distribution_advanced = 0.0
            # 1. 价格以下区域 (Bins 0, 1, 2)
            # 增加(Pos) -> 吸筹, 减少(Neg) -> 派发 * 0.8
            for i in range(3):
                absorption_advanced += pos_sums[i] * weights[i]
                distribution_advanced += neg_sums[i] * weights[i] * 0.8
            # 2. 价格以上区域 (Bins 3, 4)
            # 增加(Pos) -> 派发, 减少(Neg) -> 吸筹 * 0.7
            for i in range(3, 5):
                distribution_advanced += pos_sums[i] * weights[i]
                absorption_advanced += neg_sums[i] * weights[i] * 0.7
            # 3. 深度获利区 (Bin 5) - 特殊逻辑
            i = 5
            w = weights[i]
            inc_sum = pos_sums[i]
            dec_sum = neg_sums[i]
            if inc_sum > dec_sum:
                # 增加多于减少 -> 派发主导
                distribution_advanced += inc_sum * w * 1.2
                absorption_advanced += dec_sum * w * 0.4
            else:
                # 减少多于增加 -> 洗盘/换手
                distribution_advanced += inc_sum * w * 0.6
                absorption_advanced += dec_sum * w * 0.9
            # 计算其他指标
            game_intensity, breakout_potential, energy_concentration = self._calculate_energy_indicators(
                changes, price_grid, reference_price, stock_code, trade_date
            )
            key_battle_zones = self._identify_key_battle_zones(changes, price_grid, reference_price, stock_code, trade_date)
            return {
                'absorption_energy': min(100, max(0.01, absorption_advanced)),
                'distribution_energy': min(100, max(0.01, distribution_advanced)),
                'net_energy_flow': absorption_advanced - distribution_advanced,
                'game_intensity': min(1.0, max(0, game_intensity)),
                'key_battle_zones': key_battle_zones,
                'breakout_potential': min(100, breakout_potential),
                'energy_concentration': min(1.0, max(0, energy_concentration)),
                'reference_price': reference_price,
                'original_current_price': current_price,
            }
        except Exception as e:
            print(f"❌ [探针-能量场] {stock_code} {trade_date} 计算异常: {e}")
            return self._get_default_energy()

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
    
    def _calculate_energy_indicators(self, changes: np.ndarray, price_grid: np.ndarray, 
                                         current_price: float, stock_code: str = "", trade_date: str = "") -> tuple:
        """调试版的能量指标计算 - 优化突破势能"""
        # 1. 计算能量集中度（先计算，因为后面会用到）
        abs_changes = np.abs(changes)
        total_energy = np.sum(abs_changes)
        energy_concentration = 0.0
        if total_energy > 0:
            # 计算top 20%的变化占比
            sorted_indices = np.argsort(abs_changes)[::-1]
            top_count = max(1, int(len(changes) * 0.2))
            top_energy = np.sum(abs_changes[sorted_indices[:top_count]])
            energy_concentration = top_energy / total_energy
        # 2. 博弈强度（活跃区域比例）
        active_threshold = 0.2  # 变化绝对值大于0.2%的区域
        active_mask = np.abs(changes) > active_threshold
        active_count = np.sum(active_mask)
        total_count = len(changes)
        game_intensity = active_count / total_count * 2.0 if total_count > 0 else 0
        # 3. 优化突破势能计算（A股特性）
        above_mask = price_grid > current_price
        below_mask = price_grid < current_price
        # 上方吸收（价格以上筹码增加）
        absorption_above = np.sum(changes[above_mask & (changes > 0)])
        # 上方派发（价格以上筹码减少）
        distribution_above = np.sum(np.abs(changes[above_mask & (changes < 0)]))
        # 下方吸收（价格以下筹码增加）
        absorption_below = np.sum(changes[below_mask & (changes > 0)])
        # 下方派发（价格以下筹码减少）
        distribution_below = np.sum(np.abs(changes[below_mask & (changes < 0)]))
        # 突破势能计算优化：
        # 1. 下方支撑强度 = 下方吸收 / (下方派发 + 1e-10)
        # 2. 上方突破压力 = 上方派发 / (上方吸收 + 1e-10)
        # 3. 净突破能量 = (吸收_above - 派发_above) * 支撑强度 * 系数
        if distribution_below > 0:
            support_strength = absorption_below / distribution_below
        else:
            support_strength = absorption_below * 2 + 1  # 无派发时支撑更强
        # 限制支撑强度范围
        support_strength = min(3.0, max(0.1, support_strength))
        # 计算净上方能量
        net_above = absorption_above - distribution_above
        # 突破势能 = 净上方能量 * 支撑强度 * 放大系数
        # A股特性：突破需要较强的能量
        if net_above > 0:
            breakout_potential = net_above * support_strength * 10
        else:
            # 上方净减少，突破可能性低
            breakout_potential = max(0, net_above) * 5
        # 能量集中度加成：能量越集中，突破越容易
        if energy_concentration > 0.8:
            breakout_potential *= (1 + energy_concentration)
        return game_intensity, breakout_potential, energy_concentration

    def _identify_key_battle_zones(self, changes: np.ndarray, price_grid: np.ndarray, current_price: float, stock_code: str = "", trade_date: str = "") -> List[Dict]:
        """
        版本: v2.1
        说明: 关键博弈区域识别（Numpy优化版）
        修改思路: 优化邻域对抗强度的计算，使用Numpy切片和向量化操作替代列表推导式。
        """
        battle_zones = []
        min_intensity = 0.5
        try:
            # 筛选出显著变化的索引
            significant_indices = np.where(np.abs(changes) > min_intensity)[0]
            for i in significant_indices:
                change = changes[i]
                price = price_grid[i]
                # 定义邻域范围
                start_idx = max(0, i - 2)
                end_idx = min(len(changes), i + 3)
                neighborhood = changes[start_idx:end_idx]
                # 向量化计算对抗强度
                # 寻找符号相反的变化 (乘积 < 0)
                opponent_mask = (neighborhood * change) < 0
                if np.any(opponent_mask):
                    opponent_changes = neighborhood[opponent_mask]
                    opponent_avg = np.mean(np.abs(opponent_changes))
                    battle_intensity = abs(change) + opponent_avg * 0.5
                    if battle_intensity > min_intensity * 1.5:
                        battle_zones.append({
                            'price': float(price),
                            'battle_intensity': float(battle_intensity),
                            'type': 'absorption' if change > 0 else 'distribution',
                            'position': 'below_current' if price < current_price else 'above_current',
                            'distance_to_current': float((price - current_price) / current_price),
                        })
            # 按强度排序并限制数量
            battle_zones.sort(key=lambda x: x['battle_intensity'], reverse=True)
            return battle_zones[:5]
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












