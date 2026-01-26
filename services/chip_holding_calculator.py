# services/chip_holding_service.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from utils.model_helpers import get_chip_holding_matrix_model_by_code

logger = logging.getLogger(__name__)

class ChipHoldingService:
    """
    基于1分钟数据和逐笔数据的精确筹码持有时间计算服务
    """
    def __init__(self, use_tick_data: bool = True):
        """
        初始化计算服务
        Args:
            use_tick_data: 是否使用逐笔数据增强计算
        """
        self.use_tick_data = use_tick_data
        self.price_grid_size = 200  # 价格网格数量
        self.max_holding_days = 250  # 最大追踪持有天数
        # 从model_helpers导入必要的函数
        from utils.model_helpers import (
            get_minute_data_model_by_code_and_timelevel,
            get_stock_tick_data_model_by_code,
            get_cyq_chips_model_by_code,
            get_daily_data_model_by_code
        )
        self.get_minute_data_model = get_minute_data_model_by_code_and_timelevel
        self.get_tick_data_model = get_stock_tick_data_model_by_code
        self.get_chips_model = get_cyq_chips_model_by_code
        self.get_daily_data_model = get_daily_data_model_by_code

    async def calculate_holding_matrix_daily_async(
        self,
        stock_code: str,
        trade_date: str,
        lookback_days: int = 60
    ) -> Dict[str, any]:
        """
        计算单日筹码持有时间矩阵（异步版本）
        """
        try:
            logger.info(f"开始计算 {stock_code} {trade_date} 的筹码持有时间矩阵")
            # 1. 获取基础数据（异步）
            data_dict = await self._fetch_required_data(stock_code, trade_date, lookback_days)
            if not data_dict:
                logger.error(f"获取基础数据失败: {stock_code} {trade_date}")
                return self._get_default_result()
            # 2. 建立价格网格（同步计算）
            price_grid, chip_dist_matrix = await sync_to_async(self._build_price_grid_and_chip_matrix)(
                data_dict['chip_dists'], data_dict['price_range']
            )
            # 3. 计算分钟级成交量分布（同步计算）
            minute_volume_dist = await sync_to_async(self._calculate_minute_volume_distribution)(
                data_dict['minute_data'], price_grid
            )
            # 4. 如果使用逐笔数据，进行增强计算
            if self.use_tick_data and data_dict['tick_data'] is not None:
                enhanced_dist = await sync_to_async(self._enhance_with_tick_data)(
                    minute_volume_dist, data_dict['tick_data'], price_grid
                )
            else:
                enhanced_dist = minute_volume_dist
            # 5. 计算换手率矩阵（同步计算）
            turnover_matrix = await sync_to_async(self._calculate_turnover_matrix)(
                enhanced_dist, chip_dist_matrix, data_dict['float_shares']
            )
            # 6. 优化换手概率参数（同步计算）
            optimal_params = await sync_to_async(self._optimize_turnover_parameters)(
                turnover_matrix, chip_dist_matrix, data_dict['daily_turnover']
            )
            # 7. 计算持有时间矩阵（同步计算）
            holding_matrix = await sync_to_async(self._calculate_holding_matrix)(
                chip_dist_matrix, turnover_matrix, optimal_params
            )
            # 8. 计算衍生因子（同步计算）
            factors = await sync_to_async(self._calculate_holding_factors)(
                holding_matrix, chip_dist_matrix, data_dict
            )
            # 9. 验证结果（同步计算）
            validation = await sync_to_async(self._validate_results)(
                holding_matrix, factors, data_dict
            )
            result = {
                'stock_code': stock_code,
                'trade_date': trade_date,
                'holding_matrix': holding_matrix,
                'price_grid': price_grid,
                'factors': factors,
                'validation': validation,
                'calc_status': 'success',
                'calc_time': datetime.now()
            }
            logger.info(f"计算完成 {stock_code} {trade_date}: "
                       f"短线筹码={factors.get('short_term_ratio', 0):.2%}, "
                       f"长线筹码={factors.get('long_term_ratio', 0):.2%}")
            return result
        except Exception as e:
            logger.error(f"计算筹码持有矩阵失败 {stock_code} {trade_date}: {e}", exc_info=True)
            return self._get_default_result(stock_code, trade_date)

    def calculate_holding_matrix_daily(
        self,
        stock_code: str,
        trade_date: str,
        lookback_days: int = 60
    ) -> Dict[str, any]:
        """
        计算单日筹码持有时间矩阵（主入口函数，同步包装）
        """
        try:
            # 创建事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.calculate_holding_matrix_daily_async(stock_code, trade_date, lookback_days)
                )
            finally:
                loop.close()
            return result
        except Exception as e:
            logger.error(f"计算筹码持有矩阵失败 {stock_code} {trade_date}: {e}")
            return self._get_default_result(stock_code, trade_date)

    def calculate_batch_holding_matrices(
        self,
        stock_codes: List[str],
        trade_date: str,
        max_workers: int = 4
    ) -> Dict[str, Dict]:
        """
        批量计算多只股票的持有时间矩阵
        Args:
            stock_codes: 股票代码列表
            trade_date: 交易日期
            max_workers: 最大并行工作数
        Returns:
            Dict: 股票代码到结果的映射
        """
        import concurrent.futures
        from tqdm import tqdm
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.calculate_holding_matrix_daily, code, trade_date): code
                for code in stock_codes
            }
            # 处理结果
            for future in tqdm(concurrent.futures.as_completed(future_to_code), 
                              total=len(stock_codes), desc="计算持有矩阵"):
                code = future_to_code[future]
                try:
                    result = future.result()
                    results[code] = result
                except Exception as e:
                    logger.error(f"批量计算失败 {code}: {e}")
                    results[code] = self._get_default_result(code, trade_date)
        return results
    
    async def _fetch_required_data(
        self,
        stock_code: str,
        trade_date: str,
        lookback_days: int
    ) -> Dict[str, any]:
        """
        获取计算所需的所有数据（异步版本）
        """
        try:
            from django.db.models import Q
            import pytz
            from stock_models.time_trade import StockDailyBasic
            # 转换日期
            trade_date_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
            start_date = trade_date_dt - timedelta(days=lookback_days)
            data = {}
            # 1. 获取1分钟数据
            minute_model = self.get_minute_data_model(stock_code, '1')
            if minute_model:
                minute_qs = minute_model.objects.filter(
                    stock__stock_code=stock_code,
                    trade_time__date=trade_date_dt
                ).order_by('trade_time')
                minute_records = await sync_to_async(list)(minute_qs.values('trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount'))
                data['minute_data'] = pd.DataFrame(minute_records) if minute_records else None
            # 2. 获取逐笔数据（如果可用）
            if self.use_tick_data:
                tick_model = self.get_tick_data_model(stock_code)
                if tick_model:
                    tick_qs = tick_model.objects.filter(
                        stock__stock_code=stock_code,
                        trade_time__date=trade_date_dt
                    ).order_by('trade_time')[:50000]
                    tick_records = await sync_to_async(list)(tick_qs.values('trade_time', 'price', 'volume', 'type'))
                    data['tick_data'] = pd.DataFrame(tick_records) if tick_records else None
            # 3. 获取筹码分布数据
            chips_model = self.get_chips_model(stock_code)
            chip_dist_current_qs = chips_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time=trade_date_dt
            ).values('price', 'percent')
            chip_dist_historical_qs = chips_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_date,
                trade_time__lt=trade_date_dt
            ).order_by('trade_time')
            chip_dist_current = await sync_to_async(list)(chip_dist_current_qs)
            chip_dist_historical = await sync_to_async(list)(chip_dist_historical_qs.values('trade_time', 'price', 'percent'))
            data['chip_dist_current'] = pd.DataFrame(list(chip_dist_current))
            data['chip_dists'] = chip_dist_historical
            # 4. 获取日线数据（用于换手率）
            daily_model = self.get_daily_data_model(stock_code)
            daily_qs = daily_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_date,
                trade_time__lte=trade_date_dt
            ).order_by('trade_time').values('trade_time', 'vol', 'amount')
            daily_data = await sync_to_async(list)(daily_qs)
            data['daily_data'] = pd.DataFrame(list(daily_data))
            # 5. 获取自由流通股本
            basic_qs = StockDailyBasic.objects.filter(
                stock__stock_code=stock_code,
                trade_time=trade_date_dt
            )
            basic_data = await sync_to_async(basic_qs.first)()
            data['float_shares'] = float(basic_data.free_share) * 10000 if basic_data and basic_data.free_share else 0
            # 6. 计算价格范围
            if not data['chip_dist_current'].empty:
                data['price_range'] = (
                    data['chip_dist_current']['price'].min(),
                    data['chip_dist_current']['price'].max()
                )
            # 7. 计算日换手率
            if not data['daily_data'].empty and data['float_shares'] > 0:
                data['daily_turnover'] = data['daily_data']['vol'] * 100 / data['float_shares']
            return data
        except Exception as e:
            logger.error(f"获取数据失败 {stock_code}: {e}", exc_info=True)
            return {}

    def _build_price_grid_and_chip_matrix(
        self,
        chip_dists: List[Dict],
        price_range: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        建立价格网格和筹码分布矩阵
        """
        try:
            min_price, max_price = price_range
            price_grid = np.linspace(min_price, max_price, self.price_grid_size)
            # 将历史筹码分布插值到价格网格
            chip_matrix = np.zeros((len(chip_dists), len(price_grid)))
            for i, chip_dist in enumerate(chip_dists):
                df = pd.DataFrame(chip_dist)
                if not df.empty:
                    # 线性插值
                    f = interp1d(df['price'], df['percent'], 
                                bounds_error=False, fill_value=0)
                    chip_matrix[i, :] = f(price_grid)
            # 归一化
            row_sums = chip_matrix.sum(axis=1, keepdims=True)
            chip_matrix = np.divide(chip_matrix, row_sums, 
                                   out=np.zeros_like(chip_matrix), 
                                   where=row_sums != 0)
            return price_grid, chip_matrix
        except Exception as e:
            logger.error(f"建立价格网格失败: {e}")
            return np.array([]), np.array([])
    
    def _calculate_minute_volume_distribution(
        self,
        minute_data: pd.DataFrame,
        price_grid: np.ndarray
    ) -> np.ndarray:
        """
        使用1分钟数据计算成交量分布
        """
        try:
            if minute_data is None or minute_data.empty:
                return np.zeros(len(price_grid))
            volume_dist = np.zeros(len(price_grid))
            for _, row in minute_data.iterrows():
                minute_volume = row['vol'] * 100  # 转换为股
                low_price = row['low']
                high_price = row['high']
                # 找到价格区间对应的网格索引
                start_idx = np.searchsorted(price_grid, low_price, side='left')
                end_idx = np.searchsorted(price_grid, high_price, side='right')
                if start_idx < end_idx:
                    # 在价格区间内均匀分布成交量
                    interval_volume = minute_volume / (end_idx - start_idx)
                    volume_dist[start_idx:end_idx] += interval_volume
                else:
                    # 价格区间很窄，直接分配到最近的网格
                    mid_price = (low_price + high_price) / 2
                    nearest_idx = np.argmin(np.abs(price_grid - mid_price))
                    volume_dist[nearest_idx] += minute_volume
            return volume_dist
        except Exception as e:
            logger.error(f"计算分钟成交量分布失败: {e}")
            return np.zeros(len(price_grid))
    
    def _enhance_with_tick_data(
        self,
        base_dist: np.ndarray,
        tick_data: pd.DataFrame,
        price_grid: np.ndarray
    ) -> np.ndarray:
        """
        使用逐笔数据增强成交量分布
        """
        try:
            if tick_data is None or tick_data.empty:
                return base_dist
            # 基于逐笔数据计算更精确的分布
            tick_prices = tick_data['price'].values
            tick_volumes = tick_data['volume'].values * 100  # 转换为股
            # 使用核密度估计
            if len(tick_prices) > 10:
                # 训练KDE模型
                kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
                # 按成交量加权
                sample_weights = tick_volumes / tick_volumes.sum()
                # 需要重采样以应用权重
                resampled_prices = []
                for price, weight in zip(tick_prices, sample_weights):
                    # 按权重比例重复价格
                    n_repeats = max(1, int(weight * 1000))
                    resampled_prices.extend([price] * n_repeats)
                if resampled_prices:
                    kde.fit(np.array(resampled_prices).reshape(-1, 1))
                    # 计算每个网格点的密度
                    log_density = kde.score_samples(price_grid.reshape(-1, 1))
                    tick_density = np.exp(log_density)
                    # 归一化并乘以总成交量
                    total_volume = tick_volumes.sum()
                    tick_dist = tick_density / tick_density.sum() * total_volume
                    # 合并基础分布和逐笔分布（加权平均）
                    alpha = 0.7  # 逐笔数据权重
                    enhanced_dist = alpha * tick_dist + (1 - alpha) * base_dist
                    return enhanced_dist
            return base_dist
        except Exception as e:
            logger.error(f"逐笔数据增强失败: {e}")
            return base_dist
    
    def _calculate_turnover_matrix(
        self,
        volume_dist: np.ndarray,
        chip_matrix: np.ndarray,
        float_shares: float
    ) -> np.ndarray:
        """
        计算换手率矩阵
        """
        try:
            if float_shares <= 0 or len(volume_dist) == 0:
                return np.zeros_like(chip_matrix)
            # 计算每日换手率
            daily_turnover_rate = volume_dist.sum() / float_shares
            # 计算各价格区间的相对换手率
            chip_dist_current = chip_matrix[-1, :] if chip_matrix.shape[0] > 0 else np.ones(len(volume_dist))
            chip_dist_current = np.maximum(chip_dist_current, 1e-6)  # 避免除零
            # 各价格区间换手率 = 成交量分布 / 筹码分布
            price_turnover = volume_dist / (chip_dist_current * float_shares)
            price_turnover = np.nan_to_num(price_turnover, nan=0, posinf=0, neginf=0)
            # 创建换手率矩阵（与筹码矩阵同形状）
            turnover_matrix = np.tile(price_turnover, (chip_matrix.shape[0], 1))
            # 应用时间衰减（越久远的数据影响越小）
            time_weights = np.exp(-np.arange(chip_matrix.shape[0]) / 30)  # 30日衰减
            turnover_matrix = turnover_matrix * time_weights[:, np.newaxis]
            return turnover_matrix
        except Exception as e:
            logger.error(f"计算换手率矩阵失败: {e}")
            return np.zeros_like(chip_matrix)
    
    def _optimize_turnover_parameters(
        self,
        turnover_matrix: np.ndarray,
        chip_matrix: np.ndarray,
        daily_turnover_series: pd.Series
    ) -> Dict[str, float]:
        """
        优化换手概率参数
        """
        try:
            if turnover_matrix.size == 0 or chip_matrix.size == 0:
                return {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
            # 简化优化：使用最小二乘法
            def objective(params):
                alpha, beta = params
                # 预测换手率 = alpha * 成交量分布 + beta * 筹码集中度
                chip_concentration = np.std(chip_matrix, axis=1)
                predicted = alpha * turnover_matrix.mean(axis=1) + beta * chip_concentration
                # 与实际日换手率比较
                if daily_turnover_series is not None and len(daily_turnover_series) == len(predicted):
                    actual = daily_turnover_series.values[-len(predicted):]
                    loss = np.mean((predicted - actual) ** 2)
                else:
                    loss = 1.0
                return loss
            # 初始猜测和边界
            initial_guess = [0.5, 0.3]
            bounds = [(0, 1), (0, 1)]
            # 优化
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            return {
                'alpha': result.x[0],
                'beta': result.x[1],
                'gamma': 0.2,  # 固定时间衰减参数
                'success': result.success,
                'loss': result.fun
            }
        except Exception as e:
            logger.error(f"优化参数失败: {e}")
            return {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
    
    def _calculate_holding_matrix(
        self,
        chip_matrix: np.ndarray,
        turnover_matrix: np.ndarray,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        计算持有时间矩阵
        """
        try:
            n_days = chip_matrix.shape[0]
            n_prices = chip_matrix.shape[1]
            # 初始化持有时间矩阵 [价格区间 × 持有天数]
            holding_matrix = np.zeros((n_prices, self.max_holding_days))
            # 初始假设：所有筹码持有0天
            holding_matrix[:, 0] = chip_matrix[-1, :] if n_days > 0 else np.ones(n_prices) / n_prices
            if n_days < 2:
                return holding_matrix
            # 模拟历史换手过程
            for day in range(1, min(n_days, self.max_holding_days)):
                # 获取当天的换手率
                turnover_rate = turnover_matrix[day, :]
                # 应用参数调整
                adjusted_turnover = params['alpha'] * turnover_rate + params['beta']
                adjusted_turnover = np.clip(adjusted_turnover, 0, 0.99)  # 限制在0-0.99
                # 更新持有时间矩阵
                for price_idx in range(n_prices):
                    # 未换手的筹码持有时间增加
                    for holding_day in range(self.max_holding_days - 1, 0, -1):
                        holding_matrix[price_idx, holding_day] = (
                            holding_matrix[price_idx, holding_day - 1] * 
                            (1 - adjusted_turnover[price_idx])
                        )
                    # 新换手的筹码持有时间为0
                    total_turnover = np.sum(
                        holding_matrix[:, 0] * adjusted_turnover
                    )
                    # 按成交量比例重新分配换手筹码
                    if total_turnover > 0:
                        turnover_dist = turnover_matrix[day, :] / np.sum(turnover_matrix[day, :])
                        holding_matrix[price_idx, 0] = total_turnover * turnover_dist[price_idx]
                    else:
                        holding_matrix[price_idx, 0] = 0
            # 归一化
            row_sums = holding_matrix.sum(axis=1, keepdims=True)
            holding_matrix = np.divide(holding_matrix, row_sums, 
                                      out=np.zeros_like(holding_matrix), 
                                      where=row_sums != 0)
            return holding_matrix
        except Exception as e:
            logger.error(f"计算持有时间矩阵失败: {e}")
            return np.zeros((chip_matrix.shape[1], self.max_holding_days))

    def _calculate_holding_factors(
        self,
        holding_matrix: np.ndarray,
        chip_matrix: np.ndarray,
        data_dict: Dict[str, any]
    ) -> Dict[str, float]:
        """计算持有时间相关因子"""
        try:
            factors = {}
            if holding_matrix.size == 0:
                return self._get_default_factors()
            # 1. 短线筹码比例（<5日）
            short_term_mask = np.arange(self.max_holding_days) < 5
            short_term_ratio = np.sum(holding_matrix[:, short_term_mask]) / np.sum(holding_matrix)
            factors['short_term_ratio'] = float(short_term_ratio)
            # 2. 中线筹码比例（5-60日）
            mid_term_mask = (np.arange(self.max_holding_days) >= 5) & (np.arange(self.max_holding_days) < 60)
            mid_term_ratio = np.sum(holding_matrix[:, mid_term_mask]) / np.sum(holding_matrix)
            factors['mid_term_ratio'] = float(mid_term_ratio)
            # 3. 长线筹码比例（>60日）
            long_term_mask = np.arange(self.max_holding_days) >= 60
            long_term_ratio = np.sum(holding_matrix[:, long_term_mask]) / np.sum(holding_matrix)
            factors['long_term_ratio'] = float(long_term_ratio)
            # 4. 平均持有时间
            holding_days = np.arange(self.max_holding_days)
            weighted_days = np.sum(holding_matrix * holding_days[np.newaxis, :], axis=1)
            total_weights = np.sum(holding_matrix, axis=1)
            avg_holding_days = np.sum(weighted_days) / np.sum(total_weights) if np.sum(total_weights) > 0 else 0
            factors['avg_holding_days'] = float(avg_holding_days)
            # 5. 筹码集中度调整的持有时间
            if chip_matrix.shape[0] > 0:
                chip_concentration = np.std(chip_matrix[-1, :]) / np.mean(chip_matrix[-1, :])
                factors['concentration_adjusted_holding'] = float(avg_holding_days * (1 + chip_concentration))
            # 6. 高位筹码沉淀比例（90%分位以上）
            if 'price_grid' in data_dict and 'chip_dist_current' in data_dict:
                price_grid = data_dict['price_grid']
                chip_dist_current = data_dict['chip_dist_current']
                if len(price_grid) > 0 and not chip_dist_current.empty:
                    # 找到90%分位价格
                    sorted_prices = np.sort(price_grid)
                    idx_90 = int(len(sorted_prices) * 0.9)
                    price_90 = sorted_prices[idx_90]
                    # 计算价格高于90%分位的筹码比例
                    high_mask = chip_dist_current['price'] >= price_90
                    if high_mask.any():
                        high_chip_ratio = chip_dist_current.loc[high_mask, 'percent'].sum() / chip_dist_current['percent'].sum()
                        factors['high_position_lock_ratio_90'] = float(high_chip_ratio)
                    else:
                        factors['high_position_lock_ratio_90'] = 0.0
            # 7. 主力成本区间锁定比例（50分位±10%）
            if 'chip_dist_current' in data_dict:
                chip_dist_current = data_dict['chip_dist_current']
                if not chip_dist_current.empty:
                    # 获取50分位成本
                    sorted_chips = chip_dist_current.sort_values('price')
                    cumsum = sorted_chips['percent'].cumsum()
                    idx_50 = (cumsum >= 0.5).idxmax() if (cumsum >= 0.5).any() else -1
                    if idx_50 >= 0:
                        cost_50pct = sorted_chips.loc[idx_50, 'price']
                        # 计算50分位±10%区间筹码
                        lower_bound = cost_50pct * 0.9
                        upper_bound = cost_50pct * 1.1
                        main_mask = (chip_dist_current['price'] >= lower_bound) & (chip_dist_current['price'] <= upper_bound)
                        main_range_ratio = chip_dist_current.loc[main_mask, 'percent'].sum() / chip_dist_current['percent'].sum()
                        factors['main_cost_range_ratio'] = float(main_range_ratio)
                    else:
                        factors['main_cost_range_ratio'] = 0.0
            # 8. 换手率调整因子
            if 'daily_turnover' in data_dict and isinstance(data_dict['daily_turnover'], pd.Series):
                recent_turnover = data_dict['daily_turnover'].iloc[-5:].mean() if len(data_dict['daily_turnover']) >= 5 else 0
                factors['turnover_adjustment'] = float(recent_turnover)
                # 高换手率下短线筹码比例应该更高
                if recent_turnover > 0.05:  # 5%以上换手
                    factors['short_term_ratio'] = min(factors['short_term_ratio'] * 1.5, 0.95)
            # 9. 价格位置调整
            if 'price_range' in data_dict and chip_matrix.shape[0] > 0:
                price_min, price_max = data_dict['price_range']
                current_chip = chip_matrix[-1, :] if chip_matrix.shape[0] > 0 else np.zeros(chip_matrix.shape[1])
                # 计算筹码价格重心
                price_center = np.sum(current_chip * np.linspace(price_min, price_max, len(current_chip))) / np.sum(current_chip)
                price_position = (price_center - price_min) / (price_max - price_min) if price_max > price_min else 0.5
                factors['price_position'] = float(price_position)
                # 价格高位通常长线筹码较少
                if price_position > 0.7:
                    factors['long_term_ratio'] = max(factors['long_term_ratio'] * 0.8, 0.1)
            return factors
        except Exception as e:
            logger.error(f"计算持有因子失败: {e}")
            return self._get_default_factors()

    def _validate_results(
        self,
        holding_matrix: np.ndarray,
        factors: Dict[str, float],
        data_dict: Dict[str, any]
    ) -> Dict[str, any]:
        """
        验证计算结果合理性
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'checks_passed': 0,
            'total_checks': 5
        }
        try:
            # 检查1：持有时间矩阵总和应为1
            total_sum = np.sum(holding_matrix)
            if abs(total_sum - 1.0) > 0.01:
                validation['warnings'].append(f"持有矩阵总和异常: {total_sum:.4f}")
                validation['is_valid'] = False
            else:
                validation['checks_passed'] += 1
            # 检查2：短线+中线+长线比例应接近1
            sum_ratios = factors.get('short_term_ratio', 0) + \
                        factors.get('mid_term_ratio', 0) + \
                        factors.get('long_term_ratio', 0)
            if abs(sum_ratios - 1.0) > 0.05:
                validation['warnings'].append(f"筹码比例总和异常: {sum_ratios:.4f}")
                validation['is_valid'] = False
            else:
                validation['checks_passed'] += 1
            # 检查3：平均持有时间应在合理范围内
            avg_days = factors.get('avg_holding_days', 0)
            if avg_days < 1 or avg_days > 500:
                validation['warnings'].append(f"平均持有时间异常: {avg_days:.1f}天")
                validation['is_valid'] = False
            else:
                validation['checks_passed'] += 1
            # 检查4：与换手率一致性
            if 'daily_turnover' in data_dict and isinstance(data_dict['daily_turnover'], pd.Series):
                recent_turnover = data_dict['daily_turnover'].iloc[-5:].mean() if len(data_dict['daily_turnover']) >= 5 else 0
                expected_short_term = min(recent_turnover * 5, 0.95)  # 预期短线比例
                actual_short_term = factors.get('short_term_ratio', 0)
                if abs(actual_short_term - expected_short_term) > 0.3:
                    validation['warnings'].append(
                        f"短线比例与换手率不一致: 实际{actual_short_term:.2%}, 预期{expected_short_term:.2%}"
                    )
                else:
                    validation['checks_passed'] += 1
            # 检查5：价格区间合理性
            if 'price_range' in data_dict:
                price_min, price_max = data_dict['price_range']
                if price_max <= price_min or price_max <= 0:
                    validation['warnings'].append(f"价格区间异常: {price_min}-{price_max}")
                    validation['is_valid'] = False
                else:
                    validation['checks_passed'] += 1
            validation['score'] = validation['checks_passed'] / validation['total_checks']
            return validation
        except Exception as e:
            logger.error(f"验证结果失败: {e}")
            validation['warnings'].append(f"验证过程异常: {str(e)}")
            validation['is_valid'] = False
            return validation
    
    def _get_default_result(self, stock_code: str = "", trade_date: str = "") -> Dict[str, any]:
        """获取默认结果（计算失败时返回）"""
        return {
            'stock_code': stock_code,
            'trade_date': trade_date,
            'holding_matrix': np.array([]),
            'price_grid': np.array([]),
            'factors': self._get_default_factors(),
            'validation': {
                'is_valid': False,
                'warnings': ['计算失败'],
                'checks_passed': 0,
                'total_checks': 5,
                'score': 0
            },
            'calc_status': 'failed',
            'calc_time': datetime.now()
        }

    def _get_default_factors(self) -> Dict[str, float]:
        """获取默认因子值"""
        return {
            'short_term_ratio': 0.2,
            'mid_term_ratio': 0.3,
            'long_term_ratio': 0.5,
            'avg_holding_days': 100.0,
            'concentration_adjusted_holding': 100.0,
            'high_position_lock_ratio_90': 0.0,
            'main_cost_range_ratio': 0.5,
            'turnover_adjustment': 0.02,
            'price_position': 0.5
        }
    
    def save_holding_matrix_to_db(
        self,
        stock_code: str,
        trade_date: str,
        result: Dict[str, any]
    ) -> bool:
        """
        将持有时间矩阵保存到数据库
        需要先创建对应的数据库模型
        """
        try:
            # 压缩矩阵数据（可以保存为JSON或二进制）
            import json
            import base64
            import pickle
            ChipHoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
            # 将矩阵转换为可存储格式
            if 'holding_matrix' in result and result['holding_matrix'].size > 0:
                # 方法1：保存为JSON（适合小矩阵）
                matrix_data = {
                    'matrix': result['holding_matrix'].tolist(),
                    'price_grid': result['price_grid'].tolist() if 'price_grid' in result else []
                }
                # 方法2：保存为压缩二进制（推荐）
                matrix_bytes = pickle.dumps(result['holding_matrix'])
                compressed_data = base64.b64encode(matrix_bytes).decode('utf-8')
                # 创建或更新记录
                holding_record, created = ChipHoldingMatrixModel.objects.update_or_create(
                    stock__stock_code=stock_code,
                    trade_time=trade_date,
                    defaults={
                        'short_term_ratio': result['factors'].get('short_term_ratio', 0),
                        'long_term_ratio': result['factors'].get('long_term_ratio', 0),
                        'avg_holding_days': result['factors'].get('avg_holding_days', 0),
                        'matrix_data': json.dumps(matrix_data),  # 或保存压缩数据
                        'calc_status': result.get('calc_status', 'failed'),
                        'validation_score': result.get('validation', {}).get('score', 0)
                    }
                )
                return True
        except Exception as e:
            logger.error(f"保存持有矩阵失败 {stock_code} {trade_date}: {e}")
            return False
        return False


# 快速计算函数（保持向后兼容）
def calculate_holding_factors_daily(stock_code: str, trade_date: str) -> Dict[str, float]:
    """
    快速计算持有时间因子（简化接口）
    """
    service = ChipHoldingService(use_tick_data=True)
    result = service.calculate_holding_matrix_daily(stock_code, trade_date)
    return result.get('factors', {})