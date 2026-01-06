# services\advanced_structural_metrics_service.py
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from scipy.stats import norm, linregress
import numba
from typing import Tuple
from asgiref.sync import sync_to_async
from stock_models.stock_basic import StockInfo
from stock_models.advanced_metrics import BaseAdvancedStructuralMetrics
from utils.model_helpers import (
    get_advanced_structural_metrics_model_by_code,
    get_daily_data_model_by_code,
    get_minute_data_model_by_code_and_timelevel,
)
import pandas_ta as ta
import logging

logger = logging.getLogger("services")

@numba.njit(cache=True)
def _numba_calculate_trend_metrics(price_arr: np.ndarray) -> Tuple[float, float]:
    """
    【Numba优化版】通过线性回归计算价格序列的趋势强度和趋势质量。
    """
    cleaned_arr = price_arr[~np.isnan(price_arr)]
    if len(cleaned_arr) < 2:
        return 0.0, 0.0
    
    if np.unique(cleaned_arr).size == 1: # 如果所有价格都相同
        return 0.0, 1.0
    y = cleaned_arr
    x = np.arange(len(y), dtype=np.float64)
    
    N = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = N * sum_x2 - sum_x * sum_x
    
    if denominator == 0:
        return 0.0, 0.0 # 避免除以零
    slope = (N * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / N
    
    predicted_y = slope * x + intercept
    residuals = y - predicted_y
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    return slope, r_squared

@numba.njit(cache=True)
def _numba_calculate_mean_reversion_speed(price_arr: np.ndarray) -> float:
    """
    【Numba优化版】估算价格序列的均值回归速度。
    """
    cleaned_arr = price_arr[~np.isnan(price_arr)]
    if len(cleaned_arr) < 2:
        return 0.0
    
    price_changes = np.diff(cleaned_arr)
    lagged_prices = cleaned_arr[:-1]
    
    if len(price_changes) < 2:
        return 0.0
    y = price_changes
    x = lagged_prices
    
    N = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = N * sum_x2 - sum_x * sum_x
    
    if denominator == 0:
        return 0.0
    slope = (N * sum_xy - sum_x * sum_y) / denominator
    
    return -slope

@numba.njit(cache=True)
def _numba_calculate_tpo_metrics(close_arr: np.ndarray, vol_arr: np.ndarray) -> Tuple[float, float, float]:
    """
    【Numba优化版】基于日内分钟数据计算市场轮廓（TPO/Market Profile）的核心指标。
    """
    if len(close_arr) == 0 or vol_arr.sum() == 0:
        return np.nan, np.nan, np.nan

    # 1. 构建成交量分布图 (简化为离散价格点)
    # Numba 不直接支持 groupby，手动实现
    unique_prices = np.unique(close_arr)
    volume_profile = np.zeros(len(unique_prices), dtype=np.float64)
    
    for i, price in enumerate(unique_prices):
        volume_profile[i] = np.sum(vol_arr[close_arr == price])
    # 2. 确定VPOC
    if len(volume_profile) == 0:
        return np.nan, np.nan, np.nan
    vpoc_idx = np.argmax(volume_profile)
    vpoc = unique_prices[vpoc_idx]
    
    # 3. 计算价值区 (VAH, VAL)
    total_volume = np.sum(volume_profile)
    value_area_target_volume = total_volume * 0.7
    
    value_area_prices = np.array([vpoc])
    current_volume_in_area = volume_profile[vpoc_idx]
    
    # 获取VPOC上下方的价格索引
    prices_below_vpoc_indices = np.where(unique_prices < vpoc)[0]
    prices_above_vpoc_indices = np.where(unique_prices > vpoc)[0]
    
    # 双指针，从紧邻VPOC的价格开始
    below_ptr = len(prices_below_vpoc_indices) - 1
    above_ptr = 0
    
    while current_volume_in_area < value_area_target_volume:
        vol_below = 0.0
        if below_ptr >= 0:
            price_below_idx = prices_below_vpoc_indices[below_ptr]
            vol_below = volume_profile[price_below_idx]
        vol_above = 0.0
        if above_ptr < len(prices_above_vpoc_indices):
            price_above_idx = prices_above_vpoc_indices[above_ptr]
            vol_above = volume_profile[price_above_idx]
        if vol_below == 0 and vol_above == 0:
            break # 没有更多价格可以添加
        if vol_above > vol_below:
            value_area_prices = np.append(value_area_prices, unique_prices[prices_above_vpoc_indices[above_ptr]])
            current_volume_in_area += vol_above
            above_ptr += 1
        else:
            value_area_prices = np.append(value_area_prices, unique_prices[prices_below_vpoc_indices[below_ptr]])
            current_volume_in_area += vol_below
            below_ptr -= 1
    vah = np.max(value_area_prices)
    val = np.min(value_area_prices)
    
    return vpoc, vah, val

@numba.njit(cache=True)
def _numba_calculate_continuous_data_metrics(
    prev_close: float, today_open: float, today_close_arr: np.ndarray
) -> Tuple[float, float]:
    """
    【Numba优化版】计算跳空回报率和缺口后动量。
    """
    gap_return = np.nan
    if prev_close > 0:
        gap_return = (today_open / prev_close) - 1
    post_gap_momentum_30min = np.nan
    if today_open > 0 and len(today_close_arr) > 0:
        close_after_30_min = today_close_arr[-1]
        post_gap_momentum_30min = (close_after_30_min / today_open) - 1
    return gap_return, post_gap_momentum_30min

@numba.njit(cache=True)
def _numba_calculate_continuous_data_metrics(
    prev_close: float, today_open: float, today_close_arr: np.ndarray
) -> Tuple[float, float]:
    """
    【Numba优化版】计算跳空回报率和缺口后动量。
    """
    gap_return = np.nan
    if prev_close > 0:
        gap_return = (today_open / prev_close) - 1
    post_gap_momentum_30min = np.nan
    if today_open > 0 and len(today_close_arr) > 0:
        close_after_30_min = today_close_arr[-1]
        post_gap_momentum_30min = (close_after_30_min / today_open) - 1
    return gap_return, post_gap_momentum_30min

@numba.njit(cache=True)
def _numba_calculate_atr_interaction_metrics(
    day_high: float, day_low: float, atr_5: float, atr_14: float, atr_50: float,
    day_close: float, vwap: float
) -> Tuple[float, float, float]:
    """
    【Numba优化版】计算日内价格行为与日线ATR之间的交互指标。
    """
    intraday_range_vs_atr14 = np.nan
    close_vwap_deviation_normalized = np.nan
    volatility_expansion_ratio = np.nan

    # 1. 日内振幅 vs ATR14
    intraday_range = day_high - day_low
    if atr_14 > 0:
        intraday_range_vs_atr14 = intraday_range / atr_14

    # 2. 收盘价与VWAP的偏离度 (ATR标准化)
    if atr_14 > 0:
        close_vwap_deviation_normalized = (day_close - vwap) / atr_14

    # 3. 短期与长期波动率扩张比
    if atr_50 > 0:
        volatility_expansion_ratio = atr_5 / atr_50
    return intraday_range_vs_atr14, close_vwap_deviation_normalized, volatility_expansion_ratio

@numba.njit(cache=True)
def _numba_calculate_prev_day_interaction_metrics(
    today_vpoc: float, today_vah: float, today_val: float,
    day_close: float, day_open: float,
    prev_vpoc: float, prev_vah: float, prev_val: float, prev_atr: float
) -> Tuple[float, float, float, float]:
    """
    【Numba优化版】计算当日市场行为与前一日关键结构位（如价值区）的交互指标。
    """
    value_area_migration = np.nan
    value_area_overlap_pct = np.nan
    closing_acceptance_type = np.nan
    opening_position_vs_prev_va = np.nan

    # 2. 计算价值区迁移 (Value Area Migration)
    if not np.isnan(today_vpoc) and not np.isnan(prev_vpoc) and not np.isnan(prev_atr) and prev_atr > 0:
        value_area_migration = (today_vpoc - prev_vpoc) / prev_atr

    # 3. 计算价值区重叠度 (Value Area Overlap)
    if not np.isnan(today_vah) and not np.isnan(today_val) and not np.isnan(prev_vah) and not np.isnan(prev_val):
        today_va_height = today_vah - today_val
        if today_va_height > 0:
            overlap_width = max(0.0, min(today_vah, prev_vah) - max(today_val, prev_val))
            value_area_overlap_pct = (overlap_width / today_va_height) * 100

    # 4. 计算收盘接受度类型 (Closing Acceptance Type)
    if not np.isnan(day_close) and not np.isnan(today_vpoc) and not np.isnan(today_vah) and not np.isnan(today_val):
        if day_close > today_vah:
            closing_acceptance_type = 2.0
        elif day_close > today_vpoc:
            closing_acceptance_type = 1.0
        elif day_close < today_val:
            closing_acceptance_type = -2.0
        elif day_close < today_vpoc:
            closing_acceptance_type = -1.0
        else:
            closing_acceptance_type = 0.0

    # 5. 计算开盘位置 vs 前日价值区
    if not np.isnan(day_open) and not np.isnan(prev_vah) and not np.isnan(prev_val):
        if day_open > prev_vah:
            opening_position_vs_prev_va = 2.0
        elif day_open > prev_val:
            opening_position_vs_prev_va = 1.0
        else:
            opening_position_vs_prev_va = -1.0
    return value_area_migration, value_area_overlap_pct, closing_acceptance_type, opening_position_vs_prev_va

@numba.njit(cache=True)
def _numba_calculate_price_thrust_divergence(
    am_high: float, pm_high: float, am_low: float, pm_low: float,
    am_thrust: float, pm_thrust: float
) -> float:
    """
    【Numba优化版】计算价格推力背离。
    """
    top_divergence_score = 0.0
    bottom_divergence_score = 0.0

    # 顶背离计算 (价格新高, 动能减弱)
    if pm_high > am_high and am_thrust > 0 and pm_thrust < am_thrust:
        price_change_pct = (pm_high / am_high - 1)
        thrust_change_pct = (pm_thrust - am_thrust) / abs(am_thrust) if am_thrust != 0 else -1.0
        if thrust_change_pct < 0:
            top_divergence_score = price_change_pct / abs(thrust_change_pct) * -1

    # 底背离计算 (价格新低, 动能增强)
    if pm_low < am_low and am_thrust < 0 and pm_thrust > am_thrust:
        price_change_pct = (am_low / pm_low - 1)
        thrust_change_pct = (pm_thrust - am_thrust) / abs(am_thrust) if am_thrust != 0 else 1.0
        if thrust_change_pct > 0:
            bottom_divergence_score = price_change_pct / abs(thrust_change_pct)
    return top_divergence_score + bottom_divergence_score

@numba.njit(cache=True)
def _numba_calculate_gini(array: np.ndarray) -> float:
    """
    【Numba优化版】计算基尼系数。
    """
    if len(array) < 2 or np.sum(array) == 0:
        return 0.0
    
    sorted_array = np.sort(array)
    n = len(array)
    cum_array = np.cumsum(sorted_array)
    
    # 避免除以零
    if cum_array[-1] == 0:
        return 0.0

    return (n + 1 - 2 * np.sum(cum_array) / cum_array[-1]) / n

@numba.njit(cache=True)
def _numba_calculate_ofi_static_dynamic(
    buy_price1_arr: np.ndarray, buy_volume1_arr: np.ndarray,
    sell_price1_arr: np.ndarray, sell_volume1_arr: np.ndarray,
    prev_buy_price1_arr: np.ndarray, prev_buy_volume1_arr: np.ndarray,
    prev_sell_price1_arr: np.ndarray, prev_sell_volume1_arr: np.ndarray
) -> np.ndarray:
    """
    【Numba优化版】计算订单流失衡 (OFI) 的静态和动态部分。
    """
    n = len(buy_price1_arr)
    ofi_series = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0: # 第一个元素没有前一个状态，OFI为0
            continue
        delta_buy_price = buy_price1_arr[i] - prev_buy_price1_arr[i]
        delta_sell_price = sell_price1_arr[i] - prev_sell_price1_arr[i]
        ofi_static = 0.0
        if delta_buy_price == 0 and delta_sell_price == 0:
            ofi_static = buy_volume1_arr[i] - prev_buy_volume1_arr[i] # 假设买一价和卖一价不变时，OFI由买一量变化决定
        ofi_dynamic = 0.0
        if delta_buy_price > 0:
            ofi_dynamic += prev_buy_volume1_arr[i] # 买一价上涨，前一刻的买一量被吃掉
        elif delta_buy_price < 0:
            ofi_dynamic -= buy_volume1_arr[i] # 买一价下跌，当前买一量是新的
        if delta_sell_price > 0:
            ofi_dynamic += sell_volume1_arr[i] # 卖一价上涨，当前卖一量是新的
        elif delta_sell_price < 0:
            ofi_dynamic -= prev_sell_volume1_arr[i] # 卖一价下跌，前一刻的卖一量被吃掉
        ofi_series[i] = ofi_static + ofi_dynamic
    return ofi_series

@numba.njit(cache=True)
def _numba_calculate_vpin_buckets(
    cum_vol_arr: np.ndarray, volume_arr: np.ndarray,
    buy_vol_arr: np.ndarray, sell_vol_arr: np.ndarray,
    vpin_bucket_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    【Numba优化版】计算VPIN桶内的买卖失衡。
    返回每个桶的失衡值和桶的索引。
    """
    # 明确定义空数组，以帮助Numba进行类型推断
    empty_float_array = np.empty(0, dtype=np.float64)
    empty_int_array = np.empty(0, dtype=np.int64)

    if vpin_bucket_size <= 0:
        return empty_float_array, empty_int_array
    
    n = len(cum_vol_arr)
    # 防御性检查：如果传入的累计成交量数组为空，直接返回空数组
    if n == 0:
        return empty_float_array, empty_int_array

    # 预估最大桶数，避免动态列表增长开销
    max_buckets = int(cum_vol_arr[-1] / vpin_bucket_size) + 2
    
    # 防御性检查：如果计算出的最大桶数无效，直接返回空数组
    if max_buckets <= 0:
        return empty_float_array, empty_int_array

    bucket_imbalance = np.zeros(max_buckets, dtype=np.float64)
    bucket_buy_vol = np.zeros(max_buckets, dtype=np.float64)
    bucket_sell_vol = np.zeros(max_buckets, dtype=np.float64)
    current_bucket_idx = 0

    for i in range(n):
        bucket_idx = int(cum_vol_arr[i] / vpin_bucket_size)
        # 确保桶索引在范围内
        if bucket_idx >= max_buckets:
            # 如果超出预估范围，跳过此元素，避免索引越界
            continue 
        bucket_buy_vol[bucket_idx] += buy_vol_arr[i]
        bucket_sell_vol[bucket_idx] += sell_vol_arr[i]
        current_bucket_idx = max(current_bucket_idx, bucket_idx)
    # 截取实际使用的桶
    actual_buckets = current_bucket_idx + 1
    # 防御性检查：如果实际桶数为0或负数，返回空数组
    if actual_buckets <= 0:
        return empty_float_array, empty_int_array

    imbalance_values = bucket_buy_vol[:actual_buckets] - bucket_sell_vol[:actual_buckets]
    bucket_indices = np.arange(actual_buckets)
    return imbalance_values, bucket_indices

@numba.njit(cache=True)
def _numba_calculate_active_volume_price_efficiency(
    price_arr: np.ndarray, volume_arr: np.ndarray, price_change_arr: np.ndarray
) -> float:
    """
    【Numba优化版】计算累计推力与累计价格位移的相关性。
    """
    n = len(price_arr)
    if n < 2:
        return np.nan
    cum_thrust = np.zeros(n, dtype=np.float64)
    cum_price_change = np.zeros(n, dtype=np.float64)
    first_price = price_arr[0]
    for i in range(n):
        # 计算每笔tick的有效推力
        net_thrust_volume = volume_arr[i] * np.sign(price_change_arr[i])
        if i == 0:
            cum_thrust[i] = net_thrust_volume
            cum_price_change[i] = price_arr[i] - first_price
        else:
            cum_thrust[i] = cum_thrust[i-1] + net_thrust_volume
            cum_price_change[i] = price_arr[i] - first_price
    # 计算相关性
    # Numba 0.58+ 支持 np.corrcoef，但为了更广泛的兼容性，手动实现
    mean_cum_thrust = np.mean(cum_thrust)
    mean_cum_price_change = np.mean(cum_price_change)
    numerator = np.sum((cum_thrust - mean_cum_thrust) * (cum_price_change - mean_cum_price_change))
    denominator_thrust = np.sqrt(np.sum((cum_thrust - mean_cum_thrust)**2))
    denominator_price = np.sqrt(np.sum((cum_price_change - mean_cum_price_change)**2))
    denominator = denominator_thrust * denominator_price
    if denominator == 0:
        return 0.0 # 如果其中一个序列没有变化，相关性为0
    return numerator / denominator

@numba.njit(cache=True)
def _numba_calculate_liquidity_authenticity_score(
    buy_price1_arr: np.ndarray, buy_volume1_arr: np.ndarray,
    sell_price1_arr: np.ndarray, sell_volume1_arr: np.ndarray,
    tick_prices_arr: np.ndarray, tick_times_arr: np.ndarray,
    level5_times_arr: np.ndarray,
    buy_commitment_threshold: float, sell_commitment_threshold: float
) -> Tuple[int, int]:
    """
    【Numba优化版】计算流动性承诺-兑现分数。
    """
    fulfillments = 0
    defaults = 0
    n_level5 = len(buy_price1_arr)
    n_tick = len(tick_prices_arr)
    # 识别买方承诺（大额买单出现）
    for i in range(n_level5):
        if buy_volume1_arr[i] > buy_commitment_threshold:
            # 检查是否是新增的大单（与前一刻相比）
            if i > 0 and buy_volume1_arr[i] > buy_volume1_arr[i-1] * 2: # 简化判断为显著增加
                commit_price = buy_price1_arr[i]
                # 追踪此承诺未来20个快照
                future_snapshots_start_idx = i + 1
                future_snapshots_end_idx = min(n_level5, future_snapshots_start_idx + 20)
                pressure_found = False
                for j in range(future_snapshots_start_idx, future_snapshots_end_idx):
                    if sell_price1_arr[j] <= commit_price + 0.02: # 卖一价接近承诺价
                        pressure_found = True
                        
                        # 结局判断：是成交了还是撤单了？
                        if buy_volume1_arr[j] < buy_volume1_arr[i] * 0.5: # 大单在压力下消失
                            defaults += 1
                        else:
                            # 检查是否有真实成交 (简化为tick数据中是否有承诺价的成交)
                            # 找到level5快照时间对应的tick数据范围
                            level5_time_start = level5_times_arr[i]
                            level5_time_end = level5_times_arr[j]
                            
                            tick_start_idx = np.searchsorted(tick_times_arr, level5_time_start)
                            tick_end_idx = np.searchsorted(tick_times_arr, level5_time_end)
                            
                            found_trade = False
                            for k in range(tick_start_idx, tick_end_idx):
                                if tick_prices_arr[k] == commit_price:
                                    found_trade = True
                                    break
                            
                            if found_trade:
                                fulfillments += 1
                        break # 找到压力点后就停止追踪
                # 如果在未来20个快照内没有找到压力点，也算作违约（承诺未被测试）
                if not pressure_found:
                    defaults += 1
    # 识别卖方承诺（大额卖单出现）
    for i in range(n_level5):
        if sell_volume1_arr[i] > sell_commitment_threshold:
            if i > 0 and sell_volume1_arr[i] > sell_volume1_arr[i-1] * 2: # 简化判断为显著增加
                commit_price = sell_price1_arr[i]
                future_snapshots_start_idx = i + 1
                future_snapshots_end_idx = min(n_level5, future_snapshots_start_idx + 20)
                pressure_found = False
                for j in range(future_snapshots_start_idx, future_snapshots_end_idx):
                    if buy_price1_arr[j] >= commit_price - 0.02: # 买一价接近承诺价
                        pressure_found = True
                        
                        if sell_volume1_arr[j] < sell_volume1_arr[i] * 0.5:
                            defaults += 1
                        else:
                            level5_time_start = level5_times_arr[i]
                            level5_time_end = level5_times_arr[j]
                            
                            tick_start_idx = np.searchsorted(tick_times_arr, level5_time_start)
                            tick_end_idx = np.searchsorted(tick_times_arr, level5_time_end)
                            
                            found_trade = False
                            for k in range(tick_start_idx, tick_end_idx):
                                if tick_prices_arr[k] == commit_price:
                                    found_trade = True
                                    break
                            
                            if found_trade:
                                fulfillments += 1
                        break
                if not pressure_found:
                    defaults += 1
                    
    return fulfillments, defaults

@numba.njit(cache=True)
def _numba_calculate_vwap_reversion_corr(deviation_arr: np.ndarray) -> float:
    """
    【Numba优化版】计算VWAP均值回归相关性。
    """
    n = len(deviation_arr)
    if n < 2:
        return np.nan
    # 计算自相关系数 (lag=1)
    # corr(X_t, X_{t-1}) = cov(X_t, X_{t-1}) / (std(X_t) * std(X_{t-1}))
    # 移除NaN
    clean_deviation = deviation_arr[~np.isnan(deviation_arr)]
    if len(clean_deviation) < 2:
        return np.nan
    x_t = clean_deviation[1:]
    x_t_minus_1 = clean_deviation[:-1]
    if np.std(x_t) == 0 or np.std(x_t_minus_1) == 0:
        return 0.0 # 如果序列没有变化，自相关为0
    return np.corrcoef(x_t, x_t_minus_1)[0, 1]

class AdvancedStructuralMetricsService:
    """
    【V1.0 · 结构与行为锻造中心】
    - 核心职责: 封装所有高级结构与行为指标的加载、计算、融合与存储逻辑。
                利用分钟级数据，为日线级别锻造高保真的微观结构DNA指标。
    - 架构模式: 借鉴 AdvancedFundFlowMetricsService 的成功经验，实现业务逻辑与任务调度的完全解耦。
    """
    def __init__(self, debug_params: dict = None):
        """
        【V22.1 · 诊断驾驶舱】
        - 核心升级: 接收并存储 debug_params，为探针的精确触发提供支持。
        初始化服务，设定回溯期等基础参数
        """
        self.max_lookback_days = 300 # 为计算衍生指标所需的最大回溯天数
        self.debug_params = debug_params if debug_params is not None else {} # 接收并存储调试参数

    async def run_precomputation(self, stock_info: StockInfo, dates_to_process: pd.DatetimeIndex, daily_df_with_atr: pd.DataFrame, intraday_data_map: dict):
        """
        【V3.0 · 纯计算引擎版】高级结构与行为指标预计算总指挥
        - 核心重构: 剥离所有数据加载逻辑，直接接收由上游Celery任务预加载的 intraday_data_map。
        - 核心职责: 1. 按区块处理日期。 2. 调用指标锻造器。 3. 计算衍生指标并保存。
        """
        MetricsModel = get_advanced_structural_metrics_model_by_code(stock_info.stock_code)
        initial_history_end_date = dates_to_process.min()
        historical_metrics_df = await self._load_historical_metrics(MetricsModel, stock_info, initial_history_end_date)
        CHUNK_SIZE = 50
        all_new_core_metrics_df = pd.DataFrame()
        for i in range(0, len(dates_to_process), CHUNK_SIZE):
            chunk_dates = dates_to_process[i:i + CHUNK_SIZE]
            if chunk_dates.empty:
                continue
            chunk_intraday_map = {
                pd.to_datetime(d).date(): intraday_data_map[pd.to_datetime(d).date()]
                for d in chunk_dates if pd.to_datetime(d).date() in intraday_data_map
            }
            if not chunk_intraday_map:
                logger.warning(f"[{stock_info.stock_code}] 区块 {chunk_dates.min().date()} to {chunk_dates.max().date()} 无任何日内数据，跳过整个区块。")
                continue
            # 修改：传递 historical_metrics_df
            chunk_new_metrics_df = await self._forge_advanced_structural_metrics(chunk_intraday_map, stock_info.stock_code, daily_df_with_atr, historical_metrics_df)
            all_new_core_metrics_df = pd.concat([all_new_core_metrics_df, chunk_new_metrics_df])
        if all_new_core_metrics_df.empty:
            print(f"[{stock_info.stock_code}] [结构指标] 未能计算出任何新的核心指标，任务结束。")
            return 0
        full_sequence_for_derivatives = pd.concat([historical_metrics_df, all_new_core_metrics_df])
        full_sequence_for_derivatives.sort_index(inplace=True)
        final_metrics_df = self._calculate_derivatives(stock_info.stock_code, full_sequence_for_derivatives)
        chunk_to_save = final_metrics_df[final_metrics_df.index.isin(all_new_core_metrics_df.index)]
        total_processed_count = await self._prepare_and_save_data(stock_info, MetricsModel, chunk_to_save)
        return total_processed_count

    async def _initialize_context(self, stock_code: str, is_incremental: bool, start_date_str: str = None):
        """
        【V1.0】初始化计算上下文，确定股票实体、目标模型、计算模式和日期范围。
        """
        stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        MetricsModel = get_advanced_structural_metrics_model_by_code(stock_code)
        last_metric_date = None
        fetch_start_date = None
        if start_date_str:
            try:
                start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                is_incremental = True
                last_metric_date = start_date_obj - timedelta(days=1)
                fetch_start_date = start_date_obj - timedelta(days=self.max_lookback_days)
            except (ValueError, TypeError):
                is_incremental = True # 如果日期格式错误，退化为标准增量模式
        if is_incremental and not start_date_str:
            @sync_to_async(thread_sensitive=True)
            def get_latest_metric_async(model, stock_info_obj):
                try:
                    return model.objects.filter(stock=stock_info_obj).latest('trade_time')
                except model.DoesNotExist:
                    return None
            latest_metric = await get_latest_metric_async(MetricsModel, stock_info)
            if latest_metric:
                last_metric_date = latest_metric.trade_time
                fetch_start_date = last_metric_date - timedelta(days=self.max_lookback_days)
            else:
                is_incremental = False # 如果数据库为空，强制切换为全量模式
                fetch_start_date = None
                
        return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date

    async def _load_intraday_data_for_range(self, stock_info: StockInfo, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
        """
        【V2.4 · 范围查询修正版】
        - 核心修正: 将分钟数据回退逻辑中的 `__date__in` 查询替换为高效的 `__gte` 和 `__lte` 范围查询，根治回退加载失败的问题。
        """
        from django.utils import timezone
        from utils.model_helpers import get_stock_tick_data_model_by_code, get_stock_level5_data_model_by_code, get_daily_data_model_by_code
        from datetime import time, datetime
        intraday_data_map = {}
        start_datetime = timezone.make_aware(datetime.combine(start_date, time.min))
        end_datetime = timezone.make_aware(datetime.combine(end_date, time.max))
        TickModel = get_stock_tick_data_model_by_code(stock_info.stock_code)
        Level5Model = get_stock_level5_data_model_by_code(stock_info.stock_code)
        @sync_to_async(thread_sensitive=True)
        def get_hf_data(model, stock_pk, start_dt, end_dt):
            if not model: return pd.DataFrame()
            qs = model.objects.filter(stock_id=stock_pk, trade_time__gte=start_dt, trade_time__lt=end_dt).order_by('trade_time')
            return pd.DataFrame.from_records(qs.values())
        tick_df = await get_hf_data(TickModel, stock_info.pk, start_datetime, end_datetime)
        if not tick_df.empty:
            # 移除调试信息
            tick_df['trade_time'] = pd.to_datetime(tick_df['trade_time']).dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            level5_df = await get_hf_data(Level5Model, stock_info.pk, start_datetime, end_datetime)
            if not level5_df.empty:
                level5_df['trade_time'] = pd.to_datetime(level5_df['trade_time']).dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                tick_df = pd.merge_asof(tick_df.sort_values('trade_time'), level5_df.sort_values('trade_time'), on='trade_time', direction='backward')
                conditions = [tick_df['price'] >= tick_df['sell_price1'], tick_df['price'] <= tick_df['buy_price1']]
                choices = ['B', 'S']
                tick_df['type'] = np.select(conditions, choices, default='M')
            tick_df.set_index('trade_time', inplace=True)
            buy_vol_per_minute = tick_df[tick_df['type'] == 'B'].resample('1min')['volume'].sum()
            sell_vol_per_minute = tick_df[tick_df['type'] == 'S'].resample('1min')['volume'].sum()
            minute_df_from_hf = tick_df.resample('1min').agg(
                open=('price', 'first'), high=('price', 'max'), low=('price', 'min'),
                close=('price', 'last'), vol=('volume', 'sum'), amount=('amount', 'sum')
            ).dropna(subset=['open', 'high', 'low', 'close', 'vol', 'amount'])
            minute_df_from_hf['buy_vol_raw'] = buy_vol_per_minute
            minute_df_from_hf['sell_vol_raw'] = sell_vol_per_minute
            minute_df_from_hf.fillna({'buy_vol_raw': 0, 'sell_vol_raw': 0}, inplace=True)
            minute_df_from_hf.reset_index(inplace=True)
            minute_df_from_hf['date'] = minute_df_from_hf['trade_time'].dt.date
            intraday_data_map.update({date: group_df for date, group_df in minute_df_from_hf.groupby('date')})
        DailyModel = get_daily_data_model_by_code(stock_info.stock_code)
        all_dates_in_range_qs = DailyModel.objects.filter(stock=stock_info, trade_time__gte=start_date, trade_time__lte=end_date).values_list('trade_time', flat=True)
        all_required_dates = {d.date() for d in pd.to_datetime(await sync_to_async(list)(all_dates_in_range_qs))}
        dates_with_hf_data = set(intraday_data_map.keys())
        dates_for_fallback = sorted(list(all_required_dates - dates_with_hf_data))
        if dates_for_fallback:
            # 移除调试信息
            MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
            if MinuteModel:
                @sync_to_async(thread_sensitive=True)
                def get_minute_data_for_dates(model, stock_pk, dates_list):
                    # 核心修正：使用范围查询
                    start_dt_fallback = timezone.make_aware(datetime.combine(dates_list[0], time.min))
                    end_dt_fallback = timezone.make_aware(datetime.combine(dates_list[-1], time.max))
                    qs = model.objects.filter(stock_id=stock_pk, trade_time__gte=start_dt_fallback, trade_time__lte=end_dt_fallback)
                    qs_count = qs.count()
                    # 移除DB查询探针
                    if qs_count == 0:
                        return pd.DataFrame()
                    # 额外过滤确保只包含请求的日期
                    df = pd.DataFrame.from_records(qs.values('trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount').order_by('trade_time'))
                    df['trade_time'] = pd.to_datetime(df['trade_time']).dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                    df = df[df['trade_time'].dt.date.isin(dates_list)]
                    return df
                minute_df_fallback = await get_minute_data_for_dates(MinuteModel, stock_info.pk, dates_for_fallback)
                if not minute_df_fallback.empty:
                    cols_to_float = ['open', 'high', 'low', 'close', 'amount', 'vol']
                    for col in cols_to_float:
                        minute_df_fallback[col] = pd.to_numeric(minute_df_fallback[col], errors='coerce')
                    minute_df_fallback['date'] = minute_df_fallback['trade_time'].dt.date
                    intraday_data_map.update({date: group_df for date, group_df in minute_df_fallback.groupby('date')})
        return intraday_data_map

    async def _forge_advanced_structural_metrics(self, intraday_map: dict, stock_code: str, daily_df_with_atr: pd.DataFrame, historical_metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V46.0 · 潜龙在渊】
        - 核心升级: 在构建传递给下一日的 `prev_day_metrics` 上下文时，增加 `volume` 字段，
                    为 `equilibrium_compression_index` 的计算提供必要的“力量胶着度”评估依据。
        """
        new_metrics_data = []
        prev_day_calculated_metrics = {}
        if intraday_map:
            first_date_dt_obj = min(intraday_map.keys())
            prev_date_ts = pd.to_datetime(first_date_dt_obj) - pd.Timedelta(days=1)
            # 优先从 historical_metrics_df 中获取前一天的结构指标
            if prev_date_ts in historical_metrics_df.index:
                prev_hist_series = historical_metrics_df.loc[prev_date_ts]
                prev_day_calculated_metrics.update({
                    'vpoc': prev_hist_series.get('today_vpoc'), # 修正：使用正确的字段名 today_vpoc
                    'vah': prev_hist_series.get('today_vah'),   # 修正：使用正确的字段名 today_vah
                    'val': prev_hist_series.get('today_val'),   # 修正：使用正确的字段名 today_val
                })
            # 从 daily_df_with_atr 中获取前一天的日线基础数据
            if prev_date_ts in daily_df_with_atr.index:
                prev_daily_series = daily_df_with_atr.loc[prev_date_ts]
                prev_day_calculated_metrics.update({
                    'high': prev_daily_series.get('high_qfq'),
                    'low': prev_daily_series.get('low_qfq'),
                    'volume': prev_daily_series.get('vol'),
                    'atr_14d': prev_daily_series.get('ATR_14'),
                })
            # 确保所有关键字段都有默认的 NaN 值，以防数据缺失
            prev_day_calculated_metrics.setdefault('vpoc', np.nan)
            prev_day_calculated_metrics.setdefault('vah', np.nan)
            prev_day_calculated_metrics.setdefault('val', np.nan)
            prev_day_calculated_metrics.setdefault('high', np.nan)
            prev_day_calculated_metrics.setdefault('low', np.nan)
            prev_day_calculated_metrics.setdefault('volume', np.nan)
            prev_day_calculated_metrics.setdefault('atr_14d', np.nan)
        for trade_date_dt_obj, data_for_day in sorted(intraday_map.items()):
            current_trade_timestamp = pd.to_datetime(trade_date_dt_obj)
            if current_trade_timestamp not in daily_df_with_atr.index:
                logger.warning(f"[{stock_code}] 在日期 {trade_date_dt_obj} 对应的日线数据缺失，跳过当日计算。")
                continue
            daily_series_for_day = daily_df_with_atr.loc[current_trade_timestamp]
            canonical_minute_df = None
            tick_df_for_day = data_for_day.get('tick')
            minute_df_for_day = data_for_day.get('minute')
            if tick_df_for_day is not None and not tick_df_for_day.empty:
                resampled_df = tick_df_for_day.resample('1min').agg(
                    open=('price', 'first'),
                    high=('price', 'max'),
                    low=('price', 'min'),
                    close=('price', 'last'),
                    vol=('volume', 'sum'),
                    amount=('amount', 'sum')
                ).dropna(how='all')
                if not resampled_df.empty:
                    canonical_minute_df = resampled_df
            if canonical_minute_df is None and minute_df_for_day is not None and not minute_df_for_day.empty:
                if 'volume' in minute_df_for_day.columns and 'vol' not in minute_df_for_day.columns:
                    minute_df_for_day.rename(columns={'volume': 'vol'}, inplace=True)
                canonical_minute_df = minute_df_for_day
            if canonical_minute_df is None or canonical_minute_df.empty:
                logger.warning(f"[{stock_code}] 在日期 {trade_date_dt_obj} 缺少可用的分钟级或Tick级数据，跳过当日计算。")
                continue
            level5_df_for_day = data_for_day.get('level5')
            realtime_df_for_day = data_for_day.get('realtime')
            continuous_group = self._create_continuous_minute_data(canonical_minute_df)
            should_probe = self.debug_params.get('should_probe', False)
            probe_dates = self.debug_params.get('probe_dates', [])
            is_target_date = should_probe and (trade_date_dt_obj.strftime('%Y-%m-%d') in probe_dates)
            debug_info = {
                'is_target_date': is_target_date,
                'enable_probe': should_probe,
                'trade_date_str': trade_date_dt_obj.strftime('%Y-%m-%d'),
                'stock_code': stock_code
            }
            day_metric_dict = self._calculate_daily_structural_metrics(
                group=canonical_minute_df,
                continuous_group=continuous_group,
                tick_df=tick_df_for_day,
                level5_df=level5_df_for_day,
                realtime_df=realtime_df_for_day,
                daily_info=daily_series_for_day,
                prev_day_metrics=prev_day_calculated_metrics,
                debug_info=debug_info
            )
            day_metric_dict['trade_time'] = current_trade_timestamp
            day_metric_dict['stock_code'] = stock_code
            new_metrics_data.append(day_metric_dict)
            prev_day_calculated_metrics = {
                'vpoc': day_metric_dict.get('today_vpoc'), # 修正：使用正确的字段名 today_vpoc
                'vah': day_metric_dict.get('today_vah'),   # 修正：使用正确的字段名 today_vah
                'val': day_metric_dict.get('today_val'),   # 修正：使用正确的字段名 today_val
                'atr_14d': daily_series_for_day.get('ATR_14'),
                'high': daily_series_for_day.get('high_qfq'),
                'low': daily_series_for_day.get('low_qfq'),
                'volume': daily_series_for_day.get('vol'),
            }
        if not new_metrics_data:
            return pd.DataFrame()
        new_metrics_df = pd.DataFrame(new_metrics_data)
        new_metrics_df.set_index('trade_time', inplace=True)
        final_metrics_df = self._calculate_dynamic_evolution_factors(new_metrics_df)
        return final_metrics_df

    def _calculate_daily_structural_metrics(self, group: pd.DataFrame, continuous_group: pd.DataFrame,
                                            tick_df: pd.DataFrame | None, level5_df: pd.DataFrame | None,
                                            realtime_df: pd.DataFrame | None, daily_info: pd.Series,
                                            prev_day_metrics: dict, debug_info: dict) -> dict:
        context = {
            'group': group,
            'continuous_group': continuous_group,
            'tick_df': tick_df,
            'level5_df': level5_df,
            'realtime_df': realtime_df,
            'daily_series_for_day': daily_info,
            'prev_day_metrics': prev_day_metrics,
            'day_open_qfq': daily_info.get('open_qfq'),
            'day_high_qfq': daily_info.get('high_qfq'),
            'day_low_qfq': daily_info.get('low_qfq'),
            'day_close_qfq': daily_info.get('close_qfq'),
            'pre_close_qfq': daily_info.get('pre_close_qfq'),
            'atr_5': daily_info.get('ATR_5'),
            'atr_14': daily_info.get('ATR_14'),
            'atr_50': daily_info.get('ATR_50'),
            'total_volume_safe': group['vol'].sum() if 'vol' in group.columns and not group.empty else 0,
            'debug': debug_info,
            'turnover_rate_f': daily_info.get('turnover_rate_f'),
            'stock_code': debug_info.get('stock_code', 'N/A'), # 将 stock_code 添加到 context 顶层
        }
        energy_metrics = StructuralMetricsCalculators.calculate_energy_density_metrics(context)
        context.update(energy_metrics)
        microstructure_metrics = MicrostructureDynamicsCalculators.calculate_all(context)
        context.update(microstructure_metrics)
        profile_metrics = ThematicMetricsCalculators.calculate_market_profile_metrics(context)
        context.update(profile_metrics)
        control_metrics = StructuralMetricsCalculators.calculate_control_metrics(context)
        context.update(control_metrics)
        game_metrics = StructuralMetricsCalculators.calculate_game_efficiency_metrics(context)
        context.update(game_metrics)
        forward_metrics = ThematicMetricsCalculators.calculate_forward_looking_metrics(context)
        context.update(forward_metrics)
        battlefield_metrics = ThematicMetricsCalculators.calculate_battlefield_metrics(context)
        context.update(battlefield_metrics)
        derivative_metrics = DerivativeMetricsCalculator.calculate_divergence_metrics(context)
        context.update(derivative_metrics)
        all_metrics = {
            **energy_metrics,
            **microstructure_metrics,
            **profile_metrics,
            **control_metrics,
            **game_metrics,
            **forward_metrics,
            **battlefield_metrics,
            **derivative_metrics,
        }
        return all_metrics

    def _calculate_derivatives(self, stock_code: str, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 · 导数净化版】为所有核心结构指标计算斜率和加速度。
        - 核心修复: 引入并遵循模型的 SLOPE_ACCEL_EXCLUSIONS 列表，
                      跳过对布尔型等不适合计算导数的指标的处理，避免引入噪音和无效计算。
        """
        derivatives_df = pd.DataFrame(index=metrics_df.index)
        # 引入模型的排除列表
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedStructuralMetrics.CORE_METRICS.keys())
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedStructuralMetrics.SLOPE_ACCEL_EXCLUSIONS
        ACCEL_WINDOW = 2
        UNIFIED_PERIODS = [1, 5, 13, 21, 55]
        for col in CORE_METRICS_TO_DERIVE:
            # 检查指标是否在排除列表中
            if col in SLOPE_ACCEL_EXCLUSIONS:
                continue
            if col in metrics_df.columns:
                source_series = pd.to_numeric(metrics_df[col], errors='coerce')
                if source_series.isnull().all():
                    continue
                for p in UNIFIED_PERIODS:
                    calc_window = max(2, p) if p > 1 else 2
                    # 计算斜率
                    slope_col_name = f'{col}_slope_{p}d'
                    slope_series = ta.slope(close=source_series.astype(float), length=calc_window)
                    derivatives_df[slope_col_name] = slope_series
                    # 计算加速度
                    if slope_series is not None and not slope_series.empty:
                        accel_col_name = f'{col}_accel_{p}d'
                        derivatives_df[accel_col_name] = ta.slope(close=slope_series.astype(float), length=ACCEL_WINDOW)
        # 将衍生指标合并回原始指标DataFrame
        final_df = metrics_df.join(derivatives_df)
        return final_df

    async def _load_historical_metrics(self, model, stock_info, end_date):
        """
        【V1.1 · 索引修复版】从数据库加载并净化历史高级结构指标。
        - 核心修复: 修正了 `set_index` 的用法。旧用法会保留原始的 `trade_time` 列，导致下游 `reset_index` 操作时因列名冲突而失败。
                     新用法确保 `trade_time` 列在被设置为索引后，从DataFrame的列中被正确移除。
        """
        @sync_to_async
        def get_data():
            core_metric_cols = list(BaseAdvancedStructuralMetrics.CORE_METRICS.keys())
            required_cols = ['trade_time'] + [col for col in core_metric_cols if hasattr(model, col)]
            qs = model.objects.filter(
                stock=stock_info,
                trade_time__lt=end_date
            ).order_by('trade_time').values(*required_cols)
            return pd.DataFrame.from_records(qs)
        df = await get_data()
        if not df.empty:
            # 先将 'trade_time' 列转换为 datetime 类型
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            # 使用列名字符串作为参数，确保设置索引后该列被移除，解决下游 reset_index 冲突
            df = df.set_index('trade_time')
            # 在数据源头进行类型净化，杜绝object类型污染
            for col in df.columns:
                # 'trade_time' 已成为索引，不再是列，因此无需在循环中进行特殊处理
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _calculate_dynamic_evolution_factors(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V37.10 · 动态因子健壮性修正】
        - 核心修复: 为所有动态演化因子的计算增加防御性检查。在尝试计算滚动平均或变化率之前，
                     首先验证其依赖的源数据列是否存在。如果源列因缺少高频数据等原因未能生成，
                     则将衍生因子列直接置为NaN，从而避免KeyError，大幅提升计算流程的健壮性。
        """
        df = metrics_df.copy().sort_index()
        # 为每个衍生因子的计算增加源列存在性检查
        # 核心意愿演化
        if 'intraday_thrust_purity' in df.columns:
            df['thrust_purity_ma5'] = df['intraday_thrust_purity'].rolling(window=5, min_periods=1).mean()
        else:
            df['thrust_purity_ma5'] = np.nan
        # 吸筹行为演化
        if 'absorption_strength_index' in df.columns:
            df['absorption_strength_ma5'] = df['absorption_strength_index'].rolling(window=5, min_periods=1).mean()
        else:
            df['absorption_strength_ma5'] = np.nan
        # 情绪激进度演化
        if 'buy_sweep_intensity' in df.columns:
            df['sweep_intensity_ma5'] = df['buy_sweep_intensity'].rolling(window=5, min_periods=1).mean()
        else:
            df['sweep_intensity_ma5'] = np.nan
        # 流动性风险演化
        if 'vpin_score' in df.columns:
            df['vpin_roc3'] = df['vpin_score'].pct_change(periods=3)
        else:
            df['vpin_roc3'] = np.nan
        return df

    def _create_continuous_minute_data(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        【V31.0 · 索引访问模式统一】
        - 核心修复: 移除了方法末尾的 reset_index() 和 rename()，确保返回的DataFrame保持DatetimeIndex。
                     这是实现全系统索引访问模式统一的源头修复。
        """
        if group is None or group.empty:
            return pd.DataFrame()
        trade_date = group.index[0].date()
        morning_session = pd.to_datetime(pd.date_range(start=f'{trade_date} 09:31:00', end=f'{trade_date} 12:00:00', freq='1min'))
        afternoon_session = pd.to_datetime(pd.date_range(start=f'{trade_date} 13:01:00', end=f'{trade_date} 15:00:00', freq='1min'))
        full_day_index = morning_session.union(afternoon_session).tz_localize('Asia/Shanghai')
        continuous_group = group.reindex(full_day_index)
        price_cols = ['open', 'high', 'low', 'close']
        existing_price_cols = [col for col in price_cols if col in continuous_group.columns]
        continuous_group[existing_price_cols] = continuous_group[existing_price_cols].ffill()
        transaction_cols = ['vol', 'amount']
        existing_transaction_cols = [col for col in transaction_cols if col in continuous_group.columns]
        continuous_group[existing_transaction_cols] = continuous_group[existing_transaction_cols].fillna(0)
        continuous_group['minute_vwap'] = (continuous_group['amount'] / continuous_group['vol']).where(continuous_group['vol'] > 0, np.nan)
        # 移除 reset_index 和 rename，保持 DatetimeIndex
        return continuous_group

    def _calculate_trend_metrics(self, price_series: pd.Series) -> tuple[float, float]:
        """
        【V30.8 · 新增辅助函数】
        通过线性回归计算价格序列的趋势强度和趋势质量。
        - 趋势强度: 回归线的斜率。
        - 趋势质量: 回归的R平方值。
        :param price_series: 分钟收盘价序列。
        :return: (趋势强度, 趋势质量) 的元组。
        - 核心优化: 使用Numba优化后的趋势指标计算函数。
        """
        # 检查输入数据是否有效，至少需要两个点才能拟合一条直线
        cleaned_series = price_series.dropna()
        if len(cleaned_series) < 2:
            return 0.0, 0.0
        # 调用Numba优化函数
        return _numba_calculate_trend_metrics(cleaned_series.values)

    def _calculate_mean_reversion_speed(self, price_series: pd.Series) -> float:
        """
        【V30.9 · 新增辅助函数】
        估算价格序列的均值回归速度。
        基于Ornstein-Uhlenbeck过程的离散化模型，通过回归价格变化与滞后价格来计算。
        :param price_series: 分钟收盘价序列。
        :return: 均值回归速度。正值表示存在均值回归，值越大速度越快。
        - 核心优化: 使用Numba优化后的均值回归速度计算函数。
        """
        cleaned_series = price_series.dropna()
        if len(cleaned_series) < 2:
            return 0.0
        # 调用Numba优化函数
        return _numba_calculate_mean_reversion_speed(cleaned_series.values)

    def _calculate_tpo_metrics(self, group: pd.DataFrame) -> dict:
        """
        【V30.10 · 新增辅助函数】
        基于日内分钟数据计算市场轮廓（TPO/Market Profile）的核心指标。
        - VPOC (Volume Point of Control): 成交量最大的价格点。
        - Value Area (VA): 包含当日70%成交量的价格区域。
        :param group: 包含'close'和'vol'列的日内分钟数据DataFrame。
        :return: 包含VPOC, VAH, VAL的字典。
        - 核心优化: 使用Numba优化后的TPO指标计算函数。
        """
        if group.empty or 'vol' not in group.columns or group['vol'].sum() == 0:
            return {
                '_today_vpoc': np.nan,
                '_today_vah': np.nan,
                '_today_val': np.nan,
            }
        # 提取NumPy数组
        close_arr = group['close'].values
        vol_arr = group['vol'].values
        # 调用Numba优化函数
        vpoc, vah, val = _numba_calculate_tpo_metrics(close_arr, vol_arr)
        return {
            '_today_vpoc': vpoc,
            '_today_vah': vah,
            '_today_val': val,
        }

    def _calculate_continuous_data_metrics(self, continuous_group: pd.DataFrame) -> dict:
        """
        【V30.12 · 索引健壮性修复】
        分析跨交易日的连续数据，主要用于衡量隔夜跳空缺口及其后续影响。
        - 核心修复: 增加对DataFrame索引类型的判断。当索引为RangeIndex而非预期的DatetimeIndex时，
                    从'trade_time'列获取日期信息，以避免AttributeError。
        :param continuous_group: 由前一日后半段和当日前半段拼接的分钟数据DataFrame。
        :return: 包含跳空回报率和缺口后动量的字典。
        - 核心优化: 使用Numba优化后的跳空回报率和缺口后动量计算函数。
        """
        metrics = {
            'gap_return': np.nan,
            'post_gap_momentum_30min': np.nan,
        }
        if continuous_group is None or continuous_group.empty or len(continuous_group) < 2:
            return metrics
        # 1. 识别两个交易日
        if isinstance(continuous_group.index, pd.DatetimeIndex):
            trade_dates = continuous_group.index.date
        elif 'trade_time' in continuous_group.columns:
            trade_dates = pd.to_datetime(continuous_group['trade_time']).dt.date
        else:
            return metrics
        unique_dates = np.unique(trade_dates)
        if len(unique_dates) != 2:
            return metrics
        # 2. 拆分数据并获取关键价格
        prev_day_data = continuous_group[trade_dates == unique_dates[0]]
        today_data = continuous_group[trade_dates == unique_dates[1]]
        if prev_day_data.empty or today_data.empty:
            return metrics
        prev_close = prev_day_data['close'].iloc[-1]
        today_open = today_data['open'].iloc[0]
        # 截取今日开盘后30分钟的数据
        first_30_min_data = today_data.head(30)
        today_close_arr = first_30_min_data['close'].values if not first_30_min_data.empty else np.array([])
        # 调用Numba优化函数
        gap_return, post_gap_momentum_30min = _numba_calculate_continuous_data_metrics(
            prev_close, today_open, today_close_arr
        )
        metrics['gap_return'] = gap_return
        metrics['post_gap_momentum_30min'] = post_gap_momentum_30min
        return metrics

    def _calculate_atr_interaction_metrics(self, group: pd.DataFrame, atr_5: float, atr_14: float, atr_50: float) -> dict:
        """
        【V30.13 · 新增辅助函数】
        计算日内价格行为与日线ATR之间的交互指标。
        :param group: 日内分钟数据DataFrame。
        :param atr_5: 5日ATR。
        :param atr_14: 14日ATR。
        :param atr_50: 50日ATR。
        :return: 包含ATR交互指标的字典。
        - 核心优化: 使用Numba优化后的ATR交互指标计算函数。
        """
        metrics = {
            'intraday_range_vs_atr14': np.nan,
            'close_vwap_deviation_normalized': np.nan,
            'volatility_expansion_ratio': np.nan,
        }
        if group.empty:
            return metrics
        day_high = group['high'].max()
        day_low = group['low'].min()
        day_close = group['close'].iloc[-1]
        total_volume = group['vol'].sum()
        total_amount = group['amount'].sum()
        vwap = total_amount / total_volume if total_volume > 0 else day_close
        # 调用Numba优化函数
        intraday_range_vs_atr14, close_vwap_deviation_normalized, volatility_expansion_ratio = \
            _numba_calculate_atr_interaction_metrics(
                day_high, day_low, atr_5, atr_14, atr_50, day_close, vwap
            )
        metrics['intraday_range_vs_atr14'] = intraday_range_vs_atr14
        metrics['close_vwap_deviation_normalized'] = close_vwap_deviation_normalized
        metrics['volatility_expansion_ratio'] = volatility_expansion_ratio
        return metrics

    def _calculate_prev_day_interaction_metrics(self, group: pd.DataFrame, prev_day_metrics: dict) -> dict:
        """
        【V30.14 · 新增辅助函数】
        计算当日市场行为与前一日关键结构位（如价值区）的交互指标。
        :param group: 日内分钟数据DataFrame。
        :param prev_day_metrics: 包含前一日VPOC, VAH, VAL, ATR的字典。
        :return: 包含交互指标的字典。
        - 核心优化: 使用Numba优化后的前一日交互指标计算函数。
        """
        metrics = {
            'value_area_migration': np.nan,
            'value_area_overlap_pct': np.nan,
            'closing_acceptance_type': np.nan,
            'opening_position_vs_prev_va': np.nan,
        }
        if group.empty or not prev_day_metrics:
            return metrics
        # 1. 获取当日和前一日的关键指标
        today_tpo = self._calculate_tpo_metrics(group)
        today_vpoc = today_tpo.get('_today_vpoc')
        today_vah = today_tpo.get('_today_vah')
        today_val = today_tpo.get('_today_val')
        day_close = group['close'].iloc[-1]
        day_open = group['open'].iloc[0]
        prev_vpoc = prev_day_metrics.get('vpoc')
        prev_vah = prev_day_metrics.get('vah')
        prev_val = prev_day_metrics.get('val')
        prev_atr = prev_day_metrics.get('atr_14d')
        # 调用Numba优化函数
        value_area_migration, value_area_overlap_pct, closing_acceptance_type, opening_position_vs_prev_va = \
            _numba_calculate_prev_day_interaction_metrics(
                today_vpoc, today_vah, today_val, day_close, day_open,
                prev_vpoc, prev_vah, prev_val, prev_atr
            )
        metrics['value_area_migration'] = value_area_migration
        metrics['value_area_overlap_pct'] = value_area_overlap_pct
        metrics['closing_acceptance_type'] = closing_acceptance_type
        metrics['opening_position_vs_prev_va'] = opening_position_vs_prev_va
        return metrics

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        """
        【V30.21 · 持久化类型修复】
        - 核心修复: 移除了在保存数据时对 trade_time 字段多余的 .date() 调用，因为上游传入的值已是 date 类型。
        """
        if final_df.empty:
            return 0
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        model_fields = {f.name for f in MetricsModel._meta.get_fields() if not f.is_relation and f.name != 'id'}
        df_filtered = final_df[[col for col in final_df.columns if col in model_fields]]
        @sync_to_async(thread_sensitive=True)
        def save_atomically(model, stock_obj, df_to_save):
            processed_count = 0
            # 使用 reset_index() 来安全地将索引转换为列进行迭代
            for record_data in df_to_save.reset_index().to_dict('records'):
                # 移除多余的 .date() 调用，因为值已是 date 对象
                trade_time = record_data.pop('trade_time')
                defaults_data = {key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in record_data.items()}
                try:
                    obj, created = model.objects.update_or_create(
                        stock=stock_obj,
                        trade_time=trade_time,
                        defaults=defaults_data
                    )
                    processed_count += 1
                except Exception as e:
                    logger.error(f"[{stock_obj.stock_code}] [结构指标保存失败] 日期: {trade_time}, 错误: {e}")
            return processed_count
        processed_count = await save_atomically(MetricsModel, stock_info, df_filtered)
        return processed_count

class DerivativeMetricsCalculator:
    """
    【V36.0 · 动能背离】
    衍生指标计算器，专注于对已有的基础指标进行二次分析，以发现更深层次的模式，如“背离”。
    """
    @staticmethod
    def calculate_divergence_metrics(context: dict) -> dict:
        """
        计算各类背离指标，核心是 `price_thrust_divergence`。
        【V58.0 · 诡道归元】
        - 核心升级: 新增对“底背离”的计算逻辑，实现了对“顶背离”与“底背离”的阴阳合一，
                     使指标能同时捕捉上涨衰竭的风险和下跌企稳的机会。
        - 指标整合: 将顶、底背离得分统一至 `price_thrust_divergence`，负值为顶背离，正值为底背离。
        - 核心优化: 使用Numba优化后的价格推力背离计算函数。
        """
        tick_df = context.get('tick_df')
        group = context.get('group')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        if tick_df is None or tick_df.empty or group is None or group.empty:
            return results
        midday_break_time = time(13, 0)
        am_ticks = tick_df[tick_df.index.time < midday_break_time]
        pm_ticks = tick_df[tick_df.index.time >= midday_break_time]
        am_group = group[group.index.time < midday_break_time]
        pm_group = group[group.index.time >= midday_break_time]
        if am_ticks.empty or pm_ticks.empty or am_group.empty or pm_group.empty:
            return results
        am_thrust, pm_thrust = 0.0, 0.0 # 确保为浮点数
        am_total_vol = am_ticks['volume'].sum()
        if am_total_vol > 0:
            if 'price_change' in am_ticks.columns and not am_ticks['price_change'].isnull().all():
                self_calculated_change = am_ticks['price'].diff().fillna(0)
                zero_change_mask = am_ticks['price_change'] == 0
                effective_price_change = np.where(zero_change_mask, self_calculated_change, am_ticks['price_change'])
                net_thrust_volume = (am_ticks['volume'] * np.sign(effective_price_change)).sum()
                am_thrust = net_thrust_volume / am_total_vol
            elif 'type' in am_ticks.columns:
                am_buy_vol = am_ticks[am_ticks['type'] == 'B']['volume'].sum()
                am_sell_vol = am_ticks[am_ticks['type'] == 'S']['volume'].sum()
                am_thrust = (am_buy_vol - am_sell_vol) / am_total_vol
        pm_total_vol = pm_ticks['volume'].sum()
        if pm_total_vol > 0:
            if 'price_change' in pm_ticks.columns and not pm_ticks['price_change'].isnull().all():
                self_calculated_change = pm_ticks['price'].diff().fillna(0)
                zero_change_mask = pm_ticks['price_change'] == 0
                effective_price_change = np.where(zero_change_mask, self_calculated_change, pm_ticks['price_change'])
                net_thrust_volume = (pm_ticks['volume'] * np.sign(effective_price_change)).sum()
                pm_thrust = net_thrust_volume / pm_total_vol
            elif 'type' in pm_ticks.columns:
                pm_buy_vol = pm_ticks[pm_ticks['type'] == 'B']['volume'].sum()
                pm_sell_vol = pm_ticks[pm_ticks['type'] == 'S']['volume'].sum()
                pm_thrust = (pm_buy_vol - pm_sell_vol) / pm_total_vol
        am_high, pm_high = am_group['high'].max(), pm_group['high'].max()
        am_low, pm_low = am_group['low'].min(), pm_group['low'].min()
        # 调用Numba优化函数
        results['price_thrust_divergence'] = _numba_calculate_price_thrust_divergence(
            am_high, pm_high, am_low, pm_low, am_thrust, pm_thrust
        )
        return results

class StructuralMetricsCalculators:
    """
    【V25.0 · 计算内核剥离】
    - 核心职责: 封装所有高级结构指标的纯计算逻辑，与服务流程完全解耦。
    - 架构模式: 作为一个无状态的静态工具类，提供一系列独立的计算函数。
    """
    @staticmethod
    def calculate_energy_density_metrics(context: dict) -> dict:
        group = context['group']
        daily_series_for_day = context['daily_series_for_day']
        atr_14 = context['atr_14']
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        day_open_qfq = context['day_open_qfq']
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
        day_close_qfq = context['day_close_qfq']
        pre_close_qfq = context['pre_close_qfq']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        stock_code = debug_info.get('stock_code', 'N/A')
        results = {
            'intraday_energy_density': np.nan,
            'intraday_thrust_purity': np.nan,
            'volume_burstiness_index': np.nan,
            'auction_impact_score': np.nan,
            'dynamic_reversal_strength': np.nan,
            'reversal_conviction_rate': np.nan,
            'reversal_recovery_rate': np.nan,
            'high_level_consolidation_volume': np.nan,
            'opening_period_thrust': np.nan,
        }
        if pd.notna(atr_14) and atr_14 > 0:
            turnover_rate_f = context.get('turnover_rate_f') # 从 context 中获取 turnover_rate_f
            if pd.notna(turnover_rate_f):
                results['intraday_energy_density'] = np.log1p(turnover_rate_f) / atr_14
        if tick_df is not None and not tick_df.empty:
            total_volume = tick_df['volume'].sum()
            if total_volume > 0:
                if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
                    self_calculated_change = tick_df['price'].diff().fillna(0)
                    zero_change_mask = tick_df['price_change'] == 0
                    effective_price_change = np.where(zero_change_mask, self_calculated_change, tick_df['price_change'])
                    net_thrust_volume = (tick_df['volume'] * np.sign(effective_price_change)).sum()
                    results['intraday_thrust_purity'] = net_thrust_volume / total_volume
                elif 'type' in tick_df.columns:
                    active_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
                    active_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
                    results['intraday_thrust_purity'] = (active_buy_vol - active_sell_vol) / total_volume
        else:
            thrust_vector = (group['close'] - group['open']) * group['vol']
            absolute_energy = abs(group['close'] - group['open']) * group['vol']
            total_energy = absolute_energy.sum()
            if total_energy > 0:
                results['intraday_thrust_purity'] = thrust_vector.sum() / total_energy
        if tick_df is not None and not tick_df.empty:
            results['volume_burstiness_index'] = StructuralMetricsCalculators.calculate_gini(tick_df['volume'].values)
        else:
            results['volume_burstiness_index'] = StructuralMetricsCalculators.calculate_gini(group['vol'].values)
        if all(pd.notna(v) for v in [day_open_qfq, pre_close_qfq, atr_14]) and atr_14 > 0:
            gap_magnitude = (day_open_qfq - pre_close_qfq) / atr_14
            if tick_df is not None and not tick_df.empty and level5_df is not None and not level5_df.empty:
                opening_ticks = tick_df[tick_df.index.time < time(9, 35)]
                opening_level5 = level5_df[level5_df.index.time < time(9, 35)]
                if not opening_ticks.empty and not opening_level5.empty:
                    merged_hf = pd.merge_asof(opening_ticks.sort_index(), opening_level5.sort_index(), on='trade_time', direction='backward')
                    merged_hf['mid_price'] = (merged_hf['buy_price1'] + merged_hf['sell_price1']) / 2
                    merged_hf['prev_mid_price'] = merged_hf['mid_price'].shift(1)
                    buy_pressure = np.where(merged_hf['mid_price'] >= merged_hf['prev_mid_price'], merged_hf['buy_volume1'].shift(1), 0)
                    sell_pressure = np.where(merged_hf['mid_price'] <= merged_hf['prev_mid_price'], merged_hf['sell_volume1'].shift(1), 0)
                    merged_hf['ofi'] = buy_pressure - sell_pressure
                    opening_ofi = merged_hf['ofi'].sum()
                    opening_volume = merged_hf['volume'].sum()
                    if opening_volume > 0:
                        conviction_factor = np.tanh(opening_ofi / opening_volume)
                        results['auction_impact_score'] = gap_magnitude * (1 + conviction_factor * np.sign(gap_magnitude))
                    else:
                        results['auction_impact_score'] = gap_magnitude
                else:
                    results['auction_impact_score'] = gap_magnitude
            else:
                results['auction_impact_score'] = gap_magnitude
        try:
            from scipy.signal import find_peaks
            prominence_source = "静态回退"
            if pd.notna(atr_14) and atr_14 > 0:
                dynamic_prominence = atr_14 * 0.05
                prominence_source = f"动态ATR({atr_14:.2f}*5%)"
            else:
                dynamic_prominence = 0.01
            peaks, _ = find_peaks(group['high'], distance=5, prominence=dynamic_prominence)
            troughs, _ = find_peaks(-group['low'], distance=5, prominence=dynamic_prominence)
            if len(troughs) > 0 and len(peaks) > 0:
                reversal_details = []
                all_extrema = sorted(np.concatenate([peaks, troughs]))
                first_trough_idx = -1
                for i, extremum_pos in enumerate(all_extrema):
                    if extremum_pos in troughs:
                        first_trough_idx = i
                        break
                if first_trough_idx != -1:
                    for i in range(first_trough_idx, len(all_extrema) - 1):
                        if all_extrema[i] in troughs and all_extrema[i+1] in peaks:
                            trough_pos = all_extrema[i]
                            peak_pos = all_extrema[i+1]
                            prev_peak_candidates = peaks[peaks < trough_pos]
                            if len(prev_peak_candidates) > 0:
                                prev_peak_pos = prev_peak_candidates[-1]
                                falling_phase = group.iloc[prev_peak_pos:trough_pos+1]
                                rebounding_phase = group.iloc[trough_pos:peak_pos+1]
                                vol_fall = falling_phase['vol'].sum()
                                vol_rebound = rebounding_phase['vol'].sum()
                                if not falling_phase.empty and not rebounding_phase.empty and \
                                   vol_fall > 0 and vol_rebound > 0:
                                    vwap_fall = falling_phase['amount'].sum() / vol_fall
                                    vwap_rebound = rebounding_phase['amount'].sum() / vol_rebound
                                    if vwap_fall > 0:
                                        price_momentum = (vwap_rebound / vwap_fall - 1)
                                        fall_magnitude = group.iloc[prev_peak_pos]['high'] - group.iloc[trough_pos]['low']
                                        rebound_magnitude = group.iloc[peak_pos]['high'] - group.iloc[trough_pos]['low']
                                        recovery_rate = rebound_magnitude / fall_magnitude if fall_magnitude > 0 else 0
                                        if recovery_rate > 1:
                                            volume_factor = np.log1p(vol_rebound / vol_fall)
                                        else:
                                            volume_factor = np.log1p(vol_fall / vol_rebound)
                                        momentum = (price_momentum * volume_factor) * 100
                                        reversal_details.append({
                                            "momentum": momentum,
                                            "fall_magnitude": fall_magnitude,
                                            "rebound_magnitude": rebound_magnitude
                                        })
                if reversal_details:
                    positive_momentums = [r['momentum'] for r in reversal_details if r['momentum'] > 0]
                    negative_momentums = [r['momentum'] for r in reversal_details if r['momentum'] <= 0]
                    sum_positive_momentum = np.sum(positive_momentums)
                    sum_abs_negative_momentum = np.sum(np.abs(negative_momentums))
                    total_abs_momentum = sum_positive_momentum + sum_abs_negative_momentum
                    conviction_rate = 0.0
                    if total_abs_momentum > 0:
                        conviction_rate = sum_positive_momentum / total_abs_momentum
                    results['reversal_conviction_rate'] = conviction_rate
                    if positive_momentums:
                        raw_strength = np.mean(positive_momentums)
                        final_strength = raw_strength * conviction_rate
                        results['dynamic_reversal_strength'] = final_strength
                        successful_reversals = [r for r in reversal_details if r['momentum'] > 0]
                        recovery_ratios = [
                            r['rebound_magnitude'] / r['fall_magnitude']
                            for r in successful_reversals if r['fall_magnitude'] > 0
                        ]
                        if recovery_ratios:
                            results['reversal_recovery_rate'] = np.mean(recovery_ratios)
        except ImportError:
            pass
        price_range = day_high_qfq - day_low_qfq
        if price_range > 0 and pd.notna(atr_14) and atr_14 > 0:
            high_level_threshold = day_high_qfq - 0.25 * price_range
            volume_ratio = 0.0
            if tick_df is not None and not tick_df.empty:
                high_vol = tick_df[tick_df['price'] >= high_level_threshold]['volume'].sum()
                total_vol = tick_df['volume'].sum()
                if total_vol > 0:
                    volume_ratio = high_vol / total_vol
            else:
                total_volume = group['vol'].sum()
                if total_volume > 0:
                    volume_ratio = group[group['high'] >= high_level_threshold]['vol'].sum() / total_volume
            distance_from_threshold = day_close_qfq - high_level_threshold
            normalized_distance = distance_from_threshold / atr_14
            confirmation_factor = np.tanh(normalized_distance)
            results['high_level_consolidation_volume'] = volume_ratio * confirmation_factor
        if tick_df is not None and not tick_df.empty:
            opening_ticks = tick_df.between_time('09:30:00', '09:59:59')
            if not opening_ticks.empty:
                opening_total_vol = opening_ticks['volume'].sum()
                if opening_total_vol > 0:
                    if 'price_change' not in opening_ticks.columns:
                        opening_ticks['price_change'] = opening_ticks['price'].diff().fillna(0)
                    self_calculated_change = opening_ticks['price'].diff().fillna(0)
                    zero_change_mask = opening_ticks['price_change'] == 0
                    effective_price_change = np.where(zero_change_mask, self_calculated_change, opening_ticks['price_change'])
                    net_thrust_volume = (opening_ticks['volume'] * np.sign(effective_price_change)).sum()
                    results['opening_period_thrust'] = net_thrust_volume / opening_total_vol
                elif 'type' in opening_ticks.columns:
                    opening_buy_vol = opening_ticks[opening_ticks['type'] == 'B']['volume'].sum()
                    opening_sell_vol = opening_ticks[opening_ticks['type'] == 'S']['volume'].sum()
                    results['opening_period_thrust'] = (opening_buy_vol - opening_sell_vol) / opening_total_vol
        return results

    @staticmethod
    def calculate_control_metrics(context: dict) -> dict:
        group = context['group']
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        daily_info = context['daily_series_for_day']
        day_open_qfq = context['day_open_qfq']
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        total_volume_safe = context['total_volume_safe']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        stock_code = debug_info.get('stock_code', 'N/A')
        results = {
            'cost_dispersion_index': np.nan,
            'intraday_pnl_imbalance': np.nan,
            'mean_reversion_frequency': np.nan,
            'trend_efficiency_ratio': np.nan,
            'pullback_depth_ratio': np.nan,
            'opening_impulse_efficiency': np.nan,
            'midday_narrow_range_gravity': np.nan,
            'tail_acceleration_efficiency': np.nan,
            'closing_conviction_score': np.nan,
            'absorption_strength_index': np.nan,
        }
        if group.empty or total_volume_safe == 0 or not pd.notna(atr_14) or atr_14 == 0:
            return results
        dispersion_raw = np.nan
        if tick_df is not None and not tick_df.empty:
            weighted_price_mean = np.average(tick_df['price'], weights=tick_df['volume'])
            variance = np.average((tick_df['price'] - weighted_price_mean)**2, weights=tick_df['volume'])
            dispersion_raw = np.sqrt(variance)
            results['cost_dispersion_index'] = dispersion_raw / atr_14
        else:
            weighted_price_mean = np.average(group['close'], weights=group['vol'])
            variance = np.average((group['close'] - weighted_price_mean)**2, weights=group['vol'])
            dispersion_raw = np.sqrt(variance)
            results['cost_dispersion_index'] = dispersion_raw / atr_14
        first_half = continuous_group[continuous_group.index.time < time(11, 30)]
        second_half = continuous_group[continuous_group.index.time >= time(13, 0)]
        if not first_half.empty and not second_half.empty and first_half['vol'].sum() > 0 and second_half['vol'].sum() > 0:
            vwap_first = (first_half['amount'].sum() / first_half['vol'].sum())
            vwap_second = (second_half['amount'].sum() / second_half['vol'].sum())
            results['intraday_pnl_imbalance'] = (vwap_second - vwap_first) / atr_14
        if 'minute_vwap' in continuous_group.columns and not continuous_group['minute_vwap'].isnull().all():
            if tick_df is not None and not tick_df.empty:
                continuous_group.index.name = 'trade_time'
                merged_df = pd.merge_asof(tick_df.sort_index(), continuous_group[['minute_vwap']].sort_index(), on='trade_time', direction='backward')
                merged_df['position'] = np.sign(merged_df['price'] - merged_df['minute_vwap'])
                crossings = (merged_df['position'].diff().abs() == 2).sum()
                results['mean_reversion_frequency'] = (crossings / len(tick_df)) * 1000 if len(tick_df) > 0 else 0
            else:
                position = np.sign(continuous_group['close'] - continuous_group['minute_vwap'])
                crossings = (position.diff().abs() == 2).sum()
                results['mean_reversion_frequency'] = (crossings / len(continuous_group)) * 100 if len(continuous_group) > 0 else 0
        price_change = day_close_qfq - day_open_qfq
        sum_abs_minute_change = (continuous_group['high'] - continuous_group['low']).sum()
        if sum_abs_minute_change > 0:
            er_raw = abs(price_change) / sum_abs_minute_change
            thrust_purity = context.get('intraday_thrust_purity', 0)
            results['trend_efficiency_ratio'] = er_raw * (1 + thrust_purity) * np.sign(price_change)
        minute_return = continuous_group['close'].pct_change().fillna(0)
        minute_volume = continuous_group['vol']
        advancing_mask = continuous_group['close'] > continuous_group['open']
        declining_mask = continuous_group['close'] < continuous_group['open']
        if advancing_mask.sum() > 2 and declining_mask.sum() > 2:
            corr_adv = minute_return[advancing_mask].corr(minute_volume[advancing_mask])
            corr_dec = minute_return[declining_mask].corr(minute_volume[declining_mask])
            corr_adv = corr_adv if pd.notna(corr_adv) else 0
            corr_dec = corr_dec if pd.notna(corr_dec) else 0
            trend_direction = np.sign(day_close_qfq - day_open_qfq) if (day_close_qfq != day_open_qfq) else 1
            results['pullback_depth_ratio'] = (corr_adv - corr_dec) * trend_direction
        open_period_df = continuous_group.between_time('09:30', '10:00')
        mid_period_df = continuous_group.between_time('10:01', '14:29')
        tail_period_df = continuous_group.between_time('14:30', '15:00')
        if not open_period_df.empty and total_volume_safe > 0:
            open_vol_ratio = open_period_df['vol'].sum() / total_volume_safe
            if open_vol_ratio > 0:
                open_price_change = open_period_df['close'].iloc[-1] - open_period_df['open'].iloc[0]
                price_change_norm = open_price_change / atr_14
                results['opening_impulse_efficiency'] = price_change_norm / open_vol_ratio
        if not mid_period_df.empty and not open_period_df.empty and not tail_period_df.empty:
            volatility_mid = mid_period_df['close'].pct_change().std()
            active_period_df = pd.concat([open_period_df, tail_period_df])
            volatility_active = active_period_df['close'].pct_change().std()
            if pd.notna(volatility_mid) and pd.notna(volatility_active) and volatility_active > 0:
                results['midday_narrow_range_gravity'] = 1 - (volatility_mid / volatility_active)
        if not tail_period_df.empty and total_volume_safe > 0:
            tail_vol_ratio = tail_period_df['vol'].sum() / total_volume_safe
            if tail_vol_ratio > 0:
                tail_price_change = tail_period_df['close'].iloc[-1] - tail_period_df['open'].iloc[0]
                price_change_norm = tail_price_change / atr_14
                results['tail_acceleration_efficiency'] = price_change_norm / tail_vol_ratio
        if not tail_period_df.empty and not mid_period_df.empty and mid_period_df['vol'].mean() > 0:
            accel_ratio = tail_period_df['vol'].mean() / mid_period_df['vol'].mean()
            tail_thrust_purity = np.nan
            if tick_df is not None:
                tail_ticks = tick_df.between_time('14:30', '15:00')
                if not tail_ticks.empty and tail_ticks['volume'].sum() > 0:
                    tail_total_vol = tail_ticks['volume'].sum()
                    if 'price_change' not in tail_ticks.columns:
                        tail_ticks['price_change'] = tail_ticks['price'].diff().fillna(0)
                    self_calculated_change = tail_ticks['price'].diff().fillna(0)
                    zero_change_mask = tail_ticks['price_change'] == 0
                    effective_price_change = np.where(zero_change_mask, self_calculated_change, tail_ticks['price_change'])
                    net_thrust_vol = (tail_ticks['volume'] * np.sign(effective_price_change)).sum()
                    tail_thrust_purity = net_thrust_vol / tail_total_vol
                elif 'type' in tail_ticks.columns:
                    buy_vol = tail_ticks[tail_ticks['type'] == 'B']['volume'].sum()
                    sell_vol = tail_ticks[tail_ticks['type'] == 'S']['volume'].sum()
                    tail_thrust_purity = (buy_vol - sell_vol) / tail_total_vol
            vpoc = context.get('_today_vpoc', np.nan)
            if pd.notna(vpoc):
                deviation_magnitude = (day_close_qfq - vpoc) / atr_14
                tail_force_factor = np.log1p(accel_ratio)
                conviction_purity = tail_thrust_purity if pd.notna(tail_thrust_purity) else np.sign(day_close_qfq - vpoc)
                results['closing_conviction_score'] = deviation_magnitude * tail_force_factor * conviction_purity
        if tick_df is not None and not tick_df.empty:
            # 确保 price_diff 列存在
            if 'price_diff' not in tick_df.columns:
                tick_df['price_diff'] = tick_df['price'].diff()
            down_moves = tick_df[tick_df['price_diff'] < 0].copy() # 确保是副本，避免SettingWithCopyWarning
            if not down_moves.empty:
                mf_buy_on_dip = down_moves[down_moves['amount'] > 200000]['volume'].sum()
                total_vol_on_dip = down_moves['volume'].sum()
                if total_vol_on_dip > 0:
                    results['absorption_strength_index'] = mf_buy_on_dip / total_vol_on_dip
        else:
            down_minutes = group[group['close'] < group['open']].copy() # 确保是副本
            if not down_minutes.empty:
                if 'main_force_buy_vol' in down_minutes.columns: # 假设 main_force_buy_vol 是分钟数据中主力买入量
                    mf_buy_on_dip = down_minutes['main_force_buy_vol'].sum()
                    total_vol_on_dip = down_minutes['vol'].sum()
                    if total_vol_on_dip > 0:
                        results['absorption_strength_index'] = mf_buy_on_dip / total_vol_on_dip
        return results

    @staticmethod
    def calculate_game_efficiency_metrics(context: dict) -> dict:
        """
        【V65.0 · 博弈穿透】
        - 核心升维: 全面重构博弈效率指标，引入“冲击成本”和“路径效率”概念。
        - `breakthrough/defense_cost_index` 新增: 基于高频Tick，精确计算多空双方在推动价格时
                     付出的平均“滑点成本”，量化突破与防御的真实代价。
        - `trend_asymmetry_index` 新增: 比较上涨与下跌分钟的“趋势效率”，从“幅度”不对称性
                     的度量，升维至“路径”顺畅度的不对称性分析。
        """
        group = context['group']
        tick_df = context.get('tick_df')
        day_open_qfq = context['day_open_qfq']
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        intraday_thrust_purity = context.get('intraday_thrust_purity')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        stock_code = debug_info.get('stock_code', 'N/A') # 获取 stock_code
        results = {}
        if group.empty:
            return results
        # 1. 升维：趋势不对称指数 (Trend Asymmetry Index) - 分钟级
        up_minutes = group[group['close'] > group['open']]
        down_minutes = group[group['close'] < group['open']]
        if not up_minutes.empty and not down_minutes.empty:
            up_path = (up_minutes['high'] - up_minutes['low']).sum()
            up_net_change = (up_minutes['close'] - up_minutes['open']).sum()
            down_path = (down_minutes['high'] - down_minutes['low']).sum()
            down_net_change = abs(down_minutes['close'] - down_minutes['open']).sum()
            eff_up = up_net_change / up_path if up_path > 0 else 0
            eff_down = down_net_change / down_path if down_path > 0 else 0
            if eff_up > 0 and eff_down > 0:
                results['trend_asymmetry_index'] = np.log(eff_up / eff_down)
        # 2. 保留：推力效能分 (Thrust Efficiency Score)
        if all(pd.notna(v) for v in [day_close_qfq, day_open_qfq, atr_14, intraday_thrust_purity]) and atr_14 > 0:
            price_change_in_atr = (day_close_qfq - day_open_qfq) / atr_14
            effort_factor = 1 - abs(intraday_thrust_purity) + 1e-9
            results['thrust_efficiency_score'] = price_change_in_atr / effort_factor
        # 3. 升维：突破/防御成本指数 (Breakthrough/Defense Cost Index) - 高频
        if tick_df is not None and not tick_df.empty and len(tick_df) > 1 and pd.notna(atr_14) and atr_14 > 0:
            tick_df['prev_price'] = tick_df['price'].shift(1)
            tick_df['price_diff'] = tick_df['price'] - tick_df['prev_price']
            # 突破成本 (多方)
            up_thrust_ticks = tick_df[tick_df['price_diff'] > 0]
            if not up_thrust_ticks.empty:
                total_up_vol = up_thrust_ticks['volume'].sum()
                if total_up_vol > 0:
                    weighted_avg_slippage_up = np.average(up_thrust_ticks['price_diff'], weights=up_thrust_ticks['volume'])
                    results['breakthrough_cost_index'] = weighted_avg_slippage_up / atr_14
            # 防御成本 (空方)
            down_thrust_ticks = tick_df[tick_df['price_diff'] < 0]
            if not down_thrust_ticks.empty:
                total_down_vol = down_thrust_ticks['volume'].sum()
                if total_down_vol > 0:
                    weighted_avg_slippage_down = np.average(abs(down_thrust_ticks['price_diff']), weights=down_thrust_ticks['volume'])
                    results['defense_cost_index'] = weighted_avg_slippage_down / atr_14
        return results

    @staticmethod
    def calculate_gini(array: np.ndarray) -> float:
        """
        【V22.0 · 计算内核静态化】
        计算基尼系数
        - 核心优化: 使用Numba优化后的Gini系数计算函数。
        """
        return _numba_calculate_gini(array)

class ThematicMetricsCalculators:
    """
    【V28.0 · 主题内核归一】
    - 核心职责: 封装所有基于特定市场理论（如市场剖面、行为金融学）的主题指标计算逻辑。
    - 架构模式: 作为一个无状态的静态工具类，提供一系列独立的、按主题划分的计算函数。
    """
    @staticmethod
    def calculate_market_profile_metrics(context: dict) -> dict:
        group = context['group']
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
        day_close_qfq = context['day_close_qfq']
        total_volume_safe = context['total_volume_safe']
        atr_14 = context['atr_14']
        prev_day_metrics = context['prev_day_metrics']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        stock_code = context.get('stock_code', 'N/A') # 从 context 中获取 stock_code
        results = {
            'volume_profile_entropy': np.nan,
            'value_area_migration': np.nan,
            'value_area_overlap_pct': np.nan,
            'closing_acceptance_type': np.nan,
            'equilibrium_compression_index': np.nan,
        }
        today_vpoc = np.nan
        if tick_df is not None and not tick_df.empty and tick_df['volume'].sum() > 0:
            vp_hf = tick_df.groupby('price')['volume'].sum()
            if not vp_hf.empty:
                today_vpoc = vp_hf.idxmax()
                total_volume = tick_df['volume'].sum()
                vp_prob = vp_hf[vp_hf > 0] / total_volume
                entropy = -np.sum(vp_prob * np.log2(vp_prob))
                max_entropy = np.log2(len(vp_prob))
                results['volume_profile_entropy'] = entropy / max_entropy if max_entropy > 0 else 0.0
        if continuous_group['vol'].sum() > 0:
            try:
                bins = pd.cut(continuous_group['close'], bins=20, duplicates='drop')
                vp_minute = continuous_group.groupby(bins)['vol'].sum()
            except ValueError:
                vp_minute = continuous_group.groupby('close')['vol'].sum()
            if pd.isna(today_vpoc) and not vp_minute.empty:
                vpoc_interval = vp_minute.idxmax()
                today_vpoc = vpoc_interval.mid if hasattr(vpoc_interval, 'mid') else vpoc_interval
            vpoc_interval_for_va = vp_minute.idxmax() if not vp_minute.empty else np.nan
            today_vah, today_val = ThematicMetricsCalculators._calculate_value_area(vp_minute, continuous_group['vol'].sum(), vpoc_interval_for_va)
        else:
            today_vah, today_val = np.nan, np.nan
        prev_vpoc, prev_atr = prev_day_metrics.get('vpoc'), prev_day_metrics.get('atr_14d')
        if all(pd.notna(v) for v in [today_vpoc, prev_vpoc, prev_atr]) and prev_atr > 0:
            results['value_area_migration'] = (today_vpoc - prev_vpoc) / prev_atr
        prev_vah, prev_val = prev_day_metrics.get('vah'), prev_day_metrics.get('val')
        if all(pd.notna(v) for v in [today_vah, today_val, prev_vah, prev_val]) and (today_vah - today_val) > 0:
            overlap_width = max(0, min(today_vah, prev_vah) - max(today_val, prev_val))
            results['value_area_overlap_pct'] = (overlap_width / (today_vah - today_val)) * 100
        if all(pd.notna(v) for v in [day_close_qfq, today_vpoc, today_vah, today_val]):
            if day_close_qfq > today_vah: results['closing_acceptance_type'] = 2
            elif day_close_qfq > today_vpoc: results['closing_acceptance_type'] = 1
            elif day_close_qfq < today_val: results['closing_acceptance_type'] = -2
            elif day_close_qfq < today_vpoc: results['closing_acceptance_type'] = -1
            else: results['closing_acceptance_type'] = 0
        prev_high = prev_day_metrics.get('high')
        prev_low = prev_day_metrics.get('low')
        prev_volume = prev_day_metrics.get('volume')
        prev_vpoc = prev_day_metrics.get('vpoc')
        if all(pd.notna(v) for v in [prev_high, prev_low, prev_vpoc, prev_volume, today_vpoc, day_high_qfq, day_low_qfq, total_volume_safe]):
            if day_high_qfq <= prev_high and day_low_qfq >= prev_low:

                prev_range = prev_high - prev_low
                today_range = day_high_qfq - day_low_qfq
                if prev_range > 0 and prev_volume > 0:
                    space_compression = 1 - (today_range / prev_range)
                    positional_balance = 1 - (abs(today_vpoc - prev_vpoc) / prev_range)
                    volume_intensity = np.tanh((total_volume_safe / prev_volume) - 1)
                    score = space_compression * positional_balance * (1 + volume_intensity)
                    results['equilibrium_compression_index'] = score
        results['today_vpoc'] = today_vpoc
        results['today_vah'] = today_vah  
        results['today_val'] = today_val  
        return results

    @staticmethod
    def calculate_forward_looking_metrics(context: dict) -> dict:
        group = context['group']
        continuous_group = context['continuous_group']
        level5_df = context.get('level5_df')
        day_open_qfq = context['day_open_qfq']
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
        day_close_qfq = context['day_close_qfq']
        pre_close_qfq = context['pre_close_qfq']
        atr_5 = context['atr_5']
        atr_14 = context['atr_14']
        atr_50 = context['atr_50']
        intraday_thrust_purity = context.get('intraday_thrust_purity', 0.0)
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        stock_code = context.get('stock_code', 'N/A') # 从 context 中获取 stock_code
        results = {}
        auction_period_df = group[group.index.time >= time(14, 57)]
        if not auction_period_df.empty and not continuous_group.empty:
            close_before_auction_series = continuous_group.loc[continuous_group.index.time < time(14, 57, 0)]['close']
            if not close_before_auction_series.empty:
                close_before_auction = close_before_auction_series.iloc[-1]
                if pd.notna(close_before_auction) and close_before_auction > 0:
                    auction_price_change = (day_close_qfq / close_before_auction - 1) * 100
                    if level5_df is not None and not level5_df.empty:
                        pre_auction_period_df = group[(group.index.time >= time(14, 27)) & (group.index.time < time(14, 57))]
                        avg_vol_pre_auction = pre_auction_period_df['vol'].mean() if not pre_auction_period_df.empty else 0
                        auction_volume = auction_period_df['vol'].sum()
                        volume_surprise_factor = auction_volume / avg_vol_pre_auction if avg_vol_pre_auction > 0 else 1.0
                        last_snapshot_series = level5_df.loc[level5_df.index.time < time(14, 57, 0)]
                        last_snapshot = last_snapshot_series.iloc[-1] if not last_snapshot_series.empty else None
                        pre_auction_tension = 0
                        if last_snapshot is not None:
                            b1_v, a1_v = last_snapshot.get('buy_volume1', 0), last_snapshot.get('sell_volume1', 0)
                            if (b1_v + a1_v) > 0:
                                pre_auction_tension = (b1_v - a1_v) / (b1_v + a1_v)
                        tension_factor = np.exp(pre_auction_tension)
                        results['auction_showdown_score'] = auction_price_change * np.log1p(volume_surprise_factor) * tension_factor
                    else:
                        avg_vol_minute_continuous = continuous_group['vol'].mean()
                        if avg_vol_minute_continuous > 0:
                            auction_volume_multiple = (auction_period_df['vol'].sum() / 3) / avg_vol_minute_continuous
                            results['auction_showdown_score'] = auction_price_change * np.log1p(auction_volume_multiple)
        if all(pd.notna(v) for v in [day_high_qfq, day_low_qfq, day_open_qfq, day_close_qfq, pre_close_qfq, atr_14]) and atr_14 > 0:
            base_shock = (day_close_qfq - pre_close_qfq) / atr_14
            intraday_range = day_high_qfq - day_low_qfq
            if intraday_range > 0:
                path_efficiency_factor = (day_close_qfq - day_open_qfq) / intraday_range
            else:
                path_efficiency_factor = np.sign(day_close_qfq - day_open_qfq) if day_close_qfq != day_open_qfq else 0
            thrust_purity_factor = intraday_thrust_purity if pd.notna(intraday_thrust_purity) else 0
            conviction_weight = (path_efficiency_factor + thrust_purity_factor)
            results['shock_conviction_score'] = base_shock * conviction_weight
        if all(pd.notna(v) for v in [atr_5, atr_50]) and atr_50 > 0:
            results['volatility_expansion_ratio'] = atr_5 / atr_50
        return results

    @staticmethod
    def calculate_battlefield_metrics(context: dict) -> dict:
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        prev_day_metrics = context.get('prev_day_metrics', {})
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        stock_code = context.get('stock_code', 'N/A') # 从 context 中获取 stock_code
        results = {}
        from scipy.stats import linregress # 确保 linregress 被导入
        if not continuous_group.empty and len(continuous_group) > 1 and pd.notna(atr_14) and atr_14 > 0:
            am_session = continuous_group.between_time('09:30', '11:30')
            pm_session = continuous_group.between_time('13:00', '15:00')
            if len(am_session) > 2 and len(pm_session) > 2:
                y_am = am_session['close'].values
                x_am = np.arange(len(y_am))
                slope_am, _, _, _, _ = linregress(x_am, y_am)
                y_pm = pm_session['close'].values
                x_pm = np.arange(len(y_pm))
                slope_pm, _, _, _, _ = linregress(x_pm, y_pm)
                results['trend_acceleration_score'] = (slope_pm - slope_am) / atr_14
        def _calculate_thrust_purity_for_period(period_df: pd.DataFrame, period_ticks: pd.DataFrame | None) -> float:
            if period_ticks is not None and not period_ticks.empty:
                total_vol = period_ticks['volume'].sum()
                if total_vol > 0:
                    if 'price_change' not in period_ticks.columns:
                        period_ticks['price_change'] = period_ticks['price'].diff().fillna(0)
                    net_thrust_vol = (period_ticks['volume'] * np.sign(period_ticks['price_change'])).sum()
                    return net_thrust_vol / total_vol
            if not period_df.empty:
                total_vol = period_df['vol'].sum()
                if total_vol > 0:
                    thrust_vector = (period_df['close'] - period_df['open']) * period_df['vol']
                    absolute_energy = abs(period_df['close'] - period_df['open']) * period_df['vol']
                    total_energy = absolute_energy.sum()
                    if total_energy > 0:
                        return thrust_vector.sum() / total_energy
            return 0.0
        pre_charge_df = continuous_group.between_time('13:30', '14:29')
        final_charge_df = continuous_group.between_time('14:30', '15:00')
        pre_charge_ticks = tick_df.between_time('13:30', '14:29') if tick_df is not None else None
        final_charge_ticks = tick_df.between_time('14:30', '15:00') if tick_df is not None else None
        if not pre_charge_df.empty and not final_charge_df.empty:
            purity_pre = _calculate_thrust_purity_for_period(pre_charge_df, pre_charge_ticks)
            purity_final = _calculate_thrust_purity_for_period(final_charge_df, final_charge_ticks)
            vol_pre = pre_charge_df['vol'].sum()
            vol_final = final_charge_df['vol'].sum()
            if vol_pre > 0:
                vol_ratio = vol_final / vol_pre
                results['final_charge_intensity'] = (purity_final - purity_pre) * np.log1p(vol_ratio)
        open_rhythm_df = continuous_group.between_time('09:30', '10:00')
        mid_rhythm_df = continuous_group.between_time('10:00', '14:30')
        tail_rhythm_df = continuous_group.between_time('14:30', '15:00')
        if not open_rhythm_df.empty and not mid_rhythm_df.empty and not tail_rhythm_df.empty:
            avg_vol_open = open_rhythm_df['vol'].mean()
            avg_vol_mid = mid_rhythm_df['vol'].mean()
            avg_vol_tail = tail_rhythm_df['vol'].mean()
            avg_vol_ends = (avg_vol_open + avg_vol_tail) / 2
            if avg_vol_ends > 1e-9:
                results['volume_structure_skew'] = (avg_vol_mid - avg_vol_ends) / avg_vol_ends
            else:
                results['volume_structure_skew'] = 0.0
        prev_day_high = prev_day_metrics.get('high')
        day_high_qfq = context['day_high_qfq']
        if tick_df is not None and not tick_df.empty and pd.notna(prev_day_high) and pd.notna(atr_14) and atr_14 > 0:
            if day_high_qfq > prev_day_high:
                breakthrough_zone_ticks = tick_df[tick_df['price'] >= prev_day_high].copy()
                if not breakthrough_zone_ticks.empty:
                    total_breakthrough_vol = breakthrough_zone_ticks['volume'].sum()
                    if total_breakthrough_vol > 0:
                        breakthrough_zone_ticks['price_change'] = breakthrough_zone_ticks['price'].diff().fillna(0)
                        net_thrust_vol = (breakthrough_zone_ticks['volume'] * np.sign(breakthrough_zone_ticks['price_change'])).sum()
                        breakthrough_thrust_purity = net_thrust_vol / total_breakthrough_vol
                        confirmation_raw = (day_close_qfq - prev_day_high) / atr_14
                        confirmation_factor = np.tanh(confirmation_raw)
                        score = confirmation_factor * (1 + breakthrough_thrust_purity)
                        results['breakthrough_conviction_score'] = score
        prev_day_low = prev_day_metrics.get('low')
        day_low_qfq = context['day_low_qfq']
        if tick_df is not None and not tick_df.empty and pd.notna(prev_day_low) and pd.notna(atr_14) and atr_14 > 0:
            if day_low_qfq < prev_day_low:
                defense_zone_ticks = tick_df[tick_df['price'] <= prev_day_low].copy()
                if not defense_zone_ticks.empty:
                    total_defense_vol = defense_zone_ticks['volume'].sum()
                    if total_defense_vol > 0:
                        defense_zone_ticks['price_change'] = defense_zone_ticks['price'].diff().fillna(0)
                        net_thrust_vol = (defense_zone_ticks['volume'] * np.sign(defense_zone_ticks['price_change'])).sum()
                        defense_thrust_purity = net_thrust_vol / total_defense_vol
                        rejection_raw = (day_close_qfq - prev_day_low) / atr_14
                        rejection_factor = np.tanh(rejection_raw)
                        score = rejection_factor * (1 + defense_thrust_purity)
                        results['defense_solidity_score'] = score
        prev_high = prev_day_metrics.get('high')
        prev_low = prev_day_metrics.get('low')
        prev_volume = prev_day_metrics.get('volume')
        prev_vpoc = prev_day_metrics.get('vpoc')
        today_vpoc = context.get('_today_vpoc')
        total_volume_safe = context['total_volume_safe']
        if all(pd.notna(v) for v in [prev_high, prev_low, prev_vpoc, prev_volume, today_vpoc]):
            if day_high_qfq <= prev_high and day_low_qfq >= prev_low:
                prev_range = prev_high - prev_low
                today_range = day_high_qfq - day_low_qfq
                if prev_range > 0 and prev_volume > 0:
                    space_compression = 1 - (today_range / prev_range)
                    positional_balance = 1 - (abs(today_vpoc - prev_vpoc) / prev_range)
                    volume_intensity = np.tanh((total_volume_safe / prev_volume) - 1)
                    score = space_compression * positional_balance * (1 + volume_intensity)
                    results['equilibrium_compression_index'] = score
        return results

    @staticmethod
    def _calculate_value_area(vp: pd.Series, total_volume: float, vpoc_interval: pd.Interval) -> tuple:
        """计算日内价值区域 (VAH/VAL)"""
        if vp.empty or total_volume == 0 or pd.isna(vpoc_interval):
            return np.nan, np.nan
        value_area_target_volume = total_volume * 0.7
        vp_sorted_by_price = vp.sort_index()
        try:
            poc_idx = vp_sorted_by_price.index.get_loc(vpoc_interval)
        except KeyError:
            return np.nan, np.nan
        current_volume = vp_sorted_by_price.iloc[poc_idx]
        low_idx, high_idx = poc_idx, poc_idx
        while current_volume < value_area_target_volume and (low_idx > 0 or high_idx < len(vp_sorted_by_price) - 1):
            vol_above = vp_sorted_by_price.iloc[high_idx + 1] if high_idx < len(vp_sorted_by_price) - 1 else -1
            vol_below = vp_sorted_by_price.iloc[low_idx - 1] if low_idx > 0 else -1
            if vol_above > vol_below:
                high_idx += 1
                current_volume += vol_above
            else:
                low_idx -= 1
                current_volume += vol_below
        val = vp_sorted_by_price.index[low_idx].left
        vah = vp_sorted_by_price.index[high_idx].right
        return vah, val

class MicrostructureDynamicsCalculators:
    """
    【V27.0 · 微观动力学内核】
    - 核心职责: 封装所有基于高频数据的微观结构指标计算逻辑，实现最终的内核分离。
    - 架构模式: 作为一个无状态的静态工具类，提供一系列独立的、可诊断的计算函数。
    """
    @staticmethod
    def calculate_all(context: dict) -> dict:
        """主入口：编排所有微观动力学指标的计算"""
        results = {}
        results.update(MicrostructureDynamicsCalculators._calculate_ofi_and_sweeps(context))
        results.update(MicrostructureDynamicsCalculators._calculate_vpin(context))
        results.update(MicrostructureDynamicsCalculators._calculate_hf_mechanics(context))
        results.update(MicrostructureDynamicsCalculators._calculate_liquidity_metrics(context))
        results.update(MicrostructureDynamicsCalculators._calculate_vwap_reversion(context))
        return results

    @staticmethod
    def _calculate_ofi_and_sweeps(context: dict) -> dict:
        """计算订单流失衡(OFI)与扫单强度"""
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        total_volume = context.get('total_volume_safe')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'order_flow_imbalance_score': np.nan,
            'buy_sweep_intensity': np.nan,
            'sell_sweep_intensity': np.nan,
        }
        if tick_df is None or tick_df.empty or total_volume == 0:
            return results
        # 订单流失衡 (OFI)
        if level5_df is not None and not level5_df.empty and len(level5_df) > 1:
            df = level5_df[['buy_price1', 'buy_volume1', 'sell_price1', 'sell_volume1']].copy()
            df_prev = df.shift(1).fillna(0) # 填充NaN以避免Numba处理NaN
            # 提取NumPy数组
            buy_price1_arr = df['buy_price1'].values
            buy_volume1_arr = df['buy_volume1'].values
            sell_price1_arr = df['sell_price1'].values
            sell_volume1_arr = df['sell_volume1'].values
            prev_buy_price1_arr = df_prev['buy_price1'].values
            prev_buy_volume1_arr = df_prev['buy_volume1'].values
            prev_sell_price1_arr = df_prev['sell_price1'].values
            prev_sell_volume1_arr = df_prev['sell_volume1'].values
            # 调用Numba优化函数
            ofi_series_numba = _numba_calculate_ofi_static_dynamic(
                buy_price1_arr, buy_volume1_arr,
                sell_price1_arr, sell_volume1_arr,
                prev_buy_price1_arr, prev_buy_volume1_arr,
                prev_sell_price1_arr, prev_sell_volume1_arr
            )
            total_ofi = np.nansum(ofi_series_numba)
            if total_volume > 0:
                results['order_flow_imbalance_score'] = total_ofi / total_volume
        # 扫单强度 (Sweep Intensity)
        buy_sweep_vol, sell_sweep_vol = 0, 0
        min_sweep_len = 3
        tick_df['block'] = (tick_df['type'] != tick_df['type'].shift()).cumsum()
        tick_df['block_size'] = tick_df.groupby('block')['type'].transform('size')
        sweep_candidates = tick_df[(tick_df['block_size'] >= min_sweep_len) & (tick_df['type'].isin(['B', 'S']))]
        if not sweep_candidates.empty:
            for _, group_sweep in sweep_candidates.groupby('block'):
                trade_type = group_sweep['type'].iloc[0]
                prices = group_sweep['price']
                if trade_type == 'B' and prices.is_monotonic_increasing:
                    buy_sweep_vol += group_sweep['volume'].sum()
                elif trade_type == 'S' and prices.is_monotonic_decreasing:
                    sell_sweep_vol += group_sweep['volume'].sum()
        total_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
        total_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
        if total_buy_vol > 0:
            results['buy_sweep_intensity'] = buy_sweep_vol / total_buy_vol
        if total_sell_vol > 0:
            results['sell_sweep_intensity'] = sell_sweep_vol / total_sell_vol
        return results

    @staticmethod
    def _calculate_vpin(context: dict) -> dict:
        """计算VPIN (Volume-Synchronized Probability of Informed Trading)"""
        tick_df = context.get('tick_df')
        total_volume = context.get('total_volume_safe')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {'vpin_score': np.nan}
        if tick_df is None or tick_df.empty or total_volume == 0:
            return results
        vpin_bucket_size = total_volume / 50
        vpin_window = 10
        if vpin_bucket_size > 0:
            tick_df['buy_vol'] = np.where(tick_df['type'] == 'B', tick_df['volume'], 0)
            tick_df['sell_vol'] = np.where(tick_df['type'] == 'S', tick_df['volume'], 0)
            tick_df['cum_vol'] = tick_df['volume'].cumsum()
            # 提取NumPy数组
            cum_vol_arr = tick_df['cum_vol'].values
            volume_arr = tick_df['volume'].values
            buy_vol_arr = tick_df['buy_vol'].values
            sell_vol_arr = tick_df['sell_vol'].values
            # 调用Numba优化函数
            imbalance_values, bucket_indices = _numba_calculate_vpin_buckets(
                cum_vol_arr, volume_arr, buy_vol_arr, sell_vol_arr, vpin_bucket_size
            )
            if len(imbalance_values) > vpin_window:
                # 将Numba结果转换回Pandas Series进行后续滚动计算
                bucket_imbalance_series = pd.Series(imbalance_values, index=bucket_indices)
                imbalance_std = bucket_imbalance_series.rolling(window=vpin_window).std().bfill()
                abs_imbalance = bucket_imbalance_series.abs()
                sigma_imbalance = imbalance_std.replace(0, np.nan)
                z_score = abs_imbalance / sigma_imbalance
                vpin_series = z_score.apply(lambda z: norm.cdf(z) if pd.notna(z) else np.nan)
                results['vpin_score'] = vpin_series.mean()
        return results

    @staticmethod
    def _calculate_hf_mechanics(context: dict) -> dict:
        """
        【V61.0 · 博弈精研】
        - `active_volume_price_efficiency` 升维: 逻辑彻底重构。不再计算静态的“终局”比值，
                     而是通过计算日内“累计推力”与“累计价格位移”两条曲线的相关系数，
                     来动态追溯推力的“过程有效性”，洞察主力资金的控盘合力。
        - 核心优化: 使用Numba优化后的相关性计算函数。
        """
        tick_df = context.get('tick_df')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'active_volume_price_efficiency': np.nan,
        }
        if tick_df is None or tick_df.empty or len(tick_df) < 2:
            return results
        # 1. 计算每笔tick的有效推力
        price_arr = tick_df['price'].values
        volume_arr = tick_df['volume'].values
        # 确保 price_change 存在且有效
        price_change_arr = np.zeros_like(price_arr, dtype=np.float64)
        if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
            self_calculated_change = np.diff(price_arr, prepend=price_arr[0]) # 计算实际价格变化
            zero_change_mask = (tick_df['price_change'].values == 0)
            price_change_arr = np.where(zero_change_mask, self_calculated_change, tick_df['price_change'].values)
        else: # 回退逻辑，直接使用价格变化
            price_change_arr = np.diff(price_arr, prepend=price_arr[0])
        # 调用Numba优化函数
        correlation = _numba_calculate_active_volume_price_efficiency(
            price_arr, volume_arr, price_change_arr
        )
        results['active_volume_price_efficiency'] = correlation
        return results

    @staticmethod
    def _calculate_liquidity_metrics(context: dict) -> dict:
        """
        【V70.0 · 流动性验真】
        - 核心升维: 彻底重构 `liquidity_authenticity_score`。不再依赖静态盘口形态，而是
                     引入“流动性承诺-兑现”动态追踪模型。通过识别盘口异常大额挂单，并追踪
                     其在价格压力下的最终结局（真实成交或提前撤单），深度量化挂单的“诚意”，
                     从而辨别“铁壁”与“幻象”。
        - 核心优化: 使用Numba优化后的流动性承诺-兑现分数计算函数。
        """
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        realtime_df = context.get('realtime_df')
        daily_series_for_day = context.get('daily_series_for_day')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'market_impact_cost': np.nan,
            'liquidity_slope': np.nan,
            'liquidity_authenticity_score': np.nan,
        }
        if tick_df is None or tick_df.empty or level5_df is None or level5_df.empty or len(level5_df) < 2:
            return results
        # --- 保留 market_impact_cost 和 liquidity_slope 的计算逻辑 ---
        column_rename_map = {**{f'buy_price{i}': f'b{i}_p' for i in range(1, 6)}, **{f'buy_volume{i}': f'b{i}_v' for i in range(1, 6)}, **{f'sell_price{i}': f'a{i}_p' for i in range(1, 6)}, **{f'sell_volume{i}': f'a{i}_v' for i in range(1, 6)}}
        level5_df_renamed = level5_df.copy().rename(columns=column_rename_map)
        if realtime_df is not None and not realtime_df.empty:
            snapshot_df = pd.merge_asof(realtime_df.sort_index(), level5_df_renamed.sort_index(), on='trade_time', direction='backward')
            snapshot_df['snapshot_volume'] = snapshot_df['volume'].diff().fillna(0).clip(lower=0)
            total_amount = daily_series_for_day.get('amount', 0)
            if total_amount > 0:
                standard_amount = float(total_amount) * 0.001
                impact_costs, weights_for_costs = [], []
                slopes, weights_for_slopes = [], []
                for _, row in snapshot_df.iterrows():
                    snapshot_volume = row['snapshot_volume']
                    if snapshot_volume <= 0: continue
                    amount_to_fill, filled_amount, filled_volume = standard_amount, 0, 0
                    for i in range(1, 6):
                        price, vol = row.get(f'a{i}_p'), row.get(f'a{i}_v', 0) * 100
                        if pd.isna(price): continue
                        value = float(price) * vol
                        if amount_to_fill > value:
                            filled_amount += value; filled_volume += vol; amount_to_fill -= value
                        else:
                            filled_volume += amount_to_fill / float(price); filled_amount += amount_to_fill; break
                    if filled_volume > 0:
                        mid_price = (row.get('b1_p', 0) + row.get('a1_p', 0)) / 2
                        if mid_price > 0:
                            cost = ((filled_amount / filled_volume) / float(mid_price) - 1) * 100
                            impact_costs.append(cost)
                            weights_for_costs.append(snapshot_volume)
                    mid_price = (row.get('b1_p', 0) + row.get('a1_p', 0)) / 2
                    if mid_price > 0:
                        ask_x = [(float(row.get(f'a{i}_p', mid_price)) - mid_price) / mid_price for i in range(1, 6)]
                        ask_y = np.cumsum([row.get(f'a{i}_v', 0) * 100 for i in range(1, 6)])
                        if np.std(ask_x) > 0:
                            slope = linregress(ask_x, ask_y).slope
                            slopes.append(slope)
                            weights_for_slopes.append(snapshot_volume)
                if impact_costs and sum(weights_for_costs) > 0:
                    results['market_impact_cost'] = np.average(impact_costs, weights=weights_for_costs)
                if slopes and sum(weights_for_slopes) > 0:
                    results['liquidity_slope'] = np.average(slopes, weights=weights_for_slopes)
        # --- 新增 `liquidity_authenticity_score` 的升维计算逻辑 ---
        # 提取NumPy数组
        buy_price1_arr = level5_df['buy_price1'].values
        buy_volume1_arr = level5_df['buy_volume1'].values
        sell_price1_arr = level5_df['sell_price1'].values
        sell_volume1_arr = level5_df['sell_volume1'].values
        tick_prices_arr = tick_df['price'].values
        tick_times_arr = tick_df.index.values.astype(np.int64) # 将Timestamp转换为int64
        level5_times_arr = level5_df.index.values.astype(np.int64) # 将Timestamp转换为int64
        b1_vol_mean, b1_vol_std = buy_volume1_arr.mean(), buy_volume1_arr.std()
        a1_vol_mean, a1_vol_std = sell_volume1_arr.mean(), sell_volume1_arr.std()
        buy_commitment_threshold = b1_vol_mean + 2 * b1_vol_std
        sell_commitment_threshold = a1_vol_mean + 2 * a1_vol_std
        # 调用Numba优化函数
        fulfillments, defaults = _numba_calculate_liquidity_authenticity_score(
            buy_price1_arr, buy_volume1_arr,
            sell_price1_arr, sell_volume1_arr,
            tick_prices_arr, tick_times_arr,
            level5_times_arr,
            buy_commitment_threshold, sell_commitment_threshold
        )
        total_events = fulfillments + defaults
        if total_events > 0:
            results['liquidity_authenticity_score'] = fulfillments / total_events
        else:
            results['liquidity_authenticity_score'] = 0.5 # 无事件发生，给予中性分
        return results

    @staticmethod
    def _calculate_vwap_reversion(context: dict) -> dict:
        """计算VWAP均值回归相关性"""
        minute_df = context.get('continuous_group')
        results = {'vwap_mean_reversion_corr': np.nan}
        if minute_df is not None and not minute_df.empty and 'minute_vwap' in minute_df.columns and len(minute_df) > 1:
            daily_vwap = (minute_df['amount'].sum() / minute_df['vol'].sum()) if minute_df['vol'].sum() > 0 else np.nan
            if pd.notna(daily_vwap):
                deviation = minute_df['minute_vwap'] - daily_vwap
                # 提取NumPy数组
                deviation_arr = deviation.values
                # 调用Numba优化函数
                results['vwap_mean_reversion_corr'] = _numba_calculate_vwap_reversion_corr(deviation_arr)
        return results






