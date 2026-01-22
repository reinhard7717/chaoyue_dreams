# services\advanced_structural_metrics_service.py
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from scipy.stats import norm, linregress
import numba
from numba import jit, prange, float64, int64
from typing import Tuple
from asgiref.sync import sync_to_async
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale
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
                    # 使用范围查询
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
                    'vpoc': prev_hist_series.get('today_vpoc'), # 使用正确的字段名 today_vpoc
                    'vah': prev_hist_series.get('today_vah'),   # 使用正确的字段名 today_vah
                    'val': prev_hist_series.get('today_val'),   # 使用正确的字段名 today_val
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
                'vpoc': day_metric_dict.get('today_vpoc'), # 使用正确的字段名 today_vpoc
                'vah': day_metric_dict.get('today_vah'),   # 使用正确的字段名 today_vah
                'val': day_metric_dict.get('today_val'),   # 使用正确的字段名 today_val
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

@jit(nopython=True, cache=True)
def _numba_calculate_thrust_purity(prices, volumes, price_changes=None, trade_types=None):
    """使用numba优化的推力纯度计算核心函数"""
    n = len(prices)
    if n == 0:
        return np.nan
    
    total_volume = np.sum(volumes)
    if total_volume <= 0:
        return np.nan
    
    if price_changes is not None and np.any(~np.isnan(price_changes)):
        # 如果有价格变化数据
        net_thrust = 0.0
        for i in prange(n):
            price_change = price_changes[i]
            if np.isnan(price_change) or price_change == 0:
                # 计算实际价格变化
                if i > 0:
                    actual_change = prices[i] - prices[i-1]
                else:
                    actual_change = 0
            else:
                actual_change = price_change
            net_thrust += volumes[i] * np.sign(actual_change)
        return net_thrust / total_volume
    elif trade_types is not None:
        # 使用买卖方向数据
        buy_volume = 0.0
        sell_volume = 0.0
        for i in prange(n):
            if trade_types[i] == 'B':  # 注意：numba需要数值类型
                buy_volume += volumes[i]
            elif trade_types[i] == 'S':
                sell_volume += volumes[i]
        return (buy_volume - sell_volume) / total_volume
    else:
        # 使用价格变化向量
        net_thrust = 0.0
        for i in prange(n):
            if i > 0:
                price_change = prices[i] - prices[i-1]
                net_thrust += volumes[i] * np.sign(price_change)
        return net_thrust / total_volume

@jit(nopython=True, cache=True)
def _numba_calculate_burstiness_index(volumes):
    """使用numba优化的成交量爆发性指数计算"""
    n = len(volumes)
    if n <= 1:
        return 0.0
    
    mean_volume = np.mean(volumes)
    if mean_volume <= 0:
        return 0.0
    
    # 计算变异系数
    std_volume = np.std(volumes)
    cv = std_volume / mean_volume
    
    # 计算峰度系数
    if n >= 4 and std_volume > 0:
        deviations = volumes - mean_volume
        m4 = np.mean(deviations**4)
        m2 = np.mean(deviations**2)
        kurtosis = m4 / (m2**2) - 3
    else:
        kurtosis = 0
    
    # 计算爆发比例
    burst_mask = volumes > 2 * mean_volume
    burst_ratio = np.sum(burst_mask) / n if n > 0 else 0
    
    # 计算爆发强度
    if burst_ratio > 0:
        burst_volumes = volumes[burst_mask]
        burst_intensity = np.mean(burst_volumes) / mean_volume
    else:
        burst_intensity = 1.0
    
    # 综合计算
    result = 0.4 * np.log1p(cv) + \
             0.3 * np.log1p(max(0, kurtosis)) + \
             0.2 * burst_ratio + \
             0.1 * np.log1p(burst_intensity)
    
    return np.tanh(result)

@jit(nopython=True, cache=True)
def _numba_calculate_ofi(mid_prices, buy_volumes, sell_volumes, bid_ask_spreads):
    """使用numba优化的订单流不平衡计算"""
    n = len(mid_prices)
    ofi_sum = 0.0
    for i in prange(1, n):
        if mid_prices[i] > mid_prices[i-1]:
            # 价格上涨，买方压力
            spread = bid_ask_spreads[i-1]
            if spread > 0:
                ofi_sum += buy_volumes[i-1] / spread
        elif mid_prices[i] < mid_prices[i-1]:
            # 价格下跌，卖方压力
            spread = bid_ask_spreads[i-1]
            if spread > 0:
                ofi_sum -= sell_volumes[i-1] / spread
    return ofi_sum

@jit(nopython=True, cache=True)
def _numba_find_reversals(highs, lows, closes, volumes, amounts, atr_14):
    """使用numba优化的反转检测和计算函数"""
    n = len(highs)
    if n < 10:  # 数据太少
        return 0.0, 0.0, 0.0
    
    # 使用简化的极值点检测
    peaks = []
    troughs = []
    
    # 简单峰值检测（可考虑更复杂的算法）
    for i in range(5, n-5):
        if highs[i] >= np.max(highs[i-5:i+5]):
            peaks.append(i)
        if lows[i] <= np.min(lows[i-5:i+5]):
            troughs.append(i)
    
    if len(peaks) < 2 or len(troughs) < 2:
        return 0.0, 0.0, 0.0
    
    # 合并排序 - 修复concatenate参数类型问题，使用元组而非列表
    peaks_arr = np.array(peaks)
    troughs_arr = np.array(troughs)
    all_extrema = np.sort(np.concatenate((peaks_arr, troughs_arr)))
    
    total_momentum = 0.0
    positive_count = 0
    recovery_sum = 0.0
    valid_reversals = 0
    
    for i in range(1, len(all_extrema)-1):
        current = all_extrema[i]
        next_ext = all_extrema[i+1]
        # 检查是否为低点-高点模式
        # 优化：避免使用 'in' 操作符和列表推导式，改用显式循环
        is_trough = False
        for t in troughs:
            if t == current:
                is_trough = True
                break
        if not is_trough:
            continue
        is_next_peak = False
        for p in peaks:
            if p == next_ext:
                is_next_peak = True
                break
        if is_next_peak:
            # 找前一个高点
            prev_peak = -1
            for p in peaks:
                if p < current:
                    if p > prev_peak:
                        prev_peak = p
            if prev_peak == -1:
                continue
            # 计算下跌阶段
            fall_start = prev_peak
            fall_end = current
            fall_length = fall_end - fall_start
            if fall_length < 3:
                continue
            # 计算反弹阶段
            rebound_start = current
            rebound_end = next_ext
            rebound_length = rebound_end - rebound_start
            if rebound_length < 3:
                continue
            # 计算阶段VWAP和成交量
            fall_vol = np.sum(volumes[fall_start:fall_end+1])
            rebound_vol = np.sum(volumes[rebound_start:rebound_end+1])
            if fall_vol == 0 or rebound_vol == 0:
                continue
            # 计算价格变动
            price_change_fall = highs[fall_start] - lows[fall_end]
            price_change_rebound = highs[rebound_end] - lows[rebound_start]
            if price_change_fall <= 0:
                continue
            # 计算恢复率
            recovery_ratio = price_change_rebound / price_change_fall
            # 计算成交量比率
            volume_ratio = np.log1p(rebound_vol / fall_vol)
            # 计算综合动量
            momentum = recovery_ratio * np.tanh(volume_ratio) * 100
            total_momentum += momentum
            valid_reversals += 1
            if momentum > 0:
                positive_count += 1
                recovery_sum += recovery_ratio
    if valid_reversals == 0:
        return 0.0, 0.0, 0.0
    # 计算最终指标
    avg_momentum = total_momentum / valid_reversals
    conviction_rate = positive_count / valid_reversals if valid_reversals > 0 else 0.0
    recovery_rate = recovery_sum / positive_count if positive_count > 0 else 0.0
    
    return avg_momentum * conviction_rate, conviction_rate, recovery_rate

@jit(nopython=True, cache=True)
def _numba_calculate_high_level_volume(prices, volumes, high_threshold, close_price, atr_14):
    """使用numba优化的高位成交量计算"""
    n = len(prices)
    if n == 0:
        return 0.0
    
    total_volume = np.sum(volumes)
    if total_volume <= 0:
        return 0.0
    
    # 计算高位成交量
    high_vol = 0.0
    for i in prange(n):
        if prices[i] >= high_threshold:
            high_vol += volumes[i]
    
    volume_ratio = high_vol / total_volume
    
    # 计算确认因子
    distance = close_price - high_threshold
    if atr_14 > 0:
        normalized_distance = distance / atr_14
    else:
        normalized_distance = distance / (np.max(prices) - np.min(prices) + 1e-6)
    
    confirmation_factor = np.tanh(normalized_distance)
    
    return volume_ratio * confirmation_factor

@jit(nopython=True, cache=True)
def _numba_calculate_opening_thrust(opening_prices, opening_volumes, price_changes=None, trade_types=None):
    """使用numba优化的开盘推力计算"""
    return _numba_calculate_thrust_purity(
        opening_prices, opening_volumes, price_changes, trade_types
    )

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
        # 1. 日内能量密度（向量化）
        if pd.notna(atr_14) and atr_14 > 0:
            turnover_rate_f = context.get('turnover_rate_f')
            if pd.notna(turnover_rate_f):
                results['intraday_energy_density'] = np.log1p(turnover_rate_f) / atr_14
        # 2. 日内推力纯度（向量化 + numba优化）
        if tick_df is not None and not tick_df.empty:
            # 准备数据为numpy数组
            tick_prices = tick_df['price'].values.astype(np.float64)
            tick_volumes = tick_df['volume'].values.astype(np.float64)
            # 根据可用数据选择计算方式
            if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
                price_changes = tick_df['price_change'].values.astype(np.float64)
                results['intraday_thrust_purity'] = _numba_calculate_thrust_purity(
                    tick_prices, tick_volumes, price_changes, None
                )
            elif 'type' in tick_df.columns:
                # 转换为数值类型（B=1, S=-1, 其他=0）
                trade_types_numeric = np.zeros(len(tick_df), dtype=np.int8)
                for i, val in enumerate(tick_df['type'].values):
                    if val == 'B':
                        trade_types_numeric[i] = 1
                    elif val == 'S':
                        trade_types_numeric[i] = -1
                # 使用买卖方向计算
                buy_mask = trade_types_numeric == 1
                sell_mask = trade_types_numeric == -1
                if np.any(buy_mask) or np.any(sell_mask):
                    buy_vol = np.sum(tick_volumes[buy_mask])
                    sell_vol = np.sum(tick_volumes[sell_mask])
                    total_vol = np.sum(tick_volumes)
                    if total_vol > 0:
                        results['intraday_thrust_purity'] = (buy_vol - sell_vol) / total_vol
            else:
                results['intraday_thrust_purity'] = _numba_calculate_thrust_purity(
                    tick_prices, tick_volumes, None, None
                )
        else:
            # 使用分钟数据
            group_prices = group['close'].values.astype(np.float64)
            group_volumes = group['vol'].values.astype(np.float64)
            results['intraday_thrust_purity'] = _numba_calculate_thrust_purity(
                group_prices, group_volumes, None, None
            )
        # 3. 成交量爆发性指数（向量化 + numba优化）
        if tick_df is not None and not tick_df.empty:
            tick_volumes = tick_df['volume'].values.astype(np.float64)
            results['volume_burstiness_index'] = _numba_calculate_burstiness_index(tick_volumes)
        else:
            group_volumes = group['vol'].values.astype(np.float64)
            results['volume_burstiness_index'] = _numba_calculate_burstiness_index(group_volumes)
        # 4. 集合竞价影响分数（向量化 + numba优化）
        if all(pd.notna(v) for v in [day_open_qfq, pre_close_qfq, atr_14]) and atr_14 > 0:
            gap_magnitude = (day_open_qfq - pre_close_qfq) / atr_14
            if tick_df is not None and not tick_df.empty and level5_df is not None and not level5_df.empty:
                # 准备数据
                auction_ticks = tick_df[tick_df.index.time < time(9, 35)]
                auction_level5 = level5_df[level5_df.index.time < time(9, 35)]
                if not auction_ticks.empty and not auction_level5.empty:
                    # 合并数据（优化版：使用向量化合并）
                    merged_idx = pd.merge_asof(
                        auction_ticks.sort_index(),
                        auction_level5.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction='backward',
                        tolerance=pd.Timedelta('2s')
                    )
                    if not merged_idx.empty:
                        # 提取numpy数组
                        buy_price1 = merged_idx['buy_price1'].values.astype(np.float64)
                        sell_price1 = merged_idx['sell_price1'].values.astype(np.float64)
                        buy_volume1 = merged_idx['buy_volume1'].values.astype(np.float64)
                        sell_volume1 = merged_idx['sell_volume1'].values.astype(np.float64)
                        # 计算中间价和价差
                        mid_prices = (buy_price1 + sell_price1) / 2
                        bid_ask_spreads = sell_price1 - buy_price1
                        # 使用numba计算OFI
                        opening_ofi = _numba_calculate_ofi(
                            mid_prices, buy_volume1, sell_volume1, bid_ask_spreads
                        )
                        opening_volume = np.sum(merged_idx['volume'].values.astype(np.float64))
                        if opening_volume > 0:
                            conviction_factor = np.tanh(opening_ofi / opening_volume)
                            results['auction_impact_score'] = gap_magnitude * (1 + conviction_factor * np.sign(gap_magnitude))
                        else:
                            results['auction_impact_score'] = gap_magnitude
                    else:
                        results['auction_impact_score'] = gap_magnitude
                else:
                    results['auction_impact_score'] = gap_magnitude
            else:
                results['auction_impact_score'] = gap_magnitude
        # 5. 动态反转强度、反转确信率、反转恢复率（numba优化）
        try:
            # 准备数据为numpy数组
            group_highs = group['high'].values.astype(np.float64)
            group_lows = group['low'].values.astype(np.float64)
            group_closes = group['close'].values.astype(np.float64)
            group_volumes = group['vol'].values.astype(np.float64)
            group_amounts = group['amount'].values.astype(np.float64)
            # 使用numba优化的反转计算
            if pd.notna(atr_14) and atr_14 > 0:
                strength, conviction, recovery = _numba_find_reversals(
                    group_highs, group_lows, group_closes, group_volumes, group_amounts, atr_14
                )
                results['dynamic_reversal_strength'] = strength
                results['reversal_conviction_rate'] = conviction
                results['reversal_recovery_rate'] = recovery
        except Exception as e:
            if enable_probe:
                print(f"反转指标计算异常: {e}")
            results['dynamic_reversal_strength'] = np.nan
            results['reversal_conviction_rate'] = np.nan
            results['reversal_recovery_rate'] = np.nan
        # 6. 高位整理成交量（numba优化）
        price_range = day_high_qfq - day_low_qfq
        if price_range > 0 and pd.notna(atr_14) and atr_14 > 0:
            high_level_threshold = day_high_qfq - 0.25 * price_range
            if tick_df is not None and not tick_df.empty:
                # 使用tick数据
                tick_prices = tick_df['price'].values.astype(np.float64)
                tick_volumes = tick_df['volume'].values.astype(np.float64)
                results['high_level_consolidation_volume'] = _numba_calculate_high_level_volume(
                    tick_prices, tick_volumes, high_level_threshold, day_close_qfq, atr_14
                )
            else:
                # 使用分钟数据
                group_highs = group['high'].values.astype(np.float64)
                group_volumes = group['vol'].values.astype(np.float64)
                results['high_level_consolidation_volume'] = _numba_calculate_high_level_volume(
                    group_highs, group_volumes, high_level_threshold, day_close_qfq, atr_14
                )
        # 7. 开盘期间推力（numba优化）
        if tick_df is not None and not tick_df.empty:
            # 提取开盘期间数据
            opening_mask = (tick_df.index.time >= time(9, 30)) & (tick_df.index.time <= time(9, 59, 59))
            opening_ticks = tick_df[opening_mask]
            if not opening_ticks.empty:
                # 准备数据
                opening_prices = opening_ticks['price'].values.astype(np.float64)
                opening_volumes = opening_ticks['volume'].values.astype(np.float64)
                # 根据可用数据选择计算方式
                if 'price_change' in opening_ticks.columns:
                    price_changes = opening_ticks['price_change'].values.astype(np.float64)
                    results['opening_period_thrust'] = _numba_calculate_opening_thrust(
                        opening_prices, opening_volumes, price_changes, None
                    )
                elif 'type' in opening_ticks.columns:
                    # 使用买卖方向计算
                    buy_mask = opening_ticks['type'] == 'B'
                    sell_mask = opening_ticks['type'] == 'S'
                    opening_total_vol = np.sum(opening_volumes)
                    if opening_total_vol > 0:
                        buy_vol = np.sum(opening_volumes[buy_mask])
                        sell_vol = np.sum(opening_volumes[sell_mask])
                        results['opening_period_thrust'] = (buy_vol - sell_vol) / opening_total_vol
                else:
                    results['opening_period_thrust'] = _numba_calculate_opening_thrust(
                        opening_prices, opening_volumes, None, None
                    )
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

@jit(nopython=True, cache=True)
def _numba_calculate_vpin_buckets(cum_vol_arr, volume_arr, buy_vol_arr, sell_vol_arr, bucket_size):
    """Numba优化的VPIN桶计算核心逻辑"""
    num_ticks = len(cum_vol_arr)
    imbalance_values = []
    bucket_indices = []
    current_bucket_vol = 0.0
    current_bucket_buy = 0.0
    current_bucket_sell = 0.0
    bucket_counter = 0
    
    for i in range(num_ticks):
        current_bucket_vol += volume_arr[i]
        current_bucket_buy += buy_vol_arr[i]
        current_bucket_sell += sell_vol_arr[i]
        if current_bucket_vol >= bucket_size:
            if current_bucket_vol > 0:
                imbalance = abs(current_bucket_buy - current_bucket_sell) / current_bucket_vol
            else:
                imbalance = 0.0
            imbalance_values.append(imbalance)
            bucket_indices.append(bucket_counter)
            bucket_counter += 1
            overflow_vol = current_bucket_vol - bucket_size
            if overflow_vol > 0:
                ratio = overflow_vol / current_bucket_vol
                current_bucket_buy = buy_vol_arr[i] * ratio
                current_bucket_sell = sell_vol_arr[i] * ratio
                current_bucket_vol = overflow_vol
            else:
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                current_bucket_vol = 0.0
    
    if current_bucket_vol > 0 and current_bucket_vol >= bucket_size * 0.5:
        if current_bucket_vol > 0:
            imbalance = abs(current_bucket_buy - current_bucket_sell) / current_bucket_vol
        else:
            imbalance = 0.0
        imbalance_values.append(imbalance)
        bucket_indices.append(bucket_counter)
    
    return np.array(imbalance_values), np.array(bucket_indices)

@jit(cache=True, fastmath=True)
def _numba_calculate_active_volume_price_efficiency(
    price_arr: np.ndarray,
    volume_arr: np.ndarray,
    price_change_arr: np.ndarray,
    time_seconds_arr: np.ndarray = None,
    volume_threshold_ratio: float = 0.001
) -> float:
    """
    【V62.0 · 博弈精研 - 高精度版】
    核心逻辑重构：从简单的相关系数升级为加权动态有效性评估
    1. 引入成交量权重：大单推力权重更高，小单噪声权重降低
    2. 时间衰减修正：早盘推力和尾盘推力具有不同时效性
    3. 方向一致性检验：避免价格震荡导致的假性相关
    4. 异常值鲁棒处理：使用中位数替代均值进行中心化
    
    参数说明：
    price_arr: 价格序列
    volume_arr: 成交量序列
    price_change_arr: 价格变化序列（带符号）
    time_seconds_arr: 时间戳（秒），用于时间衰减计算
    volume_threshold_ratio: 成交量阈值比例，过滤噪声小单
    
    返回：
    correlation: 经加权和时间修正后的推力-位移相关系数
    """
    n = len(price_arr)
    if n < 10:  # 最少需要10个tick点以保证统计意义
        return 0.0
    
    # 1. 计算成交量阈值，过滤噪声小单
    total_volume = np.sum(volume_arr)
    volume_threshold = total_volume * volume_threshold_ratio
    valid_mask = volume_arr >= volume_threshold
    
    # 如果没有有效数据点，返回0
    valid_count = np.sum(valid_mask)
    if valid_count < 5:
        return 0.0
    
    # 2. 提取有效数据点
    valid_price_changes = price_change_arr[valid_mask]
    valid_volumes = volume_arr[valid_mask]
    
    # 3. 计算推力向量（带方向的正负推力）
    # 推力 = 价格变化方向 * log(成交量) * 价格变化绝对值
    # 使用对数成交量平滑极端大单影响
    log_volumes = np.log1p(valid_volumes)  # log(1+x)避免为0
    price_directions = np.sign(valid_price_changes)
    price_abs_changes = np.abs(valid_price_changes)
    
    # 处理价格变化为0的情况（价格不变但可能有成交量）
    zero_change_mask = price_abs_changes == 0
    if np.any(zero_change_mask):
        # 价格不变时，推力为0（无效推力）
        thrust = np.where(zero_change_mask, 0.0, 
                         price_directions * log_volumes * price_abs_changes)
    else:
        thrust = price_directions * log_volumes * price_abs_changes
    
    # 4. 计算位移向量（价格变化的累计效应）
    # 位移 = 价格变化的累计和，反映价格变动的净效果
    displacement = np.cumsum(valid_price_changes)
    
    # 5. 时间衰减权重（如果提供了时间序列）
    if time_seconds_arr is not None and len(time_seconds_arr) == n:
        valid_times = time_seconds_arr[valid_mask]
        if len(valid_times) > 0:
            # 标准化时间到[0,1]区间
            time_min = valid_times[0]
            time_max = valid_times[-1]
            if time_max > time_min:
                normalized_time = (valid_times - time_min) / (time_max - time_min)
                # 时间衰减因子：尾盘权重略高于早盘（1.0-1.2）
                time_weights = 1.0 + 0.2 * normalized_time
                # 应用时间权重到推力
                thrust = thrust * time_weights
    
    # 6. 计算加权的累计推力和累计位移
    # 累计推力 = 推力序列的加权累计和
    cumulative_thrust = np.cumsum(thrust)
    
    # 7. 鲁棒性相关系数计算（使用中位数中心化）
    # 避免极端值对相关系数的过度影响
    thrust_median = np.median(cumulative_thrust)
    disp_median = np.median(displacement)
    
    # 中心化序列
    thrust_centered = cumulative_thrust - thrust_median
    disp_centered = displacement - disp_median
    
    # 8. 计算加权协方差和方差
    # 使用有效数据点的数量作为权重基准
    weights = np.ones_like(thrust_centered) / valid_count
    
    # 加权协方差
    cov_weighted = np.sum(weights * thrust_centered * disp_centered)
    
    # 加权方差
    var_thrust_weighted = np.sum(weights * thrust_centered * thrust_centered)
    var_disp_weighted = np.sum(weights * disp_centered * disp_centered)
    
    # 9. 计算相关系数，确保分母不为0
    denominator = np.sqrt(var_thrust_weighted * var_disp_weighted)
    if denominator > 1e-10:
        correlation = cov_weighted / denominator
        # 限制相关系数在[-1, 1]范围内
        correlation = max(min(correlation, 1.0), -1.0)
    else:
        correlation = 0.0
    
    return float(correlation)

@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _numba_calculate_liquidity_authenticity_score(
    buy_price1_arr, buy_volume1_arr, sell_price1_arr, sell_volume1_arr,
    tick_prices_arr, tick_times_arr, level5_times_arr,
    buy_commitment_threshold, sell_commitment_threshold
):
    """
    Numba优化计算流动性诚意分
    追踪大额挂单的命运：是真实成交（兑现）还是临阵脱逃（违约）
    """
    # 确保数组长度一致，避免越界
    n_level5 = min(len(buy_price1_arr), len(buy_volume1_arr), len(sell_price1_arr), len(sell_volume1_arr))
    n_tick = min(len(tick_prices_arr), len(tick_times_arr))
    
    fulfillments = 0
    defaults = 0
    
    # 定义时间窗口：大单出现后60秒（20个3秒周期）内追踪其命运
    time_window_ns = 60_000_000_000  # 60秒，纳秒单位
    
    # 存储大单信息：时间戳、价格、方向、原始挂单量
    commitments = []
    
    # 步骤1：识别异常大单（承诺）
    for i in prange(n_level5):
        # 识别买方大单
        if buy_volume1_arr[i] >= buy_commitment_threshold and buy_price1_arr[i] > 0:
            commitments.append((level5_times_arr[i], buy_price1_arr[i], 1, buy_volume1_arr[i]))
        # 识别卖方大单
        if sell_volume1_arr[i] >= sell_commitment_threshold and sell_price1_arr[i] > 0:
            commitments.append((level5_times_arr[i], sell_price1_arr[i], -1, sell_volume1_arr[i]))
    
    # 步骤2：追踪每个大单的命运
    for i in prange(len(commitments)):
        commit_time, commit_price, direction, commit_volume = commitments[i]
        # 在tick数据中寻找后续的价格触及和成交情况
        found_touch = False
        found_fulfill = False
        remaining_volume = commit_volume
        # 遍历tick数据，寻找在时间窗口内的相关成交
        for j in range(n_tick):
            tick_time = tick_times_arr[j]
            tick_price = tick_prices_arr[j]
            # 只检查时间窗口内的tick
            if tick_time <= commit_time:
                continue
            if tick_time - commit_time > time_window_ns:
                break
            # 检查价格是否触及大单价位
            if direction == 1:  # 买方大单
                if tick_price <= commit_price:  # 卖价触及买单价位
                    found_touch = True
                    # 模拟成交：每次tick减去相应成交量（简化模型）
                    # 这里假设每次tick都会消耗一部分挂单
                    remaining_volume = max(0, remaining_volume - 100)  # 假设每次tick成交100手
                    if remaining_volume <= 0:
                        found_fulfill = True
                        break
            else:  # 卖方大单
                if tick_price >= commit_price:  # 买价触及卖单价位
                    found_touch = True
                    remaining_volume = max(0, remaining_volume - 100)  # 假设每次tick成交100手
                    if remaining_volume <= 0:
                        found_fulfill = True
                        break
        # 步骤3：判断大单命运
        if found_touch:
            if found_fulfill:
                fulfillments += 1  # 完全成交，兑现
            else:
                defaults += 1  # 触及但未完全成交，违约（可能是撤单）
        # 未触及价格的情况不计入统计
    
    return fulfillments, defaults

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
        """计算订单流失衡(OFI)与扫单强度 - 精确重构版"""
        # 提取所需数据
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        total_volume = context.get('total_volume_safe')
        debug_info = context.get('debug', {})
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'order_flow_imbalance_score': np.nan,
            'buy_sweep_intensity': np.nan,
            'sell_sweep_intensity': np.nan,
        }
        # 数据验证与准备
        if tick_df is None or tick_df.empty or total_volume <= 0:
            return results
        # 精确计算订单流失衡(OFI) - 使用全五档数据
        if level5_df is not None and not level5_df.empty and len(level5_df) > 1:
            # 准备五档买卖盘口数据
            buy_prices = level5_df[[f'buy_price{i}' for i in range(1, 6)]].values
            buy_volumes = level5_df[[f'buy_volume{i}' for i in range(1, 6)]].values
            sell_prices = level5_df[[f'sell_price{i}' for i in range(1, 6)]].values
            sell_volumes = level5_df[[f'sell_volume{i}' for i in range(1, 6)]].values
            # 动态权重分配：离最优价越近权重越高，考虑档位深度衰减
            weights = np.array([0.35, 0.25, 0.15, 0.15, 0.10])  # 买卖权重对称
            # 计算每个时间点的订单流失衡
            ofi_values = np.zeros(len(level5_df))
            for i in range(1, len(level5_df)):
                # 提取当前和前一时点的盘口数据
                curr_buy_vol = buy_volumes[i]
                prev_buy_vol = buy_volumes[i-1]
                curr_sell_vol = sell_volumes[i]
                prev_sell_vol = sell_volumes[i-1]
                curr_buy_prices = buy_prices[i]
                prev_buy_prices = buy_prices[i-1]
                curr_sell_prices = sell_prices[i]
                prev_sell_prices = sell_prices[i-1]
                # 逐档计算订单流变化，考虑价格变动的影响
                for level in range(5):
                    # 买单失衡计算：考虑挂单量变化和价格位置变化
                    if curr_buy_prices[level] == prev_buy_prices[level]:
                        # 同一价格档位，直接计算挂单量变化
                        buy_flow = curr_buy_vol[level] - prev_buy_vol[level]
                    else:
                        # 价格档位变动，视为撤单后重新挂单
                        buy_flow = curr_buy_vol[level]  # 新价格档位的全部挂单视为新增需求
                    # 卖单失衡计算：与买单相反
                    if curr_sell_prices[level] == prev_sell_prices[level]:
                        sell_flow = curr_sell_vol[level] - prev_sell_vol[level]
                    else:
                        sell_flow = curr_sell_vol[level]
                    # 加权累加：买单增加或卖单减少为买方压力
                    ofi_values[i] += weights[level] * (buy_flow - sell_flow)
                # 考虑大单失衡的额外权重：如果单档变化超过平均水平的3倍，给予额外权重
                avg_volume_change = np.mean(np.abs(ofi_values[max(0, i-10):i])) if i > 10 else 1
                large_order_mask = np.abs(ofi_values[i]) > 3 * avg_volume_change
                if large_order_mask:
                    ofi_values[i] *= 1.2  # 大单失衡给予20%额外权重
            # 剔除开盘和收盘特殊时段的噪声（前5分钟和后5分钟）
            if len(level5_df) > 60:  # 假设每分钟有多个盘口快照
                # 找到开盘后5分钟和收盘前5分钟的位置
                open_cutoff = int(len(level5_df) * 0.1)  # 前10%作为开盘噪声
                close_cutoff = int(len(level5_df) * 0.9)  # 后10%作为收盘噪声
                ofi_values[:open_cutoff] = ofi_values[:open_cutoff] * 0.5  # 开盘时段减权
                ofi_values[close_cutoff:] = ofi_values[close_cutoff:] * 0.5  # 收盘时段减权
            # 累计全日OFI并标准化
            total_ofi = np.sum(ofi_values)
            # 使用总成交量标准化，考虑流通盘规模调整
            results['order_flow_imbalance_score'] = total_ofi / total_volume * 1000  # 放大1000倍便于观察
        # 精确计算扫单强度 - 使用3秒聚合的tick数据
        # 数据预处理：确保时间排序和价格有效性
        tick_df = tick_df.copy()
        tick_df.sort_values('time', inplace=True)
        # 识别主动买卖方向
        tick_df['is_buy'] = tick_df['type'].apply(lambda x: 1 if x == 'B' else 0)
        tick_df['is_sell'] = tick_df['type'].apply(lambda x: 1 if x == 'S' else 0)
        # 计算每笔交易相对于前一笔的价格变化
        tick_df['price_change'] = tick_df['price'].diff()
        tick_df['prev_price'] = tick_df['price'].shift(1)
        # 定义扫单的严格条件
        buy_sweep_volume = 0
        sell_sweep_volume = 0
        total_buy_volume = tick_df[tick_df['is_buy'] == 1]['volume'].sum()
        total_sell_volume = tick_df[tick_df['is_sell'] == 1]['volume'].sum()
        # 扫单检测参数
        min_sweep_length = 3  # 连续扫单最小长度
        price_momentum_threshold = 0.001  # 价格动量阈值（0.1%）
        volume_concentration_threshold = 0.7  # 成交量集中度阈值
        # 识别买入扫单序列
        buy_sequences = []
        current_sequence = []
        for idx, row in tick_df[tick_df['is_buy'] == 1].iterrows():
            if not current_sequence:
                current_sequence.append(row)
                continue
            # 检查是否满足扫单连续性条件
            last_row = current_sequence[-1]
            # 条件1：价格持续上涨
            price_increasing = row['price'] > last_row['price']
            # 条件2：时间连续性（3秒聚合数据内保持连续性）
            time_gap = (row['time'] - last_row['time']).total_seconds()
            time_continuous = time_gap <= 3.5  # 允许微小时间间隔
            # 条件3：成交量不低于前一笔的50%
            volume_sustained = row['volume'] >= last_row['volume'] * 0.5
            if price_increasing and time_continuous and volume_sustained:
                current_sequence.append(row)
            else:
                if len(current_sequence) >= min_sweep_length:
                    buy_sequences.append(current_sequence.copy())
                current_sequence = [row]
        # 处理最后一个序列
        if len(current_sequence) >= min_sweep_length:
            buy_sequences.append(current_sequence)
        # 计算买入扫单强度
        for seq in buy_sequences:
            if len(seq) < 2:
                continue
            # 计算序列特征
            seq_prices = [r['price'] for r in seq]
            seq_volumes = [r['volume'] for r in seq]
            price_momentum = (seq_prices[-1] - seq_prices[0]) / seq_prices[0]
            # 条件1：价格动量超过阈值
            if price_momentum < price_momentum_threshold:
                continue
            # 条件2：成交量集中在前几笔（扫单特征）
            first_half_volume = sum(seq_volumes[:int(len(seq_volumes)*0.5)])
            total_seq_volume = sum(seq_volumes)
            volume_concentration = first_half_volume / total_seq_volume if total_seq_volume > 0 else 0
            if volume_concentration < volume_concentration_threshold:
                continue
            # 条件3：排除尾盘拉升（如果是收盘前10分钟，减权处理）
            last_time = seq[-1]['time']
            if hasattr(last_time, 'hour'):
                if last_time.hour == 14 and last_time.minute >= 50:
                    # 尾盘拉升，给予50%折扣
                    buy_sweep_volume += total_seq_volume * 0.5
                else:
                    buy_sweep_volume += total_seq_volume
        # 识别卖出扫单序列（逻辑对称但方向相反）
        sell_sequences = []
        current_sequence = []
        for idx, row in tick_df[tick_df['is_sell'] == 1].iterrows():
            if not current_sequence:
                current_sequence.append(row)
                continue
            last_row = current_sequence[-1]
            price_decreasing = row['price'] < last_row['price']
            time_gap = (row['time'] - last_row['time']).total_seconds()
            time_continuous = time_gap <= 3.5
            volume_sustained = row['volume'] >= last_row['volume'] * 0.5
            if price_decreasing and time_continuous and volume_sustained:
                current_sequence.append(row)
            else:
                if len(current_sequence) >= min_sweep_length:
                    sell_sequences.append(current_sequence.copy())
                current_sequence = [row]
        if len(current_sequence) >= min_sweep_length:
            sell_sequences.append(current_sequence)
        # 计算卖出扫单强度
        for seq in sell_sequences:
            if len(seq) < 2:
                continue
            seq_prices = [r['price'] for r in seq]
            seq_volumes = [r['volume'] for r in seq]
            price_momentum = (seq_prices[0] - seq_prices[-1]) / seq_prices[0]  # 注意方向
            if price_momentum < price_momentum_threshold:
                continue
            first_half_volume = sum(seq_volumes[:int(len(seq_volumes)*0.5)])
            total_seq_volume = sum(seq_volumes)
            volume_concentration = first_half_volume / total_seq_volume if total_seq_volume > 0 else 0
            if volume_concentration < volume_concentration_threshold:
                continue
            # 开盘跳水检查
            first_time = seq[0]['time']
            if hasattr(first_time, 'hour'):
                if first_time.hour == 9 and first_time.minute <= 30:
                    # 开盘跳水，给予50%折扣
                    sell_sweep_volume += total_seq_volume * 0.5
                else:
                    sell_sweep_volume += total_seq_volume
        # 计算最终扫单强度指标
        if total_buy_volume > 0:
            results['buy_sweep_intensity'] = buy_sweep_volume / total_buy_volume
        if total_sell_volume > 0:
            results['sell_sweep_intensity'] = sell_sweep_volume / total_sell_volume
        # 添加质量检查：如果扫单强度异常高，进行合理性调整
        if results['buy_sweep_intensity'] > 0.5:  # 如果超过50%，可能计算有误
            results['buy_sweep_intensity'] = min(results['buy_sweep_intensity'], 0.5)
        if results['sell_sweep_intensity'] > 0.5:
            results['sell_sweep_intensity'] = min(results['sell_sweep_intensity'], 0.5)
        return results

    @staticmethod
    def _calculate_vpin(context: dict) -> dict:
        """计算精确的知情交易概率指标(VPIN)，基于A股市场特性优化，包含中性'M'数据的精确处理"""
        tick_df = context.get('tick_df')
        total_volume = context.get('total_volume_safe')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {'vpin_score': np.nan}
        if tick_df is None or tick_df.empty or total_volume <= 0:
            return results
        tick_df = tick_df.sort_index() if hasattr(tick_df.index, 'name') and tick_df.index.name == 'time' else tick_df
        target_bucket_count = 50
        vpin_bucket_size = total_volume / target_bucket_count
        if vpin_bucket_size < 100:
            target_bucket_count = max(20, int(total_volume / 100))
            vpin_bucket_size = total_volume / target_bucket_count
        vpin_window = 10
        # 对中性'M'数据的精细化处理方法：
        # 方法1：基于价格变动的推断法（首选，最精确）
        # 方法2：基于相邻买卖单的比例分配法（备选）
        # 首先确保数据包含必要的列
        required_columns = ['type', 'volume']
        if not all(col in tick_df.columns for col in required_columns):
            return results
        # 初始化买卖成交量数组
        tick_df['buy_vol'] = 0.0
        tick_df['sell_vol'] = 0.0
        # 处理已知的B和S类型
        is_buy = tick_df['type'] == 'B'
        is_sell = tick_df['type'] == 'S'
        is_neutral = tick_df['type'] == 'M'
        tick_df.loc[is_buy, 'buy_vol'] = tick_df.loc[is_buy, 'volume']
        tick_df.loc[is_sell, 'sell_vol'] = tick_df.loc[is_sell, 'volume']
        # 处理中性'M'数据 - 方法1：基于价格变动的推断法
        if is_neutral.any() and 'price' in tick_df.columns:
            # 计算价格变动：当前价格与前一笔价格的差异
            price_series = tick_df['price'].astype(float)
            price_diff = price_series.diff()
            # 对于每一笔中性交易，根据价格变动方向推断买卖方向
            for idx in tick_df[is_neutral].index:
                current_idx = tick_df.index.get_loc(idx)
                # 获取当前价格和前一笔价格
                current_price = price_series.iloc[current_idx]
                # 如果无法获取前一笔价格，则使用比例分配法
                if current_idx == 0 or pd.isna(price_diff.iloc[current_idx]):
                    # 方法2：使用相邻买卖单比例分配
                    # 计算当前时间点前后一段时间窗口内的买卖比例
                    window_size = 5  # 使用前后5笔交易作为参考窗口
                    start_idx = max(0, current_idx - window_size)
                    end_idx = min(len(tick_df), current_idx + window_size + 1)
                    # 计算窗口内B和S类型的成交量
                    window_data = tick_df.iloc[start_idx:end_idx]
                    buy_in_window = window_data.loc[window_data['type'] == 'B', 'volume'].sum()
                    sell_in_window = window_data.loc[window_data['type'] == 'S', 'volume'].sum()
                    total_bs_in_window = buy_in_window + sell_in_window
                    if total_bs_in_window > 0:
                        buy_ratio = buy_in_window / total_bs_in_window
                        sell_ratio = sell_in_window / total_bs_in_window
                    else:
                        buy_ratio = sell_ratio = 0.5
                    # 按比例分配中性交易量
                    neutral_volume = tick_df.loc[idx, 'volume']
                    tick_df.loc[idx, 'buy_vol'] = neutral_volume * buy_ratio
                    tick_df.loc[idx, 'sell_vol'] = neutral_volume * sell_ratio
                else:
                    # 根据价格变动方向推断
                    price_change = price_diff.iloc[current_idx]
                    neutral_volume = tick_df.loc[idx, 'volume']
                    if price_change > 0:
                        # 价格上涨，推断为买方驱动
                        tick_df.loc[idx, 'buy_vol'] = neutral_volume * 0.7  # 70%分配给买方
                        tick_df.loc[idx, 'sell_vol'] = neutral_volume * 0.3  # 30%分配给卖方
                    elif price_change < 0:
                        # 价格下跌，推断为卖方驱动
                        tick_df.loc[idx, 'buy_vol'] = neutral_volume * 0.3  # 30%分配给买方
                        tick_df.loc[idx, 'sell_vol'] = neutral_volume * 0.7  # 70%分配给卖方
                    else:
                        # 价格不变，平均分配
                        tick_df.loc[idx, 'buy_vol'] = neutral_volume * 0.5
                        tick_df.loc[idx, 'sell_vol'] = neutral_volume * 0.5
        elif is_neutral.any():
            # 如果没有价格数据，使用相邻买卖单比例分配法
            for idx in tick_df[is_neutral].index:
                current_idx = tick_df.index.get_loc(idx)
                window_size = 10
                start_idx = max(0, current_idx - window_size)
                end_idx = min(len(tick_df), current_idx + window_size + 1)
                window_data = tick_df.iloc[start_idx:end_idx]
                buy_in_window = window_data.loc[window_data['type'] == 'B', 'volume'].sum()
                sell_in_window = window_data.loc[window_data['type'] == 'S', 'volume'].sum()
                total_bs_in_window = buy_in_window + sell_in_window
                if total_bs_in_window > 0:
                    buy_ratio = buy_in_window / total_bs_in_window
                    sell_ratio = sell_in_window / total_bs_in_window
                else:
                    buy_ratio = sell_ratio = 0.5
                neutral_volume = tick_df.loc[idx, 'volume']
                tick_df.loc[idx, 'buy_vol'] = neutral_volume * buy_ratio
                tick_df.loc[idx, 'sell_vol'] = neutral_volume * sell_ratio
        tick_df['cum_vol'] = tick_df['volume'].cumsum()
        cum_vol_arr = tick_df['cum_vol'].values.astype(np.float64)
        volume_arr = tick_df['volume'].values.astype(np.float64)
        buy_vol_arr = tick_df['buy_vol'].values.astype(np.float64)
        sell_vol_arr = tick_df['sell_vol'].values.astype(np.float64)
        imbalance_values, bucket_indices = _numba_calculate_vpin_buckets(
            cum_vol_arr, volume_arr, buy_vol_arr, sell_vol_arr, vpin_bucket_size
        )
        if len(imbalance_values) > vpin_window:
            bucket_imbalance_series = pd.Series(imbalance_values, index=bucket_indices)
            imbalance_std = bucket_imbalance_series.rolling(
                window=vpin_window, 
                min_periods=int(vpin_window * 0.7)
            ).std().bfill()
            epsilon = 1e-10
            imbalance_std = imbalance_std.replace(0, epsilon)
            abs_imbalance = bucket_imbalance_series.abs()
            z_score = abs_imbalance / imbalance_std
            vpin_series = z_score.apply(lambda z: norm.cdf(z) if pd.notna(z) else np.nan)
            results['vpin_score'] = float(vpin_series.mean())
        return results

    @staticmethod
    def _calculate_hf_mechanics(context: dict) -> dict:
        """
        【V62.0 · 博弈精研 - 高精度版】
        重构要点：
        1. 推力计算精细化：考虑价格变化幅度、成交量规模、价格不变的特殊情况
        2. 时间维度引入：早盘和尾盘的推力具有不同时效性，加入时间衰减因子
        3. 噪声过滤：基于成交量阈值过滤小单噪声，避免随机小单干扰主力行为判断
        4. 鲁棒性增强：使用中位数中心化替代均值，抵抗极端值影响
        5. 方向一致性检验：确保推力方向与价格变化方向严格一致
        返回：
        active_volume_price_efficiency: 精细化计算后的主动量价效能指标
        """
        tick_df = context.get('tick_df')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {'active_volume_price_efficiency': np.nan}
        # 1. 数据基础校验
        if tick_df is None or tick_df.empty or len(tick_df) < 20:
            # 至少需要20个tick点以保证统计意义
            return results
        # 2. 提取核心数据数组
        price_arr = tick_df['price'].values.astype(np.float64)
        volume_arr = tick_df['volume'].values.astype(np.float64)
        # 3. 精确计算价格变化序列
        # 优先使用原始price_change列，但需验证其准确性
        if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
            raw_price_change = tick_df['price_change'].values.astype(np.float64)
            # 验证原始price_change的准确性：与计算值对比
            calculated_change = np.diff(price_arr, prepend=price_arr[0])
            # 如果原始值异常（如全为0但价格实际有变化），则使用计算值
            zero_mask = raw_price_change == 0
            if np.all(zero_mask) and not np.all(calculated_change == 0):
                price_change_arr = calculated_change
            else:
                # 混合使用：原始值为0时用计算值，否则用原始值
                price_change_arr = np.where(zero_mask, calculated_change, raw_price_change)
        else:
            price_change_arr = np.diff(price_arr, prepend=price_arr[0])
        # 4. 提取时间序列（用于时间衰减计算）
        time_seconds_arr = None
        if 'time' in tick_df.columns:
            # 转换时间到秒数，用于计算时间衰减
            try:
                if isinstance(tick_df['time'].iloc[0], str):
                    # 假设时间格式为HH:MM:SS或HHMMSS
                    time_strs = tick_df['time'].values
                    seconds_list = []
                    for t in time_strs:
                        if ':' in t:
                            h, m, s = t.split(':')
                        elif len(t) == 6:
                            h, m, s = t[:2], t[2:4], t[4:6]
                        else:
                            h, m, s = 9, 30, 0  # 默认开盘时间
                        total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
                        seconds_list.append(total_seconds)
                    time_seconds_arr = np.array(seconds_list, dtype=np.float64)
            except:
                time_seconds_arr = None
        # 5. 调用高精度Numba函数计算
        correlation = _numba_calculate_active_volume_price_efficiency(
            price_arr=price_arr,
            volume_arr=volume_arr,
            price_change_arr=price_change_arr,
            time_seconds_arr=time_seconds_arr,
            volume_threshold_ratio=0.0005  # 更严格的阈值，过滤更多噪声
        )
        # 6. 后处理：根据市场阶段调整
        # 获取日线数据判断市场状态
        day_open = context.get('day_open_qfq', np.nan)
        day_close = context.get('day_close_qfq', np.nan)
        if not np.isnan(day_open) and not np.isnan(day_close):
            daily_return = (day_close - day_open) / day_open
            # 如果日内波动极小（<0.5%），降低指标灵敏度
            if abs(daily_return) < 0.005:
                correlation = correlation * 0.7  # 压缩信号强度
        results['active_volume_price_efficiency'] = correlation
        return results

    @staticmethod
    def _calculate_liquidity_metrics(context: dict) -> dict:
        """
        【V70.1 · 流动性验真·精细化重构版】
        针对3秒聚合高频数据进行精细化计算，以精确性为第一要务
        重构思路：
        1. market_impact_cost：使用更精确的冲击成本模型，考虑实际成交分布
        2. liquidity_slope：优化价格归一化方法，消除绝对价格影响
        3. liquidity_authenticity_score：使用更精细的大单识别和追踪逻辑
        """
        # 提取关键数据
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        realtime_df = context.get('realtime_df')
        group = context.get('group')
        continuous_group = context.get('continuous_group')
        daily_series_for_day = context.get('daily_series_for_day')
        debug_info = context.get('debug', {})
        # 初始化结果字典
        results = {
            'market_impact_cost': np.nan,
            'liquidity_slope': np.nan,
            'liquidity_authenticity_score': np.nan,
        }
        # 数据校验
        if tick_df is None or tick_df.empty or level5_df is None or level5_df.empty or len(level5_df) < 10:
            # 数据不足时返回中性值
            results['market_impact_cost'] = 0.15  # 1.5%的冲击成本
            results['liquidity_slope'] = 1.0e7  # 中等斜率
            results['liquidity_authenticity_score'] = 0.5  # 中性分数
            return results
        # 提取日级数据用于标准化
        day_open = context.get('day_open_qfq', np.nan)
        day_close = context.get('day_close_qfq', np.nan)
        day_high = context.get('day_high_qfq', np.nan)
        day_low = context.get('day_low_qfq', np.nan)
        pre_close = context.get('pre_close_qfq', np.nan)
        atr_14 = context.get('atr_14', np.nan)
        # 计算日内波动率作为标准化基准
        if pd.notna(day_high) and pd.notna(day_low) and pd.notna(pre_close) and pre_close > 0:
            daily_range_pct = (day_high - day_low) / pre_close * 100
        else:
            daily_range_pct = 2.0  # 默认2%的日波动
        # 标准化基准价格：使用VWAP作为更稳定的基准
        if group is not None and not group.empty and 'close' in group.columns and 'vol' in group.columns:
            total_volume = group['vol'].sum()
            if total_volume > 0:
                vwap = (group['close'] * group['vol']).sum() / total_volume
            else:
                vwap = day_close if pd.notna(day_close) else pre_close
        else:
            vwap = day_close if pd.notna(day_close) else pre_close
        # ==================== 1. 市场冲击成本精细化计算 ====================
        if level5_df is not None and not level5_df.empty and realtime_df is not None and not realtime_df.empty:
            # 重命名列以便处理
            column_rename_map = {
                **{f'buy_price{i}': f'b{i}_p' for i in range(1, 6)},
                **{f'buy_volume{i}': f'b{i}_v' for i in range(1, 6)},
                **{f'sell_price{i}': f'a{i}_p' for i in range(1, 6)},
                **{f'sell_volume{i}': f'a{i}_v' for i in range(1, 6)}
            }
            level5_df_renamed = level5_df.copy().rename(columns=column_rename_map)
            # 精确对齐时间戳：3秒聚合数据需要精确匹配
            snapshot_df = pd.merge_asof(
                realtime_df.sort_index(),
                level5_df_renamed.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest',  # 使用最近邻匹配，更适合3秒数据
                tolerance=pd.Timedelta('5s')  # 允许5秒的时间偏差
            )
            # 计算每个快照的增量成交量
            if 'volume' in snapshot_df.columns:
                snapshot_df['snapshot_volume'] = snapshot_df['volume'].diff().fillna(snapshot_df['volume'].iloc[0] if len(snapshot_df) > 0 else 0)
                # 过滤异常值：成交量不能为负，也不能异常大
                volume_median = snapshot_df['snapshot_volume'].median()
                volume_std = snapshot_df['snapshot_volume'].std()
                if volume_std > 0:
                    snapshot_df['snapshot_volume'] = snapshot_df['snapshot_volume'].clip(
                        lower=0,
                        upper=volume_median + 3 * volume_std
                    )
            else:
                # 如果没有volume列，使用近似估计
                total_volume = context.get('total_volume_safe', 0)
                if total_volume > 0 and len(snapshot_df) > 0:
                    snapshot_df['snapshot_volume'] = total_volume / len(snapshot_df)
                else:
                    snapshot_df['snapshot_volume'] = 0
            # 确定冲击订单规模：使用日成交额的0.1%作为标准冲击量
            total_amount = daily_series_for_day.get('amount', 0) if daily_series_for_day is not None else 0
            if total_amount <= 0 and group is not None and 'amount' in group.columns:
                total_amount = group['amount'].sum()
            if total_amount > 0:
                # 冲击量：取0.1%的日成交额，但不超过日成交额的1%
                standard_amount = float(total_amount) * 0.001
                # 进一步标准化：确保冲击量在合理范围内
                if vwap > 0:
                    standard_shares = standard_amount / vwap
                    # 限制冲击股数在日成交量的0.5%以内
                    max_shares = total_volume * 0.005 if total_volume > 0 else standard_shares
                    standard_shares = min(standard_shares, max_shares)
                    standard_amount = standard_shares * vwap
                impact_costs = []
                weights = []
                for idx, row in snapshot_df.iterrows():
                    snapshot_volume = row['snapshot_volume']
                    if snapshot_volume <= 0:
                        continue
                    # 计算中间价作为基准
                    b1_p = row.get('b1_p')
                    a1_p = row.get('a1_p')
                    if pd.isna(b1_p) or pd.isna(a1_p) or b1_p <= 0 or a1_p <= 0:
                        continue
                    mid_price = (b1_p + a1_p) / 2
                    # 模拟买入冲击成本（冲击卖单簿）
                    amount_to_fill = standard_amount
                    filled_amount = 0.0
                    filled_shares = 0.0
                    # 遍历卖1到卖5
                    for i in range(1, 6):
                        price_key = f'a{i}_p'
                        volume_key = f'a{i}_v'
                        price = row.get(price_key)
                        volume = row.get(volume_key, 0)
                        if pd.isna(price) or price <= 0 or pd.isna(volume) or volume <= 0:
                            continue
                        # 将挂单手数转换为股数（A股1手=100股）
                        shares_available = float(volume) * 100
                        value_available = price * shares_available
                        if amount_to_fill >= value_available:
                            # 完全吃掉这一档
                            filled_amount += value_available
                            filled_shares += shares_available
                            amount_to_fill -= value_available
                        else:
                            # 部分吃掉这一档
                            shares_to_fill = amount_to_fill / price
                            filled_amount += amount_to_fill
                            filled_shares += shares_to_fill
                            amount_to_fill = 0
                            break
                    if filled_shares > 0 and mid_price > 0:
                        # 计算实际成交均价
                        actual_avg_price = filled_amount / filled_shares
                        # 计算冲击成本百分比
                        cost_pct = (actual_avg_price / mid_price - 1) * 100
                        # 标准化冲击成本：除以日波动率，消除市场整体波动影响
                        if daily_range_pct > 0:
                            normalized_cost = cost_pct / daily_range_pct
                        else:
                            normalized_cost = cost_pct
                        impact_costs.append(normalized_cost)
                        weights.append(snapshot_volume)
                if impact_costs and sum(weights) > 0:
                    # 使用成交量加权平均，但过滤异常值
                    costs_array = np.array(impact_costs)
                    weights_array = np.array(weights)
                    # 过滤异常值：去除成本超过3倍标准差的值
                    if len(costs_array) >= 5:
                        cost_mean = np.mean(costs_array)
                        cost_std = np.std(costs_array)
                        if cost_std > 0:
                            valid_mask = np.abs(costs_array - cost_mean) <= 3 * cost_std
                            costs_array = costs_array[valid_mask]
                            weights_array = weights_array[valid_mask]
                    if len(costs_array) > 0 and sum(weights_array) > 0:
                        results['market_impact_cost'] = np.average(costs_array, weights=weights_array)
        # ==================== 2. 流动性斜率精细化计算 ====================
        if level5_df is not None and not level5_df.empty:
            slopes = []
            slope_weights = []
            for idx, row in level5_df.iterrows():
                # 获取中间价
                b1_p = row.get('buy_price1')
                a1_p = row.get('sell_price1')
                if pd.isna(b1_p) or pd.isna(a1_p) or b1_p <= 0 or a1_p <= 0:
                    continue
                mid_price = (b1_p + a1_p) / 2
                # 准备卖单簿数据（计算向上冲击的流动性）
                ask_prices = []
                ask_cum_volumes = []
                cum_volume = 0
                for i in range(1, 6):
                    price_key = f'sell_price{i}'
                    volume_key = f'sell_volume{i}'
                    price = row.get(price_key)
                    volume = row.get(volume_key, 0)
                    if pd.isna(price) or price <= 0 or pd.isna(volume) or volume <= 0:
                        # 如果某一档缺失，使用线性插值
                        if i == 1:
                            price = a1_p
                        else:
                            prev_price = ask_prices[-1] if ask_prices else a1_p
                            price = prev_price * 1.001  # 假设1‰的价差
                        volume = 0
                    # 价格偏移（相对于中间价的百分比）
                    price_offset = (price - mid_price) / mid_price
                    # 累计挂单量（股数）
                    cum_volume += float(volume) * 100
                    ask_prices.append(price_offset)
                    ask_cum_volumes.append(cum_volume)
                # 准备买单簿数据（计算向下冲击的流动性）
                bid_prices = []
                bid_cum_volumes = []
                cum_volume = 0
                for i in range(1, 6):
                    price_key = f'buy_price{i}'
                    volume_key = f'buy_volume{i}'
                    price = row.get(price_key)
                    volume = row.get(volume_key, 0)
                    if pd.isna(price) or price <= 0 or pd.isna(volume) or volume <= 0:
                        if i == 1:
                            price = b1_p
                        else:
                            prev_price = bid_prices[-1] if bid_prices else b1_p
                            price = prev_price * 0.999  # 假设1‰的价差
                        volume = 0
                    price_offset = (price - mid_price) / mid_price
                    cum_volume += float(volume) * 100
                    bid_prices.append(price_offset)
                    bid_cum_volumes.append(cum_volume)
                # 合并买卖双方数据（价格偏移取绝对值，因为流动性是关于中间价对称的）
                all_prices = []
                all_cum_volumes = []
                # 买方：价格偏移为负，按绝对值从小到大排序
                for i in range(len(bid_prices)-1, -1, -1):
                    all_prices.append(abs(bid_prices[i]))
                    all_cum_volumes.append(bid_cum_volumes[i])
                # 卖方：价格偏移为正
                for i in range(len(ask_prices)):
                    all_prices.append(abs(ask_prices[i]))
                    all_cum_volumes.append(ask_cum_volumes[i])
                # 进行线性回归计算斜率
                if len(all_prices) >= 3 and np.std(all_prices) > 1e-10:
                    try:
                        slope, _, _, _, _ = linregress(all_prices, all_cum_volumes)
                        # 标准化斜率：除以当日VWAP和总成交量
                        if vwap > 0 and total_volume > 0:
                            # 理论基准：在1%的价格变动内应该能成交多少比例的日成交量
                            theoretical_slope = total_volume / 0.01
                            if theoretical_slope > 0:
                                normalized_slope = slope / theoretical_slope
                                slopes.append(normalized_slope)
                                # 使用该时刻的买卖价差作为权重：价差越小，权重越大
                                spread_pct = (a1_p - b1_p) / mid_price
                                if spread_pct > 0:
                                    weight = 1.0 / spread_pct
                                    slope_weights.append(weight)
                    except:
                        continue
            if slopes and slope_weights and sum(slope_weights) > 0:
                # 流动性斜率：数值越小表示流动性越好（价格变动一点就能遇到大量挂单）
                # 我们取倒数，使得指标越大表示流动性越好
                slopes_array = np.array(slopes)
                weights_array = np.array(slope_weights)
                # 过滤异常值
                if len(slopes_array) >= 5:
                    slope_mean = np.mean(slopes_array)
                    slope_std = np.std(slopes_array)
                    if slope_std > 0:
                        valid_mask = np.abs(slopes_array - slope_mean) <= 3 * slope_std
                        slopes_array = slopes_array[valid_mask]
                        weights_array = weights_array[valid_mask]
                if len(slopes_array) > 0:
                    avg_slope = np.average(slopes_array, weights=weights_array)
                    # 取倒数并取对数，使得指标更稳定
                    if avg_slope > 0:
                        results['liquidity_slope'] = 1.0 / avg_slope
                    else:
                        results['liquidity_slope'] = 0.0
        # ==================== 3. 流动性诚意分精细化计算 ====================
        if tick_df is not None and not tick_df.empty and level5_df is not None and not level5_df.empty:
            # 准备数据数组
            buy_price1_arr = level5_df['buy_price1'].values.astype(np.float64)
            buy_volume1_arr = level5_df['buy_volume1'].values.astype(np.float64)
            sell_price1_arr = level5_df['sell_price1'].values.astype(np.float64)
            sell_volume1_arr = level5_df['sell_volume1'].values.astype(np.float64)
            # 确保tick数据有时间索引
            if tick_df.index is None or len(tick_df.index) == 0:
                return results
            tick_prices_arr = tick_df['price'].values.astype(np.float64)
            tick_times_arr = tick_df.index.values.astype(np.int64)  # 纳秒时间戳
            # 确保level5数据有时间索引
            if level5_df.index is None or len(level5_df.index) == 0:
                return results
            level5_times_arr = level5_df.index.values.astype(np.int64)
            # 计算大单阈值：使用动态阈值，考虑日内波动
            if len(buy_volume1_arr) >= 20 and len(sell_volume1_arr) >= 20:
                # 使用滚动窗口计算动态阈值
                window_size = min(20, len(buy_volume1_arr))
                # 买方阈值
                buy_rolling_mean = np.convolve(buy_volume1_arr, np.ones(window_size)/window_size, mode='valid')
                buy_rolling_std = np.array([np.std(buy_volume1_arr[max(0, i-window_size+1):i+1]) 
                                           for i in range(window_size-1, len(buy_volume1_arr))])
                if len(buy_rolling_mean) > 0 and len(buy_rolling_std) > 0:
                    buy_commitment_threshold = buy_rolling_mean[-1] + 2.5 * buy_rolling_std[-1]
                else:
                    buy_commitment_threshold = np.mean(buy_volume1_arr) + 2.5 * np.std(buy_volume1_arr)
                # 卖方阈值
                sell_rolling_mean = np.convolve(sell_volume1_arr, np.ones(window_size)/window_size, mode='valid')
                sell_rolling_std = np.array([np.std(sell_volume1_arr[max(0, i-window_size+1):i+1]) 
                                            for i in range(window_size-1, len(sell_volume1_arr))])
                if len(sell_rolling_mean) > 0 and len(sell_rolling_std) > 0:
                    sell_commitment_threshold = sell_rolling_mean[-1] + 2.5 * sell_rolling_std[-1]
                else:
                    sell_commitment_threshold = np.mean(sell_volume1_arr) + 2.5 * np.std(sell_volume1_arr)
            else:
                # 数据不足时使用简单阈值
                buy_commitment_threshold = np.mean(buy_volume1_arr) + 2.5 * np.std(buy_volume1_arr)
                sell_commitment_threshold = np.mean(sell_volume1_arr) + 2.5 * np.std(sell_volume1_arr)
            # 确保阈值合理
            buy_commitment_threshold = max(buy_commitment_threshold, np.percentile(buy_volume1_arr, 75))
            sell_commitment_threshold = max(sell_commitment_threshold, np.percentile(sell_volume1_arr, 75))
            # 调用Numba优化函数
            fulfillments, defaults = _numba_calculate_liquidity_authenticity_score(
                buy_price1_arr, buy_volume1_arr,
                sell_price1_arr, sell_volume1_arr,
                tick_prices_arr, tick_times_arr,
                level5_times_arr,
                buy_commitment_threshold, sell_commitment_threshold
            )
            # 计算诚意分，使用贝叶斯平滑处理小样本问题
            total_events = fulfillments + defaults
            if total_events > 0:
                # 使用贝叶斯先验：假设市场平均有60%的诚意
                prior_success = 6
                prior_failure = 4
                smoothed_score = (fulfillments + prior_success) / (total_events + prior_success + prior_failure)
                results['liquidity_authenticity_score'] = smoothed_score
            else:
                # 无事件发生，给予市场中性分
                results['liquidity_authenticity_score'] = 0.6
        # 最终结果合理性检查
        if pd.isna(results['market_impact_cost']):
            results['market_impact_cost'] = 0.15  # 默认1.5%冲击成本
        if pd.isna(results['liquidity_slope']):
            results['liquidity_slope'] = 1.0e7  # 默认中等斜率
        if pd.isna(results['liquidity_authenticity_score']):
            results['liquidity_authenticity_score'] = 0.6  # 默认中性偏诚意
        return results

    @staticmethod
    def _calculate_vwap_reversion(context: dict) -> dict:
        """计算VWAP均值回归相关性 - 精细化高频版本"""
        # 初始化结果
        results = {'vwap_mean_reversion_corr': np.nan}
        # 优先级1：使用3秒高频tick数据（最精确）
        tick_df = context.get('tick_df')
        if tick_df is not None and not tick_df.empty and len(tick_df) > 10:
            try:
                # 精确计算日内VWAP时间序列（3秒级别）
                tick_df = tick_df.copy()
                tick_df['cum_amount'] = tick_df['amount'].cumsum()
                tick_df['cum_volume'] = tick_df['volume'].cumsum()
                tick_df['intraday_vwap'] = tick_df['cum_amount'] / tick_df['cum_volume']
                # 计算全天的整体VWAP（分母使用总成交量确保精确）
                total_volume = context.get('total_volume_safe')
                if pd.notna(total_volume) and total_volume > 0:
                    daily_vwap = tick_df['amount'].sum() / total_volume
                else:
                    daily_vwap = tick_df['amount'].sum() / tick_df['volume'].sum()
                if pd.notna(daily_vwap):
                    # 计算每个3秒bar的VWAP与全天VWAP的偏差
                    deviation = tick_df['intraday_vwap'] - daily_vwap
                    # 移除极端的异常值（超过5倍标准差）
                    std_dev = deviation.std()
                    if pd.notna(std_dev) and std_dev > 0:
                        mask = np.abs(deviation) <= 5 * std_dev
                        deviation = deviation[mask]
                    if len(deviation) > 2:
                        # 使用滚动窗口计算自相关性（更稳健）
                        lag = 1  # 使用1个周期滞后
                        # 使用精确的皮尔逊相关系数公式
                        x = deviation.iloc[:-lag].values
                        y = deviation.iloc[lag:].values
                        # 确保长度匹配
                        min_len = min(len(x), len(y))
                        x = x[:min_len]
                        y = y[:min_len]
                        if min_len > 10:
                            # 计算相关系数
                            x_mean = np.mean(x)
                            y_mean = np.mean(y)
                            x_std = np.std(x)
                            y_std = np.std(y)
                            
                            if x_std > 1e-10 and y_std > 1e-10:
                                # 使用向量化计算提高精度
                                covariance = np.mean((x - x_mean) * (y - y_mean))
                                correlation = covariance / (x_std * y_std)
                                results['vwap_mean_reversion_corr'] = float(correlation)
            except Exception as e:
                # 如果高频计算失败，降级到分钟级
                pass
        # 优先级2：使用连续分钟数据（降级处理）
        if pd.isna(results['vwap_mean_reversion_corr']):
            minute_df = context.get('continuous_group')
            if minute_df is not None and not minute_df.empty:
                try:
                    # 使用分钟级VWAP，确保数据质量
                    required_cols = ['minute_vwap', 'amount', 'vol']
                    if all(col in minute_df.columns for col in required_cols):
                        # 精确计算日度VWAP（考虑成交量权重）
                        total_amount = minute_df['amount'].sum()
                        total_volume = minute_df['vol'].sum()
                        if total_volume > 0:
                            daily_vwap = total_amount / total_volume
                        else:
                            daily_vwap = np.nan
                        if pd.notna(daily_vwap):
                            deviation = minute_df['minute_vwap'] - daily_vwap
                            # 数据清洗：移除开盘和收盘的特殊时段（前5分钟和最后5分钟）
                            if len(deviation) > 10:
                                deviation = deviation.iloc[5:-5]
                            
                            if len(deviation) > 2:
                                # 使用偏差的百分比变化而不是绝对值（更稳定）
                                deviation_pct = deviation / daily_vwap
                                # 计算滞后相关性（确保时间对齐）
                                lag = 1
                                x = deviation_pct.iloc[:-lag].values
                                y = deviation_pct.iloc[lag:].values
                                min_len = min(len(x), len(y))
                                x = x[:min_len]
                                y = y[:min_len]
                                if min_len > 5:
                                    # 使用加权相关系数（成交量加权）
                                    volumes = minute_df['vol'].iloc[lag:lag+min_len].values
                                    weights = volumes / volumes.sum() if volumes.sum() > 0 else None
                                    if weights is not None:
                                        # 加权相关系数计算
                                        x_weighted_mean = np.average(x, weights=weights)
                                        y_weighted_mean = np.average(y, weights=weights)
                                        x_weighted_var = np.average((x - x_weighted_mean)**2, weights=weights)
                                        y_weighted_var = np.average((y - y_weighted_mean)**2, weights=weights)
                                        if x_weighted_var > 1e-12 and y_weighted_var > 1e-12:
                                            cov_weighted = np.average((x - x_weighted_mean) * (y - y_weighted_mean), weights=weights)
                                            correlation = cov_weighted / np.sqrt(x_weighted_var * y_weighted_var)
                                            results['vwap_mean_reversion_corr'] = float(correlation)
                                    else:
                                        # 退化为普通相关系数
                                        x_mean = np.mean(x)
                                        y_mean = np.mean(y)
                                        x_std = np.std(x)
                                        y_std = np.std(y)
                                        if x_std > 1e-10 and y_std > 1e-10:
                                            covariance = np.mean((x - x_mean) * (y - y_mean))
                                            correlation = covariance / (x_std * y_std)
                                            results['vwap_mean_reversion_corr'] = float(correlation)
                except Exception as e:
                    results['vwap_mean_reversion_corr'] = np.nan
        # 后处理：确保相关性在[-1, 1]范围内
        if pd.notna(results['vwap_mean_reversion_corr']):
            corr = results['vwap_mean_reversion_corr']
            if corr < -1.0:
                results['vwap_mean_reversion_corr'] = -1.0
            elif corr > 1.0:
                results['vwap_mean_reversion_corr'] = 1.0
                
        return results





