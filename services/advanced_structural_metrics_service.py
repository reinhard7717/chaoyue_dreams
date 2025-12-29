# services\advanced_structural_metrics_service.py
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
import numba
from typing import Tuple
from asgiref.sync import sync_to_async
from stock_models.stock_basic import StockInfo
from stock_models.advanced_metrics import BaseAdvancedStructuralMetrics
from services.structural_metrics_calculators import StructuralMetricsCalculators
from services.microstructure_dynamics_calculators import MicrostructureDynamicsCalculators
from services.derivative_metrics_calculators import DerivativeMetricsCalculator
from services.thematic_metrics_calculators import ThematicMetricsCalculators
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
            # 核心变更：不再自己加载数据，直接使用传入的 intraday_data_map
            # 我们需要根据当前区块的日期来过滤这个大的 map
            chunk_intraday_map = {
                pd.to_datetime(d).date(): intraday_data_map[pd.to_datetime(d).date()]
                for d in chunk_dates if pd.to_datetime(d).date() in intraday_data_map
            }
            if not chunk_intraday_map:
                logger.warning(f"[{stock_info.stock_code}] 区块 {chunk_dates.min().date()} to {chunk_dates.max().date()} 无任何日内数据，跳过整个区块。")
                continue
            chunk_new_metrics_df = await self._forge_advanced_structural_metrics(chunk_intraday_map, stock_info.stock_code, daily_df_with_atr)
            all_new_core_metrics_df = pd.concat([all_new_core_metrics_df, chunk_new_metrics_df])
        if all_new_core_metrics_df.empty:
            logger.info(f"[{stock_info.stock_code}] [结构指标] 未能计算出任何新的核心指标，任务结束。")
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

    async def _forge_advanced_structural_metrics(self, intraday_map: dict, stock_code: str, daily_df_with_atr: pd.DataFrame) -> pd.DataFrame:
        """
        【V46.0 · 潜龙在渊】
        - 核心升级: 在构建传递给下一日的 `prev_day_metrics` 上下文时，增加 `volume` 字段，
                    为 `equilibrium_compression_index` 的计算提供必要的“力量胶着度”评估依据。
        """
        new_metrics_data = []
        prev_day_metrics = {}
        if intraday_map:
            # 确保 first_date 是 Timestamp 类型，以便与 daily_df_with_atr 的 DatetimeIndex 兼容
            first_date_ts = pd.to_datetime(min(intraday_map.keys())) # 将 datetime.date 转换为 Timestamp
            prev_date_ts = first_date_ts - pd.Timedelta(days=1) # prev_date_ts 现在是 Timestamp
            # 假设 daily_df_with_atr 的索引在 tasks/stock_analysis_tasks.py 中已修正为 DatetimeIndex (Timestamps)
            if prev_date_ts in daily_df_with_atr.index:
                prev_day_series = daily_df_with_atr.loc[prev_date_ts]
                prev_day_metrics = {
                    'high': prev_day_series.get('high_qfq'),
                    'low': prev_day_series.get('low_qfq'),
                    'volume': prev_day_series.get('volume'),
                }
        # 遍历 intraday_map，其键 trade_date_dt_obj 仍是 datetime.date 对象
        for trade_date_dt_obj, data_for_day in sorted(intraday_map.items()):
            # 将当前的 datetime.date 对象转换为 pandas.Timestamp，用于后续操作
            current_trade_timestamp = pd.to_datetime(trade_date_dt_obj)
            # 使用 Timestamp 进行 daily_df_with_atr 的索引查找
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
            target_date_str = self.debug_params.get('target_date')
            # 调试信息中的日期字符串仍使用原始的 datetime.date 对象
            is_target_date = target_date_str == trade_date_dt_obj.strftime('%Y-%m-%d')
            debug_info = {
                'is_target_date': is_target_date,
                'enable_probe': self.debug_params.get('enable_asm_probe', False),
                'trade_date_str': trade_date_dt_obj.strftime('%Y-%m-%d')
            }
            day_metric_dict = self._calculate_daily_structural_metrics(
                group=canonical_minute_df,
                continuous_group=continuous_group,
                tick_df=tick_df_for_day,
                level5_df=level5_df_for_day,
                realtime_df=realtime_df_for_day,
                daily_info=daily_series_for_day,
                prev_day_metrics=prev_day_metrics,
                debug_info=debug_info
            )
            # 确保 trade_time 字段是 Timestamp 类型
            day_metric_dict['trade_time'] = current_trade_timestamp
            day_metric_dict['stock_code'] = stock_code
            new_metrics_data.append(day_metric_dict)
            # 在传递给下一日的上下文中增加 volume
            prev_day_metrics = {
                'vpoc': day_metric_dict.get('_today_vpoc'),
                'vah': day_metric_dict.get('_today_vah'),
                'val': day_metric_dict.get('_today_val'),
                'atr_14d': daily_series_for_day.get('ATR_14'),
                'high': daily_series_for_day.get('high_qfq'),
                'low': daily_series_for_day.get('low_qfq'),
                'volume': daily_series_for_day.get('volume'),
            }
        if not new_metrics_data:
            return pd.DataFrame()
        new_metrics_df = pd.DataFrame(new_metrics_data)
        # new_metrics_df 的索引现在将是 DatetimeIndex (Timestamps)
        new_metrics_df.set_index('trade_time', inplace=True)
        final_metrics_df = self._calculate_dynamic_evolution_factors(new_metrics_df)
        return final_metrics_df

    def _calculate_daily_structural_metrics(self, group: pd.DataFrame, continuous_group: pd.DataFrame,
                                            tick_df: pd.DataFrame | None, level5_df: pd.DataFrame | None,
                                            realtime_df: pd.DataFrame | None, daily_info: pd.Series,
                                            prev_day_metrics: dict, debug_info: dict) -> dict:
        """
        【V49.0 · 宗门归位】
        - 核心修正: 彻底修复了因调用错误的计算器类而导致的 `AttributeError`。现在，每个指标
                     计算函数都从其正确的“宗门”（即对应的Calculator类）中调用，确保了整个
                     计算流程的正确性和模块化。
        - 核心重构: 重新梳理并确立了正确的计算顺序，确保了指标间的依赖关系得到满足。
                     例如，先计算基础的能量和微观动力学指标，再计算依赖其输出的市场剖面和
                     控盘指标，保证了数据流的逻辑自洽。
        """
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
        }
        # 修正了所有计算函数的调用，使其“宗门归位”
        # 1. 基础层：计算最底层的能量、微观动力学和订单流指标
        energy_metrics = StructuralMetricsCalculators.calculate_energy_density_metrics(context)
        context.update(energy_metrics)
        microstructure_metrics = MicrostructureDynamicsCalculators.calculate_all(context)
        context.update(microstructure_metrics)
        # 2. 结构层：计算市场剖面、控盘、博弈效率等核心结构指标
        profile_metrics = ThematicMetricsCalculators.calculate_market_profile_metrics(context)
        context.update(profile_metrics)
        control_metrics = StructuralMetricsCalculators.calculate_control_metrics(context)
        context.update(control_metrics)
        game_metrics = StructuralMetricsCalculators.calculate_game_efficiency_metrics(context)
        context.update(game_metrics)
        # 3. 策略层：计算前瞻性、战场博弈等策略意图指标
        forward_metrics = ThematicMetricsCalculators.calculate_forward_looking_metrics(context)
        context.update(forward_metrics)
        battlefield_metrics = ThematicMetricsCalculators.calculate_battlefield_metrics(context)
        context.update(battlefield_metrics)
        # 4. 衍生层：计算背离等衍生指标
        derivative_metrics = DerivativeMetricsCalculator.calculate_divergence_metrics(context)
        context.update(derivative_metrics)
        # 5. 整合所有宗门的计算结果
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










