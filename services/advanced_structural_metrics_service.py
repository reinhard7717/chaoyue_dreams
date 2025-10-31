# services\advanced_structural_metrics_service.py
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from functools import reduce
from django.db import transaction
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

class AdvancedStructuralMetricsService:
    """
    【V1.0 · 结构与行为锻造中心】
    - 核心职责: 封装所有高级结构与行为指标的加载、计算、融合与存储逻辑。
                利用分钟级数据，为日线级别锻造高保真的微观结构DNA指标。
    - 架构模式: 借鉴 AdvancedFundFlowMetricsService 的成功经验，实现业务逻辑与任务调度的完全解耦。
    """
    def __init__(self):
        """初始化服务，设定回溯期等基础参数"""
        self.max_lookback_days = 300 # 为计算衍生指标所需的最大历史回溯天数

    async def run_precomputation(self, stock_code: str, is_incremental: bool, start_date_str: str = None):
        """
        【V1.2 · 原料审计版】高级结构与行为指标预计算总指挥
        - 核心升级: 在锻造指标前，增加对分钟级原料数据的存在性校验。
        """
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental, start_date_str
        )
        if not is_incremental_final:
            await sync_to_async(MetricsModel.objects.filter(stock=stock_info).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        else:
            rollback_start_date = fetch_start_date if fetch_start_date else start_date_str
            if rollback_start_date:
                await sync_to_async(MetricsModel.objects.filter(stock=stock_info, trade_time__gte=rollback_start_date).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info, trade_time__gte=fetch_start_date).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if dates_to_process.empty:
            logger.info(f"[{stock_code}] [结构指标] 没有需要处理的日期，任务结束。")
            return 0
        initial_history_end_date = dates_to_process.min()
        historical_metrics_df = await self._load_historical_metrics(MetricsModel, stock_info, initial_history_end_date)
        CHUNK_SIZE = 50
        all_new_core_metrics_df = pd.DataFrame()
        for i in range(0, len(dates_to_process), CHUNK_SIZE):
            chunk_dates = dates_to_process[i:i + CHUNK_SIZE]
            if chunk_dates.empty:
                continue
            minute_data_map = await self._load_minute_data_for_range(stock_info, chunk_dates.min(), chunk_dates.max())
            # 审计分钟数据是否覆盖了区块内的所有日期
            processed_dates_in_chunk = set(minute_data_map.keys())
            required_dates_in_chunk = set(chunk_dates.date)
            missing_dates = required_dates_in_chunk - processed_dates_in_chunk
            if missing_dates:
                for missing_date in sorted(list(missing_dates)):
                    logger.warning(f"[{stock_code}] [{missing_date}] 跳过结构指标计算，缺失当日全部的分钟数据。")
            if not minute_data_map:
                logger.warning(f"[{stock_code}] 区块 {chunk_dates.min().date()} to {chunk_dates.max().date()} 无任何分钟数据，跳过整个区块。")
                continue
            
            # 将 stock_code 传递给锻造引擎
            chunk_new_metrics_df = self._forge_advanced_structural_metrics(minute_data_map, stock_code)
            
            all_new_core_metrics_df = pd.concat([all_new_core_metrics_df, chunk_new_metrics_df])
        if all_new_core_metrics_df.empty:
            logger.info(f"[{stock_code}] [结构指标] 未能计算出任何新的核心指标，任务结束。")
            return 0
        full_sequence_for_derivatives = pd.concat([historical_metrics_df, all_new_core_metrics_df])
        full_sequence_for_derivatives.sort_index(inplace=True)
        final_metrics_df = self._calculate_derivatives(stock_code, full_sequence_for_derivatives)
        chunk_to_save = final_metrics_df[final_metrics_df.index.isin(all_new_core_metrics_df.index)]
        total_processed_count = await self._prepare_and_save_data(stock_info, MetricsModel, chunk_to_save)
        logger.info(f"[{stock_code}] [结构指标] 成功处理并保存了 {total_processed_count} 条高级结构与行为指标。")
        return total_processed_count

    async def _initialize_context(self, stock_code: str, is_incremental: bool, start_date_str: str = None):
        """
        【V1.0 · 新增】初始化计算上下文，确定股票实体、目标模型、计算模式和日期范围。
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

    async def _load_minute_data_for_range(self, stock_info: StockInfo, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
        """
        【V1.4 · 升序修正版】一次性加载指定日期范围内的所有分钟线数据，并按日期分组。
        - 核心修复: 查询时强制使用 .order_by('trade_time')，确保从数据库源头获取升序排列的数据。
        """
        from django.utils import timezone
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        if not MinuteModel:
            return {}
        @sync_to_async(thread_sensitive=True)
        def get_data(model, stock_pk, start_dt, end_dt):
            # 强制按交易时间升序排序
            qs = model.objects.filter(
                stock_id=stock_pk,
                trade_time__gte=start_dt,
                trade_time__lt=end_dt
            ).values('trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount').order_by('trade_time')
            
            return pd.DataFrame.from_records(qs)
        start_datetime = timezone.make_aware(datetime.combine(start_date, time.min))
        end_datetime = timezone.make_aware(datetime.combine(end_date, time.max))
        minute_df = await get_data(MinuteModel, stock_info.pk, start_datetime, end_datetime)
        if minute_df.empty:
            return {}
        minute_df['trade_time'] = pd.to_datetime(minute_df['trade_time'])
        if minute_df['trade_time'].dt.tz is not None:
            minute_df['trade_time'] = minute_df['trade_time'].dt.tz_convert('Asia/Shanghai')
        else:
            minute_df['trade_time'] = minute_df['trade_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        cols_to_float = ['open', 'high', 'low', 'close', 'amount', 'vol']
        for col in cols_to_float:
            if col in minute_df.columns:
                minute_df[col] = pd.to_numeric(minute_df[col], errors='coerce')
        minute_df['date'] = minute_df['trade_time'].dt.date
        return {date: group_df for date, group_df in minute_df.groupby('date')}

    def _forge_advanced_structural_metrics(self, minute_data_map: dict, stock_code: str) -> pd.DataFrame:
        """
        【V5.0 · 战场动力学整合版】
        - 核心革命: 引入 `_calculate_intraday_battlefield_dynamics` 统一计算单元，实现对日内博弈的深度刻画。
        - 核心升级: 将 `rebound_momentum`, `high_level_consolidation_volume`, `opening_period_thrust` 等全新指标整合进最终结果。
        """
        # [代码修改开始]
        if not minute_data_map:
            return pd.DataFrame()
        daily_metrics = []
        all_dates = sorted(minute_data_map.keys())
        from stock_models.time_trade import StockDailyBasic
        DailyModel = get_daily_data_model_by_code(stock_code)
        history_start_date = min(all_dates) - timedelta(days=50)
        daily_data_qs = DailyModel.objects.filter(
            stock__stock_code=stock_code,
            trade_time__gte=history_start_date,
            trade_time__lte=max(all_dates)
        ).values('trade_time', 'high', 'low', 'open', 'close', 'pre_close', 'high_qfq', 'low_qfq', 'close_qfq', 'pre_close_qfq').order_by('trade_time')
        daily_df = pd.DataFrame.from_records(list(daily_data_qs))
        if daily_df.empty:
            logger.warning(f"[{stock_code}] 无法加载日线行情数据，部分指标无法计算。")
            return pd.DataFrame()
        daily_df['trade_time'] = pd.to_datetime(daily_df['trade_time']).dt.date
        daily_df = daily_df.set_index('trade_time')
        daily_basic_qs = StockDailyBasic.objects.filter(
            stock__stock_code=stock_code,
            trade_time__gte=history_start_date,
            trade_time__lte=max(all_dates)
        ).values('trade_time', 'turnover_rate_f').order_by('trade_time')
        daily_basic_df = pd.DataFrame.from_records(list(daily_basic_qs))
        if not daily_basic_df.empty:
            daily_basic_df['trade_time'] = pd.to_datetime(daily_basic_df['trade_time']).dt.date
            daily_basic_df = daily_basic_df.set_index('trade_time')
            daily_df = daily_df.join(daily_basic_df, how='left')
        daily_df.ta.atr(high=daily_df['high_qfq'], low=daily_df['low_qfq'], close=daily_df['close_qfq'], length=14, append=True)
        base_atr = daily_df['ATRr_14']
        for date, group in minute_data_map.items():
            if group.empty or len(group) < 10:
                logger.warning(f"[{stock_code}] [{date}] 跳过结构指标计算，分钟数据不足。")
                continue
            try:
                daily_series_for_day = daily_df.loc[date]
                base_atr_for_day = base_atr.get(date)
            except KeyError:
                logger.warning(f"[{stock_code}] [{date}] 无法找到对应的日线数据或ATR，跳过当天计算。")
                continue
            group = group.sort_values(by='trade_time').reset_index(drop=True)
            total_volume = group['vol'].sum()
            total_volume_safe = total_volume if total_volume > 0 else np.nan
            # --- 调用全新的“能量密度”和“战场动力学”计算核心 ---
            energy_metrics = self._calculate_energy_density_metrics(daily_series_for_day, group, base_atr_for_day)
            battlefield_metrics = self._calculate_intraday_battlefield_dynamics(group, total_volume)
            # --- 计算其他维度的指标 (保留原有逻辑) ---
            continuous_mask = group['trade_time'].dt.time < time(14, 57, 0)
            continuous_group = group[continuous_mask].copy()
            if continuous_group.empty: continue
            day_open, day_high, day_low, day_close = group['open'].iloc[0], group['high'].max(), group['low'].min(), group['close'].iloc[-1]
            continuous_group['minute_vwap'] = continuous_group['amount'] / continuous_group['vol'].replace(0, np.nan)
            continuous_group = continuous_group.fillna(method='ffill').fillna(day_open)
            daily_vwap = (continuous_group['amount']).sum() / continuous_group['vol'].sum() if continuous_group['vol'].sum() > 0 else day_close
            vwap_dev = (continuous_group['minute_vwap'] - daily_vwap) * continuous_group['vol']
            vwap_deviation_area = vwap_dev.sum() / (daily_vwap * total_volume_safe) if pd.notna(daily_vwap) and daily_vwap > 0 and pd.notna(total_volume_safe) else 0
            trend_persistence_index = self._calculate_hurst(continuous_group['minute_vwap'])
            time_series = continuous_group['trade_time']
            seconds_from_start = (time_series - time_series.iloc[0]).dt.total_seconds()
            seconds_from_start[time_series.dt.time >= time(13,0)] -= 5400
            weighted_time = (seconds_from_start * continuous_group['vol']).sum()
            volume_weighted_time_index = weighted_time / (14400 * total_volume_safe) if pd.notna(total_volume_safe) else 0.5
            price_change_sign = np.sign(continuous_group['minute_vwap'].diff().fillna(0))
            vpa_efficiency = (price_change_sign * continuous_group['vol']).sum() / total_volume_safe if pd.notna(total_volume_safe) else 0
            vp = continuous_group.groupby(pd.cut(continuous_group['close'], bins=20))['vol'].sum()
            vpoc_interval = vp.idxmax() if not vp.empty else np.nan
            vpoc = vpoc_interval.mid if pd.notna(vpoc_interval) else day_close
            intraday_vah, intraday_val = self._calculate_value_area(vp, continuous_group['vol'].sum(), vpoc_interval)
            is_bullish_divergence, is_bearish_divergence = self._detect_intraday_divergence(continuous_group)
            am_df = group[group['trade_time'].dt.time < pd.to_datetime('12:00').time()]
            pm_df = group[group['trade_time'].dt.time >= pd.to_datetime('13:00').time()]
            am_vol_sum = am_df['vol'].sum()
            pm_vol_sum = pm_df['vol'].sum()
            vwap_am = am_df['amount'].sum() / am_vol_sum if am_vol_sum > 0 else 0
            vwap_pm = pm_df['amount'].sum() / pm_vol_sum if pm_vol_sum > 0 else 0
            am_pm_vwap_ratio_val = (vwap_pm / vwap_am - 1) * 100 if vwap_am > 0 and vwap_pm > 0 else np.nan
            # --- 组装所有指标 ---
            day_metric_dict = {
                'trade_time': date,
                **energy_metrics,
                **battlefield_metrics, # 注入全新的战场动力学指标
                'vwap_deviation_area': vwap_deviation_area,
                'trend_persistence_index': trend_persistence_index,
                'volume_weighted_time_index': volume_weighted_time_index,
                'vpa_efficiency': vpa_efficiency,
                'intraday_vpoc': vpoc,
                'intraday_vah': intraday_vah,
                'intraday_val': intraday_val,
                'upper_shadow_volume_ratio': group[group['high'] > max(day_open, day_close)]['vol'].sum() / total_volume_safe if pd.notna(total_volume_safe) else 0,
                'lower_shadow_volume_ratio': group[group['low'] < min(day_open, day_close)]['vol'].sum() / total_volume_safe if pd.notna(total_volume_safe) else 0,
                'true_daily_cmf': (((group['close'] - group['low']) - (group['high'] - group['close'])) / (group['high'] - group['low']).replace(0, np.nan)).fillna(0).dot(group['vol']) / total_volume_safe if pd.notna(total_volume_safe) else 0,
                'intraday_trend_efficiency': abs(day_close - day_open) / continuous_group['minute_vwap'].diff().abs().sum() if continuous_group['minute_vwap'].diff().abs().sum() > 0 else 0,
                'intraday_reversal_intensity': ((day_close - day_low) - (day_high - day_close)) / (day_high - day_low) if (day_high - day_low) > 0 else 0,
                'close_vs_vpoc_ratio': day_close / vpoc if vpoc > 0 else 1.0,
                'am_pm_volume_ratio': pm_vol_sum / am_vol_sum if am_vol_sum > 0 else np.nan,
                'am_pm_vwap_ratio': am_pm_vwap_ratio_val,
                'is_intraday_bullish_divergence': is_bullish_divergence,
                'is_intraday_bearish_divergence': is_bearish_divergence,
            }
            daily_metrics.append(day_metric_dict)
        if not daily_metrics:
            return pd.DataFrame()
        result_df = pd.DataFrame(daily_metrics)
        return result_df.set_index(pd.to_datetime(result_df['trade_time']))
        # [代码修改结束]

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

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        """
        【V1.0 · 新增】准备数据并以原子方式批量保存到数据库。
        """
        if final_df.empty:
            return 0
        from django.db.models import DecimalField
        from decimal import Decimal, ROUND_HALF_UP
        # 替换无穷大值为NaN
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 筛选出模型中存在的列
        model_fields = {f.name for f in MetricsModel._meta.get_fields() if not f.is_relation and f.name != 'id'}
        df_filtered = final_df[[col for col in final_df.columns if col in model_fields]]
        # 转换为字典列表
        records_list = df_filtered.to_dict('records')
        @sync_to_async(thread_sensitive=True)
        def save_atomically(model, stock_obj, records_to_process):
            processed_count = 0
            # 使用 Django 的 bulk_create 提高性能
            objs_to_create = []
            for record_data in records_to_process:
                # 确保 trade_time 是 date 对象
                trade_time = record_data.pop('trade_time').date()
                # 清理 NaN 值
                defaults_data = {key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in record_data.items()}
                
                objs_to_create.append(
                    model(
                        stock=stock_obj,
                        trade_time=trade_time,
                        **defaults_data
                    )
                )
            try:
                # 批量创建，忽略冲突（因为我们已经在前面删除了）
                model.objects.bulk_create(objs_to_create, ignore_conflicts=True)
                processed_count = len(objs_to_create)
            except Exception as e:
                logger.error(f"[{stock_obj.stock_code}] [结构指标批量保存失败] 错误: {e}")
            return processed_count
        records_for_atomic_save = []
        for record_date, record_data in zip(df_filtered.index, records_list):
            record_data['trade_time'] = record_date
            records_for_atomic_save.append(record_data)
            
        processed_count = await save_atomically(MetricsModel, stock_info, records_for_atomic_save)
        return processed_count

    async def _load_historical_metrics(self, model, stock_info, end_date):
        """
        【V1.0 · 新增】从数据库加载并净化历史高级结构指标。
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
            df = df.set_index(pd.to_datetime(df['trade_time']))
            # 在数据源头进行类型净化，杜绝object类型污染
            for col in df.columns:
                if col != 'trade_time':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _calculate_value_area(self, vp: pd.Series, total_volume: float, vpoc_interval: pd.Interval) -> tuple:
        """
        【V2.3 · 优化】计算日内价值区域 (VAH/VAL)
        - 优化点: 确保 VAH/VAL 边界的精确性。
        """
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
        # 确保边界是价格区间的精确边界
        val = vp_sorted_by_price.index[low_idx].left
        vah = vp_sorted_by_price.index[high_idx].right
        
        return vah, val

    def _detect_intraday_divergence(self, group: pd.DataFrame) -> tuple:
        """
        【V2.3 · 优化】检测日内价格与RSI的背离
        - 优化点: 
            1. 使用分钟VWAP作为价格代理，更具代表性。
            2. 引入 scipy.signal.find_peaks 进行更可靠的波峰波谷检测。
            3. 移除冗余的 minute_vwap 计算。
        """
        from scipy.signal import find_peaks
        from itertools import combinations
        is_bullish_divergence = False
        is_bearish_divergence = False
        if len(group) < 30:
            return is_bullish_divergence, is_bearish_divergence
        # 直接使用主函数计算好的 minute_vwap
        price_series = group['minute_vwap']
        
        rsi_series = ta.rsi(price_series, length=14).dropna()
        if rsi_series.empty:
            return is_bullish_divergence, is_bearish_divergence
        # 对齐价格和RSI序列
        aligned_price = price_series.loc[rsi_series.index]
        aligned_rsi = rsi_series
        # 使用 find_peaks 寻找波峰和波谷
        price_low_indices, _ = find_peaks(-aligned_price.values, distance=10, prominence=0.005) # 至少间隔10分钟，显著性0.5%
        rsi_low_indices, _ = find_peaks(-aligned_rsi.values, distance=10, prominence=1.0) # RSI显著性1.0
        price_high_indices, _ = find_peaks(aligned_price.values, distance=10, prominence=0.005)
        rsi_high_indices, _ = find_peaks(aligned_rsi.values, distance=10, prominence=1.0)
        # 将索引转换为时间戳索引
        price_lows = aligned_price.iloc[price_low_indices]
        rsi_lows = aligned_rsi.iloc[rsi_low_indices]
        price_highs = aligned_price.iloc[price_high_indices]
        rsi_highs = aligned_rsi.iloc[rsi_high_indices]
        # --- 检测底部背离 (价格创新低，RSI未创新低) ---
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            for (t1, p1), (t2, p2) in combinations(price_lows.items(), 2):
                if t2 < t1: continue
                if p2 < p1:
                    rsi_l1 = rsi_lows[rsi_lows.index <= t1].iloc[-1] if not rsi_lows[rsi_lows.index <= t1].empty else np.nan
                    rsi_l2 = rsi_lows[rsi_lows.index >= t2].iloc[0] if not rsi_lows[rsi_lows.index >= t2].empty else np.nan
                    if pd.notna(rsi_l1) and pd.notna(rsi_l2) and rsi_l2 > rsi_l1:
                        is_bullish_divergence = True
                        break
        # --- 检测顶部背离 (价格创新高，RSI未创新高) ---
        if not is_bullish_divergence and len(price_highs) >= 2 and len(rsi_highs) >= 2:
            for (t1, p1), (t2, p2) in combinations(price_highs.items(), 2):
                if t2 < t1: continue
                if p2 > p1:
                    rsi_h1 = rsi_highs[rsi_highs.index <= t1].iloc[-1] if not rsi_highs[rsi_highs.index <= t1].empty else np.nan
                    rsi_h2 = rsi_highs[rsi_highs.index >= t2].iloc[0] if not rsi_highs[rsi_highs.index >= t2].empty else np.nan
                    if pd.notna(rsi_h1) and pd.notna(rsi_h2) and rsi_h2 < rsi_h1:
                        is_bearish_divergence = True
                        break
        return is_bullish_divergence, is_bearish_divergence

    def _calculate_gini(self, array: np.ndarray) -> float:
        """计算基尼系数"""
        # [代码新增开始]
        if array is None or len(array) < 2 or np.sum(array) == 0:
            return 0.0
        sorted_array = np.sort(array)
        n = len(array)
        cum_array = np.cumsum(sorted_array, dtype=float)
        return (n + 1 - 2 * np.sum(cum_array) / cum_array[-1]) / n
        # [代码新增结束]

    def _calculate_hurst(self, series: pd.Series) -> float:
        """计算Hurst指数"""
        # [代码新增开始]
        if len(series) < 100:
            return np.nan
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
        # [代码新增结束]

    def _calculate_energy_density_metrics(self, daily_series: pd.Series, minute_df: pd.DataFrame, atr_14d: float) -> dict:
        """
        【V1.0 · 新增 · 能量密度合成计算单元】
        整合所有与日内“能量”相关的核心指标计算。
        """
        # [代码新增开始]
        # 准备分钟数据，确保数据清洁
        if minute_df is None or minute_df.empty:
            return {}
        
        # 确保分钟数据按时间排序
        minute_df = minute_df.sort_values(by='trade_time').reset_index(drop=True)
        
        # 预计算分钟VWAP，如果不存在
        if 'minute_vwap' not in minute_df.columns:
            minute_df['minute_vwap'] = minute_df['amount'] / minute_df['vol'].replace(0, np.nan)
            minute_df['minute_vwap'].fillna(method='ffill', inplace=True)

        # 调用各个重构后的计算函数
        energy_density = self._calculate_energy_density(daily_series, atr_14d)
        thrust_purity = self._calculate_thrust_purity(minute_df)
        volume_burstiness = self._calculate_volume_burstiness(minute_df)
        auction_impact = self._calculate_auction_impact(minute_df, atr_14d)

        # 组装结果字典
        results = {
            'intraday_energy_density': energy_density,
            'intraday_thrust_purity': thrust_purity,
            'volume_burstiness_index': volume_burstiness,
            'auction_impact_score': auction_impact,
        }
        return results
        # [代码新增结束]

    def _calculate_intraday_battlefield_dynamics(self, minute_df: pd.DataFrame, total_daily_volume: float) -> dict:
        """
        【V1.0 · 新增 · 日内战场动力学模型】
        - 核心思想: 引入反转韧性、高位换手意愿、开盘突袭纯度三个指标，从能量的释放方式、方向和效率上深度刻画日内博弈。
        - 新增指标:
          - rebound_momentum: 反转动能。量化下跌被承接并转为反攻的动能强度。
          - high_level_consolidation_volume: 高位整固成交量占比。衡量市场对高价区的接受度和认可度。
          - opening_period_thrust: 开盘期推力。判断开盘30分钟的真实意图和量价合力方向。
        """
        # [代码新增开始]
        results = {
            'rebound_momentum': 0.0,
            'high_level_consolidation_volume': 0.0,
            'opening_period_thrust': 0.0,
        }
        if minute_df.empty or total_daily_volume == 0:
            return results
        # --- 1. 计算反转动能 (Rebound Momentum) ---
        # 找到日内最低点时刻
        low_idx = minute_df['low'].idxmin()
        if low_idx > 0 and low_idx < len(minute_df) - 1:
            # 分割为下跌阶段和反弹阶段
            falling_phase = minute_df.iloc[:low_idx+1]
            rebounding_phase = minute_df.iloc[low_idx+1:]
            # 计算各阶段的VWAP和成交量占比
            vwap_fall = (falling_phase['amount']).sum() / falling_phase['vol'].sum() if falling_phase['vol'].sum() > 0 else np.nan
            vwap_rebound = (rebounding_phase['amount']).sum() / rebounding_phase['vol'].sum() if rebounding_phase['vol'].sum() > 0 else np.nan
            vol_ratio_rebound = rebounding_phase['vol'].sum() / total_daily_volume
            if pd.notna(vwap_fall) and pd.notna(vwap_rebound) and vwap_fall > 0:
                # 反转动能 = (反弹VWAP / 下跌VWAP - 1) * 反弹成交量占比
                # 这个公式同时捕捉了反弹的“价格力度”和“成交量参与度”
                results['rebound_momentum'] = (vwap_rebound / vwap_fall - 1) * vol_ratio_rebound * 100
        # --- 2. 计算高位整固成交量 (High-Level Consolidation Volume) ---
        day_high = minute_df['high'].max()
        day_low = minute_df['low'].min()
        price_range = day_high - day_low
        if price_range > 0:
            # 定义高位区域为当天振幅的顶部25%区间
            high_level_threshold = day_high - 0.25 * price_range
            high_level_volume = minute_df[minute_df['high'] >= high_level_threshold]['vol'].sum()
            results['high_level_consolidation_volume'] = high_level_volume / total_daily_volume
        # --- 3. 计算开盘期推力 (Opening Period Thrust) ---
        opening_period_df = minute_df[minute_df['trade_time'].dt.time < time(9, 59, 59)]
        if not opening_period_df.empty:
            # 复用“推力纯度”的计算逻辑，但仅应用于开盘30分钟
            opening_thrust_vector = (opening_period_df['close'] - opening_period_df['open']) * opening_period_df['vol']
            opening_absolute_energy = abs(opening_period_df['close'] - opening_period_df['open']) * opening_period_df['vol']
            total_opening_energy = opening_absolute_energy.sum()
            if total_opening_energy > 0:
                results['opening_period_thrust'] = opening_thrust_vector.sum() / total_opening_energy
        return results
        # [代码新增结束]







