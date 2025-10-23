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
        【V1.1 · 标准化流程版】高级结构与行为指标预计算总指挥
        - 核心升级: 调整了分钟数据的加载和处理流程，与筹码/资金流服务完全对齐。
                      现在一次性加载区块数据并预分组为字典，再传递给锻造引擎。
        """
        # 步骤 1: 初始化上下文，确定计算范围和目标数据表
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental, start_date_str
        )
        # 步骤 2: 根据模式（全量/增量）确定需要处理的日期列表
        if not is_incremental_final:
            # 全量模式：删除旧数据，获取该股票所有历史交易日
            await sync_to_async(MetricsModel.objects.filter(stock=stock_info).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        else:
            # 增量/部分全量模式：删除指定日期之后的数据，获取需要重新计算的日期范围
            rollback_start_date = fetch_start_date if fetch_start_date else start_date_str
            if rollback_start_date:
                await sync_to_async(MetricsModel.objects.filter(stock=stock_info, trade_time__gte=rollback_start_date).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info, trade_time__gte=fetch_start_date).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if dates_to_process.empty:
            logger.info(f"[{stock_code}] [结构指标] 没有需要处理的日期，任务结束。")
            return 0
        # 步骤 3: 加载历史指标数据，为计算衍生指标（斜率、加速度）做准备
        initial_history_end_date = dates_to_process.min()
        historical_metrics_df = await self._load_historical_metrics(MetricsModel, stock_info, initial_history_end_date)
        # 步骤 4: 分块处理，避免一次性加载过多分钟数据到内存
        CHUNK_SIZE = 50 # 每次处理50个交易日
        all_new_core_metrics_df = pd.DataFrame()
        for i in range(0, len(dates_to_process), CHUNK_SIZE):
            chunk_dates = dates_to_process[i:i + CHUNK_SIZE]
            if chunk_dates.empty:
                continue
            # [代码修改开始]
            # 步骤 4.1: 为当前块加载所需的分钟级原始数据，并预分组为字典
            minute_data_map = await self._load_minute_data_for_range(stock_info, chunk_dates.min(), chunk_dates.max())
            if not minute_data_map:
                continue
            # 步骤 4.2: 调用核心锻造引擎，计算高保真结构指标
            chunk_new_metrics_df = self._forge_advanced_structural_metrics(minute_data_map)
            # [代码修改结束]
            all_new_core_metrics_df = pd.concat([all_new_core_metrics_df, chunk_new_metrics_df])
        if all_new_core_metrics_df.empty:
            logger.info(f"[{stock_code}] [结构指标] 未能计算出任何新的核心指标，任务结束。")
            return 0
        # 步骤 5: 拼接历史数据与新数据，为计算衍生指标构建完整序列
        full_sequence_for_derivatives = pd.concat([historical_metrics_df, all_new_core_metrics_df])
        full_sequence_for_derivatives.sort_index(inplace=True)
        # 步骤 6: 计算所有核心指标的斜率和加速度
        final_metrics_df = self._calculate_derivatives(stock_code, full_sequence_for_derivatives)
        # 步骤 7: 筛选出本次计算的新数据，并存入数据库
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
        【V1.2 · 标准化加载版】一次性加载指定日期范围内的所有分钟线数据，并按日期分组。
        - 核心升级: 与筹码/资金流服务的数据加载模式完全对齐，返回一个按日期分组的字典。
        - 核心修复: 在数据加载后，立即将所有价格和金额相关的列强制转换为float类型，
                      从源头杜绝 decimal.Decimal 与 float 混合运算导致的 TypeError。
        """
        from django.utils import timezone
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        if not MinuteModel:
            return {}
        @sync_to_async(thread_sensitive=True)
        def get_data(model, stock_pk, start_dt, end_dt):
            qs = model.objects.filter(
                stock_id=stock_pk,
                trade_time__gte=start_dt,
                trade_time__lt=end_dt
            ).values('trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount')
            return pd.DataFrame.from_records(qs)
        start_datetime = timezone.make_aware(datetime.combine(start_date, time.min))
        end_datetime = timezone.make_aware(datetime.combine(end_date, time.max))
        minute_df = await get_data(MinuteModel, stock_info.pk, start_datetime, end_datetime)
        if minute_df.empty:
            return {}
        minute_df['trade_time'] = pd.to_datetime(minute_df['trade_time'])
        # 实施“数据类型净化协议”
        cols_to_float = ['open', 'high', 'low', 'close', 'amount', 'vol']
        for col in cols_to_float:
            if col in minute_df.columns:
                minute_df[col] = pd.to_numeric(minute_df[col], errors='coerce')
        minute_df['date'] = minute_df['trade_time'].dt.date
        return {date: group_df for date, group_df in minute_df.groupby('date')}

    def _forge_advanced_structural_metrics(self, minute_data_map: dict) -> pd.DataFrame:
        """
        【V2.1 · 标准化输入版】高级结构与行为指标锻造核心引擎
        - 核心升级: 输入参数从单个DataFrame变更为按日期分组的字典 `minute_data_map`，
                      与筹码/资金流服务的锻造引擎接口完全对齐。
        """
        if not minute_data_map:
            return pd.DataFrame()
        daily_metrics = []
        for date, group in minute_data_map.items():
            if group.empty or len(group) < 10: # 至少需要10分钟数据
                continue
            # --- 基础变量计算 ---
            day_open, day_high, day_low, day_close = group['open'].iloc[0], group['high'].max(), group['low'].min(), group['close'].iloc[-1]
            day_range = day_high - day_low
            day_range_safe = day_range if day_range > 0 else np.nan
            total_volume = group['vol'].sum()
            total_volume_safe = total_volume if total_volume > 0 else np.nan
            # --- V1.0 指标计算 (略作优化) ---
            vwap = (group['amount']).sum() / total_volume_safe if pd.notna(total_volume_safe) else day_close
            vwap_pos = (vwap - day_low) / day_range_safe if pd.notna(day_range_safe) else 0.5
            body_high, body_low = max(day_open, day_close), min(day_open, day_close)
            upper_shadow_mask, lower_shadow_mask = group['high'] > body_high, group['low'] < body_low
            upper_shadow_vol_ratio = group[upper_shadow_mask]['vol'].sum() / total_volume_safe if pd.notna(total_volume_safe) else 0
            lower_shadow_vol_ratio = group[lower_shadow_mask]['vol'].sum() / total_volume_safe if pd.notna(total_volume_safe) else 0
            mfm = ((group['close'] - group['low']) - (group['high'] - group['close'])) / (group['high'] - group['low']).replace(0, np.nan)
            true_daily_cmf = (mfm.fillna(0) * group['vol']).sum() / total_volume_safe if pd.notna(total_volume_safe) else 0
            path_length = group['close'].diff().abs().sum()
            intraday_trend_efficiency = abs(day_close - day_open) / path_length if path_length > 0 else 0
            intraday_reversal_intensity = ((day_close - day_low) - (day_high - day_close)) / day_range_safe if pd.notna(day_range_safe) else 0
            # --- V2.0 指标计算 ---
            # 1. 日内价值区间 (简易实现)
            vp = group.groupby(pd.cut(group['close'], bins=20))['vol'].sum()
            vpoc = vp.idxmax().mid if not vp.empty else day_close
            close_vs_vpoc_ratio = day_close / vpoc if vpoc > 0 else 1.0
            # 2. 上下午成交量/VWAP比
            am_mask = group['trade_time'].dt.hour < 12
            pm_mask = ~am_mask
            am_vol = group[am_mask]['vol'].sum()
            pm_vol = group[pm_mask]['vol'].sum()
            am_pm_volume_ratio = pm_vol / am_vol if am_vol > 0 else np.nan
            am_vwap = (group[am_mask]['amount']).sum() / am_vol if am_vol > 0 else np.nan
            pm_vwap = (group[pm_mask]['amount']).sum() / pm_vol if pm_vol > 0 else np.nan
            am_pm_vwap_ratio = pm_vwap / am_vwap if pd.notna(am_vwap) and am_vwap > 0 else np.nan
            # 3. 日内成交量基尼系数
            vol_array = group['vol'].dropna().values
            if len(vol_array) > 1:
                sorted_vol = np.sort(vol_array)
                cum_vol = np.cumsum(sorted_vol)
                n = len(vol_array)
                intraday_volume_gini = (n + 1 - 2 * np.sum(cum_vol) / cum_vol[-1]) / n if cum_vol[-1] > 0 else 0
            else:
                intraday_volume_gini = 0
            # 4. 日内趋势线性度(R²)
            x = np.arange(len(group))
            y = group['close'].values
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            intraday_trend_linearity = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
            # 5. 成交量加权时间指数
            time_as_float = (group['trade_time'] - group['trade_time'].iloc[0]).dt.total_seconds()
            volume_weighted_time_index = (time_as_float * group['vol']).sum() / (time_as_float.max() * total_volume_safe) if pd.notna(total_volume_safe) and time_as_float.max() > 0 else 0.5
            daily_metrics.append({
                'trade_time': date,
                'volume_weighted_close_position': vwap_pos,
                'upper_shadow_volume_ratio': upper_shadow_vol_ratio,
                'lower_shadow_volume_ratio': lower_shadow_vol_ratio,
                'true_daily_cmf': true_daily_cmf,
                'intraday_trend_efficiency': intraday_trend_efficiency,
                'intraday_reversal_intensity': intraday_reversal_intensity,
                'intraday_vpoc': vpoc,
                'intraday_vah': np.nan, # 完整实现较复杂，暂留空
                'intraday_val': np.nan, # 完整实现较复杂，暂留空
                'close_vs_vpoc_ratio': close_vs_vpoc_ratio,
                'am_pm_volume_ratio': am_pm_volume_ratio,
                'am_pm_vwap_ratio': am_pm_vwap_ratio,
                'intraday_volume_gini': intraday_volume_gini,
                'intraday_trend_linearity': intraday_trend_linearity,
                'volume_weighted_time_index': volume_weighted_time_index,
                'is_intraday_bullish_divergence': False, # 完整实现较复杂，暂留空
                'is_intraday_bearish_divergence': False, # 完整实现较复杂，暂留空
            })
        if not daily_metrics:
            return pd.DataFrame()
        return pd.DataFrame(daily_metrics).set_index(pd.to_datetime(pd.DataFrame(daily_metrics)['trade_time']))

    def _calculate_derivatives(self, stock_code: str, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 · 新增】为所有核心结构指标计算斜率和加速度。
        """
        derivatives_df = pd.DataFrame(index=metrics_df.index)
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedStructuralMetrics.CORE_METRICS.keys())
        ACCEL_WINDOW = 2
        UNIFIED_PERIODS = [1, 5, 13, 21, 55]

        for col in CORE_METRICS_TO_DERIVE:
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
