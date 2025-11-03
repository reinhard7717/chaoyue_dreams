# services/advanced_chip_metrics_service.py
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from functools import reduce
from django.db import transaction
from asgiref.sync import sync_to_async
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockDailyBasic, StockCyqPerf
from stock_models.advanced_metrics import BaseAdvancedChipMetrics
from utils.model_helpers import (
    get_advanced_chip_metrics_model_by_code,
    get_cyq_chips_model_by_code,
    get_daily_data_model_by_code,
    get_minute_data_model_by_code_and_timelevel,
)
from services.chip_feature_calculator import ChipFeatureCalculator
import logging
logger = logging.getLogger(__name__)

class AdvancedChipMetricsService:
    """
    【V1.0 · 兵工厂模式】高级筹码指标服务
    - 核心职责: 封装所有高级筹码指标的加载、计算、融合与存储逻辑。
    - 架构优势: 实现业务逻辑与任务调度的完全解耦，并为“深度融合”架构提供标准接口。
    """
    def __init__(self):
        self.max_lookback_days = 200

    async def run_precomputation(self, stock_code: str, is_incremental: bool, start_date_str: str = None, fund_flow_attributed_minute_map: dict = None):
        """
        【V1.0 · 融合接口版】服务层主执行器。
        - 新增: 接收由资金流服务预先计算好的 `fund_flow_attributed_minute_map`。
        """
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental, start_date_str
        )
        # 统一以日线数据确定处理范围
        DailyModel = get_daily_data_model_by_code(stock_code)
        date_filter = {'stock': stock_info}
        if fetch_start_date:
            date_filter['trade_time__gte'] = fetch_start_date
        if last_metric_date and is_incremental_final:
            date_filter['trade_time__gt'] = last_metric_date
        all_dates_qs = DailyModel.objects.filter(**date_filter).values_list('trade_time', flat=True).order_by('trade_time')
        dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if dates_to_process.empty:
            logger.info(f"[{stock_code}] [筹码服务] 无需计算的日期，任务终止。")
            return 0
        # 统一加载所有数据
        data_dfs = await self._load_all_sources(stock_info, dates_to_process.min(), dates_to_process.max())
        merged_df = self._preprocess_and_merge_data(stock_code, data_dfs)
        minute_data_map = await self._load_minute_data_for_range(stock_info, dates_to_process.min(), dates_to_process.max())
        # 核心指标锻造
        base_metrics_df = self._synthesize_and_forge_metrics(
            stock_info, merged_df, minute_data_map, fund_flow_attributed_minute_map
        )
        if base_metrics_df.empty:
            logger.info(f"[{stock_code}] [筹码服务] 未能计算出任何新的基础指标。")
            return 0
        # 衍生指标升维
        historical_metrics_df = await self._load_historical_metrics(MetricsModel, stock_info, base_metrics_df.index.min())
        full_sequence_df = pd.concat([historical_metrics_df, base_metrics_df]).sort_index()
        final_metrics_df = self._calculate_derivatives(full_sequence_df)
        # 精确切分并保存
        chunk_to_save = final_metrics_df[final_metrics_df.index.isin(base_metrics_df.index)]
        processed_count = await self._prepare_and_save_data(stock_info, MetricsModel, chunk_to_save)
        return processed_count

    async def _initialize_context(self, stock_code: str, is_incremental: bool, start_date_str: str = None):
        """初始化任务上下文，确定计算模式和日期范围。"""
        stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        MetricsModel = get_advanced_chip_metrics_model_by_code(stock_code)
        last_metric_date = None
        fetch_start_date = None
        if start_date_str:
            try:
                start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                is_incremental = True # 强制为增量模式，以便进行回滚删除
                last_metric_date = start_date_obj - timedelta(days=1)
                fetch_start_date = start_date_obj - timedelta(days=self.max_lookback_days)
                deleted_count, _ = await sync_to_async(MetricsModel.objects.filter(stock=stock_info, trade_time__gte=start_date_obj).delete)()
                logger.info(f"[{stock_code}] [筹码服务] 部分全量模式，已回滚删除从 {start_date_obj} 开始的 {deleted_count} 条旧指标。")
            except (ValueError, TypeError):
                logger.error(f"[{stock_code}] 提供的起始日期 '{start_date_str}' 格式错误，将忽略。")
                is_incremental = True
        if is_incremental and not start_date_str:
            latest_metric = await sync_to_async(
                lambda: MetricsModel.objects.filter(stock=stock_info).only('trade_time').latest('trade_time')
            )()
            if latest_metric:
                last_metric_date = latest_metric.trade_time
                fetch_start_date = last_metric_date - timedelta(days=self.max_lookback_days)
            else:
                is_incremental = False
        return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date

    async def _load_all_sources(self, stock_info: StockInfo, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """统一加载并审计所有筹码计算所需的数据源。"""
        @sync_to_async(thread_sensitive=True)
        def get_data_async(model, stock_info_obj, fields: tuple = None, date_field='trade_time', start_dt=None, end_dt=None):
            if not model: return pd.DataFrame()
            qs = model.objects.filter(stock=stock_info_obj, **{f'{date_field}__gte': start_dt, f'{date_field}__lte': end_dt})
            return pd.DataFrame.from_records(qs.values(*fields) if fields else qs.values())
        chip_model = get_cyq_chips_model_by_code(stock_info.stock_code)
        daily_data_model = get_daily_data_model_by_code(stock_info.stock_code)
        data_tasks = {
            "cyq_chips": get_data_async(chip_model, stock_info, fields=('trade_time', 'price', 'percent'), start_dt=start_date, end_dt=end_date),
            "daily_data": get_data_async(daily_data_model, stock_info, fields=('trade_time', 'close_qfq', 'vol', 'high_qfq', 'low_qfq'), start_dt=start_date, end_dt=end_date),
            "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=('trade_time', 'float_share', 'circ_mv'), start_dt=start_date, end_dt=end_date),
            "cyq_perf": get_data_async(StockCyqPerf, stock_info, start_dt=start_date, end_dt=end_date),
        }
        results = await asyncio.gather(*data_tasks.values())
        data_dfs = dict(zip(data_tasks.keys(), results))
        # 数据审计
        for name, df in data_dfs.items():
            if df is None or df.empty:
                raise ValueError(f"[审计失败] 核心数据源 '{name}' 在日期范围 {start_date.date()} to {end_date.date()} 为空！")
        return data_dfs

    def _preprocess_and_merge_data(self, stock_code: str, data_dfs: dict, close_map: dict, date_20d_ago_map: dict, atr_map: dict, high_20d_map: dict, low_20d_map: dict, high_5d_map: dict, low_5d_map: dict, turnover_vol_5d_map: dict) -> pd.DataFrame:
        """【V3.0 · 5日窗口注入版】新增对5日高低价及成交量区间的注入。"""
        # 核心修改: 更新方法签名以接收5日窗口数据映射
        cyq_chips_df = data_dfs['cyq_chips'].copy()
        daily_data_df = data_dfs['daily_data'].copy()
        daily_basic_df = data_dfs['daily_basic'].copy()
        cyq_perf_df = data_dfs['cyq_perf'].copy()
        for df in [cyq_chips_df, daily_data_df, daily_basic_df, cyq_perf_df]:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
        daily_data_df.set_index('trade_time', inplace=True)
        daily_basic_df.set_index('trade_time', inplace=True)
        cyq_perf_df.drop(columns=['id', 'stock_id'], errors='ignore', inplace=True)
        cyq_perf_df.set_index('trade_time', inplace=True)
        daily_combined_df = daily_data_df.join([daily_basic_df, cyq_perf_df], how='left')
        merged_df = pd.merge(cyq_chips_df, daily_combined_df.reset_index(), on='trade_time', how='right')
        merged_df.sort_values(by=['trade_time', 'price'], inplace=True)
        merged_df['prev_20d_trade_time'] = merged_df['trade_time'].map(date_20d_ago_map)
        merged_df['prev_20d_close'] = merged_df['prev_20d_trade_time'].map(close_map)
        merged_df['atr_14d'] = merged_df['trade_time'].map(atr_map)
        merged_df['high_20d'] = merged_df['trade_time'].map(high_20d_map)
        merged_df['low_20d'] = merged_df['trade_time'].map(low_20d_map)
        # 核心新增: 将5日窗口数据映射到合并后的DataFrame中
        merged_df['high_5d'] = merged_df['trade_time'].map(high_5d_map)
        merged_df['low_5d'] = merged_df['trade_time'].map(low_5d_map)
        merged_df['turnover_vol_5d'] = merged_df['trade_time'].map(turnover_vol_5d_map)
        merged_df.drop(columns=['prev_20d_trade_time'], inplace=True)
        return merged_df

    def _synthesize_and_forge_metrics(self, stock_info: StockInfo, merged_df: pd.DataFrame, minute_data_map: dict, fund_flow_attributed_minute_map: dict, memory: dict = None, historical_components: pd.DataFrame = None) -> tuple[pd.DataFrame, dict, list]:
        """
        【V3.3 · 记忆种子注入版】
        - 核心修复: 为 `chip_fatigue_index` 的跨日记忆提供一个 0.0 的默认“种子”值，彻底修复其迭代计算链。
        """
        stock_code = stock_info.stock_code
        all_metrics_list = []
        failures_list = []
        prev_metrics = memory.copy() if memory is not None else {}
        grouped_data = merged_df.groupby('trade_time')
        required_daily_chip_cols = ['close_qfq', 'vol', 'float_share', 'circ_mv', 'weight_avg', 'winner_rate', 'pre_close_qfq']
        is_first_day_in_batch = True
        hist_comp_dict = historical_components.to_dict('index') if historical_components is not None and not historical_components.empty else {}
        for i, (trade_date, daily_full_df) in enumerate(grouped_data):
            context_data = daily_full_df.iloc[0].to_dict()
            chip_data_for_calc = daily_full_df[['price', 'percent']].dropna()
            if chip_data_for_calc.empty:
                reason = "当日源筹码分布(cyq_chips)数据缺失"
                logger.warning(f"[{stock_code}] [{trade_date.date()}] 预警：{reason}。")
                failures_list.append({'stock_code': stock_code, 'trade_date': str(trade_date.date()), 'reason': reason})
                prev_metrics = {
                    'concentration_90pct': None, 'winner_avg_cost': None, 'chip_distribution': chip_data_for_calc,
                    'close_price': context_data.get('close_qfq'), 'prev_20d_close': context_data.get('prev_20d_close'),
                    'high_20d': context_data.get('high_20d'), 'low_20d': context_data.get('low_20d'),
                    'total_chip_volume': context_data.get('float_share', 0) * 10000, 'chip_fatigue_index': None,
                    'recent_closes_queue': [], 'dominant_peak_cost': None, 'atr_14d': None,
                }
                if is_first_day_in_batch: is_first_day_in_batch = False
                continue
            missing_keys = [key for key in required_daily_chip_cols if key not in context_data or pd.isna(context_data[key])]
            if not chip_data_for_calc.empty and chip_data_for_calc['percent'].sum() < 0.1:
                missing_keys.append('valid_chip_distribution')
            if missing_keys:
                reason = f"缺失核心原料数据: {missing_keys}"
                logger.warning(f"[{stock_code}] [{trade_date.date()}] 跳过筹码计算，{reason}")
                failures_list.append({'stock_code': stock_code, 'trade_date': str(trade_date.date()), 'reason': reason})
                continue
            cyq_perf_keys = ['weight_avg', 'winner_rate', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'prev_20d_close', 'open_qfq']
            context_for_calc = {key: context_data.get(key) for key in cyq_perf_keys}
            close_price_today = context_data.get('close_qfq')
            recent_closes_list = prev_metrics.get('recent_closes_queue', [])
            if len(recent_closes_list) >= 10:
                recent_closes_list.pop(0)
            recent_closes_list.append(close_price_today)
            total_chip_volume_today = context_data.get('float_share', 0) * 10000
            context_for_calc.update({
                'close_price': close_price_today, 'high_price': context_data.get('high_qfq'),
                'low_price': context_data.get('low_qfq'), 'open_price': context_data.get('open_qfq'),
                'pre_close': context_data.get('pre_close_qfq'), 'daily_turnover_volume': context_data.get('vol', 0) * 100,
                'total_chip_volume': total_chip_volume_today, 'stock_code': stock_code, 'trade_date': trade_date.date(),
                'circ_mv': context_data.get('circ_mv'), 'is_first_day_in_batch': is_first_day_in_batch,
                'high_20d': context_data.get('high_20d'), 'low_20d': context_data.get('low_20d'),
                'atr_14d': context_data.get('atr_14d'),
                'high_5d': context_data.get('high_5d'), 'low_5d': context_data.get('low_5d'),
                'turnover_vol_5d': context_data.get('turnover_vol_5d'),
                'prev_concentration_90pct': prev_metrics.get('concentration_90pct'),
                'prev_winner_avg_cost': prev_metrics.get('winner_avg_cost'),
                'prev_chip_distribution': prev_metrics.get('chip_distribution'),
                'prev_dominant_peak_cost': prev_metrics.get('dominant_peak_cost'),
                'prev_day_20d_ago_close': prev_metrics.get('prev_20d_close'),
                'prev_high_20d': prev_metrics.get('high_20d'), 'prev_low_20d': prev_metrics.get('low_20d'),
                'prev_total_chip_volume': prev_metrics.get('total_chip_volume'),
                # [代码修改开始]
                # 修复记忆链：为疲劳度指数提供一个 0.0 的默认“种子”值
                'prev_chip_fatigue_index': prev_metrics.get('chip_fatigue_index', 0.0),
                # [代码修改结束]
                'recent_10d_closes': recent_closes_list,
                'prev_atr_14d': prev_metrics.get('atr_14d'),
            })
            if hist_comp_dict:
                historical_data_for_day = {k: v for k, v in hist_comp_dict.items() if k < trade_date}
                if historical_data_for_day:
                    context_for_calc['historical_components'] = pd.DataFrame.from_dict(historical_data_for_day, orient='index')
            if fund_flow_attributed_minute_map and trade_date in fund_flow_attributed_minute_map:
                enhanced_minute_data = fund_flow_attributed_minute_map[trade_date]
            else:
                raw_minute_data_for_day = minute_data_map.get(trade_date.date(), pd.DataFrame())
                enhanced_minute_data = self._enhance_minute_data_fallback(raw_minute_data_for_day)
            context_for_calc['minute_data'] = enhanced_minute_data
            calculator = ChipFeatureCalculator(chip_data_for_calc, context_for_calc)
            daily_metrics = calculator.calculate_all_metrics()
            if daily_metrics:
                daily_metrics['trade_time'] = trade_date
                daily_metrics['prev_20d_close'] = context_data.get('prev_20d_close')
                all_metrics_list.append(daily_metrics)
                if 'historical_components' in context_for_calc:
                    today_metrics_for_hist = {k: [daily_metrics.get(k)] for k in context_for_calc['historical_components'].columns}
                    today_df = pd.DataFrame(today_metrics_for_hist, index=[trade_date])
                    hist_comp_dict.update(today_df.to_dict('index'))
            prev_metrics = {
                'concentration_90pct': daily_metrics.get('concentration_90pct') if daily_metrics else None,
                'winner_avg_cost': daily_metrics.get('winner_avg_cost') if daily_metrics else None,
                'chip_distribution': chip_data_for_calc,
                'dominant_peak_cost': daily_metrics.get('dominant_peak_cost') if daily_metrics else None,
                'close_price': context_data.get('close_qfq'),
                'prev_20d_close': context_data.get('prev_20d_close'), 'high_20d': context_data.get('high_20d'),
                'low_20d': context_data.get('low_20d'), 'total_chip_volume': total_chip_volume_today,
                'chip_fatigue_index': daily_metrics.get('chip_fatigue_index') if daily_metrics else None,
                'recent_closes_queue': recent_closes_list,
                'atr_14d': context_data.get('atr_14d'),
            }
            if is_first_day_in_batch: is_first_day_in_batch = False
        if not all_metrics_list:
            return pd.DataFrame(), prev_metrics, failures_list
        return pd.DataFrame(all_metrics_list).set_index('trade_time'), prev_metrics, failures_list

    def _enhance_minute_data_fallback(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 · 竞价隔离版】当资金流数据缺失时，提供一个基础的分钟数据增强。
        - 核心修复: 过滤掉集合竞价时段的数据。
        """
        if minute_df.empty: return minute_df
        df = minute_df.copy()
        # 过滤集合竞价时段
        CONTINUOUS_TRADING_END_TIME = time(14, 57, 0)
        df = df[df['trade_time'].dt.time < CONTINUOUS_TRADING_END_TIME].copy()
        if df.empty: return df
        
        df['amount_yuan'] = pd.to_numeric(df['amount'], errors='coerce')
        df['vol_shares'] = pd.to_numeric(df['vol'], errors='coerce')
        df['minute_vwap'] = df['amount_yuan'] / df['vol_shares'].replace(0, np.nan)
        return df

    async def _load_minute_data_for_range(self, stock_info: StockInfo, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        【V1.1 · 时区修正版】一次性加载指定日期范围内的所有分钟线数据，并按日期分组。
        - 核心修复: 强制将时间戳转换为北京时间，并确保升序排列。
        """
        from django.utils import timezone
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        if not MinuteModel: return {}
        @sync_to_async(thread_sensitive=True)
        def get_data(model, stock_pk, start_dt, end_dt):
            # 强制按交易时间升序排序
            qs = model.objects.filter(stock_id=stock_pk, trade_time__gte=start_dt, trade_time__lt=end_dt).values('trade_time', 'amount', 'vol', 'open', 'close', 'high', 'low').order_by('trade_time')
            
            return pd.DataFrame.from_records(qs)
        start_datetime = timezone.make_aware(datetime.combine(start_date, time.min))
        end_datetime = timezone.make_aware(datetime.combine(end_date, time.max))
        minute_df = await get_data(MinuteModel, stock_info.pk, start_datetime, end_datetime)
        if minute_df.empty: return {}
        minute_df['trade_time'] = pd.to_datetime(minute_df['trade_time'])
        # 强制将时间戳转换为北京时间
        if minute_df['trade_time'].dt.tz is not None:
            minute_df['trade_time'] = minute_df['trade_time'].dt.tz_convert('Asia/Shanghai')
        else:
            minute_df['trade_time'] = minute_df['trade_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        
        minute_df['date'] = minute_df['trade_time'].dt.date
        return {date: group_df for date, group_df in minute_df.groupby('date')}

    async def _load_historical_metrics(self, model, stock_info, end_date):
        """从数据库加载历史高级筹码指标。"""
        @sync_to_async
        def get_data():
            core_metric_cols = list(BaseAdvancedChipMetrics.CORE_METRICS.keys())
            required_cols = ['trade_time'] + [col for col in core_metric_cols if hasattr(model, col)]
            qs = model.objects.filter(stock=stock_info, trade_time__lt=end_date).order_by('trade_time')
            return pd.DataFrame.from_records(qs.values(*required_cols))
        df = await get_data()
        if not df.empty:
            df = df.set_index(pd.to_datetime(df['trade_time']))
            for col in df.columns:
                if col != 'trade_time':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _calculate_derivatives(self, consensus_df: pd.DataFrame) -> pd.DataFrame:
        """【V2.2 · 导数定律统一版】修正加速度计算窗口，与资金流服务保持一致。"""
        derivatives_df = pd.DataFrame(index=consensus_df.index)
        import pandas_ta as ta
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedChipMetrics.SLOPE_ACCEL_EXCLUSIONS
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedChipMetrics.CORE_METRICS.keys())
        UNIFIED_PERIODS = BaseAdvancedChipMetrics.UNIFIED_PERIODS
        # 为加速度定义一个独立的、符合数学定义的短窗口
        ACCEL_WINDOW = 2
        
        for col in CORE_METRICS_TO_DERIVE:
            if col in consensus_df.columns and col not in SLOPE_ACCEL_EXCLUSIONS and col not in BaseAdvancedChipMetrics.BOOLEAN_FIELDS:
                source_series = pd.to_numeric(consensus_df[col], errors='coerce')
                if source_series.isnull().all():
                    continue
                for p in UNIFIED_PERIODS:
                    calc_window = 2 if p == 1 else p
                    slope_col_name = f'{col}_slope_{p}d'
                    slope_series = ta.slope(close=source_series, length=calc_window)
                    derivatives_df[slope_col_name] = slope_series
                    if slope_series is not None and not slope_series.empty:
                        accel_col_name = f'{col}_accel_{p}d'
                        # 强制为加速度计算使用独立的短窗口 ACCEL_WINDOW
                        derivatives_df[accel_col_name] = ta.slope(close=slope_series.astype(float), length=ACCEL_WINDOW)
        return derivatives_df

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        """准备并以“更新或创建”的方式原子化保存数据。"""
        if final_df.empty: return 0
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 健壮性修复：确保所有布尔字段在保存前都有确定的值 (True/False)，而不是 NaN。
        # NaN 在保存时会变成 NULL，导致数据库 NOT NULL 约束错误。
        boolean_fields = BaseAdvancedChipMetrics.BOOLEAN_FIELDS
        for col in boolean_fields:
            if col in final_df.columns:
                # 将 NaN 值填充为 False，这是布尔字段最安全的默认值
                final_df[col] = final_df[col].fillna(False)
        
        model_fields = {f.name for f in MetricsModel._meta.get_fields() if not f.is_relation and f.name != 'id'}
        df_filtered = final_df[[col for col in final_df.columns if col in model_fields]]
        records_list = df_filtered.to_dict('records')
        @sync_to_async(thread_sensitive=True)
        def save_atomically(model, stock_obj, records_to_process):
            processed_count = 0
            for record_data in records_to_process:
                trade_time = record_data.pop('trade_time').date()
                defaults_data = {key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in record_data.items()}
                try:
                    obj, created = model.objects.update_or_create(stock=stock_obj, trade_time=trade_time, defaults=defaults_data)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"[{stock_obj.stock_code}] [筹码保存失败] 日期: {trade_time}, 错误: {e}")
            return processed_count
        records_for_atomic_save = []
        for record_date, record_data in zip(df_filtered.index, records_list):
            record_data['trade_time'] = record_date
            records_for_atomic_save.append(record_data)
        processed_count = await save_atomically(MetricsModel, stock_info, records_for_atomic_save)
        return processed_count
