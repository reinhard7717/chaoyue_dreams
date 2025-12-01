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
from services.structural_metrics_calculators import StructuralMetricsCalculators
from services.microstructure_dynamics_calculators import MicrostructureDynamicsCalculators
from services.thematic_metrics_calculators import ThematicMetricsCalculators
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
        【V30.6 · 数据源融合修正】
        - 核心修复: 修复了因架构重构导致的数据融合逻辑缺失，该缺失导致在某些情况下传入的分钟数据缺少'vol'列。
        - 解决方案: 在服务层内部重建数据源融合逻辑，优先使用高保真的Tick数据通过resample生成标准分钟数据，否则回退到预计算的分钟数据，确保下游计算总能获得格式统一的输入。
        """
        new_metrics_data = []
        prev_day_metrics = {}
        for trade_date, data_for_day in intraday_map.items():
            daily_series_for_day = daily_df_with_atr.loc[trade_date]
            atr_5 = daily_series_for_day.get('ATR_5', np.nan)
            atr_14 = daily_series_for_day.get('ATR_14', np.nan)
            atr_50 = daily_series_for_day.get('ATR_50', np.nan)
            
            # --- 新增代码块：数据源融合与标准化 ---
            canonical_minute_df = None
            tick_df_for_day = data_for_day.get('tick')
            minute_df_for_day = data_for_day.get('minute')

            if tick_df_for_day is not None and not tick_df_for_day.empty:
                # 优先级1：使用Tick数据生成分钟线
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
                # 优先级2：使用预计算的分钟线
                # 确保列名统一，以防万一
                if 'volume' in minute_df_for_day.columns and 'vol' not in minute_df_for_day.columns:
                    minute_df_for_day.rename(columns={'volume': 'vol'}, inplace=True)
                canonical_minute_df = minute_df_for_day

            if canonical_minute_df is None or canonical_minute_df.empty:
                logger.warning(f"[{stock_code}] 在日期 {trade_date} 缺少可用的分钟级或Tick级数据，跳过当日计算。")
                continue
            # --- 融合逻辑结束 ---

            level5_df_for_day = data_for_day.get('level5')
            realtime_df_for_day = data_for_day.get('realtime')
            
            # 修改代码行：使用融合后的标准分钟数据
            continuous_group = self._create_continuous_minute_data(canonical_minute_df)
            day_metric_dict = self._calculate_daily_structural_metrics(
                canonical_minute_df, continuous_group, daily_series_for_day, atr_5, atr_14, atr_50, prev_day_metrics,
                tick_df_for_day, level5_df_for_day, realtime_df_for_day
            )
            day_metric_dict['trade_date'] = trade_date
            day_metric_dict['stock_code'] = stock_code
            new_metrics_data.append(day_metric_dict)
            prev_day_metrics = {
                'vpoc': day_metric_dict.get('_today_vpoc'),
                'vah': day_metric_dict.get('_today_vah'),
                'val': day_metric_dict.get('_today_val'),
                'atr_14d': atr_14,
            }
        if not new_metrics_data:
            return pd.DataFrame()
        new_metrics_df = pd.DataFrame(new_metrics_data)
        new_metrics_df.set_index('trade_date', inplace=True)
        final_metrics_df = self._calculate_dynamic_evolution_factors(new_metrics_df)
        final_metrics_df.reset_index(inplace=True)
        return final_metrics_df

    def _calculate_daily_structural_metrics(
        self, group: pd.DataFrame, continuous_group: pd.DataFrame, daily_series: pd.Series,
        atr_5: float, atr_14: float, atr_50: float, prev_day_metrics: dict,
        tick_df: pd.DataFrame = None, level5_df: pd.DataFrame = None, realtime_df: pd.DataFrame = None
    ) -> dict:
        """
        【V30.15 · 计算内核调用修复】
        - 核心修复: 修正了此方法期望'trade_time'为列，而上游传入的DataFrame已将其作为索引导致的KeyError。
        - 核心修复: 新增对外部计算内核的调用，修复因重构导致的指标计算逻辑缺失。
        """
        if group is None or group.empty:
            return {}
        trade_date_obj = group.index[0].date()
        trade_date_str = trade_date_obj.strftime('%Y-%m-%d')
        metrics = {}
        # 1. 基础统计指标
        metrics['mean_price'] = group['close'].mean()
        metrics['median_price'] = group['close'].median()
        metrics['std_price'] = group['close'].std()
        total_volume = group['vol'].sum()
        total_amount = group['amount'].sum()
        metrics['total_volume'] = total_volume
        metrics['total_amount'] = total_amount
        # 2. VWAP (Volume Weighted Average Price)
        metrics['vwap'] = total_amount / total_volume if total_volume > 0 else group['close'].iloc[-1]
        # 3. 波动率指标
        log_returns = np.log(group['close'] / group['close'].shift(1)).dropna()
        metrics['volatility_realized'] = log_returns.std() * np.sqrt(240)  # 年化波动率
        # 4. 趋势与反转指标
        price_trend_strength, price_trend_quality = self._calculate_trend_metrics(group['close'])
        metrics['price_trend_strength'] = price_trend_strength
        metrics['price_trend_quality'] = price_trend_quality
        # 5. 均值回归特性
        metrics['mean_reversion_speed'] = self._calculate_mean_reversion_speed(group['close'])
        # 6. 价值区分析 (TPO/MP)
        tpo_metrics = self._calculate_tpo_metrics(group)
        metrics.update(tpo_metrics)
        # 7. 连续数据分析
        if continuous_group is not None and not continuous_group.empty:
            continuous_metrics = self._calculate_continuous_data_metrics(continuous_group)
            metrics.update(continuous_metrics)
        # 8. 与ATR的交互
        atr_interaction_metrics = self._calculate_atr_interaction_metrics(group, atr_5, atr_14, atr_50)
        metrics.update(atr_interaction_metrics)
        # 9. 与前一日指标的交互
        prev_day_interaction_metrics = self._calculate_prev_day_interaction_metrics(group, prev_day_metrics)
        metrics.update(prev_day_interaction_metrics)
        # 新增代码块：构建上下文并调用外部计算内核
        # 10. 调用外部计算内核
        target_date_str = self.debug_params.get('target_date')
        is_target_date = target_date_str == trade_date_str if target_date_str else False
        context = {
            'group': group,
            'continuous_group': continuous_group,
            'daily_series_for_day': daily_series,
            'day_open_qfq': daily_series.get('open_qfq'),
            'day_high_qfq': daily_series.get('high_qfq'),
            'day_low_qfq': daily_series.get('low_qfq'),
            'day_close_qfq': daily_series.get('close_qfq'),
            'pre_close_qfq': daily_series.get('pre_close_qfq'),
            'atr_5': atr_5,
            'atr_14': atr_14,
            'atr_50': atr_50,
            'total_volume_safe': total_volume if total_volume > 0 else 1,
            'prev_day_metrics': prev_day_metrics,
            'tick_df': tick_df,
            'level5_df': level5_df,
            'realtime_df': realtime_df,
            'debug': {
                'enable_probe': self.debug_params.get('enable_asm_probe', False),
                'is_target_date': is_target_date,
                'trade_date_str': trade_date_str,
            }
        }
        # 调用能量密度与战场动力学计算内核
        metrics.update(StructuralMetricsCalculators.calculate_energy_density_metrics(context))
        metrics.update(StructuralMetricsCalculators.calculate_control_metrics(context))
        metrics.update(StructuralMetricsCalculators.calculate_game_efficiency_metrics(context))
        # 调用微观动力学计算内核
        metrics.update(MicrostructureDynamicsCalculators.calculate_all(context))
        # 调用主题指标计算内核
        metrics.update(ThematicMetricsCalculators.calculate_market_profile_metrics(context))
        metrics.update(ThematicMetricsCalculators.calculate_forward_looking_metrics(context))
        metrics.update(ThematicMetricsCalculators.calculate_battlefield_metrics(context))
        return metrics

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
        【V30.0 · 动态演化因子】
        - 核心职责: 计算核心结构指标的时间序列衍生因子，捕捉其动态演化趋势。
        """
        df = metrics_df.copy().sort_index()
        # 核心意愿演化
        df['thrust_purity_ma5'] = df['intraday_thrust_purity'].rolling(window=5, min_periods=1).mean()
        # 吸筹行为演化
        df['absorption_strength_ma5'] = df['absorption_strength_index'].rolling(window=5, min_periods=1).mean()
        # 情绪激进度演化
        df['sweep_intensity_ma5'] = df['buy_sweep_intensity'].rolling(window=5, min_periods=1).mean()
        # 流动性风险演化
        df['vpin_roc3'] = df['vpin_score'].pct_change(periods=3)
        return df

    def _create_continuous_minute_data(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        【V30.5 · 数据填充逻辑修正】
        - 核心修复: 修正了因错误地对'vol'和'amount'等事件数据使用前向填充(ffill)导致的逻辑错误和KeyError。
        - 实现逻辑: 1. reindex后，对不同性质的列采用不同填充策略。
                     2. 价格等'状态'列使用 .ffill()。
                     3. 成交量/额等'事件'列使用 .fillna(0)。
        """
        if group is None or group.empty:
            return pd.DataFrame()
        trade_date = group.index[0].date()
        morning_session = pd.to_datetime(pd.date_range(start=f'{trade_date} 09:31:00', end=f'{trade_date} 12:00:00', freq='1min'))
        afternoon_session = pd.to_datetime(pd.date_range(start=f'{trade_date} 13:01:00', end=f'{trade_date} 15:00:00', freq='1min'))
        full_day_index = morning_session.union(afternoon_session).tz_localize('Asia/Shanghai')
        # 步骤1: 重新索引以创建包含NaN的完整时间序列
        continuous_group = group.reindex(full_day_index)
        # 步骤2: 对价格相关的“状态”列进行前向填充
        price_cols = ['open', 'high', 'low', 'close']
        existing_price_cols = [col for col in price_cols if col in continuous_group.columns]
        continuous_group[existing_price_cols] = continuous_group[existing_price_cols].ffill()
        # 步骤3: 对成交量/额相关的“事件”列填充0
        transaction_cols = ['vol', 'amount']
        existing_transaction_cols = [col for col in transaction_cols if col in continuous_group.columns]
        continuous_group[existing_transaction_cols] = continuous_group[existing_transaction_cols].fillna(0)
        # 步骤4: 安全地计算分钟VWAP
        continuous_group['minute_vwap'] = (continuous_group['amount'] / continuous_group['vol']).where(continuous_group['vol'] > 0, np.nan)
        # 步骤5: 恢复'trade_time'为列
        continuous_group.reset_index(inplace=True)
        continuous_group.rename(columns={'index': 'trade_time'}, inplace=True)
        return continuous_group

    def _calculate_trend_metrics(self, price_series: pd.Series) -> tuple[float, float]:
        """
        【V30.8 · 新增辅助函数】
        通过线性回归计算价格序列的趋势强度和趋势质量。
        - 趋势强度: 回归线的斜率。
        - 趋势质量: 回归的R平方值。
        :param price_series: 分钟收盘价序列。
        :return: (趋势强度, 趋势质量) 的元组。
        """
        # 检查输入数据是否有效，至少需要两个点才能拟合一条直线
        cleaned_series = price_series.dropna()
        if len(cleaned_series) < 2:
            return 0.0, 0.0
        # 如果所有价格都相同，则没有趋势，但趋势质量是完美的（一条平线）
        if cleaned_series.nunique() == 1:
            return 0.0, 1.0
        y = cleaned_series.values
        x = np.arange(len(y))
        # 使用numpy的polyfit进行1次多项式拟合（即线性回归）
        slope, intercept = np.polyfit(x, y, 1)
        # 计算R平方值来评估拟合优度（趋势质量）
        predicted_y = slope * x + intercept
        residuals = y - predicted_y
        ss_res = np.sum(residuals**2)  # 残差平方和
        ss_tot = np.sum((y - np.mean(y))**2)  #总体平方和
        if ss_tot == 0:
            # 避免除以零，这种情况在nunique检查中已处理，但作为双重保险
            return 0.0, 1.0
        r_squared = 1 - (ss_res / ss_tot)
        return slope, r_squared

    def _calculate_mean_reversion_speed(self, price_series: pd.Series) -> float:
        """
        【V30.9 · 新增辅助函数】
        估算价格序列的均值回归速度。
        基于Ornstein-Uhlenbeck过程的离散化模型，通过回归价格变化与滞后价格来计算。
        :param price_series: 分钟收盘价序列。
        :return: 均值回归速度。正值表示存在均值回归，值越大速度越快。
        """
        # 清理数据并确保有足够的数据点进行回归
        cleaned_series = price_series.dropna()
        if len(cleaned_series) < 2:
            return 0.0
        # 计算价格变化 ΔP(t)
        price_changes = cleaned_series.diff().dropna()
        # 获取滞后价格 P(t-1)
        lagged_prices = cleaned_series.shift(1).dropna()
        # 确保两者对齐
        if len(price_changes) != len(lagged_prices):
            # 在diff和shift之后，长度应该是一样的，但为了健壮性再做一次对齐
            common_index = price_changes.index.intersection(lagged_prices.index)
            price_changes = price_changes.loc[common_index]
            lagged_prices = lagged_prices.loc[common_index]
        if len(price_changes) < 2:
            return 0.0
        # 对 ΔP(t) = α + β * P(t-1) + ε 进行线性回归
        # y 是价格变化, x 是滞后价格
        y = price_changes.values
        x = lagged_prices.values
        # 使用numpy的polyfit进行线性回归，得到斜率β
        slope, _ = np.polyfit(x, y, 1)
        # 均值回归速度定义为 -slope
        return -slope

    def _calculate_tpo_metrics(self, group: pd.DataFrame) -> dict:
        """
        【V30.10 · 新增辅助函数】
        基于日内分钟数据计算市场轮廓（TPO/Market Profile）的核心指标。
        - VPOC (Volume Point of Control): 成交量最大的价格点。
        - Value Area (VA): 包含当日70%成交量的价格区域。
        :param group: 包含'close'和'vol'列的日内分钟数据DataFrame。
        :return: 包含VPOC, VAH, VAL的字典。
        """
        if group.empty or 'vol' not in group.columns or group['vol'].sum() == 0:
            return {
                '_today_vpoc': np.nan,
                '_today_vah': np.nan,
                '_today_val': np.nan,
            }
        # 1. 构建成交量分布图
        volume_profile = group.groupby('close')['vol'].sum().sort_index()
        if volume_profile.empty:
            return {
                '_today_vpoc': np.nan,
                '_today_vah': np.nan,
                '_today_val': np.nan,
            }
        # 2. 确定VPOC
        vpoc = volume_profile.idxmax()
        # 3. 计算价值区 (VAH, VAL)
        total_volume = volume_profile.sum()
        value_area_target_volume = total_volume * 0.7
        # 初始化搜索范围
        value_area_prices = [vpoc]
        current_volume_in_area = volume_profile.loc[vpoc]
        # 获取VPOC上下方的价格索引
        prices_below_vpoc = volume_profile.index[volume_profile.index < vpoc]
        prices_above_vpoc = volume_profile.index[volume_profile.index > vpoc]
        # 双指针，从紧邻VPOC的价格开始
        below_ptr = len(prices_below_vpoc) - 1
        above_ptr = 0
        while current_volume_in_area < value_area_target_volume:
            vol_below = 0
            if below_ptr >= 0:
                price_below = prices_below_vpoc[below_ptr]
                vol_below = volume_profile.loc[price_below]
            vol_above = 0
            if above_ptr < len(prices_above_vpoc):
                price_above = prices_above_vpoc[above_ptr]
                vol_above = volume_profile.loc[price_above]
            if vol_below == 0 and vol_above == 0:
                break # 没有更多价格可以添加
            # 贪心策略：总是添加下一个成交量更大的价格点
            if vol_above > vol_below:
                value_area_prices.append(price_above)
                current_volume_in_area += vol_above
                above_ptr += 1
            else:
                value_area_prices.append(price_below)
                current_volume_in_area += vol_below
                below_ptr -= 1
        vah = max(value_area_prices)
        val = min(value_area_prices)
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
        """
        metrics = {
            'gap_return': np.nan,
            'post_gap_momentum_30min': np.nan,
        }
        if continuous_group is None or continuous_group.empty or len(continuous_group) < 2:
            return metrics
        # 1. 识别两个交易日
        # 修改代码行：增加索引类型检查，使其更健壮
        if isinstance(continuous_group.index, pd.DatetimeIndex):
            trade_dates = continuous_group.index.date
        elif 'trade_time' in continuous_group.columns:
            trade_dates = pd.to_datetime(continuous_group['trade_time']).dt.date
        else:
            # 如果既没有时间戳索引，也没有trade_time列，则无法继续
            return metrics
        unique_dates = np.unique(trade_dates)
        if len(unique_dates) != 2:
            # 数据不符合跨日连续数据的定义
            return metrics
        # 2. 拆分数据并获取关键价格
        prev_day_data = continuous_group[trade_dates == unique_dates[0]]
        today_data = continuous_group[trade_dates == unique_dates[1]]
        if prev_day_data.empty or today_data.empty:
            return metrics
        prev_close = prev_day_data['close'].iloc[-1]
        today_open = today_data['open'].iloc[0]
        # 3. 计算跳空回报率
        if prev_close > 0:
            metrics['gap_return'] = (today_open / prev_close) - 1
        # 4. 计算缺口后30分钟动量
        if today_open > 0:
            # 截取今日开盘后30分钟的数据
            first_30_min_data = today_data.head(30)
            if not first_30_min_data.empty:
                close_after_30_min = first_30_min_data['close'].iloc[-1]
                metrics['post_gap_momentum_30min'] = (close_after_30_min / today_open) - 1
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
        """
        metrics = {
            'intraday_range_vs_atr14': np.nan,
            'close_vwap_deviation_normalized': np.nan,
            'volatility_expansion_ratio': np.nan,
        }
        if group.empty:
            return metrics
        # 1. 日内振幅 vs ATR14
        day_high = group['high'].max()
        day_low = group['low'].min()
        intraday_range = day_high - day_low
        if pd.notna(atr_14) and atr_14 > 0:
            metrics['intraday_range_vs_atr14'] = intraday_range / atr_14
        # 2. 收盘价与VWAP的偏离度 (ATR标准化)
        day_close = group['close'].iloc[-1]
        total_volume = group['vol'].sum()
        total_amount = group['amount'].sum()
        vwap = total_amount / total_volume if total_volume > 0 else day_close
        if pd.notna(atr_14) and atr_14 > 0:
            metrics['close_vwap_deviation_normalized'] = (day_close - vwap) / atr_14
        # 3. 短期与长期波动率扩张比
        if pd.notna(atr_5) and pd.notna(atr_50) and atr_50 > 0:
            metrics['volatility_expansion_ratio'] = atr_5 / atr_50
        return metrics

    def _calculate_prev_day_interaction_metrics(self, group: pd.DataFrame, prev_day_metrics: dict) -> dict:
        """
        【V30.14 · 新增辅助函数】
        计算当日市场行为与前一日关键结构位（如价值区）的交互指标。
        :param group: 日内分钟数据DataFrame。
        :param prev_day_metrics: 包含前一日VPOC, VAH, VAL, ATR的字典。
        :return: 包含交互指标的字典。
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
        # 2. 计算价值区迁移 (Value Area Migration)
        if all(pd.notna(v) for v in [today_vpoc, prev_vpoc, prev_atr]) and prev_atr > 0:
            metrics['value_area_migration'] = (today_vpoc - prev_vpoc) / prev_atr
        # 3. 计算价值区重叠度 (Value Area Overlap)
        if all(pd.notna(v) for v in [today_vah, today_val, prev_vah, prev_val]):
            today_va_height = today_vah - today_val
            if today_va_height > 0:
                overlap_width = max(0, min(today_vah, prev_vah) - max(today_val, prev_val))
                metrics['value_area_overlap_pct'] = (overlap_width / today_va_height) * 100
        # 4. 计算收盘接受度类型 (Closing Acceptance Type)
        if all(pd.notna(v) for v in [day_close, today_vpoc, today_vah, today_val]):
            if day_close > today_vah:
                metrics['closing_acceptance_type'] = 2  # 强势接受于价值区之上
            elif day_close > today_vpoc:
                metrics['closing_acceptance_type'] = 1  # 接受于价值区上半区
            elif day_close < today_val:
                metrics['closing_acceptance_type'] = -2 # 强势拒绝于价值区之下
            elif day_close < today_vpoc:
                metrics['closing_acceptance_type'] = -1 # 接受于价值区下半区
            else:
                metrics['closing_acceptance_type'] = 0  # 接受于VPOC
        # 5. 计算开盘位置 vs 前日价值区
        if all(pd.notna(v) for v in [day_open, prev_vah, prev_val]):
            if day_open > prev_vah:
                metrics['opening_position_vs_prev_va'] = 2 # 开在价值区之上
            elif day_open > prev_val:
                metrics['opening_position_vs_prev_va'] = 1 # 开在价值区之内
            else:
                metrics['opening_position_vs_prev_va'] = -1 # 开在价值区之下
        return metrics

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        """
        【V19.7 · 索引健壮性修复版】
        - 核心修复: 修复了因错误地使用 CORE_METRICS 筛选待保存列而导致所有衍生指标（斜率、加速度）被丢弃的严重BUG。
        - 核心逻辑: 改为直接从模型元数据 `MetricsModel._meta.get_fields()` 获取所有已定义的字段名进行筛选。
        - 核心优化: 统一并简化了保存逻辑，使用 `reset_index()` 来安全地处理 `trade_time`，避免冲突。
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
                trade_time = record_data.pop('trade_time').date()
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










