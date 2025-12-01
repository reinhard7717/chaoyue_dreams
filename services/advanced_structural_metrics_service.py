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

    def _calculate_daily_structural_metrics(self, group: pd.DataFrame, continuous_group: pd.DataFrame, daily_series_for_day: pd.Series, atr_5: float, atr_14: float, atr_50: float, prev_day_metrics: dict, tick_df_for_day: pd.DataFrame = None, level5_df_for_day: pd.DataFrame = None, realtime_df_for_day: pd.DataFrame = None) -> dict:
        """
        【V28.0 · 主题内核归一】
        - 核心重构: 迁移所有剩余的计算逻辑至 ThematicMetricsCalculators。
                     此方法已达到其最终形态：一个纯粹的、由内核调用组成的流程编排器。
        """
        results = {}
        group['amount'] = pd.to_numeric(group['amount'], errors='coerce')
        group['vol'] = pd.to_numeric(group['vol'], errors='coerce')
        total_volume = group['vol'].sum()
        total_volume_safe = total_volume if total_volume > 0 else np.nan
        day_open_qfq, day_high_qfq, day_low_qfq, day_close_qfq, pre_close_qfq = (
            daily_series_for_day.get('open_qfq'), daily_series_for_day.get('high_qfq'),
            daily_series_for_day.get('low_qfq'), daily_series_for_day.get('close_qfq'),
            daily_series_for_day.get('pre_close_qfq')
        )
        trade_date_str = group['trade_time'].iloc[0].strftime('%Y-%m-%d')
        is_target_date = trade_date_str == self.debug_params.get('target_date')
        enable_probe = self.debug_params.get('enable_asm_probe', False)
        context = {
            'group': group,
            'continuous_group': continuous_group,
            'daily_series_for_day': daily_series_for_day,
            'atr_5': atr_5, 'atr_14': atr_14, 'atr_50': atr_50,
            'prev_day_metrics': prev_day_metrics,
            'tick_df': tick_df_for_day,
            'level5_df': level5_df_for_day,
            'realtime_df': realtime_df_for_day,
            'total_volume_safe': total_volume_safe,
            'day_open_qfq': day_open_qfq, 'day_high_qfq': day_high_qfq,
            'day_low_qfq': day_low_qfq, 'day_close_qfq': day_close_qfq,
            'pre_close_qfq': pre_close_qfq,
            'debug': {
                'is_target_date': is_target_date,
                'enable_probe': enable_probe,
                'trade_date_str': trade_date_str,
            }
        }
        # --- 1. 能量密度与战场动力学 ---
        results.update(StructuralMetricsCalculators.calculate_energy_density_metrics(context))
        # --- 2. 趋势、节律与代理筹码 ---
        results.update(StructuralMetricsCalculators.calculate_control_metrics(context))
        # --- 3. 博弈效能 ---
        results.update(StructuralMetricsCalculators.calculate_game_efficiency_metrics(context))
        # --- 4. 微观结构动力学 ---
        results.update(MicrostructureDynamicsCalculators.calculate_all(context))
        # --- 5. 市场剖面与价值区 ---
        market_profile_results = ThematicMetricsCalculators.calculate_market_profile_metrics(context)
        results.update(market_profile_results)
        # --- 6. 前瞻性与收盘博弈 ---
        results.update(ThematicMetricsCalculators.calculate_forward_looking_metrics(context))
        # --- 7. 战场分析仪 ---
        results.update(ThematicMetricsCalculators.calculate_battlefield_metrics(context))
        return results

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










