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

    async def _forge_advanced_structural_metrics(self, intraday_data_map: dict, stock_code: str, daily_df_with_atr: pd.DataFrame) -> pd.DataFrame:
        """
        【V19.3 · 索引修复版】
        - 核心修正: 修复了 set_index 未能正确移除原始 'trade_time' 列，导致下游 reset_index 操作失败的BUG。
        - 核心逻辑: 采用更标准的 `set_index('trade_time')` 写法，确保索引设置的原子性和正确性。
        - 兼容性修复: 增加成交量列名智能识别逻辑，兼容 'vol' 和 'volume' 两种列名，并统一重命名为 'vol'，以根治 KeyError。
        """
        if not intraday_data_map:
            return pd.DataFrame()
        daily_metrics = []
        all_dates = sorted(intraday_data_map.keys())
        daily_df = daily_df_with_atr
        from stock_models.time_trade import StockDailyBasic
        daily_basic_qs = StockDailyBasic.objects.filter(
            stock__stock_code=stock_code,
            trade_time__gte=min(all_dates),
            trade_time__lte=max(all_dates)
        ).values('trade_time', 'turnover_rate_f').order_by('trade_time')
        daily_basic_df = pd.DataFrame.from_records(await sync_to_async(list)(daily_basic_qs))
        if not daily_basic_df.empty:
            daily_basic_df['trade_time'] = pd.to_datetime(daily_basic_df['trade_time']).dt.date
            daily_basic_df = daily_basic_df.set_index('trade_time')
            daily_df = daily_df.join(daily_basic_df, how='left')
        atr_5 = daily_df['ATR_5']
        atr_14 = daily_df['ATR_14']
        atr_50 = daily_df['ATR_50']
        prev_day_metrics = {}
        for date, data_bundle in intraday_data_map.items():
            group = data_bundle.get('minute')
            tick_df_for_day = data_bundle.get('tick')
            level5_df_for_day = data_bundle.get('level5')
            realtime_df_for_day = data_bundle.get('realtime')
            if group is None or group.empty or len(group) < 10:
                logger.warning(f"[{stock_code}] [{date}] 跳过结构指标计算，分钟级数据不足。")
                continue
            group.reset_index(inplace=True)
            volume_col_name = None
            if 'volume' in group.columns:
                volume_col_name = 'volume'
            elif 'vol' in group.columns:
                volume_col_name = 'vol'
            if volume_col_name and volume_col_name != 'vol':
                group.rename(columns={volume_col_name: 'vol'}, inplace=True)
            elif not volume_col_name:
                logger.error(f"[{stock_code}] [{date}] 分钟数据中缺少成交量列 ('volume' 或 'vol')，跳过当天计算。列: {group.columns.tolist()}")
                continue
            try:
                daily_series_for_day = daily_df.loc[date]
                atr_5_for_day = atr_5.get(date)
                atr_14_for_day = atr_14.get(date)
                atr_50_for_day = atr_50.get(date)
            except KeyError:
                logger.warning(f"[{stock_code}] [{date}] 无法找到对应的日线数据或ATR，跳过当天计算。")
                continue
            group = group.sort_values(by='trade_time').reset_index(drop=True)
            continuous_mask = group['trade_time'].dt.time < time(14, 57, 0)
            continuous_group = group[continuous_mask].copy()
            if continuous_group.empty: continue
            continuous_group['amount'] = pd.to_numeric(continuous_group['amount'], errors='coerce')
            continuous_group['vol'] = pd.to_numeric(continuous_group['vol'], errors='coerce')
            continuous_group['minute_vwap'] = continuous_group['amount'] / continuous_group['vol'].replace(0, np.nan)
            continuous_group['minute_vwap'].fillna(method='ffill', inplace=True)
            continuous_group['minute_vwap'].fillna(group['open'].iloc[0], inplace=True)
            day_metric_dict = self._calculate_daily_structural_metrics(
                group, continuous_group, daily_series_for_day, atr_5_for_day, atr_14_for_day, atr_50_for_day, prev_day_metrics,
                tick_df_for_day=tick_df_for_day,
                level5_df_for_day=level5_df_for_day,
                realtime_df_for_day=realtime_df_for_day
            )
            day_metric_dict['trade_time'] = date
            prev_day_metrics = {
                'vpoc': day_metric_dict.pop('_today_vpoc', np.nan),
                'vah': day_metric_dict.pop('_today_vah', np.nan),
                'val': day_metric_dict.pop('_today_val', np.nan),
                'atr_14d': atr_14_for_day
            }
            daily_metrics.append(day_metric_dict)
        if not daily_metrics:
            return pd.DataFrame()
        result_df = pd.DataFrame(daily_metrics)
        # 确保 'trade_time' 列在设置索引后被移除
        result_df['trade_time'] = pd.to_datetime(result_df['trade_time'])
        return result_df.set_index('trade_time')

    def _calculate_daily_structural_metrics(self, group: pd.DataFrame, continuous_group: pd.DataFrame, daily_series_for_day: pd.Series, atr_5: float, atr_14: float, atr_50: float, prev_day_metrics: dict, tick_df_for_day: pd.DataFrame = None, level5_df_for_day: pd.DataFrame = None, realtime_df_for_day: pd.DataFrame = None) -> dict:
        """
        【V19.3 · 职责集中重构版】
        - 核心重构: 剥离所有基于Tick数据的指标计算逻辑（active_volume_price_efficiency等），将其移至 _calculate_microstructure_metrics。
        - 核心目标: 此方法现在专注于基于分钟线和日线数据的结构指标计算，职责更清晰。
        """
        results = {}
        # 初始化所有指标，包括新的高频指标
        results['auction_impact_score'] = np.nan
        results['price_shock_factor'] = np.nan
        results['volatility_expansion_ratio'] = np.nan
        results['trend_quality_score'] = np.nan
        results['closing_momentum_index'] = np.nan
        results['volume_structure_skew'] = np.nan
        results['active_volume_price_efficiency'] = np.nan
        results['absorption_strength_index'] = np.nan
        results['distribution_pressure_index'] = np.nan
        results['order_flow_imbalance_score'] = np.nan
        results['buy_sweep_intensity'] = np.nan
        results['sell_sweep_intensity'] = np.nan
        results['vpin_score'] = np.nan
        results['vwap_mean_reversion_corr'] = np.nan
        group['amount'] = pd.to_numeric(group['amount'], errors='coerce')
        group['vol'] = pd.to_numeric(group['vol'], errors='coerce')
        total_volume = group['vol'].sum()
        total_volume_safe = total_volume if total_volume > 0 else np.nan
        day_open_qfq, day_high_qfq, day_low_qfq, day_close_qfq, pre_close_qfq = (
            daily_series_for_day.get('open_qfq'), daily_series_for_day.get('high_qfq'),
            daily_series_for_day.get('low_qfq'), daily_series_for_day.get('close_qfq'),
            daily_series_for_day.get('pre_close_qfq')
        )
        # --- 1. 能量密度与战场动力学 ---
        if pd.notna(atr_14) and atr_14 > 0:
            turnover_rate = pd.to_numeric(daily_series_for_day.get('turnover_rate_f'), errors='coerce')
            if pd.notna(turnover_rate):
                results['intraday_energy_density'] = np.log1p(turnover_rate) / atr_14
        # 修正：intraday_thrust_purity 现在只使用分钟线数据计算，不再依赖旧的 is_hf_data
        thrust_vector = (group['close'] - group['open']) * group['vol']
        absolute_energy = abs(group['close'] - group['open']) * group['vol']
        total_energy = absolute_energy.sum()
        if total_energy > 0:
            results['intraday_thrust_purity'] = thrust_vector.sum() / total_energy
        results['volume_burstiness_index'] = self._calculate_gini(group['vol'].values)
        if all(pd.notna(v) for v in [day_open_qfq, pre_close_qfq, atr_14]) and atr_14 > 0:
            results['auction_impact_score'] = (day_open_qfq - pre_close_qfq) / atr_14
        low_idx = group['low'].idxmin()
        if low_idx > 0 and low_idx < len(group) - 1:
            falling_phase = group.iloc[:low_idx+1]
            rebounding_phase = group.iloc[low_idx+1:]
            vwap_fall = (falling_phase['amount']).sum() / falling_phase['vol'].sum() if falling_phase['vol'].sum() > 0 else np.nan
            vwap_rebound = (rebounding_phase['amount']).sum() / rebounding_phase['vol'].sum() if rebounding_phase['vol'].sum() > 0 else np.nan
            if pd.notna(vwap_fall) and pd.notna(vwap_rebound) and vwap_fall > 0 and total_volume > 0:
                results['rebound_momentum'] = (vwap_rebound / vwap_fall - 1) * (rebounding_phase['vol'].sum() / total_volume) * 100
        price_range = day_high_qfq - day_low_qfq
        if price_range > 0 and total_volume > 0:
            high_level_threshold = day_high_qfq - 0.25 * price_range
            results['high_level_consolidation_volume'] = group[group['high'] >= high_level_threshold]['vol'].sum() / total_volume
        opening_period_df = group[group['trade_time'].dt.time < time(9, 59, 59)]
        if not opening_period_df.empty:
            opening_thrust_vector = (opening_period_df['close'] - opening_period_df['open']) * opening_period_df['vol']
            opening_absolute_energy = abs(opening_period_df['close'] - opening_period_df['open']) * opening_period_df['vol']
            if opening_absolute_energy.sum() > 0:
                results['opening_period_thrust'] = opening_thrust_vector.sum() / opening_absolute_energy.sum()
        # --- 2. 趋势、节律与代理筹码 ---
        if (day_high_qfq - day_low_qfq) > 0:
            if day_close_qfq > day_open_qfq:
                running_max = group['high'].cummax()
                results['pullback_depth_ratio'] = ((running_max - group['low']) / running_max).max()
            else:
                running_min = group['low'].cummin()
                results['pullback_depth_ratio'] = ((group['high'] - running_min) / running_min).max()
        total_path = group['close'].diff().abs().sum()
        if total_path > 0:
            results['trend_efficiency_ratio'] = abs(day_close_qfq - day_open_qfq) / total_path
        vwap_ma20 = continuous_group['minute_vwap'].rolling(window=20, min_periods=1).mean()
        position = np.sign(continuous_group['minute_vwap'] - vwap_ma20)
        results['mean_reversion_frequency'] = position.diff().ne(0).sum() / 4.0
        opening_df_rhythm = group[group['trade_time'].dt.time < time(10, 0)]
        midday_df_rhythm = group[(group['trade_time'].dt.time >= time(10, 0)) & (group['trade_time'].dt.time < time(14, 30))]
        tail_df_rhythm = group[group['trade_time'].dt.time >= time(14, 30)]
        avg_vol_opening = opening_df_rhythm['vol'].mean() if not opening_df_rhythm.empty else 0
        avg_vol_midday = midday_df_rhythm['vol'].mean() if not midday_df_rhythm.empty else 0
        avg_vol_tail = tail_df_rhythm['vol'].mean() if not tail_df_rhythm.empty else 0
        avg_vol_rest = (midday_df_rhythm['vol'].sum() + tail_df_rhythm['vol'].sum()) / (len(midday_df_rhythm) + len(tail_df_rhythm)) if (len(midday_df_rhythm) + len(tail_df_rhythm)) > 0 else 0
        if avg_vol_opening > 0 and avg_vol_rest > 0:
            results['opening_volume_impulse'] = avg_vol_opening / avg_vol_rest
        avg_vol_active = (opening_df_rhythm['vol'].sum() + tail_df_rhythm['vol'].sum()) / (len(opening_df_rhythm) + len(tail_df_rhythm)) if (len(opening_df_rhythm) + len(tail_df_rhythm)) > 0 else 0
        if avg_vol_midday > 0 and avg_vol_active > 0:
            results['midday_consolidation_level'] = avg_vol_midday / avg_vol_active
        pre_tail_df = group[(group['trade_time'].dt.time >= time(13, 0)) & (group['trade_time'].dt.time < time(14, 30))]
        avg_vol_pre_tail = pre_tail_df['vol'].mean() if not pre_tail_df.empty else 0
        if avg_vol_tail > 0 and avg_vol_pre_tail > 0:
            results['tail_volume_acceleration'] = avg_vol_tail / avg_vol_pre_tail
        daily_vwap = group['amount'].sum() / total_volume if total_volume > 0 else day_close_qfq
        if total_volume > 0:
            vp_proxy = continuous_group.groupby(pd.cut(continuous_group['close'], bins=20, duplicates='drop'))['vol'].sum()
            vp_prob = vp_proxy[vp_proxy > 0] / total_volume
            if not vp_prob.empty:
                entropy = -np.sum(vp_prob * np.log2(vp_prob))
                max_entropy = np.log2(len(vp_prob))
                results['volume_profile_entropy'] = entropy / max_entropy if max_entropy > 0 else 0.0
            winners_vol = continuous_group[continuous_group['minute_vwap'] < day_close_qfq]['vol'].sum()
            losers_vol = continuous_group[continuous_group['minute_vwap'] > day_close_qfq]['vol'].sum()
            if (winners_vol + losers_vol) > 0:
                results['intraday_pnl_imbalance'] = (winners_vol - losers_vol) / (winners_vol + losers_vol)
            if pd.notna(atr_14) and atr_14 > 0:
                weighted_variance = ((continuous_group['minute_vwap'] - daily_vwap)**2 * continuous_group['vol']).sum() / total_volume
                results['cost_dispersion_index'] = np.sqrt(weighted_variance) / atr_14
        # --- 3. VPOC、价值区与博弈效率 ---
        vp = continuous_group.groupby(pd.cut(continuous_group['close'], bins=20, duplicates='drop'))['vol'].sum()
        vpoc_interval = vp.idxmax() if not vp.empty else np.nan
        today_vpoc = vpoc_interval.mid if pd.notna(vpoc_interval) else day_close_qfq
        vpoc_volume_ratio = vp.max() / continuous_group['vol'].sum() if not vp.empty and continuous_group['vol'].sum() > 0 else 0
        if pd.notna(atr_14) and atr_14 > 0:
            deviation_magnitude = (day_close_qfq - today_vpoc) / atr_14
            results['vpoc_deviation_magnitude'] = deviation_magnitude
            results['vpoc_consensus_strength'] = vpoc_volume_ratio
            tail_period_df = group[group['trade_time'].dt.time >= time(14, 45)]
            if not tail_period_df.empty and not continuous_group.empty and continuous_group['vol'].mean() > 0:
                tail_force_factor = np.log1p(tail_period_df['vol'].mean() / continuous_group['vol'].mean())
                results['closing_conviction_score'] = deviation_magnitude * tail_force_factor
        today_vah, today_val = self._calculate_value_area(vp, continuous_group['vol'].sum(), vpoc_interval)
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
        # --- 4. 传统博弈效率 (兼容所有数据源) ---
        continuous_group['price_diff'] = continuous_group['close'] - continuous_group['open']
        up_minutes = continuous_group[continuous_group['price_diff'] > 0]
        if not up_minutes.empty and up_minutes['vol'].sum() > 0 and pd.notna(total_volume_safe) and day_open_qfq > 0:
            normalized_price_gain = up_minutes['price_diff'].sum() / day_open_qfq
            normalized_volume_cost = up_minutes['vol'].sum() / total_volume_safe
            if normalized_volume_cost > 0: results['upward_thrust_efficacy'] = normalized_price_gain / normalized_volume_cost
        down_minutes = continuous_group[continuous_group['price_diff'] < 0]
        if not down_minutes.empty and abs(down_minutes['price_diff']).sum() > 0 and pd.notna(total_volume_safe) and day_open_qfq > 0:
            normalized_price_drop = abs(down_minutes['price_diff']).sum() / day_open_qfq
            normalized_volume_cost = down_minutes['vol'].sum() / total_volume_safe
            if normalized_price_drop > 0: results['downward_absorption_efficacy'] = normalized_volume_cost / normalized_price_drop
        up_eff, down_eff = results.get('upward_thrust_efficacy'), results.get('downward_absorption_efficacy')
        if all(pd.notna(v) for v in [up_eff, down_eff]) and up_eff > 0 and down_eff > 0:
            results['net_vpa_score'] = np.log(up_eff / down_eff)
        if len(continuous_group) >= 30 and pd.notna(atr_14) and atr_14 > 0:
            from scipy.signal import find_peaks
            from itertools import combinations
            price_series = continuous_group['minute_vwap']
            rsi_series = ta.rsi(price_series, length=14).dropna()
            if not rsi_series.empty:
                aligned_price, aligned_rsi = price_series.loc[rsi_series.index], rsi_series
                price_low_indices, _ = find_peaks(-aligned_price.values, distance=15, prominence=aligned_price.std()*0.5)
                rsi_low_indices, _ = find_peaks(-aligned_rsi.values, distance=15, prominence=aligned_rsi.std()*0.5)
                price_high_indices, _ = find_peaks(aligned_price.values, distance=15, prominence=aligned_price.std()*0.5)
                rsi_high_indices, _ = find_peaks(aligned_rsi.values, distance=15, prominence=aligned_rsi.std()*0.5)
                bullish_strengths, bearish_strengths = [], []
                if len(price_low_indices) >= 2 and len(rsi_low_indices) >= 2:
                    for i1, i2 in combinations(price_low_indices, 2):
                        if aligned_price.iloc[i2] < aligned_price.iloc[i1]:
                            try:
                                r1 = aligned_rsi.iloc[rsi_low_indices[np.abs(rsi_low_indices - i1).argmin()]]
                                r2 = aligned_rsi.iloc[rsi_low_indices[np.abs(rsi_low_indices - i2).argmin()]]
                                if r2 > r1: bullish_strengths.append(((aligned_price.iloc[i1] - aligned_price.iloc[i2]) / atr_14) * (r2 - r1))
                            except IndexError: continue
                if len(price_high_indices) >= 2 and len(rsi_high_indices) >= 2:
                    for i1, i2 in combinations(price_high_indices, 2):
                        if aligned_price.iloc[i2] > aligned_price.iloc[i1]:
                            try:
                                r1 = aligned_rsi.iloc[rsi_high_indices[np.abs(rsi_high_indices - i1).argmin()]]
                                r2 = aligned_rsi.iloc[rsi_high_indices[np.abs(rsi_high_indices - i2).argmin()]]
                                if r2 < r1: bearish_strengths.append(((aligned_price.iloc[i2] - aligned_price.iloc[i1]) / atr_14) * (r2 - r1))
                            except IndexError: continue
                if bullish_strengths: results['divergence_conviction_score'] = max(bullish_strengths)
                elif bearish_strengths: results['divergence_conviction_score'] = min(bearish_strengths)
            returns = continuous_group['minute_vwap'].pct_change().fillna(0)
            weights = continuous_group['vol']
            if weights.sum() > 0:
                weighted_mean = np.average(returns, weights=weights)
                weighted_var = np.average((returns - weighted_mean)**2, weights=weights)
                if weighted_var > 0:
                    weighted_std = np.sqrt(weighted_var)
                    results['volatility_skew_index'] = np.average(((returns - weighted_mean) / weighted_std)**3, weights=weights)
        # --- 5. 前瞻性与收盘博弈分析 ---
        auction_period_df = group[group['trade_time'].dt.time >= time(14, 57)]
        if not auction_period_df.empty and not continuous_group.empty:
            close_before_auction = continuous_group['close'].iloc[-1]
            if pd.notna(close_before_auction) and close_before_auction > 0:
                auction_price_change = (day_close_qfq / close_before_auction - 1) * 100
                avg_vol_minute_continuous = continuous_group['vol'].mean()
                if avg_vol_minute_continuous > 0:
                    auction_volume_multiple = (auction_period_df['vol'].sum() / 3) / avg_vol_minute_continuous
                    results['auction_showdown_score'] = auction_price_change * np.log1p(auction_volume_multiple)
        if all(pd.notna(v) for v in [day_high_qfq, day_low_qfq, pre_close_qfq, atr_14]) and atr_14 > 0:
            true_range = max(day_high_qfq, pre_close_qfq) - min(day_low_qfq, pre_close_qfq)
            shock = true_range / atr_14
            direction = np.sign(day_close_qfq - day_open_qfq) if day_close_qfq != day_open_qfq else 1
            results['price_shock_factor'] = shock * direction
        if all(pd.notna(v) for v in [atr_5, atr_50]) and atr_50 > 0:
            results['volatility_expansion_ratio'] = atr_5 / atr_50
        # --- 6. V18.0 战场分析仪指标 ---
        if not continuous_group.empty and len(continuous_group) > 1:
            from scipy.stats import linregress
            vwap_series = continuous_group['minute_vwap'].dropna()
            if len(vwap_series) > 2:
                x = np.arange(len(vwap_series))
                slope, intercept, r_value, p_value, std_err = linregress(x, vwap_series)
                linearity = r_value**2
                vwap_max = vwap_series.max()
                vwap_min = vwap_series.min()
                vwap_range = vwap_max - vwap_min
                if vwap_range > 0:
                    if day_close_qfq > day_open_qfq:
                        pullback_control = ((vwap_series - vwap_min) / vwap_range).mean()
                    else:
                        pullback_control = ((vwap_max - vwap_series) / vwap_range).mean()
                    trend_quality = linearity * pullback_control
                    direction = np.sign(day_close_qfq - day_open_qfq) if day_close_qfq != day_open_qfq else 1
                    results['trend_quality_score'] = trend_quality * direction
            tail_df = continuous_group[continuous_group['trade_time'].dt.time >= time(14, 0)]
            if not tail_df.empty and pd.notna(atr_14) and atr_14 > 0 and total_volume_safe > 0:
                vwap_tail = (tail_df['amount'].sum() / tail_df['vol'].sum()) if tail_df['vol'].sum() > 0 else np.nan
                vwap_full = (continuous_group['amount'].sum() / continuous_group['vol'].sum()) if continuous_group['vol'].sum() > 0 else np.nan
                if all(pd.notna(v) for v in [vwap_tail, vwap_full]):
                    momentum_deviation = (vwap_tail - vwap_full) / atr_14
                    vol_ratio_tail = tail_df['vol'].sum() / total_volume_safe
                    results['closing_momentum_index'] = momentum_deviation * np.log1p(vol_ratio_tail)
            open_rhythm_df = continuous_group[continuous_group['trade_time'].dt.time < time(10, 0)]
            mid_rhythm_df = continuous_group[(continuous_group['trade_time'].dt.time >= time(10, 0)) & (continuous_group['trade_time'].dt.time < time(14, 30))]
            tail_rhythm_df = continuous_group[continuous_group['trade_time'].dt.time >= time(14, 30)]
            if not open_rhythm_df.empty and not mid_rhythm_df.empty and not tail_rhythm_df.empty:
                avg_vol_open = open_rhythm_df['vol'].mean()
                avg_vol_mid = mid_rhythm_df['vol'].mean()
                avg_vol_tail = tail_rhythm_df['vol'].mean()
                avg_vol_ends = (avg_vol_open + avg_vol_tail) / 2
                if avg_vol_ends > 0:
                    results['volume_structure_skew'] = avg_vol_mid / avg_vol_ends
        # --- 7. 微观结构动力学 (Microstructure Dynamics) ---
        # 调用重构后的微观结构计算中心
        microstructure_metrics = self._calculate_microstructure_metrics(
            tick_df=tick_df_for_day,
            level5_df=level5_df_for_day,
            realtime_df=realtime_df_for_day,
            minute_df=continuous_group,
            total_volume=total_volume_safe,
            group=group,
            daily_series_for_day=daily_series_for_day,
            atr_14=atr_14
        )
        results.update(microstructure_metrics)
        results['_today_vpoc'] = today_vpoc
        results['_today_vah'] = today_vah
        results['_today_val'] = today_val
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

    def _calculate_gini(self, array: np.ndarray) -> float:
        """计算基尼系数"""
        if array is None or len(array) < 2 or np.sum(array) == 0:
            return 0.0
        sorted_array = np.sort(array)
        n = len(array)
        cum_array = np.cumsum(sorted_array, dtype=float)
        return (n + 1 - 2 * np.sum(cum_array) / cum_array[-1]) / n

    def _calculate_microstructure_metrics(self, tick_df: pd.DataFrame, level5_df: pd.DataFrame, realtime_df: pd.DataFrame, minute_df: pd.DataFrame, total_volume: float, group: pd.DataFrame, daily_series_for_day: pd.Series, atr_14: float) -> dict:
        """
        【V21.9 · 全函数类型净化版】
        - 核心修复: 修复了 `liquidity_authenticity_score` 计算中 `Decimal` 与 `float` 的除法 `TypeError`。
        - 核心强化: 主动审查并修复了 `market_impact_cost` 计算循环中所有潜在的 `Decimal` 与 `float` 混合运算 `TypeError`，
                    通过在计算前进行显式的 `float()` 类型转换，确保了整个函数的数值计算类型安全。
        """
        from scipy.stats import norm, linregress
        results = {
            'order_flow_imbalance_score': np.nan,
            'buy_sweep_intensity': np.nan,
            'sell_sweep_intensity': np.nan,
            'vpin_score': np.nan,
            'vwap_mean_reversion_corr': np.nan,
            'active_volume_price_efficiency': np.nan,
            'absorption_strength_index': np.nan,
            'distribution_pressure_index': np.nan,
            'market_impact_cost': np.nan,
            'liquidity_slope': np.nan,
            'liquidity_authenticity_score': np.nan,
        }
        column_rename_map = {
            **{f'buy_price{i}': f'b{i}_p' for i in range(1, 6)},
            **{f'buy_volume{i}': f'b{i}_v' for i in range(1, 6)},
            **{f'sell_price{i}': f'a{i}_p' for i in range(1, 6)},
            **{f'sell_volume{i}': f'a{i}_v' for i in range(1, 6)},
        }
        if tick_df is None or tick_df.empty or total_volume == 0:
            return results
        # --- 意图与执行交叉验证：流动性真实性评分 ---
        if level5_df is not None and not level5_df.empty:
            level5_df_renamed = level5_df.copy()
            level5_df_renamed.rename(columns=column_rename_map, inplace=True)
            tick_df_sorted = tick_df.sort_index()
            level5_df_sorted = level5_df_renamed.sort_index()
            level5_cols = list(column_rename_map.values())
            merged_before = pd.merge_asof(tick_df_sorted, level5_df_sorted, on='trade_time', direction='backward')
            rename_map_pre = {col: f"{col}_pre" for col in level5_cols if col in merged_before.columns}
            merged_before.rename(columns=rename_map_pre, inplace=True)
            merged_after = pd.merge_asof(tick_df_sorted, level5_df_sorted, on='trade_time', direction='forward')
            rename_map_post = {col: f"{col}_post" for col in level5_cols if col in merged_after.columns}
            merged_after.rename(columns=rename_map_post, inplace=True)
            merged_full = pd.DataFrame()
            if not merged_before.empty and not merged_after.empty:
                merged_before.set_index('trade_time', inplace=True)
                merged_after.set_index('trade_time', inplace=True)
                post_cols = list(rename_map_post.values())
                cols_to_join = [col for col in post_cols if col in merged_after.columns]
                merged_full = merged_before.join(merged_after[cols_to_join])
            if not merged_full.empty:
                required_join_cols = ['a1_v_pre', 'b1_v_pre', 'a1_p_post', 'a1_p_pre', 'a1_v_post', 'b1_p_post', 'b1_p_pre', 'b1_v_post']
                if not all(col in merged_full.columns for col in required_join_cols):
                    # 移除调试信息
                    pass
                else:
                    merged_full.dropna(subset=required_join_cols, inplace=True)
                    if not merged_full.empty:
                        merged_full['is_buy_impact'] = (merged_full['type'] == 'B') & (merged_full['volume'] > 0.3 * merged_full['a1_v_pre'])
                        merged_full['is_sell_impact'] = (merged_full['type'] == 'S') & (merged_full['volume'] > 0.3 * merged_full['b1_v_pre'])
                        impact_ticks = merged_full[merged_full['is_buy_impact'] | merged_full['is_sell_impact']].copy()
                        scores = []
                        if not impact_ticks.empty:
                            buy_impacts = impact_ticks[impact_ticks['is_buy_impact']]
                            deception_mask_buy = (buy_impacts['a1_p_post'] == buy_impacts['a1_p_pre']) & (buy_impacts['a1_v_post'] >= 0.8 * buy_impacts['a1_v_pre'])
                            deception_scores = -buy_impacts.loc[deception_mask_buy, 'amount']
                            scores.append(deception_scores)
                            sell_impacts = impact_ticks[impact_ticks['is_sell_impact']]
                            support_mask_sell = (sell_impacts['b1_p_post'] == sell_impacts['b1_p_pre']) & (sell_impacts['b1_v_post'] >= 0.8 * sell_impacts['b1_v_pre'])
                            support_scores = sell_impacts.loc[support_mask_sell, 'amount']
                            scores.append(support_scores)
                            all_scores = pd.concat(scores)
                            if not all_scores.empty and all_scores.abs().sum() > 0:
                                total_day_amount = daily_series_for_day.get('amount', 0)
                                if total_day_amount > 0:
                                    # 将 all_scores.sum() 也转换为 float
                                    results['liquidity_authenticity_score'] = float(all_scores.sum()) / float(total_day_amount)
        # 1. VWAP均值回归相关性
        if minute_df is not None and not minute_df.empty and 'minute_vwap' in minute_df.columns and len(minute_df) > 1:
            daily_vwap = (minute_df['amount'].sum() / minute_df['vol'].sum()) if minute_df['vol'].sum() > 0 else np.nan
            if pd.notna(daily_vwap):
                deviation = minute_df['minute_vwap'] - daily_vwap
                results['vwap_mean_reversion_corr'] = deviation.autocorr(lag=1)
        # 2. 订单流失衡 (OFI)
        if level5_df is not None and not level5_df.empty and len(level5_df) > 1:
            df = level5_df[['buy_price1', 'buy_volume1', 'sell_price1', 'sell_volume1']].copy()
            df_prev = df.shift(1)
            delta_buy_price = df['buy_price1'] - df_prev['buy_price1']
            delta_sell_price = df['sell_price1'] - df_prev['sell_price1']
            ofi_static = np.where((delta_buy_price == 0) & (delta_sell_price == 0), df['buy_volume1'] - df_prev['buy_volume1'], 0)
            ofi_dynamic = np.where(delta_buy_price > 0, df_prev['buy_volume1'], 0)
            ofi_dynamic = np.where(delta_buy_price < 0, -df['buy_volume1'], ofi_dynamic)
            ofi_dynamic = np.where(delta_sell_price > 0, ofi_dynamic + df['sell_volume1'], ofi_dynamic)
            ofi_dynamic = np.where(delta_sell_price < 0, ofi_dynamic - df_prev['sell_volume1'], ofi_dynamic)
            ofi_series = ofi_static + ofi_dynamic
            if ofi_series.size > 0:
                results['order_flow_imbalance_score'] = np.nansum(ofi_series) / total_volume
        # 3. 扫单强度 (Sweep Intensity)
        buy_sweep_vol = 0
        sell_sweep_vol = 0
        min_sweep_len = 3
        tick_df['block'] = (tick_df['type'] != tick_df['type'].shift()).cumsum()
        tick_df['block_size'] = tick_df.groupby('block')['type'].transform('size')
        sweep_candidates = tick_df[(tick_df['block_size'] >= min_sweep_len) & (tick_df['type'].isin(['B', 'S']))]
        if not sweep_candidates.empty:
            for block_id, group_sweep in sweep_candidates.groupby('block'):
                trade_type = group_sweep['type'].iloc[0]
                prices = group_sweep['price']
                if trade_type == 'B' and prices.is_monotonic_increasing:
                    buy_sweep_vol += group_sweep['volume'].sum()
                elif trade_type == 'S' and prices.is_monotonic_decreasing:
                    sell_sweep_vol += group_sweep['volume'].sum()
        total_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
        total_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
        if total_buy_vol > 0: results['buy_sweep_intensity'] = buy_sweep_vol / total_buy_vol
        if total_sell_vol > 0: results['sell_sweep_intensity'] = sell_sweep_vol / total_sell_vol
        # 4. VPIN
        vpin_bucket_size = total_volume / 50
        vpin_window = 10
        if vpin_bucket_size > 0:
            tick_df['buy_vol'] = np.where(tick_df['type'] == 'B', tick_df['volume'], 0)
            tick_df['sell_vol'] = np.where(tick_df['type'] == 'S', tick_df['volume'], 0)
            tick_df['cum_vol'] = tick_df['volume'].cumsum()
            tick_df['bucket'] = (tick_df['cum_vol'] // vpin_bucket_size).astype(int)
            bucket_imbalance = tick_df.groupby('bucket').agg(buy_vol=('buy_vol', 'sum'), sell_vol=('sell_vol', 'sum'))
            bucket_imbalance['imbalance'] = bucket_imbalance['buy_vol'] - bucket_imbalance['sell_vol']
            if len(bucket_imbalance) > vpin_window:
                imbalance_std = bucket_imbalance['imbalance'].rolling(window=vpin_window).std().bfill()
                abs_imbalance = bucket_imbalance['imbalance'].abs()
                sigma_imbalance = imbalance_std.replace(0, np.nan)
                z_score = abs_imbalance / sigma_imbalance
                vpin_series = z_score.apply(lambda z: norm.cdf(z) if pd.notna(z) else np.nan)
                results['vpin_score'] = vpin_series.mean()
        # 5. 高频力学指标 (原位于 _calculate_daily_structural_metrics)
        day_open_qfq = daily_series_for_day.get('open_qfq')
        day_close_qfq = daily_series_for_day.get('close_qfq')
        net_active_volume = total_buy_vol - total_sell_vol
        price_change_in_atr = (day_close_qfq - day_open_qfq) / atr_14 if pd.notna(atr_14) and atr_14 > 0 else 0
        if net_active_volume != 0 and total_volume > 0:
            results['active_volume_price_efficiency'] = price_change_in_atr / (net_active_volume / total_volume)
        tick_df.index = tick_df.index.tz_convert('Asia/Shanghai')
        down_minutes_df = group[group['close'] < group['open']]
        up_minutes_df = group[group['close'] > group['open']]
        if not down_minutes_df.empty:
            active_buy_on_dip = 0
            active_sell_on_dip = 0
            for _, minute_row in down_minutes_df.iterrows():
                minute_start = minute_row['trade_time']
                minute_end = minute_start + pd.Timedelta(minutes=1)
                ticks_in_minute = tick_df[(tick_df.index >= minute_start) & (tick_df.index < minute_end)]
                active_buy_on_dip += ticks_in_minute[ticks_in_minute['type'] == 'B']['volume'].sum()
                active_sell_on_dip += ticks_in_minute[ticks_in_minute['type'] == 'S']['volume'].sum()
            if active_sell_on_dip > 0:
                results['absorption_strength_index'] = active_buy_on_dip / active_sell_on_dip
        if not up_minutes_df.empty:
            active_sell_on_rally = 0
            active_buy_on_rally = 0
            for _, minute_row in up_minutes_df.iterrows():
                minute_start = minute_row['trade_time']
                minute_end = minute_start + pd.Timedelta(minutes=1)
                ticks_in_minute = tick_df[(tick_df.index >= minute_start) & (tick_df.index < minute_end)]
                active_sell_on_rally += ticks_in_minute[ticks_in_minute['type'] == 'S']['volume'].sum()
                active_buy_on_rally += ticks_in_minute[ticks_in_minute['type'] == 'B']['volume'].sum()
            if active_buy_on_rally > 0:
                results['distribution_pressure_index'] = active_sell_on_rally / active_buy_on_rally
        # --- 实时盘口结构指标 ---
        if realtime_df is not None and not realtime_df.empty and level5_df is not None and not level5_df.empty:
            level5_df_processed = level5_df.copy()
            level5_df_processed.rename(columns=column_rename_map, inplace=True)
            numeric_cols = list(column_rename_map.values())
            for col in numeric_cols:
                if col in level5_df_processed.columns:
                    level5_df_processed[col] = pd.to_numeric(level5_df_processed[col], errors='coerce')
            combined_df = realtime_df.join(level5_df_processed)
            required_calc_cols = ['volume', 'b1_p', 'a1_p'] + numeric_cols
            if not all(col in combined_df.columns for col in required_calc_cols):
                return results
            combined_df.dropna(subset=required_calc_cols, inplace=True)
            if combined_df.empty:
                return results
            snapshot_volumes = combined_df['volume'].diff().fillna(0).clip(lower=0)
            # 1. 市场冲击成本 (Market Impact Cost)
            total_amount = daily_series_for_day.get('amount', 0)
            if not pd.notna(total_amount) or total_amount <= 0:
                # 移除调试探针
                pass
            if total_amount > 0:
                standard_amount = float(total_amount) * 0.001
                impact_costs = []
                for idx, row in combined_df.iterrows():
                    amount_to_fill = standard_amount
                    filled_amount = 0
                    filled_volume = 0
                    for i in range(1, 6):
                        price = row[f'a{i}_p']
                        vol = row[f'a{i}_v'] * 100
                        # 预防性修复：将 price 转换为 float
                        value = float(price) * vol
                        if amount_to_fill > value:
                            filled_amount += value
                            filled_volume += vol
                            amount_to_fill -= value
                        else:
                            # 预防性修复：将 price 转换为 float
                            filled_volume += amount_to_fill / float(price)
                            filled_amount += amount_to_fill
                            break
                    if filled_volume > 0:
                        exec_price = filled_amount / filled_volume
                        mid_price = (row['b1_p'] + row['a1_p']) / 2
                        if mid_price > 0:
                            # 预防性修复：将 mid_price 转换为 float
                            impact_costs.append((exec_price / float(mid_price) - 1) * 100)
                if impact_costs and snapshot_volumes.sum() > 0:
                    results['market_impact_cost'] = np.average(impact_costs, weights=snapshot_volumes.iloc[:len(impact_costs)])
            # 2. 盘口深度斜率 (Liquidity Slope)
            slopes = []
            for idx, row in combined_df.iterrows():
                mid_price = (row['b1_p'] + row['a1_p']) / 2
                if mid_price > 0:
                    # 预防性修复：将价格转换为 float 以便 scipy 处理
                    mid_price_float = float(mid_price)
                    ask_points_x = [(float(row[f'a{i}_p']) - mid_price_float) / mid_price_float for i in range(1, 6)]
                    ask_points_y = np.cumsum([row[f'a{i}_v'] * 100 for i in range(1, 6)])
                    if len(ask_points_x) > 1 and np.std(ask_points_x) > 0:
                        slope, _, _, _, _ = linregress(ask_points_x, ask_points_y)
                        slopes.append(slope)
            if slopes and snapshot_volumes.sum() > 0:
                results['liquidity_slope'] = np.average(slopes, weights=snapshot_volumes.iloc[:len(slopes)])
        return results

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










