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
        【V1.3 · 异步改造版】高级结构与行为指标预计算总指挥
        - 核心修复: 使用 await 调用改造后的异步锻造引擎 _forge_advanced_structural_metrics。
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
            processed_dates_in_chunk = set(minute_data_map.keys())
            required_dates_in_chunk = set(chunk_dates.date)
            missing_dates = required_dates_in_chunk - processed_dates_in_chunk
            if missing_dates:
                for missing_date in sorted(list(missing_dates)):
                    logger.warning(f"[{stock_code}] [{missing_date}] 跳过结构指标计算，缺失当日全部的分钟数据。")
            if not minute_data_map:
                logger.warning(f"[{stock_code}] 区块 {chunk_dates.min().date()} to {chunk_dates.max().date()} 无任何分钟数据，跳过整个区块。")
                continue
            # [代码修改开始]
            # 修复：使用 await 调用异步改造后的锻造引擎
            chunk_new_metrics_df = await self._forge_advanced_structural_metrics(minute_data_map, stock_code)
            # [代码修改结束]
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

    async def _forge_advanced_structural_metrics(self, minute_data_map: dict, stock_code: str) -> pd.DataFrame:
        # [代码修改开始]
        """
        【V17.1 · 异步改造版】
        - 核心修复: 将此方法改造为异步函数，并使用 sync_to_async 封装所有阻塞的数据库查询，解决 SynchronousOnlyOperation 错误。
        """
        # [代码修改结束]
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
        # [代码修改开始]
        # 修复：使用 sync_to_async 封装阻塞的数据库查询
        daily_df = pd.DataFrame.from_records(await sync_to_async(list)(daily_data_qs))
        # [代码修改结束]
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
        # [代码修改开始]
        # 修复：使用 sync_to_async 封装阻塞的数据库查询
        daily_basic_df = pd.DataFrame.from_records(await sync_to_async(list)(daily_basic_qs))
        # [代码修改结束]
        if not daily_basic_df.empty:
            daily_basic_df['trade_time'] = pd.to_datetime(daily_basic_df['trade_time']).dt.date
            daily_basic_df = daily_basic_df.set_index('trade_time')
            daily_df = daily_df.join(daily_basic_df, how='left')
        daily_df.ta.atr(high=daily_df['high_qfq'], low=daily_df['low_qfq'], close=daily_df['close_qfq'], length=14, append=True)
        base_atr = daily_df['ATRr_14']
        prev_day_metrics = {}
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
            continuous_mask = group['trade_time'].dt.time < time(14, 57, 0)
            continuous_group = group[continuous_mask].copy()
            if continuous_group.empty: continue
            continuous_group['minute_vwap'] = continuous_group['amount'] / continuous_group['vol'].replace(0, np.nan)
            continuous_group['minute_vwap'].fillna(method='ffill', inplace=True)
            continuous_group['minute_vwap'].fillna(group['open'].iloc[0], inplace=True)
            day_metric_dict = self._compute_all_structural_metrics(group, continuous_group, daily_series_for_day, base_atr_for_day, prev_day_metrics)
            day_metric_dict['trade_time'] = date
            prev_day_metrics = {
                'vpoc': day_metric_dict.pop('_today_vpoc', np.nan),
                'vah': day_metric_dict.pop('_today_vah', np.nan),
                'val': day_metric_dict.pop('_today_val', np.nan),
                'atr_14d': base_atr_for_day
            }
            daily_metrics.append(day_metric_dict)
        if not daily_metrics:
            return pd.DataFrame()
        result_df = pd.DataFrame(daily_metrics)
        return result_df.set_index(pd.to_datetime(result_df['trade_time']))

    def _compute_all_structural_metrics(self, group: pd.DataFrame, continuous_group: pd.DataFrame, daily_series_for_day: pd.Series, base_atr_for_day: float, prev_day_metrics: dict) -> dict:
        """
        【V1.0 · 新增 · 统一结构计算引擎】
        - 核心整合: 将所有分散的结构指标计算逻辑内聚于此，形成一个自包含的、职责单一的计算核心。
        - 包含逻辑: 能量密度、战场动力学、趋势、成交节律、VPOC、代理筹码、量价效能、高级博弈效率、价值区动态等。
        """
        results = {}
        total_volume = group['vol'].sum()
        total_volume_safe = total_volume if total_volume > 0 else np.nan
        day_open, day_high, day_low, day_close = group['open'].iloc[0], group['high'].max(), group['low'].min(), group['close'].iloc[-1]
        # --- 1. 能量密度与战场动力学 ---
        # 能量密度
        if pd.notna(base_atr_for_day) and base_atr_for_day > 0:
            turnover_rate = pd.to_numeric(daily_series_for_day.get('turnover_rate_f'), errors='coerce')
            if pd.notna(turnover_rate):
                results['intraday_energy_density'] = np.log1p(turnover_rate) / base_atr_for_day
        # 推力纯度
        thrust_vector = (group['close'] - group['open']) * group['vol']
        absolute_energy = abs(group['close'] - group['open']) * group['vol']
        total_energy = absolute_energy.sum()
        if total_energy > 0:
            results['intraday_thrust_purity'] = thrust_vector.sum() / total_energy
        # 成交量爆裂度
        results['volume_burstiness_index'] = self._calculate_gini(group['vol'].values)
        # 集合竞价冲击
        auction_df = group[group['trade_time'].dt.time < time(9, 30)]
        if not auction_df.empty and pd.notna(base_atr_for_day) and base_atr_for_day > 0:
            auction_vol = auction_df['vol'].sum()
            auction_range = auction_df['high'].max() - auction_df['low'].min()
            if total_volume > 0:
                results['auction_impact_score'] = (auction_range / base_atr_for_day) * (auction_vol / total_volume)
        # 战场动力学
        low_idx = group['low'].idxmin()
        if low_idx > 0 and low_idx < len(group) - 1:
            falling_phase = group.iloc[:low_idx+1]
            rebounding_phase = group.iloc[low_idx+1:]
            vwap_fall = (falling_phase['amount']).sum() / falling_phase['vol'].sum() if falling_phase['vol'].sum() > 0 else np.nan
            vwap_rebound = (rebounding_phase['amount']).sum() / rebounding_phase['vol'].sum() if rebounding_phase['vol'].sum() > 0 else np.nan
            if pd.notna(vwap_fall) and pd.notna(vwap_rebound) and vwap_fall > 0 and total_volume > 0:
                results['rebound_momentum'] = (vwap_rebound / vwap_fall - 1) * (rebounding_phase['vol'].sum() / total_volume) * 100
        price_range = day_high - day_low
        if price_range > 0 and total_volume > 0:
            high_level_threshold = day_high - 0.25 * price_range
            results['high_level_consolidation_volume'] = group[group['high'] >= high_level_threshold]['vol'].sum() / total_volume
        opening_period_df = group[group['trade_time'].dt.time < time(9, 59, 59)]
        if not opening_period_df.empty:
            opening_thrust_vector = (opening_period_df['close'] - opening_period_df['open']) * opening_period_df['vol']
            opening_absolute_energy = abs(opening_period_df['close'] - opening_period_df['open']) * opening_period_df['vol']
            if opening_absolute_energy.sum() > 0:
                results['opening_period_thrust'] = opening_thrust_vector.sum() / opening_absolute_energy.sum()
        # --- 2. 趋势、节律与代理筹码 ---
        # 趋势动力学
        if (day_high - day_low) > 0:
            if day_close > day_open:
                running_max = group['high'].cummax()
                results['pullback_depth_ratio'] = ((running_max - group['low']) / running_max).max()
            else:
                running_min = group['low'].cummin()
                results['pullback_depth_ratio'] = ((group['high'] - running_min) / running_min).max()
        total_path = group['close'].diff().abs().sum()
        if total_path > 0:
            results['trend_efficiency_ratio'] = abs(day_close - day_open) / total_path
        vwap_ma20 = continuous_group['minute_vwap'].rolling(window=20, min_periods=1).mean()
        position = np.sign(continuous_group['minute_vwap'] - vwap_ma20)
        results['mean_reversion_frequency'] = position.diff().ne(0).sum() / 4.0
        # 成交节律
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
        # 代理筹码动力学
        daily_vwap = group['amount'].sum() / total_volume if total_volume > 0 else day_close
        if total_volume > 0:
            vp_proxy = continuous_group.groupby(pd.cut(continuous_group['close'], bins=20))['vol'].sum()
            vp_prob = vp_proxy[vp_proxy > 0] / total_volume
            if not vp_prob.empty:
                entropy = -np.sum(vp_prob * np.log2(vp_prob))
                max_entropy = np.log2(len(vp_prob))
                results['volume_profile_entropy'] = entropy / max_entropy if max_entropy > 0 else 0.0
            winners_vol = continuous_group[continuous_group['minute_vwap'] < day_close]['vol'].sum()
            losers_vol = continuous_group[continuous_group['minute_vwap'] > day_close]['vol'].sum()
            if (winners_vol + losers_vol) > 0:
                results['intraday_pnl_imbalance'] = (winners_vol - losers_vol) / (winners_vol + losers_vol)
            if pd.notna(base_atr_for_day) and base_atr_for_day > 0:
                weighted_variance = ((continuous_group['minute_vwap'] - daily_vwap)**2 * continuous_group['vol']).sum() / total_volume
                results['cost_dispersion_index'] = np.sqrt(weighted_variance) / base_atr_for_day
        # --- 3. VPOC、价值区与博弈效率 ---
        # VPOC
        vp = continuous_group.groupby(pd.cut(continuous_group['close'], bins=20))['vol'].sum()
        vpoc_interval = vp.idxmax() if not vp.empty else np.nan
        today_vpoc = vpoc_interval.mid if pd.notna(vpoc_interval) else day_close
        vpoc_volume_ratio = vp.max() / continuous_group['vol'].sum() if not vp.empty and continuous_group['vol'].sum() > 0 else 0
        if pd.notna(base_atr_for_day) and base_atr_for_day > 0:
            deviation_magnitude = (day_close - today_vpoc) / base_atr_for_day
            results['vpoc_deviation_magnitude'] = deviation_magnitude
            results['vpoc_consensus_strength'] = vpoc_volume_ratio
            tail_period_df = group[group['trade_time'].dt.time >= time(14, 45)]
            if not tail_period_df.empty and not continuous_group.empty and continuous_group['vol'].mean() > 0:
                tail_force_factor = np.log1p(tail_period_df['vol'].mean() / continuous_group['vol'].mean())
                results['closing_conviction_score'] = deviation_magnitude * tail_force_factor
        # 价值区动态
        today_vah, today_val = self._calculate_value_area(vp, continuous_group['vol'].sum(), vpoc_interval)
        prev_vpoc, prev_atr = prev_day_metrics.get('vpoc'), prev_day_metrics.get('atr_14d')
        if all(pd.notna(v) for v in [today_vpoc, prev_vpoc, prev_atr]) and prev_atr > 0:
            results['value_area_migration'] = (today_vpoc - prev_vpoc) / prev_atr
        prev_vah, prev_val = prev_day_metrics.get('vah'), prev_day_metrics.get('val')
        if all(pd.notna(v) for v in [today_vah, today_val, prev_vah, prev_val]) and (today_vah - today_val) > 0:
            overlap_width = max(0, min(today_vah, prev_vah) - max(today_val, prev_val))
            results['value_area_overlap_pct'] = (overlap_width / (today_vah - today_val)) * 100
        if all(pd.notna(v) for v in [day_close, today_vpoc, today_vah, today_val]):
            if day_close > today_vah: results['closing_acceptance_type'] = 2
            elif day_close > today_vpoc: results['closing_acceptance_type'] = 1
            elif day_close < today_val: results['closing_acceptance_type'] = -2
            elif day_close < today_vpoc: results['closing_acceptance_type'] = -1
            else: results['closing_acceptance_type'] = 0
        # 量价效能
        continuous_group['price_diff'] = continuous_group['close'] - continuous_group['open']
        up_minutes = continuous_group[continuous_group['price_diff'] > 0]
        if not up_minutes.empty and up_minutes['vol'].sum() > 0 and pd.notna(total_volume_safe):
            normalized_price_gain = up_minutes['price_diff'].sum() / day_open
            normalized_volume_cost = up_minutes['vol'].sum() / total_volume_safe
            if normalized_volume_cost > 0: results['upward_thrust_efficacy'] = normalized_price_gain / normalized_volume_cost
        down_minutes = continuous_group[continuous_group['price_diff'] < 0]
        if not down_minutes.empty and abs(down_minutes['price_diff']).sum() > 0 and pd.notna(total_volume_safe):
            normalized_price_drop = abs(down_minutes['price_diff']).sum() / day_open
            normalized_volume_cost = down_minutes['vol'].sum() / total_volume_safe
            if normalized_price_drop > 0: results['downward_absorption_efficacy'] = normalized_volume_cost / normalized_price_drop
        up_eff, down_eff = results.get('upward_thrust_efficacy'), results.get('downward_absorption_efficacy')
        if all(pd.notna(v) for v in [up_eff, down_eff]) and up_eff > 0 and down_eff > 0:
            results['net_vpa_score'] = np.log(up_eff / down_eff)
        # 高级博弈效率
        if len(continuous_group) >= 30 and pd.notna(base_atr_for_day) and base_atr_for_day > 0:
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
                                if r2 > r1: bullish_strengths.append(((aligned_price.iloc[i1] - aligned_price.iloc[i2]) / base_atr_for_day) * (r2 - r1))
                            except IndexError: continue
                if len(price_high_indices) >= 2 and len(rsi_high_indices) >= 2:
                    for i1, i2 in combinations(price_high_indices, 2):
                        if aligned_price.iloc[i2] > aligned_price.iloc[i1]:
                            try:
                                r1 = aligned_rsi.iloc[rsi_high_indices[np.abs(rsi_high_indices - i1).argmin()]]
                                r2 = aligned_rsi.iloc[rsi_high_indices[np.abs(rsi_high_indices - i2).argmin()]]
                                if r2 < r1: bearish_strengths.append(((aligned_price.iloc[i2] - aligned_price.iloc[i1]) / base_atr_for_day) * (r2 - r1))
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
        # --- 4. 新增：收盘竞价摊牌分析 ---
        auction_period_df = group[group['trade_time'].dt.time >= time(14, 57)]
        if not auction_period_df.empty and not continuous_group.empty:
            close_before_auction = continuous_group['close'].iloc[-1]
            if pd.notna(close_before_auction) and close_before_auction > 0:
                auction_price_change = (day_close / close_before_auction - 1) * 100
                # 使用全天（连续交易时段）的平均每分钟成交量作为基准
                avg_vol_minute_continuous = continuous_group['vol'].mean()
                if avg_vol_minute_continuous > 0:
                    # 竞价成交量是其时段内总成交量，为匹配每分钟基准，除以3分钟
                    auction_volume_multiple = (auction_period_df['vol'].sum() / 3) / avg_vol_minute_continuous
                    results['auction_showdown_score'] = auction_price_change * np.log1p(auction_volume_multiple)
        # 存储当天计算的VPOC/VAH/VAL，供下一天使用
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

    def _calculate_gini(self, array: np.ndarray) -> float:
        """计算基尼系数"""
        if array is None or len(array) < 2 or np.sum(array) == 0:
            return 0.0
        sorted_array = np.sort(array)
        n = len(array)
        cum_array = np.cumsum(sorted_array, dtype=float)
        return (n + 1 - 2 * np.sum(cum_array) / cum_array[-1]) / n












