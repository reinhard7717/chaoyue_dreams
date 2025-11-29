# services/fund_flow_service.py

import asyncio
import logging
from django.utils import timezone
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from functools import reduce
from django.db import transaction
from asgiref.sync import sync_to_async
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockDailyBasic
from stock_models.advanced_metrics import BaseAdvancedFundFlowMetrics
from utils.model_helpers import (
    get_advanced_fund_flow_metrics_model_by_code,
    get_fund_flow_model_by_code,
    get_fund_flow_ths_model_by_code,
    get_fund_flow_dc_model_by_code,
    get_daily_data_model_by_code,
    get_minute_data_model_by_code_and_timelevel,
)

logger = logging.getLogger('services')

class AdvancedFundFlowMetricsService:
    """
    【V1.0 · 兵工厂模式】高级资金流指标服务
    - 核心职责: 封装所有高级资金流指标的加载、计算、融合与存储逻辑。
    - 架构优势: 实现业务逻辑与任务调度的完全解耦。
    """
    def __init__(self, debug_params: dict = None): # 新增 debug_params 参数
        self.max_lookback_days = 300
        self.debug_params = debug_params if debug_params is not None else {}

    def _get_safe_numeric_series(self, df: pd.DataFrame, col_name: str, default_value=0) -> pd.Series:
        """
        【V2.0 · 单行兼容版】类型安全的列获取辅助函数。
        修正了对单行DataFrame处理时返回标量导致后续链式调用失败的BUG。
        """
        # 彻底修正单行DataFrame问题
        if col_name not in df.columns:
            # 如果列不存在，创建一个填充了默认值的Series
            return pd.Series(default_value, index=df.index, dtype=float)
        # 使用 df[col_name] 保证返回的是Series，而不是标量，从根本上解决问题
        series = df[col_name]
        # 先转换为数值类型，再填充NaN
        return pd.to_numeric(series, errors='coerce').fillna(default_value)

    def _get_numeric_series_with_nan(self, df: pd.DataFrame, col_name: str) -> pd.Series:
        """
        安全地获取一个列作为数值型Series，并保留NaN。
        对单行DataFrame具有鲁棒性。
        """
        if col_name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        # 使用 df[col_name] 保证返回的是Series，而不是标量
        series = df[col_name]
        return pd.to_numeric(series, errors='coerce')

    async def run_precomputation(self, stock_code: str, is_incremental: bool, start_date_str: str = None, preloaded_minute_data: pd.DataFrame = None):
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental, start_date_str
        )
        if not is_incremental_final:
            await sync_to_async(MetricsModel.objects.filter(stock=stock_info).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        else:
            mode = "部分全量" if start_date_str else "增量"
            rollback_start_date = fetch_start_date if fetch_start_date else start_date_str
            if rollback_start_date:
                await sync_to_async(MetricsModel.objects.filter(stock=stock_info, trade_time__gte=rollback_start_date).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info, trade_time__gte=fetch_start_date).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if dates_to_process.empty:
            return 0
        initial_history_end_date = dates_to_process.min()
        historical_metrics_df = await self._load_historical_metrics(MetricsModel, stock_info, initial_history_end_date)
        CHUNK_SIZE = 50
        all_new_core_metrics_df = pd.DataFrame()
        for i in range(0, len(dates_to_process), CHUNK_SIZE):
            chunk_dates = dates_to_process[i:i + CHUNK_SIZE]
            if chunk_dates.empty:
                continue
            chunk_start_date, chunk_end_date = chunk_dates.min(), chunk_dates.max()
            chunk_raw_data_df = await self._load_and_merge_sources(stock_info, start_date=chunk_start_date, end_date=chunk_end_date)
            if chunk_raw_data_df.empty:
                continue
            # 核心修正：移除独立的 daily_vwap 计算步骤，将其整合到核心合成方法中
            self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, chunk_raw_data_df.index)
            chunk_new_metrics_df, _, _ = self._synthesize_and_forge_metrics(stock_code, chunk_raw_data_df)
            all_new_core_metrics_df = pd.concat([all_new_core_metrics_df, chunk_new_metrics_df])
        if hasattr(self, '_minute_df_daily_grouped'):
            del self._minute_df_daily_grouped
        if all_new_core_metrics_df.empty:
            return 0
        full_sequence_for_derivatives = pd.concat([historical_metrics_df, all_new_core_metrics_df])
        full_sequence_for_derivatives.sort_index(inplace=True)
        final_metrics_df = self._calculate_derivatives(stock_code, full_sequence_for_derivatives)
        chunk_to_save = final_metrics_df[final_metrics_df.index.isin(all_new_core_metrics_df.index)]
        total_processed_count = await self._prepare_and_save_data(stock_info, MetricsModel, chunk_to_save)
        return total_processed_count

    async def _initialize_context(self, stock_code: str, is_incremental: bool, start_date_str: str = None):
        from datetime import datetime
        stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        MetricsModel = get_advanced_fund_flow_metrics_model_by_code(stock_code)
        last_metric_date = None
        fetch_start_date = None
        # 移除所有调试性质的print语句
        if start_date_str:
            try:
                start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                is_incremental = True
                last_metric_date = start_date_obj - timedelta(days=1)
                fetch_start_date = start_date_obj - timedelta(days=self.max_lookback_days)
            except (ValueError, TypeError):
                is_incremental = True
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
                is_incremental = False
                fetch_start_date = None
        return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date
        
    async def _load_and_merge_sources(self, stock_info, data_dfs: dict, base_daily_df: pd.DataFrame):
        """
        【V2.4 · 净流量悖论修复版】
        - 核心修复: 解决了“净流量悖论”。在 `standardize_and_prepare` 中，彻底移除了为 THS 和 DC 数据源
                     从“净额”数据反推“买入/卖出毛坯额”的错误逻辑。
        - 核心思想: 停止凭空捏造数据。系统现在只使用 Tushare 提供的真实“毛坯”数据进行需要 gross flow 的计算。
                     如果 Tushare 数据缺失，相关指标将正确地输出为空(NaN)，而不是基于虚假数据得出错误结论，
                     从根本上保证了下游概率成本等核心指标的数据纯净性。
        """
        def standardize_and_prepare(df: pd.DataFrame, source: str) -> pd.DataFrame:
            if df.empty: return df
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            cols_to_numeric = [col for col in df.columns if col != 'trade_time' and 'code' not in col and 'name' not in col]
            df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
            required_amount_cols = [
                'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
                'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount',
            ]
            required_net_amount_cols = [
                'net_mf_amount', 'net_amount', 'net_amount_main', 'net_amount_xl',
                'net_amount_lg', 'net_amount_md', 'net_amount_sm', 'trade_count'
            ]
            for col in required_amount_cols + required_net_amount_cols:
                if col not in df.columns:
                    df[col] = 0.0
            if source == 'tushare':
                df['net_flow_tushare'] = df['net_mf_amount']
                df['main_force_net_flow_tushare'] = (df['buy_lg_amount'] - df['sell_lg_amount']) + (df['buy_elg_amount'] - df['sell_elg_amount'])
                df['retail_net_flow_tushare'] = (df['buy_sm_amount'] - df['sell_sm_amount']) + (df['buy_md_amount'] - df['sell_md_amount'])
                df['net_xl_amount_tushare'] = df['buy_elg_amount'] - df['sell_elg_amount']
                df['net_lg_amount_tushare'] = df['buy_lg_amount'] - df['sell_lg_amount']
                df['net_md_amount_tushare'] = df['buy_md_amount'] - df['sell_md_amount']
                df['net_sh_amount_tushare'] = df['buy_sm_amount'] - df['sell_sm_amount']
                return df
            elif source == 'ths':
                df['net_flow_ths'] = df['net_amount']
                df['main_force_net_flow_ths'] = df['buy_lg_amount']
                df['retail_net_flow_ths'] = df['buy_md_amount'] + df['buy_sm_amount']
                # [修改代码块] 移除所有从净额反推毛坯额的错误逻辑
                # 不再捏造 buy/sell amount，只保留真实的 net amount
                df['net_lg_amount_ths'] = df['buy_lg_amount']
                df['net_md_amount_ths'] = df['buy_md_amount']
                df['net_sh_amount_ths'] = df['buy_sm_amount']
                # 确保 buy/sell amount 列存在但为空，以维持 schema 一致性，但不填充虚假数据
                for col in ['buy_lg_amount', 'sell_lg_amount', 'buy_md_amount', 'sell_md_amount', 'buy_sm_amount', 'sell_sm_amount']:
                    if col not in df.columns: df[col] = np.nan
                return df
            elif source == 'dc':
                df['main_force_net_flow_dc'] = df['net_amount']
                df['retail_net_flow_dc'] = df['net_amount_md'] + df['net_amount_sm']
                df['net_flow_dc'] = df['main_force_net_flow_dc'] + df['retail_net_flow_dc']
                # [修改代码块] 移除所有从净额反推毛坯额的错误逻辑
                # 不再捏造 buy/sell amount，只保留真实的 net amount
                df['net_xl_amount_dc'] = df['net_amount_xl']
                df['net_lg_amount_dc'] = df['net_amount_lg']
                df['net_md_amount_dc'] = df['net_amount_md']
                df['net_sh_amount_dc'] = df['net_amount_sm']
                # 确保 buy/sell amount 列存在但为空
                for col in ['buy_elg_amount', 'sell_elg_amount', 'buy_lg_amount', 'sell_lg_amount', 'buy_md_amount', 'sell_md_amount', 'buy_sm_amount', 'sell_sm_amount']:
                    if col not in df.columns: df[col] = np.nan
                return df
            return df
        df_tushare = standardize_and_prepare(data_dfs['tushare'], 'tushare')
        df_ths = standardize_and_prepare(data_dfs['ths'], 'ths')
        df_dc = standardize_and_prepare(data_dfs['dc'], 'dc')
        if df_tushare.empty:
            return pd.DataFrame()
        merged_df = df_tushare
        other_flow_dfs = [df for df in [df_ths, df_dc] if not df.empty]
        if other_flow_dfs:
            for right_df in other_flow_dfs:
                overlap_cols = merged_df.columns.intersection(right_df.columns).drop('trade_time', errors='ignore')
                right_df_cleaned = right_df.drop(columns=overlap_cols, errors='ignore')
                merged_df = pd.merge(merged_df, right_df_cleaned, on='trade_time', how='left')
        merged_df = merged_df.sort_values('trade_time').set_index('trade_time')
        if not base_daily_df.empty:
            base_daily_df_copy = base_daily_df.copy()
            merged_df.index = pd.to_datetime(merged_df.index).normalize()
            base_daily_df_copy.index = pd.to_datetime(base_daily_df_copy.index).normalize()
            overlap_cols = merged_df.columns.intersection(base_daily_df_copy.columns)
            merged_df = merged_df.join(base_daily_df_copy.drop(columns=overlap_cols, errors='ignore'), how='left')
        return merged_df

    async def _get_daily_grouped_minute_data(self, stock_info: StockInfo, date_index: pd.DatetimeIndex, fetch_full_cols: bool = True, tick_data_map: dict = None, level5_data_map: dict = None, minute_data_map: dict = None):
        """
        【V1.14 · 日内数据回退增强版】不再查询数据库，仅处理由上游任务传入的日内数据maps。
        - 核心重构: 移除所有数据库查询逻辑，职责单一化为数据处理与聚合。
        - 核心逻辑: 遍历所需日期，优先尝试逐笔数据，若处理失败则回退到分钟数据。
        - 核心修复: 修正逐笔数据与Level5数据合并后，价格、成交量、成交额列名未被 `suffixes` 参数重命名的问题。
                    这些列名应保持原始名称，避免 `KeyError`。
        - 核心增强: 引入逐笔数据处理失败回退机制，确保分钟数据在逐笔数据不可用时能被利用。
        """
        import pandas as pd
        from django.utils import timezone
        if date_index.empty:
            return {}
        intraday_data_map = {}
        for date_obj in date_index.date:
            processed_with_tick_data = False
            if tick_data_map and date_obj in tick_data_map:
                try:
                    tick_df = tick_data_map[date_obj].copy()
                    if not all(col in tick_df.columns for col in ['price', 'volume', 'amount']):
                        logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 逐笔数据缺少'price', 'volume'或'amount'列，将尝试回退到分钟数据。")
                        raise ValueError("Missing essential tick data columns")
                    current_price_col = 'price'
                    current_volume_col = 'volume'
                    current_amount_col = 'amount'
                    has_original_type = 'type' in tick_df.columns
                    if level5_data_map and date_obj in level5_data_map:
                        level5_df = level5_data_map[date_obj]
                        tick_df_sorted = tick_df.sort_index()
                        level5_df_sorted = level5_df.sort_index()
                        merged_df_temp = pd.merge_asof(
                            tick_df_sorted.reset_index(),
                            level5_df_sorted.reset_index(),
                            on='trade_time',
                            direction='backward',
                            suffixes=('_tick', '_level5')
                        )
                        tick_df = merged_df_temp.set_index('trade_time')
                        if 'sell_price1' in tick_df.columns and 'buy_price1' in tick_df.columns:
                            conditions = [tick_df[current_price_col] >= tick_df['sell_price1'], tick_df[current_price_col] <= tick_df['buy_price1']]
                            choices = ['B', 'S']
                            tick_df['type'] = np.select(conditions, choices, default='M')
                        else:
                            logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 合并Level5数据后缺少买卖价格，无法重新评估'type'。")
                            if not has_original_type:
                                tick_df['type'] = 'M'
                    else:
                        if not has_original_type:
                            logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 缺少Level5数据且原始逐笔数据无'type'列，'type'将默认为'M'。")
                            tick_df['type'] = 'M'
                    if 'type' not in tick_df.columns:
                        logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 逐笔数据无'type'列，无法计算买卖量。")
                        buy_vol_per_minute = pd.Series(0, index=tick_df.index).resample('1min').sum()
                        sell_vol_per_minute = pd.Series(0, index=tick_df.index).resample('1min').sum()
                    else:
                        buy_vol_per_minute = tick_df[tick_df['type'] == 'B'].resample('1min')[current_volume_col].sum()
                        sell_vol_per_minute = tick_df[tick_df['type'] == 'S'].resample('1min')[current_volume_col].sum()
                    minute_df_from_ticks = tick_df.resample('1min').agg(
                        open=(current_price_col, 'first'), high=(current_price_col, 'max'), low=(current_price_col, 'min'),
                        close=(current_price_col, 'last'), vol=(current_volume_col, 'sum'), amount=(current_amount_col, 'sum')
                    ).dropna(subset=['open', 'high', 'low', 'close', 'vol', 'amount'])
                    minute_df_from_ticks['buy_vol_raw'] = buy_vol_per_minute
                    minute_df_from_ticks['sell_vol_raw'] = sell_vol_per_minute
                    minute_df_from_ticks.fillna(0, inplace=True)
                    intraday_data_map[date_obj] = self._group_minute_data_from_df(minute_df_from_ticks)
                    processed_with_tick_data = True
                except Exception as e:
                    logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 逐笔数据处理失败: {e}，将尝试回退到分钟数据。")
                    processed_with_tick_data = False
            if not processed_with_tick_data and minute_data_map and date_obj in minute_data_map:
                intraday_data_map[date_obj] = self._group_minute_data_from_df(minute_data_map[date_obj])
            elif not processed_with_tick_data:
                pass # 修改行：移除了此处的print调试信息
        return intraday_data_map

    def _calculate_all_metrics_for_day(self, stock_code: str, daily_data_series: pd.Series, intraday_data: pd.DataFrame, attributed_minute_df: pd.DataFrame, probabilistic_costs_dict: dict, tick_data_for_day: pd.DataFrame, level5_data_for_day: pd.DataFrame, realtime_data_for_day: pd.DataFrame, debug_mode: bool = False) -> tuple[dict, None]:
        """
        【V1.1 · 记忆注入版】
        - 核心修正: 在调用行为指标计算引擎前，将包含 `prev_metrics` 的完整 `daily_data_series` 传递下去，
                     确保时间序列指标（如CMF斜率）能够获取到前一日的状态。
        """
        day_metrics = {}
        daily_derived_metrics = self._calculate_daily_derived_metrics(daily_data_series, debug_mode=debug_mode)
        day_metrics.update(daily_derived_metrics)
        day_metrics.update(probabilistic_costs_dict)
        prob_costs_series = pd.Series(probabilistic_costs_dict)
        prob_costs_df_for_agg = pd.DataFrame([prob_costs_series], index=[daily_data_series.name])
        daily_df_for_agg = pd.DataFrame([daily_data_series.to_dict()], index=[daily_data_series.name])
        aggregate_pvwap_costs_df = self._calculate_aggregate_pvwap_costs(prob_costs_df_for_agg, daily_df_for_agg, debug_mode=debug_mode)
        if not aggregate_pvwap_costs_df.empty:
            day_metrics.update(aggregate_pvwap_costs_df.iloc[0].to_dict())
        # [修改代码块] 此处的 daily_data_series 已经包含了从上游传入的 prev_metrics
        updated_daily_data_series = pd.Series({**daily_data_series.to_dict(), **day_metrics}, name=daily_data_series.name)
        main_force_net_flow_calibrated = daily_derived_metrics.get('main_force_net_flow_calibrated')
        behavioral_metrics = self._compute_all_behavioral_metrics(
            attributed_minute_df, updated_daily_data_series,
            tick_data=tick_data_for_day,
            level5_data=level5_data_for_day,
            realtime_data=realtime_data_for_day,
            main_force_net_flow_calibrated=main_force_net_flow_calibrated,
            debug_mode=debug_mode
        )
        day_metrics.update(behavioral_metrics)
        day_metrics['trade_time'] = daily_data_series.name
        return day_metrics, None

    def _synthesize_and_forge_metrics(self, stock_code: str, merged_df: pd.DataFrame, tick_data_map: dict = None, level5_data_map: dict = None, minute_data_map: dict = None, realtime_data_map: dict = None, memory: dict = None) -> tuple[pd.DataFrame, dict, list, dict]:
        """
        【V12.0 · 记忆总线兼容版】
        - 核心升级: 新增 `memory` 参数和第四个返回值 `prev_metrics`，以完全兼容“统一记忆总线”架构。
        - 核心实现: 实现了完整的跨区块记忆加载、更新与返回逻辑，确保时间序列指标（如CMF）在全量回测中的计算连续性。
        """
        all_metrics_list = []
        attributed_minute_data_map = {}
        failures = []
        # [修改代码块] 初始化记忆
        prev_metrics = memory.copy() if memory is not None else {}
        num_days = len(merged_df)
        enable_probe = self.debug_params.get('enable_mfca_probe', False)
        target_date_str = self.debug_params.get('target_date')
        for i, (trade_date, daily_data_series) in enumerate(merged_df.iterrows()):
            debug_mode = (i == num_days - 1)
            date_obj = trade_date.date()
            is_target_date = str(date_obj) == target_date_str
            if enable_probe and is_target_date:
                print(f"\n[探针 A.1 - {stock_code} @ {date_obj}] 进入资金流服务 _synthesize_and_forge_metrics...")
            daily_amount = pd.to_numeric(daily_data_series.get('amount'), errors='coerce') * 1000
            daily_vol_shares = pd.to_numeric(daily_data_series.get('vol'), errors='coerce') * 100
            if pd.notna(daily_amount) and pd.notna(daily_vol_shares) and daily_vol_shares > 0:
                daily_data_series['daily_vwap'] = daily_amount / daily_vol_shares
            else:
                daily_data_series['daily_vwap'] = np.nan
            intraday_data = self._minute_df_daily_grouped.get(date_obj)
            if intraday_data is None or intraday_data.empty:
                failures.append({'stock_code': stock_code, 'trade_date': str(date_obj), 'reason': '当日分钟线/逐笔聚合数据缺失'})
                continue
            # [修改代码块] 将 prev_metrics 注入到 daily_data_series 中，供下游使用
            daily_data_series_with_mem = pd.concat([daily_data_series, pd.Series(prev_metrics, name=daily_data_series.name)])
            attribution_weights_df = self._calculate_intraday_attribution_weights(intraday_data, daily_data_series_with_mem)
            probabilistic_costs_dict, attributed_minute_df = self._calculate_probabilistic_costs(stock_code, attribution_weights_df, daily_data_series_with_mem, debug_mode=debug_mode)
            day_metrics, _ = self._calculate_all_metrics_for_day(
                stock_code, daily_data_series_with_mem, intraday_data, attributed_minute_df, probabilistic_costs_dict,
                tick_data_for_day=tick_data_map.get(date_obj),
                level5_data_for_day=level5_data_map.get(date_obj),
                realtime_data_for_day=realtime_data_map.get(date_obj),
                debug_mode=debug_mode
            )
            if enable_probe and is_target_date:
                print(f"[探针 A.2 - {stock_code} @ {date_obj}] 资金流指标计算完成。")
                print(f"  - 计算出的 avg_cost_main_buy: {day_metrics.get('avg_cost_main_buy')}")
            all_metrics_list.append(day_metrics)
            attributed_minute_data_map[date_obj] = attributed_minute_df.copy(deep=True)
            # [修改代码块] 更新并传递记忆
            next_prev_metrics = {
                'holistic_cmf': day_metrics.get('holistic_cmf'),
                'main_force_cmf': day_metrics.get('main_force_cmf'),
            }
            prev_metrics = next_prev_metrics
        if not all_metrics_list:
            return pd.DataFrame(), {}, failures, prev_metrics
        final_metrics_df = pd.DataFrame(all_metrics_list)
        final_metrics_df.set_index('trade_time', inplace=True)
        return final_metrics_df, attributed_minute_data_map, failures, prev_metrics

    def _calculate_daily_derived_metrics(self, daily_data_series: pd.Series, debug_mode: bool = False) -> dict:
        """
        【V2.1 · 信念指数逻辑迁移版】
        - 核心重构: 移除了旧的 `main_force_conviction_index` 计算逻辑。
                     新的动态信念指数依赖高频和分钟级数据，其计算已迁移至 `_compute_all_behavioral_metrics` 核心引擎中。
        """
        results = {}
        WAN = 10000.0
        def get_calibrated_value(target_col_name: str):
            consensus_map = {
                'net_flow_calibrated': ('net_flow_tushare', ['net_flow_ths', 'net_flow_dc']),
                'main_force_net_flow_calibrated': ('main_force_net_flow_tushare', ['main_force_net_flow_ths', 'main_force_net_flow_dc']),
                'retail_net_flow_calibrated': ('retail_net_flow_tushare', ['retail_net_flow_ths', 'retail_net_flow_dc']),
                'net_xl_amount_calibrated': ('net_xl_amount_tushare', ['net_xl_amount_dc']),
                'net_lg_amount_calibrated': ('net_lg_amount_tushare', ['net_lg_amount_ths', 'net_lg_amount_dc']),
                'net_md_amount_calibrated': ('net_md_amount_tushare', ['net_md_amount_ths', 'net_md_amount_dc']),
                'net_sh_amount_calibrated': ('net_sh_amount_tushare', ['net_sh_amount_ths', 'net_sh_amount_dc']),
            }
            if target_col_name not in consensus_map:
                return np.nan
            base_col, confirm_cols = consensus_map[target_col_name]
            base_value = pd.to_numeric(daily_data_series.get(base_col), errors='coerce')
            if pd.isna(base_value):
                for conf_col in confirm_cols:
                    alt_value = pd.to_numeric(daily_data_series.get(conf_col), errors='coerce')
                    if pd.notna(alt_value):
                        base_value = alt_value
                        break
            if pd.notna(base_value):
                confirmation_score = sum(1 for conf_col in confirm_cols if pd.notna(daily_data_series.get(conf_col)) and np.sign(base_value) == np.sign(pd.to_numeric(daily_data_series.get(conf_col), errors='coerce')))
                available_sources = sum(1 for conf_col in confirm_cols if pd.notna(daily_data_series.get(conf_col)))
                calibration_factor = (1 + confirmation_score) / (1 + available_sources) if available_sources > 0 else 1.0
                return base_value * calibration_factor
            return np.nan
        for col_name in ['net_flow_calibrated', 'main_force_net_flow_calibrated', 'retail_net_flow_calibrated', 'net_xl_amount_calibrated', 'net_lg_amount_calibrated', 'net_md_amount_calibrated', 'net_sh_amount_calibrated']:
            results[col_name] = get_calibrated_value(col_name)
        turnover_amount_yuan = pd.to_numeric(daily_data_series.get('amount'), errors='coerce') * 1000
        try:
            if turnover_amount_yuan > 0:
                base_flow_yuan = pd.to_numeric(daily_data_series.get('main_force_net_flow_tushare'), errors='coerce') * WAN
                confirm_flows_yuan = [pd.to_numeric(daily_data_series.get(c), errors='coerce') * WAN for c in ['main_force_net_flow_ths', 'main_force_net_flow_dc']]
                if pd.notna(base_flow_yuan):
                    deviations = [abs(conf_flow - base_flow_yuan) / turnover_amount_yuan for conf_flow in confirm_flows_yuan if pd.notna(conf_flow)]
                    results['flow_credibility_index'] = (1.0 - np.mean(deviations)) * 100 if deviations else 50.0
            else:
                results['flow_credibility_index'] = np.nan
        except Exception:
            results['flow_credibility_index'] = np.nan
        try:
            mf_flow_calibrated = results.get('main_force_net_flow_calibrated')
            retail_flow_calibrated = results.get('retail_net_flow_calibrated')
            if turnover_amount_yuan > 0 and pd.notna(mf_flow_calibrated) and pd.notna(retail_flow_calibrated):
                mf_flow_yuan = mf_flow_calibrated * WAN
                retail_flow_yuan = retail_flow_calibrated * WAN
                battle_volume_yuan = min(abs(mf_flow_yuan), abs(retail_flow_yuan))
                battle_turnover_yuan = 2 * battle_volume_yuan
                results['mf_retail_battle_intensity'] = (battle_turnover_yuan / turnover_amount_yuan) * 100
            else:
                results['mf_retail_battle_intensity'] = np.nan
        except Exception:
            results['mf_retail_battle_intensity'] = np.nan
        try:
            mf_buy_yuan = np.nansum([pd.to_numeric(daily_data_series.get('buy_lg_amount'), errors='coerce'), pd.to_numeric(daily_data_series.get('buy_elg_amount'), errors='coerce')]) * WAN
            mf_sell_yuan = np.nansum([pd.to_numeric(daily_data_series.get('sell_lg_amount'), errors='coerce'), pd.to_numeric(daily_data_series.get('sell_elg_amount'), errors='coerce')]) * WAN
            mf_total_activity_yuan = mf_buy_yuan + mf_sell_yuan
            if turnover_amount_yuan > 0:
                results['main_force_activity_ratio'] = (mf_total_activity_yuan / turnover_amount_yuan) * 100
                results['main_force_buy_rate_consensus'] = (mf_buy_yuan / turnover_amount_yuan) * 100
            else:
                results['main_force_activity_ratio'] = np.nan
                results['main_force_buy_rate_consensus'] = np.nan
            mf_net_calibrated = results.get('main_force_net_flow_calibrated')
            if mf_total_activity_yuan > 0 and pd.notna(mf_net_calibrated):
                results['main_force_flow_directionality'] = (mf_net_calibrated * WAN / mf_total_activity_yuan) * 100
            else:
                results['main_force_flow_directionality'] = np.nan
        except Exception:
            results['main_force_activity_ratio'] = np.nan
            results['main_force_buy_rate_consensus'] = np.nan
            results['main_force_flow_directionality'] = np.nan
        # 修改代码块：移除旧的信念指数计算逻辑，它将被新的动态信念指数在 `_compute_all_behavioral_metrics` 中取代
        results['main_force_conviction_index'] = np.nan
        try:
            mf_flow_calibrated = results.get('main_force_net_flow_calibrated')
            retail_flow_calibrated = results.get('retail_net_flow_calibrated')
            if pd.notna(mf_flow_calibrated) and pd.notna(retail_flow_calibrated):
                mf_flow_yuan = mf_flow_calibrated * WAN
                retail_flow_yuan = retail_flow_calibrated * WAN
                total_opinionated_flow_yuan = abs(mf_flow_yuan) + abs(retail_flow_yuan)
                if total_opinionated_flow_yuan > 0:
                    dominance_ratio = abs(retail_flow_yuan) / total_opinionated_flow_yuan
                    divergence_penalty = 1 if np.sign(mf_flow_yuan) != np.sign(retail_flow_yuan) and mf_flow_yuan != 0 and retail_flow_yuan != 0 else 0
                    results['retail_flow_dominance_index'] = np.sign(retail_flow_yuan) * dominance_ratio * (1 + divergence_penalty) * 100
                else:
                    results['retail_flow_dominance_index'] = np.nan
            else:
                results['retail_flow_dominance_index'] = np.nan
        except Exception:
            results['retail_flow_dominance_index'] = np.nan
        try:
            pct_change = pd.to_numeric(daily_data_series.get('pct_change'), errors='coerce')
            mf_flow_calibrated = results.get('main_force_net_flow_calibrated')
            if pd.notna(pct_change) and pd.notna(mf_flow_calibrated) and turnover_amount_yuan > 0:
                mf_flow_yuan = mf_flow_calibrated * WAN
                standardized_mf_flow = mf_flow_yuan / turnover_amount_yuan
                if abs(standardized_mf_flow) > 1e-6:
                    results['main_force_price_impact_ratio'] = (pct_change / 100) / standardized_mf_flow
                else:
                    results['main_force_price_impact_ratio'] = 0.0 if pct_change == 0 else np.inf * np.sign(pct_change)
            else:
                results['main_force_price_impact_ratio'] = np.nan
        except Exception:
            results['main_force_price_impact_ratio'] = np.nan
        return results

    def _calculate_probabilistic_costs(self, stock_code: str, minute_data_for_day: pd.DataFrame, daily_data: pd.Series, debug_mode: bool = False) -> tuple[dict, pd.DataFrame]:
        """
        【V6.15 · 诊断探针植入版】
        - 核心增强: 植入诊断探针，用于在debug模式下打印计算概率成本的关键输入与输出，定位成本指标计算失败的根源。
        """
        if minute_data_for_day is None or minute_data_for_day.empty:
            return {}, pd.DataFrame()
        day_results = {}
        cost_types = ['sm_buy', 'sm_sell', 'md_buy', 'md_sell', 'lg_buy', 'lg_sell', 'elg_buy', 'elg_sell']
        df_to_attribute = minute_data_for_day
        # 移除了所有与debug_mode和探针相关的print语句
        for cost_type in cost_types:
            size, direction = cost_type.split('_')
            db_vol_key = f'{direction}_{size}_vol'
            daily_vol_shares = pd.to_numeric(daily_data.get(db_vol_key), errors='coerce') * 100
            if pd.isna(daily_vol_shares) or daily_vol_shares == 0:
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                continue
            weight_col = f'{size}_{direction}_weight'
            if weight_col not in df_to_attribute.columns:
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                continue
            weight_series = df_to_attribute[weight_col]
            if weight_series.sum() < 1e-9:
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                continue
            attributed_vol = weight_series * daily_vol_shares
            df_to_attribute[f'{cost_type}_vol_attr'] = attributed_vol
            attributed_value = attributed_vol * df_to_attribute['minute_vwap']
            total_attributed_value = attributed_value.sum()
            total_attributed_vol = attributed_vol.sum()
            calculated_cost = total_attributed_value / total_attributed_vol if total_attributed_vol > 0 else np.nan
            day_results[f'avg_cost_{cost_type}'] = calculated_cost
        fully_attributed_df = self._attribute_minute_volume_to_players(df_to_attribute)
        return day_results, fully_attributed_df

    def _calculate_aggregate_pvwap_costs(self, pvwap_df: pd.DataFrame, daily_df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """
        【V2.1 · 最终修复版】
        - 核心修复: 修正了此前版本中，已计算的聚合成本（avg_cost_main_buy/sell）未被包含在返回结果中的致命缺陷。
                     现在，这些关键的中间成本被正确地添加到返回的DataFrame中，从而打通了整个计算链路的“最后一公里”，
                     确保下游指标（如retail_fomo_premium_index）能够获取到它们所依赖的数据。
        """
        if pvwap_df.empty or daily_df.empty:
            return pd.DataFrame()
        temp_df = pvwap_df.copy()
        result_agg_df = pd.DataFrame(index=pvwap_df.index)
        def weighted_average_cost(cost_keys, vol_keys):
            total_value = 0
            total_volume = 0
            for cost_key, vol_key in zip(cost_keys, vol_keys):
                cost = pd.to_numeric(temp_df.get(cost_key, np.nan).iloc[0], errors='coerce')
                vol = pd.to_numeric(daily_df.get(vol_key, 0).iloc[0], errors='coerce') * 100
                if pd.notna(cost) and pd.notna(vol) and vol > 0:
                    total_value += cost * vol
                    total_volume += vol
            return total_value / total_volume if total_volume > 0 else np.nan
        temp_df['avg_cost_main_buy'] = weighted_average_cost(['avg_cost_lg_buy', 'avg_cost_elg_buy'], ['buy_lg_vol', 'buy_elg_vol'])
        temp_df['avg_cost_main_sell'] = weighted_average_cost(['avg_cost_lg_sell', 'avg_cost_elg_sell'], ['sell_lg_vol', 'sell_elg_vol'])
        temp_df['avg_cost_retail_buy'] = weighted_average_cost(['avg_cost_sm_buy', 'avg_cost_md_buy'], ['buy_sm_vol', 'buy_md_vol'])
        temp_df['avg_cost_retail_sell'] = weighted_average_cost(['avg_cost_sm_sell', 'avg_cost_md_sell'], ['sell_sm_vol', 'sell_md_vol'])
        
        # 【最终修复】将计算出的聚合成本添加到返回结果中
        result_agg_df['avg_cost_main_buy'] = temp_df['avg_cost_main_buy']
        result_agg_df['avg_cost_main_sell'] = temp_df['avg_cost_main_sell']
        result_agg_df['avg_cost_retail_buy'] = temp_df['avg_cost_retail_buy']
        result_agg_df['avg_cost_retail_sell'] = temp_df['avg_cost_retail_sell']

        temp_df['daily_vwap'] = daily_df['daily_vwap']
        temp_df['atr_14d'] = daily_df['atr_14d']
        try:
            alpha = (temp_df['avg_cost_main_buy'] - temp_df['avg_cost_main_sell']) / temp_df['daily_vwap']
            result_agg_df['main_force_cost_alpha'] = alpha * 100
        except Exception:
            result_agg_df['main_force_cost_alpha'] = np.nan
        try:
            beta = (temp_df['avg_cost_retail_buy'] - temp_df['avg_cost_retail_sell']) / temp_df['daily_vwap']
            result_agg_df['retail_cost_beta'] = beta * 100
        except Exception:
            result_agg_df['retail_cost_beta'] = np.nan
        try:
            t0_spread = (temp_df['avg_cost_main_sell'] - temp_df['avg_cost_main_buy']) / temp_df['daily_vwap']
            result_agg_df['main_force_t0_spread_ratio'] = t0_spread * 100
            execution_alpha = (temp_df['avg_cost_main_sell'] - temp_df['daily_vwap']) / temp_df['atr_14d']
            result_agg_df['main_force_execution_alpha'] = execution_alpha
            efficiency = t0_spread / execution_alpha.replace(0, np.nan)
            result_agg_df['main_force_t0_efficiency'] = efficiency
        except Exception:
            result_agg_df['main_force_t0_spread_ratio'] = np.nan
            result_agg_df['main_force_execution_alpha'] = np.nan
            result_agg_df['main_force_t0_efficiency'] = np.nan
        try:
            mf_cost_premium = (temp_df['avg_cost_main_buy'] / temp_df['daily_vwap'] - 1)
            retail_cost_discount = (1 - temp_df['avg_cost_retail_sell'] / temp_df['daily_vwap'])
            temperature = mf_cost_premium - retail_cost_discount
            result_agg_df['flow_temperature_premium'] = temperature * 100
        except Exception:
            result_agg_df['flow_temperature_premium'] = np.nan
        return result_agg_df

    def _attribute_minute_volume_to_players(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1】将基础成交量归因为主力/散户的核心辅助函数。
        - 核心职责: 聚合基础的 *_vol_attr 列，生成 main_force_* 和 retail_* 级别的成交量列。
        """
        df = minute_df.copy()
        # 修改行：移除了所有与debug_params和probe_dates相关的探针初始化代码
        df['main_force_buy_vol'] = df.get('lg_buy_vol_attr', 0) + df.get('elg_buy_vol_attr', 0)
        df['main_force_sell_vol'] = df.get('lg_sell_vol_attr', 0) + df.get('elg_sell_vol_attr', 0)
        df['main_force_net_vol'] = df['main_force_buy_vol'] - df['main_force_sell_vol']
        df['retail_buy_vol'] = df.get('sm_buy_vol_attr', 0) + df.get('md_buy_vol_attr', 0)
        df['retail_sell_vol'] = df.get('sm_sell_vol_attr', 0) + df.get('md_sell_vol_attr', 0)
        df['retail_net_vol'] = df['retail_buy_vol'] - df['retail_sell_vol']
        # 修改行：移除了检查归因后成交量的探针print语句
        return df

    def _calculate_derivatives(self, stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
        derivatives_df = pd.DataFrame(index=consensus_df.index)
        import pandas_ta as ta
        # 修改行：移除了所有与debug_params和probe_dates相关的探针初始化代码
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedFundFlowMetrics.SLOPE_ACCEL_EXCLUSIONS
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
        ACCEL_WINDOW = 2
        sum_cols = [
            'net_flow_calibrated', 'main_force_net_flow_calibrated', 'retail_net_flow_calibrated',
            'net_xl_amount_calibrated', 'net_lg_amount_calibrated', 'net_md_amount_calibrated',
            'net_sh_amount_calibrated', 'main_force_on_peak_flow',
        ]
        UNIFIED_PERIODS = [1, 5, 13, 21, 55]
        for p in UNIFIED_PERIODS:
            if p <= 1: continue
            min_p = max(2, int(p * 0.8))
            for col in sum_cols:
                if col in consensus_df.columns:
                    source_series_for_sum = pd.to_numeric(consensus_df[col], errors='coerce')
                    # 修改行：移除了检查数据源缺失值的探针print语句
                    sum_col_name = f'{col}_sum_{p}d'
                    derivatives_df[sum_col_name] = source_series_for_sum.rolling(window=p, min_periods=min_p).sum()
                else:
                    pass # 修改行：移除了检查数据源列是否存在的探针print语句
        all_cols_to_derive = CORE_METRICS_TO_DERIVE + list(derivatives_df.columns)
        for col in all_cols_to_derive:
            base_col_name = col.split('_sum_')[0] if '_sum_' in col else col
            if base_col_name in SLOPE_ACCEL_EXCLUSIONS:
                continue
            if col in consensus_df.columns:
                source_series = pd.to_numeric(consensus_df[col], errors='coerce')
            elif col in derivatives_df.columns:
                source_series = derivatives_df[col]
            else:
                continue
            # 修改行：移除了检查数据源是否全为缺失值的探针print语句
            if source_series.isnull().all():
                continue
            for p in UNIFIED_PERIODS:
                calc_window = max(2, p)
                slope_col_name = f'{col}_slope_{p}d'
                slope_series = ta.slope(close=source_series.astype(float), length=calc_window)
                derivatives_df[slope_col_name] = slope_series
                if slope_series is not None and not slope_series.empty:
                    accel_col_name = f'{col}_accel_{p}d'
                    derivatives_df[accel_col_name] = ta.slope(close=slope_series.astype(float), length=ACCEL_WINDOW)
        return derivatives_df

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        records_to_save_df = final_df
        stock_code = stock_info.stock_code
        is_target_date_in_df = False
        if self.debug_params and self.debug_params.get('target_date'):
            target_date = pd.to_datetime(self.debug_params['target_date']).date()
            if target_date in records_to_save_df.index.date:
                is_target_date_in_df = True
        enable_probe = self.debug_params.get('enable_mfca_probe', False)
        if records_to_save_df.empty:
            return 0
        from django.db.models import DecimalField
        from decimal import Decimal, ROUND_HALF_UP
        decimal_fields = [f.name for f in MetricsModel._meta.get_fields() if isinstance(f, DecimalField)]
        for col in decimal_fields:
            if col in records_to_save_df.columns:
                records_to_save_df[col] = pd.to_numeric(records_to_save_df[col], errors='coerce')
                records_to_save_df[col] = records_to_save_df[col].replace([np.inf, -np.inf], np.nan)
        records_to_save_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        model_fields = {f.name for f in MetricsModel._meta.get_fields() if not f.is_relation and f.name != 'id'}
        df_filtered = records_to_save_df[[col for col in records_to_save_df.columns if col in model_fields]]
        # 修改代码块：升级探针，精确定位丢失的字段
        if enable_probe and is_target_date_in_df:
            print(f"\n[探针 S.1 - {stock_code} @ {self.debug_params['target_date']}] 进入 `_prepare_and_save_data` 保存前最终诊断...")
            filtered_cols_set = set(df_filtered.columns)
            model_fields_set = set(model_fields)
            print(f"  - 准备保存的DataFrame列数: {len(filtered_cols_set)}")
            print(f"  - 模型定义的字段数 (不含关系): {len(model_fields_set)}")
            if 'lower_shadow_absorption_strength' in filtered_cols_set:
                # 新增探针：打印目标字段的实际值
                target_row = df_filtered.loc[df_filtered.index.date == target_date]
                if not target_row.empty:
                    target_value = target_row['lower_shadow_absorption_strength'].iloc[0]
                    print(f"  - [关键检查] 'lower_shadow_absorption_strength' 存在于待保存数据中，值为: {target_value}")
            else:
                print("  - [关键警告] 'lower_shadow_absorption_strength' 在待保存数据中已丢失！")
            # 新增探针：通过集合运算找出丢失的字段
            missing_fields = model_fields_set - filtered_cols_set - {'trade_time', 'id', 'stock_id'} # 排除非数据字段
            if missing_fields:
                print(f"  - [!!!] 根本原因定位：模型中定义但数据中缺失的字段为: {missing_fields}")
            else:
                print("  - [检查通过] 数据列与模型字段匹配。")
        records_list = df_filtered.to_dict('records')
        @sync_to_async(thread_sensitive=True)
        def save_atomically(model, stock_obj, records_to_process):
            processed_count = 0
            for i, record_data in enumerate(records_to_process):
                trade_time = record_data.pop('trade_time').date()
                defaults_data = {key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in record_data.items()}
                for key, value in defaults_data.items():
                    if key in decimal_fields and pd.notna(value):
                        defaults_data[key] = Decimal(str(value)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
                try:
                    obj, created = model.objects.update_or_create(
                        stock=stock_obj,
                        trade_time=trade_time,
                        defaults=defaults_data
                    )
                    processed_count += 1
                except Exception as e:
                    logger.error(f"[{stock_obj.stock_code}] [资金流保存失败] 日期: {trade_time}, 错误: {e}")
            return processed_count
        records_for_atomic_save = []
        for record_date, record_data in zip(df_filtered.index, records_list):
            record_data['trade_time'] = record_date
            records_for_atomic_save.append(record_data)
        processed_count = await save_atomically(MetricsModel, stock_info, records_for_atomic_save)
        return processed_count

    def _calculate_advanced_behavioral_metrics(self, daily_df: pd.DataFrame, minute_df_attributed_grouped: dict) -> pd.DataFrame:
        """
        【V28.0 · 行为计算核心整合版】
        - 核心重构: 废弃所有零散的行为计算方法，引入统一的计算引擎 `_compute_all_behavioral_metrics`。
        - 核心思想: 本方法负责数据准备与调度，将所有分钟级行为指标的计算逻辑内聚到单一引擎中。
        """
        if not minute_df_attributed_grouped:
            return pd.DataFrame(index=daily_df.index)
        all_results = {}
        for date, daily_data in daily_df.iterrows():
            if date not in minute_df_attributed_grouped:
                continue
            minute_data = minute_df_attributed_grouped[date].copy()
            if minute_data.empty:
                continue
            # 调用统一计算引擎
            day_results = self._compute_all_behavioral_metrics(minute_data, daily_data)
            day_results['trade_time'] = date
            all_results[date] = day_results
        if not all_results:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(all_results, orient='index').set_index('trade_time')

    def _compute_all_behavioral_metrics(self, intraday_data: pd.DataFrame, daily_data: pd.Series, tick_data: pd.DataFrame = None, level5_data: pd.DataFrame = None, realtime_data: pd.DataFrame = None, main_force_net_flow_calibrated: float = None, debug_mode: bool = False) -> dict:
        """
        【V35.0 · 动态信念重构版】
        - 核心重构: 废弃原有的日线级 `main_force_conviction_index`，引入基于高频博弈的“动态主力信念指数”全新算法，
                     从攻击性、成本容忍度、韧性三个维度深度刻画主力信念。
        - 核心修复: 修正了 `dip_absorption_power` 中 `VolumeDominance` 组件因数据源口径不一导致的计算错误。
        - 核心增强: 为新的动态信念指数植入专属诊断探针。
        """
        from scipy.signal import find_peaks
        from datetime import time
        import numpy as np
        import pandas_ta as ta
        is_target_date = str(daily_data.name.date()) == self.debug_params.get('target_date')
        enable_probe = self.debug_params.get('enable_mfca_probe', False)
        if enable_probe and is_target_date:
            print(f"\n[探针 - {daily_data.name.date()}] 进入 `_compute_all_behavioral_metrics` 行为指标计算引擎...")
        results = {}
        if intraday_data.empty:
            return results
        daily_total_volume = daily_data.get('vol', 0) * 100
        daily_total_amount = pd.to_numeric(daily_data.get('amount', 0), errors='coerce') * 1000
        daily_vwap = daily_total_amount / daily_total_volume if daily_total_volume > 0 else np.nan
        atr = daily_data.get('atr_14d')
        day_open, day_close = daily_data.get('open_qfq'), daily_data.get('close_qfq')
        day_high, day_low = daily_data.get('high_qfq'), daily_data.get('low_qfq')
        pre_close = daily_data.get('pre_close_qfq')
        hf_analysis_df = pd.DataFrame()
        if tick_data is not None and not tick_data.empty and level5_data is not None and not level5_data.empty:
            merged_hf = pd.merge_asof(
                tick_data.sort_index(), level5_data.sort_index(),
                left_index=True, right_index=True, direction='backward'
            ).dropna(subset=['buy_price1', 'sell_price1', 'amount', 'volume'])
            if realtime_data is not None and not realtime_data.empty and not merged_hf.empty:
                realtime_prepped = realtime_data[['volume']].copy()
                realtime_prepped['snapshot_time'] = realtime_prepped.index
                merged_hf = pd.merge_asof(
                    merged_hf,
                    realtime_prepped,
                    left_index=True,
                    right_index=True,
                    direction='backward',
                    suffixes=('_tick', '_realtime')
                )
            if not merged_hf.empty:
                merged_hf.rename(columns={'volume_tick': 'volume'}, inplace=True)
                merged_hf['mid_price'] = (merged_hf['buy_price1'] + merged_hf['sell_price1']) / 2
                merged_hf['prev_mid_price'] = merged_hf['mid_price'].shift(1)
                buy_pressure = np.where(merged_hf['mid_price'] >= merged_hf['prev_mid_price'], merged_hf['buy_volume1'].shift(1), 0)
                sell_pressure = np.where(merged_hf['mid_price'] <= merged_hf['prev_mid_price'], merged_hf['sell_volume1'].shift(1), 0)
                merged_hf['ofi'] = buy_pressure - sell_pressure
                is_main_force_trade = merged_hf['amount'] > 200000
                is_retail_trade = merged_hf['amount'] < 50000
                merged_hf['main_force_ofi'] = np.where(is_main_force_trade, merged_hf['ofi'], 0)
                merged_hf['retail_ofi'] = np.where(is_retail_trade, merged_hf['ofi'], 0)
                merged_hf['mid_price_change'] = merged_hf['mid_price'].diff()
                if 'volume_realtime' in merged_hf.columns and 'snapshot_time' in merged_hf.columns:
                    snapshot_changed_mask = merged_hf['snapshot_time'] != merged_hf['snapshot_time'].shift(1)
                    volume_delta = merged_hf['volume_realtime'].diff().fillna(0)
                    merged_hf['market_vol_delta'] = np.where(snapshot_changed_mask, volume_delta, 0)
                hf_analysis_df = merged_hf
                results['main_force_ofi'] = hf_analysis_df['main_force_ofi'].sum()
                results['retail_ofi'] = hf_analysis_df['retail_ofi'].sum()
        # 修改代码块：重构动态主力信念指数 (dynamic_main_force_conviction_index)
        if not hf_analysis_df.empty:
            # 1. 攻击性 (Aggressiveness): 主力累积OFI的日内趋势斜率
            mf_ofi_cumsum = hf_analysis_df['main_force_ofi'].cumsum().fillna(0)
            aggressiveness_score = ta.slope(mf_ofi_cumsum, length=len(mf_ofi_cumsum)).iloc[-1] if len(mf_ofi_cumsum) > 1 else 0
            aggressiveness_component = np.tanh(aggressiveness_score / 1000) if pd.notna(aggressiveness_score) else 0.0
            # 2. 成本容忍度 (Cost Tolerance)
            avg_cost_main_buy = daily_data.get('avg_cost_main_buy')
            avg_cost_main_sell = daily_data.get('avg_cost_main_sell')
            cost_tolerance_component = 0.0
            if pd.notna(avg_cost_main_buy) and pd.notna(avg_cost_main_sell) and avg_cost_main_sell > 0:
                cost_tolerance_component = (avg_cost_main_buy / avg_cost_main_sell) - 1
            # 3. 韧性 (Resilience)
            market_pressure_zone = hf_analysis_df['ofi'] < 0
            mf_resilience_ofi = hf_analysis_df.loc[market_pressure_zone, 'main_force_ofi'].clip(lower=0).sum()
            total_mf_positive_ofi = hf_analysis_df['main_force_ofi'].clip(lower=0).sum()
            resilience_component = mf_resilience_ofi / total_mf_positive_ofi if total_mf_positive_ofi > 0 else 0.0
            # 最终裁决：三维信念融合
            results['main_force_conviction_index'] = (0.4 * aggressiveness_component + 0.4 * cost_tolerance_component + 0.2 * resilience_component) * 100
            if enable_probe and is_target_date:
                print(f"  [探针] main_force_conviction_index (动态信念) 计算:")
                print(f"    - Aggressiveness (OFI Slope): {aggressiveness_component:.4f}, Cost Tolerance (Buy/Sell Cost): {cost_tolerance_component:.4f}, Resilience (Absorption): {resilience_component:.4f}")
                print(f"    -> Final Score: {results['main_force_conviction_index']:.2f}")
        else:
            results['main_force_conviction_index'] = np.nan
        if not hf_analysis_df.empty:
            large_orders_df = hf_analysis_df[hf_analysis_df['amount'] > 200000]
            if not large_orders_df.empty:
                results['observed_large_order_size_avg'] = large_orders_df['amount'].mean()
            up_ticks = hf_analysis_df[hf_analysis_df['mid_price_change'] > 0]
            down_ticks = hf_analysis_df[hf_analysis_df['mid_price_change'] < 0]
            if not up_ticks.empty and not down_ticks.empty and up_ticks['mid_price_change'].sum() > 0 and down_ticks['mid_price_change'].abs().sum() > 0:
                vol_per_tick_up = up_ticks['volume'].sum() / (up_ticks['mid_price_change'].sum() * 100)
                vol_per_tick_down = down_ticks['volume'].sum() / (down_ticks['mid_price_change'].abs().sum() * 100)
                if vol_per_tick_down > 1e-6:
                    asymmetry_ratio = vol_per_tick_up / vol_per_tick_down
                    results['micro_price_impact_asymmetry'] = np.log1p(asymmetry_ratio) if asymmetry_ratio > 0 else np.nan
            hf_analysis_df['prev_a1_p'] = hf_analysis_df['sell_price1'].shift(1)
            hf_analysis_df['prev_b1_p'] = hf_analysis_df['buy_price1'].shift(1)
            ask_clearing_mask = (hf_analysis_df['type'] == 'B') & (hf_analysis_df['price'] == hf_analysis_df['prev_a1_p'])
            ask_clearing_vol = hf_analysis_df.loc[ask_clearing_mask, 'volume'].sum()
            bid_clearing_mask = (hf_analysis_df['type'] == 'S') & (hf_analysis_df['price'] == hf_analysis_df['prev_b1_p'])
            bid_clearing_vol = hf_analysis_df.loc[bid_clearing_mask, 'volume'].sum()
            total_cleared_vol = ask_clearing_vol + bid_clearing_vol
            if daily_total_volume > 0:
                results['order_book_clearing_rate'] = (total_cleared_vol / daily_total_volume) * 100
            try:
                wash_trade_vol = 0
                if 'market_vol_delta' in hf_analysis_df.columns:
                    df_wash = hf_analysis_df[['type', 'volume', 'price', 'market_vol_delta']].copy()
                    df_wash['direction'] = df_wash['type'].map({'B': 1, 'S': -1}).fillna(0)
                    df_wash['prev_direction'] = df_wash['direction'].shift(1)
                    df_wash['prev_volume'] = df_wash['volume'].shift(1)
                    df_wash['price_change_ratio'] = df_wash['price'].pct_change().abs()
                    wash_mask = (
                        (df_wash['direction'] * df_wash['prev_direction'] == -1) &
                        (np.abs(df_wash['volume'] - df_wash['prev_volume']) / df_wash['prev_volume'] < 0.2) &
                        (df_wash['price_change_ratio'] < 0.001) &
                        (df_wash['market_vol_delta'] > df_wash['volume'] * 2)
                    )
                    wash_trade_vol = df_wash.loc[wash_mask, 'market_vol_delta'].sum()
                if daily_total_volume > 0:
                    results['wash_trade_intensity'] = (wash_trade_vol / daily_total_volume) * 100
            except Exception:
                results['wash_trade_intensity'] = np.nan
            try:
                df_book = hf_analysis_df.copy()
                weighted_buy_vol = pd.Series(0, index=df_book.index); weighted_sell_vol = pd.Series(0, index=df_book.index)
                total_buy_value = pd.Series(0, index=df_book.index); total_sell_value = pd.Series(0, index=df_book.index)
                for i in range(1, 6):
                    weight = 1 / i
                    weighted_buy_vol += df_book[f'buy_volume{i}'] * weight; weighted_sell_vol += df_book[f'sell_volume{i}'] * weight
                    total_buy_value += df_book[f'buy_volume{i}'] * df_book[f'buy_price{i}']; total_sell_value += df_book[f'sell_volume{i}'] * df_book[f'sell_price{i}']
                df_book['imbalance'] = (weighted_buy_vol - weighted_sell_vol) / (weighted_buy_vol + weighted_sell_vol).replace(0, np.nan)
                df_book['liquidity_supply_ratio'] = total_buy_value / total_sell_value.replace(0, np.nan)
                time_diffs = df_book.index.to_series().diff().dt.total_seconds().fillna(0)
                if time_diffs.sum() > 0:
                    results['order_book_imbalance'] = np.average(df_book['imbalance'].dropna(), weights=time_diffs[df_book['imbalance'].notna()]) * 100
                    results['order_book_liquidity_supply'] = np.average(df_book['liquidity_supply_ratio'].dropna(), weights=time_diffs[df_book['liquidity_supply_ratio'].notna()])
                if 'market_vol_delta' in df_book.columns and df_book['imbalance'].var() > 0 and df_book['market_vol_delta'].var() > 0:
                    results['imbalance_effectiveness'] = df_book['imbalance'].corr(df_book['market_vol_delta'])
            except Exception:
                results['order_book_imbalance'] = np.nan; results['order_book_liquidity_supply'] = np.nan; results['imbalance_effectiveness'] = np.nan
            try:
                df_static = hf_analysis_df.copy()
                large_order_threshold_value = 500000
                pressure_mask = (df_static['sell_volume1'] * df_static['sell_price1'] > large_order_threshold_value) | (df_static['sell_volume2'] * df_static['sell_price2'] > large_order_threshold_value)
                support_mask = (df_static['buy_volume1'] * df_static['buy_price1'] > large_order_threshold_value) | (df_static['buy_volume2'] * df_static['buy_price2'] > large_order_threshold_value)
                time_diffs = df_static.index.to_series().diff().dt.total_seconds().fillna(0)
                pressure_strength = 0; support_strength = 0
                if 'market_vol_delta' in df_static.columns:
                    market_activity = df_static['market_vol_delta'].rolling(window=20, min_periods=1).mean().replace(0, np.nan)
                    activity_factor = 1 / np.log1p(market_activity)
                    pressure_strength = (time_diffs * activity_factor)[pressure_mask].sum()
                    support_strength = (time_diffs * activity_factor)[support_mask].sum()
                else:
                    pressure_strength = time_diffs[pressure_mask].sum(); support_strength = time_diffs[support_mask].sum()
                total_trading_seconds = (df_static.index.max() - df_static.index.min()).total_seconds()
                if total_trading_seconds > 0:
                    results['large_order_pressure'] = (pressure_strength / total_trading_seconds) * 100
                    results['large_order_support'] = (support_strength / total_trading_seconds) * 100
            except Exception:
                results['large_order_pressure'] = np.nan; results['large_order_support'] = np.nan
            try:
                df_exhaust = hf_analysis_df.copy()
                df_exhaust['prev_a1_v'] = df_exhaust['sell_volume1'].shift(1)
                df_exhaust['prev_b1_v'] = df_exhaust['buy_volume1'].shift(1)
                buy_exhaustion_mask = df_exhaust['sell_price1'] > df_exhaust['prev_a1_p']
                buy_exhausted_vol = df_exhaust.loc[buy_exhaustion_mask, 'prev_a1_v'].sum()
                sell_exhaustion_mask = df_exhaust['buy_price1'] < df_exhaust['prev_b1_p']
                sell_exhausted_vol = df_exhaust.loc[sell_exhaustion_mask, 'prev_b1_v'].sum()
                if daily_total_volume > 0:
                    results['buy_quote_exhaustion_rate'] = (buy_exhausted_vol / daily_total_volume) * 100
                    results['sell_quote_exhaustion_rate'] = (sell_exhausted_vol / daily_total_volume) * 100
            except Exception:
                results['buy_quote_exhaustion_rate'] = np.nan; results['sell_quote_exhaustion_rate'] = np.nan
        if pd.notna(daily_vwap) and daily_total_volume > 0 and pd.notna(atr) and atr > 0 and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            price_deviation_value = (intraday_data['minute_vwap'] - daily_vwap) * intraday_data['vol_shares']
            results['vwap_control_strength'] = price_deviation_value.sum() / (atr * daily_total_volume)
            price_dev_series = intraday_data['minute_vwap'] - daily_vwap
            mf_net_flow_series = intraday_data['main_force_net_vol']
            if price_dev_series.var() != 0 and mf_net_flow_series.var() != 0 and len(price_dev_series) > 1:
                correlation = price_dev_series.corr(mf_net_flow_series)
                results['main_force_vwap_guidance'] = correlation if pd.notna(correlation) else np.nan
            position_vs_vwap = np.sign(intraday_data['minute_vwap'] - daily_vwap)
            crossings = position_vs_vwap.diff().ne(0)
            results['vwap_crossing_intensity'] = intraday_data.loc[crossings, 'vol_shares'].sum() / daily_total_volume
            twap = intraday_data['minute_vwap'].mean()
            if pd.notna(twap) and twap > 0:
                results['vwap_structure_skew'] = (daily_vwap - twap) / twap * 100
        opening_battle_df = intraday_data[(intraday_data.index.time >= time(9, 30)) & (intraday_data.index.time <= time(9, 45))]
        if not opening_battle_df.empty and len(opening_battle_df) > 1 and pd.notna(atr) and atr > 0 and 'close' in opening_battle_df.columns and 'open' in opening_battle_df.columns and 'vol_shares' in opening_battle_df.columns and 'minute_vwap' in opening_battle_df.columns and 'main_force_net_vol' in opening_battle_df.columns:
            price_gain = (opening_battle_df['close'].iloc[-1] - opening_battle_df['open'].iloc[0]) / atr
            battle_amount = (opening_battle_df['vol_shares'] * opening_battle_df['minute_vwap']).sum()
            if battle_amount > 0:
                mf_power = opening_battle_df['main_force_net_vol'].sum() * opening_battle_df['minute_vwap'].mean() / battle_amount
                results['opening_battle_result'] = np.sign(price_gain) * np.sqrt(abs(price_gain)) * (1 + mf_power) * 100
        if pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low) and 'high' in intraday_data.columns and 'low' in intraday_data.columns and 'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            body_high, body_low = max(day_open, day_close), min(day_open, day_close)
            upper_shadow_df = intraday_data[intraday_data['high'] > body_high]
            if not upper_shadow_df.empty and upper_shadow_df['vol_shares'].sum() > 0:
                mf_sell_in_shadow = abs(upper_shadow_df[upper_shadow_df['main_force_net_vol'] < 0]['main_force_net_vol'].sum())
                results['upper_shadow_selling_pressure'] = (mf_sell_in_shadow / upper_shadow_df['vol_shares'].sum())
            lower_shadow_df = intraday_data[intraday_data['low'] < body_low]
            day_range = day_high - day_low
            if not lower_shadow_df.empty and lower_shadow_df['vol_shares'].sum() > 0 and day_range > 0:
                shadow_volume = lower_shadow_df['vol_shares'].sum()
                mf_net_in_shadow = lower_shadow_df['main_force_net_vol'].sum()
                FlowComponent = np.tanh(mf_net_in_shadow / shadow_volume)
                RecoveryComponent = (body_low - day_low) / day_range
                OFI_Component = 0.0
                if not hf_analysis_df.empty:
                    hf_shadow_zone = hf_analysis_df[hf_analysis_df['price'] < body_low]
                    if not hf_shadow_zone.empty:
                        mf_ofi_in_shadow = hf_shadow_zone['main_force_ofi'].sum()
                        total_abs_ofi_day = hf_analysis_df['ofi'].abs().sum()
                        if total_abs_ofi_day > 0:
                            OFI_Component = mf_ofi_in_shadow / total_abs_ofi_day
                results['lower_shadow_absorption_strength'] = (0.5 * FlowComponent + 0.3 * RecoveryComponent + 0.2 * OFI_Component) * 100
                if enable_probe and is_target_date:
                    print(f"  [探针] lower_shadow_absorption_strength 计算:")
                    print(f"    - body_low: {body_low:.2f}, day_low: {day_low:.2f}, day_high: {day_high:.2f}, day_range: {day_range:.2f}")
                    print(f"    - shadow_volume: {shadow_volume:,.0f}, mf_net_in_shadow: {mf_net_in_shadow:,.0f}")
                    print(f"    - FlowComponent: {FlowComponent:.4f}, RecoveryComponent: {RecoveryComponent:.4f}, OFI_Component: {OFI_Component:.4f}")
                    print(f"    -> Final Score: {results['lower_shadow_absorption_strength']:.2f}")
            else:
                results['lower_shadow_absorption_strength'] = np.nan
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and 'minute_vwap' in continuous_trading_df.columns:
            peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
            troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            results['dip_absorption_power'] = np.nan
            results['rally_distribution_pressure'] = np.nan
            if not hf_analysis_df.empty and pd.notna(daily_vwap):
                absorption_zone_hf = hf_analysis_df[hf_analysis_df['price'] < daily_vwap]
                if not absorption_zone_hf.empty:
                    offensive_ofi = absorption_zone_hf['main_force_ofi'].clip(lower=0).sum()
                    total_abs_ofi_day = hf_analysis_df['ofi'].abs().sum()
                    OffensiveOFI_Component = offensive_ofi / total_abs_ofi_day if total_abs_ofi_day > 0 else 0.0
                    mf_buy_cost_in_dip = (absorption_zone_hf['price'] * absorption_zone_hf['main_force_ofi'].clip(lower=0)).sum() / offensive_ofi if offensive_ofi > 0 else np.nan
                    CostAdvantage_Component = (daily_vwap - mf_buy_cost_in_dip) / atr if pd.notna(mf_buy_cost_in_dip) and pd.notna(atr) and atr > 0 else 0.0
                    # 修改代码块：修复 dip_absorption_power 的 VolumeDominance 计算
                    total_mf_buy_vol_day = intraday_data['main_force_buy_vol'].sum()
                    # 使用分钟级归因数据重新计算分子，确保口径统一
                    mf_buy_vol_in_dip = intraday_data[intraday_data['minute_vwap'] < daily_vwap]['main_force_buy_vol'].sum()
                    VolumeDominance_Component = mf_buy_vol_in_dip / total_mf_buy_vol_day if total_mf_buy_vol_day > 0 else 0.0
                    results['dip_absorption_power'] = (0.5 * OffensiveOFI_Component + 0.3 * CostAdvantage_Component + 0.2 * VolumeDominance_Component) * 100
                    if enable_probe and is_target_date:
                        print(f"  [探针] dip_absorption_power (战略吸筹) 计算:")
                        print(f"    - OffensiveOFI: {OffensiveOFI_Component:.4f}, CostAdvantage: {CostAdvantage_Component:.4f}, VolumeDominance (已修复): {VolumeDominance_Component:.4f}")
                        print(f"    -> Final Score: {results['dip_absorption_power']:.2f}")
                total_rally_price_change = 0
                total_deceptive_pressure = 0
                for i in range(len(turning_points) - 1):
                    start_idx, end_idx = turning_points[i], turning_points[i+1]
                    window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                    if window_df.empty or len(window_df) < 2: continue
                    if window_df['minute_vwap'].iloc[-1] > window_df['minute_vwap'].iloc[0]:
                        start_time, end_time = window_df.index[0], window_df.index[-1]
                        rally_hf_df = hf_analysis_df[(hf_analysis_df.index >= start_time) & (hf_analysis_df.index <= end_time)]
                        if not rally_hf_df.empty:
                            price_change_in_rally = window_df['minute_vwap'].iloc[-1] - window_df['minute_vwap'].iloc[0]
                            total_rally_price_change += price_change_in_rally
                            mf_ofi_in_rally = rally_hf_df['main_force_ofi'].sum()
                            OFI_Divergence = -mf_ofi_in_rally / rally_hf_df['ofi'].abs().sum() if rally_hf_df['ofi'].abs().sum() > 0 else 0
                            mf_t0_eff = daily_data.get('main_force_t0_efficiency', 0)
                            CostDecay_Component = np.tanh(mf_t0_eff) if pd.notna(mf_t0_eff) else 0
                            mf_net_sell_vol_in_rally = abs(intraday_data.loc[start_time:end_time, 'main_force_net_vol'].clip(upper=0).sum())
                            DistributionEfficiency = (mf_net_sell_vol_in_rally / daily_total_volume) / (price_change_in_rally / atr) if price_change_in_rally > 0 and pd.notna(atr) and atr > 0 else 0
                            deceptive_pressure_score = (0.5 * OFI_Divergence + 0.3 * CostDecay_Component + 0.2 * DistributionEfficiency)
                            total_deceptive_pressure += deceptive_pressure_score * price_change_in_rally
                if total_rally_price_change > 0:
                    results['rally_distribution_pressure'] = (total_deceptive_pressure / total_rally_price_change) * 100
                    if enable_probe and is_target_date:
                        print(f"  [探针] rally_distribution_pressure (诡道派发) 计算:")
                        print(f"    - Weighted Avg Deceptive Pressure: {total_deceptive_pressure / total_rally_price_change:.4f}")
                        print(f"    -> Final Score: {results['rally_distribution_pressure']:.2f}")
            else: 
                if pd.notna(daily_vwap):
                    absorption_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] < daily_vwap]
                    if not absorption_zone_df.empty and 'main_force_net_vol' in absorption_zone_df.columns and 'vol_shares' in absorption_zone_df.columns and 'minute_vwap' in absorption_zone_df.columns:
                        mf_absorption_df = absorption_zone_df[absorption_zone_df['main_force_net_vol'] > 0]
                        if not mf_absorption_df.empty:
                            absorption_vol = mf_absorption_df['main_force_net_vol'].sum()
                            if absorption_vol > 0 and pd.notna(atr) and atr > 0:
                                absorption_cost = (mf_absorption_df['minute_vwap'] * mf_absorption_df['main_force_net_vol']).sum() / absorption_vol
                                cost_deviation = (daily_vwap - absorption_cost) / atr
                                if daily_total_volume > 0:
                                    results['dip_absorption_power'] = (absorption_vol / daily_total_volume) * cost_deviation * 100
                rally_dist_vol, total_rally_vol = 0, 0
                for i in range(len(turning_points) - 1):
                    start_idx, end_idx = turning_points[i], turning_points[i+1]
                    window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                    if window_df.empty or len(window_df) < 2 or 'minute_vwap' not in window_df.columns or 'main_force_net_vol' not in window_df.columns or 'vol_shares' not in window_df.columns:
                        continue
                    if window_df['minute_vwap'].iloc[-1] > window_df['minute_vwap'].iloc[0]:
                        total_rally_vol += window_df['vol_shares'].sum()
                        mf_net_vol = window_df['main_force_net_vol'].sum()
                        if mf_net_vol < 0:
                            rally_dist_vol += abs(mf_net_vol)
                if total_rally_vol > 0:
                    results['rally_distribution_pressure'] = (rally_dist_vol / total_rally_vol) * 100
        if not hf_analysis_df.empty and 'main_force_ofi' in hf_analysis_df.columns and 'mid_price_change' in hf_analysis_df.columns:
            mf_ofi_series = hf_analysis_df['main_force_ofi']
            price_change_series = hf_analysis_df['mid_price_change']
            if mf_ofi_series.var() > 0 and price_change_series.var() > 0:
                results['ofi_price_impact_factor'] = mf_ofi_series.corr(price_change_series)
        if len(intraday_data) >= 10 and pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low) and pd.notna(atr) and atr > 0 and daily_total_volume > 0 and 'main_force_buy_vol' in intraday_data.columns and 'main_force_sell_vol' in intraday_data.columns:
            mf_activity_ratio = (intraday_data['main_force_buy_vol'].sum() + intraday_data['main_force_sell_vol'].sum()) / daily_total_volume
            if mf_activity_ratio > 0:
                price_outcome = (day_close - day_open) / atr
                results['trend_conviction_ratio'] = price_outcome / mf_activity_ratio
            day_range = day_high - day_low
            if day_range > 0:
                is_v_shape = (day_close - day_open) > 0
                turn_point_idx = np.argmin(intraday_data['low'].values) if is_v_shape else np.argmax(intraday_data['high'].values)
                if 0 < turn_point_idx < len(intraday_data) - 1:
                    initial_phase = intraday_data.iloc[:turn_point_idx]
                    reversal_phase = intraday_data.iloc[turn_point_idx:]
                    vol_initial, vol_reversal = initial_phase['vol_shares'].sum(), reversal_phase['vol_shares'].sum()
                    if vol_initial > 0 and vol_reversal > 0:
                        turn_point_vwap = intraday_data['minute_vwap'].iloc[turn_point_idx]
                        price_recovery = abs(day_close - turn_point_vwap) / day_range
                        vol_shift = np.log1p(vol_reversal / vol_initial)
                        reversal_mf_net_vol = reversal_phase['main_force_net_vol'].sum()
                        reversal_conviction = reversal_mf_net_vol / vol_reversal if vol_reversal > 0 else 0
                        power_score = price_recovery * vol_shift * reversal_conviction
                        results['reversal_power_index'] = power_score if is_v_shape else -power_score
        if 'minute_vwap' in intraday_data.columns and 'high' in intraday_data.columns and 'low' in intraday_data.columns and 'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            df_cmf = intraday_data.copy()
            cols_to_process = ['high', 'low', 'minute_vwap']
            for col in cols_to_process:
                df_cmf[col] = pd.to_numeric(df_cmf[col], errors='coerce')
            df_cmf.dropna(subset=cols_to_process, inplace=True)
            if not df_cmf.empty:
                price_range_cmf = df_cmf['high'] - df_cmf['low']
                mfm = ((df_cmf['minute_vwap'] - df_cmf['low']) - (df_cmf['high'] - df_cmf['minute_vwap'])) / price_range_cmf.replace(0, np.nan)
                mfm.fillna(0, inplace=True)
                if df_cmf['vol_shares'].sum() > 0:
                    results['holistic_cmf'] = (mfm * df_cmf['vol_shares']).sum() / df_cmf['vol_shares'].sum()
                if df_cmf['main_force_net_vol'].abs().sum() > 0:
                    results['main_force_cmf'] = (mfm * df_cmf['main_force_net_vol']).sum() / df_cmf['main_force_net_vol'].abs().sum()
                results['cmf_divergence_score'] = results.get('main_force_cmf', np.nan) - results.get('holistic_cmf', np.nan)
        if 'main_force_net_vol' in intraday_data.columns and pd.notna(day_close) and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
            vp_global = intraday_data.groupby(pd.cut(intraday_data['minute_vwap'], bins=30, duplicates='drop'))['vol_shares'].sum()
            if not vp_global.empty:
                vpoc_interval = vp_global.idxmax()
                global_vpoc_price = vpoc_interval.mid
                peak_zone_df = intraday_data[(intraday_data['minute_vwap'] >= vpoc_interval.left) & (intraday_data['minute_vwap'] < vpoc_interval.right)]
                if not peak_zone_df.empty:
                    mf_net_vol_on_peak = peak_zone_df['main_force_net_vol'].sum()
                    if daily_total_amount > 0:
                        normalized_mf_on_peak_flow = np.tanh((mf_net_vol_on_peak * global_vpoc_price) / daily_total_amount)
                        results['main_force_on_peak_flow'] = normalized_mf_on_peak_flow
            mf_net_buy_df = intraday_data[intraday_data['main_force_net_vol'] > 0]
            if not mf_net_buy_df.empty:
                vp_mf = mf_net_buy_df.groupby(pd.cut(mf_net_buy_df['minute_vwap'], bins=30, duplicates='drop'))['main_force_net_vol'].sum()
                if not vp_mf.empty:
                    mf_vpoc = vp_mf.idxmax().mid
                    results['main_force_vpoc'] = mf_vpoc
                    if pd.notna(global_vpoc_price) and global_vpoc_price > 0 and pd.notna(mf_vpoc):
                        results['mf_vpoc_premium'] = (mf_vpoc / global_vpoc_price - 1) * 100
        if 'main_force_net_vol' in intraday_data.columns and 'retail_net_vol' in intraday_data.columns:
            mf_net_series = intraday_data['main_force_net_vol']
            retail_net_series = intraday_data['retail_net_vol']
            if mf_net_series.var() != 0 and retail_net_series.var() != 0 and len(mf_net_series) > 1:
                rolling_corr = mf_net_series.rolling(window=30).corr(retail_net_series)
                results['mf_retail_liquidity_swap_corr'] = rolling_corr.mean()
        if pd.notna(daily_vwap) and daily_total_volume > 0 and pd.notna(atr) and atr > 0 and not continuous_trading_df.empty and 'close' in continuous_trading_df.columns and 'open' in continuous_trading_df.columns and 'vol_shares' in continuous_trading_df.columns and 'minute_vwap' in continuous_trading_df.columns and 'main_force_net_vol' in continuous_trading_df.columns:
            up_minutes = continuous_trading_df[continuous_trading_df['close'] > continuous_trading_df['open']]
            down_minutes = continuous_trading_df[continuous_trading_df['close'] < continuous_trading_df['open']]
            if not up_minutes.empty and not down_minutes.empty:
                up_price_change = (up_minutes['close'] - up_minutes['open']).sum()
                down_price_change = (down_minutes['open'] - down_minutes['close']).sum()
                avg_up_speed = up_price_change / len(up_minutes) if len(up_minutes) > 0 else 0
                avg_down_speed = down_price_change / len(down_minutes) if len(down_minutes) > 0 else 0
                if avg_up_speed > 0 and avg_down_speed > 0:
                    results['volatility_asymmetry_index'] = np.log(avg_up_speed / avg_down_speed)
            day_range = day_high - day_low
            if day_range > 0:
                range_pos_factor = ((day_close - day_low) / day_range) * 2 - 1
                value_dev_factor = np.tanh((day_close - daily_vwap) / atr)
                force_balance_factor = intraday_data['main_force_net_vol'].sum() / daily_total_volume if daily_total_volume > 0 else 0
                results['closing_price_deviation_score'] = (0.5 * range_pos_factor + 0.3 * value_dev_factor + 0.2 * force_balance_factor) * 100
            auction_df = intraday_data[intraday_data.index.time >= time(14, 57)]
            if not auction_df.empty and not continuous_trading_df.empty and pd.notna(atr) and atr > 0:
                pre_auction_close = continuous_trading_df['close'].iloc[-1]
                auction_vol = auction_df['vol_shares'].sum()
                avg_minute_vol = continuous_trading_df['vol_shares'].mean()
                PriceImpact = (day_close - pre_auction_close) / atr if pd.notna(pre_auction_close) else 0.0
                VolumeAnomaly = np.log1p((auction_vol / 3) / avg_minute_vol) if avg_minute_vol > 0 else 0.0
                ForceIntent = 0.0
                mf_ofi_auction = 0.0
                total_abs_ofi_day = 0.0
                if not hf_analysis_df.empty:
                    hf_auction_zone = hf_analysis_df[hf_analysis_df.index.time >= time(14, 57)]
                    if not hf_auction_zone.empty:
                        mf_ofi_auction = hf_auction_zone['main_force_ofi'].sum()
                        total_abs_ofi_day = hf_analysis_df['ofi'].abs().sum()
                        if total_abs_ofi_day > 0:
                            ForceIntent = mf_ofi_auction / total_abs_ofi_day
                results['closing_auction_ambush'] = PriceImpact * VolumeAnomaly * (1 + ForceIntent) * 100
                if enable_probe and is_target_date:
                    print(f"  [探针] closing_auction_ambush 计算:")
                    print(f"    - pre_auction_close: {pre_auction_close:.2f}, day_close: {day_close:.2f}, atr: {atr:.2f}")
                    print(f"    - auction_vol: {auction_vol:,.0f}, avg_minute_vol: {avg_minute_vol:,.0f}")
                    print(f"    - mf_ofi_auction: {mf_ofi_auction:,.0f}, total_abs_ofi_day: {total_abs_ofi_day:,.0f}")
                    print(f"    - PriceImpact: {PriceImpact:.4f}, VolumeAnomaly: {VolumeAnomaly:.4f}, ForceIntent: {ForceIntent:.4f}")
                    print(f"    -> Final Score: {results['closing_auction_ambush']:.2f}")
            else:
                results['closing_auction_ambush'] = np.nan
            panic_vol, total_panic_vol = 0, 0
            for i in range(len(turning_points) - 1):
                start_idx, end_idx = turning_points[i], turning_points[i+1]
                window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                if window_df.empty or len(window_df) < 2 or 'minute_vwap' not in window_df.columns or 'main_force_net_vol' not in window_df.columns or 'vol_shares' not in window_df.columns:
                    continue
                mf_net_vol = window_df['main_force_net_vol'].sum()
                window_vol = window_df['vol_shares'].sum()
                if window_df['minute_vwap'].iloc[-1] <= window_df['minute_vwap'].iloc[0]:
                    total_panic_vol += window_vol
                    if mf_net_vol < 0:
                        panic_vol += abs(mf_net_vol)
            if total_panic_vol > 0:
                results['panic_selling_cascade'] = (panic_vol / total_panic_vol) * 100
            posturing_df = continuous_trading_df[continuous_trading_df.index.time >= time(14, 30)]
            if pd.notna(daily_vwap) and atr > 0 and not posturing_df.empty and 'vol_shares' in posturing_df.columns and 'minute_vwap' in posturing_df.columns and 'main_force_net_vol' in posturing_df.columns:
                posturing_vwap = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum() / posturing_df['vol_shares'].sum()
                price_posture = (posturing_vwap - daily_vwap) / atr
                posturing_amount = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum()
                if posturing_amount > 0:
                    force_posture = (posturing_df['main_force_net_vol'].sum() * posturing_vwap) / posturing_amount
                    results['pre_closing_posturing'] = (0.6 * price_posture + 0.4 * force_posture) * 100
            day_range = day_high - day_low
            if day_range > 0:
                fomo_zone_threshold = day_low + 0.75 * day_range
                fomo_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] > fomo_zone_threshold]
                if not fomo_zone_df.empty and 'retail_net_vol' in fomo_zone_df.columns and 'retail_buy_vol' in continuous_trading_df.columns and 'minute_vwap' in fomo_zone_df.columns:
                    fomo_retail_df = fomo_zone_df[fomo_zone_df['retail_net_vol'] > 0]
                    if not fomo_retail_df.empty:
                        fomo_vol = fomo_retail_df['retail_net_vol'].sum()
                        total_retail_buy_vol = continuous_trading_df[continuous_trading_df['retail_buy_vol'] > 0]['retail_buy_vol'].sum()
                        if fomo_vol > 0 and total_retail_buy_vol > 0:
                            cost_fomo = (fomo_retail_df['minute_vwap'] * fomo_retail_df['retail_net_vol']).sum() / fomo_vol
                            cost_mf_sell = daily_data.get('avg_cost_main_sell')
                            if pd.notna(cost_mf_sell) and cost_mf_sell > 0:
                                premium = (cost_fomo / cost_mf_sell - 1)
                                results['retail_fomo_premium_index'] = premium * (fomo_vol / total_retail_buy_vol) * 100
                panic_zone_threshold = day_low + 0.25 * day_range
                panic_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] < panic_zone_threshold]
                if not panic_zone_df.empty and 'retail_net_vol' in panic_zone_df.columns and 'retail_sell_vol' in continuous_trading_df.columns and 'minute_vwap' in panic_zone_df.columns:
                    panic_retail_df = panic_zone_df[panic_zone_df['retail_net_vol'] < 0]
                    if not panic_retail_df.empty:
                        panic_vol = abs(panic_retail_df['retail_net_vol'].sum())
                        total_retail_sell_vol = continuous_trading_df[continuous_trading_df['retail_sell_vol'] > 0]['retail_sell_vol'].sum()
                        if panic_vol > 0 and total_retail_sell_vol > 0:
                            cost_panic = (panic_retail_df['minute_vwap'] * abs(panic_retail_df['retail_net_vol'])).sum() / panic_vol
                            cost_mf_buy = daily_data.get('avg_cost_main_buy')
                            if pd.notna(cost_mf_buy) and cost_mf_buy > 0:
                                discount = (cost_mf_buy - cost_panic) / cost_mf_buy
                                results['retail_panic_surrender_index'] = discount * (panic_vol / total_retail_sell_vol) * 100
        if not intraday_data.empty and 'main_force_net_vol' in intraday_data.columns and 'open' in intraday_data.columns and 'close' in intraday_data.columns:
            dip_or_flat_df = intraday_data[intraday_data['close'] <= intraday_data['open']]
            if not dip_or_flat_df.empty:
                total_vol_dip = dip_or_flat_df['vol_shares'].sum()
                if total_vol_dip > 0:
                    mf_net_buy_on_dip = dip_or_flat_df['main_force_net_vol'].clip(lower=0).sum()
                    results['hidden_accumulation_intensity'] = (mf_net_buy_on_dip / total_vol_dip) * 100
        if not intraday_data.empty and 'main_force_net_vol' in intraday_data.columns and 'close' in intraday_data.columns:
            intraday_data['price_change'] = intraday_data['close'].diff()
            if pd.notna(atr) and atr > 0:
                intraday_data['price_change_norm'] = intraday_data['price_change'] / atr
                mf_net_vol_series = intraday_data['main_force_net_vol']
                price_change_norm_series = intraday_data['price_change_norm']
                if mf_net_vol_series.var() > 0 and price_change_norm_series.var() > 0:
                    results['microstructure_efficiency_index'] = mf_net_vol_series.corr(price_change_norm_series)
        WAN = 10000.0
        try:
            pct_change = pd.to_numeric(daily_data.get('pct_change'), errors='coerce')
            circ_mv_yuan = pd.to_numeric(daily_data.get('circ_mv'), errors='coerce') * WAN
            if pd.notna(circ_mv_yuan) and circ_mv_yuan > 0 and pd.notna(main_force_net_flow_calibrated) and pd.notna(pct_change):
                mf_flow_yuan = main_force_net_flow_calibrated * WAN
                flow_input = mf_flow_yuan / circ_mv_yuan
                if abs(flow_input) > 1e-9:
                    efficiency = (pct_change / 100) / flow_input
                    results['flow_efficiency_index'] = np.sign(efficiency) * np.log1p(abs(efficiency))
        except Exception:
            results['flow_efficiency_index'] = np.nan
        try:
            trade_count = pd.to_numeric(daily_data.get('trade_count'), errors='coerce')
            turnover_amount_yuan = pd.to_numeric(daily_data.get('amount'), errors='coerce') * 1000
            if pd.notna(trade_count) and trade_count > 0 and pd.notna(turnover_amount_yuan) and turnover_amount_yuan > 0:
                results['inferred_active_order_size'] = turnover_amount_yuan / trade_count
        except Exception:
            results['inferred_active_order_size'] = np.nan
        return results

    def _calculate_intraday_attribution_weights(self, intraday_data_for_day: pd.DataFrame, daily_data: pd.Series) -> pd.DataFrame:
        """
        【V9.5 · 逐笔数据兼容版 - 价格范围零值修复】
        - 核心革命: 废弃“一体适用”的权重模型，为超大单、大单、中单、小单引入各自独特的、基于行为特征的权重分配逻辑。
        - 核心思想:
          - 超大单(ELG) -> 脉冲修正: 权重集中在成交量和振幅剧增的“暴力分钟”。
          - 大单(LG) -> VWAP修正: 权重与价格偏离VWAP的程度相关，体现战术意图。
          - 中单(MD) -> 动量修正: 权重与短期价格动量相关，体现追涨杀跌特性。
          - 小单(SM) -> 基准压力: 沿用原有的K线形态压力模型作为基准。
        - 核心修复: 修复了 `price_range` 为零时导致的 `decimal.InvalidOperation` 错误。
        - 【修正】修复 `impulse_modifier` 计算中 `price_range` 的错误使用。
        """
        df = intraday_data_for_day.copy()
        # 修改行：移除了所有与debug_params和probe_dates相关的探针初始化代码
        if 'vol_shares' not in df.columns or df['vol_shares'].sum() < 1e-6 or len(df) < 5:
            for size in ['sm', 'md', 'lg', 'elg']:
                df[f'{size}_buy_weight'] = 0; df[f'{size}_sell_weight'] = 0
            return df
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        price_range = df['high'] - df['low']
        buy_pressure_proxy_ratio = np.where(
            price_range != 0,
            (df['close'] - df['low']) / price_range,
            0.5
        )
        conditions = [
            price_range > 0,
            (price_range == 0) & (df['close'] > df['open']),
            (price_range == 0) & (df['close'] < df['open'])
        ]
        choices = [
            buy_pressure_proxy_ratio,
            1.0,
            0.0
        ]
        buy_pressure_proxy = np.select(conditions, choices, default=0.5)
        # 修改行：移除了检查中间计算结果的探针print语句
        vol_ma = df['vol_shares'].rolling(window=20, min_periods=1).mean()
        range_ma = price_range.rolling(window=20, min_periods=1).mean()
        impulse_modifier = (df['vol_shares'] / vol_ma) * (price_range / range_ma.replace(0, 1e-9))
        impulse_modifier = impulse_modifier.fillna(1).clip(0, 10)
        daily_vwap = daily_data.get('daily_vwap')
        if pd.notna(daily_vwap):
            vwap_deviation = (df['minute_vwap'] - daily_vwap) / daily_vwap
            lg_buy_modifier = np.exp(-np.maximum(0, vwap_deviation) * 5)
            lg_sell_modifier = np.exp(np.minimum(0, vwap_deviation) * 5)
        else:
            lg_buy_modifier = pd.Series(1.0, index=df.index); lg_sell_modifier = pd.Series(1.0, index=df.index)
        momentum_modifier = df['minute_vwap'].pct_change().rolling(window=5).mean().fillna(0)
        md_buy_modifier = np.exp(momentum_modifier * 50)
        md_sell_modifier = np.exp(-momentum_modifier * 50)
        sm_buy_score = df['vol_shares'] * buy_pressure_proxy
        sm_sell_score = df['vol_shares'] * (1 - buy_pressure_proxy)
        md_buy_score = sm_buy_score * md_buy_modifier
        md_sell_score = sm_sell_score * md_sell_modifier
        lg_buy_score = sm_buy_score * lg_buy_modifier
        lg_sell_score = sm_sell_score * lg_sell_modifier
        elg_buy_score = sm_buy_score * impulse_modifier
        elg_sell_score = sm_sell_score * impulse_modifier
        scores = {
            'sm': (sm_buy_score, sm_sell_score), 'md': (md_buy_score, md_sell_score),
            'lg': (lg_buy_score, lg_sell_score), 'elg': (elg_buy_score, elg_sell_score)
        }
        for size, (buy_score, sell_score) in scores.items():
            total_buy_score = buy_score.sum()
            df[f'{size}_buy_weight'] = buy_score / total_buy_score if total_buy_score > 1e-9 else 0
            total_sell_score = sell_score.sum()
            df[f'{size}_sell_weight'] = sell_score / total_sell_score if total_sell_score > 1e-9 else 0
            # 修改行：移除了检查score总和的探针print语句
        return df

    async def _load_historical_metrics(self, model, stock_info, end_date):
        """
        【V2.2 · 索引修复版】从数据库加载并净化历史高级资金流指标。
        - 核心修复: 修正 set_index 的用法，确保 trade_time 列在成为索引后被正确移除。
        """
        @sync_to_async
        def get_data():
            core_metric_cols = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
            required_cols = ['trade_time'] + [col for col in core_metric_cols if hasattr(model, col)]
            qs = model.objects.filter(
                stock=stock_info, 
                trade_time__lt=end_date
            ).order_by('trade_time')
            return pd.DataFrame.from_records(qs.values(*required_cols))
        df = await get_data()
        if not df.empty:
            # 修复：分两步操作，先转换类型，再用列名设置索引，确保 'trade_time' 列被正确移除
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.set_index('trade_time')
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _calculate_microstructure_signals(self, stock_code: str, daily_intraday_df: pd.DataFrame, daily_level5_df: pd.DataFrame, daily_total_volume: float) -> dict:
        """
        【V2.0 · 微观结构情报引擎版】
        - 核心重构: 全面升级为基于Tick和Level5数据的高级博弈指标计算中心。
        - 核心引入: 实现了更精确的订单流失衡(OFI)计算，并区分主力与散户OFI。
        - 核心增强: 优化了对倒强度、盘口失衡的计算逻辑，引入时间加权和价格惩罚项。
        - 核心输出: 锻造出'main_force_ofi', 'retail_ofi', 'wash_trade_intensity'等高保真微观博弈指标。
        """
        results = {
            'wash_trade_intensity': np.nan,
            'order_book_imbalance': np.nan,
            'large_order_pressure': np.nan,
            'large_order_support': np.nan,
            'main_force_ofi': np.nan,
            'retail_ofi': np.nan,
        }
        if daily_intraday_df is None or daily_intraday_df.empty:
            daily_intraday_df = pd.DataFrame()
        if daily_level5_df is None or daily_level5_df.empty:
            daily_level5_df = pd.DataFrame()
        if daily_intraday_df.empty or daily_level5_df.empty or daily_total_volume <= 0:
            return results
        # --- 数据预处理 ---
        tick_df = daily_intraday_df.copy()
        level5_df = daily_level5_df.copy()
        for col in ['price', 'volume', 'amount']:
            if col in tick_df.columns:
                tick_df[col] = pd.to_numeric(tick_df[col], errors='coerce')
        for i in range(1, 6):
            for prefix in ['buy_price', 'buy_volume', 'sell_price', 'sell_volume']:
                col_name = f'{prefix}{i}'
                if col_name in level5_df.columns:
                    level5_df[col_name] = pd.to_numeric(level5_df[col_name], errors='coerce')
        # --- 1. 订单流失衡 (OFI) ---
        # 引入更精确的OFI计算逻辑
        merged_df = pd.merge_asof(tick_df.sort_index(), level5_df.sort_index(), left_index=True, right_index=True, direction='backward').dropna()
        if not merged_df.empty and 'buy_price1' in merged_df.columns and 'sell_price1' in merged_df.columns:
            merged_df['mid_price'] = (merged_df['buy_price1'] + merged_df['sell_price1']) / 2
            merged_df['prev_mid_price'] = merged_df['mid_price'].shift(1)
            # 计算OFI
            buy_pressure = np.where(merged_df['mid_price'] >= merged_df['prev_mid_price'], merged_df['buy_volume1'].shift(1), 0)
            sell_pressure = np.where(merged_df['mid_price'] <= merged_df['prev_mid_price'], merged_df['sell_volume1'].shift(1), 0)
            merged_df['ofi'] = buy_pressure - sell_pressure
            # 按交易金额区分主力与散户
            is_main_force_trade = merged_df['amount'] > 200000  # 阈值：20万元
            main_force_ofi_series = merged_df.loc[is_main_force_trade, 'ofi']
            retail_ofi_series = merged_df.loc[~is_main_force_trade, 'ofi']
            # 对OFI进行归一化处理，使其具有可比性
            total_ofi_range = merged_df['ofi'].abs().sum()
            if total_ofi_range > 0:
                results['main_force_ofi'] = main_force_ofi_series.sum() / total_ofi_range
                results['retail_ofi'] = retail_ofi_series.sum() / total_ofi_range
        # --- 2. 对倒强度 (Wash Trade Intensity) ---
        # 引入价格变动作为惩罚项
        if 'type' in tick_df.columns:
            tick_df['direction'] = tick_df['type'].map({'B': 1, 'S': -1, 'M': 0})
            tick_df['reversal'] = (tick_df['direction'] * tick_df['direction'].shift(1)) < 0
            tick_df['price_change_abs'] = tick_df['price'].diff().abs().fillna(0)
            # 价格变化越小，对倒嫌疑越大
            price_penalty = np.exp(-tick_df['price_change_abs'] * 100)
            wash_trade_vol = tick_df[tick_df['reversal']]['volume'] * price_penalty[tick_df['reversal']]
            results['wash_trade_intensity'] = wash_trade_vol.sum() / (daily_total_volume / 100) if daily_total_volume > 0 else np.nan
        # --- 3. 盘口失衡度 (Order Book Imbalance) ---
        # 引入时间加权
        if not level5_df.empty and 'buy_volume1' in level5_df.columns:
            level5_df['buy_vol_total'] = level5_df[[f'buy_volume{i}' for i in range(1, 6)]].sum(axis=1)
            level5_df['sell_vol_total'] = level5_df[[f'sell_volume{i}' for i in range(1, 6)]].sum(axis=1)
            total_book_vol = level5_df['buy_vol_total'] + level5_df['sell_vol_total']
            level5_df['imbalance'] = (level5_df['buy_vol_total'] - level5_df['sell_vol_total']) / total_book_vol.replace(0, np.nan)
            level5_df.dropna(subset=['imbalance'], inplace=True)
            if not level5_df.empty:
                time_diffs = level5_df.index.to_series().diff().dt.total_seconds().fillna(0)
                if time_diffs.sum() > 0:
                    results['order_book_imbalance'] = np.average(level5_df['imbalance'], weights=time_diffs) * 100
        # --- 4. 大单压制与支撑强度 (保留原逻辑) ---
        large_order_threshold_value = 500000
        pressure_strength = 0
        support_strength = 0
        if not level5_df.empty and 'sell_price1' in level5_df.columns:
            time_diffs_seconds = level5_df.index.to_series().diff().dt.total_seconds().values
            for i in range(1, len(level5_df)):
                row = level5_df.iloc[i]
                duration = time_diffs_seconds[i]
                for p, v in [('sell_price1', 'sell_volume1'), ('sell_price2', 'sell_volume2')]:
                    if pd.notna(row[p]) and pd.notna(row[v]) and row[p] * row[v] * 100 > large_order_threshold_value:
                        pressure_strength += row[v] * 100 * duration
                for p, v in [('buy_price1', 'buy_volume1'), ('buy_price2', 'buy_volume2')]:
                    if pd.notna(row[p]) and pd.notna(row[v]) and row[p] * row[v] * 100 > large_order_threshold_value:
                        support_strength += row[v] * 100 * duration
            if daily_total_volume > 0:
                total_seconds = 4 * 60 * 60 # 交易总秒数
                results['large_order_pressure'] = pressure_strength / (daily_total_volume * total_seconds) * 100
                results['large_order_support'] = support_strength / (daily_total_volume * total_seconds) * 100
        # 移除所有调试性质的print探针
        return results

    def _group_minute_data_from_df(self, minute_df: pd.DataFrame):
        """【V1.15 · 数据完整性修复版 - 辅助列添加 - 智能列名识别】从预加载的DataFrame构建按日分组的数据。
        - 核心职责: 确保传入的DataFrame保持 `trade_time` 作为 `DatetimeIndex`，并正确处理时区，添加 `amount_yuan`, `vol_shares`, `minute_vwap`, `vol_weight` 等辅助列。
        - 核心修复: 不再修改DataFrame的索引，仅添加辅助列。
        - 【修正】智能识别成交量列名（'volume' 或 'vol'），并统一为 'vol_shares'。
        - 【修正】根据最新澄清，统一处理时区，确保最终输出为北京时间。
        """
        from django.utils import timezone
        if minute_df is None or minute_df.empty:
            return pd.DataFrame()
        df = minute_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'trade_time' in df.columns:
                df['trade_time'] = pd.to_datetime(df['trade_time'])
                df = df.set_index('trade_time')
            else:
                logger.warning("DataFrame passed to _group_minute_data_from_df has no 'trade_time' column and no DatetimeIndex.")
                return pd.DataFrame()
        # 统一处理时区，确保最终输出为北京时间
        if df.index.tz is None:
            # 如果意外是 naive，假定它是 UTC（因为DAO层应该输出UTC aware，但可能在某些操作后丢失时区信息）
            df.index = df.index.tz_localize('UTC', ambiguous='infer').tz_convert(timezone.get_current_timezone())
        else:
            # 如果已经是 aware，直接转换为目标时区
            df.index = df.index.tz_convert(timezone.get_current_timezone())
        volume_col_name = None
        if 'volume' in df.columns:
            volume_col_name = 'volume'
        elif 'vol' in df.columns:
            volume_col_name = 'vol'
        else:
            logger.error(f"DataFrame缺少成交量列 ('volume' 或 'vol')，无法处理。列名: {df.columns.tolist()}")
            return pd.DataFrame()
        df['amount_yuan'] = pd.to_numeric(df['amount'], errors='coerce')
        df['vol_shares'] = pd.to_numeric(df[volume_col_name], errors='coerce')
        df['minute_vwap'] = df['amount_yuan'] / df['vol_shares'].replace(0, np.nan)
        current_day_total_vol = df['vol_shares'].sum()
        df['vol_weight'] = df['vol_shares'] / current_day_total_vol if current_day_total_vol > 0 else 0
        return df

    def _calculate_realtime_orderbook_signals(self, level5_df: pd.DataFrame, daily_total_volume: float) -> dict:
        """
        【V1.1 · 命名统一修复版】计算基于五档盘口快照的资金意图指标。
        - 核心修复: 将输入参数从 `realtime_df` 修正为 `level5_df`，以反映正确的数据源。
        - 核心修复: 将内部所有列名引用（如 'b1_p', 'a1_v'）更新为与 Django 模型
                     `BaseStockLevel5Data` 一致的名称（如 'buy_price1', 'sell_volume1'），
                     彻底解决 `KeyError`。
        """
        results = {
            'order_book_liquidity_supply': np.nan,
            'buy_quote_exhaustion_rate': np.nan,
            'sell_quote_exhaustion_rate': np.nan,
        }
        if level5_df is None or level5_df.empty or daily_total_volume <= 0:
            return results
        # --- 1. 盘口流动性供给 ---
        bid_supply = pd.Series(0, index=level5_df.index, dtype=float)
        ask_supply = pd.Series(0, index=level5_df.index, dtype=float)
        for i in range(1, 6):
            # 更新列名为Django模型定义的名称
            bid_price_col = f'buy_price{i}'
            bid_vol_col = f'buy_volume{i}'
            ask_price_col = f'sell_price{i}'
            ask_vol_col = f'sell_volume{i}'
            if all(c in level5_df.columns for c in [bid_price_col, bid_vol_col, ask_price_col, ask_vol_col]):
                bid_supply += level5_df[bid_price_col] * level5_df[bid_vol_col] * 100
                ask_supply += level5_df[ask_price_col] * level5_df[ask_vol_col] * 100
        time_diffs = level5_df.index.to_series().diff().dt.total_seconds().fillna(0)
        if time_diffs.sum() > 0:
            avg_bid_supply = np.average(bid_supply, weights=time_diffs)
            avg_ask_supply = np.average(ask_supply, weights=time_diffs)
            if avg_ask_supply > 0:
                results['order_book_liquidity_supply'] = avg_bid_supply / avg_ask_supply
        # --- 2. 报价消耗率 ---
        df = level5_df.copy()
        # 更新列名为Django模型定义的名称
        df['prev_a1_p'] = df['sell_price1'].shift(1)
        df['prev_b1_p'] = df['buy_price1'].shift(1)
        df['prev_a1_v'] = df['sell_volume1'].shift(1)
        df['prev_b1_v'] = df['buy_volume1'].shift(1)
        # 买方消耗：当前卖一价 > 上一刻卖一价，意味着上一刻的卖一盘被击穿
        buy_exhaustion_mask = df['sell_price1'] > df['prev_a1_p']
        buy_exhausted_vol = (df.loc[buy_exhaustion_mask, 'prev_a1_v'] * 100).sum()
        # 卖方消耗：当前买一价 < 上一刻买一价，意味着上一刻的买一盘被击穿
        sell_exhaustion_mask = df['buy_price1'] < df['prev_b1_p']
        sell_exhausted_vol = (df.loc[sell_exhaustion_mask, 'prev_b1_v'] * 100).sum()
        results['buy_quote_exhaustion_rate'] = (buy_exhausted_vol / daily_total_volume) * 100
        results['sell_quote_exhaustion_rate'] = (sell_exhausted_vol / daily_total_volume) * 100
        return results







