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
        【V2.3 · 资金流统一校准增强版】
        - 核心重构: 剥离日线和基础面数据的合并逻辑，改为接收由上游任务统一准备好的 `base_daily_df`。
        - 核心修复: 在合并前，主动检查并移除与 `base_daily_df` 的重叠列，根除 'columns overlap' 错误。
        - 【增强】统一 `main_force_net_flow` 和 `retail_net_flow` 的计算逻辑，使其在所有数据源中都尽可能地代表
                  **大单+特大单的净流入** 和 **中单+小单的净流入**，并修正 DC 的总净流入计算。
        - 【修正】确保所有数据源都包含 `buy_elg_amount` 和 `sell_elg_amount` 列，即使值为0。
        """
        def standardize_and_prepare(df: pd.DataFrame, source: str) -> pd.DataFrame:
            if df.empty: return df
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            cols_to_numeric = [col for col in df.columns if col != 'trade_time' and 'code' not in col and 'name' not in col]
            df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
            # 确保所有必要的买卖额列都存在，并填充0
            required_amount_cols = [
                'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
                'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount',
            ]
            # 确保所有净流入列也存在，并填充0
            required_net_amount_cols = [
                'net_mf_amount', 'net_amount', 'net_amount_main', 'net_amount_xl',
                'net_amount_lg', 'net_amount_md', 'net_amount_sm', 'trade_count'
            ]
            for col in required_amount_cols + required_net_amount_cols:
                if col not in df.columns:
                    df[col] = 0.0 # 填充默认值0.0
            # 结束
            if source == 'tushare':
                df['net_flow_tushare'] = df['net_mf_amount']
                # Tushare 的主力净流入 = 大单净流入 + 特大单净流入
                df['main_force_net_flow_tushare'] = (df['buy_lg_amount'] - df['sell_lg_amount']) + (df['buy_elg_amount'] - df['sell_elg_amount'])
                df['retail_net_flow_tushare'] = (df['buy_sm_amount'] - df['sell_sm_amount']) + (df['buy_md_amount'] - df['sell_md_amount'])
                df['net_xl_amount_tushare'] = df['buy_elg_amount'] - df['sell_elg_amount']
                df['net_lg_amount_tushare'] = df['buy_lg_amount'] - df['sell_lg_amount']
                df['net_md_amount_tushare'] = df['buy_md_amount'] - df['sell_md_amount']
                df['net_sh_amount_tushare'] = df['buy_sm_amount'] - df['sell_sm_amount']
                return df
            elif source == 'ths':
                # THS 的 buy_lg_amount, buy_md_amount, buy_sm_amount 已经是净流入额
                df['net_flow_ths'] = df['net_amount'] # THS 的 net_amount 是总净流入
                df['main_force_net_flow_ths'] = df['buy_lg_amount'] # THS 没有特大单，只用大单作为主力
                df['retail_net_flow_ths'] = df['buy_md_amount'] + df['buy_sm_amount']
                # 明确添加 elg 相关的列，即使 THS 没有，也填充0
                df['buy_elg_amount'] = 0.0
                df['sell_elg_amount'] = 0.0
                # 确保 lg, md, sm 的买卖额也存在，即使 THS 只有净额
                # 假设净流入为正时是买入，为负时是卖出
                df['sell_lg_amount'] = df['buy_lg_amount'].clip(upper=0).abs()
                df['buy_lg_amount'] = df['buy_lg_amount'].clip(lower=0)
                df['sell_md_amount'] = df['buy_md_amount'].clip(upper=0).abs()
                df['buy_md_amount'] = df['buy_md_amount'].clip(lower=0)
                df['sell_sm_amount'] = df['buy_sm_amount'].clip(upper=0).abs()
                df['buy_sm_amount'] = df['buy_sm_amount'].clip(lower=0)
                # 结束
                df['net_lg_amount_ths'] = df['buy_lg_amount'] - df['sell_lg_amount'] # 重新计算净额以保持一致性
                df['net_md_amount_ths'] = df['buy_md_amount'] - df['sell_md_amount']
                df['net_sh_amount_ths'] = df['buy_sm_amount'] - df['sell_sm_amount']
                return df
            elif source == 'dc':
                # DC 的 net_amount 是主力净流入，buy_elg_amount, buy_lg_amount, buy_md_amount, buy_sm_amount 都是净流入额
                df['main_force_net_flow_dc'] = df['net_amount'] # DC 的 net_amount 就是主力净流入
                df['retail_net_flow_dc'] = df['net_amount_md'] + df['net_amount_sm'] # 散户净流入 = 中单 + 小单
                # 修正：DC 的总净流入应该由主力净流入和散户净流入组成
                df['net_flow_dc'] = df['main_force_net_flow_dc'] + df['retail_net_flow_dc']
                # 确保买卖额也存在，即使 DC 只有净额
                df['sell_elg_amount'] = df['net_amount_xl'].clip(upper=0).abs()
                df['buy_elg_amount'] = df['net_amount_xl'].clip(lower=0)
                df['sell_lg_amount'] = df['net_amount_lg'].clip(upper=0).abs()
                df['buy_lg_amount'] = df['net_amount_lg'].clip(lower=0)
                df['sell_md_amount'] = df['net_amount_md'].clip(upper=0).abs()
                df['buy_md_amount'] = df['net_amount_md'].clip(lower=0)
                df['sell_sm_amount'] = df['net_amount_sm'].clip(upper=0).abs()
                df['buy_sm_amount'] = df['net_amount_sm'].clip(lower=0)
                # 结束
                df['net_xl_amount_dc'] = df['net_amount_xl']
                df['net_lg_amount_dc'] = df['net_amount_lg']
                df['net_md_amount_dc'] = df['net_amount_md']
                df['net_sh_amount_dc'] = df['net_amount_sm']
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
                # 在合并前，主动检查并移除与 `merged_df` 的重叠列，避免 'columns overlap' 错误
                overlap_cols = merged_df.columns.intersection(right_df.columns).drop('trade_time', errors='ignore')
                right_df_cleaned = right_df.drop(columns=overlap_cols, errors='ignore')
                merged_df = pd.merge(merged_df, right_df_cleaned, on='trade_time', how='left')
        merged_df = merged_df.sort_values('trade_time').set_index('trade_time')
        if not base_daily_df.empty:
            overlap_cols = merged_df.columns.intersection(base_daily_df.columns).drop('trade_time', errors='ignore')
            merged_df = merged_df.join(base_daily_df.drop(columns=overlap_cols, errors='ignore'), how='left')
        return merged_df

    async def _get_daily_grouped_minute_data(self, stock_info: StockInfo, date_index: pd.DatetimeIndex, fetch_full_cols: bool = True, tick_data_map: dict = None, level5_data_map: dict = None, minute_data_map: dict = None):
        """
        【V1.14 · 日内数据回退增强版 - 探针增强】不再查询数据库，仅处理由上游任务传入的日内数据maps。
        - 核心重构: 移除所有数据库查询逻辑，职责单一化为数据处理与聚合。
        - 核心逻辑: 遍历所需日期，优先尝试逐笔数据，若处理失败则回退到分钟数据。
        - 核心修复: 修正逐笔数据与Level5数据合并后，价格、成交量、成交额列名未被 `suffixes` 参数重命名的问题。
                    这些列名应保持原始名称，避免 `KeyError`。
        - 核心增强: 引入逐笔数据处理失败回退机制，确保分钟数据在逐笔数据不可用时能被利用。
        - 【新增探针】增加探针，确认 `intraday_data_map` 是否成功填充了指定日期的数据，并检查其 `vol_shares` 总和及行数。
        """
        import pandas as pd
        from django.utils import timezone
        if date_index.empty:
            return {}
        intraday_data_map = {}
        # 新增行：获取调试参数
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == date_index.date[0]: # 假设date_index通常是单日
                is_probe_date = True
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
                print(f"调试信息: [{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 未找到任何预加载的日内数据。")
            # 新增探针：检查最终的日内数据
            if is_probe_date and date_obj == probe_date_naive:
                final_intraday_df = intraday_data_map.get(date_obj)
                if final_intraday_df is not None and not final_intraday_df.empty:
                    print(f"    -> [资金流服务探针] @ {date_obj}: 成功加载日内数据。行数: {len(final_intraday_df)}, vol_shares总和: {final_intraday_df['vol_shares'].sum():.2f}。")
                else:
                    print(f"    -> [资金流服务探针] @ {date_obj}: 未能加载日内数据或数据为空。")
        return intraday_data_map

    def _calculate_all_metrics_for_day(self, stock_code: str, daily_data_series: pd.Series, intraday_data: pd.DataFrame, attributed_minute_df: pd.DataFrame, probabilistic_costs_dict: dict, tick_data_for_day: pd.DataFrame, level5_data_for_day: pd.DataFrame) -> tuple[dict, None]:
        """
        【V1.2 · 聚合成本计算集成版】
        - 核心职责: 作为单日所有高级资金流指标计算的总调度中心。
        - 核心逻辑: 依次调用日线、行为、微观结构等专项计算模块，并将结果聚合。
        - 核心修正: 在调用 `_compute_all_behavioral_metrics` 之前，集成 `_calculate_aggregate_pvwap_costs` 的计算结果。
        """
        # 1. 初始化当日指标字典
        day_metrics = {}
        # 2. 计算基于日线数据的衍生指标
        daily_derived_metrics = self._calculate_daily_derived_metrics(daily_data_series)
        day_metrics.update(daily_derived_metrics)
        # 3. 合并预先计算好的概率成本
        day_metrics.update(probabilistic_costs_dict)
        # 新增 - 计算聚合的PVWAP成本并合并到 day_metrics
        # _calculate_aggregate_pvwap_costs 期望一个 DataFrame 作为 pvwap_df 参数
        # 这里我们将 probabilistic_costs_dict 转换为 Series，再转换为 DataFrame，以便传递
        prob_costs_series = pd.Series(probabilistic_costs_dict)
        # 确保索引是 DatetimeIndex，因为 _calculate_aggregate_pvwap_costs 期望索引
        prob_costs_df_for_agg = pd.DataFrame([prob_costs_series], index=[daily_data_series.name])
        # _calculate_aggregate_pvwap_costs 还需要 daily_df，这里传入 daily_data_series
        # 但 daily_data_series 是 Series，需要转换为 DataFrame
        daily_df_for_agg = pd.DataFrame([daily_data_series.to_dict()], index=[daily_data_series.name])
        aggregate_pvwap_costs_df = self._calculate_aggregate_pvwap_costs(prob_costs_df_for_agg, daily_df_for_agg)
        # 将计算出的聚合成本从 DataFrame 转换为字典并合并到 day_metrics
        if not aggregate_pvwap_costs_df.empty:
            day_metrics.update(aggregate_pvwap_costs_df.iloc[0].to_dict())
        # 4. 计算基于分钟数据的行为指标
        # 注意：_compute_all_behavioral_metrics 需要归因后的分钟数据
        # 此时 daily_data_series 已经包含了 day_metrics 中所有更新的指标
        # 所以这里需要将 day_metrics 重新合并到 daily_data_series，或者直接传递 day_metrics
        # 为了避免修改 daily_data_series 的原始结构，我们创建一个新的 Series 传递
        updated_daily_data_series = pd.Series({**daily_data_series.to_dict(), **day_metrics}, name=daily_data_series.name) # 修改行：在创建Series时，通过name参数保留日期索引
        behavioral_metrics = self._compute_all_behavioral_metrics(attributed_minute_df, updated_daily_data_series)
        day_metrics.update(behavioral_metrics)
        # 5. 计算基于高频数据的微观结构信号
        daily_total_volume = daily_data_series.get('vol', 0) * 100
        microstructure_signals = self._calculate_microstructure_signals(stock_code, tick_data_for_day, level5_data_for_day, daily_total_volume)
        day_metrics.update(microstructure_signals)
        # 6. 填充 trade_time 字段
        day_metrics['trade_time'] = daily_data_series.name
        # 7. 返回最终的指标字典
        return day_metrics, None

    def _synthesize_and_forge_metrics(self, stock_code: str, merged_df: pd.DataFrame, tick_data_map: dict = None, level5_data_map: dict = None, minute_data_map: dict = None) -> tuple[pd.DataFrame, dict, list]:
        """
        【V10.10 · 数据流修复版 - 探针增强 - 跨服务数据完整性修复】
        - 核心修正: 调整对 `_calculate_probabilistic_costs` 的调用，以接收其返回的成本字典和完整归因后的分钟DataFrame，确保下游计算能获得必要的 `main_force_net_vol` 等列。
        - 【新增探针】检查 `attributed_minute_df` 中的主力/散户买卖量。
        - 【关键修复】在将 `attributed_minute_df` 存储到 `attributed_minute_data_map` 之前，进行深拷贝，确保跨服务传递时数据完整性。
        """
        all_metrics_list = []
        attributed_minute_data_map = {}
        failures = []
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date_global = False
        probe_date_naive = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            is_probe_date_global = True
        if is_probe_date_global:
            print(f"    -> [资金流合成探针-初始化] debug_params: {debug_params}, probe_date_naive: {probe_date_naive}")
        for trade_date, daily_data_series in merged_df.iterrows():
            date_obj = trade_date.date()
            is_current_probe_date = is_probe_date_global and (probe_date_naive == date_obj)
            if is_current_probe_date:
                print(f"    -> [资金流合成探针] @ {date_obj}: is_current_probe_date is TRUE.")
            intraday_data = self._minute_df_daily_grouped.get(date_obj)
            if intraday_data is None or intraday_data.empty:
                print(f"日内数据特征准备跳过，原因：日内数据(intraday_data)为空")
                failures.append({'stock_code': stock_code, 'trade_date': str(date_obj), 'reason': '当日分钟线/逐笔聚合数据缺失'})
                continue
            attribution_weights_df = self._calculate_intraday_attribution_weights(intraday_data, daily_data_series)
            probabilistic_costs_dict, attributed_minute_df = self._calculate_probabilistic_costs(stock_code, attribution_weights_df, daily_data_series)
            if is_current_probe_date:
                print(f"    -> [资金流合成探针] @ {date_obj}: attributed_minute_df 检查。")
                print(f"       - attributed_minute_df['main_force_sell_vol'] sum: {attributed_minute_df['main_force_sell_vol'].sum():.2f}")
                print(f"       - attributed_minute_df['retail_sell_vol'] sum: {attributed_minute_df['retail_sell_vol'].sum():.2f}")
            day_metrics, _ = self._calculate_all_metrics_for_day(
                stock_code, daily_data_series, intraday_data, attributed_minute_df, probabilistic_costs_dict,
                tick_data_map.get(date_obj) if tick_data_map else None,
                level5_data_map.get(date_obj) if level5_data_map else None
            )
            all_metrics_list.append(day_metrics)
            # 在存储到 map 之前进行深拷贝
            attributed_minute_data_map[date_obj] = attributed_minute_df.copy(deep=True)
        if not all_metrics_list:
            return pd.DataFrame(), {}, failures
        final_metrics_df = pd.DataFrame(all_metrics_list)
        final_metrics_df.set_index('trade_time', inplace=True)
        return final_metrics_df, attributed_minute_data_map, failures

    def _calculate_daily_derived_metrics(self, daily_data_series: pd.Series) -> dict:
        results = {}
        trade_date = daily_data_series.name.date()
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        consensus_map = {
            'net_flow_calibrated': ('net_flow_tushare', ['net_flow_ths', 'net_flow_dc']),
            'main_force_net_flow_calibrated': ('main_force_net_flow_tushare', ['main_force_net_flow_ths', 'main_force_net_flow_dc']),
            'retail_net_flow_calibrated': ('retail_net_flow_tushare', ['retail_net_flow_ths', 'retail_net_flow_dc']),
            'net_xl_amount_calibrated': ('net_xl_amount_tushare', ['net_xl_amount_dc']),
            'net_lg_amount_calibrated': ('net_lg_amount_tushare', ['net_lg_amount_ths', 'net_lg_amount_dc']),
            'net_md_amount_calibrated': ('net_md_amount_tushare', ['net_md_amount_ths', 'net_md_amount_dc']),
            'net_sh_amount_calibrated': ('net_sh_amount_tushare', ['net_sh_amount_ths', 'net_sh_amount_dc']),
        }
        for target_col, (base_col, confirm_cols) in consensus_map.items():
            base_value = pd.to_numeric(daily_data_series.get(base_col), errors='coerce')
            if is_probe_date and pd.isna(base_value):
                print(f"    -> [资金流校准探针] @ {trade_date}: 基础列 '{base_col}' 缺失。")
            if pd.isna(base_value):
                for conf_col in confirm_cols:
                    alt_value = pd.to_numeric(daily_data_series.get(conf_col), errors='coerce')
                    if is_probe_date and pd.isna(alt_value):
                        print(f"    -> [资金流校准探针] @ {trade_date}: 确认列 '{conf_col}' 缺失。")
                    if pd.notna(alt_value):
                        base_value = alt_value
                        break
            if pd.notna(base_value):
                confirmation_score = sum(1 for conf_col in confirm_cols if pd.notna(daily_data_series.get(conf_col)) and np.sign(base_value) == np.sign(pd.to_numeric(daily_data_series.get(conf_col), errors='coerce')))
                available_sources = sum(1 for conf_col in confirm_cols if pd.notna(daily_data_series.get(conf_col)))
                calibration_factor = (1 + confirmation_score) / (1 + available_sources) if available_sources > 0 else 1.0
                results[target_col] = base_value * calibration_factor
            else:
                results[target_col] = np.nan
        turnover_amount_yuan = pd.to_numeric(daily_data_series.get('amount'), errors='coerce') * 1000
        if is_probe_date and pd.isna(daily_data_series.get('amount')):
            print(f"    -> [日线衍生探针] @ {trade_date}: 原始列 'amount' 缺失。")
        if is_probe_date and turnover_amount_yuan <= 0:
            print(f"    -> [日线衍生探针] @ {trade_date}: 'turnover_amount_yuan' 为 {turnover_amount_yuan}，无法计算依赖其的指标。")
        if turnover_amount_yuan > 0:
            base_flow = pd.to_numeric(daily_data_series.get('main_force_net_flow_tushare'), errors='coerce')
            if is_probe_date and pd.isna(base_flow):
                print(f"    -> [日线衍生探针] @ {trade_date}: 'main_force_net_flow_tushare' 缺失。")
            confirm_flows = [pd.to_numeric(daily_data_series.get(c), errors='coerce') for c in ['main_force_net_flow_ths', 'main_force_net_flow_dc']]
            if pd.notna(base_flow):
                deviations = [abs(conf_flow - base_flow) / turnover_amount_yuan for conf_flow in confirm_flows if pd.notna(conf_flow)]
                results['flow_credibility_index'] = (1.0 - np.mean(deviations)) * 100 if deviations else 50.0
            mf_flow = results.get('main_force_net_flow_calibrated')
            retail_flow = results.get('retail_net_flow_calibrated')
            if is_probe_date and pd.isna(mf_flow):
                print(f"    -> [日线衍生探针] @ {trade_date}: 'main_force_net_flow_calibrated' 缺失。")
            if is_probe_date and pd.isna(retail_flow):
                print(f"    -> [日线衍生探针] @ {trade_date}: 'retail_net_flow_calibrated' 缺失。")
            if pd.notna(mf_flow) and pd.notna(retail_flow):
                total_flow_abs = abs(mf_flow) + abs(retail_flow)
                if total_flow_abs > 0:
                    battle_volume = min(abs(mf_flow), abs(retail_flow))
                    results['mf_retail_battle_intensity'] = np.sign(mf_flow) * (battle_volume / (turnover_amount_yuan / 10000)) * 100
                else:
                    results['mf_retail_battle_intensity'] = 0.0
            else:
                results['mf_retail_battle_intensity'] = np.nan
        mf_buy = np.nansum([
            pd.to_numeric(daily_data_series.get('buy_lg_amount'), errors='coerce'),
            pd.to_numeric(daily_data_series.get('buy_elg_amount'), errors='coerce')
        ])
        mf_sell = np.nansum([
            pd.to_numeric(daily_data_series.get('sell_lg_amount'), errors='coerce'),
            pd.to_numeric(daily_data_series.get('sell_elg_amount'), errors='coerce')
        ])
        if is_probe_date and pd.isna(daily_data_series.get('buy_lg_amount')):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'buy_lg_amount' 缺失。")
        if is_probe_date and pd.isna(daily_data_series.get('buy_elg_amount')):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'buy_elg_amount' 缺失。")
        if is_probe_date and pd.isna(daily_data_series.get('sell_lg_amount')):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'sell_lg_amount' 缺失。")
        if is_probe_date and pd.isna(daily_data_series.get('sell_elg_amount')):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'sell_elg_amount' 缺失。")
        mf_total_activity = mf_buy + mf_sell
        total_turnover_wan = turnover_amount_yuan / 10000
        if is_probe_date and total_turnover_wan <= 0:
            print(f"    -> [日线衍生探针] @ {trade_date}: 'total_turnover_wan' 为 {total_turnover_wan}，无法计算 'main_force_activity_ratio'。")
        if total_turnover_wan > 0:
            results['main_force_activity_ratio'] = (mf_total_activity / total_turnover_wan) * 100
        else:
            results['main_force_activity_ratio'] = np.nan
        if is_probe_date and mf_total_activity <= 0:
            print(f"    -> [日线衍生探针] @ {trade_date}: 'mf_total_activity' 为 {mf_total_activity}，无法计算 'main_force_flow_directionality'。")
        if mf_total_activity > 0:
            mf_net_calibrated = results.get('main_force_net_flow_calibrated')
            if is_probe_date and pd.isna(mf_net_calibrated):
                print(f"    -> [日线衍生探针] @ {trade_date}: 'main_force_net_flow_calibrated' 缺失，无法计算 'main_force_flow_directionality'。")
            if pd.notna(mf_net_calibrated):
                results['main_force_flow_directionality'] = (mf_net_calibrated / mf_total_activity) * 100
            else:
                results['main_force_flow_directionality'] = np.nan
        else:
            results['main_force_flow_directionality'] = np.nan
        def get_directionality(buy_c, sell_c):
            b = np.nan_to_num(pd.to_numeric(daily_data_series.get(buy_c), errors='coerce'))
            s = np.nan_to_num(pd.to_numeric(daily_data_series.get(sell_c), errors='coerce'))
            if is_probe_date and pd.isna(daily_data_series.get(buy_c)):
                print(f"    -> [日线衍生探针] @ {trade_date}: '{buy_c}' 缺失。")
            if is_probe_date and pd.isna(daily_data_series.get(sell_c)):
                print(f"    -> [日线衍生探针] @ {trade_date}: '{sell_c}' 缺失。")
            return (b - s) / (b + s) if (b + s) > 0 else 0.0
        xl_directionality = get_directionality('buy_elg_amount', 'sell_elg_amount')
        lg_directionality = get_directionality('buy_lg_amount', 'sell_lg_amount')
        results['main_force_conviction_index'] = ((xl_directionality + lg_directionality) / 2.0) * (1.0 - abs(xl_directionality - lg_directionality)) * 100
        mf_flow = results.get('main_force_net_flow_calibrated')
        retail_flow = results.get('retail_net_flow_calibrated')
        if is_probe_date and pd.isna(mf_flow):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'main_force_net_flow_calibrated' 缺失。")
        if is_probe_date and pd.isna(retail_flow):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'retail_net_flow_calibrated' 缺失。")
        if pd.notna(mf_flow) and pd.notna(retail_flow):
            total_opinionated_flow = abs(mf_flow) + abs(retail_flow)
            if total_opinionated_flow > 0:
                dominance_ratio = abs(retail_flow) / total_opinionated_flow
                divergence_penalty = 1 if np.sign(mf_flow) != np.sign(retail_flow) and mf_flow != 0 and retail_flow != 0 else 0
                results['retail_flow_dominance_index'] = np.sign(retail_flow) * dominance_ratio * (1 + divergence_penalty) * 100
            else:
                results['retail_flow_dominance_index'] = np.nan
        else:
            results['retail_flow_dominance_index'] = np.nan
        pct_change = pd.to_numeric(daily_data_series.get('pct_change'), errors='coerce')
        if is_probe_date and pd.isna(pct_change):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'pct_change' 缺失。")
        if is_probe_date and pd.isna(mf_flow):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'main_force_net_flow_calibrated' 缺失。")
        if is_probe_date and total_turnover_wan <= 0:
            print(f"    -> [日线衍生探针] @ {trade_date}: 'total_turnover_wan' 为 {total_turnover_wan}，无法计算 'main_force_price_impact_ratio'。")
        if pd.notna(pct_change) and pd.notna(mf_flow) and total_turnover_wan > 0:
            standardized_mf_flow = mf_flow / total_turnover_wan
            if abs(standardized_mf_flow) > 1e-6:
                results['main_force_price_impact_ratio'] = (pct_change / 100) / standardized_mf_flow
            else:
                results['main_force_price_impact_ratio'] = 0.0 if pct_change == 0 else np.inf * np.sign(pct_change)
        else:
            results['main_force_price_impact_ratio'] = np.nan
        if is_probe_date and total_turnover_wan <= 0:
            print(f"    -> [日线衍生探针] @ {trade_date}: 'total_turnover_wan' 为 {total_turnover_wan}，无法计算 'main_force_buy_rate_consensus'。")
        if total_turnover_wan > 0:
            results['main_force_buy_rate_consensus'] = (mf_buy / total_turnover_wan) * 100
        else:
            results['main_force_buy_rate_consensus'] = np.nan
        results['flow_efficiency_index'] = np.nan
        circ_mv = pd.to_numeric(daily_data_series.get('circ_mv'), errors='coerce')
        if is_probe_date and pd.isna(circ_mv):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'circ_mv' 缺失。")
        if is_probe_date and circ_mv <= 0:
            print(f"    -> [日线衍生探针] @ {trade_date}: 'circ_mv' 为 {circ_mv}，无法计算 'flow_efficiency_index'。")
        if is_probe_date and pd.isna(mf_flow):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'main_force_net_flow_calibrated' 缺失，无法计算 'flow_efficiency_index'。")
        if is_probe_date and pd.isna(pct_change):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'pct_change' 缺失，无法计算 'flow_efficiency_index'。")
        if pd.notna(circ_mv) and circ_mv > 0 and pd.notna(mf_flow) and pd.notna(pct_change):
            flow_input = mf_flow / circ_mv
            if abs(flow_input) > 1e-9:
                efficiency = (pct_change / 100) / flow_input
                results['flow_efficiency_index'] = np.sign(efficiency) * np.log1p(abs(efficiency))
            else:
                results['flow_efficiency_index'] = np.nan
        else:
            results['flow_efficiency_index'] = np.nan
        results['inferred_active_order_size'] = np.nan
        trade_count = pd.to_numeric(daily_data_series.get('trade_count'), errors='coerce')
        if is_probe_date and pd.isna(trade_count):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'trade_count' 缺失。")
        if is_probe_date and trade_count <= 0:
            print(f"    -> [日线衍生探针] @ {trade_date}: 'trade_count' 为 {trade_count}，无法计算 'inferred_active_order_size'。")
        if is_probe_date and pd.isna(turnover_amount_yuan):
            print(f"    -> [日线衍生探针] @ {trade_date}: 'turnover_amount_yuan' 缺失，无法计算 'inferred_active_order_size'。")
        if is_probe_date and turnover_amount_yuan <= 0:
            print(f"    -> [日线衍生探针] @ {trade_date}: 'turnover_amount_yuan' 为 {turnover_amount_yuan}，无法计算 'inferred_active_order_size'。")
        if pd.notna(trade_count) and trade_count > 0 and pd.notna(turnover_amount_yuan) and turnover_amount_yuan > 0:
            results['inferred_active_order_size'] = turnover_amount_yuan * 1000 / trade_count
        else:
            results['inferred_active_order_size'] = np.nan
        return results

    def _calculate_probabilistic_costs(self, stock_code: str, minute_data_for_day: pd.DataFrame, daily_data: pd.Series) -> tuple[dict, pd.DataFrame]:
        """
        【V6.14 · 完整归因返回版 - 探针增强】
        - 核心重构: 修改方法职责，使其在计算完概率成本后，继续调用 `_attribute_minute_volume_to_players` 进行主力/散户级别的成交量聚合。
        - 核心修复: 更改返回签名，同时返回成本指标字典和被完整归因（包含 `main_force_net_vol` 等列）的分钟DataFrame，修复数据流中断问题。
        - 【新增探针】增加探针，检查 `daily_vol_shares` 和 `weight_series.sum()` 的值，并打印 `daily_data` 中所有 `*_vol` 列。
        """
        trade_date = daily_data.name.date()
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        if is_probe_date:
            print(f"    -> [概率成本探针] @ {trade_date}: 检查 daily_data 中的成交量。")
            for col in ['buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol', 'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol']:
                print(f"       - daily_data['{col}']: {daily_data.get(col, 'N/A')}")
        if minute_data_for_day is None or minute_data_for_day.empty:
            if is_probe_date:
                print(f"    -> [概率成本探针] @ {trade_date}: 'minute_data_for_day' 为空，跳过计算。")
            return {}, pd.DataFrame()
        day_results = {}
        cost_types = ['sm_buy', 'sm_sell', 'md_buy', 'md_sell', 'lg_buy', 'lg_sell', 'elg_buy', 'elg_sell']
        df_to_attribute = minute_data_for_day
        for cost_type in cost_types:
            size, direction = cost_type.split('_')
            db_vol_key = f'{direction}_{size}_vol'
            daily_vol_shares = pd.to_numeric(daily_data.get(db_vol_key), errors='coerce') * 100 # 转换为股数
            if is_probe_date:
                print(f"    -> [概率成本探针] @ {trade_date}: '{cost_type}' - 每日总成交量 ('{db_vol_key}'): {daily_vol_shares:.2f}。")
            if pd.isna(daily_vol_shares) or daily_vol_shares == 0:
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                if is_probe_date:
                    print(f"    -> [概率成本探针] @ {trade_date}: '{cost_type}' - 每日总成交量为0或缺失，'{cost_type}_vol_attr' 设为0。")
                continue
            weight_col = f'{size}_{direction}_weight'
            if weight_col not in df_to_attribute.columns:
                if is_probe_date:
                    print(f"    -> [概率成本探针] @ {trade_date}: '{cost_type}' - 缺少权重列 '{weight_col}'。")
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                continue
            weight_series = df_to_attribute[weight_col]
            if is_probe_date:
                print(f"    -> [概率成本探针] @ {trade_date}: '{cost_type}' - 权重序列 ('{weight_col}') 总和: {weight_series.sum():.6f}。")
            if weight_series.sum() < 1e-9: # 检查权重总和是否过小
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                if is_probe_date:
                    print(f"    -> [概率成本探针] @ {trade_date}: '{cost_type}' - 权重序列总和过小，'{cost_type}_vol_attr' 设为0。")
                continue
            attributed_vol = weight_series * daily_vol_shares
            df_to_attribute[f'{cost_type}_vol_attr'] = attributed_vol
            if is_probe_date:
                print(f"    -> [概率成本探针] @ {trade_date}: '{cost_type}' - 归因成交量 ('{cost_type}_vol_attr') 总和: {attributed_vol.sum():.2f}。")
            attributed_value = attributed_vol * df_to_attribute['minute_vwap']
            total_attributed_value = attributed_value.sum()
            total_attributed_vol = attributed_vol.sum()
            day_results[f'avg_cost_{cost_type}'] = total_attributed_value / total_attributed_vol if total_attributed_vol > 0 else np.nan
        fully_attributed_df = self._attribute_minute_volume_to_players(df_to_attribute)
        return day_results, fully_attributed_df

    def _calculate_aggregate_pvwap_costs(self, pvwap_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        temp_df = pvwap_df.copy()
        trade_date = temp_df.index[0].date() if not temp_df.empty else 'UNKNOWN'
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        cols_to_join = [
            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',
            'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
            'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount'
        ]
        existing_cols_to_join = [col for col in cols_to_join if col in daily_df.columns]
        agg_cols = [
            'avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell',
            'main_force_cost_alpha', 'retail_cost_beta', 'main_force_t0_spread_ratio',
            'main_force_execution_alpha', 'main_force_t0_efficiency', 'flow_temperature_premium'
        ]
        if not existing_cols_to_join:
            if is_probe_date:
                print(f"    -> [聚合成本探针] @ {trade_date}: 'daily_df' 缺少所有资金流成交量/额列。")
            return pd.DataFrame(columns=agg_cols, index=pvwap_df.index)
        temp_df = temp_df.join(daily_df[existing_cols_to_join])
        def weighted_avg_cost(cost_cols, vol_cols):
            numerator = pd.Series(0.0, index=temp_df.index)
            denominator = pd.Series(0.0, index=temp_df.index)
            for cost_col, vol_col in zip(cost_cols, vol_cols):
                if cost_col in temp_df.columns and vol_col in temp_df.columns:
                    cost = temp_df[cost_col]
                    volume_data = temp_df[vol_col]
                    if isinstance(volume_data, pd.Series):
                        volume_shares = pd.to_numeric(volume_data, errors='coerce').fillna(0) * 100
                    else:
                        volume_shares = pd.to_numeric(volume_data, errors='coerce')
                        volume_shares = 0 if pd.isna(volume_shares) else volume_shares * 100
                    value_contribution = (cost * volume_shares).fillna(0)
                    numerator += value_contribution
                    denominator += volume_shares.where(cost.notna(), 0)
                else:
                    if is_probe_date:
                        print(f"    -> [聚合成本探针] @ {trade_date}: 缺少列 '{cost_col}' 或 '{vol_col}'。")
            return numerator / denominator.replace(0, np.nan)
        result_agg_df = pd.DataFrame(index=pvwap_df.index)
        result_agg_df['avg_cost_main_buy'] = weighted_avg_cost(['avg_cost_lg_buy', 'avg_cost_elg_buy'], ['buy_lg_vol', 'buy_elg_vol'])
        result_agg_df['avg_cost_main_sell'] = weighted_avg_cost(['avg_cost_lg_sell', 'avg_cost_elg_sell'], ['sell_lg_vol', 'sell_elg_vol'])
        result_agg_df['avg_cost_retail_buy'] = weighted_avg_cost(['avg_cost_sm_buy', 'avg_cost_md_buy'], ['buy_sm_vol', 'buy_md_vol'])
        result_agg_df['avg_cost_retail_sell'] = weighted_avg_cost(['avg_cost_sm_sell', 'avg_cost_md_sell'], ['sell_sm_vol', 'sell_md_vol'])
        # 修改行：修正daily_vwap的来源，应从daily_df计算而不是pvwap_df获取
        amount = pd.to_numeric(daily_df.get('amount'), errors='coerce') * 1000
        volume = pd.to_numeric(daily_df.get('vol'), errors='coerce') * 100
        daily_vwap = amount / volume.replace(0, np.nan)
        if is_probe_date and pd.isna(daily_vwap).any():
            print(f"    -> [聚合成本探针] @ {trade_date}: 'daily_vwap' 存在缺失值。")
        if is_probe_date and (daily_vwap == 0).any():
            print(f"    -> [聚合成本探针] @ {trade_date}: 'daily_vwap' 存在0值。")
        if 'avg_cost_main_buy' in result_agg_df.columns and daily_vwap is not None and not daily_vwap.empty:
            safe_vwap = daily_vwap.replace(0, np.nan)
            result_agg_df['main_force_cost_alpha'] = ((safe_vwap - result_agg_df['avg_cost_main_buy']) / safe_vwap) * 100
        else:
            result_agg_df['main_force_cost_alpha'] = np.nan
        if 'avg_cost_retail_buy' in result_agg_df.columns and 'avg_cost_main_sell' in result_agg_df.columns:
            safe_main_sell_cost = result_agg_df['avg_cost_main_sell'].replace(0, np.nan)
            result_agg_df['retail_cost_beta'] = ((result_agg_df['avg_cost_retail_buy'] - safe_main_sell_cost) / safe_main_sell_cost) * 100
        else:
            result_agg_df['retail_cost_beta'] = np.nan
        # flow_temperature_premium
        result_agg_df['flow_temperature_premium'] = np.nan
        if 'avg_cost_main_buy' in result_agg_df.columns and daily_vwap is not None and not daily_vwap.empty:
            safe_vwap = daily_vwap.replace(0, np.nan)
            if is_probe_date and pd.isna(result_agg_df['avg_cost_main_buy']).any():
                print(f"    -> [聚合成本探针] @ {trade_date}: 'avg_cost_main_buy' 存在缺失值。")
            if pd.notna(result_agg_df['avg_cost_main_buy']).all() and pd.notna(safe_vwap).all():
                result_agg_df['flow_temperature_premium'] = (result_agg_df['avg_cost_main_buy'] / safe_vwap - 1) * 100
        if 'avg_cost_main_sell' in result_agg_df.columns and 'avg_cost_main_buy' in result_agg_df.columns and daily_vwap is not None and not daily_vwap.empty:
            t0_spread = result_agg_df['avg_cost_main_sell'] - result_agg_df['avg_cost_main_buy']
            spread_ratio = (t0_spread / daily_vwap.replace(0, np.nan)) * 100
            result_agg_df['main_force_t0_spread_ratio'] = spread_ratio
        else:
            result_agg_df['main_force_t0_spread_ratio'] = np.nan
        # main_force_execution_alpha
        result_agg_df['main_force_execution_alpha'] = np.nan
        mf_buy_vol_series = self._get_numeric_series_with_nan(temp_df, 'buy_lg_vol').fillna(0) + self._get_numeric_series_with_nan(temp_df, 'buy_elg_vol').fillna(0)
        mf_sell_vol_series = self._get_numeric_series_with_nan(temp_df, 'sell_lg_vol').fillna(0) + self._get_numeric_series_with_nan(temp_df, 'sell_elg_vol').fillna(0)
        if is_probe_date and (mf_buy_vol_series == 0).all():
            print(f"    -> [聚合成本探针] @ {trade_date}: 'mf_buy_vol_series' 全为0。")
        if is_probe_date and (mf_sell_vol_series == 0).all():
            print(f"    -> [聚合成本探针] @ {trade_date}: 'mf_sell_vol_series' 全为0。")
        total_mf_vol = (mf_buy_vol_series + mf_sell_vol_series) * 100 # 转换为股数
        if daily_vwap is not None and not daily_vwap.empty and \
           'avg_cost_main_buy' in result_agg_df.columns and 'avg_cost_main_sell' in result_agg_df.columns:
            safe_vwap = daily_vwap.replace(0, np.nan)
            alpha_buy = ((safe_vwap - result_agg_df['avg_cost_main_buy']) / safe_vwap).fillna(0)
            alpha_sell = ((result_agg_df['avg_cost_main_sell'] - safe_vwap) / safe_vwap).fillna(0)
            weighted_alpha = (alpha_buy * mf_buy_vol_series + alpha_sell * mf_sell_vol_series)
            if is_probe_date and (total_mf_vol == 0).any():
                print(f"    -> [聚合成本探针] @ {trade_date}: 'total_mf_vol' 存在0值，导致 'main_force_execution_alpha' 分母为0。")
            result_agg_df['main_force_execution_alpha'] = (weighted_alpha / np.where(total_mf_vol == 0, np.nan, total_mf_vol)) * 100
        # main_force_t0_efficiency
        result_agg_df['main_force_t0_efficiency'] = np.nan
        mf_buy_amount_series = self._get_numeric_series_with_nan(temp_df, 'buy_lg_amount').fillna(0) + self._get_numeric_series_with_nan(temp_df, 'buy_elg_amount').fillna(0)
        mf_sell_amount_series = self._get_numeric_series_with_nan(temp_df, 'sell_lg_amount').fillna(0) + self._get_numeric_series_with_nan(temp_df, 'sell_elg_amount').fillna(0)
        if is_probe_date and (mf_buy_amount_series == 0).all():
            print(f"    -> [聚合成本探针] @ {trade_date}: 'mf_buy_amount_series' 全为0。")
        if is_probe_date and (mf_sell_amount_series == 0).all():
            print(f"    -> [聚合成本探针] @ {trade_date}: 'mf_sell_amount_series' 全为0。")
        # 将 pd.min 替换为 np.minimum
        t0_vol = np.minimum(mf_buy_vol_series, mf_sell_vol_series)
        if 'avg_cost_main_sell' in result_agg_df.columns and 'avg_cost_main_buy' in result_agg_df.columns:
            t0_profit = (result_agg_df['avg_cost_main_sell'] - result_agg_df['avg_cost_main_buy']) * t0_vol
            total_mf_amount = (mf_buy_amount_series + mf_sell_amount_series) * 10000 # 转换为元
            if is_probe_date and (total_mf_amount == 0).any():
                print(f"    -> [聚合成本探针] @ {trade_date}: 'total_mf_amount' 存在0值，导致 'main_force_t0_efficiency' 分母为0。")
            result_agg_df['main_force_t0_efficiency'] = (t0_profit / np.where(total_mf_amount == 0, np.nan, total_mf_amount)) * 100
        if 'market_cost_battle_premium' in result_agg_df.columns:
            result_agg_df = result_agg_df.drop(columns=['market_cost_battle_premium'])
        return result_agg_df

    def _attribute_minute_volume_to_players(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 · 探针增强版】将基础成交量归因为主力/散户的核心辅助函数。
        - 核心职责: 聚合基础的 *_vol_attr 列，生成 main_force_* 和 retail_* 级别的成交量列。
        - 【新增探针】检查归因后的主力/散户买卖量。
        """
        df = minute_df.copy()
        trade_date = df.index[0].date() if not df.empty else 'UNKNOWN'
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        df['main_force_buy_vol'] = df.get('lg_buy_vol_attr', 0) + df.get('elg_buy_vol_attr', 0)
        df['main_force_sell_vol'] = df.get('lg_sell_vol_attr', 0) + df.get('elg_sell_vol_attr', 0)
        df['main_force_net_vol'] = df['main_force_buy_vol'] - df['main_force_sell_vol']
        df['retail_buy_vol'] = df.get('sm_buy_vol_attr', 0) + df.get('md_buy_vol_attr', 0)
        df['retail_sell_vol'] = df.get('sm_sell_vol_attr', 0) + df.get('md_sell_vol_attr', 0)
        df['retail_net_vol'] = df['retail_buy_vol'] - df['retail_sell_vol']
        if is_probe_date:
            print(f"    -> [归因聚合探针] @ {trade_date}: _attribute_minute_volume_to_players 内部检查。")
            print(f"       - lg_sell_vol_attr sum: {df.get('lg_sell_vol_attr', pd.Series(0.0, index=df.index)).sum():.2f}")
            print(f"       - elg_sell_vol_attr sum: {df.get('elg_sell_vol_attr', pd.Series(0.0, index=df.index)).sum():.2f}")
            print(f"       - sm_sell_vol_attr sum: {df.get('sm_sell_vol_attr', pd.Series(0.0, index=df.index)).sum():.2f}")
            print(f"       - md_sell_vol_attr sum: {df.get('md_sell_vol_attr', pd.Series(0.0, index=df.index)).sum():.2f}")
            print(f"       - main_force_sell_vol sum (after agg): {df['main_force_sell_vol'].sum():.2f}")
            print(f"       - retail_sell_vol sum (after agg): {df['retail_sell_vol'].sum():.2f}")
        return df

    def _calculate_derivatives(self, stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
        derivatives_df = pd.DataFrame(index=consensus_df.index)
        import pandas_ta as ta
        trade_date = consensus_df.index[-1].date() if not consensus_df.empty else 'UNKNOWN'
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
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
                    if is_probe_date and source_series_for_sum.isnull().any():
                        print(f"    -> [衍生指标探针] @ {trade_date}: 累计值计算源 '{col}' 存在缺失值。")
                    sum_col_name = f'{col}_sum_{p}d'
                    derivatives_df[sum_col_name] = source_series_for_sum.rolling(window=p, min_periods=min_p).sum()
                else:
                    if is_probe_date:
                        print(f"    -> [衍生指标探针] @ {trade_date}: 累计值计算源 '{col}' 不存在于 'consensus_df'。")
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
            if is_probe_date and source_series.isnull().all():
                print(f"    -> [衍生指标探针] @ {trade_date}: 斜率/加速度计算源 '{col}' 全为缺失值。")
            if source_series.isnull().all():
                continue
            for p in UNIFIED_PERIODS:
                calc_window = max(2, p) if p > 1 else 2
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
        # 移除所有探针性质的print语句
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

    def _compute_all_behavioral_metrics(self, intraday_data: pd.DataFrame, daily_data: pd.Series) -> dict:
        from scipy.signal import find_peaks
        results = {}
        trade_date = daily_data.name.date()
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        if intraday_data.empty:
            if is_probe_date:
                print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 为空，跳过所有计算。")
            return results
        daily_total_volume = daily_data.get('vol', 0) * 100
        daily_total_amount = pd.to_numeric(daily_data.get('amount', 0), errors='coerce') * 1000
        daily_vwap = np.nan
        if daily_total_volume > 0:
            daily_vwap = daily_total_amount / daily_total_volume
        atr = daily_data.get('atr_14d')
        day_open, day_close = daily_data.get('open_qfq'), daily_data.get('close_qfq')
        day_high, day_low = daily_data.get('high_qfq'), daily_data.get('low_qfq')
        pre_close = daily_data.get('pre_close_qfq')
        results['vwap_control_strength'] = np.nan
        results['main_force_vwap_guidance'] = np.nan
        results['vwap_crossing_intensity'] = np.nan
        results['vwap_structure_skew'] = np.nan
        results['opening_battle_result'] = np.nan
        results['upper_shadow_selling_pressure'] = np.nan
        results['lower_shadow_absorption_strength'] = np.nan
        results['reversal_power_index'] = np.nan
        results['trend_conviction_ratio'] = np.nan
        results['holistic_cmf'] = np.nan
        results['main_force_cmf'] = np.nan
        results['cmf_divergence_score'] = np.nan
        results['main_force_on_peak_flow'] = np.nan
        results['main_force_vpoc'] = np.nan
        results['mf_vpoc_premium'] = np.nan
        results['mf_retail_liquidity_swap_corr'] = np.nan
        results['asymmetric_volume_thrust'] = np.nan
        results['closing_auction_ambush'] = np.nan
        results['closing_price_deviation_score'] = np.nan
        results['dip_absorption_power'] = np.nan
        results['rally_distribution_pressure'] = np.nan
        results['panic_selling_cascade'] = np.nan
        results['pre_closing_posturing'] = np.nan
        results['retail_fomo_premium_index'] = np.nan
        results['retail_panic_surrender_index'] = np.nan
        results['volatility_asymmetry_index'] = np.nan
        # results['opening_gap_defense_strength'] = np.nan # 撤回
        # results['upward_impulse_purity'] = np.nan # 撤回
        # 逐一检查核心依赖
        if is_probe_date and pd.isna(daily_vwap):
            print(f"    -> [行为指标探针] @ {trade_date}: 'daily_vwap' 缺失。")
        if is_probe_date and daily_total_volume <= 0:
            print(f"    -> [行为指标探针] @ {trade_date}: 'daily_total_volume' 为 {daily_total_volume}，无法计算依赖其的指标。")
        if is_probe_date and pd.isna(atr):
            print(f"    -> [行为指标探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr <= 0:
            print(f"    -> [行为指标探针] @ {trade_date}: 'atr_14d' 为 {atr}，无法计算依赖其的指标。")
        if is_probe_date and 'minute_vwap' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'minute_vwap' 列。")
        if is_probe_date and 'vol_shares' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'vol_shares' 列。")
        if is_probe_date and 'main_force_net_vol' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'main_force_net_vol' 列。")
        if pd.notna(daily_vwap) and daily_total_volume > 0 and pd.notna(atr) and atr > 0 and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            price_deviation_value = (intraday_data['minute_vwap'] - daily_vwap) * intraday_data['vol_shares']
            results['vwap_control_strength'] = price_deviation_value.sum() / (atr * daily_total_volume)
            price_dev_series = intraday_data['minute_vwap'] - daily_vwap
            mf_net_flow_series = intraday_data['main_force_net_vol']
            if is_probe_date and price_dev_series.var() == 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'price_dev_series' 方差为0，无法计算相关性。")
            if is_probe_date and mf_net_flow_series.var() == 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'mf_net_flow_series' 方差为0，无法计算相关性。")
            if price_dev_series.var() != 0 and mf_net_flow_series.var() != 0 and len(price_dev_series) > 1:
                correlation = price_dev_series.corr(mf_net_flow_series)
                results['main_force_vwap_guidance'] = correlation if pd.notna(correlation) else np.nan
            else:
                results['main_force_vwap_guidance'] = np.nan
            position_vs_vwap = np.sign(intraday_data['minute_vwap'] - daily_vwap)
            crossings = position_vs_vwap.diff().ne(0)
            results['vwap_crossing_intensity'] = intraday_data.loc[crossings, 'vol_shares'].sum() / daily_total_volume
            twap = intraday_data['minute_vwap'].mean()
            if is_probe_date and pd.isna(twap):
                print(f"    -> [行为指标探针] @ {trade_date}: 'twap' 缺失。")
            if is_probe_date and twap <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'twap' 为 {twap}，无法计算 'vwap_structure_skew'。")
            if pd.notna(twap) and twap > 0:
                results['vwap_structure_skew'] = (daily_vwap - twap) / twap * 100
            else:
                results['vwap_structure_skew'] = np.nan
        else:
            results['vwap_control_strength'] = np.nan
            results['main_force_vwap_guidance'] = np.nan
            results['vwap_crossing_intensity'] = np.nan
            results['vwap_structure_skew'] = np.nan
        # 使用 intraday_data.index 访问时间
        opening_battle_df = intraday_data[(intraday_data.index.time >= time(9, 30)) & (intraday_data.index.time <= time(9, 45))]
        if is_probe_date and opening_battle_df.empty:
            print(f"    -> [行为指标探针] @ {trade_date}: 'opening_battle_df' 为空。")
        if is_probe_date and len(opening_battle_df) <= 1:
            print(f"    -> [行为指标探针] @ {trade_date}: 'opening_battle_df' 长度不足2。")
        if is_probe_date and pd.isna(atr):
            print(f"    -> [行为指标探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr <= 0:
            print(f"    -> [行为指标探针] @ {trade_date}: 'atr_14d' 为 {atr}，无法计算 'opening_battle_result'。")
        if is_probe_date and 'close' not in opening_battle_df.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'opening_battle_df' 缺少 'close' 列。")
        if is_probe_date and 'open' not in opening_battle_df.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'opening_battle_df' 缺少 'open' 列。")
        if is_probe_date and 'vol_shares' not in opening_battle_df.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'opening_battle_df' 缺少 'vol_shares' 列。")
        if is_probe_date and 'minute_vwap' not in opening_battle_df.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'opening_battle_df' 缺少 'minute_vwap' 列。")
        if is_probe_date and 'main_force_net_vol' not in opening_battle_df.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'opening_battle_df' 缺少 'main_force_net_vol' 列。")
        if not opening_battle_df.empty and len(opening_battle_df) > 1 and pd.notna(atr) and atr > 0 and \
           'close' in opening_battle_df.columns and 'open' in opening_battle_df.columns and \
           'vol_shares' in opening_battle_df.columns and 'minute_vwap' in opening_battle_df.columns and \
           'main_force_net_vol' in opening_battle_df.columns:
            price_gain = (opening_battle_df['close'].iloc[-1] - opening_battle_df['open'].iloc[0]) / atr
            battle_amount = (opening_battle_df['vol_shares'] * opening_battle_df['minute_vwap']).sum()
            if is_probe_date and battle_amount <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'opening_battle_df' 的 'battle_amount' 为 {battle_amount}。")
            if battle_amount > 0:
                mf_power = opening_battle_df['main_force_net_vol'].sum() * opening_battle_df['minute_vwap'].mean() / battle_amount
                results['opening_battle_result'] = np.sign(price_gain) * np.sqrt(abs(price_gain)) * (1 + mf_power) * 100
            else:
                results['opening_battle_result'] = np.nan
        else:
            results['opening_battle_result'] = np.nan
        if is_probe_date and pd.isna(day_open):
            print(f"    -> [行为指标探针] @ {trade_date}: 'day_open' 缺失。")
        if is_probe_date and pd.isna(day_close):
            print(f"    -> [行为指标探针] @ {trade_date}: 'day_close' 缺失。")
        if is_probe_date and pd.isna(day_high):
            print(f"    -> [行为指标探针] @ {trade_date}: 'day_high' 缺失。")
        if is_probe_date and pd.isna(day_low):
            print(f"    -> [行为指标探针] @ {trade_date}: 'day_low' 缺失。")
        if is_probe_date and 'high' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'high' 列。")
        if is_probe_date and 'low' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'low' 列。")
        if is_probe_date and 'vol_shares' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'vol_shares' 列。")
        if is_probe_date and 'main_force_net_vol' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'main_force_net_vol' 列。")
        if pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low) and \
           'high' in intraday_data.columns and 'low' in intraday_data.columns and \
           'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            body_high, body_low = max(day_open, day_close), min(day_open, day_close)
            upper_shadow_df = intraday_data[intraday_data['high'] > body_high]
            if is_probe_date and upper_shadow_df.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'upper_shadow_df' 为空。")
            if is_probe_date and upper_shadow_df['vol_shares'].sum() <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'upper_shadow_df' 的 'vol_shares' 总和为 {upper_shadow_df['vol_shares'].sum()}。")
            if not upper_shadow_df.empty and upper_shadow_df['vol_shares'].sum() > 0:
                mf_sell_in_shadow = abs(upper_shadow_df[upper_shadow_df['main_force_net_vol'] < 0]['main_force_net_vol'].sum())
                results['upper_shadow_selling_pressure'] = (mf_sell_in_shadow / upper_shadow_df['vol_shares'].sum())
            else:
                results['upper_shadow_selling_pressure'] = np.nan
            lower_shadow_df = intraday_data[intraday_data['low'] < body_low]
            if is_probe_date and lower_shadow_df.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'lower_shadow_df' 为空。")
            if is_probe_date and lower_shadow_df['vol_shares'].sum() <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'lower_shadow_df' 的 'vol_shares' 总和为 {lower_shadow_df['vol_shares'].sum()}。")
            if not lower_shadow_df.empty and lower_shadow_df['vol_shares'].sum() > 0:
                mf_net_flow = lower_shadow_df['main_force_net_vol'].sum()
                results['lower_shadow_absorption_strength'] = mf_net_flow / lower_shadow_df['vol_shares'].sum()
            else:
                results['lower_shadow_absorption_strength'] = np.nan
        else:
            results['upper_shadow_selling_pressure'] = np.nan
            results['lower_shadow_absorption_strength'] = np.nan
        if is_probe_date and len(intraday_data) < 10:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 长度不足10。")
        if is_probe_date and pd.isna(day_open):
            print(f"    -> [行为指标探针] @ {trade_date}: 'day_open' 缺失。")
        if is_probe_date and pd.isna(day_close):
            print(f"    -> [行为指标探针] @ {trade_date}: 'day_close' 缺失。")
        if is_probe_date and pd.isna(day_high):
            print(f"    -> [行为指标探针] @ {trade_date}: 'day_high' 缺失。")
        if is_probe_date and pd.isna(day_low):
            print(f"    -> [行为指标探针] @ {trade_date}: 'day_low' 缺失。")
        if is_probe_date and pd.isna(atr):
            print(f"    -> [行为指标探针] @ {trade_date}: 'atr_14d' 缺失。")
        if is_probe_date and atr <= 0:
            print(f"    -> [行为指标探针] @ {trade_date}: 'atr_14d' 为 {atr}，无法计算 'trend_conviction_ratio'。")
        if is_probe_date and daily_total_volume <= 0:
            print(f"    -> [行为指标探针] @ {trade_date}: 'daily_total_volume' 为 {daily_total_volume}，无法计算 'trend_conviction_ratio'。")
        if is_probe_date and 'main_force_buy_vol' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'main_force_buy_vol' 列。")
        if is_probe_date and 'main_force_sell_vol' not in intraday_data.columns:
            print(f"    -> [行为指标探针] @ {trade_date}: 'intraday_data' 缺少 'main_force_sell_vol' 列。")
        if len(intraday_data) >= 10 and pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low) and pd.notna(atr) and atr > 0 and daily_total_volume > 0 and \
           'main_force_buy_vol' in intraday_data.columns and 'main_force_sell_vol' in intraday_data.columns:
            mf_activity_ratio = (intraday_data['main_force_buy_vol'].sum() + intraday_data['main_force_sell_vol'].sum()) / daily_total_volume
            if is_probe_date and mf_activity_ratio <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'mf_activity_ratio' 为 {mf_activity_ratio}，无法计算 'trend_conviction_ratio'。")
            if mf_activity_ratio > 0:
                price_outcome = (day_close - day_open) / atr
                results['trend_conviction_ratio'] = price_outcome / mf_activity_ratio
            else:
                results['trend_conviction_ratio'] = np.nan
            day_range = day_high - day_low
            if is_probe_date and day_range <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'day_range' 为 {day_range}。")
            if day_range > 0:
                is_v_shape = (day_close - day_open) > 0
                turn_point_idx = np.argmin(intraday_data['low'].values) if is_v_shape else np.argmax(intraday_data['high'].values)
                if 0 < turn_point_idx < len(intraday_data) - 1:
                    initial_phase = intraday_data.iloc[:turn_point_idx]
                    reversal_phase = intraday_data.iloc[turn_point_idx:]
                    vol_initial, vol_reversal = initial_phase['vol_shares'].sum(), reversal_phase['vol_shares'].sum()
                    if is_probe_date and vol_initial <= 0:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'initial_phase' 的 'vol_shares' 总和为 {vol_initial}。")
                    if is_probe_date and vol_reversal <= 0:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'reversal_phase' 的 'vol_shares' 总和为 {vol_reversal}。")
                    if vol_initial > 0 and vol_reversal > 0:
                        turn_point_vwap = intraday_data['minute_vwap'].iloc[turn_point_idx]
                        price_recovery = abs(day_close - turn_point_vwap) / day_range
                        vol_shift = np.log1p(vol_reversal / vol_initial)
                        reversal_mf_net_vol = reversal_phase['main_force_net_vol'].sum()
                        reversal_conviction = reversal_mf_net_vol / vol_reversal if vol_reversal > 0 else 0
                        power_score = price_recovery * vol_shift * reversal_conviction
                        results['reversal_power_index'] = power_score if is_v_shape else -power_score
                    else:
                        results['reversal_power_index'] = np.nan
                else:
                    results['reversal_power_index'] = np.nan
            else:
                results['reversal_power_index'] = np.nan
        else:
            results['trend_conviction_ratio'] = np.nan
            results['reversal_power_index'] = np.nan
        if 'minute_vwap' in intraday_data.columns and 'high' in intraday_data.columns and 'low' in intraday_data.columns and 'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            df_cmf = intraday_data.copy()
            cols_to_process = ['high', 'low', 'minute_vwap']
            for col in cols_to_process:
                df_cmf[col] = pd.to_numeric(df_cmf[col], errors='coerce')
            df_cmf.dropna(subset=cols_to_process, inplace=True)
            if is_probe_date and df_cmf.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'df_cmf' 为空。")
            if not df_cmf.empty:
                price_range_cmf = df_cmf['high'] - df_cmf['low']
                mfm = ((df_cmf['minute_vwap'] - df_cmf['low']) - (df_cmf['high'] - df_cmf['minute_vwap'])) / price_range_cmf.replace(0, np.nan)
                mfm.fillna(0, inplace=True)
                if is_probe_date and df_cmf['vol_shares'].sum() <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'df_cmf' 的 'vol_shares' 总和为 {df_cmf['vol_shares'].sum()}。")
                if df_cmf['vol_shares'].sum() > 0:
                    results['holistic_cmf'] = (mfm * df_cmf['vol_shares']).sum() / df_cmf['vol_shares'].sum()
                else:
                    results['holistic_cmf'] = np.nan
                if is_probe_date and df_cmf['main_force_net_vol'].abs().sum() <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'df_cmf' 的 'main_force_net_vol' 绝对值总和为 {df_cmf['main_force_net_vol'].abs().sum()}。")
                if df_cmf['main_force_net_vol'].abs().sum() > 0:
                    results['main_force_cmf'] = (mfm * df_cmf['main_force_net_vol']).sum() / df_cmf['main_force_net_vol'].abs().sum()
                else:
                    results['main_force_cmf'] = np.nan
                results['cmf_divergence_score'] = results.get('main_force_cmf', np.nan) - results.get('holistic_cmf', np.nan)
            else:
                results['holistic_cmf'] = np.nan
                results['main_force_cmf'] = np.nan
                results['cmf_divergence_score'] = np.nan
        else:
            results['holistic_cmf'] = np.nan
            results['main_force_cmf'] = np.nan
            results['cmf_divergence_score'] = np.nan
        if 'main_force_net_vol' in intraday_data.columns and pd.notna(day_close) and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
            vp_global = intraday_data.groupby(pd.cut(intraday_data['minute_vwap'], bins=30, duplicates='drop'))['vol_shares'].sum()
            if is_probe_date and vp_global.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'vp_global' 为空。")
            if not vp_global.empty:
                vpoc_interval = vp_global.idxmax()
                global_vpoc_price = vpoc_interval.mid
                peak_zone_df = intraday_data[
                    (intraday_data['minute_vwap'] >= vpoc_interval.left) &
                    (intraday_data['minute_vwap'] < vpoc_interval.right)
                ]
                if is_probe_date and peak_zone_df.empty:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'peak_zone_df' 为空。")
                if not peak_zone_df.empty:
                    mf_net_vol_on_peak = peak_zone_df['main_force_net_vol'].sum()
                    if is_probe_date and daily_total_amount <= 0:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'daily_total_amount' 为 {daily_total_amount}。")
                    if daily_total_amount > 0:
                        normalized_mf_on_peak_flow = np.tanh((mf_net_vol_on_peak * global_vpoc_price) / daily_total_amount)
                        results['main_force_on_peak_flow'] = normalized_mf_on_peak_flow
                    else:
                        results['main_force_on_peak_flow'] = np.nan
                else:
                    results['main_force_on_peak_flow'] = np.nan
            else:
                results['main_force_on_peak_flow'] = np.nan
            mf_net_buy_df = intraday_data[intraday_data['main_force_net_vol'] > 0]
            if is_probe_date and mf_net_buy_df.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'mf_net_buy_df' 为空。")
            if not mf_net_buy_df.empty:
                vp_mf = mf_net_buy_df.groupby(pd.cut(mf_net_buy_df['minute_vwap'], bins=30, duplicates='drop'))['main_force_net_vol'].sum()
                if is_probe_date and vp_mf.empty:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'vp_mf' 为空。")
                if not vp_mf.empty:
                    mf_vpoc = vp_mf.idxmax().mid
                    results['main_force_vpoc'] = mf_vpoc
                    if is_probe_date and pd.isna(global_vpoc_price):
                        print(f"    -> [行为指标探针] @ {trade_date}: 'global_vpoc_price' 缺失。")
                    if is_probe_date and global_vpoc_price <= 0:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'global_vpoc_price' 为 {global_vpoc_price}。")
                    if is_probe_date and pd.isna(mf_vpoc):
                        print(f"    -> [行为指标探针] @ {trade_date}: 'mf_vpoc' 缺失。")
                    if pd.notna(global_vpoc_price) and global_vpoc_price > 0 and pd.notna(mf_vpoc):
                        results['mf_vpoc_premium'] = (mf_vpoc / global_vpoc_price - 1) * 100
                    else:
                        results['mf_vpoc_premium'] = np.nan
                else:
                    results['main_force_vpoc'] = np.nan
                    results['mf_vpoc_premium'] = np.nan
            else:
                results['main_force_vpoc'] = np.nan
                results['mf_vpoc_premium'] = np.nan
        else:
            results['main_force_on_peak_flow'] = np.nan
            results['main_force_vpoc'] = np.nan
            results['mf_vpoc_premium'] = np.nan
        if 'main_force_net_vol' in intraday_data.columns and 'retail_net_vol' in intraday_data.columns:
            mf_net_series = intraday_data['main_force_net_vol']
            retail_net_series = intraday_data['retail_net_vol']
            if is_probe_date and mf_net_series.var() == 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'mf_net_series' 方差为0。")
            if is_probe_date and retail_net_series.var() == 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'retail_net_series' 方差为0。")
            if mf_net_series.var() != 0 and retail_net_series.var() != 0 and len(mf_net_series) > 1:
                rolling_corr = mf_net_series.rolling(window=30).corr(retail_net_series)
                results['mf_retail_liquidity_swap_corr'] = rolling_corr.mean()
            else:
                results['mf_retail_liquidity_swap_corr'] = np.nan
        else:
            results['mf_retail_liquidity_swap_corr'] = np.nan
        # 使用 intraday_data.index 访问时间
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if is_probe_date and continuous_trading_df.empty:
            print(f"    -> [行为指标探针] @ {trade_date}: 'continuous_trading_df' 为空。")
        if pd.notna(daily_vwap) and daily_total_volume > 0 and pd.notna(atr) and atr > 0 and not continuous_trading_df.empty and \
           'close' in continuous_trading_df.columns and 'open' in continuous_trading_df.columns and \
           'vol_shares' in continuous_trading_df.columns and 'minute_vwap' in continuous_trading_df.columns and \
           'main_force_net_vol' in continuous_trading_df.columns:
            up_minutes = continuous_trading_df[continuous_trading_df['close'] > continuous_trading_df['open']]
            down_minutes = continuous_trading_df[continuous_trading_df['close'] < continuous_trading_df['open']]
            if is_probe_date and up_minutes.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'up_minutes' 为空。")
            if is_probe_date and down_minutes.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'down_minutes' 为空。")
            if not up_minutes.empty and not down_minutes.empty:
                up_price_change = (up_minutes['close'] - up_minutes['open']).sum()
                up_vol = up_minutes['vol_shares'].sum()
                down_price_change = (down_minutes['open'] - down_minutes['close']).sum()
                down_vol = down_minutes['vol_shares'].sum()
                if is_probe_date and up_vol <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'up_vol' 为 {up_vol}。")
                if is_probe_date and down_vol <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'down_vol' 为 {down_vol}。")
                if up_vol > 0 and down_vol > 0:
                    upward_efficacy = up_price_change / up_vol
                    downward_efficacy = down_price_change / down_vol
                    if is_probe_date and downward_efficacy <= 1e-9:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'downward_efficacy' 为 {downward_efficacy}。")
                    if downward_efficacy > 1e-9:
                        thrust_ratio = upward_efficacy / downward_efficacy
                        results['asymmetric_volume_thrust'] = np.log(thrust_ratio) if thrust_ratio > 0 else -np.log(-thrust_ratio) if thrust_ratio < 0 else 0.0
                    else:
                        results['asymmetric_volume_thrust'] = np.nan
                else:
                    results['asymmetric_volume_thrust'] = np.nan
                avg_up_speed = up_price_change / len(up_minutes) if len(up_minutes) > 0 else 0
                avg_down_speed = down_price_change / len(down_minutes) if len(down_minutes) > 0 else 0
                if is_probe_date and avg_up_speed <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'avg_up_speed' 为 {avg_up_speed}。")
                if is_probe_date and avg_down_speed <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'avg_down_speed' 为 {avg_down_speed}。")
                if avg_up_speed > 0 and avg_down_speed > 0:
                    results['volatility_asymmetry_index'] = np.log(avg_up_speed / avg_down_speed)
                else:
                    results['volatility_asymmetry_index'] = np.nan
            else:
                results['asymmetric_volume_thrust'] = np.nan
                results['volatility_asymmetry_index'] = np.nan
            day_range = day_high - day_low
            if is_probe_date and day_range <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'day_range' 为 {day_range}。")
            if day_range > 0:
                range_pos_factor = ((day_close - day_low) / day_range) * 2 - 1
                value_dev_factor = np.tanh((day_close - daily_vwap) / atr)
                force_balance_factor = intraday_data['main_force_net_vol'].sum() / daily_total_volume if daily_total_volume > 0 else 0
                results['closing_price_deviation_score'] = (0.5 * range_pos_factor + 0.3 * value_dev_factor + 0.2 * force_balance_factor) * 100
            else:
                results['closing_price_deviation_score'] = np.nan
            # 使用 intraday_data.index 访问时间
            auction_df = intraday_data[intraday_data.index.time >= time(14, 57)]
            if is_probe_date and auction_df.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'auction_df' 为空。")
            if is_probe_date and continuous_trading_df.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'continuous_trading_df' 为空。")
            if not auction_df.empty and not continuous_trading_df.empty and \
               'close' in continuous_trading_df.columns and 'vol_shares' in continuous_trading_df.columns and \
               'vol' in auction_df.columns and 'high' in auction_df.columns and 'low' in auction_df.columns and \
               'minute_vwap' in auction_df.columns and 'main_force_net_vol' in auction_df.columns:
                pre_auction_close = continuous_trading_df['close'].iloc[-1]
                auction_vol = auction_df['vol_shares'].sum()
                avg_minute_vol = continuous_trading_df['vol_shares'].mean()
                if is_probe_date and auction_vol <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'auction_vol' 为 {auction_vol}。")
                if is_probe_date and avg_minute_vol <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'avg_minute_vol' 为 {avg_minute_vol}。")
                if auction_vol > 0 and avg_minute_vol > 0:
                    price_impact = (day_close - pre_auction_close) / atr
                    volume_impact = np.log1p((auction_vol / 3) / avg_minute_vol)
                    auction_amount = (auction_df['vol_shares'] * auction_df['minute_vwap']).sum()
                    if is_probe_date and auction_amount <= 0:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'auction_amount' 为 {auction_amount}。")
                    if auction_amount > 0:
                        mf_participation = auction_df['main_force_net_vol'].sum() * auction_df['minute_vwap'].mean() / auction_amount
                        results['closing_auction_ambush'] = price_impact * volume_impact * (1 + mf_participation) * 100
                    else:
                        results['closing_auction_ambush'] = np.nan
                else:
                    results['closing_auction_ambush'] = np.nan
            else:
                results['closing_auction_ambush'] = np.nan
            absorption_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] < daily_vwap]
            if is_probe_date and absorption_zone_df.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'absorption_zone_df' 为空。")
            if not absorption_zone_df.empty and \
               'main_force_net_vol' in absorption_zone_df.columns and 'vol_shares' in absorption_zone_df.columns and \
               'minute_vwap' in absorption_zone_df.columns:
                mf_absorption_df = absorption_zone_df[absorption_zone_df['main_force_net_vol'] > 0]
                if is_probe_date and mf_absorption_df.empty:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'mf_absorption_df' 为空。")
                if not mf_absorption_df.empty:
                    absorption_vol = mf_absorption_df['main_force_net_vol'].sum()
                    if is_probe_date and absorption_vol <= 0:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'absorption_vol' 为 {absorption_vol}。")
                    if absorption_vol > 0:
                        absorption_cost = (mf_absorption_df['minute_vwap'] * mf_absorption_df['main_force_net_vol']).sum() / absorption_vol
                        cost_deviation = (daily_vwap - absorption_cost) / atr
                        if is_probe_date and daily_total_volume <= 0:
                            print(f"    -> [行为指标探针] @ {trade_date}: 'daily_total_volume' 为 {daily_total_volume}。")
                        if daily_total_volume > 0:
                            results['dip_absorption_power'] = (absorption_vol / daily_total_volume) * cost_deviation * 100
                        else:
                            results['dip_absorption_power'] = np.nan
                    else:
                        results['dip_absorption_power'] = np.nan
                else:
                    results['dip_absorption_power'] = np.nan
            else:
                results['dip_absorption_power'] = np.nan
            peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
            troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            rally_dist_vol, panic_vol, total_rally_vol, total_panic_vol = 0, 0, 0, 0
            for i in range(len(turning_points) - 1):
                start_idx, end_idx = turning_points[i], turning_points[i+1]
                window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                if is_probe_date and window_df.empty:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'window_df' 为空。")
                if is_probe_date and len(window_df) < 2:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'window_df' 长度不足2。")
                if window_df.empty or len(window_df) < 2 or \
                   'minute_vwap' not in window_df.columns or 'main_force_net_vol' not in window_df.columns or \
                   'vol_shares' not in window_df.columns:
                    continue
                mf_net_vol = window_df['main_force_net_vol'].sum()
                window_vol = window_df['vol_shares'].sum()
                if window_df['minute_vwap'].iloc[-1] > window_df['minute_vwap'].iloc[0]:
                    total_rally_vol += window_vol
                    if mf_net_vol < 0:
                        rally_dist_vol += abs(mf_net_vol)
                else:
                    total_panic_vol += window_vol
                    if mf_net_vol < 0:
                        panic_vol += abs(mf_net_vol)
            if is_probe_date and total_rally_vol <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'total_rally_vol' 为 {total_rally_vol}。")
            if total_rally_vol > 0:
                results['rally_distribution_pressure'] = (rally_dist_vol / total_rally_vol) * 100
            else:
                results['rally_distribution_pressure'] = np.nan
            if is_probe_date and total_panic_vol <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'total_panic_vol' 为 {total_panic_vol}。")
            if total_panic_vol > 0:
                results['panic_selling_cascade'] = (panic_vol / total_panic_vol) * 100
            else:
                results['panic_selling_cascade'] = np.nan
            # 使用 intraday_data.index 访问时间
            posturing_df = continuous_trading_df[continuous_trading_df.index.time >= time(14, 30)]
            if is_probe_date and posturing_df.empty:
                print(f"    -> [行为指标探针] @ {trade_date}: 'posturing_df' 为空。")
            if pd.notna(daily_vwap) and atr > 0 and not posturing_df.empty and \
               'vol_shares' in posturing_df.columns and 'minute_vwap' in posturing_df.columns and \
               'main_force_net_vol' in posturing_df.columns:
                posturing_vwap = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum() / posturing_df['vol_shares'].sum()
                price_posture = (posturing_vwap - daily_vwap) / atr
                posturing_amount = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum()
                if is_probe_date and posturing_amount <= 0:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'posturing_amount' 为 {posturing_amount}。")
                if posturing_amount > 0:
                    force_posture = (posturing_df['main_force_net_vol'].sum() * posturing_vwap) / posturing_amount
                    results['pre_closing_posturing'] = (0.6 * price_posture + 0.4 * force_posture) * 100
                else:
                    results['pre_closing_posturing'] = np.nan
            else:
                results['pre_closing_posturing'] = np.nan
            if is_probe_date and day_range <= 0:
                print(f"    -> [行为指标探针] @ {trade_date}: 'day_range' 为 {day_range}。")
            if day_range > 0:
                fomo_zone_threshold = day_low + 0.75 * day_range
                fomo_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] > fomo_zone_threshold]
                if is_probe_date and fomo_zone_df.empty:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'fomo_zone_df' 为空。")
                if not fomo_zone_df.empty and \
                   'retail_net_vol' in fomo_zone_df.columns and 'retail_buy_vol' in continuous_trading_df.columns and \
                   'minute_vwap' in fomo_zone_df.columns:
                    fomo_retail_df = fomo_zone_df[fomo_zone_df['retail_net_vol'] > 0]
                    if is_probe_date and fomo_retail_df.empty:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'fomo_retail_df' 为空。")
                    if not fomo_retail_df.empty:
                        fomo_vol = fomo_retail_df['retail_net_vol'].sum()
                        total_retail_buy_vol = continuous_trading_df[continuous_trading_df['retail_buy_vol'] > 0]['retail_buy_vol'].sum()
                        if is_probe_date and fomo_vol <= 0:
                            print(f"    -> [行为指标探针] @ {trade_date}: 'fomo_vol' 为 {fomo_vol}。")
                        if is_probe_date and total_retail_buy_vol <= 0:
                            print(f"    -> [行为指标探针] @ {trade_date}: 'total_retail_buy_vol' 为 {total_retail_buy_vol}。")
                        if fomo_vol > 0 and total_retail_buy_vol > 0:
                            cost_fomo = (fomo_retail_df['minute_vwap'] * fomo_retail_df['retail_net_vol']).sum() / fomo_vol
                            cost_mf_sell = daily_data.get('avg_cost_main_sell')
                            if is_probe_date and pd.isna(cost_mf_sell): # 增加 pd.isna() 检查
                                print(f"    -> [行为指标探针] @ {trade_date}: 'avg_cost_main_sell' 缺失。")
                            if pd.notna(cost_mf_sell) and cost_mf_sell > 0:
                                premium = (cost_fomo / cost_mf_sell - 1)
                                results['retail_fomo_premium_index'] = premium * (fomo_vol / total_retail_buy_vol) * 100
                            else:
                                results['retail_fomo_premium_index'] = np.nan
                        else:
                            results['retail_fomo_premium_index'] = np.nan
                    else:
                        results['retail_fomo_premium_index'] = np.nan
                else:
                    results['retail_fomo_premium_index'] = np.nan
                panic_zone_threshold = day_low + 0.25 * day_range
                panic_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] < panic_zone_threshold]
                if is_probe_date and panic_zone_df.empty:
                    print(f"    -> [行为指标探针] @ {trade_date}: 'panic_zone_df' 为空。")
                if not panic_zone_df.empty and \
                   'retail_net_vol' in panic_zone_df.columns and 'retail_sell_vol' in continuous_trading_df.columns and \
                   'minute_vwap' in panic_zone_df.columns:
                    panic_retail_df = panic_zone_df[panic_zone_df['retail_net_vol'] < 0]
                    if is_probe_date and panic_retail_df.empty:
                        print(f"    -> [行为指标探针] @ {trade_date}: 'panic_retail_df' 为空。")
                    if not panic_retail_df.empty:
                        panic_vol = abs(panic_retail_df['retail_net_vol'].sum())
                        total_retail_sell_vol = continuous_trading_df[continuous_trading_df['retail_sell_vol'] > 0]['retail_sell_vol'].sum()
                        if is_probe_date and panic_vol <= 0:
                            print(f"    -> [行为指标探针] @ {trade_date}: 'panic_vol' 为 {panic_vol}。")
                        if is_probe_date and total_retail_sell_vol <= 0:
                            print(f"    -> [行为指标探针] @ {trade_date}: 'total_retail_sell_vol' 为 {total_retail_sell_vol}。")
                        if panic_vol > 0 and total_retail_sell_vol > 0:
                            cost_panic = (panic_retail_df['minute_vwap'] * abs(panic_retail_df['retail_net_vol'])).sum() / panic_vol
                            cost_mf_buy = daily_data.get('avg_cost_main_buy')
                            if is_probe_date and pd.isna(cost_mf_buy): # 增加 pd.isna() 检查
                                print(f"    -> [行为指标探针] @ {trade_date}: 'avg_cost_main_buy' 缺失。")
                            if pd.notna(cost_mf_buy) and cost_mf_buy > 0:
                                discount = (cost_mf_buy - cost_panic) / cost_mf_buy
                                results['retail_panic_surrender_index'] = discount * (panic_vol / total_retail_sell_vol) * 100
                            else:
                                results['retail_panic_surrender_index'] = np.nan
                        else:
                            results['retail_panic_surrender_index'] = np.nan
                    else:
                        results['retail_panic_surrender_index'] = np.nan
                else:
                    results['retail_panic_surrender_index'] = np.nan
            else:
                results['retail_fomo_premium_index'] = np.nan
                results['retail_panic_surrender_index'] = np.nan
        return results

    def _calculate_intraday_attribution_weights(self, intraday_data_for_day: pd.DataFrame, daily_data: pd.Series) -> pd.DataFrame:
        """
        【V9.5 · 逐笔数据兼容版 - 价格范围零值修复 - 探针增强】
        - 核心革命: 废弃“一体适用”的权重模型，为超大单、大单、中单、小单引入各自独特的、基于行为特征的权重分配逻辑。
        - 核心思想:
          - 超大单(ELG) -> 脉冲修正: 权重集中在成交量和振幅剧增的“暴力分钟”。
          - 大单(LG) -> VWAP修正: 权重与价格偏离VWAP的程度相关，体现战术意图。
          - 中单(MD) -> 动量修正: 权重与短期价格动量相关，体现追涨杀跌特性。
          - 小单(SM) -> 基准压力: 沿用原有的K线形态压力模型作为基准。
        - 核心修复: 修复了 `price_range` 为零时导致的 `decimal.InvalidOperation` 错误。
        - 【新增探针】增加探针，检查提前返回的条件是否被触发，以及 `buy_pressure_proxy` 和 `sell_score` 的中间计算结果。
        - 【修正】修复 `impulse_modifier` 计算中 `price_range` 的错误使用。
        """
        df = intraday_data_for_day.copy()
        trade_date = daily_data.name.date()
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        if is_probe_date:
            print(f"    -> [归因权重探针] @ {trade_date}: 开始计算归因权重。")
            print(f"       - 传入df 'vol_shares' 总和: {df['vol_shares'].sum():.6f}")
            print(f"       - 传入df 行数: {len(df)}")
        if 'vol_shares' not in df.columns or df['vol_shares'].sum() < 1e-6 or len(df) < 5:
            if is_probe_date:
                print(f"    -> [归因权重探针] @ {trade_date}: 提前返回！原因：'vol_shares'总和 ({df['vol_shares'].sum():.6f}) < 1e-6 或 行数 ({len(df)}) < 5。")
            for size in ['sm', 'md', 'lg', 'elg']:
                df[f'{size}_buy_weight'] = 0; df[f'{size}_sell_weight'] = 0
            return df
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        price_range = df['high'] - df['low']
        if is_probe_date:
            print(f"       - price_range min: {price_range.min():.4f}, max: {price_range.max():.4f}, mean: {price_range.mean():.4f}")
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
        if is_probe_date:
            print(f"       - buy_pressure_proxy min: {buy_pressure_proxy.min():.4f}, max: {buy_pressure_proxy.max():.4f}, mean: {buy_pressure_proxy.mean():.4f}")
            print(f"       - (1 - buy_pressure_proxy) min: {(1 - buy_pressure_proxy).min():.4f}, max: {(1 - buy_pressure_proxy).max():.4f}, mean: {(1 - buy_pressure_proxy).mean():.4f}")
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
            if is_probe_date:
                print(f"       - {size}_sell_score sum: {sell_score.sum():.6f}")
                print(f"       - {size}_buy_score sum: {buy_score.sum():.6f}")
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
        results = {
            'wash_trade_intensity': np.nan,
            'order_book_imbalance': np.nan,
            'large_order_pressure': np.nan,
            'large_order_support': np.nan,
        }
        # 修复：在方法入口处，将 None 转换为空的 DataFrame
        if daily_intraday_df is None:
            daily_intraday_df = pd.DataFrame()
        if daily_level5_df is None:
            daily_level5_df = pd.DataFrame()
        trade_date = daily_intraday_df.index[0].date() if not daily_intraday_df.empty else (daily_level5_df.index[0].date() if not daily_level5_df.empty else 'UNKNOWN')
        debug_params = self.debug_params if hasattr(self, 'debug_params') else {}
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        if is_probe_date and daily_intraday_df.empty:
            print(f"    -> [微观结构探针] @ {trade_date}: 'daily_intraday_df' 为空。")
        if is_probe_date and daily_level5_df.empty:
            print(f"    -> [微观结构探针] @ {trade_date}: 'daily_level5_df' 为空。")
        if is_probe_date and daily_total_volume <= 0:
            print(f"    -> [微观结构探针] @ {trade_date}: 'daily_total_volume' 为 {daily_total_volume}。")
        if daily_intraday_df.empty or daily_level5_df.empty or daily_total_volume <= 0:
            return results
        daily_intraday_df = daily_intraday_df.copy()
        for col in ['price', 'volume', 'amount']:
            if col in daily_intraday_df.columns:
                daily_intraday_df[col] = pd.to_numeric(daily_intraday_df[col], errors='coerce')
        daily_level5_df = daily_level5_df.copy()
        for i in range(1, 6):
            for prefix in ['buy_price', 'buy_volume', 'sell_price', 'sell_volume']:
                col_name = f'{prefix}{i}'
                if col_name in daily_level5_df.columns:
                    daily_level5_df[col_name] = pd.to_numeric(daily_level5_df[col_name], errors='coerce')
        def _aggregate_minute_group(group_df: pd.DataFrame) -> pd.Series:
            if group_df.empty:
                return pd.Series({
                    'vol': 0.0,
                    'vwap': np.nan,
                    'reversals': 0.0,
                    'trades': 0.0,
                    'vwap_std': 0.0
                })
            temp_df = group_df[['price', 'volume', 'reversal']].copy()
            temp_df['price'] = pd.to_numeric(temp_df['price'], errors='coerce')
            temp_df['volume'] = pd.to_numeric(temp_df['volume'], errors='coerce')
            temp_df.dropna(subset=['price', 'volume'], inplace=True)
            current_vol = temp_df['volume'].sum()
            current_vwap = np.nan
            if current_vol > 0:
                current_vwap = np.average(temp_df['price'], weights=temp_df['volume'])
            return pd.Series({
                'vol': current_vol,
                'vwap': current_vwap,
                'reversals': temp_df['reversal'].sum(),
                'trades': len(temp_df),
                'vwap_std': temp_df['price'].std() if len(temp_df) > 1 else 0.0
            })
        # 1. 计算主力对倒强度 (Wash Trade Intensity)
        if 'type' not in daily_intraday_df.columns:
            if is_probe_date:
                print(f"    -> [微观结构探针] @ {trade_date}: 'daily_intraday_df' 缺少 'type' 列，无法计算 'wash_trade_intensity'。")
            results['wash_trade_intensity'] = np.nan
        else:
            daily_intraday_df['direction'] = daily_intraday_df['type'].map({'B': 1, 'S': -1, 'M': 0})
            daily_intraday_df['reversal'] = (daily_intraday_df['direction'] * daily_intraday_df['direction'].shift(1)) < 0
            minute_agg = daily_intraday_df.resample('1min').apply(_aggregate_minute_group)
            minute_agg.dropna(subset=['vol', 'vwap', 'reversals', 'trades'], inplace=True)
            if is_probe_date and minute_agg.empty:
                print(f"    -> [微观结构探针] @ {trade_date}: 'minute_agg' 为空。")
            if not minute_agg.empty:
                minute_agg['wash_score'] = (minute_agg['vol'] / daily_total_volume) * (minute_agg['reversals'] / minute_agg['trades'].replace(0, 1)) * np.exp(-100 * minute_agg['vwap_std'] / minute_agg['vwap'].replace(0, 1))
                results['wash_trade_intensity'] = minute_agg['wash_score'].sum() * 100
            else:
                results['wash_trade_intensity'] = np.nan
        # 2. 计算五档盘口失衡度 (Order Book Imbalance)
        level5_cols_check = ['buy_volume1', 'buy_volume2', 'buy_volume3', 'buy_volume4', 'buy_volume5',
                             'sell_volume1', 'sell_volume2', 'sell_volume3', 'sell_volume4', 'sell_volume5']
        missing_level5_cols = [col for col in level5_cols_check if col not in daily_level5_df.columns]
        if missing_level5_cols:
            if is_probe_date:
                print(f"    -> [微观结构探针] @ {trade_date}: 'daily_level5_df' 缺少 '{missing_level5_cols}' 列，无法计算 'order_book_imbalance'。")
            results['order_book_imbalance'] = np.nan
        else: # 只有当所有列都存在时才计算
            daily_level5_df['buy_vol_total'] = daily_level5_df[['buy_volume1', 'buy_volume2', 'buy_volume3', 'buy_volume4', 'buy_volume5']].sum(axis=1)
            daily_level5_df['sell_vol_total'] = daily_level5_df[['sell_volume1', 'sell_volume2', 'sell_volume3', 'sell_volume4', 'sell_volume5']].sum(axis=1)
            total_book_vol = daily_level5_df['buy_vol_total'] + daily_level5_df['sell_vol_total']
            daily_level5_df['imbalance'] = (daily_level5_df['buy_vol_total'] - daily_level5_df['sell_vol_total']) / total_book_vol.replace(0, np.nan)
            daily_level5_df.dropna(subset=['imbalance'], inplace=True)
            if is_probe_date and daily_level5_df.empty:
                print(f"    -> [微观结构探针] @ {trade_date}: 'daily_level5_df' (dropna) 为空。")
            if not daily_level5_df.empty:
                time_diffs = daily_level5_df.index.to_series().diff().dt.total_seconds().fillna(0)
                results['order_book_imbalance'] = np.average(daily_level5_df['imbalance'], weights=time_diffs) * 100
            else:
                results['order_book_imbalance'] = np.nan
        # 3. 计算大单压制与支撑强度 (Large Order Pressure/Support)
        large_order_threshold_value = 500000 # 定义大单门槛为50万元
        pressure_strength = 0
        support_strength = 0
        level5_price_vol_check_large_order = ['sell_price1', 'sell_volume1', 'sell_price2', 'sell_volume2',
                                              'buy_price1', 'buy_volume1', 'buy_price2', 'buy_volume2']
        missing_level5_cols_large_order = [col for col in level5_price_vol_check_large_order if col not in daily_level5_df.columns]
        if missing_level5_cols_large_order:
            if is_probe_date:
                print(f"    -> [微观结构探针] @ {trade_date}: 'daily_level5_df' 缺少列 {missing_level5_cols_large_order}，无法计算大单压制/支撑。")
            results['large_order_pressure'] = np.nan
            results['large_order_support'] = np.nan
        elif not daily_level5_df.empty:
            time_diffs_seconds = daily_level5_df.index.to_series().diff().dt.total_seconds().values
            for i in range(1, len(daily_level5_df)):
                row = daily_level5_df.iloc[i]
                duration = time_diffs_seconds[i]
                # 计算压制强度
                for p, v in [('sell_price1', 'sell_volume1'), ('sell_price2', 'sell_volume2')]:
                    if is_probe_date and pd.isna(row[p]):
                        print(f"    -> [微观结构探针] @ {trade_date}: Level5 '{p}' 缺失。")
                    if is_probe_date and pd.isna(row[v]):
                        print(f"    -> [微观结构探针] @ {trade_date}: Level5 '{v}' 缺失。")
                    if pd.notna(row[p]) and pd.notna(row[v]) and row[p] * row[v] * 100 > large_order_threshold_value:
                        pressure_strength += row[v] * 100 * duration
                # 计算支撑强度
                for p, v in [('buy_price1', 'buy_volume1'), ('buy_price2', 'buy_volume2')]:
                    if is_probe_date and pd.isna(row[p]):
                        print(f"    -> [微观结构探针] @ {trade_date}: Level5 '{p}' 缺失。")
                    if is_probe_date and pd.isna(row[v]):
                        print(f"    -> [微观结构探针] @ {trade_date}: Level5 '{v}' 缺失。")
                    if pd.notna(row[p]) and pd.notna(row[v]) and row[p] * row[v] * 100 > large_order_threshold_value:
                        support_strength += row[v] * 100 * duration
            if daily_total_volume > 0:
                results['large_order_pressure'] = pressure_strength / (daily_total_volume * 240 * 60) * 100
                results['large_order_support'] = support_strength / (daily_total_volume * 240 * 60) * 100
            else:
                results['large_order_pressure'] = np.nan
                results['large_order_support'] = np.nan
        else:
            results['large_order_pressure'] = np.nan
            results['large_order_support'] = np.nan
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








